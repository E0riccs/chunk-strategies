import os
import datetime
import uuid

from src.rag_utils.vector_store_handler import VectorStoreHandler
from src.rag_utils.reranker import Reranker
from src.llm_handler import LLM_handler

from src.rag_utils.response_model import RagAnswerModel

class RAGHandler:
    """
        RAG处理器：负责向量数据库管理、文档检索、重排序和问答生成
        封装了完整的RAG流程，对外提供简洁的接口
    """
    def __init__(self,
                vector_store_base_persist_dir='db/chroma_db',
                reranker_model_name='default_model'):
        """
        1. 完善向量数据库配置，随实验运行建立实例
        2. 随实验运行建立重排序实例
        3. 随实验运行建立问答 LLM 实例

        Args:
            vector_store_base_persist_dir (str): 向量数据库基础持久化目录
        """

        # 向量数据库
        self.vector_store_base_persist_dir = vector_store_base_persist_dir

    def _setup_vector_store(self,
                            file_type_name: str,
                            chunking_strategy_name: str,
                            vector_store_collection_name_prefix: str = "experiment",
                            use_api_embeddings: bool = False,
                            embedding_model_name: str = 'default_embedding',
                            api_platform: str = 'siliconflow') -> str:
        """
        1. 为特定实验设置向量数据库操作实例；
        2. 创建唯一的 collection
        
        Args:
            file_type_name (str): 文件类型名称
            chunking_strategy_name (str): 分块策略名称
            vector_store_collection_name_prefix (str): collection名称前缀
            
        Returns:
            str: 创建的collection名称
        """


        # 1. collection
        # 清理名称用于文件路径
        self.file_type_name = file_type_name
        self.chunking_strategy_name = chunking_strategy_name

        safe_file_type_name = self.file_type_name.replace(' ', '_').lower()
        safe_strategy_name = self.chunking_strategy_name.replace(' ', '_').lower()

        # 为每个文件类型+策略组合创建唯一的collection，避免干扰并允许干净的重新运行
        collection_name_suffix = f"{safe_file_type_name}_{safe_strategy_name}"
        # 进一步清理collection_name_suffix，ChromaDB有限制
        collection_name_suffix = collection_name_suffix.replace('-', '_')  # 替换连字符
        current_collection_name = f"{vector_store_collection_name_prefix}_{collection_name_suffix}"
        
        # 确保collection名称对ChromaDB有效（例如，长度、字符）
        current_collection_name = current_collection_name[:60]  # collection名称最大长度为63
        current_collection_name = ''.join(c if c.isalnum() or c in ['_', '.'] else '_' for c in current_collection_name)
        if not current_collection_name[0].isalnum() or not current_collection_name[-1].isalnum():
             current_collection_name = 'c' + current_collection_name[1:-1] + 'c'  # 确保开始/结束是字母数字

        self.collection_name = current_collection_name
        print(f"Initializing VectorStore for collection: {current_collection_name}")
    
        # 2. vector store handler
        self.vector_store_handler = VectorStoreHandler(
            collection_name=current_collection_name,
            persist_directory=os.path.join(self.vector_store_base_persist_dir, current_collection_name),
            use_api_embeddings=use_api_embeddings,
            embedding_model_name=embedding_model_name,
            api_platform=api_platform
        )
        
        return current_collection_name

    def _setup_reranker(self, model_name):
        """
        设置重排序器
        """
        self.reranker = Reranker(reranker_model_name=model_name)


    def add_documents_to_vector_store(self, 
                                    chunks: list, 
                                    original_text_length: int) -> None:
        """
        将文档块添加到向量数据库
        
        Args:
            chunks (list): 文档块列表
            original_text_length (int): 原始文本长度
        """
        if not self.vector_store_handler:
            raise ValueError("Vector store handler not initialized. Call _setup_vector_store first.")
            
        print(f"Adding {len(chunks)} chunks to vector store...")
        chunk_metadatas = [
            {
                "source_file_type": self.file_type_name,
                "chunking_strategy": self.chunking_strategy_name,
                "chunk_index": i,
                "original_text_length": original_text_length
            } for i in range(len(chunks))
        ]
        
        # 为每个块生成唯一ID，确保它们可以被单独引用/更新
        chunk_ids = [f"{self.collection_name}_chunk_{uuid.uuid4()}" for _ in range(len(chunks))]
        self.vector_store_handler.add_documents(chunks, metadatas=chunk_metadatas, ids=chunk_ids)
        print(f"Vector store now contains {self.vector_store_handler.get_collection_count()} documents.")

    def answer_question(self, 
                        question_text: str, 
                        retrieval_n_results: int = 10, 
                        qa_model_id: str = 'default_model',
                        if_rerank: bool = False,
                        reranker_top_n: int = 3,
                        reranker_method_name: str = 'default_reranker') -> dict:
        """
        Answers a question using the RAG pipeline: retrieve, (optionally) rerank, then generate answer.
        Filter documents is not available in this version.

        Args:
            question_text (str): The question to answer.
            retrieval_n_results (int): Number of documents to retrieve from vector store.
            if_rerank (bool): Whether to rerank retrieved documents.
            reranker_top_n (int): Number of documents to keep after reranking. Only used if reranker is available.
            qa_model_id (str): The model ID to be used by LLM_handler for generating the answer.
            reranker_method_name (str): The method name to be used by Reranker for reranking.

        Returns:
            dict: A dictionary containing the final answer, context, and other details.
        """
        if not question_text:
            print("Error: Question text cannot be empty.")
            return RagAnswerModel.create_response(
                final_answer="Error: No question provided.",
                retrieved_documents_count=0,
                reranked_documents_count=0,
                context_for_answer=""
            )

        # 1. Retrieve documents
        print(f"Retrieving documents for question: '{question_text}'")
        retrieved_docs_result = self.vector_store_handler.query_documents(
            query_text=question_text,
            n_results=retrieval_n_results
        )
        
        retrieved_documents = []
        if retrieved_docs_result and retrieved_docs_result.get('documents') and retrieved_docs_result['documents'][0]:
            retrieved_documents = retrieved_docs_result['documents'][0]
        else:
            return RagAnswerModel.create_response(
                final_answer="Could not retrieve relevant documents to answer the question.",
                retrieved_documents_count=0,
                reranked_documents_count=0,
                context_for_answer=""
            )

        # 2. Rerank documents (if reranker is available)
        # TODO: reranker is not tested in this version
        if if_rerank:
            self._setup_reranker(reranker_method_name)
            documents_count = 0
            if self.reranker and retrieved_documents:
                print(f"Reranking {len(retrieved_documents)} documents...")
                reranked_docs = self.reranker.rerank_documents(
                    query=question_text,
                    documents=retrieved_documents,
                    top_n=reranker_top_n,
                    return_documents=True
                )
            if reranked_docs:
                context_for_llm = reranked_docs
                documents_count = len(reranked_docs)
                print(f"Reranked down to {documents_count} documents.")
            else:
                context_for_llm = retrieved_documents
                documents_count = len(context_for_llm)
                print("Reranking did not return any documents. Using original retrieved documents.")
        else:
            context_for_llm = retrieved_documents
            documents_count = len(context_for_llm)
            print("Reranker not available. Skipping reranking step.")
        
        # 3. Generate answer using LLM
        chat_llm_handler = LLM_handler(model_id = qa_model_id)

        print(f"Generating answer using LLM (model: {qa_model_id}) with {documents_count} documents as context...")

        answer = chat_llm_handler.chat(question = question_text, document = context_for_llm, task = "RAGAnswer") # Assuming a RAG-specific prompt
        
        # If answer_from_json returns a dict, extract the relevant part
        if isinstance(final_answer, dict) and 'answer' in final_answer:
            final_answer = final_answer['answer']
        elif isinstance(final_answer, dict) and 'response' in final_answer: # Common key for response text
            final_answer = final_answer['response']
        # Add more robust extraction if needed based on actual LLM response structure

        print(f"Generated final answer: {final_answer[:100]}...")

        return {
            "final_answer": final_answer,
            "retrieved_documents_count": len(retrieved_documents),
            "reranked_documents_count": reranked_documents_count,
            "context_for_answer": context_for_llm # or context_str for the string version
        }

# Example Usage (Conceptual - requires actual instances of handlers)
if __name__ == '__main__':
    pass