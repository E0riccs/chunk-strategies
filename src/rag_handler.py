import os
import datetime
import uuid

import chromadb.utils.embedding_functions as embedding_functions
from src.rag_utils.vector_store_handler import VectorStoreHandler
from src.rag_utils.reranker import Reranker

class RAGHandler:
    """
        RAG处理器：负责向量数据库管理、文档检索、重排序和问答生成
        封装了完整的RAG流程，对外提供简洁的接口
    """
    def __init__(self,
                 vector_store_base_persist_dir: str):
        """
        1. 完善向量数据库配置，随实验运行建立实例
        2. 完善重排序配置，随实验运行建立实例

        Args:
            vector_store_base_persist_dir (str): 向量数据库基础持久化目录
        """
        # 向量数据库
        self.vector_store_base_persist_dir = vector_store_base_persist_dir

        # 重排序
        self.reranker = None


    def _setup_vector_store(self,
                            file_type_name: str,
                            chunking_strategy_name: str,
                            vector_store_collection_name_prefix: str = "experiment") -> str:
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
        safe_file_type_name = file_type_name.replace(' ', '_').lower()
        safe_strategy_name = chunking_strategy_name.replace(' ', '_').lower()

        # 为每个文件类型+策略组合创建唯一的collection，避免干扰并允许干净的重新运行
        collection_name_suffix = f"{safe_file_type_name}_{safe_strategy_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        # 进一步清理collection_name_suffix，ChromaDB有限制
        collection_name_suffix = collection_name_suffix.replace('-', '_')  # 替换连字符
        current_collection_name = f"{vector_store_collection_name_prefix}_{collection_name_suffix}"
        
        # 确保collection名称对ChromaDB有效（例如，长度、字符）
        current_collection_name = current_collection_name[:60]  # collection名称最大长度为63
        current_collection_name = ''.join(c if c.isalnum() or c in ['_', '.'] else '_' for c in current_collection_name)
        if not current_collection_name[0].isalnum() or not current_collection_name[-1].isalnum():
             current_collection_name = 'c' + current_collection_name[1:-1] + 'c'  # 确保开始/结束是字母数字

        print(f"Initializing VectorStore for collection: {current_collection_name}")
        
        # 2. embedding



        # 3. vector store handler
        self.vector_store_handler = VectorStoreHandler(
            persist_directory=os.path.join(self.vector_store_base_persist_dir, current_collection_name),
            collection_name=current_collection_name,
            embedding_model_name=current_embedding_function.model_name if hasattr(current_embedding_function, 'model_name') else 'sentence-transformers/all-MiniLM-L6-v2',
            openai_api_key=current_embedding_function.api_key if is_openai_embed and hasattr(current_embedding_function, 'api_key') else None,
            openai_embedding_model=current_embedding_function.model_name if is_openai_embed and hasattr(current_embedding_function, 'model_name') else 'text-embedding-ada-002',
            use_openai_embeddings=is_openai_embed
        )
        
        return current_collection_name

    def _setup_reranker(self):
        """
        设置重排序器
        """
        self.reranker = Reranker()


    def add_documents_to_vector_store(self, 
                                    chunks: list, 
                                    file_type_name: str, 
                                    chunking_strategy_name: str, 
                                    original_text_length: int) -> None:
        """
        将文档块添加到向量数据库
        
        Args:
            chunks (list): 文档块列表
            file_type_name (str): 文件类型名称
            chunking_strategy_name (str): 分块策略名称
            original_text_length (int): 原始文本长度
        """
        if not self.vector_store_handler:
            raise ValueError("Vector store handler not initialized. Call setup_vector_store_for_experiment first.")
            
        print(f"Adding {len(chunks)} chunks to vector store...")
        chunk_metadatas = [
            {
                "source_file_type": file_type_name,
                "chunking_strategy": chunking_strategy_name,
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
                        reranker_top_n: int = 3,
                        vector_store_filter: dict = None,
                        qa_model_id: str = 'default_model') -> dict:
        """
        Answers a question using the RAG pipeline: retrieve, (optionally) rerank, then generate answer.

        Args:
            question_text (str): The question to answer.
            retrieval_n_results (int): Number of documents to retrieve from vector store.
            reranker_top_n (int): Number of documents to keep after reranking. Only used if reranker is available.
            vector_store_filter (dict, optional): Filter for vector store retrieval.
            qa_model_id (str): The model ID to be used by LLM_handler for generating the answer.

        Returns:
            dict: A dictionary containing the final answer, context, and other details.
        """
        if not question_text:
            print("Error: Question text cannot be empty.")
            return {
                "final_answer": "Error: No question provided.",
                "retrieved_documents_count": 0,
                "reranked_documents_count": 0,
                "context_for_answer": ""
            }

        # 1. Retrieve documents
        print(f"Retrieving documents for question: '{question_text}'")
        retrieved_docs_result = self.vector_store_handler.query_documents(
            query_text=question_text,
            n_results=retrieval_n_results,
            where_filter=vector_store_filter,
            include=["documents", "metadatas"] # Ensure documents are included
        )
        
        retrieved_documents = []
        if retrieved_docs_result and retrieved_docs_result.get('documents') and retrieved_docs_result['documents'][0]:
            retrieved_documents = retrieved_docs_result['documents'][0]
            print(f"Retrieved {len(retrieved_documents)} documents from vector store.")
        else:
            print("No documents retrieved from vector store.")
            return {
                "final_answer": "Could not retrieve relevant documents to answer the question.",
                "retrieved_documents_count": 0,
                "reranked_documents_count": 0,
                "context_for_answer": ""
            }

        # 2. Rerank documents (if reranker is available)
        context_for_llm = retrieved_documents
        reranked_documents_count = 0
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
                reranked_documents_count = len(reranked_docs)
                print(f"Reranked down to {len(context_for_llm)} documents.")
            else:
                print("Reranking did not return any documents. Using original retrieved documents.")
        elif not self.reranker:
            print("Reranker not available. Skipping reranking step.")
        
        # 3. Generate answer using LLM
        print(f"Generating answer using LLM (model: {qa_model_id}) with {len(context_for_llm)} documents as context...")
        
        # The llm_handler.answer_question_with_context method needs to be defined or adapted.
        # For now, let's assume it takes the question and a list of context strings.
        # This is a placeholder for where the actual LLM call in llm_handler would be invoked.
        # We need to ensure llm_handler has a method that fits this RAG flow.
        # The original `answer_question_rag` was in `llm_handler`, we are moving its logic here.

        prompt = self.llm_handler._load_prompt("RAGAnswer") # Assuming a RAG-specific prompt
        if not prompt:
            print("Warning: RAGAnswer prompt not found. Using a generic approach.")
            # Fallback prompt or structure if specific RAG prompt is missing
            context_str = "\n\n---\n\n".join(context_for_llm)
            user_content = f"Question: {question_text}\n\nContext:\n{context_str}\n\nAnswer:"
        else:
            context_str = "\n\n---\n\n".join(context_for_llm)
            # Use the build_content method from llm_handler if it's suitable
            user_content = self.llm_handler.build_content(prompt, original_text=context_str, question=question_text)
            # Adjust build_content or create a new method in llm_handler for RAG-specific prompt formatting
            # For example, the prompt might have placeholders like {{question}} and {{context}}
            user_content = prompt.replace("{{question}}", question_text).replace("{{context}}", context_str)

        # Use the specified qa_model_id for this call
        # This requires llm_handler to be able to switch models or use a specific one for a call
        # For simplicity, let's assume llm_handler's current model is used, or it handles model switching internally.
        # If llm_handler needs to be re-initialized or have a method to set model temporarily, that's an extension.
        raw_llm_response = self.llm_handler._call_llm(user_content) # Using the _call_llm method
        final_answer = self.llm_handler.api.answer_from_json(raw_llm_response) # Assuming this extracts the answer string
        
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