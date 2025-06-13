import os

import chromadb
from chromadb.utils import embedding_functions
from src.rag_utils.embeddings import extract_embedding_from_json
from src.rag_utils.response_model import APIResponseModel

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys
sys.path.insert(0, project_root)

from src.rag_utils.api_factory import APIFactory


class VectorStoreHandler:
    def __init__(self,
                 collection_name="rag_chunks", 
                 persist_directory="db/chroma_db", 
                 use_api_embeddings= True, 
                 embedding_model_name= None,
                 api_platform = None):
        """
        Initializes the VectorStoreHandler.

        Args:
            persist_directory (str): Directory to persist ChromaDB data.
            collection_name (str): Name of the collection in ChromaDB.
            use_api_embeddings (bool): If True, uses API embeddings. Otherwise, uses SentenceTransformer.
            embedding_model_name (str): Name of the model for embeddings.(API)
            api_platform (str): API platform for embeddings.(API)
        """
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            print(f"Created persistence directory: {persist_directory}")

        self.chroma_compatible_api = True

        self.client = chromadb.PersistentClient(path=persist_directory) # 持久化保存
        self.collection_name = collection_name

        if use_api_embeddings:
            # api embedding
            if not embedding_model_name:
                raise ValueError("embedding_model_name must be specified when use_api_embeddings is True")
            if not api_platform:
                raise ValueError("api_platform must be specified when use_api_embeddings is True")

            if self.is_in_chroma_api_embeddings(api_platform):
                # TODO not tested, not work without environment variables
                self.chroma_compatible_api = True
                self.embedding_function = embedding_functions.known_embedding_functions[api_platform](
                    model_name=embedding_model_name
                )
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function 
                )# with embedding function
                print(f"Using Api embeddings (chromadb compatible) bound with model: {self.embedding_model_name}")
            else:
                self.chroma_compatible_api = False
                self.embedding_function = APIFactory(model_id=embedding_model_name).create_api()
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name
                )# without embedding function

            self.embedding_model_name = embedding_model_name
            print(f"Using Api embeddings with model: {self.embedding_model_name}")
        else:
            # local embedding
            self.chroma_compatible_api = True
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name='all-MiniLM-L6-v2'
            )
            self.embedding_model_name = 'SentenceTransformer-all-MiniLM-L6-v2'
            print(f"Using embeddings with model: {self.embedding_model_name}")

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

        print(f"Successfully connected to collection '{self.collection.name}' with {self.collection.count()} documents.")

    def is_in_chroma_api_embeddings(self, platform_name:str):
        return platform_name in embedding_functions.known_embedding_functions
        
    def add_documents(self, chunks, metadatas=None, ids=None):
        """
        Adds documents (chunks) to the ChromaDB collection.

        Args:
            chunks (list of str): The text content of the documents.
            metadatas (list of dict, optional): Metadata associated with each document.
            ids (list of str, optional): Unique IDs for each document. If None, generated automatically.
            embeddings (list of embeddings, optional): Embeddings for each document. If None, generated automatically.
        """
        if not chunks:
            print("No chunks provided to add.")
            return

        if not ids:
            start_id_num = self.collection.count() # Simplistic way to avoid collision on subsequent calls
            ids = [f"doc_{start_id_num + i}" for i in range(len(chunks))]
        
        if metadatas and len(chunks) != len(metadatas):
            raise ValueError("Number of chunks must match number of metadatas.")
        if ids and len(chunks) != len(ids):
            raise ValueError("Number of chunks must match number of ids.")

        try:
            if self.chroma_compatible_api:
                self.collection.upsert(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                embeddings = []
                for chunk in chunks:
                    response = self.embedding_function.get_embedding(chunk)
                    response = self.embedding_function.embed_answer_from_json(response)
                    embeddings.append(extract_embedding_from_json(response[APIResponseModel.RESPONSE_CONTENT]))
                self.collection.upsert(
                    documents=chunks,
                    metadatas=metadatas,
                    embeddings= embeddings,
                    ids=ids
                )
            print(f"Added {len(chunks)} documents to collection '{self.collection.name}'. Total documents: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            # Potentially log more details or re-raise specific exceptions

    def query_documents(self, query_text, n_results=5, where_filter=None, include=["metadatas", "documents", "distances"]):
        """
        Queries the ChromaDB collection for documents similar to the query_text.

        Args:
            query_text (str): The text to search for.
            n_results (int): The number of results to return.
            where_filter (dict, optional): A filter to apply to the metadata. 
                                         Example: {"source_file_type": "chapter_text"}
            include (list of str): A list of what to include in the results. 
                                   Defaults to ["metadatas", "documents", "distances"].

        Returns:
            dict: The query results from ChromaDB, or an empty dict if no results or error.
                  Example structure for results['documents'][0], results['metadatas'][0] etc.
        """
        if not query_text:
            print("Query text cannot be empty.")
            return {}
        
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self.collection.count()), # Cannot request more results than available
                where=where_filter,
                include=include
            )
            # print(f"Query returned {len(results.get('documents', [[]])[0])} results.")
            return results
        except Exception as e:
            print(f"Error querying documents from ChromaDB: {e}")
            return {}

    def get_collection_count(self):
        """Returns the number of documents in the collection."""
        return self.collection.count()

    def clear_collection(self):
        """Deletes all documents from the current collection."""
        print(f"Attempting to clear collection: {self.collection_name}")
        # ChromaDB doesn't have a direct 'clear' for all items. 
        # We need to delete the collection and recreate it.
        self.client.delete_collection(name=self.collection_name)
        print(f"Deleted collection: {self.collection_name}")
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        print(f"Recreated collection: {self.collection.name}. Current count: {self.collection.count()}")

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # connect to db
    vector_store = VectorStoreHandler(
        collection_name="test1", 
        persist_directory="db/chroma_db", 
        use_api_embeddings=True, 
        embedding_model_name="default_embedding", 
        api_platform="siliconflow")

    # TODO 测试向量数据库于此

    