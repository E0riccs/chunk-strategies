import chromadb
from chromadb.utils import embedding_functions
import os

# from langchain_community.vectorstores import Chroma # Alternative using LangChain wrapper
# from langchain_openai import OpenAIEmbeddings # Alternative for embeddings

class VectorStoreHandler:
    def __init__(self, 
                 collection_name="rag_chunks", 
                 persist_directory="db/chroma_db", 
                 embedding_model_name="all-MiniLM-L6-v2", 
                 openai_api_key=None, 
                 openai_embedding_model="text-embedding-ada-002",
                 use_openai_embeddings=False): 
        """
        Initializes the VectorStoreHandler.

        Args:
            persist_directory (str): Directory to persist ChromaDB data.
            collection_name (str): Name of the collection in ChromaDB.
            embedding_model_name (str): Name of the SentenceTransformer model for embeddings.
                                         Used if use_openai_embeddings is False.
            openai_api_key (str, optional): OpenAI API key. Required if use_openai_embeddings is True.
            openai_embedding_model (str): Name of the OpenAI embedding model.
                                          Used if use_openai_embeddings is True.
            use_openai_embeddings (bool): If True, uses OpenAI embeddings. Otherwise, uses SentenceTransformer.
        """
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            print(f"Created persistence directory: {persist_directory}")

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name

        if use_openai_embeddings:
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when use_openai_embeddings is True.")
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=openai_embedding_model
            )
            print(f"Using OpenAI embeddings with model: {openai_embedding_model}")
        else:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
            print(f"Using SentenceTransformer embeddings with model: {embedding_model_name}")

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        print(f"Successfully connected to collection '{self.collection.name}' with {self.collection.count()} documents.")

    def add_documents(self, chunks, metadatas=None, ids=None):
        """
        Adds documents (chunks) to the ChromaDB collection.

        Args:
            chunks (list of str): The text content of the documents.
            metadatas (list of dict, optional): Metadata associated with each document.
            ids (list of str, optional): Unique IDs for each document. If None, generated automatically.
        """
        if not chunks:
            print("No chunks provided to add.")
            return

        if not ids:
            # Generate unique IDs if not provided, to avoid collisions if re-adding similar content
            # A more robust ID generation might be needed depending on use case (e.g., hash of content + metadata)
            start_id_num = self.collection.count() # Simplistic way to avoid collision on subsequent calls
            ids = [f"doc_{start_id_num + i}" for i in range(len(chunks))]
        
        if metadatas and len(chunks) != len(metadatas):
            raise ValueError("Number of chunks must match number of metadatas.")
        if ids and len(chunks) != len(ids):
            raise ValueError("Number of chunks must match number of ids.")

        try:
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
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
    # Ensure you have an OpenAI API key set as an environment variable 
    # if you want to test with use_openai_embeddings=True
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    print("Initializing VectorStoreHandler (default SentenceTransformer embeddings)...")
    # Relative path for persistence from the script's location if run directly
    # For project structure, this path might need adjustment or be absolute
    script_dir = os.path.dirname(os.path.abspath(__file__))
    persist_path = os.path.join(os.path.dirname(script_dir), 'db', 'chroma_test_db')
    
    vector_store = VectorStoreHandler(persist_directory=persist_path, collection_name="test_collection")
    print(f"Initial count: {vector_store.get_collection_count()}")

    # Clear collection for a fresh start if it exists from previous runs
    if vector_store.get_collection_count() > 0:
        print("Clearing existing test collection...")
        vector_store.clear_collection()
        print(f"Count after clearing: {vector_store.get_collection_count()}")

    sample_chunks = [
        "This is the first document about apples.",
        "The second document discusses bananas and their properties.",
        "Oranges are a great source of Vitamin C, making this the third document.",
        "Apples and oranges are both fruits."
    ]
    sample_metadatas = [
        {"source": "doc_A", "topic": "fruit"},
        {"source": "doc_B", "topic": "fruit"},
        {"source": "doc_C", "topic": "fruit"},
        {"source": "doc_A", "topic": "comparison"}
    ]
    sample_ids = ["id1", "id2", "id3", "id4"]

    print("\nAdding documents...")
    vector_store.add_documents(sample_chunks, sample_metadatas, sample_ids)
    print(f"Count after adding: {vector_store.get_collection_count()}")

    print("\nQuerying for 'apples'...")
    results_apple = vector_store.query_documents("apples", n_results=2)
    if results_apple and results_apple.get('documents'):
        for i, doc in enumerate(results_apple['documents'][0]):
            print(f"  Result {i+1}: {doc}")
            print(f"    Metadata: {results_apple['metadatas'][0][i]}")
            print(f"    Distance: {results_apple['distances'][0][i]}")
    else:
        print("No results for 'apples'.")

    print("\nQuerying for 'vitamin C' with filter source='doc_C'...")
    results_vitamin_c = vector_store.query_documents("vitamin C", n_results=1, where_filter={"source": "doc_C"})
    if results_vitamin_c and results_vitamin_c.get('documents'):
        for i, doc in enumerate(results_vitamin_c['documents'][0]):
            print(f"  Result {i+1}: {doc}")
            print(f"    Metadata: {results_vitamin_c['metadatas'][0][i]}")
    else:
        print("No results for 'vitamin C' with the specified filter.")
    
    print("\nQuerying for 'bananas' with filter topic='fruit'...")
    results_bananas = vector_store.query_documents("bananas", n_results=2, where_filter={"topic": "fruit"})
    if results_bananas and results_bananas.get('documents'):
        for i, doc in enumerate(results_bananas['documents'][0]):
            print(f"  Result {i+1}: {doc}")
            print(f"    Metadata: {results_bananas['metadatas'][0][i]}")
    else:
        print("No results for 'bananas' with the specified filter.")

    # Test OpenAI embeddings if API key is available
    # if OPENAI_API_KEY:
    #     print("\n--- Testing with OpenAI Embeddings ---")
    #     openai_persist_path = os.path.join(os.path.dirname(script_dir), 'db', 'chroma_openai_test_db')
    #     vector_store_openai = VectorStoreHandler(
    #         persist_directory=openai_persist_path, 
    #         collection_name="openai_test_collection",
    #         use_openai_embeddings=True,
    #         openai_api_key=OPENAI_API_KEY
    #     )
    #     if vector_store_openai.get_collection_count() > 0:
    #         vector_store_openai.clear_collection()
    #     vector_store_openai.add_documents(sample_chunks, sample_metadatas, sample_ids)
    #     results_openai = vector_store_openai.query_documents("apples", n_results=2)
    #     if results_openai and results_openai.get('documents'):
    #         for i, doc in enumerate(results_openai['documents'][0]):
    #             print(f"  OpenAI Result {i+1}: {doc}")
    # else:
    #     print("\nSkipping OpenAI embeddings test as OPENAI_API_KEY is not set.")

    print("\nVectorStoreHandler example usage finished.")