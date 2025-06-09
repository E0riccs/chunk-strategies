import os
import cohere

class Reranker:
    def __init__(self, cohere_api_key=None, model='rerank-english-v2.0'): # You can also use 'rerank-multilingual-v2.0'
        """
        Initializes the Reranker with a Cohere API key.

        Args:
            cohere_api_key (str, optional): The Cohere API key. 
                                          If None, it tries to read from COHERE_API_KEY environment variable.
            model (str): The reranking model to use.
        """
        if cohere_api_key is None:
            cohere_api_key = os.getenv('COHERE_API_KEY')
        
        if not cohere_api_key:
            raise ValueError("Cohere API key not provided and not found in COHERE_API_KEY environment variable.")
        
        self.co = cohere.Client(cohere_api_key)
        self.model = model
        print(f"Cohere Reranker initialized with model: {self.model}")

    def rerank_documents(self, query, documents, top_n=None, return_documents=True):
        """
        Reranks a list of documents based on a query using the Cohere Rerank API.

        Args:
            query (str): The search query.
            documents (list of str): A list of documents (strings) to rerank.
            top_n (int, optional): The number of reranked documents to return. 
                                   If None, returns all reranked documents.
            return_documents (bool): If True, returns the document texts. Otherwise, returns full result objects.

        Returns:
            list: A list of reranked document texts (if return_documents is True) 
                  or a list of Cohere rerank result objects (if return_documents is False).
                  Returns an empty list if an error occurs or no documents are provided.
        """
        if not documents:
            print("No documents provided for reranking.")
            return []
        if not query:
            print("No query provided for reranking. Returning original documents.")
            return documents # Or an empty list, depending on desired behavior

        try:
            rerank_results = self.co.rerank(
                query=query,
                documents=documents,
                top_n=top_n,
                model=self.model,
                return_documents=False # Get full result objects to access scores and indices
            )
            
            # The API returns a RerankResponse object which has a 'results' attribute.
            # Each item in 'results' has 'document' (if requested), 'index', and 'relevance_score'.
            
            if return_documents:
                # Sort the original documents based on the reranked order
                # The rerank_results.results are already sorted by relevance
                reranked_docs_content = [documents[result.index] for result in rerank_results.results]
                # print(f"Reranked {len(reranked_docs_content)} documents for query: '{query}'")
                return reranked_docs_content
            else:
                # print(f"Reranked {len(rerank_results.results)} document objects for query: '{query}'")
                return rerank_results.results # Return the full result objects

        except cohere.CohereAPIError as e:
            print(f"Cohere API error during reranking: {e}")
            # Fallback: return original documents or an empty list
            # For now, returning original documents might be safer for pipeline continuation
            # return documents 
            return [] # Or return empty to signal failure
        except Exception as e:
            print(f"An unexpected error occurred during reranking: {e}")
            return []

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # IMPORTANT: Set your COHERE_API_KEY environment variable for this example to run
    # export COHERE_API_KEY='your_cohere_api_key_here'
    
    try:
        reranker = Reranker()
        print("Reranker initialized successfully.")

        sample_query = "What are the benefits of eating apples?"
        sample_documents = [
            "Apples are a good source of fiber and vitamin C.",
            "Bananas are high in potassium.",
            "Eating apples daily can help with digestion and boost immunity.",
            "Oranges are known for their high vitamin C content.",
            "An apple a day keeps the doctor away is a famous saying."
        ]

        print(f"\nOriginal documents (count: {len(sample_documents)}):")
        for i, doc in enumerate(sample_documents):
            print(f"  {i+1}. {doc}")

        # Rerank and get top 3 document texts
        print(f"\nReranking for query: '{sample_query}', top_n=3")
        top_3_reranked_docs = reranker.rerank_documents(sample_query, sample_documents, top_n=3)
        if top_3_reranked_docs:
            print("Top 3 reranked documents:")
            for i, doc in enumerate(top_3_reranked_docs):
                print(f"  {i+1}. {doc}")
        else:
            print("Reranking returned no documents.")

        # Rerank and get all document texts
        print(f"\nReranking for query: '{sample_query}', all documents")
        all_reranked_docs = reranker.rerank_documents(sample_query, sample_documents)
        if all_reranked_docs:
            print("All reranked documents:")
            for i, doc in enumerate(all_reranked_docs):
                print(f"  {i+1}. {doc}")
        else:
            print("Reranking returned no documents.")

        # Rerank and get full result objects (including scores)
        print(f"\nReranking for query: '{sample_query}', top_n=3, return full objects")
        top_3_reranked_objects = reranker.rerank_documents(sample_query, sample_documents, top_n=3, return_documents=False)
        if top_3_reranked_objects:
            print("Top 3 reranked result objects:")
            for i, result in enumerate(top_3_reranked_objects):
                print(f"  {i+1}. Index: {result.index}, Score: {result.relevance_score:.4f}")
                # Access original document text via index if needed: sample_documents[result.index]
                print(f"     Document: {sample_documents[result.index]}")
        else:
            print("Reranking returned no result objects.")
            
        # Test with empty documents
        print("\nTesting with empty documents list...")
        empty_docs_result = reranker.rerank_documents(sample_query, [])
        print(f"Result for empty documents: {empty_docs_result}")

        # Test with empty query
        print("\nTesting with empty query...")
        empty_query_result = reranker.rerank_documents("", sample_documents)
        print(f"Result for empty query (should be original documents or empty based on implementation):")
        # for i, doc in enumerate(empty_query_result):
        #     print(f"  {i+1}. {doc}")

    except ValueError as e:
        print(f"Error initializing Reranker: {e}")
        print("Please ensure COHERE_API_KEY is set as an environment variable.")
    except Exception as e:
        print(f"An unexpected error occurred during Reranker example: {e}")