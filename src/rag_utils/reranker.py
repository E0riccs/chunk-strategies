import os
from src.rag_utils.api_factory import APIFactory
from src.logger import setup_logger

class Reranker:
    def __init__(self, 
                 reranker_model_name='default_reranker'):
        """
        Initializes the Reranker.

        Args:
            reranker_model_name (str): The reranking model to use.
        """
        self.logger = setup_logger(__name__)
        api_factory = APIFactory(config_path='config/llm_info.yaml', model_id=reranker_model_name, api_type='reranker')
        api = api_factory.create_api()
        
        self.api = api

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
            self.logger.warning("No documents provided for reranking.")
            return []
        if not query:
            self.logger.warning("No query provided for reranking. Returning original documents.")
            return documents # Or an empty list, depending on desired behavior

        try:
            self.logger.info(f"Reranking {len(documents)} documents for query: '{query}'")
            rerank_results = self.api.rerank_documents(
                query=query,
                documents=documents,
                top_n=top_n,
                return_documents=False # Get full result objects to access scores and indices
            )
            
            if return_documents:
                # Sort the original documents based on the reranked order
                # The rerank_results.results are already sorted by relevance
                reranked_docs_content = [documents[result.index] for result in rerank_results.results]
                self.logger.info(f"Reranked {len(reranked_docs_content)} documents for query: '{query}'")
                return reranked_docs_content
            else:
                self.logger.info(f"Reranked {len(rerank_results.results)} document objects for query: '{query}'")
                return rerank_results.results # Return the full result objects
            
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during reranking: {e}")
            return []

# Example Usage (for testing purposes)
if __name__ == '__main__':  
    # IMPORTANT: Set your COHERE_API_KEY environment variable for this example to run
    # export COHERE_API_KEY='your_cohere_api_key_here'
    
    try:
        reranker = Reranker()
        logger = setup_logger(__name__)
        logger.info("Reranker initialized successfully.")

        sample_query = "What are the benefits of eating apples?"
        sample_documents = [
            "Apples are a good source of fiber and vitamin C.",
            "Bananas are high in potassium.",
            "Eating apples daily can help with digestion and boost immunity.",
            "Oranges are known for their high vitamin C content.",
            "An apple a day keeps the doctor away is a famous saying."
        ]

        logger.info(f"\nOriginal documents (count: {len(sample_documents)}):")
        for i, doc in enumerate(sample_documents):
            logger.info(f"  {i+1}. {doc}")

        # Rerank and get top 3 document texts
        logger.info(f"\nReranking for query: '{sample_query}', top_n=3")
        top_3_reranked_docs = reranker.rerank_documents(sample_query, sample_documents, top_n=3)
        if top_3_reranked_docs:
            logger.info("Top 3 reranked documents:")
            for i, doc in enumerate(top_3_reranked_docs):
                logger.info(f"  {i+1}. {doc}")
        else:
            logger.info("Reranking returned no documents.")

        # Rerank and get all document texts
        logger.info(f"\nReranking for query: '{sample_query}', all documents")
        all_reranked_docs = reranker.rerank_documents(sample_query, sample_documents)
        if all_reranked_docs:
            logger.info("All reranked documents:")
            for i, doc in enumerate(all_reranked_docs):
                logger.info(f"  {i+1}. {doc}")
        else:
            logger.info("Reranking returned no documents.")

        # Rerank and get full result objects (including scores)
        logger.info(f"\nReranking for query: '{sample_query}', top_n=3, return full objects")
        top_3_reranked_objects = reranker.rerank_documents(sample_query, sample_documents, top_n=3, return_documents=False)
        if top_3_reranked_objects:
            logger.info("Top 3 reranked result objects:")
            for i, result in enumerate(top_3_reranked_objects):
                logger.info(f"  {i+1}. Index: {result.index}, Score: {result.relevance_score:.4f}")
                # Access original document text via index if needed: sample_documents[result.index]
                logger.info(f"     Document: {sample_documents[result.index]}")
        else:
            logger.info("Reranking returned no result objects.")
            
        # Test with empty documents
        logger.info("\nTesting with empty documents list...")
        empty_docs_result = reranker.rerank_documents(sample_query, [])
        logger.info(f"Result for empty documents: {empty_docs_result}")

        # Test with empty query
        logger.info("\nTesting with empty query...")
        empty_query_result = reranker.rerank_documents("", sample_documents)
        logger.info(f"Result for empty query (should be original documents or empty based on implementation):")
        # for i, doc in enumerate(empty_query_result):
        #     print(f"  {i+1}. {doc}")

    except ValueError as e:
        logger.error(f"Error initializing Reranker: {e}")
        logger.error("Please ensure COHERE_API_KEY is set as an environment variable.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Reranker example: {e}")