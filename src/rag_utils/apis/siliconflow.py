import requests
import json
from ..rag_model import APIResponseModel

class SiliconflowAPI:
    def __init__(self, model_id, url, api_key, model_name, **kwargs):
        self.model_id = model_id
        self.url = url
        self.api_key = api_key
        self.model_name = model_name
        self.kwargs = kwargs

    def get_embedding(self, 
                user_content):
        """
            Network request for embedding generation.
        """
        payload = {
            "model": self.model_name,
            "input": user_content,
            "encoding_format": "float"
        }

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.request("POST", self.url, json=payload, headers=headers)
        print("Embedding response with code: ", response.status_code)

        return response


    def rerank_documents(self,
                        query:str,
                        documents:list[str],
                        top_n:int = 6,
                        return_documents:bool=False,
                        max_chunks_per_doc:int=1024,
                        overlap_tokens:int=80):
        """
            Network request for document reranking.

            Args:
                query (str): The query to rerank documents for.
                documents (list of str): A list of documents (strings) to rerank.
                top_n (int): The number of reranked documents to return. 
                return_documents (bool): If false, the response does not include document text; if true, it includes the input document text.
                max_chunks_per_doc (int, optional): The maximum number of chunks per document.
                overlap_tokens (int, optional): The number of overlapping tokens between chunks.
        """
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": return_documents,
            "max_chunks_per_doc": max_chunks_per_doc,
            "overlap_tokens": overlap_tokens
        }

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.request("POST", self.url, json=payload, headers=headers)
        print("Rerank response with code: ", response.status_code)

        return response



    def embed_result_from_json(self, json_response):
        """
            Parse the JSON response from the embedding API.
            In this function, we do not extract the real response content (e.g. QAs) but only standardize the response format.
        """

        json_response = json_response.text 
        try:
            response_data = json.loads(json_response)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return None

        # Extract data safely using .get() method
        tokens_used = response_data.get('usage', {}).get('total_tokens')
        embedding_model = response_data.get('model', {})
        llm_platform = 'siliconflow'  # This is specific to this class
        
        data = response_data.get('data', [])
        if data and isinstance(data, list) and len(data) > 0:
            embedding = data[0].get('embedding', {})
        else:
            embedding = None
        
        # Use the standardized response model
        return APIResponseModel(
            tokens_used=tokens_used,
            model=embedding_model,
            api_platform=llm_platform,
            response_content=embedding
        )

    def rerank_answer_from_json(self, json_response):
        """
            Parse the JSON response from the reranking API.
            In this function, we do not extract the real response content (e.g. QAs) but only standardize the response format.
        """
        pass
