import requests

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
            "encoding_format": "base64"
        }

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.request("POST", self.url, json=payload, headers=headers)
        print("Response with code: ", response.status_code)

        return response


    def rerank_documents(self):
        """
            Network request for document reranking.
        """
        pass

    def answer_from_json(self, json_response):
        """
            Parse the JSON response from the API.
        """
        pass
