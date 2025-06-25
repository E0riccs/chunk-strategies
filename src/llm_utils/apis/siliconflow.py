import requests
import json
from src.utils.logger import setup_logger
from ..llm_model import LLMResponseModel

class SiliconflowAPI:
    def __init__(self, model_id, url, api_key, model_name, **kwargs):
        self.model_id = model_id
        self.url = url
        self.api_key = api_key
        self.model_name = model_name
        self.kwargs = kwargs
        self.logger = setup_logger(__name__)

    def send_message(self, 
                    user_content, 
                    stream=False, 
                    max_tokens=8192,  # The maximum number of tokens to generate.
                    thinking_budget=4096,   # Maximum number of tokens for chain-of-thought output. This field applies to all Reasoning models.
                    min_p=0.05, 
                    stop=None,  # Up to 4 sequences where the API will stop generating further tokens.
                    temperature=0.7, 
                    top_p=0.7, 
                    top_k=50, 
                    frequency_penalty=0.5, 
                    n=1, 
                    response_format={"type": "text"}, 
                    # response_format={"type": "json_object"}
                    tools=None):
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            "stream": stream,
            "max_tokens": max_tokens,
            "thinking_budget": thinking_budget,
            "min_p": min_p,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "n": n,
            "response_format": response_format
        }
        if tools:
            payload["tools"] = tools
        if self.kwargs.get('json_schema'):
            payload['response_format'] = self.kwargs['json_schema']

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.request("POST", self.url, json=payload, headers=headers)
        self.logger.info(f"Response with code: {response.status_code}")

        return response


    def answer_from_json(self, json_response):
        """
            Extract the llm answer from the json response.
            In this function, we do not extract the real answer content (e.g. QAs) but only standardize the answer format.
        """
        json_response = json_response.text 
        try:
            response_data = json.loads(json_response)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding JSON: {e}")
            return None

        # Extract data safely using .get() method
        tokens_used = response_data.get('usage', {}).get('total_tokens')
        gen_model = response_data.get('model')
        llm_platform = 'siliconflow'  # This is specific to this class
        
        choices = response_data.get('choices', [])
        if choices and isinstance(choices, list) and len(choices) > 0:
            message = choices[0].get('message', {})
            ans_content = message.get('content')
            thinking_content = message.get('reasoning_content')
        else:
            ans_content = None
            thinking_content = None
        
        # Use the standardized response model
        response = LLMResponseModel(
            tokens_used=tokens_used,
            gen_model=gen_model,
            llm_platform=llm_platform,
            ans_content=ans_content,
            thinking_content=thinking_content
        )

        return response