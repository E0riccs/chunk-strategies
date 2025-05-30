import requests
import os
import sys

class SiliconflowAPI:
    def __init__(self, model_id, url, api_key, model_name):
        self.model_id = model_id
        self.url = url
        self.api_key = api_key
        self.model_name = model_name

    def send_message(self, 
                    user_content, 
                    stream=False, max_tokens=512, thinking_budget=4096, min_p=0.05, stop=None, temperature=0.7, top_p=0.7, top_k=50, frequency_penalty=0.5, n=1, response_format={"type": "text"}, tools=None):
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

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        response = requests.request("POST", self.url, json=payload, headers=headers)
        return response.text