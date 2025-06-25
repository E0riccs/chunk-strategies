import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
import sys
sys.path.insert(0, project_root)

from src.rag_utils.apis import siliconflow
from src.utils.utils import load_yaml_config
from src.utils.logger import setup_logger

class APIFactory:
    def __init__(self, model_id, config_path='config/llm_info.yaml',api_type='embedding'):
        """
        Initialize API client.

        Args:
            config_path (str): Path to the configuration file containing model information.
            model_id (str): Model ID to use for the API.
            api_type (str): Type of API to use for the API in ['embedding', 'reranker']
        Raises:
            ValueError: If the given model_id is not found in the configuration file.
        """

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        abs_config_path = os.path.join(project_root, config_path)

        self.logger = setup_logger(__name__)
        self.model_id = model_id
        self.config = load_yaml_config(abs_config_path)
        self.model_config = self._get_model_config(model_id, api_type)
        if not self.model_config:
            self.logger.error(f"Model with id '{model_id}' not found in {abs_config_path}")
            raise ValueError(f"Model with id '{model_id}' not found in {abs_config_path}")

        self.url = self.model_config.get('end_point')
        self.api_key = self.model_config.get('api_key')
        self.api_platform = self.model_config.get('platform')
        self.model_name = self.model_config.get('model_name')
    
    def _get_model_config(self, model_id, api_type):
        for model_info in self.config.get(api_type+'_models', []):
            if model_info.get('model') == model_id:
                return model_info
        return None
    
    def create_api(self, **kwargs):
        if self.api_platform == 'siliconflow':
            return siliconflow.SiliconflowAPI(self.model_id, self.url, self.api_key, self.model_name, **kwargs)
        else:
            self.logger.error(f"Unknown api_platform: {self.api_platform}")
            raise ValueError(f"Unknown api_platform: {self.api_platform}")


if __name__ == '__main__':
    # Example usage:
    try: 
        # Create an instance of the API client
        # Adjust the config_path to be relative to the project root when running this script directly.
        json_schema = {}

        api = APIFactory(config_path='config/llm_info.yaml', model_id='default_embedding', api_type='embedding').create_api()
        
        # Make a chat request
        user_query = "这是一个测试消息！"
        response = api.get_embedding(user_query)
        response = api.embed_answer_from_json(response)

        print(response)


        api = APIFactory(config_path='config/llm_info.yaml', model_id='default_reranker', api_type='reranker').create_api()
        user_query = "天气如何"
        documents = ["今天天气真好","阳光明媚","适合出去散步","希望下午也能保持这样的好天气"]
        response = api.rerank_documents(user_query, documents)
        response = api.rerank_answer_from_json(response)

        logger.info(response)


    except Exception as e:
        print(f"An error occurred: {e}")