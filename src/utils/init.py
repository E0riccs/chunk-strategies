import os
import yaml
from src.utils.logger import setup_logger

DEFAULT_EXPERIMENTS_CONFIG_PATH = 'config/experiments_to_run.yaml'
DEFAULT_LLM_CONFIG_PATH = 'config/llm_info.yaml'

def create_default_llm_config(logger):
    """
    创建默认的LLM配置文件
    """
    llm_config_path = DEFAULT_LLM_CONFIG_PATH
    if not os.path.exists(llm_config_path):
        example_llm_config = {
            'models':[
                {
                    'model': 'custom_model_name1',
                    'model_name': 'gpt-3.5-turbo',
                    'platform': 'openai',
                    'description': 'free.',
                    'api_key': 'your_api_key_00000000000000',
                    'end_pointy': 'https://api.openai.com/v1/chat/completions'
                },
                {
                    'model': 'custom_model_name2',
                    'model_name': 'openai',
                    'platform': 'siliconflow',
                    'description': 'not free. 1.89¥/1M tokens',
                    'api_key': 'your_api_key_00000000000000',
                    'end_pointy': 'https://api.openai.com/v1/chat/completions'
                }
            ]
        }
        try:
            with open(llm_config_path, 'w', encoding='utf-8') as f_yaml:
                yaml.dump(example_llm_config, f_yaml, default_flow_style=False, sort_keys=False)
            logger.info(f"Created default LLM config: {llm_config_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating default {DEFAULT_LLM_CONFIG_PATH}: {e}")
            return False
    return True

def create_default_experiments_config(logger):
    """
    创建默认的实验配置文件
    """
    exp_config_path = DEFAULT_EXPERIMENTS_CONFIG_PATH

    if not os.path.exists(exp_config_path):
        example_experiments_config = {
            'experiments': [
                {
                    'file_type': 'chapter_text', 
                    'chunking_strategy': 'simple_chunk_100_overlap_10',
                    'rerank': False
                },
                {
                    'file_type': 'itemized_text', 
                    'chunking_strategy': 'recursive_char_split_150_overlap_15',
                    'rerank': False
                },
            ],
            'results': {
                'columns_order': [
                    'timestamp', 'file_type_name', 'chunking_strategy_name', 
                    'total_processing_time_seconds', 'llm_evaluation_score_1_to_5', 
                    'avg_cosine_similarity_chunks_vs_original', 'number_of_chunks',
                    'chunking_method', 'chunking_params', 'file_type_description', 
                    'test_file_path', 'chunks_output_file', 'evaluation_module_runtime_seconds'
                ]
            }
        }
        try:
            with open(exp_config_path, 'w', encoding='utf-8') as f_yaml:
                yaml.dump(example_experiments_config, f_yaml, default_flow_style=False, sort_keys=False)
            logger.info(f"Created default experiments config: {exp_config_path}")
            return True
        except Exception as e:
            logger.error(f"Error creating default {DEFAULT_EXPERIMENTS_CONFIG_PATH}: {e}")
            return False
    return True

def check_required_configs(logger):
    """
    检查所有必需的配置文件是否存在
    """
    required_configs = [
        'config/chunking_strategies.yaml',
        'config/file_types.yaml',
        'config/prompts.md'
    ]
    all_exist = True
    for config_file in required_configs:
        if not os.path.exists(config_file):
            logger.error(f"Missing required config file: {config_file}")
            all_exist = False
    return all_exist

def initialize_project():
    """
    初始化项目，创建必要的默认配置文件
    """
    logger = setup_logger(__name__)

    # 检查必需的配置文件
    if not check_required_configs(logger):
        return False
    
    # 创建默认配置文件
    if not create_default_llm_config(logger):
        return False
    
    if not create_default_experiments_config(logger):
        return False
    
    return True