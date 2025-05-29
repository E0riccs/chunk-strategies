import os
import argparse
from src.experiment_runner import ExperimentRunner
from src.llm_evaler import LLMEvaler

DEFAULT_EXPERIMENTS_CONFIG_PATH = 'config/experiments_to_run.yaml'
DEFAULT_LLM_CONFIG_PATH = 'config/llm.yaml'

def main():
    parser = argparse.ArgumentParser(description="Run text chunking experiments.")

    # 1. 读取参数
    parser.add_argument(
        '--run_all_from_config',
        type=str,
        metavar='CONFIG_FILE_PATH',
        help='Path to a YAML file defining a batch of experiments to run (e.g., config/experiments_to_run.yaml). '
             'If provided, --file_type and --strategy are ignored.'
    )

    parser.add_argument(
        '--api_key',
        type=str,
        default=OPENAI_API_KEY,
        help='OpenAI API key. Defaults to OPENAI_API_KEY environment variable.'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save experiment results. Default: results/'
    )

    args = parser.parse_args()

    if not args.api_key:
        print("Warning: OpenAI API key not provided. LLM evaluations will be skipped.")
        print("You can set the OPENAI_API_KEY environment variable or use the --api_key argument.")

    # 2. 针对原材料生成 QA 对
    llmer = LLMEvaler()
    
    abs_default_llm_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_LLM_CONFIG_PATH)
    
    if not os.path.exists(abs_default_llm_config_path):
        example_llm_config = {
            'models':[
                {
                    'model': 'model_large',
                    'model_name': 'gpt-3.5-turbo',
                    'api_key': 'sk-0000000000000000000000'
                },
                {
                    'model': 'model_medium',
                    'model_name': 'gpt-4o',
                    'api_key': 'sk-0000000000000000000000'
                }
            ]
        }
        import yaml
        try:
            with open(abs_default_llm_config_path, 'w', encoding='utf-8') as f_yaml:
                yaml.dump(example_llm_config, f_yaml, default_flow_style=False, sort_keys=False)
            print(f"Created default LLM config: {abs_default_llm_config_path}")
        except Exception as e:
            print(f"Error creating default {DEFAULT_LLM_CONFIG_PATH}: {e}")
            return # Exit if cannot create default config
    
    llmer._generate_qa_pairs()
    

    # 3. 运行实验
    runner = ExperimentRunner(
        file_types_config_path='config/file_types.yaml',
        chunking_strategies_config_path='config/chunking_strategies.yaml',
        results_dir=args.results_dir,
        openai_api_key=args.api_key
    )

    if args.run_all_from_config:
        print(f"Running all experiments from config file: {args.run_all_from_config}")
        runner.run_all_experiments_from_config(experiments_config_path=args.run_all_from_config)
    else:
        print("No specific experiment requested. Using/Creating a default 'experiments_to_run.yaml' and running it.")
        
        # Create a default experiments_to_run.yaml if it doesn't exist and run it
        # This makes it easier for the user to get started if they run main.py without args
        abs_default_exp_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_EXPERIMENTS_CONFIG_PATH)

        if not os.path.exists(abs_default_exp_config_path):
            example_experiments_config = {
                'experiments': [
                    {'file_type': 'chapter_text', 'chunking_strategy': 'simple_chunk_100_overlap_10'},
                    {'file_type': 'itemized_text', 'chunking_strategy': 'recursive_char_split_150_overlap_15'},
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
            import yaml
            try:
                with open(abs_default_exp_config_path, 'w', encoding='utf-8') as f_yaml:
                    yaml.dump(example_experiments_config, f_yaml, default_flow_style=False, sort_keys=False)
                print(f"Created default experiments config: {abs_default_exp_config_path}")
            except Exception as e:
                print(f"Error creating default {DEFAULT_EXPERIMENTS_CONFIG_PATH}: {e}")
                return # Exit if cannot create default config
        
        runner.run_all_experiments_from_config(experiments_config_path=DEFAULT_EXPERIMENTS_CONFIG_PATH)

    print("\nMain script execution finished.")

if __name__ == "__main__":
    main()