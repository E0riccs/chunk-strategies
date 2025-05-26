import os
import argparse
from src.experiment_runner import ExperimentRunner
from src.utils import load_yaml_config # For potentially loading API key or other main configs

# It's good practice to allow API key to be set via environment variable
# or passed as an argument, or even from a main config file (not implemented here for simplicity)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    parser = argparse.ArgumentParser(description="Run text chunking experiments.")

    # 手动指定单一测试项目
    parser.add_argument(
        '--file_type',
        type=str,
        help='Specific file type name to process (must be defined in config/file_types.yaml).'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        help='Specific chunking strategy name to use (must be defined in config/chunking_strategies.yaml).'
    )

    # 全部测试项目
    parser.add_argument(
        '--run_all_from_config',
        type=str,
        metavar='CONFIG_FILE_PATH',
        help='Path to a YAML file defining a batch of experiments to run (e.g., config/experiments_to_run.yaml). '
             'If provided, --file_type and --strategy are ignored.'
    )

    # 其他参数
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

    runner = ExperimentRunner(
        file_types_config_path='config/file_types.yaml',
        chunking_strategies_config_path='config/chunking_strategies.yaml',
        results_dir=args.results_dir,
        openai_api_key=args.api_key
    )

    if args.run_all_from_config:
        print(f"Running all experiments from config file: {args.run_all_from_config}")
        runner.run_all_experiments_from_config(experiments_config_path=args.run_all_from_config)
    elif args.file_type and args.strategy:
        print(f"Running single experiment: File Type='{args.file_type}', Strategy='{args.strategy}'")
        runner.run_experiment(file_type_name=args.file_type, chunking_strategy_name=args.strategy)
        runner.save_all_results_summary() # Save summary even for a single run
    else:
        # 未指定测试项目，使用默认测试项（若无则创建）
        print("No specific experiment requested. Creating a default 'experiments_to_run.yaml' and running it.")
        print("Use --file_type and --strategy for a single run, or --run_all_from_config for batch runs.")
        
        # Create a default experiments_to_run.yaml if it doesn't exist and run it
        # This makes it easier for the user to get started if they run main.py without args
        default_experiments_config_path = 'config/experiments_to_run.yaml'
        abs_default_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), default_experiments_config_path)

        if not os.path.exists(abs_default_config_path):
            example_experiments_config = {
                'experiments': [
                    {'file_type': 'chapter_text', 'chunking_strategy': 'simple_chunk_100_overlap_0'},
                    {'file_type': 'itemized_text', 'chunking_strategy': 'recursive_char_split_150_overlap_15'},
                ]
            }
            import yaml
            try:
                with open(abs_default_config_path, 'w', encoding='utf-8') as f_yaml:
                    yaml.dump(example_experiments_config, f_yaml, default_flow_style=False, sort_keys=False)
                print(f"Created default experiments config: {abs_default_config_path}")
            except Exception as e:
                print(f"Error creating default {default_experiments_config_path}: {e}")
                return # Exit if cannot create default config
        
        runner.run_all_experiments_from_config(experiments_config_path=default_experiments_config_path)

    print("\nMain script execution finished.")
    print(f"Check the '{os.path.abspath(args.results_dir)}' directory for output files.")

if __name__ == "__main__":
    main()