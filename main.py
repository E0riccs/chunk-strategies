import argparse
from src.experiment_runner import ExperimentRunner
from src.llm_handler import LLM_handler
from src.utils.logger import setup_logger
from src.utils.init import initialize_project, DEFAULT_EXPERIMENTS_CONFIG_PATH

def main():
    parser = argparse.ArgumentParser(description="Run text chunking experiments.")
    logger = setup_logger(__name__)

    # 1. 读取参数
    parser.add_argument(
        '--run_all_from_config',
        type=str,
        metavar='CONFIG_FILE_PATH',
        help='Path to a YAML file defining a batch of experiments to run (e.g., config/experiments_to_run.yaml). '
             'If provided, --file_type and --strategy are ignored.'
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Directory to save experiment results. Default: results/'
    )
    # 如果已存在qa文件，重建/后续添加/跳过策略
    parser.add_argument(
        '--gen_qa',
        type=str,
        choices=['rebuild', 'append', 'skip'],
        default='skip',
        help='Strategy for generating QA pairs from raw materials. '
    )

    parser.add_argument(
        '--model_id',
        type=str,
        default='default_model',
        help='ID of the model to use for LLM calls. Default: default_model'
    )
    parser.add_argument(
        '--gen_qa_model_id',
        type=str,
        default='model_gen_qa2',
        help='ID of the model to use for LLM calls. Default: model_gen_qa2'
    )

    args = parser.parse_args()
    
    # 2. 初始化项目，创建必要的默认配置文件
    if not initialize_project():
        logger.error("Failed to initialize project.")
        return
        
    # 3. 针对原材料生成 QA 对s
    llmer_gen = LLM_handler(config_path= 'config', model_id = args.gen_qa_model_id)
    
    llmer_gen.generate_qa_pairs(build_strategy=args.gen_qa)
    

    # 4. 运行实验
    runner = ExperimentRunner(
        file_types_config_path='config/file_types.yaml',
        chunking_strategies_config_path='config/chunking_strategies.yaml',
        results_dir = args.results_dir,
        llm_config_path = 'config/llm_info.yaml'
    )

    if args.run_all_from_config:
        logger.info(f"Running all experiments from config file: {args.run_all_from_config}")
        runner.run_all_experiments_from_config(experiments_config_path=args.run_all_from_config)
    else:
        logger.info("No specific experiment requested. Using default 'experiments_to_run.yaml' and running it.")
        runner.run_all_experiments_from_config(experiments_config_path=DEFAULT_EXPERIMENTS_CONFIG_PATH)

    logger.info("\nMain script execution finished.")

if __name__ == "__main__":
    main()