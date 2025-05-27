import os
import time
import datetime
import pandas as pd
from src.file_handler import FileHandler
from src.chunker import Chunker
from src.evaluator import Evaluator
from src.utils import save_text_to_file, load_yaml_config

class ExperimentRunner:
    def __init__(self, 
                 file_types_config_path='config/file_types.yaml',
                 chunking_strategies_config_path='config/chunking_strategies.yaml',
                 results_dir='results',
                 openai_api_key=None):
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.file_handler = FileHandler(config_path=self._abs_path(file_types_config_path))
        self.chunker = Chunker(config_path=self._abs_path(chunking_strategies_config_path))
        self.evaluator = Evaluator(llm_api_key=openai_api_key)
        
        self.results_dir = self._abs_path(results_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory: {self.results_dir}")

        self.experiments_config = None

        self.all_results_data = []

    def _abs_path(self, relative_path):
        """Converts a path relative to project root to an absolute path."""
        return os.path.join(self.base_dir, relative_path)

    def run_experiment(self, file_type_name, chunking_strategy_name):
        """Runs a single experiment for a given file type and chunking strategy."""
        print(f"\n--- Running Experiment ---")
        print(f"File Type: {file_type_name}")
        print(f"Chunking Strategy: {chunking_strategy_name}")

        # 1. Load original text
        original_text = self.file_handler.load_test_data(file_type_name)
        if original_text is None:
            print(f"Failed to load data for {file_type_name}. Skipping experiment.")
            return None
        
        file_type_details = self.file_handler.get_file_type_details(file_type_name)
        strategy_details = self.chunker.get_strategy_details(chunking_strategy_name)

        if not file_type_details or not strategy_details:
            print("Invalid file type or strategy name. Skipping experiment.")
            return None

        # 2. Chunk the text
        start_chunk_time = time.time()
        chunks = self.chunker.chunk(original_text, chunking_strategy_name)
        end_chunk_time = time.time()
        chunking_duration = end_chunk_time - start_chunk_time

        if chunks is None:
            print(f"Failed to chunk text using {chunking_strategy_name}. Skipping experiment.")
            return None
        
        print(f"Successfully chunked text into {len(chunks)} chunks in {chunking_duration:.4f}s.")

        # 3. Evaluate the chunks
        # The 'processing_time' for evaluation refers to the chunking time itself
        metrics = self.evaluator.evaluate(original_text, chunks, chunking_duration)
        print("Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

        # 4. Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Sanitize names for file paths
        safe_file_type_name = file_type_name.replace(' ', '_').lower()
        safe_strategy_name = chunking_strategy_name.replace(' ', '_').lower()

        # Save chunked output
        chunks_filename = f"{safe_file_type_name}_{safe_strategy_name}_chunks.txt"
        chunks_filepath = os.path.join(self.results_dir, 'chunks', chunks_filename)
        save_text_to_file("\n\n---\n\n".join(chunks), chunks_filepath)

        # Prepare data for overall results table
        result_entry = {
            'timestamp': timestamp,
            'file_type_name': file_type_name,
            'file_type_description': file_type_details.get('description', 'N/A'),
            'test_file_path': file_type_details.get('test_file', 'N/A'),
            'chunking_strategy_name': chunking_strategy_name,
            'chunking_method': strategy_details.get('method', 'N/A'),
            'chunking_params': str(strategy_details.get('params', {})),
            'chunks_output_file': chunks_filename,
            **metrics # Unpack all metrics into the dictionary
        }
        self.all_results_data.append(result_entry)
        print(f"Experiment completed for {file_type_name} with {chunking_strategy_name}.")
        return result_entry

    def run_all_experiments_from_config(self, experiments_config_path='config/experiments_to_run.yaml'):
        """Runs all experiments defined in a configuration file."""
        abs_experiments_config_path = self._abs_path(experiments_config_path)
        self.experiments_config = load_yaml_config(abs_experiments_config_path)
        
        if not self.experiments_config or 'experiments' not in self.experiments_config:
            print(f"Error: Experiments configuration file not found or invalid at {abs_experiments_config_path}")
            return

        print(f"\n=== Starting Batch of Experiments from {abs_experiments_config_path} ===")
        for exp_setting in self.experiments_config.get('experiments', []):
            file_type = exp_setting.get('file_type')
            strategy = exp_setting.get('chunking_strategy')
            if file_type and strategy:
                self.run_experiment(file_type, strategy)
            else:
                print(f"Warning: Invalid experiment setting in config: {exp_setting}. Skipping.")
        
        self.save_all_results_summary()
        print("\n=== All Configured Experiments Completed ===")

    def save_all_results_summary(self):
        """Saves all accumulated experiment results to a CSV file."""
        if not self.all_results_data:
            print("No experiment results to save.")
            return

        df = pd.DataFrame(self.all_results_data)
        
        cols_order = self.experiments_config.get('results', {}).get('columns_order', [])
        # Ensure all expected columns are present, add if missing (e.g. if an eval step failed)
        for col in cols_order:
            if col not in df.columns:
                df[col] = None # or np.nan
        df = df[cols_order]

        summary_filename = f"summary/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_filepath = os.path.join(self.results_dir, summary_filename)
        
        try:
            df.to_csv(summary_filepath, index=False, encoding='utf-8')
            print(f"\nSuccessfully saved experiment summary to: {summary_filepath}")
        except IOError as e:
            print(f"Error saving summary CSV file {summary_filepath}: {e}")

if __name__ == '__main__':
    # IMPORTANT: Set your OPENAI_API_KEY environment variable for LLM evaluation
    # export OPENAI_API_KEY='your_api_key_here'
    # Or pass it directly: runner = ExperimentRunner(openai_api_key='YOUR_KEY')

    print("Initializing Experiment Runner...")
    runner = ExperimentRunner()

    # --- Option 1: Run individual experiments --- 
    # print("\n--- Running Single Experiment Example ---")
    # runner.run_experiment(file_type_name='chapter_text', 
    #                       chunking_strategy_name='simple_chunk_100_overlap_0')
    # runner.run_experiment(file_type_name='itemized_text', 
    #                       chunking_strategy_name='recursive_char_split_150_overlap_15')
    # runner.save_all_results_summary() # Save summary after manual runs

    # --- Option 2: Run experiments from a config file --- 
    # First, create an example experiments_to_run.yaml in the config directory:
    example_experiments_config = {
        'experiments': [
            {'file_type': 'chapter_text', 'chunking_strategy': 'simple_chunk_100_overlap_10'},
            {'file_type': 'chapter_text', 'chunking_strategy': 'recursive_char_split_150_overlap_15'},
            {'file_type': 'itemized_text', 'chunking_strategy': 'simple_chunk_200_overlap_20'},
            {'file_type': 'short_plain_text', 'chunking_strategy': 'simple_chunk_100_overlap_0'}
        ]
    }
    config_dir = os.path.join(runner.base_dir, 'config')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    experiments_yaml_path = os.path.join(config_dir, 'experiments_to_run.yaml')
    import yaml
    try:
        with open(experiments_yaml_path, 'w', encoding='utf-8') as f_yaml:
            yaml.dump(example_experiments_config, f_yaml, default_flow_style=False, sort_keys=False)
        print(f"Created example experiments config: {experiments_yaml_path}")
    except Exception as e:
        print(f"Error creating example experiments_to_run.yaml: {e}")

    # Now run from this config file
    print("\n--- Running Experiments from Config File --- ")
    runner.run_all_experiments_from_config(experiments_config_path='config/experiments_to_run.yaml')

    print("\nScript finished. Check the 'results' directory.")
    print("Note: If LLM evaluation shows 'N/A' or errors, ensure OPENAI_API_KEY is correctly set.")