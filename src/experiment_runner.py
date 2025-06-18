import os
import time
import datetime
import yaml
import re
import pandas as pd
import random

from src.chunker import Chunker
from src.evaluator import Evaluator
from src.file_handler import FileHandler
from src.rag_handler import RAGHandler
from src.utils import load_yaml_config

class ExperimentRunner:
    def __init__(self, 
                 file_types_config_path='config/file_types.yaml',
                 chunking_strategies_config_path='config/chunking_strategies.yaml',
                 results_dir='results',
                 llm_config_path='config/llm_info.yaml',
                 vector_store_base_persist_dir='db/chroma_db',
                 vector_store_collection_name_prefix='experiment'
                 ):
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.file_handler = FileHandler(config_path=self._abs_path(file_types_config_path))
        self.chunker = Chunker(config_path=self._abs_path(chunking_strategies_config_path))
        self.rag_handler = RAGHandler(vector_store_base_persist_dir = vector_store_base_persist_dir)

        # config
        self.experiments_config = {} # Initialize as empty dict, will be populated
        self.llm_config = load_yaml_config(self._abs_path(llm_config_path))
        self.text_config = load_yaml_config(self._abs_path(file_types_config_path))
        
        self.results_dir = self._abs_path(results_dir)
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory: {self.results_dir}")

        self.all_results_data = []

    def _abs_path(self, relative_path):
        """Converts a path relative to project root to an absolute path."""
        return os.path.join(self.base_dir, relative_path)

    def run_experiment(self, setting):
        """Runs a single experiment for a given file type and chunking strategy."""
        # 0. config
        file_type_name = setting.get('file_type')
        chunking_strategy_name = setting.get('chunking_strategy')        

        if_rerank = setting.get('rerank', False)
        if if_rerank:
            reranker_method_name = setting.get('reranker_method', '')

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

        
        # 3. RAGHandler添加文档到向量数据库
        self.rag_handler._setup_vector_store(
            file_type_name=file_type_name,
            chunking_strategy_name=chunking_strategy_name,
            use_api_embeddings=self.use_api_embeddings,
            embedding_model_name=self.embedding_model_name,
            api_platform=self.embedding_api_platform
        )
        # self.rag_handler.add_documents_to_vector_store(
        #     chunks=chunks,
        #     original_text_length=len(original_text)
        # )

        # 4. RAG QA and Evaluation
        # Load QAs from QA file
        qa_file_path = 'data/qa_pairs/' + self.file_handler.get_data_file_name(file_type_name) + '_qa_pairs.txt'
        test_questions_for_rag = [] # List of dicts: {'question': ..., 'answer': ...}
        qa_file_path = self._abs_path(qa_file_path)
        print(f"Attempting to load QAs from file: {qa_file_path}")
        # Use the RAG LLM handler to load QA pairs
        # test_questions_for_rag = self.load_qa_pairs_from_file(qa_file_path)
        test_questions_for_rag = self.load_qa_pairs_from_file(qa_file_path, 1) # test


        # 5. Answer the question with rag
        related_documents = []
        rag_results = []
        for group in test_questions_for_rag:
            doc, ans_rag = self.rag_handler.answer_question(
                question_text=group['question'],
                retrieval_n_results=10,
                reranker_top_n=3,
                # vector_store_filter=None,
                qa_model_id=self.rag_model_id
            )
            related_documents.append(doc)
            rag_results.append(ans_rag.model_dump())


        # 6. Evaluate chunking using Evaluator
        print("\n--- Evaluating Chunks --- ")
        self.evaluator = Evaluator(model_id=self.eval_model_id, exp_setting=setting) 
        chunk_eval_metrics = self.evaluator.evaluate(rag_results, test_questions_for_rag, related_documents)
        print("Chunk Evaluation Metrics (from Evaluator):")
        for key, value in chunk_eval_metrics.items():
            print(f"  {key}: {value}")

        # Combine metrics
        metrics = {
            **chunk_eval_metrics, # Unpack metrics from Evaluator
            # 'chunking_time_seconds': round(chunking_duration, 4), # This is already in chunk_eval_metrics as 'total_processing_time_seconds'
            # 'number_of_chunks': len(chunks), # This is already in chunk_eval_metrics
        }

        # 6. Save results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Chunked output is now in the vector store, not saved as a separate text file.
        # We can log the collection name or path if needed.
        # chunks_filename = f"{safe_file_type_name}_{safe_strategy_name}_chunks.txt"
        # chunks_filepath = os.path.join(self.results_dir, 'chunks', chunks_filename)
        # save_text_to_file("\n\n---\n".join(chunks), chunks_filepath) # No longer saving chunks to text file

        # Prepare data for overall results table
        result_entry = {
            'timestamp': timestamp,
            'file_type_name': file_type_name,
            'file_type_description': file_type_details.get('description', 'N/A'),
            'test_file_path': file_type_details.get('test_file', 'N/A'),
            'chunking_strategy_name': chunking_strategy_name,
            'chunking_method': strategy_details.get('method', 'N/A'),
            'chunking_params': str(strategy_details.get('params', {})),
            'vector_store_collection': self.rag_handler.collection_name,
            **metrics, # Unpack combined chunking metrics
            'rag_qa_results': rag_results 
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

        # the llm model used in all experiments
        self.llm_list = {}
        for model_dict in self.experiments_config.get('llm', []):
            self.llm_list.update({k: v for k,v in model_dict.items()})

        # embedding model detail
        if 'embedding_model' in self.llm_list:
            self.use_api_embeddings = True
            self.embedding_model_name = self.llm_list.get('embedding_model')
            for model_dicts in self.llm_config.get('embedding_models', []):
                if model_dicts.get('model') == self.embedding_model_name:
                    self.embedding_api_platform = model_dicts.get('platform')
                    break

        # rag model detail
        self.rag_model_id = self.llm_list.get('rag_model')
        for model_dicts in self.llm_config.get('llm_models', []):
            if model_dicts.get('model') == self.rag_model_id:
                self.rag_api_platform = model_dicts.get('platform')
                break
        
        # eval model detail
        self.eval_model_id = self.llm_list.get('eval_model')
        for model_dicts in self.llm_config.get('llm_models', []):
            if model_dicts.get('model') == self.eval_model_id:
                self.eval_api_platform = model_dicts.get('platform')
                break
        

        print(f"\n=== Starting Batch of Experiments from {abs_experiments_config_path} ===")
        for exp_setting in self.experiments_config.get('experiments', []):
            self.run_experiment(exp_setting)
        
        self.save_all_results_summary()
        print("\n=== All Configured Experiments Completed ===")

    def save_all_results_summary(self):
        """Saves all accumulated experiment results to a CSV file."""
        if not self.all_results_data:
            print("No experiment results to save.")
            return

        df = pd.DataFrame(self.all_results_data)
        
        # Define a more flexible column order, or let pandas decide
        # If 'columns_order' is in config, try to use it, but be robust if new columns exist
        configured_cols_order = self.experiments_config.get('results', {}).get('columns_order', [])
        if configured_cols_order:
            # Create a list of columns present in the DataFrame, ordered by configured_cols_order
            # then add any remaining columns from the DataFrame that were not in the config
            final_cols = [col for col in configured_cols_order if col in df.columns]
            remaining_cols = [col for col in df.columns if col not in final_cols]
            df = df[final_cols + remaining_cols]
        else:
            # Default ordering if no config provided (pandas default or sort alphabetically)
            df = df[sorted(df.columns)] # Example: sort alphabetically for consistency

        # Handle complex columns like 'rag_qa_results' which is a list of dicts.
        # Pandas will store it as a string representation of the list by default in CSV.
        # For better analysis, one might flatten this or save it to a separate linked file.
        # For now, we keep it as is for simplicity of this step.

        summary_filename = f"summary/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_filepath = os.path.join(self.results_dir, summary_filename)
        
        try:
            df.to_csv(summary_filepath, index=False, encoding='utf-8')
            print(f"\nSuccessfully saved experiment summary to: {summary_filepath}")
        except IOError as e:
            print(f"Error saving summary CSV file {summary_filepath}: {e}")

    def load_qa_pairs_from_file(self, file_path, qa_nums=8):
        """
        Loads QA pairs from a specified text file.
        Each Q and A should be on separate lines, prefixed with "Q: " and "A: ".
        Args:
            file_path (str): The absolute path to the QA file.
            qa_nums (int): The number of QA pairs to load.
        Returns:
            list: A list of dictionaries, where each dictionary is a QA pair {'question': str, 'answer': str}.
        """
        qa_pairs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_q = None
            for line in lines:
                line = line.strip()
                if line.startswith("Q:"):
                    current_q = line[3:].strip()
                elif line.startswith("A:") and current_q:
                    answer = line[3:].strip()

                    # Convert string representation of list to actual list
                    if answer.startswith("[") and answer.endswith("]"):
                        answer_text = answer[1:-1]
                        
                        # Find all quoted strings or items without quotes
                        items = re.findall(r'\'([^\']*?)\'', answer_text)
                        answer = items
                    else:
                        # If not in list format, treat as a single item
                        answer = [answer]

                    qa_pairs.append({"question": current_q, "answer": answer})
                    current_q = None # Reset for the next pair
                # Blank lines or other lines are ignored
        except FileNotFoundError:
            print(f"Error: QA file not found at {file_path}")
            return []
        except Exception as e:
            print(f"Error reading QA file {file_path}: {e}")
            return []
        
        if not qa_pairs:
            print(f"No QA pairs loaded from {file_path}. Ensure format is 'Q: ...' and 'A: ...'")
        else:
            print(f"Loaded {len(qa_pairs)} QA pairs from {file_path}")

        if len(qa_pairs) > qa_nums:
            # random select qa_pairs
            qa_pairs = random.sample(qa_pairs, qa_nums)
            print(f"Selected {qa_nums} QA pairs from {file_path}")
        else:
            print(f"Selected all QA pairs from {file_path}")

        return qa_pairs


if __name__ == '__main__':
    # IMPORTANT: Set your OPENAI_API_KEY environment variable for LLM evaluation
    # export OPENAI_API_KEY='your_api_key_here'
    # Or pass it directly: runner = ExperimentRunner(eval_model_id='your_chosen_llm_for_eval')
    # Ensure OPENAI_API_KEY and COHERE_API_KEY are set in your environment if using them.

    print("Initializing Experiment Runner...")
    # Example: Specify a model for evaluation/QA if different from default
    # runner = ExperimentRunner(eval_model_id='gpt-4-turbo-preview') 
    runner = ExperimentRunner()
    # You might also want to configure vector_store_persist_dir and collection_name_prefix here
    # runner = ExperimentRunner(vector_store_persist_dir='custom_db_path', vector_store_collection_name_prefix='my_exp')

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
            # {'file_type': 'itemized_text', 'chunking_strategy': 'simple_chunk_200_overlap_20'},
            # {'file_type': 'short_plain_text', 'chunking_strategy': 'simple_chunk_100_overlap_0'}
        ],
        'rag_qa_model': 'gpt-3.5-turbo', # Specify the model for RAG question answering
        'rag_eval_model': 'gpt-4',     # Specify the model for RAG evaluation
        'results': {
            'columns_order': [ # Example of preferred column order
                'timestamp', 'file_type_name', 'chunking_strategy_name',
                'number_of_chunks', 'chunking_time_seconds', 
                'vector_store_collection', 'rag_qa_results' # rag_qa_results will be complex
            ]
        }
    }
    # Note: To run RAG QA, 'file_types.yaml' should have 'test_questions' for each file_type.
    # Example for 'chapter_text' in 'file_types.yaml':
    # chapter_text:
    #   description: "A standard chapter from a book."
    #   test_file: "data/chapter_example.txt"
    #   test_questions: 
    #     - "What is the main topic of this chapter?"
    #     - { question: "Summarize the key arguments.", answer: "The key arguments are X, Y, and Z." } # Optional reference answer
    config_dir = os.path.join(runner.base_dir, 'config')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    experiments_yaml_path = os.path.join(config_dir, 'experiments_to_run.yaml')

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