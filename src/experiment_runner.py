import os
import time
import datetime
import pandas as pd
from src.file_handler import FileHandler
from src.chunker import Chunker
from src.evaluator import Evaluator
from src.vector_store_handler import VectorStoreHandler
from src.llm_handler import LLM_handler # For RAG QA and evaluation
import uuid # For generating unique IDs for chunks
from src.utils import save_text_to_file, load_yaml_config

class ExperimentRunner:
    def __init__(self, 
                 file_types_config_path='config/file_types.yaml',
                 chunking_strategies_config_path='config/chunking_strategies.yaml',
                 results_dir='results',
                 eval_model_id='default_model',
                 vector_store_persist_dir='db/chroma_main_store',
                 vector_store_collection_name_prefix='rag_exp'):
        
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.file_handler = FileHandler(config_path=self._abs_path(file_types_config_path))
        self.chunker = Chunker(config_path=self._abs_path(chunking_strategies_config_path))
        # Determine the model for the Evaluator's LLM_handler instance
        # It can be the same as the general eval_model_id or a specific one if needed
        evaluator_llm_model_id = self.experiments_config.get('evaluator_llm_model', eval_model_id) 
        self.evaluator = Evaluator(model_id=evaluator_llm_model_id) 

        # LLM_handler for RAG QA and RAG evaluation
        # It uses its own config for models, potentially different from the Evaluator's LLM
        # config_path should be the directory containing llm_info.yaml and prompts.md
        llm_handler_config_dir = self._abs_path('config') 
        self.llm_handler = LLM_handler(model_id=eval_model_id, config_path=llm_handler_config_dir)

        # Initialize VectorStoreHandler - collection name will be dynamic per experiment
        # Default to SentenceTransformer embeddings if not specified
        embedding_function_name = self.experiments_config.get('embedding_function', 'sentence-transformers/all-MiniLM-L6-v2')
        self.vector_store_handler_prototype = VectorStoreHandler(embedding_function_name=embedding_function_name)
        self.current_vector_store_handler = None # Will be set per experiment run
        
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

        # 3. Initialize Vector Store for this specific experiment run
        # Sanitize names for file paths first
        safe_file_type_name = file_type_name.replace(' ', '_').lower()
        safe_strategy_name = chunking_strategy_name.replace(' ', '_').lower()

        # This creates a unique collection for each file_type + strategy combination to avoid interference
        # and allows for clean re-runs.
        collection_name_suffix = f"{safe_file_type_name}_{safe_strategy_name}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        # Sanitize collection_name_suffix further if needed, ChromaDB has restrictions.
        collection_name_suffix = collection_name_suffix.replace('-', '_') # Replace hyphens
        current_collection_name = f"{self.vector_store_collection_name_prefix}_{collection_name_suffix}"
        
        # Ensure collection name is valid for ChromaDB (e.g., length, characters)
        # A simple truncation and character replacement might be needed for very long names.
        current_collection_name = current_collection_name[:60] # Max length for collection name is 63
        current_collection_name = ''.join(c if c.isalnum() or c in ['_', '.'] else '_' for c in current_collection_name)
        if not current_collection_name[0].isalnum() or not current_collection_name[-1].isalnum():
             current_collection_name = 'c' + current_collection_name[1:-1] + 'c' # Ensure start/end are alphanumeric


        print(f"Initializing VectorStore for collection: {current_collection_name}")
        self.current_vector_store_handler = VectorStoreHandler(
            persist_directory=os.path.join(self.vector_store_base_persist_dir, current_collection_name), # Each experiment gets its own sub-folder
            collection_name=current_collection_name # Unique collection name
            # Potentially pass openai_api_key if using OpenAI embeddings from config
        )
        # Clear the collection if it somehow exists and we want a fresh start (optional)
        # self.current_vector_store_handler.clear_collection() 

        # 4. Add chunks to Vector Store
        print(f"Adding {len(chunks)} chunks to vector store...")
        chunk_metadatas = [
            {
                "source_file_type": file_type_name,
                "chunking_strategy": chunking_strategy_name,
                "chunk_index": i,
                "original_text_length": len(original_text)
            } for i in range(len(chunks))
        ]
        # Generate unique IDs for each chunk to ensure they can be individually referenced/updated if needed
        chunk_ids = [f"{current_collection_name}_chunk_{uuid.uuid4()}" for _ in range(len(chunks))]
        self.current_vector_store_handler.add_documents(chunks, metadatas=chunk_metadatas, ids=chunk_ids)
        print(f"Vector store now contains {self.current_vector_store_handler.get_collection_count()} documents.")

        # 5. RAG QA and Evaluation (New Step)
        # This part needs a predefined set of questions, or questions generated from the original text.
        # For now, let's assume a placeholder for questions.
        # These questions could be loaded from the file_type_details or a general config.
        test_questions_source = file_type_details.get('test_questions', [])
        test_questions_for_rag = [] # This will store list of dicts: {'question': ..., 'answer': ...}

        if isinstance(test_questions_source, str): # If it's a path to a QA file
            qa_file_path = self._abs_path(test_questions_source) # Ensure absolute path
            print(f"Attempting to load test questions from file: {qa_file_path}")
            loaded_qas = self.llm_handler.load_qa_pairs_from_file(qa_file_path)
            # loaded_qas is a list of {'question': ..., 'answer': ...}
            test_questions_for_rag = loaded_qas
        elif isinstance(test_questions_source, list):
            # Handles list of strings (questions only) or list of dicts (q/a pairs)
            for item in test_questions_source:
                if isinstance(item, str):
                    test_questions_for_rag.append({'question': item, 'answer': None}) # No reference answer
                elif isinstance(item, dict) and 'question' in item:
                    # Ensure 'answer' key exists, defaulting to None if not present
                    test_questions_for_rag.append({'question': item['question'], 'answer': item.get('answer')})
            print(f"Loaded test questions from config: {len(test_questions_for_rag)} questions")
        else:
            print(f"Warning: 'test_questions' format in file_types.yaml for {file_type_name} is not recognized or is empty. RAG QA will be skipped.")

        rag_results = []
        if self.current_vector_store_handler and test_questions_for_rag:
            print(f"\n--- Starting RAG QA for {len(test_questions_for_rag)} questions --- ")
            for q_data in test_questions_for_rag:
                question_text = q_data.get('question')
                reference_answer = q_data.get('answer') # Optional reference answer, might be None
                
                if not question_text:
                    print("Skipping empty or invalid question data.")
                    continue

                rag_output = self.llm_handler.answer_question_rag(
                    question_text=question_text,
                    vector_store_handler=self.current_vector_store_handler,
                    retrieval_n_results=strategy_details.get('rag_retrieval_n_results', 10),
                    reranker_top_n=strategy_details.get('rag_reranker_top_n', 3),
                    qa_model_id=self.experiments_config.get('rag_qa_model', 'default_model'),
                    vector_store_filter=None
                )
                
                # Evaluate the RAG answer using llm_handler
                rag_evaluation_metrics = self.llm_handler.evaluate_rag_answer(
                    question=question_text,
                    generated_answer=rag_output['final_answer'],
                    reference_answer=reference_answer, 
                    eval_model_id=self.experiments_config.get('rag_eval_model', 'default_model')
                )
                
                rag_results.append({
                    'question': question_text,
                    'reference_answer': reference_answer if reference_answer else 'N/A',
                    'generated_answer': rag_output['final_answer'],
                    'retrieved_docs_count': rag_output['retrieved_documents_count'],
                    'reranked_docs_count': rag_output['reranked_documents_count'],
                    'evaluation_score': rag_evaluation_metrics.get('score', 'N/A'),
                    'evaluation_reason': rag_evaluation_metrics.get('reason', 'N/A'),
                    'evaluation_critique': rag_evaluation_metrics.get('critique', 'N/A')
                })
            print("--- RAG QA Completed ---")
        else:
            if not test_questions_for_rag: # Corrected variable name
                print("Skipping RAG QA as no test questions are defined or loaded.")
            if not self.current_vector_store_handler:
                 print("Skipping RAG QA as vector store handler is not initialized.")

        # 6. Evaluate chunking using Evaluator
        print("\n--- Evaluating Chunks --- ")
        chunk_eval_metrics = self.evaluator.evaluate(original_text, chunks, chunking_duration)
        print("Chunk Evaluation Metrics (from Evaluator):")
        for key, value in chunk_eval_metrics.items():
            print(f"  {key}: {value}")

        # Combine metrics
        metrics = {
            **chunk_eval_metrics, # Unpack metrics from Evaluator
            # 'chunking_time_seconds': round(chunking_duration, 4), # This is already in chunk_eval_metrics as 'total_processing_time_seconds'
            # 'number_of_chunks': len(chunks), # This is already in chunk_eval_metrics
        }

        # 7. Save results
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
            'vector_store_collection': current_collection_name,
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