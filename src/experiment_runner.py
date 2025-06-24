import os
import time
import datetime
import yaml
import re
import pandas as pd
import random

from src.chunker import Chunker
from src.eval_utils.evaluator import Evaluator
from src.file_handler import FileHandler
from src.rag_handler import RAGHandler
from src.eval_utils.eval_save import EvalSaver

from src.utils import load_yaml_config
from src.logger import setup_logger

class ExperimentRunner:
    def __init__(self, 
                 file_types_config_path='config/file_types.yaml',
                 chunking_strategies_config_path='config/chunking_strategies.yaml',
                 results_dir='results',
                 llm_config_path='config/llm_info.yaml',
                 vector_store_base_persist_dir='db/chroma_db',
                 vector_store_collection_name_prefix='experiment'
                 ):
        self.logger = setup_logger(__name__)
        
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
            self.logger.info(f"Created results directory: {self.results_dir}")

    def _abs_path(self, relative_path):
        """Converts a path relative to project root to an absolute path."""
        return os.path.join(self.base_dir, relative_path)

    def run_experiment(self, setting):
        """Runs a single experiment for a given file type and chunking strategy."""
        # 0. config
        file_type_name = setting.get('file_type')
        chunking_strategy_name = setting.get('chunking_strategy')        

        if_rerank = setting.get('rerank', False)
        retrieval_n_results = setting.get('retrieval_n_results', 8)
        if if_rerank:
            reranker_method_name = setting.get('reranker_method', '')
            reranker_top_n = setting.get('reranker_top_n', 4)
        else:
            reranker_method_name = 'Null'
            reranker_top_n = 0

        self.logger.info(f"\n--- Running Experiment ---")
        self.logger.info(f"File Type: {file_type_name}")
        self.logger.info(f"Chunking Strategy: {chunking_strategy_name}")

        # 1. Load original text
        original_text = self.file_handler.load_test_data(file_type_name)
        if original_text is None:
            self.logger.error(f"Failed to load data for {file_type_name}. Skipping experiment.")
            return None
        
        file_type_details = self.file_handler.get_file_type_details(file_type_name)
        strategy_details = self.chunker.get_strategy_details(chunking_strategy_name)

        if not file_type_details or not strategy_details:
            self.logger.error("Invalid file type or strategy name. Skipping experiment.")
            return None

        # 2. Chunk the text
        start_chunk_time = time.time()
        chunks = self.chunker.chunk(original_text, chunking_strategy_name)
        end_chunk_time = time.time()
        chunking_duration = end_chunk_time - start_chunk_time

        if chunks is None:
            self.logger.error(f"Failed to chunk text using {chunking_strategy_name}. Skipping experiment.")
            return None
        
        self.logger.info(f"Successfully chunked text into {len(chunks)} chunks in {chunking_duration:.4f}s.")

        
        # 3. RAGHandler添加文档到向量数据库
        self.rag_handler.setup_vector_store(
            file_type_name=file_type_name,
            chunking_strategy_name=chunking_strategy_name,
            use_api_embeddings=self.use_api_embeddings,
            embedding_model_name=self.embedding_model_name,
            api_platform=self.embedding_api_platform
        )
        self.rag_handler.add_documents_to_vector_store(
            chunks=chunks,
            original_text_length=len(original_text)
        )

        # 4. RAG QA and Evaluation
        # Load QAs from QA file
        qa_file_path = 'data/qa_pairs/' + self.file_handler.get_data_file_name(file_type_name) + '_qa_pairs.txt'
        qa_file_path = self._abs_path(qa_file_path)
        self.logger.info(f"Attempting to load QAs from file: {qa_file_path}")
        # Use the RAG LLM handler to load QA pairs
        test_questions_for_rag = self.load_qa_pairs_from_file(qa_file_path, 2) # test
        # test_questions_for_rag = self.load_qa_pairs_from_file(qa_file_path)


        # 5. Answer the question with rag
        related_documents = []
        rag_results = []
        for group in test_questions_for_rag:
            doc, ans_rag = self.rag_handler.answer_question(
                question_text=group['question'],
                retrieval_n_results = retrieval_n_results,
                reranker_top_n=reranker_top_n,
                # vector_store_filter=None,
                qa_model_id=self.rag_model_id
            )
            related_documents.append(doc)
            rag_results.append(ans_rag.model_dump())


        # 6. Evaluate chunking using Evaluator
        self.logger.info("\n--- Evaluating Chunks --- ")
        self.evaluator = Evaluator(model_id=self.eval_model_id, exp_setting=setting) 
        chunk_eval_metrics = self.evaluator.evaluate(rag_results, test_questions_for_rag, related_documents)

        # 7. save results
        result_data_to_save = {
            'file_type_name': file_type_name,
            'chunking_strategy_name': chunking_strategy_name,
            'chunking_method': strategy_details.get('method', 'N/A'),
            'chunk_size': strategy_details.get('params', {}).get('chunk_size'),
            'chunk_overlap': strategy_details.get('params', {}).get('chunk_overlap'),
            'chunk_times_s': chunking_duration,
            'chunk_retrieval_nums': retrieval_n_results,
            'reranker_model': reranker_method_name,
            'reranker_nums': reranker_top_n,
            'eval_metrics': chunk_eval_metrics,
            'chunking_params': str(strategy_details.get('params', {})),
        }
        self.eval_saver.add_one_result(**result_data_to_save)

    def run_all_experiments_from_config(self, experiments_config_path='config/experiments_to_run.yaml'):
        """Runs all experiments defined in a configuration file."""
        abs_experiments_config_path = self._abs_path(experiments_config_path)
        self.experiments_config = load_yaml_config(abs_experiments_config_path)
        if not self.experiments_config or 'experiments' not in self.experiments_config:
            self.logger.error(f"Error: Experiments configuration file not found or invalid at {abs_experiments_config_path}")
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

        # eval saver
        self.eval_saver = EvalSaver(results_dir=self.results_dir, experiments_config=self.experiments_config)

        self.logger.info(f"\n=== Starting Batch of {len(self.experiments_config.get('experiments', []))} Experiments from {abs_experiments_config_path} ===")
        for exp_setting in self.experiments_config.get('experiments', []):
            self.run_experiment(exp_setting)
        
        self.logger.info("\n=== All Configured Experiments Completed ===")

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
            self.logger.error(f"Error: QA file not found at {file_path}")
            return []
        except Exception as e:
            self.logger.error(f"Error reading QA file {file_path}: {e}")
            return []
        
        if not qa_pairs:
            self.logger.warning(f"No QA pairs loaded from {file_path}. Ensure format is 'Q: ...' and 'A: ...'")
        else:
            self.logger.info(f"Loaded {len(qa_pairs)} QA pairs from {file_path}")

        if len(qa_pairs) > qa_nums:
            # random select qa_pairs
            qa_pairs = random.sample(qa_pairs, qa_nums)
            self.logger.info(f"Selected {qa_nums} QA pairs from {file_path}")
        else:
            self.logger.info(f"Selected all QA pairs from {file_path}")

        return qa_pairs