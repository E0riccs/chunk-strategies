import os
import time

from src.utils.logger import setup_logger
from src.utils.utils import load_yaml_config
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from src.rag_utils.out_import import ChunkLoader


class SpliterFactory:
    def __init__(self):
        self.logger = setup_logger(__name__)

    def _simple_split(self, text, chunk_size, chunk_overlap):
        """A basic character-based splitter."""
        splitter = CharacterTextSplitter(
            separator="\n", # 次要分割依据
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=True,
            length_function=len,
            is_separator_regex=False, # 分割符是否是正则表达式
        )
        return splitter.split_text(text)

    def _recursive_character_text_split(self, text, chunk_size, chunk_overlap, separators=None):
        """Uses Langchain's RecursiveCharacterTextSplitter."""
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )
        return splitter.split_text(text)

    def create_spliter(self, method_name):
        if method_name == 'simple_split':
            return self._simple_split
        elif method_name == 'recursive_character_text_splitter':
            return self._recursive_character_text_split
        else:
            self.logger.error(f"Chunking method: {method_name} is not supported.")
            raise ValueError(f"Chunking method: {method_name} is not supported.")

class Chunker:
    def __init__(self, config_path='config/chunking_strategies.yaml'):
        self.logger = setup_logger(__name__)

        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_config_path = os.path.join(self.base_dir, config_path)
        self.strategies_config = load_yaml_config(abs_config_path)
        if not self.strategies_config or 'chunking_strategies' not in self.strategies_config:
            raise ValueError("Invalid or missing chunking strategies configuration.")

        self.spliter_factory = SpliterFactory()

    def get_strategy_details(self, strategy_name):
        """Retrieves details for a given chunking strategy name."""
        for strategy in self.strategies_config.get('chunking_strategies', []):
            if strategy.get('name') == strategy_name:
                return strategy
            
        self.logger.error(f"Error: Chunking strategy '{strategy_name}' not found in configuration.")
        return None
        

    def _chunk_load(self):
        """
            Load existing chunks from a source(different file type).
            Args:
                None
            Return:
                A tuple of (chunking_duration = 0, chunks(list)).
        """

        loader = ChunkLoader(self.strategy_details)
        chunks = loader.load()

        return 0, chunks

    
    def _chunk_split(self, text):
        """
            Split the given text into chunks using the specified strategy.
            Args:
                text: The text to be chunked.
            Return:
                A tuple of (chunking_duration, chunks(list)).
        """
        method_name = self.strategy_details.get('method')
        params = self.strategy_details.get('params', {})
        spliter = self.spliter_factory.create_spliter(method_name)
        
        start_time = time.time()
        res = spliter(text, **params)
        end_time = time.time()

        return end_time - start_time, res

    def chunk(self, text, strategy_name):
        """
            Make chunk results(Split or Load).
            Args:
                text: The text to be chunked.
                strategy_name: The name of the chunking strategy to use.
            Return:
                A tuple of (chunking_duration, chunks) from class method.
        """
        self.strategy_details = self.get_strategy_details(strategy_name)
        if not self.strategy_details:
            return None

        method_name = self.strategy_details.get('method')
        if method_name == 'outside':
            # outside 必备参数
            if self.strategy_details.get('source_type') is None:
                self.logger.error(f"Chunking strategy '{strategy_name}' need more params.")
                raise ValueError("Error: Chunking strategy '{strategy_name}' need more params.")
            return self._chunk_load()
        else:
            return self._chunk_split(text)
            

if __name__ == '__main__':
    # Example usage
    # This assumes the script is run from the project root or PYTHONPATH is set correctly
    logger = setup_logger(__name__)
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        config_file_path = os.path.join(project_root, 'config/chunking_strategies.yaml')
        
        chunker = Chunker(config_path=config_file_path)
        sample_text = "This is a sample text for chunking.\nIt has multiple lines.\nAnd some more content to make it long enough for splitting. We need to see how different strategies work. This is the first paragraph.\n\nThis is the second paragraph. It also contains several sentences. The goal is to test the chunking mechanisms effectively."

        logger.info("\nTesting simple_chunk_100_overlap_0...")
        chunks1 = chunker.chunk(sample_text, 'simple_chunk_100_overlap_0')
        if chunks1:
            logger.info(f"Strategy: simple_chunk_100_overlap_0, Chunks: {len(chunks1)}")
            for i, chunk_text in enumerate(chunks1):
                logger.info(f"  Chunk {i+1}: '{chunk_text[:50]}...' (Length: {len(chunk_text)})")
        
        logger.info("\nTesting recursive_char_split_150_overlap_15...")
        chunks2 = chunker.chunk(sample_text, 'recursive_char_split_150_overlap_15')
        if chunks2:
            logger.info(f"Strategy: recursive_char_split_150_overlap_15, Chunks: {len(chunks2)}")
            for i, chunk_text in enumerate(chunks2):
                logger.info(f"  Chunk {i+1}: '{chunk_text[:50]}...' (Length: {len(chunk_text)})")

        logger.info("\nTesting non-existent strategy...")
        chunks_non_existent = chunker.chunk(sample_text, 'non_existent_strategy')
        if not chunks_non_existent:
            logger.info("Correctly handled non-existent strategy.")
            
    except ValueError as e:
        logger.error(f"Initialization Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")