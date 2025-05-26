from src.utils import load_yaml_config
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import os

ok_method_name = ['simple_split', 'recursive_character_text_splitter']

class SpliterFactory:
    def _simple_split(self, text, chunk_size, chunk_overlap):
        """A basic character-based splitter."""
        splitter = CharacterTextSplitter(
            separator = "\n\n", # Default, can be parameterized if needed
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
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
        if method_name not in ok_method_name:
            raise ValueError(f"Invalid chunking method: {method_name}, please use one of {ok_method_name}")
        if method_name == 'simple_split':
            return self._simple_split
        elif method_name == 'recursive_character_text_splitter':
            return self._recursive_character_text_split

class Chunker:
    def __init__(self, config_path='config/chunking_strategies.yaml'):
        # Construct absolute path for config_path relative to project root
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_config_path = os.path.join(self.base_dir, config_path)
        self.strategies_config = load_yaml_config(abs_config_path)
        if not self.strategies_config or 'chunking_strategies' not in self.strategies_config:
            raise ValueError("Invalid or missing chunking strategies configuration.")

        self.spliter_factory = SpliterFactory()

    def get_strategy_details(self, strategy_name):
        dic = {}
        """Retrieves details for a given chunking strategy name."""
        for strategy in self.strategies_config.get('chunking_strategies', []):
            if strategy.get('name') == strategy_name:
                dic['method'] = strategy.get('method')
                dic['params'] = strategy.get('params', {})
                return dic
            
        print(f"Error: Chunking strategy '{strategy_name}' not found in configuration.")
        return None

    def chunk(self, text, strategy_name):
        """Chunks the given text using the specified strategy."""
        strategy_details = self.get_strategy_details(strategy_name)
        if not strategy_details:
            return None

        method_name = strategy_details.get('method')
        params = strategy_details.get('params', {})

        spliter = self.spliter_factory.create_spliter(method_name)

        if method_name in ok_method_name:
            return spliter(text, **params)
        else:
            print(f"Error: Unknown chunking method '{method_name}' for strategy '{strategy_name}'.")
            return None

if __name__ == '__main__':
    # Example usage
    # This assumes the script is run from the project root or PYTHONPATH is set correctly
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        config_file_path = os.path.join(project_root, 'config/chunking_strategies.yaml')
        
        chunker = Chunker(config_path=config_file_path)
        sample_text = "This is a sample text for chunking.\nIt has multiple lines.\nAnd some more content to make it long enough for splitting. We need to see how different strategies work. This is the first paragraph.\n\nThis is the second paragraph. It also contains several sentences. The goal is to test the chunking mechanisms effectively."

        print("\nTesting simple_chunk_100_overlap_0...")
        chunks1 = chunker.chunk(sample_text, 'simple_chunk_100_overlap_0')
        if chunks1:
            print(f"Strategy: simple_chunk_100_overlap_0, Chunks: {len(chunks1)}")
            for i, chunk_text in enumerate(chunks1):
                print(f"  Chunk {i+1}: '{chunk_text[:50]}...' (Length: {len(chunk_text)})")
        
        print("\nTesting recursive_char_split_150_overlap_15...")
        chunks2 = chunker.chunk(sample_text, 'recursive_char_split_150_overlap_15')
        if chunks2:
            print(f"Strategy: recursive_char_split_150_overlap_15, Chunks: {len(chunks2)}")
            for i, chunk_text in enumerate(chunks2):
                print(f"  Chunk {i+1}: '{chunk_text[:50]}...' (Length: {len(chunk_text)})")

        print("\nTesting non-existent strategy...")
        chunks_non_existent = chunker.chunk(sample_text, 'non_existent_strategy')
        if not chunks_non_existent:
            print("Correctly handled non-existent strategy.")
            
    except ValueError as e:
        print(f"Initialization Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")