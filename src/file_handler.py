from src.utils.utils import load_yaml_config
from src.utils.logger import setup_logger
import os

class FileHandler:
    def __init__(self, config_path='config/file_types.yaml'):
        self.logger = setup_logger(__name__)
        self.file_types_config = load_yaml_config(config_path)
        if not self.file_types_config or 'file_types' not in self.file_types_config:
            raise ValueError("Invalid or missing file types configuration.")
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_file_type_details(self, type_name):
        """Retrieves details for a given file type name."""
        for ft in self.file_types_config.get('file_types', []):
            if ft.get('name') == type_name:
                return ft
        self.logger.error(f"File type '{type_name}' not found in configuration.")
        return None

    def load_test_data(self, type_name):
        """
            Loads the test data content for a given file type name.
            None/Empty file is forbidden.
        """
        file_type_details = self.get_file_type_details(type_name)
        if file_type_details and 'test_file' in file_type_details:
            # Construct absolute path for test_file
            test_file_path = os.path.join(self.base_dir, file_type_details['test_file'])
            try:
                with open(test_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content
            except FileNotFoundError:
                self.logger.error(f"Test data file not found at {test_file_path}")
                return None
            except IOError as e:
                self.logger.error(f"Error reading test data file {test_file_path}: {e}")
                return None
        else:
            self.logger.error(f"Could not load test data for file type '{type_name}'. Details missing or invalid.")
            return None

    def get_data_file_name(self, file_type):
        """
            get and format the data file name for a given file type.
        """
        file_type_details = self.get_file_type_details(file_type)
        if file_type_details and 'test_file' in file_type_details:
            path = os.path.basename(file_type_details['test_file'])
            return os.path.splitext(path)[0]
        elif file_type == 'out_file':
            return 'out_file'
        else:
            self.logger.log(f"Could not get data file name for file type '{file_type}'. Details missing or invalid.")
            return None

if __name__ == '__main__':
    # Example usage
    # Adjust the path to config/file_types.yaml if running this script directly from src
    # This assumes the script is run from the project root or PYTHONPATH is set correctly
    try:
        # Correct path when running from project root: 'config/file_types.yaml'
        # If running from src/: '../config/file_types.yaml'
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_script_dir)
        config_file_path = os.path.join(project_root, 'config/file_types.yaml')

        handler = FileHandler(config_path=config_file_path)

        # Initialize logger
        logger = setup_logger(__name__)
        
        # Test getting file type details
        chapter_details = handler.get_file_type_details('chapter_text')
        if chapter_details:
            logger.info(f"Chapter Text Details: {chapter_details}")

        # Test loading data
        logger.info("\nLoading chapter text data...")
        chapter_data = handler.load_test_data('chapter_text')
        if chapter_data:
            logger.info(f"Successfully loaded data for chapter_text (first 100 chars):\n{chapter_data[:100]}...")

        logger.info("\nLoading itemized text data...")
        itemized_data = handler.load_test_data('itemized_text')
        if itemized_data:
            logger.info(f"Successfully loaded data for itemized_text (first 100 chars):\n{itemized_data[:100]}...")

        logger.info("\nLoading short plain text data...")
        short_data = handler.load_test_data('short_plain_text')
        if short_data:
            logger.info(f"Successfully loaded data for short_plain_text:\n{short_data}")
        
        logger.info("\nTesting non-existent file type...")
        non_existent_data = handler.load_test_data('non_existent_type')
        if not non_existent_data:
            logger.info("Correctly handled non-existent file type.")

    except ValueError as e:
        logger.error(f"Initialization Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")