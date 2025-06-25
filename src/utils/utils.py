import yaml
from src.utils.logger import setup_logger

def load_yaml_config(file_path):
    logger = setup_logger(__name__)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Error: The file {file_path} was not found.")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        return None

def save_text_to_file(text, file_path):
    """Saves text content to a file."""
    logger = setup_logger(__name__)
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Successfully saved to {file_path}")
    except IOError as e:
        logger.error(f"Error saving file {file_path}: {e}")


if __name__ == '__main__':
    # Example usage (optional, for testing)
    test_yaml_path = '../config/file_types.yaml'
    config_data = load_yaml_config(test_yaml_path)
    if config_data:
        print("Loaded YAML config:", config_data)

    save_text_to_file("This is a test content.", "../results/test_output.txt")