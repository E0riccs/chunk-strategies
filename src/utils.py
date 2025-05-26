import yaml

def load_yaml_config(file_path):
    """Loads a YAML configuration file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        return None

def save_text_to_file(text, file_path):
    """Saves text content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Successfully saved to {file_path}")
    except IOError as e:
        print(f"Error saving file {file_path}: {e}")


if __name__ == '__main__':
    # Example usage (optional, for testing)
    test_yaml_path = '../config/file_types.yaml'
    config_data = load_yaml_config(test_yaml_path)
    if config_data:
        print("Loaded YAML config:", config_data)

    save_text_to_file("This is a test content.", "../results/test_output.txt")