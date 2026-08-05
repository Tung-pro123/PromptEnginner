import os
import yaml

def load_yaml(file_path):
    """
    Loads a YAML configuration file safely.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found at: {file_path}")
    
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file: {exc}")
