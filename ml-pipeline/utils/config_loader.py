import yaml

class ConfigLoader:
    @staticmethod
    def load_config(config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)