import yaml

def load_config():
    """Load configuration from YAML file."""
    try:
        with open("config/config.yaml", "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print("Configuration file not found.")
        return None
    except yaml.YAMLError as exc:
        print(f"Error in configuration file: {exc}")
        return None