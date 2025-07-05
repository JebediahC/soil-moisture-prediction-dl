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

class Logger:
    """Simple logger class to handle logging messages."""
    
    def __init__(self, name="Logger", log_path=None):
        self.name = name
        self.log_path = log_path
        if log_path:
            self.log_file = open(log_path, "a")
        else:
            self.log_file = None

    def _gettime(self):
        """Get the current time formatted as a string with milliseconds."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _print_to_console(self, message):
        print(f"[{self._gettime()}] {message}")
    
    def _write_to_file(self, message):
        if self.log_file:
            self.log_file.write(f"[{self._gettime()}] {message}\n")
            self.log_file.flush()
        else:
            self._print_to_console(message)

    def close(self):
        """Close the log file if it was opened."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

    def info(self, message):
        self._write_to_file(f"[INFO] {message}")
    
    def error(self, message):
        self._write_to_file(f"[ERROR] {message}")

    def debug(self, message):
        self._write_to_file(f"[DEBUG] {message}")

class SimpleLogger(Logger):
    """A simple logger that extends the Logger class."""

    def __init__(self, name="SimpleLogger", log_path=None):
        super().__init__(name, log_path=log_path)

    def log(self, message):
        self.info(message)

class Config:
    """Configuration class to hold parameters."""
    
    def __init__(self, config):
        self.raw_path = config.get("raw_path", "data/raw")
        self.processed_path = config.get("processed_path", "data/processed")
        self.model_path = config.get("model_path", "models")
        self.results_path = config.get("results_path", "results")
        self.per_year_outputs_path = config.get("per_year_outputs_path", "data/per_year_outputs")
        self.model = config.get("model", "empty_model")