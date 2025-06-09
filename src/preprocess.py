import utils

config = utils.load_config()

def run_preprocessing():
    
    input_path = config["raw_path"]
    output_path = config["processed_path"]
    print(f"Preprocessing data from {input_path} to {output_path}...")
    # Preprocessing logic here
    print(f"Data preprocessing completed. Processed data saved to {output_path}.")
