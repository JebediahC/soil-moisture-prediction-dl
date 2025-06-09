import utils
# import torch

config = utils.load_config()

def load_model(path):
    model = None
    print(f"Loading model from {path}...")
    if config["model"] == "empty_model":
        print("Using empty model as placeholder.")
    else:
        raise NotImplementedError("Model loading not implemented for this type.")
    print("Model loaded successfully.")
    return model