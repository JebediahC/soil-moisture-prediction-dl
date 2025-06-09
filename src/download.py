import utils

config = utils.load_config()

def download():
    date = config["date"]
    print(f"Downloading data for {date}...")

    # download logic here

    print(f"Data for {date} downloaded successfully.")