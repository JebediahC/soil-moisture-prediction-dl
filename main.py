## entrance of the program

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

from src.utils import SimpleLogger
import src.utils as utils

# from src.preprocess import Preprocessor

config = utils.load_config()

logger = SimpleLogger(name="Logger-main", log_path="logs/logger_main_private7.5.log")

# Add memory_profiler for memory usage
# try:
#     from memory_profiler import memory_usage
# except ImportError:
#     memory_usage = None
#     logger.warning("Warning: memory_profiler is not installed. Install it with 'pip install memory_profiler' for memory usage tracking.")

VAR_MAP = {
    "swvl1": "swvl1",
    "ro": "ro",
    "e": "e",
    "tp": "tp"
}

# --- STEP 1: PROCESS & SAVE DAILY AVERAGED FILES ---


def process_month_to_daily_mean(month_str):
    input_path = os.path.join(config["raw_folder"], f"{month_str}.nc")
    output_path = os.path.join(config["intermediate_folder"], f"{month_str}.npy")
    ds = xr.open_dataset(input_path)
    var_stack = []

    for var in config["use_channels"]:
        var_data = ds[VAR_MAP[var]].values  # (time, lat, lon)
        total_hours = var_data.shape[0]
        days = total_hours // 24
        daily_data = var_data[:days * 24].reshape(days, 24, *var_data.shape[1:]).mean(axis=1)  # (days, lat, lon)
        daily_data = np.nan_to_num(daily_data, nan=0.0)
        var_stack.append(daily_data)

    daily_array = np.stack(var_stack, axis=-1)  # (days, lat, lon, channels)
    # daily_array = np.transpose(daily_array, (1, 2, 0, 3))  # to (lat, lon, days, channels)
    np.save(output_path, daily_array)
    log_msg = f"Saved daily mean to {output_path} shape={daily_array.shape}"
    logger.info(log_msg)
if __name__ == "__main__":
    months = config["months"]
    for month in months:
        process_month_to_daily_mean(month)

