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









