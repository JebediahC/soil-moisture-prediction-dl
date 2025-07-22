import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import json

from src.utils import SimpleLogger
import src.utils as utils

# from src.preprocess import Preprocessor

config = utils.load_config()

logger = SimpleLogger(name="Logger-main", log_path=config["log_path"])


VAR_MAP = {
    "swvl1": "swvl1",
    "ro": "ro",
    "e": "e",
    "tp": "tp"
}

CHANNELS = {
    "swvl1": 0,
    "ro": 1,
    "e": 2,
    "tp": 3
}

from src.visualize import ArrayInfoDisplayer

array_info_displayer = ArrayInfoDisplayer(logger=logger)

# shows the config we are using
logger.info("Using config:\n" + json.dumps(config, indent=2, ensure_ascii=False))


# --- STEP 1: LAT/LON and NetCDF info---
# Show NetCDF file format info
sample_nc = xr.open_dataset(os.path.join(config["raw_folder"], f"{config['months'][0]}.nc"))

lat = sample_nc['latitude'].values
lon = sample_nc['longitude'].values

print(sample_nc)
print("\nNetCDF file info:")
sample_nc.info()


# --- STEP 2: PROCESS & SAVE DAILY AVERAGED FILES ---
def process_month_to_daily_mean(month_str):
    input_path = os.path.join(config["raw_folder"], f"{month_str}.nc")
    output_path = os.path.join(config["intermediate_folder"], f"{month_str}.npy")
    if os.path.exists(output_path):
        logger.info(f"File {output_path} already exists, skipping processing.")
        return
    ds = xr.open_dataset(input_path)
    var_stack = []

    for var in config["use_channels"]:
        var_data = ds[VAR_MAP[var]].values  # (time, lat, lon)
        total_hours = var_data.shape[0]
        days = total_hours // 24
        daily_data = var_data[:days * 24].reshape(days, 24, *var_data.shape[1:]).mean(axis=1)  # (days, lat, lon)
        daily_data = np.nan_to_num(daily_data, nan=0.0)
        # 应用掩膜，海洋区域置零
        # daily_data = daily_data * land_mask[None, :, :]
        var_stack.append(daily_data)

    daily_array = np.stack(var_stack, axis=-1)  # (days, lat, lon, channels)
    array_info_displayer.print_info(daily_array, name=f"Processed daily_array for {month_str}")
    np.save(output_path, daily_array)
    log_msg = f"Saved daily mean to {output_path} shape={daily_array.shape}"
    logger.info(log_msg)


# check if precessed folder exists, if not create it
if not os.path.exists(config["intermediate_folder"]):
    raise FileNotFoundError(f"Intermediate folder {config['intermediate_folder']} does not exist. Please create it first.")

months = config["months"]
for month in months:
    process_month_to_daily_mean(month)


# --- STEP 3: LOAD ALL MONTHLY FILES & CONCAT ---
def load_combined_data():
    monthly_arrays = []
    for month in config["months"]:
        arr = np.load(os.path.join(config["intermediate_folder"], f"{month}.npy"))
        monthly_arrays.append(arr)
    final_array = np.concatenate(monthly_arrays, axis=0)  # concat over day axis
    print(f"Final combined shape: {final_array.shape}")  # (days, lat, lon, channels)
    return final_array

data_array = load_combined_data()  # (T, H, W, C)

# Step 3: 自动生成land_mask（所有通道都为0的像元视为海洋）
# land_mask shape: (H, W)，1为陆地，0为海洋
land_mask = (np.any(data_array != 0, axis=(0, 3))).astype(np.float32)
print(f"Auto-generated land_mask shape: {land_mask.shape}, 陆地像元数: {np.sum(land_mask)}")

# 可视化掩膜
plt.figure(figsize=(8, 6))
plt.imshow(land_mask, cmap='gray')
plt.title('Auto-generated Land Mask (1=Land, 0=Ocean)')
plt.xlabel('Longitude Index')
plt.ylabel('Latitude Index')
plt.colorbar(label='Mask Value')
plt.show()

# 降分辨率可视化
# downsampled_array = array_info_displayer.downsample_channels(data_array, target_hw=30)
# array_info_displayer.print_info(downsampled_array, name="Downsampled Array (T, H, W, C)")
# aframe = downsampled_array[100,:,:,0]


# --- STEP 4: Transform to Tensor ---
def prepare_tensor_data(data):
    # (T, H, W, C) -> (T, C, H, W)
    return np.transpose(data, (0, 3, 1, 2))

tensor_data = prepare_tensor_data(data_array)
print(f"tensor data shape: {tensor_data.shape}")  # (T, C, H, W)


# --- STEP 5: DATASET & DATALOADER Test ---

import matplotlib as mpl
mpl.rcParams['animation.embed_limit'] = 50  # 单位是MB，设置为50MB

from src.datasets import SoilDataset

dataset = SoilDataset(tensor_data, config["input_days"], config["predict_days"])

total = len(dataset)
train_size = int(0.7 * total)
val_size = int(0.2 * total)
test_size = total - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
print(f"Dataset split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)

# ---------------- visualization ----------------
from src.visualize import Visualizer
x_sample, y_sample = dataset[10]


print(f"Sample x shape: {x_sample.shape}, y shape: {y_sample.shape}")
visualizer = Visualizer(lat, lon)
x_to_map = np.transpose(x_sample[-1], (1, 2, 0))
print(f"x_to_map shape: {x_to_map.shape}")  # Should be (H, W, C)
visualizer.plot_variable_day(x_to_map, title="The first sample Day", channel_index=CHANNELS["swvl1"])
# visualizer.plot_tensor_sample(x_sample, y_sample)

from IPython.display import HTML

# Show animation inline for a specific channel (e.g., swvl1)

# Generate animations for different channels and display them separately

# anim_swvl1 = visualizer.animate_tensor_sample(x_sample, y_sample, channel_index=CHANNELS["swvl1"])

# anim_ro_in, anim_ro_out = visualizer.animate_tensor_sample(x_sample, y_sample, channel_index=CHANNELS["e"])

# Display each animation in its own output cell to avoid overlap
# display(HTML(anim_swvl1.to_jshtml()))
# display(HTML(anim_ro_in.to_jshtml()))
# display(HTML(anim_ro_out.to_jshtml()))


x, y = dataset.__getitem__(0)  # Test if dataset works
print(x.shape, y.shape)

import importlib
import src.visualize
importlib.reload(src.visualize)
from src.visualize import Visualizer

# Select only the first channel from x to match y's channel dimension
x_first_channel = x[:, 0:1, :, :]  # shape: (input_days, 1, H, W)
print(f"x_first_channel shape: {x_first_channel.shape}")  # Should be (input_days, 1, H, W)
xy_combined = torch.cat([x_first_channel, y], dim=0)
print(xy_combined.shape)


# --- STEP 7-1: TRAINING LOOP part 1 ---
from src.models import ConvLSTM, CNN3D_LSTM, CNN3D_LSTM_Lite, CNN3D_LSTM_Micro, get_model
from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用配置文件中的模型类型创建模型
model_type = config.get("model_type", "convlstm")  # 默认使用convlstm
print(f"Using model type: {model_type}")

if model_type.lower() == "convlstm":
    model = ConvLSTM(input_dim=len(config["use_channels"])).to(device)
elif model_type.lower() == "3dcnn_lstm":
    model = CNN3D_LSTM(input_dim=len(config["use_channels"])).to(device)
elif model_type.lower() == "3dcnn_lstm_lite":
    model = CNN3D_LSTM_Lite(input_dim=len(config["use_channels"])).to(device)
elif model_type.lower() == "3dcnn_lstm_micro":
    model = CNN3D_LSTM_Micro(input_dim=len(config["use_channels"])).to(device)
else:
    # 使用工厂函数创建模型
    model = get_model(model_type, input_dim=len(config["use_channels"])).to(device)
    logger.warning(f"Using factory function to create model of type: {model_type}")

logger.info(f"Created model of type: {model_type}")

# 1. 显示模型结构和参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {model}")
print(f"Total parameters: {total_params:,d} ({total_params/1e6:.2f}M)")
print(f"Model device: {device}")


import datetime

# --- STEP 7-2: TRAINING LOOP part 2 ---

run_name = config["run_name"]

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='none')  # 用于mask后再mean

# TensorBoard writer
writer = SummaryWriter(log_dir=f"runs/{model_type}_{run_name}")

# Early stopping参数
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# 获取陆地掩膜（假设land_mask已在前面定义，shape: (lat, lon)）
# mask shape: (1, 1, H, W)，与y/pred最后两个维度一致
land_mask_tensor = torch.tensor(land_mask, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)



for epoch in range(config["epochs"]):
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        # 3. 损失函数计算前应用掩膜
        mask = land_mask_tensor.expand_as(y)
        loss_map = criterion(pred, y) * mask
        loss = loss_map.sum() / mask.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_train_loss = epoch_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_train_loss, epoch)

    # 验证集loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val in DataLoader(val_set, batch_size=config["batch_size"]):
            x_val, y_val = x_val.to(device), y_val.to(device)
            pred_val = model(x_val)
            mask_val = land_mask_tensor.expand_as(y_val)
            loss_map_val = criterion(pred_val, y_val) * mask_val
            loss_val = loss_map_val.sum() / mask_val.sum()
            val_loss += loss_val.item()
    avg_val_loss = val_loss / len(val_set)
    writer.add_scalar('Loss/val', avg_val_loss, epoch)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 2. Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        save_path = os.path.join("params", f"{model_type}_best_{timestamp}_{run_name}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved to {save_path}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    # 定期保存检查点
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join("params", f"{model_type}_{timestamp}_{run_name}_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

writer.close()


def load_model(model, path="soil_conv_lstm.pth", device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model


def create_model_from_config(model_type=None, input_dim=None):
    """
    根据配置创建模型的便捷函数
    
    Args:
        model_type: 模型类型，如果为None则从config读取
        input_dim: 输入维度，如果为None则从config计算
    
    Returns:
        创建的模型实例
    """
    if model_type is None:
        model_type = config.get("model_type", "convlstm")
    if input_dim is None:
        input_dim = len(config["use_channels"])
    
    return get_model(model_type, input_dim)