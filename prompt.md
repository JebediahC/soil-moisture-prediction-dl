我们准备进行Soil Moisture Prediction系统的编写，请你首先帮我生成一个代码框架，然后我会引导你一步步完善这个系统。

这个系统由以下部分组成
- Raw Dataset Download
- Preprocess Dataset
- Virtualization
- Input trained models and data
- Prediction
- Evaluation

系统配置：
- 语言：python
- 可以单独运行其中一部分
- 

数据详情：
- Ingest ERA5-Land input data
- Layers: 4(volumetric soil water layer 1, total runoff, total evaporation, total precipitation)
- spatial resolution(0.1°)
- Input window 10 days
- format: NetCDF
- Region: Europe (35°N–72°N, 25°W–45°E)


## Raw Dataset Download

```pseudo
# Generate pseudo codes
```
参考[download data](#download-data-from-cdsclimate-data-store-api)

## Preprocess Dataset
```pseudo
# Generate pseudo codes
```
参考[proprocess](#preprocess-data)

## Virtualization
data format(after preprocessing)：
- shape: 370(latitude 37° and 0.1° resolution)×700(longitude 70° and 0.1° resolution)×4(4 parameters)×10(10 days input window and )

Display maps for each layer


## Input trained models and data
Input data format(after preprocessing)

Load model


## Prediction
input: 
data format(after preprocessing)：
- shape: 370(latitude 37° and 0.1° resolution)×700(longitude 70° and 0.1° resolution)×4(4 parameters)×10(10 days input window and )


## Evaluation
use this Evaluation Metrics to evaluate the accuracy:
- RMSE
- R2
- MAE
- MSE

display maps to virtualize the layer

## Reference

### Download data from CDS(Climate Data Store) API

```python
import cdsapi

dataset = "reanalysis-era5-land"
request = {
    "variable": [
        "volumetric_soil_water_layer_1",
        "runoff",
        "total_evaporation",
        "total_precipitation"
    ],
    "year": "2025",
    "month": "05",
    "day": [
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [72, -25, 35, 45]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
```

### Preprocess data

```python
import os
os.environ['PROJ_LIB'] = r"E:\anaconda3\envs\era5env\Library\share\proj"

import xarray as xr
import rioxarray
import numpy as np

input_folder = r"F:\ESPACE\seminar\data"
output_folder = r"F:\ESPACE\seminar\tif_monthly"
os.makedirs(output_folder, exist_ok=True)

# 遍历每月 .nc 文件
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".nc"):
        print(f"\n📦 Processing {filename} ...")
        file_path = os.path.join(input_folder, filename)
        ds = xr.open_dataset(file_path)
        if 'valid_time' in ds.dims:
            ds = ds.rename({'valid_time': 'time'})
        # 日尺度变量处理
        sm = ds['swvl1'].resample(time='1D').mean()
        precip = ds['tp'].resample(time='1D').sum()
        evap = ds['e'].resample(time='1D').sum()
        runoff = ds['ro'].resample(time='1D').sum()

        # 要输出的变量字典
        variables = {
            'soil': sm,
            'precip': precip,
            'evap': evap,
            'runoff': runoff
        }

        for varname, var in variables.items():
            # 生成多波段 tif：每个 time 为一个波段
            bands = []
            for t in var.time:
                da = var.sel(time=t)
                da.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
                da.rio.write_crs("EPSG:4326", inplace=True)
                bands.append(da.values[np.newaxis, :, :])  # 保留波段维度

            # 合并为 3D ndarray：[bands, y, x]
            data_3d = np.concatenate(bands, axis=0)

            # 创建 DataArray 写入 tif
            stacked = xr.DataArray(
                data_3d,
                dims=("band", "latitude", "longitude"),
                coords={
                    "band": np.arange(1, len(bands) + 1),
                    "latitude": da.latitude,
                    "longitude": da.longitude
                }
            )
            stacked.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
            stacked.rio.write_crs("EPSG:4326", inplace=True)

            # 构建输出路径
            base_name = filename.replace(".nc", "")  # e.g. 2020-1
            out_path = os.path.join(output_folder, f"{varname}_{base_name}.tif")
            stacked.rio.to_raster(out_path)
            print(f"✅ Saved {out_path}")
```