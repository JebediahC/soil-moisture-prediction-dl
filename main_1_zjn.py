
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
