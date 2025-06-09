import os
import xarray as xr
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow  # 如果没有报错，说明安装成功

# 设置路径
data_folder = r"F:\ESPACE\seminar\data"
output_folder = r"F:\ESPACE\seminar\per_year_outputs"
os.makedirs(output_folder, exist_ok=True)

# 只处理 2020 年所有 .nc 文件
file_list = sorted([
    os.path.join(data_folder, f)
    for f in os.listdir(data_folder)
    if f.endswith('.nc') and f.startswith('2024')
])

# 加速读取多个 NetCDF 文件并按时间合并
ds_all = xr.open_mfdataset(file_list, combine='by_coords', parallel=True)

# 获取所有唯一日期
time_values = pd.to_datetime(ds_all['valid_time'].values)
unique_days = sorted(set([t.date() for t in time_values if t.year == 2024]))

# 存储每日数据
all_days_data = []

for day in tqdm(unique_days, desc="Processing each day in 2024"):
    day_slice = ds_all.sel(valid_time=slice(f"{day} 00:00", f"{day} 23:59"))

    # 聚合：单位转换并保留维度
    swvl1_day = day_slice['swvl1'].mean(dim='valid_time', skipna=True)
    tp_day = day_slice['tp'].sum(dim='valid_time', skipna=True) * 1000
    e_day = day_slice['e'].sum(dim='valid_time', skipna=True) * 1000
    ro_day = day_slice['ro'].sum(dim='valid_time', skipna=True) * 1000

    # 合并为 DataFrame
    ds_day = xr.Dataset({
        'swvl1_m3m3': swvl1_day,
        'tp_mm_day': tp_day,
        'e_mm_day': e_day,
        'ro_mm_day': ro_day
    })

    df_day = ds_day.to_dataframe().reset_index()
    df_day = df_day[df_day['swvl1_m3m3'] > 0].dropna()

    # 添加日期信息
    df_day['date'] = str(day)
    df_day['year'] = day.year
    df_day['month'] = day.month
    df_day['doy'] = day.timetuple().tm_yday

    all_days_data.append(df_day)

# 拼接全年所有网格点数据
df_2020 = pd.concat(all_days_data, ignore_index=True)

# 保存为 Parquet 格式（压缩，列式存储）
output_parquet = os.path.join(output_folder, "daily_pixel_data_2024.parquet")
df_2020.to_parquet(output_parquet, index=False)

print(f"✅ 已保存至: {output_parquet}")

