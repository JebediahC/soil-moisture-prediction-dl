import os
import pandas as pd
import yaml
# import pyarraw.parquet as pq

config = yaml.safe_load(open("config/config.yaml"))

per_year_outputs_path = config["per_year_outputs_path"]
parquet_files = [f for f in os.listdir(per_year_outputs_path) if f.endswith('.parquet')]
print("parquet file lists: ", parquet_files)

read_file_path = os.path.join(per_year_outputs_path, parquet_files[0])
print("read: ", read_file_path)

input("press enter to read the selected parquet file...")

# 读取 parquet 文件
df = pd.read_parquet(per_year_outputs_path)

# 查看前几行数据
print(df.head())
# 查看所有列名和数据类型
print(df.dtypes)

# 查看总行数和内存使用
print(df.info())

# 查看描述统计（单位是否合适）
print(df.describe())
# 唯一日期数量
print(f"共包含 {df['date'].nunique()} 个日期")

# 每月条数分布
print(df.groupby('month').size())

# 每天多少个网格点
print(df.groupby('date').size().head())
print(f"纬度范围: {df['latitude'].min()} ~ {df['latitude'].max()}")
print(f"经度范围: {df['longitude'].min()} ~ {df['longitude'].max()}")

print(df.isnull().sum())

