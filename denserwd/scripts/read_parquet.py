import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet("/mnt/storage/public_data/dataset/physical-intelligence/libero/data/chunk-000/episode_000325.parquet")

# 查看前几行
print(df.head())
print(df.columns)
print(len(df))