import pandas as pd
import os

DATA_PATH = "data/hpc/job_table.parquet"

if not os.path.exists(DATA_PATH):
    print(f"Dataset not found: {DATA_PATH}")
    exit(1)

print("Loading dataset...")
df = pd.read_parquet(DATA_PATH)

print("\n================ DATASET SHAPE ================")
print(df.shape)

print("\n================ COLUMNS ================")
for i, col in enumerate(df.columns):
    print(f"{i+1}. {col} | {df[col].dtype}")

print("\n================ FIRST 5 ROWS ================")
print(df.head())

print("\n================ NUMERIC COLUMNS ================")
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
for col in numeric_cols:
    print(col)

print("\nTotal numeric columns:", len(numeric_cols))

print("\n================ MISSING VALUES ================")
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False).head(30))

print("\n================ BASIC STATS ================")
print(df[numeric_cols].describe().T.head(30))
