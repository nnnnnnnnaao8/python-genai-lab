import pandas as pd

df = pd.DataFrame({
    "category": ["A","A","B","B","B", None],
    "value": ["1", "2", "3", None, "3", "1"]
})
print("raw:", df.shape)

# 1) 欠損を落とす（valueのNaN）
df1 = df.dropna(subset=["value"])
print("dropna:", df1.shape)

# 2) 数値化（object→数値へ）
df1["value"] = pd.to_numeric(df1["value"])  # ← etl.py でもやってたやつ
print("dtypes:", df1.dtypes.to_dict())

# 3) 重複を落とす
df2 = df1.drop_duplicates()
print("dedup:", df2.shape)

# 4) 集計（平均/中央値）
mean_tbl = df2.groupby("category").agg(count=("value","count"), agg_value=("value","mean")).reset_index()
med_tbl  = df2.groupby("category").agg(count=("value","count"), agg_value=("value","median")).reset_index()

print("\nmean:\n", mean_tbl)
print("\nmedian:\n", med_tbl)
