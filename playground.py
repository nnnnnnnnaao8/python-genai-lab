#%% セル1：パラメータ
from pathlib import Path
IN  = Path("data/raw/sample.csv")
OUT = Path("data/processed/summary.csv")
FIG = Path("artifacts/eda_fig.png")

#%% セル2：ETLの中身を関数呼び出しで（引数なしでOK）
from src.etl import DataLoader, Cleaner, aggregate, Plotter
import pandas as pd

df = DataLoader().load(IN)
df_clean = Cleaner(drop_na_cols=["value"], drop_duplicates=True).clean(df)
summary = aggregate(df_clean, mode="median")
OUT.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUT, index=False)
Plotter().plot(df_clean, FIG)

#%% セル3：結果を確認（Data Viewerで開ける）
import pandas as pd
display(summary.head())     # 右上のグリッドアイコンで表表示

# %%
