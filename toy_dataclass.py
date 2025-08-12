from dataclasses import dataclass

@dataclass
class CleanerConfig:
    drop_na_cols: list[str] | None = None
    drop_duplicates: bool = True

cfg = CleanerConfig(drop_na_cols=["value"], drop_duplicates=False)
print(cfg)                 # ← 設定の入れ物（自動で見やすく表示される）
print(cfg.drop_na_cols)    # ← ドットで取り出せる
