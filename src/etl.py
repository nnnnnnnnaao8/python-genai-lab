
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class DataLoader:
    def load(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)


@dataclass
class Cleaner:
    drop_na_cols: list[str] | None = None
    drop_duplicates: bool = True

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.drop_na_cols:
            out = out.dropna(subset=self.drop_na_cols)
        if self.drop_duplicates:
            out = out.drop_duplicates()
        for col in out.columns:
            if out[col].dtype == "object":
                try:
                    out[col] = pd.to_numeric(out[col])
                except Exception:
                    pass
        return out


@dataclass
class Plotter:
    def plot(self, df: pd.DataFrame, fig_path: Path) -> None:
        if "value" not in df.columns:
            return
        plt.figure()
        df["value"].hist(bins=10)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path)
        plt.close()


def aggregate(df: pd.DataFrame, mode: str = "mean") -> pd.DataFrame:
    if not {"category", "value"}.issubset(df.columns):
        raise ValueError("Required columns: 'category', 'value'")
    if mode not in {"mean", "median"}:
        raise ValueError("mode must be 'mean' or 'median'")
    op = "mean" if mode == "mean" else "median"
    grouped = (
        df.groupby("category", dropna=False)
          .agg(count=("value", "count"), agg_value=("value", op))
          .reset_index()
          .sort_values(["count", "agg_value"], ascending=[False, False])
    )
    return grouped


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal ETL pipeline")
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--fig", dest="fig_path", required=False, type=Path, default=Path("artifacts/eda_fig.png"))
    ap.add_argument("--drop-duplicates", dest="drop_duplicates", action="store_true")
    ap.add_argument("--no-drop-duplicates", dest="drop_duplicates", action="store_false")
    ap.add_argument("--agg", choices=["mean", "median"], default="mean")
    ap.add_argument("--verbose", action="store_true", help="途中経過を表示")
    ap.add_argument("--show", action="store_true", help="図を画面に表示")
    ap.set_defaults(drop_duplicates=True)
    args = ap.parse_args()

    loader = DataLoader()
    cleaner = Cleaner(drop_na_cols=["value"], drop_duplicates=args.drop_duplicates)
    plotter = Plotter()

    if args.verbose: print(f"[1/4] Load: {args.in_path}")
    df = loader.load(args.in_path)

    if args.verbose: print("[2/4] Clean")
    df_clean = cleaner.clean(df)

    if args.verbose: print(f"[3/4] Aggregate -> {args.out_path} (mode={args.agg})")
    summary = aggregate(df_clean, mode=args.agg)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_path, index=False)

    if args.verbose: print(f"[4/4] Figure -> {args.fig_path}")
    plotter.plot(df_clean, args.fig_path)

    if args.show:
        plt.figure()
        df_clean["value"].hist(bins=10)
        plt.show()

    print(f"Done: wrote {args.out_path} and {args.fig_path}")


if __name__ == "__main__":
    main()
