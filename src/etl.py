
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime as dt



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
class Standardizer:
    col: str = "value"
    make_new_col: bool = True  # Trueなら value_z を追加

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.col not in out.columns:
            return out
        s = out[self.col]
        mu, sigma = float(s.mean()), float(s.std() or 0.0)
        if sigma == 0.0:
            return out
        z = (s - mu) / sigma
        if self.make_new_col:
            out[self.col + "_z"] = z
        else:
            out[self.col] = z
        return out


@dataclass
class Plotter:
    def plot(self, df: pd.DataFrame, fig_path: Path, box_path: Path | None = None) -> None:
        if "value" in df.columns:
            plt.figure()
            df["value"].hist(bins=10)
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path)
            plt.close()
        if box_path is not None and "value" in df.columns:
            plt.figure()
            df["value"].plot(kind="box")
            box_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(box_path)
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
    ap.add_argument("--scale", action="store_true", help="value列をzスコアで標準化（value_zを追加）")
    ap.add_argument("--no-scale", dest="scale", action="store_false")
    ap.add_argument("--fig2", dest="fig2_path", type=Path, required=False, help="箱ひげ図の保存先（省略可）")
    ap.add_argument("--report", type=Path, default=None, help="実行レポートを保存する先（.txt推奨）")
    ap.add_argument("--open", dest="auto_open", action="store_true", help="終了後に成果物を自動で開く（Windows）")



    args = ap.parse_args()

    loader = DataLoader()
    cleaner = Cleaner(drop_na_cols=["value"], drop_duplicates=args.drop_duplicates)


    plotter = Plotter()

    if args.verbose: print(f"[1/4] Load: {args.in_path}")
    df = loader.load(args.in_path)

    if args.verbose: print("[2/4] Clean")
    df_clean = cleaner.clean(df)
    
    std = Standardizer(col="value", make_new_col=True)
    if args.scale:
        if args.verbose: print("[2.5/4] Standardize (z-score) -> value_z")
        df_clean = std.transform(df_clean)

    if args.verbose: print(f"[3/4] Aggregate -> {args.out_path} (mode={args.agg})")
    summary = aggregate(df_clean, mode=args.agg)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_path, index=False)

    if args.verbose: print(f"[4/4] Figure -> {args.fig_path}")
    plotter.plot(df_clean, args.fig_path, box_path=args.fig2_path)

    # ---- 実感の出る要約を作成 ----
    raw_rows = len(df)
    after_dropna = len(df.dropna(subset=["value"])) if "value" in df.columns else raw_rows
    after_clean = len(df_clean)
    dup_removed = max(after_dropna - after_clean, 0) if args.drop_duplicates else 0

    stats_lines = []
    if "value" in df_clean.columns:
        s = df_clean["value"].dropna()
        if len(s) > 0:
            stats_lines.append(f"value: count={len(s)} mean={s.mean():.2f} median={s.median():.2f} std={s.std():.2f} min={s.min():.2f} max={s.max():.2f}")
    if "category" in df_clean.columns:
        top = df_clean["category"].value_counts(dropna=False).head(3)
        top_str = ", ".join([f"{idx}:{cnt}" for idx, cnt in top.items()])
        stats_lines.append(f"top categories: {top_str}")

    summary_head = ""
    try:
        summary_head = summary.head(5).to_csv(index=False).strip()
    except Exception:
        pass

    digest = (
        "=== ETL RUN SUMMARY ===\n"
        f"when      : {dt.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"in        : {args.in_path}\n"
        f"out       : {args.out_path}\n"
        f"figure    : {args.fig_path}\n"
        f"mode      : agg={args.agg} drop_duplicates={args.drop_duplicates} scale={getattr(args,'scale',False)}\n"
        f"rows      : raw={raw_rows} -> dropna={after_dropna} -> clean={after_clean} (dups_removed~{dup_removed})\n"
        + ("\n".join(stats_lines) + "\n" if stats_lines else "")
        + ("--- summary head (first 5 rows) ---\n" + summary_head + "\n" if summary_head else "")
    )

    print(digest)

    # レポート保存（指定があれば）
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(digest)
        print(f"[report] wrote {args.report}")

    # 自動で開く（Windows想定）
    if getattr(args, "auto_open", False):
        try:
            os.startfile(str(args.fig_path))        # 図
        except Exception:
            pass
        try:
            os.startfile(str(args.out_path))        # CSV
        except Exception:
            pass
        if args.report is not None:
            try:
                os.startfile(str(args.report))      # レポート
            except Exception:
                pass


    if args.show:
        plt.figure()
        df_clean["value"].hist(bins=10)
        plt.show()

    print(f"Done: wrote {args.out_path} and {args.fig_path}")


if __name__ == "__main__":
    main()
