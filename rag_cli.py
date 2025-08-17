# rag_cli.py
# かんたん対話CLI：質問→ヒット上位→回答（出典つき）
from __future__ import annotations
from pathlib import Path
import argparse, csv, time, re
from datetime import datetime

from src.rag_baseline import retrieve, retrieve_hybrid, answer

def parse_terms(s: str | None) -> list[str]:
    if not s: return []
    return [t.strip() for t in s.split(",") if t.strip()]

def print_hits(hits: list[dict], show_scores: bool = True, max_chars: int = 160):
    for i, h in enumerate(hits, start=1):
        tail = f"  [sim={h.get('sim',0):.3f} kw={h.get('kw',0):.1f} score={h.get('score',0):.3f}]" if show_scores else ""
        snippet = (h["text"][:max_chars] + "...") if len(h["text"]) > max_chars else h["text"]
        print(f"[{i}] {h['source']} p.{h['page']}  id={h['id']}\n    {snippet}{tail}")

def log_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file: w.writeheader()
        w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="RAG 対話CLI（Q> で質問を入力）")
    ap.add_argument("--k", type=int, default=5, help="最終的に使う件数")
    ap.add_argument("--pool", type=int, default=100, help="再ランク前の候補数")
    ap.add_argument("--kw-boost", type=float, default=0.9, help="キーワード加点の強さ")
    ap.add_argument("--hybrid", action="store_true", help="ハイブリッド検索（推奨）")
    ap.add_argument("--llm", action="store_true", help="LLM要約をON（APIキー必要）")
    ap.add_argument("--terms", type=str, default="RAG, 検索, 生成, コンテキスト, 文脈, ベクトル, 近傍, チャンク, オーバーラップ",
                    help="カンマ区切りの加点キーワード")
    args = ap.parse_args()

    terms = parse_terms(args.terms)
    print("=== RAG CLI ===")
    print("使い方：そのまま質問を入力。コマンド: :hits でヒット一覧のみ, :q で終了")
    print(f"(k={args.k}, pool={args.pool}, kw-boost={args.kw_boost}, hybrid={args.hybrid}, llm={args.llm})\n")

    log_csv = Path("artifacts/cli_logs.csv")

    while True:
        q = input("Q> ").strip()
        if not q: 
            continue
        if q in {":q", ":quit", "exit"}:
            print("bye.")
            break

        t0 = time.time()
        # まずヒットを確認（ハイブリッド推奨）
        if args.hybrid:
            hits = retrieve_hybrid(q, k=args.k, terms=terms, pool=args.pool, kw_boost=args.kw_boost)
        else:
            hits = retrieve(q, k=args.k)
        t1 = time.time()

        print_hits(hits, show_scores=args.hybrid)
        ans, _ = answer(q, k=args.k, use_llm=args.llm,
                        use_hybrid=args.hybrid, terms=terms,
                        pool=args.pool, kw_boost=args.kw_boost)
        t2 = time.time()

        print("\n=== answer ===")
        print(ans)
        print(f"\n(time: search={t1-t0:.2f}s, answer={t2-t1:.2f}s)")

        # ログ保存（後で見返せる）
        log_row(log_csv, {
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "q": q, "k": args.k, "pool": args.pool, "kw_boost": args.kw_boost,
            "hybrid": args.hybrid, "llm": args.llm,
            "n_hits": len(hits),
        })

if __name__ == "__main__":
    main()
