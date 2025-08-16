# src/eval_rag.py
# ============================================================
# RAG 簡易評価（CSV: id,question,expected_short）
# - 各質問に対して answer() を実行
# - 出力と簡易スコア（類似率×0.6 + キーワード一致率×0.4）
# - CSVに書き出し、SUMMARY を表示（pass@0.6）
# ============================================================

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse
import csv
import difflib
import re
import time

from src.rag_baseline import answer  # 既存の関数を利用

# ---------- テキスト正規化 ----------
def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def jp_sentences(text: str) -> list[str]:
    t = _norm(text)
    sents = re.findall(r"[^。！？\n]{2,}[。！？]", t) or []
    tail = re.sub(r".*[。！？]", "", t)
    if len(tail) >= 6: sents.append(tail)
    return sents



# ---------- 回答の整形（出典/改行などを削る） ----------
def clean_answer_text(s: str) -> str:
    s = _norm(s)
    s = re.sub(r"\[出典:.*?\]$", "", s)          # 末尾の出典を除去
    s = re.sub(r"（出典.*?）$", "", s)
    s = s.splitlines()[0]                        # 1行目だけ
    m = re.search(r"。", s)                      # 最初の1文
    return (s[:m.end()] if m else s[:120]).strip()

STOP = {"こと","ため","など","の","に","は","が","を","で","と","や","も","です","ます","する"}
def terms_from_expected(exp: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9一-龥ぁ-んァ-ヶー]{2,}", exp)  # 2文字以上を雑に抽出
    out = [t for t in toks if t not in STOP]
    return list(dict.fromkeys(out))  # 重複除去・順序維持

# ---------- 簡易スコア ----------
def simple_score(expected: str, actual: str) -> tuple[float, float, float, int, int]:
    """
    戻り値:
      (score, char_ratio, kw_ratio, kw_total, kw_hit)
      score      : 最終スコア（0〜1目安）
      char_ratio : 文字列類似（SequenceMatcher）
      kw_ratio   : キーワード一致率（expected の語が actual に何割含まれるか）
    """
    e = _norm(expected)
    a = _norm(actual)

    char_ratio = difflib.SequenceMatcher(None, e, a).ratio()  # 0〜1

    toks = re.findall(r"[A-Za-z0-9一-龥ぁ-んァ-ヶー]+", e)
    kw_total = len(toks)
    kw_hit = sum(1 for t in toks if t and t in a)
    kw_ratio = kw_hit / max(1, kw_total)

    score = round(char_ratio * 0.6 + kw_ratio * 0.4, 3)
    return score, char_ratio, kw_ratio, kw_total, kw_hit

# ---------- CLI ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 簡易評価（CSVを回して結果を保存）")
    ap.add_argument("--in", dest="in_path", required=True, type=Path, help="評価CSV: id,question,expected_short")
    ap.add_argument("--out", dest="out_path", type=Path, default=None, help="出力CSV（未指定なら timestamp パス）")
    ap.add_argument("--k", type=int, default=3, help="検索件数")
    ap.add_argument("--hybrid", action="store_true", help="埋め込み＋キーワード加点の再ランキングを使う")
    ap.add_argument("--terms", type=str, default="RAG, LLM, 検索, 生成", help="カンマ区切りの加点キーワード（固定）")
    ap.add_argument("--pool", type=int, default=20, help="再ランキング前に集める候補件数")
    ap.add_argument("--kw-boost", type=float, default=0.4, help="キーワード加点の重み")
    ap.add_argument("--llm", action="store_true", help="LLM要約で回答（APIキー必須）")
    ap.add_argument("--sent-split", action="store_true",
                help="チャンクを文字数ではなく文単位（。！？/改行）で分割する")
    ap.add_argument("--dyn-terms", action="store_true",
                help="expected_short から自動で加点用語を抽出して再ランキングに使う")

    args = ap.parse_args()

    # 出力パス
    if args.out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_path = Path(f"artifacts/eval/eval_{ts}.csv")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    # 入力読み込み
    rows = []
    with args.in_path.open(encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            rows.append(row)

    base_terms_default = [s.strip() for s in (args.terms or "").split(",") if s.strip()]

    results = []
    passed = 0
    n = len(rows)

    for i, row in enumerate(rows, start=1):
        qid = row.get("id", f"Q{i}")
        q   = row.get("question", "")
        exp = row.get("expected_short", "")

        # 質問ごとの加点語
        per_q_terms = terms_from_expected(exp) if args.dyn_terms else base_terms_default
        
        # === 時間計測開始
        t0 = time.time()

        # 回答を取得
        ans_text_raw, hits = answer(
        q, k=args.k, use_llm=args.llm,
        use_hybrid=args.hybrid,
        terms=per_q_terms,  # ← ここを per_q_terms に
        pool=args.pool, kw_boost=args.kw_boost
        )
        # === 経過時間
        sec = time.time() - t0

        # ★ここから（このブロック全部が for の内側）
        candidates = []
        candidates.append(("answer_raw", ans_text_raw))
        candidates.append(("answer_clean", clean_answer_text(ans_text_raw)))
        for j, h in enumerate((hits or [])[:args.k], start=1):
            txt = _norm(h.get("text", ""))
            candidates.append((f"hit{j}_head", txt[:120]))
            sents = jp_sentences(txt)
            def contains_term(s): return any(t in s for t in per_q_terms) if per_q_terms else False
            sents_sorted = sorted(sents, key=lambda s: (not contains_term(s), -len(s)))
            for k2, s in enumerate(sents_sorted[:5], start=1):
                candidates.append((f"hit{j}_sent{k2}", s[:120]))


        best = None
        for tag, cand in candidates:
            sc, char_r, kw_r, kw_tot, kw_hit = simple_score(exp, cand)
            item = (sc, tag, cand, char_r, kw_r, kw_tot, kw_hit)
            if (best is None) or (sc > best[0]):
                best = item

        score, tag, cand_used, char_r, kw_r, kw_tot, kw_hit = best
        ok = 1 if score >= 0.6 else 0
        passed += ok

        sources = "; ".join([f"{h['source']} p.{h['page']}" for h in (hits or [])])
        results.append({
            "id": qid,
            "question": q,
            "expected_short": exp,
            "answer": ans_text_raw,
            "judge_from": tag,
            "candidate_used": cand_used,
            "sources": sources,
            "score": f"{score:.3f}",
            "char_ratio": f"{char_r:.3f}",
            "kw_ratio": f"{kw_r:.3f}",
            "kw_hit": kw_hit,
            "kw_total": kw_tot,
            "sec": f"{sec:.2f}",
        })
        print(f"[{i}/{n}] {qid}: score={score:.3f} ({'OK' if ok else 'NG'})  from={tag}  time={sec:.2f}s")
        # ★ここまで


    # 出力（Excel 文字化け対策で UTF-8 BOM）
    if results:
        with args.out_path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    rate = (passed / n) * 100 if n else 0.0
    print("=== SUMMARY ===")
    print(f"out      : {args.out_path}")
    print(f"n        : {n}")
    print(f"pass@0.6 : {passed} ({rate:.1f}%)")
    print(f"options  : k={args.k} hybrid={args.hybrid} terms={base_terms_default} pool={args.pool} kw_boost={args.kw_boost} llm={args.llm} dyn_terms={args.dyn_terms}")

if __name__ == "__main__":
    main()
