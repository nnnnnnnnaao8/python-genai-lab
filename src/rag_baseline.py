# src/rag_baseline.py
# ============================================================
# 最小RAG（Retrieval-Augmented Generation）の骨格
#  - 文書(PDF/TXT)からテキストを取り出し → チャンクに分割
#  - 文章ベクトルに変換してローカルDB(Chroma)へ保存（索引）
#  - 質問をベクトル化して上位k件を検索
#  - （任意）LLMで要約回答。常に出典を末尾に残す
# ------------------------------------------------------------
# 使いどころ：
#    python -m src.rag_baseline --rebuild
#    python -m src.rag_baseline --q "RAGとは？" --k 3 [--llm]
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse, re
import fnmatch  # ← 先頭付近に追加
from pypdf import PdfReader                   # PDFからテキスト抽出（画像PDFは×）
import chromadb                               # ローカル永続のベクトルDB
from chromadb.utils import embedding_functions

import re  # 既にあるはず

# 日本語/英数字の割合をざっくり測る
_JP_OK = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF\u0030-\u0039A-Za-z\s。、，．・：；？！「」『』（）\-\u3000-\u303F\uFF10-\uFF19\uFF21-\uFF3A\uFF41-\uFF5A]")

def jp_ratio(text: str) -> float:
    if not text: 
        return 0.0
    valid = len(_JP_OK.findall(text))
    return valid / max(1, len(text))

def is_garbled(text: str, min_ratio: float = 0.4) -> bool:
    # 制御文字は空白へ
    text = re.sub(r"[\u0000-\u001F]", " ", text)
    return jp_ratio(text) < min_ratio


# ---------- パスやモデルの設定 ----------
DOC_DIR   = Path("data/raw/docs")             # 読み込む資料の置き場（*.pdf, *.txt）
INDEX_DIR = Path("data/index/rag_demo")       # ベクトル索引の保存先（ローカルフォルダ）
COLL_NAME = "docs"                            # コレクション名（Chromaの論理名）
# EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 英語強め・軽量
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 多言語（日本語OK）


# ---------- 文字整形＆分割（チャンク化） ----------
def norm_ws(text: str) -> str:
    """改行や連続スペースを1つにして読みやすく整形。検索の安定化にも効く。"""
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, size: int = 700, overlap: int = 100) -> List[str]:
    """
    長い本文を部分的に重ねながら分割する。
      - size: 1チャンクの長さ（文字）
      - overlap: 隣のチャンクと重ねる長さ（コンテキスト切れ防止）
    """
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i+size])
        i += max(size - overlap, 1)  # 後ろへ進む。重なり分だけ戻りが小さくなるイメージ
    return [norm_ws(t) for t in out if t.strip()]

# ---------- 資料の取り込み（Extract） ----------
def read_docs(include: str | None = None) -> List[Tuple[str, int, str]]:
    items: List[Tuple[str,int,str]] = []
    # PDF
    for pdf in sorted(DOC_DIR.glob("*.pdf")):
        name = pdf.name
        if include and not (include.lower() in name.lower()):
            continue
        txt_ok = False
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(pdf))
            for p in range(len(doc)):
                page = doc.load_page(p)
                txt = page.get_text("text") or ""
                if txt.strip():
                    items.append((name, p+1, norm_ws(txt)))
                    txt_ok = True
        except Exception:
            txt_ok = False
        if not txt_ok:
            # フォールバック：pypdf
            try:
                from pypdf import PdfReader
                r = PdfReader(str(pdf))
                for p, page in enumerate(r.pages, start=1):
                    txt = page.extract_text() or ""
                    if txt.strip():
                        items.append((name, p, norm_ws(txt)))
            except Exception:
                pass

    # TXT
    for txt in sorted(DOC_DIR.glob("*.txt")):
        name = txt.name
        if include and not (include.lower() in name.lower()):
            continue
        t = txt.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(t, size=2000, overlap=0), start=1):
            items.append((name, i, chunk))
    return items


# ---------- 索引の作り直し（Index作成：Transform+Load） ----------
def rebuild_index(chunk: int = 700, overlap: int = 100, include: str | None = None, min_jp_ratio: float = 0.4) -> int:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection(COLL_NAME)
    except Exception:
        pass
    emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
    coll = client.create_collection(COLL_NAME, embedding_function=emb)

    rows = []
    skipped = 0
    for src, page, text in read_docs(include=include):
        for j, t in enumerate(chunk_text(text, size=chunk, overlap=overlap), start=1):
            if is_garbled(t, min_ratio=min_jp_ratio):
                skipped += 1
                continue
            rid = f"{src}#p{page}#c{j}"
            rows.append((rid, t, {"source": src, "page": page, "chunk": j}))

    if not rows:
        print(f"[index] built 0 chunks (skipped {skipped} garbled)")
        return 0

    ids, docs, metas = zip(*rows)
    coll.add(ids=list(ids), documents=list(docs), metadatas=list(metas))
    print(f"[index] built {len(rows)} chunks (skipped {skipped} garbled) at {INDEX_DIR}\\{COLL_NAME}")
    return len(rows)



# ---------- 既存の索引を開く ----------
def _open_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
    return client.get_collection(COLL_NAME, embedding_function=emb)

# ---------- 検索（Retrieve） ----------
def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    クエリ文をベクトル化して上位k件のチャンクを返す。
    返り値は {id, text, distance, source, page, chunk} の辞書リスト。
    """
    coll = _open_collection()
    res = coll.query(query_texts=[query], n_results=k)

    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "distance": float(res["distances"][0][i]) if "distances" in res else None,
            "source": res["metadatas"][0][i]["source"],
            "page": int(res["metadatas"][0][i]["page"]),
            "chunk": int(res["metadatas"][0][i]["chunk"]),
        })
    return hits

# ---------- ハイブリッド再ランキング（埋め込み＋キーワード加点） ----------
def _keyword_score(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    t = text.lower()
    score = 0.0
    for kw in terms:
        kw = kw.strip()
        if not kw:
            continue
        # 大文字小文字を無視して部分一致（全角/半角は簡易対応）
        k = kw.lower()
        if k in t:
            score += 1.0
        # RAG の全角表記など超簡易置換（必要に応じて足す）
        if k == "rag" and "ＲＡＧ" in text:
            score += 1.0
    return score

def retrieve_hybrid(query: str, k: int = 3, terms: list[str] | None = None,
                    pool: int = 20, kw_boost: float = 0.3) -> list[dict[str, any]]:
    base_hits = retrieve(query, k=pool)
    if not base_hits:
        return []

    terms = terms or []

    # 距離の min/max を使って 0〜1 に正規化（小さい距離 = 類似度1に近い）
    ds = [float(h.get("distance")) for h in base_hits if h.get("distance") is not None]
    if ds:
        dmin, dmax = min(ds), max(ds)
        rng = (dmax - dmin) if dmax > dmin else 1.0
    else:
        dmin, rng = 0.0, 1.0

    rescored = []
    for h in base_hits:
        dist = h.get("distance")
        if dist is None:
            sim = 0.0
        else:
            d = float(dist)
            sim = (dmax - d) / rng  # 0〜1にマップ（1がベスト）
        kw  = _keyword_score(h["text"], terms)
        # キーワード加点が強すぎる時は上限をかける（任意）
        kw_capped = min(kw, 5.0)
        total = sim + kw_boost * kw_capped

        hh = dict(h)
        hh["sim"] = sim
        hh["kw"]  = kw
        hh["score"] = total
        rescored.append(hh)

    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored[:k]



# ---------- 回答（Generate） ----------
def answer(
    query: str,
    k: int = 3,
    use_llm: bool = False,
    use_hybrid: bool = False,             # ← 追加
    terms: list[str] | None = None,       # ← 追加（キーワード）
    pool: int = 20,                       # ← 追加（候補件数）
    kw_boost: float = 0.3                 # ← 追加（加点の強さ）
) -> tuple[str, list[dict[str, any]]]:
    hits = (
        retrieve_hybrid(query, k=k, terms=terms, pool=pool, kw_boost=kw_boost)
        if use_hybrid else
        retrieve(query, k=k)
    )
    if not use_llm or not hits:
        snippet = hits[0]["text"][:400] if hits else "（該当なし）"
        srcs = ", ".join([f"{h['source']} p.{h['page']}" for h in hits])
        return f"{snippet}\n\n[出典: {srcs}]", hits
  

    # --- ここからLLM要約（APIキーがあればON） ---
    try:
        from openai import OpenAI
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key: raise RuntimeError("NO_KEY")

        client = OpenAI(api_key=api_key)
        # 検索ヒットを「出典番号つきコンテキスト」に整形
        context = "\n\n".join([f"[{i+1}] {h['source']} p.{h['page']}: {h['text']}" for i, h in enumerate(hits)])
        prompt = (
            "次のコンテキストだけを根拠に、質問に日本語で簡潔に答えてください。"
            "最後に出典番号を角括弧で示してください。\n\n"
            f"質問: {query}\n\nコンテキスト:\n{context}"
        )

        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":"あなたは事実に忠実なアシスタントです。"},
                {"role":"user","content":prompt}
            ],
            temperature=0.2,
        )
        text = chat.choices[0].message.content.strip()
        srcs = ", ".join([f"{h['source']} p.{h['page']}" for h in hits])
        return f"{text}\n\n[出典: {srcs}]", hits

    except Exception:
        # LLMが失敗したら抽出型へフォールバック（止めない）
        snippet = hits[0]["text"][:400] if hits else "（該当なし）"
        srcs = ", ".join([f"{h['source']} p.{h['page']}" for h in hits])
        return f"{snippet}\n\n[出典: {srcs}]", hits

# ---------- CLI（コマンドライン） ----------
def main():
    ap = argparse.ArgumentParser(description="最小RAG（1PDF/TXTから回答）")
    ap.add_argument("--rebuild", action="store_true", help="索引を作り直す")
    ap.add_argument("--chunk", type=int, default=700, help="チャンク長（文字）")
    ap.add_argument("--overlap", type=int, default=100, help="オーバーラップ（文字）")
    ap.add_argument("--q", type=str, help="質問文")
    ap.add_argument("--k", type=int, default=3, help="検索件数")
    ap.add_argument("--llm", action="store_true", help="LLMで要約回答（APIキー必須）")
    ap.add_argument("--min-jp-ratio", type=float, default=0.4, help="この比率未満のチャンクは文字化け扱いで除外")
    ap.add_argument("--include", type=str, default=None, help="索引対象をファイル名で部分一致/ワイルドカード指定（例: sample.txt / *.pdf）")
    ap.add_argument("--hybrid", action="store_true", help="埋め込みにキーワード加点で再ランキング")
    ap.add_argument("--terms", type=str, default="RAG, LLM, 検索, 生成, ベクトル, 埋め込み",
                help="カンマ区切りの加点キーワード")
    ap.add_argument("--pool", type=int, default=20, help="再ランキング前に集める件数")
    ap.add_argument("--kw-boost", type=float, default=0.3, help="キーワード加点の重み（大きいほど強く効く）")

    
    args = ap.parse_args()


    # 索引作成（--rebuild を付けたときだけ実行）
    if args.rebuild:
        n = rebuild_index(chunk=args.chunk, overlap=args.overlap, include=args.include, min_jp_ratio=args.min_jp_ratio)
      #  print(f"[index] built {n} chunks at {INDEX_DIR}\\{COLL_NAME}")# printはrebuild_index側が出すので任意


    # 質問（--q を付けたときだけ実行）
    if args.q:
        terms = [s.strip() for s in (args.terms or "").split(",") if s.strip()]
        hits = retrieve_hybrid(args.q, k=args.k, terms=terms, pool=args.pool, kw_boost=args.kw_boost) \
            if args.hybrid else retrieve(args.q, k=args.k)
        for i, h in enumerate(hits, start=1):
            print(
                f"[{i}] {h['source']} p.{h['page']}  id={h['id']}\n    {h['text'][:160]}..."
                f"{('  [sim=%.3f kw=%.1f score=%.3f]' % (h.get('sim',0), h.get('kw',0), h.get('score',0))) if args.hybrid else ''}"
            )
        ans, _ = answer(args.q, k=args.k, use_llm=args.llm,
                        use_hybrid=args.hybrid, terms=terms,
                        pool=args.pool, kw_boost=args.kw_boost)
        print("\n=== answer ===")
        print(ans)


if __name__ == "__main__":
    main()