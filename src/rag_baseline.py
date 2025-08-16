# src/rag_baseline.py
# ============================================================
# 最小RAG（Retrieval-Augmented Generation）
#  - 文書(PDF/TXT)からテキスト抽出 → チャンク化（文字数 or 文単位）
#  - ベクトル化して Chroma に格納（索引）
#  - 質問の近傍検索（ハイブリッド再ランキング可）
#  - （任意）LLM要約、常に出典を末尾に残す
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse
import re
import fnmatch

from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# ---------- 日本語率の簡易判定 ----------
_JP_OK = re.compile(
    r"[\u3040-\u30FF\u4E00-\u9FFF\u0030-\u0039A-Za-z\s。、，．・：；？！「」『』（）\-\u3000-\u303F\uFF10-\uFF19\uFF21-\uFF3A\uFF41-\uFF5A]"
)

def jp_ratio(text: str) -> float:
    if not text:
        return 0.0
    valid = len(_JP_OK.findall(text))
    return valid / max(1, len(text))

def is_garbled(text: str, min_ratio: float = 0.4) -> bool:
    text = re.sub(r"[\u0000-\u001F]", " ", text)  # 制御文字は空白へ
    return jp_ratio(text) < min_ratio

# ---------- 文分割（日本語ざっくり） ----------
def split_sentences_jp(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())
    sents = re.findall(r"[^。！？\n]{2,}[。！？]", t) or []
    tail = re.sub(r".*[。！？]", "", t)
    if len(tail) >= 6:
        sents.append(tail)
    return [s.strip() for s in sents if len(s.strip()) >= 6]

# ---------- パスやモデル ----------
DOC_DIR   = Path("data/raw/docs")
INDEX_DIR = Path("data/index/rag_demo")
COLL_NAME = "docs"
# 英語軽量: "sentence-transformers/all-MiniLM-L6-v2"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 多言語対応

# ---------- 整形＆文字数チャンク ----------
def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def chunk_text(text: str, size: int = 700, overlap: int = 100) -> List[str]:
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        out.append(text[i:i+size])
        i += max(size - overlap, 1)
    return [norm_ws(t) for t in out if t.strip()]

# ---------- include のマッチ（部分一致 or ワイルドカード） ----------
def _match_include(name: str, include: str | None) -> bool:
    if not include:
        return True
    if any(ch in include for ch in "*?[]"):
        return fnmatch.fnmatch(name, include)
    return include.lower() in name.lower()

# ---------- 資料取り込み（ページ/大きめチャンク単位まで） ----------
def read_docs(include: str | None = None) -> List[Tuple[str, int, str]]:
    """
    返り値: [(source_name, page_or_seq, text), ...]
    後段で size/overlap または文分割で最終チャンク化する。
    """
    items: List[Tuple[str,int,str]] = []

    # PDF
    for pdf in sorted(DOC_DIR.glob("*.pdf")):
        name = pdf.name
        if not _match_include(name, include):
            continue
        txt_ok = False
        # 1st: PyMuPDF（あれば）
        try:
            import fitz  # type: ignore
            doc = fitz.open(str(pdf))
            for p in range(len(doc)):
                page = doc.load_page(p)
                txt = page.get_text("text") or ""
                if txt.strip():
                    items.append((name, p+1, norm_ws(txt)))
                    txt_ok = True
        except Exception:
            txt_ok = False
        # 2nd: pypdf フォールバック
        if not txt_ok:
            try:
                r = PdfReader(str(pdf))
                for p, page in enumerate(r.pages, start=1):
                    txt = page.extract_text() or ""
                    if txt.strip():
                        items.append((name, p, norm_ws(txt)))
            except Exception:
                pass

    # TXT（ここでは丸ごと読み、後段で最終チャンク化）
    for txt in sorted(DOC_DIR.glob("*.txt")):
        name = txt.name
        if not _match_include(name, include):
            continue
        t = txt.read_text(encoding="utf-8", errors="ignore")
        items.append((name, 1, norm_ws(t)))

    return items

# ---------- 索引の作り直し ----------
def rebuild_index(
    chunk: int = 700,
    overlap: int = 100,
    include: str | None = None,
    min_jp_ratio: float = 0.4,
    sent_split: bool = False,
) -> int:
    """
    sent_split=True の場合は文単位でチャンク化する。
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        client.delete_collection(COLL_NAME)
    except Exception:
        pass

    emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
    coll = client.create_collection(COLL_NAME, embedding_function=emb)

    rows: list[tuple[str, str, dict]] = []
    skipped = 0

    for src, page, text in read_docs(include=include):
        # 最終チャンク化（文字数 or 文）
        parts = split_sentences_jp(text) if sent_split else chunk_text(text, size=chunk, overlap=overlap)
        for j, t in enumerate(parts, start=1):
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
    mode = "sent-split" if sent_split else f"chars(size={chunk},overlap={overlap})"
    print(f"[index] built {len(rows)} chunks (skipped {skipped} garbled) at {INDEX_DIR}\\{COLL_NAME}  mode={mode} include={include or '*'}")
    return len(rows)

# ---------- 既存の索引を開く ----------
def _open_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
    return client.get_collection(COLL_NAME, embedding_function=emb)

# ---------- 検索 ----------
def retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    coll = _open_collection()
    res = coll.query(query_texts=[query], n_results=k)
    hits: list[dict[str, Any]] = []
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

# ---------- ハイブリッド再ランキング ----------
def _keyword_score(text: str, terms: list[str]) -> float:
    if not terms:
        return 0.0
    t = text.lower()
    score = 0.0
    for kw in terms:
        k = kw.strip().lower()
        if not k:
            continue
        if k in t:
            score += 1.0
        if k == "rag" and "ＲＡＧ" in text:
            score += 1.0
    return score

def retrieve_hybrid(query: str, k: int = 3, terms: list[str] | None = None,
                    pool: int = 20, kw_boost: float = 0.3) -> list[dict[str, Any]]:
    base_hits = retrieve(query, k=pool)
    if not base_hits:
        return []

    terms = terms or []
    ds = [float(h.get("distance")) for h in base_hits if h.get("distance") is not None]
    if ds:
        dmin, dmax = min(ds), max(ds)
        rng = (dmax - dmin) if dmax > dmin else 1.0
    else:
        dmin, rng = 0.0, 1.0

    rescored = []
    for h in base_hits:
        dist = h.get("distance")
        sim = 0.0 if dist is None else (dmax - float(dist)) / rng  # 0〜1（1が良い）
        kw  = _keyword_score(h["text"], terms)
        kw_capped = min(kw, 5.0)
        total = sim + kw_boost * kw_capped

        hh = dict(h)
        hh["sim"] = sim
        hh["kw"] = kw
        hh["score"] = total
        rescored.append(hh)

    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored[:k]

# ---------- 回答 ----------
def answer(
    query: str,
    k: int = 3,
    use_llm: bool = False,
    use_hybrid: bool = False,
    terms: list[str] | None = None,
    pool: int = 20,
    kw_boost: float = 0.3,
) -> tuple[str, list[dict[str, Any]]]:
    hits = retrieve_hybrid(query, k=k, terms=terms or [], pool=pool, kw_boost=kw_boost) if use_hybrid else retrieve(query, k=k)
    if not use_llm or not hits:
        snippet = hits[0]["text"][:400] if hits else "（該当なし）"
        srcs = ", ".join([f"{h['source']} p.{h['page']}" for h in hits])
        return f"{snippet}\n\n[出典: {srcs}]", hits

    # --- LLM要約（キーがあれば） ---
    try:
        from openai import OpenAI
        import os
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("NO_KEY")
        client = OpenAI(api_key=api_key)
        context = "\n\n".join([f"[{i+1}] {h['source']} p.{h['page']}: {h['text']}" for i, h in enumerate(hits)])
        prompt = (
            "次のコンテキストだけを根拠に、質問に日本語で簡潔に答えてください。"
            "最後に出典番号を角括弧で示してください。\n\n"
            f"質問: {query}\n\nコンテキスト:\n{context}"
        )
        chat = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは事実に忠実なアシスタントです。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = chat.choices[0].message.content.strip()
        srcs = ", ".join([f"{h['source']} p.{h['page']}" for h in hits])
        return f"{text}\n\n[出典: {srcs}]", hits
    except Exception:
        snippet = hits[0]["text"][:400] if hits else "（該当なし）"
        srcs = ", ".join([f"{h['source']} p.{h['page']}" for h in hits])
        return f"{snippet}\n\n[出典: {srcs}]", hits

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="最小RAG（1PDF/TXTから回答）")
    ap.add_argument("--rebuild", action="store_true", help="索引を作り直す")
    ap.add_argument("--chunk", type=int, default=700, help="チャンク長（文字）")
    ap.add_argument("--overlap", type=int, default=100, help="オーバーラップ（文字）")
    ap.add_argument("--q", type=str, help="質問文")
    ap.add_argument("--k", type=int, default=3, help="検索件数")
    ap.add_argument("--llm", action="store_true", help="LLMで要約回答（APIキー必須）")
    ap.add_argument("--min-jp-ratio", type=float, default=0.4, help="この比率未満は除外")
    ap.add_argument("--include", type=str, default=None, help="索引対象ファイル名の部分一致/ワイルドカード（例: sample.txt / *.pdf）")
    ap.add_argument("--hybrid", action="store_true", help="埋め込み＋キーワード加点で再ランキング")
    ap.add_argument("--terms", type=str, default="RAG, LLM, 検索, 生成, ベクトル, 埋め込み", help="カンマ区切りの加点キーワード")
    ap.add_argument("--pool", type=int, default=20, help="再ランキング前の候補件数")
    ap.add_argument("--kw-boost", type=float, default=0.3, help="キーワード加点の重み")
    ap.add_argument("--sent-split", action="store_true", help="文単位でチャンク化する（。！？/改行で分割）")
    args = ap.parse_args()

    # 索引（--rebuild）
    if args.rebuild:
        rebuild_index(
            chunk=args.chunk,
            overlap=args.overlap,
            include=args.include,
            min_jp_ratio=args.min_jp_ratio,
            sent_split=args.sent_split,      # ← 渡す
        )

    # 質問（--q）
    if args.q:
        terms = [s.strip() for s in (args.terms or "").split(",") if s.strip()]
        hits = retrieve_hybrid(args.q, k=args.k, terms=terms, pool=args.pool, kw_boost=args.kw_boost) if args.hybrid else retrieve(args.q, k=args.k)
        for i, h in enumerate(hits, start=1):
            extra = f"  [sim={h.get('sim',0):.3f} kw={h.get('kw',0):.1f} score={h.get('score',0):.3f}]" if args.hybrid else ""
            print(f"[{i}] {h['source']} p.{h['page']}  id={h['id']}\n    {h['text'][:160]}...{extra}")
        ans, _ = answer(args.q, k=args.k, use_llm=args.llm, use_hybrid=args.hybrid, terms=terms, pool=args.pool, kw_boost=args.kw_boost)
        print("\n=== answer ===")
        print(ans)

if __name__ == "__main__":
    main()
