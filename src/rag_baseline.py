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

from pypdf import PdfReader                   # PDFからテキスト抽出（画像PDFは×）
import chromadb                               # ローカル永続のベクトルDB
from chromadb.utils import embedding_functions

# ---------- パスやモデルの設定 ----------
DOC_DIR   = Path("data/raw/docs")             # 読み込む資料の置き場（*.pdf, *.txt）
INDEX_DIR = Path("data/index/rag_demo")       # ベクトル索引の保存先（ローカルフォルダ）
COLL_NAME = "docs"                            # コレクション名（Chromaの論理名）
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 埋め込みモデル（軽量で実用）

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
def read_docs() -> List[Tuple[str, int, str]]:
    """
    DOC_DIR 内のPDF/TXTを読み、(ファイル名, ページ番号, テキスト) のリストにして返す。
    注意：画像だけのPDFは pypdf ではテキストが取れない（OCRが必要）。
    """
    items: List[Tuple[str,int,str]] = []

    # PDFは1ページ単位で読む（ページ番号をメタデータに持たせるため）
    for pdf in sorted(DOC_DIR.glob("*.pdf")):
        r = PdfReader(str(pdf))
        for p, page in enumerate(r.pages, start=1):
            txt = page.extract_text() or ""   # テキストが無ければ空
            if txt.strip():
                items.append((pdf.name, p, norm_ws(txt)))

    # TXTは「仮想的なページ」に分けて扱う（2000文字ごとに区切る）
    for txt in sorted(DOC_DIR.glob("*.txt")):
        t = txt.read_text(encoding="utf-8", errors="ignore")
        # TXTは大きめに割っておき、あとでさらに chunk_text する
        for i, chunk in enumerate(chunk_text(t, size=2000, overlap=0), start=1):
            items.append((txt.name, i, chunk))

    return items

# ---------- 索引の作り直し（Index作成：Transform+Load） ----------
def rebuild_index(chunk: int = 700, overlap: int = 100) -> int:
    """
    すべてのPDF/TXTを読み、チャンク化 → 埋め込み → Chromaへ格納する。
    戻り値：追加したチャンク数（0なら読み取れたテキストが無い可能性）
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))

    # 同名コレクションを一旦削除してから作り直す（再現性のため）
    try:
        client.delete_collection(COLL_NAME)
    except Exception:
        pass

    # 埋め込み関数（SentenceTransformer）を用意
    emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)

    # 新しいコレクション（実データ置き場）を作成
    coll = client.create_collection(COLL_NAME, embedding_function=emb)

    rows = []
    for src, page, text in read_docs():
        # 読み取った本文をさらに細かくチャンク化（検索精度のため）
        for j, t in enumerate(chunk_text(text, size=chunk, overlap=overlap), start=1):
            rid = f"{src}#p{page}#c{j}"       # 一意なID（ファイル名＋ページ＋チャンク番号）
            rows.append((rid, t, {"source": src, "page": page, "chunk": j}))

    if not rows:
        return 0

    # まとめてDBに投入（add時に自動でベクトル化され保存される）
    ids, docs, metas = zip(*rows)
    coll.add(ids=list(ids), documents=list(docs), metadatas=list(metas))
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

# ---------- 回答（Generate） ----------
def answer(query: str, k: int = 3, use_llm: bool = False) -> Tuple[str, List[Dict[str,Any]]]:
    """
    検索→（任意でLLM要約）→ 出典つきの短い回答テキストを返す。
    use_llm=False のときは最上位スニペットをそのまま返す（抽出型）。
    """
    hits = retrieve(query, k=k)
    if not use_llm or not hits:
        # LLMを使わない（またはヒットなし）→ 抽出型で返す
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
    args = ap.parse_args()

    # 索引作成（--rebuild を付けたときだけ実行）
    if args.rebuild:
        n = rebuild_index(chunk=args.chunk, overlap=args.overlap)
        print(f"[index] built {n} chunks at {INDEX_DIR}\\{COLL_NAME}")

    # 質問（--q を付けたときだけ実行）
    if args.q:
        hits = retrieve(args.q, k=args.k)
        for i, h in enumerate(hits, start=1):
            print(f"[{i}] {h['source']} p.{h['page']}  id={h['id']}\n    {h['text'][:160]}...")
        ans, _ = answer(args.q, k=args.k, use_llm=args.llm)
        print("\n=== answer ===")
        print(ans)

if __name__ == "__main__":
    main()