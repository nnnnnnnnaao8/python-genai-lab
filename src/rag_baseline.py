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
# jp_ratio: 渡された文字列内の「日本語/許容記号」の比率（0〜1）を返すヘルパー。
# この内容は使える文字範囲を指定するための文字クラス、具体的には日本語、英数字、記号類を許可するパターンになっている。
#ちなみにPython では re.compile() を使って、正規表現を パターンオブジェクト にしておくことができます。
#⇒そのオブジェクトには 作成したjp_ratio関数内にある .findall() などのメソッドが生えており、re.findallの形にする必要がない
_JP_OK = re.compile(
    r"[\u3040-\u30FF\u4E00-\u9FFF\u0030-\u0039A-Za-z\s。、，．・：；？！「」『』（）\-\u3000-\u303F\uFF10-\uFF19\uFF21-\uFF3A\uFF41-\uFF5A]"
)
#上記は「日本語文章でよく使う文字だけを許可するフィルター」。
#例えば入力文字列のバリデーションや「変な制御文字・絵文字・特殊記号を弾く」用途でよく使われます。


#日本語率がどれほどかを計算している。
def jp_ratio(text: str) -> float:
    if not text:
        return 0.0
    valid = len(_JP_OK.findall(text)) #findallはreモジュールにある関数で指定した正規表現パターンに
                                       #「マッチする部分文字列」をすべてリストで返すもの
                                       #よってvalidは、引数textのうち日本語、許容記号がどれほどあるかを示す
    return valid / max(1, len(text))

# is_garbled: 制御文字を間引いた上で日本語率が閾値未満かどうかを真偽で返す（ノイズ除外）。
#引数はテキストと、最小日本語率(閾値)設定
def is_garbled(text: str, min_ratio: float = 0.4) -> bool:
    text = re.sub(r"[\u0000-\u001F]", " ", text)  # 制御文字は空白へ \u0000-\u001Fが制御文字　re.subは変数textの制御文字を空白""にという関数
    return jp_ratio(text) < min_ratio

# ---------- 文分割（日本語ざっくり） ----------
# split_sentences_jp: 「。！？」などで大雑把に文分割し、短すぎる断片を除いて文リストを返す。
def split_sentences_jp(text: str) -> list[str]:
    t = re.sub(r"\s+", " ", (text or "").strip())#.strip()は文字列両端の空白改行を削除するメソッド
             #r"\s" → 空白（スペース、タブ、改行など）。「+」は同じ種類の文字がまとめて繰り返されている部分をひとまとまりでとらえるということ
            #従ってやってることは、変数tに引数text or 何もない文""の文字列両端の空白改行を削除した後に、改行と空白⇒" "を何もない⇒""状態にしたものを入れている
    sents = re.findall(r"[^。！？\n]{2,}[。！？]", t) or [] #「少なくとも2文字以上の本文 + 文末記号（。 or ！ or ？）」という“文”っぽい塊を、t からすべて抜き出す。
               #or [] は、findall が None を返すことはないけど、空リストに対して後続で安全に処理したい“保険”としてよく書かれるイディオム(慣用句)。                                            
    tail = re.sub(r".*[。！？]", "", t)
                #r".*[。！？]" は、文字列全体のうち「最後に出現する 文末記号（。！？） まで」を貪欲にマッチします。
                #.* が貪欲（greedy）なので、一番後ろの 。/！/？ まで飲み込みます。
                #それを ""（空文字）に置換するので、**最後の文末記号“より後ろ”にある残り（端切れ）**だけが tail に残る、という仕掛け。
    if len(tail) >= 6:  #端切れが6文字以上のとき、それも文にするよって内容
        sents.append(tail)
    return [s.strip() for s in sents if len(s.strip()) >= 6]#sents内の各文をstrip()して前後の空白削除。6文字未満は除外して返す。最終的に残るのは整形済みの文リスト


# #上記「def split_sentences_jp(text: str) -> list[str]:」について小さな注意点・改善アイデア
# 欧文の文末（. ! ?）は対象外：必要なら [。！？.!?] のように拡張を。
# 連続する記号（例：！！）：現在は1文字ずつ区切るので、意図に応じて {1,2} など調整可。
# 改行は既にスペースに畳んでいる：[^。！？\n] の \n は外しても同じ挙動。
# 貪欲 .* の性質：tail 取得では「最後の」文末まで飛ぶのが狙いどおり。ただ、途中に非常に長いテキストがあっても問題ないかは要件次第。
# パフォーマンス：頻繁に使うなら re.compile() しておくとわずかに効率化。(←いみわかんね)

# ---------- パスやモデル ----------
DOC_DIR   = Path("data/raw/docs")
INDEX_DIR = Path("data/index/rag_demo")
COLL_NAME = "docs"
# 英語軽量: "sentence-transformers/all-MiniLM-L6-v2"
EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 多言語対応

# ---------- 整形＆文字数チャンク ----------
# norm_ws: 連続空白を1つにし前後空白を削って正規化した文字列を返す。
def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# chunk_text: 文字数ベースで size 長・overlap 付きの窓をスライドし、正規化したチャンク文字列リストを返す。
def chunk_text(text: str, size: int = 700, overlap: int = 100) -> List[str]:
    out: List[str] = []    #出力用のリストoutを空で用意する。List[str]は文字列リストで返すよってこと
    i = 0                  
    n = len(text)   #引数で入力された文字列の長さをnに格納
    while i < n:
        out.append(text[i:i+size])      #開始位置iからsize文字分の部分文字列を切り出し
        i += max(size - overlap, 1)     #次のチャンク開始位置を進める、チャンクの幅(オーバーラップ残して進む)デフォルトでは700-100＝600 overlap最小は1
    return [norm_ws(t) for t in out if t.strip()] #if t.strip() → 空白だけのチャンクは除外。 norm_ws(t) → 文字列内の空白を正規化する関数



# ---------- include のマッチ（部分一致 or ワイルドカード）(ファイル名フィルタ関数) ----------
# _match_include: include がワイルドカードなら fnmatch、そうでなければ部分一致でファイル名をフィルタする。
#部分一致はただ含まれているかどうかで、ワイルドカードは
def _match_include(name: str, include: str | None) -> bool: #nameはファイル名 includeは指定したフィルタ条件(ワイルドカードか部分文字列かNone)　戻り値True or False
    if not include: #includeがNoneや空文字なら フィルタ条件なし なので全部マッチ(条件に合致)とみなしてTrue
        return True
    if any(ch in include for ch in "*?[]"): #include に * ? [] のどれかが入っていたら「ワイルドカード指定」だと判定。
                                            # → fnmatch.fnmatch(name, include) でマッチ判定する。
                                            #    fnmatch はシェルのファイルパターン（glob）風にマッチングする関数。
                                           #ちなみに、*..0文字以上の任意の文字列。?..任意の一文字
        return fnmatch.fnmatch(name, include)
    return include.lower() in name.lower() #それ以外の場合、部分一致と判定 大小文字区別せずに includeがnameの中に含まれてればTrue そうでなければFalse

# ---------- 資料取り込み（ページ/大きめチャンク単位まで） ----------
# read_docs: PDF はページ単位、TXT は全文を 1 単位として読み、(source_name, page_or_seq, text) のタプル一覧を返す。
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
# rebuild_index: 資料読込→（文 or 文字）チャンク化→日本語率でスキップ→埋め込み→Chromaに add までを一括実行し、作成チャンク数を返す。
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
# _open_collection: 永続ディレクトリからコレクションを取得して返す（同一埋め込み関数で開く）。
def _open_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    emb = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
    return client.get_collection(COLL_NAME, embedding_function=emb)

# ---------- 検索 ----------
# retrieve: 質問に対し Chroma から上位 k 件を取り出し、{id,text,distance,source,page,chunk} を持つ辞書リストで返す。
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
# _keyword_score: terms 中のキーワードが含まれるたびに1点ずつ（上限5）を加点する単純スコアラー。
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

# retrieve_hybrid: 上位 pool 件の距離を 0〜1 に正規化→キーワード加点×kw_boost を合算→score 降順で並べ替え、上位 k 件を返す。
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
# answer: （非LLM時）先頭ヒット本文の先頭約400文字＋出典一覧の文字列と、採用ヒットのリスト（辞書）を返す；LLM時は要約＋出典文字列。
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
# main: CLI 引数を受け取り、--rebuild で索引再作成、--q で（ハイブリッド可）検索→ヒット表示→answer を出力する実行入口。
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
