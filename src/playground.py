# RAGデバッグ用 playground.py
# 目的: src/rag_baseline.py の各関数を個別に叩いて、printで中身を確認する最小スクリプト
# 使い方: プロジェクトのルート( src/ がある階層 )で以下を実行
#   python playground.py
# 必要に応じて INCLUDE, SAMPLE_QUERY を自分の環境に合わせて変更してください。


#----------------------------------対話型を通すため
# --- 1) ルートに移動して 2) import パスに追加 ---
import os, sys
ROOT = r"C:\Users\81804\Desktop\genai_restart_scaffold_20250811_084323"  # ← プロジェクトのルート
os.chdir(ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import rag_baseline as R  # これで通るはず

#---------------------------------------




from pprint import pprint
from pathlib import Path

# プロジェクト直下で実行している想定 (src/ が隣にある)
from src import rag_baseline as R

# ----------------------- 設定 -----------------------
INCLUDE = None           # 例: "concepts.txt" や "*.pdf" にすると対象を絞れる
SAMPLE_QUERY = "RAGとは何？"  # 適宜変更
# --------------------------------------------------

def hr(title: str):
    print("\n" + "="*12 + f" {title} " + "="*12)


def trunc(s: str, n: int = 160) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "..."


if __name__ == "__main__":
    # 0) 低レベル関数: jp_ratio / is_garbled / split_sentences_jp / chunk_text / _match_include / norm_ws
    hr("jp_ratio / is_garbled")
    t1 = "これはAIのテスト123"
    t2 = "%%%%%"  # 日本語/許容文字が少ない例
    print("t1=", t1)
    print("jp_ratio(t1)=", R.jp_ratio(t1))
    print("is_garbled(t1, 0.4)=", R.is_garbled(t1, 0.4))
    print("t2=", t2)
    print("jp_ratio(t2)=", R.jp_ratio(t2))
    print("is_garbled(t2, 0.4)=", R.is_garbled(t2, 0.4))

    hr("split_sentences_jp / chunk_text / norm_ws")
    long_text = "これはRAGの解説です。検索と生成を組み合わせます！\n改行も正規化されます。文単位の分割も試してみましょう。"
    print("norm_ws(long_text)=", R.norm_ws(long_text))
    print("split_sentences_jp(long_text)=")
    pprint(R.split_sentences_jp(long_text))
    print("chunk_text(long_text, size=16, overlap=4)=")
    pprint(R.chunk_text(long_text, size=16, overlap=4))

#     hr("_match_include")
#     print("_match_include('sample.pdf', None)=", R._match_include('sample.pdf', None))
#     print("_match_include('sample.pdf', 'sample')=", R._match_include('sample.pdf', 'sample'))
#     print("_match_include('sample.pdf', '*.pdf')=", R._match_include('sample.pdf', '*.pdf'))
#     print("_match_include('sample.pdf', '*.txt')=", R._match_include('sample.pdf', '*.txt'))

#     # 1) read_docs: PDF はページ単位, TXT は全文1塊で取得
#     hr("read_docs(include)")
#     try:
#         docs = R.read_docs(include=INCLUDE)
#         print(f"len(docs)={len(docs)}  (各要素: (source, page_or_seq, text))")
#         for i, (src, page, text) in enumerate(docs[:3], start=1):
#             print(f"[{i}] {src} p.{page}: {trunc(text)}")
#     except Exception as e:
#         print("read_docs error:", e)

#     # 2) rebuild_index: 索引を作成（文分割/Sent Splitオンとオフのどちらでも可）
#     hr("rebuild_index (build index)")
#     try:
#         # 少し軽めのパラメータ
#         built = R.rebuild_index(chunk=400, overlap=80, include=INCLUDE, sent_split=False)
#         print("built chunks:", built)
#     except Exception as e:
#         print("rebuild_index error:", e)

#     # 3) retrieve: 上位k件を {id,text,distance,source,page,chunk} 形式で取得
#     hr("retrieve")
#     try:
#         hits = R.retrieve(SAMPLE_QUERY, k=5)
#         print(f"k={len(hits)}")
#         for i, h in enumerate(hits, start=1):
#             print(f"[{i}] {h['source']} p.{h['page']}  id={h['id']}  dist={h.get('distance')}")
#             print("    ", trunc(h['text']))
#     except Exception as e:
#         print("retrieve error:", e)

#     # 4) retrieve_hybrid: 距離正規化 + KW加点で再ランキング
#     hr("retrieve_hybrid")
#     try:
#         terms = ["RAG", "検索", "埋め込み"]
#         hh = R.retrieve_hybrid(SAMPLE_QUERY, k=5, terms=terms, pool=20, kw_boost=0.6)
#         print(f"k={len(hh)}")
#         for i, h in enumerate(hh, start=1):
#             print(f"[{i}] score={h['score']:.3f} sim={h.get('sim',0):.3f} kw={h.get('kw',0):.1f}  {h['source']} p.{h['page']}  id={h['id']}")
#             print("    ", trunc(h['text']))
#     except Exception as e:
#         print("retrieve_hybrid error:", e)

#     # 5) answer: 非LLMモード（本文先頭+出典）を確認
#     hr("answer (non-LLM)")
#     try:
#         ans, used = R.answer(SAMPLE_QUERY, k=5, use_llm=False, use_hybrid=True, terms=["RAG","検索"], pool=20, kw_boost=0.6)
#         print(ans)
#         print("-- used hits --")
#         for i, h in enumerate(used, start=1):
#             print(f"[{i}] {h['source']} p.{h['page']}  id={h['id']}")
#     except Exception as e:
#         print("answer error:", e)

#     # 参考) 既存の索引を直接開く（通常は内部で呼ばれる）
#     hr("_open_collection (reference)")
#     try:
#         coll = R._open_collection()
#         print("collection opened:", coll.name)
#     except Exception as e:
#         print("open_collection error:", e)

#     hr("done")
