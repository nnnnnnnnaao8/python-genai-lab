#%% パラメータ
Q = "RAGとは何？"
K = 3

#%% 索引を作る（最初だけ）
from src.rag_baseline import rebuild_index
print("chunks:", rebuild_index(chunk=700, overlap=100))

#%% 検索→回答（抽出型）
from src.rag_baseline import retrieve, answer
hits = retrieve(Q, k=K)
hits[:2]

#%% LLMで要約（APIキーあれば）
ans, _ = answer(Q, k=K, use_llm=True)
print(ans)
