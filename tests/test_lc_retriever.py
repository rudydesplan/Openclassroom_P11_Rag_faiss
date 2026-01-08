# test_lc_retriever.py

from retrieval.langchain_dense_retriever import LCDenseRetriever
from retrieval.dense_retriever import DenseRetriever

# 1) Backend FAISS local
dense = DenseRetriever(
    model_name="BAAI/bge-m3",
    device="cpu",
    top_k=5
)

# 2) Wrapper LangChain
lc_ret = LCDenseRetriever.from_json_store(
    retriever=dense,
    store_path="outputs/uid_text_store.json",
    top_k=5
)

query = "atelier pour enfants à Paris"

print("OK – Avant get_relevant_documents")

docs = lc_ret.get_relevant_documents(query)

print("OK – Après get_relevant_documents")
print("\n===== DOCS =====\n")

for d in docs:
    print("--- Document ---")
    print("UID:", d.metadata["uid"])
    print("Score:", d.metadata["dense_rank_score"])
    print("Excerpt:", d.page_content[:200])
    print()
