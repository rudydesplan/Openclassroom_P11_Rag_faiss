# test_reranker_real.py

from reranking.langchain_reranker import LCReranker
from retrieval.langchain_dense_retriever import LCDenseRetriever
from retrieval.dense_retriever import DenseRetriever
from pathlib import Path

print("\n=== TEST RERANKER BGE (SUR VRAIS DOCUMENTS) ===\n")

# ---------------------------------------------------------
# 1) Charger le backend dense (FAISS + BGE)
# ---------------------------------------------------------
dense_backend = DenseRetriever(
    faiss_index_path="outputs/faiss.index",
    faiss_mapping_path="outputs/faiss_mapping.json",
    model_name="BAAI/bge-m3",
    device="cpu"
)

# ---------------------------------------------------------
# 2) Charger le LangChain retriever
# ---------------------------------------------------------
store_path = Path("outputs/uid_text_store.json")
lc_retriever = LCDenseRetriever.from_json_store(
    retriever=dense_backend,
    store_path=store_path,
    top_k=10
)

print("[OK] LC Retriever initialisé avec FAISS & uid_text_store.json\n")

# ---------------------------------------------------------
# 3) Query réelle
# ---------------------------------------------------------
query = "événements artistiques pour enfants à Paris"
print(f"[QUERY] {query}\n")

# ---------------------------------------------------------
# 4) Récupération *réelle* des documents (dense search)
# ---------------------------------------------------------
docs = lc_retriever.get_relevant_documents(query)

print(f"[INFO] {len(docs)} documents récupérés depuis FAISS.")
for i, d in enumerate(docs[:5], start=1):
    print(f"  Doc {i} – UID {d.metadata['uid']}, score dense={d.metadata['dense_rank_score']:.4f}")

# ---------------------------------------------------------
# 5) Reranker BGE-large
# ---------------------------------------------------------
reranker = LCReranker(
    model_name="BAAI/bge-reranker-large",
    top_k=5
)

print("\n[CALL] Exécution du reranker réel...\n")

ranked_docs = reranker.invoke({
    "query": query,
    "documents": docs
})

# ---------------------------------------------------------
# 6) Affichage du classement final
# ---------------------------------------------------------
print("\n=== RÉSULTATS RERANKING (RÉELS) ===\n")

for i, doc in enumerate(ranked_docs, start=1):
    score = doc.metadata.get("rerank_score", None)
    uid = doc.metadata.get("uid", "UNKNOWN")

    print(f"{i}. Score = {score:.4f}")
    print(f"   UID: {uid}")
    print(f"   Excerpt: {doc.page_content[:200]}...\n")

print("\n=== FIN DU TEST RERANKER ===\n")
