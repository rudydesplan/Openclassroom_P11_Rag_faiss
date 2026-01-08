# test_prompt_template_dynamic.py

from pathlib import Path

from retrieval.dense_retriever import DenseRetriever
from retrieval.langchain_dense_retriever import LCDenseRetriever
from reranking.langchain_reranker import LCReranker
from context.langchain_context_builder import LCContextBuilder
from chat_prompt_template import get_prompt


print("\n=== TEST PROMPT TEMPLATE (DYNAMIQUE, RÉEL, PULS-EVENTS) ===\n")

# ---------------------------------------------------------
# 1) Charger backend dense (FAISS + BGE-m3)
# ---------------------------------------------------------
dense_backend = DenseRetriever(
    faiss_index_path="outputs/faiss.index",
    faiss_mapping_path="outputs/faiss_mapping.json",
    model_name="BAAI/bge-m3",
    device="cpu"
)

store_path = Path("outputs/uid_text_store.jsonl")

# ---------------------------------------------------------
# 2) LangChain retriever wrapper
# ---------------------------------------------------------
lc_retriever = LCDenseRetriever.from_json_store(
    retriever=dense_backend,
    store_path=store_path,
    top_k=10
)

print("[OK] LC Retriever initialisé.\n")

# ---------------------------------------------------------
# 3) Query utilisateur réelle
# ---------------------------------------------------------
query = "événements artistiques pour enfants à Paris"
print(f"[QUERY] {query}\n")

# ---------------------------------------------------------
# 4) Dense search FAISS
# ---------------------------------------------------------
docs = lc_retriever.get_relevant_documents(query)

print(f"[INFO] {len(docs)} documents récupérés.\n")
for i, d in enumerate(docs[:5], start=1):
    print(f"  Doc {i} — UID {d.metadata['uid']} — score dense={d.metadata['dense_rank_score']:.4f}")

# ---------------------------------------------------------
# 5) Reranking (cross-encoder)
# ---------------------------------------------------------
reranker = LCReranker(
    model_name="BAAI/bge-reranker-large",
    device="cpu",
    top_k=5
)

print("\n[CALL] Reranking...\n")

reranked_docs = reranker.invoke({
    "query": query,
    "documents": docs
})

print("\n=== TOP DOCUMENTS APRÈS RERANKING ===\n")
for i, d in enumerate(reranked_docs, start=1):
    print(f"{i}. UID {d.metadata['uid']} — rerank_score={d.metadata.get('rerank_score'):.4f}")

# ---------------------------------------------------------
# 6) Context Builder
# ---------------------------------------------------------
context_builder = LCContextBuilder(
    max_tokens=1500,
    top_n=5,
    include_scores=True
)

print("\n[CALL] Construction du contexte optimisé...\n")

context = context_builder.invoke({
    "query": query,
    "documents": reranked_docs
})

print("[OK] Contexte généré.\n")

# ---------------------------------------------------------
# 7) Compiler le ChatPromptTemplate dynamique
# ---------------------------------------------------------
print("[CALL] Compilation du ChatPromptTemplate...\n")

prompt = get_prompt()

messages = prompt.format_messages(
    context=context,
    question=query
)

# ---------------------------------------------------------
# 8) Affichage final
# ---------------------------------------------------------
print("\n===== SYSTEM MESSAGE =====\n")
print(messages[0].content)

print("\n===== USER MESSAGE =====\n")
print(messages[1].content)

print("\n=== FIN DU TEST CHAT PROMPT TEMPLATE DYNAMIQUE ===\n")
