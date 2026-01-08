import os
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-9f8086e2-07c2-485a-81ff-4dbf49dfedd0"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-95a05033-df84-44b6-b66b-90b2fe405d24"
os.environ["LANGFUSE_BASE_URL"] = "https://cloud.langfuse.com"

from datetime import datetime, UTC
SESSION_ID = f"retriever_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

import orjson
from pathlib import Path
from langfuse import get_client
from retrieval.dense_retriever import get_default_dense_retriever
from retrieval.langchain_dense_retriever import LCDenseRetriever
from retriever_metrics import compute_retriever_metrics
from retrieve_with_eval import retrieve_with_eval
from langfuse import Langfuse


DATASET_PATH = Path("synthetic_eval_dataset/langsmith_eval.jsonl")

dataset = []
with DATASET_PATH.open("rb") as f:
    for line in f:
        if line.strip():
            dataset.append(orjson.loads(line))

# -------------------------------------------------
langfuse = get_client()

# -------------------------------------------------
# Build retriever (same as your RAG pipeline)
# -------------------------------------------------
dense_backend = get_default_dense_retriever()

retriever = LCDenseRetriever.from_json_store(
    retriever=dense_backend,
    store_path="outputs/uid_text_store.json",
    top_k=5
)

# -------------------------------------------------
# One evaluation example
# -------------------------------------------------
for i, item in enumerate(dataset[:10000], start=1):
    retrieve_with_eval(
        retriever=retriever,
        query=item["inputs"]["question"],
        ground_truth_uid=item["metadata"]["uid"],
        k=5,
        session_id=SESSION_ID
    )

    if i % 10 == 0:
        print(f"Processed {i}/10000")

langfuse.flush()
