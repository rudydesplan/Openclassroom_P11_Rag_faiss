from datetime import datetime, UTC
from pathlib import Path
import re
import orjson
from pathlib import Path

from langfuse import get_client, propagate_attributes

from retrieval.dense_retriever import get_default_dense_retriever
from retrieval.langchain_dense_retriever import LCDenseRetriever
from reranking.langchain_reranker import LCReranker
from context.context_builder import LCContextBuilder

import os

# =================================================
# Langfuse env (cloud or self-hosted)
# =================================================
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-404a98f4-e236-4ecf-956b-25441e740b66"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-8f656873-4f40-4ccc-acab-7419029a3e2b"
os.environ["LANGFUSE_BASE_URL"] = "https://us.cloud.langfuse.com"


def evaluate_reranker_rank_only(
    query: str,
    ground_truth_uid: str,
    reranked_docs,
):
    """
    Logs ONLY reranker_rank into Langfuse.
    """
    with langfuse.start_as_current_span(
        name="reranker_evaluation",
        input={"query": query},
    ) as span:

        rerank_uids = [d.metadata.get("uid") for d in reranked_docs]

        if ground_truth_uid in rerank_uids:
            reranker_rank = rerank_uids.index(ground_truth_uid) + 1
            span.score(
                name="reranker_rank",
                value=reranker_rank,
                data_type="NUMERIC",
            )
        else:
            reranker_rank = None

        span.update(
            output={"reranker_rank": reranker_rank},
            metadata={
                "stage": "reranker",
                "top_k": TOP_K,
            },
        )


langfuse = get_client()

SESSION_ID = f"context_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


# =================================================
# Dataset loading
# =================================================

dataset = []
with DATASET_PATH.open("rb") as f:
    for line in f:
        if line.strip():
            dataset.append(orjson.loads(line))

print(f"Loaded dataset with {len(dataset)} samples")


# =================================================
# Config
# =================================================
TOP_K = 5
MAX_TOKENS = 2500
DANGEROUS_GAP_THRESHOLD = 0.05
LIMIT = 5000


# =================================================
# Helpers
# =================================================
def extract_context_uids(context: str):
    """
    Extract UIDs from TOON-encoded context.
    Simple heuristic: '"uid":"XXXX"' pattern.
    """
    return re.findall(r'\n\s*"(\d+)",', context)


def extract_rerank_scores(docs):
    """
    Returns list of rerank scores aligned with docs.
    """
    return [d.metadata.get("rerank_score", 0.0) for d in docs]


# =================================================
# Context evaluation
# =================================================
def evaluate_context(
    query: str,
    ground_truth_uid: str,
    reranked_docs,
    context_str: str,
):
    with langfuse.start_as_current_span(
        name="context_evaluation",
        input={"query": query},
    ) as span:

        context_uids = extract_context_uids(context_str)

        # -----------------------------
        # 1️⃣ Context Recall
        # -----------------------------
        context_recall = int(ground_truth_uid in context_uids)
        span.score(
            name="context_recall",
            value=context_recall,
            data_type="BOOLEAN",
        )

        # -----------------------------
        # 2️⃣ Context Position
        # -----------------------------
        if ground_truth_uid in context_uids:
            context_position = context_uids.index(ground_truth_uid) + 1
            span.score(
                name="context_position",
                value=context_position,
                data_type="NUMERIC",
            )
        else:
            context_position = None

        # -----------------------------
        # 3️⃣ Dangerous Context Injection
        # -----------------------------
        dangerous = 0

        if context_position is not None:
            rerank_scores = extract_rerank_scores(reranked_docs)

            for idx, doc in enumerate(reranked_docs):
                if doc.metadata.get("uid") == ground_truth_uid:
                    break

                gap = rerank_scores[0] - rerank_scores[idx]
                if gap >= DANGEROUS_GAP_THRESHOLD:
                    dangerous = 1
                    break

        span.score(
            name="dangerous_context_injection",
            value=dangerous,
            data_type="BOOLEAN",
        )

        span.update(
            output={
                "context_recall": context_recall,
                "context_position": context_position,
                "dangerous_context": dangerous,
            },
            metadata={
                "stage": "context",
                "max_tokens": MAX_TOKENS,
                "threshold_gap": DANGEROUS_GAP_THRESHOLD,
            },
        )


# =================================================
# Main loop
# =================================================
def run_eval(dataset):

    dense_backend = get_default_dense_retriever()
    retriever = LCDenseRetriever.from_json_store(
        retriever=dense_backend,
        store_path="outputs/uid_text_store.json",
        top_k=TOP_K,
    )

    reranker = LCReranker(
        model_name="BAAI/bge-reranker-large",
        device="cuda",
        top_k=TOP_K,
    )

    context_builder = LCContextBuilder(
        max_tokens=MAX_TOKENS,
        top_n=TOP_K,
        include_scores=True,
        output_format="toon",
    )

    for i, item in enumerate(dataset[0:LIMIT], start=1):

        query = item["inputs"]["question"]
        gt_uid = item["metadata"]["uid"]

        dense_docs = retriever.invoke(query)
        
        reranked_docs = reranker.invoke(
            {"query": query, "documents": dense_docs}
        )

        context_str = context_builder.invoke(
            {"query": query, "documents": reranked_docs}
        )

        with propagate_attributes(session_id=SESSION_ID):
            with langfuse.start_as_current_span(
                name="context_eval_trace",
                input={"query": query},
            ):

                evaluate_reranker_rank_only(
                    query=query,
                    ground_truth_uid=gt_uid,
                    reranked_docs=reranked_docs,
                )

                
                evaluate_context(
                    query=query,
                    ground_truth_uid=gt_uid,
                    reranked_docs=reranked_docs,
                    context_str=context_str,
                )

        if i % 100 == 0:
            print(f"Processed {i}/{LIMIT}")

    langfuse.flush()
    print("\n✅ Context evaluation completed")
    print(f"Session ID: {SESSION_ID}")
