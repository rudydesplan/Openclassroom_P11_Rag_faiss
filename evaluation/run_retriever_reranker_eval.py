import os
from datetime import datetime, UTC
from pathlib import Path
import orjson

from langfuse import get_client, propagate_attributes

from retriever_metrics import compute_retriever_metrics
from retrieval.dense_retriever import get_default_dense_retriever
from retrieval.langchain_dense_retriever import LCDenseRetriever
from reranking.langchain_reranker import LCReranker


# =================================================
# Langfuse env (cloud or self-hosted)
# =================================================
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-404a98f4-e236-4ecf-956b-25441e740b66"
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-8f656873-4f40-4ccc-acab-7419029a3e2b"
os.environ["LANGFUSE_BASE_URL"] = "https://us.cloud.langfuse.com"

langfuse = get_client()

SESSION_ID = f"retriever_eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"


# =================================================
# Dataset loading (JSONL)
# =================================================
DATASET_PATH = Path("synthetic_eval_dataset/langsmith_eval.jsonl")

dataset = []
with DATASET_PATH.open("rb") as f:
    for line in f:
        if line.strip():
            dataset.append(orjson.loads(line))

print(f"Loaded dataset with {len(dataset)} samples")


# =================================================
# 1️⃣ PURE RETRIEVAL (NO LOGGING)
# =================================================
def retrieve_docs(retriever, query: str):
    docs = retriever.invoke(query)
    return docs


# =================================================
# 2️⃣ RETRIEVER EVALUATION (SPAN)
# =================================================
def evaluate_retriever(docs, query: str, ground_truth_uid: str, k: int):
    with langfuse.start_as_current_span(
        name="dense_retrieval",
        input={"query": query},
    ) as span:

        metrics = compute_retriever_metrics(
            retrieved_docs=docs,
            ground_truth_uid=ground_truth_uid,
            k=k,
        )

        span.score(
            name="retriever_recall_at_k",
            value=metrics["recall_at_k"],
            data_type="BOOLEAN",
        )

        if metrics["rank"] is not None:
            span.score(
                name="retriever_rank",
                value=metrics["rank"],
                data_type="NUMERIC",
            )

        if metrics["score_gap"] is not None:
            span.score(
                name="retriever_score_gap",
                value=metrics["score_gap"],
                data_type="NUMERIC",
            )

        span.update(
            output={"retrieved_uids": [d.metadata["uid"] for d in docs]},
            metadata={"top_k": k, "stage": "retriever"},
        )

        return metrics


# =================================================
# 3️⃣ RERANKER EVALUATION (SPAN)
# =================================================
def evaluate_reranker(
    reranker,
    query: str,
    ground_truth_uid: str,
    dense_docs,
    k: int,
):
    with langfuse.start_as_current_span(
        name="reranker_evaluation",
        input={"query": query},
    ) as span:

        reranked_docs = reranker.invoke(
            {"query": query, "documents": dense_docs}
        )

        rerank_uids = [d.metadata["uid"] for d in reranked_docs]
        rerank_scores = [d.metadata["rerank_score"] for d in reranked_docs]

        if ground_truth_uid in rerank_uids:
            rank_rerank = rerank_uids.index(ground_truth_uid) + 1
            gt_score = rerank_scores[rerank_uids.index(ground_truth_uid)]
            score_gap_rerank = rerank_scores[0] - gt_score
        else:
            rank_rerank = None
            score_gap_rerank = None

        dense_uids = [d.metadata["uid"] for d in dense_docs]
        rank_dense = (
            dense_uids.index(ground_truth_uid) + 1
            if ground_truth_uid in dense_uids
            else None
        )

        delta_rank = (
            rank_dense - rank_rerank
            if rank_dense is not None and rank_rerank is not None
            else None
        )

        if rank_rerank is not None:
            span.score(
                name="reranker_rank",
                value=rank_rerank,
                data_type="NUMERIC",
            )

        if score_gap_rerank is not None:
            span.score(
                name="reranker_score_gap",
                value=score_gap_rerank,
                data_type="NUMERIC",
            )

        if delta_rank is not None:
            span.score(
                name="reranker_delta_rank",
                value=delta_rank,
                data_type="NUMERIC",
            )

        span.update(
            output={
                "dense_rank": rank_dense,
                "rerank_rank": rank_rerank,
                "delta_rank": delta_rank,
            },
            metadata={
                "top_k": k,
                "stage": "reranker",
                "reranker_model": "bge-reranker-large",
            },
        )

        return reranked_docs


# =================================================
# 4️⃣ MAIN BATCH LOOP (TRACE ROOT)
# =================================================
def run_eval(dataset, top_k=5, limit=5000):

    dense_backend = get_default_dense_retriever()
    retriever = LCDenseRetriever.from_json_store(
        retriever=dense_backend,
        store_path="outputs/uid_text_store.json",
        top_k=top_k,
    )

    reranker = LCReranker(
        model_name="BAAI/bge-reranker-large",
        device="cuda",
        top_k=top_k,
    )

    for i, item in enumerate(dataset[:limit], start=1):

        query = item["inputs"]["question"]
        gt_uid = item["metadata"]["uid"]

        dense_docs = retrieve_docs(retriever, query)

        # ✅ Correct v3 pattern:
        # session_id via propagation
        # root span = trace
        with propagate_attributes(session_id=SESSION_ID):
            with langfuse.start_as_current_span(
                name="retrieval_rerank_eval",
                input={"query": query},
            ):
                evaluate_retriever(
                    docs=dense_docs,
                    query=query,
                    ground_truth_uid=gt_uid,
                    k=top_k,
                )

                evaluate_reranker(
                    reranker=reranker,
                    query=query,
                    ground_truth_uid=gt_uid,
                    dense_docs=dense_docs,
                    k=top_k,
                )

        if i % 10 == 0:
            print(f"Processed {i}/{limit}")

    langfuse.flush()
    print("\n✅ Evaluation completed")
    print(f"Session ID: {SESSION_ID}")


# =================================================
# 5️⃣ ENTRY POINT
# =================================================
if __name__ == "__main__":
    run_eval(dataset)
