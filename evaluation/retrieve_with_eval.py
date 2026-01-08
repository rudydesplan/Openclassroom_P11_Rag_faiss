from langfuse import get_client, propagate_attributes
from retriever_metrics import compute_retriever_metrics

langfuse = get_client()

def retrieve_with_eval(
    retriever,
    query: str,
    ground_truth_uid: str,
    k: int,
    session_id: str
):
    # Inject session_id into context (doc-approved way)
    with propagate_attributes(session_id=session_id):
        with langfuse.start_as_current_observation(
            name="dense_retrieval",
            as_type="span",
            input={"query": query}
        ) as span:

            docs = retriever.get_relevant_documents(query)

            metrics = compute_retriever_metrics(
                retrieved_docs=docs,
                ground_truth_uid=ground_truth_uid,
                k=k
            )

            # --- Log Recall@k ---
            span.score(
                name="retriever_recall_at_k",
                value=metrics["recall_at_k"],
                data_type="BOOLEAN"
            )

            # --- Log Rank ---
            if metrics["rank"] is not None:
                span.score(
                    name="retriever_rank",
                    value=metrics["rank"],
                    data_type="NUMERIC"
                )

            # --- Log Score gap ---
            if metrics["score_gap"] is not None:
                span.score(
                    name="retriever_score_gap",
                    value=metrics["score_gap"],
                    data_type="NUMERIC"
                )

            span.update(output={
                "retrieved_uids": [d.metadata["uid"] for d in docs]
            })

            return docs
