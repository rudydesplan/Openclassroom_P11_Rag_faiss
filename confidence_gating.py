# confidence_gating.py

def production_confidence_gated_rerank(
    dense_docs,
    reranked_docs,
    confidence_threshold=0.05,
):
    """
    Production-safe confidence gating (NO ground truth).

    Returns:
      final_docs: list
      reranker_gated: bool
      confidence_gap: float | None
    """

    if len(reranked_docs) < 2:
        return reranked_docs, False, None

    scores = [d.metadata.get("rerank_score", 0.0) for d in reranked_docs]
    score_gap = scores[0] - scores[1]

    dense_top_uid = dense_docs[0].metadata.get("uid")
    rerank_top_uid = reranked_docs[0].metadata.get("uid")

    reranker_disagrees = dense_top_uid != rerank_top_uid

    # 🚦 Gate if confident AND disagreement
    if score_gap >= confidence_threshold and reranker_disagrees:
        return dense_docs[:len(reranked_docs)], True, score_gap

    return reranked_docs, False, score_gap
