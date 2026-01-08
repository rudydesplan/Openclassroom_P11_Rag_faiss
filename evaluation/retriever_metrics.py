from typing import List, Dict, Optional
from langchain_core.documents import Document


def compute_retriever_metrics(
    retrieved_docs: List[Document],
    ground_truth_uid: str,
    k: int,
) -> Dict[str, Optional[float]]:
    """
    Compute Recall@k, Rank, Score Gap for a retriever run.
    """

    uids = [d.metadata["uid"] for d in retrieved_docs]
    scores = [d.metadata["dense_rank_score"] for d in retrieved_docs]

    # --- Recall@k ---
    recall_at_k = 1 if ground_truth_uid in uids[:k] else 0

    # --- Rank ---
    if ground_truth_uid in uids:
        rank = uids.index(ground_truth_uid) + 1  # 1-based
        gt_score = scores[uids.index(ground_truth_uid)]
    else:
        rank = None
        gt_score = None

    # --- Score gap ---
    if gt_score is not None:
        score_gap = scores[0] - gt_score
    else:
        score_gap = None

    return {
        "recall_at_k": recall_at_k,
        "rank": rank,
        "score_gap": score_gap,
    }
