import faiss
import orjson
import numpy as np
from pathlib import Path
from FlagEmbedding import FlagModel

# =====================================================
# 🔒 SINGLE SOURCE OF TRUTH — CONFIG
# =====================================================
DEFAULT_RETRIEVER_CONFIG = dict(
    model_name="BAAI/bge-m3",
    device="cpu",
    faiss_index_path="outputs/faiss.index",
    faiss_mapping_path="outputs/faiss_mapping.json",
    top_k=5,
    max_length=8192,
)

class DenseRetriever:
    """
    Dense retriever using BGE-M3 + FAISS inner-product index.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        faiss_index_path: str,
        faiss_mapping_path: str,
        top_k: int,
        max_length: int,
    ):
        self.top_k = top_k
        self.max_length = max_length

        # -----------------------
        # Load embedding model
        # -----------------------
        self.model = FlagModel(
            model_name,
            device=device,
            use_fp16=False  # CPU-safe
        )

        # -----------------------
        # Load FAISS index
        # -----------------------
        self.dense_index = faiss.read_index(str(faiss_index_path))

        # -----------------------
        # Load mapping
        # -----------------------
        mapping = orjson.loads(Path(faiss_mapping_path).read_bytes())
        self.uids = mapping["uids"]

        print(f"[DenseRetriever] Loaded {len(self.uids)} UIDs")
        print(f"[DenseRetriever] FAISS index dimension: {self.dense_index.d}")


    # -----------------------------------------------------
    # Normalize vectors (cosine similarity via inner product)
    # -----------------------------------------------------
    @staticmethod
    def normalize(v):
        v = np.array(v, dtype="float32")
        n = np.linalg.norm(v)
        return v / max(n, 1e-12)


    # -----------------------------------------------------
    # Main search function
    # -----------------------------------------------------
    def search(self, query: str):
        """
        Returns top_k results ranked by dense similarity only.
        """

        # 1) Encode query into dense vector
        q_vec = self.model.encode_queries(
            [query],
            batch_size=1,
            max_length=self.max_length
        )[0]

        q_vec = self.normalize(q_vec).astype("float32").reshape(1, -1)

        # 2) FAISS search
        dense_scores, dense_ids = self.dense_index.search(q_vec, self.top_k)

        dense_scores = dense_scores[0]
        dense_ids = dense_ids[0]

        # 3) Build result list
        results = []
        for rank, (doc_id, score) in enumerate(zip(dense_ids, dense_scores)):
            uid = self.uids[doc_id]

            results.append({
                "uid": uid,
                "doc_id": int(doc_id),
                "rank": rank,
                "dense_rank_score": float(score)
            })

        return results

def get_default_dense_retriever() -> DenseRetriever:
    """
    Always returns a DenseRetriever with project-wide defaults.
    """
    return DenseRetriever(**DEFAULT_RETRIEVER_CONFIG)

# ---------------------------------------------------------
# Quick CLI for manual testing
# ---------------------------------------------------------
if __name__ == "__main__":
    retriever = get_default_dense_retriever()

    while True:
        q = input("\nQuery > ").strip()
        if not q:
            break

        results = retriever.search(q)

        print("\n--- RESULTS ---")
        for r in results:
            print(
                f"{r['rank'] + 1}. "
                f"UID={r['uid']}  "
                f"score={r['dense_rank_score']:.4f}"
            )