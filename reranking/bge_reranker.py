#bge_reranker.py

from typing import List
from langchain_core.documents import Document
from FlagEmbedding import FlagReranker


class BGEReranker:
    """
    Reranker BGE basé sur Cross-Encoder (très haute précision)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = "cpu"):
        self.model = FlagReranker(model_name, device=device)

    def rerank(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """
        Rerank documents based on query relevance using the Cross-Encoder.

        :param query: la question utilisateur
        :param documents: liste de Document (sortis de HybridRetriever)
        :param top_k: optionnel (prend tout si None)
        """

        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]

        # Scores cross-encoder
        scores = self.model.compute_score(pairs)  # renvoie une liste de floats

        # Associer score + document
        ranked = list(zip(documents, scores))

        # Trier par score décroissant
        ranked_sorted = sorted(ranked, key=lambda x: x[1], reverse=True)

        # Couper si demandé
        if top_k is not None:
            ranked_sorted = ranked_sorted[:top_k]

        # Ajouter le score de reranking dans metadata
        output = []
        for doc, score in ranked_sorted:
            md = dict(doc.metadata)
            md["rerank_score"] = float(score)

            output.append(
                Document(
                    page_content=doc.page_content,
                    metadata=md
                )
            )

        return output
