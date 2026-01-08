# langchain_reranker.py

from __future__ import annotations
from typing import List

from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.runnables import Runnable
from langsmith import traceable
from reranking.bge_reranker import BGEReranker  # backend local (cross-encoder)
                                      # :contentReference[oaicite:1]{index=1}


class LCReranker(Runnable):
    """
    Wrapper LangChain autour de BGEReranker.
    
    Prend :
        - input: { "query": str, "documents": List[Document] }
    Retourne :
        - List[Document] rerankés, enrichis avec rerank_score.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = "cpu", top_k: int = None):
        super().__init__()
        self.reranker = BGEReranker(model_name=model_name, device=device)
        self.top_k = top_k  # nombre de documents à conserver après rerank

    # ----------------------------------------------------------------------
    # LCEL : méthode principale
    # ----------------------------------------------------------------------
    @traceable(name="BGE-Reranker")
    def invoke(
        self,
        inputs: dict,
        *,
        config=None,
        run_manager: CallbackManagerForChainRun | None = None
    ) -> List[Document]:

        query: str = inputs["query"]
        docs: List[Document] = inputs["documents"]

        if not docs:
            return []

        reranked_docs = self.reranker.rerank(
            query=query,
            documents=docs,
            top_k=self.top_k
        )

        return reranked_docs

    # ----------------------------------------------------------------------
    # Batch mode (LCEL compatibility)
    # ----------------------------------------------------------------------
    def batch(
        self,
        inputs_list: List[dict],
        *,
        config=None,
        run_manager=None
    ) -> List[List[Document]]:

        outputs = []
        for inputs in inputs_list:
            outputs.append(self.invoke(inputs, config=config, run_manager=run_manager))
        return outputs

    # ----------------------------------------------------------------------
    # Async compatibility
    # ----------------------------------------------------------------------
    async def ainvoke(
        self,
        inputs: dict,
        *,
        config=None,
        run_manager=None
    ) -> List[Document]:
        return self.invoke(inputs, config=config, run_manager=run_manager)

if __name__ == "__main__":
    print("\n[SELF-TEST] Chargement du modèle BGE Reranker...\n")

    reranker = LCReranker(
        model_name="BAAI/bge-reranker-large",
        device="cpu",
        top_k=5
    )

    print("[OK] Modèle 'bge-reranker-large' chargé correctement.\n")
    print("Utilisez maintenant :  python test_reranker_real.py\n")