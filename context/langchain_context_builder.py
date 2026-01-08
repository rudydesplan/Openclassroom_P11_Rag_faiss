# langchain_context_builder.py

from __future__ import annotations
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManagerForChainRun
from langsmith import traceable
# backend context builder local
from context.context_builder import build_context    # :contentReference[oaicite:1]{index=1}


class LCContextBuilder(Runnable):
    """
    Wrapper LangChain pour transformer une liste de Documents
    en un contexte optimisé pour le LLM.

    Input attendu :
        {
            "query": "string",
            "documents": [Document, Document, ...]
        }

    Output :
        String (contexte final)
    """

    def __init__(
        self,
        max_tokens: int = 2400,
        top_n: int | None = None,
        include_scores: bool = True,
        output_format: str = "toon"
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.top_n = top_n
        self.include_scores = include_scores
        self.output_format = output_format

    # ----------------------------------------------------------------------
    # Méthode principale : invoke()  → utilisée par LCEL & LangChain
    # ----------------------------------------------------------------------
    @traceable(name="ContextBuilder")
    def invoke(
        self,
        inputs: Dict[str, Any],
        *,
        config=None,
        run_manager: CallbackManagerForChainRun | None = None
    ) -> str:

        query: str = inputs["query"]
        documents: List[Document] = inputs["documents"]

        if not documents:
            return ""

        # Appel du backend local
        context_str = build_context(
            query=query,
            documents=documents,
            max_tokens=self.max_tokens,
            top_n=self.top_n,
            include_scores=self.include_scores,
            output_format=self.output_format
        )

        return context_str

    # ----------------------------------------------------------------------
    # Batch mode (LCEL compatibility)
    # ----------------------------------------------------------------------
    def batch(
        self,
        inputs_list: List[Dict[str, Any]],
        *,
        config=None,
        run_manager=None
    ) -> List[str]:

        outputs = []
        for inputs in inputs_list:
            outputs.append(self.invoke(inputs, config=config, run_manager=run_manager))
        return outputs

    # ----------------------------------------------------------------------
    # Async mode (LCEL)
    # ----------------------------------------------------------------------
    async def ainvoke(
        self,
        inputs: Dict[str, Any],
        *,
        config=None,
        run_manager=None
    ) -> str:

        return self.invoke(inputs, config=config, run_manager=run_manager)
