# langchain_dense_retriever.py

from __future__ import annotations
from typing import List, Dict

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langsmith import traceable

from retrieval.dense_retriever import DenseRetriever
import orjson
from pathlib import Path


class LCDenseRetriever(BaseRetriever):
    """
    LangChain wrapper around the local dense retriever (FAISS + BGE).
    """

    retriever: DenseRetriever
    uid_text_store: Dict[str, str]
    top_k: int = 5

    # --------------------------------------------------------------------
    # UTIL: auto-generate uid_text_store.json if missing
    # --------------------------------------------------------------------
    @staticmethod
    def _ensure_uid_text_store(store_path: Path):
        """
        If outputs/uid_text_store.json does not exist:
        → generate it from outputs/documents_for_faiss.jsonl
        """
        if store_path.exists():
            return

        print(f"[INFO] {store_path} missing. Auto-generating...")

        jsonl_path = Path("outputs/documents_for_faiss.jsonl")
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"[ERROR] Cannot generate {store_path}: {jsonl_path} not found."
            )

        uid_to_text: Dict[str, str] = {}

        with jsonl_path.open("rb") as f:
            for line in f:
                if line.strip():
                    obj = orjson.loads(line)
                    uid_to_text[obj["uid"]] = obj["text"]

        store_path.write_bytes(
            orjson.dumps(uid_to_text, option=orjson.OPT_INDENT_2)
        )

        print(f"[OK] uid_text_store.json generated ({len(uid_to_text)} entries).")

    # --------------------------------------------------------------------
    # FACTORY: load UID → text store
    # --------------------------------------------------------------------
    @classmethod
    def from_json_store(
        cls,
        retriever: DenseRetriever,
        store_path: str,
        top_k: int = 5,
    ) -> "LCDenseRetriever":

        retriever.top_k = top_k
        store_path = Path(store_path)

        cls._ensure_uid_text_store(store_path)

        uid_text_store = orjson.loads(store_path.read_bytes())
        print(f"[INFO] uid_text_store loaded: {len(uid_text_store)} entries.")

        return cls(
            retriever=retriever,
            uid_text_store=uid_text_store,
            top_k=top_k,
        )

    # --------------------------------------------------------------------
    # CORE: LangChain calls THIS via invoke()
    # --------------------------------------------------------------------
    @traceable(name="DenseRetriever")
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:

        results = self.retriever.search(query)

        docs: List[Document] = []
        for r in results[: self.top_k]:
            uid = r["uid"]

            try:
                text = self.uid_text_store[uid]
            except KeyError:
                raise KeyError(
                    f"[ERROR] UID {uid} missing from uid_text_store.json"
                )

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "uid": uid,
                        "dense_rank_score": r["dense_rank_score"],
                        "doc_id": r["doc_id"],
                    },
                )
            )

        return docs

    # --------------------------------------------------------------------
    # ASYNC variant (LCEL-compatible)
    # --------------------------------------------------------------------
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> List[Document]:
        return self._get_relevant_documents(query)


# --------------------------------------------------------------------
# CLI utility
# --------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    store_path = Path("outputs/uid_text_store.json")

    if "--build-store" in sys.argv:
        LCDenseRetriever._ensure_uid_text_store(store_path)
        print("[OK] uid_text_store.json generated via --build-store")
        exit(0)

    print("No action specified.")
    print("Usage:")
    print("  python langchain_dense_retriever.py --build-store")
