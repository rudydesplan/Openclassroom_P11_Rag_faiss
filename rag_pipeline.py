from __future__ import annotations

from langsmith import traceable

from retrieval.langchain_dense_retriever import LCDenseRetriever
from reranking.langchain_reranker import LCReranker
from context.langchain_context_builder import LCContextBuilder
from chat_prompt_template import get_prompt
from llm_provider import get_llm

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import (
    RunnableWithMessageHistory,
    RunnablePassthrough,
    RunnableLambda,
)

# ✅ SINGLE SOURCE OF TRUTH
from retrieval.dense_retriever import get_default_dense_retriever
from confidence_gating import production_confidence_gated_rerank


# -------------------------------------------------------------------------
# RAG pipeline builder (LangChain ≥ 1.0 compliant)
# -------------------------------------------------------------------------
@traceable(name="PulsEvents-RAG-Pipeline")
def get_rag_pipeline(
    retriever_top_k: int = 5,
    reranker_top_k: int = 5,
    max_context_tokens: int = 24000,
    temperature: float = 0.1,
):
    """
    Build a full RAG pipeline with per-query LLM selection.
    """

    # ---------------------------------------------------------
    # 1) Dense Retriever
    # ---------------------------------------------------------
    dense_backend = get_default_dense_retriever()
    dense_backend.top_k = retriever_top_k

    retriever = LCDenseRetriever.from_json_store(
        retriever=dense_backend,
        store_path="outputs/uid_text_store.json",
        top_k=retriever_top_k,
    )

    # ---------------------------------------------------------
    # 2) Reranker
    # ---------------------------------------------------------
    reranker = LCReranker(
        model_name="BAAI/bge-reranker-large",
        device="cpu",
        top_k=reranker_top_k,
    )

    # ---------------------------------------------------------
    # 3) Context Builder
    # ---------------------------------------------------------
    context_builder = LCContextBuilder(
        max_tokens=max_context_tokens,
        include_scores=True,
        top_n=None,
        output_format="toon",
    )

    # ---------------------------------------------------------
    # 4) Prompt
    # ---------------------------------------------------------
    prompt = get_prompt()

    # ---------------------------------------------------------
    # 5) Confidence gating step
    # ---------------------------------------------------------
    def confidence_gate_step(x):
        final_docs, reranker_gated, confidence_gap = (
            production_confidence_gated_rerank(
                dense_docs=x["dense_docs"],
                reranked_docs=x["documents"],
                confidence_threshold=0.05,
            )
        )

        return {
            **x,
            "documents": final_docs,
            "confidence_gap": confidence_gap,
        }

    # ---------------------------------------------------------
    # 6) Dynamic LLM step (per query)
    # ---------------------------------------------------------
    def llm_step(x):
        llm = get_llm(
            model=x["model"],
            temperature=temperature,
            max_tokens=24000,
        )
        return llm.invoke(x["prompt_input"])

    # ---------------------------------------------------------
    # 7) LCEL composition (CORRECT STYLE)
    # ---------------------------------------------------------
    rag_chain = (

        # -------------------------------------------------
        # 1) Dense retrieval
        # -------------------------------------------------
        RunnablePassthrough.assign(
            dense_docs=lambda x: retriever.invoke(x["query"])
        )

        # -------------------------------------------------
        # 2) Reranking
        # -------------------------------------------------
        | RunnablePassthrough.assign(
            documents=lambda x: reranker.invoke({
                "query": x["query"],
                "documents": x["dense_docs"],
            })
        )

        # -------------------------------------------------
        # 3) Confidence gating
        # -------------------------------------------------
        | RunnableLambda(confidence_gate_step)

        # -------------------------------------------------
        # 4) Context building
        # -------------------------------------------------
        | RunnablePassthrough.assign(
            context=lambda x: context_builder.invoke({
                "query": x["query"],
                "documents": x["documents"],
            })
        )

        # -------------------------------------------------
        # 5) Prompt formatting
        # -------------------------------------------------
        | RunnableLambda(
            lambda x: {
                **x,
                "prompt_input": prompt.format(
                    question=x["query"],
                    context=x["context"],
                )
            }
        )

        # -------------------------------------------------
        # 6) Final LLM call (dynamic model)
        # -------------------------------------------------
        | RunnablePassthrough.assign(
            answer=RunnableLambda(llm_step)
        )
    )

    # ---------------------------------------------------------
    # 8) Conversation memory
    # ---------------------------------------------------------
    session_store = {}

    def get_session_history(session_id: str):
        if session_id not in session_store:
            session_store[session_id] = ChatMessageHistory()
        return session_store[session_id]

    conversation_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversation_chain


if __name__ == "__main__":
    pipe = get_rag_pipeline()
