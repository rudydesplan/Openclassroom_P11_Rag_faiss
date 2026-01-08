# context_builder.py

from langchain_core.documents import Document
from typing import List, Tuple, Dict, Any
import re
import tiktoken
from toon_format import encode
from langsmith import traceable

# --- Tokenizer OpenAI-like (fonctionne pour Gemma/Llama aussi pour estimation) ---
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Compte tokens approximatifs → utile pour trimming."""
    return len(_tokenizer.encode(text))


def trim_to_token_limit(text: str, max_tokens: int) -> str:
    """Coupe proprement un texte trop long selon un budget de tokens."""
    tokens = _tokenizer.encode(text)

    if len(tokens) <= max_tokens:
        return text

    clipped = tokens[:max_tokens]
    return _tokenizer.decode(clipped)


# ============================================================
# Configuration (best practice RAG)
# ============================================================

# Portion du budget tokens réservée au top-1 reranké
TOP_DOC_TOKEN_RATIO = 0.42   # 35% du budget total


# ============================================================
# Utilitaires internes
# ============================================================

def clean_text(text: str) -> str:
    """Nettoyage léger du texte."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def deduplicate_documents(
    docs: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Supprime les doublons exacts de contenu.
    Préserve l'ordre.
    """
    seen = set()
    out = []
    for text, md in docs:
        key = text.strip()
        if key and key not in seen:
            seen.add(key)
            out.append((text, md))
    return out


# ============================================================
# Fonction principale
# ============================================================

def build_context(
    query: str,
    documents: List[Any],
    max_tokens: int = 2400,
    top_n: int = 5,
    include_scores: bool = True,
    output_format: str = "toon",
) -> str:
    """
    Construit un contexte optimisé avec :
    - injection garantie du document reranké top-1
    - budget tokens réservé pour ce document
    - troncature contrôlée
    """

    if not documents:
        return encode({"documents": []})

    # --------------------------------------------------------
    # 1️⃣ Préparer (text, metadata)
    # --------------------------------------------------------
    cleaned_docs: List[Tuple[str, Dict[str, Any]]] = []

    for d in documents[:top_n]:
        text = clean_text(d.page_content)
        md = dict(d.metadata or {})
        cleaned_docs.append((text, md))

    # --------------------------------------------------------
    # 2️⃣ Déduplication
    # --------------------------------------------------------
    cleaned_docs = deduplicate_documents(cleaned_docs)

    if not cleaned_docs:
        return encode({"documents": []})

    # --------------------------------------------------------
    # 3️⃣ Séparer top-1 reranké
    # --------------------------------------------------------
    # Hypothèse valide dans ton pipeline :
    # les documents arrivent déjà triés par reranker
    top_doc = cleaned_docs[0]
    other_docs = cleaned_docs[1:]

    # --------------------------------------------------------
    # 4️⃣ Allocation du budget tokens
    # --------------------------------------------------------
    top_doc_budget = int(max_tokens * TOP_DOC_TOKEN_RATIO)
    remaining_budget = max_tokens - top_doc_budget

    # Sécurité
    top_doc_budget = max(64, top_doc_budget)
    remaining_budget = max(0, remaining_budget)

    per_other_limit = (
        remaining_budget // max(1, len(other_docs))
        if other_docs
        else 0
    )

    # --------------------------------------------------------
    # 5️⃣ Construction des documents TOON
    # --------------------------------------------------------
    documents_out = []

    # --- Top-1 garanti ---
    top_text, top_md = top_doc
    documents_out.append({
        "uid": top_md.get("uid", ""),
        "dense_score": top_md.get("dense_rank_score") if include_scores else None,
        "rerank_score": top_md.get("rerank_score") if include_scores else None,
        "content": trim_to_token_limit(top_text, top_doc_budget),
    })

    # --- Autres documents ---
    for text, md in other_docs:
        documents_out.append({
            "uid": md.get("uid", ""),
            "dense_score": md.get("dense_rank_score") if include_scores else None,
            "rerank_score": md.get("rerank_score") if include_scores else None,
            "content": trim_to_token_limit(text, per_other_limit),
        })

    # --------------------------------------------------------
    # 6️⃣ Encodage final
    # --------------------------------------------------------
    return encode({"documents": documents_out})
