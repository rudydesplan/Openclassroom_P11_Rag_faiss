import pytest
from langchain_core.documents import Document
from context.context_builder import build_context
from toon_format import encode,decode


def test_context_builder_outputs_valid_toon():
    # --- 1) Préparer des faux documents ---
    docs = [
        Document(
            page_content="Atelier artistique pour enfants",
            metadata={
                "uid": "EVT-001",
                "dense_rank_score": 0.9123,
                "rerank_score": 0.8877,
            },
        ),
        Document(
            page_content="Spectacle de marionnettes à Paris",
            metadata={
                "uid": "EVT-002",
                "dense_rank_score": 0.8321,
                "rerank_score": 0.8012,
            },
        ),
    ]

    # --- 2) Appel du context builder en mode TOON ---
    toon_context = build_context(
        query="événements pour enfants",
        documents=docs,
        max_tokens=2000,
        top_n=None,
        include_scores=True,
        output_format="toon",
    )

    # --- 3) Vérification structurelle : doit commencer par "documents[" ---
    assert toon_context.startswith("documents["), "Le contexte TOON doit commencer par un header TOON."

    # --- 4) Vérification des UIDs ---
    assert "EVT-001" in toon_context
    assert "EVT-002" in toon_context

    # --- 5) Vérification des scores ---
    assert "0.9123" in toon_context
    assert "0.8877" in toon_context

    # --- 6) Vérification : decode() doit retransformer en Python ---
    decoded = decode(toon_context)
    assert isinstance(decoded, dict)
    assert "documents" in decoded
    assert len(decoded["documents"]) == 2

    # --- 7) Vérification contenu décodé ---
    row1 = decoded["documents"][0]
    assert row1["uid"] == "EVT-001"
    assert row1["dense_score"] == 0.9123
    assert row1["rerank_score"] == 0.8877
    assert "Atelier artistique" in row1["content"]


    # --- 8) Vérification : pas de Markdown dans le contexte ---
    assert "### DOCUMENT" not in toon_context, "Le contexte ne doit plus contenir de Markdown."


    print("\n[OK] Test TOON : le context_builder génère un TOON valide et décodable.")