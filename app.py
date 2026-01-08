import os
import re
from typing import Any, List, Union
import chainlit as cl
from langchain_core.messages import BaseMessage
from chainlit.input_widget import Select, Switch

# ==========================================================
# LangSmith configuration
# ==========================================================
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "pr-overcooked-champagne-38"

# ==========================================================
# RAG imports
# ==========================================================
from rag_pipeline import get_rag_pipeline
from llm_provider import list_available_models

# ==========================================================
# Load RAG pipeline
# ==========================================================
conversation_chain = get_rag_pipeline()

# ==========================================================
# UTILITIES: Robust Text Normalization
# ==========================================================
def normalize_to_text(value: Any) -> str:
    """
    Recursively converts any LCEL output (Str, Message, List, Dict) 
    into a flat string. Prevents the 'List to Str' concatenation error.
    """
    if value is None:
        return ""
    
    # 1. Direct String
    if isinstance(value, str):
        return value
    
    # 2. Lists/Tuples -> Recursive Join
    if isinstance(value, (list, tuple)):
        return "".join(normalize_to_text(v) for v in value)
    
    # 3. LangChain Messages -> Recursive on content (content can be list)
    if isinstance(value, BaseMessage):
        return normalize_to_text(value.content)
    
    # 4. Dictionaries -> Look for standard text keys
    if isinstance(value, dict):
        if "answer" in value:
            return normalize_to_text(value["answer"])
        if "text" in value:
            return normalize_to_text(value["text"])
        # If no known key, ignore it to prevent leaking raw JSON
        return ""

    # 5. Fallback
    return str(value)

# ==========================================================
# UTILITIES: Address & UID Extraction
# ==========================================================
def extract_address_from_content(content: str) -> str:
    """Extracts address from raw page_content if metadata is missing."""
    if not content:
        return "Adresse non précisée"
        
    for line in content.splitlines():
        low = line.lower()
        if low.startswith(("localisation", "adresse", "lieu")):
            parts = line.split(":", 1)
            if len(parts) > 1:
                return parts[1].strip()
    return "Adresse non précisée"

def extract_uids(text: str) -> List[str]:
    """Finds UIDs mentioned in the LLM response."""
    return list(
        set(
            re.findall(
                r"UID[:\s]+([A-Za-z0-9\-]+)",
                text,
                flags=re.IGNORECASE,
            )
        )
    )

# ==========================================================
# EVENT CARD BUILDER
# ==========================================================
def build_event_card(doc) -> str:
    uid = doc.metadata.get("uid", "?")
    title = doc.metadata.get("title", "Événement")
    content = doc.page_content or ""
    
    # Smart address lookup
    address = extract_address_from_content(content)
    if address == "Adresse non précisée":
        address = doc.metadata.get("address", "Adresse inconnue")

    maps_link = doc.metadata.get("maps_link", "")

    return f"""
### 🎫 {title}
**UID :** {uid}  
📍 **Adresse :** {address}

👉 [Ouvrir dans Google Maps]({maps_link})
"""

# ==========================================================
# RESET MEMORY ACTION
# ==========================================================
@cl.action_callback("reset_memory")
async def reset_memory(action: cl.Action):
    session_id = cl.user_session.get("id")
    conversation_chain._message_history_store[session_id] = (
        conversation_chain.message_history_factory(session_id)
    )
    await action.send("🧠 La mémoire a été réinitialisée !")

# ==========================================================
# CHAT START
# ==========================================================
@cl.on_chat_start
async def on_chat_start():
    settings = await cl.ChatSettings(
        [
            Select(
                id="model",
                label="🤖 Modèle LLM",
                values=list_available_models(),
                initial_index=0,
            ),
            Switch(
                id="voice_mode",
                label="🎤 Activer le mode vocal",
                initial=False,
            ),
        ]
    ).send()

    cl.user_session.set("model", settings.get("model", "gemini-2.5-flash"))

    await cl.Message(
        content=(
            "👋 **Bienvenue sur Puls-Events !**\n\n"
            "Sélectionne un modèle LLM puis pose-moi une question.\n"
            "Je te recommanderai des événements pertinents et personnalisés."
        ),
        author="Puls-Events",
        actions=[
            cl.Action(
                name="reset_memory",
                label="🔄 Réinitialiser la mémoire",
                value="reset",
                payload={"value": "reset"},
            )
        ],
    ).send()

# ==========================================================
# SETTINGS UPDATE
# ==========================================================
@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("model", settings.get("model"))

# ==========================================================
# MAIN MESSAGE HANDLER
# ==========================================================
@cl.on_message
async def on_message(message: cl.Message):
    query = message.content
    session_id = cl.user_session.get("id")
    model = cl.user_session.get("model", "gemini-2.5-flash")

    llm_msg = cl.Message(
        content="",
        author=f"Puls-Events ({model})",
    )
    
    generated_text = ""

    # 1. STREAMING RESPONSE
    async for chunk in conversation_chain.astream(
        {"query": query, "model": model},
        config={"configurable": {"session_id": session_id}},
    ):
        # --- SAFE TOKEN EXTRACTION ---
        token = ""
        
        # Case A: String chunk
        if isinstance(chunk, str):
            token = chunk
        
        # Case B: Dict with 'answer' key
        elif isinstance(chunk, dict) and "answer" in chunk:
            token = normalize_to_text(chunk["answer"])
        
        # --- STRICT TYPE CHECK ---
        # Ensure token is strictly a string before adding
        if token and isinstance(token, str):
            generated_text += token
            await llm_msg.stream_token(token)

    await llm_msg.send()

    # 2. EXTRACT UIDS & RETRIEVE DOCS
    uids = extract_uids(generated_text)
    
    if uids:
        # Fetch the context used for this answer using ainvoke
        result = await conversation_chain.ainvoke(
            {"query": query, "model": model},
            config={"configurable": {"session_id": session_id}},
            return_only_outputs=True,
        )
        
        docs = result.get("documents", [])
        docs_by_uid = {doc.metadata.get("uid"): doc for doc in docs}

        # 3. GENERATE CARDS
        for uid in uids:
            doc = docs_by_uid.get(uid)
            if doc:
                await cl.Message(
                    content=build_event_card(doc),
                    author="Événement",
                ).send()