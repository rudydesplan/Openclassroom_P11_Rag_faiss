import os, time
os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
os.environ.pop("GOOGLE_AUTH_CREDENTIALS", None)
os.environ.pop("GOOGLE_CLOUD_PROJECT", None)
os.environ["GOOGLE_API_KEY"] = "AIzaSyAJXVf-S5P04pN4xqoROkWXQyOOh8FZ7Xo"

from typing import Literal , List
import google.genai as genai
from langsmith.wrappers import wrap_gemini
from google.genai.types import HttpOptions
from langchain_google_genai import ChatGoogleGenerativeAI
from google.ai.generativelanguage_v1beta.types import Content


# ---------------------------------------------------------------------
# 1) Liste stricte des modèles disponibles pour le POC Puls-Events
# ---------------------------------------------------------------------
AVAILABLE_MODELS = Literal[
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

# ---------------------------------------------------------------------
# 2) Factory LLM principale
# ---------------------------------------------------------------------
def get_llm(
    model: AVAILABLE_MODELS,
    temperature: float = 0.1,
    max_tokens: int = 24000,
):
    """
    Factory pour instancier proprement un modèle Gemini avec LangChain.

    Paramètres
    ----------
    model : str
        "gemini-2.5-flash" → modèle rapide, fiable, non streaming.

    temperature : float
        Contrôle la créativité (0.0–0.3 recommandé pour RAG factuel).

    max_tokens : int
        Taille maximale de génération. Suffisant pour résumer plusieurs événements.

    Retour
    ------
    ChatGoogleGenerativeAI
        Instance LLM prête à être utilisée dans le pipeline LangChain.
    """

    # Instanciation du modèle Gemini via LangChain
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        streaming=False,   # Flash n’utilise pas le streaming
    )

    return llm

# -------------------------------------------------------------
#   Compteur officiel des tokens Gemini
# -------------------------------------------------------------
def count_gemini_tokens(model: str, text: str) -> int:
    """
    Compte les tokens d'un texte selon la tokenisation officielle du SDK
    google-genai (API v1).
    """
    try:
        client = genai.Client(http_options=HttpOptions(api_version="v1"))
        resp = client.models.count_tokens(model=model, contents=text)
        return resp.total_tokens
    except Exception:
        return len(text.split())  # fallback approximatif

# ---------------------------------------------------------------------
# 3) Benchmark interne complet du modèle
# ---------------------------------------------------------------------
def benchmark_llm(llm):
    """
    Mesure :
      - latence réseau (time-to-first-token)
      - latence totale de génération
      - nombre de tokens générés
      - vitesse en tokens/seconde

    Retourne un dict exploitable dans logs ou monitoring.
    """

    query = (
        "Tu es Gemini 3-pro-preview, presente toi et précise que tu es utilisé "
        "dans un POC Puls-Events. Ne dépasse pas 2 phrases."
    )

    # ---- 1) CALCUL DES TOKENS DU PROMPT ----
    # ---- Tokens du prompt ----
    prompt_tokens = count_gemini_tokens(llm.model, query)

    start = time.time()
    response = llm.invoke(query)
    latency = time.time() - start

    # ---- Tokens de la réponse ----
    response_tokens = count_gemini_tokens(llm.model, response.content)

    tps = response_tokens / latency if latency > 0 else None

    return {
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "latency_total": latency,
        "tps": tps,
        "response": response,
    }

# ---------------------------------------------------------------------
# 3) Helper
# ---------------------------------------------------------------------
def list_available_models() -> List[str]:
    return [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]


if __name__ == "__main__":
    from llm_provider import (
        list_available_models,
        get_llm,
        benchmark_llm,
    )

    models = list_available_models()

    while True:
        print("\n==============================")
        print("Available Gemini models:")
        for i, m in enumerate(models, start=1):
            print(f"{i}. {m}")
        print("q. Quit")
        print("==============================")

        choice = input("Choose a model to test: ").strip()

        if choice.lower() == "q":
            print("Exiting.")
            break

        try:
            model = models[int(choice) - 1]
        except (ValueError, IndexError):
            print("❌ Invalid selection")
            continue

        print(f"\n🔧 Instantiating LLM → {model}")
        llm = get_llm(model=model)

        print("LLM instantiated:", llm)

        try:
            # -------------------------------------------------
            # 1) Connectivity check
            # -------------------------------------------------
            print("\n🔌 Connectivity check...")
            response = llm.invoke(
                "Bonjour ! Es-tu bien connecté ? "
                "Répond simplement : Connexion Gemini OK."
            )
            print("[RESPONSE]", response.content)

            # -------------------------------------------------
            # 2) Benchmark
            # -------------------------------------------------
            print("\n📊 Benchmarking model...\n")

            results = benchmark_llm(llm)

            print("[MODEL RESPONSE]")
            print(results["response"])

            print("\n[STATS]")
            print(f"Prompt tokens        : {results['prompt_tokens']}")
            print(f"Response tokens      : {results['response_tokens']}")
            print(
                f"Total tokens         : "
                f"{results['prompt_tokens'] + results['response_tokens']}"
            )
            print(f"Latency total        : {results['latency_total']:.3f} sec")

            if results["tokens_per_second"] is not None:
                print(
                    f"Tokens / second      : "
                    f"{results['tokens_per_second']:.2f}"
                )

            print("\n=== END BENCHMARK ===\n")

        except Exception as e:
            print("\n❌ ERROR: Unable to call Gemini.")
            print("Details:", e)