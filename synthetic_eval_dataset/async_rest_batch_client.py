import asyncio
import requests
import orjson
import os
import time
import random

API_KEY = "AIzaSyAJXVf-S5P04pN4xqoROkWXQyOOh8FZ7Xo"

ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{}:batchGenerateContent?key={}"
)

MAX_RETRIES = 3

SYSTEM_PROMPT = """
Tu génères des données d’évaluation pour un système de Retrieval-Augmented Generation (RAG)
qui répond à des questions sur des événements publics.
Pour chaque événement, génère EXACTEMENT 3 questions-réponses différentes.

La description d’événement fournie suit un format semi-structuré avec des champs tels que :
- Titre :
- Description :
- Description longue :
- Détail des conditions :
- Mots clés :
- Dates :
- Horaires détaillés :
- Mode de participation :
- Lien d'accès en ligne :
- Inscription et contact :
- Localisation :
- Accès transport :
- Accessibilité :
- Agenda d'origine :
- Statut :

Ta tâche :
1. Lire attentivement TOUTES les informations présentes dans le texte.
2. Générer 3 questions naturelles qu’un utilisateur français typique pourrait poser sur cet événement.
3. Fournir les expected_answer STRICTEMENT basées sur les informations présentes dans le texte.
4. Ne JAMAIS inventer un détail qui n’est pas explicitement mentionné.
5. Utiliser UNIQUEMENT les champs présents dans le texte fourni.
6. Te concentrer sur des questions réalistes concernant :
   - la date / l’horaire
   - le lieu
   - le mode de participation
   - les informations de contact ou d’inscription
   - les détails, objectifs ou mots-clés de l’événement
7. Produire UNIQUEMENT un JSON valide au format :
{
  "query": [
    "question 1",
    "question 2",
    "question 3"
  ],
  "expected_answer": [
    "réponse 1",
    "réponse 2",
    "réponse 3"
  ]
}

RÈGLES DE FORMATAGE IMPORTANTES :
- Les deux tableaux doivent avoir EXACTEMENT la même longueur.
- Ne JAMAIS afficher ```json ni aucun formatage Markdown.
- Ne JAMAIS entourer le JSON avec des blocs de code.
- Produire UNIQUEMENT du JSON brut — rien avant, rien après.

Langue : Francais
"""

USER_TEMPLATE = """
Voici la description d'un événement :

{context}

Produis un JSON contenant :
{{
  "query": [
    "question 1",
    "question 2",
    "question 3",
    "question 4",
    "question 5"
  ],
  "expected_answer": [
    "réponse 1",
    "réponse 2",
    "réponse 3",
    "réponse 4",
    "réponse 5"
  ]
}}
"""


class AsyncRestBatchClient:

    def __init__(self, model="models/gemini-2.5-flash-preview-09-2025"):
        self.model = model
        self.url = ENDPOINT_TEMPLATE.format(model, API_KEY)

    def _build_requests(self, texts):
        out = []
        for t in texts:
            out.append({
                "contents": [
                    {"role": "system", "parts": [{"text": SYSTEM_PROMPT}]},
                    {"role": "user",   "parts": [{"text": USER_TEMPLATE.format(context=t)}]},
                ]
            })
        return out

    async def batch_generate(self, texts):
        return await asyncio.to_thread(self._run_sync_batch, texts)

    def _run_sync_batch(self, texts):
        body = {"requests": self._build_requests(texts)}

        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.post(
                    self.url,
                    data=orjson.dumps(body),
                    headers={"Content-Type": "application/json"},
                    timeout=180,
                )

                if resp.status_code != 200:
                    print("\n🔴 SERVER ERROR RESPONSE:")
                    print("Status:", resp.status_code)
                    print("Headers:", resp.headers)
                    print("Body:", resp.text)
                    raise RuntimeError(f"HTTP {resp.status_code}")

                data = resp.json()
                results = []

                for item in data.get("responses", []):
                    cand = item.get("candidates", [])
                    if not cand:
                        results.append(None)
                        continue

                    text_output = cand[0]["content"]["parts"][0]["text"]

                    try:
                        start = text_output.index("{")
                        end = text_output.rindex("}") + 1
                        results.append(orjson.loads(text_output[start:end]))
                    except Exception:
                        results.append(None)

                return results

            except Exception as e:
                wait = (attempt + 1) * random.uniform(1.2, 2.0)
                print(f"⚠ REST batch error: {e} → retry in {wait:.1f}s")
                time.sleep(wait)

        print("❌ Too many batch retries")
        return [None] * len(texts)