import time
from pathlib import Path
import google.generativeai as genai
import orjson

# ===============================================================
# 1. CONFIGURE GEMINI
# ===============================================================

genai.configure(api_key="YOUR_GOOGLE_API_KEY")

model = genai.GenerativeModel("gemini-1.5-flash")  # Fast, cheap, accurate


# ===============================================================
# 2. SYNTHETIC GENERATION PROMPT (optimized for your dataset)
# ===============================================================

SYSTEM_PROMPT = """
You generate evaluation data for a Retrieval-Augmented Generation (RAG) system
that answers questions about public events.

The event description provided follows a semi-structured format with fields such as:
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

Your task:
1. Carefully read ALL the information in the text.
2. Generate ONE natural question that a typical French user might ask.
3. Provide the correct expected_answer STRICTLY based on the text.
4. Do NOT invent any detail not explicitly mentioned.
5. Use ONLY the available information.
6. Focus on realistic questions (dates, location, contact, participation mode, purpose).
7. Output ONLY valid JSON with:
{
  "query": "...",
  "expected_answer": "..."
}
"""

USER_PROMPT_TEMPLATE = """
Voici la description d'un événement :

{context}

Produis un JSON contenant :
- "query" : une question naturelle qu'un utilisateur poserait
- "expected_answer" : la réponse correcte basée uniquement sur le texte fourni
"""


# ===============================================================
# 3. FUNCTION — generate Q/A using Gemini
# ===============================================================

def generate_qa(context: str, retries=3):
    """
    Generate one Q/A pair grounded in the event context.
    Handles Gemini formatting variations (extra text, embedded JSON, etc.).
    Uses orjson for fast parsing.
    """
    prompt = SYSTEM_PROMPT + USER_PROMPT_TEMPLATE.format(context=context)

    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            # Try direct JSON parse
            try:
                data = orjson.loads(text)
                if "query" in data and "expected_answer" in data:
                    return data
            except Exception:
                pass

            # Fallback: extract JSON substring manually
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = orjson.loads(text[start:end])
                return data
            except Exception:
                pass

        except Exception as e:
            print(f"⚠ Gemini error (attempt {attempt+1}): {e}")
            time.sleep(1)

    print("❌ Failed to parse JSON for this event.")
    return None


# ===============================================================
# 4. PROCESS JSONL INPUT → GENERATE SYNTHETIC Q/A → OUTPUT JSONL
# ===============================================================

INPUT_FILE = Path("\outputs\documents_for_faiss.jsonl")
OUTPUT_FILE = Path("\outputs\synthetic_eval_dataset.jsonl")

count = 0
skipped = 0

with INPUT_FILE.open("r", encoding="utf-8") as f_in, \
     OUTPUT_FILE.open("wb") as f_out:  # write binary for orjson

    for line in f_in:
        try:
            item = orjson.loads(line)
        except Exception:
            skipped += 1
            continue

        uid = item.get("uid")
        text = item.get("text", "").strip()

        if not text:
            skipped += 1
            continue

        print(f"\n⏳ Processing UID {uid}...")

        qa = generate_qa(text)

        if qa:
            record = {
                "uid": uid,
                "text": text,
                "query": qa["query"],
                "expected_answer": qa["expected_answer"]
            }

            # Write JSONL line using orjson
            f_out.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
            print(f"✔ OK → generated Q/A for UID {uid}")
            count += 1

        else:
            print(f"❌ Skipped UID {uid}")
            skipped += 1


# ===============================================================
# 5. SUMMARY
# ===============================================================

print("\n==================== DONE ====================")
print(f"✔ Generated Q/A pairs: {count}")
print(f"⚠ Skipped: {skipped}")
print(f"📄 Output file: {OUTPUT_FILE}")
print("===============================================")
