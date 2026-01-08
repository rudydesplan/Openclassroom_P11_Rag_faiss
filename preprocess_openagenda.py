import sys
import json
import orjson
from loguru import logger
import pandas as pd
from dateutil import parser
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta , timezone
from pathlib import Path

FRENCH_TO_EN = {
    "janvier": "january",
    "février": "february",
    "fevrier": "february",
    "mars": "march",
    "avril": "april",
    "mai": "may",
    "juin": "june",
    "juillet": "july",
    "août": "august",
    "aout": "august",
    "septembre": "september",
    "octobre": "october",
    "novembre": "november",
    "décembre": "december",
    "decembre": "december"
}

def to_local_date(ts):
    #print("to_local_date() INPUT:", ts)

    if pd.isna(ts):
        #print(" → OUTPUT: None (NaT)")
        return None

    out = ts.tz_convert("Europe/Paris").date()
    #print(" → OUTPUT:", out)
    return out


def normalize_french_date(s: str) -> str:
    #print("normalize_french_date() INPUT:", s)

    if not isinstance(s, str):
        #print(" → OUTPUT:", s, "(not a string)")
        return s

    out = s.lower()
    for fr, en in FRENCH_TO_EN.items():
        if fr in out:
            #print(f"   replacing '{fr}' → '{en}'")
            out = out.replace(fr, en)

    #print(" → OUTPUT:", out)
    return out


def parse_french_datetime(date_str: str):
    #print("\nparse_french_datetime() INPUT:", date_str)

    if not isinstance(date_str, str) or not date_str.strip():
        #print(" → OUTPUT: None (empty or non-string)")
        return None

    try:
        norm = normalize_french_date(date_str)
        #print(" normalized:", norm)

        dt = parser.parse(norm, dayfirst=True, fuzzy=True)
        #print(" parsed (naive or aware):", dt)

        # Add timezone if missing
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("Europe/Paris"))
            #print(" added tzinfo (Paris):", dt)

        dt_utc = dt.astimezone(timezone.utc)
        #print(" converted to UTC:", dt_utc)

        return dt_utc

    except Exception as e:
        #print(" → OUTPUT: None (PARSE ERROR)", e)
        return None



def parse_french_datetime_series(series: pd.Series) -> pd.Series:
    #print("\nparse_french_datetime_series() START")
    #print(series)

    def parse_one(val):
        #print("  parse_one() val =", val)
        dt = parse_french_datetime(val)
        #print("   → parsed =", dt)
        return pd.NaT if dt is None else dt

    out = series.apply(parse_one)
    #print("\n  After apply:", out)

    out2 = pd.to_datetime(out, utc=True)
    #print("  After to_datetime (UTC enforced):", out2)

    #print("parse_french_datetime_series() END\n")
    return out2


def filter_by_date(df: pd.DataFrame, now=None) -> pd.DataFrame:
    #print("\n================= FILTER DEBUG START =================")

    initial_count = len(df)

    # Injected test time
    if now is None:
        now = datetime.now(timezone.utc)

    now = pd.Timestamp(now).tz_convert("UTC")
    #print("now =", now, "(UTC)")

    one_year_ago = (now - pd.Timedelta(days=365))
    #print("one_year_ago =", one_year_ago)

    cutoff_date = one_year_ago.date()
    #print("cutoff_date =", cutoff_date)


    #print("one_year_ago =", one_year_ago)
    #print("cutoff_date =", cutoff_date)

    # Parse French dates
    df["firstdate_begin_dt"] = parse_french_datetime_series(df.get("firstdate_begin"))
    df["firstdate_end_dt"]   = parse_french_datetime_series(df.get("firstdate_end"))
    df["lastdate_begin_dt"]  = parse_french_datetime_series(df.get("lastdate_begin"))
    df["lastdate_end_dt"]    = parse_french_datetime_series(df.get("lastdate_end"))

    #print("\n-- Parsed datetimes --")
    #print(df[["firstdate_begin", "firstdate_begin_dt"]])
    #print(df[["firstdate_end", "firstdate_end_dt"]])
    #print(df[["lastdate_begin", "lastdate_begin_dt"]])
    #print(df[["lastdate_end", "lastdate_end_dt"]])

    # Apply priority rules
    df["start_dt"] = df["firstdate_begin_dt"].combine_first(df["lastdate_begin_dt"])
    df["end_dt"]   = df["lastdate_end_dt"].combine_first(df["firstdate_end_dt"])

    #print("\nstart_dt =", df["start_dt"].iloc[0])
    #print("end_dt   =", df["end_dt"].iloc[0])

    # Convert to DATE only (no timezone)
    df["start_date"] = df["start_dt"].apply(to_local_date)
    df["end_date"]   = df["end_dt"].apply(to_local_date)

    #print("\nstart_date =", df["start_date"].iloc[0])
    #print("end_date   =", df["end_date"].iloc[0])

    # Compute KEEP MASK
    cond_end_ok = df["end_date"].notna() & (df["end_date"] >= cutoff_date)
    cond_start_ok = (
        df["end_date"].isna()
        & df["start_date"].notna()
        & (df["start_date"] >= cutoff_date)
    )

    #print("\ncond_end_ok   =", cond_end_ok.iloc[0])
    #print("cond_start_ok =", cond_start_ok.iloc[0])

    keep_mask = cond_end_ok | cond_start_ok

    #print("keep_mask =", keep_mask.iloc[0])

    filtered = df[keep_mask].copy().reset_index(drop=True)

    #print("\n=== FILTERED RESULT ===")
    #print(filtered)
    #print("================= FILTER DEBUG END =================\n")

    return filtered



# ----------------------------------------------------------
# 1. Charger les données JSON
# ----------------------------------------------------------

def load_openagenda_json(json_path: str) -> pd.DataFrame:
    """
    Charge un fichier JSON contenant une liste d'événements OpenAgenda.
    """
    #with open(json_path, "r", encoding="utf-8") as f:
    #    data = json.load(f)

    with open(json_path, "rb") as f:
        data = orjson.loads(f.read())
    
    df = pd.DataFrame(data)
    logger.info(f"{len(data)} événements chargés depuis le fichier JSON.")
    return df

# ----------------------------------------------------------
# 4. Désérialiser les champs JSON internes (status, timings...)
# ----------------------------------------------------------
def safe_json_parse(value):
    if value is None:
        return None
    if isinstance(value, dict) or isinstance(value, list):
        return value
    try:
        return orjson.loads(value)
    except Exception as e:
        logger.warning(f"Élément invalide rencontré lors du parsing JSON pour la valeur: {value}")
        # logger.exception(e) 
        return None

def extract_registration_contact(reg_list):
    """
    Extrait tous les types de contact ('link', 'phone', 'email')
    et les formate pour le RAG (ex: "Lien: URL\nTéléphone: X").
    """
    if not isinstance(reg_list, list):
        return ""
    
    parts = []
    for item in reg_list:
        if isinstance(item, dict) and 'type' in item and 'value' in item:
            # Capitalisation pour une meilleure lisibilité dans le RAG document
            item_type = item['type'].capitalize() 
            item_value = item['value']
            parts.append(f"{item_type}: {item_value}")
            
    return "\n".join(parts)

def deserialize_json_fields(df: pd.DataFrame) -> pd.DataFrame:
    json_fields = ["status", "attendancemode", "timings", "registration"]

    for field in json_fields:
        df[field] = df[field].apply(safe_json_parse)

    # Extraction : status_label_fr
    df["status_label_fr"] = df["status"].apply(
        lambda x: x.get("label", {}).get("fr") if isinstance(x, dict) else None
    )

    # Extraction attendancemode
    df["attendancemode_fr"] = df["attendancemode"].apply(
        lambda x: x.get("label", {}).get("fr") if isinstance(x, dict) else None
    )

    #Extraction des liens d'inscription
    df["registration_contact"] = df["registration"].apply(extract_registration_contact)

    return df

def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace None par chaîne vide pour éviter du bruit dans le RAG.
    """
    df = df.fillna("")
    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    COLUMNS_TO_DROP = ["location_coordinates","status", "accessibility", "attendancemode", "registration", "contributor_email" , "contributor_contactnumber" , "contributor_contactname" , "contributor_contactposition" , "contributor_organization" , "category" ]
    return df.drop(columns=COLUMNS_TO_DROP, errors='ignore')

# ----------------------------------------------------------
# 6. Construire le document RAG
# ----------------------------------------------------------

def build_rag_document(row) -> str:
    """
    Construit un texte riche et cohérent à partir de 56 champs.
    """
    # Gestion propre de l'âge
    age_info = ""
    if row.get('age_min') or row.get('age_max'):
        age_info = f"Public : De {row.get('age_min', '?')} à {row.get('age_max', '?')} ans"

    # --- 1. Logique de sélection de la meilleure image ---
    image_url = ""
    
    if row.get('image'):
        image_url = row['image']
    elif row.get('originalimage'):
        image_url = row['originalimage']
    elif row.get('location_image'):
        image_url = row['location_image']
        
    image_info = ""
    if image_url:
        image_info = f"Image Cover : {image_url}"


    # Création d'un lien Google Maps ---
    maps_link = ""
    coords = row.get('location_coordinates')
    if isinstance(coords, dict) and 'lat' in coords and 'lon' in coords:
        lat = coords['lat']
        lon = coords['lon']
        # Lien universel Google Maps Search
        maps_link = f"Plan d'accès : https://www.google.com/maps/search/?api=1&query={lat},{lon}"

    parts = [
        f"Titre : {row['title_fr']}",
        f"Description : {row['description_fr']}",
        f"Description longue : {row['longdescription_fr']}",
        f"Détail des conditions : {row['conditions_fr']}",
        age_info,
        f"Mots clés : {', '.join(row['keywords_fr']) if isinstance(row['keywords_fr'], list) else row['keywords_fr']}",
        f"Dates : {row['daterange_fr']}",
        f"Horaires détaillés : {row['timings']}",
        f"Mode de participation : {row.get('attendancemode_fr', '')}",
        f"Lien d'accès en ligne : {row['onlineaccesslink']}",
        f"Inscription et contact : {row['registration_contact']}",
        f"Localisation : {row['location_name']} — {row['location_address']} — {row['location_postalcode']} {row['location_city']} — {row['location_department']} ({row['location_region']})",
        maps_link,
        f"Accès transport : {row['location_access_fr']}",
        f"Accessibilité : {row['accessibility_label_fr']}",
        f"Agenda d'origine : {row['originagenda_title']}",
        f"Statut : {row.get('status_label_fr', '')}",
        image_info
    ]

    return "\n".join(p for p in parts if p)

def sanitize_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit toutes les colonnes datetime/ Timestamp en chaînes ISO8601,
    et les dates en 'YYYY-MM-DD'.
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        elif pd.api.types.is_object_dtype(df[col]):
            # Cas possible : datetime.date ou Timestamp isolé
            df[col] = df[col].apply(
                lambda x: x.isoformat() if hasattr(x, "isoformat") else x
            )

    return df


def generate_rag_documents(df: pd.DataFrame):
    """
    Générateur qui produit des dictionnaires (uid, text) prêts pour l'export JSONL.
    Ceci évite de stocker la colonne complète 'rag_document' en mémoire.
    """
    for index, row in df.iterrows():
        # Utilise la fonction build_rag_document existante
        rag_text = build_rag_document(row) 
        
        yield {
            "uid": row["uid"],
            "text": rag_text
        }

# ----------------------------------------------------------
# 7. Export des fichiers finaux
# ----------------------------------------------------------

def export_clean_dataset(df: pd.DataFrame, out_dir="outputs"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Export du dataset complet en JSON...")
    
    df = sanitize_for_json(df)

    # Convert DataFrame → Python list[dict]
    data = df.to_dict(orient="records")

    # Serialize using orjson (UTF-8, pretty)
    json_bytes = orjson.dumps(
        data,
        option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_NON_STR_KEYS
    )

    # Write to disk
    with open(f"{out_dir}/clean_evenements-publics-openagenda.json", "wb") as f:
        f.write(json_bytes)

    logger.success("Export JSON terminé.")

    # JSONL pour FAISS (idx, uid, text) - Utilisation du générateur + ORJSON
    logger.info("Démarrage de l'export JSONL avec générateur (optimisation mémoire).")

    documents_generator = generate_rag_documents(df) 
    
    with open(f"{out_dir}/documents_for_faiss.jsonl", "wb") as f:
        for item in documents_generator:
            # orjson.dumps (très rapide) + OPT_APPEND_NEWLINE (format JSONL)
            f.write(orjson.dumps(item, option=orjson.OPT_APPEND_NEWLINE)) 
            
    logger.success(f"Export JSONL réussi.")

# ----------------------------------------------------------
# FONCTION PRINCIPALE
# ----------------------------------------------------------

def preprocess(json_path: str):
    # Configuration initiale du logger)
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info(f"Démarrage du pré-traitement des données OpenAgenda.")
    logger.info(f"Fichier d'entrée: {json_path}")

    try:
        df = load_openagenda_json(json_path)
        df = filter_by_date(df)
        df = deserialize_json_fields(df)
        df = drop_unnecessary_columns(df)
        df = clean_missing_values(df)
        export_clean_dataset(df)

        logger.success(f"Pré-traitement terminé. {len(df)} événements conservés et exportés.")
        return df

    except Exception as e:
        logger.error(f"Erreur fatale lors du pipeline de pré-traitement: {e}")
        raise

if __name__ == "__main__":
    preprocess("evenements-publics-openagenda.json")
