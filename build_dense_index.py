import warnings
warnings.filterwarnings("ignore")

import json
import orjson
import numpy as np
from pathlib import Path
import faiss
from FlagEmbedding import FlagModel
from tqdm import tqdm

# ------------------------
# Paths
# ------------------------

INPUT_PATH = Path("outputs/documents_for_faiss.jsonl")
FAISS_INDEX_PATH = Path("outputs/faiss.index")
FAISS_MAPPING_PATH = Path("outputs/faiss_mapping.json")

# ------------------------
# Model & Config
# ------------------------

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda"                 # GPU
USE_FP16 = True                 # huge speedup on T4
BATCH_SIZE = 256                # GPU can handle large batches
MAX_LENGTH = 2048               # enough for most chunks

# ------------------------
# Utilities
# ------------------------

def load_documents(path):
    """
    Streaming JSONL loader to avoid loading 72k docs in RAM.
    Yields one doc at a time.
    """
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2 normalization for FAISS inner-product index.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


# ------------------------
# Main Builder
# ------------------------

def build_dense_index():
    print("\n--- BUILD DENSE INDEX (BGE-M3) ---")

    # Load dense embedding model
    model = FlagModel(
        MODEL_NAME,
        device=DEVICE,
        use_fp16=USE_FP16
    )
    print(f"Model loaded: {MODEL_NAME}")
    
    model.encode_corpus = model.encode_corpus_single

    uids = []
    dense_index = None
    global_id = 0

    batch_texts = []
    batch_uids = []

    # ----------------------------
    # Stream input file in batches
    # ----------------------------
    for doc in load_documents(INPUT_PATH):
        batch_texts.append(doc["text"])
        batch_uids.append(doc["uid"])

        # Process full batch
        if len(batch_texts) == BATCH_SIZE:
            dense_index = process_batch(
                model, batch_texts, batch_uids, dense_index, global_id
            )

            global_id += len(batch_texts)
            uids.extend(batch_uids)

            batch_texts = []
            batch_uids = []

    # Process last partial batch
    if batch_texts:
        dense_index = process_batch(
            model, batch_texts, batch_uids, dense_index, global_id
        )

        global_id += len(batch_texts)
        uids.extend(batch_uids)

    # ----------------------------
    # Save FAISS index
    # ----------------------------
    print("Saving FAISS index...")
    faiss.write_index(dense_index, str(FAISS_INDEX_PATH))

    # Save UID mapping
    mapping = {"uids": uids}
    FAISS_MAPPING_PATH.write_bytes(
        orjson.dumps(mapping, option=orjson.OPT_INDENT_2)
    )

    print("\n✨ DENSE INDEX READY (FAISS + BGE-M3) ✨\n")


# ----------------------------
# Helper: process batch
# ----------------------------

def process_batch(model, texts, uids_batch, dense_index, start_id):
    print(f"Processing batch at doc {start_id} ({len(texts)} items)")

    # ----- Dense encoding -----
    dense_vecs = model.encode_corpus(
        texts,
        batch_size=len(texts),
        max_length=MAX_LENGTH
    )
    
    dense_vecs = np.array(dense_vecs, dtype="float32")
    dense_vecs = normalize(dense_vecs)

    # ----- Create FAISS index if first batch -----
    if dense_index is None:
        dim = dense_vecs.shape[1]
        print(f"📐 Creating FAISS CPU index dim={dim}")
    
        # Kaggle does not support FAISS GPU → CPU version is perfect
        dense_index = faiss.IndexFlatIP(dim)

    # ----- Add vectors to FAISS -----
    dense_index.add(dense_vecs)

    return dense_index


# ----------------------------
if __name__ == "__main__":
    build_dense_index()
