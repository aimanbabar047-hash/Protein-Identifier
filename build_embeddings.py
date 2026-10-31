"""
build_embeddings.py
====================
1. Computes an ESM2 embedding ("fingerprint") for every protein in the
   reference database and saves them to disk (one-time job).
2. Runs a quick demo: finds the closest database proteins to a query
   using cosine similarity.

Run from the project folder:
    python build_embeddings.py
"""

import os
import json
import time

if os.name == "nt" and os.path.exists("D:\\"):
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")  # keep model files on D: drive
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

from database import load_database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "data", "reference_proteins.fasta")
EXTRA_DATABASE_PATH = os.path.join(BASE_DIR, "data", "insulin_family.fasta")
OUT_EMBEDDINGS = os.path.join(BASE_DIR, "data", "embeddings.npy")
OUT_IDS = os.path.join(BASE_DIR, "data", "embedding_ids.json")

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_LEN = 1022  # ESM2 can read at most 1022 amino acids at a time

print("Loading ESM2 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME)
model.eval()


def embed(sequence):
    """Turns one protein sequence into a single 320-number fingerprint."""
    inputs = tokenizer(sequence[:MAX_LEN], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Drop the start/end tokens, then average over all amino acids
    return outputs.last_hidden_state[0, 1:-1].mean(dim=0).numpy()


# ---------- Part 1: fingerprints for the whole database ----------
database = load_database(DATABASE_PATH, limit=500, extra_fasta_path=EXTRA_DATABASE_PATH)
ids = list(database.keys())
print(f"Embedding {len(ids)} proteins (this can take a few minutes)...")

start = time.time()
vectors = []
for n, protein_id in enumerate(ids, start=1):
    vectors.append(embed(database[protein_id]))
    if n % 50 == 0:
        print(f"  {n}/{len(ids)} done ({time.time() - start:.0f}s)")

matrix = np.vstack(vectors)
np.save(OUT_EMBEDDINGS, matrix)
with open(OUT_IDS, "w") as f:
    json.dump(ids, f)

print(f"Saved embeddings {matrix.shape} to {OUT_EMBEDDINGS}")
print(f"Total time: {time.time() - start:.0f} seconds")

# ---------- Part 2: quick similarity demo ----------
normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def top_matches(query_sequence, n=5):
    """Returns the n database proteins closest to the query (cosine similarity)."""
    q = embed(query_sequence)
    q = q / np.linalg.norm(q)
    similarities = normalized @ q
    best = np.argsort(similarities)[::-1][:n]
    return [(ids[i], float(similarities[i])) for i in best]


mouse_insulin = "MALWMRLLPLLALLALWAPAPTQAFVKQHLCGPHLVEALYLVCGERGFFYTPKSRREVEDPQVAQLELGGGPGAGDLQTLALEVAQQKRGIVDQCCTSICSLYQLENYCN"

print()
print("Demo: query = MOUSE insulin. Closest database proteins by embedding:")
for protein_id, score in top_matches(mouse_insulin, n=5):
    print(f"  {protein_id}   similarity = {score:.4f}")
