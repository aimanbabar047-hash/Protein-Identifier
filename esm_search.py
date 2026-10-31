"""
esm_search.py
==============
Embedding-based protein search using the ESM2 protein language model.

Needs the files produced by build_embeddings.py:
    data/embeddings.npy       (one 320-number fingerprint per protein)
    data/embedding_ids.json   (protein IDs, in the same order)
"""

import os
import json

# Keep the downloaded model on the D: drive on this machine (saves C: space).
# On other machines this is skipped and the default Hugging Face cache is used.
if os.name == "nt" and os.path.exists("D:\\"):
    os.environ.setdefault("HF_HOME", r"D:\hf_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "data", "embeddings.npy")
IDS_PATH = os.path.join(BASE_DIR, "data", "embedding_ids.json")

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
MAX_LEN = 1022  # ESM2 can read at most 1022 amino acids at a time

_tokenizer = None
_model = None


def _load_model():
    """Loads the ESM2 model the first time it is needed."""
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoTokenizer, EsmModel
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = EsmModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def embed(sequence):
    """Turns one protein sequence into a single 320-number fingerprint."""
    import torch
    tokenizer, model = _load_model()
    inputs = tokenizer(sequence[:MAX_LEN], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Drop the start/end tokens, then average over all amino acids
    return outputs.last_hidden_state[0, 1:-1].mean(dim=0).numpy()


def load_index():
    """Loads the saved database fingerprints. Returns (ids, normalized_matrix)."""
    matrix = np.load(EMBEDDINGS_PATH)
    with open(IDS_PATH) as f:
        ids = json.load(f)
    normalized = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return ids, normalized


def top_matches(query_sequence, ids, normalized, n=5):
    """Returns the n closest database proteins as [(protein_id, cosine_similarity), ...]."""
    q = embed(query_sequence)
    q = q / np.linalg.norm(q)
    similarities = normalized @ q
    best = np.argsort(similarities)[::-1][:n]
    return [(ids[i], float(similarities[i])) for i in best]
