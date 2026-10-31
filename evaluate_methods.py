"""
evaluate_methods.py
====================
Compares two search methods on the same task:
  A) k-mer filter + Needleman-Wunsch  (the app's current method)
  B) ESM2 embedding + cosine similarity

Test: take proteins from the database, randomly damage a fraction of their
amino acids, and check whether each method still finds the original protein.

Run from the project folder:
    python evaluate_methods.py
"""

import random
import time

import os

from database import load_database
from kmer_search import smart_find_best_match
import esm_search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "data", "reference_proteins.fasta")
EXTRA_DATABASE_PATH = os.path.join(BASE_DIR, "data", "insulin_family.fasta")

NUM_PROTEINS = 12                  # proteins to test (more = slower)
MUTATION_LEVELS = [0.2, 0.4, 0.6]  # fraction of letters to damage
MIN_LEN, MAX_LEN = 50, 300         # keep sequences short so NW stays fast
AMINO = "ACDEFGHIKLMNPQRSTVWY"

rng = random.Random(42)  # fixed seed, so results are repeatable


def mutate(sequence, fraction):
    """Replaces a given fraction of positions with a different random amino acid."""
    letters = list(sequence)
    count = int(len(letters) * fraction)
    for pos in rng.sample(range(len(letters)), count):
        letters[pos] = rng.choice([a for a in AMINO if a != letters[pos]])
    return "".join(letters)


database = load_database(DATABASE_PATH, limit=500, extra_fasta_path=EXTRA_DATABASE_PATH)
ids, normalized = esm_search.load_index()

# The saved fingerprints must be in the same order as the database
assert ids == list(database.keys()), "embedding_ids.json does not match the database - rerun build_embeddings.py"

candidates = [pid for pid, seq in database.items() if MIN_LEN <= len(seq) <= MAX_LEN]
chosen = rng.sample(candidates, NUM_PROTEINS)
print(f"Testing {NUM_PROTEINS} proteins at mutation levels {MUTATION_LEVELS}")
print("This takes several minutes (the Needleman-Wunsch part is slow)...\n")

results = {}
start = time.time()
for level in MUTATION_LEVELS:
    nw_correct = 0
    emb_correct = 0
    for pid in chosen:
        query = mutate(database[pid], level)

        nw_id, _, _ = smart_find_best_match(query, database, k=3, top_candidates=20)
        if nw_id == pid:
            nw_correct += 1

        emb_id = esm_search.top_matches(query, ids, normalized, n=1)[0][0]
        if emb_id == pid:
            emb_correct += 1

    results[level] = (nw_correct, emb_correct)
    print(f"  mutation {int(level * 100)}% done ({time.time() - start:.0f}s)")

print()
print("=" * 58)
print(f"{'Letters damaged':<18}{'k-mer + NW':<20}{'Embedding (ESM2)':<20}")
print("-" * 58)
for level, (nw_c, emb_c) in results.items():
    print(f"{int(level * 100):>3}%{'':<14}{nw_c}/{NUM_PROTEINS}{'':<15}{emb_c}/{NUM_PROTEINS}")
print("=" * 58)
print("Each cell = how many of the damaged proteins were correctly identified.")
