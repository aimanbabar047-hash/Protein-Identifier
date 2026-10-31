"""
evaluate_relatives.py
======================
Second test: can each method find RELATIVES (other insulin-family proteins)?

Each insulin-family protein is used as a query, with itself removed from the
database. We then check whether the best matches are other family members.

Run from the project folder:
    python evaluate_relatives.py
"""

import os

from database import load_database
from kmer_search import smart_find_best_match
import esm_search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "data", "reference_proteins.fasta")
EXTRA_DATABASE_PATH = os.path.join(BASE_DIR, "data", "insulin_family.fasta")

database = load_database(DATABASE_PATH, limit=500, extra_fasta_path=EXTRA_DATABASE_PATH)
family = load_database(EXTRA_DATABASE_PATH, limit=None)   # only the insulin family
family_ids = set(family.keys())

ids, normalized = esm_search.load_index()
assert ids == list(database.keys()), "embedding_ids.json does not match the database - rerun build_embeddings.py"

print(f"Insulin family size: {len(family_ids)} proteins")
print(f"Database size: {len(database)} proteins")
print("Each family member is used as a query (itself removed). Takes a minute or two...\n")

nw_top1_hits = 0
emb_top1_hits = 0
emb_top5_total = 0
max_top5_total = 0

print(f"{'Query':<10}{'k-mer+NW best':<18}{'family?':<10}{'Embedding best':<18}{'family?':<10}{'family in emb top-5'}")
print("-" * 82)

for qid, qseq in family.items():
    # Method A: k-mer filter + NW, with the query removed from the database
    db_without = {pid: seq for pid, seq in database.items() if pid != qid}
    nw_best, _, _ = smart_find_best_match(qseq, db_without, k=3, top_candidates=20)
    nw_is_family = nw_best in family_ids

    # Method B: embeddings, dropping the query itself from the ranking
    ranked = [pid for pid, _ in esm_search.top_matches(qseq, ids, normalized, n=len(ids))]
    ranked = [pid for pid in ranked if pid != qid]
    emb_best = ranked[0]
    emb_is_family = emb_best in family_ids
    family_in_top5 = sum(1 for pid in ranked[:5] if pid in family_ids)

    nw_top1_hits += nw_is_family
    emb_top1_hits += emb_is_family
    emb_top5_total += family_in_top5
    max_top5_total += min(5, len(family_ids) - 1)

    print(f"{qid:<10}{nw_best:<18}{'yes' if nw_is_family else 'no':<10}"
          f"{emb_best:<18}{'yes' if emb_is_family else 'no':<10}{family_in_top5}/5")

n = len(family)
print("=" * 82)
print(f"Best match is another family member:  k-mer+NW {nw_top1_hits}/{n}   Embedding {emb_top1_hits}/{n}")
print(f"Embedding: family members in its top 5: {emb_top5_total}/{max_top5_total}")
print("(By pure chance, a random top-5 would contain about 0.1 family members per query.)")
