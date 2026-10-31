"""
kmer_search.py
===============
Fast pre-filtering using k-mer (Jaccard) similarity, followed by
accurate Needleman-Wunsch alignment on the top candidates only.
This mirrors the "seed and extend" strategy used by tools like BLAST.
"""

from nw_algorithm import needleman_wunsch


def get_kmers(sequence, k=3):
    """Returns the set of all overlapping k-length substrings of a sequence."""
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmers.add(sequence[i:i + k])
    return kmers


def kmer_similarity(seq1, seq2, k=3):
    """
    Computes normalized (Jaccard) k-mer similarity between two sequences.
    Returns a value between 0 (no overlap) and 1 (identical k-mer sets).
    """
    kmers1 = get_kmers(seq1, k)
    kmers2 = get_kmers(seq2, k)

    common_kmers = kmers1 & kmers2
    total_kmers = kmers1 | kmers2

    if len(total_kmers) == 0:
        return 0

    return len(common_kmers) / len(total_kmers)


def smart_find_best_match(query_sequence, database, k=3, top_candidates=20):
    """
    Two-step search:
      1. Rank all database proteins by fast k-mer similarity.
      2. Run accurate Needleman-Wunsch only on the top candidates.

    Parameters
    ----------
    query_sequence : str
    database : dict {protein_id: sequence}
    k : int - k-mer size
    top_candidates : int - number of top candidates to run NW on

    Returns
    -------
    best_id : str
    best_score : int
    best_alignment : tuple (aligned_seq1, aligned_seq2)
    """

    # Step A: fast k-mer similarity ranking
    similarity_scores = []
    for protein_id, db_sequence in database.items():
        sim_score = kmer_similarity(query_sequence, db_sequence, k=k)
        similarity_scores.append((protein_id, sim_score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarity_scores[:top_candidates]

    # Step B: accurate NW alignment on top candidates only
    best_score = float('-inf')
    best_id = None
    best_alignment = None

    for protein_id, _ in top_matches:
        db_sequence = database[protein_id]
        aligned1, aligned2, score, _, _ = needleman_wunsch(query_sequence, db_sequence)

        if score > best_score:
            best_score = score
            best_id = protein_id
            best_alignment = (aligned1, aligned2)

    return best_id, best_score, best_alignment
