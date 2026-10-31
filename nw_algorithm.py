"""
nw_algorithm.py
================
Needleman-Wunsch Global Alignment Algorithm.
Contains the core function used to align two protein sequences.
"""

import numpy as np


def needleman_wunsch(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-2):
    """
    Globally aligns two sequences using the Needleman-Wunsch algorithm.

    Parameters
    ----------
    seq1, seq2 : str
        The two sequences to align.
    match_score : int
        Score awarded when two characters match.
    mismatch_score : int
        Score awarded when two characters differ.
    gap_penalty : int
        Score penalty for inserting a gap.

    Returns
    -------
    aligned_seq1, aligned_seq2 : str
        The aligned sequences (with '-' representing gaps).
    final_score : int
        The optimal alignment score.
    score_matrix : np.ndarray
        The full dynamic programming score matrix.
    traceback_matrix : np.ndarray
        Matrix of directions ('D', 'U', 'L') used to reconstruct the alignment.
    """

    n = len(seq1)
    m = len(seq2)

    def compare_score(char1, char2):
        return match_score if char1 == char2 else mismatch_score

    # Step 1: Initialize matrices
    score_matrix = np.zeros((n + 1, m + 1), dtype=int)
    traceback_matrix = np.full((n + 1, m + 1), '', dtype=object)

    # Step 2: Fill base case (first row/column)
    for i in range(1, n + 1):
        score_matrix[i][0] = score_matrix[i - 1][0] + gap_penalty
        traceback_matrix[i][0] = 'U'

    for j in range(1, m + 1):
        score_matrix[0][j] = score_matrix[0][j - 1] + gap_penalty
        traceback_matrix[0][j] = 'L'

    # Step 3: Fill the rest of the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = score_matrix[i - 1][j - 1] + compare_score(seq1[i - 1], seq2[j - 1])
            up = score_matrix[i - 1][j] + gap_penalty
            left = score_matrix[i][j - 1] + gap_penalty

            best_score = max(diagonal, up, left)
            score_matrix[i][j] = best_score

            if best_score == diagonal:
                traceback_matrix[i][j] = 'D'
            elif best_score == up:
                traceback_matrix[i][j] = 'U'
            else:
                traceback_matrix[i][j] = 'L'

    # Step 4: Traceback to reconstruct the alignment
    aligned_seq1 = ""
    aligned_seq2 = ""

    i, j = n, m
    while i > 0 or j > 0:
        direction = traceback_matrix[i][j]

        if direction == 'D':
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            i -= 1
            j -= 1
        elif direction == 'U':
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            i -= 1
        else:  # 'L'
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            j -= 1

    final_score = score_matrix[n][m]

    return aligned_seq1, aligned_seq2, final_score, score_matrix, traceback_matrix
