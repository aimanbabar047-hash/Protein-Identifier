"""
database.py
============
Loads the reference protein database from a FASTA file into a
simple dictionary: {protein_id: sequence}
"""

from Bio import SeqIO


def load_database(fasta_path, limit=500, extra_fasta_path=None):
    """
    Reads a FASTA file and returns a dictionary of protein_id -> sequence.

    Parameters
    ----------
    fasta_path : str
        Path to the main FASTA file (e.g. downloaded from UniProt).
    limit : int
        Maximum number of sequences to load from the main file (keeps
        things fast for a student project / demo; set to None to load
        everything).
    extra_fasta_path : str or None
        Optional path to a second, smaller FASTA file whose sequences
        are always included in full (no limit applied) -- useful for
        adding a handful of biologically relevant proteins (e.g. the
        insulin family) so the demo has some meaningful matches, not
        just random reviewed proteins.

    Returns
    -------
    dict {protein_id: sequence}
    """
    protein_database = {}

    for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
        if limit is not None and i >= limit:
            break

        # UniProt headers look like: sp|A0A087X1C5|CP2D7_HUMAN
        # We want the accession number (middle part) as the ID.
        parts = record.id.split("|")
        protein_id = parts[1] if len(parts) > 1 else record.id

        protein_database[protein_id] = str(record.seq)

    if extra_fasta_path is not None:
        for record in SeqIO.parse(extra_fasta_path, "fasta"):
            parts = record.id.split("|")
            protein_id = parts[1] if len(parts) > 1 else record.id
            protein_database[protein_id] = str(record.seq)

    return protein_database
