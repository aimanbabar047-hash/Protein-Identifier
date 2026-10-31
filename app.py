"""
app.py
=======
Streamlit web app for Protein Identification using Needleman-Wunsch
alignment, k-mer based fast search, and AI-generated explanations.

Run with:
    streamlit run app.py
"""

import streamlit as st
from database import load_database
from kmer_search import smart_find_best_match
from ai_explainer import explain_result
from nw_algorithm import needleman_wunsch
from visualizer import render_matrix_html

# Standard 20 amino acid letters (single-letter codes)
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Protein Identifier",
    page_icon="\U0001F9EC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Overall page accent color */
    .stApp {
        background-color: #fafbfc;
    }

    /* Section headers */
    h1, h2, h3 {
        color: #1a2b4c;
    }

    /* Metric-style result cards */
    .result-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .result-label {
        font-size: 13px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 4px;
    }
    .result-value {
        font-size: 22px;
        font-weight: 700;
        color: #1a2b4c;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background-color: #10213f;
    }
    section[data-testid="stSidebar"] * {
        color: #e5e9f2 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load the reference database (cached so it only loads once)
# ---------------------------------------------------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "data", "reference_proteins.fasta")
EXTRA_DATABASE_PATH = os.path.join(BASE_DIR, "data", "insulin_family.fasta")


@st.cache_data
def get_database():
    return load_database(DATABASE_PATH, limit=500, extra_fasta_path=EXTRA_DATABASE_PATH)


with st.spinner("Loading reference protein database..."):
    protein_database = get_database()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## \U0001F9EC Protein Identifier")
    st.markdown(
        "A tool that identifies proteins using the **Needleman-Wunsch** "
        "global alignment algorithm, a fast **k-mer based search**, and "
        "**AI-generated explanations**."
    )
    st.divider()
    st.markdown("### Database")
    st.metric("Reference proteins loaded", len(protein_database))
    st.caption("Source: UniProt (reviewed, model organisms)")
    st.divider()
    st.markdown("### How it works")
    st.markdown(
        "1. Paste a protein sequence (FASTA)\n"
        "2. Fast k-mer filtering shortlists candidates\n"
        "3. Needleman-Wunsch finds the best alignment\n"
        "4. Gemini explains the result in plain English"
    )
    st.divider()
    st.caption("Created by Aiman Babar - BS Bioinformatics")

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def parse_fasta_input(text):
    """
    Extracts a plain sequence from pasted FASTA text.
    Ignores the '>' header line if present, and removes whitespace.
    """
    lines = text.strip().splitlines()
    sequence_lines = [line.strip() for line in lines if not line.startswith(">")]
    return "".join(sequence_lines).upper()


def validate_protein_sequence(sequence):
    """
    Checks whether a sequence looks like a valid protein sequence.

    Returns
    -------
    (is_valid, message) : tuple(bool, str)
        is_valid is True if the sequence passes all checks.
        message contains an explanation if it fails (empty string if valid).
    """
    if len(sequence) == 0:
        return False, "Please paste a protein sequence."

    if len(sequence) < 3:
        return False, "Sequence is too short. Please enter at least 3 amino acids."

    # Check for DNA sequences FIRST (before the amino-acid check), because
    # A, T, G, C happen to also be valid single-letter amino acid codes
    # (Alanine, Threonine, Glycine, Cysteine) -- so a pure-ATGC sequence
    # would otherwise pass the amino-acid check without ever being flagged.
    # We only apply this heuristic for longer sequences, since a very short
    # sequence made only of A/T/G/C could plausibly be a real (if unusual)
    # short protein rather than DNA.
    unique_letters = set(sequence)
    if len(sequence) >= 10 and unique_letters.issubset(set("ATGC")):
        return False, (
            "This looks like a DNA sequence (only contains A, T, G, C), "
            "not a protein sequence. Please paste an amino acid sequence instead."
        )

    # Check for invalid characters (anything not a standard amino acid letter)
    invalid_chars = set(sequence) - VALID_AMINO_ACIDS
    if invalid_chars:
        return False, (
            f"Invalid character(s) found: {', '.join(sorted(invalid_chars))}. "
            "A protein sequence should only contain the 20 standard amino acid "
            "letters (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)."
        )

    return True, ""


def result_card(label, value):
    """Renders a small styled card for displaying a result metric."""
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Main content - Tabs
# ---------------------------------------------------------
st.title("\U0001F9EC Protein Identifier")

tab_search, tab_visualizer = st.tabs(["\U0001F50D Identify Protein", "\U0001F4D0 Matrix Visualizer"])

# ===========================================================
# TAB 1: Main protein search
# ===========================================================
with tab_search:
    st.write(
        "Paste a protein sequence (FASTA format) below. The app will find "
        "the closest matching protein in the reference database and "
        "explain the result."
    )

    user_input = st.text_area(
        "Paste your FASTA sequence here:",
        height=180,
        placeholder=">MyProtein\nMEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIE...",
    )

    search_clicked = st.button("\U0001F50D Identify Protein", type="primary")

    if search_clicked:
        query_sequence = parse_fasta_input(user_input)
        is_valid, error_message = validate_protein_sequence(query_sequence)

        if not is_valid:
            st.error(error_message)
        else:
            with st.spinner("Searching for the best match..."):
                best_id, best_score, best_alignment = smart_find_best_match(
                    query_sequence, protein_database, k=3, top_candidates=20
                )

            aligned1, aligned2 = best_alignment

            st.subheader("Result")
            col1, col2 = st.columns(2)
            with col1:
                result_card("Best Matching Protein ID", best_id)
            with col2:
                result_card("Alignment Score", best_score)

            st.write("")
            with st.expander("View Alignment"):
                st.code(f"{aligned1}\n{aligned2}", language="text")

            with st.spinner("Generating explanation..."):
                explanation = explain_result(best_id, best_score, aligned1, aligned2)

            st.subheader("AI Explanation")
            st.info(explanation)

# ===========================================================
# TAB 2: Matrix Visualizer (teaching demo)
# ===========================================================
with tab_visualizer:
    st.write(
        "Enter two **short** sequences (max ~12 letters each) to see the "
        "full score matrix and traceback path, just like it's drawn by "
        "hand on paper."
    )

    col1, col2 = st.columns(2)
    with col1:
        demo_seq1 = st.text_input("Sequence 1", value="GATTACA", max_chars=12)
    with col2:
        demo_seq2 = st.text_input("Sequence 2", value="GCATGCU", max_chars=12)

    if st.button("Show Matrix"):
        if len(demo_seq1) == 0 or len(demo_seq2) == 0:
            st.error("Please enter both sequences.")
        else:
            aligned1, aligned2, score, score_matrix, traceback_matrix = needleman_wunsch(
                demo_seq1.upper(), demo_seq2.upper()
            )

            col1, col2 = st.columns(2)
            with col1:
                result_card("Final Score", score)
            with col2:
                result_card("Aligned Sequences", f"{aligned1} / {aligned2}")

            st.write("")
            st.caption(
                "The highlighted (yellow) cells show the optimal traceback "
                "path -- the route the algorithm follows backward through "
                "the matrix to build the final alignment. "
                "\u2196 = match/mismatch, \u2191 = gap in sequence 2, \u2190 = gap in sequence 1."
            )

            matrix_html = render_matrix_html(demo_seq1.upper(), demo_seq2.upper(), score_matrix, traceback_matrix)
            st.markdown(matrix_html, unsafe_allow_html=True)
