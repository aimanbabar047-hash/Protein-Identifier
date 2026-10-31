"""
ai_explainer.py
================
Uses Google's Gemini API to explain an alignment result in
simple, non-technical language.

The API key is NOT stored in this file. It is read from:
  1. Streamlit secrets  (.streamlit/secrets.toml  ->  GEMINI_API_KEY = "...")
  2. or an environment variable named GEMINI_API_KEY
"""

import os
import time
from google import genai
from google.genai.errors import ServerError

MODEL_NAME = "gemini-3.8-flash"


def _get_api_key():
    """Looks for the Gemini API key in Streamlit secrets, then environment."""
    try:
        import streamlit as st
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.environ.get("GEMINI_API_KEY")


def explain_result(protein_id, score, aligned_seq1, aligned_seq2, max_retries=3):
    """
    Explains the alignment result using Gemini, in simple English.
    Retries automatically if Gemini's servers are temporarily overloaded
    (503 UNAVAILABLE), which can happen on the free tier.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "AI explanation is unavailable: no Gemini API key found. "
            "Add GEMINI_API_KEY to .streamlit/secrets.toml (see README)."
        )

    client = genai.Client(api_key=api_key)

    prompt = f"""
I am building a bioinformatics tool that aligns protein sequences.

Here are the results:
- Best matching protein ID: {protein_id}
- Alignment score: {score}
- Aligned sequence 1: {aligned_seq1}
- Aligned sequence 2: {aligned_seq2}

Explain this result to a non-technical user in 2-3 lines:
1. What does this score mean (is it a good match or not)
2. What do the gaps in the alignment mean (if any)
3. What is the practical significance of this result

Keep the response short and in simple English.
"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return response.text

        except ServerError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return (
                "The AI explanation service is currently overloaded "
                "(Gemini servers are busy). The alignment result above "
                "is still valid - please try again in a moment."
            )
