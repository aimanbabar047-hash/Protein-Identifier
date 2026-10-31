# 🧬 Protein Identifier

A web app that identifies a protein from its amino acid sequence by aligning it against a reference database, and explains the result in plain English.

Built by **Aiman Babar** (BS Bioinformatics).

## What it does

1. You paste a protein sequence (FASTA or raw).
2. A fast **k-mer (Jaccard) filter** shortlists the most similar reference proteins.
3. The **Needleman-Wunsch** global alignment algorithm (implemented from scratch) aligns your sequence against each candidate and picks the best score.
4. **Google Gemini** explains the score and alignment in simple language.
5. A **Matrix Visualizer** tab shows the full score matrix and traceback path for short sequences, like the hand-drawn version taught in class.

## Features

- Needleman-Wunsch implemented from scratch (score matrix + traceback)
- K-mer pre-filtering (seed-and-filter idea similar to BLAST); in my tests, search time dropped from ~159 s to ~5 s on 500 proteins
- Input validation (lowercase handling, invalid characters, DNA detection)
- Interactive matrix + traceback visualization for learning
- AI-generated plain-English explanation

## Project structure

```
app.py                 Streamlit user interface
nw_algorithm.py        Needleman-Wunsch alignment
kmer_search.py         k-mer filtering + best-match search
database.py            FASTA loading
ai_explainer.py        Gemini explanation
visualizer.py          Matrix / traceback HTML rendering
data/                  Reference FASTA files + saved ESM2 embeddings

Experiments (not needed to run the app):
esm_search.py          ESM2 embedding search
build_embeddings.py    Computes embeddings for the reference database
evaluate_methods.py    Test 1: damaged copies
evaluate_relatives.py  Test 2: finding relatives
```

## Run it locally

```bash
pip install -r requirements.txt
```

Add your Gemini API key (free from https://aistudio.google.com/apikey):

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Replace the placeholder with your key

Then:

```bash
streamlit run app.py
```

## Reference data

500 reviewed (Swiss-Prot) human protein sequences from UniProt, plus 12 insulin-family proteins so the demo includes meaningful matches.

## Experiment: alignment vs. protein language model embeddings

I also tested whether ESM2 embeddings (`esm2_t6_8M_UR50D`, mean-pooled, cosine similarity) could do the search better than k-mer + Needleman-Wunsch. Both tests are small, so treat them as indications, not proof.

**Test 1: re-identifying damaged copies.** 12 database proteins, a fraction of residues replaced at random, then searched against the database. Numbers show correctly identified proteins out of 12.

| Residues changed | k-mer + NW | Embedding |
|---|---|---|
| 20% | 12/12 | 9/12 |
| 40% | 12/12 | 3/12 |
| 60% | 8/12 | 0/12 |

**Test 2: finding relatives.** Each of the 12 insulin-family proteins was used as a query (itself removed) to see whether the best match was another family member.

- Best match is a family member: k-mer + NW 10/12, embedding 8/12
- Embedding top-5 contained 23 family members out of 60 slots (about 1-2 expected by chance)
- Neither method failed on every query: both right on 6, only embedding right on 2, only NW right on 4, and no query was missed by both

**Conclusion.** Alignment is better at identifying a protein or a close copy of it. Embeddings are weaker at that but do find relatives, and they succeeded on some queries where alignment did not, so the two are complementary. The embedding method is therefore kept as an experiment (not used by the app) for now.

To reproduce (needs extra packages: `pip install torch transformers`):

```bash
python build_embeddings.py
python evaluate_methods.py
python evaluate_relatives.py
```

## Limitations

- Needleman-Wunsch uses simple match/mismatch/gap scoring, not a substitution matrix like BLOSUM62
- Global alignment always reports some "best" match, even when it is poor; a low score means no real match
- Global alignment only; local alignment (Smith-Waterman) would suit partial matches better
- The reference database is small (about 500 proteins), so most queries will not find a close match
- Experiments use 12 proteins, random mutations, and the smallest ESM2 model
- The AI explanation is a helper and can occasionally be imprecise

## Future work

- A "no confident match" warning for low scores
- An optional "similar proteins" panel using ESM2 embeddings as a second opinion
- BLOSUM62 scoring and affine gap penalties
- Smith-Waterman local alignment
- Larger reference database and a larger ESM2 model
