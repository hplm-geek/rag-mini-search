# rag-mini-search

A command-line search tool that demonstrates two retrieval modes side by side:

- **Keyword search** — inverted index with TF-IDF scoring
- **Vector search** — dense embeddings via `sentence-transformers` + cosine similarity

The corpus is 20 plain-text sales call transcripts in `data/`.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** The vector model (`all-MiniLM-L6-v2`, ~80 MB) is downloaded automatically
> from Hugging Face on the first vector search run. No API key required.

---

## Usage

```bash
# Run both modes (default)
python main.py "your query here"

# Keyword only
python main.py "your query here" --mode keyword

# Vector only
python main.py "your query here" --mode vector

# Control number of results returned
python main.py "your query here" --top 10

# Point at a different corpus directory
python main.py "your query here" --corpus /path/to/docs
```

### Example output

```
[Keyword Search] query: "systems delegation"
--------------------------------------------------
  1. transcript_01.txt          score: 0.0321
  2. transcript_07.txt          score: 0.0198
  3. transcript_03.txt          score: 0.0142

[Vector Search]  query: "systems delegation"
  (loading model — first run may download ~80 MB)
--------------------------------------------------
  1. transcript_01.txt          score: 0.6814
  2. transcript_04.txt          score: 0.5903
  3. transcript_11.txt          score: 0.5721
```

---

## Project structure

```
rag-mini-search/
├── data/                    # 20 plain-text corpus documents
│   ├── transcript_01.txt
│   └── ...
├── search_engine/
│   ├── __init__.py
│   ├── indexer.py           # InvertedIndex — tokenize, build, TF-IDF search
│   └── vectorizer.py        # VectorIndex — embed, cosine similarity search
├── main.py                  # CLI entry point (argparse)
├── test_search.py           # pytest test suite
├── requirements.txt
└── README.md
```

---

## Running tests

```bash
pip install pytest
pytest test_search.py -v
```

The test suite covers:
- Index builds and contains expected token count
- Keyword search returns results and scores are positive floats
- Results are ranked in descending score order
- Unknown queries return empty results gracefully
- Vector scores fall within the valid cosine similarity range `[-1.0, 1.0]`

---

## How it works

### Keyword search (`InvertedIndex`)

1. Each document is tokenised: lowercased, punctuation stripped, stopwords removed.
2. An in-memory index maps `token → {doc_id: raw_count}`.
3. At query time, each query token's TF-IDF contribution is summed per document.
   - **TF** = raw count / document length (normalised)
   - **IDF** = log((N + 1) / (df + 1)) + 1  (smoothed)
4. Documents are ranked by total score, descending.

### Vector search (`VectorIndex`)

1. All documents are encoded once at index-build time using `all-MiniLM-L6-v2`.
2. The query string is encoded to a dense vector at search time.
3. Cosine similarity is computed between the query vector and every document vector.
4. Documents are ranked by similarity, descending.

The two modes complement each other: keyword search rewards exact term overlap,
vector search captures semantic similarity even when vocabulary differs.

## License
MIT License — built by Sean O'Meara
