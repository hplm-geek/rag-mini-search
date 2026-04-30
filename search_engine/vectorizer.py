"""Vector similarity search using sentence-transformers and cosine similarity."""

import os
import numpy as np


class VectorIndex:
    """
    Encodes a corpus of .txt documents into dense vectors using a local
    sentence-transformers model, then ranks documents by cosine similarity
    to a query embedding.

    The model is loaded lazily on first call to build() so imports are fast.
    """

    # Lightweight model that runs fully locally with no API key
    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # loaded on first build()
        self.doc_names: list[str] = []
        self.embeddings: np.ndarray | None = None  # shape (N, D)

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def build(self, corpus_dir: str) -> None:
        """Embed all .txt files in corpus_dir. Downloads model on first run (~80 MB)."""
        self._load_model()

        txt_files = sorted(
            f for f in os.listdir(corpus_dir) if f.endswith(".txt")
        )
        if not txt_files:
            raise ValueError(f"No .txt files found in {corpus_dir}")

        texts = []
        for fname in txt_files:
            path = os.path.join(corpus_dir, fname)
            with open(path, "r", encoding="utf-8") as fh:
                texts.append(fh.read())
            self.doc_names.append(fname)

        # encode returns a numpy array of shape (N, embedding_dim)
        self.embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def search(self, query: str, top_n: int = 5) -> list[tuple[str, float]]:
        """
        Return up to top_n (doc_name, cosine_similarity) pairs, highest similarity first.
        Scores are in the range [-1.0, 1.0]; practical range is roughly [0.0, 1.0].
        """
        if self.embeddings is None or len(self.doc_names) == 0:
            raise RuntimeError("Index is empty — call build() first.")

        self._load_model()
        query_vec = self._model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]

        scores = _cosine_similarity(query_vec, self.embeddings)
        top_indices = np.argsort(scores)[::-1][:top_n]

        return [
            (self.doc_names[i], round(float(scores[i]), 4))
            for i in top_indices
        ]


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a single query vector and a matrix of doc vectors."""
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    doc_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    return doc_norms @ query_norm
