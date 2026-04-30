"""Inverted index keyword search over a corpus of plain-text documents."""

import os
import re
import math
from collections import defaultdict


# Words that carry no signal for search ranking
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "it", "its", "this", "that", "i", "you", "we", "they", "he",
    "she", "not", "so", "if", "as", "up", "out", "my", "your", "our",
    "their", "about", "just", "like", "what", "when", "how", "all", "no",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    return [t.strip("'") for t in tokens if t not in _STOPWORDS and len(t) > 1]


class InvertedIndex:
    """
    Builds a token → {doc_id: term_frequency} index from a directory of .txt files.
    Scores queries using TF-IDF: term frequency weighted by inverse document frequency.
    """

    def __init__(self):
        # index[term] = {doc_id: raw_count}
        self.index: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.doc_lengths: dict[str, int] = {}
        self.doc_names: dict[str, str] = {}  # doc_id → filename
        self._num_docs: int = 0

    def build(self, corpus_dir: str) -> None:
        """Index all .txt files found in corpus_dir."""
        txt_files = sorted(
            f for f in os.listdir(corpus_dir) if f.endswith(".txt")
        )
        if not txt_files:
            raise ValueError(f"No .txt files found in {corpus_dir}")

        for fname in txt_files:
            doc_id = fname[:-4]  # strip .txt
            path = os.path.join(corpus_dir, fname)
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()

            tokens = tokenize(text)
            self.doc_names[doc_id] = fname
            self.doc_lengths[doc_id] = len(tokens)

            for token in tokens:
                self.index[token][doc_id] += 1

        self._num_docs = len(txt_files)

    def search(self, query: str, top_n: int = 5) -> list[tuple[str, float]]:
        """
        Return up to top_n (doc_name, score) pairs ranked by TF-IDF.
        Score is the sum of TF-IDF contributions across all query terms.
        """
        if self._num_docs == 0:
            raise RuntimeError("Index is empty — call build() first.")

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: dict[str, float] = defaultdict(float)

        for term in query_terms:
            if term not in self.index:
                continue

            # IDF: log( N / df ) where df = number of docs containing the term
            df = len(self.index[term])
            idf = math.log((self._num_docs + 1) / (df + 1)) + 1.0

            for doc_id, raw_count in self.index[term].items():
                # Normalised TF: raw count / total tokens in document
                tf = raw_count / max(self.doc_lengths[doc_id], 1)
                scores[doc_id] += tf * idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.doc_names[doc_id], round(score, 4)) for doc_id, score in ranked[:top_n]]
