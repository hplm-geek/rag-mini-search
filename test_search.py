"""Basic tests for rag-mini-search."""

import os
import pytest

from search_engine import InvertedIndex, VectorIndex

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# Inverted index tests
# ---------------------------------------------------------------------------

def test_index_builds_and_contains_terms():
    """Index should contain tokens from the corpus after build()."""
    idx = InvertedIndex()
    idx.build(CORPUS_DIR)
    # The transcripts contain common business words
    assert len(idx.index) > 0, "Index should not be empty after build"
    assert idx._num_docs == 20, "Should have indexed all 20 documents"


def test_keyword_search_returns_results():
    """A relevant query should return at least one result."""
    idx = InvertedIndex()
    idx.build(CORPUS_DIR)
    results = idx.search("systems", top_n=5)
    assert len(results) > 0, "Expected results for query 'systems'"


def test_keyword_results_are_tuples_with_score():
    """Each result should be a (filename, float_score) tuple."""
    idx = InvertedIndex()
    idx.build(CORPUS_DIR)
    results = idx.search("coaching", top_n=3)
    for doc_name, score in results:
        assert isinstance(doc_name, str)
        assert isinstance(score, float)
        assert score > 0.0, "Score should be positive for a matching document"


def test_keyword_results_ranked_descending():
    """Results should be ordered highest score first."""
    idx = InvertedIndex()
    idx.build(CORPUS_DIR)
    results = idx.search("invoice billing", top_n=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "Results must be in descending score order"


def test_keyword_unknown_query_returns_empty():
    """A query with no matching tokens should return an empty list."""
    idx = InvertedIndex()
    idx.build(CORPUS_DIR)
    results = idx.search("zzzznonexistenttoken9999")
    assert results == [], "Unknown query should return empty results"


# ---------------------------------------------------------------------------
# Vector index tests
# ---------------------------------------------------------------------------

def test_vector_search_returns_results():
    """Vector search should return results for any non-empty query."""
    idx = VectorIndex()
    idx.build(CORPUS_DIR)
    results = idx.search("time management and productivity", top_n=5)
    assert len(results) == 5, "Should return exactly top_n results"


def test_vector_scores_in_valid_range():
    """Cosine similarity scores should be in [-1.0, 1.0]."""
    idx = VectorIndex()
    idx.build(CORPUS_DIR)
    results = idx.search("delegation and systems", top_n=5)
    for _, score in results:
        assert -1.0 <= score <= 1.0, f"Score {score} out of cosine similarity range"


def test_vector_results_ranked_descending():
    """Vector results should be ordered highest similarity first."""
    idx = VectorIndex()
    idx.build(CORPUS_DIR)
    results = idx.search("client onboarding process", top_n=5)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True), "Vector results must be in descending order"
