"""
rag-mini-search — CLI entry point.

Usage:
    python main.py "your query here" --mode keyword
    python main.py "your query here" --mode vector
    python main.py "your query here" --mode both
"""

import argparse
import os
import sys

from search_engine import InvertedIndex, VectorIndex

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "data")


def build_keyword_index() -> InvertedIndex:
    idx = InvertedIndex()
    idx.build(CORPUS_DIR)
    return idx


def build_vector_index() -> VectorIndex:
    idx = VectorIndex()
    idx.build(CORPUS_DIR)
    return idx


def print_results(results: list[tuple[str, float]], mode: str) -> None:
    if not results:
        print("  (no results)")
        return
    for rank, (doc_name, score) in enumerate(results, start=1):
        print(f"  {rank}. {doc_name:<25}  score: {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a local corpus of .txt documents using keyword or vector search."
    )
    parser.add_argument("query", help="Search query string")
    parser.add_argument(
        "--mode",
        choices=["keyword", "vector", "both"],
        default="both",
        help="Search mode (default: both)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--corpus",
        default=CORPUS_DIR,
        metavar="DIR",
        help=f"Path to corpus directory (default: {CORPUS_DIR})",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.corpus):
        print(f"Error: corpus directory not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    run_keyword = args.mode in ("keyword", "both")
    run_vector = args.mode in ("vector", "both")

    if run_keyword:
        print(f'\n[Keyword Search] query: "{args.query}"')
        print("-" * 50)
        idx = build_keyword_index()
        results = idx.search(args.query, top_n=args.top)
        print_results(results, "keyword")

    if run_vector:
        print(f'\n[Vector Search]  query: "{args.query}"')
        print("  (loading model — first run may download ~80 MB)")
        print("-" * 50)
        idx = build_vector_index()
        results = idx.search(args.query, top_n=args.top)
        print_results(results, "vector")

    print()


if __name__ == "__main__":
    main()
