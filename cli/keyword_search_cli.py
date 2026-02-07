#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    bm25_idf_command,
    bm25_tf_command,
    bm25search_command,
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)
from lib.search_utils import BM25_B, BM25_K1


def run_build(_: argparse.Namespace) -> None:
    print("Building inverted index...")
    build_command()
    print("Inverted index built successfully.")


def run_search(args: argparse.Namespace) -> None:
    print("Searching for:", args.query)
    results = search_command(args.query)
    for i, res in enumerate(results, 1):
        print(f"{i}. ({res['id']}) {res['title']}")


def run_tf(args: argparse.Namespace) -> None:
    tf = tf_command(args.doc_id, args.term)
    print(f"Term frequency of '{args.term}' in document '{args.doc_id}': {tf}")


def run_idf(args: argparse.Namespace) -> None:
    idf = idf_command(args.term)
    print(f"Inverse document frequency of '{args.term}': {idf:.2f}")


def run_tfidf(args: argparse.Namespace) -> None:
    tf_idf = tfidf_command(args.doc_id, args.term)
    print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")


def run_bm25idf(args: argparse.Namespace) -> None:
    bm25idf = bm25_idf_command(args.term)
    print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")


def run_bm25tf(args: argparse.Namespace) -> None:
    bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
    print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")


def run_bm25search(args: argparse.Namespace) -> None:
    print("Searching for:", args.query)
    results = bm25search_command(args.query)
    for i, res in enumerate(results, 1):
        print(f"{i}. ({res['id']}) {res['title']} - Score: {res['score']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    build_parser.set_defaults(func=run_build)

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.set_defaults(func=run_search)

    tf_parser = subparsers.add_parser(
        "tf", help="Get term frequency for a given document ID and term"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")
    tf_parser.set_defaults(func=run_tf)

    idf_parser = subparsers.add_parser(
        "idf", help="Get inverse document frequency for a given term"
    )
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")
    idf_parser.set_defaults(func=run_idf)

    tf_idf_parser = subparsers.add_parser(
        "tfidf", help="Get TF-IDF score for a given document ID and term"
    )
    tf_idf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_idf_parser.add_argument("term", type=str, help="Term to get TF-IDF score for")
    tf_idf_parser.set_defaults(func=run_tfidf)

    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )
    bm25_idf_parser.set_defaults(func=run_bm25idf)

    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter"
    )
    bm25_tf_parser.set_defaults(func=run_bm25tf)

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.set_defaults(func=run_bm25search)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
