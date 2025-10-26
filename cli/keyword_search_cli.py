import argparse
from typing import Optional

from lib import (
    cmd_build,
    DEFAULT_SEARCH_LIMIT,
    cmd_search,
    cmd_tf,
    cmd_idf,
    cmd_tf_idf,
    cmd_bm25_idf
)


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def add_search(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("search", help="BM25 search over movies")
    p.add_argument("query", type=str, help="Search query string")
    p.add_argument(
        "-k", "--limit",
        type=positive_int,
        default=DEFAULT_SEARCH_LIMIT,
        metavar="N",
        help="Number of results to return",
    )
    p.set_defaults(func=lambda a: cmd_search(a.query, k=a.limit))


def add_tf(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("tf", help="Term frequency for a document")
    p.add_argument("doc_id", type=int, help="Document ID")
    p.add_argument("term", type=str, help="Term to look up")
    p.set_defaults(func=lambda a: cmd_tf(a.doc_id, a.term))


def add_idf(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("idf", help="Inverse document frequency of a term")
    p.add_argument("term", type=str, help="Term to look up")
    p.set_defaults(func=lambda a: cmd_idf(a.term))


def add_tfidf(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("tfidf", help="TF-IDF of a term in a document")
    p.add_argument("doc_id", type=int, help="Document ID")
    p.add_argument("term", type=str, help="Term to look up")
    p.set_defaults(func=lambda a: cmd_tf_idf(a.doc_id, a.term))


def add_bm25idf(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    p.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    p.set_defaults(func=lambda a: cmd_bm25_idf(a.term))


def add_build(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("build", help="Build the search index")
    p.set_defaults(func=lambda a: cmd_build())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kwsearch",
        description="Keyword Search CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # required=True works on modern Python; fall back for older versions if needed
    subparsers = parser.add_subparsers(
        title="commands", dest="command", required=True
    )

    add_build(subparsers)
    add_search(subparsers)
    add_tf(subparsers)
    add_idf(subparsers)
    add_tfidf(subparsers)
    add_bm25idf(subparsers)

    parser.add_argument(
        "-V", "--version",
        action="version",
        version="kwsearch 1.0.0",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    
    # Each subcommand sets a .func; calling it performs the action
    result = args.func(args)
    return 0 if result is None else int(bool(result))


if __name__ == "__main__":
    raise SystemExit(main())
