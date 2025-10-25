import sys
import argparse

from lib.command_build import build_index
from lib.command_search import search_movies, DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Build search index")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help=f"Number of results to return (default: {DEFAULT_SEARCH_LIMIT})",
    )

    args = parser.parse_args()

    
    match args.command:
        case "search":
            print(f"Searching for: {args.query}...")
            search_movies(args.query, k=args.k)
        case "build":
            print(f"Building search index...")
            build_index()
        case _:
            parser.print_help()
            sys.exit(0)


if __name__ == "__main__":
    main()

