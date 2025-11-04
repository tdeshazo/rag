import argparse
from typing import Optional

from lib.sem_cmds import (
    verify_model, embed_text, verify_embeddings, embed_query_text, search,
    cmd_word_chunk, cmd_sentence_chunk
)


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def add_verify(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("verify", help="Semantic Search CLI")
    p.set_defaults(func=lambda a: verify_model())


def add_verify_embeddings(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("verify_embeddings", help="Semantic Search CLI")
    p.set_defaults(func=lambda a: verify_embeddings())


def add_embed_text(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("embed_text", help="Create text embeddings")
    p.add_argument("text", type=str, help="Text to generate embeddings for")
    p.set_defaults(func=lambda a: embed_text(a.text))


def add_embed_text(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("embedquery", help="Create text embeddings")
    p.add_argument("text", type=str, help="Text to generate embeddings for")
    p.set_defaults(func=lambda a: embed_query_text(a.text))


def add_search(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("search", help="Create text embeddings")
    p.add_argument("query", type=str)
    p.add_argument(
        "--limit",
        type=positive_int,
        default=5,
        metavar="N",
        help="Number of results to return",
    )
    p.set_defaults(func=lambda a: search(a.query, a.limit))


def add_chunk(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("chunk", help="Chunk text")
    p.add_argument("text", type=str)
    p.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        metavar="N",
        help="Number of chunks to split text",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=0,
        metavar="N",
        help="Number of words to overlap",
    )
    p.set_defaults(func=lambda a: cmd_word_chunk(a.text, a.chunk_size, a.overlap))

def add_semantic_chunk(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("semantic_chunk", help="Chunk text")
    p.add_argument("text", type=str)
    p.add_argument(
        "--max-chunk-size",
        type=int,
        default=200,
        metavar="N",
        help="Number of chunks to split text",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=0,
        metavar="N",
        help="Number of words to overlap",
    )
    p.set_defaults(func=lambda a: cmd_sentence_chunk(a.text, a.max_chunk_size, a.overlap))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        # prog="semanticsearch",
        description="Keyword Search CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # required=True works on modern Python; fall back for older versions if needed
    subparsers = parser.add_subparsers(
        title="commands", dest="command", required=True
    )

    add_verify(subparsers)
    add_embed_text(subparsers)
    add_verify_embeddings(subparsers)
    add_search(subparsers)
    add_chunk(subparsers)
    add_semantic_chunk(subparsers)

    parser.add_argument(
        "-V", "--version",
        action="version",
        version="semsearch 1.0.0",
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
