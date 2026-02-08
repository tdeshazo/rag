#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    chunk_command,
    embed_chunks_command,
    embed_query_command,
    embed_text_command,
    search_chunked_command,
    semantic_chunk_command,
    semantic_search_command,
    verify_command,
    verify_embeddings_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify", help="Verify that the embedding model is loaded"
    )
    verify_parser.set_defaults(func=lambda _: verify_command())

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")
    single_embed_parser.set_defaults(
        func=lambda args: embed_text_command(args.text)
    )

    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset"
    )
    verify_embeddings_parser.set_defaults(
        func=lambda _: verify_embeddings_command()
    )

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    embed_query_parser.set_defaults(
        func=lambda args: embed_query_command(args.query)
    )

    search_parser = subparsers.add_parser(
        "search", help="Search for movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    search_parser.set_defaults(
        func=lambda args: semantic_search_command(args.query, args.limit)
    )

    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size chunks with optional overlap"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200, help="Size of each chunk in words"
    )
    chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of words to overlap between chunks",
    )
    chunk_parser.set_defaults(
        func=lambda args: chunk_command(args.text, args.chunk_size, args.overlap)
    )

    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text on sentence boundaries to preserve meaning"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4,
        help="Maximum size of each chunk in sentences",
    )
    semantic_chunk_parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="Number of sentences to overlap between chunks",
    )
    semantic_chunk_parser.set_defaults(
        func=lambda args: semantic_chunk_command(
            args.text,
            args.max_chunk_size,
            args.overlap,
        )
    )

    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="Generate embeddings for chunked documents"
    )
    embed_chunks_parser.set_defaults(func=lambda _: embed_chunks_command())

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search using chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    search_chunked_parser.set_defaults(
        func=lambda args: search_chunked_command(args.query, args.limit)
    )

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
