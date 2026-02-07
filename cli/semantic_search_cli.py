#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    chunk_text,
    embed_chunks_command,
    embed_query_text,
    embed_text,
    search_chunked_command,
    semantic_chunk_text,
    semantic_search,
    verify_embeddings,
    verify_model,
)


def run_verify(_: argparse.Namespace) -> None:
    result = verify_model()
    print(f"Model loaded: {result['model']}")
    print(f"Max sequence length: {result['max_seq_length']}")


def run_embed_text(args: argparse.Namespace) -> None:
    result = embed_text(args.text)
    embedding = result["embedding"]
    print(f"Text: {result['text']}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def run_verify_embeddings(_: argparse.Namespace) -> None:
    result = verify_embeddings()
    print(f"Number of docs:   {result['num_documents']}")
    print(
        f"Embeddings shape: {result['num_vectors']} vectors in {result['num_dimensions']} dimensions"
    )


def run_embed_query(args: argparse.Namespace) -> None:
    result = embed_query_text(args.query)
    embedding = result["embedding"]
    print(f"Query: {result['query']}")
    print(f"First 5 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def run_search(args: argparse.Namespace) -> None:
    result = semantic_search(args.query, args.limit)
    print(f"Query: {result['query']}")
    print(f"Top {len(result['results'])} results:")
    print()
    for i, item in enumerate(result["results"], 1):
        print(f"{i}. {item['title']} (score: {item['score']:.4f})")
        print(f"   {item['description'][:100]}...")
        print()


def run_chunk(args: argparse.Namespace) -> None:
    result = chunk_text(args.text, args.chunk_size, args.overlap)
    print(f"Chunking {result['text_length']} characters")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"{i}. {chunk}")


def run_semantic_chunk(args: argparse.Namespace) -> None:
    result = semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
    print(f"Semantically chunking {result['text_length']} characters")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"{i}. {chunk}")


def run_embed_chunks(_: argparse.Namespace) -> None:
    embeddings = embed_chunks_command()
    print(f"Generated {len(embeddings)} chunked embeddings")


def run_search_chunked(args: argparse.Namespace) -> None:
    result = search_chunked_command(args.query, args.limit)
    print(f"Query: {result['query']}")
    print("Results:")
    for i, res in enumerate(result["results"], 1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser(
        "verify", help="Verify that the embedding model is loaded"
    )
    verify_parser.set_defaults(func=run_verify)

    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")
    single_embed_parser.set_defaults(func=run_embed_text)

    verify_embeddings_parser = subparsers.add_parser(
        "verify_embeddings", help="Verify embeddings for the movie dataset"
    )
    verify_embeddings_parser.set_defaults(func=run_verify_embeddings)

    embed_query_parser = subparsers.add_parser(
        "embedquery", help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    embed_query_parser.set_defaults(func=run_embed_query)

    search_parser = subparsers.add_parser(
        "search", help="Search for movies using semantic search"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    search_parser.set_defaults(func=run_search)

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
    chunk_parser.set_defaults(func=run_chunk)

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
    semantic_chunk_parser.set_defaults(func=run_semantic_chunk)

    embed_chunks_parser = subparsers.add_parser(
        "embed_chunks", help="Generate embeddings for chunked documents"
    )
    embed_chunks_parser.set_defaults(func=run_embed_chunks)

    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search using chunked embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Number of results to return"
    )
    search_chunked_parser.set_defaults(func=run_search_chunked)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
