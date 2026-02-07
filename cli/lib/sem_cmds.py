import re
from textwrap import shorten
from typing import Callable, Optional

from .defaults import DEFAULT_MODEL, DEFAULT_SEARCH_LIMIT
from .file_utils import load_movies
from .semantic_search import SemanticSearch
from .chunked_semantic_search import ChunkedSemanticSearch


def verify_model(model: str = DEFAULT_MODEL) -> None:
    try:
        st = SemanticSearch(model)
    except Exception as e:
        print(f"Error loading verifying model: {e}")
    else:
        print(f"Model loaded: {model}")
        print(f"Max sequence length: {st.model.max_seq_length}")


def embed_text(query: str) -> None:
    try:
        st = SemanticSearch()
    except Exception as e:
        print(f"Error loading verifying model")
    else:
        embedding = st.generate_embedding(query)
        print(f"Text: {query}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings() -> None:
    documents = load_movies()
    st = SemanticSearch()
    embeddings = st.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def embed_query_text(query: str) -> None:
    try:
        st = SemanticSearch()
    except Exception as e:
        print(f"Error loading verifying model")
    else:
        embedding = st.generate_embedding(query)
        print(f"Query: {query}")
        print(f"First 5 dimensions: {embedding[:5]}")
        print(f"Dimensions: {embedding.shape}")


def search(query: str, limit: int = 5):
    documents = load_movies()
    st = SemanticSearch()
    st.load_or_create_embeddings(documents)
    results = st.search(query, limit)

    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        score = float(r.get("score", 0.0))
        description = r.get("description", "") or ""
        # Normalize whitespace and truncate with an ellipsis
        desc_line = " ".join(description.split())
        desc_line = shorten(desc_line, width=120, placeholder="...")

        print(f"{i}. {title} (score: {score:.4f})")
        print(f"   {desc_line}")
        if i < len(results):
            print()
    return


def _chunk(
        text: str,
        split:Callable,
        max_chunk_size: int = 4,
        overlap: int = 0,
        fmt: Optional[str] = None
    ):
    split_text = split(text)
    step = max_chunk_size - overlap
    chunks = [' '.join(split_text[i:i + max_chunk_size])
            for i in range(0, len(split_text), step)]
    if fmt:
        data = {"chars": len(text)}
        print(fmt.format(**data))
    for i, line in enumerate(chunks, start=1):
        print(f"{i}. {line}")
    

def cmd_word_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0):
    return _chunk(
        text,
        split=lambda t: t.split(),
        max_chunk_size=max_chunk_size,
        overlap=overlap,
        fmt="Chunking {chars} characters:",
    )

 
def cmd_sentence_chunk(text: str, max_chunk_size: int = 4, overlap: int = 0):
    # return _chunk(
    #     text,
    #     split=lambda t: re.split(r"(?<=[.!?])\s+", t),
    #     max_chunk_size=max_chunk_size,
    #     overlap=overlap,
    #     fmt="Semantically chunking {chars} characters:",
    # )
    chunks = ChunkedSemanticSearch.chunk(
        text, max_chunk_size, overlap
    )
    print(f"Semantically chunking {len(text)} characters:")
    for i, line in enumerate(chunks, start=1):
        print(f"{i}. {line}")


def cmd_embed_chunks() -> None:
    documents = load_movies()
    cs = ChunkedSemanticSearch()
    embeddings = cs.load_or_create_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")


def cmd_search_chunks(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    documents = load_movies()
    st = ChunkedSemanticSearch()
    st.load_or_create_embeddings(documents)
    results = st.search(query, limit)

    for i, r in enumerate(results, start=1):
        title = r.get("title", "")
        score = float(r.get("score", 0.0))
        description = r.get("description", "") or ""
        # Normalize whitespace and truncate with an ellipsis
        desc_line = " ".join(description.split())
        desc_line = shorten(desc_line, width=120, placeholder="...")

        print(f"{i}. {title} (score: {score:.4f})")
        print(f"   {desc_line}")
        if i < len(results):
            print()
    return

