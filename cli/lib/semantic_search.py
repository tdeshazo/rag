from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from .search_utils import (
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DOCUMENT_PREVIEW_LENGTH,
    MOVIE_EMBEDDINGS_PATH,
    load_cache,
    load_movies,
    save_cache,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True, frozen=True)
class ChunkMeta:
    movie_idx: int
    chunk_idx: int
    total_chunks: int

    def to_dict(self) -> dict[str, int]:
        return {
            "movie_idx": self.movie_idx,
            "chunk_idx": self.chunk_idx,
            "total_chunks": self.total_chunks,
        }

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "ChunkMeta":
        return cls(
            movie_idx=int(data["movie_idx"]),
            chunk_idx=int(data["chunk_idx"]),
            total_chunks=int(data["total_chunks"]),
        )


class SemanticSearch:
    __slots__ = ("model", "embeddings", "documents", "document_map")

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("cannot generate embedding for empty text")
        return self.model.encode([text])[0]

    def _add_document(self, doc):
        self.docmap[doc['id']] = doc
        return f"{doc['title']}: {doc['description']}"

    def build_embeddings(self, documents):
        self.documents = documents
        docs = [self._add_document(doc) for doc in documents]
        self.embeddings = self.model.encode(docs,
                                            convert_to_numpy=True,
                                            show_progress_bar=True)
        save_cache(self.embeddings, "movie_embeddings.npy")
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        cached_embeddings = load_cache(MOVIE_EMBEDDINGS_PATH, force=True)
        if cached_embeddings is not None:
            self.embeddings = cached_embeddings
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def verify_model() -> dict:
    search_instance = SemanticSearch()
    return {
        "model": search_instance.model,
        "max_seq_length": search_instance.model.max_seq_length,
    }


def embed_text(text: str) -> dict:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)
    return {"text": text, "embedding": embedding}


def verify_embeddings() -> dict:
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    return {
        "num_documents": len(documents),
        "num_vectors": embeddings.shape[0],
        "num_dimensions": embeddings.shape[1],
    }


def embed_query_text(query: str) -> dict:
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)
    return {"query": query, "embedding": embedding}


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)
    return {"query": query, "results": results}


def fixed_size_chunking(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i: i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    return {"text_length": len(text), "chunks": chunks}


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    strip = str.strip
    text = strip(text)
    if not text:
        return []
    sentences = list(
        filter(None, map(str.strip, _SENTENCE_SPLIT_RE.split(text))))

    n = len(sentences)
    if n == 0:
        return []

    step = max_chunk_size - overlap
    if step <= 0:
        raise ValueError(
            "overlap must be < max_chunk_size (otherwise the loop can't advance).")

    chunks: list[str] = []
    append = chunks.append
    join = " ".join

    i = 0
    while True:
        j = i + max_chunk_size
        if j >= n:
            j = n
        if chunks and (j - i) <= overlap:
            break
        append(join(sentences[i:j]))
        if j == n:
            break
        i += step

    return chunks


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    return {"text_length": len(text), "chunks": chunks}


class ChunkedSemanticSearch(SemanticSearch):
    __slots__ = ("chunk_embeddings", "chunk_metadata")

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata: list[ChunkMeta] | None = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        all_chunks = []
        chunk_metadata: list[ChunkMeta] = []

        for i, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(
                text,
                max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE,
                overlap=DEFAULT_CHUNK_OVERLAP,
            )

            all_chunks.extend(chunks)
            num_chunks = len(chunks)
            chunk_metadata.extend(
                ChunkMeta(movie_idx=i, chunk_idx=j, total_chunks=num_chunks)
                for j in range(num_chunks)
            )

        self.chunk_embeddings = self.model.encode(
            all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        save_cache(self.chunk_embeddings, CHUNK_EMBEDDINGS_PATH)
        save_cache(
            {
                "chunks": [meta.to_dict() for meta in chunk_metadata],
                "total_chunks": len(all_chunks),
            },
            CHUNK_METADATA_PATH,
        )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        chunk_embeddings = load_cache(CHUNK_EMBEDDINGS_PATH, force=True)
        chunk_metadata = load_cache(CHUNK_METADATA_PATH, force=True)
        if chunk_embeddings is not None and chunk_metadata is not None:
            self.chunk_embeddings = chunk_embeddings
            self.chunk_metadata = [
                ChunkMeta.from_dict(meta) for meta in chunk_metadata["chunks"]
            ]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )
        # Compute query embedding
        q = self.generate_embedding(query)
        q = np.asarray(q, dtype=np.float32).ravel()

        # Ensure embeddings are a 2D numpy array
        emb = np.asarray(self.chunk_embeddings, dtype=np.float32)
        if emb.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        n = emb.shape[0]
        if n == 0:
            return []

        # Vectorized cosine similarity
        qnorm = np.linalg.norm(q)
        emb_norms = np.linalg.norm(emb, axis=1)
        denom = emb_norms * (qnorm if qnorm != 0 else 1.0)

        sims = np.zeros(n, dtype=np.float32)
        nz = denom > 0
        if np.any(nz):
            sims[nz] = emb[nz].dot(q) / denom[nz]

        k = min(limit, n)
        if k == 0:
            return []

        # Map indices to movie ids using the SAME order embeddings were built
        scores = {}
        for i, chunk in enumerate(self.chunk_metadata):
            movie_id = chunk.movie_idx
            scores[movie_id] = max(scores.get(movie_id, 0.0), float(sims[i]))

        # Sort scores by value
        sorted_scores = sorted(
            scores.items(), key=lambda x: x[1], reverse=True)

        # Add info for top k results
        results = []
        docmap = self.documents
        for i, score in sorted_scores[:k]:
            doc = docmap[i]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                "score": score,
            })

        return results

# ----- CLI COMMANDS -----


def embed_chunks_command() -> np.ndarray:
    movies = load_movies()
    searcher = ChunkedSemanticSearch()
    embeddings = searcher.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")
    return embeddings


def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = ChunkedSemanticSearch()
    searcher.load_or_create_chunk_embeddings(movies)
    results = searcher.search_chunks(query, limit)
    payload = {"query": query, "results": results}
    print(f"Query: {payload['query']}")
    print("Results:")
    for i, res in enumerate(payload["results"], 1):
        preview = res.get("document", res.get("description", ""))
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {preview}...")
    return payload


def verify_command() -> dict:
    result = verify_model()
    print(f"Model loaded: {result['model']}")
    print(f"Max sequence length: {result['max_seq_length']}")
    return result


def embed_text_command(text: str) -> dict:
    result = embed_text(text)
    embedding = result["embedding"]
    print(f"Text: {result['text']}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
    return result


def verify_embeddings_command() -> dict:
    result = verify_embeddings()
    print(f"Number of docs:   {result['num_documents']}")
    print(
        f"Embeddings shape: {result['num_vectors']} vectors in {result['num_dimensions']} dimensions"
    )
    return result


def embed_query_command(query: str) -> dict:
    result = embed_query_text(query)
    embedding = result["embedding"]
    print(f"Query: {result['query']}")
    print(f"First 5 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")
    return result


def semantic_search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    result = semantic_search(query, limit)
    print(f"Query: {result['query']}")
    print(f"Top {len(result['results'])} results:")
    print()
    for i, item in enumerate(result["results"], 1):
        print(f"{i}. {item['title']} (score: {item['score']:.4f})")
        print(f"   {item['description'][:100]}...")
        print()
    return result


def chunk_command(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    result = chunk_text(text, chunk_size, overlap)
    print(f"Chunking {result['text_length']} characters")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"{i}. {chunk}")
    return result


def semantic_chunk_command(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> dict:
    result = semantic_chunk_text(text, max_chunk_size, overlap)
    print(f"Semantically chunking {result['text_length']} characters")
    for i, chunk in enumerate(result["chunks"], 1):
        print(f"{i}. {chunk}")
    return result
