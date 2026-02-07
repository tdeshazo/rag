from __future__ import annotations

import logging
import re

import numpy as np

from .defaults import (
    DEFAULT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    SCORE_PRECISION
)
from .semantic_search import SemanticSearch
from .file_utils import save_cache, load_cache

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name=DEFAULT_MODEL) -> None:
        logger.debug(
            "Initializing ChunkedSemanticSearch with model=%s", model_name)
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    @staticmethod
    def chunk(
        text: str,
        max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP
    ) -> list[str]:
        strip = str.strip
        text = strip(text)
        if not text:
            return []
        sentences = list(filter(None, map(str.strip, _SENTENCE_SPLIT_RE.split(text))))
        
        n = len(sentences)
        if n == 0:
            return []

        step = max_chunk_size - overlap
        if step <= 0:
            raise ValueError("overlap must be < max_chunk_size (otherwise the loop can't advance).")

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

    def build_chunk_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        logger.debug("build_chunk_embeddings: %d documents", len(documents))
        all_chunks = []
        metadata = []

        for i, doc in enumerate(documents):
            text = doc.get("description") or ""
            if not text.strip():
                logger.debug(
                    "build_chunk_embeddings: skipping doc idx=%d (empty description)", i)
                continue
            chunks = self.chunk(text)
            all_chunks.extend(chunks)
            num_chunks = len(chunks)
            metadata.extend(
                [{'movie_idx': i, 'chunk_idx': j, 'total_chunks': num_chunks}
                 for j in range(num_chunks)])

        logger.debug("build_chunk_embeddings: total chunks=%d",
                     len(all_chunks))
        self.chunk_embeddings = self.model.encode(
            all_chunks,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        save_cache(self.chunk_embeddings, "chunk_embeddings.npy")
        logger.debug(
            "build_chunk_embeddings: embeddings shape=%s",
            getattr(self.chunk_embeddings, "shape", None),
        )

        self.chunk_metadata = metadata
        save_cache({"chunks": metadata, "total_chunks": len(
            all_chunks)}, "chunk_metadata.json")
        logger.debug(
            "build_chunk_embeddings: metadata saved (chunks=%d)", len(metadata))

        return self.chunk_embeddings

    def load_or_create_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        logger.debug("load_or_create_chunk_embeddings: attempting cache load")
        self.chunk_embeddings = load_cache("chunk_embeddings.npy", force=True)
        if self.chunk_embeddings is not None:
            self.docmap = {doc['id']: doc for doc in documents}
            self.chunk_metadata = load_cache("chunk_metadata.json")["chunks"]
            logger.debug(
                "load_or_create_chunk_embeddings: cache hit (docmap=%d, chunks=%d)",
                len(self.docmap),
                len(self.chunk_metadata),
            )
            return self.chunk_embeddings
        logger.debug("load_or_create_chunk_embeddings: cache miss, rebuilding")
        return self.build_chunk_embeddings(documents)

    def search(self, query: str, limit: int=10):
        if self.chunk_embeddings is None or self.documents is None:
            logger.debug("search: embeddings/documents not loaded")
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        # Validate and clamp limit
        if not isinstance(limit, int) or limit <= 0:
            logger.debug("search: invalid limit=%r, defaulting to 5", limit)
            limit = 10

        # Compute query embedding
        logger.debug("search: query length=%d, limit=%d", len(query), limit)
        q = self.generate_embedding(query)
        q = np.asarray(q, dtype=np.float32).ravel()

        # Ensure embeddings are a 2D numpy array
        emb = np.asarray(self.chunk_embeddings, dtype=np.float32)
        if emb.ndim != 2:
            logger.debug("search: embeddings ndim=%d", emb.ndim)
            raise ValueError("Embeddings must be a 2D array.")

        n = emb.shape[0]
        if n == 0:
            logger.debug("search: empty embeddings array")
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
        logger.debug("search: computing top-%d of %d", k, n)

        # Map indices to movie ids using the SAME order embeddings were built
        scores = {}
        for i, chunk in enumerate(self.chunk_metadata):
            movie_id = chunk["movie_idx"]
            scores[movie_id] = max(scores.get(movie_id, 0.0), float(sims[i]))
        
        # Sort scores by value
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Add info for top k results
        results = []
        docmap = self.documents
        for i, score in sorted_scores[:k]:
            doc = docmap[i]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "description": doc["description"],
                "score": round(score, SCORE_PRECISION),
            })

        return results
    