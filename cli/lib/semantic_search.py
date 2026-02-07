import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .defaults import DEFAULT_MODEL
from .file_utils import save_cache, load_cache

logger = logging.getLogger(__name__)


class SemanticSearch:
    def __init__(self, model=DEFAULT_MODEL):
        logger.debug("Initializing SemanticSearch with model=%s", model)
        self.model: SentenceTransformer = SentenceTransformer(model)
        self.chunk_embeddings: list | None = None
        self.documents: list[dict[str, Any]] | None = None
        self.docmap = {}

    def generate_embedding(self, text: str):
        text = text.strip()
        if not text:
            logger.debug("generate_embedding: empty text after strip")
            raise ValueError
        logger.debug("generate_embedding: encoding text length=%d", len(text))
        return self.model.encode([text])[0]

    def __add_document(self, doc):
        self.docmap[doc['id']] = doc
        return f"{doc['title']}: {doc['description']}"

    def build_embeddings(self, documents: list[dict[str, Any]]):
        self.documents = documents
        logger.debug("build_embeddings: %d documents", len(documents))
        docs = [self.__add_document(doc) for doc in documents]
        self.chunk_embeddings = self.model.encode(docs,
                                            convert_to_numpy=True,
                                            show_progress_bar=True)
        logger.debug("build_embeddings: embeddings shape=%s", getattr(self.chunk_embeddings, "shape", None))
        save_cache(self.chunk_embeddings, "movie_embeddings.npy")
        logger.debug("build_embeddings: embeddings cached")
        return self.chunk_embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        logger.debug("load_or_create_embeddings: attempting cache load")
        self.chunk_embeddings = load_cache("movie_embeddings.npy", force=True)
        if self.chunk_embeddings is not None:
            self.docmap = {doc['id']: doc for doc in documents}
            logger.debug("load_or_create_embeddings: cache hit, docmap size=%d", len(self.docmap))
            return self.chunk_embeddings
        logger.debug("load_or_create_embeddings: cache miss, rebuilding")
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int=5):
        if self.chunk_embeddings is None or self.documents is None:
            logger.debug("search: embeddings/documents not loaded")
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        # Validate and clamp limit
        if not isinstance(limit, int) or limit <= 0:
            logger.debug("search: invalid limit=%r, defaulting to 5", limit)
            limit = 5

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

        # Top-k indices sorted by similarity desc
        topk_idx = np.argpartition(sims, -k)[-k:]
        topk_idx = topk_idx[np.argsort(sims[topk_idx])[::-1]]

        # Map indices to doc ids using the SAME order embeddings were built
        doc_ids = [doc["id"] for doc in self.documents]

        results = []
        for i in topk_idx:
            doc_id = doc_ids[i]
            d = self.docmap[doc_id]
            results.append({
                "score": float(sims[i]),
                "title": d["title"],
                "description": d["description"],
            })
        if results:
            logger.debug("search: top score=%.6f", results[0]["score"])
        return results
