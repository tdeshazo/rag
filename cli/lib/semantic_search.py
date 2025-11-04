from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from .file_utils import save_cache, load_cache

DEFAULT_MODEL = 'all-MiniLM-L6-v2'


class SemanticSearch:
    def __init__(self, model=DEFAULT_MODEL):
        self.model: SentenceTransformer = SentenceTransformer(model)
        self.embeddings: list | None = None
        self.documents: list[dict[str, Any]] | None = None
        self.docmap = {}

    def generate_embedding(self, text: str):
        text = text.strip()
        if not text:
            raise ValueError
        return self.model.encode([text])[0]

    def __add_document(self, doc):
        self.docmap[doc['id']] = doc
        return f"{doc['title']}: {doc['description']}"

    def build_embeddings(self, documents: list[dict[str, Any]]):
        self.documents = documents
        docs = [self.__add_document(doc) for doc in documents]
        self.embeddings = self.model.encode(docs,
                                            convert_to_numpy=True,
                                            show_progress_bar=True)
        save_cache(self.embeddings, "movie_embeddings.npy")
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.embeddings = load_cache("movie_embeddings.npy", force=True)
        if self.embeddings is not None:
            self.docmap = {doc['id']: doc for doc in documents}
            return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int=5):
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        # Validate and clamp limit
        if not isinstance(limit, int) or limit <= 0:
            limit = 5

        # Compute query embedding
        q = self.generate_embedding(query)
        q = np.asarray(q, dtype=np.float32).ravel()

        # Ensure embeddings are a 2D numpy array
        emb = np.asarray(self.embeddings, dtype=np.float32)
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
        return results
