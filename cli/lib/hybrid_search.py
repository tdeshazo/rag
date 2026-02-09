from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .llm import LlmClient
from .search_utils import (
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    format_search_result,
    load_movies,
)


@dataclass(slots=True)
class HybridAccumulator:
    title: str
    document: str
    bm25_score: float = 0.0
    semantic_score: float = 0.0


class HybridSearch:
    __slots__ = ("documents", "semantic_search", "idx")

    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * 500)

        combined = combine_search_results(
            bm25_results, semantic_results, alpha)
        return combined[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * 500)

        combined = fuse_search_results(bm25_results, semantic_results, k)
        return combined[:limit]

# ===== Weighted Search Helpers =====


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    def norm(s): return (s - min_score) / (max_score - min_score)
    return [norm(score) for score in scores]


def normalize_search_results(results: list[dict]) -> list[dict]:
    scores = [result["score"] for result in results]
    normalized: list[float] = normalize_scores(scores)
    for i, result in enumerate(results):
        result["normalized_score"] = normalized[i]

    return results


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score


def combine_search_results(
    bm25_results: list[dict], semantic_results: list[dict], alpha: float = DEFAULT_ALPHA
) -> list[dict]:
    bm25_normalized = normalize_search_results(bm25_results)
    semantic_normalized = normalize_search_results(semantic_results)
    combined_scores: dict[int, HybridAccumulator] = {}

    for result in bm25_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = HybridAccumulator(
                title=result["title"],
                document=result["document"],
            )
        combined_scores[doc_id].bm25_score = max(
            combined_scores[doc_id].bm25_score, result["normalized_score"]
        )

    for result in semantic_normalized:
        doc_id = result["id"]
        if doc_id not in combined_scores:
            combined_scores[doc_id] = HybridAccumulator(
                title=result["title"],
                document=result["document"],
            )
        combined_scores[doc_id].semantic_score = max(
            combined_scores[doc_id].semantic_score, result["normalized_score"]
        )

    hybrid_results = []
    for doc_id, data in combined_scores.items():
        score_value = hybrid_score(data.bm25_score, data.semantic_score, alpha)
        result = format_search_result(
            doc_id=doc_id,
            title=data.title,
            document=data.document,
            score=score_value,
            bm25_score=data.bm25_score,
            semantic_score=data.semantic_score,
        )
        hybrid_results.append(result)

    return sorted(hybrid_results, key=lambda x: x["score"], reverse=True)

# ===== RRF Search Helpers =====


def rrf_score(rank, k=60):
    return 1 / (k + rank)


def rank_search_results(results: list[dict], k=60) -> list[dict]:
    for i, result in enumerate(results, start=1):
        result["rank"] = rrf_score(i, k)


def fuse_search_results(
    bm25_results: list[dict], semantic_results: list[dict], k: float = 60
) -> list[dict]:
    combined_ranks: dict[int, HybridAccumulator] = {}

    for i, result in enumerate(bm25_results, start=1):
        doc_id = result["id"]
        if doc_id not in combined_ranks:
            combined_ranks[doc_id] = HybridAccumulator(
                title=result["title"],
                document=result["document"],
            )
        combined_ranks[doc_id].bm25_score = max(
            combined_ranks[doc_id].bm25_score, i
        )

    for i, result in enumerate(semantic_results, start=1):
        doc_id = result["id"]
        if doc_id not in combined_ranks:
            combined_ranks[doc_id] = HybridAccumulator(
                title=result["title"],
                document=result["document"],
            )
        combined_ranks[doc_id].semantic_score = max(
            combined_ranks[doc_id].semantic_score, i
        )

    fused_results = []
    for doc_id, data in combined_ranks.items():
        score_value = rrf_score(data.bm25_score, k) + \
            rrf_score(data.semantic_score, k)
        result = format_search_result(
            doc_id=doc_id,
            title=data.title,
            document=data.document,
            score=score_value,
            bm25_score=data.bm25_score,
            semantic_score=data.semantic_score,
        )
        fused_results.append(result)

    return sorted(fused_results, key=lambda x: x["score"], reverse=True)

# ===== CLI COMMANDS =====


def normalize_command(scores: list[float]) -> list[float]:
    normalized = normalize_scores(scores)
    for score in normalized:
        print(f"* {score:.4f}")
    return normalized


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: int = DEFAULT_SEARCH_LIMIT
) -> dict:
    movies = load_movies()
    searcher = HybridSearch(movies)
    original_query = query
    search_limit = limit
    results = searcher.weighted_search(query, alpha, search_limit)
    payload = {
        "original_query": original_query,
        "query": query,
        "alpha": alpha,
        "results": results,
    }
    print(
        f"Weighted Hybrid Search Results for '{payload['query']}' (alpha={payload['alpha']}):"
    )
    print(
        f"  Alpha {payload['alpha']}: {int(payload['alpha'] * 100)}% Keyword, {int((1 - payload['alpha']) * 100)}% Semantic"
    )
    for i, res in enumerate(payload["results"], 1):
        print(f"{i}. {res['title']}")
        print(f"   Hybrid Score: {res.get('score', 0):.3f}")
        metadata = res.get("metadata", {})
        if "bm25_score" in metadata and "semantic_score" in metadata:
            print(
                f"   BM25: {metadata['bm25_score']:.3f}, Semantic: {metadata['semantic_score']:.3f}"
            )
        print(f"   {res['document'][:100]}...")
        print()
    return payload


def rrf_search_command(
    query: str, k: float = 60, limit: int = DEFAULT_SEARCH_LIMIT, mode = ""
) -> dict:
    
    if mode != "":
        client = LlmClient()
        query = client.enhance(query, mode)

    movies = load_movies()
    searcher = HybridSearch(movies)
    original_query = query
    search_limit = limit
    results = searcher.rrf_search(query, k, search_limit)
    payload = {
        "original_query": original_query,
        "query": query,
        "k": k,
        "results": results,
    }
    print(
        f"Weighted Hybrid Search Results for '{payload['query']}' (k={payload['k']}):"
    )
    for i, res in enumerate(payload["results"], 1):
        print(f"{i}. {res['title']}")
        print(f"   RRF Score: {res.get('score', 0):.3f}")
        metadata = res.get("metadata", {})
        if "bm25_score" in metadata and "semantic_score" in metadata:
            print(
                f"   BM25 Rank: {metadata['bm25_score']}, Semantic Rank: {metadata['semantic_score']}"
            )
        print(f"   {res['document'][:100]}...")
        print()
    return payload
