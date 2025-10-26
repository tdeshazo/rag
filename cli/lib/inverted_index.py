from __future__ import annotations

import math
from typing import Any, DefaultDict
from collections import defaultdict, Counter

from .file_utils import load_movies, save_cache, load_cache
from .search_utils import tokenize_text


class InvertedIndex:

    def __init__(self) -> "InvertedIndex":
        self.index: DefaultDict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Any] = {}
        self.term_frequencies: dict[int, Counter] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """Add a document id and tokenized text to the index"""
        tokens = tokenize_text(text)
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term) -> list[int]:
        """Retrieve document ids matching the search term"""
        matches = sorted(list(self.index.get(term, set())))
        return matches

    def get_tf(self, doc_id: int, term: str) -> int:
        if not self.term_frequencies[doc_id]:
            raise ValueError(f"Invald document ID: {doc_id}")
        return self.term_frequencies.get(doc_id, {}).get(term, 0)

    def get_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N + 1) / (df + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)        

    def build(self) -> None:
        """Build the index and docmap from disk"""
        
        movies = load_movies()

        for m in movies:
            self.docmap[m['id']] = m
            self.__add_document(m['id'], f"{m['title']} {m['description']}")

    def save(self) -> None:
        """Save index and docmap to disk cache"""
        save_cache(self.index, "index")
        save_cache(self.docmap, "docmap")
        save_cache(self.term_frequencies, "term_frequencies")
    
    def load(self):
        """Load cached index and docmap from disk."""
        cached_index = load_cache("index")
        cached_docmap = load_cache("docmap")
        cached_tf = load_cache("term_frequencies")

        # Recast index as defaultdict in case unpickled as normal dict
        self.index = defaultdict(set, cached_index)
        self.docmap = cached_docmap
        self.term_frequencies = cached_tf