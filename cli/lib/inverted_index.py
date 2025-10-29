from __future__ import annotations

import math
from typing import Any, DefaultDict
from collections import defaultdict, Counter

from .file_utils import load_movies, save_cache, load_cache
from .tokenizer import tokenize_text


class InvertedIndex:

    def __init__(self) -> "InvertedIndex":
        self.index: DefaultDict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Any] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """Add a document id and tokenized text to the index"""
        tokens = tokenize_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        lengths = list(self.doc_lengths.values())
        if not lengths:
            return 0.0
        return sum(lengths) / len(lengths)

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

    def get_bm25_tf(self, doc_id: int, term: str, k1: float, b: float) -> float:
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0.0)
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        try:
            return (tf * (k1 + 1)) / (tf + k1 * length_norm)
        except ZeroDivisionError:
            return 0.0

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
        save_cache(self.doc_lengths, "doc_lengths")
    
    def load(self):
        """Load cached index and docmap from disk."""
        self.index = defaultdict(set, load_cache("index"))
        self.docmap = load_cache("docmap")
        self.term_frequencies = load_cache("term_frequencies")
        self.doc_lengths = load_cache("doc_lengths")
