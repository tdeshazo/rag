from __future__ import annotations

from typing import Any, DefaultDict
from collections import defaultdict

from .file_utils import load_movies, save_cache, load_cache
from .tokenizer import tokenize_text


class InvertedIndex:

    def __init__(self) -> "InvertedIndex":
        self.index: DefaultDict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Any] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        """Add a document id and tokenized text to the index"""
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)

    def get_documents(self, term) -> list[int]:
        """Retrieve document ids matching the search term"""
        matches = sorted(list(self.index.get(term, set())))
        return matches

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
    
    def load(self):
        """Load cached index and docmap from disk."""
        cached_index = load_cache("index")
        cached_docmap = load_cache("docmap")

        # Recast index as defaultdict in case unpickled as normal dict
        self.index = defaultdict(set, cached_index)
        self.docmap = cached_docmap