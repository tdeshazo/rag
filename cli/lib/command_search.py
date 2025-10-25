from __future__ import annotations

import sys

from .tokenizer import tokenize_text
from .inverted_index import InvertedIndex


DEFAULT_SEARCH_LIMIT = 5


def get_matching_docs(index: InvertedIndex, query: str, k:int) -> list[int]:
    q_tokens = tokenize_text(query)
    print(q_tokens)
    if not q_tokens or not index.index or k <=0:
        return []

    matches: list[int] = []
    for token in q_tokens:
        match = index.get_documents(token)
        matches.extend(match)
        if len(matches) >= k:
            break
    return matches


def search_movies(query: str, k: int):
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error loading cache: {e}")
        sys.exit(1)

    matches = get_matching_docs(index, query, k)

    for i in range(min(k, len(matches))):
        doc = index.docmap[matches[i]]
        print(matches[i], doc['title'])
