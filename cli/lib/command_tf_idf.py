from __future__ import annotations

import sys

from .search_utils import tokenize_text
from .inverted_index import InvertedIndex


def cmd_tf_idf(doc_id: int, term: str):
    token = tokenize_text(term)
    if len(token) > 1:
        print(f"Requires one token, but received {len(token)}")
        sys.exit(1)

    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        print(f"Error loading cache: {e}")
        sys.exit(1)

    try:
        idf = index.get_idf(token[0])
        tf = index.get_tf(doc_id, token[0])
        tf_idf = idf * tf
    except ValueError as e:
        print(e)
        sys.exit(1)

    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
