from __future__ import annotations

from .inverted_index import InvertedIndex


def build_index() -> None:
    indexer = InvertedIndex()
    indexer.build()
    indexer.save()
