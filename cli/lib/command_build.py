from __future__ import annotations

from .inverted_index import InvertedIndex


def cmd_build() -> None:
    indexer = InvertedIndex()
    indexer.build()
    indexer.save()
