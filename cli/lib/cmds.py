from __future__ import annotations

import sys
from typing import Callable, Any, Optional, Dict

from .tokenizer import tokenize_text
from .inverted_index import InvertedIndex


DEFAULT_SEARCH_LIMIT = 1


def cmd_build() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def _die(msg: str, code: int = 1):
    print(msg, file=sys.stderr)
    sys.exit(code)


def _load_existing() -> InvertedIndex:
    index = InvertedIndex()
    try:
        index.load()
    except FileNotFoundError as e:
        _die(f"Error loading cache: {e}")
    return index


def _one_token(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        _die(f"Requires one token, but received {len(tokens)}")
    return tokens[0]


def _run_with_index_and_term(
    term: str,
    compute: Callable[[InvertedIndex, str], Any],
    *,
    fmt: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Validate single-token input, load index, compute value, print or format."""
    term = _one_token(term)
    index = _load_existing()
    try:
        value = compute(index, term)
    except ValueError as e:
        _die(str(e))
    if fmt:
        data = {"term": term, "value": value}
        if context:
            data.update(context)
        print(fmt.format(**data))
    else:
        print(value)


def cmd_bm25_idf(term: str) -> None:
    _run_with_index_and_term(
        term,
        lambda idx, t: idx.get_bm25_idf(t),
        fmt="BM25 IDF score of '{term}': {value:.2f}",
    )


def cmd_tf(doc_id: int, term: str) -> None:
    _run_with_index_and_term(
        term,
        lambda idx, t: idx.get_tf(doc_id, t),
        fmt=None,
    )


def cmd_idf(term: str) -> None:
    _run_with_index_and_term(
        term,
        lambda idx, t: idx.get_idf(t),
        fmt="Inverse document frequency of '{term}': {value:.2f}",
    )


def cmd_tf_idf(doc_id: int, term: str) -> None:
    _run_with_index_and_term(
        term,
        lambda idx, t: idx.get_tf(doc_id, t) * idx.get_idf(t),
        fmt="TF-IDF score of '{term}' in document '{doc_id}': {value:.2f}",
        context={"doc_id": doc_id},
    )


def cmd_bm25_idf(term: str) -> None:
    _run_with_index_and_term(
        term,
        lambda idx, t: idx.get_bm25_idf(t),
        fmt="BM25 IDF score of '{term}': {value:.2f}",
    )


def cmd_bm25_tf(doc_id: int, term: str, k1: int, b: int) -> None:
        _run_with_index_and_term(
        term,
        lambda idx, t: idx.get_bm25_tf(doc_id, t, k1, b),
        fmt="BM25 TF score of '{term}' in document '{doc_id}': {value:.2f}",
        context={"doc_id": doc_id},
    )


def cmd_search(query: str, k: int) -> None:
    index = _load_existing()
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

    for i in range(min(k, len(matches))):
        doc = index.docmap[matches[i]]
        print(matches[i], doc['title'])
