from __future__ import annotations

import string

from nltk.stem import PorterStemmer

from .file_utils import load_stopwords


_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def tokenize_text(text: str) -> set[str]:
    """Lowercase, remove punctuation, split on whitespace, remove stop words, return unique tokens."""
    stopwords: frozenset[str] = load_stopwords()
    unique_tokens = set(text.casefold().translate(_PUNCT_TABLE).split()) - stopwords

    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in unique_tokens if token]
