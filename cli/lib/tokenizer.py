from __future__ import annotations

import string

from nltk.stem import PorterStemmer

from .file_utils import load_stopwords


_PUNCT_TABLE = str.maketrans('', '', string.punctuation)
_STOPWORDS: list[str] = load_stopwords()


def tokenize_text(text: str) -> list[str]:
    """Lowercase, remove punctuation, split on whitespace, remove stop words, return tokens."""
    tokens = text.casefold().translate(_PUNCT_TABLE).split()
    stopwords = _STOPWORDS
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens if token and token not in stopwords]
