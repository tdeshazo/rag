import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ALPHA = 0.5

DEFAULT_SEARCH_LIMIT = 5
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


def format_search_result(
    doc_id: str, title: str, document: str, score: float, **metadata: Any
) -> dict[str, Any]:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }

def _cache_path(name: str | os.PathLike[str]) -> Path:
    """
    Internal helper: normalize a cache key or filename
    to a full path like <PROJECT_ROOT>/cache/<name>.pkl
    """
    p = Path(name)
    if not p.suffix:
        p = p.with_suffix(".pkl")
    if p.is_absolute():
        return p
    return Path(CACHE_DIR) / p.name


def _save_pkl(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def _save_nmpy(obj: Any, path: Path) -> None:
    np.save(path, obj)


def _load_nmpy(path: Path) -> Any:
    return np.load(path)


def _save_json(obj: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_cache(obj: Any, name: str | os.PathLike[str]) -> str:
    """
    Serialize obj to pickle in CACHE_DIR.

    Returns the full path written.
    """
    path = _cache_path(name)

    # make sure cache dir exists
    path.parent.mkdir(parents=True, exist_ok=True)

    match path.suffix:
        case ".pkl":
            _save_pkl(obj, path)
        case ".npy":
            _save_nmpy(obj, path)
        case ".json":
            _save_json(obj, path)
        case _:
            raise ValueError
    return str(path)
    

def load_cache(name: str | os.PathLike[str], *, force: bool = False) -> Any:
    """
    Load and return a cached pickle object from CACHE_DIR.

    Raises FileNotFoundError if the file is missing.
    """
    path = _cache_path(name)
    if not path.exists():
        if force:
            return None
        raise FileNotFoundError(f"Missing cache file '{path.name}'")

    match path.suffix:
        case ".pkl":
            return _load_pkl(path)
        case ".npy":
            return _load_nmpy(path)
        case ".json":
            return _load_json(path)
        case _:
            return None
