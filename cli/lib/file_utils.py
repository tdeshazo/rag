from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOPWORDS_PATH = PROJECT_ROOT / "data" / "stopwords.txt"
CACHE_DIR = PROJECT_ROOT / "cache"


def load_movies() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Missing data 'movies.json'")
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> frozenset[str]:
    if not STOPWORDS_PATH.exists():
        raise FileNotFoundError("Missing data 'stopwords.txt'")
    with open(STOPWORDS_PATH, "r") as f:
        return frozenset(f.read().splitlines())


def _cache_path(name: str | Path) -> Path:
    """
    Internal helper: normalize a cache key or filename
    to a full path like <PROJECT_ROOT>/cache/<name>.pkl
    """
    # Accept either bare key "tfidf" or full path
    p = Path(name)
    if p.suffix != ".pkl":
        p = p.with_suffix(".pkl")
    return CACHE_DIR / p.name


def save_cache(obj: Any, name: str | Path) -> Path:
    """
    Serialize obj to pickle in CACHE_DIR.

    Returns the full path written.
    """
    path = _cache_path(name)
    # make sure cache dir exists
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_cache(name: str | Path) -> Any:
    """
    Load and return a cached pickle object from CACHE_DIR.

    Raises FileNotFoundError if the file is missing.
    """
    path = _cache_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache file '{path.name}'")
    with open(path, "rb") as f:
        return pickle.load(f)