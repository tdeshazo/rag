from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

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


def load_stopwords() -> list[str]:
    if not STOPWORDS_PATH.exists():
        raise FileNotFoundError("Missing data 'stopwords.txt'")
    with open(STOPWORDS_PATH, "r") as f:
        return f.read().splitlines()


def _cache_path(name: str | Path) -> Path:
    """
    Internal helper: normalize a cache key or filename
    to a full path like <PROJECT_ROOT>/cache/<name>.pkl
    """
    # Accept either bare key "tfidf" or full path

    p = Path(name)
    return CACHE_DIR / p.name

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


def save_cache(obj: Any, name: str | Path) -> Path:
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
    return path
    

def load_cache(name: str | Path, *, force: bool=False) -> Any:
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
            
