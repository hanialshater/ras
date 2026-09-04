"""Simple on-disk NumPy cache for expensive embedding stages."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Any
import numpy as np


def cache_key(meta: Dict[str, Any], n: int = 16) -> str:
    payload = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:n]


def load_or_compute_array(cache_dir: str | Path, prefix: str, meta: Dict[str, Any], fn: Callable[[], np.ndarray]) -> np.ndarray:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(meta)
    arr_path = cache_dir / f"{prefix}_{key}.npy"
    meta_path = cache_dir / f"{prefix}_{key}.json"
    if arr_path.exists():
        return np.load(arr_path, mmap_mode=None)
    arr = np.asarray(fn())
    np.save(arr_path, arr)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return arr


__all__ = ["cache_key", "load_or_compute_array"]
