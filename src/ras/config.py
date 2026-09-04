"""Configuration loading and deterministic run identifiers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping")
    return cfg


def config_hash(cfg: Dict[str, Any], n: int = 10) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:n]


__all__ = ["load_config", "config_hash"]
