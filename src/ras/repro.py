"""Reproducibility helpers."""
from __future__ import annotations
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def environment_manifest() -> Dict[str, Any]:
    manifest: Dict[str, Any] = {"python": sys.version, "platform": platform.platform(), "git_sha": git_sha()}
    for name in ["numpy", "pandas", "sklearn", "torch", "transformers", "sentence_transformers", "datasets"]:
        try:
            mod = __import__(name)
            manifest[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            manifest[name] = None
    try:
        import torch
        manifest["cuda_available"] = bool(torch.cuda.is_available())
        manifest["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return manifest


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


__all__ = ["git_sha", "environment_manifest", "write_json"]
