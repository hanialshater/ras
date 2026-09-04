"""Metrics and confidence intervals used by RSA experiments."""
from __future__ import annotations
from typing import Dict, Iterable
import numpy as np
from .core import best_f1_threshold, metric_row


def bootstrap_mean_ci(values: Iterable[float], seed: int = 7, n_boot: int = 2000, alpha: float = 0.05) -> Dict[str, float]:
    x = np.asarray(list(values), dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return {
        "mean": float(x.mean()),
        "lo": float(np.quantile(samples, alpha / 2)),
        "hi": float(np.quantile(samples, 1 - alpha / 2)),
        "n": int(len(x)),
    }


def binary_ranking_stats(truth: np.ndarray, selected: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    selected = np.asarray(selected, dtype=np.int64)
    total = int(truth.sum())
    hits = int(truth[selected].sum()) if len(selected) else 0
    k = int(len(selected))
    max_hits = min(k, total)
    recall = hits / total if total else np.nan
    purity = hits / k if k else np.nan
    max_recall = max_hits / total if total else np.nan
    return {
        "hits": hits,
        "k": k,
        "total_true": total,
        "recall": float(recall),
        "purity": float(purity),
        "max_recall": float(max_recall),
        "recall_efficiency": float(recall / max_recall) if max_recall and np.isfinite(max_recall) else np.nan,
    }


__all__ = ["best_f1_threshold", "metric_row", "bootstrap_mean_ci", "binary_ranking_stats"]
