"""Metrics and confidence intervals used by RSA experiments."""
from __future__ import annotations
from typing import Dict, Iterable
import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def best_f1_threshold(scores: np.ndarray, truth: np.ndarray, n_grid: int = 160) -> float:
    scores = np.asarray(scores)
    truth = np.asarray(truth, dtype=bool)
    if len(scores) == 0:
        return 0.0
    thresholds = np.unique(scores) if np.unique(scores).size <= n_grid else np.unique(np.quantile(scores, np.linspace(0.005, 0.995, n_grid)))
    vals = [f1_score(truth, scores >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(vals))])


def metric_row(truth: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(scores) >= threshold
    return {
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "ap": float(average_precision_score(truth, scores)) if truth.any() else float("nan"),
    }


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
