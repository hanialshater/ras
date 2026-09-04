"""Compositional semantic query algebra."""
from __future__ import annotations
from typing import Iterable, Mapping
import numpy as np
from .core import log_sigmoid, softmin_scores, correlation_weights


def compose_logprob(calibrated_logits: np.ndarray, signs: Iterable[int]) -> np.ndarray:
    """Compose calibrated concept logits using log-probability AND/NOT."""
    L = np.asarray(calibrated_logits, dtype=np.float64)
    signs = np.asarray(list(signs), dtype=np.int8)
    if L.ndim != 2 or L.shape[1] != len(signs):
        raise ValueError("calibrated_logits and signs disagree")
    return log_sigmoid(L * signs[None, :]).sum(axis=1)


def compose_query(
    calibrated_logits: np.ndarray,
    name_to_idx: Mapping[str, int],
    positive: Iterable[str] = (),
    negative: Iterable[str] = (),
) -> np.ndarray:
    """Compose named positive and negative predicates into one query score."""
    positive = list(positive)
    negative = list(negative)
    names = positive + negative
    if not names:
        return np.zeros(len(calibrated_logits), dtype=np.float64)
    idx = [name_to_idx[n] for n in names]
    signs = [1] * len(positive) + [-1] * len(negative)
    return compose_logprob(np.asarray(calibrated_logits)[:, idx], signs)


__all__ = ["compose_logprob", "compose_query", "softmin_scores", "correlation_weights"]
