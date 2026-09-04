"""Compositional semantic query algebra."""
from __future__ import annotations
from typing import Iterable, Mapping
import numpy as np
from .numerics import log_sigmoid


def compose_logprob(calibrated_logits: np.ndarray, signs: Iterable[int]) -> np.ndarray:
    L = np.asarray(calibrated_logits, dtype=np.float64)
    signs = np.asarray(list(signs), dtype=np.int8)
    if L.ndim != 2 or L.shape[1] != len(signs):
        raise ValueError("calibrated_logits and signs disagree")
    return log_sigmoid(L * signs[None, :]).sum(axis=1)


def compose_query(calibrated_logits: np.ndarray, name_to_idx: Mapping[str, int], positive: Iterable[str] = (), negative: Iterable[str] = ()) -> np.ndarray:
    positive = list(positive)
    negative = list(negative)
    names = positive + negative
    if not names:
        return np.zeros(len(calibrated_logits), dtype=np.float64)
    idx = [name_to_idx[n] for n in names]
    signs = [1] * len(positive) + [-1] * len(negative)
    return compose_logprob(np.asarray(calibrated_logits)[:, idx], signs)


def softmin_scores(L: np.ndarray, tau: float) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    tau = max(float(tau), 1e-8)
    A = -L / tau
    m = np.max(A, axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(A - m).sum(axis=1))
    return -tau * lse + tau * np.log(L.shape[1])


__all__ = ["compose_logprob", "compose_query", "softmin_scores"]
