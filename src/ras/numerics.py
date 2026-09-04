"""Numerically stable scalar/vector transforms used by RSA."""
from __future__ import annotations
import math
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_sigmoid(x: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -np.asarray(x, dtype=np.float64))


def logit(p: float, eps: float = 1e-8) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return math.log(p / (1.0 - p))
