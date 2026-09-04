"""Universal random low-bit semantic substrate."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import numpy as np


@dataclass
class WhiteningState:
    mean: np.ndarray
    eigvecs: np.ndarray
    scales: np.ndarray
    gamma: float
    renormalize: bool = True


@dataclass
class QuantizedSubstrate:
    name: str
    bits: int
    n_bins: int
    Q_fit: np.ndarray
    Q_cal: np.ndarray
    Q_test: np.ndarray
    projection: np.ndarray
    projection_kind: str
    quantizer_kind: str
    bin_centroids: np.ndarray
    item_bytes_theoretical: float
    whitening: Optional[WhiteningState] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(d, d)).astype(np.float32)
    q, _ = np.linalg.qr(a)
    return q.astype(np.float32)


def independent_random_dictionary(d: int, m: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(d, m)).astype(np.float32)
    a /= np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-12)
    return a.astype(np.float32)


def fit_power_whitener(X: np.ndarray, gamma: float, eps_ratio: float = 1e-4, renormalize: bool = True) -> WhiteningState:
    mean = X.mean(axis=0).astype(np.float32)
    xc = X - mean
    cov = (xc.T @ xc) / max(len(X) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    eigvals = np.maximum(eigvals, eigvals.max() * eps_ratio + 1e-12)
    scales = np.power(eigvals, -0.5 * gamma)
    return WhiteningState(mean, eigvecs.astype(np.float32), scales.astype(np.float32), float(gamma), bool(renormalize))


def apply_power_whitener(X: np.ndarray, state: WhiteningState) -> np.ndarray:
    tmp = (X - state.mean) @ state.eigvecs
    out = (tmp * state.scales[None, :]) @ state.eigvecs.T
    if state.renormalize:
        out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


def _uniform_fit(Z: np.ndarray, bits: int, clip_q: Tuple[float, float] = (0.005, 0.995)):
    levels = 2 ** bits
    lo = np.quantile(Z, clip_q[0], axis=0).astype(np.float32)
    hi = np.quantile(Z, clip_q[1], axis=0).astype(np.float32)
    return lo, hi, np.maximum(hi - lo, 1e-8), levels


def _uniform_apply(Z, state):
    lo, hi, span, levels = state
    u = (np.clip(Z, lo, hi) - lo) / span
    return np.rint(u * (levels - 1)).astype(np.uint8)


def _quantile_edges(Z: np.ndarray, levels: int) -> np.ndarray:
    qs = np.arange(1, levels, dtype=np.float64) / levels
    return np.quantile(Z, qs, axis=0).T.astype(np.float32)


def _quantile_apply(Z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    q = np.empty(Z.shape, dtype=np.uint8)
    for j in range(Z.shape[1]):
        q[:, j] = np.searchsorted(edges[j], Z[:, j], side="right").astype(np.uint8)
    return q


def _centroids(Z: np.ndarray, Q: np.ndarray, n_bins: int) -> np.ndarray:
    out = np.zeros((Q.shape[1], n_bins), dtype=np.float32)
    gm = Z.mean(axis=0)
    for j in range(Q.shape[1]):
        counts = np.bincount(Q[:, j], minlength=n_bins).astype(np.float64)
        sums = np.bincount(Q[:, j], weights=Z[:, j], minlength=n_bins)
        out[j] = np.divide(sums, counts, out=np.full(n_bins, gm[j], dtype=np.float64), where=counts > 0)
    return out


def build_substrate(X_fit, X_cal, X_test, *, name: str, seed: int, bits: int = 4, m: Optional[int] = None, projection_kind: str = "orthogonal", quantizer_kind: str = "quantile", whitening_gamma: Optional[float] = None, whitening_renormalize: bool = True, projection: Optional[np.ndarray] = None) -> QuantizedSubstrate:
    d = X_fit.shape[1]
    m = d if m is None else int(m)
    rng = np.random.default_rng(seed)
    whitening = None
    if whitening_gamma is None:
        Xf, Xc, Xt = X_fit, X_cal, X_test
    else:
        whitening = fit_power_whitener(X_fit, whitening_gamma, renormalize=whitening_renormalize)
        Xf, Xc, Xt = (apply_power_whitener(X, whitening) for X in (X_fit, X_cal, X_test))
    if projection is None:
        if projection_kind == "orthogonal":
            if m != d:
                raise ValueError("Orthogonal projection requires m == d")
            projection = random_orthogonal(d, rng)
        elif projection_kind == "independent":
            projection = independent_random_dictionary(d, m, rng)
        else:
            raise ValueError(projection_kind)
    Zf, Zc, Zt = [(X @ projection).astype(np.float32) for X in (Xf, Xc, Xt)]
    if bits == 1:
        Qf, Qc, Qt = [(Z >= 0).astype(np.uint8) for Z in (Zf, Zc, Zt)]
        n_bins = 2
        quantizer_kind = "sign"
    else:
        n_bins = 2 ** bits
        if quantizer_kind == "uniform":
            state = _uniform_fit(Zf, bits)
            Qf, Qc, Qt = [_uniform_apply(Z, state) for Z in (Zf, Zc, Zt)]
        elif quantizer_kind == "quantile":
            edges = _quantile_edges(Zf, n_bins)
            Qf, Qc, Qt = [_quantile_apply(Z, edges) for Z in (Zf, Zc, Zt)]
        else:
            raise ValueError(quantizer_kind)
    return QuantizedSubstrate(name, bits, n_bins, Qf, Qc, Qt, projection.astype(np.float32), projection_kind, quantizer_kind, _centroids(Zf, Qf, n_bins), m * bits / 8.0, whitening, {"m": m, "d": d, "seed": seed, "whitening_gamma": whitening_gamma})


def geometry_correlation(X_test: np.ndarray, substrate: QuantizedSubstrate, seed: int = 7, n_pairs: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, len(X_test), size=n_pairs)
    b = rng.integers(0, len(X_test), size=n_pairs)
    cosine = np.sum(X_test[a] * X_test[b], axis=1)
    if substrate.bits == 1:
        surrogate = np.mean(substrate.Q_test[a] == substrate.Q_test[b], axis=1)
    else:
        surrogate = -np.mean(np.abs(substrate.Q_test[a].astype(np.int16) - substrate.Q_test[b].astype(np.int16)), axis=1)
    return float(np.corrcoef(cosine, surrogate)[0, 1])
