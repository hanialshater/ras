"""Centered one-bit semantic substrates and BBQ-inspired linear scoring.

This module deliberately separates three ideas:

1. ``CenteredBinaryEncoder`` is the concept-independent transformation fitted
   once for a catalog.  It stores a centroid and an optional orthogonal
   projection.
2. ``CenteredBinaryCode`` is the historical fit/cal/test experiment container.
3. ``score_compiled_linear`` evaluates a full-precision linear semantic head on
   the compressed representation.  Predicate weights may optionally be
   quantized to int4, matching the asymmetric 1-bit-document / 4-bit-query
   spirit of Elastic BBQ.

This is *BBQ-inspired*, not a byte-for-byte reimplementation of Lucene BBQ.
The exact Lucene estimator uses additional correction terms and a specialized
bitwise kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .substrate import random_orthogonal


@dataclass
class CenteredBinaryEncoder:
    """Concept-independent encoder fitted once and reused for every predicate."""

    centroid: np.ndarray
    projection: np.ndarray
    projection_kind: str
    seed: int
    with_corrections: bool = True

    @property
    def d(self) -> int:
        return int(self.centroid.shape[0])

    @property
    def packed_bytes(self) -> int:
        return (self.d + 7) // 8

    @property
    def item_bytes_theoretical(self) -> float:
        return float(self.packed_bytes + (8 if self.with_corrections else 0))


@dataclass
class CenteredBinaryCode:
    name: str
    Q_fit: np.ndarray
    Q_cal: np.ndarray
    Q_test: np.ndarray
    correction_fit: np.ndarray
    correction_cal: np.ndarray
    correction_test: np.ndarray
    centroid: np.ndarray
    projection: np.ndarray
    projection_kind: str
    item_bytes_theoretical: float
    meta: dict

    @property
    def d(self) -> int:
        return int(self.Q_fit.shape[1])


def two_level_corrections(residual: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Least-squares two-level reconstruction values per item.

    For each item, all negative residual coordinates are reconstructed by their
    mean and all positive coordinates by their mean.  This is the optimal
    two-level reconstruction conditional on the sign partition.
    """
    pos = bits.astype(bool)
    neg = ~pos
    pos_n = pos.sum(axis=1)
    neg_n = neg.sum(axis=1)
    pos_sum = np.where(pos, residual, 0.0).sum(axis=1)
    neg_sum = np.where(neg, residual, 0.0).sum(axis=1)
    hi = np.divide(pos_sum, pos_n, out=np.zeros_like(pos_sum, dtype=np.float32), where=pos_n > 0)
    lo = np.divide(neg_sum, neg_n, out=np.zeros_like(neg_sum, dtype=np.float32), where=neg_n > 0)
    return np.column_stack([lo, hi]).astype(np.float32)


# Backward-compatible private name used by older experiment code.
_two_level_corrections = two_level_corrections


def fit_centered_binary_encoder(
    X_fit: np.ndarray,
    *,
    seed: int = 7,
    projection_kind: Literal["identity", "orthogonal"] = "identity",
    with_corrections: bool = True,
) -> CenteredBinaryEncoder:
    """Fit the catalog-wide binary encoder on representative embeddings."""
    X_fit = np.asarray(X_fit, dtype=np.float32)
    if X_fit.ndim != 2:
        raise ValueError("X_fit must be a 2D [items, dimensions] array")
    d = int(X_fit.shape[1])
    if projection_kind == "identity":
        projection = np.eye(d, dtype=np.float32)
    elif projection_kind == "orthogonal":
        projection = random_orthogonal(d, np.random.default_rng(int(seed)))
    else:
        raise ValueError(projection_kind)
    z_fit = (X_fit @ projection).astype(np.float32)
    centroid = z_fit.mean(axis=0).astype(np.float32)
    return CenteredBinaryEncoder(
        centroid=centroid,
        projection=projection.astype(np.float32),
        projection_kind=str(projection_kind),
        seed=int(seed),
        with_corrections=bool(with_corrections),
    )


def encode_centered_binary(
    X: np.ndarray,
    encoder: CenteredBinaryEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode arbitrary items using a previously fitted catalog encoder.

    Returns unpacked sign bits with shape ``[n, d]`` and two per-item correction
    values with shape ``[n, 2]``.  Use :func:`pack_document_bits` for the serving
    representation.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != encoder.d:
        raise ValueError(f"X must have shape [n, {encoder.d}]")
    z = (X @ encoder.projection).astype(np.float32)
    residual = z - encoder.centroid
    q = (residual > 0).astype(np.uint8)
    if encoder.with_corrections:
        corrections = two_level_corrections(residual, q)
    else:
        corrections = np.zeros((len(q), 2), dtype=np.float32)
    return q, corrections


def build_centered_binary_code(
    X_fit: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 7,
    projection_kind: Literal["identity", "orthogonal"] = "identity",
    with_corrections: bool = True,
) -> CenteredBinaryCode:
    """Historical fit/cal/test wrapper around the reusable encoder API."""
    encoder = fit_centered_binary_encoder(
        X_fit,
        seed=int(seed),
        projection_kind=projection_kind,
        with_corrections=bool(with_corrections),
    )
    Qf, Cf = encode_centered_binary(X_fit, encoder)
    Qc, Cc = encode_centered_binary(X_cal, encoder)
    Qt, Ct = encode_centered_binary(X_test, encoder)
    correction_bytes = 8.0 if with_corrections else 0.0
    return CenteredBinaryCode(
        name=f"centered_binary_{projection_kind}",
        Q_fit=Qf,
        Q_cal=Qc,
        Q_test=Qt,
        correction_fit=Cf,
        correction_cal=Cc,
        correction_test=Ct,
        centroid=encoder.centroid,
        projection=encoder.projection,
        projection_kind=encoder.projection_kind,
        item_bytes_theoretical=float(encoder.packed_bytes + correction_bytes),
        meta={"d": encoder.d, "seed": int(seed), "correction_bytes": correction_bytes},
    )


def quantize_weight_int4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Uniform per-predicate unsigned int4 quantization of a weight vector."""
    w = np.asarray(weight, dtype=np.float32)
    lo = float(w.min())
    hi = float(w.max())
    span = max(hi - lo, 1e-12)
    scale = span / 15.0
    q = np.clip(np.rint((w - lo) / scale), 0, 15).astype(np.uint8)
    decoded = (lo + scale * q.astype(np.float32)).astype(np.float32)
    return q, decoded, lo, float(scale)


def score_compiled_linear(
    Q: np.ndarray,
    corrections: np.ndarray,
    code: CenteredBinaryCode,
    weight: np.ndarray,
    intercept: float,
    *,
    int4_query: bool = False,
) -> np.ndarray:
    """Approximate ``x @ weight + intercept`` from centered one-bit codes."""
    wz = (code.projection.T @ np.asarray(weight, dtype=np.float32)).astype(np.float32)
    if int4_query:
        _, wz, _, _ = quantize_weight_int4(wz)
    pos_weight_sum = Q.astype(np.float32) @ wz
    sum_w = float(wz.sum())
    lo = corrections[:, 0]
    hi = corrections[:, 1]
    base = float(intercept) + float(code.centroid @ wz)
    return (base + lo * (sum_w - pos_weight_sum) + hi * pos_weight_sum).astype(np.float32)


def pack_document_bits(Q: np.ndarray) -> np.ndarray:
    """Pack document sign bits into bytes for export to a native executor."""
    return np.packbits(np.asarray(Q, dtype=np.uint8), axis=1, bitorder="little")


def int4_weight_bitplanes(qweight: np.ndarray) -> np.ndarray:
    """Return four packed bit planes for a 0..15 predicate-weight vector."""
    q = np.asarray(qweight, dtype=np.uint8)
    planes = [np.packbits((q >> b) & 1, bitorder="little") for b in range(4)]
    return np.stack(planes).astype(np.uint8)


__all__ = [
    "CenteredBinaryEncoder",
    "CenteredBinaryCode",
    "fit_centered_binary_encoder",
    "encode_centered_binary",
    "two_level_corrections",
    "build_centered_binary_code",
    "quantize_weight_int4",
    "score_compiled_linear",
    "pack_document_bits",
    "int4_weight_bitplanes",
]
