"""Centered one-bit semantic substrates and BBQ-inspired linear scoring.

This module deliberately separates two ideas:

1. ``CenteredBinaryCode`` stores one bit per projected embedding dimension after
   subtracting a fit-set centroid.  Two per-item least-squares correction values
   store the mean residual value on the negative and positive sides.
2. ``score_compiled_linear`` evaluates a full-precision linear semantic head on
   that compressed representation.  The predicate weights may optionally be
   quantized to int4, matching the asymmetric 1-bit-document / 4-bit-query spirit
   of Elastic BBQ.

This is *BBQ-inspired*, not a byte-for-byte reimplementation of Lucene BBQ.
It is intended as a controlled semantic-predicate baseline: can a centered
binary document code plus tiny corrective state preserve a learned linear
predicate?  The exact Lucene estimator uses additional correction terms and a
specialized bitwise kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .substrate import random_orthogonal


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


def _two_level_corrections(residual: np.ndarray, bits: np.ndarray) -> np.ndarray:
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


def build_centered_binary_code(
    X_fit: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int = 7,
    projection_kind: Literal["identity", "orthogonal"] = "identity",
    with_corrections: bool = True,
) -> CenteredBinaryCode:
    """Fit a global centroid and store one sign bit per projected dimension."""
    d = int(X_fit.shape[1])
    if projection_kind == "identity":
        projection = np.eye(d, dtype=np.float32)
    elif projection_kind == "orthogonal":
        projection = random_orthogonal(d, np.random.default_rng(int(seed)))
    else:
        raise ValueError(projection_kind)

    Zf = (X_fit @ projection).astype(np.float32)
    Zc = (X_cal @ projection).astype(np.float32)
    Zt = (X_test @ projection).astype(np.float32)
    centroid = Zf.mean(axis=0).astype(np.float32)
    Rf, Rc, Rt = Zf - centroid, Zc - centroid, Zt - centroid
    Qf, Qc, Qt = [(r > 0).astype(np.uint8) for r in (Rf, Rc, Rt)]

    if with_corrections:
        Cf = _two_level_corrections(Rf, Qf)
        Cc = _two_level_corrections(Rc, Qc)
        Ct = _two_level_corrections(Rt, Qt)
        correction_bytes = 8.0  # two f32 values per item
    else:
        Cf = np.zeros((len(Qf), 2), dtype=np.float32)
        Cc = np.zeros((len(Qc), 2), dtype=np.float32)
        Ct = np.zeros((len(Qt), 2), dtype=np.float32)
        correction_bytes = 0.0

    return CenteredBinaryCode(
        name=f"centered_binary_{projection_kind}",
        Q_fit=Qf,
        Q_cal=Qc,
        Q_test=Qt,
        correction_fit=Cf,
        correction_cal=Cc,
        correction_test=Ct,
        centroid=centroid,
        projection=projection.astype(np.float32),
        projection_kind=projection_kind,
        item_bytes_theoretical=d / 8.0 + correction_bytes,
        meta={"d": d, "seed": int(seed), "correction_bytes": correction_bytes},
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
    """Approximate ``x @ weight + intercept`` from centered one-bit codes.

    A row residual is reconstructed with two values, ``lo`` on zero bits and
    ``hi`` on one bits.  For an orthogonal projection ``z=xR``, the equivalent
    linear weight is ``R.T @ weight``.
    """
    wz = (code.projection.T @ np.asarray(weight, dtype=np.float32)).astype(np.float32)
    if int4_query:
        _, wz, _, _ = quantize_weight_int4(wz)

    # Reconstructed residual dot product:
    # lo * sum_{bit=0} w + hi * sum_{bit=1} w.
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
    "CenteredBinaryCode",
    "build_centered_binary_code",
    "quantize_weight_int4",
    "score_compiled_linear",
    "pack_document_bits",
    "int4_weight_bitplanes",
]
