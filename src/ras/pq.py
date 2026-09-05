"""PQ scoring with equally quantized predicate heads and explicit activation."""
from __future__ import annotations
import time
import numpy as np
from .binary import quantize_weight_int4


def train_pq(x_fit, *arrays, m=64, seed=7):
    import faiss
    x_fit = np.ascontiguousarray(x_fit, dtype=np.float32)
    if x_fit.shape[1] % m:
        raise ValueError("PQ subspaces must divide embedding dimension")
    if len(x_fit) < 256:
        raise ValueError("8-bit PQ needs at least 256 fit rows")
    pq = faiss.ProductQuantizer(x_fit.shape[1], m, 8)
    pq.cp.seed = int(seed)
    pq.train(x_fit)
    codebook = faiss.vector_to_array(pq.centroids).reshape(m, 256, x_fit.shape[1] // m).copy()
    codes = [pq.compute_codes(np.ascontiguousarray(x, dtype=np.float32)) for x in arrays]
    return codebook, codes


def compile_pq_head(codebook, weight, intercept, *, int4=False):
    w = np.asarray(weight, dtype=np.float32)
    metadata = {"intercept": float(intercept), "weight_format": "fp32"}
    if int4:
        q, w, lo, scale = quantize_weight_int4(w)
        padded = np.pad(q, (0, len(q) % 2))
        metadata.update(
            weight_format="int4", packed_weights=(padded[::2] | (padded[1::2] << 4)),
            weight_lo=lo, weight_scale=scale,
        )
    else:
        metadata["weights"] = w.copy()
    t0 = time.perf_counter()
    lut = np.einsum("mkd,md->mk", codebook, w.reshape(codebook.shape[0], -1)).astype(np.float32)
    metadata["activation_ms"] = (time.perf_counter() - t0) * 1000
    metadata["stored_bytes"] = (len(w) + 1) // 2 + 20 if int4 else w.nbytes + 12
    metadata["active_bytes"] = lut.nbytes + 12
    return lut, metadata


def score_pq(codes, lut, intercept):
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != lut.shape[0] or lut.shape[1] != 256:
        raise ValueError("only one-byte PQ subcodes are supported")
    scores = np.full(len(codes), intercept, dtype=np.float32)
    for j in range(lut.shape[0]):
        scores += lut[j, codes[:, j]]
    return scores
