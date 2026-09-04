"""Sparse semantic predicate compilers over quantized RSA coordinates."""
from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import List
import numpy as np
from .numerics import sigmoid, logit


@dataclass
class LUTFactorModel:
    idx: np.ndarray
    table: np.ndarray
    intercept: float
    n_bins: int
    name: str = "lut_factor"


@dataclass
class BoostedLUTModel:
    unary_idx: List[int]
    unary_tables: List[np.ndarray]
    pair_idx: List[tuple[int, int]]
    pair_tables: List[np.ndarray]
    intercept: float
    n_bins: int
    name: str = "residual_boosted_lut"


def _histograms(Q: np.ndarray, y: np.ndarray, n_bins: int):
    d = Q.shape[1]
    pos = Q[y]
    neg = Q[~y]
    hp = np.empty((d, n_bins), dtype=np.float64)
    hn = np.empty_like(hp)
    for j in range(d):
        hp[j] = np.bincount(pos[:, j], minlength=n_bins)
        hn[j] = np.bincount(neg[:, j], minlength=n_bins)
    return hp, hn


def _llr_stats(Q: np.ndarray, y: np.ndarray, n_bins: int, smoothing: str = "empirical_bayes", alpha: float = 1.0, eb_tau: float = 16.0):
    hp, hn = _histograms(Q, y, n_bins)
    if smoothing == "laplace":
        pp = (hp + alpha) / (hp.sum(axis=1, keepdims=True) + n_bins * alpha)
        pn = (hn + alpha) / (hn.sum(axis=1, keepdims=True) + n_bins * alpha)
    elif smoothing == "empirical_bayes":
        global_counts = hp + hn
        global_p = (global_counts + alpha) / (global_counts.sum(axis=1, keepdims=True) + n_bins * alpha)
        pp = (hp + eb_tau * global_p) / (hp.sum(axis=1, keepdims=True) + eb_tau)
        pn = (hn + eb_tau * global_p) / (hn.sum(axis=1, keepdims=True) + eb_tau)
    else:
        raise ValueError(smoothing)
    llr = np.log(np.maximum(pp, 1e-12) / np.maximum(pn, 1e-12))
    strength = np.sum((pp - pn) * llr, axis=1)
    mix = 0.5 * (pp + pn)
    mi = 0.5 * (
        np.sum(pp * np.log(np.maximum(pp, 1e-12) / np.maximum(mix, 1e-12)), axis=1)
        + np.sum(pn * np.log(np.maximum(pn, 1e-12) / np.maximum(mix, 1e-12)), axis=1)
    )
    return llr, strength, mi


def learn_llr_factor(Q: np.ndarray, y: np.ndarray, *, k: int = 28, n_bins: int, smoothing: str = "laplace", alpha: float = 1.0, eb_tau: float = 16.0, **_) -> LUTFactorModel:
    y = np.asarray(y, dtype=bool)
    llr, strength, _ = _llr_stats(Q, y, n_bins, smoothing=smoothing, alpha=alpha, eb_tau=eb_tau)
    idx = np.argsort(strength)[-min(k, Q.shape[1]):]
    prior = (float(y.sum()) + 0.5) / (len(y) + 1.0)
    return LUTFactorModel(np.asarray(idx, dtype=np.int32), llr[idx].astype(np.float32), logit(prior), n_bins, f"llr_{smoothing}")


def score_factor(Q: np.ndarray, model: LUTFactorModel) -> np.ndarray:
    score = np.full(len(Q), model.intercept, dtype=np.float32)
    for local, j in enumerate(model.idx):
        score += model.table[local, Q[:, int(j)]]
    return score


def _newton(qj, g, h, n_bins, l2):
    G = np.bincount(qj, weights=g, minlength=n_bins).astype(np.float64)
    H = np.bincount(qj, weights=h, minlength=n_bins).astype(np.float64)
    table = G / (H + l2)
    gain = 0.5 * float(np.sum(G * G / (H + l2)))
    return table, gain


def _candidate_pool(Q, y, n_bins, pool_size):
    _, strength, _ = _llr_stats(Q, y, n_bins, smoothing="empirical_bayes", eb_tau=float(n_bins))
    return np.argsort(strength)[-min(pool_size, Q.shape[1]):][::-1].astype(np.int32)


def fit_boosted_lut(Q: np.ndarray, y: np.ndarray, *, k: int = 28, n_bins: int, candidate_pool: int = 128, l2: float = 6.0, learning_rate: float = 0.55, refine_passes: int = 1) -> BoostedLUTModel:
    y = np.asarray(y, dtype=np.float64)
    prior = (y.sum() + 0.5) / (len(y) + 1.0)
    intercept = logit(float(prior))
    logits = np.full(len(y), intercept, dtype=np.float64)
    remaining = _candidate_pool(Q, y.astype(bool), n_bins, candidate_pool).tolist()
    unary_idx, unary_tables = [], []
    for _ in range(min(k, len(remaining))):
        p = np.clip(sigmoid(logits), 1e-6, 1 - 1e-6)
        g = y - p
        h = p * (1 - p)
        best_pos, best_gain, best_table = None, -np.inf, None
        if n_bins == 2 and remaining:
            rem = np.asarray(remaining, dtype=np.int32)
            qr = Q[:, rem].astype(np.float64)
            G1 = qr.T @ g
            H1 = qr.T @ h
            G0 = g.sum() - G1
            H0 = h.sum() - H1
            gains = 0.5 * (G0 * G0 / (H0 + l2) + G1 * G1 / (H1 + l2))
            pos = int(np.argmax(gains))
            best_pos = pos
            best_gain = float(gains[pos])
            best_table = np.array([G0[pos] / (H0[pos] + l2), G1[pos] / (H1[pos] + l2)])
        else:
            for pos, j in enumerate(remaining):
                table, gain = _newton(Q[:, j], g, h, n_bins, l2)
                if gain > best_gain:
                    best_gain, best_pos, best_table = gain, pos, table
        if best_pos is None:
            break
        j = int(remaining.pop(best_pos))
        table = (learning_rate * best_table).astype(np.float32)
        unary_idx.append(j)
        unary_tables.append(table)
        logits += table[Q[:, j]]
    for _ in range(max(0, refine_passes)):
        for t, j in enumerate(unary_idx):
            old = unary_tables[t].astype(np.float64)
            base = logits - old[Q[:, j]]
            p = np.clip(sigmoid(base), 1e-6, 1 - 1e-6)
            fresh, _ = _newton(Q[:, j], y - p, p * (1 - p), n_bins, l2)
            fresh = fresh.astype(np.float32)
            unary_tables[t] = fresh
            logits = base + fresh[Q[:, j]]
    return BoostedLUTModel(unary_idx, unary_tables, [], [], float(intercept), n_bins)


def score_boosted(Q: np.ndarray, model: BoostedLUTModel) -> np.ndarray:
    score = np.full(len(Q), model.intercept, dtype=np.float32)
    for j, table in zip(model.unary_idx, model.unary_tables):
        score += table[Q[:, j]]
    for (j, k), table in zip(model.pair_idx, model.pair_tables):
        score += table[Q[:, j], Q[:, k]]
    return score


def add_pair_interactions(Q: np.ndarray, y: np.ndarray, model: BoostedLUTModel, *, n_pairs: int = 4, pair_pool: int = 16, l2: float = 10.0, learning_rate: float = 0.5) -> BoostedLUTModel:
    y = np.asarray(y, dtype=np.float64)
    logits = score_boosted(Q, model).astype(np.float64)
    candidates = list(itertools.combinations(model.unary_idx[:min(pair_pool, len(model.unary_idx))], 2))
    pair_idx = list(model.pair_idx)
    pair_tables = [p.copy() for p in model.pair_tables]
    n_bins = model.n_bins
    for _ in range(min(n_pairs, len(candidates))):
        p = np.clip(sigmoid(logits), 1e-6, 1 - 1e-6)
        g = y - p
        h = p * (1 - p)
        best_pos, best_gain, best_table = None, -np.inf, None
        for pos, (j, k) in enumerate(candidates):
            joint = Q[:, j].astype(np.int64) * n_bins + Q[:, k].astype(np.int64)
            G = np.bincount(joint, weights=g, minlength=n_bins * n_bins)
            H = np.bincount(joint, weights=h, minlength=n_bins * n_bins)
            gain = 0.5 * float(np.sum(G * G / (H + l2)))
            if gain > best_gain:
                best_gain, best_pos = gain, pos
                best_table = (G / (H + l2)).reshape(n_bins, n_bins)
        if best_pos is None:
            break
        pair = candidates.pop(best_pos)
        table = (learning_rate * best_table).astype(np.float32)
        pair_idx.append((int(pair[0]), int(pair[1])))
        pair_tables.append(table)
        logits += table[Q[:, pair[0]], Q[:, pair[1]]]
    return BoostedLUTModel(list(model.unary_idx), [t.copy() for t in model.unary_tables], pair_idx, pair_tables, model.intercept, n_bins, f"{model.name}+{len(pair_idx)}pairs")
