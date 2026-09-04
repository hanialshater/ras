"""Deterministic compound-query benchmark construction."""
from __future__ import annotations
import itertools
from dataclasses import dataclass, asdict
from typing import Sequence
import numpy as np
import pandas as pd
from .teachers import LATENT_SPECS


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    text: str
    exact: tuple[tuple[str, str], ...]
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    fit_truth: int

    def to_dict(self):
        d = asdict(self)
        d["exact"] = [list(x) for x in self.exact]
        return d


def apply_exact(df: pd.DataFrame, filters: Sequence[tuple[str, str]]) -> np.ndarray:
    m = np.ones(len(df), dtype=bool)
    for field, value in filters:
        m &= df[field].to_numpy() == value
    return m


def exact_filter_candidates(df_fit: pd.DataFrame, min_count: int = 200) -> list[tuple[tuple[str, str], ...]]:
    fields = ["baseColour", "subCategory", "articleType", "gender", "masterCategory"]
    singles = []
    for field in fields:
        for value, count in df_fit[field].value_counts().items():
            if int(count) >= min_count and str(value) != "Unknown":
                singles.append(((field, str(value)),))
    pairs = []
    for a, b in itertools.combinations(singles, 2):
        fa, fb = a[0][0], b[0][0]
        if fa == fb:
            continue
        if not ({fa, fb} & {"subCategory", "articleType", "masterCategory"}):
            continue
        filters = tuple(sorted((a[0], b[0])))
        if apply_exact(df_fit, filters).sum() >= min_count:
            pairs.append(filters)
    return list(dict.fromkeys([tuple()] + singles + pairs))


def _query_text(exact, positive, negative, spec_map):
    chunks = [spec_map[n]["query"] for n in positive]
    for _, value in exact:
        chunks.append(str(value).replace("=", " ").replace("_", " ").lower())
    chunks.extend([f"not {spec_map[n]['query']}" for n in negative])
    return " ".join(chunks)


def generate_query_benchmark(
    df_fit: pd.DataFrame,
    y_fit: np.ndarray,
    *,
    n_queries: int,
    seed: int,
    min_fit_truth: int = 100,
    max_positive_latents: int = 3,
    allow_negative: bool = True,
    specs=LATENT_SPECS,
) -> list[QuerySpec]:
    """Generate benchmark queries using fit data only, never test labels."""
    rng = np.random.default_rng(seed)
    names = [s["name"] for s in specs]
    name_to_idx = {n: i for i, n in enumerate(names)}
    spec_map = {s["name"]: s for s in specs}
    exacts = exact_filter_candidates(df_fit, min_count=max(2 * min_fit_truth, 200))
    seen = set()
    out = []
    max_attempts = max(20000, n_queries * 500)
    for _ in range(max_attempts):
        exact = exacts[int(rng.integers(0, len(exacts)))]
        npos = int(rng.integers(1, max_positive_latents + 1))
        positive = tuple(sorted(rng.choice(names, size=npos, replace=False).tolist()))
        remaining = [n for n in names if n not in positive]
        negative = tuple()
        if allow_negative and remaining and rng.random() < 0.55:
            negative = (str(rng.choice(remaining)),)
        key = (exact, positive, negative)
        if key in seen:
            continue
        seen.add(key)
        truth = apply_exact(df_fit, exact)
        for name in positive:
            truth &= y_fit[:, name_to_idx[name]]
        for name in negative:
            truth &= ~y_fit[:, name_to_idx[name]]
        fit_truth = int(truth.sum())
        if fit_truth < min_fit_truth:
            continue
        qid = f"q{len(out):04d}"
        out.append(QuerySpec(qid, _query_text(exact, positive, negative, spec_map), exact, positive, negative, fit_truth))
        if len(out) >= n_queries:
            break
    if len(out) < n_queries:
        raise RuntimeError(f"Only generated {len(out)} valid queries; lower min_fit_truth or n_queries")
    return out


__all__ = ["QuerySpec", "apply_exact", "exact_filter_candidates", "generate_query_benchmark"]
