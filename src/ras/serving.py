"""Reference serving API for semantic candidate pruning.

The host search stack owns ANN retrieval and exact filtering.  It passes row IDs
into this module, which evaluates independently compiled semantic predicates and
returns the highest-scoring survivors.  This makes the low-bit layer a sidecar,
not a replacement search engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np

from .semantic_index import BinarySemanticIndex
from .semantic_program import BinarySemanticProgram, ProgramStore


@dataclass(frozen=True)
class PredicateRef:
    name: str
    positive: bool = True


@dataclass
class QueryResult:
    row_ids: np.ndarray
    scores: np.ndarray
    semantic_ms: float
    topk_ms: float
    input_candidates: int
    predicates: int


class SemanticExecutor:
    """Memory-mapped semantic sidecar executor."""

    def __init__(self, index: BinarySemanticIndex, programs: ProgramStore):
        self.index = index
        self.programs = programs
        self._cache: dict[str, BinarySemanticProgram] = {}

    @classmethod
    def open(cls, index_dir: str, program_dir: str) -> "SemanticExecutor":
        return cls(BinarySemanticIndex.load(index_dir, mmap=True), ProgramStore(program_dir))

    def program(self, name: str) -> BinarySemanticProgram:
        if name not in self._cache:
            p = self.programs.load(name)
            if p.dim != self.index.dim or p.packed_bytes != self.index.manifest.packed_bytes:
                raise ValueError(f"program {name!r} was compiled for a different semantic index")
            self._cache[name] = p
        return self._cache[name]

    def plan(self, positive: Iterable[str] = (), negative: Iterable[str] = ()) -> list[PredicateRef]:
        """Order likely-selective predicates first for native early-exit kernels."""
        refs = [PredicateRef(str(n), True) for n in positive] + [PredicateRef(str(n), False) for n in negative]

        def expected_acceptance(ref: PredicateRef) -> float:
            p = self.program(ref.name).positive_rate
            return float(p if ref.positive else 1.0 - p)

        return sorted(refs, key=expected_acceptance)

    def score_candidates(
        self,
        candidate_ids: np.ndarray,
        *,
        positive: Iterable[str] = (),
        negative: Iterable[str] = (),
    ) -> np.ndarray:
        ids = np.asarray(candidate_ids, dtype=np.int64)
        positive = tuple(str(x) for x in positive)
        negative = tuple(str(x) for x in negative)
        if ids.ndim != 1:
            raise ValueError("candidate_ids must be a 1D array of index row IDs")
        if len(ids) == 0:
            return np.empty(0, dtype=np.float32)
        if ids.min() < 0 or ids.max() >= self.index.n_items:
            raise IndexError("candidate row ID outside semantic index")
        refs = self.plan(positive, negative)
        if not refs:
            return np.zeros(len(ids), dtype=np.float32)

        bits = self.index.bits[ids]
        corrections = self.index.corrections[ids]
        total = np.zeros(len(ids), dtype=np.float64)
        for ref in refs:
            logits = self.program(ref.name).calibrated_logits(bits, corrections).astype(np.float64)
            # log sigma(z) for positive predicates; log sigma(-z) for negation.
            z = logits if ref.positive else -logits
            total += -np.logaddexp(0.0, -z)
        return total.astype(np.float32)

    def topk(
        self,
        candidate_ids: np.ndarray,
        *,
        positive: Iterable[str] = (),
        negative: Iterable[str] = (),
        k: int = 100,
    ) -> QueryResult:
        ids = np.asarray(candidate_ids, dtype=np.int64)
        positive = tuple(str(x) for x in positive)
        negative = tuple(str(x) for x in negative)
        t0 = time.perf_counter()
        scores = self.score_candidates(ids, positive=positive, negative=negative)
        t1 = time.perf_counter()
        k = max(0, min(int(k), len(ids)))
        if k == 0:
            out_ids = np.empty(0, dtype=np.int64)
            out_scores = np.empty(0, dtype=np.float32)
        elif k == len(ids):
            order = np.argsort(-scores, kind="stable")
            out_ids, out_scores = ids[order], scores[order]
        else:
            part = np.argpartition(scores, -k)[-k:]
            order = part[np.argsort(-scores[part], kind="stable")]
            out_ids, out_scores = ids[order], scores[order]
        t2 = time.perf_counter()
        return QueryResult(
            row_ids=out_ids,
            scores=out_scores,
            semantic_ms=(t1 - t0) * 1000.0,
            topk_ms=(t2 - t1) * 1000.0,
            input_candidates=int(len(ids)),
            predicates=int(len(positive) + len(negative)),
        )


__all__ = ["PredicateRef", "QueryResult", "SemanticExecutor"]
