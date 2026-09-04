"""Compile and persist tiny semantic predicates for a binary semantic index.

A Binary1-LS2-int4 program stores four packed bit planes (the 384 int4 weights)
and six f32 scoring/calibration scalars.  For D=384 this is 192 + 24 = 216
scoring-payload bytes per predicate, excluding planner/filesystem metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from .binary import int4_weight_bitplanes, quantize_weight_int4
from .calibration import fit_scalar_calibrator
from .semantic_index import BinarySemanticIndex


PROGRAM_FORMAT_VERSION = 1
_POPCOUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


@dataclass
class BinarySemanticProgram:
    name: str
    dim: int
    packed_bytes: int
    bitplanes: np.ndarray
    weight_lo: float
    weight_scale: float
    base: float
    sum_w: float
    calibration_a: float = 1.0
    calibration_b: float = 0.0
    positive_rate: float = 0.5

    @property
    def program_bytes_theoretical(self) -> int:
        return int(self.bitplanes.size + 6 * 4)

    def raw_scores(self, bits: np.ndarray, corrections: np.ndarray) -> np.ndarray:
        """Reference NumPy implementation of the packed popcount kernel."""
        b = np.asarray(bits, dtype=np.uint8)
        c = np.asarray(corrections, dtype=np.float32)
        if b.ndim != 2 or b.shape[1] != self.packed_bytes:
            raise ValueError("packed item bits disagree with program dimension")
        if c.shape != (len(b), 2):
            raise ValueError("corrections must have shape [n, 2]")

        pos_count = _POPCOUNT[b].sum(axis=1, dtype=np.int32)
        counts = []
        for plane in self.bitplanes:
            counts.append(_POPCOUNT[np.bitwise_and(b, plane[None, :])].sum(axis=1, dtype=np.int32))
        weighted_q = counts[0] + 2 * counts[1] + 4 * counts[2] + 8 * counts[3]
        pos_w = self.weight_lo * pos_count.astype(np.float32) + self.weight_scale * weighted_q.astype(np.float32)
        lo = c[:, 0]
        hi = c[:, 1]
        return (self.base + lo * (self.sum_w - pos_w) + hi * pos_w).astype(np.float32)

    def calibrated_logits(self, bits: np.ndarray, corrections: np.ndarray) -> np.ndarray:
        s = self.raw_scores(bits, corrections)
        return (self.calibration_a * s + self.calibration_b).astype(np.float32)


def compile_linear_program(
    index: BinarySemanticIndex,
    *,
    name: str,
    weight: np.ndarray,
    intercept: float,
    calibration_a: float = 1.0,
    calibration_b: float = 0.0,
    positive_rate: float = 0.5,
) -> BinarySemanticProgram:
    """Compile an arbitrary linear semantic head into the index's int4 program."""
    w = np.asarray(weight, dtype=np.float32)
    if w.shape != (index.dim,):
        raise ValueError(f"weight must have shape ({index.dim},)")
    wz = (index.encoder.projection.T @ w).astype(np.float32)
    qweight, decoded, lo, scale = quantize_weight_int4(wz)
    planes = int4_weight_bitplanes(qweight)
    base = float(intercept) + float(index.encoder.centroid @ decoded)
    return BinarySemanticProgram(
        name=str(name),
        dim=index.dim,
        packed_bytes=index.manifest.packed_bytes,
        bitplanes=np.ascontiguousarray(planes, dtype=np.uint8),
        weight_lo=float(lo),
        weight_scale=float(scale),
        base=float(base),
        sum_w=float(decoded.sum()),
        calibration_a=float(calibration_a),
        calibration_b=float(calibration_b),
        positive_rate=float(positive_rate),
    )


def fit_binary_predicate(
    index: BinarySemanticIndex,
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    fit_indices: np.ndarray | None = None,
    calibration_indices: np.ndarray | None = None,
    seed: int = 7,
    alpha: float = 1e-4,
) -> BinarySemanticProgram:
    """Fit a supervised linear predicate and compile it to a tiny int4 program.

    The full-precision embedding is required only offline while fitting the
    semantic head.  Online inference uses only the binary sidecar index.
    Calibration is fitted on *compiled binary scores*, matching the paper's
    Binary1-LS2-int4 protocol.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    if x.shape != (index.n_items, index.dim):
        raise ValueError("embeddings must align exactly with the semantic index")
    if y.shape != (index.n_items,):
        raise ValueError("labels must contain one value per indexed item")
    if np.unique(y).size < 2:
        raise ValueError("predicate labels must contain both classes")

    all_idx = np.arange(index.n_items, dtype=np.int64)
    if fit_indices is None and calibration_indices is None:
        fit_indices, calibration_indices = train_test_split(
            all_idx,
            test_size=0.2,
            random_state=int(seed),
            stratify=y,
        )
    elif fit_indices is None or calibration_indices is None:
        raise ValueError("provide both fit_indices and calibration_indices, or neither")
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    calibration_indices = np.asarray(calibration_indices, dtype=np.int64)
    if np.intersect1d(fit_indices, calibration_indices).size:
        raise ValueError("fit and calibration indices must be disjoint")

    clf = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        alpha=float(alpha),
        max_iter=1500,
        random_state=int(seed),
        tol=1e-4,
    )
    clf.fit(x[fit_indices], y[fit_indices])
    program = compile_linear_program(
        index,
        name=name,
        weight=np.asarray(clf.coef_[0], dtype=np.float32),
        intercept=float(clf.intercept_[0]),
        positive_rate=float(y[fit_indices].mean()),
    )

    cal_raw = program.raw_scores(index.bits[calibration_indices], index.corrections[calibration_indices])
    calibrator = fit_scalar_calibrator(cal_raw, y[calibration_indices])
    program.calibration_a = float(calibrator.a)
    program.calibration_b = float(calibrator.b)
    program.positive_rate = float(y[calibration_indices].mean())
    return program


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("program names may contain only letters, digits, '_', '-', and '.'")
    return name


class ProgramStore:
    """Filesystem-backed collection of independently deployable predicates."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def save(self, program: BinarySemanticProgram, *, overwrite: bool = True) -> Path:
        name = _safe_name(program.name)
        root = self.path / name
        if root.exists() and not overwrite:
            raise FileExistsError(root)
        root.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(program.bitplanes, dtype=np.uint8).tofile(root / "bitplanes.u8")
        # Native executor reads these seven f32 values directly.  The first six
        # are the 216-byte scoring payload accounting; positive_rate is planner
        # metadata used only to order likely-selective predicates first.
        np.asarray(
            [
                program.weight_lo,
                program.weight_scale,
                program.base,
                program.sum_w,
                program.calibration_a,
                program.calibration_b,
                program.positive_rate,
            ],
            dtype=np.float32,
        ).tofile(root / "scalars.f32")
        meta = {
            "version": PROGRAM_FORMAT_VERSION,
            "name": program.name,
            "dim": int(program.dim),
            "packed_bytes": int(program.packed_bytes),
            "weight_lo": float(program.weight_lo),
            "weight_scale": float(program.weight_scale),
            "base": float(program.base),
            "sum_w": float(program.sum_w),
            "calibration_a": float(program.calibration_a),
            "calibration_b": float(program.calibration_b),
            "positive_rate": float(program.positive_rate),
            "program_bytes_theoretical": int(program.program_bytes_theoretical),
        }
        (root / "manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return root

    def load(self, name: str) -> BinarySemanticProgram:
        root = self.path / _safe_name(name)
        meta = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if int(meta["version"]) != PROGRAM_FORMAT_VERSION:
            raise ValueError(f"unsupported program version {meta['version']}")
        packed = int(meta["packed_bytes"])
        planes = np.fromfile(root / "bitplanes.u8", dtype=np.uint8).reshape(4, packed)
        return BinarySemanticProgram(
            name=str(meta["name"]),
            dim=int(meta["dim"]),
            packed_bytes=packed,
            bitplanes=planes,
            weight_lo=float(meta["weight_lo"]),
            weight_scale=float(meta["weight_scale"]),
            base=float(meta["base"]),
            sum_w=float(meta["sum_w"]),
            calibration_a=float(meta["calibration_a"]),
            calibration_b=float(meta["calibration_b"]),
            positive_rate=float(meta.get("positive_rate", 0.5)),
        )

    def names(self) -> list[str]:
        return sorted(p.name for p in self.path.iterdir() if p.is_dir() and (p / "manifest.json").exists())


__all__ = [
    "BinarySemanticProgram",
    "ProgramStore",
    "compile_linear_program",
    "fit_binary_predicate",
    "PROGRAM_FORMAT_VERSION",
]
