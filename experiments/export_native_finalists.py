"""Export real learned finalist representations/programs for native CPU benchmarking.

Besides the historical flat finalist files, this script writes a portable
Binary1-LS2-int4 semantic sidecar and both forms of the PQ64 semantic program:

* persistent: one 384-D FP32 linear head per concept + one shared PQ codebook;
* active native: one materialized 64x256 FP32 LUT per active concept.

This distinction prevents the native LUT footprint from being mistaken for a
mandatory persistent cost per concept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.binary_bbq_predicates import fit_bbq_like_linear, fit_rsa_random_bits
from experiments.large_scale_search import (
    _metadata_df,
    _retrieval_embeddings,
    _select_dataset,
    _teacher_embeddings_and_scores,
)
from experiments.reviewer_baselines import fit_linear_proxy
from ras.binary import build_centered_binary_code, pack_document_bits
from ras.calibration import fit_scalar_calibrator
from ras.config import load_config
from ras.semantic_index import BinarySemanticIndex
from ras.semantic_program import ProgramStore, compile_linear_program
from ras.splits import make_protocol_split
from ras.teachers import LATENT_SPECS, labels_from_fit_threshold


def _write(path: Path, array, dtype) -> None:
    a = np.ascontiguousarray(np.asarray(array, dtype=dtype))
    a.tofile(path)


def _pack2(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.uint8)
    if q.ndim != 2 or q.shape[1] % 4:
        raise ValueError("RSA2 codes must have shape [n, d] with d divisible by 4")
    z = q.reshape(q.shape[0], q.shape[1] // 4, 4)
    return (
        z[:, :, 0]
        | (z[:, :, 1] << 2)
        | (z[:, :, 2] << 4)
        | (z[:, :, 3] << 6)
    ).astype(np.uint8)


def _compile_pq_native(
    xfit: np.ndarray,
    xtest: np.ndarray,
    coefs: np.ndarray,
    intercepts: np.ndarray,
    cfg,
):
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("Install faiss-cpu before running the native exporter") from e

    m = int(cfg["pq"]["m"])
    nbits = int(cfg["pq"]["nbits"])
    d = int(xfit.shape[1])
    if d % m:
        raise ValueError("embedding dimension must be divisible by pq.m")
    pq = faiss.ProductQuantizer(d, m, nbits)
    pq.train(np.ascontiguousarray(xfit.astype(np.float32)))
    codes = pq.compute_codes(np.ascontiguousarray(xtest.astype(np.float32))).astype(np.uint8)
    ksub = 1 << nbits
    dsub = d // m
    centroids = faiss.vector_to_array(pq.centroids).reshape(m, ksub, dsub).astype(np.float32)
    luts = []
    for w in coefs:
        luts.append(
            np.einsum(
                "mkd,md->mk", centroids, w.reshape(m, dsub), optimize=True
            ).astype(np.float32)
        )
    return (
        codes,
        np.stack(luts),
        np.asarray(intercepts, dtype=np.float32),
        centroids,
    )


def _serialize_rsa2(programs: dict):
    names = [s["name"] for s in LATENT_SPECS]
    unary_idx, unary_tables, pair_idx, pair_tables, intercepts = [], [], [], [], []
    for name in names:
        p = programs[name]
        if len(p["unary_idx"]) != 24 or len(p["pair_idx"]) != 2:
            raise RuntimeError(
                f"Expected 24 unary + 2 pair terms for {name}, "
                f"got {len(p['unary_idx'])}+{len(p['pair_idx'])}"
            )
        unary_idx.append(p["unary_idx"])
        unary_tables.append(p["unary_tables"])
        pair_idx.append(p["pair_idx"])
        pair_tables.append(p["pair_tables"])
        intercepts.append(p["intercept"])
    return (
        np.asarray(unary_idx, dtype=np.uint16),
        np.asarray(unary_tables, dtype=np.float32),
        np.asarray(pair_idx, dtype=np.uint16),
        np.asarray(pair_tables, dtype=np.float32).reshape(len(names), 2, 16),
        np.asarray(intercepts, dtype=np.float32),
    )


def export(config_path: str, out_dir: str) -> Path:
    cfg = load_config(config_path)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, keep = _select_dataset(cfg)
    df = _metadata_df(ds, keep)
    _, x = _retrieval_embeddings(ds, df, cfg)
    teacher_scores = _teacher_embeddings_and_scores(ds, cfg, device)

    seed = int(cfg["benchmark"]["seeds"][0])
    split = make_protocol_split(len(df), seed, strict=True)
    xfit, xcal, xtest = x[split.fit_idx], x[split.cal_idx], x[split.test_idx]
    yfit, ycal, ytest, _ = labels_from_fit_threshold(
        teacher_scores,
        split.fit_idx,
        split.cal_idx,
        split.test_idx,
        prevalence=float(cfg["teacher"].get("positive_prevalence", 0.4)),
    )

    # FP32 supervised semantic heads.
    _, _, coefs, intercepts = fit_linear_proxy(
        xfit, xcal, xtest, yfit, ycal, ytest, seed
    )
    _write(root / "fp32_items.f32", xtest, np.float32)
    _write(root / "fp32_weights.f32", coefs, np.float32)
    _write(root / "fp32_intercepts.f32", intercepts, np.float32)

    # PQ64: export both compact persistent state and the materialized native LUT.
    pq_codes, pq_luts, pq_intercepts, pq_codebook = _compile_pq_native(
        xfit, xtest, coefs, intercepts, cfg
    )
    _write(root / "pq64_codes.u8", pq_codes, np.uint8)
    _write(root / "pq64_luts.f32", pq_luts, np.float32)
    _write(root / "pq64_intercepts.f32", pq_intercepts, np.float32)
    _write(root / "pq64_weights.f32", coefs, np.float32)
    _write(root / "pq64_codebook.f32", pq_codebook, np.float32)

    # BBQ-inspired centered sign bits + per-item LS2 corrections + int4 query.
    bbq = build_centered_binary_code(
        xfit, xcal, xtest, seed=seed, projection_kind="identity", with_corrections=True
    )
    _, _, bbq_export = fit_bbq_like_linear(
        bbq, ycal, ytest, coefs, intercepts, seed, int4_query=True
    )
    if bbq_export is None:
        raise RuntimeError("BBQ int4 export unexpectedly missing")
    qweights = np.asarray(bbq_export["qweights"], dtype=np.uint8)
    wlo = np.asarray(bbq_export["weight_lo"], dtype=np.float32)
    wscale = np.asarray(bbq_export["weight_scale"], dtype=np.float32)
    decoded = wlo[:, None] + wscale[:, None] * qweights.astype(np.float32)
    sum_w = decoded.sum(axis=1).astype(np.float32)
    base = (
        np.asarray(intercepts, dtype=np.float32)
        + decoded @ bbq.centroid.astype(np.float32)
    ).astype(np.float32)
    _write(root / "bbq_bits.u8", pack_document_bits(bbq.Q_test), np.uint8)
    _write(root / "bbq_corrections.f32", bbq.correction_test, np.float32)
    _write(root / "bbq_weight_bitplanes.u8", bbq_export["bitplanes"], np.uint8)
    _write(root / "bbq_weight_lo.f32", wlo, np.float32)
    _write(root / "bbq_weight_scale.f32", wscale, np.float32)
    _write(root / "bbq_base.f32", base, np.float32)
    _write(root / "bbq_sum_w.f32", sum_w, np.float32)

    # Portable BYO sidecar using the same first-split Binary1-LS2-int4 design.
    sidecar_index = BinarySemanticIndex.build(
        root / "sidecar_index",
        xtest,
        fit_embeddings=xfit,
        seed=seed,
        projection_kind="identity",
        overwrite=True,
    )
    store = ProgramStore(root / "sidecar_programs")
    packed_cal = pack_document_bits(bbq.Q_cal)
    for c, spec in enumerate(LATENT_SPECS):
        program = compile_linear_program(
            sidecar_index,
            name=spec["name"],
            weight=np.asarray(coefs[c], dtype=np.float32),
            intercept=float(intercepts[c]),
            positive_rate=float(ycal[:, c].mean()),
        )
        cal_raw = program.raw_scores(packed_cal, bbq.correction_cal)
        cal = fit_scalar_calibrator(cal_raw, ycal[:, c])
        program.calibration_a = float(cal.a)
        program.calibration_b = float(cal.b)
        program.positive_rate = float(ycal[:, c].mean())
        store.save(program)

    # RSA2 real 2-bit quantile codes + learned sparse programs.
    _, _, rsa2_programs, rsa2 = fit_rsa_random_bits(
        xfit, xcal, xtest, yfit, ycal, ytest, seed, cfg, bits=2
    )
    ui, ut, pi, pt, bi = _serialize_rsa2(rsa2_programs)
    _write(root / "rsa2_codes.u8", _pack2(rsa2.Q_test), np.uint8)
    _write(root / "rsa2_unary_idx.u16", ui, np.uint16)
    _write(root / "rsa2_unary_tables.f32", ut, np.float32)
    _write(root / "rsa2_pair_idx.u16", pi, np.uint16)
    _write(root / "rsa2_pair_tables.f32", pt, np.float32)
    _write(root / "rsa2_intercepts.f32", bi, np.float32)

    pq_shared_codebook_bytes = int(np.asarray(pq_codebook, dtype=np.float32).nbytes)
    manifest = {
        "source_items": int(len(xtest)),
        "embedding_dim": int(xtest.shape[1]),
        "concepts": [s["name"] for s in LATENT_SPECS],
        "n_concepts": len(LATENT_SPECS),
        "seed": seed,
        "bytes_per_item": {
            "fp32_linear": 1536,
            "pq64": 64,
            "bbq1_ls2_int4q": 56,
            "rsa2": 96,
        },
        "persistent_program_bytes_per_concept": {
            "fp32_linear": 1548,
            "pq64": 1548,
            "bbq1_ls2_int4q": 216,
            "rsa2": 580,
        },
        "active_program_bytes_per_concept": {
            "fp32_linear": 1548,
            "pq64": 65548,
            "bbq1_ls2_int4q": 216,
            "rsa2": 580,
        },
        "shared_representation_bytes": {"pq64_codebook": pq_shared_codebook_bytes},
        # Backward-compatible key for native-executor consumers. It means ACTIVE
        # state, not mandatory persistent state.
        "program_bytes_per_concept": {
            "fp32_linear": 1548,
            "pq64": 65548,
            "bbq1_ls2_int4q": 216,
            "rsa2": 580,
        },
        "pq64": {
            "persistent_head": "pq64_weights.f32 + pq64_intercepts.f32",
            "shared_codebook": "pq64_codebook.f32",
            "active_lut": "pq64_luts.f32",
            "note": "The 65,548-byte LUT is activation-time state and can be materialized from the compact head plus shared codebook.",
        },
        "portable_sidecar": {
            "index": "sidecar_index",
            "programs": "sidecar_programs",
            "note": "Candidate row IDs from any host ANN/filter layer can be scored without loading FP32 item embeddings.",
        },
        "note": "Real held-out item representations and learned first-seed programs; Rust may tile rows only to enlarge the resident working set.",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("Native assets:", root)
    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/binary_bbq.yaml")
    p.add_argument("--out-dir", default="results/native_finalists_first_seed")
    a = p.parse_args()
    export(a.config, a.out_dir)


if __name__ == "__main__":
    main()
