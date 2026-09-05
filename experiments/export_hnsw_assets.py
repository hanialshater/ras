"""Export only the assets needed by the live semantic-HNSW benchmarks.

This avoids the PQ/RSA/native-finalist work in export_native_finalists.py and
prints explicit stage markers so Colab users can see progress.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

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


def _stage(name: str, t0: float | None = None) -> float:
    now = time.time()
    if t0 is None:
        print(f"[stage] {name}", flush=True)
    else:
        print(f"[stage] {name} done in {now - t0:.1f}s", flush=True)
    return now


def _write(path: Path, array, dtype) -> None:
    np.ascontiguousarray(np.asarray(array, dtype=dtype)).tofile(path)


def export(config_path: str, out_dir: str) -> Path:
    cfg = load_config(config_path)
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[stage] device={device}", flush=True)

    t = _stage("load fashion dataset")
    ds, keep = _select_dataset(cfg)
    df = _metadata_df(ds, keep)
    _stage(f"dataset ready: {len(df):,} products", t)

    t = _stage("encode MiniLM retrieval embeddings")
    _, x = _retrieval_embeddings(ds, df, cfg)
    _stage("MiniLM embeddings", t)

    t = _stage("compute/load CLIP teacher scores")
    teacher_scores = _teacher_embeddings_and_scores(ds, cfg, device)
    _stage("CLIP teacher scores", t)

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

    t = _stage("fit FP32 semantic heads")
    _, _, coefs, intercepts = fit_linear_proxy(
        xfit, xcal, xtest, yfit, ycal, ytest, seed
    )
    _stage("FP32 semantic heads", t)

    _write(root / "fp32_items.f32", xtest, np.float32)

    t = _stage("build centered Binary1-LS2 code")
    bbq = build_centered_binary_code(
        xfit, xcal, xtest, seed=seed, projection_kind="identity", with_corrections=True
    )
    _stage("Binary1-LS2 code", t)

    t = _stage("write portable sidecar index")
    sidecar_index = BinarySemanticIndex.build(
        root / "sidecar_index",
        xtest,
        fit_embeddings=xfit,
        seed=seed,
        projection_kind="identity",
        overwrite=True,
    )
    _stage("sidecar index", t)

    t = _stage("compile and calibrate 8 semantic programs")
    store = ProgramStore(root / "sidecar_programs")
    packed_cal = pack_document_bits(bbq.Q_cal)
    for c, spec in enumerate(LATENT_SPECS):
        print(f"  [program {c+1}/{len(LATENT_SPECS)}] {spec['name']}", flush=True)
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
    _stage("semantic programs", t)

    manifest = {
        "source_items": int(len(xtest)),
        "embedding_dim": int(xtest.shape[1]),
        "concepts": [s["name"] for s in LATENT_SPECS],
        "seed": seed,
        "bytes_per_item": 56,
        "program_bytes_per_concept": 216,
        "purpose": "focused assets for live semantic-HNSW reviewer benchmark",
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"[done] HNSW assets: {root}", flush=True)
    return root


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/binary_bbq.yaml")
    p.add_argument("--out-dir", default="results/hnsw_assets")
    a = p.parse_args()
    export(a.config, a.out_dir)


if __name__ == "__main__":
    main()
