"""Focused review experiments; synthetic mode validates execution without model downloads."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

from ras.accounting import compiler_footprints, deployment_memory_rows
from ras.binary import pack_document_bits
from ras.calibration import fit_scalar_calibrator
from ras.composition import compose_query
from ras.config import load_config
from ras.evaluation import aggregate
from ras.metrics import best_f1_threshold, metric_row, binary_ranking_stats
from ras.pq import train_pq, compile_pq_head, score_pq
from ras.queries import generate_query_benchmark, apply_exact
from ras.repro import environment_manifest, write_json
from ras.semantic_index import BinarySemanticIndex
from ras.semantic_program import ProgramStore, compile_linear_program
from ras.splits import make_protocol_split
from ras.teacher_specs import LATENT_SPECS

NAMES = [s["name"] for s in LATENT_SPECS]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def data(cfg, synthetic):
    if synthetic:
        rng = np.random.default_rng(700)
        x = rng.normal(size=(1600, 384)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        teacher = x @ rng.normal(size=(384, 8)) + rng.normal(scale=.15, size=(len(x), 8))
        df = pd.DataFrame({
            "id": np.arange(len(x)), "baseColour": "Black", "gender": "Unisex",
            "subCategory": "Shoes", "articleType": "Casual Shoes", "masterCategory": "Footwear",
        })
        return x, teacher, df, None
    # GPU/model dependencies are needed only for the real dataset.
    import torch
    from experiments.large_scale_search import (
        _metadata_df, _retrieval_embeddings, _select_dataset, _teacher_embeddings_and_scores,
    )
    ds, keep = _select_dataset(cfg)
    df = _metadata_df(ds, keep)
    model, x = _retrieval_embeddings(ds, df, cfg)
    teacher = _teacher_embeddings_and_scores(ds, cfg, "cuda" if torch.cuda.is_available() else "cpu")
    return x, teacher, df, model


def calibrate(raw_cal, raw_test, ycal):
    cal = fit_scalar_calibrator(raw_cal, ycal)
    return cal, cal.transform(raw_test).astype(np.float32)


def evaluate(x, df, y, scores, queries, query_vectors, retention, pool_size, seed):
    rows, query_rows = [], []
    mapping = {n: i for i, n in enumerate(NAMES)}
    for qi, query in enumerate(queries):
        dense = x @ query_vectors[qi]
        ann = np.argsort(-dense, kind="stable")[:pool_size]
        ann = ann[apply_exact(df.iloc[ann], query.exact)]
        truth = np.ones(len(ann), dtype=bool)
        for name in query.positive:
            truth &= y[ann, mapping[name]]
        for name in query.negative:
            truth &= ~y[ann, mapping[name]]
        query_rows.append({
            **query.to_dict(), "seed": seed, "pool_size": len(ann),
            "total_true": int(truth.sum()), "empty_candidate_pool": len(ann) == 0,
        })
        if not len(ann):
            continue
        values = {"dense": dense[ann], "oracle": truth.astype(np.float32)}
        for method, logits in scores.items():
            values[method] = compose_query(logits[ann], mapping, query.positive, query.negative)
        for fraction in retention:
            k = max(1, min(len(ann), round(len(ann) * fraction)))
            for method, value in values.items():
                chosen = np.argsort(-value, kind="stable")[:k]
                rows.append({
                    "seed": seed, "query_id": query.query_id, "retention": fraction,
                    "method": method, "pool_size": len(ann),
                    **binary_ranking_stats(truth, chosen),
                })
    return pd.DataFrame(rows), pd.DataFrame(query_rows)


def seed_run(cfg, root, seed, x, teacher, df, model, synthetic, export_assets):
    out = root / f"seed_{seed}"
    if (out / "complete.json").exists():
        print(f"[resume] seed {seed} already complete", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    split = make_protocol_split(len(x), seed, strict=True)
    xf, xc, xt = x[split.fit_idx], x[split.cal_idx], x[split.test_idx]
    threshold = np.quantile(teacher[split.fit_idx], 1 - float(cfg["teacher"]["positive_prevalence"]), axis=0)
    labels = teacher >= threshold
    yf, yc, yt = labels[split.fit_idx], labels[split.cal_idx], labels[split.test_idx]
    print(f"[seed {seed}] fit={len(xf)} calibration={len(xc)} test={len(xt)}", flush=True)
    index = BinarySemanticIndex.build(out / "binary_index", xt, fit_embeddings=xf, overwrite=True)
    cal_bits, cal_corr = index.encode_batch(xc)
    store = ProgramStore(out / "programs")
    # One codebook/codes shared by both PQ head precision variants.
    print(f"[seed {seed}] train shared PQ codebook", flush=True)
    codebook, (qcal, qtest) = train_pq(xf, xc, xt, m=64, seed=seed)
    scores = {n: [] for n in ["linear_fp32", "bbq1_ls2_int4q", "pq64_linear_lut", "pq64_int4_head"]}
    metrics, activations, weights, intercepts, calibration_rows = [], [], [], [], []
    for i, name in enumerate(NAMES):
        head = SGDClassifier(loss="log_loss", class_weight="balanced", alpha=1e-4,
                             max_iter=1500, random_state=seed + i)
        head.fit(xf, yf[:, i].astype(int))
        w = head.coef_[0].astype(np.float32)
        intercept = float(head.intercept_[0])
        weights.append(w); intercepts.append(intercept)
        raw = {"linear_fp32": (head.decision_function(xc), head.decision_function(xt))}
        program = compile_linear_program(index, name=name, weight=w, intercept=intercept)
        raw["bbq1_ls2_int4q"] = (
            program.raw_scores(cal_bits, cal_corr), program.raw_scores(index.bits, index.corrections),
        )
        for method, int4 in [("pq64_linear_lut", False), ("pq64_int4_head", True)]:
            lut, meta = compile_pq_head(codebook, w, intercept, int4=int4)
            activations.append({"seed": seed, "concept": name, "method": method,
                                **{k: v for k, v in meta.items() if not isinstance(v, np.ndarray)}})
            np.save(out / f"{method}_{name}_lut.npy", lut, allow_pickle=False)
            if int4:
                meta["packed_weights"].tofile(out / f"{method}_{name}_weights.u8")
            raw[method] = (score_pq(qcal, lut, intercept), score_pq(qtest, lut, intercept))
        for method, (sc, st) in raw.items():
            cal, calibrated = calibrate(sc, st, yc[:, i])
            scores[method].append(calibrated)
            metrics.append({
                "seed": seed, "concept": name, "method": method,
                **metric_row(yt[:, i], st, best_f1_threshold(sc, yc[:, i])),
            })
            calibration_rows.append({"seed": seed, "concept": name, "method": method,
                                     "a": cal.a, "b": cal.b})
            if method == "bbq1_ls2_int4q":
                program.calibration_a, program.calibration_b = cal.a, cal.b
                program.positive_rate = float(yc[:, i].mean())
                store.save(program)
        print(f"[seed {seed}] compiled {name}", flush=True)
    scores = {name: np.column_stack(cols).astype(np.float32) for name, cols in scores.items()}
    # These are real on-disk score-table baselines, not additional learned models.
    for method in ["linear_fp32", "bbq1_ls2_int4q"]:
        path = out / f"materialized_{method}.f32"
        scores[method].tofile(path)
        stored = np.memmap(path, dtype=np.float32, mode="r", shape=scores[method].shape)
        np.testing.assert_array_equal(stored, scores[method])
        scores["materialized_" + method] = stored
    query_count = 20 if synthetic else int(cfg["benchmark"]["n_queries"])
    queries = generate_query_benchmark(
        df.iloc[split.fit_idx].reset_index(drop=True), yf,
        n_queries=query_count, seed=seed + 404,
        min_fit_truth=10 if synthetic else int(cfg["benchmark"]["min_fit_truth"]),
        max_positive_latents=int(cfg["benchmark"].get("max_positive_latents", 3)),
    )
    if synthetic:
        rng = np.random.default_rng(seed + 991)
        query_vectors = rng.normal(size=(len(queries), x.shape[1])).astype(np.float32)
        query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    else:
        query_vectors = model.encode([q.text for q in queries], convert_to_numpy=True,
                                     normalize_embeddings=True).astype(np.float32)
    per_query, query_meta = evaluate(
        xt, df.iloc[split.test_idx].reset_index(drop=True), yt, scores, queries, query_vectors,
        cfg["benchmark"]["retention"], int(cfg["benchmark"]["ann_pool"]), seed,
    )
    per_query.to_csv(out / "per_query.csv", index=False)
    query_meta.to_json(out / "queries.json", orient="records", indent=2)
    pd.DataFrame(metrics).to_csv(out / "predicate_metrics.csv", index=False)
    pd.DataFrame(activations).to_csv(out / "pq_activation.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(out / "calibrators.csv", index=False)
    np.savez_compressed(
        out / "replay.npz", fit_indices=split.fit_idx, calibration_indices=split.cal_idx,
        test_indices=split.test_idx, teacher_thresholds=threshold, teacher_labels=yt,
        query_vectors=query_vectors, weights=np.stack(weights), intercepts=intercepts,
        pq_codebook=codebook, pq_test_codes=qtest, **scores,
    )
    if export_assets:
        import shutil
        assets = root / "hnsw_assets"
        assets.mkdir(exist_ok=True)
        xt.astype("<f4").tofile(assets / "fp32_items.f32")
        shutil.copytree(out / "binary_index", assets / "sidecar_index", dirs_exist_ok=True)
        shutil.copytree(out / "programs", assets / "sidecar_programs", dirs_exist_ok=True)
        # Both table baselines: exact compiled logits and stronger FP32-head logits.
        scores["bbq1_ls2_int4q"].astype("<f4").tofile(assets / "binary_logits.f32")
        scores["linear_fp32"].astype("<f4").tofile(assets / "linear_logits.f32")
        write_json(assets / "manifest.json", {"seed": seed, "n_items": len(xt), "concepts": NAMES,
                                            "source": "synthetic" if synthetic else "fashion"})
    write_json(out / "complete.json", {"seed": seed, "queries": len(queries)})


def bundle(root):
    # Cache/model weights are excluded. All raw result/replay/native assets are included.
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and "cache" not in p.relative_to(root).parts and p.suffix != ".zip"
                   and p.name != "checksums.json")
    write_json(root / "checksums.json", {str(p.relative_to(root)): digest(p) for p in files})
    archive = root / "ras_review_results.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in files + [root / "checksums.json"]:
            z.write(p, p.relative_to(root))
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/binary_bbq.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--hnsw-queries", type=int, default=1000)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--overfetch-multipliers", default=".75,1,1.5,2,3,4,6,8")
    parser.add_argument("--ef-multipliers", default="1,2")
    parser.add_argument("--skip-hnsw", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    seeds = [7] if args.synthetic else cfg["benchmark"]["seeds"]
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",")]
    env = environment_manifest()
    request = {"config": cfg, "seeds": seeds, "synthetic": args.synthetic,
               "commit": env["git_sha"], "hnsw_queries": args.hnsw_queries,
               "timing_repeats": args.timing_repeats, "overfetch": args.overfetch_multipliers,
               "ef_multipliers": args.ef_multipliers}
    previous = root / "request.json"
    if previous.exists() and json.loads(previous.read_text()) != request:
        raise RuntimeError("Output belongs to another configuration/commit; choose a new output directory.")
    write_json(previous, request)
    write_json(root / "environment.json", env)
    cfg["cache_dir"] = str(root / "cache")
    import faiss
    faiss.omp_set_num_threads(1)
    env["faiss"] = getattr(faiss, "__version__", "unknown")
    write_json(root / "environment.json", env)
    (root / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    if not all((root / f"seed_{s}" / "complete.json").exists() for s in seeds):
        x, teacher, df, model = data(cfg, args.synthetic)
        # Persist exact source identity/order without requiring the images in the result ZIP.
        df.to_json(root / "catalog_metadata.json", orient="records", indent=2)
        x.astype("<f4").tofile(root / "retrieval_embeddings.f32")
        teacher.astype("<f4").tofile(root / "teacher_scores.f32")
        write_json(root / "source_arrays.json", {
            "dtype": "<f4", "retrieval_shape": list(x.shape),
            "teacher_shape": list(teacher.shape), "concepts": NAMES,
        })
        for i, seed in enumerate(seeds):
            seed_run(cfg, root, int(seed), x, teacher, df, model, args.synthetic, i == 0)
    raw = pd.concat([pd.read_csv(root / f"seed_{s}" / "per_query.csv") for s in seeds], ignore_index=True)
    raw.to_csv(root / "per_query.csv", index=False)
    summary, paired = aggregate(raw, cfg, reference="bbq1_ls2_int4q")
    summary.to_csv(root / "summary.csv", index=False)
    paired.to_csv(root / "paired_deltas.csv", index=False)
    footprints = compiler_footprints()
    write_json(root / "compiler_payloads.json", {k: vars(v) for k, v in footprints.items()})
    pd.DataFrame([
        row for n in [15_426, 5_000_000] for count in [8, 32, 128, 512, 2048, 100_000]
        for row in deployment_memory_rows(n, count)
    ]).to_csv(root / "deployment_memory.csv", index=False)
    # Vocabulary sweeps above are storage arithmetic, not new learned concepts.
    write_json(root / "scope.json", {
        "quality": "synthetic validation only" if args.synthetic else "CLIP-defined fashion semantics; conditional candidate-pool recall",
        "vocabulary": "eight learned concepts; larger vocabulary figures are arithmetic only",
        "materialized_quality": "FP32 tables reproduce their source logits exactly",
        "timing": "single-thread resident warm-cache prototype; activation is NumPy timing",
    })
    if not args.skip_hnsw and not (root / "hnsw" / "environment.json").exists():
        cmd = [sys.executable, "-u", "-m", "experiments.semantic_hnsw_reviewer_sweep",
               "--assets-dir", str(root / "hnsw_assets"), "--output-dir", str(root / "hnsw"),
               "--config", args.config, "--queries", str(args.hnsw_queries),
               "--repeats", str(args.timing_repeats), "--ef-multipliers", args.ef_multipliers,
               "--overfetch-multipliers", args.overfetch_multipliers,
               "--dot-evals", "10000" if args.synthetic else "2000000"]
        if args.synthetic:
            cmd += ["--k", "10", "--ef", "32"]
        subprocess.run(cmd, check=True)
    archive = bundle(root)
    print("\nRecall/purity at 20% retention:", flush=True)
    print(summary[summary.retention.eq(.2)].to_string(index=False))
    print(f"\nRESULT_BUNDLE={archive}", flush=True)


if __name__ == "__main__":
    main()
