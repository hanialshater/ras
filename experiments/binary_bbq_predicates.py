"""Binary / BBQ-inspired semantic predicate experiment.

This experiment is the kill test for the low-bit semantic-substrate thesis.  It
compares, under the same independent CLIP-teacher protocol:

* FP32 MiniLM linear semantic heads;
* PQ64 + compiled linear LUTs;
* a centered 1-bit document representation with two per-item reconstruction
  corrections, evaluated with FP32 or int4 predicate weights (BBQ-inspired,
  not an exact Lucene BBQ implementation);
* sparse learned predicates on centered 1-bit identity and random projections;
* sparse learned predicates on random 2-bit and 4-bit substrates;
* zero-shot MiniLM prompt vectors, dense ordering, and an oracle.

All supervised methods use strict fit/calibration/test splits.  Compound query
quality is measured inside the exact-filtered ANN pool, matching the paper.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.large_scale_search import (
    _metadata_df,
    _retrieval_embeddings,
    _select_dataset,
    _teacher_embeddings_and_scores,
)
from experiments.reviewer_baselines import (
    aggregate,
    eval_queries,
    fit_linear_proxy,
    fit_pq64_linear,
    zero_shot_scores,
)
from ras.binary import (
    build_centered_binary_code,
    int4_weight_bitplanes,
    pack_document_bits,
    quantize_weight_int4,
    score_compiled_linear,
)
from ras.calibration import fit_scalar_calibrator
from ras.config import config_hash, load_config
from ras.metrics import best_f1_threshold, metric_row
from ras.predicates import add_pair_interactions, fit_boosted_lut, score_boosted
from ras.queries import generate_query_benchmark
from ras.repro import environment_manifest, write_json
from ras.splits import make_protocol_split
from ras.substrate import build_substrate
from ras.teachers import LATENT_SPECS, labels_from_fit_threshold


def _calibrate(cal_scores: np.ndarray, test_scores: np.ndarray, ycal: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [fit_scalar_calibrator(cal_scores[:, c], ycal[:, c]).transform(test_scores[:, c]) for c in range(ycal.shape[1])]
    )


def _rsa_program_bytes(n_bins: int, cfg) -> int:
    """Approximate serialized program bytes for one concept.

    Includes f32 LUT values, u16 unary indices, u16 pair indices, an intercept,
    and two scalar calibration parameters.  It intentionally excludes generic
    container/allocator overhead.
    """
    k = int(cfg["rsa"]["k_coords"])
    m = int(cfg["rsa"].get("pair_luts", 0))
    values = k * n_bins + m * n_bins * n_bins + 3
    indices = 2 * k + 4 * m
    return int(4 * values + indices)


def _fit_sparse_codes(Qfit, Qcal, Qtest, yfit, ycal, ytest, seed, cfg, method, n_bins):
    cal_cols, test_cols, rows, programs = [], [], [], {}
    rc = cfg["rsa"]
    for c, spec in enumerate(LATENT_SPECS):
        unary = fit_boosted_lut(
            Qfit,
            yfit[:, c],
            k=int(rc["k_coords"]),
            n_bins=int(n_bins),
            candidate_pool=int(rc["candidate_pool"]),
            refine_passes=1,
        )
        model = add_pair_interactions(
            Qfit,
            yfit[:, c],
            unary,
            n_pairs=int(rc.get("pair_luts", 0)),
            pair_pool=12,
        )
        sc = score_boosted(Qcal, model)
        st = score_boosted(Qtest, model)
        row = metric_row(ytest[:, c], st, best_f1_threshold(sc, ycal[:, c]))
        row.update({"concept": spec["name"], "method": method, "seed": int(seed)})
        rows.append(row)
        cal_cols.append(sc)
        test_cols.append(st)
        programs[spec["name"]] = {
            "unary_idx": [int(j) for j in model.unary_idx],
            "unary_tables": [np.asarray(t, dtype=float).tolist() for t in model.unary_tables],
            "pair_idx": [[int(a), int(b)] for a, b in model.pair_idx],
            "pair_tables": [np.asarray(t, dtype=float).tolist() for t in model.pair_tables],
            "intercept": float(model.intercept),
        }
    cal_scores = np.column_stack(cal_cols)
    test_scores = np.column_stack(test_cols)
    return _calibrate(cal_scores, test_scores, ycal), pd.DataFrame(rows), programs


def fit_rsa1_centered(xfit, xcal, xtest, yfit, ycal, ytest, seed, cfg, projection_kind):
    code = build_centered_binary_code(
        xfit,
        xcal,
        xtest,
        seed=int(seed),
        projection_kind=projection_kind,
        with_corrections=False,
    )
    method = f"rsa1_centered_{'random' if projection_kind == 'orthogonal' else 'identity'}"
    scores, rows, programs = _fit_sparse_codes(
        code.Q_fit, code.Q_cal, code.Q_test, yfit, ycal, ytest, seed, cfg, method, 2
    )
    return scores, rows, programs, code


def fit_rsa_random_bits(xfit, xcal, xtest, yfit, ycal, ytest, seed, cfg, bits):
    rc = cfg["rsa"]
    substrate = build_substrate(
        xfit,
        xcal,
        xtest,
        name=f"random_{bits}bit_{rc.get('quantizer', 'quantile')}",
        seed=int(seed),
        bits=int(bits),
        projection_kind="orthogonal",
        quantizer_kind=str(rc.get("quantizer", "quantile")),
    )
    method = f"rsa{bits}_random"
    scores, rows, programs = _fit_sparse_codes(
        substrate.Q_fit,
        substrate.Q_cal,
        substrate.Q_test,
        yfit,
        ycal,
        ytest,
        seed,
        cfg,
        method,
        substrate.n_bins,
    )
    return scores, rows, programs, substrate


def fit_bbq_like_linear(code, ycal, ytest, coefs, intercepts, seed, *, int4_query):
    method = "bbq1_ls2_int4q" if int4_query else "bbq1_ls2_f32q"
    cal_cols, test_cols, rows = [], [], []
    qweights, qlos, qscales = [], [], []
    for c, spec in enumerate(LATENT_SPECS):
        sc = score_compiled_linear(
            code.Q_cal,
            code.correction_cal,
            code,
            coefs[c],
            intercepts[c],
            int4_query=bool(int4_query),
        )
        st = score_compiled_linear(
            code.Q_test,
            code.correction_test,
            code,
            coefs[c],
            intercepts[c],
            int4_query=bool(int4_query),
        )
        row = metric_row(ytest[:, c], st, best_f1_threshold(sc, ycal[:, c]))
        row.update({"concept": spec["name"], "method": method, "seed": int(seed)})
        rows.append(row)
        cal_cols.append(sc)
        test_cols.append(st)
        if int4_query:
            wz = (code.projection.T @ coefs[c]).astype(np.float32)
            q, _, lo, scale = quantize_weight_int4(wz)
            qweights.append(q)
            qlos.append(lo)
            qscales.append(scale)
    cal_scores = np.column_stack(cal_cols)
    test_scores = np.column_stack(test_cols)
    export = None
    if int4_query:
        qweights = np.stack(qweights)
        export = {
            "qweights": qweights,
            "weight_lo": np.asarray(qlos, dtype=np.float32),
            "weight_scale": np.asarray(qscales, dtype=np.float32),
            "bitplanes": np.stack([int4_weight_bitplanes(q) for q in qweights]),
        }
    return _calibrate(cal_scores, test_scores, ycal), pd.DataFrame(rows), export


def _method_meta(cfg):
    d = 384
    k = int(cfg["rsa"]["k_coords"])
    pairs = int(cfg["rsa"].get("pair_luts", 0))
    pq_m = int(cfg["pq"]["m"])
    pq_bits = int(cfg["pq"]["nbits"])
    pq_program = pq_m * (1 << pq_bits) * 4 + 12
    return pd.DataFrame(
        [
            {"method": "dense", "bytes_per_item": 1536, "program_bytes_per_concept": 0, "notes": "MiniLM cosine ordering"},
            {"method": "zero_shot_name", "bytes_per_item": 1536, "program_bytes_per_concept": 1536, "notes": "training-free concept vector"},
            {"method": "zero_shot_prompt_diff", "bytes_per_item": 1536, "program_bytes_per_concept": 1536, "notes": "training-free positive-minus-negative prompt vector"},
            {"method": "linear_fp32", "bytes_per_item": 1536, "program_bytes_per_concept": d * 4 + 12, "notes": "supervised FP32 linear head"},
            {"method": "pq64_linear_lut", "bytes_per_item": pq_m * pq_bits / 8, "program_bytes_per_concept": pq_program, "notes": "FP32 linear head compiled into all PQ subspaces"},
            {"method": "bbq1_ls2_f32q", "bytes_per_item": d / 8 + 8, "program_bytes_per_concept": d * 4 + 20, "notes": "BBQ-inspired centered 1-bit docs + two LS2 corrections; FP32 predicate weight; not Lucene BBQ"},
            {"method": "bbq1_ls2_int4q", "bytes_per_item": d / 8 + 8, "program_bytes_per_concept": d / 2 + 24, "notes": "BBQ-inspired centered 1-bit docs + two LS2 corrections; int4 predicate weight; not Lucene BBQ"},
            {"method": "rsa1_centered_identity", "bytes_per_item": d / 8, "program_bytes_per_concept": _rsa_program_bytes(2, cfg), "notes": f"centered 1-bit sparse learned program, {k} unary + {pairs} pair terms"},
            {"method": "rsa1_centered_random", "bytes_per_item": d / 8, "program_bytes_per_concept": _rsa_program_bytes(2, cfg), "notes": f"centered random-orthogonal 1-bit sparse learned program, {k} unary + {pairs} pairs"},
            {"method": "rsa2_random", "bytes_per_item": d * 2 / 8, "program_bytes_per_concept": _rsa_program_bytes(4, cfg), "notes": "random-orthogonal 2-bit quantile sparse program"},
            {"method": "rsa4_random", "bytes_per_item": d * 4 / 8, "program_bytes_per_concept": _rsa_program_bytes(16, cfg), "notes": "current random-orthogonal 4-bit quantile sparse program"},
            {"method": "oracle", "bytes_per_item": np.nan, "program_bytes_per_concept": np.nan, "notes": "teacher-truth upper bound inside ANN pool"},
        ]
    )


def _plot(summary, meta, root: Path):
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    order = [
        "dense",
        "zero_shot_prompt_diff",
        "rsa1_centered_identity",
        "rsa1_centered_random",
        "bbq1_ls2_int4q",
        "rsa2_random",
        "rsa4_random",
        "pq64_linear_lut",
        "linear_fp32",
        "oracle",
    ]
    for metric in ["recall", "purity"]:
        plt.figure(figsize=(9, 6))
        for method in order:
            g = summary[(summary.method == method) & (summary.metric == metric)].sort_values("retention")
            if len(g):
                plt.plot(g.retention, g["mean"], marker="o", label=method)
        plt.xscale("log")
        plt.xlabel("Fraction of exact-filtered ANN pool kept")
        plt.ylabel(metric.title())
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out / f"binary_{metric}.png", dpi=180)
        plt.close()

    r = summary[(summary.metric == "recall") & (summary.retention == 0.2)][["method", "mean"]].rename(columns={"mean": "recall"})
    p = summary[(summary.metric == "purity") & (summary.retention == 0.2)][["method", "mean"]].rename(columns={"mean": "purity"})
    frontier = meta.merge(r, on="method", how="left").merge(p, on="method", how="left")
    frontier.to_csv(root / "pareto_at_20pct.csv", index=False)

    f = frontier.dropna(subset=["recall", "bytes_per_item"])
    plt.figure(figsize=(8, 6))
    for row in f.itertuples():
        plt.scatter(row.bytes_per_item, row.recall)
        plt.annotate(row.method, (row.bytes_per_item, row.recall), fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("Semantic execution bytes / item")
    plt.ylabel("Recall at 20% retention")
    plt.tight_layout()
    plt.savefig(out / "quality_vs_item_bytes.png", dpi=180)
    plt.close()


def run(config_path: str, output_dir: str | None = None):
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_binary_" + config_hash(cfg)
    root = Path(output_dir or cfg["results_dir"]) / run_id
    root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, root / "config.yaml")
    write_json(root / "environment.json", environment_manifest())

    ds, keep = _select_dataset(cfg)
    df = _metadata_df(ds, keep)
    retrieval_model, x = _retrieval_embeddings(ds, df, cfg)
    teacher_scores = _teacher_embeddings_and_scores(ds, cfg, device)

    all_q, all_pred, all_queries = [], [], []
    programs_all = {}
    native_export_done = False

    for seed in cfg["benchmark"]["seeds"]:
        split = make_protocol_split(len(df), int(seed), strict=True)
        xfit, xcal, xtest = x[split.fit_idx], x[split.cal_idx], x[split.test_idx]
        df_fit = df.iloc[split.fit_idx].reset_index(drop=True)
        df_test = df.iloc[split.test_idx].reset_index(drop=True)
        yfit, ycal, ytest, _ = labels_from_fit_threshold(
            teacher_scores,
            split.fit_idx,
            split.cal_idx,
            split.test_idx,
            prevalence=float(cfg["teacher"].get("positive_prevalence", 0.4)),
        )
        method_scores = {}

        linear, rows, coefs, intercepts = fit_linear_proxy(xfit, xcal, xtest, yfit, ycal, ytest, int(seed))
        method_scores["linear_fp32"] = linear
        all_pred.append(rows)

        pq, rows, _ = fit_pq64_linear(xfit, xcal, xtest, yfit, ycal, ytest, int(seed), coefs, intercepts, cfg)
        method_scores["pq64_linear_lut"] = pq
        all_pred.append(rows)

        zs, rows = zero_shot_scores(retrieval_model, xcal, xtest, ycal, ytest, int(seed))
        method_scores.update(zs)
        all_pred.append(rows)

        # Centered 1-bit sparse programs.  Corrections are deliberately disabled:
        # RSA1 must stand on the binary substrate itself.
        rsa1_id, rows, programs, code1_id = fit_rsa1_centered(
            xfit, xcal, xtest, yfit, ycal, ytest, int(seed), cfg, "identity"
        )
        method_scores["rsa1_centered_identity"] = rsa1_id
        all_pred.append(rows)
        programs_all[f"{seed}_rsa1_centered_identity"] = programs

        rsa1_r, rows, programs, code1_r = fit_rsa1_centered(
            xfit, xcal, xtest, yfit, ycal, ytest, int(seed), cfg, "orthogonal"
        )
        method_scores["rsa1_centered_random"] = rsa1_r
        all_pred.append(rows)
        programs_all[f"{seed}_rsa1_centered_random"] = programs

        for bits in [2, 4]:
            s, rows, programs, _ = fit_rsa_random_bits(xfit, xcal, xtest, yfit, ycal, ytest, int(seed), cfg, bits)
            method_scores[f"rsa{bits}_random"] = s
            all_pred.append(rows)
            programs_all[f"{seed}_rsa{bits}_random"] = programs

        # BBQ-inspired controlled baseline: identity-projected, globally centered
        # documents, 1 sign bit / dimension, and two per-item two-level correction
        # values.  We test both FP32 and int4 predicate weights.
        bbq_code = build_centered_binary_code(
            xfit, xcal, xtest, seed=int(seed), projection_kind="identity", with_corrections=True
        )
        bbq_f32, rows, _ = fit_bbq_like_linear(
            bbq_code, ycal, ytest, coefs, intercepts, int(seed), int4_query=False
        )
        method_scores["bbq1_ls2_f32q"] = bbq_f32
        all_pred.append(rows)
        bbq_i4, rows, bbq_export = fit_bbq_like_linear(
            bbq_code, ycal, ytest, coefs, intercepts, int(seed), int4_query=True
        )
        method_scores["bbq1_ls2_int4q"] = bbq_i4
        all_pred.append(rows)

        if not native_export_done and bbq_export is not None:
            np.savez_compressed(
                root / "native_export_first_seed.npz",
                bbq_test_bits=pack_document_bits(bbq_code.Q_test),
                bbq_test_corrections=bbq_code.correction_test,
                bbq_centroid=bbq_code.centroid,
                bbq_int4_weights=bbq_export["qweights"],
                bbq_weight_lo=bbq_export["weight_lo"],
                bbq_weight_scale=bbq_export["weight_scale"],
                bbq_weight_bitplanes=bbq_export["bitplanes"],
                rsa1_random_test_bits=pack_document_bits(code1_r.Q_test),
                test_indices=split.test_idx,
            )
            native_export_done = True

        queries = generate_query_benchmark(
            df_fit,
            yfit,
            n_queries=int(cfg["benchmark"]["n_queries"]),
            seed=int(seed) + 404,
            min_fit_truth=int(cfg["benchmark"].get("min_fit_truth", 100)),
            max_positive_latents=int(cfg["benchmark"].get("max_positive_latents", 3)),
            allow_negative=bool(cfg["benchmark"].get("allow_negative", True)),
        )
        all_queries.extend([{**q.to_dict(), "seed": int(seed)} for q in queries])
        all_q.append(eval_queries(retrieval_model, xtest, df_test, ytest, method_scores, queries, cfg, int(seed)))

    perq = pd.concat(all_q, ignore_index=True)
    pred = pd.concat(all_pred, ignore_index=True)
    summary, deltas = aggregate(perq, cfg, reference="rsa4_random")
    meta = _method_meta(cfg)

    perq.to_csv(root / "per_query.csv", index=False)
    pred.to_csv(root / "predicate_metrics.csv", index=False)
    summary.to_csv(root / "summary.csv", index=False)
    deltas.to_csv(root / "paired_deltas.csv", index=False)
    meta.to_csv(root / "method_meta.csv", index=False)
    pd.DataFrame(all_queries).to_csv(root / "queries.csv", index=False)
    write_json(root / "binary_programs.json", programs_all)
    _plot(summary, meta, root)

    pred_summary = pred.groupby("method")[["f1", "ap"]].mean().reset_index()
    pred_summary.to_csv(root / "predicate_summary.csv", index=False)
    headline = {
        "run_id": run_id,
        "n_products": len(df),
        "seeds": cfg["benchmark"]["seeds"],
        "queries_per_seed": int(cfg["benchmark"]["n_queries"]),
        "predicate_mean_f1": pred.groupby("method").f1.mean().to_dict(),
        "predicate_mean_ap": pred.groupby("method").ap.mean().to_dict(),
    }
    for frac in [0.4, 0.2, 0.1]:
        g = summary[(summary.retention == frac) & (summary.metric == "recall")]
        headline[f"recall_at_{frac}"] = {r.method: float(r.mean) for r in g.itertuples()}
    write_json(root / "headline.json", headline)
    print(json.dumps(headline, indent=2))
    print("Results:", root)
    return root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/binary_bbq.yaml")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
