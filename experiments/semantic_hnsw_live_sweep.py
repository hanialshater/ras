"""Reproduce the fair live semantic-HNSW selectivity sweep.

This script is the non-notebook entry point for the systems result reported in
``paper/icml/data/semantic_hnsw_live_fair_full.csv``. It exports real held-out
assets, builds the Rust executable, derives semantic gates for requested
selectivities, runs the same-dot HNSW baselines and live compiled predicates,
and writes raw/summary/fairness CSVs plus an environment manifest.

The Rust executable is expected to use normalized-dot geometry for both the
library HNSW baseline and the extracted-graph traversal. The harness verifies
that the exported MiniLM vectors are unit-normalized before running.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from experiments.export_native_finalists import export as export_native
from ras import SemanticExecutor


DEFAULT_FRACTIONS = (0.50, 0.20, 0.10, 0.05, 0.02)


def _csv_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _fractions(value: str) -> list[float]:
    out = [float(x) for x in _csv_list(value)]
    if not out or any(not (0.0 < x <= 1.0) for x in out):
        raise ValueError("fractions must be comma-separated values in (0, 1]")
    return out


def _cpu_model() -> str:
    p = Path("/proc/cpuinfo")
    if p.exists():
        for line in p.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _rust_version() -> str:
    try:
        return subprocess.check_output(["rustc", "--version"], text=True).strip()
    except Exception:
        return "unknown"


def _build_rust(repo_root: Path) -> Path:
    manifest = repo_root / "rust" / "semantic_engine" / "Cargo.toml"
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(manifest),
            "--bin",
            "semantic_hnsw_live",
        ],
        check=True,
    )
    binary = repo_root / "rust" / "semantic_engine" / "target" / "release" / "semantic_hnsw_live"
    if not binary.exists():
        raise FileNotFoundError(binary)
    return binary


def _verify_normalized(items_path: Path, dim: int = 384, tol: float = 2e-3) -> dict[str, float]:
    items = np.fromfile(items_path, dtype=np.float32)
    if items.size % dim:
        raise ValueError(f"{items_path} is not divisible by dim={dim}")
    x = items.reshape(-1, dim)
    norms = np.linalg.norm(x, axis=1)
    max_error = float(np.max(np.abs(norms - 1.0)))
    if max_error >= tol:
        raise ValueError(
            f"semantic HNSW fair benchmark requires unit-normalized vectors; max |norm-1|={max_error:.6g}"
        )
    return {
        "n_items": int(len(x)),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
        "max_abs_norm_error": max_error,
    }


def _gate_table(
    assets: Path,
    positive: list[str],
    negative: list[str],
    fractions: list[float],
    k: int,
) -> pd.DataFrame:
    executor = SemanticExecutor.open(str(assets / "sidecar_index"), str(assets / "sidecar_programs"))
    ids = np.arange(executor.index.n_items, dtype=np.int64)
    n_pred = len(positive) + len(negative)
    sem_mean = executor.score_candidates(ids, positive=positive, negative=negative) / max(1, n_pred)

    rows: list[dict[str, float | int]] = []
    for f in fractions:
        if f * len(ids) < k + 1:
            continue
        gate = float(np.quantile(sem_mean, 1.0 - f))
        mask = sem_mean >= gate
        rows.append(
            {
                "target_fraction": float(f),
                "gate_logprob": gate,
                "actual_fraction": float(mask.mean()),
                "eligible_items": int(mask.sum()),
            }
        )
    if not rows:
        raise ValueError("no requested fraction leaves enough eligible items for k")
    return pd.DataFrame(rows).sort_values("target_fraction", ascending=False).reset_index(drop=True)


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["target_fraction", "method"])
        .agg(
            queries=("query_id", "count"),
            mean_latency_ms=("latency_ms", "mean"),
            p50_latency_ms=("latency_ms", "median"),
            p95_latency_ms=("latency_ms", lambda x: np.quantile(x, 0.95)),
            mean_recall_at_k=("recall_at_k", "mean"),
            mean_returned=("returned", "mean"),
            mean_visited=("visited", "mean"),
            mean_semantic_evals=("semantic_evals", "mean"),
            mean_predicate_evals=("predicate_evals", "mean"),
            mean_dense_pruned_before_semantic=("dense_pruned_before_semantic", "mean"),
            qualified_fraction=("qualified_fraction", "mean"),
            live_match_rate=("live_matches_materialized", "mean"),
        )
        .reset_index()
    )


def _fairness(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for f in sorted(summary.target_fraction.unique(), reverse=True):
        live = summary[(summary.target_fraction == f) & (summary.method == "semantic_hnsw_live")].iloc[0]
        filt = summary[
            (summary.target_fraction == f) & (summary.method == "hnsw_filtered_materialized")
        ].iloc[0]
        mat = summary[
            (summary.target_fraction == f) & (summary.method == "custom_hnsw_materialized")
        ].iloc[0]
        overhead = float(live.mean_latency_ms - mat.mean_latency_ms)
        predicate_evals = float(live.mean_predicate_evals)
        rows.append(
            {
                "target_fraction": float(f),
                "live_latency_ms": float(live.mean_latency_ms),
                "filtered_same_dot_ms": float(filt.mean_latency_ms),
                "live_over_filtered_same_dot": float(live.mean_latency_ms / filt.mean_latency_ms),
                "live_minus_filtered_recall": float(live.mean_recall_at_k - filt.mean_recall_at_k),
                "live_recall": float(live.mean_recall_at_k),
                "filtered_recall": float(filt.mean_recall_at_k),
                "live_program_overhead_ms": overhead,
                "predicate_evals": predicate_evals,
                "approx_ns_per_predicate_eval": float(overhead * 1e6 / max(1.0, predicate_evals)),
                "exact_id_parity": float(live.live_match_rate),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> Path:
    repo_root = Path(args.repo_root).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assets = out / "assets"

    if args.force_export and assets.exists():
        shutil.rmtree(assets)
    if not assets.exists():
        t0 = time.time()
        export_native(args.config, str(assets))
        print(f"export_seconds={time.time() - t0:.2f}")

    norm_stats = _verify_normalized(assets / "fp32_items.f32")
    positive = _csv_list(args.positive)
    negative = _csv_list(args.negative)
    fractions = _fractions(args.fractions)
    gates = _gate_table(assets, positive, negative, fractions, args.k)
    gates.to_csv(out / "gates.csv", index=False)

    binary = _build_rust(repo_root)
    all_runs: list[pd.DataFrame] = []
    for row in gates.itertuples(index=False):
        fraction = float(row.target_fraction)
        gate = float(row.gate_logprob)
        run_csv = out / f"run_{fraction:.3f}.csv"
        cmd = [
            str(binary),
            "--assets", str(assets),
            "--programs", str(assets / "sidecar_programs"),
            "--positive", args.positive,
            "--negative", args.negative,
            "--queries", str(args.queries),
            "--k", str(args.k),
            "--ef", str(args.ef),
            "--m", str(args.m),
            "--ef-construction", str(args.ef_construction),
            "--gate-logprob", str(gate),
            "--postfilter-oversample", str(args.postfilter_oversample),
            "--out", str(run_csv),
        ]
        print("RUN", " ".join(cmd))
        subprocess.run(cmd, check=True)
        frame = pd.read_csv(run_csv)
        frame["target_fraction"] = fraction
        frame["gate_logprob"] = gate
        all_runs.append(frame)

    raw = pd.concat(all_runs, ignore_index=True)
    raw.to_csv(out / "raw.csv", index=False)
    summary = _summary(raw)
    summary.to_csv(out / "summary.csv", index=False)
    fair = _fairness(summary)
    fair.to_csv(out / "fairness.csv", index=False)

    manifest = {
        "repo_commit": _git_commit(repo_root),
        "config": str(Path(args.config).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "rustc": _rust_version(),
        "positive": positive,
        "negative": negative,
        "fractions": fractions,
        "queries": args.queries,
        "k": args.k,
        "ef": args.ef,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "postfilter_oversample": args.postfilter_oversample,
        "normalization": norm_stats,
        "timing_scope": "same normalized-dot graph; live Binary1-LS2-int4 predicate execution is inside timed traversal",
    }
    (out / "environment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nFAIRNESS SUMMARY")
    print(fair.to_string(index=False))
    print(f"\nartifacts={out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--config", default="configs/binary_bbq.yaml")
    p.add_argument("--output-dir", default="results/semantic_hnsw_live_fair")
    p.add_argument("--positive", default="minimalist,office_appropriate")
    p.add_argument("--negative", default="technical_sporty")
    p.add_argument("--fractions", default=",".join(str(x) for x in DEFAULT_FRACTIONS))
    p.add_argument("--queries", type=int, default=100)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--ef", type=int, default=128)
    p.add_argument("--m", type=int, default=24)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--postfilter-oversample", type=int, default=8)
    p.add_argument("--force-export", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
