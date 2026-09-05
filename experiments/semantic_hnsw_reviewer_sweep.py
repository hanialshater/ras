"""Reviewer-proof semantic-HNSW benchmark.

This experiment addresses two specific questions left open by the first live-HNSW
result:

1. Does integrated live semantic traversal still look useful against a
   selectivity-aware HNSW over-fetch baseline rather than a fixed oversample?
2. How expensive is one compiled semantic predicate relative to the exact
   384-dimensional normalized-dot kernel used by the custom traversal?

The script deliberately reuses ``semantic_hnsw_live`` rather than introducing a
new search algorithm. Each invocation compares live traversal and over-fetch on
THE SAME HNSW graph. We sweep several over-fetch multipliers around K/selectivity,
run multiple predetermined predicate sets, and use 1,000 queries by default.

Recall in this experiment is *traversal recall*: brute-force top-K dense neighbors
among items passing the same compiled predicate. It is not end-to-end relevance.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from experiments.export_native_finalists import export as export_native
from experiments.semantic_hnsw_live_sweep import (
    _cpu_model,
    _git_commit,
    _rust_version,
    _verify_normalized,
)
from ras import SemanticExecutor


DEFAULT_FRACTIONS = (0.50, 0.20, 0.10, 0.05, 0.02)
DEFAULT_MULTIPLIERS = (0.75, 1.0, 1.5, 2.0)
DEFAULT_PREDICATE_SETS = (
    {"name": "office_minimal_not_sporty", "positive": ["minimalist", "office_appropriate"], "negative": ["technical_sporty"]},
    {"name": "elegant_quiet_not_chunky", "positive": ["elegant", "quiet_luxury"], "negative": ["chunky"]},
    {"name": "retro_relaxed_not_office", "positive": ["retro", "relaxed"], "negative": ["office_appropriate"]},
)


def _csv_floats(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _build_rust(repo_root: Path) -> tuple[Path, Path]:
    manifest = repo_root / "rust" / "semantic_engine" / "Cargo.toml"
    subprocess.run(
        [
            "cargo", "build", "--release", "--manifest-path", str(manifest),
            "--bin", "semantic_hnsw_live", "--bin", "dot_cost",
        ],
        check=True,
    )
    root = repo_root / "rust" / "semantic_engine" / "target" / "release"
    return root / "semantic_hnsw_live", root / "dot_cost"


def _semantic_scores(
    assets: Path,
    positive: list[str],
    negative: list[str],
) -> np.ndarray:
    executor = SemanticExecutor.open(str(assets / "sidecar_index"), str(assets / "sidecar_programs"))
    ids = np.arange(executor.index.n_items, dtype=np.int64)
    n_pred = len(positive) + len(negative)
    return executor.score_candidates(ids, positive=positive, negative=negative) / max(1, n_pred)


def _gate_rows(scores: np.ndarray, fractions: list[float], k: int) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for f in fractions:
        if f * len(scores) < k + 1:
            continue
        gate = float(np.quantile(scores, 1.0 - f))
        actual = float(np.mean(scores >= gate))
        rows.append(
            {
                "target_fraction": float(f),
                "gate_logprob": gate,
                "actual_fraction": actual,
                "eligible_items": int(np.sum(scores >= gate)),
            }
        )
    return rows


def _parse_dot_ns(stdout: str) -> float:
    m = re.search(r"ns_per_dot=([0-9.]+)", stdout)
    if not m:
        raise RuntimeError(f"could not parse dot benchmark output: {stdout}")
    return float(m.group(1))


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["predicate_set", "target_fraction", "actual_fraction", "overfetch_multiplier", "ask", "method"]
    return (
        raw.groupby(keys, dropna=False)
        .agg(
            queries=("query_id", "count"),
            mean_latency_ms=("latency_ms", "mean"),
            p50_latency_ms=("latency_ms", "median"),
            p95_latency_ms=("latency_ms", lambda x: np.quantile(x, 0.95)),
            traversal_recall_at_50=("recall_at_k", "mean"),
            mean_returned=("returned", "mean"),
            mean_predicate_evals=("predicate_evals", "mean"),
            mean_semantic_evals=("semantic_evals", "mean"),
            mean_dense_pruned=("dense_pruned_before_semantic", "mean"),
            exact_id_parity=("live_matches_materialized", "mean"),
        )
        .reset_index()
    )


def _pair_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """Same-run comparison: over-fetch vs live on the exact same built graph."""
    rows: list[dict[str, float | str | int]] = []
    group_cols = ["predicate_set", "target_fraction", "actual_fraction", "overfetch_multiplier", "ask"]
    for key, g in summary.groupby(group_cols, dropna=False):
        by = {r.method: r for r in g.itertuples(index=False)}
        if "semantic_hnsw_live" not in by or "hnsw_postfilter_materialized" not in by:
            continue
        live = by["semantic_hnsw_live"]
        over = by["hnsw_postfilter_materialized"]
        mat = by.get("custom_hnsw_materialized")
        overhead = float(live.mean_latency_ms - mat.mean_latency_ms) if mat is not None else float("nan")
        ns_pred = (
            overhead * 1e6 / max(1.0, float(live.mean_predicate_evals))
            if mat is not None else float("nan")
        )
        rows.append(
            {
                "predicate_set": key[0],
                "target_fraction": float(key[1]),
                "actual_fraction": float(key[2]),
                "overfetch_multiplier": float(key[3]),
                "ask": int(key[4]),
                "live_ms": float(live.mean_latency_ms),
                "overfetch_ms": float(over.mean_latency_ms),
                "live_traversal_recall": float(live.traversal_recall_at_50),
                "overfetch_traversal_recall": float(over.traversal_recall_at_50),
                "overfetch_minus_live_recall": float(over.traversal_recall_at_50 - live.traversal_recall_at_50),
                "overfetch_over_live_latency": float(over.mean_latency_ms / live.mean_latency_ms),
                "live_program_overhead_ms": overhead,
                "mean_predicate_evals": float(live.mean_predicate_evals),
                "approx_ns_per_predicate_eval": ns_pred,
            }
        )
    return pd.DataFrame(rows)


def _matched_frontier(pairs: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Pick the fastest over-fetch point whose recall is within tolerance of live."""
    rows: list[pd.Series] = []
    for (_, _), g in pairs.groupby(["predicate_set", "target_fraction"]):
        ok = g[g.overfetch_minus_live_recall >= -tolerance]
        if len(ok):
            rows.append(ok.sort_values("overfetch_ms").iloc[0])
        else:
            # Keep the best-recall point so a failed match is explicit rather than hidden.
            best = g.sort_values(["overfetch_traversal_recall", "overfetch_ms"], ascending=[False, True]).iloc[0].copy()
            best["ask"] = int(best["ask"])
            rows.append(best)
    out = pd.DataFrame(rows).reset_index(drop=True)
    if len(out):
        out["matched_within_tolerance"] = out.overfetch_minus_live_recall >= -tolerance
    return out


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
    live_bin, dot_bin = _build_rust(repo_root)

    dot_run = subprocess.run(
        [str(dot_bin), "--items", str(assets / "fp32_items.f32"), "--evals", str(args.dot_evals)],
        check=True, text=True, capture_output=True,
    )
    print(dot_run.stdout.strip())
    dot_ns = _parse_dot_ns(dot_run.stdout)

    fractions = _csv_floats(args.fractions)
    multipliers = _csv_floats(args.overfetch_multipliers)
    predicate_sets = list(DEFAULT_PREDICATE_SETS)
    if args.predicate_sets_json:
        predicate_sets = json.loads(Path(args.predicate_sets_json).read_text())

    raw_frames: list[pd.DataFrame] = []
    gate_records: list[dict[str, object]] = []

    for ps in predicate_sets:
        name = str(ps["name"])
        positive = list(ps.get("positive", []))
        negative = list(ps.get("negative", []))
        scores = _semantic_scores(assets, positive, negative)
        gates = _gate_rows(scores, fractions, args.k)
        if not gates:
            raise RuntimeError(f"no valid gates for predicate set {name}")

        pos_arg = ",".join(positive)
        neg_arg = ",".join(negative)
        for gate_row in gates:
            gate_records.append({"predicate_set": name, **gate_row})
            target = float(gate_row["target_fraction"])
            actual = float(gate_row["actual_fraction"])
            gate = float(gate_row["gate_logprob"])

            for mult in multipliers:
                # Existing Rust post-filter asks for k * integer_oversample. Choose
                # the smallest integer whose ask is at least mult * k/selectivity.
                oversample = max(1, int(math.ceil(mult / actual)))
                ask = min(int(norm_stats["n_items"]), args.k * oversample)
                actual_mult = ask * actual / args.k
                run_csv = out / f"run_{name}_{target:.3f}_x{mult:.2f}.csv"
                cmd = [
                    str(live_bin),
                    "--assets", str(assets),
                    "--programs", str(assets / "sidecar_programs"),
                    "--positive", pos_arg,
                    "--negative", neg_arg,
                    "--queries", str(args.queries),
                    "--k", str(args.k),
                    "--ef", str(args.ef),
                    "--m", str(args.m),
                    "--ef-construction", str(args.ef_construction),
                    "--gate-logprob", str(gate),
                    "--postfilter-oversample", str(oversample),
                    "--out", str(run_csv),
                ]
                print(
                    f"RUN set={name} target={target:.3f} actual={actual:.4f} "
                    f"requested_x={mult:.2f} actual_x={actual_mult:.2f} ask={ask}"
                )
                subprocess.run(cmd, check=True)
                frame = pd.read_csv(run_csv)
                frame["predicate_set"] = name
                frame["target_fraction"] = target
                frame["actual_fraction"] = actual
                frame["requested_overfetch_multiplier"] = mult
                frame["overfetch_multiplier"] = actual_mult
                frame["ask"] = ask
                raw_frames.append(frame)

    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(out / "raw.csv", index=False)
    pd.DataFrame(gate_records).to_csv(out / "gates.csv", index=False)

    summary = _summarize(raw)
    summary.to_csv(out / "summary.csv", index=False)
    pairs = _pair_rows(summary)
    pairs["same_kernel_dot_ns"] = dot_ns
    pairs["predicate_over_dot"] = pairs.approx_ns_per_predicate_eval / dot_ns
    pairs.to_csv(out / "same_run_pairs.csv", index=False)

    matched = _matched_frontier(pairs, args.recall_tolerance)
    matched.to_csv(out / "matched_recall.csv", index=False)

    manifest = {
        "repo_commit": _git_commit(repo_root),
        "config": str(Path(args.config).resolve()),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "rustc": _rust_version(),
        "queries": args.queries,
        "k": args.k,
        "ef": args.ef,
        "m": args.m,
        "ef_construction": args.ef_construction,
        "fractions": fractions,
        "requested_overfetch_multipliers": multipliers,
        "predicate_sets": predicate_sets,
        "recall_tolerance": args.recall_tolerance,
        "same_kernel_384d_dot_ns": dot_ns,
        "normalization": norm_stats,
        "recall_definition": "traversal recall: brute-force dense top-K among items passing the same compiled semantic predicate",
        "fairness_note": "each over-fetch/live pair is measured in the same semantic_hnsw_live invocation on the same built graph",
    }
    (out / "environment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nMATCHED-RECALL FRONTIER")
    print(matched.to_string(index=False))
    print(f"\n384D same-kernel dot: {dot_ns:.2f} ns/eval")
    print(f"artifacts={out}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--config", default="configs/binary_bbq.yaml")
    p.add_argument("--output-dir", default="results/semantic_hnsw_reviewer")
    p.add_argument("--fractions", default=",".join(map(str, DEFAULT_FRACTIONS)))
    p.add_argument("--overfetch-multipliers", default=",".join(map(str, DEFAULT_MULTIPLIERS)))
    p.add_argument("--predicate-sets-json", default=None)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--ef", type=int, default=128)
    p.add_argument("--m", type=int, default=24)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--recall-tolerance", type=float, default=0.005)
    p.add_argument("--dot-evals", type=int, default=2_000_000)
    p.add_argument("--force-export", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
