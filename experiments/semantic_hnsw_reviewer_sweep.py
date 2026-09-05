"""Reviewer-proof semantic-HNSW benchmark.

This benchmark addresses two reviewer-facing questions:

1. Does integrated live semantic traversal remain useful against a
   selectivity-aware HNSW over-fetch baseline rather than a fixed oversample?
2. How expensive is one compiled semantic predicate relative to the exact
   scalar 384-dimensional normalized-dot kernel used by the custom traversal?

The implementation is intentionally conservative. For each predetermined
predicate set it builds one HNSW graph, materializes the semantic scores once,
computes brute-force dense truth once per query, and sweeps all selectivity gates
and over-fetch budgets inside that same process. This avoids the earlier harness
bug that rebuilt HNSW and recomputed truth once per over-fetch point.

Recall is *traversal recall*: brute-force top-K dense neighbors among items that
pass the same compiled semantic predicate. It is not end-to-end relevance.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from experiments.export_hnsw_assets import export as export_hnsw_assets
from experiments.semantic_hnsw_live_sweep import (
    _cpu_model,
    _git_commit,
    _rust_version,
    _verify_normalized,
)
from ras import SemanticExecutor


DEFAULT_FRACTIONS = (0.50, 0.20, 0.10, 0.05, 0.02)
# Continue beyond 2x K/selectivity so the matched-recall frontier can actually
# be found rather than merely showing that a modest over-fetch budget fails.
DEFAULT_MULTIPLIERS = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
DEFAULT_PREDICATE_SETS = (
    {
        "name": "office_minimal_not_sporty",
        "positive": ["minimalist", "office_appropriate"],
        "negative": ["technical_sporty"],
    },
    {
        "name": "elegant_quiet_not_chunky",
        "positive": ["elegant", "quiet_luxury"],
        "negative": ["chunky"],
    },
    {
        "name": "retro_relaxed_not_office",
        "positive": ["retro", "relaxed"],
        "negative": ["office_appropriate"],
    },
)


def _csv_floats(value: str) -> list[float]:
    out = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def _build_rust(repo_root: Path) -> tuple[Path, Path]:
    manifest = repo_root / "rust" / "semantic_engine" / "Cargo.toml"
    print("[stage] build reviewer Rust binaries", flush=True)
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(manifest),
            "--bin",
            "semantic_hnsw_reviewer",
            "--bin",
            "dot_cost",
        ],
        check=True,
    )
    root = repo_root / "rust" / "semantic_engine" / "target" / "release"
    return root / "semantic_hnsw_reviewer", root / "dot_cost"


def _semantic_scores(
    assets: Path,
    positive: list[str],
    negative: list[str],
) -> np.ndarray:
    executor = SemanticExecutor.open(
        str(assets / "sidecar_index"), str(assets / "sidecar_programs")
    )
    ids = np.arange(executor.index.n_items, dtype=np.int64)
    n_pred = len(positive) + len(negative)
    return executor.score_candidates(
        ids, positive=positive, negative=negative
    ) / max(1, n_pred)


def _gate_rows(
    scores: np.ndarray, fractions: list[float], k: int
) -> list[dict[str, float | int]]:
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


def _annotate_run(
    frame: pd.DataFrame,
    *,
    predicate_set: str,
    gates: list[dict[str, float | int]],
    k: int,
) -> pd.DataFrame:
    out = frame.copy()
    gate_map = {i: row for i, row in enumerate(gates)}
    out["predicate_set"] = predicate_set
    out["target_fraction"] = out.gate_index.map(
        lambda i: float(gate_map[int(i)]["target_fraction"])
    )
    out["actual_fraction"] = out.gate_index.map(
        lambda i: float(gate_map[int(i)]["actual_fraction"])
    )
    out = out.rename(columns={"overfetch_multiplier": "requested_overfetch_multiplier"})
    out["overfetch_multiplier"] = np.where(
        out.method.eq("hnsw_overfetch_materialized"),
        out.ask.astype(float) * out.actual_fraction.astype(float) / float(k),
        0.0,
    )
    return out


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "predicate_set",
        "target_fraction",
        "actual_fraction",
        "requested_overfetch_multiplier",
        "overfetch_multiplier",
        "ask",
        "method",
    ]
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
    """Compare every over-fetch point with live traversal on the same graph/gate."""
    rows: list[dict[str, float | str | int]] = []
    for (name, target), g in summary.groupby(
        ["predicate_set", "target_fraction"], dropna=False
    ):
        live_rows = g[g.method.eq("semantic_hnsw_live")]
        mat_rows = g[g.method.eq("custom_hnsw_materialized")]
        over_rows = g[g.method.eq("hnsw_overfetch_materialized")]
        if len(live_rows) != 1 or len(mat_rows) != 1:
            raise RuntimeError(
                f"expected one live/materialized row for {name} @ {target}, "
                f"got {len(live_rows)}/{len(mat_rows)}"
            )
        live = live_rows.iloc[0]
        mat = mat_rows.iloc[0]
        overhead = float(live.mean_latency_ms - mat.mean_latency_ms)
        ns_pred = overhead * 1e6 / max(1.0, float(live.mean_predicate_evals))
        for _, over in over_rows.iterrows():
            rows.append(
                {
                    "predicate_set": str(name),
                    "target_fraction": float(target),
                    "actual_fraction": float(live.actual_fraction),
                    "requested_overfetch_multiplier": float(
                        over.requested_overfetch_multiplier
                    ),
                    "overfetch_multiplier": float(over.overfetch_multiplier),
                    "ask": int(over.ask),
                    "live_ms": float(live.mean_latency_ms),
                    "overfetch_ms": float(over.mean_latency_ms),
                    "live_traversal_recall": float(live.traversal_recall_at_50),
                    "overfetch_traversal_recall": float(over.traversal_recall_at_50),
                    "overfetch_minus_live_recall": float(
                        over.traversal_recall_at_50 - live.traversal_recall_at_50
                    ),
                    "overfetch_over_live_latency": float(
                        over.mean_latency_ms / live.mean_latency_ms
                    ),
                    "live_program_overhead_ms": overhead,
                    "mean_predicate_evals": float(live.mean_predicate_evals),
                    "approx_ns_per_predicate_eval": ns_pred,
                    "exact_id_parity": float(live.exact_id_parity),
                }
            )
    return pd.DataFrame(rows)


def _matched_frontier(pairs: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Pick the fastest over-fetch point whose recall is within tolerance of live."""
    rows: list[pd.Series] = []
    for (_, _), g in pairs.groupby(["predicate_set", "target_fraction"]):
        ok = g[g.overfetch_minus_live_recall >= -tolerance]
        if len(ok):
            rows.append(ok.sort_values(["overfetch_ms", "ask"]).iloc[0])
        else:
            best = g.sort_values(
                ["overfetch_traversal_recall", "overfetch_ms"],
                ascending=[False, True],
            ).iloc[0].copy()
            rows.append(best)
    out = pd.DataFrame(rows).reset_index(drop=True)
    if len(out):
        out["matched_within_tolerance"] = (
            out.overfetch_minus_live_recall >= -tolerance
        )
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
        export_hnsw_assets(args.config, str(assets))
        print(f"[stage] export_seconds={time.time() - t0:.2f}", flush=True)
    else:
        print(f"[stage] reusing assets: {assets}", flush=True)

    norm_stats = _verify_normalized(assets / "fp32_items.f32")
    reviewer_bin, dot_bin = _build_rust(repo_root)

    dot_run = subprocess.run(
        [
            str(dot_bin),
            "--items",
            str(assets / "fp32_items.f32"),
            "--evals",
            str(args.dot_evals),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    print(dot_run.stdout.strip(), flush=True)
    dot_ns = _parse_dot_ns(dot_run.stdout)

    fractions = _csv_floats(args.fractions)
    multipliers = _csv_floats(args.overfetch_multipliers)
    predicate_sets = list(DEFAULT_PREDICATE_SETS)
    if args.predicate_sets_json:
        predicate_sets = json.loads(Path(args.predicate_sets_json).read_text())

    raw_frames: list[pd.DataFrame] = []
    gate_records: list[dict[str, object]] = []

    for set_idx, ps in enumerate(predicate_sets, start=1):
        name = str(ps["name"])
        positive = list(ps.get("positive", []))
        negative = list(ps.get("negative", []))
        print(
            f"\n[set {set_idx}/{len(predicate_sets)}] {name} "
            f"(+{','.join(positive)} -{','.join(negative)})",
            flush=True,
        )
        scores = _semantic_scores(assets, positive, negative)
        gates = _gate_rows(scores, fractions, args.k)
        if not gates:
            raise RuntimeError(f"no valid gates for predicate set {name}")
        for row in gates:
            gate_records.append({"predicate_set": name, **row})

        run_csv = out / f"run_{name}.csv"
        cmd = [
            str(reviewer_bin),
            "--assets",
            str(assets),
            "--programs",
            str(assets / "sidecar_programs"),
            "--positive",
            ",".join(positive),
            "--negative",
            ",".join(negative),
            "--queries",
            str(args.queries),
            "--k",
            str(args.k),
            "--ef",
            str(args.ef),
            "--m",
            str(args.m),
            "--ef-construction",
            str(args.ef_construction),
            "--gates",
            ",".join(str(float(r["gate_logprob"])) for r in gates),
            "--overfetch-multipliers",
            ",".join(str(x) for x in multipliers),
            "--progress-every",
            str(args.progress_every),
            "--out",
            str(run_csv),
        ]
        subprocess.run(cmd, check=True)
        frame = _annotate_run(
            pd.read_csv(run_csv), predicate_set=name, gates=gates, k=args.k
        )
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
        "recall_definition": (
            "traversal recall: brute-force dense top-K among items passing the "
            "same compiled semantic predicate"
        ),
        "fairness_note": (
            "for each predicate set, one HNSW graph is built and all gates, live "
            "traversals, materialized traversals, and over-fetch budgets are measured "
            "inside that process; brute-force dense scores are computed once per query"
        ),
        "program_cost_note": (
            "approx_ns_per_predicate_eval is the incremental live-minus-materialized "
            "custom traversal time divided by live predicate invocations; it is an "
            "end-to-end incremental estimate, not a standalone instruction benchmark"
        ),
    }
    (out / "environment.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\nMATCHED-RECALL FRONTIER", flush=True)
    print(matched.to_string(index=False), flush=True)
    print(f"\n384D same-kernel dot: {dot_ns:.2f} ns/eval", flush=True)
    print(f"artifacts={out}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--config", default="configs/binary_bbq.yaml")
    p.add_argument("--output-dir", default="results/semantic_hnsw_reviewer")
    p.add_argument("--fractions", default=",".join(map(str, DEFAULT_FRACTIONS)))
    p.add_argument(
        "--overfetch-multipliers", default=",".join(map(str, DEFAULT_MULTIPLIERS))
    )
    p.add_argument("--predicate-sets-json", default=None)
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--ef", type=int, default=128)
    p.add_argument("--m", type=int, default=24)
    p.add_argument("--ef-construction", type=int, default=200)
    p.add_argument("--recall-tolerance", type=float, default=0.005)
    p.add_argument("--dot-evals", type=int, default=2_000_000)
    p.add_argument("--progress-every", type=int, default=100)
    p.add_argument("--force-export", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
