"""Large-scale independent-teacher search benchmark for Random Semantic Algebra.

This is the paper-grade benchmark harness. It:
  * fixes a product population once;
  * caches MiniLM and CLIP embeddings;
  * repeats strict fit/cal/test splits over multiple seeds;
  * generates compound queries using fit data only;
  * compares dense-only, RSA, full-precision linear semantic proxy, and oracle;
  * sweeps candidate retention budgets;
  * writes per-query rows, aggregate CIs, manifests, and figures.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier
from transformers import AutoProcessor, CLIPModel

from ras.cache import load_or_compute_array
from ras.calibration import fit_scalar_calibrator
from ras.composition import compose_query
from ras.config import config_hash, load_config
from ras.core import make_protocol_split
from ras.metrics import binary_ranking_stats, bootstrap_mean_ci, best_f1_threshold, metric_row
from ras.predicates import add_pair_interactions, fit_boosted_lut, score_boosted
from ras.queries import apply_exact, generate_query_benchmark
from ras.repro import environment_manifest, write_json
from ras.retrieval import encode_queries, encode_titles
from ras.substrate import build_substrate
from ras.teachers import LATENT_SPECS, encode_clip_images_dataset, labels_from_fit_threshold, teacher_score_matrix


def _select_dataset(cfg: Dict[str, Any]):
    dc = cfg["data"]
    ds = load_dataset(dc["dataset"], split="train")
    keep = ["id", "gender", "masterCategory", "subCategory", "articleType", "baseColour", "season", "usage", "productDisplayName", "image"]
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    n = dc.get("n_products")
    if n is not None and len(ds) > int(n):
        rng = np.random.default_rng(int(dc.get("data_seed", 7)))
        ds = ds.select(rng.choice(len(ds), size=int(n), replace=False).tolist())
    return ds, keep


def _metadata_df(ds, keep):
    df = ds.remove_columns(["image"]).to_pandas()
    for c in keep:
        if c in df.columns and c != "id":
            df[c] = df[c].fillna("Unknown").astype(str)
    return df


def _retrieval_embeddings(ds, df, cfg):
    rc = cfg["retrieval"]
    meta = {"dataset": cfg["data"]["dataset"], "n": len(ds), "data_seed": cfg["data"].get("data_seed", 7), "model": rc["model"], "batch_size": rc.get("batch_size", 256)}
    holder = {}
    def compute():
        model, x = encode_titles(rc["model"], df["productDisplayName"].tolist(), int(rc.get("batch_size", 256)))
        holder["model"] = model
        return x
    x = load_or_compute_array(cfg["cache_dir"], "retrieval", meta, compute)
    model = holder.get("model") or SentenceTransformer(rc["model"])
    return model, np.asarray(x, dtype=np.float32)


def _teacher_embeddings_and_scores(ds, cfg, device):
    tc = cfg["teacher"]
    meta = {"dataset": cfg["data"]["dataset"], "n": len(ds), "data_seed": cfg["data"].get("data_seed", 7), "model": tc["model"], "batch_size": tc.get("batch_size", 128)}
    def compute_img():
        model = CLIPModel.from_pretrained(tc["model"]).to(device).eval()
        processor = AutoProcessor.from_pretrained(tc["model"])
        return encode_clip_images_dataset(model, processor, ds, device, int(tc.get("batch_size", 128)))
    image_embs = load_or_compute_array(cfg["cache_dir"], "clip_images", meta, compute_img)
    score_meta = {**meta, "latent_specs": LATENT_SPECS}
    def compute_scores():
        model = CLIPModel.from_pretrained(tc["model"]).to(device).eval()
        processor = AutoProcessor.from_pretrained(tc["model"])
        return teacher_score_matrix(np.asarray(image_embs), model, processor, device, LATENT_SPECS)
    return np.asarray(load_or_compute_array(cfg["cache_dir"], "teacher_scores", score_meta, compute_scores), dtype=np.float32)


def _fit_rsa(xfit, xcal, xtest, yfit, ycal, ytest, seed, cfg):
    rc = cfg["rsa"]
    substrate = build_substrate(xfit, xcal, xtest, name=f"orth{xfit.shape[1]}_{rc['bits']}bit_{rc['quantizer']}", seed=int(seed), bits=int(rc["bits"]), projection_kind="orthogonal", quantizer_kind=str(rc["quantizer"]))
    cal_scores, test_scores, metrics = [], [], []
    for c, spec in enumerate(LATENT_SPECS):
        unary = fit_boosted_lut(substrate.Q_fit, yfit[:, c], k=int(rc["k_coords"]), n_bins=substrate.n_bins, candidate_pool=int(rc["candidate_pool"]), refine_passes=1)
        model = add_pair_interactions(substrate.Q_fit, yfit[:, c], unary, n_pairs=int(rc.get("pair_luts", 0)), pair_pool=12)
        sc = score_boosted(substrate.Q_cal, model); st = score_boosted(substrate.Q_test, model)
        row = metric_row(ytest[:, c], st, best_f1_threshold(sc, ycal[:, c])); row.update({"concept": spec["name"], "method": "rsa"}); metrics.append(row)
        cal_scores.append(sc); test_scores.append(st)
    cal_scores = np.column_stack(cal_scores); test_scores = np.column_stack(test_scores)
    calibrators = [fit_scalar_calibrator(cal_scores[:, c], ycal[:, c]) for c in range(ycal.shape[1])]
    ltest = np.column_stack([calibrators[c].transform(test_scores[:, c]) for c in range(ycal.shape[1])])
    return ltest, pd.DataFrame(metrics), substrate.item_bytes_theoretical


def _fit_linear_proxy(xfit, xcal, xtest, yfit, ycal, ytest, seed):
    cal_scores, test_scores, metrics = [], [], []
    for c, spec in enumerate(LATENT_SPECS):
        clf = SGDClassifier(loss="log_loss", class_weight="balanced", alpha=1e-4, max_iter=1200, random_state=int(seed) + c)
        clf.fit(xfit, yfit[:, c].astype(int))
        sc = clf.decision_function(xcal); st = clf.decision_function(xtest)
        row = metric_row(ytest[:, c], st, best_f1_threshold(sc, ycal[:, c])); row.update({"concept": spec["name"], "method": "linear_fp32_proxy"}); metrics.append(row)
        cal_scores.append(sc); test_scores.append(st)
    cal_scores = np.column_stack(cal_scores); test_scores = np.column_stack(test_scores)
    calibrators = [fit_scalar_calibrator(cal_scores[:, c], ycal[:, c]) for c in range(ycal.shape[1])]
    return np.column_stack([calibrators[c].transform(test_scores[:, c]) for c in range(ycal.shape[1])]), pd.DataFrame(metrics)


def _evaluate_queries(model, xtest, df_test, ytest, rsa_logits, linear_logits, queries, cfg, seed):
    bc = cfg["benchmark"]; ann_pool = int(bc["ann_pool"]); retention = [float(x) for x in bc["retention"]]
    name_to_idx = {s["name"]: i for i, s in enumerate(LATENT_SPECS)}
    qemb = encode_queries(model, [q.text for q in queries])
    rows, query_rows = [], []
    for qi, query in enumerate(queries):
        dense = xtest @ qemb[qi]; ann = np.argsort(dense)[::-1][:ann_pool]
        pool_df = df_test.iloc[ann].reset_index(drop=True); pool_dense = dense[ann]; pool_y = ytest[ann]; pool_rsa = rsa_logits[ann]; pool_linear = linear_logits[ann]
        mask = apply_exact(pool_df, query.exact)
        pool_df = pool_df.loc[mask].reset_index(drop=True); pool_dense = pool_dense[mask]; pool_y = pool_y[mask]; pool_rsa = pool_rsa[mask]; pool_linear = pool_linear[mask]
        if len(pool_df) == 0: continue
        truth = np.ones(len(pool_df), dtype=bool)
        for name in query.positive: truth &= pool_y[:, name_to_idx[name]]
        for name in query.negative: truth &= ~pool_y[:, name_to_idx[name]]
        rsa_score = compose_query(pool_rsa, name_to_idx, query.positive, query.negative); linear_score = compose_query(pool_linear, name_to_idx, query.positive, query.negative)
        orders = {"dense": np.argsort(pool_dense)[::-1], "rsa": np.argsort(rsa_score)[::-1], "linear_fp32_proxy": np.argsort(linear_score)[::-1], "oracle": np.concatenate([np.flatnonzero(truth), np.flatnonzero(~truth)])}
        query_rows.append({**query.to_dict(), "seed": int(seed), "ann_after_exact": int(len(pool_df)), "pool_true": int(truth.sum())})
        for frac in retention:
            k = max(1, min(len(pool_df), int(round(len(pool_df) * frac))))
            for method, order in orders.items():
                rows.append({"seed": int(seed), "query_id": query.query_id, "query": query.text, "retention": frac, "method": method, "pool_size": len(pool_df), **binary_ranking_stats(truth, order[:k])})
    return pd.DataFrame(rows), pd.DataFrame(query_rows)


def _aggregate(per_query, cfg):
    nboot = int(cfg["benchmark"].get("bootstrap_samples", 2000)); rows, delta_rows = [], []
    for (method, retention), g in per_query.groupby(["method", "retention"], sort=True):
        valid = g[g["total_true"] > 0]
        for metric in ["recall", "purity", "recall_efficiency"]:
            rows.append({"method": method, "retention": retention, "metric": metric, **bootstrap_mean_ci(valid[metric].to_numpy(), seed=991, n_boot=nboot)})
    pivot = per_query[per_query.total_true > 0].pivot_table(index=["seed", "query_id", "retention"], columns="method", values=["recall", "purity"], aggfunc="first")
    for retention in sorted(per_query.retention.unique()):
        try: sub = pivot.xs(retention, level="retention")
        except KeyError: continue
        for method in ["rsa", "linear_fp32_proxy"]:
            for metric in ["recall", "purity"]:
                if (metric, method) in sub.columns and (metric, "dense") in sub.columns:
                    d = (sub[(metric, method)] - sub[(metric, "dense")]).dropna().to_numpy()
                    delta_rows.append({"method": method, "retention": retention, "metric": f"delta_{metric}_vs_dense", **bootstrap_mean_ci(d, seed=992, n_boot=nboot)})
    return pd.DataFrame(rows), pd.DataFrame(delta_rows)


def _plot_summary(summary, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in [("recall", "Mean recall of teacher-defined relevant items", "recall_vs_retention.png"), ("purity", "Mean purity of retained candidates", "purity_vs_retention.png"), ("recall_efficiency", "Recall / maximum possible recall at budget", "efficiency_vs_retention.png")]:
        plt.figure(figsize=(7, 5))
        for method in ["dense", "rsa", "linear_fp32_proxy", "oracle"]:
            g = summary[(summary.method == method) & (summary.metric == metric)].sort_values("retention")
            if len(g):
                plt.plot(g.retention, g["mean"], marker="o", label=method); plt.fill_between(g.retention, g.lo, g.hi, alpha=0.12)
        plt.xscale("log"); plt.xlabel("Fraction of exact-filtered ANN pool kept"); plt.ylabel(ylabel); plt.title(metric.replace("_", " ").title()); plt.legend(); plt.tight_layout(); plt.savefig(out_dir / filename, dpi=180); plt.close()


def run(config_path: str, output_dir: str | None = None):
    cfg = load_config(config_path); device = "cuda" if torch.cuda.is_available() else "cpu"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + config_hash(cfg)
    root = Path(output_dir or cfg["results_dir"]) / run_id; root.mkdir(parents=True, exist_ok=False); shutil.copy2(config_path, root / "config.yaml"); write_json(root / "environment.json", environment_manifest())
    ds, keep = _select_dataset(cfg); df = _metadata_df(ds, keep); retrieval_model, x = _retrieval_embeddings(ds, df, cfg); teacher_scores = _teacher_embeddings_and_scores(ds, cfg, device)
    all_rows, all_query_rows, all_predicate_rows = [], [], []
    for seed in cfg["benchmark"]["seeds"]:
        split = make_protocol_split(len(df), int(seed), strict=True); xfit, xcal, xtest = x[split.fit_idx], x[split.cal_idx], x[split.test_idx]; df_fit = df.iloc[split.fit_idx].reset_index(drop=True); df_test = df.iloc[split.test_idx].reset_index(drop=True)
        yfit, ycal, ytest, _ = labels_from_fit_threshold(teacher_scores, split.fit_idx, split.cal_idx, split.test_idx, prevalence=float(cfg["teacher"].get("positive_prevalence", 0.40)))
        rsa_ltest, rsa_metrics, item_bytes = _fit_rsa(xfit, xcal, xtest, yfit, ycal, ytest, int(seed), cfg); linear_ltest, linear_metrics = _fit_linear_proxy(xfit, xcal, xtest, yfit, ycal, ytest, int(seed))
        rsa_metrics["seed"] = int(seed); linear_metrics["seed"] = int(seed); all_predicate_rows += rsa_metrics.to_dict("records") + linear_metrics.to_dict("records")
        queries = generate_query_benchmark(df_fit, yfit, n_queries=int(cfg["benchmark"]["n_queries"]), seed=int(seed) + 404, min_fit_truth=int(cfg["benchmark"].get("min_fit_truth", 100)), max_positive_latents=int(cfg["benchmark"].get("max_positive_latents", 3)), allow_negative=bool(cfg["benchmark"].get("allow_negative", True)))
        perq, qmeta = _evaluate_queries(retrieval_model, xtest, df_test, ytest, rsa_ltest, linear_ltest, queries, cfg, int(seed)); perq["rsa_item_bytes"] = item_bytes; all_rows.append(perq); all_query_rows.append(qmeta)
    per_query = pd.concat(all_rows, ignore_index=True); query_meta = pd.concat(all_query_rows, ignore_index=True); predicate_metrics = pd.DataFrame(all_predicate_rows); summary, deltas = _aggregate(per_query, cfg)
    per_query.to_csv(root / "per_query.csv", index=False); query_meta.to_csv(root / "queries.csv", index=False); predicate_metrics.to_csv(root / "predicate_metrics.csv", index=False); summary.to_csv(root / "summary.csv", index=False); deltas.to_csv(root / "paired_deltas.csv", index=False); _plot_summary(summary, root / "figures")
    headline = {"run_id": run_id, "n_products": len(df), "n_seeds": len(cfg["benchmark"]["seeds"]), "queries_per_seed": int(cfg["benchmark"]["n_queries"]), "evaluated_rows": len(per_query), "predicate_mean_f1": predicate_metrics.groupby("method").f1.mean().to_dict(), "predicate_mean_ap": predicate_metrics.groupby("method").ap.mean().to_dict()}
    for frac in [0.4, 0.2, 0.1]:
        g = summary[(summary.retention == frac) & (summary.metric == "recall")]; headline[f"recall_at_{frac}"] = {r.method: r.mean for r in g.itertuples()}
    write_json(root / "headline.json", headline); print("Results:", root); print(json.dumps(headline, indent=2)); return root


def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/large_scale.yaml"); p.add_argument("--output-dir", default=None); args = p.parse_args(); run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
