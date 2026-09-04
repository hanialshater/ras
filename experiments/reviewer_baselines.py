"""Reviewer-requested baselines for Random Semantic Algebra.

Adds the natural competitors missing from the first draft:
  * zero-shot MiniLM concept vectors (name and prompt-difference variants),
  * no-rotation 4-bit LUT programs,
  * PCA-rotation 4-bit LUT programs,
  * a small full-precision MLP ceiling,
  * a 64-byte Product Quantization (PQ64) representation with a compiled
    linear semantic head (64 LUT reads / concept),
  * the existing random-rotation RSA and FP32 linear proxy.

All supervised methods use strict fit/cal/test splits. Compound query templates
are generated from fit labels only. Search quality is evaluated inside the exact-
filtered ANN pool, matching the main paper protocol.
"""
from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import SGDClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.large_scale_search import _select_dataset, _metadata_df, _retrieval_embeddings, _teacher_embeddings_and_scores
from ras.calibration import fit_scalar_calibrator
from ras.composition import compose_query
from ras.config import load_config, config_hash
from ras.metrics import best_f1_threshold, metric_row, binary_ranking_stats, bootstrap_mean_ci
from ras.predicates import fit_boosted_lut, add_pair_interactions, score_boosted
from ras.queries import apply_exact, generate_query_benchmark
from ras.repro import environment_manifest, write_json
from ras.retrieval import encode_queries
from ras.splits import make_protocol_split
from ras.substrate import build_substrate
from ras.teachers import LATENT_SPECS, labels_from_fit_threshold


def _calibrate(cal_scores, test_scores, ycal):
    return np.column_stack([fit_scalar_calibrator(cal_scores[:, c], ycal[:, c]).transform(test_scores[:, c]) for c in range(ycal.shape[1])])


def fit_lut_variant(xfit, xcal, xtest, yfit, ycal, ytest, seed, cfg, variant):
    rc = cfg["rsa"]
    d = xfit.shape[1]
    if variant == "random":
        projection = None; projection_kind = "orthogonal"
    elif variant == "identity":
        projection = np.eye(d, dtype=np.float32); projection_kind = "identity"
    elif variant == "pca":
        xc = xfit - xfit.mean(axis=0, keepdims=True)
        cov = (xc.T @ xc) / max(len(xc) - 1, 1)
        _, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        projection = eigvecs[:, ::-1].astype(np.float32); projection_kind = "pca"
    else:
        raise ValueError(variant)
    substrate = build_substrate(xfit, xcal, xtest, name=f"{variant}_{d}_{rc['bits']}bit_{rc['quantizer']}", seed=int(seed), bits=int(rc["bits"]), projection_kind=projection_kind, quantizer_kind=str(rc["quantizer"]), projection=projection)
    cal_scores, test_scores, rows, selected = [], [], [], {}
    for c, spec in enumerate(LATENT_SPECS):
        unary = fit_boosted_lut(substrate.Q_fit, yfit[:, c], k=int(rc["k_coords"]), n_bins=substrate.n_bins, candidate_pool=int(rc["candidate_pool"]), refine_passes=1)
        model = add_pair_interactions(substrate.Q_fit, yfit[:, c], unary, n_pairs=int(rc.get("pair_luts", 0)), pair_pool=12)
        sc = score_boosted(substrate.Q_cal, model); st = score_boosted(substrate.Q_test, model)
        row = metric_row(ytest[:, c], st, best_f1_threshold(sc, ycal[:, c])); method = f"lut_{variant}"
        row.update({"concept": spec["name"], "method": method, "seed": int(seed)}); rows.append(row)
        cal_scores.append(sc); test_scores.append(st)
        selected[spec["name"]] = {"unary": [int(j) for j in model.unary_idx], "pairs": [[int(a), int(b)] for a, b in model.pair_idx]}
    cal_scores = np.column_stack(cal_scores); test_scores = np.column_stack(test_scores)
    return _calibrate(cal_scores, test_scores, ycal), pd.DataFrame(rows), selected, substrate.item_bytes_theoretical


def fit_linear_proxy(xfit, xcal, xtest, yfit, ycal, ytest, seed):
    cal_scores, test_scores, rows, coefs, intercepts = [], [], [], [], []
    for c, spec in enumerate(LATENT_SPECS):
        clf = SGDClassifier(loss="log_loss", class_weight="balanced", alpha=1e-4, max_iter=1500, random_state=int(seed) + c)
        clf.fit(xfit, yfit[:, c].astype(int))
        sc = clf.decision_function(xcal); st = clf.decision_function(xtest)
        row = metric_row(ytest[:, c], st, best_f1_threshold(sc, ycal[:, c])); row.update({"concept": spec["name"], "method": "linear_fp32", "seed": int(seed)})
        rows.append(row); cal_scores.append(sc); test_scores.append(st); coefs.append(clf.coef_[0].astype(np.float32)); intercepts.append(float(clf.intercept_[0]))
    cal_scores = np.column_stack(cal_scores); test_scores = np.column_stack(test_scores)
    return _calibrate(cal_scores, test_scores, ycal), pd.DataFrame(rows), np.stack(coefs), np.asarray(intercepts)


class TinyMLP(nn.Module):
    def __init__(self, d, hidden, out):
        super().__init__(); self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, out))
    def forward(self, x): return self.net(x)


def fit_mlp(xfit, xcal, xtest, yfit, ycal, ytest, seed, cfg, device):
    mc = cfg["mlp"]; torch.manual_seed(int(seed)); np.random.seed(int(seed))
    model = TinyMLP(xfit.shape[1], int(mc["hidden"]), yfit.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(mc["lr"]), weight_decay=float(mc["weight_decay"])); loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(xfit.astype(np.float32)), torch.from_numpy(yfit.astype(np.float32)))
    loader = DataLoader(ds, batch_size=int(mc["batch_size"]), shuffle=True, drop_last=False)
    xcal_t = torch.from_numpy(xcal.astype(np.float32)).to(device); ycal_t = torch.from_numpy(ycal.astype(np.float32)).to(device)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(int(mc["epochs"])):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device); opt.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val = float(loss_fn(model(xcal_t), ycal_t).item())
        if val < best - 1e-5:
            best = val; best_state = deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
            if bad >= int(mc["patience"]): break
    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    def pred(x):
        out = []; bs = int(mc.get("eval_batch_size", 4096))
        with torch.no_grad():
            for s in range(0, len(x), bs): out.append(model(torch.from_numpy(x[s:s+bs].astype(np.float32)).to(device)).cpu().numpy())
        return np.vstack(out)
    sc, st = pred(xcal), pred(xtest); rows = []
    for c, spec in enumerate(LATENT_SPECS):
        row = metric_row(ytest[:, c], st[:, c], best_f1_threshold(sc[:, c], ycal[:, c])); row.update({"concept": spec["name"], "method": "mlp_fp32_64", "seed": int(seed)}); rows.append(row)
    return _calibrate(sc, st, ycal), pd.DataFrame(rows)


def fit_pq64_linear(xfit, xcal, xtest, yfit, ycal, ytest, seed, coefs, intercepts, cfg):
    try:
        import faiss
    except ImportError as e:
        raise RuntimeError("PQ baseline requires faiss-cpu; install `pip install faiss-cpu`.") from e
    pc = cfg["pq"]; m = int(pc["m"]); nbits = int(pc["nbits"])
    if xfit.shape[1] % m != 0: raise ValueError("embedding dimension must be divisible by pq.m")
    pq = faiss.ProductQuantizer(xfit.shape[1], m, nbits); pq.train(np.ascontiguousarray(xfit.astype(np.float32)))
    qcal = pq.compute_codes(np.ascontiguousarray(xcal.astype(np.float32))); qtest = pq.compute_codes(np.ascontiguousarray(xtest.astype(np.float32)))
    ksub = 1 << nbits; dsub = xfit.shape[1] // m; centroids = faiss.vector_to_array(pq.centroids).reshape(m, ksub, dsub).astype(np.float32)
    scols, tcols, rows = [], [], []
    for c, spec in enumerate(LATENT_SPECS):
        w = coefs[c].reshape(m, dsub); lut = np.einsum("mkd,md->mk", centroids, w, optimize=True).astype(np.float32)
        sc = np.full(len(qcal), intercepts[c], dtype=np.float32); st = np.full(len(qtest), intercepts[c], dtype=np.float32)
        for j in range(m): sc += lut[j, qcal[:, j]]; st += lut[j, qtest[:, j]]
        row = metric_row(ytest[:, c], st, best_f1_threshold(sc, ycal[:, c])); row.update({"concept": spec["name"], "method": f"pq{m}_linear_lut", "seed": int(seed)}); rows.append(row); scols.append(sc); tcols.append(st)
    sc = np.column_stack(scols); st = np.column_stack(tcols)
    return _calibrate(sc, st, ycal), pd.DataFrame(rows), m * nbits / 8.0


def zero_shot_scores(model, xcal, xtest, ycal, ytest, seed):
    v_name = encode_queries(model, [s["query"] for s in LATENT_SPECS]); prompt_vecs = []
    for s in LATENT_SPECS:
        vp = encode_queries(model, s["pos"]).mean(axis=0); vn = encode_queries(model, s["neg"]).mean(axis=0); v = vp - vn; v /= max(np.linalg.norm(v), 1e-12); prompt_vecs.append(v.astype(np.float32))
    v_prompt = np.stack(prompt_vecs); methods = {}; rows = []
    for method, vec in [("zero_shot_name", v_name), ("zero_shot_prompt_diff", v_prompt)]:
        sc = xcal @ vec.T; st = xtest @ vec.T
        for c, spec in enumerate(LATENT_SPECS):
            row = metric_row(ytest[:, c], st[:, c], best_f1_threshold(sc[:, c], ycal[:, c])); row.update({"concept": spec["name"], "method": method, "seed": int(seed)}); rows.append(row)
        methods[method] = st.astype(np.float32)
    return methods, pd.DataFrame(rows)


def eval_queries(retrieval_model, xtest, df_test, ytest, method_scores, queries, cfg, seed):
    bc = cfg["benchmark"]; ann_pool = int(bc["ann_pool"]); retention = [float(x) for x in bc["retention"]]; name_to_idx = {s["name"]: i for i, s in enumerate(LATENT_SPECS)}
    qemb = encode_queries(retrieval_model, [q.text for q in queries]); rows = []
    for qi, query in enumerate(queries):
        dense = xtest @ qemb[qi]; ann = np.argsort(dense)[::-1][:ann_pool]; pool_df = df_test.iloc[ann].reset_index(drop=True); mask = apply_exact(pool_df, query.exact)
        if not mask.any(): continue
        ann = ann[mask]; pool_dense = dense[ann]; pool_y = ytest[ann]; truth = np.ones(len(ann), dtype=bool)
        for name in query.positive: truth &= pool_y[:, name_to_idx[name]]
        for name in query.negative: truth &= ~pool_y[:, name_to_idx[name]]
        scores = {"dense": pool_dense, "oracle": truth.astype(np.float32)}
        for method, full in method_scores.items():
            sub = full[ann]
            if method.startswith("zero_shot"):
                s = np.zeros(len(ann), dtype=np.float64)
                for name in query.positive: s += sub[:, name_to_idx[name]]
                for name in query.negative: s -= sub[:, name_to_idx[name]]
            else:
                s = compose_query(sub, name_to_idx, query.positive, query.negative)
            scores[method] = s
        for frac in retention:
            k = max(1, min(len(ann), int(round(len(ann) * frac))))
            for method, s in scores.items():
                order = np.argsort(s)[::-1][:k]
                rows.append({"seed": int(seed), "query_id": query.query_id, "query": query.text, "retention": frac, "method": method, "pool_size": len(ann), "n_latents": len(query.positive) + len(query.negative), **binary_ranking_stats(truth, order)})
    return pd.DataFrame(rows)


def aggregate(per_query, cfg, reference="lut_random"):
    nboot = int(cfg["benchmark"].get("bootstrap_samples", 2000)); rows, deltas = [], []; valid = per_query[per_query.total_true > 0]
    for (method, retention), g in valid.groupby(["method", "retention"]):
        for metric in ["recall", "purity", "recall_efficiency"]: rows.append({"method": method, "retention": retention, "metric": metric, **bootstrap_mean_ci(g[metric].to_numpy(), seed=881, n_boot=nboot)})
    pivot = valid.pivot_table(index=["seed","query_id","retention"], columns="method", values=["recall","purity"], aggfunc="first"); methods = sorted(valid.method.unique())
    for retention in sorted(valid.retention.unique()):
        sub = pivot.xs(retention, level="retention")
        for method in methods:
            for ref in ["dense", reference, "linear_fp32"]:
                if method == ref: continue
                for metric in ["recall", "purity"]:
                    if (metric, method) in sub.columns and (metric, ref) in sub.columns:
                        d = (sub[(metric, method)] - sub[(metric, ref)]).dropna().to_numpy(); deltas.append({"method": method, "reference": ref, "retention": retention, "metric": f"delta_{metric}", **bootstrap_mean_ci(d, seed=882, n_boot=nboot)})
    return pd.DataFrame(rows), pd.DataFrame(deltas)


def prevalence_depth(per_query):
    g = per_query[(per_query.method.isin(["dense", "lut_random"])) & (per_query.retention == 0.2)]
    p = g.pivot_table(index=["seed","query_id","n_latents","pool_size","total_true"], columns="method", values="recall", aggfunc="first").reset_index(); p = p.dropna(subset=["dense","lut_random"])
    p["delta_recall"] = p["lut_random"] - p["dense"]; p["prevalence"] = p.total_true / p.pool_size; p["prevalence_bin"] = pd.qcut(p.prevalence, 4, duplicates="drop")
    out = p.groupby(["prevalence_bin","n_latents"], observed=True).delta_recall.agg(["count","mean","std"]).reset_index(); return p, out


def plot_summary(summary, out):
    out.mkdir(parents=True, exist_ok=True); methods = ["dense","zero_shot_name","zero_shot_prompt_diff","lut_identity","lut_pca","lut_random","pq64_linear_lut","linear_fp32","mlp_fp32_64","oracle"]
    for metric in ["recall","purity"]:
        plt.figure(figsize=(9,6))
        for method in methods:
            g = summary[(summary.method == method) & (summary.metric == metric)].sort_values("retention")
            if len(g): plt.plot(g.retention, g["mean"], marker="o", label=method)
        plt.xscale("log"); plt.xlabel("Fraction of exact-filtered ANN pool kept"); plt.ylabel(metric.title()); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(out/f"reviewer_{metric}.png", dpi=180); plt.close()


def run(config_path, output_dir=None):
    cfg = load_config(config_path); device = "cuda" if torch.cuda.is_available() else "cpu"; run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_reviewer_" + config_hash(cfg)
    root = Path(output_dir or cfg["results_dir"]) / run_id; root.mkdir(parents=True, exist_ok=False); shutil.copy2(config_path, root/"config.yaml"); write_json(root/"environment.json", environment_manifest())
    ds, keep = _select_dataset(cfg); df = _metadata_df(ds, keep); retrieval_model, x = _retrieval_embeddings(ds, df, cfg); teacher_scores = _teacher_embeddings_and_scores(ds, cfg, device)
    all_q, all_pred, all_query_meta, selected_all = [], [], [], {}; meta_rows = [
        {"method":"dense","bytes_per_item":1536,"supervision":"none"},{"method":"zero_shot_name","bytes_per_item":1536,"supervision":"none"},{"method":"zero_shot_prompt_diff","bytes_per_item":1536,"supervision":"none"},{"method":"lut_identity","bytes_per_item":192,"supervision":"teacher"},{"method":"lut_pca","bytes_per_item":192,"supervision":"teacher"},{"method":"lut_random","bytes_per_item":192,"supervision":"teacher"},{"method":"pq64_linear_lut","bytes_per_item":64,"supervision":"teacher"},{"method":"linear_fp32","bytes_per_item":1536,"supervision":"teacher"},{"method":"mlp_fp32_64","bytes_per_item":1536,"supervision":"teacher"}]
    for seed in cfg["benchmark"]["seeds"]:
        split = make_protocol_split(len(df), int(seed), strict=True); xfit,xcal,xtest = x[split.fit_idx],x[split.cal_idx],x[split.test_idx]; df_fit=df.iloc[split.fit_idx].reset_index(drop=True); df_test=df.iloc[split.test_idx].reset_index(drop=True)
        yfit,ycal,ytest,_ = labels_from_fit_threshold(teacher_scores, split.fit_idx, split.cal_idx, split.test_idx, prevalence=float(cfg["teacher"].get("positive_prevalence",0.4))); method_scores = {}
        for variant in ["identity","pca","random"]:
            scores, rows, selected, _ = fit_lut_variant(xfit,xcal,xtest,yfit,ycal,ytest,int(seed),cfg,variant); method_scores[f"lut_{variant}"] = scores; all_pred.append(rows); selected_all[f"{seed}_{variant}"] = selected
        linear, rows, coefs, intercepts = fit_linear_proxy(xfit,xcal,xtest,yfit,ycal,ytest,int(seed)); method_scores["linear_fp32"] = linear; all_pred.append(rows)
        mlp, rows = fit_mlp(xfit,xcal,xtest,yfit,ycal,ytest,int(seed),cfg,device); method_scores["mlp_fp32_64"] = mlp; all_pred.append(rows)
        pq, rows, _ = fit_pq64_linear(xfit,xcal,xtest,yfit,ycal,ytest,int(seed),coefs,intercepts,cfg); method_scores["pq64_linear_lut"] = pq; all_pred.append(rows)
        zs, rows = zero_shot_scores(retrieval_model,xcal,xtest,ycal,ytest,int(seed)); method_scores.update(zs); all_pred.append(rows)
        queries = generate_query_benchmark(df_fit,yfit,n_queries=int(cfg["benchmark"]["n_queries"]),seed=int(seed)+404,min_fit_truth=int(cfg["benchmark"].get("min_fit_truth",100)),max_positive_latents=int(cfg["benchmark"].get("max_positive_latents",3)),allow_negative=bool(cfg["benchmark"].get("allow_negative",True))); all_query_meta.extend([{**q.to_dict(),"seed":int(seed)} for q in queries]); all_q.append(eval_queries(retrieval_model,xtest,df_test,ytest,method_scores,queries,cfg,int(seed)))
    perq=pd.concat(all_q,ignore_index=True); pred=pd.concat(all_pred,ignore_index=True); summary,deltas=aggregate(perq,cfg); depth_raw,depth_strat=prevalence_depth(perq)
    perq.to_csv(root/"per_query.csv",index=False); pred.to_csv(root/"predicate_metrics.csv",index=False); summary.to_csv(root/"summary.csv",index=False); deltas.to_csv(root/"paired_deltas.csv",index=False); pd.DataFrame(meta_rows).to_csv(root/"method_meta.csv",index=False); pd.DataFrame(all_query_meta).to_csv(root/"queries.csv",index=False); depth_raw.to_csv(root/"depth_prevalence_raw.csv",index=False); depth_strat.to_csv(root/"depth_prevalence_stratified.csv",index=False); write_json(root/"selected_coordinates.json",selected_all); plot_summary(summary,root/"figures")
    headline={"run_id":run_id,"n_products":len(df),"seeds":cfg["benchmark"]["seeds"],"queries_per_seed":int(cfg["benchmark"]["n_queries"]),"predicate_mean_f1":pred.groupby("method").f1.mean().to_dict(),"predicate_mean_ap":pred.groupby("method").ap.mean().to_dict()}
    for frac in [0.4,0.2,0.1]:
        g=summary[(summary.retention==frac)&(summary.metric=="recall")]; headline[f"recall_at_{frac}"]={r.method:float(r.mean) for r in g.itertuples()}
    write_json(root/"headline.json",headline); print(json.dumps(headline,indent=2)); print("Results:",root); return root


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/reviewer_baselines.yaml"); p.add_argument("--output-dir",default=None); a=p.parse_args(); run(a.config,a.output_dir)

if __name__ == "__main__": main()
