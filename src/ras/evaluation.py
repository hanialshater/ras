"""Aggregate conditional recall and all-query purity without dropping empty truth sets."""
from __future__ import annotations
import pandas as pd
from .metrics import bootstrap_mean_ci


def aggregate(per_query, cfg, reference="lut_random"):
    nboot = int(cfg["benchmark"].get("bootstrap_samples", 2000))
    rows, deltas = [], []
    for (method, retention), group in per_query.groupby(["method", "retention"]):
        for metric in ["recall", "purity", "recall_efficiency"]:
            eligible = group if metric == "purity" else group[group.total_true > 0]
            rows.append({
                "method": method, "retention": retention, "metric": metric,
                "queries_total": len(group),
                "queries_zero_truth": int((group.total_true == 0).sum()),
                **bootstrap_mean_ci(eligible[metric], seed=881, n_boot=nboot),
            })
    for metric in ["recall", "purity"]:
        eligible = per_query if metric == "purity" else per_query[per_query.total_true > 0]
        pivot = eligible.pivot_table(
            index=["seed", "query_id", "retention"], columns="method",
            values=metric, aggfunc="first",
        )
        for retention in sorted(eligible.retention.unique()):
            group = pivot.xs(retention, level="retention")
            for method in sorted(group.columns):
                for ref in dict.fromkeys(["dense", reference, "linear_fp32"]):
                    if method == ref or ref not in group:
                        continue
                    values = (group[method] - group[ref]).dropna()
                    deltas.append({
                        "method": method, "reference": ref, "retention": retention,
                        "metric": f"delta_{metric}",
                        **bootstrap_mean_ci(values, seed=882, n_boot=nboot),
                    })
    return pd.DataFrame(rows), pd.DataFrame(deltas)
