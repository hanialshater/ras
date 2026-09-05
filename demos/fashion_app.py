"""Interactive fashion demo for compiled semantic search.

The visual demo is intentionally separated from the Rust systems benchmark:

1. MiniLM title embeddings retrieve a broad held-out candidate pool.
2. Ordinary catalog metadata stays as exact filters.
3. Natural-language soft constraints are parsed into reusable semantic predicates.
4. The same held-out candidates are ranked side-by-side by Binary1-LS2-int4,
   PQ64, dense retrieval, FP32 semantic heads, and RSA2.
5. A systems panel reports the reviewer-controlled live-HNSW result and persistent
   item/program payloads. Those numbers are not the Python UI latency.

The visible catalog is the strict held-out test split, so the semantic heads are
not trained on the products shown in the app.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from experiments.binary_bbq_predicates import fit_bbq_like_linear, fit_rsa_random_bits
from experiments.large_scale_search import (
    _metadata_df,
    _retrieval_embeddings,
    _select_dataset,
    _teacher_embeddings_and_scores,
)
from experiments.reviewer_baselines import fit_linear_proxy, fit_pq64_linear
from ras.accounting import METHOD_FOOTPRINTS, memory_rows
from ras.binary import build_centered_binary_code
from ras.composition import compose_query
from ras.config import load_config
from ras.retrieval import encode_queries
from ras.splits import make_protocol_split
from ras.teachers import LATENT_SPECS, labels_from_fit_threshold


METHOD_META = {
    "bbq1_ls2_int4q": {
        "label": "Binary1-LS2-int4",
        "item_bytes": METHOD_FOOTPRINTS["binary1_ls2_int4"].item_bytes,
        "program_bytes": METHOD_FOOTPRINTS["binary1_ls2_int4"].program_bytes,
    },
    "pq64_linear_lut": {
        "label": "PQ64 compiled linear",
        "item_bytes": METHOD_FOOTPRINTS["pq64_linear_lut"].item_bytes,
        "program_bytes": METHOD_FOOTPRINTS["pq64_linear_lut"].program_bytes,
    },
    "rsa2_random": {
        "label": "RSA2 sparse LUT",
        "item_bytes": METHOD_FOOTPRINTS["rsa2_random"].item_bytes,
        "program_bytes": METHOD_FOOTPRINTS["rsa2_random"].program_bytes,
    },
    "linear_fp32": {
        "label": "FP32 linear",
        "item_bytes": METHOD_FOOTPRINTS["linear_fp32"].item_bytes,
        "program_bytes": METHOD_FOOTPRINTS["linear_fp32"].program_bytes,
    },
    "dense": {
        "label": "Dense MiniLM",
        "item_bytes": METHOD_FOOTPRINTS["dense_minilm"].item_bytes,
        "program_bytes": 0,
    },
}

# Reviewer-controlled HNSW benchmark. Values are means over three predetermined
# three-predicate sets, 1,000 queries per set, K=50 and EF=128. The over-fetch
# column is the largest completed selectivity-aware budget (~2 * K/selectivity).
# Traversal recall is relative to brute-force dense top-K under the SAME compiled
# predicate, not end-to-end semantic relevance.
SYSTEM_LATENCY_ROWS = [
    {"eligible": "50%", "live_ms": 2.226, "overfetch_2x_ms": 1.615, "live_traversal_recall@50": 0.9833, "overfetch_2x_traversal_recall@50": 0.7923},
    {"eligible": "20%", "live_ms": 4.580, "overfetch_2x_ms": 3.462, "live_traversal_recall@50": 0.9826, "overfetch_2x_traversal_recall@50": 0.7098},
    {"eligible": "10%", "live_ms": 6.791, "overfetch_2x_ms": 5.953, "live_traversal_recall@50": 0.9794, "overfetch_2x_traversal_recall@50": 0.7180},
    {"eligible": "5%", "live_ms": 9.820, "overfetch_2x_ms": 10.269, "live_traversal_recall@50": 0.9752, "overfetch_2x_traversal_recall@50": 0.7482},
    {"eligible": "2%", "live_ms": 14.568, "overfetch_2x_ms": 20.142, "live_traversal_recall@50": 0.9778, "overfetch_2x_traversal_recall@50": 0.8420},
]

LATENT_ALIASES = {
    "quiet_luxury": ["quiet luxury", "understated luxury", "subtle luxury", "premium understated"],
    "office_appropriate": ["office appropriate", "office", "workwear", "professional", "business casual", "business"],
    "technical_sporty": ["technical sporty", "sporty", "athletic", "running", "training", "performance"],
    "minimalist": ["minimalist", "minimal", "clean simple", "understated"],
    "retro": ["retro", "vintage", "old school", "old-school"],
    "elegant": ["elegant", "dressy", "refined", "sophisticated"],
    "relaxed": ["relaxed", "laid back", "laid-back", "easygoing"],
    "chunky": ["chunky", "bulky", "thick sole", "thick-soled", "oversized"],
}

GENDER_ALIASES = {
    "men": "Men", "mens": "Men", "male": "Men",
    "women": "Women", "womens": "Women", "female": "Women",
    "boys": "Boys", "girls": "Girls", "unisex": "Unisex",
}

CATEGORY_ALIASES = {
    "shoe": ("subCategory", "Shoes"),
    "shoes": ("subCategory", "Shoes"),
    "footwear": ("masterCategory", "Footwear"),
    "apparel": ("masterCategory", "Apparel"),
    "clothing": ("masterCategory", "Apparel"),
    "accessories": ("masterCategory", "Accessories"),
    "accessory": ("masterCategory", "Accessories"),
    "bags": ("subCategory", "Bags"),
    "bag": ("subCategory", "Bags"),
    "topwear": ("subCategory", "Topwear"),
    "bottomwear": ("subCategory", "Bottomwear"),
}


@dataclass
class DemoState:
    dataset: object
    df_test: pd.DataFrame
    test_idx: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    retrieval_model: object
    method_scores: Dict[str, np.ndarray]
    name_to_idx: Dict[str, int]
    config: dict


def _contains(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])", text) is not None


def parse_latents(query: str) -> tuple[list[str], list[str]]:
    text = query.lower().replace("’", "'")
    negative: list[str] = []
    consumed = text
    for name, aliases in LATENT_ALIASES.items():
        found = False
        for alias in aliases:
            for phrase in (f"not {alias}", f"without {alias}", f"non {alias}", f"non-{alias}"):
                if _contains(consumed, phrase):
                    negative.append(name)
                    consumed = re.sub(re.escape(phrase), " ", consumed, flags=re.IGNORECASE)
                    found = True
                    break
            if found:
                break

    positive: list[str] = []
    for name, aliases in LATENT_ALIASES.items():
        if name not in negative and any(_contains(consumed, alias) for alias in aliases):
            positive.append(name)
    return positive, negative


def parse_exact_filters(query: str, df: pd.DataFrame) -> dict[str, str]:
    text = query.lower().replace("'", "").replace("’", "")
    filters: dict[str, str] = {}

    if "baseColour" in df.columns:
        colors = sorted(
            [str(x) for x in df.baseColour.dropna().unique() if str(x) != "Unknown"],
            key=len,
            reverse=True,
        )
        for color in colors:
            if _contains(text, color.lower()):
                filters["baseColour"] = color
                break

    for alias, value in GENDER_ALIASES.items():
        if _contains(text, alias):
            filters["gender"] = value
            break

    for alias, (field, value) in sorted(CATEGORY_ALIASES.items(), key=lambda x: len(x[0]), reverse=True):
        if _contains(text, alias):
            if field in df.columns and value in set(df[field].astype(str)):
                filters[field] = value
            break

    if "articleType" in df.columns:
        vals = sorted(
            [str(x) for x in df.articleType.dropna().unique() if str(x) != "Unknown"],
            key=len,
            reverse=True,
        )
        for value in vals:
            phrase = value.lower().replace("-", " ")
            if len(phrase) >= 5 and _contains(text.replace("-", " "), phrase):
                filters["articleType"] = value
                break
    return filters


def _override_filter(filters: dict[str, str], field: str, value: str | None) -> None:
    if value and value != "Any":
        filters[field] = value


def prepare_demo(config_path: str = "configs/binary_bbq_smoke.yaml") -> DemoState:
    """Train finalist methods and return a strict held-out interactive catalog."""
    cfg = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = int(cfg["benchmark"]["seeds"][0])

    print("[demo] loading fashion catalog")
    ds, keep = _select_dataset(cfg)
    df = _metadata_df(ds, keep)
    print("[demo] MiniLM title embeddings")
    retrieval_model, x = _retrieval_embeddings(ds, df, cfg)
    print("[demo] CLIP image teacher (cached after first run)")
    teacher_scores = _teacher_embeddings_and_scores(ds, cfg, device)

    split = make_protocol_split(len(df), seed, strict=True)
    xfit, xcal, xtest = x[split.fit_idx], x[split.cal_idx], x[split.test_idx]
    yfit, ycal, ytest, _ = labels_from_fit_threshold(
        teacher_scores,
        split.fit_idx,
        split.cal_idx,
        split.test_idx,
        prevalence=float(cfg["teacher"].get("positive_prevalence", 0.40)),
    )

    print("[demo] fitting FP32 semantic predicates")
    linear, _, coefs, intercepts = fit_linear_proxy(xfit, xcal, xtest, yfit, ycal, ytest, seed)
    print("[demo] compiling PQ64 predicates")
    pq, _, _ = fit_pq64_linear(xfit, xcal, xtest, yfit, ycal, ytest, seed, coefs, intercepts, cfg)
    print("[demo] compiling Binary1-LS2-int4 predicates")
    binary_code = build_centered_binary_code(
        xfit, xcal, xtest, seed=seed, projection_kind="identity", with_corrections=True
    )
    binary, _, _ = fit_bbq_like_linear(
        binary_code, ycal, ytest, coefs, intercepts, seed, int4_query=True
    )
    print("[demo] fitting RSA2 sparse predicates")
    rsa2, _, _, _ = fit_rsa_random_bits(xfit, xcal, xtest, yfit, ycal, ytest, seed, cfg, 2)

    state = DemoState(
        dataset=ds,
        df_test=df.iloc[split.test_idx].reset_index(drop=True),
        test_idx=np.asarray(split.test_idx),
        x_test=np.asarray(xtest, dtype=np.float32),
        y_test=np.asarray(ytest, dtype=bool),
        retrieval_model=retrieval_model,
        method_scores={
            "bbq1_ls2_int4q": binary,
            "pq64_linear_lut": pq,
            "rsa2_random": rsa2,
            "linear_fp32": linear,
        },
        name_to_idx={s["name"]: i for i, s in enumerate(LATENT_SPECS)},
        config=cfg,
    )
    print(f"[demo] ready: {len(state.df_test):,} held-out products")
    return state


def _truth_mask(
    y: np.ndarray,
    name_to_idx: Mapping[str, int],
    positive: Iterable[str],
    negative: Iterable[str],
) -> np.ndarray:
    truth = np.ones(len(y), dtype=bool)
    used = False
    for name in positive:
        truth &= y[:, name_to_idx[name]]
        used = True
    for name in negative:
        truth &= ~y[:, name_to_idx[name]]
        used = True
    return truth if used else np.ones(len(y), dtype=bool)


def _caption(
    state: DemoState,
    item_local: int,
    score: float,
    is_truth: bool,
    show_truth: bool,
) -> tuple[object, str]:
    row = state.df_test.iloc[int(item_local)]
    original = int(state.test_idx[int(item_local)])
    image = state.dataset[original]["image"]
    title = str(row.get("productDisplayName", "product"))
    bits = [str(row.get("baseColour", "")), str(row.get("gender", "")), str(row.get("articleType", ""))]
    bits = [b for b in bits if b and b != "Unknown"]
    badge = " · teacher ✓" if show_truth and is_truth else (" · teacher ✗" if show_truth else "")
    return image, f"{title}\n{' · '.join(bits)}{badge}\nscore {score:.3f}"


def systems_memory_table(n_items: int = 5_000_000, n_concepts: int = 100_000) -> pd.DataFrame:
    keep = {"Binary1-LS2-int4", "PQ64 compiled linear", "RSA2 sparse LUT", "FP32 linear"}
    df = pd.DataFrame(memory_rows(n_items, n_concepts))
    df = df[df.method.isin(keep)].copy()
    return df[[
        "method",
        "item_B",
        "persistent_predicate_B",
        "active_predicate_B",
        "item_payload_MB",
        "program_payload_MB",
        "shared_payload_MB",
        "total_payload_MB",
    ]]


def systems_latency_table() -> pd.DataFrame:
    return pd.DataFrame(SYSTEM_LATENCY_ROWS)


def build_app(state: DemoState):
    import gradio as gr

    colors = ["Any"] + sorted([str(x) for x in state.df_test.baseColour.unique() if str(x) != "Unknown"])
    genders = ["Any"] + sorted([str(x) for x in state.df_test.gender.unique() if str(x) != "Unknown"])
    masters = ["Any"] + sorted([str(x) for x in state.df_test.masterCategory.unique() if str(x) != "Unknown"])

    def run_search(query, gender, color, master, ann_pool, top_k):
        t0 = time.perf_counter()
        query = (query or "").strip() or "minimalist black office shoes not sporty"

        positive, negative = parse_latents(query)
        filters = parse_exact_filters(query, state.df_test)
        _override_filter(filters, "gender", gender)
        _override_filter(filters, "baseColour", color)
        _override_filter(filters, "masterCategory", master)

        q = encode_queries(state.retrieval_model, [query])[0]
        dense_all = state.x_test @ q
        ann_pool = min(int(ann_pool), len(dense_all))
        ann = np.argsort(dense_all)[::-1][:ann_pool]

        mask = np.ones(len(ann), dtype=bool)
        pool_df = state.df_test.iloc[ann]
        for field, value in filters.items():
            if field in pool_df.columns:
                mask &= pool_df[field].astype(str).to_numpy() == str(value)
        pool = ann[mask]

        outputs_empty = [[] for _ in range(5)]
        if len(pool) == 0:
            return f"**No candidates after exact filters.** `{filters}`", pd.DataFrame(), *outputs_empty

        truth = _truth_mask(state.y_test[pool], state.name_to_idx, positive, negative)
        semantic_used = bool(positive or negative)
        top_k = max(1, min(int(top_k), len(pool)))

        method_scores: dict[str, np.ndarray] = {"dense": dense_all[pool]}
        for method, logits in state.method_scores.items():
            method_scores[method] = (
                compose_query(logits[pool], state.name_to_idx, positive, negative)
                if semantic_used
                else dense_all[pool]
            )

        methods = ["bbq1_ls2_int4q", "pq64_linear_lut", "dense", "linear_fp32", "rsa2_random"]
        galleries = []
        metric_rows = []
        for method in methods:
            scores = method_scores[method]
            order = np.argsort(scores)[::-1][:top_k]
            gallery = [
                _caption(state, int(pool[pos]), float(scores[pos]), bool(truth[pos]), semantic_used)
                for pos in order
            ]
            galleries.append(gallery)
            meta = METHOD_META[method]
            metric_rows.append(
                {
                    "method": meta["label"],
                    "teacher_hits@k": int(truth[order].sum()) if semantic_used else np.nan,
                    "precision@k": float(truth[order].mean()) if semantic_used else np.nan,
                    "item_B": meta["item_bytes"],
                    "stored_predicate_B": meta["program_bytes"],
                }
            )

        exact_txt = ", ".join(f"{k}={v}" for k, v in filters.items()) or "none"
        sem_parts = [f"+{x}" for x in positive] + [f"−{x}" for x in negative]
        sem_txt = " ∧ ".join(sem_parts) or "none"
        elapsed = (time.perf_counter() - t0) * 1000
        teacher_n = int(truth.sum()) if semantic_used else "n/a"
        status = (
            f"### Query plan\n"
            f"**Dense retrieval:** `{query}`  \n"
            f"**Exact filters:** `{exact_txt}`  \n"
            f"**Soft predicates:** `{sem_txt}`  \n"
            f"ANN pool **{ann_pool:,}** → exact-filtered **{len(pool):,}** · "
            f"teacher conjunction positives **{teacher_n}** · UI pipeline **{elapsed:.1f} ms**\n\n"
            f"_UI latency is Python demo latency, not the Rust HNSW systems measurement below._"
        )
        return status, pd.DataFrame(metric_rows), *galleries

    with gr.Blocks(title="Compiled Semantic Fashion Search") as demo:
        gr.Markdown(
            "# Compiled Semantic Fashion Search\n"
            "**Dense geometry finds the neighborhood. Exact fields stay exact. Tiny compiled programs enforce soft intent.**  \n"
            "Try constraints such as `minimalist`, `office`, `quiet luxury`, or `not sporty`. "
            "The products below are held out from semantic-predicate training."
        )

        with gr.Row():
            query = gr.Textbox(
                label="Natural-language search / LLM-style plan",
                value="minimalist black office shoes not sporty",
                placeholder="e.g. elegant women shoes not sporty",
                scale=4,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        with gr.Row():
            gender = gr.Dropdown(genders, value="Any", label="Gender")
            color = gr.Dropdown(colors, value="Any", label="Color")
            master = gr.Dropdown(masters, value="Any", label="Category")
            ann_pool = gr.Slider(100, min(1000, len(state.df_test)), value=min(500, len(state.df_test)), step=50, label="Dense candidate pool")
            top_k = gr.Slider(4, 20, value=12, step=1, label="Show top K")

        gr.Examples(
            examples=[
                ["minimalist black office shoes not sporty"],
                ["elegant women shoes not sporty"],
                ["retro casual shoes"],
                ["chunky sporty shoes"],
                ["quiet luxury black accessories"],
            ],
            inputs=[query],
        )

        status = gr.Markdown()
        metrics = gr.Dataframe(label="Held-out top-K quality + persistent payload", interactive=False, wrap=True)

        with gr.Accordion("Systems result: memory and reviewer-controlled HNSW", open=True):
            gr.Markdown(
                "**Memory scenario:** 5M items + 100k independently stored soft concepts; persistent and active program bytes are shown separately. "
                "Payload excludes HNSW graph edges and container overhead.  \n"
                "**Traversal scenario:** strict held-out graph, 1,000 queries per predicate set, three predetermined three-predicate sets, K=50, EF=128. "
                "The comparison shown is the largest completed selectivity-aware over-fetch point (~2× K/selectivity). "
                "No tested over-fetch point through 2× matched live Traversal Recall@50 within 0.005."
            )
            with gr.Row():
                gr.Dataframe(value=systems_memory_table(), label="Persistent item + concept payload", interactive=False, wrap=True)
                gr.Dataframe(value=systems_latency_table(), label="Live semantic HNSW vs 2× over-fetch", interactive=False, wrap=True)

        gr.Markdown("## Main semantic comparison")
        with gr.Row():
            binary_gallery = gr.Gallery(label="Binary1-LS2-int4 · 56 B/item · 216 B/predicate", columns=3, height=560, object_fit="contain")
            pq_gallery = gr.Gallery(label="PQ64 · 64 B/item · 1.55 KB stored head · 65.5 KB active LUT", columns=3, height=560, object_fit="contain")

        with gr.Accordion("More baselines", open=False):
            with gr.Row():
                dense_gallery = gr.Gallery(label="Dense MiniLM", columns=3, height=500, object_fit="contain")
                fp_gallery = gr.Gallery(label="FP32 linear semantic head", columns=3, height=500, object_fit="contain")
            rsa_gallery = gr.Gallery(label="RSA2 sparse LUT", columns=4, height=500, object_fit="contain")

        outputs = [status, metrics, binary_gallery, pq_gallery, dense_gallery, fp_gallery, rsa_gallery]
        inputs = [query, gender, color, master, ann_pool, top_k]
        search_btn.click(run_search, inputs=inputs, outputs=outputs)
        query.submit(run_search, inputs=inputs, outputs=outputs)
        demo.load(
            lambda: run_search(
                "minimalist black office shoes not sporty",
                "Any", "Any", "Any", min(500, len(state.df_test)), 12,
            ),
            outputs=outputs,
        )

    return demo


__all__ = [
    "DemoState",
    "prepare_demo",
    "build_app",
    "parse_latents",
    "parse_exact_filters",
    "systems_memory_table",
    "systems_latency_table",
]
