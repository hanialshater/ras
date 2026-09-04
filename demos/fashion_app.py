"""Interactive fashion search demo for compact semantic predicate execution.

The demo is deliberately faithful to the paper protocol:

1. MiniLM title embeddings retrieve a broad candidate pool.
2. Ordinary catalog metadata applies exact filters.
3. Latent predicates are parsed from the query and composed independently.
4. The same held-out candidates are ranked side-by-side by dense retrieval,
   FP32 linear predicates, PQ64 compiled predicates, BBQ-inspired 1-bit docs
   with int4 predicate weights, and sparse RSA2 programs.

The visible catalog is the held-out test split, so the semantic heads are not
trained on the products shown in the app.
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
from ras.binary import build_centered_binary_code
from ras.composition import compose_query
from ras.config import load_config
from ras.retrieval import encode_queries
from ras.splits import make_protocol_split
from ras.teachers import LATENT_SPECS, labels_from_fit_threshold


METHOD_META = {
    "bbq1_ls2_int4q": {"label": "BBQ1 + int4", "item_bytes": 56, "program_bytes": 216},
    "pq64_linear_lut": {"label": "PQ64", "item_bytes": 64, "program_bytes": 65548},
    "rsa2_random": {"label": "RSA2", "item_bytes": 96, "program_bytes": 580},
    "linear_fp32": {"label": "FP32 linear", "item_bytes": 1536, "program_bytes": 1548},
    "dense": {"label": "Dense MiniLM", "item_bytes": 1536, "program_bytes": 0},
}

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
    "men": "Men",
    "mens": "Men",
    "male": "Men",
    "women": "Women",
    "womens": "Women",
    "female": "Women",
    "boys": "Boys",
    "girls": "Girls",
    "unisex": "Unisex",
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

    # Find negations first so "not sporty" never becomes both positive and negative.
    for name, aliases in LATENT_ALIASES.items():
        found = False
        for alias in aliases:
            patterns = [f"not {alias}", f"without {alias}", f"non {alias}", f"non-{alias}"]
            for phrase in patterns:
                if _contains(consumed, phrase):
                    negative.append(name)
                    consumed = re.sub(re.escape(phrase), " ", consumed, flags=re.IGNORECASE)
                    found = True
                    break
            if found:
                break

    positive: list[str] = []
    for name, aliases in LATENT_ALIASES.items():
        if name in negative:
            continue
        if any(_contains(consumed, alias) for alias in aliases):
            positive.append(name)
    return positive, negative


def parse_exact_filters(query: str, df: pd.DataFrame) -> dict[str, str]:
    text = query.lower().replace("'", "").replace("’", "")
    filters: dict[str, str] = {}

    # Catalog colors are exact filters. Prefer longer names such as "off white".
    if "baseColour" in df.columns:
        colors = sorted([str(x) for x in df.baseColour.dropna().unique() if str(x) != "Unknown"], key=len, reverse=True)
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

    # Long article-type names can be typed literally ("casual shoes", "sports shoes", ...).
    if "articleType" in df.columns:
        vals = sorted([str(x) for x in df.articleType.dropna().unique() if str(x) != "Unknown"], key=len, reverse=True)
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
    """Train the compact finalist methods and return a held-out interactive catalog."""
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

    print("[demo] compiling PQ64 semantic predicates")
    pq, _, _ = fit_pq64_linear(xfit, xcal, xtest, yfit, ycal, ytest, seed, coefs, intercepts, cfg)

    print("[demo] compiling BBQ-inspired 1-bit docs + int4 predicates")
    bbq_code = build_centered_binary_code(
        xfit, xcal, xtest, seed=seed, projection_kind="identity", with_corrections=True
    )
    bbq, _, _ = fit_bbq_like_linear(
        bbq_code, ycal, ytest, coefs, intercepts, seed, int4_query=True
    )

    print("[demo] fitting sparse RSA2 predicates")
    rsa2, _, _, _ = fit_rsa_random_bits(xfit, xcal, xtest, yfit, ycal, ytest, seed, cfg, 2)

    state = DemoState(
        dataset=ds,
        df_test=df.iloc[split.test_idx].reset_index(drop=True),
        test_idx=np.asarray(split.test_idx),
        x_test=np.asarray(xtest, dtype=np.float32),
        y_test=np.asarray(ytest, dtype=bool),
        retrieval_model=retrieval_model,
        method_scores={
            "bbq1_ls2_int4q": bbq,
            "pq64_linear_lut": pq,
            "rsa2_random": rsa2,
            "linear_fp32": linear,
        },
        name_to_idx={s["name"]: i for i, s in enumerate(LATENT_SPECS)},
        config=cfg,
    )
    print(f"[demo] ready: {len(state.df_test):,} held-out products")
    return state


def _truth_mask(y: np.ndarray, name_to_idx: Mapping[str, int], positive: Iterable[str], negative: Iterable[str]) -> np.ndarray:
    truth = np.ones(len(y), dtype=bool)
    used = False
    for name in positive:
        truth &= y[:, name_to_idx[name]]
        used = True
    for name in negative:
        truth &= ~y[:, name_to_idx[name]]
        used = True
    return truth if used else np.ones(len(y), dtype=bool)


def _caption(state: DemoState, item_local: int, score: float, is_truth: bool, show_truth: bool) -> tuple[object, str]:
    row = state.df_test.iloc[int(item_local)]
    original = int(state.test_idx[int(item_local)])
    image = state.dataset[original]["image"]
    title = str(row.get("productDisplayName", "product"))
    bits = [str(row.get("baseColour", "")), str(row.get("gender", "")), str(row.get("articleType", ""))]
    bits = [b for b in bits if b and b != "Unknown"]
    badge = " · teacher ✓" if show_truth and is_truth else (" · teacher ✗" if show_truth else "")
    return image, f"{title}\n{' · '.join(bits)}{badge}\nscore {score:.3f}"


def build_app(state: DemoState):
    import gradio as gr

    colors = ["Any"] + sorted([str(x) for x in state.df_test.baseColour.unique() if str(x) != "Unknown"])
    genders = ["Any"] + sorted([str(x) for x in state.df_test.gender.unique() if str(x) != "Unknown"])
    masters = ["Any"] + sorted([str(x) for x in state.df_test.masterCategory.unique() if str(x) != "Unknown"])

    def run_search(query, gender, color, master, ann_pool, top_k):
        t0 = time.perf_counter()
        query = (query or "").strip()
        if not query:
            query = "minimalist black office shoes not sporty"

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
            status = f"**No candidates after filters.** Parsed exact filters: `{filters}`"
            return status, pd.DataFrame(), *outputs_empty

        truth = _truth_mask(state.y_test[pool], state.name_to_idx, positive, negative)
        semantic_used = bool(positive or negative)
        top_k = max(1, min(int(top_k), len(pool)))

        method_scores: dict[str, np.ndarray] = {"dense": dense_all[pool]}
        for method, logits in state.method_scores.items():
            if semantic_used:
                method_scores[method] = compose_query(logits[pool], state.name_to_idx, positive, negative)
            else:
                method_scores[method] = dense_all[pool]

        methods = ["bbq1_ls2_int4q", "pq64_linear_lut", "dense", "linear_fp32", "rsa2_random"]
        galleries = []
        metric_rows = []
        for method in methods:
            scores = method_scores[method]
            order = np.argsort(scores)[::-1][:top_k]
            show_truth = semantic_used
            gallery = [
                _caption(state, int(pool[pos]), float(scores[pos]), bool(truth[pos]), show_truth)
                for pos in order
            ]
            galleries.append(gallery)
            hits = int(truth[order].sum()) if semantic_used else np.nan
            precision = float(truth[order].mean()) if semantic_used else np.nan
            meta = METHOD_META[method]
            metric_rows.append(
                {
                    "method": meta["label"],
                    "teacher_hits@k": hits,
                    "precision@k": precision,
                    "item_B": meta["item_bytes"],
                    "predicate_B": meta["program_bytes"],
                }
            )

        exact_txt = ", ".join(f"{k}={v}" for k, v in filters.items()) or "none"
        sem_parts = [f"+{x}" for x in positive] + [f"−{x}" for x in negative]
        sem_txt = " ∧ ".join(sem_parts) or "none parsed (semantic methods fall back to dense order)"
        elapsed = (time.perf_counter() - t0) * 1000
        teacher_n = int(truth.sum()) if semantic_used else "n/a"
        status = (
            f"**Parsed query**  \n"
            f"Exact: `{exact_txt}`  \n"
            f"Latent: `{sem_txt}`  \n"
            f"ANN pool: **{ann_pool:,}** → exact-filtered: **{len(pool):,}** · "
            f"teacher conjunction positives: **{teacher_n}** · Python demo latency: **{elapsed:.1f} ms**"
        )
        return status, pd.DataFrame(metric_rows), *galleries

    with gr.Blocks(title="Compact Semantic Fashion Search") as demo:
        gr.Markdown(
            "# Compact Semantic Fashion Search\n"
            "Real fashion images, held-out products, and the same semantic predicates used in the experiments. "
            "Dense MiniLM retrieves broadly; exact catalog filters stay exact; then compact semantic programs "
            "execute constraints such as **minimalist**, **office**, and **not sporty**."
        )
        with gr.Row():
            query = gr.Textbox(
                label="Search",
                value="minimalist black office shoes not sporty",
                placeholder="e.g. elegant women shoes not sporty",
                scale=4,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        with gr.Row():
            gender = gr.Dropdown(genders, value="Any", label="Gender")
            color = gr.Dropdown(colors, value="Any", label="Color")
            master = gr.Dropdown(masters, value="Any", label="Category")
            ann_pool = gr.Slider(100, min(1000, len(state.df_test)), value=min(500, len(state.df_test)), step=50, label="ANN candidates")
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
        metrics = gr.Dataframe(label="Top-K comparison", interactive=False, wrap=True)

        gr.Markdown("## Main comparison")
        with gr.Row():
            bbq_gallery = gr.Gallery(label="BBQ1 + int4 · 56 B/item · 216 B/predicate", columns=3, height=560, object_fit="contain")
            pq_gallery = gr.Gallery(label="PQ64 · 64 B/item · ~65 KB/predicate", columns=3, height=560, object_fit="contain")

        with gr.Accordion("More baselines", open=False):
            with gr.Row():
                dense_gallery = gr.Gallery(label="Dense MiniLM", columns=3, height=500, object_fit="contain")
                fp_gallery = gr.Gallery(label="FP32 linear semantic head", columns=3, height=500, object_fit="contain")
            rsa_gallery = gr.Gallery(label="RSA2 sparse LUT", columns=4, height=500, object_fit="contain")

        outputs = [status, metrics, bbq_gallery, pq_gallery, dense_gallery, fp_gallery, rsa_gallery]
        inputs = [query, gender, color, master, ann_pool, top_k]
        search_btn.click(run_search, inputs=inputs, outputs=outputs)
        query.submit(run_search, inputs=inputs, outputs=outputs)
        demo.load(
            lambda: run_search("minimalist black office shoes not sporty", "Any", "Any", "Any", min(500, len(state.df_test)), 12),
            outputs=outputs,
        )

    return demo


__all__ = ["DemoState", "prepare_demo", "build_app", "parse_latents", "parse_exact_filters"]
