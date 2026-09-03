"""Independent-teacher search pilot for Random Semantic Algebra.

Retrieval / item substrate: MiniLM title embeddings.
Latent semantic teacher: CLIP image semantics.
Online executor: 4-bit RSA code + sparse LUT programs.

This script reproduces the stricter search pilot described in the paper. It is
intentionally small enough for a single-GPU Colab-style run.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, CLIPModel

from src.rsa_v2 import (
    add_pair_interactions,
    build_substrate,
    fit_boosted_lut,
    fit_scalar_calibrator,
    log_sigmoid,
    make_protocol_split,
    metric_row,
    score_boosted,
)

SEED = 7
N_PRODUCTS = 8_000
ANN_POOL = 500
K_COORDS = 24
PAIR_LUTS = 2
CANDIDATE_POOL = 96
KEEP_SWEEP = (1.0, 0.40, 0.20, 0.10, 0.05)

rng = np.random.default_rng(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


LATENT_SPECS = [
    {
        "name": "minimalist",
        "pos": [
            "a minimalist understated fashion item",
            "a clean simple minimal shoe",
            "simple sleek understated design",
        ],
        "neg": [
            "a busy ornate decorative fashion item",
            "heavily embellished flashy design",
            "complex colorful overdesigned fashion item",
        ],
    },
    {
        "name": "office_appropriate",
        "pos": [
            "office appropriate professional fashion item",
            "formal polished shoe suitable for work",
            "smart business casual fashion item",
        ],
        "neg": [
            "casual beachwear party fashion item",
            "athletic sporty performance shoe",
            "very relaxed informal fashion item",
        ],
    },
    {
        "name": "technical_sporty",
        "pos": [
            "technical sporty athletic performance shoe",
            "running or training shoe",
            "performance sportswear style",
        ],
        "neg": [
            "classic non athletic fashion shoe",
            "formal lifestyle shoe",
            "fashion item not meant for sports",
        ],
    },
    {
        "name": "retro",
        "pos": [
            "retro vintage style fashion item",
            "old school nostalgic sneaker design",
            "vintage inspired fashion product",
        ],
        "neg": [
            "modern contemporary fashion item",
            "sleek current style design",
            "plain contemporary minimalist shoe",
        ],
    },
    {
        "name": "elegant",
        "pos": [
            "elegant refined fashion item",
            "dressy polished stylish shoe",
            "sophisticated graceful design",
        ],
        "neg": [
            "rugged sporty rough casual item",
            "clunky practical athletic design",
            "very relaxed sloppy style",
        ],
    },
    {
        "name": "relaxed",
        "pos": [
            "relaxed casual laid back fashion item",
            "easygoing casual comfort style",
            "informal everyday shoe",
        ],
        "neg": [
            "structured formal dressy fashion item",
            "strict professional shoe",
            "ceremonial formal style",
        ],
    },
    {
        "name": "chunky",
        "pos": [
            "chunky bulky thick soled shoe",
            "heavy oversized fashion silhouette",
            "thick robust sneaker design",
        ],
        "neg": [
            "sleek slim streamlined shoe",
            "light delicate narrow silhouette",
            "thin elegant shoe",
        ],
    },
    {
        "name": "quiet_luxury",
        "pos": [
            "quiet luxury understated premium fashion item",
            "subtle high end refined design",
            "premium understated leather shoe",
        ],
        "neg": [
            "loud flashy logo heavy fashion item",
            "cheap looking overly branded design",
            "very bright attention grabbing style",
        ],
    },
]


EXACT_TOKEN_MAP = {
    "black": ("baseColour", "Black"),
    "white": ("baseColour", "White"),
    "brown": ("baseColour", "Brown"),
    "blue": ("baseColour", "Blue"),
    "shoe": ("subCategory", "Shoes"),
    "shoes": ("subCategory", "Shoes"),
    "shirt": ("articleType", "Shirts"),
    "shirts": ("articleType", "Shirts"),
    "tshirt": ("articleType", "Tshirts"),
    "tshirts": ("articleType", "Tshirts"),
}

LATENT_ALIASES = {
    "minimalist": "minimalist",
    "minimal": "minimalist",
    "office": "office_appropriate",
    "office appropriate": "office_appropriate",
    "formal": "office_appropriate",
    "sporty": "technical_sporty",
    "technical": "technical_sporty",
    "retro": "retro",
    "elegant": "elegant",
    "relaxed": "relaxed",
    "casual": "relaxed",
    "chunky": "chunky",
    "quiet luxury": "quiet_luxury",
    "luxury": "quiet_luxury",
}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


@torch.no_grad()
def encode_clip_images(model, processor, images, batch_size=128):
    """Use CLIP's vision tower directly to avoid high-level API drift."""
    out = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        vision_out = model.vision_model(pixel_values=pixel_values)
        emb = model.visual_projection(vision_out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.cpu().numpy().astype(np.float32))
    return np.vstack(out)


@torch.no_grad()
def encode_clip_text(model, processor, texts):
    """Use CLIP's text tower directly to avoid high-level API drift."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    emb = model.text_projection(text_out.pooler_output)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


def mean_prompt(model, processor, prompts):
    v = encode_clip_text(model, processor, prompts).mean(axis=0, keepdims=True)
    return l2_normalize(v)[0]


def teacher_scores(image_embs, model, processor):
    scores = []
    for spec in LATENT_SPECS:
        pos = mean_prompt(model, processor, spec["pos"])
        neg = mean_prompt(model, processor, spec["neg"])
        scores.append((image_embs @ pos - image_embs @ neg).astype(np.float32))
    return np.column_stack(scores)


def labels_from_fit_threshold(scores, split, prevalence=0.40):
    fit, cal, test, rows = [], [], [], []
    for c, spec in enumerate(LATENT_SPECS):
        sfit = scores[split.fit_idx, c]
        threshold = float(np.quantile(sfit, 1.0 - prevalence))
        yf = scores[split.fit_idx, c] >= threshold
        yc = scores[split.cal_idx, c] >= threshold
        yt = scores[split.test_idx, c] >= threshold
        fit.append(yf)
        cal.append(yc)
        test.append(yt)
        rows.append(
            {
                "latent_concept": spec["name"],
                "fit_prevalence": float(yf.mean()),
                "cal_prevalence": float(yc.mean()),
                "test_prevalence": float(yt.mean()),
                "teacher_threshold": threshold,
            }
        )
    return np.column_stack(fit), np.column_stack(cal), np.column_stack(test), pd.DataFrame(rows)


def fast_best_f1_threshold(scores, truth, n_grid=160):
    scores = np.asarray(scores)
    y = np.asarray(truth, dtype=np.int8)
    thresholds = np.unique(np.quantile(scores, np.linspace(0.005, 0.995, n_grid)))
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_y = y[order]
    prefix = np.concatenate(([0], np.cumsum(sorted_y)))
    total = prefix[-1]
    idx = np.searchsorted(sorted_scores, thresholds, side="left")
    tp = total - prefix[idx]
    pred_pos = len(scores) - idx
    denom = pred_pos + total
    f1 = np.divide(2.0 * tp, denom, out=np.zeros(len(thresholds)), where=denom > 0)
    return float(thresholds[np.argmax(f1)])


def parse_query(query):
    q = " " + query.lower().strip() + " "
    positive, negative, exact, handled = [], [], [], set()
    for phrase, concept in LATENT_ALIASES.items():
        token = f" not {phrase} "
        if token in q:
            negative.append(concept)
            handled.add(phrase)
            q = q.replace(token, " ")
    for phrase, concept in LATENT_ALIASES.items():
        if phrase not in handled and f" {phrase} " in q:
            positive.append(concept)
    for token in q.replace(",", " ").split():
        if token in EXACT_TOKEN_MAP:
            exact.append(EXACT_TOKEN_MAP[token])
    return list(dict.fromkeys(positive)), list(dict.fromkeys(negative)), list(dict.fromkeys(exact))


def apply_exact(df, filters):
    keep = np.ones(len(df), dtype=bool)
    for field, value in filters:
        keep &= df[field].to_numpy() == value
    return keep


def semantic_program_scores(calibrated_logits, positive, negative, name_to_idx):
    score = np.zeros(len(calibrated_logits), dtype=np.float64)
    for name in positive:
        score += log_sigmoid(calibrated_logits[:, name_to_idx[name]])
    for name in negative:
        score += log_sigmoid(-calibrated_logits[:, name_to_idx[name]])
    return score


@dataclass
class State:
    text_model: SentenceTransformer
    x_test: np.ndarray
    df_test: pd.DataFrame
    y_test: np.ndarray
    calibrated_logits_test: np.ndarray
    name_to_idx: dict


def build_state() -> State:
    keep = [
        "id",
        "gender",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "usage",
        "productDisplayName",
        "image",
    ]
    ds = load_dataset("ashraq/fashion-product-images-small", split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    if len(ds) > N_PRODUCTS:
        ds = ds.select(rng.choice(len(ds), size=N_PRODUCTS, replace=False).tolist())

    # pandas for metadata only; pull decoded images from the Dataset object.
    df = ds.to_pandas()
    for c in keep:
        if c not in ("id", "image"):
            df[c] = df[c].fillna("Unknown").astype(str)

    text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    x = text_model.encode(
        df.productDisplayName.tolist(),
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    split = make_protocol_split(len(df), SEED, strict=True)
    xfit, xcal, xtest = x[split.fit_idx], x[split.cal_idx], x[split.test_idx]

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    images = [ds[i]["image"] for i in range(len(ds))]
    image_embs = encode_clip_images(clip, processor, images)
    t_scores = teacher_scores(image_embs, clip, processor)
    yfit, ycal, ytest, prevalence = labels_from_fit_threshold(t_scores, split)
    print("Teacher labels")
    print(prevalence.to_string(index=False))

    substrate = build_substrate(
        xfit,
        xcal,
        xtest,
        name="orth384_4bit_quantile",
        seed=SEED,
        bits=4,
        projection_kind="orthogonal",
        quantizer_kind="quantile",
    )

    cal_scores, test_scores, metric_rows, calibrators = [], [], [], []
    for c, spec in enumerate(LATENT_SPECS):
        unary = fit_boosted_lut(
            substrate.Q_fit,
            yfit[:, c],
            k=K_COORDS,
            n_bins=substrate.n_bins,
            candidate_pool=CANDIDATE_POOL,
            refine_passes=1,
        )
        model = add_pair_interactions(
            substrate.Q_fit,
            yfit[:, c],
            unary,
            n_pairs=PAIR_LUTS,
            pair_pool=12,
        )
        sc = score_boosted(substrate.Q_cal, model)
        st = score_boosted(substrate.Q_test, model)
        threshold = fast_best_f1_threshold(sc, ycal[:, c])
        m = metric_row(ytest[:, c], st, threshold)
        m.update({"latent_concept": spec["name"], "LUT_ops": K_COORDS + PAIR_LUTS})
        metric_rows.append(m)
        cal_scores.append(sc)
        test_scores.append(st)
        calibrators.append(fit_scalar_calibrator(sc, ycal[:, c]))

    board = pd.DataFrame(metric_rows).sort_values("f1", ascending=False)
    print("\nHeld-out latent predicate approximation")
    print(board.to_string(index=False))
    print("mean F1", board.f1.mean(), "mean AP", board.ap.mean())

    test_scores = np.column_stack(test_scores)
    calibrated_test = np.column_stack(
        [cal.transform(test_scores[:, c]) for c, cal in enumerate(calibrators)]
    )

    return State(
        text_model=text_model,
        x_test=xtest,
        df_test=df.iloc[split.test_idx].reset_index(drop=True),
        y_test=ytest,
        calibrated_logits_test=calibrated_test,
        name_to_idx={s["name"]: i for i, s in enumerate(LATENT_SPECS)},
    )


def evaluate_query(state: State, query, keep_frac=0.20, ann_pool=ANN_POOL):
    positive, negative, exact = parse_query(query)
    query_vec = state.text_model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)[0]
    dense = state.x_test @ query_vec
    ann = np.argsort(dense)[::-1][:ann_pool]

    pool_df = state.df_test.iloc[ann].reset_index(drop=True)
    pool_dense = dense[ann]
    pool_y = state.y_test[ann]
    pool_l = state.calibrated_logits_test[ann]

    mask = apply_exact(pool_df, exact)
    pool_df = pool_df.loc[mask].reset_index(drop=True)
    pool_dense = pool_dense[mask]
    pool_y = pool_y[mask]
    pool_l = pool_l[mask]

    truth = np.ones(len(pool_df), dtype=bool)
    for name in positive:
        truth &= pool_y[:, state.name_to_idx[name]]
    for name in negative:
        truth &= ~pool_y[:, state.name_to_idx[name]]

    k = max(1, int(round(len(pool_df) * keep_frac)))
    dense_sel = np.argsort(pool_dense)[::-1][:k]
    sem = semantic_program_scores(pool_l, positive, negative, state.name_to_idx)
    rsa_sel = np.argsort(sem)[::-1][:k]

    total = int(truth.sum())

    def stats(sel):
        hits = int(truth[sel].sum())
        return {
            "hits": hits,
            "recall": hits / total if total else np.nan,
            "purity": hits / len(sel) if len(sel) else np.nan,
        }

    return {
        "query": query,
        "positive": positive,
        "negative": negative,
        "exact": exact,
        "pool_size": len(pool_df),
        "pool_true": total,
        "keep_frac": keep_frac,
        "kept": k,
        "dense": stats(dense_sel),
        "rsa": stats(rsa_sel),
    }


def retention_sweep(state: State, query):
    rows = []
    for frac in KEEP_SWEEP:
        r = evaluate_query(state, query, keep_frac=frac)
        rows.append(
            {
                "keep_frac": frac,
                "dense_recall": r["dense"]["recall"],
                "rsa_recall": r["rsa"]["recall"],
                "dense_purity": r["dense"]["purity"],
                "rsa_purity": r["rsa"]["purity"],
                "dense_hits": r["dense"]["hits"],
                "rsa_hits": r["rsa"]["hits"],
                "pool_true": r["pool_true"],
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    plt.figure(figsize=(7, 5))
    plt.plot(df.keep_frac, df.dense_recall, marker="o", label="Dense-only recall")
    plt.plot(df.keep_frac, df.rsa_recall, marker="o", label="RSA recall")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Fraction of filtered ANN pool kept")
    plt.ylabel("Recall of teacher-defined relevant items")
    plt.title("Candidate retention sweep")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(df.keep_frac, df.dense_purity, marker="o", label="Dense-only purity")
    plt.plot(df.keep_frac, df.rsa_purity, marker="o", label="RSA purity")
    plt.xscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Fraction of filtered ANN pool kept")
    plt.ylabel("Purity of retained candidates")
    plt.title("Purity vs retained fraction")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return df


if __name__ == "__main__":
    print("device:", device)
    state = build_state()
    query = "minimalist black office shoes not sporty"
    print("\n20% candidate budget")
    print(evaluate_query(state, query, keep_frac=0.20))
    print("\nRetention sweep")
    retention_sweep(state, query)
