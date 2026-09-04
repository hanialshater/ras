"""Independent semantic teachers used for latent predicate supervision."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch

LATENT_SPECS = [
    {"name": "minimalist", "query": "minimalist", "pos": ["a minimalist understated fashion item", "a clean simple minimal shoe", "simple sleek understated design"], "neg": ["a busy ornate decorative fashion item", "heavily embellished flashy design", "complex colorful overdesigned fashion item"]},
    {"name": "office_appropriate", "query": "office", "pos": ["office appropriate professional fashion item", "formal polished shoe suitable for work", "smart business casual fashion item"], "neg": ["casual beachwear party fashion item", "athletic sporty performance shoe", "very relaxed informal fashion item"]},
    {"name": "technical_sporty", "query": "sporty", "pos": ["technical sporty athletic performance shoe", "running or training shoe", "performance sportswear style"], "neg": ["classic non athletic fashion shoe", "formal lifestyle shoe", "fashion item not meant for sports"]},
    {"name": "retro", "query": "retro", "pos": ["retro vintage style fashion item", "old school nostalgic sneaker design", "vintage inspired fashion product"], "neg": ["modern contemporary fashion item", "sleek current style design", "plain contemporary minimalist shoe"]},
    {"name": "elegant", "query": "elegant", "pos": ["elegant refined fashion item", "dressy polished stylish shoe", "sophisticated graceful design"], "neg": ["rugged sporty rough casual item", "clunky practical athletic design", "very relaxed sloppy style"]},
    {"name": "relaxed", "query": "relaxed", "pos": ["relaxed casual laid back fashion item", "easygoing casual comfort style", "informal everyday shoe"], "neg": ["structured formal dressy fashion item", "strict professional shoe", "ceremonial formal style"]},
    {"name": "chunky", "query": "chunky", "pos": ["chunky bulky thick soled shoe", "heavy oversized fashion silhouette", "thick robust sneaker design"], "neg": ["sleek slim streamlined shoe", "light delicate narrow silhouette", "thin elegant shoe"]},
    {"name": "quiet_luxury", "query": "quiet luxury", "pos": ["quiet luxury understated premium fashion item", "subtle high end refined design", "premium understated leather shoe"], "neg": ["loud flashy logo heavy fashion item", "cheap looking overly branded design", "very bright attention grabbing style"]},
]


def _norm_np(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


@torch.no_grad()
def encode_clip_images_dataset(model, processor, dataset, device: str, batch_size: int = 128) -> np.ndarray:
    """Encode a Hugging Face Dataset image column without materializing all PIL images."""
    out = []
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        images = dataset[start:stop]["image"]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        vision_out = model.vision_model(pixel_values=pixel_values)
        emb = model.visual_projection(vision_out.pooler_output)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.cpu().numpy().astype(np.float32))
        if start == 0 or (start // batch_size) % 25 == 0:
            print(f"CLIP images {stop}/{len(dataset)}")
    return np.vstack(out)


@torch.no_grad()
def encode_clip_text(model, processor, texts: Sequence[str], device: str) -> np.ndarray:
    inputs = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    emb = model.text_projection(text_out.pooler_output)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


def teacher_score_matrix(image_embeddings: np.ndarray, model, processor, device: str, specs=LATENT_SPECS) -> np.ndarray:
    cols = []
    for spec in specs:
        p = _norm_np(encode_clip_text(model, processor, spec["pos"], device).mean(axis=0, keepdims=True))[0]
        n = _norm_np(encode_clip_text(model, processor, spec["neg"], device).mean(axis=0, keepdims=True))[0]
        cols.append((image_embeddings @ p - image_embeddings @ n).astype(np.float32))
    return np.column_stack(cols)


def labels_from_fit_threshold(scores: np.ndarray, fit_idx: np.ndarray, cal_idx: np.ndarray, test_idx: np.ndarray, prevalence: float = 0.40, specs=LATENT_SPECS):
    fit, cal, test, rows = [], [], [], []
    for c, spec in enumerate(specs):
        threshold = float(np.quantile(scores[fit_idx, c], 1.0 - prevalence))
        yf = scores[fit_idx, c] >= threshold
        yc = scores[cal_idx, c] >= threshold
        yt = scores[test_idx, c] >= threshold
        fit.append(yf); cal.append(yc); test.append(yt)
        rows.append({
            "latent_concept": spec["name"],
            "threshold": threshold,
            "fit_prevalence": float(yf.mean()),
            "cal_prevalence": float(yc.mean()),
            "test_prevalence": float(yt.mean()),
        })
    return np.column_stack(fit), np.column_stack(cal), np.column_stack(test), rows


__all__ = ["LATENT_SPECS", "encode_clip_images_dataset", "encode_clip_text", "teacher_score_matrix", "labels_from_fit_threshold"]
