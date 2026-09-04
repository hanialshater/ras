"""Retrieval encoders and scoring helpers."""
from __future__ import annotations
from typing import Sequence
import numpy as np
from sentence_transformers import SentenceTransformer


def encode_titles(model_name: str, titles: Sequence[str], batch_size: int = 256) -> tuple[SentenceTransformer, np.ndarray]:
    model = SentenceTransformer(model_name)
    x = model.encode(list(titles), batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    return model, x


def encode_queries(model: SentenceTransformer, queries: Sequence[str], batch_size: int = 256) -> np.ndarray:
    return model.encode(list(queries), batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)


def dense_scores(item_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    return np.asarray(item_embeddings) @ np.asarray(query_embedding)


__all__ = ["encode_titles", "encode_queries", "dense_scores"]
