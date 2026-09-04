"""Cheap smoke checks for the public Gradio demo.

These tests deliberately avoid model/dataset downloads. They exercise the query
parser and instantiate the UI with a synthetic state, which catches broken
imports and Gradio API drift in normal CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from demos.fashion_app import DemoState, build_app, parse_exact_filters, parse_latents


def _tiny_state() -> DemoState:
    # Keep >=100 rows because the real UI's ANN-pool slider starts at 100.
    n = 200
    df = pd.DataFrame(
        {
            "baseColour": (["Black", "Blue"] * (n // 2)),
            "gender": (["Women", "Men"] * (n // 2)),
            "masterCategory": (["Footwear", "Apparel"] * (n // 2)),
            "subCategory": (["Shoes", "Topwear"] * (n // 2)),
            "articleType": (["Casual Shoes", "Tshirts"] * (n // 2)),
            "productDisplayName": (["Black casual shoe", "Blue tee"] * (n // 2)),
        }
    )
    return DemoState(
        dataset=None,
        df_test=df,
        test_idx=np.arange(len(df), dtype=np.int64),
        x_test=np.zeros((len(df), 384), dtype=np.float32),
        y_test=np.zeros((len(df), 8), dtype=bool),
        retrieval_model=None,
        method_scores={},
        name_to_idx={},
        config={},
    )


def test_query_parser_handles_positive_negative_and_exact_filters() -> None:
    state = _tiny_state()
    pos, neg = parse_latents("minimalist black office shoes not sporty")
    assert "minimalist" in pos
    assert "office_appropriate" in pos
    assert "technical_sporty" in neg
    filters = parse_exact_filters("black women casual shoes", state.df_test)
    assert filters["baseColour"] == "Black"
    assert filters["gender"] == "Women"
    assert filters["articleType"] == "Casual Shoes"


def test_demo_ui_constructs_without_loading_models() -> None:
    # Instantiating the Blocks tree is enough to catch broken imports/component
    # signatures while keeping CI independent of Hugging Face/model downloads.
    app = build_app(_tiny_state())
    assert app is not None
