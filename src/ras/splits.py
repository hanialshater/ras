"""Deterministic train/calibration/test protocols."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split


@dataclass
class ProtocolSplit:
    fit_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray
    report_train_idx: np.ndarray


def make_protocol_split(n: int, seed: int = 7, strict: bool = False) -> ProtocolSplit:
    all_idx = np.arange(n)
    report_train_idx, test_idx = train_test_split(all_idx, test_size=0.35, random_state=seed)
    if strict:
        fit_idx, cal_idx = train_test_split(report_train_idx, test_size=0.20, random_state=seed + 101)
    else:
        fit_idx = report_train_idx.copy()
        cal_idx = report_train_idx.copy()
    return ProtocolSplit(np.asarray(fit_idx), np.asarray(cal_idx), np.asarray(test_idx), np.asarray(report_train_idx))
