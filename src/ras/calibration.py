"""Score calibration utilities."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from .numerics import logit


@dataclass
class ScalarCalibrator:
    a: float
    b: float

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return self.a * np.asarray(scores) + self.b


def fit_scalar_calibrator(scores: np.ndarray, y: np.ndarray) -> ScalarCalibrator:
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2 or np.std(scores) < 1e-12:
        p = (y.sum() + 0.5) / (len(y) + 1.0)
        return ScalarCalibrator(0.0, logit(float(p)))
    lr = LogisticRegression(C=1000.0, solver="lbfgs", max_iter=1000)
    lr.fit(np.asarray(scores).reshape(-1, 1), y)
    return ScalarCalibrator(float(lr.coef_[0, 0]), float(lr.intercept_[0]))


def calibrate_score_matrices(scores_cal, scores_test, y_cal):
    lc, lt, cals = [], [], []
    for c in range(y_cal.shape[1]):
        cal = fit_scalar_calibrator(scores_cal[:, c], y_cal[:, c])
        cals.append(cal)
        lc.append(cal.transform(scores_cal[:, c]))
        lt.append(cal.transform(scores_test[:, c]))
    return np.column_stack(lc), np.column_stack(lt), cals
