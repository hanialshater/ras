"""Random Semantic Algebra research package."""

from .substrate import QuantizedSubstrate, build_substrate, geometry_correlation
from .predicates import (
    BoostedLUTModel,
    LUTFactorModel,
    add_pair_interactions,
    compile_linear_teacher,
    fit_boosted_lut,
    learn_llr_factor,
    score_boosted,
    score_factor,
    train_linear_teachers,
)
from .calibration import ScalarCalibrator, fit_scalar_calibrator
from .composition import compose_logprob, compose_query
from .metrics import best_f1_threshold, metric_row

__all__ = [
    "QuantizedSubstrate", "build_substrate", "geometry_correlation",
    "BoostedLUTModel", "LUTFactorModel", "add_pair_interactions",
    "compile_linear_teacher", "fit_boosted_lut", "learn_llr_factor",
    "score_boosted", "score_factor", "train_linear_teachers",
    "ScalarCalibrator", "fit_scalar_calibrator", "compose_logprob",
    "compose_query", "best_f1_threshold", "metric_row",
]
