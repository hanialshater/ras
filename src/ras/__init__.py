"""Random Semantic Algebra research package."""

from .substrate import QuantizedSubstrate, build_substrate, geometry_correlation
from .predicates import (
    BoostedLUTModel,
    LUTFactorModel,
    add_pair_interactions,
    fit_boosted_lut,
    learn_llr_factor,
    score_boosted,
    score_factor,
)
from .calibration import ScalarCalibrator, fit_scalar_calibrator
from .composition import compose_logprob, compose_query
from .metrics import best_f1_threshold, metric_row
from .splits import ProtocolSplit, make_protocol_split
from .semantic_index import BinarySemanticIndex, SemanticIndexManifest
from .semantic_program import (
    BinarySemanticProgram,
    ProgramStore,
    compile_linear_program,
    fit_binary_predicate,
)
from .serving import PredicateRef, QueryResult, SemanticExecutor

__all__ = [
    "QuantizedSubstrate", "build_substrate", "geometry_correlation",
    "BoostedLUTModel", "LUTFactorModel", "add_pair_interactions",
    "fit_boosted_lut", "learn_llr_factor", "score_boosted", "score_factor",
    "ScalarCalibrator", "fit_scalar_calibrator", "compose_logprob", "compose_query",
    "best_f1_threshold", "metric_row", "ProtocolSplit", "make_protocol_split",
    "BinarySemanticIndex", "SemanticIndexManifest", "BinarySemanticProgram",
    "ProgramStore", "compile_linear_program", "fit_binary_predicate",
    "PredicateRef", "QueryResult", "SemanticExecutor",
]
