"""Universal low-bit semantic substrate."""
from .core import (
    QuantizedSubstrate,
    WhiteningState,
    apply_power_whitener,
    build_substrate,
    fit_power_whitener,
    geometry_correlation,
    independent_random_dictionary,
    random_orthogonal,
)

__all__ = [
    "QuantizedSubstrate", "WhiteningState", "apply_power_whitener",
    "build_substrate", "fit_power_whitener", "geometry_correlation",
    "independent_random_dictionary", "random_orthogonal",
]
