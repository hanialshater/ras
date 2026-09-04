"""Compatibility namespace for the refactored research package.

Paper experiments should import focused modules such as :mod:`ras.substrate`,
:mod:`ras.predicates`, and :mod:`ras.composition`. The exact historical v2
snapshot remains available as top-level module :mod:`rsa_v2` for reproducing
legacy tables, but new code does not depend on it.
"""
from .splits import ProtocolSplit, make_protocol_split

__all__ = ["ProtocolSplit", "make_protocol_split"]
