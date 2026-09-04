"""Compatibility bridge to the exact RSA v2 research snapshot.

New experiments should import from the focused modules in :mod:`ras` rather than
from this module directly. The legacy `rsa_v2.py` snapshot is kept byte-exact so
published numbers remain reproducible while the public API is refactored.
"""
from rsa_v2 import *  # noqa: F401,F403
