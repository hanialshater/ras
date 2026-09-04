"""Memory accounting helpers used by the paper, demo, and benchmarks.

The numbers here are theoretical payload bytes for the learned item representation
and one independently deployable semantic predicate. They intentionally exclude
HNSW graph edges, generic allocator/container overhead, file-system metadata,
and model weights used only offline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodFootprint:
    key: str
    label: str
    item_bytes: int
    program_bytes: int

    def total_bytes(self, n_items: int, n_concepts: int) -> int:
        return int(n_items) * self.item_bytes + int(n_concepts) * self.program_bytes


METHOD_FOOTPRINTS: dict[str, MethodFootprint] = {
    "binary1_ls2_int4": MethodFootprint(
        "binary1_ls2_int4", "Binary1-LS2-int4", item_bytes=56, program_bytes=216
    ),
    "pq64_linear_lut": MethodFootprint(
        "pq64_linear_lut", "PQ64 compiled linear", item_bytes=64, program_bytes=65_548
    ),
    "rsa2_random": MethodFootprint(
        "rsa2_random", "RSA2 sparse LUT", item_bytes=96, program_bytes=580
    ),
    "rsa4_random": MethodFootprint(
        "rsa4_random", "RSA4 sparse LUT", item_bytes=192, program_bytes=3_652
    ),
    "linear_fp32": MethodFootprint(
        "linear_fp32", "FP32 linear", item_bytes=1_536, program_bytes=1_548
    ),
    "dense_minilm": MethodFootprint(
        "dense_minilm", "Dense MiniLM", item_bytes=1_536, program_bytes=0
    ),
}


def decimal_mb(n_bytes: int) -> float:
    """Convert bytes to decimal MB, matching the paper's storage arithmetic."""
    return float(n_bytes) / 1_000_000.0


def decimal_gb(n_bytes: int) -> float:
    """Convert bytes to decimal GB, matching the paper's storage arithmetic."""
    return float(n_bytes) / 1_000_000_000.0


def memory_rows(n_items: int, n_concepts: int) -> list[dict[str, float | int | str]]:
    """Return a presentation-friendly joint item/program memory table."""
    rows: list[dict[str, float | int | str]] = []
    for method in METHOD_FOOTPRINTS.values():
        item_total = int(n_items) * method.item_bytes
        program_total = int(n_concepts) * method.program_bytes
        total = item_total + program_total
        rows.append(
            {
                "method": method.label,
                "item_B": method.item_bytes,
                "predicate_B": method.program_bytes,
                "item_payload_MB": decimal_mb(item_total),
                "program_payload_MB": decimal_mb(program_total),
                "total_payload_MB": decimal_mb(total),
                "total_payload_GB": decimal_gb(total),
            }
        )
    return rows


__all__ = [
    "MethodFootprint",
    "METHOD_FOOTPRINTS",
    "decimal_mb",
    "decimal_gb",
    "memory_rows",
]
