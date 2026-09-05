"""Memory accounting helpers used by the paper, demo, and benchmarks.

The default accounting is *persistent* semantic payload: item representation,
independently stored predicate programs, and any one-time shared representation
state. Some executors materialize larger transient state when a predicate is
activated; ``active_program_bytes`` records that separately.

All numbers intentionally exclude HNSW graph edges, generic allocator/container
overhead, file-system metadata, and model weights used only offline.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodFootprint:
    key: str
    label: str
    item_bytes: int
    # Persistently stored bytes per independently deployable predicate.
    program_bytes: int
    # Runtime bytes for one active predicate when different from persistent state.
    active_program_bytes: int | None = None
    # One-time representation state shared by all predicates (for example PQ codebook).
    shared_bytes: int = 0

    @property
    def active_bytes(self) -> int:
        return self.program_bytes if self.active_program_bytes is None else self.active_program_bytes

    def total_bytes(self, n_items: int, n_concepts: int) -> int:
        """Persistent item + concept store + one-time shared state."""
        return (
            int(n_items) * self.item_bytes
            + int(n_concepts) * self.program_bytes
            + self.shared_bytes
        )


# D=384. PQ64 uses m=64, 8-bit subcodes and dsub=6, so its shared float
# codebook is 64 * 256 * 6 * 4 = 393,216 bytes. The native PQ executor used in
# the paper materializes a 64 * 256 f32 LUT per active predicate plus scalar
# state (65,548 B), but that LUT does not need to be persisted for every concept.
PQ64_SHARED_CODEBOOK_BYTES = 64 * 256 * 6 * 4


METHOD_FOOTPRINTS: dict[str, MethodFootprint] = {
    "binary1_ls2_int4": MethodFootprint(
        "binary1_ls2_int4",
        "Binary1-LS2-int4",
        item_bytes=56,
        program_bytes=216,
    ),
    "pq64_linear_lut": MethodFootprint(
        "pq64_linear_lut",
        "PQ64 compiled linear",
        item_bytes=64,
        program_bytes=1_548,
        active_program_bytes=65_548,
        shared_bytes=PQ64_SHARED_CODEBOOK_BYTES,
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
    """Return a presentation-friendly *persistent* item/program memory table."""
    rows: list[dict[str, float | int | str]] = []
    for method in METHOD_FOOTPRINTS.values():
        item_total = int(n_items) * method.item_bytes
        program_total = int(n_concepts) * method.program_bytes
        shared_total = method.shared_bytes
        total = item_total + program_total + shared_total
        rows.append(
            {
                "method": method.label,
                "item_B": method.item_bytes,
                # Backward-compatible column name; this now means persistent bytes.
                "predicate_B": method.program_bytes,
                "persistent_predicate_B": method.program_bytes,
                "active_predicate_B": method.active_bytes,
                "shared_B": shared_total,
                "item_payload_MB": decimal_mb(item_total),
                "program_payload_MB": decimal_mb(program_total),
                "shared_payload_MB": decimal_mb(shared_total),
                "total_payload_MB": decimal_mb(total),
                "total_payload_GB": decimal_gb(total),
            }
        )
    return rows


__all__ = [
    "MethodFootprint",
    "METHOD_FOOTPRINTS",
    "PQ64_SHARED_CODEBOOK_BYTES",
    "decimal_mb",
    "decimal_gb",
    "memory_rows",
]


# A packed 384-D int4 PQ head plus offset, scale, intercept and two calibration
# scalars: 192 + 20 B. The active FP32 LUT has the same size as FP32-head PQ.
METHOD_FOOTPRINTS["pq64_int4_head"] = MethodFootprint(
    "pq64_int4_head", "PQ64 + int4 head", 64, 212, 65_548,
    PQ64_SHARED_CODEBOOK_BYTES,
)


def compiler_footprints(dim=384, pq_m=64, pq_bits=8, k_coords=24, pair_luts=2):
    """Shared payload definitions for experiment exporters; packed bytes only."""
    def sparse(bits):
        bins = 1 << bits
        return 4 * (k_coords * bins + pair_luts * bins * bins + 3) + 2 * k_coords + 4 * pair_luts
    packed = (dim + 7) // 8
    out = {
        "dense": MethodFootprint("dense", "Dense", dim * 4, 0),
        "zero_shot_name": MethodFootprint("zero_shot_name", "Zero-shot name", dim * 4, dim * 4),
        "zero_shot_prompt_diff": MethodFootprint("zero_shot_prompt_diff", "Zero-shot prompts", dim * 4, dim * 4),
        "linear_fp32": MethodFootprint("linear_fp32", "FP32 linear", dim * 4, dim * 4 + 12),
        "bbq1_ls2_f32q": MethodFootprint("bbq1_ls2_f32q", "Binary1 FP32 head", packed + 8, dim * 4 + 20),
        "bbq1_ls2_int4q": MethodFootprint("bbq1_ls2_int4q", "Binary1 int4 head", packed + 8, 4 * packed + 24),
    }
    for name, head_bytes in [("pq64_linear_lut", dim * 4 + 12), ("pq64_int4_head", (dim + 1) // 2 + 20)]:
        out[name] = MethodFootprint(
            name, name, (pq_m * pq_bits + 7) // 8, head_bytes,
            pq_m * (1 << pq_bits) * 4 + 12, (1 << pq_bits) * dim * 4,
        )
    for name, bits in [("rsa1_centered_identity", 1), ("rsa1_centered_random", 1),
                       ("rsa2_random", 2), ("rsa4_random", 4)]:
        out[name] = MethodFootprint(name, name, (dim * bits + 7) // 8, sparse(bits))
    return out


def deployment_memory_rows(n_items, n_concepts, dim=384):
    """Incremental memory when host retrieval already retains FP32 item vectors."""
    host = int(n_items) * dim * 4
    selected = compiler_footprints(dim)
    rows = []
    for key in ["linear_fp32", "bbq1_ls2_int4q", "pq64_linear_lut", "pq64_int4_head"]:
        method = selected[key]
        extra = method.total_bytes(n_items, n_concepts)
        if key == "linear_fp32":
            extra -= host
        rows.append({
            "method": key, "n_items": n_items, "n_concepts": n_concepts,
            "host_vectors_B": host, "incremental_persistent_B": extra,
            "combined_persistent_B": host + extra,
            "active_predicate_B": method.active_bytes,
        })
    # Learned logits are precomputed on items. Concept updates require a catalog scan.
    extra = int(n_items) * int(n_concepts) * 4
    rows.append({
        "method": "materialized_fp32_logits", "n_items": n_items, "n_concepts": n_concepts,
        "host_vectors_B": host, "incremental_persistent_B": extra,
        "combined_persistent_B": host + extra, "active_predicate_B": 0,
    })
    return rows
