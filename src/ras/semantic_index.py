"""Portable semantic sidecar index for bring-your-own embeddings.

The index intentionally does not perform ANN retrieval.  A host search engine
keeps ownership of retrieval and exact filters, then passes candidate row IDs to
the semantic executor.  Adding a new semantic predicate never re-encodes items.

Directory format (version 1)::

    index/
      manifest.json
      bits.u8              # packed sign bits, row-major
      corrections.f32      # two LS2 values per item
      centroid.f32
      projection.f32       # omitted for identity projection
      item_ids.npy         # optional external IDs

For D=384 with corrections the serving code is 48 bytes of sign bits plus two
f32 correction values = 56 bytes/item.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal, Sequence

import numpy as np

from .binary import (
    CenteredBinaryEncoder,
    encode_centered_binary,
    fit_centered_binary_encoder,
    pack_document_bits,
)


FORMAT_VERSION = 1


@dataclass(frozen=True)
class SemanticIndexManifest:
    version: int
    n_items: int
    dim: int
    packed_bytes: int
    correction_values: int
    projection_kind: str
    seed: int
    item_bytes_theoretical: float
    has_item_ids: bool


@dataclass
class BinarySemanticIndex:
    """Loaded binary semantic index.

    ``bits`` and ``corrections`` may be NumPy memmaps, so loading a large index
    does not require copying the whole catalog into process-private memory.
    Candidate IDs are zero-based row ordinals supplied by the host retrieval
    system.  Optional external IDs are metadata only.
    """

    path: Path
    manifest: SemanticIndexManifest
    encoder: CenteredBinaryEncoder
    bits: np.ndarray
    corrections: np.ndarray
    item_ids: np.ndarray | None = None

    @property
    def n_items(self) -> int:
        return self.manifest.n_items

    @property
    def dim(self) -> int:
        return self.manifest.dim

    @classmethod
    def build(
        cls,
        path: str | Path,
        embeddings: np.ndarray,
        *,
        fit_embeddings: np.ndarray | None = None,
        item_ids: Sequence | np.ndarray | None = None,
        seed: int = 7,
        projection_kind: Literal["identity", "orthogonal"] = "identity",
        overwrite: bool = False,
    ) -> "BinarySemanticIndex":
        """Build a concept-independent semantic sidecar from arbitrary embeddings.

        ``fit_embeddings`` may be a representative sample used only to fit the
        centroid/projection.  When omitted, the full supplied catalog is used.
        The strongest current paper result uses ``projection_kind='identity'``.
        """
        path = Path(path)
        if path.exists() and any(path.iterdir()) and not overwrite:
            raise FileExistsError(f"index directory is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)

        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or len(x) == 0:
            raise ValueError("embeddings must be a non-empty [items, dimensions] array")
        fit_x = x if fit_embeddings is None else np.asarray(fit_embeddings, dtype=np.float32)
        if fit_x.ndim != 2 or fit_x.shape[1] != x.shape[1]:
            raise ValueError("fit_embeddings must have the same embedding dimension")

        encoder = fit_centered_binary_encoder(
            fit_x,
            seed=int(seed),
            projection_kind=projection_kind,
            with_corrections=True,
        )
        q, corrections = encode_centered_binary(x, encoder)
        packed = np.ascontiguousarray(pack_document_bits(q), dtype=np.uint8)
        corrections = np.ascontiguousarray(corrections, dtype=np.float32)

        packed.tofile(path / "bits.u8")
        corrections.tofile(path / "corrections.f32")
        np.ascontiguousarray(encoder.centroid, dtype=np.float32).tofile(path / "centroid.f32")
        if encoder.projection_kind != "identity":
            np.ascontiguousarray(encoder.projection, dtype=np.float32).tofile(path / "projection.f32")

        ids = None
        if item_ids is not None:
            ids = np.asarray(item_ids)
            if len(ids) != len(x):
                raise ValueError("item_ids length must equal number of embeddings")
            if ids.dtype == object:
                ids = ids.astype(str)
            np.save(path / "item_ids.npy", ids, allow_pickle=False)

        manifest = SemanticIndexManifest(
            version=FORMAT_VERSION,
            n_items=int(len(x)),
            dim=int(x.shape[1]),
            packed_bytes=int(packed.shape[1]),
            correction_values=2,
            projection_kind=str(projection_kind),
            seed=int(seed),
            item_bytes_theoretical=float(packed.shape[1] + 8),
            has_item_ids=bool(item_ids is not None),
        )
        (path / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
        return cls.load(path)

    @classmethod
    def load(cls, path: str | Path, *, mmap: bool = True) -> "BinarySemanticIndex":
        path = Path(path)
        raw = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        manifest = SemanticIndexManifest(**raw)
        if manifest.version != FORMAT_VERSION:
            raise ValueError(f"unsupported semantic index version {manifest.version}")

        centroid = np.fromfile(path / "centroid.f32", dtype=np.float32)
        if centroid.size != manifest.dim:
            raise ValueError("centroid.f32 has unexpected size")
        if manifest.projection_kind == "identity":
            projection = np.eye(manifest.dim, dtype=np.float32)
        else:
            projection = np.fromfile(path / "projection.f32", dtype=np.float32).reshape(manifest.dim, manifest.dim)
        encoder = CenteredBinaryEncoder(
            centroid=centroid,
            projection=projection,
            projection_kind=manifest.projection_kind,
            seed=manifest.seed,
            with_corrections=True,
        )

        mode = "r" if mmap else None
        if mmap:
            bits = np.memmap(
                path / "bits.u8",
                mode="r",
                dtype=np.uint8,
                shape=(manifest.n_items, manifest.packed_bytes),
            )
            corrections = np.memmap(
                path / "corrections.f32",
                mode="r",
                dtype=np.float32,
                shape=(manifest.n_items, manifest.correction_values),
            )
        else:
            bits = np.fromfile(path / "bits.u8", dtype=np.uint8).reshape(manifest.n_items, manifest.packed_bytes)
            corrections = np.fromfile(path / "corrections.f32", dtype=np.float32).reshape(
                manifest.n_items, manifest.correction_values
            )

        item_ids = None
        if manifest.has_item_ids:
            item_ids = np.load(path / "item_ids.npy", allow_pickle=False, mmap_mode=mode)
        return cls(path=path, manifest=manifest, encoder=encoder, bits=bits, corrections=corrections, item_ids=item_ids)

    def encode_batch(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Encode new/updated items using the already-fitted catalog transform."""
        q, corrections = encode_centered_binary(embeddings, self.encoder)
        return pack_document_bits(q), corrections

    def external_ids(self, row_ids: np.ndarray) -> np.ndarray:
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if self.item_ids is None:
            return row_ids
        return np.asarray(self.item_ids[row_ids])


__all__ = ["SemanticIndexManifest", "BinarySemanticIndex", "FORMAT_VERSION"]
