"""Minimal bring-your-own-data example.

Expected files:
  embeddings.npy             float32 [N, D]
  office.npy                 0/1 [N]
  minimalist.npy             0/1 [N]
  sporty.npy                 0/1 [N]
  candidates.npy             integer row IDs from your retrieval system

Replace the file names and predicate names with your own data.  The host ANN
engine only needs to pass integer row IDs to the semantic sidecar.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

from ras import BinarySemanticIndex, ProgramStore, SemanticExecutor, fit_binary_predicate


ROOT = Path("byo_demo")
X = np.load("embeddings.npy").astype(np.float32)

# Build once per embedding/index generation.
index = BinarySemanticIndex.build(ROOT / "index", X, projection_kind="identity", overwrite=True)
store = ProgramStore(ROOT / "programs")

# Compile predicates independently.  Adding another concept does not touch the
# item code written above.
for name, label_file in {
    "office": "office.npy",
    "minimalist": "minimalist.npy",
    "sporty": "sporty.npy",
}.items():
    y = np.load(label_file).astype(np.int8)
    store.save(fit_binary_predicate(index, X, y, name=name))

# These IDs would normally come from ANN/lexical retrieval + exact filters.
candidate_ids = np.load("candidates.npy").astype(np.int64)
executor = SemanticExecutor.open(str(ROOT / "index"), str(ROOT / "programs"))
result = executor.topk(
    candidate_ids,
    positive=["minimalist", "office"],
    negative=["sporty"],
    k=min(1000, len(candidate_ids)),
)

print("semantic_ms:", result.semantic_ms)
print("topk_ms:", result.topk_ms)
print("top row IDs:", result.row_ids[:20])
