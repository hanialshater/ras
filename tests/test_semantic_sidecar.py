from __future__ import annotations

import hashlib

import numpy as np

from ras.semantic_index import BinarySemanticIndex
from ras.semantic_program import ProgramStore, compile_linear_program, fit_binary_predicate
from ras.serving import SemanticExecutor


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_portable_index_and_compiled_score_match(tmp_path):
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    index = BinarySemanticIndex.build(tmp_path / "index", x, item_ids=np.arange(1000, 1080))
    assert index.manifest.packed_bytes == 4
    assert index.manifest.item_bytes_theoretical == 12.0

    w = rng.normal(size=32).astype(np.float32)
    p = compile_linear_program(index, name="concept", weight=w, intercept=0.2)
    assert p.program_bytes_theoretical == 40  # 4 planes x 4 bytes + 6 f32 scalars

    ids = np.arange(25)
    got = p.raw_scores(index.bits[ids], index.corrections[ids])

    # Decode the int4 bit planes and independently evaluate the same two-level
    # reconstruction formula used by the native kernel.
    plane_bits = np.stack([
        np.unpackbits(row, bitorder="little")[: index.dim] for row in p.bitplanes
    ])
    qweight = sum((plane_bits[b].astype(np.uint8) << b) for b in range(4))
    decoded = p.weight_lo + p.weight_scale * qweight.astype(np.float32)
    qitems = np.unpackbits(np.asarray(index.bits[ids]), axis=1, bitorder="little")[:, : index.dim]
    pos = qitems.astype(np.float32) @ decoded
    lo = np.asarray(index.corrections[ids, 0])
    hi = np.asarray(index.corrections[ids, 1])
    expected = p.base + lo * (p.sum_w - pos) + hi * pos
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(index.external_ids(np.array([0, 3])), np.array([1000, 1003]))


def test_add_predicates_without_reindex_and_execute_topk(tmp_path):
    rng = np.random.default_rng(9)
    x = rng.normal(size=(300, 48)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    index_dir = tmp_path / "index"
    index = BinarySemanticIndex.build(index_dir, x)
    bits_before = _sha(index_dir / "bits.u8")

    w1 = rng.normal(size=48)
    w2 = rng.normal(size=48)
    y1 = (x @ w1 > np.median(x @ w1)).astype(np.int8)
    y2 = (x @ w2 > np.median(x @ w2)).astype(np.int8)
    store = ProgramStore(tmp_path / "programs")
    store.save(fit_binary_predicate(index, x, y1, name="clean", seed=1))
    store.save(fit_binary_predicate(index, x, y2, name="formal", seed=2))

    # Predicate deployment changes only the program store.
    assert _sha(index_dir / "bits.u8") == bits_before
    assert set(store.names()) == {"clean", "formal"}
    assert (tmp_path / "programs" / "clean" / "scalars.f32").stat().st_size == 7 * 4

    executor = SemanticExecutor.open(str(index_dir), str(tmp_path / "programs"))
    candidates = np.arange(index.n_items)
    result = executor.topk(candidates, positive=["clean", "formal"], k=30)
    assert len(result.row_ids) == 30
    assert np.all(result.scores[:-1] >= result.scores[1:])
    assert result.predicates == 2
    assert result.semantic_ms >= 0.0 and result.topk_ms >= 0.0

    neg = executor.score_candidates(candidates[:20], positive=["clean"], negative=["formal"])
    assert neg.shape == (20,)
    assert np.isfinite(neg).all()
