from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from ras import BinarySemanticIndex, ProgramStore, SemanticExecutor, compile_linear_program
from ras.accounting import deployment_memory_rows
from ras.evaluation import aggregate
from ras.pq import compile_pq_head, score_pq


def setup_index(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(80, 31)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    index = BinarySemanticIndex.build(tmp_path / "index", x)
    p = compile_linear_program(index, name="concept", weight=rng.normal(size=31), intercept=.2)
    return x, index, p


def test_rejects_equal_dimension_wrong_encoder(tmp_path):
    x, index, p = setup_index(tmp_path)
    other = BinarySemanticIndex.build(tmp_path / "other", x, projection_kind="orthogonal")
    store = ProgramStore(tmp_path / "programs")
    store.save(p)
    with pytest.raises(ValueError, match="different semantic index"):
        SemanticExecutor(other, store).score_candidates(np.arange(10), positive=["concept"])


def test_program_overwrite_refreshes_existing_executor(tmp_path):
    _, index, p = setup_index(tmp_path)
    store = ProgramStore(tmp_path / "programs")
    store.save(p)
    live = SemanticExecutor(index, store)
    old = live.score_candidates(np.arange(10), positive=["concept"])
    # Simulate a publisher in another process/store instance.
    ProgramStore(store.path).save(replace(p, base=p.base + 3))
    new = live.score_candidates(np.arange(10), positive=["concept"])
    fresh = SemanticExecutor(index, store).score_candidates(np.arange(10), positive=["concept"])
    assert not np.array_equal(old, new)
    np.testing.assert_array_equal(new, fresh)


@pytest.mark.parametrize("name", [".", "..", "../escape", "/absolute"])
def test_program_name_cannot_escape(tmp_path, name):
    _, _, p = setup_index(tmp_path)
    with pytest.raises(ValueError):
        ProgramStore(tmp_path / "programs").save(replace(p, name=name))


def test_symlink_program_cannot_escape(tmp_path):
    _, _, p = setup_index(tmp_path)
    store = ProgramStore(tmp_path / "programs")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.path / p.name).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        store.save(p)
    assert list(outside.iterdir()) == []


def test_purity_includes_zero_relevant_queries():
    frame = pd.DataFrame([
        dict(seed=7, query_id="positive", method="dense", retention=.2,
             total_true=5, recall=1., purity=1., recall_efficiency=1.),
        dict(seed=7, query_id="empty", method="dense", retention=.2,
             total_true=0, recall=np.nan, purity=0., recall_efficiency=np.nan),
    ])
    summary, _ = aggregate(frame, {"benchmark": {"bootstrap_samples": 20}})
    by_metric = summary.set_index("metric")
    assert by_metric.loc["purity", "mean"] == .5
    assert by_metric.loc["purity", "n"] == 2
    assert by_metric.loc["recall", "n"] == 1


def test_pq_int4_scores_equal_reconstructed_vector_dot():
    rng = np.random.default_rng(5)
    book = rng.normal(size=(4, 256, 6)).astype(np.float32)
    codes = rng.integers(0, 256, size=(70, 4), dtype=np.uint8)
    w = rng.normal(size=24).astype(np.float32)
    lut, metadata = compile_pq_head(book, w, .4, int4=True)
    packed = metadata["packed_weights"]
    q = np.empty(24, dtype=np.uint8)
    q[::2], q[1::2] = packed & 15, packed >> 4
    decoded = metadata["weight_lo"] + metadata["weight_scale"] * q
    recon = np.concatenate([book[j, codes[:, j]] for j in range(4)], axis=1)
    np.testing.assert_allclose(score_pq(codes, lut, .4), recon @ decoded + .4, atol=4e-6, rtol=1e-5)
    assert metadata["stored_bytes"] == 32


def test_resident_vector_accounting_and_small_vocabulary():
    rows = {r["method"]: r for r in deployment_memory_rows(5_000_000, 100_000)}
    assert rows["linear_fp32"]["combined_persistent_B"] == 7_834_800_000
    assert rows["bbq1_ls2_int4q"]["combined_persistent_B"] == 7_981_600_000
    small = {r["method"]: r for r in deployment_memory_rows(5_000_000, 8)}
    assert small["materialized_fp32_logits"]["incremental_persistent_B"] == 160_000_000


def test_legacy_program_requires_recompilation(tmp_path):
    import json
    _, _, p = setup_index(tmp_path)
    store = ProgramStore(tmp_path / "programs")
    path = store.save(p)
    meta = json.loads((path / "manifest.json").read_text())
    meta["version"] = 1
    (path / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="recompile"):
        store.load(p.name)
