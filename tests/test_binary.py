import numpy as np

from ras.binary import (
    build_centered_binary_code,
    int4_weight_bitplanes,
    pack_document_bits,
    quantize_weight_int4,
    score_compiled_linear,
)


def test_centered_binary_shapes_and_storage():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 16)).astype(np.float32)
    code = build_centered_binary_code(x[:30], x[30:40], x[40:], projection_kind="identity")
    assert code.Q_fit.shape == (30, 16)
    assert code.correction_test.shape == (10, 2)
    assert set(np.unique(code.Q_test)).issubset({0, 1})
    assert code.item_bytes_theoretical == 10.0  # 16 bits + two f32 corrections


def test_two_level_linear_reconstruction_beats_sign_only_on_training_distribution():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 32)).astype(np.float32)
    w = rng.normal(size=32).astype(np.float32)
    b = 0.2
    code = build_centered_binary_code(x[:120], x[120:160], x[160:], projection_kind="identity")
    approx = score_compiled_linear(code.Q_test, code.correction_test, code, w, b)
    exact = x[160:] @ w + b
    assert np.corrcoef(approx, exact)[0, 1] > 0.75


def test_int4_and_bitplane_export():
    rng = np.random.default_rng(2)
    w = rng.normal(size=384).astype(np.float32)
    q, decoded, lo, scale = quantize_weight_int4(w)
    assert q.min() >= 0 and q.max() <= 15
    assert decoded.shape == w.shape
    assert scale > 0
    planes = int4_weight_bitplanes(q)
    assert planes.shape == (4, 48)

    bits = rng.integers(0, 2, size=(7, 384), dtype=np.uint8)
    packed = pack_document_bits(bits)
    assert packed.shape == (7, 48)
    recovered = np.unpackbits(packed, axis=1, bitorder="little")[:, :384]
    np.testing.assert_array_equal(recovered, bits)
