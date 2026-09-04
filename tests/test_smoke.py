import numpy as np
from ras.substrate import build_substrate
from ras.predicates import fit_boosted_lut, score_boosted
from ras.calibration import fit_scalar_calibrator
from ras.composition import compose_logprob


def _toy(seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(1200, 32)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    y = (x[:, 0] + 0.6 * x[:, 1] > 0).astype(bool)
    return x[:800], x[800:1000], x[1000:], y[:800], y[800:1000], y[1000:]


def test_substrate_and_boosted_lut_smoke():
    xf, xc, xt, yf, yc, yt = _toy()
    s = build_substrate(xf, xc, xt, name="toy", seed=7, bits=4, projection_kind="orthogonal", quantizer_kind="quantile")
    assert s.Q_fit.shape == (800, 32)
    m = fit_boosted_lut(s.Q_fit, yf, k=8, n_bins=16, candidate_pool=16)
    sc = score_boosted(s.Q_cal, m)
    st = score_boosted(s.Q_test, m)
    assert np.isfinite(sc).all() and np.isfinite(st).all()
    cal = fit_scalar_calibrator(sc, yc)
    assert np.isfinite(cal.transform(st)).all()


def test_logprob_composition_prefers_joint_high_scores():
    L = np.array([[4.0, 4.0], [4.0, -4.0], [-4.0, 4.0], [-4.0, -4.0]])
    assert np.argmax(compose_logprob(L, [1, 1])) == 0
    assert np.argmax(compose_logprob(L, [1, -1])) == 1
