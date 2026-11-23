import numpy as np
import pytest

from qlab.rough import RBergomi


def test_volterra_variance_and_martingales():
    m = RBergomi(H=0.1)
    t, Y, V, S = m.paths(1.0, n_paths=40_000, seed=1)
    for k in [10, 100, 312]:
        assert Y[:, k].var() == pytest.approx(t[k] ** (2 * m.H), rel=0.04)    # Var(Y_t) = t^{2H}
        assert V[:, k].mean() == pytest.approx(m.xi0, rel=0.05)                # forward variance is matched
    assert S[:, -1].mean() == pytest.approx(1.0, abs=0.005)


def test_zero_vol_of_vol_is_black_scholes():
    m = RBergomi(eta=0.0, xi0=0.2 ** 2)
    iv = m.smile(0.5, [-0.2, 0.0, 0.2], n_paths=100_000)
    assert iv == pytest.approx(0.2, abs=0.003)


def test_skew_is_negative_and_explodes_faster_when_rougher():
    Ts = np.array([0.04, 0.16, 0.64])
    slope = {}
    for H in (0.05, 0.3):
        m = RBergomi(H=H)
        sk = np.array([m.atm_skew(T, n_paths=40_000, seed=3) for T in Ts])
        assert np.all(sk < 0)
        slope[H] = np.polyfit(np.log(Ts), np.log(-sk), 1)[0]
    assert slope[0.05] < slope[0.3] - 0.1
    assert -0.6 < slope[0.05] < -0.2
