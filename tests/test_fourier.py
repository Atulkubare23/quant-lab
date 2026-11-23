import numpy as np
import pytest

from qlab import fourier as F

S, r, q = 100.0, 0.03, 0.01
KS = np.array([70, 85, 100, 115, 130.0])

MODELS = [
    F.BS(0.25), F.Merton(0.15, 0.5, -0.1, 0.15), F.Kou(0.15, 1.0, 0.4, 10, 5), F.VG(0.2, 0.2, -0.15),
    F.NIG(15.0, -5.0, 0.5), F.CGMY(1.0, 8.0, 10.0, 0.5), F.Heston(0.04, 2.0, 0.04, 0.5, -0.7),
    F.Bates(0.04, 2.0, 0.04, 0.5, -0.7, 0.3, -0.1, 0.1),
]


@pytest.mark.parametrize("T", [0.25, 1.0, 3.0])
def test_three_pricers_match_black_scholes(T):
    exact = np.array([F.bs_price(S, k, T, r, q, 0.25) for k in KS])
    m = F.BS(0.25)
    assert F.cos(m, S, KS, T, r, q) == pytest.approx(exact, abs=1e-7)
    assert F.lewis(m, S, KS, T, r, q) == pytest.approx(exact, abs=1e-7)
    assert F.carr_madan(m, S, KS, T, r, q) == pytest.approx(exact, abs=2e-3)


def test_merton_matches_the_poisson_series():
    m = F.Merton(0.15, 0.5, -0.1, 0.15)
    exact = np.array([F.merton_series(S, k, 1.0, r, q, m) for k in KS])
    assert F.cos(m, S, KS, 1.0, r, q) == pytest.approx(exact, abs=1e-7)
    assert F.lewis(m, S, KS, 1.0, r, q) == pytest.approx(exact, abs=1e-7)


@pytest.mark.parametrize("m", MODELS, ids=lambda m: type(m).__name__)
def test_every_model_is_a_martingale_and_the_pricers_agree(m):
    T = 1.0
    assert m.cf(np.array(-1j), T, r, q).real == pytest.approx(np.exp((r - q) * T), rel=1e-10)   # E[S_T/S_0]
    assert abs(m.cf(np.array(0.0), T, r, q) - 1) < 1e-12
    c, l, f = F.cos(m, S, KS, T, r, q), F.lewis(m, S, KS, T, r, q), F.carr_madan(m, S, KS, T, r, q)
    assert c == pytest.approx(l, abs=1e-5)
    assert f == pytest.approx(l, abs=5e-3)
    put = F.cos(m, S, KS, T, r, q, call=False)
    assert c - put == pytest.approx(S * np.exp(-q * T) - KS * np.exp(-r * T), abs=1e-10)       # parity
    assert np.all(np.diff(c) < 0) and np.all(np.diff(c, 2) > 0)                                 # no arbitrage


def test_negative_jumps_and_negative_rho_give_a_put_skew():
    for m in (F.Merton(0.15, 0.5, -0.1, 0.15), F.Heston(0.04, 2.0, 0.04, 0.5, -0.7), F.VG(0.2, 0.2, -0.15)):
        iv = F.smile(m, S, KS, 1.0, r, q)
        assert iv[0] > iv[2] > iv[-1] or iv[0] > iv[2]
    flat = F.smile(F.BS(0.2), S, KS, 1.0, r, q)
    assert flat == pytest.approx(0.2, abs=1e-6)
