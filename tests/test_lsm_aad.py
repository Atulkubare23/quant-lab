import numpy as np
import pytest

from qlab import aad
from qlab.fourier import bs_price
from qlab.lsm import crr_american, lsm


def test_lsm_american_put_matches_the_tree_from_below():
    for S, sig in [(36, 0.2), (44, 0.4)]:
        v, se = lsm(S, 40, sig, 0.06, 0.0, 1.0, 50, n_paths=100_000)
        tree = crr_american(S, 40, sig, 0.06, 0.0, 1.0)
        assert tree - 0.06 < v < tree + 3 * se                  # low-biased, close
        assert v > bs_price(S, 40, 1.0, 0.06, 0.0, sig, call=False) + 0.05   # early exercise is worth something


def test_american_call_without_dividends_is_european():
    v = crr_american(100, 100, 0.25, 0.05, 0.0, 1.0, call=True)
    assert v == pytest.approx(bs_price(100, 100, 1.0, 0.05, 0.0, 0.25), abs=2e-3)


def test_bermudan_max_call_benchmark():
    v, se = lsm([100, 100], 100, 0.2, 0.05, 0.10, 3.0, 9, "maxcall", np.eye(2), 200_000)
    assert v == pytest.approx(13.90, abs=0.12)


def test_adjoint_greeks_equal_bump_greeks_with_common_random_numbers():
    n, rho = 4, 0.3
    corr = np.full((n, n), rho) + (1 - rho) * np.eye(n)
    S0, sig, w = np.array([90, 100, 110, 95.0]), np.array([0.2, 0.25, 0.3, 0.35]), np.full(n, 0.25)
    args = dict(r=0.03, w=w, K=100, T=1.0, corr=corr, n_paths=50_000, seed=7)
    v, g = aad.basket_call(S0, sig, **args)
    h = 1e-4
    for i in range(n):
        up, dn = S0.copy(), S0.copy()
        up[i] += h; dn[i] -= h
        fd = (aad.basket_call(up, sig, adjoint=False, **args) - aad.basket_call(dn, sig, adjoint=False, **args)) / (2 * h)
        assert g["delta"][i] == pytest.approx(fd, rel=1e-5)
        su, sd = sig.copy(), sig.copy()
        su[i] += h; sd[i] -= h
        fd = (aad.basket_call(S0, su, adjoint=False, **args) - aad.basket_call(S0, sd, adjoint=False, **args)) / (2 * h)
        assert g["vega"][i] == pytest.approx(fd, rel=1e-4)


def test_adjoint_single_asset_recovers_black_scholes_greeks():
    from scipy.stats import norm
    S, K, T, r, s = 100.0, 105.0, 1.0, 0.02, 0.2
    v, g = aad.basket_call([S], [s], r, [1.0], K, T, np.eye(1), n_paths=400_000, seed=1)
    d1 = (np.log(S / K) + (r + s * s / 2) * T) / (s * np.sqrt(T))
    assert v == pytest.approx(bs_price(S, K, T, r, 0, s), rel=1e-2)
    assert g["delta"][0] == pytest.approx(norm.cdf(d1), rel=1e-2)
    assert g["vega"][0] == pytest.approx(S * norm.pdf(d1) * np.sqrt(T), rel=2e-2)


def test_asian_tape_with_fifty_two_steps():
    v, g = aad.asian_call(100, 0.2, 0.03, 100, 1.0, n_steps=52, n_paths=50_000, seed=2)
    v_up = aad.asian_call(100.01, 0.2, 0.03, 100, 1.0, n_steps=52, n_paths=50_000, seed=2)[0]
    v_dn = aad.asian_call(99.99, 0.2, 0.03, 100, 1.0, n_steps=52, n_paths=50_000, seed=2)[0]
    assert g["delta"] == pytest.approx((v_up - v_dn) / 0.02, rel=1e-4)
    assert 0 < v < bs_price(100, 100, 1.0, 0.03, 0, 0.2)        # averaging lowers the value
