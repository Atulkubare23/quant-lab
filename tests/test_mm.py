import numpy as np
import pytest

from qlab.mm import Market, as_quotes, simulate, summary


def test_quotes_skew_away_from_inventory():
    m = Market()
    b0, a0 = as_quotes(100, 0, 0.5, m, 0.1)
    b1, a1 = as_quotes(100, 5, 0.5, m, 0.1)
    assert (a0 + b0) / 2 == pytest.approx(100)
    assert b1 < b0 and a1 < a0 and (a1 - b1) == pytest.approx(a0 - b0)   # long: cheaper on both sides


def test_paper_table_gamma_0_1():
    """Avellaneda & Stoikov (2008) table 2: inventory strategy 65.00 / 6.60 / 2.80, symmetric 68.44 / 12.70 / 8.66."""
    s = summary(Market(), 0.1, n_paths=1000, seed=0)
    assert s["as"]["mean"] == pytest.approx(65.0, abs=1.5)
    assert s["as"]["std"] == pytest.approx(6.6, abs=1.0)
    assert s["as"]["q_std"] == pytest.approx(2.8, abs=0.6)
    assert s["symmetric"]["mean"] == pytest.approx(68.44, abs=1.5)
    assert s["symmetric"]["std"] == pytest.approx(12.7, abs=1.5)
    assert s["as"]["std"] < 0.6 * s["symmetric"]["std"]


def test_glft_respects_the_inventory_limit():
    pnl, q = simulate(Market(dt=0.01), "glft", 0.1, n_paths=200, seed=1, Q=4)
    assert np.all(np.abs(q) <= 4)
    assert pnl.mean() > 0
