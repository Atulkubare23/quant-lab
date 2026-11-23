from math import sqrt

import numpy as np
import pytest

from qlab import frtb as F
from qlab.fourier import bs_price


def test_girr_correlations_reproduce_mar21_table_2():
    table = {(0.25, 0.5): 0.970, (0.25, 2): 0.811, (1, 5): 0.887, (2, 30): 0.657, (5, 10): 0.970, (15, 20): 0.990, (0.25, 10): 0.40}
    for (a, b), v in table.items():
        assert F.girr_tenor_corr(a, b) == pytest.approx(v, abs=5e-4)


def test_scenarios_and_single_factor_charges():
    assert F.scen(0.5, "high") == pytest.approx(0.625) and F.scen(0.9, "high") == 1.0
    assert F.scen(0.5, "low") == pytest.approx(0.375) and F.scen(0.9, "low") == pytest.approx(0.8)
    s = [F.Sens("GIRR", "delta", "BRL", 1e8, tenor=5)]
    assert F.delta_vega(s, "GIRR", "delta") == pytest.approx(0.011 * 1e8)
    s = [F.Sens("GIRR", "delta", "USD", 1e8, tenor=5)]
    assert F.delta_vega(s, "GIRR", "delta") == pytest.approx(0.011 * 1e8 / sqrt(2))     # specified currency


def test_two_tenor_hand_calculation():
    s = [F.Sens("GIRR", "delta", "BRL", 1e8, tenor=1), F.Sens("GIRR", "delta", "BRL", -1e8, tenor=5)]
    w1, w2, r = 0.016e8, -0.011e8, F.girr_tenor_corr(1, 5)
    for sc in ("medium", "high", "low"):
        rr = float(F.scen(r, sc))
        assert F.delta_vega(s, "GIRR", "delta", sc) == pytest.approx(sqrt(w1 * w1 + w2 * w2 + 2 * rr * w1 * w2))


def test_equity_buckets_other_sector_and_cross_bucket():
    s = [F.Sens("EQ", "delta", 11, 1e6, name="A"), F.Sens("EQ", "delta", 11, -1e6, name="B")]
    assert F.delta_vega(s, "EQ", "delta") == pytest.approx(2 * 0.70e6)                   # no offset in bucket 11
    s = [F.Sens("EQ", "delta", 12, 1e6, name="SPX"), F.Sens("EQ", "delta", 13, -1e6, name="EEM")]
    a, b = 0.15e6, -0.25e6
    assert F.delta_vega(s, "EQ", "delta") == pytest.approx(sqrt(a * a + b * b + 2 * 0.75 * a * b))
    assert F.eq_gamma(3, 7) == 0.15 and F.eq_gamma(3, 12) == 0.45 and F.eq_gamma(11, 12) == 0.0


def test_curvature_charges_short_options_only():
    n, S0, rw = 10_000, 100.0, F.EQ_RW[12]
    for sign in (1, -1):
        price = lambda S: sign * n * bs_price(S, 100, 1.0, 0.0, 0.0, 0.2)
        d = (price(S0 * 1.0001) - price(S0 * 0.9999)) / 0.0002
        up, dn = F.curvature_cvr([price], S0, rw, d)
        c = F.curvature([F.Curvature("EQ", 12, "SPX", up, dn)], "EQ")
        if sign > 0:
            assert up < 0 and dn < 0 and c == 0
        else:
            assert c == pytest.approx(max(up, dn)) and c > 0


def test_capital_takes_the_worst_scenario():
    s = [F.Sens("FX", "delta", "EUR", 1e7), F.Sens("FX", "delta", "TRY", -1e7)]
    r = F.capital(s)
    assert r["capital"] == max(v["total"] for v in r["scenarios"].values())
    assert r["binding"] == "low"                                                          # offsetting positions
