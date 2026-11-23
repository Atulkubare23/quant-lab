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


# ---------------------------------------------------------------- CSR and commodity (MAR21.51-21.85)
def test_csr_sector_gamma_reproduces_table_5_and_rating_split():
    assert F.csr_gamma(1, 2) == pytest.approx(0.75)
    assert F.csr_gamma(4, 13) == pytest.approx(0.5 * 0.20)           # different rating and sector
    assert F.csr_gamma(3, 11) == pytest.approx(0.5)                   # same sector, IG vs HY
    assert F.csr_gamma(5, 17) == pytest.approx(0.45) and F.csr_gamma(16, 3) == 0.0
    assert F.csr_gamma(17, 18) == pytest.approx(0.75) and F.csr_gamma(7, 8) == pytest.approx(0.05)
    for b in range(1, 19):
        for c in range(1, 19):
            assert F.csr_gamma(b, c) == F.csr_gamma(c, b)


def test_csr_intra_bucket_name_tenor_basis():
    a = F.Sens("CSR", "delta", 4, 1e6, curve="bond", tenor=5, name="ACME")
    b = F.Sens("CSR", "delta", 4, -1e6, curve="CDS", tenor=10, name="WIDGETCO")
    w = 0.03e6
    rho = 0.35 * 0.65 * 0.999
    assert F.delta_vega([a, b], "CSR", "delta") == pytest.approx(sqrt(2 * w * w - 2 * rho * w * w))
    c = F.Sens("CTP", "delta", 4, 1e6, curve="bond", tenor=5, name="ACME")
    d = F.Sens("CTP", "delta", 4, 1e6, curve="CDS", tenor=5, name="ACME")
    w = 0.05e6
    assert F.delta_vega([c, d], "CTP", "delta") == pytest.approx(sqrt(2 * w * w + 2 * 0.99 * w * w))
    other = [F.Sens("CSR", "delta", 16, 1e6, tenor=1, name="X"), F.Sens("CSR", "delta", 16, -1e6, tenor=1, name="Y")]
    assert F.delta_vega(other, "CSR", "delta") == pytest.approx(2 * 0.12e6)


def test_securitisation_non_ctp_weights_and_bucket_25():
    assert F.NCTP_RW[9] == pytest.approx(1.25 * 0.009) and F.NCTP_RW[17] == pytest.approx(1.75 * 0.009)
    s = [F.Sens("NCTP", "delta", 1, 1e6, tenor=5, name="T1"), F.Sens("NCTP", "delta", 4, 1e6, tenor=5, name="T2"),
         F.Sens("NCTP", "delta", 25, 1e6, tenor=5, name="T3")]
    k1, k4, k25 = 0.009e6, 0.02e6, 0.035e6
    assert F.delta_vega(s, "NCTP", "delta") == pytest.approx(sqrt(k1 * k1 + k4 * k4) + k25)   # gamma 0, no offset with 25


def test_commodity_correlations_and_buckets():
    wti = F.Sens("COMM", "delta", 2, 1e6, tenor=1, name="WTI", location="Cushing")
    brent = F.Sens("COMM", "delta", 2, -1e6, tenor=5, name="Brent", location="Le Havre")
    w = 0.35e6
    rho = 0.95 * 0.99 * 0.999
    assert F.delta_vega([wti, brent], "COMM", "delta") == pytest.approx(sqrt(2 * w * w - 2 * rho * w * w))
    gold = F.Sens("COMM", "delta", 7, 1e6, tenor=1, name="Gold")
    other = F.Sens("COMM", "delta", 11, 1e6, tenor=1, name="Potash")
    g, o = 0.20e6, 0.50e6
    assert F.delta_vega([wti, gold], "COMM", "delta") == pytest.approx(sqrt(w * w + g * g + 2 * 0.2 * w * g))
    assert F.delta_vega([gold, other], "COMM", "delta") == pytest.approx(sqrt(g * g + o * o))


def test_curvature_and_vega_for_new_classes_use_squared_name_correlation():
    c = [F.Curvature("COMM", 2, "WTI", 1e5, -1e4), F.Curvature("COMM", 2, "Brent", 2e5, -2e4)]
    r = 0.95 ** 2
    assert F.curvature(c, "COMM") == pytest.approx(sqrt(1e10 + 4e10 + 2 * r * 2e10))
    v = [F.Sens("CSR", "vega", 4, 1e6, tenor=1, name="A"), F.Sens("CSR", "vega", 4, 1e6, tenor=3, name="B")]
    rho = F.vega_maturity_corr(1, 3) * 0.35
    assert F.delta_vega(v, "CSR", "vega") == pytest.approx(sqrt(2e12 + 2 * rho * 1e12))
    assert F.curvature_rw("GIRR", "USD") == 0.017 and F.curvature_rw("COMM", 3) == 0.60


# ---------------------------------------------------------------- DRC (MAR22) and RRAO (MAR23)
def test_jtd_lgd_pnl_and_maturity_scaling():
    assert F.JTD("X", 10e6, -1e6, seniority="senior").gross() == pytest.approx(6.5e6)
    assert F.JTD("X", 10e6, 0, seniority="covered").gross() == pytest.approx(2.5e6)
    assert F.JTD("X", -10e6, 0.5e6, seniority="non-senior").gross() == pytest.approx(-9.5e6)
    assert F.JTD("X", 10e6, -11e6, seniority="equity").gross() == 0.0             # floored at zero
    assert F.JTD("X", 10e6, 0, seniority="equity", maturity=0.5).gross() == pytest.approx(5e6)
    assert F.JTD("X", 10e6, 0, seniority="equity", maturity=0.01).gross() == pytest.approx(2.5e6)  # 3m floor


def test_index_future_against_stock_offsets_fully():
    # MAR22 FAQ: a short 1-month future against a long stock both count at the 3-month floor
    pos = [F.JTD("ACME", 10, seniority="equity", maturity=0.25), F.JTD("ACME", -10, seniority="equity", maturity=1 / 12)]
    assert F.drc_nonsec(pos)["total"] == pytest.approx(0.0)


def test_seniority_rule_for_offsetting():
    # a senior short may not offset a non-senior long; a non-senior short may offset a senior long
    nets = F.net_jtd([F.JTD("X", 10, seniority="non-senior"), F.JTD("X", -10, seniority="senior")])
    assert nets[("corporates", "X", "BBB")] == pytest.approx((10, -7.5))
    nets = F.net_jtd([F.JTD("X", 10, seniority="senior"), F.JTD("X", -10, seniority="non-senior")])
    assert nets[("corporates", "X", "BBB")] == pytest.approx((0, -2.5))


def test_drc_bucket_hbr_by_hand():
    pos = [F.JTD("A", 100, rating="BBB", seniority="equity"), F.JTD("B", -50, rating="AA", seniority="equity"),
           F.JTD("S", 40, rating="A", seniority="equity", bucket="sovereigns")]
    r = F.drc_nonsec(pos)
    h = 100 / 150
    assert r["hbr"]["corporates"] == pytest.approx(h)
    assert r["buckets"]["corporates"] == pytest.approx(0.06 * 100 - h * 0.02 * 50)
    assert r["buckets"]["sovereigns"] == pytest.approx(0.03 * 40)
    assert r["total"] == pytest.approx(r["buckets"]["corporates"] + 1.2)
    # a fully short bucket carries no charge
    assert F.drc_nonsec([F.JTD("B", -50, seniority="equity")])["total"] == 0.0


def test_securitisation_offsets_only_within_tranche():
    same = [F.Tranche("pool1 A", 100, 0.10, "RMBS EU"), F.Tranche("pool1 A", -100, 0.10, "RMBS EU")]
    assert F.drc_sec_nonctp(same)["total"] == 0.0
    diff = [F.Tranche("pool1 A", 100, 0.10, "RMBS EU"), F.Tranche("pool1 B", -100, 0.10, "RMBS EU")]
    assert F.drc_sec_nonctp(diff)["total"] == pytest.approx(10 - 0.5 * 10)


def test_ctp_negative_bucket_counts_half():
    # index 1 net long 100, index 2 net short 100, same RW: HBR 0.5 across the CTP
    t = [F.Tranche("CDX IG 0-3", 100, 0.5, "CDX IG"), F.Tranche("iTraxx 0-3", -100, 0.5, "iTraxx")]
    r = F.drc_ctp(t)
    assert r["hbr"] == pytest.approx(0.5)
    assert r["buckets"] == pytest.approx({"CDX IG": 50, "iTraxx": -25})
    assert r["total"] == pytest.approx(50 - 12.5)
    assert F.drc_ctp([F.Tranche("x", -100, 0.5, "CDX IG")])["total"] == 0.0


def test_rrao_and_total():
    items = [F.Residual(1e8, exotic=True), F.Residual(1e8, exotic=False),
             F.Residual(1e8, exotic=False, back_to_back=True), F.Residual(1e8, exotic=False, listed_or_cleared=True),
             F.Residual(1e8, exotic=True, listed_or_cleared=True)]
    assert F.rrao(items) == pytest.approx(1e6 + 1e5 + 1e6)
    s = [F.Sens("FX", "delta", "EUR", 1e6)]
    r = F.standardised_approach(s, nonsec=[F.JTD("A", 100, seniority="equity")], residual=items[:1])
    assert r["total"] == pytest.approx(r["SBM"] + 6 + 1e6)
    assert r["SBM"] == pytest.approx(F.sbm(s)["capital"])
