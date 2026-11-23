"""SIMM engine mechanics. The shipped parameters are placeholders, so these tests check the algebra
(hand calculations from whatever parameters are loaded), not ISDA's numbers."""
import copy
from math import sqrt

import numpy as np
import pytest
from scipy.stats import norm

from qlab import simm as M

P = M.params()


def test_placeholder_status_is_carried_through():
    assert "PLACEHOLDER" in P["status"]
    assert M.simm([M.Sens("FX", "delta", "EUR", 1e5)])["status"] == P["status"]


def test_ir_delta_hand_calculation_with_subcurves_and_concentration():
    rw = P["IR"]["rw"]["regular"]
    i2, i10 = M.IR_TENORS.index("2y"), M.IR_TENORS.index("10y")
    s = [M.Sens("IR", "delta", "USD", 10_000, label1="2y"), M.Sens("IR", "delta", "USD", -6_000, label1="10y", subcurve="3M")]
    w = np.array([rw[i2] * 10_000, rw[i10] * -6_000])
    r = P["IR"]["tenor_corr"][i2][i10] * P["IR"]["phi"]
    assert M.ir_margin(s, P, "delta") == pytest.approx(sqrt(w @ np.array([[1, r], [r, 1]]) @ w))
    q = copy.deepcopy(P)
    q["IR"]["delta_threshold"]["regular"] = 1_000.0                    # total |s| = 4,000 -> CR = 2
    assert M.ir_margin(s, q, "delta") == pytest.approx(2 * M.ir_margin(s, P, "delta"))


def test_ir_cross_currency_uses_gamma_and_concentration_ratio():
    rw5 = P["IR"]["rw"]["regular"][M.IR_TENORS.index("5y")]
    s = [M.Sens("IR", "delta", "USD", 1e4, label1="5y"), M.Sens("IR", "delta", "EUR", 1e4, label1="5y")]
    k = rw5 * 1e4
    assert M.ir_margin(s, P, "delta") == pytest.approx(sqrt(2 * k * k + 2 * P["IR"]["gamma"] * k * k))


def test_curvature_lambda_theta():
    s = [M.Sens("IR", "vega", "USD", 1e6, label1="1y")]
    cvr = M.scaling(1.0) * 1e6
    z = norm.ppf(0.995)
    assert M.curvature_margin(s, P, "IR") == pytest.approx(z * z * cvr * P["IR"]["curvature_scale"])   # theta = 0
    neg = [M.Sens("IR", "vega", "USD", -1e6, label1="1y")]           # theta = -1 -> lambda = 1 -> margin 0
    assert M.curvature_margin(neg, P, "IR") == 0.0
    assert M.scaling(14 / 365) == 0.5 and M.scaling(1.0) == pytest.approx(0.5 * 14 / 365)


def test_totals_combine_risk_classes_with_psi():
    s = [M.Sens("IR", "delta", "USD", 1e4, label1="5y"), M.Sens("FX", "delta", "EUR", 1e5)]
    r = M.simm(s, P)
    ir, fx = r["risk_classes"]["IR"], r["risk_classes"]["FX"]
    assert fx == pytest.approx(P["FX"]["rw"] * 1e5)
    assert r["products"]["RatesFX"] == pytest.approx(sqrt(ir * ir + fx * fx + 2 * P["psi"]["IR"]["FX"] * ir * fx))
    e = M.simm(s + [M.Sens("EQ", "delta", "X", 1e4, bucket="residual")], P)
    assert e["SIMM"] == pytest.approx(r["SIMM"] + P["EQ"]["rw"]["residual"] * 1e4)


# ---------------------------------------------------------------- credit and commodity
def _two(a, b, r):
    return sqrt(a * a + b * b + 2 * r * a * b)


def test_credit_qualifying_same_and_different_issuer():
    rw = P["CRQ"]["rw"]["4"]
    same, diff = P["CRQ"]["intra_corr"]["same"], P["CRQ"]["intra_corr"]["diff"]
    s = [M.Sens("CRQ", "delta", "A/sen", 1_000, bucket="4", label1="5y"),
         M.Sens("CRQ", "delta", "A/sen", -400, bucket="4", label1="10y"),
         M.Sens("CRQ", "delta", "B/sen", 700, bucket="4", label1="5y")]
    w = np.array([rw * 1_000, rw * -400, rw * 700])
    rho = np.array([[1, same, diff], [same, 1, diff], [diff, diff, 1]])
    assert M.bucketed_margin(s, P, "CRQ", "delta") == pytest.approx(sqrt(w @ rho @ w))


def test_credit_concentration_is_per_issuer_across_tenors():
    q = copy.deepcopy(P)
    q["CRQ"]["delta_threshold"]["4"] = 100.0           # issuer A nets 400 -> CR 2; B 100 -> CR 1
    s = [M.Sens("CRQ", "delta", "A", 600, bucket="4", label1="5y"), M.Sens("CRQ", "delta", "A", -200, bucket="4", label1="10y"),
         M.Sens("CRQ", "delta", "B", 100, bucket="4", label1="5y")]
    rw, same, diff = q["CRQ"]["rw"]["4"], q["CRQ"]["intra_corr"]["same"], q["CRQ"]["intra_corr"]["diff"]
    w = np.array([rw * 600 * 2, rw * -200 * 2, rw * 100])
    f = 1 / 2                                            # min(CR) / max(CR) between A and B
    rho = np.array([[1, same, diff * f], [same, 1, diff * f], [diff * f, diff * f, 1]])
    assert M.bucketed_margin(s, q, "CRQ", "delta") == pytest.approx(sqrt(w @ rho @ w))


def test_credit_buckets_gamma_residual_and_base_correlation():
    a = M.Sens("CRQ", "delta", "A", 1_000, bucket="1", label1="5y")
    b = M.Sens("CRQ", "delta", "B", -500, bucket="7", label1="5y")
    ka, kb = P["CRQ"]["rw"]["1"] * 1_000, P["CRQ"]["rw"]["7"] * 500
    g = P["CRQ"]["inter_corr"][0][6]
    assert M.bucketed_margin([a, b], P, "CRQ", "delta") == pytest.approx(sqrt(ka * ka + kb * kb - 2 * g * ka * kb))
    r = M.Sens("CRQ", "delta", "C", 300, bucket="residual", label1="5y")
    kr = P["CRQ"]["rw"]["residual"] * 300
    assert M.bucketed_margin([a, b, r], P, "CRQ", "delta") == pytest.approx(sqrt(ka * ka + kb * kb - 2 * g * ka * kb) + kr)
    bc = [M.Sens("CRQ", "basecorr", "CDX IG", 1e4), M.Sens("CRQ", "basecorr", "iTraxx", 2e4)]
    kbc = P["CRQ"]["basecorr_rw"] * _two(1e4, 2e4, P["CRQ"]["basecorr_corr"])
    assert M.bucketed_margin(bc, P, "CRQ", "delta") == pytest.approx(kbc)
    assert M.bucketed_margin([a] + bc, P, "CRQ", "delta") == pytest.approx(sqrt(ka * ka + kbc * kbc))


def test_credit_non_qualifying():
    c = P["CRNQ"]
    s = [M.Sens("CRNQ", "delta", "pool1", 100, bucket="1", label1="5y"), M.Sens("CRNQ", "delta", "pool2", 100, bucket="2", label1="5y")]
    k1, k2 = c["rw"]["1"] * 100, c["rw"]["2"] * 100
    assert M.bucketed_margin(s, P, "CRNQ", "delta") == pytest.approx(_two(k1, k2, c["inter_corr"][0][1]))


def test_commodity_delta_intra_inter_and_other_bucket():
    c = P["CM"]
    s = [M.Sens("CM", "delta", "WTI", 1e4, bucket="2"), M.Sens("CM", "delta", "Brent", -6e3, bucket="2")]
    k = c["rw"]["2"] * _two(1e4, -6e3, c["intra_corr"]["2"])
    assert M.bucketed_margin(s, P, "CM", "delta") == pytest.approx(k)
    g = M.Sens("CM", "delta", "Gold", 5e3, bucket="12")
    kg, sb = c["rw"]["12"] * 5e3, c["rw"]["2"] * 4e3
    assert M.bucketed_margin(s + [g], P, "CM", "delta") == pytest.approx(sqrt(k * k + kg * kg + 2 * c["inter_corr"][1][11] * sb * kg))
    # bucket 16 has zero gamma with everything: pure quadrature
    o = M.Sens("CM", "delta", "Other", 5e3, bucket="16")
    ko = c["rw"]["16"] * 5e3
    assert M.bucketed_margin(s + [o], P, "CM", "delta") == pytest.approx(sqrt(k * k + ko * ko))


def test_commodity_vega_sums_expiries_and_curvature():
    c = P["CM"]
    v = [M.Sens("CM", "vega", "WTI", 1e5, bucket="2", label1="6m"), M.Sens("CM", "vega", "WTI", 5e4, bucket="2", label1="1y")]
    assert M.bucketed_margin(v, P, "CM", "vega") == pytest.approx(c["vrw"] * c["hvr"] * 1.5e5)
    cvr = M.scaling(0.5) * 1e5 + M.scaling(1.0) * 5e4                  # one risk factor; theta = 0
    z = norm.ppf(0.995)
    assert M.curvature_margin(v, P, "CM") == pytest.approx(z * z * cvr)


def test_curvature_residual_kept_separate():
    a = [M.Sens("CRQ", "vega", "A", 1e6, bucket="3", label1="1y")]
    r = [M.Sens("CRQ", "vega", "R", -1e6, bucket="residual", label1="1y")]
    # the short-vol residual cannot offset the long-vol bucket
    assert M.curvature_margin(a + r, P, "CRQ") == pytest.approx(M.curvature_margin(a, P, "CRQ"))
    assert M.curvature_margin(r, P, "CRQ") == 0.0


def test_product_classes_and_override():
    s = [M.Sens("CRQ", "delta", "A", 1_000, bucket="4", label1="5y"), M.Sens("CM", "delta", "WTI", 1e4, bucket="2"),
         M.Sens("EQ", "delta", "X", 1e4, bucket="5")]
    r = M.simm(s, P)
    assert set(r["products"]) == {"Credit", "Commodity", "Equity"}
    assert r["SIMM"] == pytest.approx(sum(r["risk_classes"].values()))       # one class per product: no psi
    # booking the equity trade in the Credit product class diversifies it with credit via psi
    s[2].product = "Credit"
    r2 = M.simm(s, P)
    crq, eq = r["risk_classes"]["CRQ"], r["risk_classes"]["EQ"]
    assert r2["products"]["Credit"] == pytest.approx(sqrt(crq * crq + eq * eq + 2 * P["psi"]["CRQ"]["EQ"] * crq * eq))
    assert P["psi"]["CM"]["IR"] == P["psi"]["IR"]["CM"]


# ---------------------------------------------------------------- add-ons and product-class multipliers
def test_no_addons_changes_nothing():
    s = [M.Sens("CRQ", "delta", "A", 1_000, bucket="4", label1="5y"), M.Sens("FX", "delta", "EUR", 1e5)]
    r = M.simm(s, P)
    assert r["SIMM"] == pytest.approx(sum(r["products"].values()))
    assert r["addon"]["total"] == 0.0 and r["multipliers"] == {}


def test_multiplier_scales_only_its_product_class():
    s = [M.Sens("CRQ", "delta", "A", 1_000, bucket="4", label1="5y"), M.Sens("FX", "delta", "EUR", 1e5)]
    base = M.simm(s, P)
    r = M.simm(s, P, [M.AddOn("multiplier", "Credit", 1.5), M.AddOn("multiplier", "Equity", 2.0)])
    assert r["products_after_multiplier"]["Credit"] == pytest.approx(1.5 * base["products"]["Credit"])
    assert r["products_after_multiplier"]["RatesFX"] == pytest.approx(base["products"]["RatesFX"])
    assert r["SIMM"] == pytest.approx(base["SIMM"] + 0.5 * base["products"]["Credit"])   # no Equity trades: no effect


def test_fixed_and_notional_addons():
    s = [M.Sens("FX", "delta", "EUR", 1e5)]
    base = M.simm(s, P)["SIMM"]
    a = [M.AddOn("fixed", amount=1e5), M.AddOn("fixed", amount=5e4),
         M.AddOn("factor", "Bermudan", 0.005), M.AddOn("notional", "Bermudan", 6e7), M.AddOn("notional", "Bermudan", -4e7),
         M.AddOn("factor", "Quanto", 0.01)]                                      # factor without notional: zero
    r = M.simm(s, P, a)
    assert r["addon"]["notional"] == pytest.approx({"Bermudan": 0.005 * 1e8, "Quanto": 0.0})
    assert r["addon"]["total"] == pytest.approx(1.5e5 + 5e5)
    assert r["SIMM"] == pytest.approx(base + 6.5e5)


def test_addon_input_errors():
    with pytest.raises(ValueError):
        M.addon_margin([M.AddOn("notional", "Bermudan", 1e7)])                  # no factor
    with pytest.raises(ValueError):
        M.multipliers([M.AddOn("multiplier", "Rates", 1.2)])                    # not a product class
    with pytest.raises(ValueError):
        M.multipliers([M.AddOn("multiplier", "Credit", 1.2), M.AddOn("multiplier", "Credit", 1.3)])
    with pytest.raises(ValueError):
        M.addon_margin([M.AddOn("bogus")])
