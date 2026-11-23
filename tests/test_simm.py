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
