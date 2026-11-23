from math import exp

import numpy as np
import pytest

from qlab.hullwhite import HW, Tree, bachelier_swaption, calibrate_sigma

P0 = lambda t: exp(-0.035 * t - 0.001 * t * t / 2)
M = HW(0.05, 0.01, P0)
PAY = [6, 7, 8, 9, 10.0]
F = (P0(5) - P0(10)) / sum(P0(t) for t in PAY)


def test_tree_reprices_the_curve_and_jamshidian():
    tree = Tree(M, 10, 50)
    for T in [0.5, 2, 5, 10]:
        assert tree.zero(T) == pytest.approx(P0(T), rel=1e-12)
    for K in [F - 0.01, F, F + 0.01]:
        for payer in (True, False):
            assert tree.bermudan([5], PAY, K, payer) == pytest.approx(M.swaption(5, PAY, K, payer), rel=3e-3)


def test_monte_carlo_is_unbiased_for_bonds_and_swaptions():
    x, D = M.simulate([1, 5], 400_000, seed=1)
    assert D[:, 0].mean() == pytest.approx(P0(1), rel=2e-3)
    assert (D[:, 0] * M.bond(1, 7, x[:, 0])).mean() == pytest.approx(P0(7), rel=2e-3)
    swap = 1 - sum(0.0433 * M.bond(5, T, x[:, 1]) for T in PAY) - M.bond(5, 10, x[:, 1])
    mc = (D[:, 1] * np.maximum(swap, 0)).mean()
    assert mc == pytest.approx(M.swaption(5, PAY, 0.0433), rel=1.5e-2)


def test_parity_bounds_and_bermudan_premium():
    rec, pay = M.swaption(5, PAY, F, False), M.swaption(5, PAY, F, True)
    assert rec == pytest.approx(pay, rel=1e-10)                         # ATM forward: payer = receiver
    assert M.zbc(1, 3, 0.95) - M.zbp(1, 3, 0.95) == pytest.approx(P0(3) - 0.95 * P0(1), abs=1e-14)
    tree = Tree(M, 10, 50)
    eur = [tree.bermudan([e], PAY[PAY.index(e + 1):], F) for e in [5, 6, 7, 8, 9]]
    berm = tree.bermudan([5, 6, 7, 8, 9], PAY, F)
    assert berm > max(eur) and berm < sum(eur)
    tiny = HW(0.05, 1e-6, P0)
    assert tiny.swaption(5, PAY, F - 0.01) == pytest.approx(0.01 * sum(P0(t) for t in PAY), rel=1e-4)


def test_caps_and_calibration():
    K, t1, t2 = 0.03, 2.0, 2.25
    floorlet = (1 + K * 0.25) * M.zbc(t1, t2, 1 / (1 + K * 0.25))
    assert M.caplet(t1, t2, K) - floorlet == pytest.approx(P0(t1) - (1 + K * 0.25) * P0(t2), abs=1e-14)
    assert M.cap(5, 0.0) > P0(0.25) - P0(5)          # Gaussian rates can go negative: a 0% cap beats the float leg
    target = bachelier_swaption(F, F, 5, 0.009, sum(P0(t) for t in PAY))
    s = calibrate_sigma(0.05, P0, 5, PAY, F, target)
    assert HW(0.05, s, P0).swaption(5, PAY, F) == pytest.approx(target, rel=1e-8)
