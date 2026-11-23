from math import exp

import numpy as np
import pytest

from qlab.hullwhite import HW
from qlab.xva import CSA, Swap, exposures, xva

P0 = lambda t: exp(-0.035 * t)
M = HW(0.05, 0.01, P0)
K = (1 - P0(10)) / sum(P0(t) for t in range(1, 11))
TIMES = np.array([0.5, 1.0, 2.5, 3.0, 6.0, 9.5])


@pytest.fixture(scope="module")
def payer():
    return exposures(M, [Swap(K, 0, 10)], TIMES, n_paths=200_000, seed=3)


def test_discounted_value_is_a_martingale(payer):
    dV = (payer.D * payer.V).mean(0)
    for k, t in enumerate(TIMES):
        rest = [T for T in range(1, 11) if T > t]
        last = max([T for T in range(0, 11) if T <= t])
        fwd = P0(last) - P0(10) - K * sum(P0(T) for T in rest) if t > 0 else 0
        if t == last:                                       # on a reset date the float leg is exactly 1 - P(t, end)
            assert dV[k] == pytest.approx(fwd, abs=4e-4)


def test_discounted_ee_on_reset_dates_is_the_swaption(payer):
    p = payer.profile()
    for t in (1.0, 3.0, 6.0):
        k = list(TIMES).index(t)
        swpn = M.swaption(t, [T for T in range(int(t) + 1, 11)], K)
        assert p["dEE"][k] == pytest.approx(swpn, rel=0.02)


def test_collateral_cuts_exposure_and_xva_is_symmetric():
    book = [Swap(K, 0, 10, 1e8)]
    un = exposures(M, book, np.linspace(0.25, 10, 40), CSA(), 20000)
    co = exposures(M, book, np.linspace(0.25, 10, 40), CSA(1e6, 1e5), 20000)
    assert co.profile()["EE"].max() < 0.25 * un.profile()["EE"].max()
    rec = exposures(M, [Swap(K, 0, 10, 1e8, payer=False)], np.linspace(0.25, 10, 40), CSA(), 20000)
    a = xva(un, hazard_c=0.03, hazard_own=0.01)
    b = xva(rec, hazard_c=0.01, hazard_own=0.03)
    assert a["CVA"] == pytest.approx(b["DVA"], rel=1e-9)      # same paths, mirrored trade and names
    assert a["CVA"] > 0 and a["FVA"] > 0
