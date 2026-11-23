from math import exp

import numpy as np
import pytest

from qlab.fmm import FMM

P0 = lambda t: exp(-0.035 * t - 0.0015 * t * t / 2)
T = np.arange(1, 11, dtype=float)


@pytest.fixture(scope="module")
def m():
    return FMM(P0, T, 0.25, beta=0.1, shift=0.01)


def test_numeraire_reprices_the_curve(m):
    sim = m.simulate(100_000, seed=4)
    for k in [1, 3, 7, 10]:
        assert np.mean(1 / sim["B"][:, k]) == pytest.approx(P0(T[k - 1]), rel=2e-3)


def test_caplets_on_compounded_rates_match_the_closed_form(m):
    for j in [0, 2, 6]:
        K = m.R0[j]
        assert m.caplet(j, K, 100_000, seed=1) == pytest.approx(m.caplet_black(j, K), rel=0.025)
    # the accruing period's rate still moves: a 0-1y caplet is worth more than if it fixed at the start
    assert m.caplet_black(0, m.R0[0]) > 0


def test_swaptions_match_rebonato_and_bermudan_bounds():
    one = FMM(P0, T, 0.25, beta=0.0, shift=0.01)          # one factor: Rebonato is near exact
    S = one._swap(3, 10)[2]
    for e in (3, 6):
        assert one.swaption(e, 10, S, n_paths=60_000) == pytest.approx(one.swaption_rebonato(e, 10, S), rel=0.02)
    eu = [one.swaption(e, 10, S, n_paths=40_000) for e in range(3, 10)]
    berm = one.bermudan(3, 10, S, n_paths=40_000)
    assert max(eu) < berm < sum(eu)
    assert one.bermudan(9, 10, S, n_paths=40_000) == pytest.approx(eu[-1], rel=0.03)   # one date = European
    flat = FMM(P0, T, 0.002, beta=0.0, shift=0.01)        # no vol: best intrinsic exercise
    eu0 = [flat.swaption(e, 10, S, n_paths=20_000) for e in range(3, 10)]
    assert flat.bermudan(3, 10, S, n_paths=20_000) == pytest.approx(max(eu0), rel=0.01)
