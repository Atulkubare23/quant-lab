"""Deep hedging trains for ~30 s per problem; the checks are about the economics, not the digits."""
import numpy as np
import pytest
from scipy.stats import norm

from qlab import aad
from qlab.deephedge import DeepHedger, Problem, bs_call, bs_delta_policy, evaluate


def test_network_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 3))
    W = aad.Var(rng.standard_normal((3, 4)))
    f = lambda W: aad.mean(aad.sigmoid(aad.tanh(aad.matmul(aad.Var(x), W))) * aad.abs_(aad.concat([aad.Var(x), aad.Var(x[:, :1])], 1)[:, :4] + 0.3))
    y = f(W)
    y.backward()
    g = W.adj.copy()
    h = 1e-6
    for i, j in [(0, 0), (1, 2), (2, 3)]:
        Wp, Wm = W.value.copy(), W.value.copy()
        Wp[i, j] += h; Wm[i, j] -= h
        fd = (f(aad.Var(Wp)).value - f(aad.Var(Wm)).value) / (2 * h)
        assert g[i, j] == pytest.approx(fd, rel=1e-6)


@pytest.fixture(scope="module")
def no_cost():
    p = Problem()
    return p, DeepHedger(p, seed=1).train(iters=300, batch=2048)


def test_without_costs_the_network_learns_black_scholes(no_cost):
    p, dh = no_cost
    deep, bs = evaluate(dh.policy, p), evaluate(bs_delta_policy(p), p)
    assert deep["price"] == pytest.approx(bs["price"], abs=0.08)
    assert deep["price"] == pytest.approx(bs_call(100, 100, p.T, 0.2), rel=0.06)
    S = np.linspace(95, 105, 11)
    t = 15 / 365
    d1 = (np.log(S / 100) + 0.5 * 0.04 * (p.T - t)) / (0.2 * np.sqrt(p.T - t))
    assert np.mean(np.abs(dh.delta(S, t) - norm.cdf(d1))) < 0.06


def test_with_costs_the_network_beats_delta_hedging():
    p = Problem(cost=0.005)
    dh = DeepHedger(p, seed=2).train(iters=300, batch=2048)
    deep, bs = evaluate(dh.policy, p), evaluate(bs_delta_policy(p), p)
    assert deep["price"] < bs["price"] - 0.2
