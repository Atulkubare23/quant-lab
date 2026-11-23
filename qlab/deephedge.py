"""Deep hedging (Buehler, Gonon, Teichmann & Wood 2019) in numpy, trained on the qlab.aad tape.

A network maps (log-moneyness, time left, current position) to the hedge position at each
rebalancing date. It is trained to minimise a convex risk measure of the hedged P&L

    X = -Z + sum_k delta_k (S_{k+1} - S_k) - c sum_k |delta_k - delta_{k-1}| S_k

where Z is the option payoff and c a proportional transaction cost (positions start and end flat).
With the entropic measure rho(X) = log E[exp(-lam X)] / lam, rho(X) is the indifference price: the
cash the hedger must receive to be as well off as without the trade. Without costs, in Black-Scholes,
the optimal hedge is the Black-Scholes delta and the price is the Black-Scholes price, which is how
the tests check the training. With costs the network learns a no-trade band; under Heston (one
hedging instrument, so the market is incomplete) it beats the model-free Black-Scholes delta hedge.
Paths are simulated under the risk-neutral measure with r = 0, as in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np
from scipy.stats import norm

from . import aad
from .aad import Var


# ---------------------------------------------------------------- markets
def gbm(S0, sigma, T, n_steps, n_paths, rng):
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    lr = -0.5 * sigma ** 2 * dt + sigma * sqrt(dt) * z
    return S0 * np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(lr, 1)], 1))


def heston(S0, v0, kappa, theta, xi, rho, T, n_steps, n_paths, rng, sub=4):
    """Full-truncation Euler with sub-steps; returns S on the hedging grid."""
    m = n_steps * sub
    dt = T / m
    S = np.full(n_paths, float(S0))
    v = np.full(n_paths, float(v0))
    out = [S.copy()]
    for i in range(m):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + sqrt(1 - rho * rho) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0)
        S = S * np.exp(-0.5 * vp * dt + np.sqrt(vp * dt) * z1)
        v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp * dt) * z2
        if (i + 1) % sub == 0:
            out.append(S.copy())
    return np.stack(out, 1)


# ---------------------------------------------------------------- network and optimiser
class MLP:
    def __init__(self, sizes, rng):
        self.params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.params += [Var(rng.standard_normal((a, b)) * sqrt(1.0 / a)), Var(np.zeros((1, b)))]

    def __call__(self, x):
        h = x
        n = len(self.params) // 2
        for i in range(n):
            h = aad.matmul(h, self.params[2 * i]) + self.params[2 * i + 1]
            if i < n - 1:
                h = aad.tanh(h)
        return h


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p.value) for p in params]
        self.v = [np.zeros_like(p.value) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, p in enumerate(self.p):
            g = p.adj if p.adj is not None else 0.0
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.value = p.value - self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------------------------------------------------------- hedging problem
@dataclass
class Problem:
    K: float = 100.0
    T: float = 30 / 365
    n_steps: int = 30
    cost: float = 0.0           # proportional cost per unit of notional traded
    lam: float = 1.0            # entropic risk aversion
    market: str = "gbm"
    S0: float = 100.0
    sigma: float = 0.2
    heston: tuple = (0.04, 1.0, 0.04, 2.0, -0.7)   # v0, kappa, theta, xi, rho

    def paths(self, n, rng):
        if self.market == "gbm":
            return gbm(self.S0, self.sigma, self.T, self.n_steps, n, rng)
        return heston(self.S0, *self.heston, self.T, self.n_steps, n, rng)

    def payoff(self, S):
        return np.maximum(S[:, -1] - self.K, 0)


def _features(S_k, t_k, prev, p: Problem):
    n = len(S_k)
    return np.column_stack([np.log(S_k / p.K) / (p.sigma * sqrt(p.T)), np.full(n, (p.T - t_k) / p.T)]), prev


def pnl(policy, S, p: Problem, tape=True):
    """Hedged P&L per path. policy(features: ndarray, prev: Var|ndarray) -> position."""
    n, m = S.shape[0], p.n_steps
    dt = p.T / m
    prev = Var(np.zeros((n, 1))) if tape else np.zeros((n, 1))
    gains = 0.0
    costs = 0.0
    for k in range(m):
        f, _ = _features(S[:, k], k * dt, None, p)
        pos = policy(f, prev)
        dS = (S[:, k + 1] - S[:, k])[:, None]
        gains = gains + pos * dS
        if p.cost:
            trade = aad.abs_(pos - prev) if tape else np.abs(pos - prev)
            costs = costs + trade * (p.cost * S[:, k][:, None])
        prev = pos
    if p.cost:
        close = aad.abs_(prev) if tape else np.abs(prev)
        costs = costs + close * (p.cost * S[:, -1][:, None])
    return gains - costs - p.payoff(S)[:, None]


def entropic(X, lam, tape=True):
    if tape:
        y = X * (-lam)
        c = float(y.value.max())
        return (aad.log(aad.mean(aad.exp(y - c))) + c) * (1 / lam)
    y = -lam * X
    c = y.max()
    return (np.log(np.mean(np.exp(y - c))) + c) / lam


class DeepHedger:
    def __init__(self, p: Problem, hidden=(32, 32), seed=0):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.net = MLP([3, *hidden, 1], self.rng)
        self.history = []

    def policy(self, f, prev):
        if isinstance(prev, Var):
            x = aad.concat([Var(f), prev], axis=1)
            return self.net(x) + 0.5                      # start near an ATM delta
        x = Var(np.column_stack([f, prev]))
        return self.net(x).value + 0.5

    def train(self, iters=600, batch=4096, lr=5e-3):
        opt = Adam(self.net.params, lr)
        for i in range(iters):
            if i == int(iters * 0.7):
                opt.lr = lr / 5
            S = self.p.paths(batch, self.rng)
            loss = entropic(pnl(self.policy, S, self.p), self.p.lam)
            loss.backward()
            opt.step()
            self.history.append(float(loss.value))
        return self

    def price(self, n=100_000, seed=123):
        S = self.p.paths(n, np.random.default_rng(seed))
        return float(entropic(pnl(self.policy, S, self.p, tape=False), self.p.lam, tape=False))

    def delta(self, S, t, prev=None):
        S = np.atleast_1d(np.asarray(S, float))
        f, _ = _features(S, t, None, self.p)
        pv = np.full((len(S), 1), 0.5 if prev is None else prev)
        return self.policy(f, pv)[:, 0]


def bs_delta_policy(p: Problem, sigma=None):
    sig = p.sigma if sigma is None else sigma
    def pol(f, prev):
        lm, tl = f[:, 0] * p.sigma * sqrt(p.T), f[:, 1] * p.T
        tl = np.maximum(tl, 1e-12)
        d1 = (lm + 0.5 * sig * sig * tl) / (sig * np.sqrt(tl))
        return norm.cdf(d1)[:, None]
    return pol


def evaluate(policy, p: Problem, n=100_000, seed=123):
    S = p.paths(n, np.random.default_rng(seed))
    X = pnl(policy, S, p, tape=False)
    return {"price": float(entropic(X, p.lam, tape=False)), "std": float(X.std()),
            "cvar95": float(-np.mean(np.sort(X[:, 0])[: n // 20]))}


def bs_call(S, K, T, sigma):
    d1 = (log(S / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrt(T))
    return S * norm.cdf(d1) - K * norm.cdf(d1 - sigma * sqrt(T))


if __name__ == "__main__":
    import time
    for label, p in [("Black-Scholes, no costs", Problem()),
                     ("Black-Scholes, 0.5% costs", Problem(cost=0.005)),
                     ("Heston, no costs", Problem(market="heston"))]:
        t0 = time.perf_counter()
        dh = DeepHedger(p).train()
        deep = evaluate(dh.policy, p)
        bs = evaluate(bs_delta_policy(p), p)
        print(f"{label:28s} trained in {time.perf_counter() - t0:4.0f}s | indifference price: deep {deep['price']:.3f}, "
              f"BS delta {bs['price']:.3f} | P&L std deep {deep['std']:.3f}, BS {bs['std']:.3f}")
    print(f"Black-Scholes price {bs_call(100, 100, 30 / 365, 0.2):.3f}")
