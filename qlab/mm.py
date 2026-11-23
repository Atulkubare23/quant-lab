"""Market making: Avellaneda & Stoikov (2008) against a symmetric quoter.

Mid-price is arithmetic Brownian motion, dS = sigma dW. A quote at distance delta from the mid is
hit with intensity A exp(-k delta). The optimal (exponential-utility, risk aversion gamma) quotes:

    reservation price  r = S - q gamma sigma^2 (T - t)
    total spread       delta_a + delta_b = gamma sigma^2 (T - t) + (2/gamma) ln(1 + gamma/k)

so the quoter skews both prices away from its inventory q. The benchmark quotes the same total
spread centred on the mid. Also: the Gueant-Lehalle-Fernandez-Tapia (2013) asymptotic quotes with
an inventory limit, which do not depend on the horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np


@dataclass
class Market:
    S0: float = 100.0
    sigma: float = 2.0          # price units per sqrt(year of the session), as in the paper
    A: float = 140.0
    k: float = 1.5
    T: float = 1.0
    dt: float = 0.005


def as_quotes(S, q, t, m: Market, gamma):
    tau = m.T - t
    r = S - q * gamma * m.sigma ** 2 * tau
    half = 0.5 * (gamma * m.sigma ** 2 * tau + 2 / gamma * log(1 + gamma / m.k))
    return r - half, r + half             # bid, ask


def glft_quotes(S, q, m: Market, gamma, Q=None):
    """Gueant-Lehalle-Fernandez-Tapia closed-form approximation (no horizon dependence)."""
    c = 1 / gamma * log(1 + gamma / m.k)
    w = sqrt(m.sigma ** 2 * gamma / (2 * m.k * m.A) * (1 + gamma / m.k) ** (1 + m.k / gamma))
    bid = S - (c + (2 * q + 1) / 2 * w)
    ask = S + (c - (2 * q - 1) / 2 * w)
    if Q is not None:
        if q >= Q:
            bid = -np.inf
        if q <= -Q:
            ask = np.inf
    return bid, ask


def simulate(m: Market, strategy="as", gamma=0.1, n_paths=1000, seed=0, Q=None):
    """Returns terminal P&L (cash + q S_T) and terminal inventory per path."""
    rng = np.random.default_rng(seed)
    n = int(round(m.T / m.dt))
    S = np.full(n_paths, m.S0)
    cash = np.zeros(n_paths)
    q = np.zeros(n_paths)
    for i in range(n):
        t = i * m.dt
        if strategy == "as":
            bid, ask = as_quotes(S, q, t, m, gamma)
        elif strategy == "symmetric":
            b, a = as_quotes(S, 0.0, t, m, gamma)
            bid, ask = S - (a - b) / 2, S + (a - b) / 2
        elif strategy == "glft":
            out = [glft_quotes(s, qq, m, gamma, Q) for s, qq in zip(S, q)]
            bid, ask = np.array([o[0] for o in out]), np.array([o[1] for o in out])
        else:
            raise ValueError(strategy)
        db, da = S - bid, ask - S
        pb = m.A * np.exp(-m.k * db) * m.dt
        pa = m.A * np.exp(-m.k * da) * m.dt
        hit_b = rng.random(n_paths) < np.clip(pb, 0, 1)
        hit_a = rng.random(n_paths) < np.clip(pa, 0, 1)
        q += hit_b.astype(float) - hit_a
        cash += np.where(hit_a, ask, 0) - np.where(hit_b, bid, 0)
        S = S + m.sigma * sqrt(m.dt) * np.where(rng.random(n_paths) < 0.5, 1.0, -1.0)
    return cash + q * S, q


def summary(m: Market, gamma, n_paths=1000, seed=0):
    out = {}
    for s in ("as", "symmetric"):
        pnl, q = simulate(m, s, gamma, n_paths, seed)
        out[s] = {"mean": pnl.mean(), "std": pnl.std(), "q_mean": q.mean(), "q_std": q.std()}
    return out


if __name__ == "__main__":
    m = Market()
    print("Avellaneda-Stoikov table (S0=100, sigma=2, A=140, k=1.5, T=1, dt=0.005), 1000 paths")
    print(f"{'gamma':>6} {'strategy':>10} {'mean P&L':>9} {'std P&L':>8} {'std q':>6}")
    for g in (0.01, 0.1, 0.5):
        for s, v in summary(m, g).items():
            print(f"{g:6} {s:>10} {v['mean']:9.2f} {v['std']:8.2f} {v['q_std']:6.2f}")
