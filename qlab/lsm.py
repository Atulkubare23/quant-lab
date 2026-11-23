"""American and Bermudan options by Longstaff-Schwartz least-squares Monte Carlo.

  gbm_paths       correlated GBM on an exercise grid (exact steps)
  lsm             regression of continuation values on basis functions of in-the-money paths;
                  the exercise rule is then applied to a fresh set of paths (low-biased estimate)
  crr_american    Cox-Ross-Rubinstein tree, the benchmark for one asset
  basis_poly      1, S, S^2, ... for one asset; for several, sorted prices with squares and cross
                  terms plus the payoff (the usual choice for max-calls)
"""
from __future__ import annotations

from math import exp, sqrt

import numpy as np


def gbm_paths(S0, sigma, r, q, T, n_steps, n_paths, corr=None, seed=0):
    S0, sigma, q = (np.atleast_1d(np.asarray(v, float)) for v in (S0, sigma, q))
    d = len(S0)
    q = np.broadcast_to(q, (d,))
    sigma = np.broadcast_to(sigma, (d,))
    L = np.linalg.cholesky(np.asarray(corr, float)) if corr is not None and d > 1 else np.eye(d)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths // 2, n_steps, d)) @ L.T
    z = np.concatenate([z, -z])                          # antithetic
    inc = (r - q - 0.5 * sigma ** 2) * dt + sigma * sqrt(dt) * z
    logS = np.log(S0) + np.cumsum(inc, axis=1)
    return np.exp(logS)                                  # (paths, steps, assets), excludes t = 0


def basis_poly(S, payoff, scale, degree=3):
    """Fixed scaling (the strike) so that training and test paths share one set of coefficients."""
    if S.shape[1] == 1:
        x = S[:, 0] / scale
        return np.column_stack([x ** k for k in range(degree + 1)])
    s = np.sort(S, axis=1)[:, ::-1] / scale
    cols = [np.ones(len(S))]
    for i in range(s.shape[1]):
        cols += [s[:, i], s[:, i] ** 2]
    for i in range(s.shape[1]):
        for j in range(i + 1, s.shape[1]):
            cols.append(s[:, i] * s[:, j])
    pn = payoff / scale
    cols += [pn, pn ** 2]
    return np.column_stack(cols)


def _rule(paths, payoff_fn, disc, basis):
    """Backward induction; returns regression coefficients per exercise date."""
    n_paths, n_steps, _ = paths.shape
    cash = payoff_fn(paths[:, -1])
    coefs = [None] * n_steps
    for k in range(n_steps - 2, -1, -1):
        cash = cash * disc
        ex = payoff_fn(paths[:, k])
        itm = ex > 0
        if itm.sum() < 10:
            continue
        X = basis(paths[itm, k], ex[itm])
        beta, *_ = np.linalg.lstsq(X, cash[itm], rcond=None)
        coefs[k] = beta
        cont = X @ beta
        stop = ex[itm] > cont
        idx = np.where(itm)[0][stop]
        cash[idx] = ex[itm][stop]
    return coefs


def lsm(S0, K, sigma, r, q, T, n_ex, payoff="put", corr=None, n_paths=100_000, basis=basis_poly, seed=0):
    """Bermudan option with n_ex equally spaced exercise dates (the last is expiry).
    payoff: 'put', 'call' (single asset) or 'maxcall' (several). Returns (price, standard error)."""
    fn = {
        "put": lambda S: np.maximum(K - S[:, 0], 0),
        "call": lambda S: np.maximum(S[:, 0] - K, 0),
        "maxcall": lambda S: np.maximum(S.max(axis=1) - K, 0),
    }[payoff]
    disc = exp(-r * T / n_ex)
    basis_k = lambda S, ex: basis(S, ex, K)
    train = gbm_paths(S0, sigma, r, q, T, n_ex, n_paths, corr, seed)
    coefs = _rule(train, fn, disc, basis_k)
    test = gbm_paths(S0, sigma, r, q, T, n_ex, n_paths, corr, seed + 1)
    n = len(test)
    value = np.zeros(n)
    alive = np.ones(n, bool)
    for k in range(n_ex):
        ex = fn(test[:, k])
        if k == n_ex - 1:
            stop = alive & (ex > 0)
        elif coefs[k] is None:
            continue
        else:
            itm = alive & (ex > 0)
            stop = np.zeros(n, bool)
            if itm.any():
                cont = basis_k(test[itm, k], ex[itm]) @ coefs[k]
                stop[np.where(itm)[0][ex[itm] > cont]] = True
        value[stop] = ex[stop] * disc ** (k + 1)
        alive &= ~stop
    # the option can also be exercised immediately at t = 0
    v = value.mean()
    S00 = np.atleast_1d(np.asarray(S0, float))[None, :]
    return max(v, float(fn(S00)[0])), value.std() / sqrt(n)


def crr_american(S, K, sigma, r, q, T, n=2000, call=False):
    dt = T / n
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    p = (exp((r - q) * dt) - d) / (u - d)
    disc = exp(-r * dt)
    ST = S * u ** np.arange(n, -n - 1, -2)
    V = np.maximum(ST - K, 0) if call else np.maximum(K - ST, 0)
    for i in range(n - 1, -1, -1):
        ST = S * u ** np.arange(i, -i - 1, -2)
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        V = np.maximum(V, (ST - K) if call else (K - ST))
    return float(V[0])


if __name__ == "__main__":
    print("American put, K=40, r=6%, T=1 (Longstaff-Schwartz table 1)")
    for S, sig in [(36, 0.2), (40, 0.2), (44, 0.4)]:
        v, se = lsm(S, 40, sig, 0.06, 0.0, 1.0, 50)
        print(f"  S={S} vol={sig:.0%}: LSM {v:.3f} +/- {se:.3f}   CRR {crr_american(S, 40, sig, 0.06, 0, 1.0):.3f}")
    v, se = lsm([100, 100], 100, 0.2, 0.05, 0.10, 3.0, 9, "maxcall", np.eye(2), 200_000)
    print(f"Bermudan max-call on two assets (Broadie-Glasserman benchmark 13.90): {v:.3f} +/- {se:.3f}")
