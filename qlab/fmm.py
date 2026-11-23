"""SOFR forward market model (Lyashenko & Mercurio 2019), shifted-lognormal, spot measure.

Backward-looking compounded rates R_j over [T_{j-1}, T_j] keep moving during their accrual period
and are only known at T_j. The FMM extends the LIBOR market model to them by letting each rate's
volatility decay linearly to zero across its accrual period:

    d ln(R_j + s) = sigma_j g_j(t) dW_j + drift,   g_j = 1 before T_{j-1}, (T_j - t)/tau in accrual, 0 after

Under the spot measure (numeraire B(T_k) = prod_{i<=k} (1 + tau R_i(T_i)), the discretely
compounded bank account) the drift of R_j sums over the rates from the one currently accruing up
to j. Correlation is exp(-beta |T_i - T_j|).

  caplet_black        closed form: shifted Black with variance sigma^2 (T_{j-1} + tau/3)
  swaption_rebonato   Rebonato's frozen-weights approximation
  FMM.simulate        predictor-corrector log-Euler
  FMM.caplet / swaption / bermudan (Longstaff-Schwartz on the swap rate and annuity)
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, sqrt

import numpy as np
from scipy.stats import norm


def black(F, K, var, call=True):
    if var <= 0:
        return max(F - K, 0) if call else max(K - F, 0)
    sd = sqrt(var)
    d1 = (log(F / K) + 0.5 * var) / sd
    d2 = d1 - sd
    return F * norm.cdf(d1) - K * norm.cdf(d2) if call else K * norm.cdf(-d2) - F * norm.cdf(-d1)


@dataclass
class FMM:
    P0: callable
    T: np.ndarray            # accrual end dates T_1..T_N (T_0 = 0)
    sigma: np.ndarray        # one lognormal vol per rate
    beta: float = 0.1
    shift: float = 0.0

    def __post_init__(self):
        self.T = np.asarray(self.T, float)
        self.sigma = np.broadcast_to(np.asarray(self.sigma, float), self.T.shape).copy()
        self.Tm = np.concatenate([[0.0], self.T[:-1]])          # accrual starts
        self.tau = self.T - self.Tm
        P = np.array([self.P0(t) for t in np.concatenate([[0.0], self.T])])
        self.R0 = (P[:-1] / P[1:] - 1) / self.tau
        self.rho = np.exp(-self.beta * np.abs(self.T[:, None] - self.T[None, :]))
        w, V = np.linalg.eigh(self.rho)                          # works for rank-deficient (beta = 0)
        self.L = V * np.sqrt(np.clip(w, 0, None))

    def g(self, t):
        return np.clip((self.T - t) / self.tau, 0, 1)

    # ------------------------------------------------------- closed forms
    def caplet_black(self, j, K):
        """Value at 0 of tau (R_j(T_j) - K)^+ paid at T_j."""
        var = self.sigma[j] ** 2 * (self.Tm[j] + self.tau[j] / 3)
        return self.tau[j] * self.P0(self.T[j]) * black(self.R0[j] + self.shift, K + self.shift, var)

    def _swap(self, e, n):
        """Swap from T_e (index into accrual starts) to T_n: rates e..n-1 (0-based)."""
        idx = np.arange(e, n)
        P = np.array([self.P0(t) for t in self.T[idx]])
        A = np.sum(self.tau[idx] * P)
        S = (self.P0(self.Tm[e]) - P[-1]) / A
        return idx, A, S, P

    def swaption_rebonato(self, e, n, K, payer=True):
        idx, A, S, P = self._swap(e, n)
        w = self.tau[idx] * P / A
        Te = self.Tm[e]
        Rs = self.R0[idx] + self.shift
        v = np.outer(w * Rs * self.sigma[idx], w * Rs * self.sigma[idx]) * self.rho[np.ix_(idx, idx)]
        var = v.sum() * Te / (S + self.shift) ** 2
        return A * black(S + self.shift, K + self.shift, var, payer)

    # ------------------------------------------------------- simulation
    def simulate(self, n_paths, steps_per_period=4, seed=0, until=None):
        """Returns dict: grid times, R on the period grid (paths, dates, rates) where dates are
        T_0..T_N, the realised rates R_j(T_j), and the numeraire B at T_0..T_N."""
        rng = np.random.default_rng(seed)
        N = len(self.T)
        until = self.T[-1] if until is None else until
        x = np.tile(np.log(self.R0 + self.shift), (n_paths, 1))
        snap = [np.exp(x) - self.shift]
        realised = np.zeros((n_paths, N))
        t = 0.0
        for k in range(N):
            if self.Tm[k] >= until - 1e-12:
                break
            h = self.tau[k] / steps_per_period
            for _ in range(steps_per_period):
                z = rng.standard_normal((n_paths, N)) @ self.L.T
                g0, g1 = self.g(t), self.g(t + h)
                sg = self.sigma * np.sqrt((g0 * g0 + g0 * g1 + g1 * g1) / 3)   # exact mean of g^2 over the step
                beta_idx = np.searchsorted(self.T, t, side="right")        # currently accruing rate
                def drift(xx):
                    R = np.exp(xx) - self.shift
                    a = self.tau * sg * (R + self.shift) / (1 + self.tau * R)  # (paths, N)
                    a[:, :beta_idx] = 0
                    # mu_j = sg_j * sum_{i=beta..j} rho_ij a_i
                    return sg * (a @ np.triu(self.rho)) - 0.5 * sg * sg
                d0 = drift(x)
                xp = x + d0 * h + sg * sqrt(h) * z
                x = x + 0.5 * (d0 + drift(xp)) * h + sg * sqrt(h) * z
                t += h
            R = np.exp(x) - self.shift
            realised[:, k] = R[:, k]
            snap.append(R.copy())
        realised_cols = len(snap) - 1
        B = np.ones((n_paths, realised_cols + 1))
        for k in range(realised_cols):
            B[:, k + 1] = B[:, k] * (1 + self.tau[k] * realised[:, k])
        return {"R": np.stack(snap, 1), "realised": realised, "B": B}

    def caplet(self, j, K, n_paths=100_000, seed=0):
        sim = self.simulate(n_paths, seed=seed, until=self.T[j])
        pay = self.tau[j] * np.maximum(sim["realised"][:, j] - K, 0)
        return float(np.mean(pay / sim["B"][:, j + 1]))

    def _swap_value_at(self, R_at, e, n, K):
        """Payer swap value at T_e (in units of 1) from the rates observed at T_e."""
        P = np.cumprod(1 / (1 + self.tau[e:n] * R_at[:, e:n]), axis=1)
        A = (self.tau[e:n] * P).sum(1)
        return 1 - P[:, -1] - K * A, A, (1 - P[:, -1]) / A

    def swaption(self, e, n, K, payer=True, n_paths=100_000, seed=0):
        sim = self.simulate(n_paths, seed=seed, until=self.Tm[e])
        V, _, _ = self._swap_value_at(sim["R"][:, e, :], e, n, K)
        V = V if payer else -V
        return float(np.mean(np.maximum(V, 0) / sim["B"][:, e]))

    def bermudan(self, first, n, K, payer=True, n_paths=100_000, seed=0):
        """Exercise at any accrual start T_first..T_{n-1} into the swap to T_n (Longstaff-Schwartz,
        coefficients fitted on one path set and applied to a fresh one)."""
        def run(sim, coefs=None):
            cash = np.zeros(n_paths)
            fitted = {}
            for e in range(n - 1, first - 1, -1):
                V, A, S = self._swap_value_at(sim["R"][:, e, :], e, n, K)
                V = V if payer else -V
                disc = 1 / sim["B"][:, e]
                ex = V * disc                          # exercise value in numeraire units
                itm = V > 0
                if e == n - 1:
                    cash = np.where(itm, ex, 0.0)
                    continue
                X = np.column_stack([np.ones(n_paths), S, S * S, A, A * S, V, V * V])
                if coefs is None:
                    beta, *_ = np.linalg.lstsq(X[itm] , cash[itm] / disc[itm], rcond=None)
                    fitted[e] = beta
                else:
                    beta = coefs[e]
                cont = X @ beta * disc
                stop = itm & (ex > cont)
                cash = np.where(stop, ex, cash)
            return cash.mean(), fitted
        _, coefs = run(self.simulate(n_paths, seed=seed, until=self.Tm[n - 1]))
        v, _ = run(self.simulate(n_paths, seed=seed + 1, until=self.Tm[n - 1]), coefs)
        return float(v)


if __name__ == "__main__":
    P0 = lambda t: exp(-0.035 * t - 0.0015 * t * t / 2)
    T = np.arange(1, 11, dtype=float)                 # annual compounded SOFR periods, 10y
    m = FMM(P0, T, 0.25, beta=0.1, shift=0.01)
    print("caplet (annual compounded SOFR, ATM): closed form vs Monte Carlo")
    for j in [0, 2, 6]:
        K = m.R0[j]
        print(f"  period {m.Tm[j]:.0f}-{m.T[j]:.0f}y  {1e4 * m.caplet_black(j, K):7.2f}bp  MC {1e4 * m.caplet(j, K, 60_000):7.2f}bp")
    idx, A, S, _ = m._swap(3, 10)
    print(f"3y x 7y payer ATM ({S:.3%}): Rebonato {1e4 * m.swaption_rebonato(3, 10, S):.1f}bp, "
          f"MC {1e4 * m.swaption(3, 10, S, n_paths=60_000):.1f}bp")
    eu = max(m.swaption(e, 10, S, n_paths=40_000) for e in range(3, 10))
    print(f"Bermudan 3nc1 into 10y: {1e4 * m.bermudan(3, 10, S, n_paths=40_000):.1f}bp (best European {1e4 * eu:.1f}bp)")
