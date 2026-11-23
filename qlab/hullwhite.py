"""Hull-White one-factor model fitted to today's discount curve.

    r(t) = x(t) + phi(t),   dx = -a x dt + sigma dW,   x(0) = 0

  bond(t, T, x)       closed-form zero-coupon bond (Brigo-Mercurio x-form)
  zbp / zbc           zero-coupon bond options
  caplet, cap         caplets as puts on zero bonds
  swaption            European swaption by Jamshidian's decomposition
  Tree                trinomial tree (Hull-White 1994) fitted to the curve; European and
                      Bermudan swaptions by backward induction
  simulate            exact joint simulation of x(t) and its integral (for Monte Carlo and xVA)
  calibrate_sigma     match a market swaption price given a mean reversion
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, log, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass
class HW:
    a: float
    sigma: float
    P0: callable          # today's discount curve, t -> P(0, t); scalar functions are vectorised

    def __post_init__(self):
        f = self.P0
        def vec(t):
            if np.ndim(t) == 0:
                return float(f(float(t)))
            return np.array([float(f(float(u))) for u in np.ravel(t)]).reshape(np.shape(t))
        self.P0 = vec

    def B(self, t, T):
        return (1 - np.exp(-self.a * (T - t))) / self.a

    def V(self, t, T):
        a, s, h = self.a, self.sigma, T - t
        return s * s / (a * a) * (h + 2 / a * np.exp(-a * h) - 1 / (2 * a) * np.exp(-2 * a * h) - 3 / (2 * a))

    def bond(self, t, T, x):
        """P(t, T) given x(t); exact fit to P0 at t = 0."""
        return self.P0(T) / self.P0(t) * np.exp(0.5 * (self.V(t, T) - self.V(0, T) + self.V(0, t)) - self.B(t, T) * x)

    def _sp(self, T, S):
        a = self.a
        return self.sigma * sqrt((1 - exp(-2 * a * T)) / (2 * a)) * self.B(T, S)

    def zbp(self, T, S, X):
        """Put, expiry T, on a zero bond maturing S, strike X."""
        sp = self._sp(T, S)
        h = log(self.P0(S) / (self.P0(T) * X)) / sp + sp / 2
        return X * self.P0(T) * norm.cdf(-h + sp) - self.P0(S) * norm.cdf(-h)

    def zbc(self, T, S, X):
        sp = self._sp(T, S)
        h = log(self.P0(S) / (self.P0(T) * X)) / sp + sp / 2
        return self.P0(S) * norm.cdf(h) - X * self.P0(T) * norm.cdf(h - sp)

    def caplet(self, t1, t2, K):
        tau = t2 - t1
        return (1 + K * tau) * self.zbp(t1, t2, 1 / (1 + K * tau))

    def cap(self, T, K, freq=4):
        ts = np.arange(1, int(round(T * freq)) + 1) / freq
        return sum(self.caplet(t0, t1, K) for t0, t1 in zip(ts[:-1], ts[1:]))

    def swaption(self, T0, pay_times, K, payer=True):
        """European swaption: exercise at T0 into a swap paying K * tau at each pay time (annual-style)."""
        ts = np.asarray(pay_times, float)
        tau = np.diff(np.concatenate([[T0], ts]))
        c = K * tau
        c[-1] += 1
        f = lambda x: float(np.sum(c * self.bond(T0, ts, x))) - 1
        xs = brentq(f, -1.0, 1.0)
        X = self.bond(T0, ts, xs)
        opt = self.zbp if payer else self.zbc
        return float(sum(ci * opt(T0, Ti, Xi) for ci, Ti, Xi in zip(c, ts, X)))

    def simulate(self, times, n_paths, seed=0):
        """Exact simulation of x and I = int_0^t x ds on the given grid. Returns (x, D) where
        D = exp(-int_0^t r ds) is the bank-account discount factor on each path."""
        rng = np.random.default_rng(seed)
        a, s = self.a, self.sigma
        times = np.asarray(times, float)
        x = np.zeros((n_paths, len(times)))
        I = np.zeros_like(x)
        prev = 0.0
        for k, t in enumerate(times):
            h = t - prev
            if h > 0:
                e, Bh = exp(-a * h), (1 - exp(-a * h)) / a
                v1 = s * s / (2 * a) * (1 - e * e)
                v2 = s * s / (a * a) * (h - 2 * Bh + (1 - e * e) / (2 * a))
                c12 = s * s / (2 * a * a) * (1 - e) ** 2
                L = np.linalg.cholesky(np.array([[v1, c12], [c12, v2]]))
                z = rng.standard_normal((n_paths, 2)) @ L.T
                xp = x[:, k - 1] if k else np.zeros(n_paths)
                Ip = I[:, k - 1] if k else np.zeros(n_paths)
                x[:, k] = xp * e + z[:, 0]
                I[:, k] = Ip + xp * Bh + z[:, 1]
            prev = t
        # E[exp(-I)] = exp(V(0,t)/2), so this D is unbiased for P0(t)
        D = np.array([self.P0(t) for t in times])[None, :] * np.exp(-I - 0.5 * self.V(0, times)[None, :])
        return x, D


class Tree:
    """Trinomial tree in x with Hull's branching; alpha_i fitted so that the tree reprices P0."""

    def __init__(self, m: HW, T: float, steps_per_year: int = 50):
        self.m, self.dt = m, 1.0 / steps_per_year
        self.N = int(round(T * steps_per_year))
        a, dt = m.a, self.dt
        self.M = exp(-a * dt) - 1
        V = m.sigma ** 2 * (1 - exp(-2 * a * dt)) / (2 * a)
        self.dx = sqrt(3 * V)
        self.jmax = max(int(ceil(0.184 / (a * dt))), 1)
        j = np.arange(-self.jmax, self.jmax + 1)
        M = self.M
        pu = 1 / 6 + (j * j * M * M + j * M) / 2
        pm = 2 / 3 - j * j * M * M
        pd = 1 / 6 + (j * j * M * M - j * M) / 2
        k = np.zeros_like(j)                 # middle child offset from j
        top, bot = j == self.jmax, j == -self.jmax
        jt, jb = j[top], j[bot]
        pu[top], pm[top], pd[top] = 7 / 6 + (jt * jt * M * M + 3 * jt * M) / 2, -1 / 3 - jt * jt * M * M - 2 * jt * M, 1 / 6 + (jt * jt * M * M + jt * M) / 2
        pu[bot], pm[bot], pd[bot] = 1 / 6 + (jb * jb * M * M - jb * M) / 2, -1 / 3 - jb * jb * M * M + 2 * jb * M, 7 / 6 + (jb * jb * M * M - 3 * jb * M) / 2
        k[top], k[bot] = -1, 1
        self.j, self.p, self.k = j, (pu, pm, pd), k
        self._fit()

    def _width(self, i):
        return min(i, self.jmax)

    def _fit(self):
        nJ, J0, dt = len(self.j), self.jmax, self.dt
        Q = np.zeros(nJ)
        Q[J0] = 1.0
        self.alpha = np.zeros(self.N + 1)
        for i in range(self.N + 1):
            w = self._width(i)
            idx = np.arange(J0 - w, J0 + w + 1)
            self.alpha[i] = (log(np.sum(Q[idx] * np.exp(-self.j[idx] * self.dx * dt))) - log(self.m.P0((i + 1) * dt))) / dt
            if i == self.N:
                break
            disc = Q * np.exp(-(self.alpha[i] + self.j * self.dx) * dt)
            Qn = np.zeros(nJ)
            for pr, off in zip(self.p, (1, 0, -1)):
                tgt = idx + self.k[idx] + off
                np.add.at(Qn, tgt, disc[idx] * pr[idx])
            Q = Qn

    def _roll(self, v, i):
        """Value at step i from values at step i+1."""
        J0, w = self.jmax, self._width(i)
        idx = np.arange(J0 - w, J0 + w + 1)
        out = np.zeros_like(v)
        exp_v = sum(pr[idx] * v[idx + self.k[idx] + off] for pr, off in zip(self.p, (1, 0, -1)))
        out[idx] = exp_v * np.exp(-(self.alpha[i] + self.j[idx] * self.dx) * self.dt)
        return out

    def zero(self, T):
        n = int(round(T / self.dt))
        v = np.ones(len(self.j))
        for i in range(n - 1, -1, -1):
            v = self._roll(v, i)
        return v[self.jmax]

    def bermudan(self, exercise_times, pay_times, K, payer=True):
        """Option to enter, at any exercise time, the swap paying K * tau on the remaining pay times.
        A single exercise time gives the European swaption."""
        dt = self.dt
        pay = np.asarray(pay_times, float)
        ex = set(int(round(t / dt)) for t in exercise_times)
        start = min(exercise_times)
        tau = np.diff(np.concatenate([[start], pay]))
        cash = {int(round(t / dt)): K * ta for t, ta in zip(pay, tau)}
        cash[int(round(pay[-1] / dt))] += 1.0
        nJ = len(self.j)
        bond = np.zeros(nJ)
        opt = np.zeros(nJ)
        n = max(cash)
        for i in range(n, -1, -1):
            if i < n:
                bond, opt = self._roll(bond, i), self._roll(opt, i)
            if i in ex:
                w = self._width(i)
                idx = np.arange(self.jmax - w, self.jmax + w + 1)
                exv = (1 - bond[idx]) if payer else (bond[idx] - 1)
                opt[idx] = np.maximum(opt[idx], exv)
            if i in cash:
                bond = bond + cash[i]
        return opt[self.jmax]


def bachelier_swaption(F, K, T, normal_vol, annuity, payer=True):
    s = normal_vol * sqrt(T)
    d = (F - K) / s
    w = 1 if payer else -1
    return annuity * (w * (F - K) * norm.cdf(w * d) + s * norm.pdf(d))


def calibrate_sigma(a, P0, T0, pay_times, K, target, payer=True):
    return brentq(lambda s: HW(a, s, P0).swaption(T0, pay_times, K, payer) - target, 1e-5, 0.1)


if __name__ == "__main__":
    P0 = lambda t: exp(-0.035 * t - 0.001 * t * t / 2)
    m = HW(0.05, 0.01, P0)
    pay = [6, 7, 8, 9, 10.0]
    ann = sum(P0(t) for t in pay)
    F = (P0(5) - P0(10)) / ann
    print(f"5y x 5y forward swap rate {F:.4%}")
    tree = Tree(m, 10, 50)
    print(f"European payer ATM: Jamshidian {m.swaption(5, pay, F):.6f}, tree {tree.bermudan([5], pay, F):.6f}")
    berm = tree.bermudan([5, 6, 7, 8, 9], pay, F)
    print(f"Bermudan payer 5nc1 into 10y: {berm:.6f}  (exercise premium {berm - m.swaption(5, pay, F):.6f})")
    target = bachelier_swaption(F, F, 5, 0.0090, ann)
    print(f"sigma calibrated to 90bp normal vol: {calibrate_sigma(0.05, P0, 5, pay, F, target):.5f}")
