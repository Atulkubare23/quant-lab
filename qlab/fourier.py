"""Fourier pricing engine: one characteristic function, three pricers.

Every model is a characteristic function of X = ln(S_T / S_0) under the risk-neutral measure,
with the martingale drift built in (E[S_T] = S_0 e^{(r-q)T}). Pricers:

  carr_madan   damped call transform on an FFT grid (Carr & Madan 1999), Simpson weights
  cos          Fang & Oosterlee (2008) cosine expansion; puts priced, calls by parity
  lewis        Lewis (2001) single integral along Im(u) = -1/2, adaptive quadrature

Models: Black-Scholes, Merton jump-diffusion, Kou double-exponential, Variance Gamma, NIG,
CGMY, Heston ("little trap" form), Bates (Heston + Merton jumps).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gamma as G
from scipy.stats import norm


# ---------------------------------------------------------------- models
class Model:
    """Levy-type models give psi(u), the exponent of one unit of time; drift is added here."""

    def psi(self, u):
        raise NotImplementedError

    def cf(self, u, T, r, q):
        u = np.asarray(u, dtype=complex)
        w = -self.psi(np.array(-1j)).real              # martingale correction: E[e^X] = e^{(r-q)T}
        return np.exp(1j * u * (r - q + w) * T + T * self.psi(u))


@dataclass
class BS(Model):
    sigma: float

    def psi(self, u):
        return -0.5 * self.sigma ** 2 * u * u


@dataclass
class Merton(Model):
    sigma: float
    lam: float
    mu_j: float
    delta_j: float

    def psi(self, u):
        return (-0.5 * self.sigma ** 2 * u * u
                + self.lam * (np.exp(1j * u * self.mu_j - 0.5 * self.delta_j ** 2 * u * u) - 1))


@dataclass
class Kou(Model):
    sigma: float
    lam: float
    p: float
    eta1: float          # > 1, mean up-jump 1/eta1
    eta2: float          # > 0, mean down-jump 1/eta2

    def psi(self, u):
        jump = self.p * self.eta1 / (self.eta1 - 1j * u) + (1 - self.p) * self.eta2 / (self.eta2 + 1j * u) - 1
        return -0.5 * self.sigma ** 2 * u * u + self.lam * jump


@dataclass
class VG(Model):
    sigma: float
    nu: float
    theta: float

    def psi(self, u):
        return -np.log(1 - 1j * u * self.theta * self.nu + 0.5 * self.sigma ** 2 * self.nu * u * u) / self.nu


@dataclass
class NIG(Model):
    alpha: float
    beta: float
    delta: float

    def psi(self, u):
        a, b = self.alpha, self.beta
        return self.delta * (np.sqrt(a * a - b * b + 0j) - np.sqrt(a * a - (b + 1j * u) ** 2))


@dataclass
class CGMY(Model):
    C: float
    G_: float
    M: float
    Y: float             # < 2, != 0, 1

    def psi(self, u):
        C, Gm, M, Y = self.C, self.G_, self.M, self.Y
        return C * G(-Y) * ((M - 1j * u) ** Y - M ** Y + (Gm + 1j * u) ** Y - Gm ** Y)


@dataclass
class Heston(Model):
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def _heston(self, u, T):
        k, th, xi, rho, v0 = self.kappa, self.theta, self.xi, self.rho, self.v0
        b = k - rho * xi * 1j * u
        d = np.sqrt(b * b + xi * xi * (1j * u + u * u))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = k * th / xi ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / xi ** 2 * (1 - e) / (1 - g * e)
        return C + D * v0

    def cf(self, u, T, r, q):
        u = np.asarray(u, dtype=complex)
        return np.exp(1j * u * (r - q) * T + self._heston(u, T))


@dataclass
class Bates(Heston):
    lam: float = 0.0
    mu_j: float = 0.0
    delta_j: float = 0.0

    def cf(self, u, T, r, q):
        u = np.asarray(u, dtype=complex)
        k = exp(self.mu_j + 0.5 * self.delta_j ** 2) - 1
        jump = self.lam * T * (np.exp(1j * u * self.mu_j - 0.5 * self.delta_j ** 2 * u * u) - 1 - 1j * u * k)
        return np.exp(1j * u * (r - q) * T + self._heston(u, T) + jump)


# ---------------------------------------------------------------- closed forms used as checks
def bs_price(S, K, T, r, q, sigma, call=True):
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if call:
        return S * exp(-q * T) * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S * exp(-q * T) * norm.cdf(-d1)


def merton_series(S, K, T, r, q, m: Merton, n_terms=80):
    """Merton (1976): a Poisson mixture of Black-Scholes prices."""
    k = exp(m.mu_j + 0.5 * m.delta_j ** 2) - 1
    lp = m.lam * (1 + k)
    out, w = 0.0, exp(-lp * T)
    for n in range(n_terms):
        sig = sqrt(m.sigma ** 2 + n * m.delta_j ** 2 / T)
        rn = r - m.lam * k + n * log(1 + k) / T
        out += w * bs_price(S, K, T, rn, q, sig)
        w *= lp * T / (n + 1)
    return out


def implied_vol(price, S, K, T, r, q, call=True):
    lo = max(S * exp(-q * T) - K * exp(-r * T), 0) if call else max(K * exp(-r * T) - S * exp(-q * T), 0)
    if price <= lo + 1e-12:
        return float("nan")
    return brentq(lambda s: bs_price(S, K, T, r, q, s, call) - price, 1e-4, 5.0, xtol=1e-12)


# ---------------------------------------------------------------- pricers
def carr_madan(model, S, K, T, r, q, alpha=1.5, N=2 ** 12, eta=0.25):
    """Calls on many strikes from one FFT. Returns an array matching K."""
    K = np.atleast_1d(np.asarray(K, float))
    lam = 2 * pi / (N * eta)
    b = N * lam / 2
    v = eta * np.arange(N)
    u = v - (alpha + 1) * 1j
    phi = np.exp(1j * u * log(S)) * model.cf(u, T, r, q)       # cf of ln S_T
    psi = exp(-r * T) * phi / (alpha ** 2 + alpha - v ** 2 + 1j * (2 * alpha + 1) * v)
    w = (3 + (-1) ** (np.arange(N) + 1)) / 3.0
    w[0] = 1 / 3
    x = np.exp(1j * b * v) * psi * eta * w
    k = -b + lam * np.arange(N)
    calls = np.exp(-alpha * k) / pi * np.fft.fft(x).real
    return np.interp(np.log(K), k, calls)


def _cumulants(model, T, r, q, h=1e-3):
    lp = lambda u: np.log(model.cf(np.array(u), T, r, q))
    c1 = ((lp(h) - lp(-h)) / (2j * h)).real
    c2 = (-(lp(h) - 2 * lp(0.0) + lp(-h)) / h ** 2).real
    c4 = ((lp(2 * h) - 4 * lp(h) + 6 * lp(0.0) - 4 * lp(-h) + lp(-2 * h)) / h ** 4).real
    return c1, max(c2, 1e-12), abs(c4)


def cos(model, S, K, T, r, q, N=256, L=12.0, call=True):
    """Fang-Oosterlee COS. Puts are expanded (bounded payoff), calls come from put-call parity."""
    K = np.atleast_1d(np.asarray(K, float))
    c1, c2, c4 = _cumulants(model, T, r, q)
    half = L * sqrt(c2 + sqrt(c4))
    a, b = c1 - half, c1 + half
    k = np.arange(N)
    w = k * pi / (b - a)
    # put payoff on y = ln(S_T/K): K (1 - e^y)^+ on [a, 0]
    def chi(c, d):
        return (1 / (1 + w ** 2)) * (np.cos(w * (d - a)) * exp(d) - np.cos(w * (c - a)) * exp(c)
                                     + w * np.sin(w * (d - a)) * exp(d) - w * np.sin(w * (c - a)) * exp(c))
    def psi_(c, d):
        out = np.empty(N)
        out[0] = d - c
        out[1:] = (np.sin(w[1:] * (d - a)) - np.sin(w[1:] * (c - a))) / w[1:]
        return out
    Vk = 2 / (b - a) * (-chi(a, 0.0) + psi_(a, 0.0))
    phi = model.cf(w, T, r, q)
    x = np.log(S / K)
    terms = phi[None, :] * np.exp(1j * w[None, :] * (x[:, None] - a)) * Vk[None, :]
    terms[:, 0] *= 0.5
    put = K * exp(-r * T) * terms.sum(axis=1).real
    if call:
        return put + S * exp(-q * T) - K * exp(-r * T)
    return put


def lewis(model, S, K, T, r, q, call=True):
    """Lewis (2001): C = S e^{-qT} - sqrt(S K) e^{-(r+q)T/2}/pi * int Re[e^{iuk} phi_X(u - i/2)]/(u^2+1/4)."""
    out = []
    for Kj in np.atleast_1d(np.asarray(K, float)):
        k = log(S / Kj) + (r - q) * T
        def f(u):
            z = complex(u, -0.5)
            phi = model.cf(np.array(z), T, r, q) * np.exp(-1j * z * (r - q) * T)   # martingale part only
            return (np.exp(1j * u * k) * phi).real / (u * u + 0.25)
        I, _ = quad(f, 0, np.inf, limit=400, epsabs=1e-12)
        c = S * exp(-q * T) - sqrt(S * Kj) * exp(-(r + q) * T / 2) / pi * I
        out.append(c if call else c - S * exp(-q * T) + Kj * exp(-r * T))
    return np.array(out)


def smile(model, S, strikes, T, r, q, method=cos):
    """Implied Black-Scholes vols of the model's calls."""
    c = method(model, S, strikes, T, r, q)
    return np.array([implied_vol(ci, S, k, T, r, q) for ci, k in zip(c, strikes)])


if __name__ == "__main__":
    S, r, q, T = 100.0, 0.03, 0.01, 1.0
    Ks = np.array([80, 90, 100, 110, 120.0])
    models = {
        "Black-Scholes": BS(0.2), "Merton": Merton(0.15, 0.5, -0.1, 0.15), "Kou": Kou(0.15, 1.0, 0.4, 10, 5),
        "VG": VG(0.2, 0.2, -0.15), "NIG": NIG(15.0, -5.0, 0.5), "CGMY": CGMY(1.0, 8.0, 10.0, 0.5),
        "Heston": Heston(0.04, 2.0, 0.04, 0.5, -0.7), "Bates": Bates(0.04, 2.0, 0.04, 0.5, -0.7, 0.3, -0.1, 0.1),
    }
    print(f"{'model':14s}" + "".join(f"{k:>9.0f}" for k in Ks) + "   (implied vol by strike)")
    for n, m in models.items():
        print(f"{n:14s}" + "".join(f"{v:9.2%}" for v in smile(m, S, Ks, T, r, q)))
