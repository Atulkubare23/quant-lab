"""Rough Bergomi (Bayer, Friz & Gatheral 2016) by the hybrid scheme (Bennedsen, Lunde & Pakkanen 2017).

    V_t = xi_0 exp(eta Y_t - eta^2/2 t^{2H}),   Y_t = sqrt(2H) int_0^t (t-s)^{H-1/2} dW_s
    dS/S = sqrt(V) (rho dW + sqrt(1 - rho^2) dW_perp)

Y is a Riemann-Liouville fractional Brownian motion with Var(Y_t) = t^{2H}. The hybrid scheme
(kappa = 1) simulates the singular first cell of the kernel exactly and the rest as a discrete
convolution with the kernel evaluated at optimal points; this is the scheme in McCrickerd &
Pakkanen's rough_bergomi repository, rewritten here.

The model gives the steep, power-law ATM skew of equity index options: skew ~ T^{H - 1/2}.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .fourier import implied_vol


def _b(k, a):
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


@dataclass
class RBergomi:
    H: float = 0.1
    eta: float = 1.9
    rho: float = -0.9
    xi0: float = 0.235 ** 2

    def paths(self, T, steps_per_year=312, n_paths=50_000, seed=0):
        """Returns (t, Y, V, S/S0) on the grid; S is driftless (price forwards)."""
        a = self.H - 0.5
        n = steps_per_year
        s = int(round(T * n))
        dt = 1.0 / n
        rng = np.random.default_rng(seed)
        cov = np.array([[dt, dt ** (a + 1) / (a + 1)], [dt ** (a + 1) / (a + 1), dt ** (2 * a + 1) / (2 * a + 1)]])
        dW1 = rng.multivariate_normal([0, 0], cov, size=(n_paths, s))          # (paths, s, 2)
        Y1 = np.zeros((n_paths, s + 1))
        Y1[:, 1:] = dW1[:, :, 1]                                                # exact near-diagonal term
        G = np.zeros(s + 1)
        k = np.arange(2, s + 1)
        G[2:] = (_b(k, a) / n) ** a                                             # kernel at optimal points
        X = dW1[:, :, 0]
        # convolution sum_{k>=2} G_k X_{i-k}, done with FFT along time
        L = 1 << int(np.ceil(np.log2(2 * s + 1)))
        conv = np.fft.irfft(np.fft.rfft(G, L)[None, :] * np.fft.rfft(X, L, axis=1), L, axis=1)[:, :s + 1]
        Y2 = np.zeros((n_paths, s + 1))
        Y2[:, 1:] = conv[:, :s]
        Y = sqrt(2 * a + 1) * (Y1 + Y2)
        t = np.linspace(0, s * dt, s + 1)
        V = self.xi0 * np.exp(self.eta * Y - 0.5 * self.eta ** 2 * t ** (2 * a + 1))
        dW2 = rng.standard_normal((n_paths, s)) * sqrt(dt)
        dZ = self.rho * X + sqrt(1 - self.rho ** 2) * dW2
        incr = np.sqrt(V[:, :-1]) * dZ - 0.5 * V[:, :-1] * dt
        S = np.exp(np.concatenate([np.zeros((n_paths, 1)), np.cumsum(incr, axis=1)], axis=1))
        return t, Y, V, S

    def smile(self, T, log_strikes, **kw):
        """Implied vols of forward-measured calls (S0 = F = 1, r = q = 0)."""
        _, _, _, S = self.paths(T, **kw)
        ST = S[:, -1]
        out = []
        for k in log_strikes:
            K = np.exp(k)
            # price OTM options (puts below the forward) for accuracy
            if K < 1:
                p = np.maximum(K - ST, 0).mean()
                out.append(implied_vol(p, 1.0, K, T, 0.0, 0.0, call=False))
            else:
                c = np.maximum(ST - K, 0).mean()
                out.append(implied_vol(c, 1.0, K, T, 0.0, 0.0, call=True))
        return np.array(out)

    def atm_skew(self, T, dk=0.02, **kw):
        iv = self.smile(T, [-dk, dk], **kw)
        return (iv[1] - iv[0]) / (2 * dk)


if __name__ == "__main__":
    m = RBergomi()
    ks = np.linspace(-0.3, 0.15, 10)
    print("rBergomi H=0.1 eta=1.9 rho=-0.9 xi0=0.235^2")
    for T in [0.05, 0.25, 1.0]:
        iv = m.smile(T, ks * sqrt(T / 0.25), n_paths=40_000)
        print(f"  T={T:<5} " + " ".join(f"{v:6.1%}" for v in iv))
    Ts = np.array([0.04, 0.08, 0.16, 0.32, 0.64])
    sk = np.array([abs(m.atm_skew(T, n_paths=60_000, seed=5)) for T in Ts])
    slope = np.polyfit(np.log(Ts), np.log(sk), 1)[0]
    print(f"ATM skew term structure: log-log slope {slope:.2f} (theory H - 1/2 = {m.H - 0.5:.2f})")
