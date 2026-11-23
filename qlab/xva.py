"""Counterparty exposure and xVA for a netting set of swaps, simulated under Hull-White.

  exposures   mark every swap on every path and date (closed-form HW bonds), net them, apply a
              collateral agreement (threshold, minimum transfer, margin period of risk)
  profile     EE, ENE, PFE(q), EPE and the discounted EE used in CVA
  xva         CVA, DVA, FVA from flat hazard rates and a funding spread

Exposure is discounted on each path with the bank account, so the discounted EE at t is a price:
for a single payer swap it equals a European payer swaption on the remaining swap, which is how
the tests check the simulation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp

import numpy as np

from .hullwhite import HW


@dataclass
class Swap:
    fixed: float
    start: float
    end: float
    notional: float = 1.0
    payer: bool = True           # pay fixed, receive float
    freq: int = 1

    @property
    def pay_times(self):
        n = int(round((self.end - self.start) * self.freq))
        return self.start + np.arange(1, n + 1) / self.freq

    @property
    def resets(self):
        return np.concatenate([[self.start], self.pay_times[:-1]])

    def mtm(self, m: HW, t, x, x_at):
        """Value at t on each path; x_at(s) gives x at an earlier grid time s (needed for the coupon
        fixed at the last reset: float leg = P(t, next) / P(reset, next) - P(t, end), single curve)."""
        ts = self.pay_times
        fut = ts[ts > t + 1e-12]
        if len(fut) == 0:
            return np.zeros_like(x)
        if t <= self.start:
            flt = m.bond(t, self.start, x) - m.bond(t, self.end, x)
            first = self.start
        else:
            reset = self.resets[ts > t + 1e-12][0]
            flt = m.bond(t, fut[0], x) / m.bond(reset, fut[0], x_at(reset)) - m.bond(t, self.end, x)
            first = reset
        taus = np.diff(np.concatenate([[first], fut]))
        fix = self.fixed * sum(tau * m.bond(t, T, x) for tau, T in zip(taus, fut))
        v = flt - fix
        return self.notional * (v if self.payer else -v)


@dataclass
class CSA:
    threshold: float = np.inf    # inf = uncollateralised
    mta: float = 0.0
    mpor: float = 10 / 252       # margin period of risk in years


@dataclass
class Result:
    times: np.ndarray
    V: np.ndarray                # netted value per path/date
    E: np.ndarray                # exposure after collateral (can be negative: owed to counterparty)
    D: np.ndarray                # bank-account discount factors

    def profile(self, q=0.975):
        pos, neg = np.maximum(self.E, 0), np.minimum(self.E, 0)
        return {
            "EE": pos.mean(0), "ENE": neg.mean(0), "PFE": np.quantile(self.E, q, axis=0),
            "dEE": (self.D * pos).mean(0), "dENE": (self.D * neg).mean(0),
        }


def exposures(m: HW, swaps, times, csa: CSA = CSA(), n_paths=20000, seed=0):
    times = np.asarray(times, float)
    lag = csa.mpor
    pts = [times] + [s.resets for s in swaps]
    if np.isfinite(csa.threshold):
        pts.append(np.maximum(times - lag, 0))
    grid = np.unique(np.round(np.concatenate(pts), 12))
    grid = grid[grid > 0]
    x, D = m.simulate(grid, n_paths, seed)
    pos = {t: k for k, t in enumerate(grid)}
    x_at = lambda u: x[:, pos[round(float(u), 12)]] if u > 0 else np.zeros(n_paths)
    Vg = np.stack([sum(s.mtm(m, t, x[:, k], x_at) for s in swaps) for k, t in enumerate(grid)], axis=1)
    times = np.round(times, 12)
    ki = [pos[t] for t in times]
    V = Vg[:, ki]
    if np.isfinite(csa.threshold):
        Vlag = np.stack([Vg[:, pos[round(max(t - lag, 0), 12)]] if t - lag > 0 else np.zeros(n_paths) for t in times], axis=1)
        C = np.sign(Vlag) * np.maximum(np.abs(Vlag) - csa.threshold, 0)          # two-way CSA
        C = np.where(np.abs(C) < csa.mta, 0, C)
        E = V - C
    else:
        E = V
    return Result(times, V, E, D[:, ki])


def xva(res: Result, hazard_c=0.02, rec_c=0.4, hazard_own=0.01, rec_own=0.4, funding_spread=0.005):
    t = res.times
    p = res.profile()
    edges = np.concatenate([[0], t])
    surv_c, surv_o = np.exp(-hazard_c * edges), np.exp(-hazard_own * edges)
    dpd_c, dpd_o = -np.diff(surv_c), -np.diff(surv_o)
    joint_c = dpd_c * surv_o[1:]             # counterparty defaults first (first-to-default)
    joint_o = dpd_o * surv_c[1:]
    dt = np.diff(edges)
    cva = (1 - rec_c) * np.sum(p["dEE"] * joint_c)
    dva = (1 - rec_own) * np.sum(-p["dENE"] * joint_o)
    fva = funding_spread * np.sum((p["dEE"] + p["dENE"]) * surv_c[1:] * surv_o[1:] * dt)
    return {"CVA": cva, "DVA": dva, "FVA": fva}


if __name__ == "__main__":
    P0 = lambda t: exp(-0.035 * t)
    m = HW(0.05, 0.01, P0)
    par = (1 - P0(10)) / sum(P0(t) for t in range(1, 11))
    book = [Swap(par, 0, 10, 1e8, True), Swap(par + 0.002, 0, 5, 6e7, False)]
    times = np.linspace(0.25, 10, 40)
    for label, csa in [("uncollateralised", CSA()), ("CSA, 1m threshold, 10d MPoR", CSA(1e6, 1e5))]:
        r = exposures(m, book, times, csa, 20000)
        p = r.profile()
        x = xva(r)
        print(f"{label:30s} peak EE {p['EE'].max():>12,.0f}  peak PFE {p['PFE'].max():>12,.0f}  "
              + "  ".join(f"{k} {v:>10,.0f}" for k, v in x.items()))
