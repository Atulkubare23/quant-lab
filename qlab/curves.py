"""Multi-curve construction: an overnight (SOFR-style) discount curve and a term forecast curve,
solved together by Newton with an exact Jacobian from dual numbers, plus par-rate risk.

Conventions are kept deliberately simple (times in years, annual fixed legs, quarterly floating,
no calendars) so that every number can be checked by hand. What carries over to a production
system is the structure: unknowns = curve nodes, equations = instruments repricing to quotes,
risk = dPV/dnodes mapped to dPV/dquotes through the inverse Jacobian.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

from .dual import Dual, exp, grad, log, value


@dataclass
class Curve:
    """Discount factors at node times; log-linear interpolation, t=0 has DF 1."""
    times: list
    dfs: list

    def df(self, t):
        if t <= 0:
            return 1.0
        ts = [0.0] + list(self.times)
        ls = [0.0] + [log(d) for d in self.dfs]
        i = min(max(bisect_right(ts, t) - 1, 0), len(ts) - 2)       # last segment extrapolates
        w = (t - ts[i]) / (ts[i + 1] - ts[i])
        return exp(ls[i] * (1 - w) + ls[i + 1] * w)

    def zero(self, t):
        return -value(log(self.df(t))) / t

    def fwd(self, t0, t1):
        return (self.df(t0) / self.df(t1) - 1) / (t1 - t0)


def _sched(T, freq):
    n = max(int(round(T * freq)), 1)
    return [T * (i + 1) / n for i in range(n)]


def annuity(disc: Curve, T, freq=1):
    ts, prev, out = _sched(T, freq), 0.0, 0.0
    for t in ts:
        out = out + (t - prev) * disc.df(t)
        prev = t
    return out


def ois_rate(c: dict, T):
    """Par fixed rate of an overnight-indexed swap: compounded overnight leg = 1 - DF(T)."""
    d = c["ois"]
    return (1 - d.df(T)) / annuity(d, T, 1 if T >= 1 else 1 / T)


def float_pv(c: dict, T, freq=4, start=0.0):
    d, f = c["ois"], c["term"]
    ts = [start + x for x in _sched(T - start, freq)] if T > start else []
    prev, out = start, 0.0
    for t in ts:
        out = out + (t - prev) * f.fwd(prev, t) * d.df(t)
        prev = t
    return out


def term_swap_rate(c: dict, T):
    """Par rate of a fixed (annual) vs term-index (quarterly) swap, discounted on the overnight curve."""
    return float_pv(c, T) / annuity(c["ois"], T, 1)


def fra_rate(c: dict, t0, t1):
    return c["term"].fwd(t0, t1)


@dataclass
class Instrument:
    name: str
    quote: float
    rate: callable          # curves -> par rate


@dataclass
class CurveSet:
    nodes: dict             # curve name -> node times
    instruments: list
    curves: dict = field(default_factory=dict)
    jacobian: np.ndarray = None     # d(rates)/d(nodes)

    def _build(self, x):
        out, i = {}, 0
        for name, ts in self.nodes.items():
            out[name] = Curve(list(ts), x[i:i + len(ts)])
            i += len(ts)
        return out

    def solve(self, tol=1e-14, max_iter=30):
        n = sum(len(t) for t in self.nodes.values())
        assert n == len(self.instruments), "square system: one instrument per node"
        x = np.concatenate([np.exp(-0.03 * np.asarray(t)) for t in self.nodes.values()])
        quotes = np.array([i.quote for i in self.instruments])
        for _ in range(max_iter):
            c = self._build(Dual.variables(x))
            r = [i.rate(c) for i in self.instruments]
            f = np.array([value(v) for v in r]) - quotes
            J = np.array([grad(v, n) for v in r])
            x = x - np.linalg.solve(J, f)
            if np.max(np.abs(f)) < tol:
                break
        self.curves = self._build(list(x))
        c = self._build(Dual.variables(x))
        self.jacobian = np.array([grad(i.rate(c), n) for i in self.instruments])
        self._x = x
        return self

    def risk(self, pv, bp=1e-4):
        """PV change per 1bp move in each quote: dPV/dq = dPV/dx . (dr/dx)^-1."""
        n = len(self._x)
        g = grad(pv(self._build(Dual.variables(self._x))), n)
        return np.linalg.solve(self.jacobian.T, g) * bp

    def pv(self, pv):
        return value(pv(self.curves))


def swap_pv(c, T, fixed, notional=1e6, receive_fixed=True):
    """Fixed (annual) vs term index (quarterly), overnight discounting."""
    s = 1 if receive_fixed else -1
    return s * notional * (fixed * annuity(c["ois"], T, 1) - float_pv(c, T))


def standard_set(ois_quotes: dict, term_quotes: dict) -> CurveSet:
    """ois_quotes {T: rate} (OIS swaps), term_quotes {T: rate} (term swaps; T < 1 treated as FRA 0xT)."""
    ins = [Instrument(f"OIS {T}y", q, (lambda T: lambda c: ois_rate(c, T))(T)) for T, q in ois_quotes.items()]
    for T, q in term_quotes.items():
        f = (lambda T: lambda c: fra_rate(c, 0.0, T))(T) if T < 1 else (lambda T: lambda c: term_swap_rate(c, T))(T)
        ins.append(Instrument(f"term {T}y", q, f))
    return CurveSet({"ois": sorted(ois_quotes), "term": sorted(term_quotes)}, ins)


SAMPLE_OIS = {0.25: 0.0430, 0.5: 0.0425, 1: 0.0410, 2: 0.0385, 3: 0.0372, 5: 0.0365, 7: 0.0368, 10: 0.0375, 20: 0.0390, 30: 0.0385}
SAMPLE_TERM = {0.25: 0.0455, 1: 0.0438, 2: 0.0412, 3: 0.0398, 5: 0.0391, 7: 0.0394, 10: 0.0401, 20: 0.0414, 30: 0.0409}


if __name__ == "__main__":
    cs = standard_set(SAMPLE_OIS, SAMPLE_TERM).solve()
    print("pillar   OIS zero   term zero   basis(bp)")
    for t in [1, 2, 5, 10, 30]:
        zo, zt = cs.curves["ois"].zero(t), cs.curves["term"].zero(t)
        print(f"{t:5}y   {zo:8.4%}   {zt:8.4%}   {1e4 * (zt - zo):8.1f}")
    trade = lambda c: swap_pv(c, 7, 0.0400)
    print(f"\nreceive 4.00% 7y vs term, 1m notional: PV {cs.pv(trade):,.0f}")
    for i, d in zip(cs.instruments, cs.risk(trade)):
        if abs(d) > 0.5:
            print(f"  {i.name:10s} {d:9.1f} per bp")
