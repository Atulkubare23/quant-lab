"""FRTB standardised approach: the sensitivities-based method (Basel Framework MAR21) for
GIRR, equity and FX, delta, vega and curvature. Parameters from the MAR21 text (in force 1 Jan 2023).

Inputs are net sensitivities (MAR21.19-21.26 definitions):
  GIRR delta   PV01 per 1bp scaled to per unit (dV / 0.0001), by currency, curve, tenor
  EQ delta     dV per 1% relative move / 0.01 (spot) or per 1bp (repo), by issuer and bucket
  FX delta     dV per 1% relative move / 0.01, by currency
  vega         sigma x dV/dsigma (vega x implied vol), by underlying and option maturity
  curvature    CVR up/down per risk factor (curvature_cvr below computes them from a pricer)

The capital is the largest of the three correlation scenarios (MAR21.6-7), summed across risk
classes and delta / vega / curvature inside each scenario.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import exp, sqrt

import numpy as np

from .sbm import across

# ---------------------------------------------------------------- parameters (MAR21)
GIRR_TENORS = [0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30]
GIRR_RW = dict(zip(GIRR_TENORS, [0.017, 0.017, 0.016, 0.013, 0.012, 0.011, 0.011, 0.011, 0.011, 0.011]))
GIRR_RW_INFL = GIRR_RW_XCCY = 0.016
SPECIFIED_CCY = {"EUR", "USD", "GBP", "AUD", "JPY", "SEK", "CAD"}
GIRR_THETA, GIRR_FLOOR, GIRR_GAMMA = 0.03, 0.40, 0.50
EQ_RW = {1: 0.55, 2: 0.60, 3: 0.45, 4: 0.55, 5: 0.30, 6: 0.35, 7: 0.40, 8: 0.50, 9: 0.70, 10: 0.50, 11: 0.70, 12: 0.15, 13: 0.25}
EQ_RHO = {**{b: 0.15 for b in (1, 2, 3, 4)}, **{b: 0.25 for b in (5, 6, 7, 8)}, 9: 0.075, 10: 0.125, 12: 0.80, 13: 0.80}
FX_RW, FX_GAMMA = 0.15, 0.60
SPECIFIED_PAIRS = {"EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "MXN", "CNY", "NZD", "RUB", "HKD", "SGD", "TRY", "KRW",
                   "SEK", "ZAR", "INR", "NOK", "BRL"}          # each against USD (MAR21.88)
VEGA_ALPHA = 0.01
VEGA_RW = {"GIRR": 1.0, "EQ_large": 0.7778, "EQ_small": 1.0, "FX": 1.0}   # min(55% sqrt(LH/10), 100%)


def eq_gamma(b, c):
    if b == 11 or c == 11:
        return 0.0
    if b <= 10 and c <= 10:
        return 0.15
    if {b, c} == {12, 13}:
        return 0.75
    return 0.45


def scen(rho, s):
    """MAR21.6: medium as given, high x1.25 capped at 1, low max(2 rho - 1, 0.75 rho)."""
    rho = np.asarray(rho, float)
    if s == "high":
        return np.minimum(1.25 * rho, 1.0)
    if s == "low":
        return np.maximum(2 * rho - 1, 0.75 * rho)
    return rho


def girr_tenor_corr(t1, t2):
    return max(exp(-GIRR_THETA * abs(t1 - t2) / min(t1, t2)), GIRR_FLOOR)


def vega_maturity_corr(t1, t2):
    return exp(-VEGA_ALPHA * abs(t1 - t2) / min(t1, t2))


# ---------------------------------------------------------------- sensitivities
@dataclass
class Sens:
    risk_class: str          # GIRR, EQ, FX
    kind: str                # delta, vega
    bucket: object           # currency (GIRR, FX) or equity bucket number
    value: float
    curve: str = "OIS"       # GIRR: curve name, or 'inflation' / 'xccy'
    tenor: float = None      # GIRR tenor, or option maturity for vega
    name: str = None         # equity issuer
    repo: bool = False       # equity repo-rate sensitivity


@dataclass
class Curvature:
    risk_class: str          # EQ or FX
    bucket: object
    name: str                # issuer / currency
    up: float                # CVR+
    down: float              # CVR-


# ---------------------------------------------------------------- correlations within buckets
def _rho_delta(a: Sens, b: Sens):
    if a.risk_class == "GIRR":
        ia, ib = a.curve in ("inflation", "xccy"), b.curve in ("inflation", "xccy")
        if a.curve == "xccy" or b.curve == "xccy":
            return 1.0 if (a.curve == b.curve == "xccy") else 0.0
        if ia and ib:
            return 0.999                                   # different inflation curves, same currency
        if ia or ib:
            return 0.40
        r = 1.0 if a.tenor == b.tenor else girr_tenor_corr(a.tenor, b.tenor)
        return r if a.curve == b.curve else r * 0.999
    if a.risk_class == "EQ":
        if a.name == b.name:
            return 1.0 if a.repo == b.repo else 0.999
        r = EQ_RHO[a.bucket]
        return r if a.repo == b.repo else r * 0.999
    return 1.0                                             # FX: one risk factor per bucket


def _rho_vega(a: Sens, b: Sens):
    m = vega_maturity_corr(a.tenor, b.tenor)
    if a.risk_class == "GIRR":
        return m                                           # (underlying-maturity factor taken as 1)
    if a.risk_class == "EQ":
        return m * (1.0 if a.name == b.name else EQ_RHO[a.bucket])
    return m


def _rw(s: Sens, specified: bool):
    if s.kind == "vega":
        if s.risk_class == "EQ":
            return VEGA_RW["EQ_small" if s.bucket in (9, 10, 11) else "EQ_large"]
        return VEGA_RW[s.risk_class]
    if s.risk_class == "GIRR":
        rw = GIRR_RW_INFL if s.curve == "inflation" else GIRR_RW_XCCY if s.curve == "xccy" else GIRR_RW[s.tenor]
        return rw / sqrt(2) if specified and s.bucket in SPECIFIED_CCY else rw
    if s.risk_class == "EQ":
        return EQ_RW[s.bucket] / (100 if s.repo else 1)
    return FX_RW / sqrt(2) if specified and s.bucket in SPECIFIED_PAIRS else FX_RW


def _gamma(rc, b, c):
    return {"GIRR": GIRR_GAMMA, "FX": FX_GAMMA}.get(rc) if rc != "EQ" else eq_gamma(b, c)


def _net(sens):
    """Net sensitivities to the same risk factor (MAR21.4(2))."""
    out = {}
    for s in sens:
        key = (s.risk_class, s.kind, s.bucket, s.curve if s.risk_class == "GIRR" else None, s.tenor, s.name, s.repo)
        if key in out:
            out[key].value += s.value
        else:
            out[key] = Sens(s.risk_class, s.kind, s.bucket, s.value, s.curve, s.tenor, s.name, s.repo)
    return list(out.values())


def delta_vega(sens, risk_class, kind, scenario="medium", specified=True):
    xs = [s for s in _net(sens) if s.risk_class == risk_class and s.kind == kind]
    by = defaultdict(list)
    for s in xs:
        by[s.bucket].append(s)
    K, S, names = [], [], []
    for b, items in by.items():
        ws = np.array([_rw(s, specified) * s.value for s in items])
        if risk_class == "EQ" and b == 11:
            k = float(np.abs(ws).sum())                       # MAR21.79: simple sum, no diversification
        else:
            f = _rho_delta if kind == "delta" else _rho_vega
            rho = np.array([[1.0 if i == j else f(x, y) for j, y in enumerate(items)] for i, x in enumerate(items)])
            rho = scen(rho, scenario)
            np.fill_diagonal(rho, 1.0)
            k = sqrt(max(0.0, float(ws @ rho @ ws)))
        K.append(k)
        S.append(float(ws.sum()))
        names.append(b)
    if not K:
        return 0.0
    g = np.array([[0.0 if i == j else _gamma(risk_class, b, c) for j, c in enumerate(names)] for i, b in enumerate(names)])
    return across(K, S, scen(g, scenario))


def _psi(a, b):
    return 0.0 if (a < 0 and b < 0) else 1.0


def curvature(cvrs, risk_class, scenario="medium"):
    xs = [c for c in cvrs if c.risk_class == risk_class]
    by = defaultdict(list)
    for c in xs:
        by[c.bucket].append(c)
    K, S, names = [], [], []
    for b, items in by.items():
        best = None
        for side in ("up", "down"):
            v = np.array([getattr(c, side) for c in items])
            if risk_class == "EQ" and b == 11:
                k = float(np.sum(np.maximum(v, 0)))
            else:
                tot = float(np.sum(np.maximum(v, 0) ** 2))
                for i, ci in enumerate(items):
                    for j, cj in enumerate(items):
                        if i != j:
                            r = (1.0 if ci.name == cj.name else EQ_RHO[b]) if risk_class == "EQ" else 1.0
                            r = float(scen(r * r, scenario))
                            tot += r * v[i] * v[j] * _psi(v[i], v[j])
                k = sqrt(max(tot, 0.0))
            if best is None or k > best[0] or (k == best[0] and side == "up" and v.sum() > 0):
                best = (k, float(v.sum()), side)
        K.append(best[0])
        S.append(best[1])
        names.append(b)
    if not K:
        return 0.0
    K, S = np.array(K), np.array(S)
    tot = float(np.sum(K * K))
    for i, b in enumerate(names):
        for j, c in enumerate(names):
            if i != j:
                g = float(scen(_gamma(risk_class, b, c) ** 2, scenario))
                tot += g * S[i] * S[j] * _psi(S[i], S[j])
    if tot < 0:
        Sc = np.clip(S, -K, K)
        tot = float(np.sum(K * K)) + sum(float(scen(_gamma(risk_class, b, c) ** 2, scenario)) * Sc[i] * Sc[j] * _psi(Sc[i], Sc[j])
                                          for i, b in enumerate(names) for j, c in enumerate(names) if i != j)
    return sqrt(max(tot, 0.0))


def curvature_cvr(pricers, x, rw, delta):
    """CVR+/- for one risk factor (MAR21.5): pricers are functions of the factor level x; delta is the
    sum of their delta sensitivities (per unit relative move, ie dV/dx * x). Relative shock rw."""
    v0 = sum(p(x) for p in pricers)
    up = -(sum(p(x * (1 + rw)) for p in pricers) - v0 - rw * delta)
    dn = -(sum(p(x * (1 - rw)) for p in pricers) - v0 + rw * delta)
    return up, dn


def capital(sens, cvrs=(), specified=True):
    out = {}
    for sc in ("medium", "high", "low"):
        parts = {}
        for rc in ("GIRR", "EQ", "FX"):
            parts[f"{rc} delta"] = delta_vega(sens, rc, "delta", sc, specified)
            parts[f"{rc} vega"] = delta_vega(sens, rc, "vega", sc, specified)
            if rc != "GIRR":
                parts[f"{rc} curvature"] = curvature(cvrs, rc, sc)
        out[sc] = {"parts": parts, "total": sum(parts.values())}
    worst = max(out, key=lambda k: out[k]["total"])
    return {"scenarios": out, "binding": worst, "capital": out[worst]["total"]}


if __name__ == "__main__":
    from .fourier import bs_price
    # a small book: a USD receiver swap ladder, two equity names, an index option position, FX
    sens = [Sens("GIRR", "delta", "USD", -6_000 / 1e-4, tenor=5), Sens("GIRR", "delta", "USD", 9_000 / 1e-4, tenor=10),
            Sens("GIRR", "delta", "EUR", -4_000 / 1e-4, tenor=2), Sens("GIRR", "delta", "USD", 1_500 / 1e-4, curve="inflation"),
            Sens("EQ", "delta", 5, 3e6, name="ACME"), Sens("EQ", "delta", 8, -1.5e6, name="BANKCO"),
            Sens("FX", "delta", "EUR", 4e6), Sens("FX", "delta", "TRY", -1e6)]
    # long 1y ATM calls on the index: delta, vega and curvature from Black-Scholes
    n, S0, vol = 50_000, 100.0, 0.2
    price = lambda S: n * bs_price(S, 100, 1.0, 0.03, 0.0, vol)
    dlt = (price(S0 * 1.01) - price(S0)) / 0.01
    vega = vol * n * (bs_price(S0, 100, 1.0, 0.03, 0.0, vol + 1e-4) - bs_price(S0, 100, 1.0, 0.03, 0.0, vol - 1e-4)) / 2e-4
    sens += [Sens("EQ", "delta", 12, dlt, name="SPX"), Sens("EQ", "vega", 12, vega, name="SPX", tenor=1.0)]
    up, dn = curvature_cvr([price], S0, EQ_RW[12], dlt)
    r = capital(sens, [Curvature("EQ", 12, "SPX", up, dn)])
    for sc, v in r["scenarios"].items():
        print(f"{sc:6s} " + "  ".join(f"{k} {x / 1e3:,.0f}k" for k, x in v["parts"].items() if x) + f"   total {v['total'] / 1e3:,.0f}k")
    print(f"SBM capital = {r['capital'] / 1e3:,.0f}k ({r['binding']} correlations bind)")
