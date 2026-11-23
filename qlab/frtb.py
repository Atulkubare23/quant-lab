"""FRTB standardised approach: the sensitivities-based method (Basel Framework MAR21) for all seven
risk classes (GIRR, CSR non-securitisation, CSR securitisation CTP and non-CTP, equity, commodity, FX),
delta, vega and curvature. Parameters from the MAR21 text (in force 1 Jan 2023).

Inputs are net sensitivities (MAR21.19-21.26 definitions):
  GIRR delta   PV01 per 1bp scaled to per unit (dV / 0.0001), by currency, curve, tenor
  EQ delta     dV per 1% relative move / 0.01 (spot) or per 1bp (repo), by issuer and bucket
  CSR delta    dV per 1bp of credit spread / 0.0001, by issuer (tranche), tenor and bond / CDS basis
  COMM delta   dV per 1% relative move / 0.01, by commodity, tenor and delivery location
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


# ---------------------------------------------------------------- CSR and commodity parameters (MAR21.51-21.85)
CSR_RW = {1: 0.005, 2: 0.01, 3: 0.05, 4: 0.03, 5: 0.03, 6: 0.02, 7: 0.015, 8: 0.025, 9: 0.02, 10: 0.04, 11: 0.12,
          12: 0.07, 13: 0.085, 14: 0.055, 15: 0.05, 16: 0.12, 17: 0.015, 18: 0.05}
CTP_RW = {1: 0.04, 2: 0.04, 3: 0.08, 4: 0.05, 5: 0.04, 6: 0.03, 7: 0.02, 8: 0.06, 9: 0.13, 10: 0.13, 11: 0.16,
          12: 0.10, 13: 0.12, 14: 0.12, 15: 0.12, 16: 0.13}
_NCTP_SENIOR = [0.009, 0.015, 0.02, 0.02, 0.008, 0.012, 0.012, 0.014]
NCTP_RW = {**{b + 1: w for b, w in enumerate(_NCTP_SENIOR)}, **{b + 9: 1.25 * w for b, w in enumerate(_NCTP_SENIOR)},
           **{b + 17: 1.75 * w for b, w in enumerate(_NCTP_SENIOR)}, 25: 0.035}
COMM_RW = {1: 0.30, 2: 0.35, 3: 0.60, 4: 0.80, 5: 0.40, 6: 0.45, 7: 0.20, 8: 0.35, 9: 0.25, 10: 0.35, 11: 0.50}
COMM_RHO = {1: 0.55, 2: 0.95, 3: 0.40, 4: 0.80, 5: 0.60, 6: 0.65, 7: 0.55, 8: 0.45, 9: 0.15, 10: 0.40, 11: 0.15}
# MAR21.57 Table 5: sector correlations between bucket groups (1/9, 2/10, ..., 7/15, 8, 16, 17, 18)
_SECTORS = ["1", "2", "3", "4", "5", "6", "7", "8", "16", "17", "18"]
_T5 = [[0.75, 0.10, 0.20, 0.25, 0.20, 0.15, 0.10, 0.0, 0.45, 0.45],
       [0.05, 0.15, 0.20, 0.15, 0.10, 0.10, 0.0, 0.45, 0.45],
       [0.05, 0.15, 0.20, 0.05, 0.20, 0.0, 0.45, 0.45],
       [0.20, 0.25, 0.05, 0.05, 0.0, 0.45, 0.45],
       [0.25, 0.05, 0.15, 0.0, 0.45, 0.45],
       [0.05, 0.20, 0.0, 0.45, 0.45],
       [0.05, 0.0, 0.45, 0.45],
       [0.0, 0.45, 0.45],
       [0.0, 0.0],
       [0.75]]
CSR_SECTOR_CORR = {}
for _i, _row in enumerate(_T5):
    for _k, _v in enumerate(_row):
        _a, _b = _SECTORS[_i], _SECTORS[_i + 1 + _k]
        CSR_SECTOR_CORR[(_a, _b)] = CSR_SECTOR_CORR[(_b, _a)] = _v
SIMPLE_SUM_BUCKET = {"EQ": 11, "CSR": 16, "CTP": 16, "NCTP": 25}
RISK_CLASSES = ("GIRR", "CSR", "CTP", "NCTP", "EQ", "COMM", "FX")


def _sector(b):
    return str(b - 8) if 9 <= b <= 15 else str(b)


def csr_gamma(b, c):
    """MAR21.57: gamma = gamma_rating x gamma_sector (also used for the CTP, MAR21.61)."""
    rating = 0.5 if (b <= 15 and c <= 15 and ((b <= 8) != (c <= 8))) else 1.0
    sb, sc = _sector(b), _sector(c)
    return rating * (1.0 if sb == sc else CSR_SECTOR_CORR[(sb, sc)])


def comm_gamma(b, c):
    return 0.0 if (b == 11 or c == 11) else 0.20


# ---------------------------------------------------------------- sensitivities
@dataclass
class Sens:
    risk_class: str          # GIRR, CSR (non-securitisation), CTP, NCTP (securitisation non-CTP), EQ, COMM, FX
    kind: str                # delta, vega
    bucket: object           # currency (GIRR, FX) or bucket number
    value: float
    curve: str = "OIS"       # GIRR curve or 'inflation' / 'xccy'; CSR basis ('bond' / 'CDS')
    tenor: float = None      # GIRR / CSR / commodity tenor, or option maturity for vega
    name: str = None         # issuer (EQ, CSR, CTP), tranche (NCTP), commodity (COMM)
    repo: bool = False       # equity repo-rate sensitivity
    location: str = None     # commodity delivery location


@dataclass
class Curvature:
    risk_class: str
    bucket: object
    name: str                # issuer / tranche / commodity / currency
    up: float                # CVR+
    down: float              # CVR-


# ---------------------------------------------------------------- correlations within buckets
def _name_corr(rc, bucket, a, b):
    """The 'name' dimension of the delta correlation (also used for vega and, squared, curvature)."""
    if a == b:
        return 1.0
    if rc == "EQ":
        return EQ_RHO[bucket]
    if rc in ("CSR", "CTP"):
        return 0.80 if bucket in (17, 18) else 0.35
    if rc == "NCTP":
        return 0.40
    if rc == "COMM":
        return COMM_RHO[bucket]
    return 1.0


def _rho_delta(a: Sens, b: Sens):
    rc = a.risk_class
    if rc == "GIRR":
        ia, ib = a.curve in ("inflation", "xccy"), b.curve in ("inflation", "xccy")
        if a.curve == "xccy" or b.curve == "xccy":
            return 1.0 if (a.curve == b.curve == "xccy") else 0.0
        if ia and ib:
            return 0.999                                   # different inflation curves, same currency
        if ia or ib:
            return 0.40
        r = 1.0 if a.tenor == b.tenor else girr_tenor_corr(a.tenor, b.tenor)
        return r if a.curve == b.curve else r * 0.999
    if rc == "EQ":
        if a.name == b.name:
            return 1.0 if a.repo == b.repo else 0.999
        r = EQ_RHO[a.bucket]
        return r if a.repo == b.repo else r * 0.999
    if rc in ("CSR", "CTP", "NCTP"):
        tenor = 1.0 if a.tenor == b.tenor else (0.80 if rc == "NCTP" else 0.65)
        basis = 1.0 if a.curve == b.curve else (0.99 if rc == "CTP" else 0.999)
        return _name_corr(rc, a.bucket, a.name, b.name) * tenor * basis
    if rc == "COMM":
        tenor = 1.0 if a.tenor == b.tenor else 0.99
        loc = 1.0 if a.location == b.location else 0.999
        return _name_corr(rc, a.bucket, a.name, b.name) * tenor * loc
    return 1.0                                             # FX: one risk factor per bucket


def _rho_vega(a: Sens, b: Sens):
    """MAR21.93-94: option-maturity correlation x the delta correlation of the underlying (name dimension)."""
    m = vega_maturity_corr(a.tenor, b.tenor)
    if a.risk_class in ("GIRR", "FX"):
        return m                                           # GIRR: underlying-maturity factor taken as 1
    return m * _name_corr(a.risk_class, a.bucket, a.name, b.name)


def _rw(s: Sens, specified: bool):
    rc = s.risk_class
    if s.kind == "vega":
        if rc == "EQ":
            return VEGA_RW["EQ_small" if s.bucket in (9, 10, 11) else "EQ_large"]
        return 1.0                                         # GIRR, CSR, COMM, FX: min(55% sqrt(LH/10), 1) = 1
    if rc == "GIRR":
        rw = GIRR_RW_INFL if s.curve == "inflation" else GIRR_RW_XCCY if s.curve == "xccy" else GIRR_RW[s.tenor]
        return rw / sqrt(2) if specified and s.bucket in SPECIFIED_CCY else rw
    if rc == "EQ":
        return EQ_RW[s.bucket] / (100 if s.repo else 1)
    if rc == "CSR":
        return CSR_RW[s.bucket]
    if rc == "CTP":
        return CTP_RW[s.bucket]
    if rc == "NCTP":
        return NCTP_RW[s.bucket]
    if rc == "COMM":
        return COMM_RW[s.bucket]
    return FX_RW / sqrt(2) if specified and s.bucket in SPECIFIED_PAIRS else FX_RW


def curvature_rw(risk_class, bucket):
    """MAR21.98-99: relative shift equal to the delta RW (EQ, FX, COMM); parallel shift at the highest
    delta RW of the bucket (GIRR: the 0.25y weight; CSR weights are flat across tenors)."""
    return {"GIRR": GIRR_RW[0.25], "EQ": EQ_RW.get(bucket), "FX": FX_RW, "CSR": CSR_RW.get(bucket),
            "CTP": CTP_RW.get(bucket), "NCTP": NCTP_RW.get(bucket), "COMM": COMM_RW.get(bucket)}[risk_class]


def _gamma(rc, b, c):
    if rc == "EQ":
        return eq_gamma(b, c)
    if rc in ("CSR", "CTP"):
        return csr_gamma(b, c)
    if rc == "NCTP":
        return 0.0
    if rc == "COMM":
        return comm_gamma(b, c)
    return {"GIRR": GIRR_GAMMA, "FX": FX_GAMMA}[rc]


def _net(sens):
    """Net sensitivities to the same risk factor (MAR21.4(2))."""
    out = {}
    for s in sens:
        key = (s.risk_class, s.kind, s.bucket, s.curve, s.tenor, s.name, s.repo, s.location)
        if key in out:
            out[key].value += s.value
        else:
            out[key] = Sens(s.risk_class, s.kind, s.bucket, s.value, s.curve, s.tenor, s.name, s.repo, s.location)
    return list(out.values())


def _combine(risk_class, K, S, names, scenario, curvature=False):
    """Across buckets; NCTP's other-sector bucket (25) is added without diversification (MAR21.71)."""
    if not K:
        return 0.0
    extra = 0.0
    if risk_class == "NCTP" and 25 in names:
        i = names.index(25)
        extra = K[i]
        K, S, names = K[:i] + K[i + 1:], S[:i] + S[i + 1:], names[:i] + names[i + 1:]
        if not K:
            return extra
    n = len(K)
    g = np.zeros((n, n))
    for i, b in enumerate(names):
        for j, c in enumerate(names):
            if i != j:
                x = _gamma(risk_class, b, c)
                g[i, j] = scen(x * x if curvature else x, scenario)
    if not curvature:
        return across(K, S, g) + extra
    Ka, Sa = np.array(K), np.array(S)
    def tot(Sv):
        return float(np.sum(Ka * Ka)) + sum(g[i, j] * Sv[i] * Sv[j] * _psi(Sv[i], Sv[j])
                                             for i in range(n) for j in range(n) if i != j)
    v = tot(Sa)
    if v < 0:
        v = tot(np.clip(Sa, -Ka, Ka))
    return sqrt(max(v, 0.0)) + extra


def delta_vega(sens, risk_class, kind, scenario="medium", specified=True):
    xs = [s for s in _net(sens) if s.risk_class == risk_class and s.kind == kind]
    by = defaultdict(list)
    for s in xs:
        by[s.bucket].append(s)
    K, S, names = [], [], []
    for b, items in by.items():
        ws = np.array([_rw(s, specified) * s.value for s in items])
        if SIMPLE_SUM_BUCKET.get(risk_class) == b:
            k = float(np.abs(ws).sum())                       # other-sector buckets: no diversification
        else:
            f = _rho_delta if kind == "delta" else _rho_vega
            rho = np.array([[1.0 if i == j else f(x, y) for j, y in enumerate(items)] for i, x in enumerate(items)])
            rho = scen(rho, scenario)
            np.fill_diagonal(rho, 1.0)
            k = sqrt(max(0.0, float(ws @ rho @ ws)))
        K.append(k)
        S.append(float(ws.sum()))
        names.append(b)
    return _combine(risk_class, K, S, names, scenario)


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
            if SIMPLE_SUM_BUCKET.get(risk_class) == b:
                k = float(np.sum(np.maximum(v, 0)))
            else:
                tot = float(np.sum(np.maximum(v, 0) ** 2))
                for i, ci in enumerate(items):
                    for j, cj in enumerate(items):
                        if i != j:
                            r = _name_corr(risk_class, b, ci.name, cj.name)
                            r = float(scen(r * r, scenario))           # MAR21.100: squared delta correlation
                            tot += r * v[i] * v[j] * _psi(v[i], v[j])
                k = sqrt(max(tot, 0.0))
            if best is None or k > best[0] or (k == best[0] and side == "up" and v.sum() > 0):
                best = (k, float(v.sum()), side)
        K.append(best[0])
        S.append(best[1])
        names.append(b)
    return _combine(risk_class, K, S, names, scenario, curvature=True)


def curvature_cvr(pricers, x, rw, delta):
    """CVR+/- for one risk factor (MAR21.5): pricers are functions of the factor level x; delta is the
    sum of their delta sensitivities (per unit relative move, ie dV/dx * x). Relative shock rw."""
    v0 = sum(p(x) for p in pricers)
    up = -(sum(p(x * (1 + rw)) for p in pricers) - v0 - rw * delta)
    dn = -(sum(p(x * (1 - rw)) for p in pricers) - v0 + rw * delta)
    return up, dn


def sbm(sens, cvrs=(), specified=True):
    out = {}
    for sc in ("medium", "high", "low"):
        parts = {}
        for rc in RISK_CLASSES:
            parts[f"{rc} delta"] = delta_vega(sens, rc, "delta", sc, specified)
            parts[f"{rc} vega"] = delta_vega(sens, rc, "vega", sc, specified)
            parts[f"{rc} curvature"] = curvature(cvrs, rc, sc)
        out[sc] = {"parts": parts, "total": sum(parts.values())}
    worst = max(out, key=lambda k: out[k]["total"])
    return {"scenarios": out, "binding": worst, "capital": out[worst]["total"]}


capital = sbm      # backwards-compatible name


# ================================================================ default risk capital (MAR22)
DRC_RW = {"AAA": 0.005, "AA": 0.02, "A": 0.03, "BBB": 0.06, "BB": 0.15, "B": 0.30, "CCC": 0.50,
          "unrated": 0.15, "defaulted": 1.0}
LGD = {"equity": 1.0, "non-senior": 1.0, "senior": 0.75, "covered": 0.25, "none": 1.0}
SENIORITY = {"equity": 0, "non-senior": 1, "senior": 2, "covered": 3, "none": 2}


@dataclass
class JTD:
    """One default exposure (MAR22.10-22.18).
    notional > 0 is long credit (loses on default), < 0 short.
    pnl: cumulative MtM already taken (market value - notional for a bond; Table 1 for derivatives).
    seniority: equity, non-senior, senior, covered (LGD 100/100/75/25%); 'none' applies no LGD
    (eg an instrument paying a fixed amount on default, MAR22.12(4))."""
    obligor: str
    notional: float
    pnl: float = 0.0
    rating: str = "BBB"
    seniority: str = "senior"
    maturity: float = 5.0
    bucket: str = "corporates"          # corporates, sovereigns, local governments (MAR22.22)

    def gross(self):
        lgd = 1.0 if self.seniority == "none" else LGD[self.seniority]
        j = lgd * self.notional + self.pnl
        j = max(j, 0.0) if self.notional > 0 else min(j, 0.0)
        return j * min(max(self.maturity, 0.25), 1.0)            # MAR22.15 and 22.18


def net_jtd(positions):
    """Offset long and short gross JTD of the same obligor (MAR22.19): a short may only offset a long
    of the same or higher seniority. Returns {(bucket, obligor, rating): (net_long, net_short <= 0)}."""
    out = {}
    by = defaultdict(list)
    for p in positions:
        by[(p.bucket, p.obligor, p.rating)].append(p)
    for key, ps in by.items():
        longs = sorted([[SENIORITY[p.seniority], p.gross()] for p in ps if p.notional > 0])
        shorts = sorted([[SENIORITY[p.seniority], -p.gross()] for p in ps if p.notional < 0], reverse=True)
        for sh in shorts:                           # most senior shorts first (they can offset the fewest longs)
            for lg in longs:                        # least senior eligible long first
                if lg[0] >= sh[0] and sh[1] > 0 and lg[1] > 0:
                    x = min(lg[1], sh[1])
                    lg[1] -= x
                    sh[1] -= x
        out[key] = (sum(l[1] for l in longs), -sum(s[1] for s in shorts))
    return out


def hbr(longs, shorts):
    """Hedge benefit ratio (MAR22.24)."""
    L, S = sum(longs), sum(abs(x) for x in shorts)
    return L / (L + S) if L + S > 0 else 0.0


def drc_nonsec(positions):
    """MAR22.20-22.25: per bucket max(sum RW*long - HBR * sum RW*|short|, 0); simple sum over buckets."""
    nets = net_jtd(positions)
    buckets = defaultdict(list)
    for (b, _, rating), (lg, sh) in nets.items():
        buckets[b].append((rating, lg, sh))
    per, hb = {}, {}
    for b, rows in buckets.items():
        h = hbr([r[1] for r in rows], [r[2] for r in rows])
        wl = sum(DRC_RW[r[0]] * r[1] for r in rows)
        ws = sum(DRC_RW[r[0]] * abs(r[2]) for r in rows)
        per[b], hb[b] = max(wl - h * ws, 0.0), h
    return {"buckets": per, "hbr": hb, "total": sum(per.values())}


@dataclass
class Tranche:
    """Securitisation exposure (MAR22.26-22.45). JTD = market value, no LGD; rw comes from the
    banking-book securitisation framework with a one-year maturity (MAR22.34) and is supplied by the
    caller. Offsetting only within the same tranche of the same pool (MAR22.29)."""
    tranche: str
    market_value: float          # > 0 long, < 0 short
    rw: float
    bucket: str                  # non-CTP: asset class x region (or corporates); CTP: the index
    maturity: float = 5.0


def _tranche_nets(tranches):
    by, rw = defaultdict(float), {}
    for t in tranches:
        by[(t.bucket, t.tranche)] += t.market_value * min(max(t.maturity, 0.25), 1.0)
        rw[(t.bucket, t.tranche)] = t.rw
    return {k: (max(v, 0.0), min(v, 0.0), rw[k]) for k, v in by.items()}


def drc_sec_nonctp(tranches):
    """MAR22.35-22.37: same bucket formula as non-securitisations, HBR per bucket."""
    buckets = defaultdict(list)
    for (b, _), row in _tranche_nets(tranches).items():
        buckets[b].append(row)
    per = {}
    for b, rows in buckets.items():
        h = hbr([r[0] for r in rows], [r[1] for r in rows])
        per[b] = max(sum(r[2] * r[0] for r in rows) - h * sum(r[2] * abs(r[1]) for r in rows), 0.0)
    return {"buckets": per, "total": sum(per.values())}


def drc_ctp(tranches):
    """MAR22.44-22.45: one HBR over the whole CTP, bucket (index) amounts may be negative, and
    total = max(sum_b [max(DRC_b, 0) + 0.5 min(DRC_b, 0)], 0)."""
    rows = list(_tranche_nets(tranches).items())
    h = hbr([r[0] for _, r in rows], [r[1] for _, r in rows])
    per = defaultdict(float)
    for (b, _), (lg, sh, rw) in rows:
        per[b] += rw * lg - h * rw * abs(sh)
    total = max(sum(max(v, 0.0) + 0.5 * min(v, 0.0) for v in per.values()), 0.0)
    return {"buckets": dict(per), "hbr": h, "total": total}


def drc(nonsec=(), sec_nonctp=(), ctp=()):
    """No diversification between the three parts (MAR22.4)."""
    a, b, c = drc_nonsec(nonsec), drc_sec_nonctp(sec_nonctp), drc_ctp(ctp)
    return {"non-securitisation": a, "securitisation non-CTP": b, "CTP": c,
            "total": a["total"] + b["total"] + c["total"]}


# ================================================================ residual risk add-on (MAR23)
@dataclass
class Residual:
    notional: float              # gross notional
    exotic: bool                 # exotic underlying (1.0%, MAR23.8) vs other residual risk (0.1%)
    back_to_back: bool = False   # exactly matched with a third party: excluded (MAR23.5)
    listed_or_cleared: bool = False   # excluded unless exotic (MAR23.5)


def rrao(items):
    tot = 0.0
    for r in items:
        if r.back_to_back or (r.listed_or_cleared and not r.exotic):
            continue
        tot += abs(r.notional) * (0.01 if r.exotic else 0.001)
    return tot


def standardised_approach(sens=(), cvrs=(), nonsec=(), sec_nonctp=(), ctp=(), residual=(), specified=True):
    """FRTB-SA capital = SBM + DRC + RRAO (MAR20.4)."""
    s = sbm(sens, cvrs, specified)
    d = drc(nonsec, sec_nonctp, ctp)
    r = rrao(residual)
    return {"SBM": s["capital"], "SBM binding": s["binding"], "DRC": d["total"], "RRAO": r,
            "total": s["capital"] + d["total"] + r, "detail": {"sbm": s, "drc": d}}


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
    # credit spreads (CS01 per bp scaled to per unit of spread), commodities
    sens += [Sens("CSR", "delta", 4, 2_000 / 1e-4, curve="bond", tenor=5, name="ACME"),
             Sens("CSR", "delta", 4, -1_500 / 1e-4, curve="CDS", tenor=5, name="ACME"),
             Sens("CSR", "delta", 12, 800 / 1e-4, tenor=3, name="JUNKCO"),
             Sens("COMM", "delta", 2, 1.2e6, tenor=1, name="WTI", location="Cushing"),
             Sens("COMM", "delta", 2, -1.0e6, tenor=1, name="Brent", location="Rotterdam"),
             Sens("COMM", "delta", 7, 0.5e6, tenor=0.5, name="Gold")]
    nonsec = [JTD("ACME", 20e6, -0.5e6, "BBB", "senior"), JTD("ACME", -10e6, 0.2e6, "BBB", "senior"),
              JTD("JUNKCO", 5e6, -0.4e6, "B", "senior", maturity=0.5),
              JTD("BANKCO", -8e6, 0.0, "A", "non-senior")]
    residual = [Residual(50e6, exotic=False), Residual(10e6, exotic=True)]
    r = standardised_approach(sens, [Curvature("EQ", 12, "SPX", up, dn)], nonsec, residual=residual)
    for sc, v in r["detail"]["sbm"]["scenarios"].items():
        print(f"{sc:6s} " + "  ".join(f"{k} {x / 1e3:,.0f}k" for k, x in v["parts"].items() if x) + f"   total {v['total'] / 1e3:,.0f}k")
    print(f"SBM {r['SBM'] / 1e3:,.0f}k ({r['SBM binding']} correlations bind) + DRC {r['DRC'] / 1e3:,.0f}k"
          f" + RRAO {r['RRAO'] / 1e3:,.0f}k = FRTB-SA {r['total'] / 1e3:,.0f}k")
