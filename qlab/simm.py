"""ISDA SIMM (initial margin for uncleared derivatives): the calculation engine.

Implements the SIMM aggregation for the Rates/FX and Equity product classes:

  IR delta      WS = RW_tenor x s x CR; sub-curves of one currency at correlation phi; inflation and
                cross-currency basis factors; currencies aggregated with gamma x g_bc, where
                g_bc = min(CR_b, CR_c) / max(CR_b, CR_c)
  IR vega       VR = HVR x VRW x (vega x implied vol) per expiry; same tenor correlations
  IR curvature  CVR = sum_k SF(t_k) VR_k with SF(t) = 0.5 min(1, 14 days / t); the lambda / theta
                adjustment from the 99.5% quantile: lambda = (z^2 - 1)(1 + theta) - theta
  EQ delta/vega/curvature  by bucket with intra- and inter-bucket correlations, residual bucket added
  FX delta      per currency, concentration thresholds
  Totals        risk classes combined with psi; product classes summed; SIMM = sum of product classes

PARAMETERS. The calibration (risk weights, correlations, thresholds) is ISDA's and is published
in its methodology document, which this session could not download (isda.org refused the request).
qlab/data/simm_params.json therefore holds PLACEHOLDER values of the right order of magnitude so the
engine runs end to end; load the official calibration into the same JSON keys before using numbers.
`params()["status"]` says which set is loaded, and every result carries it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import norm

PARAMS_FILE = Path(__file__).with_name("data") / "simm_params.json"
IR_TENORS = ["2w", "1m", "3m", "6m", "1y", "2y", "3y", "5y", "10y", "15y", "20y", "30y"]
TENOR_YEARS = dict(zip(IR_TENORS, [14 / 365, 1 / 12, 0.25, 0.5, 1, 2, 3, 5, 10, 15, 20, 30]))


def params(path=PARAMS_FILE):
    return json.loads(Path(path).read_text())


@dataclass
class Sens:
    risk_class: str            # IR, EQ, FX
    measure: str               # delta, vega
    qualifier: str             # currency (IR, FX) or issuer (EQ)
    value: float               # delta: per 1bp (IR) / per 1% (EQ, FX); vega: vega x implied vol
    bucket: str = None         # equity bucket ("1".."12", "residual")
    label1: str = None         # IR tenor / vol expiry, or "inflation" / "xccy"
    subcurve: str = "OIS"      # IR sub-curve


def _cr(total, threshold):
    return max(1.0, sqrt(abs(total) / threshold)) if threshold else 1.0


def _quad(ws, rho):
    rho = np.array(rho, float)
    np.fill_diagonal(rho, 1.0)
    return sqrt(max(0.0, float(ws @ rho @ ws)))


# ---------------------------------------------------------------- interest rates
def _ir_bucket(items, P, measure):
    """Return (K, S, CR) for one currency."""
    ir = P["IR"]
    tenor_rho = np.array(ir["tenor_corr"])
    ccy = items[0].qualifier
    vol_group = next((g for g, cs in ir["vol_groups"].items() if ccy in cs), "regular")
    rws = ir["rw"][vol_group]
    thr = ir["delta_threshold"].get(vol_group, ir["delta_threshold"]["regular"]) if measure == "delta" else ir["vega_threshold"]
    total = sum(s.value for s in items if s.label1 != "xccy")
    CR = _cr(total, thr)
    ws, keys = [], []
    for s in items:
        if measure == "vega":
            rw = ir["vrw"] * ir["hvr"]
        elif s.label1 == "inflation":
            rw = ir["rw_inflation"]
        elif s.label1 == "xccy":
            rw = ir["rw_xccy"]
        else:
            rw = rws[IR_TENORS.index(s.label1)]
        ws.append(rw * s.value * (1.0 if s.label1 == "xccy" else CR))
        keys.append(s)
    ws = np.array(ws)
    n = len(ws)
    rho = np.eye(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = keys[i], keys[j]
            if "xccy" in (a.label1, b.label1):
                r = 1.0 if a.label1 == b.label1 else ir["corr_xccy"]
            elif "inflation" in (a.label1, b.label1):
                r = 1.0 if a.label1 == b.label1 else ir["corr_inflation"]
            else:
                r = tenor_rho[IR_TENORS.index(a.label1), IR_TENORS.index(b.label1)]
                if a.subcurve != b.subcurve:
                    r *= ir["phi"]
            rho[i, j] = r
    return _quad(ws, rho), float(ws.sum()), CR


def _across(K, S, CR, gamma):
    K, S, CR = map(np.asarray, (K, S, CR))
    S = np.clip(S, -K, K)                       # SIMM always clips S_b to [-K_b, K_b]
    tot = float(np.sum(K * K))
    for i in range(len(K)):
        for j in range(len(K)):
            if i != j:
                g = min(CR[i], CR[j]) / max(CR[i], CR[j])
                tot += gamma(i, j) * g * S[i] * S[j]
    return sqrt(max(tot, 0.0))


def ir_margin(sens, P, measure):
    by = defaultdict(list)
    for s in sens:
        if s.risk_class == "IR" and s.measure == measure:
            by[s.qualifier].append(s)
    if not by:
        return 0.0
    res = [_ir_bucket(v, P, measure) for v in by.values()]
    return _across([r[0] for r in res], [r[1] for r in res], [r[2] for r in res], lambda i, j: P["IR"]["gamma"])


# ---------------------------------------------------------------- equity
def _eq_buckets(sens, P, measure):
    eq = P["EQ"]
    by = defaultdict(lambda: defaultdict(float))
    for s in sens:
        if s.risk_class == "EQ" and s.measure == measure:
            by[s.bucket][s.qualifier] += s.value
    out = {}
    for b, names in by.items():
        rw = eq["rw"][b] if measure == "delta" else eq["vrw"] * eq["hvr"]
        thr = eq["delta_threshold"][b] if measure == "delta" else eq["vega_threshold"]
        ws, crs = [], []
        for v in names.values():
            cr = _cr(v, thr)
            ws.append(rw * v * cr)
            crs.append(cr)
        ws = np.array(ws)
        rho0 = eq["intra_corr"][b]
        n = len(ws)
        rho = np.full((n, n), rho0)
        for i in range(n):
            for j in range(n):
                if i != j:
                    rho[i, j] = rho0 * min(crs[i], crs[j]) / max(crs[i], crs[j])
        out[b] = (_quad(ws, rho), float(ws.sum()), float(np.sqrt(np.mean(np.square(crs)))))
    return out


def eq_margin(sens, P, measure):
    out = _eq_buckets(sens, P, measure)
    if not out:
        return 0.0
    names = [b for b in out if b != "residual"]
    inter = P["EQ"]["inter_corr"]
    K = [out[b][0] for b in names]
    S = np.clip([out[b][1] for b in names], -np.array(K), np.array(K)) if names else []
    tot = sum(k * k for k in K)
    for i, b in enumerate(names):
        for j, c in enumerate(names):
            if i != j:
                tot += inter[int(b) - 1][int(c) - 1] * S[i] * S[j]
    return sqrt(max(tot, 0.0)) + (out["residual"][0] if "residual" in out else 0.0)


# ---------------------------------------------------------------- FX
def fx_margin(sens, P):
    fx = P["FX"]
    by = defaultdict(float)
    for s in sens:
        if s.risk_class == "FX" and s.measure == "delta":
            by[s.qualifier] += s.value
    if not by:
        return 0.0
    ws = []
    for c, v in by.items():
        rw = fx["rw_high"] if c in fx["high_vol"] else fx["rw"]
        ws.append(rw * v * _cr(v, fx["delta_threshold"]))
    ws = np.array(ws)
    rho = np.full((len(ws), len(ws)), fx["corr"])
    return _quad(ws, rho)


# ---------------------------------------------------------------- curvature
def scaling(t_years):
    return 0.5 * min(1.0, (14 / 365) / t_years)


def curvature_margin(sens, P, risk_class):
    """From vega sensitivities: CVR_k = sum_expiries SF(t) x VR (no HVR/VRW), then lambda-theta."""
    by = defaultdict(lambda: defaultdict(float))
    rc = P[risk_class]
    for s in sens:
        if s.risk_class == risk_class and s.measure == "vega":
            b = s.qualifier if risk_class == "IR" else s.bucket
            t = TENOR_YEARS.get(s.label1, None) if risk_class == "IR" else rc["vega_expiry_years"].get(s.label1)
            by[b][(s.qualifier, s.label1)] += scaling(t) * s.value
    if not by:
        return 0.0
    K, S, allcvr = [], [], []
    for b, d in by.items():
        cvr = np.array(list(d.values()))
        allcvr += list(cvr)
        r = rc["curv_intra"] if risk_class == "IR" else rc["intra_corr"][b] ** 2
        rho = np.full((len(cvr), len(cvr)), r)
        K.append(_quad(cvr, rho))
        S.append(float(np.clip(cvr.sum(), -K[-1], K[-1])))
    allcvr = np.array(allcvr)
    theta = min(allcvr.sum() / np.abs(allcvr).sum(), 0.0) if np.abs(allcvr).sum() else 0.0
    z = norm.ppf(0.995)
    lam = (z * z - 1) * (1 + theta) - theta
    g = rc["gamma"] ** 2 if risk_class == "IR" else 0.0
    tot = sum(k * k for k in K) + sum(g * S[i] * S[j] for i in range(len(K)) for j in range(len(K)) if i != j)
    m = max(allcvr.sum() + lam * sqrt(max(tot, 0.0)), 0.0)
    return m * (rc.get("curvature_scale", 1.0))


# ---------------------------------------------------------------- totals
def simm(sens, P=None):
    P = params() if P is None else P
    rc = {
        "IR": ir_margin(sens, P, "delta") + ir_margin(sens, P, "vega") + curvature_margin(sens, P, "IR"),
        "FX": fx_margin(sens, P),
        "EQ": eq_margin(sens, P, "delta") + eq_margin(sens, P, "vega") + curvature_margin(sens, P, "EQ"),
    }
    psi = P["psi"]
    def combine(classes):
        x = np.array([rc[c] for c in classes])
        m = np.array([[1.0 if a == b else psi[a][b] for b in classes] for a in classes])
        return sqrt(float(x @ m @ x))
    products = {"RatesFX": combine(["IR", "FX"]), "Equity": combine(["EQ"])}
    return {"status": P["status"], "risk_classes": rc, "products": products, "SIMM": sum(products.values())}


if __name__ == "__main__":
    P = params()
    print(f"parameter set: {P['status']}")
    book = [Sens("IR", "delta", "USD", -45_000, label1="10y"), Sens("IR", "delta", "USD", 30_000, label1="2y"),
            Sens("IR", "delta", "USD", 8_000, label1="5y", subcurve="SOFR-3M"),
            Sens("IR", "delta", "EUR", 25_000, label1="5y"), Sens("IR", "delta", "USD", 5_000, label1="inflation"),
            Sens("IR", "vega", "USD", 2.0e6, label1="1y"), Sens("IR", "vega", "USD", -0.8e6, label1="5y"),
            Sens("FX", "delta", "EUR", 3e5), Sens("FX", "delta", "JPY", -2e5),          # per 1% move
            Sens("EQ", "delta", "ACME", 5e4, bucket="5"), Sens("EQ", "delta", "SPX", -8e4, bucket="11"),
            Sens("EQ", "vega", "SPX", -1.5e5, bucket="11", label1="1y")]
    r = simm(book, P)
    print("risk classes: " + ", ".join(f"{k} {v / 1e6:.2f}m" for k, v in r["risk_classes"].items()))
    print("products:     " + ", ".join(f"{k} {v / 1e6:.2f}m" for k, v in r["products"].items()))
    print(f"SIMM IM:      {r['SIMM'] / 1e6:.2f}m")
