"""ISDA SIMM (initial margin for uncleared derivatives): the calculation engine.

Risk classes: IR, CreditQ (CRQ), CreditNonQ (CRNQ), Equity (EQ), Commodity (CM), FX.

  IR delta      WS = RW_tenor x s x CR; sub-curves of one currency at correlation phi; inflation and
                cross-currency basis factors; currencies aggregated with gamma x g_bc, where
                g_bc = min(CR_b, CR_c) / max(CR_b, CR_c)
  IR vega       VR = HVR x VRW x (vega x implied vol) per expiry; same tenor correlations
  Bucketed classes (EQ, CRQ, CRNQ, CM)
                WS_k = RW_b x s_k x CR_k, CR_k from the qualifier's net sensitivity and the bucket
                threshold; K_b = sqrt(sum WS^2 + sum rho_kl f_kl WS_k WS_l), f_kl = min(CR)/max(CR);
                K = sqrt(sum K_b^2 + sum gamma_bc S_b S_c) + K_residual, S_b clipped to [-K_b, K_b]
                Risk factors: EQ and CM one per qualifier (vega summed over expiries); credit one per
                qualifier (issuer/seniority) and tenor, rho = rho_same within an issuer, rho_diff across
  CRQ base correlation  WS = RW_bc x s per index family, correlated at rho_bc, entering the CRQ delta
                aggregation as one extra bucket with zero gamma (check this against the methodology)
  Curvature     CVR = sum SF(t) x (vega x vol), SF(t) = 0.5 min(1, 14 days / t); rho^2 within and
                gamma^2 across buckets; residual buckets on their own; the lambda / theta adjustment
                from the 99.5% quantile: lambda = (z^2 - 1)(1 + theta) - theta
  FX delta      per currency, concentration thresholds
  Totals        within a product class (RatesFX, Credit, Equity, Commodity) the six risk-class margins
                are combined with psi; SIMM = sum of product classes

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
RISK_CLASSES = ("IR", "CRQ", "CRNQ", "EQ", "CM", "FX")
BUCKETED = ("EQ", "CRQ", "CRNQ", "CM")
CREDIT = ("CRQ", "CRNQ")
DEFAULT_PRODUCT = {"IR": "RatesFX", "FX": "RatesFX", "CRQ": "Credit", "CRNQ": "Credit", "EQ": "Equity", "CM": "Commodity"}


def params(path=PARAMS_FILE):
    return json.loads(Path(path).read_text())


@dataclass
class Sens:
    risk_class: str            # IR, CRQ, CRNQ, EQ, CM, FX
    measure: str               # delta, vega; CRQ also basecorr
    qualifier: str             # currency (IR, FX), issuer (EQ), issuer/seniority (credit), commodity, index (basecorr)
    value: float               # delta: per 1bp (IR, credit) / per 1% (EQ, CM, FX, basecorr); vega: vega x implied vol
    bucket: str = None         # bucket ("1".., "residual") for EQ, CRQ, CRNQ, CM
    label1: str = None         # IR / credit tenor, vol expiry, or "inflation" / "xccy"
    subcurve: str = "OIS"      # IR sub-curve
    product: str = None        # SIMM product class; defaults by risk class


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


# ---------------------------------------------------------------- bucketed classes: EQ, CRQ, CRNQ, CM
def _factor(rc, s):
    """Risk-factor key within a bucket: credit has a tenor / expiry dimension, EQ and CM do not."""
    return (s.qualifier, s.label1) if rc in CREDIT else (s.qualifier,)


def intra_corr(P, rc, bucket, a, b):
    """Correlation between two risk-factor keys in one bucket (before the concentration factor)."""
    p = P[rc]
    if rc in CREDIT:
        same = a[0] == b[0]
        if bucket == "residual":
            return p["intra_corr"]["residual_same" if same else "residual_diff"]
        return p["intra_corr"]["same" if same else "diff"]
    return p["intra_corr"][bucket]


def inter_corr(P, rc, b, c):
    return P[rc]["inter_corr"][int(b) - 1][int(c) - 1]


def bucket_margins(sens, P, rc, measure):
    """{bucket: (K_b, S_b)} for one bucketed risk class and measure (delta or vega)."""
    p = P[rc]
    by = defaultdict(lambda: defaultdict(float))
    for s in sens:
        if s.risk_class == rc and s.measure == measure:
            by[s.bucket][_factor(rc, s)] += s.value
    out = {}
    for b, factors in by.items():
        if measure == "delta":
            rw, thr = p["rw"][b], p["delta_threshold"][b]
        else:
            rw = p["vrw"] * p.get("hvr", 1.0)
            vt = p["vega_threshold"]
            thr = vt[b] if isinstance(vt, dict) else vt
        net = defaultdict(float)
        for k, v in factors.items():
            net[k[0]] += v                              # concentration is per qualifier
        cr = {q: _cr(v, thr) for q, v in net.items()}
        keys = list(factors)
        ws = np.array([rw * factors[k] * cr[k[0]] for k in keys])
        n = len(keys)
        rho = np.eye(n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    ci, cj = cr[keys[i][0]], cr[keys[j][0]]
                    rho[i, j] = intra_corr(P, rc, b, keys[i], keys[j]) * min(ci, cj) / max(ci, cj)
        out[b] = (_quad(ws, rho), float(ws.sum()))
    return out


def _combine_buckets(out, P, rc, extra=()):
    """sqrt(sum K_b^2 + sum gamma S_b S_c) over non-residual buckets, plus the residual K.
    extra: further (K, S) groups with zero gamma to everything (CRQ base correlation)."""
    names = [b for b in out if b != "residual"]
    K = np.array([out[b][0] for b in names])
    S = np.clip([out[b][1] for b in names], -K, K) if names else np.array([])
    tot = float(np.sum(K * K)) + sum(k * k for k, _ in extra)
    for i, b in enumerate(names):
        for j, c in enumerate(names):
            if i != j:
                tot += inter_corr(P, rc, b, c) * S[i] * S[j]
    return sqrt(max(tot, 0.0)) + (out["residual"][0] if "residual" in out else 0.0)


def basecorr_margin(sens, P):
    """(K, S) of the CRQ base-correlation risk factors (one per index family), or None."""
    by = defaultdict(float)
    for s in sens:
        if s.risk_class == "CRQ" and s.measure == "basecorr":
            by[s.qualifier] += s.value
    if not by:
        return None
    ws = np.array([P["CRQ"]["basecorr_rw"] * v for v in by.values()])
    rho = np.full((len(ws), len(ws)), P["CRQ"]["basecorr_corr"])
    return _quad(ws, rho), float(ws.sum())


def bucketed_margin(sens, P, rc, measure):
    out = bucket_margins(sens, P, rc, measure)
    extra = []
    if rc == "CRQ" and measure == "delta":
        bc = basecorr_margin(sens, P)
        if bc:
            extra.append(bc)
    if not out and not extra:
        return 0.0
    return _combine_buckets(out, P, rc, extra)


def eq_margin(sens, P, measure):
    return bucketed_margin(sens, P, "EQ", measure)


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


def _lambda(cvr_sum, cvr_abs):
    theta = min(cvr_sum / cvr_abs, 0.0) if cvr_abs else 0.0
    z = norm.ppf(0.995)
    return (z * z - 1) * (1 + theta) - theta


def _ir_curvature(sens, P):
    by = defaultdict(lambda: defaultdict(float))
    rc = P["IR"]
    for s in sens:
        if s.risk_class == "IR" and s.measure == "vega":
            by[s.qualifier][s.label1] += scaling(TENOR_YEARS[s.label1]) * s.value
    if not by:
        return 0.0
    K, S, allcvr = [], [], []
    for d in by.values():
        cvr = np.array(list(d.values()))
        allcvr += list(cvr)
        K.append(_quad(cvr, np.full((len(cvr), len(cvr)), rc["curv_intra"])))
        S.append(float(np.clip(cvr.sum(), -K[-1], K[-1])))
    allcvr = np.array(allcvr)
    lam = _lambda(allcvr.sum(), np.abs(allcvr).sum())
    g = rc["gamma"] ** 2
    tot = sum(k * k for k in K) + sum(g * S[i] * S[j] for i in range(len(K)) for j in range(len(K)) if i != j)
    return max(allcvr.sum() + lam * sqrt(max(tot, 0.0)), 0.0) * rc.get("curvature_scale", 1.0)


def _bucketed_curvature(sens, P, rc):
    by = defaultdict(lambda: defaultdict(float))
    for s in sens:
        if s.risk_class == rc and s.measure == "vega":
            by[s.bucket][_factor(rc, s)] += scaling(TENOR_YEARS[s.label1]) * s.value
    if not by:
        return 0.0

    def group(buckets):
        if not buckets:
            return 0.0
        K, S, allcvr = [], [], []
        for b in buckets:
            keys = list(by[b])
            cvr = np.array([by[b][k] for k in keys])
            allcvr += list(cvr)
            rho = np.array([[intra_corr(P, rc, b, x, y) ** 2 for y in keys] for x in keys])
            K.append(_quad(cvr, rho))
            S.append(float(np.clip(cvr.sum(), -K[-1], K[-1])))
        tot = sum(k * k for k in K)
        for i, b in enumerate(buckets):
            for j, c in enumerate(buckets):
                if i != j:
                    tot += inter_corr(P, rc, b, c) ** 2 * S[i] * S[j]
        allcvr = np.array(allcvr)
        return max(allcvr.sum() + _lambda(allcvr.sum(), np.abs(allcvr).sum()) * sqrt(max(tot, 0.0)), 0.0)

    m = group([b for b in by if b != "residual"]) + group([b for b in by if b == "residual"])
    return m * P[rc].get("curvature_scale", 1.0)


def curvature_margin(sens, P, risk_class):
    """From vega sensitivities: CVR = sum over expiries SF(t) x (vega x vol), then lambda-theta."""
    return _ir_curvature(sens, P) if risk_class == "IR" else _bucketed_curvature(sens, P, risk_class)


# ---------------------------------------------------------------- totals
def risk_class_margins(sens, P):
    rc = {"IR": ir_margin(sens, P, "delta") + ir_margin(sens, P, "vega") + curvature_margin(sens, P, "IR"),
          "FX": fx_margin(sens, P)}
    for c in BUCKETED:
        rc[c] = bucketed_margin(sens, P, c, "delta") + bucketed_margin(sens, P, c, "vega") + curvature_margin(sens, P, c)
    return {c: rc[c] for c in RISK_CLASSES}


def simm(sens, P=None):
    P = params() if P is None else P
    psi = P["psi"]
    m = np.array([[1.0 if a == b else psi[a][b] for b in RISK_CLASSES] for a in RISK_CLASSES])
    by = defaultdict(list)
    for s in sens:
        by[s.product or DEFAULT_PRODUCT[s.risk_class]].append(s)
    products, detail, total_rc = {}, {}, defaultdict(float)
    for prod, items in by.items():
        rc = risk_class_margins(items, P)
        x = np.array([rc[c] for c in RISK_CLASSES])
        products[prod] = sqrt(float(x @ m @ x))
        detail[prod] = rc
        for c, v in rc.items():
            total_rc[c] += v
    return {"status": P["status"], "risk_classes": {c: v for c, v in total_rc.items() if v},
            "products": products, "by_product": detail, "SIMM": sum(products.values())}


if __name__ == "__main__":
    P = params()
    print(f"parameter set: {P['status']}")
    book = [Sens("IR", "delta", "USD", -45_000, label1="10y"), Sens("IR", "delta", "USD", 30_000, label1="2y"),
            Sens("IR", "delta", "USD", 8_000, label1="5y", subcurve="SOFR-3M"),
            Sens("IR", "delta", "EUR", 25_000, label1="5y"), Sens("IR", "delta", "USD", 5_000, label1="inflation"),
            Sens("IR", "vega", "USD", 2.0e6, label1="1y"), Sens("IR", "vega", "USD", -0.8e6, label1="5y"),
            Sens("FX", "delta", "EUR", 3e5), Sens("FX", "delta", "JPY", -2e5),          # per 1% move
            Sens("EQ", "delta", "ACME", 5e4, bucket="5"), Sens("EQ", "delta", "SPX", -8e4, bucket="11"),
            Sens("EQ", "vega", "SPX", -1.5e5, bucket="11", label1="1y"),
            # credit: CS01 per bp by tenor; an index tranche's base-correlation risk per 1%
            Sens("CRQ", "delta", "ACME/senior", 4_000, bucket="4", label1="5y"),
            Sens("CRQ", "delta", "ACME/senior", -1_500, bucket="4", label1="10y"),
            Sens("CRQ", "delta", "BANKCO/senior", -2_500, bucket="2", label1="5y"),
            Sens("CRQ", "basecorr", "CDX IG", 3e4),
            Sens("CRQ", "vega", "CDX IG", 2e5, bucket="12", label1="1y"),
            Sens("CRNQ", "delta", "RMBS pool 7", 800, bucket="1", label1="5y"),
            # commodity: per 1% price move
            Sens("CM", "delta", "WTI", 6e4, bucket="2"), Sens("CM", "delta", "Brent", -5e4, bucket="2"),
            Sens("CM", "delta", "Gold", 3e4, bucket="12"), Sens("CM", "vega", "WTI", 1e5, bucket="2", label1="6m")]
    r = simm(book, P)
    print("risk classes: " + ", ".join(f"{k} {v / 1e6:.2f}m" for k, v in r["risk_classes"].items()))
    print("products:     " + ", ".join(f"{k} {v / 1e6:.2f}m" for k, v in r["products"].items()))
    print(f"SIMM IM:      {r['SIMM'] / 1e6:.2f}m")
