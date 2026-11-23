"""SA-CCR: the Basel standardised approach for counterparty credit risk exposure (BCBS 279, CRE52).

    EAD = alpha (RC + PFE),  alpha = 1.4,  PFE = multiplier x AddOn

  adjusted notional   IR / credit: notional x supervisory duration (e^{-0.05 S} - e^{-0.05 E}) / 0.05
  supervisory delta   +-1 for linear trades; +-N(+-d1) for options with the supervisory vol
  maturity factor     sqrt(min(M, 1)) unmargined (M floored at 10 business days); 1.5 sqrt(MPoR) margined
  add-ons             IR: per currency, three maturity buckets with 1.4 / 0.6 cross terms
                      FX: per currency pair, full offset
                      credit / equity: entity add-ons combined with the single-factor formula
                      commodity: per hedging set, by commodity type, rho = 40%
  multiplier          min(1, 0.05 + 0.95 exp((V - C) / (2 x 0.95 x AddOn)))
  RC                  max(V - C, 0) unmargined; max(V - C, TH + MTA - NICA, 0) margined

Parameters are those of the Basel text; local rulebooks (CRR2, US) differ in places.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import exp, log, sqrt

from scipy.stats import norm

ALPHA = 1.4
SF = {"IR": 0.005, "FX": 0.04, "EQ_single": 0.32, "EQ_index": 0.20,
      "CR_AAA": 0.0038, "CR_AA": 0.0038, "CR_A": 0.0042, "CR_BBB": 0.0054, "CR_BB": 0.0106, "CR_B": 0.016,
      "CR_CCC": 0.06, "CR_IG_index": 0.0038, "CR_SG_index": 0.0106,
      "CO_electricity": 0.40, "CO_other": 0.18}
VOL = {"IR": 0.50, "FX": 0.15, "CR_single": 1.00, "CR_index": 0.80, "EQ_single": 1.20, "EQ_index": 0.75,
       "CO_electricity": 1.50, "CO_other": 0.70}
RHO = {"CR_single": 0.5, "CR_index": 0.8, "EQ_single": 0.5, "EQ_index": 0.8, "CO": 0.4}


@dataclass
class Trade:
    asset_class: str            # IR, FX, CR, EQ, CO
    hedging_set: str            # currency (IR), pair (FX), entity (CR/EQ), commodity set (CO)
    notional: float             # in the netting-set currency (for EQ/CO: market value of the underlying)
    mtm: float
    long: bool                  # long in the primary risk factor
    start: float = 0.0          # S (years)
    end: float = 1.0            # E (years); also the maturity M unless given
    maturity: float = None
    option: str = None          # None, 'call' or 'put'
    bought: bool = True
    price: float = None         # underlying price / forward rate (options)
    strike: float = None
    exercise: float = None      # option exercise time T
    rating: str = "A"           # credit: AAA..CCC, IG_index, SG_index
    index: bool = False         # equity / credit index
    commodity_type: str = "other"   # within a CO hedging set; 'electricity' gets the higher factor

    @property
    def M(self):
        return self.end if self.maturity is None else self.maturity


def supervisory_duration(S, E):
    return (exp(-0.05 * S) - exp(-0.05 * E)) / 0.05


def supervisory_vol(t: Trade):
    if t.asset_class == "CR":
        return VOL["CR_index" if t.index else "CR_single"]
    if t.asset_class == "EQ":
        return VOL["EQ_index" if t.index else "EQ_single"]
    if t.asset_class == "CO":
        return VOL["CO_electricity" if t.commodity_type == "electricity" else "CO_other"]
    return VOL[t.asset_class]


def supervisory_delta(t: Trade):
    if t.option is None:
        return 1.0 if t.long else -1.0
    s = supervisory_vol(t)
    T = t.exercise
    d1 = (log(t.price / t.strike) + 0.5 * s * s * T) / (s * sqrt(T))
    if t.option == "call":
        return norm.cdf(d1) if t.bought else -norm.cdf(d1)
    return -norm.cdf(-d1) if t.bought else norm.cdf(-d1)


def adjusted_notional(t: Trade):
    if t.asset_class in ("IR", "CR"):
        return t.notional * supervisory_duration(t.start, t.end)
    return t.notional


def maturity_factor(t: Trade, margined=False, mpor_days=10):
    if margined:
        return 1.5 * sqrt(mpor_days / 250)
    return sqrt(min(max(t.M, 10 / 250), 1.0))


def effective_notional(t: Trade, margined=False, mpor_days=10):
    return supervisory_delta(t) * adjusted_notional(t) * maturity_factor(t, margined, mpor_days)


def _sf(t: Trade):
    if t.asset_class == "IR":
        return SF["IR"]
    if t.asset_class == "FX":
        return SF["FX"]
    if t.asset_class == "EQ":
        return SF["EQ_index" if t.index else "EQ_single"]
    if t.asset_class == "CR":
        return SF["CR_" + t.rating]
    return SF["CO_electricity" if t.commodity_type == "electricity" else "CO_other"]


def addon(trades, margined=False, mpor_days=10):
    by = defaultdict(list)
    for t in trades:
        by[t.asset_class].append(t)
    out = {}
    # interest rates
    ir = defaultdict(lambda: [0.0, 0.0, 0.0])
    for t in by["IR"]:
        b = 0 if t.end < 1 else (1 if t.end <= 5 else 2)
        ir[t.hedging_set][b] += effective_notional(t, margined, mpor_days)
    tot = 0.0
    for D1, D2, D3 in ir.values():
        tot += SF["IR"] * sqrt(D1 * D1 + D2 * D2 + D3 * D3 + 1.4 * D1 * D2 + 1.4 * D2 * D3 + 0.6 * D1 * D3)
    out["IR"] = tot
    # FX
    fx = defaultdict(float)
    for t in by["FX"]:
        fx[t.hedging_set] += effective_notional(t, margined, mpor_days)
    out["FX"] = sum(SF["FX"] * abs(v) for v in fx.values())
    # credit and equity: single-factor aggregation over entities
    for ac in ("CR", "EQ"):
        ent = defaultdict(float)
        rho = {}
        for t in by[ac]:
            ent[t.hedging_set] += _sf(t) * effective_notional(t, margined, mpor_days)
            rho[t.hedging_set] = RHO[f"{ac}_{'index' if t.index else 'single'}"]
        sys_ = sum(rho[e] * a for e, a in ent.items())
        idio = sum((1 - rho[e] ** 2) * a * a for e, a in ent.items())
        out[ac] = sqrt(sys_ * sys_ + idio)
    # commodities: hedging set -> commodity types
    co = defaultdict(lambda: defaultdict(float))
    for t in by["CO"]:
        co[t.hedging_set][t.commodity_type] += _sf(t) * effective_notional(t, margined, mpor_days)
    tot = 0.0
    for types in co.values():
        sys_ = sum(RHO["CO"] * a for a in types.values())
        idio = sum((1 - RHO["CO"] ** 2) * a * a for a in types.values())
        tot += sqrt(sys_ * sys_ + idio)
    out["CO"] = tot
    out["total"] = sum(out.values())
    return out


@dataclass
class Collateral:
    C: float = 0.0              # net collateral held (after haircuts), + received
    margined: bool = False
    TH: float = 0.0
    MTA: float = 0.0
    NICA: float = 0.0
    mpor_days: int = 10


def ead(trades, coll: Collateral = Collateral()):
    V = sum(t.mtm for t in trades)
    a = addon(trades, coll.margined, coll.mpor_days)
    A = a["total"]
    mult = 1.0 if A == 0 else min(1.0, 0.05 + 0.95 * exp((V - coll.C) / (2 * 0.95 * A)))
    if coll.margined:
        rc = max(V - coll.C, coll.TH + coll.MTA - coll.NICA, 0.0)
    else:
        rc = max(V - coll.C, 0.0)
    out = {"V": V, "RC": rc, "addon": a, "multiplier": mult, "PFE": mult * A, "EAD": ALPHA * (rc + mult * A)}
    if coll.margined:                                    # margined EAD is capped at the unmargined one
        un = ead(trades, Collateral(coll.C))
        out["EAD"] = min(out["EAD"], un["EAD"])
    return out


def basel_examples():
    """The three unmargined netting sets of the Basel text's Annex 4a (amounts in thousands)."""
    ex1 = [
        Trade("IR", "USD", 10_000, 30, long=True, start=0, end=10),          # pay fixed 10y
        Trade("IR", "USD", 10_000, -20, long=False, start=0, end=4),         # receive fixed 4y
        Trade("IR", "EUR", 5_000, 50, long=False, start=1, end=11, maturity=1, option="put", bought=True,
              price=0.06, strike=0.05, exercise=1),                          # bought receiver 1y x 10y
    ]
    ex2 = [
        Trade("CR", "Firm A", 10_000, 20, long=True, end=3, rating="AA"),
        Trade("CR", "Firm B", 10_000, -40, long=False, end=6, rating="BBB"),
        Trade("CR", "CDX.IG", 10_000, 0, long=True, end=5, rating="IG_index", index=True),
    ]
    ex3 = [
        Trade("CO", "energy", 10_000, -50, long=True, end=0.75, commodity_type="crude oil"),
        Trade("CO", "energy", 20_000, -30, long=False, end=2, commodity_type="crude oil"),
        Trade("CO", "metals", 10_000, 100, long=True, end=5, commodity_type="silver"),
    ]
    return ex1, ex2, ex3


if __name__ == "__main__":
    ex1, ex2, ex3 = basel_examples()
    for name, ns, paper in [("Example 1 (rates)", ex1, 569), ("Example 2 (credit)", ex2, 381),
                            ("Example 3 (commodities)", ex3, 5406), ("Example 4 (1 + 2)", ex1 + ex2, None)]:
        r = ead(ns)
        print(f"{name:24s} RC {r['RC']:5.0f}  add-on {r['addon']['total']:7.1f}  multiplier {r['multiplier']:.3f}  "
              f"EAD {r['EAD']:8.1f}" + (f"   (Basel text: {paper:,})" if paper else ""))
    r = ead(ex1 + ex2, Collateral(C=40, margined=True, MTA=5))
    print(f"Example 4 margined (C = 40, MTA 5): EAD {r['EAD']:.1f}")
