"""CDS engine following the ISDA CDS Standard Model conventions (the "JPMCDS" model behind the
Markit upfront converter).

  Dates     standard coupon dates 20 Mar/Jun/Sep/Dec, adjusted to the next weekday (no holiday
            calendar here); step-in T+1; cash settlement T+3 weekdays; accrual from the last coupon
            date on or before step-in; the last period runs to maturity inclusive (one extra day);
            ACT/360 accrual, ACT/365F for curve times measured from the trade date
  Curves    discount and hazard curves both piecewise-flat forward (linear in rate x time)
  Legs      protection leg and accrual-on-default integrated exactly on every sub-interval where
            both forward rates are flat (the ISDA closed forms); accrued paid at default
  Quotes    conventional spread <-> points upfront through a flat hazard rate; clean vs dirty;
            term-structure bootstrap from par spreads; CS01, recovery sensitivity, jump-to-default
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from math import exp, log

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------- dates
def _weekday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def add_weekdays(d: date, n: int) -> date:
    while n > 0:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def imm_dates(start_year: int, end_year: int):
    return [date(y, m, 20) for y in range(start_year, end_year + 1) for m in (3, 6, 9, 12)]


def standard_maturity(trade: date, tenor_years: int) -> date:
    """Semi-annual roll (since Dec 2015): roll dates 20 Mar and 20 Sep; maturities fall on 20 Jun / 20 Dec."""
    y, md = trade.year, (trade.month, trade.day)
    if md < (3, 20):
        base = date(y - 1, 12, 20)
    elif md < (9, 20):
        base = date(y, 6, 20)
    else:
        base = date(y, 12, 20)
    return date(base.year + tenor_years, base.month, 20)


def accrual_start(stepin: date) -> date:
    """Last adjusted coupon date on or before step-in."""
    c = [_weekday(d) for d in imm_dates(stepin.year - 1, stepin.year)]
    return max(d for d in c if d <= stepin)


def schedule(stepin: date, maturity: date):
    """Accrual periods (start, end, pay) with the ISDA conventions."""
    start = accrual_start(stepin)
    unadj = [d for d in imm_dates(start.year, maturity.year) if start < _weekday(d) and d <= maturity]
    out, s = [], start
    for i, d in enumerate(unadj):
        last = i == len(unadj) - 1
        end = maturity + timedelta(days=1) if last else _weekday(d)
        pay = _weekday(d)
        out.append((s, end, pay))
        s = end
    return out


# ---------------------------------------------------------------- curves (flat forward)
@dataclass
class FlatForward:
    """Node times (years) and zero rates; r(t) t is linear between nodes, flat extrapolation of the
    last forward. Used for discount curves (rates) and survival curves (hazards)."""
    times: np.ndarray
    rates: np.ndarray

    def rt(self, t):
        ts = np.concatenate([[0.0], self.times])
        rt = np.concatenate([[0.0], self.times * self.rates])
        if t <= ts[-1]:
            return float(np.interp(t, ts, rt))
        f = (rt[-1] - rt[-2]) / (ts[-1] - ts[-2]) if len(ts) > 2 else self.rates[-1]
        return float(rt[-1] + f * (t - ts[-1]))

    def df(self, t):
        return exp(-self.rt(t))

    def knots(self):
        return list(self.times)


def flat(rate, horizon=40.0):
    return FlatForward(np.array([horizon]), np.array([rate]))


# ---------------------------------------------------------------- legs
def _yf(a: date, b: date) -> float:
    return (b - a).days / 365.0


def _grid(t0, t1, *curves):
    pts = {t0, t1}
    for c in curves:
        pts.update(t for t in c.knots() if t0 < t < t1)
    return sorted(pts)


def protection_leg(trade, stepin, maturity, disc, haz, R):
    t0, t1 = _yf(trade, stepin) - 1 / 365.0, _yf(trade, maturity + timedelta(days=1))
    t0 = max(t0, 0.0)
    pts = _grid(t0, t1, disc, haz)
    out = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        dt = b - a
        lam = (haz.rt(b) - haz.rt(a)) / dt
        fwd = (disc.rt(b) - disc.rt(a)) / dt
        s = lam + fwd
        QP = haz.df(a) * disc.df(a)
        x = s * dt
        frac = (1 - exp(-x)) / x if abs(x) > 1e-8 else 1 - x / 2
        out += lam * QP * dt * frac
    return (1 - R) * out


def premium_leg(trade, stepin, maturity, disc, haz, coupon=1.0, accrued_on_default=True):
    """Risky value of paying `coupon` per year (ACT/360), including the full first period."""
    total = 0.0
    for s, e, pay in schedule(stepin, maturity):
        acc = (e - s).days / 360.0
        te, tp = _yf(trade, e) - 1 / 365.0, _yf(trade, pay)
        total += coupon * acc * disc.df(tp) * haz.df(te)
        if accrued_on_default:
            ts = _yf(trade, s) - 1 / 365.0
            a0 = max(_yf(trade, stepin) - 1 / 365.0, ts)
            pts = _grid(a0, te, disc, haz)
            for a, b in zip(pts[:-1], pts[1:]):
                dt = b - a
                lam = (haz.rt(b) - haz.rt(a)) / dt
                fwd = (disc.rt(b) - disc.rt(a)) / dt
                k = lam + fwd
                QP = haz.df(a) * disc.df(a)
                tn = a - ts                       # time already accrued at the sub-interval start
                if abs(k * dt) > 1e-8:
                    I = tn * (1 - exp(-k * dt)) / k + (1 - exp(-k * dt) * (1 + k * dt)) / (k * k)
                else:
                    I = tn * dt + dt * dt / 2
                total += coupon * (365.0 / 360.0) * lam * QP * I
    return total


@dataclass
class CDS:
    trade: date
    maturity: date
    coupon: float = 0.01          # standard running coupon (100bp or 500bp)
    recovery: float = 0.4
    notional: float = 1e7

    @property
    def stepin(self):
        return self.trade + timedelta(days=1)

    @property
    def settle(self):
        return add_weekdays(self.trade, 3)

    def accrued(self):
        return self.coupon * (self.stepin - accrual_start(self.stepin)).days / 360.0

    def rpv01(self, disc, haz, clean=True):
        """Risky annuity per unit coupon, as of the cash settlement date."""
        dirty = premium_leg(self.trade, self.stepin, self.maturity, disc, haz) / disc.df(_yf(self.trade, self.settle))
        return dirty - (self.stepin - accrual_start(self.stepin)).days / 360.0 if clean else dirty

    def protection(self, disc, haz):
        return protection_leg(self.trade, self.stepin, self.maturity, disc, haz, self.recovery) / disc.df(_yf(self.trade, self.settle))

    def par_spread(self, disc, haz):
        return self.protection(disc, haz) / self.rpv01(disc, haz)

    def upfront(self, disc, haz):
        """Clean points upfront (fraction of notional) paid by the protection buyer."""
        return self.protection(disc, haz) - self.coupon * self.rpv01(disc, haz)

    def cash_settlement(self, disc, haz):
        """Amount the buyer pays at T+3: upfront net of the accrued the seller hands back."""
        return self.notional * (self.upfront(disc, haz) - self.accrued())

    # ---- quote conversions (flat hazard, as the Markit converter)
    def hazard_from_spread(self, spread, disc):
        f = lambda h: self.par_spread(disc, flat(h)) - spread
        return brentq(f, 1e-8, 5.0, xtol=1e-14)

    def upfront_from_spread(self, spread, disc):
        return self.upfront(disc, flat(self.hazard_from_spread(spread, disc)))

    def spread_from_upfront(self, puf, disc):
        h = brentq(lambda h: self.upfront(disc, flat(h)) - puf, 1e-8, 5.0, xtol=1e-14)
        return self.par_spread(disc, flat(h))

    def risk(self, spread, disc):
        base = self.upfront_from_spread(spread, disc)
        cs01 = self.notional * (self.upfront_from_spread(spread + 1e-4, disc) - base)
        h = self.hazard_from_spread(spread, disc)
        rec = CDS(self.trade, self.maturity, self.coupon, self.recovery + 0.01, self.notional)
        rr01 = self.notional * (rec.upfront(disc, flat(h)) - self.upfront(disc, flat(h)))
        jtd = self.notional * ((1 - self.recovery) - base)
        return {"upfront": base, "price": 100 * (1 - base), "cs01": cs01, "rec01": rr01, "jtd": jtd}


def bootstrap(trade, tenors, spreads, disc, recovery=0.4):
    """Piecewise-flat hazard curve repricing par CDS at standard maturities."""
    times, rates = [], []
    for n, s in zip(tenors, spreads):
        mat = standard_maturity(trade, n)
        c = CDS(trade, mat, s, recovery)
        t = _yf(trade, mat + timedelta(days=1))
        def f(h):
            rt_prev = times[-1] * rates[-1] if times else 0.0
            t_prev = times[-1] if times else 0.0
            z = (rt_prev + h * (t - t_prev)) / t
            curve = FlatForward(np.array(times + [t]), np.array(rates + [z]))
            return c.upfront(disc, curve)
        h = brentq(f, 1e-8, 5.0, xtol=1e-14)
        rt_prev = times[-1] * rates[-1] if times else 0.0
        t_prev = times[-1] if times else 0.0
        times.append(t)
        rates.append((rt_prev + h * (t - t_prev)) / t)
    return FlatForward(np.array(times), np.array(rates))


if __name__ == "__main__":
    trade = date(2026, 11, 24)
    disc = flat(0.038)
    mat = standard_maturity(trade, 5)
    print(f"trade {trade}, step-in {trade + timedelta(days=1)}, settle {add_weekdays(trade, 3)}, "
          f"accrual start {accrual_start(trade + timedelta(days=1))}, 5y maturity {mat}")
    for coupon, spread in [(0.01, 0.0065), (0.01, 0.0150), (0.05, 0.0420), (0.05, 0.0900)]:
        c = CDS(trade, mat, coupon)
        r = c.risk(spread, disc)
        print(f"coupon {coupon * 1e4:3.0f}bp quoted {spread * 1e4:4.0f}bp -> upfront {100 * r['upfront']:6.3f}%  "
              f"price {r['price']:7.3f}  cash {c.notional * (r['upfront'] - c.accrued()):>11,.0f}  CS01 {r['cs01']:>8,.0f}  "
              f"back to spread {1e4 * c.spread_from_upfront(r['upfront'], disc):.4f}bp")
    hz = bootstrap(trade, [1, 3, 5, 7, 10], [0.0040, 0.0060, 0.0085, 0.0100, 0.0115], disc)
    print("bootstrapped forward hazards:", np.round(np.diff(np.concatenate([[0], hz.times * hz.rates])) / np.diff(np.concatenate([[0], hz.times])), 5))
