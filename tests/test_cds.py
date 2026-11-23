from datetime import date, timedelta
from math import exp

import numpy as np
import pytest

from qlab import cds as C

TRADE = date(2026, 11, 24)
DISC = C.flat(0.038)


def test_standard_dates():
    assert C.standard_maturity(date(2026, 11, 24), 5) == date(2031, 12, 20)
    assert C.standard_maturity(date(2026, 3, 19), 5) == date(2030, 12, 20)
    assert C.standard_maturity(date(2026, 3, 20), 5) == date(2031, 6, 20)
    assert C.accrual_start(date(2026, 11, 25)) == date(2026, 9, 21)      # 20 Sep 2026 is a Sunday
    sch = C.schedule(date(2026, 11, 25), date(2031, 12, 20))
    assert sch[0][0] == date(2026, 9, 21) and sch[-1][1] == date(2031, 12, 21)   # last period includes maturity
    assert all(p.weekday() < 5 for _, _, p in sch) and len(sch) == 21


def test_quote_conventions_are_consistent():
    c = C.CDS(TRADE, C.standard_maturity(TRADE, 5), 0.01)
    assert c.upfront_from_spread(0.01, DISC) == pytest.approx(0, abs=1e-12)        # quoted = coupon: no upfront
    for s in (0.003, 0.025, 0.08):
        u = c.upfront_from_spread(s, DISC)
        assert c.spread_from_upfront(u, DISC) == pytest.approx(s, abs=1e-12)
        assert np.sign(u) == np.sign(s - 0.01)
        assert u == pytest.approx((s - 0.01) * c.rpv01(DISC, C.flat(c.hazard_from_spread(s, DISC))), rel=1e-12)


def test_credit_triangle_and_accrual_on_default():
    c = C.CDS(TRADE, C.standard_maturity(TRADE, 5), 0.01, 0.4)
    h = C.flat(0.02)
    assert c.par_spread(DISC, h) == pytest.approx(0.02 * 0.6 * 360 / 365, abs=1e-4)   # hazard x LGD, ACT/360 coupons
    with_aod = C.premium_leg(TRADE, c.stepin, c.maturity, DISC, h)
    without = C.premium_leg(TRADE, c.stepin, c.maturity, DISC, h, accrued_on_default=False)
    assert 0 < with_aod - without < 0.01 * with_aod


def test_exact_legs_match_daily_integration():
    haz = C.FlatForward(np.array([1.0, 3.0, 7.0]), np.array([0.01, 0.02, 0.03]))
    disc = C.FlatForward(np.array([0.5, 2.0, 10.0]), np.array([0.03, 0.035, 0.04]))
    c = C.CDS(TRADE, C.standard_maturity(TRADE, 5), 0.01, 0.4)
    exact = C.protection_leg(TRADE, c.stepin, c.maturity, disc, haz, 0.4)
    t0, t1 = 0.0, C._yf(TRADE, c.maturity + timedelta(days=1))
    ts = np.linspace(t0, t1, 200_001)
    Q = np.array([haz.df(t) for t in ts]); P = np.array([disc.df(t) for t in ts])
    numeric = 0.6 * np.sum(-np.diff(Q) * 0.5 * (P[1:] + P[:-1]))
    assert exact == pytest.approx(numeric, rel=1e-6)


def test_bootstrap_reprices_par_quotes_and_risk_signs():
    tenors, spreads = [1, 3, 5, 7, 10], [0.0040, 0.0060, 0.0085, 0.0100, 0.0115]
    hz = C.bootstrap(TRADE, tenors, spreads, DISC)
    for n, s in zip(tenors, spreads):
        c = C.CDS(TRADE, C.standard_maturity(TRADE, n), s)
        assert c.upfront(DISC, hz) == pytest.approx(0, abs=1e-12)
    c = C.CDS(TRADE, C.standard_maturity(TRADE, 5), 0.01)
    r = c.risk(0.0101, DISC)
    assert r["cs01"] > 0 and r["jtd"] > 0 and r["rec01"] < 0            # buyer gains when spreads widen
    assert r["cs01"] == pytest.approx(c.notional * 1e-4 * c.rpv01(DISC, C.flat(c.hazard_from_spread(0.0101, DISC))), rel=2e-3)
