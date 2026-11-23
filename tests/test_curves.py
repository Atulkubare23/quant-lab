import math

import numpy as np
import pytest

from qlab import curves as C
from qlab.dual import Dual, exp, log, sqrt


def test_dual_derivatives_match_finite_differences():
    f = lambda x, y: exp(x * y) / (1 + x) + log(y) * sqrt(x) - x ** 3 / y
    x, y = Dual.variables([0.7, 1.3])
    v = f(x, y)
    h = 1e-6
    fx = (f(0.7 + h, 1.3) - f(0.7 - h, 1.3)) / (2 * h)
    fy = (f(0.7, 1.3 + h) - f(0.7, 1.3 - h)) / (2 * h)
    assert v.x == pytest.approx(f(0.7, 1.3))
    assert v.g == pytest.approx([fx, fy], rel=1e-7)


def test_curves_reprice_every_instrument():
    cs = C.standard_set(C.SAMPLE_OIS, C.SAMPLE_TERM).solve()
    for i in cs.instruments:
        assert C.value(i.rate(cs.curves)) == pytest.approx(i.quote, abs=1e-12)
    assert C.swap_pv(cs.curves, 7, C.SAMPLE_TERM[7]) == pytest.approx(0, abs=1e-6)   # at-par swap has PV 0


def test_flat_continuous_curve_is_recovered():
    z = 0.04
    ois = {T: (1 - math.exp(-z * T)) / sum((min(T, k + 1) - k) * math.exp(-z * min(T, k + 1))
                                          for k in range(math.ceil(T))) for T in [0.5, 1, 2, 5, 10]}
    term = {T: v for T, v in ois.items() if T >= 1}
    term[0.25] = (math.exp(z * 0.25) - 1) / 0.25
    cs = C.standard_set(ois, term).solve()
    for t in [0.5, 1, 3, 7, 10]:
        assert cs.curves["ois"].zero(t) == pytest.approx(z, abs=1e-12)
        assert cs.curves["term"].zero(t) == pytest.approx(z, abs=1e-6)


def test_par_rate_risk_matches_bump_and_rebuild():
    trade = lambda c: C.swap_pv(c, 7, 0.04)
    cs = C.standard_set(C.SAMPLE_OIS, C.SAMPLE_TERM).solve()
    risk = cs.risk(trade)
    base = cs.pv(trade)
    for j in [3, 5, 6, 12, 14, 15]:                     # a mix of OIS and term pillars
        o, t = dict(C.SAMPLE_OIS), dict(C.SAMPLE_TERM)
        name = cs.instruments[j].name
        T = float(name.split()[1][:-1])
        (o if name.startswith("OIS") else t)[T] += 1e-4
        fd = C.standard_set(o, t).solve().pv(trade) - base
        assert risk[j] == pytest.approx(fd, rel=2e-3, abs=0.05)
    assert abs(risk[:10]).sum() < abs(risk[10:]).sum()       # mostly term-curve risk
    assert risk[[i.name for i in cs.instruments].index("term 7y")] < -600   # receiver loses when rates rise
