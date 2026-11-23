from datetime import date

import pytest

from qlab import calendars as K
from qlab import cds as C

D = date


def test_easter():
    for y, d in [(2008, D(2008, 3, 23)), (2024, D(2024, 3, 31)), (2025, D(2025, 4, 20)), (2026, D(2026, 4, 5)), (2038, D(2038, 4, 25))]:
        assert K.easter(y) == d


def test_target():
    assert K.TARGET.holidays(2026) == {D(2026, 1, 1), D(2026, 4, 3), D(2026, 4, 6), D(2026, 5, 1), D(2026, 12, 25)}  # 26 Dec is a Saturday
    assert D(2001, 12, 31) in K.TARGET.holidays(2001) and D(2002, 12, 31) not in K.TARGET.holidays(2002)


def test_new_york_observance():
    h22 = K.USNY.holidays(2022)
    assert {D(2022, 1, 17), D(2022, 6, 20), D(2022, 7, 4), D(2022, 11, 24), D(2022, 12, 26)} <= h22   # Juneteenth, Christmas on Sundays
    assert D(2021, 6, 18) not in K.USNY.holidays(2021) and D(2021, 7, 5) in K.USNY.holidays(2021)
    assert D(2027, 12, 24) not in K.USNY.holidays(2027)      # Saturday Christmas: the Fed stays open on Friday
    assert len(K.USNY.holidays(2026)) == 10                    # 4 Jul 2026 is a Saturday


def test_london_moves_and_one_offs():
    assert K.LONDON.holidays(2022) == {D(2022, 1, 3), D(2022, 4, 15), D(2022, 4, 18), D(2022, 5, 2), D(2022, 6, 2), D(2022, 6, 3),
                                       D(2022, 8, 29), D(2022, 9, 19), D(2022, 12, 26), D(2022, 12, 27)}
    assert D(2020, 5, 8) in K.LONDON.holidays(2020) and D(2020, 5, 4) not in K.LONDON.holidays(2020)
    assert D(2023, 5, 8) in K.LONDON.holidays(2023)
    assert {D(2026, 12, 25), D(2026, 12, 28)} <= K.LONDON.holidays(2026)   # Boxing Day Saturday -> Monday


def test_tokyo_substitute_citizens_and_special_years():
    h19 = K.TOKYO.holidays(2019)
    assert {D(2019, 4, 29), D(2019, 4, 30), D(2019, 5, 1), D(2019, 5, 2), D(2019, 5, 3), D(2019, 5, 6), D(2019, 10, 22)} <= h19
    assert D(2019, 12, 23) not in h19 and D(2019, 2, 25) not in h19      # no Emperor's Birthday in 2019
    h20 = K.TOKYO.holidays(2020)
    assert {D(2020, 2, 24), D(2020, 7, 23), D(2020, 7, 24), D(2020, 8, 10)} <= h20 and D(2020, 10, 12) not in h20
    assert {D(2015, 9, 21), D(2015, 9, 22), D(2015, 9, 23)} <= K.TOKYO.holidays(2015)   # Silver Week
    assert {D(2026, 9, 21), D(2026, 9, 22), D(2026, 9, 23), D(2026, 3, 20), D(2026, 5, 6)} <= K.TOKYO.holidays(2026)
    assert D(2026, 12, 31) in K.TOKYO.holidays(2026) and D(2026, 1, 2) in K.TOKYO.holidays(2026)


def test_adjust_add_and_joint():
    cal = K.LONDON
    assert cal.adjust(D(2026, 12, 25)) == D(2026, 12, 29)
    assert cal.adjust(D(2026, 12, 25), "preceding") == D(2026, 12, 24)
    assert cal.adjust(D(2026, 5, 30), "modified following") == D(2026, 5, 29)    # Sat 30 May: following is Mon 1 Jun, next month
    assert cal.add_business_days(D(2026, 12, 24), 3) == D(2026, 12, 31)
    assert cal.add_business_days(D(2026, 12, 31), -3) == D(2026, 12, 24)
    assert cal.business_days_between(D(2026, 12, 24), D(2026, 12, 31)) == 3
    j = K.joint(K.USNY, K.TARGET)
    assert not j.is_business_day(D(2026, 5, 1)) and not j.is_business_day(D(2026, 11, 26)) and j.is_business_day(D(2026, 11, 27))
    with pytest.raises(ValueError):
        cal.adjust(D(2026, 12, 25), "nearest")


def test_cds_dates_follow_the_calendar():
    # 20 Mar 2026 is the vernal equinox in Japan: the coupon rolls to Monday 23 Mar
    assert C.accrual_start(D(2026, 4, 1)) == D(2026, 3, 20)
    assert C.accrual_start(D(2026, 4, 1), K.TOKYO) == D(2026, 3, 23)
    mat = C.standard_maturity(D(2026, 4, 1), 5)
    for _, _, pay in C.schedule(D(2026, 4, 1), mat, K.TOKYO):
        assert K.TOKYO.is_business_day(pay)
    c = C.CDS(D(2026, 12, 24), C.standard_maturity(D(2026, 12, 24), 5), calendar=K.LONDON)
    assert c.settle == D(2026, 12, 31)
    w = C.CDS(D(2026, 3, 31), mat, calendar=K.WEEKENDS)
    t = C.CDS(D(2026, 3, 31), mat, calendar=K.TOKYO)
    assert t.accrued() == pytest.approx(w.accrued() - 3 * w.coupon / 360)    # three fewer days accrued
    disc = C.flat(0.01)
    assert t.upfront_from_spread(0.012, disc) != w.upfront_from_spread(0.012, disc)
