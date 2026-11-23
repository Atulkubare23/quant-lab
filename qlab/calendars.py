"""Business-day calendars from rules (no data files): TARGET, New York, London, Tokyo, and joints.

  WEEKENDS  Saturdays and Sundays only (the old behaviour of the CDS engine)
  TARGET    euro settlement: 1 Jan, Good Friday, Easter Monday, 1 May, 25 and 26 Dec (from 2000;
            31 Dec 1998, 1999 and 2001 as well)
  USNY      New York banking days, Federal Reserve schedule: New Year, Martin Luther King Jr (3rd Mon
            Jan), Washington's Birthday (3rd Mon Feb), Memorial (last Mon May), Juneteenth (19 Jun,
            from 2022), Independence (4 Jul), Labor (1st Mon Sep), Columbus (2nd Mon Oct), Veterans
            (11 Nov), Thanksgiving (4th Thu Nov), Christmas. A Sunday holiday moves to Monday; a
            Saturday holiday is not moved (the Fed stays open the Friday before)
  LONDON    England & Wales bank holidays: New Year (weekend -> Monday), Good Friday, Easter Monday,
            early May (1st Mon), spring (last Mon May), summer (last Mon Aug), Christmas and Boxing
            Day (the first two weekdays from 25 Dec), plus the moved and one-off days since 1995
  TOKYO     Japanese bank holidays: 1-3 Jan and 31 Dec, the national holidays with the substitute
            rule (a Sunday holiday moves to the next non-holiday) and the citizens' holiday (a day
            between two holidays), equinoxes from the standard formula (valid 1980-2099), the
            2019 enthronement and 2020-21 Olympic moves

Rules are coded for 2000 onwards; earlier years are not guaranteed.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache


def easter(y: int) -> date:
    """Gregorian Easter Sunday (anonymous / Meeus algorithm)."""
    a, b, c = y % 19, y // 100, y % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(y, month, day)


def nth_weekday(y, m, weekday, n):
    """n-th given weekday (Mon = 0) of a month; n = -1 for the last."""
    if n > 0:
        d = date(y, m, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


MON, TUE, WED, THU, FRI, SAT, SUN = range(7)


# ---------------------------------------------------------------- holiday rules, one year at a time
def _none(y):
    return set()


def _target(y):
    h = {date(y, 1, 1), date(y, 12, 25)}
    if y >= 2000:
        e = easter(y)
        h |= {e - timedelta(days=2), e + timedelta(days=1), date(y, 5, 1), date(y, 12, 26)}
    if y in (1998, 1999, 2001):
        h.add(date(y, 12, 31))
    return h


def _usny(y):
    def sunday_to_monday(d):
        return d + timedelta(days=1) if d.weekday() == SUN else d
    h = {sunday_to_monday(date(y, 1, 1)), sunday_to_monday(date(y, 7, 4)), sunday_to_monday(date(y, 11, 11)),
         sunday_to_monday(date(y, 12, 25)),
         nth_weekday(y, 1, MON, 3), nth_weekday(y, 2, MON, 3), nth_weekday(y, 5, MON, -1),
         nth_weekday(y, 9, MON, 1), nth_weekday(y, 10, MON, 2), nth_weekday(y, 11, THU, 4)}
    if y >= 2022:
        h.add(sunday_to_monday(date(y, 6, 19)))
    return {d for d in h if d.weekday() < SAT}


LONDON_MOVED = {   # year: {regular day: replacement}
    1995: {"early may": date(1995, 5, 8)},
    2002: {"spring": date(2002, 6, 4)},
    2012: {"spring": date(2012, 6, 4)},
    2020: {"early may": date(2020, 5, 8)},
    2022: {"spring": date(2022, 6, 2)},
}
LONDON_EXTRA = {date(1999, 12, 31), date(2002, 6, 3), date(2011, 4, 29), date(2012, 6, 5),
                date(2022, 6, 3), date(2022, 9, 19), date(2023, 5, 8)}


def _london(y):
    e = easter(y)
    jan1 = date(y, 1, 1)
    while jan1.weekday() >= SAT:
        jan1 += timedelta(days=1)
    moved = LONDON_MOVED.get(y, {})
    h = {jan1, e - timedelta(days=2), e + timedelta(days=1),
         moved.get("early may", nth_weekday(y, 5, MON, 1)),
         moved.get("spring", nth_weekday(y, 5, MON, -1)),
         nth_weekday(y, 8, MON, -1)}
    d, xmas = date(y, 12, 25), []
    while len(xmas) < 2:                  # Christmas and Boxing Day, substituted onto weekdays
        if d.weekday() < SAT:
            xmas.append(d)
        d += timedelta(days=1)
    h |= set(xmas)
    h |= {x for x in LONDON_EXTRA if x.year == y}
    return h


def _equinoxes(y):
    k = y - 1980
    vernal = int(20.8431 + 0.242194 * k - k // 4)
    autumnal = int(23.2488 + 0.242194 * k - k // 4)
    return date(y, 3, vernal), date(y, 9, autumnal)


TOKYO_EXTRA = {date(2019, 5, 1), date(2019, 10, 22)}


def _japan_national(y):
    """National holidays before the substitute and citizens' rules."""
    v, a = _equinoxes(y)
    h = {date(y, 1, 1), date(y, 2, 11), v, a, date(y, 4, 29), date(y, 5, 3), date(y, 5, 5),
         date(y, 11, 3), date(y, 11, 23)}
    h.add(nth_weekday(y, 1, MON, 2) if y >= 2000 else date(y, 1, 15))          # Coming of Age
    if 1989 <= y <= 2018:
        h.add(date(y, 12, 23))                                                  # Emperor's Birthday
    elif y >= 2020:
        h.add(date(y, 2, 23))
    if y >= 2007:
        h.add(date(y, 5, 4))                                                    # Greenery Day
    marine = {2020: date(2020, 7, 23), 2021: date(2021, 7, 22)}
    if y in marine:
        h.add(marine[y])
    elif y >= 2003:
        h.add(nth_weekday(y, 7, MON, 3))
    elif y >= 1996:
        h.add(date(y, 7, 20))
    mountain = {2020: date(2020, 8, 10), 2021: date(2021, 8, 8)}
    if y >= 2016:
        h.add(mountain.get(y, date(y, 8, 11)))
    h.add(nth_weekday(y, 9, MON, 3) if y >= 2003 else date(y, 9, 15))         # Respect for the Aged
    sports = {2020: date(2020, 7, 24), 2021: date(2021, 7, 23)}
    if y in sports:
        h.add(sports[y])
    else:
        h.add(nth_weekday(y, 10, MON, 2) if y >= 2000 else date(y, 10, 10))
    h |= {x for x in TOKYO_EXTRA if x.year == y}
    return h


def _tokyo(y):
    nat = _japan_national(y)
    h = set(nat)
    for d in sorted(nat):                                    # substitute holidays
        if d.weekday() == SUN:
            s = d + timedelta(days=1)
            if y >= 2007:
                while s in h:
                    s += timedelta(days=1)
            h.add(s)
    for d in sorted(nat):                                    # citizens' holidays (sandwiched days)
        mid = d + timedelta(days=1)
        if mid not in h and mid + timedelta(days=1) in nat and mid.weekday() != SUN:
            h.add(mid)
    h |= {date(y, 1, 2), date(y, 1, 3), date(y, 12, 31)}
    return h


# ---------------------------------------------------------------- the calendar object
class Calendar:
    def __init__(self, name, *rules):
        self.name = name
        self._rules = rules

    @lru_cache(maxsize=None)
    def holidays(self, year: int) -> frozenset:
        """Weekday holidays of one year (weekends are always non-business days)."""
        out = set()
        for r in self._rules:
            out |= r(year)
        return frozenset(d for d in out if d.weekday() < SAT)

    def is_business_day(self, d: date) -> bool:
        return d.weekday() < SAT and d not in self.holidays(d.year)

    def adjust(self, d: date, convention: str = "following") -> date:
        if convention == "unadjusted":
            return d
        step = -1 if convention == "preceding" else 1
        x = d
        while not self.is_business_day(x):
            x += timedelta(days=step)
        if convention == "modified following" and x.month != d.month:
            return self.adjust(d, "preceding")
        if convention not in ("following", "preceding", "modified following"):
            raise ValueError(f"unknown convention {convention!r}")
        return x

    def add_business_days(self, d: date, n: int) -> date:
        step = 1 if n >= 0 else -1
        while n:
            d += timedelta(days=step)
            if self.is_business_day(d):
                n -= step
        return d

    def business_days_between(self, a: date, b: date) -> int:
        """Business days in (a, b]."""
        return sum(self.is_business_day(a + timedelta(days=i)) for i in range(1, (b - a).days + 1))

    def __repr__(self):
        return f"Calendar({self.name})"

    def __hash__(self):
        return hash((self.name, self._rules))

    def __eq__(self, other):
        return isinstance(other, Calendar) and (self.name, self._rules) == (other.name, other._rules)


def joint(*cals: Calendar) -> Calendar:
    """A day is a business day only if it is one in every calendar."""
    return Calendar("+".join(c.name for c in cals), *[r for c in cals for r in c._rules])


WEEKENDS = Calendar("WEEKENDS", _none)
TARGET = Calendar("TARGET", _target)
USNY = Calendar("USNY", _usny)
LONDON = Calendar("LONDON", _london)
TOKYO = Calendar("TOKYO", _tokyo)
BY_CURRENCY = {"EUR": TARGET, "USD": USNY, "GBP": LONDON, "JPY": TOKYO}


if __name__ == "__main__":
    for cal in (TARGET, USNY, LONDON, TOKYO):
        print(f"{cal.name:7s} 2026: " + ", ".join(d.strftime("%d %b") for d in sorted(cal.holidays(2026))))
