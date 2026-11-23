"""SA-CCR against the five worked examples in Annex 4a of the Basel text (BCBS 279, 2014)."""
import pytest

from qlab import saccr as S


def test_supervisory_building_blocks():
    assert S.supervisory_duration(0, 10) == pytest.approx(7.869, abs=1e-3)
    assert S.supervisory_duration(1, 11) == pytest.approx(7.486, abs=1e-3)
    ex1, _, _ = S.basel_examples()
    assert S.supervisory_delta(ex1[2]) == pytest.approx(-0.27, abs=0.005)
    assert S.maturity_factor(ex1[0], margined=True, mpor_days=14) == pytest.approx(1.5 * (14 / 250) ** 0.5)


@pytest.mark.parametrize("i, addon, mult, ead", [(0, 347, 1.0, 569), (1, 282, 0.965, 381), (2, 3841, 1.0, 5406)])
def test_basel_examples_1_to_3(i, addon, mult, ead):
    r = S.ead(S.basel_examples()[i])
    assert r["addon"]["total"] == pytest.approx(addon, abs=0.5)
    assert r["multiplier"] == pytest.approx(mult, abs=5e-4)
    assert r["EAD"] == pytest.approx(ead, abs=0.6)


def test_basel_example_4_combined_and_5_margined():
    ex1, ex2, ex3 = S.basel_examples()
    assert S.ead(ex1 + ex2)["EAD"] == pytest.approx(936, abs=1)
    r = S.ead(ex1 + ex3, S.Collateral(C=200, margined=True, TH=0, MTA=5, NICA=150, mpor_days=14))
    assert r["RC"] == 0
    assert r["addon"]["IR"] == pytest.approx(123, abs=0.5) and r["addon"]["CO"] == pytest.approx(1278, abs=1)
    assert r["multiplier"] == pytest.approx(0.958, abs=5e-4)
    assert r["EAD"] == pytest.approx(1879, abs=1)
