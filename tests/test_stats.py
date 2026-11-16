import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from barca import stats

PEP, LE = "Guardiola", "Luis Enrique"


def sim(shift, n=60, seed=1, k=3):
    rng = np.random.default_rng(seed)
    era = np.array([PEP] * n + [LE] * n)
    strata = np.tile(np.arange(k).repeat(n // k), 2).astype(str)
    d = rng.normal(0, 1, 2 * n) + shift * (era == LE)
    return d, era, strata


def test_estimate_hand():
    d = np.array([1., 3, 5, 9, 2, 4])
    era = np.array([PEP, PEP, LE, LE, PEP, LE])
    st = np.array(["a", "a", "a", "a", "b", "b"])
    # a: nP=2,nL=2 w=1, diff=7-2=5 ; b: nP=1,nL=1 w=.5, diff=4-2=2
    assert stats.stratified_estimate(d, era, st) == pytest.approx((5 + 0.5 * 2) / 1.5)


def test_perm_null_not_small():
    d, era, st = sim(0.0)
    _, p = stats.stratified_permutation_test(d, era, st, n_perm=2000)
    assert p > 0.05


def test_perm_large_shift():
    d, era, st = sim(5.0)
    _, p = stats.stratified_permutation_test(d, era, st, n_perm=10000)
    assert p == pytest.approx(1 / 10001)


def test_bootstrap_covers_truth():
    d, era, st = sim(1.0, n=90)
    lo, hi = stats.stratified_bootstrap_ci(d, era, st, n_boot=2000)
    assert lo < 1.0 < hi and hi - lo < 1.5


def test_bh_matches_statsmodels():
    p = [0.001, 0.02, 0.04, 0.3, 0.6]
    assert np.allclose(stats.bh(p), multipletests(p, method="fdr_bh")[1])


def test_season_condition():
    era = np.array([PEP] * 4 + [LE] * 4)
    season = np.array(["s1", "s1", "s2", "s2", "s3", "s3", "s4", "s4"])
    d = np.array([0, 0, 1, 1, 4, 4, 5, 5.])
    ok, sm, mid = stats.season_condition(d, era, season, +1)
    assert ok and mid == pytest.approx(2.5)
    assert not stats.season_condition(d, era, season, -1)[0]
    d2 = np.array([0, 0, 3, 3, 4, 4, 5, 5.])  # Pep season 2 above midpoint 2.5
    assert not stats.season_condition(d2, era, season, +1)[0]


def test_decide():
    T, F = True, False
    s = dict(M1=T, M2=T, M3=T, M4=F, M5=T)
    assert stats.decide(s, dict(s))["H1"] and stats.decide(s, dict(s))["H2"]
    assert not stats.decide(dict(s, M3=F), dict(s))["H1"]
    assert not stats.decide(s, dict(s, M2=F))["H1"]
    assert not stats.decide(dict(M1=T, M2=T, M3=T, M4=F, M5=F), dict(s))["H2"]
    allT = dict.fromkeys(s, True)
    assert stats.decide(dict(M1=T, M2=F, M3=T, M4=T, M5=T), allT)["H2"]
    assert not stats.decide(allT, dict(allT, M1=F, M2=F))["H2"]
