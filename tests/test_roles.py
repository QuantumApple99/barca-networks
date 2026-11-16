import numpy as np
import pandas as pd
import pytest

from barca import roles


@pytest.fixture(scope="module")
def data():
    ms = roles.match_stats()
    units = roles.unit_table(ms)
    U = roles.build_player_season_features(ms=ms, units=units)
    return ms, units, U


def test_feature_table_shape_and_sanity(data):
    ms, units, U = data
    assert len(U) == 101
    assert not (units.position_group == "GK").any()
    assert U[roles.FEATURES].notna().all().all()
    assert (U.minutes >= 900).all()
    assert np.allclose(U[roles.MADE].sum(axis=1), 1)
    assert np.allclose(U[roles.RECV].sum(axis=1), 1)
    assert len(roles.FEATURES) == 25


def test_pass_totals_conserved(data):
    ms, _, _ = data
    p = pd.read_parquet(roles.io.PROCESSED / "passes.parquet")
    n = ((p.team_id == roles.io.BARCA_ID) & p.completed).sum()
    assert ms.n_made.sum() == n and ms.n_recv.sum() == n


def test_progressive_and_forward_toy():
    p = pd.DataFrame(dict(x=[60.0, 60.0, 30.0, 100.0], y=[40.0] * 4, end_x=[100.0, 61.0, 90.0, 90.0], end_y=[40.0] * 4))
    f = roles.pass_features(p)
    assert f.elig.tolist() == [1, 1, 0, 1]
    # 60->100: 60 -> 20 (<= 45) prog; 60->61: 60 -> 59 no; 30 ineligible; 100->90: 20 -> 30 no
    assert f.prog.tolist() == [1, 0, 0, 0]
    assert f.fwd.tolist() == [1, 1, 1, 0]
    assert np.allclose(f.length, [40, 1, 60, 10])


def test_cell_index_cuts():
    assert roles.cell_index([0, 40, 80, 120], [0, 0, 0, 0]).tolist() == [0, 1, 2, 2]
    assert roles.cell_index([0, 0, 0], [26.6, 26.7, 80]).tolist() == [0, 3, 6]


def test_no_eligible_passes_gives_zero_progressive():
    ms = pd.DataFrame(dict(season=["s"], player_id=[1], match_id=[1], minutes_played=[90.0], pagerank_n=[1.0],
                           betweenness=[0.1], n_made=[2], n_recv=[2], n_elig=[0], n_prog=[0], len_sum=[10.0], n_fwd=[1],
                           **{c: [0] for c in roles.MADE + roles.RECV}))
    ms.loc[0, ["made_00", "recv_00"]] = 2
    F = roles.features_from_weights(ms, [1])
    assert F.progressive_share.iloc[0] == 0


def test_identical_resample_gives_ari_one(data):
    ms, units, U = data
    ms_u = ms.merge(units.reset_index()[roles.PS_KEY], on=roles.PS_KEY).reset_index(drop=True)
    F = roles.features_from_weights(ms_u, np.ones(len(ms_u))).loc[units.index]
    assert np.allclose(F[roles.FEATURES].to_numpy(), U[roles.FEATURES].to_numpy())
    base = roles.fit_roles(U[roles.FEATURES].to_numpy())
    _, s = roles.stability(ms, units, roles.FEATURES, base["k"], base["labels"], n_boot=1, processes=1)
    # a real resample is not identical; check the identical-weights path directly
    roles._G.update(ms=ms_u, cols=roles.FEATURES, k=base["k"], base=base["labels"], order=units.index, seed=671, n_init=50)
    assert roles._boot_worker(np.ones(len(ms_u))) == 1.0


def test_resample_preserves_match_counts(data):
    ms, units, _ = data
    ms_u = ms.merge(units.reset_index()[roles.PS_KEY], on=roles.PS_KEY).reset_index(drop=True)
    w = roles.resample_matches(ms_u, units, np.random.default_rng(1))
    tot = pd.Series(w).groupby([ms_u.season, ms_u.player_id]).sum()
    cnt = ms_u.groupby(roles.PS_KEY).size()
    assert (tot == cnt).all()


def test_tv_toy():
    assert roles.tv_distance([1, 0], [0, 1]) == 1.0
    assert roles.tv_distance([.5, .5], [.5, .5]) == 0.0
    assert roles.tv_distance([.7, .3], [.5, .5]) == pytest.approx(0.2)


def test_era_mix_35_labelings(data):
    _, _, U = data
    labels = roles.order_by_size(roles.fit_roles(U[roles.FEATURES].to_numpy())["labels"])
    r = roles.era_role_mix_test(labels, U.minutes, U.season, U.era)
    assert r["n_labelings"] == 35 and len(r["tvs"]) == 35
    assert r["p"] >= 1 / 35
    assert (r["tvs"] >= r["observed_tv"] - 1e-12).sum() >= 1


def test_position_validity_perfect_and_split():
    pos = ["A"] * 6 + ["B"] * 6
    v = roles.position_validity([1] * 6 + [2] * 6, pos)
    assert v["ari"] == 1.0 and not v["groups"].split.any()
    v = roles.position_validity([1, 1, 1, 1, 2, 2] + [2] * 6, pos)    # A: 2/6 outside = 33% -> split
    assert v["groups"].set_index("position_group").split["A"]
