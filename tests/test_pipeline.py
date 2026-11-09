import networkx as nx
import pytest

from barca import networks
from barca.metrics import gini

BARCA = 217


def test_match_69276_pass_counts(tables):
    p = tables["passes"]
    p = p[p.match_id == 69276]
    assert len(p) == 1196
    assert p.has_recipient.sum() == 1138
    assert p.outcome.isna().sum() == 1022
    assert p.completed.sum() == 1022


def test_every_network_has_11_nodes(tables):
    p, xi, t = tables["passes"], tables["starting_xi"], tables["touches"]
    for (mid, tid), _ in xi.groupby(["match_id", "team_id"]):
        for w in ("own", "either"):
            assert networks.build_window_network(p, xi, mid, tid, w, t).number_of_nodes() == 11


def test_window_boundary_69299(tables):
    w = tables["windows"]
    r = w[(w.match_id == 69299) & (w.team_id == BARCA)].iloc[0]
    assert r.end_reason == "substitution"
    assert int(r.end_minute_own) == 75


def pr_gini(tables, mid):
    G = networks.build_window_network(tables["passes"], tables["starting_xi"], mid, BARCA, "own", tables["touches"])
    return gini(list(nx.pagerank(G, weight="weight").values()))


@pytest.mark.parametrize("mid,expected", [(69299, 0.230), (267183, 0.170)])
def test_pagerank_gini_reproduces_figure(tables, mid, expected):
    assert round(pr_gini(tables, mid), 3) == expected


def test_267183_first_sub_75(tables):
    w = tables["windows"]
    assert int(w[(w.match_id == 267183) & (w.team_id == BARCA)].iloc[0].end_minute_own) == 75


def test_gini_toy():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 1]) == pytest.approx(0.75)  # (n-1)/n
    assert gini([5]) == 0.0
    assert gini([0, 0]) == 0.0


def test_gini_known_value():
    # [1,2,3]: sum_i (2i-n-1) x_i / (n sum) = (-2*1+0+2*3)/(3*6) = 4/18
    assert gini([3, 1, 2]) == pytest.approx(4 / 18)


def test_opponent_strength_joined():
    import pandas as pd
    from barca import io
    m = pd.read_parquet(io.PROCESSED / "matches.parquet")
    assert m.opp_position.between(1, 20).all() and m.opp_position.ne(None).all()
    # Barcelona never plays itself; Real Madrid finished 1st or 2nd in all 7 seasons
    rm = m[m.opponent == "Real Madrid"]
    assert len(rm) > 0 and rm.opp_position.isin([1, 2]).all()
    assert m.stratum.nunique() == 6
