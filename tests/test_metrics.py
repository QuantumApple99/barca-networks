import networkx as nx
import numpy as np
import pandas as pd
import pytest

from barca import metrics, networks

BARCA = 217
E = {('A','B'):4,('B','A'):3,('B','C'):3,('C','B'):2,('A','C'):2,('C','A'):2,('C','D'):2,
     ('D','C'):1,('B','D'):2,('D','E'):4,('E','D'):2,('C','E'):2,('E','C'):1}
TOUCH_X = dict(A=50, B=60, C=70, D=100, E=85)


def toy():
    G = nx.DiGraph()
    for v in "ABCDE":
        G.add_node(v, touch_x=TOUCH_X[v], touch_y=40.0, forward_line=v in "DE")
    for (a, b), w in E.items():
        G.add_edge(a, b, weight=w)
    return G


def toy_passes():
    rows = [dict(player_id=a, recipient_id=b) for (a, b), w in E.items() for _ in range(w)]
    return pd.DataFrame(rows)


def test_m2_star_is_one():
    G = nx.DiGraph()
    G.add_nodes_from(range(5))
    for v in range(1, 5):
        G.add_edge(0, v, weight=1); G.add_edge(v, 0, weight=1)
    assert metrics.m2(G) == pytest.approx(1.0)


def test_m2_complete_is_zero():
    G = nx.complete_graph(5, create_using=nx.DiGraph)
    nx.set_edge_attributes(G, 3, "weight")
    assert metrics.m2(G) == pytest.approx(0.0)


def test_lesson06_toy():
    G = toy()
    assert metrics.m3(G) == pytest.approx(4 / 7)
    assert metrics.m4(G, toy_passes()) == pytest.approx(0.40)
    assert len(metrics.triangles(G)) == 2


def test_m3_nan_without_triangles():
    G = nx.DiGraph()
    G.add_nodes_from("ABC", touch_x=50, touch_y=40)
    G.add_edge("A", "B", weight=5); G.add_edge("B", "C", weight=5)
    assert np.isnan(metrics.m3(G))


def test_m5_toy():
    G = toy()
    p = pd.DataFrame(dict(player_id=["A"] * 4, x=[50, 50, 40, 100.0], y=[40] * 4,
                          end_x=[70, 52, 100, 115.0], end_y=[40] * 4))
    # eligible: first, second, last (x>=48). 50->70: 70 -> 50 (ok); 50->52: no; 100->115: 20->5 ok
    assert metrics.m5(G, p) == pytest.approx(2 / 3)


def test_m3_boundary_touch_not_central():
    G = toy()
    G.nodes["A"]["touch_x"] = 29.9
    assert metrics.m3(G) == 0.0


def test_real_match_69299(tables):
    p, xi, t = tables["passes"], tables["starting_xi"], tables["touches"]
    pp = p[(p.match_id == 69299) & (p.team_id == BARCA) & p.completed & p.in_window_own]
    G = networks.build_window_network(p, xi, 69299, BARCA, "own", t)
    assert round(metrics.m1(G), 3) == 0.230
    assert round(metrics.m2(G), 3) == 0.492
    assert round(metrics.m4(G, pp), 3) == 0.240
    assert round(metrics.m5(G, pp), 3) == 0.039
    assert len(pp) == 555
