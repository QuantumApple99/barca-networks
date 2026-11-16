"""Metric helpers (week 2 adds the rest)."""
import numpy as np


def gini(x):
    """Gini coefficient of non-negative values (0 = equal, ->1 = concentrated)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


# ---------------------------------------------------------------- Week 2: M1-M5
import itertools  # noqa: E402

import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402

from . import io, networks  # noqa: E402

GOAL = (120.0, 40.0)
X_ELIGIBLE = 48.0
CENTRAL_X = (30.0, 90.0)
CENTRAL_Y = (18.0, 62.0)
MIN_EDGE = 3


def m1(G):
    """Gini of PageRank (alpha 0.85, passes as weights) over all nodes, isolated included."""
    return gini(list(nx.pagerank(G, alpha=0.85, weight="weight").values()))


def m2(G):
    """Freeman betweenness centralization: sum(max c - c_v)/(n-1), normalized directed
    betweenness with distance = 1/weight (D30)."""
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    for a, b, d in G.edges(data=True):
        H.add_edge(a, b, dist=1.0 / d["weight"])
    c = np.array(list(nx.betweenness_centrality(H, weight="dist", normalized=True).values()))
    n = len(c)
    return float((c.max() - c).sum() / (n - 1))


def triangles(G, min_edge=MIN_EDGE):
    """Triangles of the undirected graph (combined weight >= min_edge): list of (nodes, weakest edge)."""
    U = {}
    for a, b, d in G.edges(data=True):
        k = frozenset((a, b))
        U[k] = U.get(k, 0) + d["weight"]
    U = {k: w for k, w in U.items() if w >= min_edge}
    adj = nx.Graph()
    adj.add_edges_from(tuple(k) for k in U)
    out = []
    for a, b, c in itertools.combinations(sorted(adj.nodes), 3):
        if adj.has_edge(a, b) and adj.has_edge(b, c) and adj.has_edge(a, c):
            w = min(U[frozenset((a, b))], U[frozenset((b, c))], U[frozenset((a, c))])
            out.append(((a, b, c), w))
    return out


def _central(nd):
    return (CENTRAL_X[0] <= nd["touch_x"] <= CENTRAL_X[1]) and (CENTRAL_Y[0] <= nd["touch_y"] <= CENTRAL_Y[1])


def m3(G, min_edge=MIN_EDGE):
    """Central triangle weight / total triangle weight; NaN if no triangles (D29)."""
    tri = triangles(G, min_edge)
    total = sum(w for _, w in tri)
    if not tri or total == 0:
        return float("nan")
    cen = sum(w for nodes, w in tri if all(_central(G.nodes[v]) for v in nodes))
    return cen / total


def m4(G, passes):
    """Completed in-window passes received by forward-line starters / all completed in-window passes.
    passes: completed in-window passes of the team (already filtered). Only starters' passes count."""
    ids = set(G.nodes)
    p = passes[passes.player_id.isin(ids)]
    if len(p) == 0:
        return float("nan")
    fwd = {v for v, d in G.nodes(data=True) if d["forward_line"]}
    return float(p.recipient_id.isin(fwd).sum() / len(p))


def m5(G, passes):
    """Among completed in-window passes starting at x >= 48, share ending >= 25% closer to (120, 40)."""
    ids = set(G.nodes)
    p = passes[passes.player_id.isin(ids) & (passes.x >= X_ELIGIBLE)]
    if len(p) == 0:
        return float("nan")
    ds = np.hypot(GOAL[0] - p.x.to_numpy(), GOAL[1] - p.y.to_numpy())
    de = np.hypot(GOAL[0] - p.end_x.to_numpy(), GOAL[1] - p.end_y.to_numpy())
    return float((de <= 0.75 * ds).mean())


def compute_team_match_metrics(window="own", barca_id=None):
    """One row per (match, team): M1..M5 plus diagnostics."""
    barca_id = barca_id or io.BARCA_ID
    L = lambda n: pd.read_parquet(io.PROCESSED / f"{n}.parquet")  # noqa: E731
    P, XI, T, W = L("passes"), L("starting_xi"), L("touches"), L("windows")
    col = networks.WINDOW_COL[window]
    P = P[P.completed & P[col]]
    T = T[T[col]]
    pg = {k: v for k, v in P.groupby(["match_id", "team_id"])}
    tg = {k: v for k, v in T.groupby(["match_id", "team_id"])}
    wm = W.set_index(["match_id", "team_id"])["window_minutes_" + window]
    rows = []
    for (mid, tid), xi in XI.groupby(["match_id", "team_id"]):
        p = pg.get((mid, tid), P.iloc[0:0])
        t = tg.get((mid, tid), T.iloc[0:0])
        t = t[t.player_id.isin(xi.player_id)]
        G = networks.build_window_network(p.assign(**{col: True}), xi, mid, tid, window, t.assign(**{col: True}))
        ps = p[p.player_id.isin(xi.player_id)]
        tri = triangles(G)
        rows.append(dict(match_id=mid, team_id=tid, is_barca=tid == barca_id,
                         M1=m1(G), M2=m2(G), M3=m3(G), M4=m4(G, p), M5=m5(G, p),
                         window_minutes=float(wm.loc[(mid, tid)]), n_completed=len(ps),
                         n_forward_line_starters=sum(d["forward_line"] for _, d in G.nodes(data=True)),
                         n_triangles=len(tri)))
    return pd.DataFrame(rows)


def build_diffs(tm):
    """One row per match: dM = Barca - opponent (NaN propagates)."""
    M = pd.read_parquet(io.PROCESSED / "matches.parquet")
    b = tm[tm.is_barca].set_index("match_id")
    o = tm[~tm.is_barca].set_index("match_id")
    out = M[["match_id", "season", "era", "stratum", "barca_home", "opp_position"]].copy()
    for k in range(1, 6):
        out[f"dM{k}"] = out.match_id.map(b[f"M{k}"] - o[f"M{k}"])
    return out


def run_stage2():
    for window, suffix in (("own", ""), ("either", "_either")):
        tm = compute_team_match_metrics(window)
        tm.to_parquet(io.PROCESSED / f"metrics_window{suffix}.parquet", index=False)
        build_diffs(tm).to_parquet(io.PROCESSED / f"o1_diffs{suffix}.parquet", index=False)
