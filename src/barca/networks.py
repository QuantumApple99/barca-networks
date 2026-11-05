"""Per-match, per-team weighted directed pass networks."""
import networkx as nx
import pandas as pd

from .positions import group_of, is_forward_line

WINDOW_COL = {"own": "in_window_own", "either": "in_window_either"}


def _attach_nodes(G, players, passes_used, touches):
    """players: DataFrame[player_id, player, position]. Adds node attributes."""
    po = passes_used.groupby("player_id")[["x", "y"]].mean()
    tm = touches.groupby("player_id")[["x", "y"]].mean() if touches is not None and len(touches) else None
    for r in players.itertuples():
        pid = r.player_id
        G.add_node(pid)
        nd = G.nodes[pid]
        nd.update(name=r.player, position=r.position, group=group_of(r.position),
                  forward_line=is_forward_line(r.position),
                  pass_x=po.at[pid, "x"] if pid in po.index else float("nan"),
                  pass_y=po.at[pid, "y"] if pid in po.index else float("nan"),
                  touch_x=tm.at[pid, "x"] if tm is not None and pid in tm.index else float("nan"),
                  touch_y=tm.at[pid, "y"] if tm is not None and pid in tm.index else float("nan"))


def _add_edges(G, passes):
    ids = set(G.nodes)
    p = passes[passes.player_id.isin(ids) & passes.recipient_id.isin(ids)
               & (passes.player_id != passes.recipient_id)]
    for (a, b), w in p.groupby(["player_id", "recipient_id"]).size().items():
        G.add_edge(int(a), int(b), weight=int(w))


def build_window_network(passes, starting_xi, match_id, team_id, window="own", touches=None):
    """Starting-XI network of completed passes before the window boundary.

    window: 'own' (team's first sub/red) or 'either' (either team's). Nodes = all 11 starters.
    passes / starting_xi / touches may be the full tables or pre-filtered frames."""
    xi = starting_xi[(starting_xi.match_id == match_id) & (starting_xi.team_id == team_id)]
    col = WINDOW_COL[window]
    p = passes[(passes.match_id == match_id) & (passes.team_id == team_id) & passes.completed & passes[col]]
    t = None
    if touches is not None:
        t = touches[(touches.match_id == match_id) & (touches.team_id == team_id) & touches[col]
                    & touches.player_id.isin(xi.player_id)]
    G = nx.DiGraph(match_id=match_id, team_id=team_id, window=window)
    _attach_nodes(G, xi, p[p.player_id.isin(xi.player_id)], t)
    _add_edges(G, p)
    return G


def build_full_network(passes, appearances, match_id, team_id, touches=None):
    """Whole-match network: every player who appeared, all completed passes."""
    ap = appearances[(appearances.match_id == match_id) & (appearances.team_id == team_id)]
    p = passes[(passes.match_id == match_id) & (passes.team_id == team_id) & passes.completed]
    t = None
    if touches is not None:
        t = touches[(touches.match_id == match_id) & (touches.team_id == team_id)]
    G = nx.DiGraph(match_id=match_id, team_id=team_id, window="full")
    _attach_nodes(G, ap[["player_id", "player", "position"]], p, t)
    _add_edges(G, p)
    return G


def edges_to_frame(G, scope=None):
    """Edge list: match_id, team_id, scope, passer_id, recipient_id, weight."""
    rows = [dict(match_id=G.graph["match_id"], team_id=G.graph["team_id"],
                 scope=scope or G.graph["window"], passer_id=a, recipient_id=b, weight=d["weight"])
            for a, b, d in G.edges(data=True)]
    return pd.DataFrame(rows, columns=["match_id", "team_id", "scope", "passer_id", "recipient_id", "weight"])
