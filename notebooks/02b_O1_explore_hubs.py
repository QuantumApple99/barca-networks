"""EXPLORATORY (not part of the pre-registered O1 test): who were Barça's passing hubs in each era?

For every Barça own-window network: each starter's PageRank and betweenness, and whether they were the
match's top player on each. Writes report/tables/o1_hubs.csv and data/processed/o1_explore_player_centrality.parquet.
Run: python notebooks/02b_O1_explore_hubs.py
"""
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from barca import io  # noqa: E402

BARCA = 217


def main(min_starts=15, top=6):
    E = pd.read_parquet(io.PROCESSED / "edges_window.parquet")
    E = E[(E.scope == "own") & (E.team_id == BARCA)]
    XI = pd.read_parquet(io.PROCESSED / "starting_xi.parquet")
    XI = XI[XI.team_id == BARCA]
    M = pd.read_parquet(io.PROCESSED / "matches.parquet")[["match_id", "era"]]
    rows = []
    for mid, xi in XI.groupby("match_id"):
        G = nx.DiGraph()
        G.add_nodes_from(xi.player_id)
        for r in E[E.match_id == mid].itertuples():
            G.add_edge(r.passer_id, r.recipient_id, weight=r.weight, dist=1 / r.weight)
        pr = nx.pagerank(G, weight="weight")
        bt = nx.betweenness_centrality(G, weight="dist", normalized=True)
        top_pr, top_bt = max(pr, key=pr.get), max(bt, key=bt.get)
        for p in xi.itertuples():
            rows.append((mid, p.player_id, p.player, pr[p.player_id], bt[p.player_id],
                         p.player_id == top_pr, p.player_id == top_bt))
    R = pd.DataFrame(rows, columns=["match_id", "player_id", "player", "pagerank", "betweenness",
                                    "top_pagerank", "top_betweenness"]).merge(M)
    R.to_parquet(io.PROCESSED / "o1_explore_player_centrality.parquet", index=False)
    n = R.groupby("era").match_id.nunique()
    g = R.groupby(["era", "player"]).agg(starts=("pagerank", "size"), pagerank=("pagerank", "mean"),
                                         betweenness=("betweenness", "mean"),
                                         top_pagerank=("top_pagerank", "sum"), top_betweenness=("top_betweenness", "sum")).reset_index()
    g = g[g.starts >= min_starts]
    g["pct_matches_top_pagerank"] = 100 * g.top_pagerank / g.era.map(n)
    g["pct_matches_top_betweenness"] = 100 * g.top_betweenness / g.era.map(n)
    out = g.sort_values(["era", "pagerank"], ascending=[True, False]).groupby("era").head(top)
    (ROOT / "report" / "tables").mkdir(parents=True, exist_ok=True)
    out.round(3).to_csv(ROOT / "report" / "tables" / "o1_hubs.csv", index=False)
    print(out.round(3).to_string(index=False))
    print("mean per-match max PageRank:", R.groupby(["era", "match_id"]).pagerank.max().groupby("era").mean().round(3).to_dict())


if __name__ == "__main__":
    main()
