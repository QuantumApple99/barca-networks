"""Stage 1: raw JSON -> data/processed/*.parquet. Run: python scripts/build_processed.py"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from barca import context, io, networks, parse  # noqa: E402


def main():
    t0 = time.time()
    ids = io.barca_match_ids()
    with Pool() as pool:
        results = pool.map(parse.parse_match, ids, chunksize=4)
    d = parse.assemble(results)
    d["matches"] = parse.parse_matches()
    assert set(d["matches"].match_id) == set(ids), "match metadata / id list mismatch"
    d["matches"] = context.add_opponent_strength(d["matches"])
    io.PROCESSED.mkdir(parents=True, exist_ok=True)

    P, XI, AP, T = d["passes"], d["starting_xi"], d["appearances"], d["touches"]
    ew, ef = [], []
    pg = {k: v for k, v in P[P.completed].groupby(["match_id", "team_id"])}
    for (mid, tid), xi in XI.groupby(["match_id", "team_id"]):
        pp = pg.get((mid, tid), P.iloc[0:0])
        tt = T[(T.match_id == mid) & (T.team_id == tid)]
        ap = AP[(AP.match_id == mid) & (AP.team_id == tid)]
        for w in ("own", "either"):
            ew.append(networks.edges_to_frame(networks.build_window_network(pp, xi, mid, tid, w, tt), scope=w))
        ef.append(networks.edges_to_frame(networks.build_full_network(pp, ap, mid, tid, tt), scope="full"))
    d["edges_window"] = pd.concat(ew, ignore_index=True)
    d["edges_full"] = pd.concat(ef, ignore_index=True)

    for name in ("matches", "passes", "starting_xi", "appearances", "touches", "windows", "edges_window", "edges_full"):
        d[name].to_parquet(io.PROCESSED / f"{name}.parquet", index=False)
        print(f"{name:13s} {len(d[name]):>8,d} rows")
    print(f"periods seen: {sorted({p for r in results for p in r['periods']})}; "
          f"duplicate event ids: {sum(r['n_dup_ids'] for r in results)}")
    print(f"stage 1 done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
