"""Week-1 validation summary. Run: python notebooks/01_validation.py (needs data/processed built)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from barca import io  # noqa: E402

pd.set_option("display.width", 160)
L = lambda n: pd.read_parquet(io.PROCESSED / f"{n}.parquet")  # noqa: E731
M, P, XI, AP, W = L("matches"), L("passes"), L("starting_xi"), L("appearances"), L("windows")
B = io.BARCA_ID
print("=== 1. matches per season (expect 31/35/33/37/38/38/34) ===")
print(M.groupby(["era", "season"]).size().to_string(), f"\ntotal {len(M)}; by era: {M.groupby('era').size().to_dict()}")

print("\n=== 2. passes per match (both teams / Barca / opponent) ===")
pm = P.groupby("match_id").size()
pb = P[P.team_id == B].groupby("match_id").size()
print(pm.describe().round(0).to_string()); print("Barca:", pb.describe().round(0)[["mean", "min", "max"]].to_dict())
print("passes per match by era:", P.merge(M[["match_id", "era"]]).groupby(["era", "match_id"]).size().groupby("era").mean().round(0).to_dict())

print("\n=== 3. excluded (non-completed) passes ===")
P["team"] = np.where(P.team_id == B, "Barca", "Opp")
ex = P.groupby("team").agg(total=("completed", "size"), completed=("completed", "sum"))
ex["excluded"] = ex.total - ex.completed; ex["excl_%"] = (100 * ex.excluded / ex.total).round(2)
print(ex.to_string())
print("excluded by reason (all teams):")
reason = np.where(P.completed, "completed", np.where(~P.has_recipient & P.outcome.isna(), "no outcome, no recipient",
                  np.where(P.outcome.notna(), "outcome=" + P.outcome.fillna(""), "other")))
print(pd.Series(reason).value_counts().to_string())
print("no recipient but no outcome key:", int((~P.has_recipient & P.outcome.isna()).sum()),
      "| recipient present but outcome key present:", int((P.has_recipient & P.outcome.notna()).sum()))
print("pass types (completed):", P[P.completed].pass_type.value_counts().to_dict())
iw = P[P.in_window_own]
print(f"in own-window passes: {len(iw):,}; completed {iw.completed.mean():.3f}; set-piece share of completed in window "
      f"{(iw[iw.completed].pass_type != 'Open Play').mean():.3f}")

print("\n=== 4. windows ===")
W = W.merge(M[["match_id", "era", "season"]]); W["team"] = np.where(W.team_id == B, "Barca", "Opp")
for c in ("window_minutes_own", "window_minutes_either"):
    print(c, W.groupby("team")[c].describe().round(1)[["mean", "25%", "50%", "75%", "min"]].to_dict("index"))
print("window_minutes_own by era/team (mean):", W.groupby(["era", "team"]).window_minutes_own.mean().round(1).to_dict())
print("end_reason own:", W.groupby(["team", "end_reason"]).size().to_dict())
print("end_reason either (per match):", W[W.team == "Barca"].end_reason_either.value_counts().to_dict())
print("who triggers either-window:", {"Barca": int((W[W.team == "Barca"].end_team_either == B).sum()),
      "Opp": int((W[W.team == "Barca"].end_team_either.notna() & (W[W.team == "Barca"].end_team_either != B)).sum())})
for thr in (30, 45):
    print(f"windows < {thr} min: own {W[W.window_minutes_own < thr].groupby('team').size().to_dict()}; "
          f"either (matches) {int((W[W.team == 'Barca'].window_minutes_either < thr).sum())}")
print("matches where EITHER team's window < 30 min:", int(W.groupby("match_id").window_minutes_own.min().lt(30).sum()))
print("shortest own windows:\n", W.nsmallest(8, "window_minutes_own")[["match_id", "team", "season", "window_minutes_own", "end_reason"]].to_string(index=False))
print("match length (min):", W.match_minutes.describe().round(1)[["min", "mean", "max"]].to_dict())

print("\n=== 5. minutes-played spot check vs substitution events ===")
rng = np.random.default_rng(0)
rows = []
for mid in rng.choice(M.match_id.values, 5, replace=False):
    ev = io.load_events(int(mid))
    s = [e for e in ev if e["type"]["name"] == "Substitution" and e["team"]["id"] == B]
    if not s: continue
    e = s[0]; rep = e["substitution"]["replacement"]
    a_on = AP[(AP.match_id == mid) & (AP.player_id == rep["id"])].iloc[0]
    a_off = AP[(AP.match_id == mid) & (AP.player_id == e["player"]["id"])].iloc[0]
    tot = W[(W.match_id == mid)].match_minutes.iloc[0]
    clock = e["minute"] + e["second"] / 60
    rows.append(dict(match=mid, off=a_off.player.split()[-1], on=a_on.player.split()[-1], sub_event_clock=round(clock, 2),
                     lineup_on_from=a_on.first_from, off_min_played=round(a_off.minutes_played, 2),
                     on_min_played=round(a_on.minutes_played, 2), sum_on_off=round(a_on.minutes_played + a_off.minutes_played, 2),
                     match_min=round(tot, 2)))
print(pd.DataFrame(rows).to_string(index=False))
print("(off + on should be ~ match_min when on replaces off and nobody else changes; off ~ clock minus P1 stoppage)")
tm = AP.groupby(["match_id", "team_id"]).minutes_played.sum().reset_index().merge(W[["match_id", "team_id", "match_minutes"]])
tm["ratio"] = tm.minutes_played / (11 * tm.match_minutes)
print("team minutes / (11 x match length): min %.3f  median %.3f  max %.3f; <0.99 in %d team-matches (red cards expected)"
      % (tm.ratio.min(), tm.ratio.median(), tm.ratio.max(), (tm.ratio < 0.99).sum()))

print("team-matches with minutes > 1.01 x expected (source lineup overlap):",
      tm[tm.ratio > 1.01][["match_id", "team_id", "ratio"]].round(3).to_dict("records"))

print("\n=== 6. data-quality checks ===")
print("starting XI rows per team-match != 11:", int((XI.groupby(["match_id", "team_id"]).size() != 11).sum()),
      "| team-matches with XI:", XI.groupby(["match_id", "team_id"]).ngroups, "(expect 492)")
print("lineup 'started' != Starting XI event:", int((AP.started != AP.in_xi_event).sum()))
print("appearances with no position group:", int(AP.position_group.isna().sum()), "| minutes <=0:", int((AP.minutes_played <= 0).sum()),
      "| minutes > match+0.5:", int((AP.merge(W[["match_id", "team_id", "match_minutes"]]).eval("minutes_played > match_minutes + 0.5")).sum()))
xs = XI.assign(k=1)[["match_id", "team_id", "player_id", "k"]]
iwp = P[P.in_window_own].merge(xs, how="left", on=["match_id", "team_id", "player_id"])
print("in-own-window passes by non-starters:", int(iwp.k.isna().sum()),
      "| completed in-window passes whose recipient is not a starter:",
      int(P[P.in_window_own & P.completed].merge(xs.rename(columns={"player_id": "recipient_id"}), how="left",
                                                  on=["match_id", "team_id", "recipient_id"]).k.isna().sum()))
print("passes with null player_id:", int(P.player_id.isna().sum()), "| passes with NaN start coords:", int(P.x.isna().sum()),
      "| self-passes (completed):", int((P.completed & (P.player_id == P.recipient_id)).sum()))
print("duplicate (match,index) pass rows:", int(P.duplicated(["match_id", "index"]).sum()))
print("StatsBomb 2015/16: Barca matches =", int((M.season_id == 27).sum()))
