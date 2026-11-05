"""Raw StatsBomb JSON -> tidy DataFrames."""
import numpy as np
import pandas as pd

from . import io
from .positions import group_of
from .windows import find_boundaries, either_boundary

TOUCH_TYPES = {"Pass", "Ball Receipt*", "Carry", "Dribble", "Shot", "Ball Recovery", "Miscontrol",
               "Dispossessed", "Clearance", "Interception", "Block", "Duel"}
BIG = 10**9


def _secs(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _clock(s):
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def _lag(e):
    return e["timestamp"]


def parse_matches():
    """One row per Barca match (246)."""
    keep = set(io.barca_match_ids())
    rows = []
    for m in io.load_match_metadata():
        if m["match_id"] not in keep:
            continue
        h, a = m["home_team"], m["away_team"]
        home = h["home_team_id"] == io.BARCA_ID
        opp = a if home else h
        rows.append(dict(
            match_id=m["match_id"], season_id=m["season"]["season_id"], season=m["season"]["season_name"],
            era=io.SEASON_ERA[m["season"]["season_id"]], date=m["match_date"],
            home_team=h["home_team_name"], away_team=a["away_team_name"],
            home_score=m["home_score"], away_score=m["away_score"], barca_home=home,
            opponent=opp["away_team_name" if home else "home_team_name"],
            opponent_id=opp["away_team_id" if home else "home_team_id"]))
    df = pd.DataFrame(rows).sort_values(["date", "match_id"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _elapsed_fn(events):
    """Return (elapsed(period, seconds_in_period_timestamp), p1_len, total) from last event per period."""
    last = {}
    for e in events:
        last[e["period"]] = max(last.get(e["period"], 0.0), _secs(e["timestamp"]))
    p1 = last.get(1, 0.0)
    extra = sum(v for k, v in last.items() if k > 2)  # extra time unlikely in league games
    total = p1 + last.get(2, 0.0) + extra

    def elapsed(period, t):
        return t if period == 1 else p1 + t  # period 2 (periods 3+ do not occur in league data)
    return elapsed, p1, total, sorted(last)


def parse_match(match_id):
    """Parse one match into dict of record lists + diagnostics."""
    events = io.load_events(match_id)
    lineups = io.load_lineups(match_id)
    elapsed, p1len, total, periods = _elapsed_fn(events)
    bounds = find_boundaries(events)
    either = either_boundary(bounds)
    last_index = events[-1]["index"]
    team_ids = sorted({e["team"]["id"] for e in events if e["type"]["name"] == "Starting XI"})

    # starting XI
    xi_rows = []
    for e in events:
        if e["type"]["name"] != "Starting XI":
            continue
        for p in e["tactics"]["lineup"]:
            xi_rows.append(dict(match_id=match_id, team_id=e["team"]["id"], player_id=p["player"]["id"],
                                player=p["player"]["name"], position=p["position"]["name"],
                                position_id=p["position"]["id"], jersey=p["jersey_number"]))

    # windows
    win_rows, own_idx, ev_by_idx = [], {}, None
    for t in team_ids:
        b = bounds.get(t)
        idx = b[0] if b else last_index + 1
        own_idx[t] = idx
    either_idx = either[0] if either else last_index + 1

    def clock_min(e):
        return e["minute"] + e["second"] / 60.0
    ev_index = {e["index"]: e for e in events}
    for t in team_ids:
        b = bounds.get(t)
        row = dict(match_id=match_id, team_id=t, end_index_own=own_idx[t],
                   end_minute_own=clock_min(b[2]) if b else total / 60.0,
                   end_reason="none" if not b else b[1], end_index_either=either_idx,
                   end_minute_either=clock_min(ev_index[either_idx]) if either else total / 60.0,
                   end_reason_either="none" if not either else either[2],
                   end_team_either=np.nan if not either else either[1])
        row["window_minutes_own"] = (elapsed(b[2]["period"], _secs(b[2]["timestamp"])) if b else total) / 60.0
        row["window_minutes_either"] = (elapsed(ev_index[either_idx]["period"],
                                                _secs(ev_index[either_idx]["timestamp"])) if either else total) / 60.0
        row["match_minutes"] = total / 60.0
        win_rows.append(row)

    # passes and touches
    pass_rows, touch_rows = [], []
    for e in events:
        name = e["type"]["name"]
        t = e["team"]["id"]
        if name == "Pass":
            p = e["pass"]
            loc, end = e.get("location") or [np.nan, np.nan], p.get("end_location") or [np.nan, np.nan]
            rec = p.get("recipient")
            pass_rows.append(dict(
                match_id=match_id, index=e["index"], period=e["period"], minute=e["minute"], second=e["second"],
                team_id=t, player_id=e.get("player", {}).get("id"), player=e.get("player", {}).get("name"),
                recipient_id=rec["id"] if rec else None, recipient=rec["name"] if rec else None,
                x=loc[0], y=loc[1], end_x=end[0], end_y=end[1],
                pass_type=p.get("type", {}).get("name", "Open Play"),
                outcome=p.get("outcome", {}).get("name"), has_recipient=rec is not None,
                completed=("outcome" not in p) and rec is not None,
                in_window_own=e["index"] < own_idx.get(t, BIG), in_window_either=e["index"] < either_idx))
        if name in TOUCH_TYPES and e.get("location") and e.get("player"):
            touch_rows.append(dict(
                match_id=match_id, team_id=t, player_id=e["player"]["id"], index=e["index"], type=name,
                x=e["location"][0], y=e["location"][1],
                outcome=e.get("ball_receipt", {}).get("outcome", {}).get("name") if name == "Ball Receipt*" else None,
                in_window_own=e["index"] < own_idx.get(t, BIG), in_window_either=e["index"] < either_idx))

    # appearances from lineups
    app_rows = []
    starters = {(r["team_id"], r["player_id"]) for r in xi_rows}
    for team in lineups:
        tid = team["team_id"]
        for pl in team["lineup"]:
            if not pl["positions"]:
                continue
            pos_min = {}
            for q in pl["positions"]:
                a = elapsed(q["from_period"], _clock(q["from"]) - (2700 if q["from_period"] == 2 else 0))
                if q["to"] is None:
                    b = total
                else:
                    b = elapsed(q["to_period"], _clock(q["to"]) - (2700 if q["to_period"] == 2 else 0))
                a, b = min(a, total), min(b, total)
                if q["from_period"] == 1 and q["to_period"] == 1:
                    b = min(b, p1len)
                pos_min[q["position"]] = pos_min.get(q["position"], 0.0) + max(b - a, 0.0) / 60.0
            main = max(pos_min, key=pos_min.get)
            app_rows.append(dict(
                match_id=match_id, team_id=tid, player_id=pl["player_id"],
                player=pl["player_name"], minutes_played=sum(pos_min.values()),
                started=pl["positions"][0]["start_reason"] == "Starting XI",
                in_xi_event=(tid, pl["player_id"]) in starters,
                position=main, position_group=group_of(main),
                first_from=pl["positions"][0]["from"], first_from_period=pl["positions"][0]["from_period"],
                n_positions=len(pl["positions"])))

    return dict(passes=pass_rows, touches=touch_rows, starting_xi=xi_rows, appearances=app_rows,
                windows=win_rows, periods=periods, n_events=len(events),
                n_dup_ids=len(events) - len({e["id"] for e in events}))


def assemble(results):
    """results: list of parse_match outputs -> dict of DataFrames."""
    out = {}
    for k in ("passes", "touches", "starting_xi", "appearances", "windows"):
        out[k] = pd.DataFrame([r for res in results for r in res[k]])
    out["passes"]["player_id"] = out["passes"]["player_id"].astype("Int64")
    out["passes"]["recipient_id"] = out["passes"]["recipient_id"].astype("Int64")
    return out
