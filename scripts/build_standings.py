#!/usr/bin/env python3
"""Build data/external/standings.csv: final La Liga table position for every team
in the 7 seasons covered by the Barca pass-network project.

Source: football-data.co.uk results CSVs (downloaded with curl into
data/external/football_data/ if missing). Team names are mapped to StatsBomb
names (from data/raw/matches/11_{season_id}.json) so the table can be joined to
the StatsBomb match data. Position 2015/16 is cross-checked against StatsBomb's
own 380 results.

Run with:  python3 -I scripts/build_standings.py
"""

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FD_DIR = ROOT / "data" / "external" / "football_data"
SB_DIR = ROOT / "data" / "raw" / "matches"
OUT = ROOT / "data" / "external" / "standings.csv"
FD_URL = "https://www.football-data.co.uk/mmz4281/{code}/SP1.csv"

# (StatsBomb season_id, season label, football-data code)
SEASONS = [
    (41, "2008/2009", "0809"),
    (21, "2009/2010", "0910"),
    (22, "2010/2011", "1011"),
    (23, "2011/2012", "1112"),
    (26, "2014/2015", "1415"),
    (27, "2015/2016", "1516"),
    (2, "2016/2017", "1617"),
]

BARCA = "Barcelona"
CROSSCHECK_SEASON_ID = 27

# football-data.co.uk name -> StatsBomb name (spelling checked against 11_*.json)
FD_TO_SB = {
    "Alaves": "Deportivo Alavés",
    "Almeria": "Almería",
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atlético Madrid",
    "Barcelona": "Barcelona",
    "Betis": "Real Betis",
    "Celta": "Celta Vigo",
    "Cordoba": "Córdoba CF",
    "Eibar": "Eibar",
    "Elche": "Elche",
    "Espanol": "Espanyol",
    "Getafe": "Getafe",
    "Granada": "Granada",
    "Hercules": "Hércules",
    "La Coruna": "RC Deportivo La Coruña",
    "Las Palmas": "Las Palmas",
    "Leganes": "Leganés",
    "Levante": "Levante UD",
    "Malaga": "Málaga",
    "Mallorca": "Mallorca",
    "Numancia": "CD Numancia de Soria",
    "Osasuna": "Osasuna",
    "Real Madrid": "Real Madrid",
    "Recreativo": "Recreativo Huelva",
    "Santander": "Racing Santander",
    "Sevilla": "Sevilla",
    "Sociedad": "Real Sociedad",
    "Sp Gijon": "Sporting Gijón",
    "Tenerife": "Tenerife",
    "Valencia": "Valencia",
    "Valladolid": "Real Valladolid",
    "Vallecano": "Rayo Vallecano",
    "Villarreal": "Villarreal",
    "Xerez": "Xerez",
    "Zaragoza": "Real Zaragoza",
}


# ---------------------------------------------------------------- download

def ensure_football_data():
    FD_DIR.mkdir(parents=True, exist_ok=True)
    for _, _, code in SEASONS:
        path = FD_DIR / f"SP1_{code}.csv"
        if path.exists() and path.stat().st_size > 0:
            continue
        subprocess.run(
            ["curl", "-sSfL", FD_URL.format(code=code), "-o", str(path)],
            check=True,
        )
        print(f"downloaded {path.name}")


# ---------------------------------------------------------------- loading

def load_statsbomb(season_id):
    with open(SB_DIR / f"11_{season_id}.json", encoding="utf-8") as f:
        return json.load(f)


def load_football_data(code):
    with open(FD_DIR / f"SP1_{code}.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    matches = []
    for r in rows:
        hg, ag = int(r["FTHG"]), int(r["FTAG"])
        expected = "H" if hg > ag else ("A" if ag > hg else "D")
        assert r["FTR"] == expected, f"FTR {r['FTR']} inconsistent with {hg}-{ag}"
        matches.append((r["HomeTeam"], r["AwayTeam"], hg, ag))
    return matches


# ---------------------------------------------------------------- table

def compute_table(matches, label):
    """matches: list of (home, away, hg, ag) with names already in one namespace.

    Returns a list of dict rows, sorted by final position (1 = champion).
    Ties on points are broken by the La Liga order: head-to-head points among
    the tied teams, head-to-head goal difference, overall goal difference,
    overall goals scored. Every resolved tie is printed.
    """
    st = {}
    for h, a, hg, ag in matches:
        for t in (h, a):
            st.setdefault(t, dict(team=t, played=0, won=0, drawn=0, lost=0,
                                  gf=0, ga=0, points=0))
        for t, gf, ga in ((h, hg, ag), (a, ag, hg)):
            s = st[t]
            s["played"] += 1
            s["gf"] += gf
            s["ga"] += ga
            if gf > ga:
                s["won"] += 1
                s["points"] += 3
            elif gf == ga:
                s["drawn"] += 1
                s["points"] += 1
            else:
                s["lost"] += 1
    for s in st.values():
        s["gd"] = s["gf"] - s["ga"]

    by_pts = {}
    for s in st.values():
        by_pts.setdefault(s["points"], []).append(s["team"])

    ordered = []
    n_ties = 0
    for pts in sorted(by_pts, reverse=True):
        group = by_pts[pts]
        if len(group) == 1:
            ordered.append(group[0])
            continue
        n_ties += 1
        members = set(group)
        h2h = {t: dict(pts=0, gf=0, ga=0) for t in group}
        for h, a, hg, ag in matches:
            if h in members and a in members:
                for t, gf, ga in ((h, hg, ag), (a, ag, hg)):
                    h2h[t]["gf"] += gf
                    h2h[t]["ga"] += ga
                    h2h[t]["pts"] += 3 if gf > ga else (1 if gf == ga else 0)

        def key(t):
            return (-h2h[t]["pts"], -(h2h[t]["gf"] - h2h[t]["ga"]),
                    -st[t]["gd"], -st[t]["gf"])

        grp_sorted = sorted(group, key=key)
        # Every adjacent pair must be separated by at least one criterion.
        reasons = []
        for x, y in zip(grp_sorted, grp_sorted[1:]):
            kx, ky = key(x), key(y)
            assert kx != ky, f"[{label}] UNRESOLVED tie on {pts} pts: {x} vs {y}"
            names = ["head-to-head points", "head-to-head GD",
                     "overall GD", "goals scored"]
            reasons.append(names[next(i for i in range(4) if kx[i] != ky[i])])
        print(f"[{label}] tie on {pts} pts ({len(group)} teams):")
        for t, why in zip(grp_sorted, reasons + [None]):
            h = h2h[t]
            print(f"    {t}: h2h pts {h['pts']}, h2h GD {h['gf'] - h['ga']:+d}, "
                  f"overall GD {st[t]['gd']:+d}, GF {st[t]['gf']}"
                  + (f"  -> separated from next by {why}" if why else ""))
        ordered.extend(grp_sorted)

    rows = []
    for pos, t in enumerate(ordered, start=1):
        r = dict(st[t])
        r["position"] = pos
        rows.append(r)
    assert len({r["position"] for r in rows}) == len(rows)
    print(f"[{label}] teams={len(rows)} matches={len(matches)} "
          f"points-ties resolved={n_ties}")
    return rows


def check_table_shape(rows, matches, label):
    assert len(rows) == 20, f"[{label}] {len(rows)} teams, expected 20"
    assert len(matches) == 380, f"[{label}] {len(matches)} matches, expected 380"
    for r in rows:
        assert r["played"] == 38, f"[{label}] {r['team']} played {r['played']}"


def tier(pos):
    if pos <= 4:
        return 1
    if pos <= 10:
        return 2
    return 3


# ---------------------------------------------------------------- main

def main():
    ensure_football_data()

    # StatsBomb names and team ids, across all seven files.
    sb_matches = {}
    id_of = {}
    for sid, _, _ in SEASONS:
        sb_matches[sid] = load_statsbomb(sid)
        for m in sb_matches[sid]:
            for side in ("home_team", "away_team"):
                name = m[side][f"{side}_name"]
                tid = m[side][f"{side}_id"]
                prev = id_of.setdefault(name, tid)
                assert prev == tid, f"id conflict for {name}: {prev} vs {tid}"
    sb_names = set(id_of)

    # Every mapped name must exist in StatsBomb data.
    for fd, sb in FD_TO_SB.items():
        assert sb in sb_names, f"mapped StatsBomb name not in data: {sb!r} (from {fd!r})"

    out_rows = []
    tables = {}
    for sid, season, code in SEASONS:
        print(f"\n=== {season} (StatsBomb {sid}, football-data {code}) ===")
        fd_matches = load_football_data(code)
        mapped = []
        for h, a, hg, ag in fd_matches:
            if h not in FD_TO_SB or a not in FD_TO_SB:
                raise KeyError(f"[{season}] unmapped football-data team: {h!r} / {a!r}")
            mapped.append((FD_TO_SB[h], FD_TO_SB[a], hg, ag))
        rows = compute_table(mapped, season)
        check_table_shape(rows, mapped, season)
        tables[sid] = rows

        # Every Barcelona opponent in StatsBomb's file must be in this season's table.
        table_names = {r["team"] for r in rows}
        sb_season_names = set()
        for m in sb_matches[sid]:
            sb_season_names.add(m["home_team"]["home_team_name"])
            sb_season_names.add(m["away_team"]["away_team_name"])
        opponents = set()
        for m in sb_matches[sid]:
            h, a = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
            if BARCA in (h, a):
                opponents.add(a if h == BARCA else h)
        missing_opp = opponents - table_names
        assert not missing_opp, f"[{season}] Barca opponents not in table: {missing_opp}"
        assert len(opponents) > 0 and BARCA in table_names
        not_in_sb_file = sorted(table_names - sb_season_names)
        print(f"[{season}] Barca opponents in StatsBomb file: {len(opponents)} "
              f"(all mapped to the table)")
        for t in not_in_sb_file:
            print(f"[{season}] NOTE: {t} is in the table but absent from 11_{sid}.json; "
                  f"StatsBomb id taken from another season's file ({id_of[t]})")

        for r in rows:
            fd_name = next(k for k, v in FD_TO_SB.items() if v == r["team"])
            out_rows.append({
                "season_id": sid,
                "season": season,
                "team_statsbomb": r["team"],
                "team_id_statsbomb": id_of.get(r["team"], ""),
                "team_footballdata": fd_name,
                "position": r["position"],
                "played": r["played"],
                "won": r["won"],
                "drawn": r["drawn"],
                "lost": r["lost"],
                "gf": r["gf"],
                "ga": r["ga"],
                "gd": r["gd"],
                "points": r["points"],
                "tier": tier(r["position"]),
            })

    # ------------------------------------------------------- cross-check 2015/16
    print("\n=== Cross-check 2015/16: StatsBomb 11_27 results vs football-data ===")
    sb27 = sb_matches[CROSSCHECK_SEASON_ID]
    assert len(sb27) == 380, f"11_27 has {len(sb27)} matches"
    sb_results = [
        (m["home_team"]["home_team_name"], m["away_team"]["away_team_name"],
         int(m["home_score"]), int(m["away_score"]))
        for m in sb27
    ]
    sb_rows = compute_table(sb_results, "xcheck StatsBomb 2015/2016")
    check_table_shape(sb_rows, sb_results, "xcheck StatsBomb 2015/2016")
    sb_pos = {r["team"]: r["position"] for r in sb_rows}
    fd_rows = tables[CROSSCHECK_SEASON_ID]
    fd_pos = {r["team"]: r["position"] for r in fd_rows}
    assert sb_pos == fd_pos, f"position mismatch: {sorted(sb_pos.items())} vs {sorted(fd_pos.items())}"
    fd_pts = {r["team"]: (r["points"], r["gd"], r["gf"]) for r in fd_rows}
    sb_pts = {r["team"]: (r["points"], r["gd"], r["gf"]) for r in sb_rows}
    assert fd_pts == sb_pts, "points/GD/GF differ between StatsBomb and football-data"
    print("cross-check PASSED: positions, points, GD and GF identical for all 20 teams")

    # ------------------------------------------------------- sanity facts
    expected_first = {
        "2008/2009": BARCA, "2009/2010": BARCA, "2010/2011": BARCA,
        "2011/2012": "Real Madrid", "2014/2015": BARCA, "2015/2016": BARCA,
        "2016/2017": "Real Madrid",
    }
    expected_second = {"2011/2012": BARCA, "2016/2017": BARCA}
    for out in out_rows:
        pos = out["position"]
        if pos == 1:
            assert expected_first[out["season"]] == out["team_statsbomb"], (
                f"{out['season']} champion is {out['team_statsbomb']}")
        if pos == 2 and out["season"] in expected_second:
            assert expected_second[out["season"]] == out["team_statsbomb"], (
                f"{out['season']} runner-up is {out['team_statsbomb']}")
    barca = {o["season"]: o["position"] for o in out_rows if o["team_statsbomb"] == BARCA}
    print("Barcelona positions:", barca)
    assert barca == {"2008/2009": 1, "2009/2010": 1, "2010/2011": 1, "2011/2012": 2,
                     "2014/2015": 1, "2015/2016": 1, "2016/2017": 2}, barca

    # ------------------------------------------------------- write
    assert len(out_rows) == 140, len(out_rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ["season_id", "season", "team_statsbomb", "team_id_statsbomb",
              "team_footballdata", "position", "played", "won", "drawn", "lost",
              "gf", "ga", "gd", "points", "tier"]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {len(out_rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
