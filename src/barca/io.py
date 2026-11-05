"""Raw-data loading. Paths are relative to the repo root (found via the package location)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
BARCA_ID = 217
SEASON_ERA = {41: "Guardiola", 21: "Guardiola", 22: "Guardiola", 23: "Guardiola",
              26: "Luis Enrique", 27: "Luis Enrique", 2: "Luis Enrique"}


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_events(match_id):
    ev = _load(RAW / "events" / f"{match_id}.json")
    return sorted(ev, key=lambda e: e["index"])


def load_lineups(match_id):
    return _load(RAW / "lineups" / f"{match_id}.json")


def load_match_metadata():
    """All raw match dicts across the 7 season files (2015/16 includes non-Barca matches)."""
    out = []
    for sid in SEASON_ERA:
        out.extend(_load(RAW / "matches" / f"11_{sid}.json"))
    return out


def barca_match_ids():
    with open(RAW / "barca_match_ids.txt") as f:
        return [int(x) for x in f.read().split()]
