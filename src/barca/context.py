"""Match context: opponent final league position, tier and O1 stratum (D10, D22)."""
import pandas as pd

from . import io

TIER_LABELS = {1: "top4", 2: "5-10", 3: "11-20"}


def tier_of(position: int) -> int:
    """D10 tiers: 1 = positions 1-4, 2 = 5-10, 3 = 11-20."""
    return 1 if position <= 4 else (2 if position <= 10 else 3)


def add_opponent_strength(matches: pd.DataFrame) -> pd.DataFrame:
    """Join each match to its opponent's final league position (data/external/standings.csv)."""
    st = pd.read_csv(io.ROOT / "data" / "external" / "standings.csv")
    st = st[["season_id", "team_id_statsbomb", "position", "points"]].rename(
        columns={"team_id_statsbomb": "opponent_id", "position": "opp_position", "points": "opp_points"})
    out = matches.merge(st, on=["season_id", "opponent_id"], how="left", validate="many_to_one")
    missing = out[out.opp_position.isna()]
    assert missing.empty, f"no standings row for: {missing[['season', 'opponent']].values.tolist()}"
    out["opp_position"] = out.opp_position.astype(int)
    out["opp_tier"] = out.opp_position.map(tier_of)
    out["stratum"] = out.barca_home.map({True: "home", False: "away"}) + "_" + out.opp_tier.map(TIER_LABELS)
    return out
