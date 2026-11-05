"""Window boundaries: a team's starting XI until its own first substitution or red card."""
RED = {"Red Card", "Second Yellow"}


def _card(e):
    for k in ("foul_committed", "bad_behaviour"):
        name = e.get(k, {}).get("card", {}).get("name")
        if name:
            return name
    return None


def find_boundaries(events):
    """Return {team_id: (index, reason, event)} for each team's first substitution / red card
    (lowest event index). Teams with none are absent."""
    out = {}
    for e in events:  # events sorted by index
        t = e["team"]["id"]
        if t in out:
            continue
        name = e["type"]["name"]
        if name == "Substitution":
            out[t] = (e["index"], "substitution", e)
        elif _card(e) in RED:
            out[t] = (e["index"], "red card", e)
    return out


def either_boundary(bounds):
    """Earliest boundary over both teams: (index, team_id, reason) or None."""
    if not bounds:
        return None
    t, (i, r, _) = min(bounds.items(), key=lambda kv: kv[1][0])
    return i, t, r
