"""StatsBomb position name -> 7 role groups."""
POSITION_GROUP = {
    "Goalkeeper": "GK",
    "Center Back": "CB", "Left Center Back": "CB", "Right Center Back": "CB",
    "Left Back": "FB", "Right Back": "FB", "Left Wing Back": "FB", "Right Wing Back": "FB",
    "Center Defensive Midfield": "DM", "Left Defensive Midfield": "DM", "Right Defensive Midfield": "DM",
    "Center Midfield": "CM", "Left Center Midfield": "CM", "Right Center Midfield": "CM",
    "Center Attacking Midfield": "CM", "Left Attacking Midfield": "CM", "Right Attacking Midfield": "CM",
    # Uncertain: wide midfielders (4-4-2 style); mapped to W.
    "Left Midfield": "W", "Right Midfield": "W",
    "Left Wing": "W", "Right Wing": "W",
    "Center Forward": "CF", "Left Center Forward": "CF", "Right Center Forward": "CF",
    "Secondary Striker": "CF", "Striker": "CF",
}
FORWARD_LINE = {"Left Wing", "Right Wing", "Center Forward", "Left Center Forward",
                "Right Center Forward", "Secondary Striker", "Striker"}
# Names present in the data whose mapping is a judgement call (reported to the manager).
UNSURE = ["Left Midfield", "Right Midfield", "Left Attacking Midfield", "Right Attacking Midfield",
          "Center Attacking Midfield", "Left Wing Back", "Right Wing Back"]


def group_of(position_name):
    return POSITION_GROUP.get(position_name)


def is_forward_line(position_name):
    return position_name in FORWARD_LINE
