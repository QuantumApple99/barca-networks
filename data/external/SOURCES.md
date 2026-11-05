# External data sources

## football-data.co.uk (La Liga results)

Final league tables are computed from match results published by football-data.co.uk
(Spanish Primera División, `SP1`). Credit: football-data.co.uk.

- URLs (fetched 2026-11-05 with `curl -sSfL`):
  - https://www.football-data.co.uk/mmz4281/0809/SP1.csv (2008/09)
  - https://www.football-data.co.uk/mmz4281/0910/SP1.csv (2009/10)
  - https://www.football-data.co.uk/mmz4281/1011/SP1.csv (2010/11)
  - https://www.football-data.co.uk/mmz4281/1112/SP1.csv (2011/12)
  - https://www.football-data.co.uk/mmz4281/1415/SP1.csv (2014/15)
  - https://www.football-data.co.uk/mmz4281/1516/SP1.csv (2015/16)
  - https://www.football-data.co.uk/mmz4281/1617/SP1.csv (2016/17)
- Local copies: `data/external/football_data/SP1_{code}.csv`.
- Output: `data/external/standings.csv`, built by `scripts/build_standings.py`.

## Table computation

- Points: 3 for a win, 1 for a draw, 0 for a loss. Each season is checked for 20 teams and 380 matches.
- Ties on points are broken in this order:
  1. head-to-head points among the tied teams,
  2. head-to-head goal difference,
  3. overall goal difference,
  4. overall goals scored.

  The head-to-head mini-league is computed once over all teams level on points. The script
  prints every tie it resolves and fails if any pair remains unresolved.

## Cross-check, 2015/16

The 2015/16 table was also computed from StatsBomb's own 380 results in `data/raw/matches/11_27.json`
(`home_score`, `away_score`). Positions, points, goal difference and goals for and against
match football-data for all 20 teams.

## Known data limitation

StatsBomb's 2008/09 file (`11_41.json`) contains 31 Barcelona matches, and it does not include
Mallorca. Mallorca's StatsBomb id (1043) comes from the other seasons' files. The league table
for 2008/09 comes from football-data.co.uk and is complete (380 matches).
