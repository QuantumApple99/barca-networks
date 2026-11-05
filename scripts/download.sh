#!/bin/zsh
# Download StatsBomb open data for Barcelona La Liga matches, 7 seasons.
# Uses curl (Python's urllib hits an SSL certificate error on the author's machine).
set -e
cd "$(dirname "$0")/.."
BASE=https://raw.githubusercontent.com/statsbomb/open-data/master/data
SEASONS=(41 21 22 23 26 27 2)
for s in $SEASONS; do
  [[ -s data/raw/matches/11_$s.json ]] || curl -sSfL "$BASE/matches/11/$s.json" -o data/raw/matches/11_$s.json
done
# Barcelona team_id = 217
python3 -c "
import json,glob
ids=[]
for f in sorted(glob.glob('data/raw/matches/11_*.json')):
    for m in json.load(open(f)):
        if 217 in (m['home_team']['home_team_id'], m['away_team']['away_team_id']): ids.append(m['match_id'])
print('\n'.join(map(str,ids)))
" > data/raw/barca_match_ids.txt
echo "matches: $(wc -l < data/raw/barca_match_ids.txt)"
cat data/raw/barca_match_ids.txt | xargs -P 8 -I{} zsh -c "[[ -s data/raw/events/{}.json ]] || curl -sSfL --retry 3 $BASE/events/{}.json -o data/raw/events/{}.json"
cat data/raw/barca_match_ids.txt | xargs -P 8 -I{} zsh -c "[[ -s data/raw/lineups/{}.json ]] || curl -sSfL --retry 3 $BASE/lineups/{}.json -o data/raw/lineups/{}.json"
echo "events: $(ls data/raw/events | wc -l)  lineups: $(ls data/raw/lineups | wc -l)"
du -sh data/raw
