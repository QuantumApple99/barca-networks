# Mining Barcelona's Passing Networks and Player Roles: Guardiola vs. Luis Enrique

SI 671 Data Mining (University of Michigan, Fall 2026) course project by Aniket Panthri.

Do Barcelona's two treble-winning sides, Guardiola (2008/09–2011/12) and Luis Enrique (2014/15–2016/17), differ measurably in how they pass? And how did player roles persist or pass on between the eras? The project mines 246 La Liga matches with classic data-mining methods:
- **graph mining:** PageRank and betweenness on per-match pass networks;
- **PCA + k-means:** to discover player roles;
- **cosine similarity:** to track role succession.

## Objectives
- **O1:** Test whether the two styles really differ: five network metrics, stratified permutation tests, bootstrap CIs, Benjamini–Hochberg.
- **O2:** Find player roles from passing behaviour: PCA + k-means on player-season profiles, with silhouette and stability checks.
- **O3:** Track how roles changed hands: cosine drift for core players, successor search for departed ones.

## Reproduce
```bash
pip install -r requirements.txt
./scripts/download.sh          # ~800 MB of StatsBomb JSON into data/raw/ (not committed)
python run_all.py              # raw → processed parquet → results and figures
pytest                         # tests
```

## Layout
- `src/barca/`: the pipeline (parsing, match windows, networks, metrics, statistics, roles, succession)
- `scripts/`: download and build scripts
- `data/processed/`: derived tables (parquet)
- `data/external/`: league standings, with their sources
- `notebooks/`: validation and one notebook per objective
- `app/`: Streamlit dashboard
- `docs/DECISIONS.md`: log of every methodological decision, with rationale
- `report/`: technical report and figures

## Data and attribution
Event data: **StatsBomb Open Data** (https://github.com/statsbomb/open-data), used under StatsBomb's open-data terms; data courtesy of StatsBomb. League tables are computed from football-data.co.uk results (see `data/external/SOURCES.md`).

## AI use
AI assistants (Claude) helped plan, write and review code and text in this project, in line with the course's AI-use policy. Design decisions are logged in `docs/DECISIONS.md` and were made or approved by the author.
