# Decision log

Every decision that shapes the data, the methods or the deliverables gets a row here, with its date, rationale and the alternatives considered. A decision is never edited after the fact: a change is recorded as a new row that **supersedes** the old one.

Status values:
- **locked**: in force;
- **superseded by Dxx**: replaced by a later decision;
- **proposed**: awaiting approval.

## Scope and framing (proposal stage, Nov 3)

| ID | Date | Decision | Rationale | Alternatives considered | Status |
|---|---|---|---|---|---|
| D01 | 2026-11-03 | Domain: sports analytics, soccer | Author's interest; rich open event data | basketball (NBA shot archetypes) | locked |
| D02 | 2026-11-03 | Solo project, laptop only, ~5 weeks | Course allows solo; no cluster needed | team of 2–3 | locked |
| D03 | 2026-11-03 | Topic: Barcelona pass networks + player roles | Fits L7 (graphs) + L4/L2 (PCA, k-means) | NBA shot archetypes; World Cup hubs | locked |
| D04 | 2026-11-03 | Comparison: Guardiola 2008/09–2011/12 vs Luis Enrique 2014/15–2016/17 (La Liga, 246 matches) | Two treble sides with famously different styles | sequence of every coach; change-point detection | locked |
| D05 | 2026-11-03 | Classic data mining only, no deep learning | Instructor requirement | — | locked |
| D06 | 2026-11-03 | Deliverables: code repo, Streamlit dashboard (3 views), report, video | Proposal (Section 3) | report only | locked |
| D07 | 2026-11-03 | Hypotheses H1/H2, five metrics, and success rules exactly as in the proposal (O1–O3) | Pre-stated in the submitted proposal | — | locked |

## Execution plan (Nov 5, approved by Aniket: "approve all")

| ID | Date | Decision | Rationale | Alternatives considered | Status |
|---|---|---|---|---|---|
| D08 | 2026-11-05 | **Network window:** each team's starting XI until that team's own first substitution or red card. Networks are built for both teams. | Keeps 11 fixed nodes per network; Barça − opponent needs both | ends at either team's first sub; full match (both kept as robustness) | locked |
| D09 | 2026-11-05 | **M3 triangle share:** undirected graph, edges with ≥ 3 passes (both directions combined); each triangle weighted by its weakest edge; M3 = weighted share of triangles whose 3 players all have their mean touch location in x 30–90, y 18–62 (120 × 80 pitch) | Captures midfield triangles by where players actually play | all three listed DM/CM by position (rejected: blind to drifting) | locked |
| D10 | 2026-11-05 | **O1 control:** stratified permutation test (10,000) and stratified bootstrap (10,000), with strata = home/away × opponent final position {1–4, 5–10, 11–20}; BH across the 5 metrics; OLS (era + home + opp_position, robust SEs) as a secondary check only | Implements the "control" promised in the proposal without changing the method | regression residuals (a method change) | locked |
| D11 | 2026-11-05 | **Opponent strength:** final league tables computed from football-data.co.uk results CSVs; 2015/16 cross-checked against StatsBomb's full-season results | Reproducible and citable | hand entry from Wikipedia (fallback) | locked |
| D12 | 2026-11-05 | **Era role-mix test:** exact permutation over the C(7,4) = 35 season labelings (minimum p ≈ 0.029) | Player-seasons aren't independent; the season is the honest unit | chi-square (rejected); shuffling over players | locked |
| D13 | 2026-11-05 | **O2 setup:** player-seasons with ≥ 900 minutes; ~25 standardized features from **full-match** networks; PCA to ≥ 80% variance; k-means k = 3–6 (n_init = 50); stability = median ARI over 200 match bootstraps; one pre-declared fallback feature set (spatial only) | Substitutes must count toward profiles; avoids feature shopping | window networks for profiles (rejected: drop substitutes) | locked |
| D14 | 2026-11-05 | **O3:** era profile = minute-weighted mean of standardized season features; cosine distance in the full standardized space | Uses all information, not 2 PCs | PCA-space distance | locked |
| D15 | 2026-11-05 | **Freeze rule:** D08–D14 are frozen before any era comparison is run (by Nov 10). Later changes need a new row plus a mention in the report. | Prevents tuning definitions to results | — | locked |
| D16 | 2026-11-05 | **Repo:** a separate git repo, **public on GitHub** from the start. Raw StatsBomb JSON is gitignored; the download script and processed data are committed, with attribution in the README. | Aniket's choice; reproducibility | private until submission (proposed, overridden) | locked |
| D17 | 2026-11-05 | **Dashboard:** local Streamlit, shown in the video; deploy to Streamlit Community Cloud only if W5 has slack | Removes deployment risk before the deadline | deploy first | locked |
| D18 | 2026-11-05 | **Schedule:** analysis done Dec 1; full report draft Dec 6; video recorded Dec 9; submit Dec 10 (hard deadline Dec 11; final exam Dec 8) | Exam-week buffer | — | locked |
| D19 | 2026-11-05 | **Pitch plots:** own matplotlib code (mplsoccer not installed) | Avoids a dependency | install mplsoccer | locked |

## Week 1 (pipeline)

| ID | Date | Decision | Rationale | Alternatives considered | Status |
|---|---|---|---|---|---|
| D20 | 2026-11-05 | **Completed pass** = `Pass` event with no `pass.outcome` key **and** a `pass.recipient`. All pass types count (open play and set pieces); set pieces are tagged so robustness can drop them. | StatsBomb convention (completed passes carry no outcome); standard in pass-network papers | open play only (kept as robustness) | proposed |
| D21 | 2026-11-05 | **Window boundary** is found by event `index` (the first `Substitution` event, or red/second-yellow card, for that team); passes with a lower index fall in the window | Avoids clock arithmetic across periods | elapsed-time cutoff | proposed |
| D22 | 2026-11-05 | **Standings:** final tables computed from football-data.co.uk (3/1/0 points). Ties broken by head-to-head points, then head-to-head GD, then overall GD, then goals for; the head-to-head away-goals step is omitted, and no tie in our seasons needed it. Verified: StatsBomb's 2015/16 results give an identical table; points for the top 4 in all 7 seasons match the published tables. | Reproducible; verified | Wikipedia hand entry | locked |
| D23 | 2026-11-05 | **Implementation details (W1):** a window with no sub or red card runs to the match end (`end_reason` = "none"); window and playing minutes are elapsed play time (first-half length + period-2 timestamp); the touches table holds on-ball events (Pass, Ball Receipt*, Carry, Dribble, Shot, Ball Recovery, Miscontrol, Dispossessed, Clearance, Interception, Block, Duel); node location = mean of the player's in-scope touches; edges skip self-passes and non-node recipients; edge lists carry a `scope` column (own / either / full) | Needed to build the pipeline; none affects a hypothesis | — | locked |
| D24 | 2026-11-05 | **Position groups:** Left/Right Midfield → W; attacking midfielders → CM; wing-backs → FB. Forward line (for M4) = Left/Right Wing, Center Forward, Left/Right Center Forward, Secondary Striker, Striker. Wing-backs are **not** forwards (Fig. 1's script counted them; corrected here). | Wing-backs are defenders by role | count anything named "Wing" | proposed |
| D25 | 2026-11-05 | **Short windows:** 27 matches have a team window under 30 min (shortest 3 min, an early red card). Keep every match in the primary analysis; robustness re-runs O1 without matches where either team's window is under 30 min. | Dropping is a post-hoc exclusion; shares from short windows are noisy, so check sensitivity | drop them from the primary analysis | proposed |
| D26 | 2026-11-05 | **Match 70220:** StatsBomb's lineup has 12 Barça players on the pitch for ~5 min (overlapping entries). Leave as is: it adds ~46 player-minutes to minutes played only and doesn't touch any network. | Negligible; no clean fix in the source | hand-correct the lineup | proposed |

### Data notes (facts, not decisions)
- 2008/09 (season 41) has only 31 of 38 Barcelona matches; **both Mallorca matches are absent** from StatsBomb's file. Coverage limitation, to be reported.
- No completed pass lacks a recipient: every pass without a recipient is incomplete, so D20's recipient clause is redundant in practice. Excluded (incomplete) passes: Barça 13.7%, opponents 28.5%. Opponent networks are therefore sparser.
- `pass.type` also takes Recovery and Interception (treated as open play). Set pieces = Corner, Free Kick, Throw-in, Goal Kick, Kick Off: 16.3% of completed in-window passes.
- Window length (own): Barça mean 60.0 min, opponents 56.6. Windows end on a red card for 8 Barça and 15 opponent teams.
- Verified by the manager: pass counts and window boundaries match the raw JSON for 24 team-matches; window edge weights match the raw completed-pass counts for 60 team-matches (0 mismatches).
