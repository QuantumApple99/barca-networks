# 2. Data and Network Construction

<!-- Draft W1, 2026-11-09. Numbers produced by `python run_all.py`; decision IDs refer to docs/DECISIONS.md. Edit into your own voice. -->

## 2.1 Source and scope

We use StatsBomb Open Data [1], which records every on-ball event in a match (passes, carries, receptions, shots and more) with pitch coordinates on a 120 × 80 grid, oriented so that each team attacks towards x = 120. We take Barcelona's La Liga matches in two eras:

| Era | Seasons | Matches (of 38 per season) |
|---|---|---|
| Guardiola | 2008/09, 2009/10, 2010/11, 2011/12 | 31, 35, 33, 37 = 136 |
| Luis Enrique | 2014/15, 2015/16, 2016/17 | 38, 38, 34 = 110 |

That is 246 of the 266 league matches played. StatsBomb's Barcelona files were collected around Lionel Messi's career, so outside 2015/16 (a complete season) they mostly cover matches Messi played. Both 2008/09 matches against Mallorca are absent. The comparison is therefore best read as "Barcelona with Messi", which describes nearly every match in both eras.

For each match we also record home/away and the opponent's final league position. We compute the final tables from the match results on football-data.co.uk, using La Liga's head-to-head tiebreak (D22). The 2015/16 table computed this way is identical to one built from StatsBomb's own complete 2015/16 results. Opponents are grouped into three tiers (finished 1–4, 5–10 or 11–20), giving six home/away × tier strata (D10). The smallest stratum holds 9 matches per era.

## 2.2 Passes

The 246 matches contain 255,199 pass events across both teams. A pass is **completed** when StatsBomb records no failed outcome for it (D20). Barcelona completed 86.3% of its passes and its opponents 71.5%, which is one reason every team-level metric is analysed as *Barcelona minus opponent* within the same match. All pass types count, including set pieces (corners, free kicks, throw-ins, goal kicks and kick-offs). Set pieces make up 16% of completed passes and are tagged so that a robustness check can remove them.

## 2.3 Pass networks

For each match and each team we build a **directed, weighted pass network**. Nodes are players, and an edge *i → j* carries the number of completed passes from *i* to *j*.

- **Team-style networks (O1)** use the **starting XI only**, from kick-off until that team's first substitution or red card (D08, D21). Cutting the network there keeps exactly 11 nodes and avoids mixing players who were never on the pitch together.
  - Barcelona's networks cover 60.0 minutes on average (median 61.8); opponents' cover 56.6.
  - 236 of Barcelona's 246 windows end on a substitution, 8 on a red card, and 2 run the full match.
  - Together they hold 133,950 completed passes: 98,289 by Barcelona and 35,661 by opponents.
  - In 27 matches one team's window lasts under 30 minutes, usually because of an early substitution or a red card. These matches stay in the main analysis, and a robustness check removes them (D25). Two alternative cut-offs are also tested: the first substitution by *either* team, and the full match.
- **Player-profile networks (O2, O3)** use the **full match** with every player who appeared, so that substitutes contribute to their season profiles (D13).

Each node also carries the player's starting position, mapped to seven groups (GK, CB, FB, DM, CM, W, CF; D24), and the player's mean touch location within the window. That location is used by the midfield-triangle metric (D09). Over the seven seasons, 69 Barcelona players appear. **108 player-seasons reach the 900-minute threshold** used for role discovery: 14, 16, 14 and 18 in Guardiola's seasons, and 17, 13 and 16 in Luis Enrique's.

## 2.4 Validation

Every stage is checked against the raw files:
- **Pass counts:** for match 69276 (Barcelona 3–0 Atlético Madrid, 2011), the pipeline reproduces the raw totals of 1,196 passes, 1,138 with a recipient and 1,022 completed.
- **Window cut-offs:** for 24 team-matches, independently recomputed boundaries match the pipeline exactly.
- **Network edges:** for 60 team-matches, network edge weights sum to the independently counted completed passes, with no mismatches.
- **Minutes played:** substitution times derived from lineups agree with substitution events to within 0.15 minutes.
- **Source data issue:** one source error was found. In match 70220, StatsBomb's lineup places 12 Barcelona players on the pitch for about five minutes. It affects minutes played only and is left as recorded (D26).

<!-- Figure candidate: Fig. 1 from the proposal (two Clásico networks), regenerated from the pipeline. -->
