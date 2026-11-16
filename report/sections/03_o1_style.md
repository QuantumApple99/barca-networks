# 3. O1: Did the Two Styles Really Differ?

<!-- Draft W2, 2026-11-16. Numbers from notebooks/02_O1_style.py (pre-registered test) and notebooks/02b_O1_explore_hubs.py (exploratory). Decision IDs refer to docs/DECISIONS.md. Edit into your own voice. -->

## 3.1 Method

For every match we compute five metrics on each team's starting-XI pass network (Section 2.3). We then take **Barcelona's value minus the opponent's** in the same match, which removes anything shared by both teams on the day: pitch, weather, referee, game state.

| Metric | Plain meaning | Hypothesis | Predicted change, Luis Enrique − Guardiola |
|---|---|---|---|
| M1 Gini of PageRank | How unequal the players' importance is in the passing | H2 | higher |
| M2 Betweenness centralization | How much play has to go through one bridge player | H1, H2 | higher |
| M3 Central-midfield triangle share | Share of closed passing triangles formed by central players (D09) | H1 | lower |
| M4 Front-line reception share | Share of passes received by the forward line (D24) | H2 | higher |
| M5 Progressive pass share | Share of passes, from outside the team's own defensive 40%, that end ≥ 25% closer to goal (D28) | H2 | higher |

**The era effect T** is the Luis Enrique minus Guardiola difference in these per-match values. It is estimated *within* six strata (home/away × opponent's final league position 1–4, 5–10, 11–20) and then averaged, so it compares like with like (D10, D31).

- **Significance:** a stratified permutation test (10,000 shuffles of the era labels within strata, two-sided; D27).
- **Uncertainty:** a stratified bootstrap 95% confidence interval (10,000 resamples).
- **Multiple testing:** Benjamini–Hochberg across the five metrics.

All of these rules, and the decision rule below, were fixed and committed to the public repository before any comparison between eras was run (D15, D27–D32).

**Decision rule.** A metric *supports* its hypothesis if its BH-adjusted p < 0.05 with the predicted sign, and all seven season averages fall on their era's predicted side of the midpoint between the two era means (D32). H1 needs both of its metrics to support it (M2, M3). H2 needs at least three of its four (M1, M2, M4, M5).

## 3.2 Results

| Metric | T (95% CI) | p | p (BH) | Predicted sign? | Season condition |
|---|---|---|---|---|---|
| M1 PageRank Gini | −0.035 (−0.051, −0.019) | < 0.001 | < 0.001 | **no, opposite** | fails |
| M2 Betweenness centralization | −0.080 (−0.113, −0.048) | < 0.001 | < 0.001 | **no, opposite** | fails |
| M3 Midfield triangle share* | −0.079 (−0.147, −0.012) | 0.024 | 0.030 | yes | fails |
| M4 Front-line reception share | +0.037 (+0.008, +0.067) | 0.019 | 0.030 | yes | fails |
| M5 Progressive pass share | +0.009 (−0.006, +0.024) | 0.28 | 0.28 | — | fails |

\* n = 236 matches; 10 team windows with no qualifying triangle, all short, are excluded from M3 only (D29). All other metrics use n = 246.

The two smallest p-values sit at the permutation floor of 1/10,001: not one of the 10,000 shuffles produced a difference as large. An ordinary regression of each metric on era, home/away and opponent position (HC3 standard errors) gives era coefficients within 0.001 of T for every metric.

**Verdict under the pre-registered rule: H1 is not supported; H2 is not supported.**

<!-- Figure: report/figures/o1_forest.png (T with 95% CI per metric). Figure: report/figures/o1_season_means.png (season means, midpoint line). -->

## 3.3 What the results say

**The concentration hypothesis is reversed.** H2 expected Luis Enrique's Barcelona to channel its passing through fewer players. The data shows the opposite, with the two clearest effects in the study:
- Guardiola's Barcelona was *more* unequal in passing importance (M1) and *more* dependent on a single bridge player (M2), relative to its opponents.
- The change comes from Barcelona itself, not the opponents. Barcelona's own PageRank Gini fell from 0.209 to 0.174, and its betweenness centralization from 0.279 to 0.213. Opponents stayed flat: 0.225 → 0.225 and 0.218 → 0.234.

**Two predicted shifts appear on average but not season by season.** Under Luis Enrique, a smaller share of Barcelona's passing triangles was central (M3), and more passes reached the forward line (M4), both as predicted. But neither holds in every season:
- 2014/15, Luis Enrique's first and treble-winning season, still looks like a Guardiola side on both M3 (0.116, above the 0.080 midpoint) and M4 (−0.054, below the −0.040 midpoint).
- On the Guardiola side, 2009/10 already falls on the Luis Enrique side of the midpoint for M3 (0.053), as does 2008/09 for M4 (+0.002).

The eras overlap, and the change looks gradual rather than a clean break at the coaching change. Under our rule this means the shift isn't consistent enough to count as support.

**Progressive passing (M5) did not change.** Relative to opponents, both sides moved the ball forward at similar rates.

## 3.4 Exploratory: who were the hubs?

*This analysis was not pre-registered. It describes the result above and does not test anything.*

To see where Guardiola-era concentration came from, we recorded which Barcelona starter had the highest PageRank and the highest betweenness in each match.

| | Guardiola (136 matches) | Luis Enrique (110 matches) |
|---|---|---|
| Most frequent top-PageRank player | Xavi: 62% of matches (Iniesta 11%) | Iniesta 18%, Messi 17%, Neymar 10% |
| Most frequent top-betweenness player | Xavi: 46% (Iniesta 8%) | Mascherano 21%, Alba 16%, Piqué 13% |
| Mean of each match's highest PageRank | 0.153 | 0.136 |

Under Guardiola, the passing network had one dominant hub, Xavi. Under Luis Enrique, the hub role rotated among several players, and Messi became the top PageRank player in 19 matches, up from 3. So "tiki-taka", in these data, means *circulation through a central organizer*, not an evenly shared ball. That is a sharper description than the popular one, and it is consistent with the single-match illustration in our proposal (Fig. 1: Gini 0.230 in 2010 vs 0.170 in 2015, with Xavi as the hub).

## 3.5 Caveats

- **Observational.** Era bundles the coach with the squad, as Xavi aged and left in 2015 and Suárez and Neymar arrived. We describe differences and claim no causes.
- **Coverage.** Outside 2015/16, the data covers mainly matches Messi played, and 2008/09 lacks both Mallorca matches (Section 2.1).
- **Non-independent matches.** Matches within a season share a squad and form, so match-level p-values may be somewhat optimistic. The season condition is the guard against this, and it is exactly where M3 and M4 fail.
- **Robustness.** The alternative network window, the short-window exclusion and the set-piece exclusion are reported in Section 6.
