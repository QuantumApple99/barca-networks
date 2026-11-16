"""Stage 3: O1 era comparison. Run: python notebooks/02_O1_style.py (needs metrics stage done)."""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from barca import io, stats  # noqa: E402

HYP = {"M1": "H2", "M2": "H1/H2", "M3": "H1", "M4": "H2", "M5": "H2"}
NAMES = {"M1": "M1 Gini of PageRank", "M2": "M2 Betweenness centralization", "M3": "M3 Midfield triangle share",
         "M4": "M4 Front-line reception", "M5": "M5 Progressive share"}
COL = {stats.ERA_PEP: "#1f4e9c", stats.ERA_LE: "#d1495b"}
TAB, FIG = ROOT / "report" / "tables", ROOT / "report" / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    D = pd.read_parquet(io.PROCESSED / "o1_diffs.parquet")
    rows, season_rows, supports, season_ok = [], [], {}, {}
    for m, sign in stats.PRED_SIGN.items():
        d = D[f"d{m}"]
        use = D[d.notna()]
        T, p = stats.stratified_permutation_test(use[f"d{m}"], use.era, use.stratum)
        lo, hi = stats.stratified_bootstrap_ci(use[f"d{m}"], use.era, use.stratum)
        raw = stats.raw_difference(use[f"d{m}"], use.era)
        ok, sm, mid = stats.season_condition(use[f"d{m}"], use.era, use.season, sign)
        ols = smf.ols(f"d{m} ~ C(era, Treatment('{stats.ERA_PEP}')) + barca_home + opp_position",
                      data=use.assign(barca_home=use.barca_home.astype(int))).fit(cov_type="HC3")
        k = [i for i in ols.params.index if "era" in i][0]
        ci = ols.conf_int().loc[k]
        for (era, season), v in sm.items():
            season_rows.append(dict(metric=m, era=era, season=season, mean_d=v, midpoint=mid))
        rows.append(dict(metric=m, hypothesis=HYP[m], predicted_sign=sign, raw_diff=raw, T=T, ci_low=lo, ci_high=hi,
                         p=p, sign_ok=bool(np.sign(T) == sign), season_condition=ok,
                         ols_era_coef=ols.params[k], ols_ci_low=ci[0], ols_ci_high=ci[1], n_matches_used=len(use)))
    R = pd.DataFrame(rows)
    R["p_BH"] = stats.bh(R.p)
    R["supports"] = (R.p_BH < 0.05) & R.sign_ok
    R = R[["metric", "hypothesis", "predicted_sign", "raw_diff", "T", "ci_low", "ci_high", "p", "p_BH", "sign_ok",
           "supports", "season_condition", "ols_era_coef", "ols_ci_low", "ols_ci_high", "n_matches_used"]]
    S = pd.DataFrame(season_rows)
    R.to_csv(TAB / "o1_results.csv", index=False); S.to_csv(TAB / "o1_season_means.csv", index=False)
    dec = stats.decide(dict(zip(R.metric, R.supports)), dict(zip(R.metric, R.season_condition)))

    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    print(R.round(4).to_string(index=False)); print(); print(S.round(4).to_string(index=False))
    print(f"\nH1 (M2 and M3): {dec['H1_text']}\nH2 (>=3 of M1,M2,M4,M5): {dec['H2_text']}")

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(R))[::-1]
    for yi, r in zip(y, R.itertuples()):
        c = "#2a7f3f" if r.supports else "#555555"
        ax.errorbar(r.T, yi, xerr=[[r.T - r.ci_low], [r.ci_high - r.T]], fmt="o", color=c, capsize=4, lw=2)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([NAMES[m] + "\n(predicted: " + ("higher" if stats.PRED_SIGN[m] > 0 else "lower") + " in LE)" for m in R.metric], fontsize=9)
    ax.set_xlabel("T: Luis Enrique minus Guardiola, in Barca minus opponent (95% CI)")
    ax.set_title("Era effect per metric (green = BH p < 0.05 in predicted direction; season condition not shown)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "o1_forest.png", dpi=200); plt.close(fig)

    seasons = sorted(S.season.unique())
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.6))
    for ax, m in zip(axes, R.metric):
        s = S[S.metric == m].set_index("season")
        ax.bar(range(len(seasons)), [s.mean_d[x] for x in seasons],
               color=[COL[s.era[x]] for x in seasons])
        ax.axhline(s.midpoint.iloc[0], color="k", ls="--", lw=1)
        ax.set_xticks(range(len(seasons))); ax.set_xticklabels([x[2:4] + "/" + x[7:] for x in seasons], rotation=60, fontsize=8)
        ax.set_title(NAMES[m], fontsize=9)
    axes[0].set_ylabel("season mean of Barca - opponent")
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in COL.values()], labels=list(COL), fontsize=8)
    fig.suptitle("Season means of d; dashed = midpoint of the two era means", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "o1_season_means.png", dpi=200); plt.close(fig)
    print(f"\nruntime {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
