"""Stage 4: O2 player roles (D33-D40). Run: python notebooks/03_O2_roles.py (needs stage 1 processed data)."""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from barca import io, roles  # noqa: E402

TAB, FIG = ROOT / "report" / "tables", ROOT / "report" / "figures"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
N_BOOT, N_INIT_BOOT = 200, 50
SHORT = {"Xavier Hernández Creus": "Xavi", "Andrés Iniesta Luján": "Iniesta", "Lionel Andrés Messi Cuccittini": "Messi",
         "Sergio Busquets i Burgos": "Busquets", "Carles Puyol i Saforcada": "Puyol", "Gerard Piqué Bernabéu": "Piqué",
         "Daniel Alves da Silva": "Alves", "David Villa Sánchez": "Villa", "Pedro Eliezer Rodríguez Ledesma": "Pedro",
         "Thierry Henry": "Henry", "Samuel Eto''o Fils": "Eto'o", "Zlatan Ibrahimović": "Ibrahimović",
         "Gnégnéri Yaya Touré": "Yaya Touré", "Eric-Sylvain Bilal Abidal": "Abidal", "Rafael Márquez Álvarez": "Márquez",
         "Javier Alejandro Mascherano": "Mascherano", "Ivan Rakitić": "Rakitić", "Luis Alberto Suárez Díaz": "Suárez",
         "Neymar da Silva Santos Junior": "Neymar", "Jordi Alba Ramos": "Alba", "Francesc Fàbregas i Soler": "Fàbregas",
         "Seydou Kéita": "Keita", "Sergi Roberto Carnicer": "Sergi Roberto", "Thiago Alcântara do Nascimento": "Thiago",
         "Maxwell Scherrer Cabelino Andrade": "Maxwell", "Bojan Krkíc Pérez": "Bojan", "Jérémy Mathieu": "Mathieu",
         "Samuel Yves Umtiti": "Umtiti", "Thomas Vermaelen": "Vermaelen", "Rafael Alcântara do Nascimento": "Rafinha",
         "Denis Suárez Fernández": "D. Suárez", "Adriano Correia Claro": "Adriano", "Arda Turan": "Arda",
         "Alexis Alejandro Sánchez Sánchez": "Alexis", "Marc Bartra Aregall": "Bartra", "Cristian Tello Herrera": "Tello",
         "Ibrahim Afellay": "Afellay", "Martín Montoya Torralbo": "Montoya", "Dmytro Chygrynskiy": "Chygrynskiy",
         "Aliaksandr Hleb": "Hleb", "Gabriel Alejandro Milito": "Milito", "Aleix Vidal Parreu": "A. Vidal",
         "André Filipe Tavares Gomes": "André Gomes", "Munir El Haddadi Mohamed": "Munir", "Jeffren Isaac Suárez Bermúdez": "Jeffren"}


def short(name):
    return SHORT.get(name, name.split()[0])


def label(r):
    return f"{short(r.player)} {r.season[2:4]}/{r.season[7:9]}"


def season_short(s):
    return f"{s[2:4]}/{s[7:9]}"


def cluster_profiles(U, cols, labels, k):
    Z = pd.DataFrame((U[cols] - U[cols].mean()) / U[cols].std(ddof=0), index=U.index)
    rows = []
    for c in range(1, k + 1):
        m = labels == c
        z = Z[m].mean()
        sub = U[m]
        em = sub.groupby("era").minutes.sum()
        pc = sub.position_group.value_counts()
        top = sub.sort_values("minutes", ascending=False).head(8)
        r = dict(cluster=f"Role {c}", size=int(m.sum()), total_minutes=float(sub.minutes.sum()))
        r.update({f"z_{f}": z[f] for f in cols})
        r["position_counts"] = "; ".join(f"{g} {n}" for g, n in pc.items())
        r["minutes_share_Guardiola_within_cluster"] = em.get("Guardiola", 0) / em.sum()
        r["minutes_share_Luis Enrique_within_cluster"] = em.get("Luis Enrique", 0) / em.sum()
        zs = z.sort_values(ascending=False)
        r["top3_z"] = "; ".join(f"{f} {v:+.2f}" for f, v in zs.head(3).items())
        r["bottom3_z"] = "; ".join(f"{f} {v:+.2f}" for f, v in zs.tail(3)[::-1].items())
        r["members_top8_minutes"] = "; ".join(f"{label(t)} ({t.position_group}, {t.minutes:.0f}')" for t in top.itertuples())
        rows.append(r)
    return pd.DataFrame(rows)


def write_run(tag, cols, res, U, suffix):
    """Write the CSVs for one run (suffix '' for the main run)."""
    fit, labels, k = res["fit"], res["labels"], res["fit"]["k"]
    kt = fit["table"].copy(); kt["chosen"] = kt.k == k
    kt.to_csv(TAB / f"o2_k_selection{suffix}.csv", index=False)
    cluster_profiles(U, cols, labels, k).to_csv(TAB / f"o2_cluster_profiles{suffix}.csv", index=False)
    res["validity"]["crosstab"].rename(index=lambda c: f"Role {c}").to_csv(TAB / f"o2_crosstab_positions{suffix}.csv")
    mix, rows = res["mix"], []
    for c, a, b in zip(mix["clusters"], mix["shares_pep"], mix["shares_le"]):
        rows.append(dict(cluster=f"Role {c}", share_Guardiola=a, share_Luis_Enrique=b, diff_LE_minus_Pep=b - a))
    rows.append(dict(cluster="TV distance", share_Guardiola=mix["observed_tv"], share_Luis_Enrique=np.nan,
                     diff_LE_minus_Pep=np.nan))
    rows.append(dict(cluster="permutation p (35 labelings)", share_Guardiola=mix["p"], share_Luis_Enrique=np.nan,
                     diff_LE_minus_Pep=np.nan))
    pd.DataFrame(rows).to_csv(TAB / f"o2_role_mix{suffix}.csv", index=False)
    pd.DataFrame(dict(labeling_pep_seasons=[", ".join(season_short(s) for s in l) for l in mix["labelings"]],
                      tv=mix["tvs"])).sort_values("tv", ascending=False).to_csv(TAB / f"o2_role_mix_perm{suffix}.csv", index=False)
    pd.DataFrame(dict(ari=res["boot_ari"])).to_csv(TAB / f"o2_boot_ari{suffix}.csv", index=False)


def summary(tag, res):
    fit, st, val, mix = res["fit"], res["stab"], res["validity"], res["mix"]
    sil = float(fit["table"].set_index("k").silhouette[fit["k"]])
    split = val["groups"][val["groups"].split].position_group.tolist()
    R = [("run", tag, "", ""),
         ("n_pca_components", fit["n_components"], "", ""),
         ("chosen_k", fit["k"], "3 <= k <= 6", "PASS" if 3 <= fit["k"] <= 6 else "FAIL"),
         ("silhouette", sil, ">= 0.25", "PASS" if sil >= 0.25 else "FAIL"),
         ("stability_median_ARI", st["median"], ">= 0.7", "PASS" if st["median"] >= 0.7 else "FAIL"),
         ("stability_p05_ARI", st["p05"], "", ""), ("stability_p95_ARI", st["p95"], "", ""),
         ("bootstrap_n_boot", st["n_boot"], "", ""), ("bootstrap_n_init", st["n_init"], "", ""),
         ("ARI_vs_positions", val["ari"], "> 0.3", "PASS" if val["ari"] > 0.3 else "FAIL"),
         ("n_split_position_groups", len(split), ">= 1", "PASS" if len(split) >= 1 else "FAIL"),
         ("split_groups", ", ".join(split), "", ""),
         ("era_mix_TV", mix["observed_tv"], "", ""),
         ("era_mix_p", mix["p"], "descriptive (min 1/35)", "")]
    return pd.DataFrame(R, columns=["criterion", "value", "threshold", "result"])


def plots(U, res, cols, suffix=""):
    fit, labels, k = res["fit"], res["labels"], res["fit"]["k"]
    cmap = plt.get_cmap("tab10")
    ev = fit["evr"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(1, len(ev) + 1), ev, color="#9aa5b1", label="per component")
    ax.plot(np.arange(1, len(ev) + 1), np.cumsum(ev), "o-", color="#1f4e9c", label="cumulative")
    ax.axhline(0.8, ls="--", color="k", lw=0.8); ax.axvline(fit["n_components"] + 0.5, ls=":", color="#d1495b")
    ax.set(xlabel="Principal component", ylabel="Explained variance ratio",
           title=f"PCA scree ({fit['n_components']} components kept for 80%)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / f"o2_scree{suffix}.png", dpi=200); plt.close(fig)

    t = fit["table"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(t.k, t.silhouette, "o-", color="#1f4e9c")
    ax.axhline(0.25, ls="--", color="k", lw=0.8); ax.scatter([k], [t.set_index("k").silhouette[k]], s=140, facecolors="none", edgecolors="#d1495b", lw=2)
    ax.set(xlabel="k", ylabel="Silhouette (PCA space)", title="Silhouette by k (dashed: 0.25)", xticks=t.k)
    fig.tight_layout(); fig.savefig(FIG / f"o2_silhouette_k{suffix}.png", dpi=200); plt.close(fig)

    T = fit["scores"]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    mk = {"Guardiola": "o", "Luis Enrique": "^"}
    for era, m in mk.items():
        for c in range(1, k + 1):
            s = (U.era.to_numpy() == era) & (labels == c)
            ax.scatter(T[s, 0], T[s, 1], marker=m, color=cmap(c - 1), s=45, alpha=0.85, edgecolor="k", lw=0.4)
    handles = [plt.Line2D([], [], marker="s", ls="", color=cmap(c - 1), label=f"Role {c} (n={int((labels == c).sum())})") for c in range(1, k + 1)]
    handles += [plt.Line2D([], [], marker=m, ls="", color="gray", label=e) for e, m in mk.items()]
    ax.legend(handles=handles, fontsize=8)
    pick = []
    for c in range(1, k + 1):
        sub = U[labels == c].sort_values("minutes", ascending=False)
        seen = set()
        for i, r in zip(sub.index, sub.itertuples()):
            if r.player not in seen and len(seen) < 3:
                seen.add(r.player); pick.append(i)
    pick = sorted(pick, key=lambda i: -U.minutes[i])[:12]
    for i in pick:
        ax.annotate(label(U.loc[i]), (T[i, 0], T[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel=f"PC1 ({fit['evr'][0]:.0%})", ylabel=f"PC2 ({fit['evr'][1]:.0%})", title=f"Barça player-season roles (k={k})")
    fig.tight_layout(); fig.savefig(FIG / f"o2_pca_roles{suffix}.png", dpi=200); plt.close(fig)

    L = pd.DataFrame(fit["pca"].components_[:2].T, index=cols, columns=["PC1", "PC2"])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for ax, pc in zip(axs, ("PC1", "PC2")):
        s = L[pc].reindex(L[pc].abs().sort_values(ascending=False).index[:10])[::-1]
        ax.barh(s.index, s.values, color=["#1f4e9c" if v > 0 else "#d1495b" for v in s.values])
        ax.set(title=f"Top loadings, {pc}", xlabel="Loading (unit-norm eigenvector)")
    fig.tight_layout(); fig.savefig(FIG / f"o2_loadings{suffix}.png", dpi=200); plt.close(fig)
    L.to_csv(TAB / f"o2_loadings{suffix}.csv")


def main():
    t0 = time.time()
    ms = roles.match_stats()
    units = roles.unit_table(ms)
    U = roles.build_player_season_features(ms=ms, units=units)
    assert len(U) == 101 and U[roles.FEATURES].notna().all().all()
    U["position_group"] = units.position_group.to_numpy()
    U.to_parquet(io.PROCESSED / "player_season_features.parquet", index=False)
    t1 = time.time(); print(f"features: {t1 - t0:.1f}s")

    res = roles.run_analysis(ms, units, U, roles.FEATURES, n_boot=N_BOOT, n_init_boot=N_INIT_BOOT)
    t2 = time.time(); print(f"main run (k-select + {N_BOOT} bootstrap reps, n_init={N_INIT_BOOT}): {t2 - t1:.1f}s")
    O = U[["season", "era", "player_id", "player", "minutes", "position_group"] + roles.FEATURES].copy()
    O["cluster"] = [f"Role {c}" for c in res["labels"]]
    O["PC1"], O["PC2"] = res["fit"]["scores"][:, 0], res["fit"]["scores"][:, 1]
    O.to_parquet(io.PROCESSED / "o2_roles.parquet", index=False)
    write_run("main (25 features)", roles.FEATURES, res, U, "")
    S = summary("main (25 features)", res)
    plots(U, res, roles.FEATURES)
    runs = [S]
    sil = float(res["fit"]["table"].set_index("k").silhouette[res["fit"]["k"]])
    if sil < 0.25:                                   # D40 fallback
        res2 = roles.run_analysis(ms, units, U, roles.SPATIAL, n_boot=N_BOOT, n_init_boot=N_INIT_BOOT)
        write_run("fallback (18 spatial)", roles.SPATIAL, res2, U, "_fallback")
        runs.append(summary("fallback (18 spatial features, D40)", res2))
        O2 = O[["season", "era", "player_id", "player", "minutes", "position_group"]].copy()
        O2["cluster"] = [f"Role {c}" for c in res2["labels"]]
        O2["PC1"], O2["PC2"] = res2["fit"]["scores"][:, 0], res2["fit"]["scores"][:, 1]
        O2.to_parquet(io.PROCESSED / "o2_roles_fallback.parquet", index=False)
        plots(U, res2, roles.SPATIAL, "_fallback")
    pd.concat(runs).to_csv(TAB / "o2_summary.csv", index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 200)
    print(pd.concat(runs).to_string(index=False))
    print(f"total O2 stage: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
