"""Week 3 / O2: player-season features, role clustering, stability, validity, era role mix (D33-D40).

Design: per (player, match) sufficient statistics are computed once (`match_stats`). A player-season's
features are a function of a multiset of its matches, so a bootstrap replicate only re-aggregates rows
(`features_from_weights`) with integer multiplicities.
"""
import itertools

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import io, networks
from .metrics import GOAL, X_ELIGIBLE

MIN_MINUTES = 900.0
X_CUTS = (40.0, 80.0)
Y_CUTS = (80.0 / 3, 160.0 / 3)
CELLS = [f"{c}{r}" for r in range(3) for c in range(3)]          # cell id = col + 3*row (col = x band)
MADE = [f"made_{c}" for c in CELLS]
RECV = [f"recv_{c}" for c in CELLS]
CENTRALITY = ["pagerank_n", "betweenness"]
VOLUME = ["passes_made_p90", "passes_recv_p90"]
STYLE = ["progressive_share", "mean_pass_length", "forward_share"]
FEATURES = CENTRALITY + VOLUME + MADE + RECV + STYLE
SPATIAL = MADE + RECV
assert len(FEATURES) == 25 and len(SPATIAL) == 18
PS_KEY = ["season", "player_id"]


# ------------------------------------------------------------------ per (player, match) statistics
def cell_index(x, y):
    """3x3 cell id (0-8) = x band + 3 * y band; cuts at 40/80 and 80/3, 160/3 (values on a cut go up)."""
    xb = (np.asarray(x) >= X_CUTS[0]).astype(int) + (np.asarray(x) >= X_CUTS[1]).astype(int)
    yb = (np.asarray(y) >= Y_CUTS[0]).astype(int) + (np.asarray(y) >= Y_CUTS[1]).astype(int)
    return xb + 3 * yb


def pass_features(p):
    """Per-pass flags: start cell, end cell, eligible, progressive, length, forward."""
    sx, sy, ex, ey = (p[c].to_numpy(float) for c in ("x", "y", "end_x", "end_y"))
    ds = np.hypot(GOAL[0] - sx, GOAL[1] - sy)
    de = np.hypot(GOAL[0] - ex, GOAL[1] - ey)
    elig = sx >= X_ELIGIBLE
    return pd.DataFrame(dict(cell_start=cell_index(sx, sy), cell_end=cell_index(ex, ey),
                             elig=elig.astype(int), prog=(elig & (de <= 0.75 * ds)).astype(int),
                             length=np.hypot(ex - sx, ey - sy), fwd=(ex > sx).astype(int)), index=p.index)


def centrality_by_match(passes, appearances, match_id, team_id=io.BARCA_ID):
    """(PageRank x n_nodes, normalized directed betweenness with distance 1/passes) per node (D33)."""
    G = networks.build_full_network(passes, appearances, match_id, team_id)
    n = G.number_of_nodes()
    pr = nx.pagerank(G, alpha=0.85, weight="weight")
    H = nx.DiGraph()
    H.add_nodes_from(G.nodes)
    for a, b, d in G.edges(data=True):
        H.add_edge(a, b, dist=1.0 / d["weight"])
    bc = nx.betweenness_centrality(H, weight="dist", normalized=True)
    return {v: (pr[v] * n, bc[v]) for v in G.nodes}


def match_stats(passes=None, appearances=None, matches=None):
    """One row per (player, match) for Barca: minutes, centrality, and pass-count sufficient statistics."""
    L = lambda n: pd.read_parquet(io.PROCESSED / f"{n}.parquet")  # noqa: E731
    passes = L("passes") if passes is None else passes
    appearances = L("appearances") if appearances is None else appearances
    matches = L("matches") if matches is None else matches
    B = io.BARCA_ID
    ap = appearances[appearances.team_id == B].merge(matches[["match_id", "season", "era"]], on="match_id")
    p = passes[(passes.team_id == B) & passes.completed].copy()
    p = p.join(pass_features(p))
    p = p[p.player_id.notna() & p.recipient_id.notna()]
    p["player_id"] = p.player_id.astype(int)
    p["recipient_id"] = p.recipient_id.astype(int)
    k = ["match_id", "player_id"]
    out = ap[["match_id", "season", "era", "player_id", "player", "position_group", "minutes_played"]].copy()
    out = out.set_index(k)
    # passes made: counts per start cell, totals, progressive stats
    made = p.groupby(k + ["cell_start"]).size().unstack(fill_value=0).reindex(columns=range(9), fill_value=0)
    made.columns = MADE
    agg = p.groupby(k).agg(n_made=("length", "size"), n_elig=("elig", "sum"), n_prog=("prog", "sum"),
                           len_sum=("length", "sum"), n_fwd=("fwd", "sum"))
    r = p.rename(columns={"recipient_id": "rid"}).groupby(["match_id", "rid", "cell_end"]).size()
    recv = r.unstack(fill_value=0).reindex(columns=range(9), fill_value=0)
    recv.columns = RECV
    recv.index = recv.index.set_names(k)
    out = out.join(made).join(agg).join(recv)
    cols = MADE + RECV + ["n_made", "n_elig", "n_prog", "len_sum", "n_fwd"]
    out[cols] = out[cols].fillna(0)
    out["n_recv"] = out[RECV].sum(axis=1)
    cen = {}
    for mid, g in ap.groupby("match_id"):
        for pid, (pr, bc) in centrality_by_match(passes, appearances, mid).items():
            cen[(mid, pid)] = (pr, bc)
    c = pd.DataFrame.from_dict(cen, orient="index", columns=CENTRALITY)
    c.index = pd.MultiIndex.from_tuples(c.index, names=k)
    out = out.join(c).reset_index()
    return out


# ------------------------------------------------------------------ aggregation to player-seasons
def features_from_weights(ms, w):
    """Player-season features from per-(player, match) rows `ms` with multiplicities `w` (same length).

    Rows with w == 0 drop out. Returns DataFrame indexed by (season, player_id) with minutes and 25 features."""
    d = ms.copy()
    d["w"] = np.asarray(w, float)
    d = d[d.w > 0]
    num = ["n_made", "n_recv", "n_elig", "n_prog", "len_sum", "n_fwd"] + MADE + RECV
    S = d[num].mul(d.w, axis=0)
    S["minutes"] = d.minutes_played * d.w
    S["pr_m"] = d.pagerank_n * d.minutes_played * d.w
    S["bc_m"] = d.betweenness * d.minutes_played * d.w
    g = S.groupby([d.season, d.player_id]).sum()
    F = pd.DataFrame(index=g.index)
    F["minutes"] = g.minutes
    F["pagerank_n"] = g.pr_m / g.minutes
    F["betweenness"] = g.bc_m / g.minutes
    F["passes_made_p90"] = g.n_made / g.minutes * 90
    F["passes_recv_p90"] = g.n_recv / g.minutes * 90
    for c in MADE:
        F[c] = g[c] / g.n_made
    for c in RECV:
        F[c] = g[c] / g.n_recv
    F["progressive_share"] = np.where(g.n_elig > 0, g.n_prog / g.n_elig.where(g.n_elig > 0, 1), 0.0)
    F["mean_pass_length"] = g.len_sum / g.n_made
    F["forward_share"] = g.n_fwd / g.n_made
    return F


def unit_table(ms):
    """Player-season rows with >= 900 minutes, non-GK, plus modal position group, era, name."""
    pos = ms.groupby(PS_KEY + ["position_group"]).minutes_played.sum().reset_index()
    pos = pos.sort_values(PS_KEY + ["minutes_played", "position_group"], ascending=[True, True, False, True])
    modal = pos.drop_duplicates(PS_KEY).set_index(PS_KEY).position_group
    mins = ms.groupby(PS_KEY).minutes_played.sum()
    meta = ms.drop_duplicates(PS_KEY).set_index(PS_KEY)[["era", "player"]]
    U = meta.join(mins.rename("minutes")).join(modal)
    U = U[(U.minutes >= MIN_MINUTES) & (U.position_group != "GK")]
    return U


def build_player_season_features(match_subset=None, ms=None, units=None):
    """Features for the unit rows. match_subset: optional {(season, player_id): list of match_ids (repeats
    allowed)}; players not in it use all their matches. Returns DataFrame (one row per unit)."""
    ms = match_stats() if ms is None else ms
    units = unit_table(ms) if units is None else units
    ms = ms.merge(units.reset_index()[PS_KEY], on=PS_KEY)
    if match_subset is None:
        w = np.ones(len(ms))
    else:
        counts = {k: pd.Series(v).value_counts() for k, v in match_subset.items()}
        w = np.array([counts[(s, p)].get(m, 0) if (s, p) in counts else 1
                      for s, p, m in zip(ms.season, ms.player_id, ms.match_id)], float)
    F = features_from_weights(ms, w)
    out = units.drop(columns="minutes").join(F, how="inner")   # minutes = resampled minutes if subset given
    return out.reset_index()


# ------------------------------------------------------------------ clustering
def n_components_80(X, thr=0.80):
    ev = PCA().fit(X).explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ev), thr - 1e-12) + 1), ev


def fit_roles(X, k_range=(3, 4, 5, 6), seed=671, n_init=50, k=None):
    """z-score -> PCA (fewest comps with >= 80% cumulative variance) -> KMeans per k.
    If k is given only that k is fitted. Returns dict with labels (0-based raw), table, pca scores."""
    Z = StandardScaler().fit_transform(X)
    nc, ev = n_components_80(Z)
    pca = PCA(n_components=nc).fit(Z)
    T = pca.transform(Z)
    rows, fits = [], {}
    for kk in ([k] if k is not None else list(k_range)):
        km = KMeans(n_clusters=kk, n_init=n_init, random_state=seed).fit(T)
        sil = silhouette_score(T, km.labels_)
        rows.append(dict(k=kk, silhouette=sil, inertia=km.inertia_))
        fits[kk] = km.labels_
    tab = pd.DataFrame(rows)
    best = tab.k.iloc[0]
    bs = tab.silhouette.iloc[0]
    for r in tab.itertuples():          # strict > : ties go to the smaller k
        if r.silhouette > bs + 1e-12:
            best, bs = r.k, r.silhouette
    return dict(k=int(best), labels=fits[int(best)], table=tab, scores=T, n_components=nc, evr=ev,
                pca=pca, Z=Z)


def order_by_size(labels):
    """Relabel clusters 1..k by decreasing size (ties: lower raw label first)."""
    vc = pd.Series(labels).value_counts()
    order = sorted(vc.index, key=lambda c: (-vc[c], c))
    m = {c: i + 1 for i, c in enumerate(order)}
    return np.array([m[c] for c in labels])


# ------------------------------------------------------------------ stability
def resample_matches(ms, units, rng):
    """Multiplicity vector over `ms` rows (ms already restricted to unit rows): each player-season
    resamples its own matches with replacement, same count."""
    w = np.zeros(len(ms))
    for idx in ms.groupby(PS_KEY).indices.values():
        draw = rng.integers(0, len(idx), len(idx))
        w[idx] += np.bincount(draw, minlength=len(idx))
    return w


_G = {}


def _init_worker(g):
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    _G.update(g)


def _boot_worker(w):
    ms, cols, k, base_labels, order, seed, n_init = (_G[x] for x in ("ms", "cols", "k", "base", "order", "seed", "n_init"))
    F = features_from_weights(ms, w).loc[order]
    lab = fit_roles(F[cols].to_numpy(), seed=seed, n_init=n_init, k=k)["labels"]
    return adjusted_rand_score(base_labels, lab)


def stability(ms, units, cols, k, base_labels, n_boot=200, seed=671, n_init=50, processes=1):
    """Bootstrap ARI of refit clustering vs base labels. `units` index order == base_labels order.
    Returns (array of ARI, summary dict)."""
    import multiprocessing as mp
    ms = ms.merge(units.reset_index()[PS_KEY], on=PS_KEY).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    ws = [resample_matches(ms, units, rng) for _ in range(n_boot)]
    g = dict(ms=ms, cols=cols, k=k, base=np.asarray(base_labels), order=units.index, seed=seed, n_init=n_init)
    if processes == 1:
        _G.update(g)
        aris = [_boot_worker(w) for w in ws]
    else:   # spawn (fork after OpenMP use in the parent can deadlock); one thread per worker
        with mp.get_context("spawn").Pool(processes, initializer=_init_worker, initargs=(g,)) as pool:
            aris = pool.map(_boot_worker, ws, chunksize=5)
    a = np.array(aris)
    return a, dict(median=float(np.median(a)), p05=float(np.percentile(a, 5)), p95=float(np.percentile(a, 95)),
                   mean=float(a.mean()), n_boot=n_boot, n_init=n_init)


# ------------------------------------------------------------------ validity vs position groups
def position_validity(labels, positions, min_group=6, split_frac=0.30):
    labels, positions = np.asarray(labels), np.asarray(positions)
    ari = adjusted_rand_score(positions, labels)
    cross = pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(positions, name="position_group"))
    rows = []
    for g in cross.columns:
        n = int(cross[g].sum())
        top = int(cross[g].max())
        out = 1 - top / n
        rows.append(dict(position_group=g, n=n, most_common_cluster=cross[g].idxmax(), share_outside=out,
                         eligible=n >= min_group, split=bool(n >= min_group and out >= split_frac - 1e-12)))
    return dict(ari=float(ari), crosstab=cross, groups=pd.DataFrame(rows))


# ------------------------------------------------------------------ era role mix
def tv_distance(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def era_role_mix_test(labels, minutes, seasons, eras, k=None):
    """Exact permutation over all C(7,4)=35 assignments of seasons to a 4-season 'Pep' group (D39).

    Returns dict(observed_tv, p, tvs (35 values, one per labeling), observed shares per era, labelings)."""
    labels, minutes = np.asarray(labels), np.asarray(minutes, float)
    seasons, eras = np.asarray(seasons), np.asarray(eras)
    cl = sorted(set(labels)) if k is None else list(range(1, k + 1))
    ss = sorted(set(seasons))
    sm = {(s, c): minutes[(seasons == s) & (labels == c)].sum() for s in ss for c in cl}

    def shares(group):
        v = np.array([sum(sm[(s, c)] for s in group) for c in cl])
        return v / v.sum()

    obs_pep = tuple(sorted(set(seasons[eras == "Guardiola"])))
    assert len(obs_pep) == 4 and len(ss) == 7
    labelings = list(itertools.combinations(ss, 4))
    tvs = []
    for pep in labelings:
        le = [s for s in ss if s not in pep]
        tvs.append(tv_distance(shares(pep), shares(le)))
    obs_le = [s for s in ss if s not in obs_pep]
    obs = tv_distance(shares(obs_pep), shares(obs_le))
    p = float(sum(t >= obs - 1e-12 for t in tvs) / len(tvs))
    return dict(observed_tv=obs, p=p, tvs=np.array(tvs), labelings=labelings, clusters=cl,
                shares_pep=shares(obs_pep), shares_le=shares(obs_le), n_labelings=len(labelings))


# ------------------------------------------------------------------ whole analysis for a feature set
BASE_N_INIT = 1000  # D41: the official fit is the lowest-inertia solution over 1,000 starts


def run_analysis(ms, units, U, cols, n_boot=200, n_init_boot=50, seed=671, processes=1):
    """U: unit rows with features (same order as units). Returns dict with all O2 results."""
    X = U[cols].to_numpy()
    fit = fit_roles(X, seed=seed, n_init=BASE_N_INIT)
    labels = order_by_size(fit["labels"])
    ari, summ = stability(ms, units, cols, fit["k"], fit["labels"], n_boot=n_boot, seed=seed,
                          n_init=n_init_boot, processes=processes)
    val = position_validity(labels, U.position_group)
    mix = era_role_mix_test(labels, U.minutes, U.season, U.era)
    return dict(fit=fit, labels=labels, stab=summ, boot_ari=ari, validity=val, mix=mix, cols=cols)
