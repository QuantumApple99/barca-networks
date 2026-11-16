"""Stratified era comparison: estimate, permutation test, bootstrap, BH, season condition, decision rule."""
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

PRED_SIGN = {"M1": 1, "M2": 1, "M3": -1, "M4": 1, "M5": 1}
ERA_PEP, ERA_LE = "Guardiola", "Luis Enrique"


def _prep(d, era, strata):
    d, era, strata = np.asarray(d, float), np.asarray(era), np.asarray(strata)
    ok = ~np.isnan(d)
    d, era, strata = d[ok], era[ok], strata[ok]
    is_le = era == ERA_LE
    codes = pd.factorize(strata)[0]
    return d, is_le, codes


def _T_batch(d, is_le_b, codes, k):
    """d: (n,), is_le_b: (B, n) bool, codes: (n,). Returns T for each of B label vectors."""
    num = np.zeros(is_le_b.shape[0]); den = 0.0
    out_num = np.zeros(is_le_b.shape[0]); out_den = np.zeros(is_le_b.shape[0])
    for s in range(k):
        m = codes == s
        le = is_le_b[:, m]
        dd = d[m]
        n_le = le.sum(1); n_pe = m.sum() - n_le
        ok = (n_le > 0) & (n_pe > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_le = (le * dd).sum(1) / n_le
            mean_pe = ((~le) * dd).sum(1) / n_pe
            w = n_pe * n_le / (n_pe + n_le)
        out_num += np.where(ok, w * (mean_le - mean_pe), 0.0)
        out_den += np.where(ok, w, 0.0)
    return out_num / out_den


def stratified_estimate(d, era, strata):
    """T = sum_s w_s (mean_LE,s - mean_Pep,s) / sum_s w_s, w_s = nP*nL/(nP+nL). NaN d dropped."""
    d, is_le, codes = _prep(d, era, strata)
    return float(_T_batch(d, is_le[None, :], codes, codes.max() + 1)[0])


def raw_difference(d, era):
    d, era = np.asarray(d, float), np.asarray(era)
    ok = ~np.isnan(d)
    return float(d[ok & (era == ERA_LE)].mean() - d[ok & (era == ERA_PEP)].mean())


def stratified_permutation_test(d, era, strata, n_perm=10000, seed=671, chunk=2000):
    """Shuffle era labels within strata; two-sided p = (1 + #{|T*|>=|T_obs|})/(n_perm+1)."""
    d, is_le, codes = _prep(d, era, strata)
    k = codes.max() + 1
    rng = np.random.default_rng(seed)
    t_obs = _T_batch(d, is_le[None, :], codes, k)[0]
    idx = [np.flatnonzero(codes == s) for s in range(k)]
    ts = []
    done = 0
    while done < n_perm:
        B = min(chunk, n_perm - done)
        lab = np.tile(is_le, (B, 1))
        for ix in idx:
            sub = lab[:, ix]
            lab[:, ix] = rng.permuted(sub, axis=1)
        ts.append(_T_batch(d, lab, codes, k))
        done += B
    ts = np.concatenate(ts)
    p = (1 + np.sum(np.abs(ts) >= abs(t_obs) - 1e-12)) / (n_perm + 1)
    return float(t_obs), float(p)


def stratified_bootstrap_ci(d, era, strata, n_boot=10000, seed=671, level=0.95, chunk=1000):
    """Resample within era x stratum cell with replacement; percentile CI of T."""
    d, is_le, codes = _prep(d, era, strata)
    k = codes.max() + 1
    cells = [np.flatnonzero((codes == s) & (is_le == e)) for s in range(k) for e in (False, True)]
    cells = [c for c in cells if len(c)]
    rng = np.random.default_rng(seed)
    ts = []
    done = 0
    while done < n_boot:
        B = min(chunk, n_boot - done)
        ix = np.empty((B, len(d)), int)
        pos = 0
        for c in cells:
            ix[:, pos:pos + len(c)] = c[rng.integers(0, len(c), size=(B, len(c)))]
            pos += len(c)
        order = np.concatenate(cells)
        db = d[ix]; lb = is_le[order][None, :].repeat(B, 0); cb = codes[order]
        # per-replicate data differ, so compute T with per-row d
        num = np.zeros(B); den = np.zeros(B)
        for s in range(k):
            m = cb == s
            le = lb[:, m]; dd = db[:, m]
            n_le = le.sum(1); n_pe = m.sum() - n_le
            with np.errstate(invalid="ignore", divide="ignore"):
                ml = (le * dd).sum(1) / n_le; mp = ((~le) * dd).sum(1) / n_pe
                w = n_pe * n_le / (n_pe + n_le)
            ok = (n_le > 0) & (n_pe > 0)
            num += np.where(ok, w * (ml - mp), 0.0); den += np.where(ok, w, 0.0)
        ts.append(num / den)
        done += B
    ts = np.concatenate(ts)
    a = (1 - level) / 2
    lo, hi = np.quantile(ts, [a, 1 - a])
    return float(lo), float(hi)


def bh(pvals):
    """Benjamini-Hochberg adjusted p-values (statsmodels fdr_bh)."""
    return multipletests(np.asarray(pvals, float), method="fdr_bh")[1]


def season_condition(d, era, season, sign):
    """D32. Returns (holds, season_means Series, midpoint). midpoint = (raw Pep mean + raw LE mean)/2 of d.
    LE seasons must lie on the predicted side (sign * (mean - mid) > 0), Pep seasons on the other side."""
    df = pd.DataFrame({"d": np.asarray(d, float), "era": np.asarray(era), "season": np.asarray(season)}).dropna(subset=["d"])
    mid = (df[df.era == ERA_PEP].d.mean() + df[df.era == ERA_LE].d.mean()) / 2
    sm = df.groupby(["era", "season"]).d.mean()
    ok = True
    for (e, s), v in sm.items():
        side = sign * (v - mid)
        ok &= side > 0 if e == ERA_LE else side < 0
    return bool(ok), sm, float(mid)


def decide(supports, season_ok):
    """supports, season_ok: dicts metric -> bool. Returns dict with H1, H2 booleans and verdict strings."""
    good = {m: bool(supports[m] and season_ok[m]) for m in supports}
    h1 = good["M2"] and good["M3"]
    h2 = sum(good[m] for m in ("M1", "M2", "M4", "M5")) >= 3
    return {"H1": h1, "H2": h2,
            "H1_text": "supported" if h1 else "not supported",
            "H2_text": "supported" if h2 else "not supported"}
