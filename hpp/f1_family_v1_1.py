#!/usr/bin/env python3
"""F1 family and household architecture of the vascular beds (aggregates only).

Question: is each vascular bed shaped more by shared adult environment (cohabitants: no shared genes) or by shared
genes plus upbringing (siblings, parent-offspring)?  Does the answer differ between leg PWV, carotid IMT, retina,
ABI, CAVI0 and ECG voltage?

Design
  * Participants: the atlas frame (`--atlas`, calendar-adjusted beds), one row per participant and variable
    (first main visit with the value).  Every variable is residualised on age, age^2, sex and age x sex and
    standardised (z).
  * Pairs: `participant_relation` (coding 045_01): 9 = shared living space (cohabitants); 3/4/5/8 = siblings
    (full, non-twin, non-identical twin, multiple birth); 6/7 = half-siblings; 1/2 = parent/child.  A pair that is
    both kin and cohabiting counts as kin.  "Partners" = cohabitants of opposite sex with |age difference| <= 15 y.
  * Pair concordance: double-entry intraclass correlation (Pearson on (z_i, z_j) and (z_j, z_i)); 95 % CI from a
    bootstrap over pairs.
  * Measurement-session control: unrelated, non-cohabiting "stranger" pairs measured on the same day (random
    disjoint pairing within each date) estimate the day/session artefact; strangers measured on different days
    are the null.  Cohabitant concordance is also split by same-day vs different-day measurement.
  * Positive controls: height (genetic; sibs >> cohabitants), BMI and BP (known spousal concordance).
  * Risk-factor-adjusted concordance: z additionally residualised on own BMI, SBP, antihypertensives, log TG,
    HbA1c and current smoking: how much of the household signal runs through shared classic risk factors.
  * Cross-bed, cross-person correlations (own bed A vs relative's bed B, oriented adverse) for cohabitants and
    siblings: is the shared factor general or bed-specific?
  * Raw vs calendar-adjusted PWV: shared visit dates make calendar drift look like household concordance.
  * v1.1 time-gap kernel: pair correlation by |dt| between the two acquisition timestamps (vas_ts; bp_ts for BP) for
    all same-day pairs of unrelated people, cohabitants and siblings, vs different days; CI from a bootstrap over
    measurement days. Strangers close in time = device/session artefact; cohabitants only = shared acute state.

Run (sandbox `pheno` env):
  python f1_family_v1_1.py --root <HPP_DATA_ROOT> \
      --out f1_family_out_v1_1 --atlas f1_atlas_v1_5.py
Privacy: counts rounded to 10, estimates need >= 50 pairs, no ids or dates leave memory.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ATLAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_atlas_v1_5.py")
if "--atlas" in sys.argv[:-1]:
    ATLAS_PATH = os.path.abspath(sys.argv[sys.argv.index("--atlas") + 1])
_spec = importlib.util.spec_from_file_location("f1_atlas", ATLAS_PATH)
A = importlib.util.module_from_spec(_spec)
sys.modules["f1_atlas"] = A
_spec.loader.exec_module(A)

VERSION = "f1_family_v1.1 2026-10-03"
MIN_PAIRS = 50
BEDS = ["pwv", "imt", "art_w", "abi_min", "cavi0", "sokolow", "vein_w", "fd"]
TRAITS = ["height", "bmi", "waist", "sbp_sit", "dbp_sit", "hr_sit", "log_tg", "ldl", "hdl", "hba1c", "log1p_ahi"]
RAW = ["pwv_raw", "imt_raw", "art_w_raw", "abi_min_raw"]
RF = ["bmi", "sbp_sit", "antihtn", "log_tg", "hba1c", "smk_current"]
KVARS = ["pwv", "cavi0", "imt", "abi_min", "art_w", "pwv_raw", "cavi0_raw", "imt_raw", "abi_min_raw", "art_w_raw",
         "sbp_sit", "dbp_sit", "hr_sit"]  # v1.1 time-gap kernel variables
BPV = ("sbp_sit", "dbp_sit", "hr_sit")  # timed by bp_ts; all others by vas_ts
KEDGES = (0, 15, 30, 60, 120, 240)  # |dt| bin edges (min), last bin open; "diff" = measured on different days
KLAB = ["0-15", "15-30", "30-60", "60-120", "120-240", "240+", "diff"]
N_DIFF = 20000  # different-day stranger pairs sampled as the null
KIN = {3: "sib", 4: "sib", 5: "sib", 8: "sib", 6: "halfsib", 7: "halfsib", 1: "po", 2: "po"}
TYPES = ["cohab", "partner", "sib", "halfsib", "po", "stranger_sameday", "stranger_diffday"]
log, rc = A.log, A.rc


def resid_z(d, y, covs):
    """Standardised residual of y on [1, covs] (rows with any missing -> NaN)."""
    cols = [y] + covs
    ok = d[cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    out = pd.Series(np.nan, index=d.index)
    if ok.sum() < 100:
        return out
    X = np.column_stack([np.ones(ok.sum())] + [d.loc[ok, c].astype(float).values for c in covs])
    yy = d.loc[ok, y].astype(float).values
    lo, hi = np.nanpercentile(yy, [0.5, 99.5])
    yy = np.clip(yy, lo, hi)
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    r = yy - X @ b
    out[ok] = (r - r.mean()) / r.std()
    return out


def person_table(fr, variables, adjust_rf):
    """One row per participant: z of every variable (first main visit with the value) and its visit date."""
    base = ["age", "sex_male"]
    rows = {}
    for v in variables:
        if v not in fr.columns:
            continue
        need = [v] + base + (RF if adjust_rf else [])
        if any(c not in fr.columns for c in need):
            continue
        d = A.one_row_per_participant(fr, need).copy()
        d["age2"] = (d["age"] - 50.0) ** 2
        d["age_sex"] = (d["age"] - 50.0) * d["sex_male"]
        z = resid_z(d, v, ["age", "age2", "sex_male", "age_sex"] + (RF if adjust_rf else []))
        rows[v] = pd.Series(z.values, index=d["participant_id"].astype(str).values)
        rows[v + "__date"] = pd.Series(d["visit_date"].values if "visit_date" in d else np.nan,
                                       index=d["participant_id"].astype(str).values)
    P = pd.DataFrame(rows)
    one = A.one_row_per_participant(fr, ["age", "sex_male"])
    P = P.join(pd.DataFrame({"age": one["age"].values, "sex_male": one["sex_male"].values},
                            index=one["participant_id"].astype(str).values), how="left")
    return P


def load_pairs(root):
    rel = A.rd(root, "participant_relation/participant_relation.parquet")
    rel = rel.reset_index()
    rel = rel[rel["relation_type"].notna() & rel["relative_participant_id"].notna()]
    a = rel["participant_id"].astype("int64").astype(str).values
    b = rel["relative_participant_id"].astype("int64").astype(str).values
    t = rel["relation_type"].astype(int).values
    types = {}
    for x, y, tt in zip(a, b, t):
        if x == y:
            continue
        k = (x, y) if x < y else (y, x)
        types.setdefault(k, set()).add(int(tt))
    out = []
    for (x, y), ts in types.items():
        kin = sorted({KIN[t_] for t_ in ts if t_ in KIN})
        if kin:
            cls = "sib" if "sib" in kin else kin[0]
        elif 9 in ts:
            cls = "cohab"
        else:
            continue
        out.append((x, y, cls))
    return pd.DataFrame(out, columns=["a", "b", "cls"])


def stranger_pairs(P, pairs, var, rng):
    """Disjoint random pairs of unrelated, non-cohabiting people measured on the same / on different days."""
    related = set(pairs["a"]) | set(pairs["b"])
    d = P.loc[P[var].notna() & ~P.index.isin(related), [var + "__date"]].dropna()
    if len(d) < 4 * MIN_PAIRS:
        return None, None
    d = d.iloc[rng.permutation(len(d))]
    same = []
    for _, g in d.groupby(var + "__date"):
        ids = list(g.index)
        same += [(ids[i], ids[i + 1]) for i in range(0, len(ids) - 1, 2)]
    ids = list(d.index)
    diff = []
    for i in range(0, len(ids) - 1, 2):
        if d.iloc[i, 0] != d.iloc[i + 1, 0]:
            diff.append((ids[i], ids[i + 1]))
    return (pd.DataFrame(same, columns=["a", "b"]) if same else None,
            pd.DataFrame(diff, columns=["a", "b"]) if diff else None)


def icc_pairs(x, y):
    xx, yy = np.concatenate([x, y]), np.concatenate([y, x])
    return float(np.corrcoef(xx, yy)[0, 1])


def pair_r(P, pr, va, vb, boot, rng):
    """Double-entry correlation of own va with relative's vb over pairs pr; bootstrap CI over pairs."""
    if pr is None or len(pr) == 0 or va not in P or vb not in P:
        return None
    x1 = P[va].reindex(pr["a"]).values
    y1 = P[vb].reindex(pr["b"]).values
    x2 = P[va].reindex(pr["b"]).values
    y2 = P[vb].reindex(pr["a"]).values
    ok = np.isfinite(x1) & np.isfinite(y1) & np.isfinite(x2) & np.isfinite(y2)
    n = int(ok.sum())
    if n < MIN_PAIRS:
        return {"n": rc(n)}
    x1, y1, x2, y2 = x1[ok], y1[ok], x2[ok], y2[ok]

    def est(ix):
        return float(np.corrcoef(np.concatenate([x1[ix], x2[ix]]), np.concatenate([y1[ix], y2[ix]]))[0, 1])
    r = est(np.arange(n))
    bs = np.array([est(rng.integers(0, n, n)) for _ in range(boot)])
    return {"n": rc(n), "r": r, "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)),
            "_bs": bs}


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")} if isinstance(d, dict) else d


def diff_r(a, b):
    """Independent-sample difference a - b from stored bootstrap draws."""
    if not a or not b or "_bs" not in a or "_bs" not in b:
        return None
    m = min(len(a["_bs"]), len(b["_bs"]))
    d = a["_bs"][:m] - b["_bs"][:m]
    return {"diff": a["r"] - b["r"], "lo": float(np.nanpercentile(d, 2.5)), "hi": float(np.nanpercentile(d, 97.5))}


def analyse(fr, pairs, boot, rng):
    res = {"pairs_n": {}, "concordance": {}, "sameday_split": {}, "rf_adjusted": {}, "cross_bed": {},
           "contrasts": {}, "raw_vs_adjusted": {}}
    allv = [v for v in BEDS + TRAITS + RAW if v in fr.columns]
    P = person_table(fr, allv, adjust_rf=False)
    Pa = person_table(fr, [b for b in BEDS if b in fr.columns], adjust_rf=True)
    # orientation for the cross-bed matrix (adverse = higher)
    for b in A.CORE:
        if b in P:
            P["o_" + b] = A.ORIENT.get(b, 1) * P[b]
    pairs = pairs[pairs["a"].isin(P.index) & pairs["b"].isin(P.index)].copy()
    sa, sb = P["sex_male"].reindex(pairs["a"]).values, P["sex_male"].reindex(pairs["b"]).values
    da = np.abs(P["age"].reindex(pairs["a"]).values - P["age"].reindex(pairs["b"]).values)
    groups = {c: pairs[pairs["cls"] == c][["a", "b"]] for c in ("cohab", "sib", "halfsib", "po")}
    groups["partner"] = pairs[(pairs["cls"] == "cohab") & (sa != sb) & (da <= 15)][["a", "b"]]
    res["pairs_n"] = {k: rc(len(v)) for k, v in groups.items()}
    if len(groups["partner"]) >= MIN_PAIRS:
        res["partner_age_diff"] = A.qtiles(pd.Series(da[(pairs["cls"] == "cohab").values & (sa != sb) & (da <= 15)]))
    keep = {}
    for v in allv:
        same, diff = stranger_pairs(P, pairs, v, rng)
        g = dict(groups, stranger_sameday=same, stranger_diffday=diff)
        row = {}
        for t in TYPES:
            row[t] = pair_r(P, g.get(t), v, v, boot, rng)
        keep[v] = row
        res["concordance"][v] = {t: strip(x) for t, x in row.items()}
        # household net of the measurement day; kin minus cohabitant
        res["contrasts"][v] = {"cohab_minus_stranger_sameday": diff_r(row["cohab"], row["stranger_sameday"]),
                               "sib_minus_cohab": diff_r(row["sib"], row["cohab"]),
                               "sib_minus_partner": diff_r(row["sib"], row["partner"])}
        # cohabitants measured on the same vs different days
        if v + "__date" in P and len(groups["cohab"]):
            c = groups["cohab"]
            d1 = P[v + "__date"].reindex(c["a"]).values
            d2 = P[v + "__date"].reindex(c["b"]).values
            same_mask = pd.notna(d1) & pd.notna(d2) & (d1 == d2)
            res["sameday_split"][v] = {"same_day": strip(pair_r(P, c[same_mask], v, v, boot, rng)),
                                       "different_day": strip(pair_r(P, c[~same_mask], v, v, boot, rng))}
    for v in [b for b in BEDS if b in Pa]:
        res["rf_adjusted"][v] = {t: strip(pair_r(Pa, groups[t], v, v, boot, rng)) for t in ("cohab", "partner", "sib")}
    oc = ["o_" + b for b in A.CORE if "o_" + b in P]
    for t in ("cohab", "sib"):
        res["cross_bed"][t] = {f"{a_[2:]}~{b_[2:]}": strip(pair_r(P, groups[t], a_, b_, max(200, boot // 5), rng))
                               for a_ in oc for b_ in oc}
    for b in A.CORE:
        if b + "_raw" in keep and b in keep:
            res["raw_vs_adjusted"][b] = {t: {"raw": strip(keep[b + "_raw"][t]), "adjusted": strip(keep[b][t])}
                                         for t in ("cohab", "stranger_sameday")}
    return res


def kernel_table(fr, v):
    """v1.1: one row per participant (first main visit with v): z as in person_table, timestamp (min), local day."""
    tc = "bp_ts" if v in BPV else "vas_ts"
    if v not in fr.columns or tc not in fr.columns:
        return None
    d = A.one_row_per_participant(fr, [v, "age", "sex_male"]).copy()
    d["age2"] = (d["age"] - 50.0) ** 2
    d["age_sex"] = (d["age"] - 50.0) * d["sex_male"]
    z = resid_z(d, v, ["age", "age2", "sex_male", "age_sex"])
    ts = pd.to_datetime(d[tc], errors="coerce", utc=True)
    k = pd.DataFrame({"z": z.values, "t": (ts - pd.Timestamp("2018-01-01", tz="UTC")).dt.total_seconds().values / 60,
                      "day": A.to_local(ts).dt.strftime("%Y-%m-%d").values}, index=d["participant_id"].astype(str).values)
    return k.dropna()


def kbins(z, t, day, I, J, B, rng):
    """v1.1: double-entry r of pairs (I, J) by |dt| bin (same day) or 'diff'; bootstrap over days (day of I)."""
    if len(I) == 0:
        return {}
    nb = len(KLAB)
    same = day[I] == day[J]
    bi = np.where(same, np.searchsorted(KEDGES, np.abs(t[I] - t[J]), side="right") - 1, nb - 1)
    _, cl = np.unique(day[I], return_inverse=True)
    D = cl.max() + 1
    x, y = z[I], z[J]
    S = np.stack([np.bincount(cl * nb + bi, weights=w, minlength=D * nb).reshape(D, nb)
                  for w in (np.ones(len(I)), x + y, x * x + y * y, x * y)], axis=-1)
    W = np.bincount((rng.integers(0, D, (B, D)) + np.arange(B)[:, None] * D).ravel(), minlength=B * D).reshape(B, D)
    T = (W @ S.reshape(D, -1)).reshape(B, nb, 4)

    def r(T):
        with np.errstate(all="ignore"):
            n = T[..., 0]
            m = T[..., 1] / (2 * n)
            return (T[..., 3] / n - m * m) / (T[..., 2] / (2 * n) - m * m)
    est, bs, n = r(S.sum(0)), r(T), S.sum(0)[:, 0]
    rcp = (lambda x: rc(x) if x < 10 ** 6 else ">=1e6")  # no 7-digit run in the outputs
    return {lab: ({"n": rcp(n[k]), "r": float(est[k]), "lo": float(np.nanpercentile(bs[:, k], 2.5)),
                   "hi": float(np.nanpercentile(bs[:, k], 97.5))} if n[k] >= MIN_PAIRS else {"n": rcp(n[k])})
            for k, lab in enumerate(KLAB)}


def time_kernel(fr, pairs, boot, rng):
    """v1.1: residual pair correlation by the time gap between the two acquisitions. Strangers = all unordered same-day
    pairs in different relation components (plus N_DIFF random different-day pairs); cohabitants and siblings as in
    load_pairs."""
    par = {}

    def find(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for x, y in zip(pairs["a"], pairs["b"]):
        rx, ry = find(x), find(y)
        if rx != ry:
            par[rx] = ry
    out, B = {}, max(200, boot)
    for v in KVARS:
        K = kernel_table(fr, v)
        if K is None or len(K) < 4 * MIN_PAIRS:
            continue
        day = pd.factorize(K["day"])[0]
        comp = pd.factorize(np.array([find(i) if i in par else i for i in K.index], dtype=object))[0]
        z, t, pos = K["z"].values, K["t"].values, {p: i for i, p in enumerate(K.index)}
        o = np.argsort(day, kind="stable")
        I, J = [np.zeros(0, int)], [np.zeros(0, int)]
        for g in np.split(o, np.flatnonzero(np.diff(day[o])) + 1):
            iu, ju = np.triu_indices(len(g), 1)
            k = comp[g[iu]] != comp[g[ju]]
            I.append(g[iu[k]])
            J.append(g[ju[k]])
        m = rng.integers(0, len(K), (2, 3 * N_DIFF))
        m = m[:, (day[m[0]] != day[m[1]]) & (comp[m[0]] != comp[m[1]])][:, :N_DIFF]
        row = {"stranger": kbins(z, t, day, np.concatenate(I + [m[0]]), np.concatenate(J + [m[1]]), B, rng)}
        for cls in ("cohab", "sib"):
            p = pairs[pairs["cls"] == cls]
            a_ = np.array([pos.get(x, -1) for x in p["a"]], dtype=int)
            b_ = np.array([pos.get(x, -1) for x in p["b"]], dtype=int)
            ok = (a_ >= 0) & (b_ >= 0)
            row[cls] = kbins(z, t, day, a_[ok], b_[ok], B, rng)
        out[v] = row
    return out


def f3(x):
    if x is None:
        return "NA"
    return f"{float(x):.3g}"


def rci(c):
    if not c or c.get("r") is None:
        return f"NA(n{c.get('n') if c else 'NA'})"
    return f"{f3(c['r'])}[{f3(c['lo'])},{f3(c['hi'])}]n{c['n']}"


def kci(c):
    """v1.1 compact cell: r[lo,hi]n, 'n<count>' below MIN_PAIRS, '-' when no pair."""
    if not c or c.get("n") is None:
        return "-"
    if c.get("r") is None:
        return f"n{c['n']}"
    return "{:.3f}[{:.3f},{:.3f}]n{}".format(*(float(c[k]) for k in ("r", "lo", "hi")), c["n"])


def dci(c):
    if not c:
        return "NA"
    return f"{f3(c['diff'])}[{f3(c['lo'])},{f3(c['hi'])}]"


def digest(res):
    L = [f"F1 FAMILY DIGEST {res.get('version')} atlas {res.get('atlas_module')} calendar_adjust "
         f"{res.get('calendar_adjust')} day {res.get('day_adjust')} session {res.get('session_adjust')} boot {res.get('boot')}",
         "pairs: " + " ".join(f"{k} {v}" for k, v in res.get("pairs_n", {}).items()),
         f"partner |age diff| q: {res.get('partner_age_diff')}",
         "# concordance r[95% CI]n (double-entry; z of age/sex residual): var | cohab | partner | sib | halfsib | po"
         " | stranger_sameday | stranger_diffday"]
    for v, row in res.get("concordance", {}).items():
        L.append(f"R {v} | " + " | ".join(rci(row.get(t)) for t in TYPES))
    L.append("# contrasts: var | cohab-stranger_sameday | sib-cohab | sib-partner")
    for v, c in res.get("contrasts", {}).items():
        L.append(f"K {v} | {dci(c.get('cohab_minus_stranger_sameday'))} | {dci(c.get('sib_minus_cohab'))} | "
                 f"{dci(c.get('sib_minus_partner'))}")
    L.append("# cohabitants measured same day vs different days: var | same | different")
    for v, c in res.get("sameday_split", {}).items():
        if v in BEDS or v in RAW:
            L.append(f"D {v} | {rci(c.get('same_day'))} | {rci(c.get('different_day'))}")
    L.append("# risk-factor-adjusted concordance (own BMI, SBP, antihtn, logTG, HbA1c, smoking): bed | cohab | partner"
             " | sib")
    for v, c in res.get("rf_adjusted", {}).items():
        L.append(f"A {v} | {rci(c.get('cohab'))} | {rci(c.get('partner'))} | {rci(c.get('sib'))}")
    for t, m in res.get("cross_bed", {}).items():
        L.append(f"# cross-bed cross-person ({t}; oriented adverse): own~relative r[CI]n")
        items = list(m.items())
        for i in range(0, len(items), 4):
            L.append(f"X {t} " + " | ".join(f"{k} {rci(c)}" for k, c in items[i:i + 4]))
    L.append("# raw vs calendar-adjusted: bed type | raw | adjusted")
    for b, m in res.get("raw_vs_adjusted", {}).items():
        for t, c in m.items():
            L.append(f"W {b} {t} | {rci(c.get('raw'))} | {rci(c.get('adjusted'))}")
    L.append("# v1.1 time-gap kernel: r[95% CI, bootstrap over days]n of same-day pairs by |dt| (min) of the acquisition "
             "timestamps (vas_ts; bp_ts for BP); diff = different days: var type | " + " | ".join(KLAB))
    for v, row in res.get("time_kernel", {}).items():
        for t_, c in row.items():
            L.append(f"T {v} {t_} | " + " | ".join(kci(c.get(k)) for k in KLAB))
    if res.get("errors"):
        L.append(f"errors: {res['errors']}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--atlas", default=ATLAS_PATH, help="atlas module file (read at import time)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists")
    t0 = time.time()
    log(f"{VERSION}; atlas module {A.VERSION}; numpy {np.__version__}; pandas {pd.__version__}; boot {a.boot}")
    rng = np.random.default_rng(a.seed)
    fr, _, _ = A.build_frame(a.root, None)
    pids = set(fr["participant_id"].astype(str))
    cal = bool(getattr(A, "CAL_ADJUST", False))
    res = {"version": VERSION, "atlas_module": A.VERSION, "calendar_adjust": cal,
           "day_adjust": cal and bool(getattr(A, "DAY_ADJUST", False)),
           "session_adjust": cal and bool(getattr(A, "SESSION_ADJUST", False)), "boot": a.boot, "errors": []}
    try:
        pairs = load_pairs(a.root)
        log(f"relation pairs: {A.cnt(len(pairs))}")
        res.update(analyse(fr, pairs, a.boot, rng))
    except Exception as e:  # keep going: report the error string only
        res["errors"].append(f"{type(e).__name__}: {str(e)[:200]}")
        log(f"ERROR {type(e).__name__}: {e}")
    try:  # v1.1: after analyse(), so the earlier sections keep their random stream
        res["time_kernel"] = time_kernel(fr, pairs, a.boot, rng)
        log("time-gap kernel done")
    except Exception as e:
        res["errors"].append(f"time_kernel {type(e).__name__}: {str(e)[:200]}")
        log(f"ERROR time_kernel {type(e).__name__}: {e}")
    res["runtime_s"] = time.time() - t0
    text = json.dumps(A.clean(res), ensure_ascii=False, indent=1, sort_keys=True, default=str)
    A.privacy_check(text, pids)
    dg = digest(json.loads(text))
    A.privacy_check(dg, pids)
    logtext = "\n".join(A.LOG) + "\n"
    A.privacy_check(logtext, pids)
    os.makedirs(a.out)
    sums = []
    for fn_, tx in {"results.json": text, "digest.txt": dg, "run_log.txt": logtext}.items():
        with open(os.path.join(a.out, fn_), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn_}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s_[:16] + "  " + s_.split()[-1] for s_ in sums))
    log(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
