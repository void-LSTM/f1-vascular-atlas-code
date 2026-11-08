#!/usr/bin/env python3
"""F1 outcome-liability atlas (package P11 'f1_outcome_liability_v1'; HPP sandbox; writes aggregates only).

P11: do published
polygenic liabilities for cardiovascular outcomes map onto the four vascular beds differently (same people), and how do
the beds relate to a published 10-year risk score?

A  outcome-liability atlas.  Outcome PGS (weights folder `pgs_weights_outcome_v1`, PGS Catalog GRCh37
   harmonised files compacted by prepare_pgs_weights_outcome.py): CAD, ischaemic stroke, precerebral-artery stenosis
   (large-artery proxy, exploratory), PAD, HF, AF, CKD, hypertension, AAA, and T2D (the PRS v1.1 T2D_gw file, byte copy).
   Genotype reading, allele matching, scoring, sample QC, relatives and the unrelated set are PRS v1.1's own functions
   (f1_prs_v1_1.py, imported by path, never modified); atlas v1.5 beds (f1_atlas_v1_5.py, imported through PRS v1.1).
   Per score: coverage; positive controls; per-bed association (pwv, imt, art_w, abi_min core; vein_w, sokolow, cavi0
   secondary; adverse-oriented, beta per SD of score) in the unrelated set, model i (age, age^2, sex, PC1-10) and
   model ii (+ visit SBP, DBP, antihypertensive); BP part = model i - model ii on the model-ii sample (paired influence
   functions); same-sample joint test of the four core beds (robust joint covariance, Wald test of equal oriented
   effects, pairwise contrasts); within full-sib pairs (>= 200 pairs).  Matrix test: all primary scores x four core
   beds in one common sample, stacked influence functions; Wald tests of (T1) within-score bed heterogeneity and (T2)
   score x bed interaction (do liabilities map to beds differently?), with a permutation null (Freedman-Lane style:
   residualised scores permuted jointly).  Reproduction gate: the T2D lines must equal the PRS v1.1 digest.
B  risk-score translation.  PREVENT 10-year total CVD risk, base model (Khan et al., Circulation 2024;149:430),
   coefficients transcribed from the published supplement as implemented in two open packages (VALIDATION.md); baseline
   visit, ages 30-79.  Risk distribution; per bed, share in the bed's adverse top decile (age/sex-residualised) within
   low (< 5 %) / intermediate (5-7.5 %) / high (>= 7.5 %) predicted risk; overlap of "adverse in any core bed" with low
   risk.  Optional: if P10 f1_atlas_outcomes_v2.py (e8cc37eaa90adef6) and P1 f1_outcomes_v1.py (9bee14ddca223c57) are
   beside this script, their outcome builders are imported by path for statin and antihypertensive starts: HR per SD
   of each bed given the PREVENT linear predictor; otherwise 'P10 module absent: skipped'.
Privacy: aggregates only; counts rounded to 10, cells < 20 suppressed (None); every regression n >= 200 (within-sib
>= 200 pairs); quantiles 5-95 % only; shares from rounded counts; no participant ids; no dates; exception text
scrubbed; every output scanned for 7+ digit runs, participant ids and ISO dates before anything is written; refuses to
overwrite --out.  Liver tables are never read.
Run: python f1_outcome_liability_v1.py --root <HPP_DATA_ROOT> --weights pgs_weights_outcome_v1 \
         --out f1_outcome_liability_out_v1
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import stats

VERSION = "f1_outcome_liability_v1.0 (2026Q4 build)"
HERE = os.path.dirname(os.path.abspath(__file__))
EXPECT = {"atlas": "62344144b83db3f2", "prs": "0fb97f1301077fd6", "p10": "e8cc37eaa90adef6", "p1": "9bee14ddca223c57"}
P = None    # PRS v1.1 module
A = None    # atlas module (P.A)
M10 = None  # P10 module (optional)
BASE = "00_00_visit"
BEDS = ["pwv", "imt", "art_w", "abi_min", "vein_w", "sokolow", "cavi0"]
CORE = ["pwv", "imt", "art_w", "abi_min"]
SH = {"pwv": "pwv", "imt": "imt", "art_w": "ret", "abi_min": "abi", "vein_w": "ven", "sokolow": "sok", "cavi0": "cav",
      "c3": "c3"}
PAIRS = [("pwv", "imt"), ("pwv", "abi_min"), ("imt", "abi_min"), ("pwv", "art_w"), ("imt", "art_w"), ("art_w", "abi_min")]
SCORE_ORDER = ["CAD", "IS", "PCS", "PAD", "HF", "AF", "CKD", "HTN", "AAA", "T2D"]
PRIMARY = ["CAD", "IS", "PAD", "HF", "AF", "CKD", "HTN", "AAA", "T2D"]   # matrix test; PCS is an exploratory proxy
OWN = {"CAD": ("ldl", 1), "IS": ("sbp_sit", 1), "PCS": ("ldl", 1), "PAD": ("ldl", 1), "HF": ("bmi", 1),
       "AF": ("ecg_hr", 0), "CKD": ("egfr", -1), "HTN": ("sbp_sit", 1), "AAA": ("ldl", 1), "T2D": ("hba1c", 1)}
PRIMARY_CONTROLS = ("CAD", "HTN", "CKD", "T2D", "AF")
EXTRA = ["sbp_sit", "dbp_sit", "ldl", "log_tg", "egfr", "hba1c", "bmi", "ecg_hr", "pr_ms", "p_ms"]
N_PCS = 10
MIN_N, MIN_PAIRS = 200, 200
WIDTH, MAXLINES = 220, 120
# PRS v1.1 digest (results/real_aggregates/hpp_run_20261004/f1_prs_v1_1_a15_digest_transcribed.txt), T2D_gw lines
REPRO = {"score": "T2D", "rows_in": "171,249", "matched": "50,952",                       # PRS:15
         "pwv_i": (0.00512, -0.0168, 0.027), "pwv_ii": (-0.00295, -0.021, 0.0151),       # PRS:51, PRS:69
         "n_common": 3180, "pwv_minus_imt": (-0.0108, -0.0533, 0.0316), "het_p": 0.576}  # PRS:105
# PREVENT base model, 10-year total CVD (Khan 2024 supplement; preventr and PooledCohort coefficient tables agree)
PREV = {
    "F": dict(age=0.7939329, nhdl=0.0305239, hdl=-0.1606857, sbp_lt=-0.2394003, sbp_ge=0.3600781, dm=0.8667604,
              smk=0.5360739, egfr_lt=0.6045917, egfr_ge=0.0433769, bptx=0.3151672, statin=-0.1477655,
              tx_sbp=-0.0663612, st_nhdl=0.1197879, age_nhdl=-0.0819715, age_hdl=0.0306769, age_sbp=-0.0946348,
              age_dm=-0.2705700, age_smk=-0.0787150, age_egfr=-0.1637806, const=-3.3077280),
    "M": dict(age=0.7688528, nhdl=0.0736174, hdl=-0.0954431, sbp_lt=-0.4347345, sbp_ge=0.3362658, dm=0.7692857,
              smk=0.4386871, egfr_lt=0.5378979, egfr_ge=0.0164827, bptx=0.2888790, statin=-0.1337349,
              tx_sbp=-0.0475924, st_nhdl=0.1502730, age_nhdl=-0.0517874, age_hdl=0.0191169, age_sbp=-0.1049477,
              age_dm=-0.2251948, age_smk=-0.0895067, age_egfr=-0.1543702, const=-3.0311680)}
PREV_RANGE = {"age": (30, 79), "tc": (130, 320), "hdl": (20, 100), "sbp": (90, 180), "egfr": (15, 140)}
STRATA = (("low", 0.0, 0.05), ("int", 0.05, 0.075), ("high", 0.075, 1.01))
CATS = (("<5", 0.0, 0.05), ("5-7.5", 0.05, 0.075), ("7.5-20", 0.075, 0.20), (">=20", 0.20, 1.01))
P10_OUT = ("statin", "ahtn")
LONG = re.compile(r"\d{7,}")
LIVER = re.compile(r"liver|hepat|mafld|nafld|steato", re.I)
DATE_RX = re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?([+-]\d{2}:?\d{2}|Z)?")
BLOCKED = []


# ============================================================================ helpers (guard as P9/P10)

def hfmt(h):
    """First 16 hex characters of a SHA-256 in groups of 6/6/4 (an unbroken hash can hold 7+ digit runs)."""
    h = str(h)[:16]
    return ".".join(h[i:i + 6] for i in range(0, len(h), 6))


def sha_file(p, n=16):
    hs = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            hs.update(blk)
    return hs.hexdigest()[:n] if n else hs.hexdigest()


def load_modules(atlas_path, prs_path):
    """Import PRS v1.1 by path; it imports the atlas by path at import time (its own `--atlas` rule, fed here so that it
    loads `atlas_path`).  The atlas reader is wrapped so that no liver table is ever read (F1 partition)."""
    global P, A
    saved = sys.argv
    sys.argv = [saved[0], "--atlas", atlas_path, "--import-only"]
    try:
        spec = importlib.util.spec_from_file_location("f1_prs_v1_1", prs_path)
        P = importlib.util.module_from_spec(spec)
        sys.modules["f1_prs_v1_1"] = P
        spec.loader.exec_module(P)
    finally:
        sys.argv = saved
    A = P.A
    orig = A.rd

    def rd(root, rel, columns=None):
        if LIVER.search(str(rel)):
            BLOCKED.append(str(rel))
            return None
        return orig(root, rel, columns)
    A.rd = rd
    return P, A


def log(m):
    A.log(m)


def rc(n):
    return A.rc(n)


def v3(x):
    return A.v3(x)


def nrc(mask):
    return rc(int(np.asarray(mask, bool).sum()))


def scrub(s):
    s = DATE_RX.sub("<date>", str(s))
    s = re.sub(r"(?<!\d)\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?", "<time>", s)
    return re.sub(r"\d{4,}", "#", s)


def err(ex):
    tb = ([f for f in traceback.extract_tb(ex.__traceback__) if f.filename == __file__] or
          traceback.extract_tb(ex.__traceback__) or [None])[-1]
    return f"{type(ex).__name__}: {scrub(str(ex))[:160]} (line {tb.lineno if tb else '?'})"


def guard(out, key, fn):
    t1 = time.time()
    try:
        out[key] = fn()
    except Exception as ex:
        out[key] = {"error": err(ex)}
        log(f"{key} FAILED: {out[key]['error']}")
    out.setdefault("timings", {})[key] = time.time() - t1
    return out[key]


def privacy_check(text, pids):
    m = LONG.search(text)
    if m:
        raise SystemExit(f"PRIVACY ABORT: digit run '{m.group(0)[:3]}...' in output")
    if DATE_RX.search(text):
        raise SystemExit("PRIVACY ABORT: a date in output")
    for p in pids:  # ids of 7+ digits are caught above (real HPP ids have 10 digits)
        if not (p.isdigit() and len(p) >= 7) and len(p) >= 4 and p in text:
            raise SystemExit("PRIVACY ABORT: participant id in output")


def clean(o):
    if isinstance(o, np.ndarray):
        o = o.tolist()
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating, int, np.integer, bool, np.bool_)):
        return A.fnum(o)
    return o


def vq(s):
    """Module version strings carry build dates; outputs show the calendar quarter only."""
    return DATE_RX.sub(lambda m: f"{m.group(0)[:4]}Q{(int(m.group(0)[5:7]) - 1) // 3 + 1}", str(s))


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fl(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return np.nan
    return x


def p2(b, se):
    return float(2 * stats.norm.sf(abs(b / se))) if se and se > 0 and np.isfinite(se) else None


def coef(b, se, n):
    return {"b": float(b), "se": float(se), "lo": float(b - 1.96 * se), "hi": float(b + 1.96 * se), "p": p2(b, se),
            "n": rc(n)}


def zcol(x):
    x = np.asarray(x, float)
    s = x.std(ddof=1)
    return (x - x.mean()) / s if s > 0 else x * np.nan


# ============================================================================ data

def load_weights(wdir):
    """Weights folder: exactly one *_manifest.tsv (score, pgs_id, trait, file, sha256); every file SHA-256 checked."""
    mpath = os.path.join(wdir, "manifest.tsv")
    if not os.path.exists(mpath):
        cands = sorted(f for f in os.listdir(wdir) if f.endswith("_manifest.tsv"))
        if len(cands) != 1:
            raise SystemExit(f"need exactly one *_manifest.tsv in the weights folder, found {len(cands)}")
        mpath = os.path.join(wdir, cands[0])
    man = pd.read_csv(mpath, sep="\t")
    info, weights = {}, {}
    order = [s for s in SCORE_ORDER if s in set(man["score"])] + [s for s in man["score"] if s not in SCORE_ORDER]
    for s in order:
        r = man[man["score"] == s].iloc[0]
        p = os.path.join(wdir, str(r["file"]))
        if not os.path.exists(p):
            info[s] = {"status": "file missing", "pgs_id": str(r["pgs_id"])}
            continue
        h = sha_file(p, None)
        info[s] = {"pgs_id": str(r["pgs_id"]), "sha256_ok": h == str(r["sha256"]), "sha16": hfmt(h)}
        if h != str(r["sha256"]):
            log(f"weights {s}: SHA-256 mismatch, skipped")
            continue
        weights[s] = pd.read_parquet(p)
    log(f"weights loaded: {len(weights)} of {len(order)} scores")
    return weights, info


def add_ecg_intervals(root, fr):
    """P-wave duration and PR interval per visit (ECG table; AF positive control)."""
    e = A.rd(root, "ecg/ecg.parquet", columns=["p_ms", "pr_ms"])
    if e is None:
        return fr
    e = A.num(e, ["p_ms", "pr_ms"])
    cols = [c for c in ("p_ms", "pr_ms") if c in e.columns]
    if not cols:
        return fr
    pv_ = A.per_visit(e, cols)
    return fr.drop(columns=[c for c in cols if c in fr.columns]).merge(pv_, on=["participant_id", "research_stage"],
                                                                       how="left")


def ols_if(y, X, j=1):
    """OLS coefficient j with its per-row influence contributions a_i e_i (HC1: var = sum IF^2 n/(n-k))."""
    n, k = X.shape
    G = np.linalg.pinv(X.T @ X)
    b = G @ (X.T @ y)
    e = y - X @ b
    a = (X @ G)[:, j]
    return float(b[j]), a * e, n, k


# ============================================================================ A: outcome-liability atlas

def sib_within(d, pairs_fs, y, x):
    """PRS v1.1 sib_within (pair differences of y on the score difference, age and sex differences), with >= 200 pairs."""
    a, b = pairs_fs["ID1"].to_numpy(), pairs_fs["ID2"].to_numpy()
    dd = d.set_index("participant_id")
    have = np.isin(a, dd.index) & np.isin(b, dd.index)
    a, b = a[have], b[have]
    if len(a) < MIN_PAIRS:
        return {"n_pairs": rc(len(a)), "status": "too few pairs"}
    ya, yb = dd.loc[a, y].to_numpy(float), dd.loc[b, y].to_numpy(float)
    xa, xb = dd.loc[a, x].to_numpy(float), dd.loc[b, x].to_numpy(float)
    dage = dd.loc[a, "age"].to_numpy(float) - dd.loc[b, "age"].to_numpy(float)
    dsex = dd.loc[a, "sex_male"].to_numpy(float) - dd.loc[b, "sex_male"].to_numpy(float)
    sdy, sdx = np.nanstd(dd[y].to_numpy(float)), np.nanstd(dd[x].to_numpy(float))
    Y, Xd = (ya - yb) / sdy, np.column_stack([(xa - xb) / sdx, dage, dsex])
    ok = np.isfinite(Y) & np.isfinite(Xd).all(axis=1)
    if ok.sum() < MIN_PAIRS:
        return {"n_pairs": rc(ok.sum()), "status": "too few pairs"}
    bb, se, _ = A.ols(Y[ok], Xd[ok])
    return {"n_pairs": rc(ok.sum()), "b": bb[0], "se": se[0], "lo": bb[0] - 1.96 * se[0], "hi": bb[0] + 1.96 * se[0],
            "p": A.p_norm(bb[0], se[0])}


def joint_core(du, s, covs):
    """Four core beds in one sample: robust joint covariance (atlas multi_fit, as PRS v1.1), Wald test of equal oriented
    effects (3 df) and pairwise contrasts."""
    oc = ["o_" + b for b in CORE]
    cm = du[oc + [s] + covs].replace([np.inf, -np.inf], np.nan).dropna()
    if len(cm) < 300:
        return {"n": rc(len(cm)), "status": "too few"}
    X = np.column_stack([np.ones(len(cm)), A.zs(cm[s]).values] + [cm[c].astype(float).values for c in covs])
    Y = np.column_stack([A.zs(cm[c]).values for c in oc])
    bvec, V = A.multi_fit(Y, X)
    con, het = A.contrasts(bvec, V, CORE, PAIRS)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    beta = {b: {"b": float(bvec[i]), "lo": float(bvec[i] - 1.96 * se[i]), "hi": float(bvec[i] + 1.96 * se[i]),
                "p": p2(bvec[i], se[i])} for i, b in enumerate(CORE)}
    return {"n": rc(len(cm)), "beta": beta, "diff": con, "heterogeneity": het}


def bp_part(du, s, covs, bp):
    """Model i minus model ii on the model-ii sample: SD of bed per SD of score that visit BP statistically accounts for
    (paired influence functions; not a mediation estimate)."""
    out = {}
    for b_ in CORE:
        y = "o_" + b_
        dd = du[[y, s] + covs + bp].replace([np.inf, -np.inf], np.nan).dropna()
        if len(dd) < MIN_N:
            out[b_] = {"n": rc(len(dd)), "status": "too few"}
            continue
        yy, xx = zcol(dd[y]), zcol(dd[s])
        C0 = [dd[c].astype(float).values for c in covs]
        b1, if1, n, _ = ols_if(yy, np.column_stack([np.ones(len(dd)), xx] + C0))
        b2, if2, _, k2 = ols_if(yy, np.column_stack([np.ones(len(dd)), xx] + C0 + [dd[c].astype(float).values for c in bp]))
        f = n / max(n - k2, 1)
        se_d = float(np.sqrt((((if1 - if2) ** 2).sum()) * f))
        se1 = float(np.sqrt((if1 ** 2).sum() * f))
        out[b_] = {"n": rc(n), "i": coef(b1, se1, n), "ii": float(b2), "diff": coef(b1 - b2, se_d, n),
                   "share": float((b1 - b2) / b1) if abs(b1) > 0 else None}
    return out


C3 = np.array([[1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1]], float)


def matrix_stats(R, E, rr, p):
    """Coefficient matrix B (K x 4) of each bed on each score (FWL on residualised data), stacked influence functions,
    robust joint covariance (4K x 4K) and the two Wald statistics."""
    n, K = R.shape
    B = (R.T @ E) / rr[:, None]
    IF = np.empty((n, K * 4))
    for k in range(K):
        a = R[:, k] / rr[k]
        IF[:, 4 * k:4 * k + 4] = a[:, None] * (E - R[:, [k]] * B[k][None, :])
    V = IF.T @ IF * n / max(n - p, 1)
    b = B.reshape(-1)
    L1 = np.kron(np.eye(K), C3)
    D = np.column_stack([np.ones(K - 1), -np.eye(K - 1)])
    L2 = np.kron(D, C3)
    return B, V, wald_stat(L1, b, V), wald_stat(L2, b, V)


def wald_stat(L, b, V):
    Lb = L @ b
    S = L @ V @ L.T
    try:
        return float(Lb @ np.linalg.solve(S, Lb)), int(L.shape[0])
    except np.linalg.LinAlgError:
        r = int(np.linalg.matrix_rank(S))
        return float(Lb @ np.linalg.pinv(S) @ Lb), r


def matrix_test(du, scores, covs, perms, seed):
    """Primary scores x four core beds, common sample: T1 = within-score bed heterogeneity (3K df), T2 = score x bed
    interaction (3(K-1) df) with robust joint covariance; permutation null: residualised scores permuted jointly."""
    sc = [s for s in scores if s in du.columns]
    oc = ["o_" + b for b in CORE]
    cm = du[oc + sc + covs].replace([np.inf, -np.inf], np.nan).dropna()
    n, K = len(cm), len(sc)
    if n < 300 or K < 2:
        return {"n": rc(n), "status": "too few"}
    Zs = np.column_stack([zcol(cm[s]) for s in sc])
    Zb = np.column_stack([zcol(cm[c]) for c in oc])
    X0 = np.column_stack([np.ones(n)] + [cm[c].astype(float).values for c in covs])
    R = Zs - X0 @ np.linalg.lstsq(X0, Zs, rcond=None)[0]
    E = Zb - X0 @ np.linalg.lstsq(X0, Zb, rcond=None)[0]
    rr = (R * R).sum(axis=0)
    p = X0.shape[1] + 1
    B, V, (t1, df1), (t2, df2) = matrix_stats(R, E, rr, p)
    rng = np.random.default_rng(seed)
    n1 = n2 = 0
    s1 = s2 = 0.0
    t0 = time.time()
    done = 0
    for _ in range(perms):
        Rp = R[rng.permutation(n)]
        _, _, (a1, _), (a2, _) = matrix_stats(Rp, E, (Rp * Rp).sum(axis=0), p)
        n1 += a1 >= t1
        n2 += a2 >= t2
        s1 += a1
        s2 += a2
        done += 1
    se = np.sqrt(np.clip(np.diag(V), 0, None)).reshape(K, 4)
    return {"n": rc(n), "scores": sc, "beds": CORE, "beta": B.tolist(), "se": se.tolist(),
            "T1_within_score_bed_heterogeneity": {"chi2": t1, "df": df1, "p": float(stats.chi2.sf(t1, df1)),
                                                  "perm_p": (1 + n1) / (1 + done), "perm_mean_over_df": s1 / done / df1
                                                  if done else None},
            "T2_score_x_bed_interaction": {"chi2": t2, "df": df2, "p": float(stats.chi2.sf(t2, df2)),
                                           "perm_p": (1 + n2) / (1 + done), "perm_mean_over_df": s2 / done / df2
                                           if done else None},
            "perms": done, "perm_seconds": time.time() - t0}


def section_a(root, fr, weights_dir, block, perms, seed):
    R = {}
    t0 = time.time()
    weights, winfo = load_weights(weights_dir)
    R["weights"] = winfo
    base = os.path.join(root, "human_genetics", "genotypes", "plink_postqc")
    fam = P.read_fam_ids(os.path.join(base, "chr_1"))
    cov, flow = P.load_samples(root, fam)
    S, coverage = P.score_all(root, weights, len(fam), block)
    R["timings_scoring_s"] = time.time() - t0
    names = list(weights)
    scores = pd.DataFrame(S, index=fam, columns=names).loc[cov.index]
    pairs = P.king_pairs(root, cov.index)
    unrel = P.unrelated_set(cov.index, pairs)
    k, i0 = pairs["Kinship"].to_numpy(), pairs["IBS0"].to_numpy()
    fs = pairs[(k > P.KIN_FIRST) & (k <= P.KIN_DUP) & (i0 >= P.IBS0_PO_MAX)]
    flow.update({"unrelated": rc(len(unrel)), "full_sib_pairs": rc(len(fs)), "related_pairs": rc(len(pairs))})
    log(f"samples: {A.cnt(len(cov))}; unrelated {A.cnt(len(unrel))}; full-sib pairs {A.cnt(len(fs))}")
    R["flow"] = flow
    R["coverage"] = P.variant_counts_text(coverage)
    R["score_corr_unrelated"] = None
    fr = add_ecg_intervals(root, fr)
    d = A.one_row_per_participant(fr, ["age", "sex_male"])
    d = d[d["participant_id"].isin(scores.index)].copy()
    d = d.join(scores, on="participant_id").join(cov[[f"pc{i}" for i in range(1, N_PCS + 1)]], on="participant_id")
    d["age2"] = (d["age"] - d["age"].mean()) ** 2
    for b_ in BEDS:
        if b_ in d.columns:
            d["o_" + b_] = A.ORIENT.get(b_, 1) * d[b_]
    base_cov = ["age", "age2", "sex_male"] + [f"pc{i}" for i in range(1, N_PCS + 1)]
    bp = ["sbp_sit", "dbp_sit", "antihtn"]
    du = d[d["participant_id"].isin(unrel)]
    R["score_corr_unrelated"] = {a: {b: float(du[a].corr(du[b])) for b in names} for a in names}
    R["positive_controls"], R["other"], R["beds_i"], R["beds_ii"], R["within_sib"] = {}, {}, {}, {}, {}
    R["joint_i"], R["joint_ii"], R["bp_part"] = {}, {}, {}
    for s in names:
        y, sign = OWN.get(s, (None, 0))
        if y in du.columns:
            R["positive_controls"][s] = {"outcome": y, "expected_sign": sign, **(P.assoc(du, y, s, base_cov) or {})}
        R["other"][s] = {y_: P.assoc(du, y_, s, base_cov) for y_ in EXTRA if y_ in du.columns}
        R["beds_i"][s], R["beds_ii"][s], R["within_sib"][s] = {}, {}, {}
        for b_ in BEDS:
            yo = "o_" + b_
            if yo not in du.columns:
                continue
            R["beds_i"][s][b_] = P.assoc(du, yo, s, base_cov)
            R["beds_ii"][s][b_] = P.assoc(du, yo, s, base_cov + bp)
            R["within_sib"][s][b_] = sib_within(d, fs, yo, s)
        R["joint_i"][s] = joint_core(du, s, base_cov)
        R["joint_ii"][s] = joint_core(du, s, base_cov + bp)
        R["bp_part"][s] = bp_part(du, s, base_cov, bp)
    prim = [s for s in PRIMARY if s in names]
    t1 = time.time()
    R["matrix_i"] = matrix_test(du, prim, base_cov, perms, seed)
    R["matrix_ii"] = matrix_test(du, prim, base_cov + bp, perms, seed + 1)
    R["timings_matrix_s"] = time.time() - t1
    R["repro"] = repro_check(R)
    ids = set(d["participant_id"].astype(str)) | set(map(str, fam)) | set(pairs["ID1"].astype(str)) | \
        set(pairs["ID2"].astype(str))
    return R, ids


def repro_check(R):
    """T2D (= PRS v1.1 T2D_gw weights) must reproduce the PRS v1.1 digest lines at 3 significant digits."""
    s = REPRO["score"]
    out = {}

    def same(c, ref):
        return isinstance(c, dict) and c.get("b") is not None and all(
            f"{fl(x):.3g}" == f"{y:.3g}" for x, y in zip((c.get("b", c.get("d")), c.get("lo"), c.get("hi")), ref))
    cv = g(R, "coverage", s) or {}
    out["coverage"] = bool(cv.get("rows_in") == REPRO["rows_in"] and cv.get("matched") == REPRO["matched"])
    out["pwv_i"] = same(g(R, "beds_i", s, "pwv"), REPRO["pwv_i"])
    out["pwv_ii"] = same(g(R, "beds_ii", s, "pwv"), REPRO["pwv_ii"])
    j = g(R, "joint_i", s) or {}
    dd = g(j, "diff", "pwv_minus_imt") or {}
    out["joint"] = bool(j.get("n") == REPRO["n_common"] and dd.get("d") is not None and
                        all(f"{fl(x):.3g}" == f"{y:.3g}" for x, y in zip((dd.get("d"), dd.get("lo"), dd.get("hi")),
                                                                        REPRO["pwv_minus_imt"]))
                        and f"{fl(g(j, 'heterogeneity', 'p')):.3g}" == f"{REPRO['het_p']:.3g}")
    out["all_ok"] = all(out.values())
    return out


# ============================================================================ B: PREVENT risk translation

def lab_thresholds(fr):
    hb = pd.to_numeric(fr.get("hba1c"), errors="coerce").median() if "hba1c" in fr.columns else np.nan
    gl = pd.to_numeric(fr.get("glucose"), errors="coerce").median() if "glucose" in fr.columns else np.nan
    return {"hba1c": 48.0 if np.isfinite(hb) and hb > 20 else 6.5, "glucose": 7.0 if np.isfinite(gl) and gl < 30 else 126.0}


def naive_ns(x):
    """Datetimes as tz-naive datetime64[ns] (HPP timestamps load as datetime64[us]; merge keys must match)."""
    t = pd.to_datetime(x, errors="coerce")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t.astype("datetime64[ns]")


def tc_hdl(root, b0):
    """Total and HDL cholesterol from one draw: nearest blood test within 180 days of the visit (atlas lab rule) with
    HDL and total (or non-HDL) cholesterol; mg/dL (mmol/L converted, factor 0.02586)."""
    bt = A.rd(root, "blood_tests/blood_tests.parquet",
              columns=["collection_date", "bt__total_cholesterol_float_value", "bt__hdl_cholesterol_float_value",
                       "bt__non_hdl_cholesterol_float_value"])
    info = {}
    if bt is None:
        return pd.DataFrame(columns=["participant_id", "tc", "hdl_b"]), {"status": "blood tests missing"}
    mp = {"bt__total_cholesterol_float_value": "tc", "bt__hdl_cholesterol_float_value": "hdl_b",
          "bt__non_hdl_cholesterol_float_value": "nhdl"}
    bt = A.num(bt, list(mp)).rename(columns=mp)
    for c in ("tc", "hdl_b", "nhdl"):
        if c not in bt.columns:
            bt[c] = np.nan
    bt["tc_src"] = np.where(bt["tc"].notna(), "measured", np.where(bt["nhdl"].notna(), "nonhdl+hdl", None))
    bt["tc"] = bt["tc"].where(bt["tc"].notna(), bt["nhdl"] + bt["hdl_b"])
    bt["lab_date"] = naive_ns(bt["collection_date"])
    bt = bt.dropna(subset=["lab_date", "tc", "hdl_b"])[["participant_id", "lab_date", "tc", "hdl_b", "tc_src"]]
    for c in ("tc", "hdl_b"):
        med = bt[c].median()
        info[c + "_unit"] = "mg/dL" if med > 20 else "mmol/L"
        if med <= 20:
            bt[c] = bt[c] / 0.02586
    left = b0[["participant_id", "visit_date"]].dropna().sort_values("visit_date")
    left = left.assign(visit_date=naive_ns(left["visit_date"]))
    bt = bt.sort_values("lab_date")
    m = pd.merge_asof(left, bt, left_on="visit_date", right_on="lab_date", by="participant_id", direction="nearest",
                      tolerance=pd.Timedelta(days=180))
    info["tc_source"] = {str(k): rc(v) for k, v in m["tc_src"].value_counts().items()}
    return m[["participant_id", "tc", "hdl_b"]], info


def curated_dm(root):
    col = "diabetes__curated_phenotype"
    d = A.rd(root, "curated_phenotypes/diabetes.parquet", columns=[col])
    if d is None or col not in d.columns:
        return set(), "missing"
    s = d[col].astype(str).str.strip().str.lower()
    return set(d.loc[(d["research_stage"] == BASE) & s.str.contains("diabet") & ~s.str.startswith(("non", "pre")),
                     "participant_id"]), "ok"


def prevent_lp(age, male, tc, hdl, sbp, bptx, statin, dm, smk, egfr):
    """PREVENT base 10-year total CVD log-odds (vectorised); cholesterol in mg/dL."""
    out = np.full(len(age), np.nan)
    for sx, key in ((1, "M"), (0, "F")):
        c = PREV[key]
        i = male == sx
        a = (age[i] - 55) / 10
        nh = (tc[i] - hdl[i]) * 0.02586 - 3.5
        hd = (hdl[i] * 0.02586 - 1.3) / 0.3
        sl = (np.minimum(sbp[i], 110) - 110) / 20
        sg = (np.maximum(sbp[i], 110) - 130) / 20
        el = (np.minimum(egfr[i], 60) - 60) / -15
        eg = (np.maximum(egfr[i], 60) - 90) / -15
        out[i] = (c["const"] + c["age"] * a + c["nhdl"] * nh + c["hdl"] * hd + c["sbp_lt"] * sl + c["sbp_ge"] * sg
                  + c["dm"] * dm[i] + c["smk"] * smk[i] + c["egfr_lt"] * el + c["egfr_ge"] * eg + c["bptx"] * bptx[i]
                  + c["statin"] * statin[i] + c["tx_sbp"] * bptx[i] * sg + c["st_nhdl"] * statin[i] * nh
                  + c["age_nhdl"] * a * nh + c["age_hdl"] * a * hd + c["age_sbp"] * a * sg + c["age_dm"] * a * dm[i]
                  + c["age_smk"] * a * smk[i] + c["age_egfr"] * a * el)
    return out


def prevent_risk(*args):
    return 1 / (1 + np.exp(-prevent_lp(*args)))


def prevent_table(root, fr):
    """Baseline PREVENT inputs and risk: ages 30-79, complete cases; TC, HDL, SBP, eGFR clipped to the published input
    ranges (counts reported); diabetes = A10 drug, HbA1c / glucose at or above the diabetes threshold, or curated
    diabetes at baseline."""
    b0 = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id").copy()
    info = {"baseline": rc(len(b0))}
    lip, li = tc_hdl(root, b0)
    info.update(li)
    b0 = b0.merge(lip, on="participant_id", how="left")
    thr = lab_thresholds(fr)
    cur, cst = curated_dm(root)
    info["thresholds"], info["curated_diabetes"] = thr, cst

    def col(c):
        return pd.to_numeric(b0[c], errors="coerce").astype(float) if c in b0.columns else pd.Series(np.nan, index=b0.index)
    dm = ((col("diabetes_any").fillna(0) == 1) | (col("hba1c") >= thr["hba1c"]) | (col("glucose") >= thr["glucose"]) |
          b0["participant_id"].isin(cur)).astype(float)
    t = pd.DataFrame({"participant_id": b0["participant_id"].values, "age": col("age").values,
                      "male": col("sex_male").values, "tc": col("tc").values, "hdl": col("hdl_b").values,
                      "sbp": col("sbp_sit").values, "bptx": col("antihtn").fillna(0).values,
                      "statin": col("statin").fillna(0).values, "dm": dm.values, "smk": col("smk_current").values,
                      "egfr": col("egfr").values})
    ina = t["age"].between(*PREV_RANGE["age"])
    info["outside_age_30_79"] = nrc(~ina & t["age"].notna())
    t = t[ina]
    need = ["male", "tc", "hdl", "sbp", "smk", "egfr"]
    info["missing_input"] = {c: nrc(t[c].isna()) for c in need}
    t = t.dropna(subset=need).copy()
    info["clipped"] = {}
    for c in ("tc", "hdl", "sbp", "egfr"):
        lo, hi = PREV_RANGE[c]
        info["clipped"][c] = nrc((t[c] < lo) | (t[c] > hi))
        t[c] = t[c].clip(lo, hi)
    t["lp"] = prevent_lp(*(t[c].to_numpy(float) for c in ("age", "male", "tc", "hdl", "sbp", "bptx", "statin", "dm",
                                                           "smk", "egfr")))
    t["risk"] = 1 / (1 + np.exp(-t["lp"]))
    info["n"] = rc(len(t))
    info["risk_q"] = A.qtiles(t["risk"])
    info["categories"] = {k: nrc((t["risk"] >= lo) & (t["risk"] < hi)) for k, lo, hi in CATS}
    info["inputs_mean"] = {c: float(t[c].mean()) for c in ("age", "tc", "hdl", "sbp", "egfr")} if len(t) >= MIN_N else None
    info["inputs_share"] = {c: float(rc(t[c].sum()) / rc(len(t))) if rc(t[c].sum()) and rc(len(t)) else None
                            for c in ("male", "bptx", "statin", "dm", "smk")}
    return t, b0, info


def adverse_top(df, bed):
    """Adverse z residualised on age, (age-50)^2 and sex in the given rows; top decile = z >= its 90th centile."""
    y = A.ORIENT.get(bed, 1) * df[bed].astype(float).values
    a = df["age"].astype(float).values
    X = np.column_stack([np.ones(len(df)), a, (a - 50) ** 2, df["male"].astype(float).values])
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    z = r / r.std()
    return z >= np.quantile(z, 0.9)


def strat(risk):
    return np.select([(risk >= lo) & (risk < hi) for _, lo, hi in STRATA], [k for k, _, _ in STRATA], "na")


def share(k, n):
    """Share from rounded counts (no information beyond the rounded counts)."""
    rk, rn = rc(k), rc(n)
    return float(rk / rn) if (rk is not None and rn and rc(n - k) is not None) else None


def section_b(root, fr):
    R = {}
    t, b0, info = prevent_table(root, fr)
    R["prevent"] = info
    m = t.merge(b0[["participant_id"] + [b for b in BEDS if b in b0.columns]], on="participant_id", how="left")
    beds = {}
    for bed in BEDS:
        if bed not in m.columns:
            beds[bed] = {"status": "missing"}
            continue
        dd = m.dropna(subset=[bed]).reset_index(drop=True)
        if len(dd) < MIN_N:
            beds[bed] = {"n": rc(len(dd)), "status": "too few"}
            continue
        top = adverse_top(dd, bed)
        st = strat(dd["risk"].values)
        rec = {"n": rc(len(dd)), "top": nrc(top), "strata": {}}
        for k, _, _ in STRATA:
            s_ = st == k
            rec["strata"][k] = {"n": nrc(s_), "top": nrc(s_ & top), "share_top": share(int((s_ & top).sum()), int(s_.sum()))}
        rec["auc_risk_for_top"] = float(A.auc(top.astype(float), dd["risk"].values)) \
            if rc(top.sum()) and rc((~top).sum()) else None
        beds[bed] = rec
    R["beds"] = beds
    dd = m.dropna(subset=CORE).reset_index(drop=True) if all(c in m.columns for c in CORE) else m.iloc[:0]
    anyc = {"n": rc(len(dd))}
    if len(dd) >= MIN_N:
        tops = np.column_stack([adverse_top(dd, b) for b in CORE])
        cnt = tops.sum(axis=1)
        st = strat(dd["risk"].values)
        anyc["count_by_stratum"] = {k: {"0": nrc((st == k) & (cnt == 0)), "1": nrc((st == k) & (cnt == 1)),
                                        "ge2": nrc((st == k) & (cnt >= 2))} for k, _, _ in STRATA}
        adv = cnt >= 1
        low = st == "low"
        anyc.update({"adverse_any": nrc(adv), "adverse_any_low": nrc(adv & low), "low": nrc(low),
                     "share_adverse_any_low": share(int((adv & low).sum()), int(adv.sum())),
                     "share_low_all": share(int(low.sum()), len(dd)),
                     "share_low_adverse_any": share(int((adv & low).sum()), int(low.sum())),
                     "share_adverse_any_all": share(int(adv.sum()), len(dd)),
                     "independence_any": float(1 - 0.9 ** 4)})
    R["any_core"] = anyc
    R["_risk"] = t[["participant_id", "lp", "risk"]]
    return R


def load_p10(p10_path, p1_path):
    """Optional P10 outcome builders (and P1 through P10), by path, only at the expected hashes; they share this run's
    atlas instance (same liver guard, one log)."""
    global M10
    if not (os.path.exists(p10_path) and os.path.exists(p1_path)):
        return "absent"
    s10, s1 = sha_file(p10_path), sha_file(p1_path)
    if s10 != EXPECT["p10"] or s1 != EXPECT["p1"]:
        return f"DIFF {hfmt(s10)} {hfmt(s1)}"
    spec = importlib.util.spec_from_file_location("f1_atlas_outcomes_v2", p10_path)
    M10 = importlib.util.module_from_spec(spec)
    sys.modules["f1_atlas_outcomes_v2"] = M10
    spec.loader.exec_module(M10)
    M10.load_modules(ATLAS_FILE, p1_path)
    M10.A = A
    M10.O.A = A
    return "ok"


def section_p10(root, fr, risk, seed):
    """Statin and antihypertensive starts (P10 definitions, P1 hazard machinery): HR per SD of each bed given the
    PREVENT linear predictor, and of the PREVENT linear predictor alone."""
    O = M10.O
    C, base_dates, prev_htn, cinfo = O.load_contacts(root, fr)
    fam = O.load_fam(root)
    ctx = O.prepare(fr, C, base_dates, prev_htn, fam, cinfo, seed)
    thr = M10.lab_thresholds(fr)
    mrows, crows = M10.med_class_rows(root), M10.cond_group_rows(root)
    C2, _ = M10.add_flags(ctx["C"], base_dates, mrows, ["m_" + k for k in M10.MED])
    med0 = {}
    if mrows is not None:
        m0 = mrows[mrows["research_stage"] == BASE].groupby("participant_id")[["m_" + k for k in M10.MED]].max()
        med0 = {k: m0["m_" + k] for k in M10.MED}
    prev = {k: M10.prevalent(crows, base_dates, "i_" + k) for k in M10.ICD}
    extra = {"med0": med0, "prev": prev, "cur": M10.curated_flags(root)}
    b, _ = M10.baseline_table(fr, ctx, extra)
    b = b.merge(risk[["participant_id", "lp"]].rename(columns={"lp": "prevent_lp"}), on="participant_id", how="left")
    out = {}
    for oc in P10_OUT:
        B = b[M10.cohort_mask(oc, b, ctx, extra, thr) & b["prevent_lp"].notna().values].reset_index(drop=True)
        PP = O.person_period(B, C2, M10.ev_fn(oc, thr))
        W, _ = O.ipw(B, C2, PP)
        Pw = O.attach_w(PP, W)
        rec = {"cohort": rc(len(B)), "risk_alone": O.hazard(Pw, B, ["prevent_lp"], [])[0]}
        for bed in CORE + ["c3"]:
            x = "o_" + bed
            rec[bed] = O.hazard(Pw, B, [x], ["prevent_lp"])[0] if x in B.columns else {"status": "missing"}
        out[oc] = rec
    return out


# ============================================================================ digest

def wrap(lines):
    out = []
    for prio, ln in lines:
        ln = re.sub(r"\s+", " ", str(ln)).strip()
        first = True
        while len(ln) > WIDTH:
            cut = ln.rfind(" ", 100, WIDTH)
            cut = WIDTH if cut < 0 else cut
            out.append((prio, ln[:cut].rstrip(), first))
            ln = "+ " + ln[cut:].strip()
            first = False
        out.append((prio, ln, first))
    return out


def shrink(wl):
    """Keep <= MAXLINES lines: drop whole logical lines of priority 3 (last first), then of priority 2, only as many as
    needed (one line of room is kept for the note); truncate as a last resort (P10 rule)."""
    groups = []
    for p_, ln, first in wl:
        if first or not groups:
            groups.append([p_, [ln]])
        else:
            groups[-1][1].append(ln)
    total = sum(len(x[1]) for x in groups)
    dropped = 0
    for lvl in (3, 2):
        i = len(groups) - 1
        while total > MAXLINES - 1 and i >= 0:
            if groups[i][0] == lvl:
                total -= len(groups[i][1])
                del groups[i]
                dropped += 1
            i -= 1
    lines = [ln for _, ls in groups for ln in ls]
    if dropped:
        pos = sum(len(ls) for _, ls in groups[:3])
        lines.insert(min(pos, len(lines)), f"SHORTENED: {dropped} lower-priority lines dropped; see results.json")
    if len(lines) > MAXLINES:
        lines = lines[:MAXLINES - 1] + [f"DIGEST TRUNCATED: {len(lines)} lines; see results.json"]
    return lines


def ci(c, key="b"):
    if not isinstance(c, dict):
        return "NA"
    v = c.get(key, c.get("d", c.get("hr")))
    if v is None:
        st = str(c.get("status") or "")
        return "NA" if not st else ("NA(<200p)" if st == "too few pairs" else
                                    "NA(<200)" if st == "too few" else f"NA({st[:10]})")
    return f"{v3(v)}[{v3(c.get('lo'))},{v3(c.get('hi'))}]"


def pstr(p):
    return "NA" if p is None else f"{float(p):.2g}"


def kvs(d):
    return " ".join(f"{k} {v}" for k, v in (d or {}).items()) if isinstance(d, dict) else "NA"


def nrange(cs):
    ns = [c.get("n", c.get("n_pairs")) for c in cs if isinstance(c, dict) and c.get("n", c.get("n_pairs")) is not None]
    return f"n {min(ns)}-{max(ns)}" if ns else "n NA"


def hzs(r, x):
    if not isinstance(r, dict):
        return "NA"
    if r.get("error"):
        return "ERR"
    st = r.get("status")
    if st not in ("ok", "not converged"):
        return "NA(<20)" if st == "too few" else f"NA({str(st)[:12]})"
    return ci(g(r, "hr", x), "hr") + ("*nc" if st == "not converged" else "")


def decisions(R):
    """Pre-stated readings (PROTOCOL 'How to read'), computed from the aggregates; priority 1."""
    L = []
    A_ = R.get("A") or {}
    if A_ and not A_.get("error"):
        for key, lab in (("matrix_i", "i"), ("matrix_ii", "ii")):
            m = A_.get(key) or {}
            t2, t1 = m.get("T2_score_x_bed_interaction") or {}, m.get("T1_within_score_bed_heterogeneity") or {}
            if not t2:
                L.append((1, f"D1{lab} matrix: {m.get('status', m.get('error', 'NA'))}"))
                continue
            differ = (t2.get("perm_p") is not None and float(t2["perm_p"]) < 0.05 and float(t2["p"]) < 0.05)
            verdict = "PROFILES DIFFER across outcome liabilities" if differ else "compatible with one shared bed profile"
            L.append((1, f"D1{lab} do outcome liabilities map to beds differently ({len(m.get('scores') or [])} scores x 4 core "
                         f"beds, model {lab}, n {m.get('n')}): score x bed interaction chi2 {v3(t2.get('chi2'))} df "
                         f"{t2.get('df')} p {pstr(t2.get('p'))} perm p {pstr(t2.get('perm_p'))} (perm null mean/df "
                         f"{v3(t2.get('perm_mean_over_df'))}) -> {verdict} | within-score bed heterogeneity chi2 "
                         f"{v3(t1.get('chi2'))} df {t1.get('df')} p {pstr(t1.get('p'))} perm p {pstr(t1.get('perm_p'))}"))
        alpha = 0.05 / len(PRIMARY)
        parts = []
        for s in SCORE_ORDER:
            j = g(A_, "joint_i", s)
            if not isinstance(j, dict) or not j.get("beta"):
                continue
            pp = g(j, "heterogeneity", "p")
            het = pp is not None and float(pp) < alpha and s in PRIMARY
            top = max(CORE, key=lambda b_: fl(g(j, "beta", b_, "b")))
            dif = j.get("diff") or {}
            best = min(dif, key=lambda k_: fl(dif[k_].get("p")) if dif[k_].get("p") is not None else 9) if dif else None
            parts.append(f"{s} p {pstr(pp)}" + (f" HET top {SH[top]} {v3(g(j, 'beta', top, 'b'))}; {best.replace('_minus_', '-').replace('abi_min', 'abi').replace('art_w', 'ret')} "
                                                f"{ci(dif.get(best), 'd')}" if het and best else ""))
        L.append((1, f"D2 per-score core-bed heterogeneity (model i, Wald 3 df; HET = p < 0.05/9 = {alpha:.2g}, PCS "
                     "exploratory): " + " | ".join(parts)))
        bp = []
        for s in SCORE_ORDER:
            for b_ in CORE:
                r = g(A_, "bp_part", s, b_) or {}
                if fl(g(r, "i", "p")) < 0.05 / (len(PRIMARY) * 4) and r.get("diff"):
                    bp.append(f"{s} {SH[b_]} i {v3(g(r, 'i', 'b'))} BP part {ci(r['diff'])} ({v3(r.get('share'))})")
        L.append((1, "D3 through visit BP (model i - model ii on the model-ii sample; cells with model-i p < 0.05/36; "
                     "share of i in brackets): " + (" | ".join(bp) if bp else "none")))
        cells, agree, excl = 0, 0, 0
        for s in PRIMARY:
            for b_ in CORE:
                c = g(A_, "beds_i", s, b_)
                w = g(A_, "within_sib", s, b_)
                if isinstance(c, dict) and fl(c.get("p")) < 0.05 / (len(PRIMARY) * 4) and isinstance(w, dict) and w.get("b") is not None:
                    cells += 1
                    agree += np.sign(fl(w["b"])) == np.sign(fl(c["b"]))
                    excl += (fl(w["lo"]) > 0) or (fl(w["hi"]) < 0)
        L.append((1, f"D4 within full-sib pairs: of {cells} (score, core bed) cells with model-i p < 0.05/36, {agree} have "
                     f"the same sign within sibships and {excl} a within-sib CI excluding 0 (pairs {g(A_, 'flow', 'full_sib_pairs')})"))
        pc = []
        for s in SCORE_ORDER:
            c = g(A_, "positive_controls", s) or {}
            if c.get("b") is None:
                continue
            sg = c.get("expected_sign") or 0
            ok = (fl(c.get("p")) < 0.05 and np.sign(fl(c["b"])) == sg) if sg else None
            tag = "" if s not in PRIMARY_CONTROLS else (" PASS" if ok else (" two-sided" if ok is None else " FAIL"))
            pc.append(f"{s}->{c.get('outcome')} {ci(c)}{tag}")
        afo = g(A_, "other", "AF") or {}
        L.append((1, "D5 positive controls (unrelated, model i; PASS = p < 0.05 in the expected direction): " + " | ".join(pc)
                     + (f" | AF->pr_ms {ci(afo.get('pr_ms'))} p_ms {ci(afo.get('p_ms'))}" if afo else "")))
    B_ = R.get("B") or {}
    if B_ and not B_.get("error"):
        a = B_.get("any_core") or {}
        L.append((1, f"D6 risk translation (PREVENT 10-y total CVD; low < 5 %): of {a.get('n')} with all four core beds, "
                     f"{a.get('adverse_any')} are in >= 1 bed's adverse top decile; {a.get('adverse_any_low')} of them have low "
                     f"predicted risk (share {v3(a.get('share_adverse_any_low'))} vs {v3(a.get('share_low_all'))} of everyone); "
                     f"low-risk participants adverse in >= 1 bed {v3(a.get('share_low_adverse_any'))} (all "
                     f"{v3(a.get('share_adverse_any_all'))}, independence {v3(a.get('independence_any'))})"))
    st = R.get("p10_status")
    if st == "ok":
        p10 = R.get("P10") or {}
        if p10.get("error"):
            L.append((1, f"D7 P10 add-on ERROR {p10['error']}"))
        else:
            L.append((1, "D7 beds given PREVENT (HR per SD of the adverse bed, adjusted for the PREVENT linear predictor; "
                         "P10 cohorts and P1 hazard): " + " | ".join(
                             f"{oc} " + " ".join(f"{SH[b_]} {hzs(g(p10, oc, b_), 'o_' + b_)}" for b_ in CORE + ['c3'])
                             for oc in P10_OUT)))
    else:
        L.append((1, f"D7 P10 module {'absent' if st == 'absent' else str(st)}: skipped"))
    return L


def jline(j):
    """Five requested contrasts (pwv-imt, pwv-abi, imt-abi, pwv-ret, imt-ret); ret-abi stays in results.json."""
    return " ".join(f"{a_.replace('abi_min', 'abi').replace('art_w', 'ret')}-{c_.replace('abi_min', 'abi').replace('art_w', 'ret')} "
                    f"{ci(g(j, 'diff', f'{a_}_minus_{c_}'), 'd')}" for a_, c_ in PAIRS[:5])


def dg_a(A_):
    L = []
    if not isinstance(A_, dict):
        return L
    if A_.get("error"):
        return [(1, f"ERROR A {A_['error']}")]
    cov = A_.get("coverage") or {}
    w = A_.get("weights") or {}
    L.append((1, "A = beta per SD of score [95% CI] of the adverse-oriented bed (SD units), unrelated; i = age age2 sex "
                 "PC1-10; ii = i + visit SBP DBP antihtn; sib = within full-sib pairs; J = core-bed joint test (Wald p "
                 "3 df, contrasts d[lo,hi]); i2 = secondary beds"))
    for s in A_.get("beds_i") or {}:
        bi = A_["beds_i"][s]
        L.append((1, f"A i {s}: " + " ".join(f"{SH[b_]} {ci(bi.get(b_))}" for b_ in CORE if b_ in bi)
                  + f" | {nrange([bi.get(b_) for b_ in CORE])}"))
    for s in A_.get("joint_i") or {}:
        j = A_["joint_i"][s]
        if not isinstance(j, dict) or not j.get("diff"):
            L.append((1, f"J i {s}: {j.get('status') if isinstance(j, dict) else 'NA'} n {g(j, 'n')}"))
            continue
        L.append((1, f"J i {s} n {j.get('n')} p {pstr(g(j, 'heterogeneity', 'p'))}: {jline(j)}"))
    for s in A_.get("beds_i") or {}:
        bi = A_["beds_i"][s]
        L.append((2, f"A i2 {s}: " + " ".join(f"{SH[b_]} {ci(bi.get(b_))}" for b_ in BEDS[4:] if b_ in bi)
                  + f" | {nrange([bi.get(b_) for b_ in BEDS[4:]])}"))
    for s in A_.get("beds_ii") or {}:
        bi = A_["beds_ii"][s]
        L.append((2, f"A ii {s}: " + " ".join(f"{SH[b_]} {ci(bi.get(b_))}" for b_ in CORE if b_ in bi)
                  + f" | {nrange([bi.get(b_) for b_ in CORE])}"))
    for s in A_.get("within_sib") or {}:
        bi = A_["within_sib"][s]
        L.append((2, f"A sib {s}: " + " ".join(f"{SH[b_]} {ci(bi.get(b_))}" for b_ in CORE if b_ in bi)
                  + f" | pairs {g(bi, 'pwv', 'n_pairs')}"))
    for s in A_.get("joint_ii") or {}:
        j = A_["joint_ii"][s]
        if not isinstance(j, dict) or not j.get("diff"):
            continue
        L.append((2, f"J ii {s} n {j.get('n')} p {pstr(g(j, 'heterogeneity', 'p'))}: {jline(j)}"))
    for s in cov:
        c = cov[s]
        L.append((2, f"cov {s} {g(w, s, 'pgs_id')} | rows in {c.get('rows_in')} | matched {c.get('matched')} | "
                     f"{v3(c.get('coverage'))} | direct {c.get('direct')} swap {c.get('swap')} flip {c.get('flip')} pal "
                     f"{c.get('pal_kept')}/{c.get('pal_dropped')} | sha {'ok' if g(w, s, 'sha256_ok') else 'DIFF'}"))
    pc = A_.get("positive_controls") or {}
    L.append((2, "controls (score -> own trait, model i): " + " | ".join(
        f"{s}->{c.get('outcome')} {ci(c)} dR2 {v3(c.get('delta_r2'))} n {c.get('n')}" for s, c in pc.items())))
    for s in A_.get("bp_part") or {}:
        r = A_["bp_part"][s]
        L.append((3, f"BP part {s} (i - ii, model-ii sample): " + " ".join(
            f"{SH[b_]} {ci(g(r, b_, 'diff'))}" for b_ in CORE)))
    for s in A_.get("other") or {}:
        o = A_["other"][s]
        L.append((3, f"other {s}: " + " ".join(f"{y} {ci(o.get(y))}" for y in EXTRA if y in o)))
    return L


def dg_b(B_):
    L = []
    if not isinstance(B_, dict):
        return L
    if B_.get("error"):
        return [(1, f"ERROR B {B_['error']}")]
    pv_ = B_.get("prevent") or {}
    q = pv_.get("risk_q") or {}
    L.append((1, f"B PREVENT base 10-y total CVD (Khan 2024), baseline, ages 30-79: n {pv_.get('n')} (baseline "
                 f"{pv_.get('baseline')}, outside age {pv_.get('outside_age_30_79')}, missing {kvs(pv_.get('missing_input'))}; "
                 f"clipped {kvs(pv_.get('clipped'))}) | risk q05/25/50/75/95 {v3(q.get('q05'))}/{v3(q.get('q25'))}/"
                 f"{v3(q.get('q50'))}/{v3(q.get('q75'))}/{v3(q.get('q95'))} | % {kvs(pv_.get('categories'))} | TC "
                 f"{kvs(pv_.get('tc_source'))} {pv_.get('tc_unit')}; shares {kvs({k: v3(v) for k, v in (pv_.get('inputs_share') or {}).items()})}"))
    L.append((1, "B top decile (adverse, age/sex-residualised; expected 0.1) by PREVENT stratum low <5% / int 5-7.5% / high "
                 ">=7.5%: n, top n, share; AUC of risk for top decile"))
    for bed in BEDS:
        r = g(B_, "beds", bed) or {}
        if not r.get("strata"):
            L.append((2, f"B {SH[bed]} {r.get('status', 'NA')} n {r.get('n')}"))
            continue
        L.append((1 if bed in CORE else 2, f"B {SH[bed]} n {r.get('n')}: " + " | ".join(
            f"{k} {g(r, 'strata', k, 'n')} top {g(r, 'strata', k, 'top')} {v3(g(r, 'strata', k, 'share_top'))}"
            for k, _, _ in STRATA) + f" | AUC {v3(r.get('auc_risk_for_top'))}"))
    a = B_.get("any_core") or {}
    cb = a.get("count_by_stratum") or {}
    L.append((1, f"B adverse core-bed count 0/1/>=2 by stratum (n {a.get('n')}): " + " | ".join(
        f"{k} {g(cb, k, '0')}/{g(cb, k, '1')}/{g(cb, k, 'ge2')}" for k, _, _ in STRATA)))
    return L


def dg_p10(R):
    p10 = R.get("P10")
    if not isinstance(p10, dict) or p10.get("error"):
        return []
    L = []
    for oc in P10_OUT:
        r = p10.get(oc) or {}
        ra = r.get("risk_alone") or {}
        L.append((2, f"B P10 {oc} cohort {r.get('cohort')}: PREVENT lp alone {hzs(ra, 'prevent_lp')} {ra.get('n')}/"
                     f"{ra.get('events')} | given PREVENT: " + " ".join(
                         f"{SH[b_]} {hzs(r.get(b_), 'o_' + b_)} {g(r, b_, 'n')}/{g(r, b_, 'events')}" for b_ in CORE + ["c3"])))
    return L


def digest(R):
    mods = R.get("modules") or {}
    A_ = R.get("A") or {}
    rp = A_.get("repro") or {}
    fw = A_.get("flow") or {}
    mi = A_.get("matrix_i") or {}
    w = A_.get("weights") or {}
    nw = sum(1 for v in w.values() if v.get("sha256_ok"))
    L = [(1, f"F1 OUTCOME LIABILITY v1 DIGEST {R.get('version')} | atlas {R.get('atlas_version')} {mods.get('atlas_sha16')} "
             f"{'ok' if mods.get('atlas_ok') else 'DIFF'} | prs {mods.get('prs_version')} {mods.get('prs_sha16')} "
             f"{'ok' if mods.get('prs_ok') else 'DIFF'} | weights {nw}/{len(w)} sha ok | P10 {R.get('p10_status', 'NA')} | seed "
             f"{R.get('seed')} perms {mi.get('perms', 'NA')} | sections {','.join(R.get('sections') or [])} | runtime "
             f"{v3(R.get('runtime_s'))}s"),
         (1, "counts rounded to 10, None = <20; beta per SD of score in SD of the adverse-oriented bed (ret = retinal artery "
             "width and abi per SD lower); unrelated set; every regression n >= 200, within-sib >= 200 pairs; shares from "
             "rounded counts; PREVENT strata low <5% int 5-7.5% high >=7.5%"),
         (1, f"check: liver tables read 0 (refused {len(R.get('liver_blocked') or [])}); frame rows {R.get('frame_rows')}; "
             + (f"flow fam {fw.get('fam')} with PCs/QC {fw.get('with_pcs_and_qc')} unrelated {fw.get('unrelated')} full-sib "
                f"pairs {fw.get('full_sib_pairs')}; matrix n {mi.get('n')}; repro PRS v1.1 T2D_gw (PRS:15,51,69,105) coverage "
                f"{'ok' if rp.get('coverage') else 'DIFF'} pwv i {ci(g(A_, 'beds_i', 'T2D', 'pwv'))} "
                f"{'ok' if rp.get('pwv_i') else 'DIFF'} pwv ii {'ok' if rp.get('pwv_ii') else 'DIFF'} joint n/contrast/Wald p "
                f"{'ok' if rp.get('joint') else 'DIFF'}; scoring {v3(A_.get('timings_scoring_s'))}s matrix "
                f"{v3(A_.get('timings_matrix_s'))}s" if A_ else "section A not run"))]
    if R.get("fatal"):
        L.append((1, f"FATAL {R['fatal']}"))
    L += decisions(R)
    L += dg_a(R.get("A"))
    L += dg_b(R.get("B"))
    L += dg_p10(R)
    L.append((3, "timings s: " + " ".join(f"{k} {v3(v)}" for k, v in (R.get("timings") or {}).items())))
    return "\n".join(shrink(wrap(L))) + "\n"


# ============================================================================ main

SECTIONS = ("A", "B")
ATLAS_FILE = None


def main():
    global ATLAS_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="HPP dataset root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", help="outcome PGS weights folder (needed for section A)")
    ap.add_argument("--atlas", default=os.path.join(HERE, "f1_atlas_v1_5.py"))
    ap.add_argument("--prs", default=os.path.join(HERE, "f1_prs_v1_1.py"))
    ap.add_argument("--p10", default=os.path.join(HERE, "f1_atlas_outcomes_v2.py"))
    ap.add_argument("--outcomes", default=os.path.join(HERE, "f1_outcomes_v1.py"))
    ap.add_argument("--seed", type=int, default=20261006, help="permutation seed")
    ap.add_argument("--p1-seed", type=int, default=20261004, help="P1 seed for the P10 add-on (usual-BP folds)")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--block", type=int, default=20000, help="max variant rows per genotype read (PRS v1.1)")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    if not os.path.isdir(a.root):
        raise SystemExit(f"dataset root not found: {a.root}")
    secs = [s.strip().upper() for s in a.sections.split(",") if s.strip()]
    for p_ in (a.atlas, a.prs):
        if not os.path.exists(p_):
            raise SystemExit(f"module not found: {p_}")
    if "A" in secs and not (a.weights and os.path.isdir(a.weights)):
        raise SystemExit("section A needs --weights <folder>")
    ATLAS_FILE = os.path.abspath(a.atlas)
    load_modules(ATLAS_FILE, os.path.abspath(a.prs))
    t0 = time.time()
    sa, sp = sha_file(a.atlas), sha_file(a.prs)
    mods = {"atlas_sha16": hfmt(sa), "prs_sha16": hfmt(sp), "prs_version": vq(P.VERSION),
            "atlas_ok": sa == EXPECT["atlas"], "prs_ok": sp == EXPECT["prs"]}
    log(f"{VERSION}; atlas {vq(A.VERSION)} {mods['atlas_sha16']}; prs {vq(P.VERSION)} {mods['prs_sha16']}; numpy "
        f"{np.__version__}; pandas {pd.__version__}; python {sys.version.split()[0]}")
    if not (mods["atlas_ok"] and mods["prs_ok"]):
        log("WARNING atlas or PRS module is not the expected version (digest header shows DIFF)")
    res = {"version": VERSION, "atlas_version": vq(A.VERSION), "modules": mods, "seed": f"{a.seed:,}",
           "sections": secs, "p10_status": "not reached" if "B" in secs else "not run"}
    pids = set()
    try:
        fr = A.build_frame(a.root, None)[0].reset_index(drop=True)
        res["frame_rows"] = rc(len(fr))
        pids |= set(fr["participant_id"].astype(str))
        if "A" in secs:
            t1 = time.time()
            RA, ids = section_a(a.root, fr, a.weights, a.block, a.perms, a.seed)
            pids |= ids
            res["A"] = RA
            res.setdefault("timings", {})["A"] = time.time() - t1
            log(f"A done in {time.time() - t1:.1f}s")
        if "B" in secs:
            t1 = time.time()
            RB = {}
            guard(RB, "B", lambda: section_b(a.root, fr))
            risk = RB["B"].pop("_risk", None) if isinstance(RB["B"], dict) else None
            res["B"] = RB["B"]
            res.setdefault("timings", {})["B"] = time.time() - t1
            st = "absent"
            try:
                st = load_p10(os.path.abspath(a.p10), os.path.abspath(a.outcomes))
            except Exception as ex:
                st = f"load error {err(ex)[:60]}"
            res["p10_status"] = st
            if st == "ok" and risk is not None:
                t2 = time.time()
                guard(res, "P10", lambda: section_p10(a.root, fr, risk, a.p1_seed))
                log(f"P10 add-on done in {time.time() - t2:.1f}s")
            elif st == "ok":
                res["P10"] = {"error": "no PREVENT risk"}
            log(f"B done in {time.time() - t1:.1f}s (P10 {st})")
    except Exception as ex:
        res["fatal"] = err(ex)
        log(f"FATAL {res['fatal']}")
    res["liver_blocked"] = sorted(set(re.sub(r"\d{5,}", "#", b) for b in BLOCKED))
    res["runtime_s"] = time.time() - t0
    text = json.dumps(clean(res), ensure_ascii=False, sort_keys=False, indent=1)
    privacy_check(text, pids)
    try:
        dg = digest(json.loads(text))
    except Exception as ex:  # a digest failure must not lose results.json
        dg = f"F1 OUTCOME LIABILITY v1 DIGEST {VERSION}\nDIGEST FAILED: {err(ex)}; see results.json\n"
        log(f"digest FAILED: {err(ex)}")
    privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines (max {max(len(x) for x in dg.splitlines())} chars); total {time.time() - t0:.1f}s")
    logtext = "\n".join(LONG.sub("#", vq(x)) for x in A.LOG) + "\n"
    privacy_check(logtext, pids)
    os.makedirs(a.out)
    sums = []
    for fn_, tx in (("results.json", text), ("digest.txt", dg), ("run_log.txt", logtext)):
        with open(os.path.join(a.out, fn_), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn_}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s[:16] + "  " + s.split()[-1] for s in sums))


if __name__ == "__main__":
    main()
