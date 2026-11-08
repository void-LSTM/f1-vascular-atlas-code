#!/usr/bin/env python3
"""F1 claim-robustness bundle P7 v1 (HPP sandbox; writes aggregates only).

Robustness checks for F1 claims C2, C5, C7, C8, C10, C11 and C13 (docs/NOVELTY_AND_NEW_FINDINGS_20261004.md, section 6.2
item 7).  Beds come from the atlas frame (--atlas, default f1_atlas_v1_5.py beside this file; build_frame(root, None)):
calendar-, clinic-day- and within-day-corrected as in atlas v1.5 (raw kept as <bed>_raw) and adverse-oriented (atlas
ORIENT).  Loaders are reused, not re-implemented: exposures from the exposome v1 builders (--exposome f1_exposome_v1.py:
assemble), the oral layer from omics v1.2 (--omics f1_omics_v1_2.py: load_oral, match, rows_for, covmat, mols), relation
pairs and person tables from family v1.1 (--family f1_family_v1_1.py: load_pairs, person_table, pair_r).

C7  Exposome bed heterogeneity in the common four-bed sample (3-df robust Wald test of equal oriented betas, BH within
    model; numerically the exposome v1 X1 joint4 test, vectorised by missingness pattern).  (a) Permutation null of the
    heterogeneity COUNT: >= 200 Freedman-Lane permutations (residual rows of the four beds from the covariates-only model,
    permuted jointly within sex x age-decile strata; exposures untouched, so their mutual dependence is kept; null = no
    conditional association with any bed), M1, M2, M3.  (b) Body-size model M2B = M2 + height + DXA total lean + leg lean
    mass + DXA legs bone area (leg length is not in the HPP schema: height is the device's path-length basis and the
    legs bone area is the leg-size proxy), against M2 in the same rows.
C8  Proportionality (rank) test of the exposure x bed coefficient matrix (pre-listed exposures x 4 beds; one model per
    exposure; M1 age + sex, M3 + BMI + SBP + antihypertensives; oriented beds z-scored once in the common sample).
    Minimum-distance statistic Q_r = min (b - vec(U V'))' W (b - vec(U V')), W = inverse joint covariance of all
    coefficients from stacked person-level influence functions; chi2 with (p - r)(q - r) df and a person-level
    bootstrap calibration (resamples recentred at the fitted rank-r matrix).  Rank-1 bed gains (PWV = 1) and the share
    of the weighted coefficient signal captured by rank 1, with bootstrap CIs; rank 1 without the age row; per-exposure
    departures: dQ = drop of the rank-1 Q when the exposure is left out (~chi2, 3 df) and the exposure's residual from
    the bed gains fitted to the other exposures (bootstrap CIs for the 8 largest dQ, M1).
C13 Age equivalents (bed ~ age + sex + exposure [+ SBP + antihypertensives]; years per stated contrast) for PWV, IMT and
    retinal artery width; PWV:IMT and PWV:retina ratios with paired-bootstrap CIs (the C8 resamples); two-bed rank-1
    tests (PWV-IMT, PWV-retina, all exposures except age) = homogeneity of the ratios across exposures.
C11 Overlap of age/sex-standardised adverse top deciles in the common sample (Jaccard per pair, union, number of beds)
    with bootstrap CIs; Gaussian-copula expectation from the observed correlations; measurement ceilings: same-bed
    retest (visit 1 vs 2) and left/right Jaccard and correlations, and the overlap expected if all beds shared one
    factor measured with that reliability (bivariate normal, Monte Carlo for the union); position of the observed
    overlap between independence (0) and the ceiling (1), with CIs that carry both the cross-sectional bootstrap and
    the bootstrap of the replicate correlations; overlap conditional on sitting SBP, DBP, antihypertensives (and BMI).
C5  Pre-stated oral taxa (P. gingivalis, T. forsythia, T. denticola, F. nucleatum; CLR as omics v1.2) x core beds:
    M3 (omics v1.2 model), then in one common sample M3, + smoking status (current, former; never = reference),
    + sitting HR, + waist and DXA VAT; M3 in never-smokers; taxa -> sitting SBP and HR; taxa by smoking status.
C2  (a) Leave-pair-out (LPO) correction: for same-day pairs (cohabitants, siblings, unrelated people) each member's
    day + session artefact (atlas v1.5 parameters) is re-estimated from the day's other measurements excluding both
    members (the atlas correction is leave-one-out, so a partner measured that day enters the correction); pair r
    and the cohabitant minus time-gap-matched stranger excess under LOO and LPO, CIs by bootstrap over days.
    (b) Calendar-gap curve: pair r by |days| between the two visits (0, 1, 2-7, 8-30, 31-180, >180) for cohabitants,
    siblings and unrelated pairs (raw PWV, corrected PWV, sitting SBP).
C10 Within-person change (exposome X3 design): d(bed)/SD(baseline bed) per SD of d(exposure) | baseline age, sex,
    follow-up, baseline bed (A), + d(SBP) (B); TOST equivalence of d(BMI), d(waist), d(VAT) at margins 0.05 and 0.10
    SD per SD and the smallest margin shown; all pairs and untreated at both visits.
Privacy: aggregates only; counts rounded to 10 (None below 20); estimates need n >= 100 (binary >= 50 per level, pairs
>= 50); no participant ids or dates; every output passes the atlas privacy scan (7+ digit runs, participant ids).
Run (pheno env, from the folder with the modules): OMP_NUM_THREADS=4 python f1_robust_v1.py --root <hpp_datasets>
    --out <new dir> --atlas f1_atlas_v1_5.py --exposome f1_exposome_v1.py --omics f1_omics_v1_2.py --family f1_family_v1_1.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import stats

VERSION = "f1_robust_v1.0.1 2026-10-04"
HERE = os.path.dirname(os.path.abspath(__file__))


def _path(flag, names):
    if flag in sys.argv[:-1]:
        return os.path.abspath(sys.argv[sys.argv.index(flag) + 1])
    for nm in names:
        if os.path.exists(os.path.join(HERE, nm)):
            return os.path.join(HERE, nm)
    return os.path.join(HERE, names[0])


ATLAS_PATH = _path("--atlas", ["f1_atlas_v1_5.py", "f1_atlas_v1.py"])
EXPO_PATH = _path("--exposome", ["f1_exposome_v1.py"])
OMICS_PATH = _path("--omics", ["f1_omics_v1_2.py", "f1_omics_v1.py"])
FAMILY_PATH = _path("--family", ["f1_family_v1_1.py", "f1_family_v1.py"])


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m  # module functions must be importable by name for multiprocessing (exposome CGM pool)
    spec.loader.exec_module(m)
    return m


A = _load(ATLAS_PATH, "f1_atlas")  # phenotype frame, privacy, regression and kernel helpers
sys.modules["f1_atlas"] = A
E = O = F = None  # exposome, omics, family modules (load_helpers(); their atlas handle is replaced by A)
LOAD_ERR, EXTRA_PIDS = {}, set()
log, rc, f3 = A.log, A.rc, A.v3
CORE = list(A.CORE)
OC = ["o_" + b for b in CORE]
MS = A.MAIN_STAGES
MIN_N, MIN_BIN, MIN_PAIRS = 100, 50, 50
PAIRS6 = [(a, b) for i, a in enumerate(CORE) for b in CORE[i + 1:]]
CC = np.array([[1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1]], float)
ORD = {"00_00_visit": 0, "01_00_call": 1, "01_01_visit": 1.5, "02_00_visit": 2, "02_01_visit": 2.5, "03_00_call": 3,
       "03_01_visit": 3.5, "04_00_visit": 4}


def load_helpers():
    """Import exposome / omics / family by path (exposome and family read --atlas from argv at import time) and point
    their atlas handle, log and count helpers at this script's atlas instance."""
    global E, O, F
    argv = sys.argv
    sys.argv = [argv[0], "--atlas", ATLAS_PATH]
    try:
        for key, path, name in (("exposome", EXPO_PATH, "f1_exposome"), ("omics", OMICS_PATH, "f1_omics"),
                                ("family", FAMILY_PATH, "f1_family")):
            try:
                m = _load(path, name)
            except BaseException as ex:  # also SystemExit from a module
                LOAD_ERR[key] = f"{os.path.basename(path)}: {type(ex).__name__}: {str(ex)[:150]}"
                continue
            m.A = A
            for k in ("log", "rc", "cnt"):
                if hasattr(m, k):
                    setattr(m, k, getattr(A, k))
            if key == "exposome":
                m.f3 = A.v3
                E = m
            elif key == "omics":
                O = m
            else:
                F = m
    finally:
        sys.argv = argv
        sys.modules["f1_atlas"] = A


# ============================================================================ helpers

def err(ex):
    tb = traceback.extract_tb(ex.__traceback__)
    return f"{type(ex).__name__}: {str(ex)[:200]} (line {tb[-1].lineno if tb else '?'})"


def guard(R, key, fn):
    t = time.time()
    try:
        R[key] = fn()
        log(f"{key} done in {time.time() - t:.1f}s")
    except Exception as ex:
        R[key] = {"error": err(ex)}
        log(f"{key} FAILED: {R[key]['error']}")
    if isinstance(R.get(key), dict):
        R[key]["seconds"] = time.time() - t


def need(mod, name):
    if mod is None:
        raise RuntimeError(f"{name} module not loaded: {LOAD_ERR.get(name, 'missing')}")
    return mod


def pn(b, se):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(2 * stats.norm.sf(abs(b / se))) if se and np.isfinite(se) and se > 0 else None


def est(b, se, n=None):
    b, se = float(b), float(se)
    if not (np.isfinite(b) and np.isfinite(se)):
        return None
    out = {"b": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": pn(b, se)}
    if n is not None:
        out["n"] = rc(n)
    return out


def bci(point, draws):
    c = A.pct(draws)
    return {"est": float(point), "ci": c} if np.isfinite(point) else None


def isbin(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return len(x) > 0 and bool(np.isin(x, (0.0, 1.0)).all())


def zr(y, age, sex, extra=None):
    """Standardised residual of y on [1, age, (age-50)^2, sex (, extra)]; NaN where anything is missing."""
    y = np.asarray(y, float)
    X = np.column_stack([np.ones(len(y)), age, (np.asarray(age, float) - 50) ** 2, sex] + ([extra] if extra is not None else []))
    m = np.isfinite(y) & np.isfinite(X).all(1)
    out = np.full(len(y), np.nan)
    if m.sum() < MIN_N:
        return out
    r = y[m] - X[m] @ A.beta_only(y[m], X[m])
    out[m] = (r - r.mean()) / r.std()
    return out


def boot_counts(rng, n):
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)


# ============================================================================ C10 within-person change: TOST

C10_BEDS = CORE + ["cavi0", "sokolow"]
C10_X = ["bmi", "waist", "log_vat", "sbp_sit"]
MARGINS = (0.05, 0.10)


def olsj(y, X, j=1):
    Gi = np.linalg.pinv(X.T @ X)
    a = (Gi @ X.T)[j]
    inf = a * (y - X @ (Gi @ (X.T @ y))) * np.sqrt(len(y) / max(len(y) - X.shape[1], 1))
    return float(a @ y), float(np.sqrt((inf ** 2).sum()))


def tost_p(b, se, d):
    return float(max(stats.norm.sf((b + d) / se), stats.norm.cdf((b - d) / se)))


def c10_tost(fr):
    s0, s1 = MS
    a = fr[fr["research_stage"] == s0].drop_duplicates("participant_id").set_index("participant_id")
    b = fr[fr["research_stage"] == s1].drop_duplicates("participant_id").set_index("participant_id")
    ids = a.index.intersection(b.index)
    a, b = a.loc[ids], b.loc[ids]
    col = lambda df, e: pd.to_numeric(df[e], errors="coerce").to_numpy(float) if e in df.columns else np.full(len(df), np.nan)
    fu = ((pd.to_datetime(b["visit_date"]) - pd.to_datetime(a["visit_date"])).dt.days / 365.25).to_numpy(float)
    age, sex = col(a, "age"), col(a, "sex_male")
    dsbp = col(b, "sbp_sit") - col(a, "sbp_sit")
    dsbp = (dsbp - np.nanmean(dsbp)) / np.nanstd(dsbp)
    unt = (col(a, "antihtn") == 0) & (col(b, "antihtn") == 0)
    out = {"n_pairs": rc(len(ids)), "n_untreated": rc(unt.sum()), "followup_years": A.qtiles(pd.Series(fu)),
           "margins": list(MARGINS), "est": {}}
    for bed in C10_BEDS:
        s = A.ORIENT.get(bed, 1)
        y0, y1 = s * col(a, bed), s * col(b, bed)
        ok = np.isfinite(y0) & np.isfinite(y1)
        if ok.sum() < MIN_N:
            continue
        sd0 = y0[ok].std()
        dy, base = (y1 - y0) / sd0, (y0 - y0[ok].mean()) / sd0
        for e in C10_X:
            dx = col(b, e) - col(a, e)
            if np.isfinite(dx).sum() < MIN_N or not np.nanstd(dx) > 0:
                continue
            dxs = dx / np.nanstd(dx)
            for mod, extra in (("A", []), ("B", [dsbp])):
                if mod == "B" and e == "sbp_sit":
                    continue
                for sub, msk in (("all", np.ones(len(ids), bool)), ("untreated", unt)):
                    X = np.column_stack([np.ones(len(ids)), dxs, age, sex, fu, base] + extra)
                    m = np.isfinite(dy) & np.isfinite(X).all(1) & msk
                    if m.sum() < MIN_N:
                        continue
                    bb, se = olsj(dy[m], X[m])
                    r = est(bb, se, m.sum())
                    if r is None:
                        continue
                    r.update(lo90=bb - 1.645 * se, hi90=bb + 1.645 * se, eq_bound=max(abs(bb - 1.645 * se), abs(bb + 1.645 * se)),
                             tost_p={f"{d:g}": tost_p(bb, se, d) for d in MARGINS})
                    out["est"][f"{e}|{bed}|{mod}|{sub}"] = r
    return out


# ============================================================================ C11 overlap of extremes

SIDES = {"pwv": ("pwv_r", "pwv_l"), "imt": ("imt_r", "imt_l"), "art_w": ("art_w_r", "art_w_l"), "abi_min": ("abi_r", "abi_l")}
TOPQ = 0.9
J_IND = 0.01 / 0.19
U_IND = 1 - 0.9 ** 4
NSIM = 400000
NSIM_B = 40000  # per-bootstrap-draw Monte Carlo of the one-factor union ceiling (common random numbers)


def tops(Zm):
    return Zm >= np.quantile(Zm, TOPQ, axis=0)[None, :]


def jac(t1, t2):
    u = (t1 | t2).sum()
    return float((t1 & t2).sum() / u) if u else np.nan


def ostats(Zm):
    """Pairwise Jaccard (PAIRS6 order), union fraction, distribution of the number of beds in the top decile."""
    T = tops(Zm)
    k = T.sum(1)
    return np.array([jac(T[:, i], T[:, j]) for i in range(4) for j in range(i + 1, 4)]), float(T.any(1).mean()), \
        np.bincount(k, minlength=5)[:5] / len(k)


def cells(Zm):
    """Small-cell flags for the point estimates: pairwise top-decile intersections and number-of-beds counts < MINCELL."""
    T = tops(Zm)
    both = np.array([(T[:, i] & T[:, j]).sum() for i in range(4) for j in range(i + 1, 4)])
    return both < A.MINCELL, np.bincount(T.sum(1), minlength=5)[:5] < A.MINCELL


def j_bvn(r):
    c = stats.norm.ppf(TOPQ)
    r = float(np.clip(r, -0.999, 0.999))
    p11 = float(stats.multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-c, -c]))
    return p11 / (2 * (1 - TOPQ) - p11)


def mc_union(R, seed):
    """Monte Carlo union and number-of-beds distribution of top deciles for a 4-variate normal with correlation R."""
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(np.asarray(R, float) + 1e-9 * np.eye(len(R)))
    T = (rng.standard_normal((NSIM, len(R))) @ Lc.T) >= stats.norm.ppf(TOPQ)
    k = T.sum(1)
    return float(T.any(1).mean()), (np.bincount(k, minlength=5)[:5] / len(k)).tolist()


def pair_z(z1, z2, boot, rng):
    ok = np.isfinite(z1) & np.isfinite(z2)
    n = int(ok.sum())
    if n < MIN_N:
        return None
    z1, z2 = z1[ok], z2[ok]
    st = lambda i: (float(np.corrcoef(z1[i], z2[i])[0, 1]), jac(z1[i] >= np.quantile(z1[i], TOPQ), z2[i] >= np.quantile(z2[i], TOPQ)))
    r, j = st(np.arange(n))
    bs = np.array([st(rng.integers(0, n, n)) for _ in range(boot)])
    small = ((z1 >= np.quantile(z1, TOPQ)) & (z2 >= np.quantile(z2, TOPQ))).sum() < A.MINCELL
    return {"n": rc(n), "r": r, "r_ci": A.pct(bs[:, 0]), "jaccard": None if small else j, "jaccard_ci": None if small else A.pct(bs[:, 1]),
            "_r_bs": bs[:, 0]}


def c11_overlap(fr, boot, seed):
    rng = np.random.default_rng([seed, 11])
    d = A.one_row_per_participant(fr, OC + ["age", "sex_male"]).reset_index(drop=True)
    n = len(d)
    out = {"n_common": rc(n), "independence": {"jaccard": J_IND, "union": U_IND}}
    if n < 200:
        return out
    age, sex = d["age"].to_numpy(float), d["sex_male"].to_numpy(float)
    Zm = np.column_stack([zr(d[c].to_numpy(float), age, sex) for c in OC])
    J, U, K = ostats(Zm)
    sj, sk = cells(Zm)
    Rm = np.corrcoef(Zm.T)
    BS = [ostats(Zm[rng.integers(0, n, n)]) for _ in range(boot)]
    Jb, Ub, Kb = np.array([b[0] for b in BS]), np.array([b[1] for b in BS]), np.array([b[2] for b in BS])
    cu, ck = mc_union(Rm, seed + 1)
    out["observed"] = {"jaccard": {f"{a}|{b}": {"est": None if sj[i] else J[i], "ci": None if sj[i] else A.pct(Jb[:, i]),
                                                 "r": float(Rm[CORE.index(a), CORE.index(b)]),
                                                 "copula": j_bvn(Rm[CORE.index(a), CORE.index(b)])} for i, (a, b) in enumerate(PAIRS6)},
                       "union": {"est": U, "ci": A.pct(Ub), "copula": cu},
                       "k_beds": {"est": [None if sk[i] else float(K[i]) for i in range(5)],
                                  "ci": [None if sk[i] else A.pct(Kb[:, i]) for i in range(5)], "copula": ck}}
    # measurement ceilings: same-bed retest (visit 1 vs 2) and left/right
    ceil = {}
    for bed in CORE:
        ob, rr = "o_" + bed, {}
        s0, s1 = [fr[fr["research_stage"] == s][["participant_id", ob, "age", "sex_male"]].dropna().drop_duplicates("participant_id")
                  .set_index("participant_id") for s in MS]
        ids = s0.index.intersection(s1.index)
        if len(ids) >= MIN_N:
            a_, b_ = s0.loc[ids], s1.loc[ids]
            rr["retest"] = pair_z(zr(a_[ob].to_numpy(float), a_["age"].to_numpy(float), a_["sex_male"].to_numpy(float)),
                                  zr(b_[ob].to_numpy(float), b_["age"].to_numpy(float), b_["sex_male"].to_numpy(float)), boot, rng)
        c1, c2 = SIDES[bed]
        if c1 in d.columns and c2 in d.columns:
            sg = A.ORIENT.get(bed, 1)
            rr["left_right"] = pair_z(zr(sg * d[c1].to_numpy(float), age, sex), zr(sg * d[c2].to_numpy(float), age, sex), boot, rng)
        ceil[bed] = rr
    out["ceilings"] = {b_: {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in rr.items() if v}
                       for b_, rr in ceil.items()}
    # one-factor ceiling; its CIs carry both the cross-sectional bootstrap (Jb, Ub) and the bootstrap of the replicate
    # correlations (paired by draw index; the replicate samples are resampled independently)
    rgrid = np.linspace(0.0, 0.999, 400)
    jgrid = np.array([j_bvn(x) for x in rgrid])
    crn = np.random.default_rng([seed, 12])
    Fz, Ez, cth = crn.standard_normal(NSIM_B), crn.standard_normal((NSIM_B, 4)), stats.norm.ppf(TOPQ)
    rho = {}
    for src in ("retest", "left_right"):
        rv = {}
        for bed in CORE:
            own = ceil[bed].get(src)
            c = own or (ceil[bed].get("left_right") if src == "retest" else None)
            if c and c.get("r") is not None and c["r"] > 0:
                rb_ = np.asarray(c["_r_bs"], float)
                rv[bed] = (float(c["r"]), src if own else "left_right", np.clip(np.where(np.isfinite(rb_), rb_, c["r"]), 1e-6, 0.999))
        if len(rv) < 4:
            continue
        Rc = np.ones((4, 4))
        for i, a in enumerate(CORE):
            for j, b in enumerate(CORE):
                if i != j:
                    Rc[i, j] = np.sqrt(rv[a][0] * rv[b][0])
        uc, kc = mc_union(Rc, seed + 2)
        nb = min(len(Jb), min(len(v[2]) for v in rv.values()))
        rhob = np.column_stack([rv[b_][2][:nb] for b_ in CORE])  # draws x beds
        ucb = np.array([((np.sqrt(rr)[None, :] * Fz[:, None] + np.sqrt(1 - rr)[None, :] * Ez) >= cth).any(1).mean() for rr in rhob])
        pp = {}
        for i, (a, b) in enumerate(PAIRS6):
            ia, ib = CORE.index(a), CORE.index(b)
            jc = j_bvn(Rc[ia, ib])
            jcb = np.interp(np.sqrt(rhob[:, ia] * rhob[:, ib]), rgrid, jgrid)
            den, denb = jc - J_IND, jcb - J_IND
            ok_ = den > 1e-6 and not sj[i]
            with np.errstate(divide="ignore", invalid="ignore"):
                pos_b = np.where(denb > 1e-6, (Jb[:nb, i] - J_IND) / denb, np.nan)
            pp[f"{a}|{b}"] = {"r_ceiling": float(Rc[ia, ib]), "jaccard_ceiling": jc, "jaccard_ceiling_ci": A.pct(jcb),
                              "position": bci((J[i] - J_IND) / den, pos_b) if ok_ else None,
                              "position_ci_jaccard_only": A.pct((Jb[:, i] - J_IND) / den) if ok_ else None}
        den = U_IND - uc
        with np.errstate(divide="ignore", invalid="ignore"):
            upb = np.where(U_IND - ucb > 1e-6, (U_IND - Ub[:nb]) / (U_IND - ucb), np.nan)
        rho[src] = {"rho": {b: v[0] for b, v in rv.items()}, "rho_source": {b: v[1] for b, v in rv.items()}, "pairs": pp,
                    "union_ceiling": uc, "union_ceiling_ci": A.pct(ucb), "k_beds_ceiling": kc,
                    "union_position": bci((U_IND - U) / den, upb) if den > 1e-6 else None,
                    "union_position_ci_union_only": A.pct((U_IND - Ub) / den) if den > 1e-6 else None}
    out["single_factor"] = rho
    # overlap conditional on blood pressure (and BMI)
    cond = {}
    for nm, extra in (("bp", ["sbp_sit", "dbp_sit", "antihtn"]), ("bp_bmi", ["sbp_sit", "dbp_sit", "antihtn", "bmi"])):
        if any(c not in d.columns for c in extra):
            continue
        m = np.isfinite(d[extra].to_numpy(float)).all(1)
        if m.sum() < 200:
            continue
        dd = d[m]
        a2, s2, Xe = dd["age"].to_numpy(float), dd["sex_male"].to_numpy(float), dd[extra].to_numpy(float)
        Z0 = np.column_stack([zr(dd[c].to_numpy(float), a2, s2) for c in OC])
        Z1 = np.column_stack([zr(dd[c].to_numpy(float), a2, s2, Xe) for c in OC])
        res = {"n": rc(m.sum())}
        for tag, Zx in (("age_sex", Z0), ("conditional", Z1)):
            Jx, Ux, _ = ostats(Zx)
            sx_ = cells(Zx)[0]
            bs = [ostats(Zx[i]) for i in (rng.integers(0, len(Zx), len(Zx)) for _ in range(boot))]
            Rx = np.corrcoef(Zx.T)
            res[tag] = {"jaccard": {f"{a}|{b}": {"est": None if sx_[i] else Jx[i], "ci": None if sx_[i] else A.pct([x[0][i] for x in bs]),
                                                 "r": float(Rx[CORE.index(a), CORE.index(b)]),
                                                 "copula": j_bvn(Rx[CORE.index(a), CORE.index(b)])} for i, (a, b) in enumerate(PAIRS6)},
                        "union": {"est": Ux, "ci": A.pct([x[1] for x in bs])}}
        cond[nm] = res
    out["conditional"] = cond
    return out


# ============================================================================ C8 proportionality + C13 age equivalents

RANK_X = ["age", "bmi", "waist", "log_vat", "leg_fat", "leg_lean", "height", "log1p_ahi", "log1p_t90", "spo2_mean", "log_tg",
          "ldl", "hdl", "hba1c", "cgm_mean", "sbp_sit", "dbp_sit", "hr_sit", "urate", "egfr", "smk_current"]
RANK_MODELS = {"M1": ["age", "sex_male"], "M3": ["age", "sex_male", "bmi", "sbp_sit", "antihtn"]}
C13_X = [("sbp_sit", 10.0), ("log_vat", None), ("waist", 10.0), ("bmi", 5.0), ("ldl", "ldl"), ("log_tg", None),
         ("hba1c", 0.5), ("log1p_ahi", None), ("log1p_t90", None)]
C13_BEDS = ["pwv", "imt", "art_w"]


def rank_sets(d, exps, covs):
    """Per exposure: rows with the exposure and covariates, design [1, x (per SD; binary 0/1), covariates]."""
    sets = []
    for e in exps:
        if e not in d.columns:
            continue
        cv = [c for c in covs if c != e and c in d.columns]
        V = d[[e] + cv].to_numpy(float)
        m = np.isfinite(V).all(1)
        if m.sum() < 200:
            continue
        x = V[m, 0]
        if isbin(x):
            if min(x.sum(), len(x) - x.sum()) < MIN_BIN:
                continue
        elif not x.std() > 0:
            continue
        else:
            x = (x - x.mean()) / x.std()
        Cv = V[m, 1:]
        Cv = Cv[:, Cv.std(0) > 0]
        X = np.column_stack([np.ones(m.sum()), x, (Cv - Cv.mean(0)) / Cv.std(0)])
        C1 = np.delete(X, 1, axis=1)
        rx = x - C1 @ np.linalg.lstsq(C1, x, rcond=None)[0]
        if (rx ** 2).sum() <= 1e-8 * ((x - x.mean()) ** 2).sum():  # collinear with the covariates
            continue
        sets.append((e, np.flatnonzero(m), X))
    return sets


def rank_fit(sets, Y, w=None):
    B = np.empty((len(sets), Y.shape[1]))
    for j, (_, idx, X) in enumerate(sets):
        Xw = X if w is None else X * w[idx, None]
        B[j] = np.linalg.lstsq(Xw.T @ X, Xw.T @ Y[idx], rcond=None)[0][1]
    return B


def rank_cov(sets, Y):
    """Joint covariance of all exposure x bed coefficients from stacked person-level influence functions (HC1)."""
    n, q = Y.shape
    IF = np.zeros((n, q * len(sets)))
    for j, (_, idx, X) in enumerate(sets):
        G = np.linalg.pinv(X.T @ X)
        Ej = Y[idx] - X @ (G @ (X.T @ Y[idx]))
        a = (G @ X.T)[1]
        IF[idx, q * j:q * j + q] = a[:, None] * Ej * np.sqrt(len(idx) / max(len(idx) - X.shape[1], 1))
    return IF.T @ IF


def md_fit(b, W, p, q, r, V0=None, iters=300, tol=1e-9):
    """Minimum-distance rank-r fit: min (b - vec(U V'))' W (b - vec(U V')), b row-major (exposure, bed); alternating GLS
    (closed-form normal equations on the (p, q, p, q) blocks of W)."""
    if r == 0:
        return float(b @ W @ b), None, None
    B = b.reshape(p, q)
    W4 = W.reshape(p, q, p, q)
    V = np.linalg.svd(B, full_matrices=False)[2][:r].T if V0 is None else V0.copy()
    Qold, Q, U = np.inf, np.inf, None
    for _ in range(iters):
        G = np.einsum("jakb,as,bt->jskt", W4, V, V).reshape(p * r, p * r)
        U = np.linalg.lstsq(G, np.einsum("jakb,as,kb->js", W4, V, B).ravel(), rcond=None)[0].reshape(p, r)
        G = np.einsum("jakb,js,kt->satb", W4, U, U).reshape(r * q, r * q)
        V = np.linalg.lstsq(G, np.einsum("jakb,js,kb->sa", W4, U, B).ravel(), rcond=None)[0].reshape(r, q).T
        res = b - (U @ V.T).ravel()
        Q = float(res @ W @ res)
        if Qold - Q < tol * (1 + Q):
            break
        Qold = Q
    return Q, U, V


def norm1(U, V):
    s = V[0, 0] if abs(V[0, 0]) > 1e-12 else 1.0
    return U * s, V / s


def c13_designs(d):
    ldl_unit = 38.67 if "ldl" in d.columns and d["ldl"].median(skipna=True) > 20 else 1.0  # 1 mmol/L in mg/dL units
    out = []
    for e, delta in C13_X:
        if e not in d.columns:
            continue
        for cond, extra in (("marg", []), ("bp", ["sbp_sit", "antihtn"])):
            if cond == "bp" and e == "sbp_sit":
                continue
            cols = ["age", "sex_male", e] + extra
            if any(c not in d.columns for c in cols):
                continue
            V = d[cols].to_numpy(float)
            m = np.isfinite(V).all(1)
            if m.sum() < 200 or not V[m, 2].std() > 0:
                continue
            dl = ldl_unit if delta == "ldl" else (float(V[m, 2].std()) if delta is None else float(delta))
            out.append((e, cond, dl, "1 SD" if delta is None else dl, np.flatnonzero(m), np.column_stack([np.ones(m.sum()), V[m]])))
    return out


def c13_years(des, Y3, w=None):
    """Years of age per contrast: rows = designs, columns = beds (b_x * delta / b_age)."""
    out = np.full((len(des), Y3.shape[1]), np.nan)
    for i, (_, _, dl, _, idx, X) in enumerate(des):
        Xw = X if w is None else X * w[idx, None]
        Bt = np.linalg.lstsq(Xw.T @ X, Xw.T @ Y3[idx], rcond=None)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            out[i] = Bt[3] * dl / Bt[1]
    return out


def sub_md(B, S, rows, cols, r=1, V0=None):
    """Rank-r MD fit of the sub-matrix B[rows][:, cols] with W = inverse of the matching block of S."""
    q = B.shape[1]
    ix = np.array([[q * j + k for k in cols] for j in rows]).ravel()
    W_ = np.linalg.pinv(S[np.ix_(ix, ix)], rcond=1e-10, hermitian=True)
    b_ = B[rows][:, cols].ravel()
    Q, U, V = md_fit(b_, W_, len(rows), len(cols), r, V0=V0)
    return {"rows": rows, "cols": cols, "W": W_, "b": b_, "Q": Q, "U": U, "V": V, "fit": (U @ V.T).ravel()}


def row_resid(bj, Sj, V):
    """Departure of one exposure row from bed gains V (q x 1) of the other rows: b_j - u_j V, u_j by GLS."""
    Wj = np.linalg.pinv(Sj)
    v = V[:, 0]
    u = float(v @ Wj @ bj) / float(v @ Wj @ v)
    return bj - u * v


def c8_rank(fr, boot, seed):
    rng = np.random.default_rng([seed, 8])
    d = A.one_row_per_participant(fr, OC + ["age", "sex_male"]).reset_index(drop=True)
    n = len(d)
    out = {"n_common": rc(n), "exposures_listed": RANK_X, "boot": boot, "models": {}}
    if n < 200:
        return out
    Y = d[OC].to_numpy(float)
    Y = (Y - Y.mean(0)) / Y.std(0)
    q, allc = 4, [0, 1, 2, 3]
    Ms = {}
    for m, covs in RANK_MODELS.items():
        sets = rank_sets(d, RANK_X, covs)
        p = len(sets)
        if p < 4:
            continue
        B = rank_fit(sets, Y)
        b = B.ravel()
        S = rank_cov(sets, Y)
        W = np.linalg.pinv(S, rcond=1e-10, hermitian=True)
        Q0 = float(b @ W @ b)
        Q1, U1, V1 = md_fit(b, W, p, q, 1)
        U1, V1 = norm1(U1, V1)
        Q2, U2, V2 = md_fit(b, W, p, q, 2)
        names = [s[0] for s in sets]
        noage = [j for j, e in enumerate(names) if e != "age"]
        sub = {"no_age": sub_md(B, S, noage, allc, V0=V1)}
        for k_ in (1, 2):  # two-bed (PWV with IMT / retina) rank-1 tests over all exposures except age (C13 ratio homogeneity)
            sub[CORE[k_]] = sub_md(B, S, noage, [0, k_])
        dq, loo, loo_t = {}, {}, {}
        for j, e in enumerate(names):  # leave-one-exposure-out: drop in rank-1 Q, and the row's departure from the others' gains
            t = sub_md(B, S, [k for k in range(p) if k != j], allc, V0=V1)
            dq[e] = Q1 - t["Q"]
            loo[e], loo_t[e] = t["V"], t
        top = sorted(names, key=lambda e: -dq[e])[:8] if m == "M1" else []
        Ms[m] = {"sets": sets, "names": names, "p": p, "B": B, "b": b, "S": S, "W": W, "Q": (Q0, Q1, Q2), "U1": U1, "V1": V1, "V2": V2,
                 "fit1": (U1 @ V1.T).ravel(), "fit2": (U2 @ V2.T).ravel(), "sub": sub, "dq": dq, "loo": loo, "loo_t": loo_t, "top": top,
                 "bs": {"Q1": [], "Q2": [], "share": [], "gain": [], "sub": {k: [] for k in sub}, "ratio": {k: [] for k in sub if k != "no_age"},
                        "res": {e: [] for e in top}}}
    i3 = [CORE.index(b_) for b_ in C13_BEDS]
    Y3 = Y[:, i3]
    des = c13_designs(d)
    yrs0 = c13_years(des, Y3)
    yb = []
    for _ in range(boot):
        w = boot_counts(rng, n)
        for m, M in Ms.items():
            p, W, bs = M["p"], M["W"], M["bs"]
            Bb = rank_fit(M["sets"], Y, w)
            bb = Bb.ravel()
            db = bb - M["b"]
            bs["Q1"].append(md_fit(M["fit1"] + db, W, p, q, 1, V0=M["V1"])[0])
            bs["Q2"].append(md_fit(M["fit2"] + db, W, p, q, 2, V0=M["V2"], iters=200)[0])
            Qb, Ub, Vb = md_fit(bb, W, p, q, 1, V0=M["V1"])
            Ub, Vb = norm1(Ub, Vb)
            bs["share"].append(1 - Qb / float(bb @ W @ bb))
            bs["gain"].append(Vb[:, 0])
            for k_, t in M["sub"].items():
                br = Bb[t["rows"]][:, t["cols"]].ravel()
                bs["sub"][k_].append(md_fit(t["fit"] + br - t["b"], t["W"], len(t["rows"]), len(t["cols"]), 1, V0=t["V"])[0])
                if k_ != "no_age":
                    Vr = md_fit(br, t["W"], len(t["rows"]), len(t["cols"]), 1, V0=t["V"])[2]
                    bs["ratio"][k_].append(Vr[1, 0] / Vr[0, 0] if abs(Vr[0, 0]) > 1e-12 else np.nan)
            for e in M["top"]:
                j, t = M["names"].index(e), M["loo_t"][e]
                Vj = md_fit(Bb[t["rows"]].ravel(), t["W"], len(t["rows"]), q, 1, V0=t["V"])[2]
                bs["res"][e].append(row_resid(Bb[j], M["S"][q * j:q * j + q, q * j:q * j + q], Vj))
        yb.append(c13_years(des, Y3, w))
    pb = lambda qs, Q: float((1 + (np.asarray(qs) >= Q).sum()) / (1 + len(qs))) if len(qs) else None
    for m, M in Ms.items():
        p, (Q0, Q1, Q2), bs = M["p"], M["Q"], M["bs"]
        df1, df2 = (p - 1) * (q - 1), (p - 2) * (q - 2)
        gain = np.array(bs["gain"]) if bs["gain"] else np.empty((0, q))
        dep = {}
        for j, e in enumerate(M["names"]):
            Sj = M["S"][q * j:q * j + q, q * j:q * j + q]
            r0 = row_resid(M["B"][j], Sj, M["loo"][e])
            v = M["loo"][e][:, 0]
            dep[e] = {"b": M["B"][j].tolist(), "dQ": float(M["dq"][e]), "p_dQ": float(stats.chi2.sf(max(M["dq"][e], 0.0), q - 1)),
                      "fit_from_others": (M["B"][j] - r0).tolist(), "gains_without": (v / v[0] if abs(v[0]) > 1e-12 else v).tolist()}
            if e in bs["res"]:
                rr = np.array(bs["res"][e]) if bs["res"][e] else np.empty((0, q))
                dep[e]["resid"] = {b_: bci(r0[i], rr[:, i] if len(rr) else []) for i, b_ in enumerate(CORE)}
        sub = {}
        for k_, t in M["sub"].items():
            dfk = (len(t["rows"]) - 1) * (len(t["cols"]) - 1)
            sub[k_] = {"Q": t["Q"], "df": dfk, "p_chi2": float(stats.chi2.sf(t["Q"], dfk)), "p_boot": pb(bs["sub"][k_], t["Q"])}
            if k_ != "no_age":
                Vt = t["V"]
                sub[k_]["ratio_common"] = bci(Vt[1, 0] / Vt[0, 0] if abs(Vt[0, 0]) > 1e-12 else np.nan, bs["ratio"][k_])
        out["models"][m] = {
            "covariates": RANK_MODELS[m], "exposures": M["names"], "p": p, "q": q,
            "n_per_exposure": {e: rc(len(s[1])) for e, s in zip(M["names"], M["sets"])},
            "Q0": Q0, "rank1": {"Q": Q1, "df": df1, "p_chi2": float(stats.chi2.sf(Q1, df1)), "p_boot": pb(bs["Q1"], Q1),
                                "null_boot_mean": float(np.mean(bs["Q1"])) if bs["Q1"] else None,
                                "share": bci(1 - Q1 / Q0, bs["share"]),
                                "gains": {b_: bci(M["V1"][i, 0], gain[:, i] if len(gain) else []) for i, b_ in enumerate(CORE)}},
            "rank1_without_age": sub["no_age"],
            "rank2": {"Q": Q2, "df": df2, "p_chi2": float(stats.chi2.sf(Q2, df2)) if df2 > 0 else None, "p_boot": pb(bs["Q2"], Q2)},
            "departures": dep, "two_bed": {k: v for k, v in sub.items() if k != "no_age"}}
    yb = np.array(yb) if yb else np.empty((0,) + yrs0.shape)
    c13 = {}
    for i, (e, cond, dl, lab, idx, _) in enumerate(des):
        r = {"contrast": lab, "n": rc(len(idx)), "years": {b_: bci(yrs0[i, k], yb[:, i, k] if len(yb) else []) for k, b_ in enumerate(C13_BEDS)}}
        for k, b_ in ((1, "imt"), (2, "art_w")):
            with np.errstate(divide="ignore", invalid="ignore"):
                rt = bci(yrs0[i, 0] / yrs0[i, k], yb[:, i, 0] / yb[:, i, k] if len(yb) else [])
            den = A.pct(yb[:, i, k]) if len(yb) else [None, None]
            if rt:
                rt["denominator_ci_includes_0"] = bool(den[0] is not None and den[0] <= 0 <= den[1])
            r["ratio_pwv_" + b_] = rt
        c13.setdefault(e, {})[cond] = r
    out["c13"] = c13
    return out


# ============================================================================ C5 oral taxa

C5_CHAIN = [("M3", "full", []), ("M3s", "S", []), ("smk", "S", ["smk_cur", "smk_former"]),
            ("smk_hr", "S", ["smk_cur", "smk_former", "hr_sit"]), ("smk_hr_adip", "S", ["smk_cur", "smk_former", "hr_sit", "waist", "log_vat"]),
            ("never", "never", [])]
C5_BASE = ["age", "sex_male", "bmi", "sbp_sit", "antihtn"]
C5_EXTRA = ["smk_cur", "smk_former", "hr_sit", "waist", "log_vat"]


def smoking_status(root, fr, info=None):
    """Per frame row: 0 never, 1 former, 2 current smoker (UK Biobank codings 100347/100348, as in the HPP dictionary:
    current status 1/2 = current; status 0 with past frequency 1/2 = former, 3/4 = never); same research stage, else
    the latest earlier stage; participants who answered 'never' at every stage are never smokers at every visit.
    info (dict, optional) receives rounded tallies of the raw codes (questionnaire rows) to verify the coding."""
    d = A.rd(root, "lifestyle_and_environment/lifestyle_and_environment.parquet",
             columns=["smoking_current_status", "smoking_past_frequency"])
    if d is None or "smoking_current_status" not in d.columns:
        return None
    st = pd.to_numeric(d["smoking_current_status"].astype(object), errors="coerce")
    pf = pd.to_numeric(d["smoking_past_frequency"].astype(object), errors="coerce") if "smoking_past_frequency" in d.columns \
        else pd.Series(np.nan, index=d.index)
    if info is not None:
        tal = lambda x, codes: [rc((x == c).sum()) for c in codes] + [rc((x.notna() & ~x.isin(codes)).sum()), rc(x.isna().sum())]
        info.update(current_status_0_1_2_other_na=tal(st, [0, 1, 2]), past_freq_if_status0_1_2_3_4_other_na=tal(pf[st == 0], [1, 2, 3, 4]))
    s3 = np.where(st.isin([1, 2]), 2.0, np.where((st == 0) & pf.isin([1, 2]), 1.0, np.where((st == 0) & pf.isin([3, 4]), 0.0, np.nan)))
    t = pd.DataFrame({"participant_id": d["participant_id"].astype(str), "o": d["research_stage"].map(ORD).astype(float), "s": s3}).dropna()
    t = t.groupby(["participant_id", "o"], as_index=False)["s"].max().sort_values("o")
    left = pd.DataFrame({"participant_id": fr["participant_id"].astype(str).to_numpy(),
                         "o": fr["research_stage"].map(ORD).astype(float).to_numpy(), "_r": np.arange(len(fr))})
    okl = left["o"].notna()
    res = np.full(len(fr), np.nan)
    if okl.any() and len(t):
        m = pd.merge_asof(left[okl].sort_values("o"), t, on="o", by="participant_id", direction="backward")
        res[m["_r"].to_numpy()] = m["s"].to_numpy(float)
    nev = t.groupby("participant_id")["s"].max() == 0
    miss = np.isnan(res) & left["participant_id"].map(nev).fillna(False).to_numpy(bool)
    res[miss] = 0.0
    return res


def c5_oral(fr, root, L=None, smk=None):
    Om = need(O, "omics")
    if L is None:
        L = Om.load_oral(root)
    EXTRA_PIDS.update(map(str, L["pid"]))
    M, minfo = Om.match(L, fr)
    fr2 = fr.copy()
    smk_info = {}
    s3 = smoking_status(root, fr, smk_info) if smk is None else smk
    if s3 is None:
        raise RuntimeError("smoking status unavailable")
    fr2["smk3"] = s3
    fr2["smk_cur"] = np.where(np.isnan(s3), np.nan, (s3 == 2).astype(float))
    fr2["smk_former"] = np.where(np.isnan(s3), np.nan, (s3 == 1).astype(float))
    js = {}
    for h in Om.HYP:
        k = [i for i, nm in enumerate(L["names"]) if h in nm]
        if k:
            js[h] = k[0]
    taxa = list(js)
    X4 = L["X"][:, [js[h] for h in taxa]]
    out = {"matched": {k: v for k, v in minfo.items() if k.startswith(("visits", "participants"))}, "taxa": taxa, "smoking_codes": smk_info,
           "prevalence": L["info"].get("hyp_prevalence"), "beds": {}}
    miss = [c for c in C5_BASE + C5_EXTRA if c not in fr2.columns]
    if any(c in miss for c in C5_BASE):
        raise RuntimeError(f"C5 base covariates missing from the frame: {miss}")
    ex_ok = [c for c in C5_EXTRA if c not in miss]  # a missing adjustment column is dropped from its step (listed)
    out["missing_covariates"] = miss
    covsets = {nm: C5_BASE + [c for c in ex if c in ex_ok] for nm, _, ex in C5_CHAIN}
    for bed in CORE:
        ob = "o_" + bed
        samples = {"full": Om.rows_for(fr2, M, [ob] + C5_BASE), "S": Om.rows_for(fr2, M, [ob] + C5_BASE + ex_ok)}
        dn = Om.rows_for(fr2, M, [ob] + C5_BASE + ["smk3"])
        samples["never"] = dn[dn["smk3"] == 0]
        rb = {"n": {k: rc(len(v)) for k, v in samples.items()}}
        if bed == "pwv":
            sS = samples["S"]
            rb["smoking_in_S"] = {"never": rc(((sS["smk_cur"] == 0) & (sS["smk_former"] == 0)).sum()),
                                  "former": rc((sS["smk_former"] == 1).sum()), "current": rc((sS["smk_cur"] == 1).sum())}
        for nm, which, _ in C5_CHAIN:
            dd = samples[which]
            if len(dd) < MIN_N:
                continue
            y = A.zs(dd[ob]).to_numpy()[:, None]
            b, se, nj, _ = Om.mols(y, X4[dd["_si"].to_numpy()], Om.covmat(dd, covsets[nm], L))
            rb[nm] = {h: est(b[i, 0], se[i, 0], len(dd)) for i, h in enumerate(taxa)}
        out["beds"][bed] = rb
    for nm, _, _ in C5_CHAIN:  # BH over taxa x core beds within each model (as omics hyp_oral)
        cells = [(bed, h) for bed in CORE for h in taxa if (out["beds"][bed].get(nm) or {}).get(h)]
        for (bed, h), qv in zip(cells, A.bh([out["beds"][bed][nm][h]["p"] for bed, h in cells])):
            out["beds"][bed][nm][h]["q"] = float(qv)
    side = {}
    for tgt in ("sbp_sit", "hr_sit"):  # taxa -> sitting SBP and HR (age, sex, BMI + read depth)
        dd = Om.rows_for(fr2, M, [tgt, "age", "sex_male", "bmi"])
        if len(dd) >= MIN_N:
            b, se, _, _ = Om.mols(A.zs(dd[tgt]).to_numpy()[:, None], X4[dd["_si"].to_numpy()], Om.covmat(dd, ["age", "sex_male", "bmi"], L))
            side[tgt] = {h: est(b[i, 0], se[i, 0], len(dd)) for i, h in enumerate(taxa)}
    dd = Om.rows_for(fr2, M, ["smk_cur", "smk_former", "age", "sex_male"])
    if len(dd) >= MIN_N and min((dd["smk_cur"] == 1).sum(), (dd["smk_former"] == 1).sum()) >= MIN_BIN:
        C = Om.covmat(dd, ["age", "sex_male"], L)
        sm = {}
        for i, h in enumerate(taxa):  # taxon (SD) ~ current + former (never = reference) + age + sex + depth
            x = X4[dd["_si"].to_numpy(), i]
            x = (x - x.mean()) / x.std()
            Xd = np.column_stack([C[:, :1], dd["smk_cur"].to_numpy(float), dd["smk_former"].to_numpy(float), C[:, 1:]])
            sm[h] = {"current": est(*olsj(x, Xd, 1), len(dd)), "former": est(*olsj(x, Xd, 2), len(dd))}
        side["by_smoking"] = sm
    out["side"] = side
    return out


# ============================================================================ C2 household: leave-pair-out and gap curve

GAP_EDGES = np.array([0, 1, 2, 8, 31, 181])
GAP_LAB = ["0", "1", "2-7", "8-30", "31-180", ">180"]
GAP_VARS = ["pwv_raw", "pwv", "sbp_sit"]
LPO_BEDS = ["pwv", "abi_min"]
DT_EDGES = np.array([0, 15, 30, 60])
DT_LAB = ["0-15", "15-30", "30-60", "60+"]
MAX_STR = 40000


def day_r(sets, D, boot, rng):
    """Double-entry r of pair sets {name: (z1, z2, day code)} with one bootstrap over days shared by all sets."""
    Wd = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(boot)]).astype(float) if boot else np.empty((0, D))

    def rr(T):
        with np.errstate(all="ignore"):
            n = T[..., 0]
            m = T[..., 1] / (2 * n)
            return (T[..., 3] / n - m * m) / (T[..., 2] / (2 * n) - m * m)
    out = {}
    for k, (z1, z2, dc) in sets.items():
        S = np.stack([np.bincount(dc, weights=w, minlength=D) for w in (np.ones(len(z1)), z1 + z2, z1 ** 2 + z2 ** 2, z1 * z2)], 1)
        out[k] = (rr(S.sum(0)), rr(Wd @ S), len(z1))
    return out


def rpack(t):
    r, bs, n = t
    if n < MIN_PAIRS:
        return {"n": rc(n)}
    return {"n": rc(n), "r": float(r), "ci": A.pct(bs)}


def lpo(fr, bed, pairs, comp, boot, rng):
    """Leave-pair-out version of the atlas v1.5 day + session correction for same-day pairs."""
    if not getattr(A, "CAL_ADJUST", False) or bed + "_raw" not in fr.columns:
        return {"status": "frame not calendar-adjusted"}
    kd = (A.DIAG.get("session_kernel") or {}).get(bed) or {}
    de = (A.DIAG.get("day_effects") or {}).get(bed) or {}
    saved = dict(A.DIAG)
    try:  # quarter-adjusted values exactly as in build_frame (calendar_adjust on the raw bed)
        f0 = fr[[c for c in ("participant_id", "research_stage", "visit_date", "vas_ts", "age", "sex_male") if c in fr.columns]].copy()
        f0[bed] = fr[bed + "_raw"].to_numpy()
        yq = A.calendar_adjust(f0)[bed].to_numpy(float)
    finally:
        A.DIAG.clear()
        A.DIAG.update(saved)
    _, t = A.calendar_quarter(fr)
    dayS = t.dt.strftime("%Y-%m-%d")
    ok = np.isfinite(yq) & fr["age"].notna().to_numpy() & fr["sex_male"].notna().to_numpy() & dayS.notna().to_numpy()
    oi = np.flatnonzero(ok)
    a_ = fr["age"].to_numpy(float)[oi]
    X = np.column_stack([np.ones(len(oi)), a_, (a_ - 50) ** 2, fr["sex_male"].to_numpy(float)[oi]])
    y = yq[oi]
    r = y - X @ A.beta_only(y, X)
    V = float(r.var())
    if "vas_ts" in fr.columns:
        ts = pd.to_datetime(fr["vas_ts"], errors="coerce", utc=True)
        tm = ((ts - ts.min()).dt.total_seconds() / 60.0).to_numpy(float)[oi]
    else:
        tm = np.full(len(oi), np.nan)
    dk = pd.factorize(dayS.to_numpy()[oi])[0]
    if kd.get("applied"):
        Vd = float(kd["V"])
        b0, s, tau = float(kd["day_share"]) * Vd, float(kd["share"]) * Vd, float(kd["tau"])
        f = min(1.0, 0.95 * V / (b0 + s))
        prm = ("kernel", tau, b0 * f, s * f)
    elif de.get("k") is not None and float(de.get("day_share", 0) or 0) > 0:
        prm = ("day", float(de["k"]))
    else:
        prm = ("none",)
    o = np.lexsort((tm, dk))
    groups = [g for g in np.split(o, np.flatnonzero(np.diff(dk[o])) + 1) if len(g) >= 2]
    gof = np.full(len(oi), -1)
    for gi, g in enumerate(groups):
        gof[g] = gi
    a_loo = np.zeros(len(oi))
    for g in groups:
        if prm[0] == "kernel":
            a_loo[g] = A.krige_day(r[g], tm[g], prm[3], prm[1], V, prm[2])
        elif prm[0] == "day":
            a_loo[g] = (r[g].sum() - r[g]) / (len(g) - 1 + prm[1])
    yv = fr[bed].to_numpy(float)[oi]
    out = {"mode": prm[0], "params": {"tau": prm[1], "b0_share": prm[2] / V, "s_share": prm[3] / V} if prm[0] == "kernel"
           else ({"k": prm[1]} if prm[0] == "day" else {}), "n_rows": rc(len(oi)),
           "loo_reproduction_max_abs_diff": float(np.nanmax(np.abs((y - a_loo) - yv))) if len(oi) else None,
           "median_rows_per_day": float(np.median([len(g) for g in groups])) if groups else None}
    if prm[0] == "none":
        out["status"] = "no day/session correction applied to this bed; LPO = LOO"

    def blup_ex(i, g, excl):
        Sx = g[~np.isin(g, excl)]
        if len(Sx) == 0 or prm[0] == "none":
            return 0.0
        if prm[0] == "day":
            return float(r[Sx].sum() / (len(Sx) + prm[1]))
        tau_, b0_, s_ = prm[1], prm[2], prm[3]
        dd_ = np.nan_to_num(np.abs(tm[Sx] - tm[i]), nan=np.inf)
        if len(Sx) > A.SESSION_MAX_NB:
            keep = np.argsort(dd_, kind="stable")[:A.SESSION_MAX_NB]
            Sx, dd_ = Sx[keep], dd_[keep]
        K = b0_ + s_ * np.nan_to_num(np.exp(-np.abs(tm[Sx][:, None] - tm[Sx][None, :]) / tau_), nan=0.0)
        np.fill_diagonal(K, V)
        c = b0_ + s_ * np.exp(-dd_ / tau_)
        return float(c @ np.linalg.solve(K, r[Sx]))
    eng = [i for g in groups for i in g]  # engine check: blup_ex excluding only the person itself must equal the atlas LOO
    if len(eng) > 300:
        eng = list(rng.choice(eng, 300, replace=False))
    big = [i for g in groups if len(g) > A.SESSION_MAX_NB + 1 for i in g[:20]]  # include the nearest-neighbour branch
    out["lpo_engine_check_max_abs_diff"] = float(max((abs(blup_ex(i, groups[gof[i]], (i,)) - a_loo[i]) for i in eng + big), default=0.0)) \
        if prm[0] != "none" else 0.0
    # rows per day: quantiles only (no maximum: a single day's count is not an aggregate); flag for the kriging branch
    out["q95_rows_per_day"] = float(np.percentile([len(g) for g in groups], 95)) if groups else None
    out["large_day_branch_checked"] = bool(big)
    # analysis rows: first main visit per participant (as family v1.1); z map as family person_table
    rows = A.one_row_per_participant(fr, [bed, "age", "sex_male"])
    ag, sx = rows["age"].to_numpy(float), rows["sex_male"].to_numpy(float)
    Xp = np.column_stack([np.ones(len(rows)), ag, (ag - 50) ** 2, sx, (ag - 50) * sx])
    yy = rows[bed].to_numpy(float)
    lo, hi = np.nanpercentile(yy, [0.5, 99.5])
    bz = np.linalg.lstsq(Xp, np.clip(yy, lo, hi), rcond=None)[0]
    rz = np.clip(yy, lo, hi) - Xp @ bz
    rm, rs = rz.mean(), rz.std()
    pos = np.full(len(fr), -1)
    pos[oi] = np.arange(len(oi))
    fidx = rows.index.to_numpy()
    prow = dict(zip(rows["participant_id"].astype(str), np.arange(len(rows))))
    zmap = lambda k, v: (np.clip(v, lo, hi) - Xp[k] @ bz - rm) / rs  # k = analysis-row number

    def build(pa, pb):
        """pa, pb: analysis-row numbers of same-day pairs -> z LOO, z LPO, day code, |dt| minutes."""
        ia, ib = pos[fidx[pa]], pos[fidx[pb]]
        z1, z2, w1, w2 = (np.empty(len(pa)) for _ in range(4))
        for t_, (ka, kb, i, j) in enumerate(zip(pa, pb, ia, ib)):
            g = groups[gof[i]]
            z1[t_], z2[t_] = zmap(ka, yv[i]), zmap(kb, yv[j])
            w1[t_] = zmap(ka, y[i] - blup_ex(i, g, (i, j)))
            w2[t_] = zmap(kb, y[j] - blup_ex(j, g, (i, j)))
        return z1, z2, w1, w2, dk[ia], np.abs(tm[ia] - tm[ib])

    def same_day(df):
        ka = df["a"].map(prow)
        kb = df["b"].map(prow)
        m = ka.notna() & kb.notna()
        ka, kb = ka[m].to_numpy(int), kb[m].to_numpy(int)
        ia, ib = pos[fidx[ka]], pos[fidx[kb]]
        k = (ia >= 0) & (ib >= 0)
        ka, kb, ia, ib = ka[k], kb[k], ia[k], ib[k]
        k = (dk[ia] == dk[ib]) & (gof[ia] >= 0)
        return ka[k], kb[k]
    sets = {cls: same_day(pairs[pairs["cls"] == cls]) for cls in ("cohab", "sib")}
    # unrelated same-day pairs (different relation components), capped at MAX_STR at random
    ar = np.arange(len(rows))
    ia_all = pos[fidx]
    okr = ia_all >= 0
    cid = pd.factorize(np.array([comp.get(p, p) for p in rows["participant_id"].astype(str)], dtype=object))[0]
    dd_ = np.where(okr, dk[np.clip(ia_all, 0, None)], -1)
    I, J = [], []
    srt = ar[okr][np.argsort(dd_[okr], kind="stable")]
    for g in np.split(srt, np.flatnonzero(np.diff(dd_[srt])) + 1):
        if len(g) < 2:
            continue
        iu, ju = np.triu_indices(len(g), 1)
        k = cid[g[iu]] != cid[g[ju]]
        I.append(g[iu[k]])
        J.append(g[ju[k]])
    I = np.concatenate(I) if I else np.zeros(0, int)
    J = np.concatenate(J) if J else np.zeros(0, int)
    out["n_unrelated_sameday_pairs"] = A.rcp(len(I))
    if len(I) > MAX_STR:
        k = rng.choice(len(I), MAX_STR, replace=False)
        I, J = I[k], J[k]
    sets["stranger"] = (I, J)
    B = {k: build(*v) for k, v in sets.items() if len(v[0])}
    D = int(dk.max()) + 1 if len(dk) else 1
    out["n_pairs"] = {k: rc(len(v[0])) for k, v in sets.items()}
    bseed = int(rng.integers(1 << 30))  # same day resamples for LOO and LPO
    for tag, (c1, c2) in (("loo", (0, 1)), ("lpo", (2, 3))):
        ss = {k: (v[c1], v[c2], v[4]) for k, v in B.items()}
        if "stranger" in B:
            dt_ = np.digitize(np.nan_to_num(B["stranger"][5], nan=-1), DT_EDGES) - 1
            for bi, lab in enumerate(DT_LAB):
                m = dt_ == bi
                ss["stranger:" + lab] = (B["stranger"][c1][m], B["stranger"][c2][m], B["stranger"][4][m])
        if "cohab" in B:
            dt_ = np.digitize(np.nan_to_num(B["cohab"][5], nan=-1), DT_EDGES) - 1
            for bi, lab in enumerate(DT_LAB):
                m = dt_ == bi
                ss["cohab:" + lab] = (B["cohab"][c1][m], B["cohab"][c2][m], B["cohab"][4][m])
        Rr = day_r(ss, D, boot, np.random.default_rng([bseed, 2]))
        res = {k: rpack(v) for k, v in Rr.items()}
        if "cohab" in B and "stranger" in B:  # stranger r matched to the cohabitants' |dt| distribution
            w, pt, bs = [], [], []
            for lab in DT_LAB:
                nc, ns = Rr["cohab:" + lab][2], Rr["stranger:" + lab][2]
                if nc and ns >= MIN_PAIRS:
                    w.append(nc)
                    pt.append(Rr["stranger:" + lab][0])
                    bs.append(Rr["stranger:" + lab][1])
            if w and Rr["cohab"][2] >= MIN_PAIRS:
                w = np.array(w, float) / sum(w)
                sm, smb = float(np.dot(w, pt)), np.array(bs).T @ w if len(bs[0]) else np.array([])
                res["stranger_matched"] = {"r": sm, "ci": A.pct(smb)}
                res["cohab_minus_stranger_matched"] = {"d": float(Rr["cohab"][0] - sm), "ci": A.pct(Rr["cohab"][1] - smb)}
        out[tag] = res
    return out


def gap_curve(fr, pairs, comp, boot, rng):
    Fm = need(F, "family")
    out = {}
    for v in GAP_VARS:
        if v not in fr.columns:
            continue
        P = Fm.person_table(fr, [v], False)
        if v not in P.columns:
            continue
        P = P[P[v].notna() & pd.to_datetime(P[v + "__date"], errors="coerce").notna()]
        dts = pd.to_datetime(P[v + "__date"], errors="coerce")
        if isinstance(dts.dtype, pd.DatetimeTZDtype):  # tz-aware dates: local calendar day
            dts = dts.dt.tz_convert(A.TZ).dt.tz_localize(None)
        dnum = pd.Series(((dts.dt.as_unit("ns").dt.normalize() - pd.Timestamp("2000-01-01")).dt.days).to_numpy(float), index=P.index)
        row = {}
        for cls in ("cohab", "sib"):
            pr = pairs[(pairs["cls"] == cls) & pairs["a"].isin(P.index) & pairs["b"].isin(P.index)].reset_index(drop=True)
            gb = np.digitize(np.abs(dnum.reindex(pr["a"]).to_numpy() - dnum.reindex(pr["b"]).to_numpy()), GAP_EDGES) - 1
            row[cls] = {lab: Fm.strip(Fm.pair_r(P, pr[gb == k], v, v, boot, rng)) for k, lab in enumerate(GAP_LAB)}
        # unrelated pairs: same-day disjoint pairs, random partners within +-1/7/30/180 days, and random global pairs
        ids = np.array(P.index, dtype=object)
        cid = np.array([comp.get(p, p) for p in ids], dtype=object)
        dv = dnum.to_numpy()
        o = np.argsort(dv, kind="stable")
        sd_ = dv[o]
        A_, B_ = [], []
        perm = rng.permutation(len(ids))
        sp = perm[np.argsort(dv[perm], kind="stable")]
        for g in np.split(sp, np.flatnonzero(np.diff(dv[sp])) + 1):
            A_.append(g[0:len(g) - 1:2])
            B_.append(g[1:len(g):2])
        for W_ in (1, 7, 30, 180):
            lo_ = np.searchsorted(sd_, dv - W_, side="left")
            hi_ = np.searchsorted(sd_, dv + W_, side="right")
            j = o[np.minimum(lo_ + (rng.random(len(ids)) * (hi_ - lo_)).astype(int), len(ids) - 1)]
            A_.append(np.arange(len(ids)))
            B_.append(j)
        A_.append(rng.integers(0, len(ids), 20000))
        B_.append(rng.integers(0, len(ids), 20000))
        ia, ib = np.concatenate(A_), np.concatenate(B_)
        k = (ia != ib) & (cid[ia] != cid[ib])
        ia, ib = ia[k], ib[k]
        gb = np.digitize(np.abs(dv[ia] - dv[ib]), GAP_EDGES) - 1
        st = {}
        for kk, lab in enumerate(GAP_LAB):
            m = np.flatnonzero(gb == kk)
            if len(m) > 20000:
                m = rng.choice(m, 20000, replace=False)
            st[lab] = Fm.strip(Fm.pair_r(P, pd.DataFrame({"a": ids[ia[m]], "b": ids[ib[m]]}), v, v, boot, rng))
        row["stranger"] = st
        out[v] = row
    return out


def c2_household(fr, root, boot, seed):
    Fm = need(F, "family")
    rng = np.random.default_rng([seed, 2])
    pairs = Fm.load_pairs(root)
    comp = A.relation_components(root) or {}
    out = {"pairs_n": {c: rc((pairs["cls"] == c).sum()) for c in ("cohab", "sib")}, "lpo": {}}
    for bed in LPO_BEDS:
        if bed not in fr.columns:
            continue
        try:
            out["lpo"][bed] = lpo(fr, bed, pairs, comp, boot, rng)
        except Exception as ex:
            out["lpo"][bed] = {"error": err(ex)}
            log(f"C2 lpo {bed} FAILED: {out['lpo'][bed]['error']}")
    try:
        out["gap_curve"] = gap_curve(fr, pairs, comp, boot, rng)
    except Exception as ex:
        out["gap_curve"] = {"error": err(ex)}
        log(f"C2 gap curve FAILED: {out['gap_curve']['error']}")
    return out


# ============================================================================ C7 heterogeneity permutation null + body size

BODY = ["height", "lean_total", "leg_lean", "legs_area"]
FL_BATCH = 10  # permutations per batch


def strata_of(age, sex):
    dec = pd.qcut(pd.Series(age).rank(method="first"), 10, labels=False).to_numpy()
    return np.nan_to_num(np.asarray(sex, float), nan=-1) * 100 + dec


def legs_area(root, d4):
    """DXA legs bone area (cm2; bone_density body_legs_area, mean per participant-stage) at the row's own stage: the
    leg-length / limb-size proxy (leg length itself is not in the HPP schema).  None if the table or field is absent."""
    b = A.rd(root, "bone_density/bone_density.parquet", columns=["body_legs_area"])
    if b is None or "body_legs_area" not in b.columns or "research_stage" not in b.columns:
        return None
    b = A.num(b, ["body_legs_area"])
    t = b.groupby(["participant_id", "research_stage"])["body_legs_area"].mean()
    key = pd.MultiIndex.from_arrays([d4["participant_id"].astype(str).to_numpy(), d4["research_stage"].astype(str).to_numpy()])
    return t.reindex(key).to_numpy(float)


def fl_resid(g, Yg):
    """Residual of Yg (K, rows, 4) on the group design [1, covariates, sex x age-decile indicators]: centring within the
    strata (the indicators), then removal of the orthonormal basis Qc of the other covariates residualised on them."""
    mu = np.add.reduceat(Yg[:, g["so"], :], g["starts"], axis=1) / g["ns"][None, :, None]
    R = Yg - mu[:, g["sc"], :]
    if g["Qc"].shape[1]:
        R = R - np.einsum("nr,krb->knb", g["Qc"], np.einsum("nr,knb->krb", g["Qc"], R))
    return R


def het_groups(d4, exps, covs, restrict=None):
    """Exposures grouped by sample (rows with the exposure, all covariates and, for dated COVID exposures, calendar
    time); per group the design [1, covariates, sex x age-decile indicators] (as exposome joint4), beds z-scored in
    the group's rows and residualised, exposures residualised."""
    Y0 = d4[OC].to_numpy(float)
    Cv = d4[covs].to_numpy(float) if covs else np.zeros((len(d4), 0))
    base = np.isfinite(Cv).all(1)
    if restrict is not None:
        base &= restrict
    ct = d4["cal_t"].to_numpy(float) if "cal_t" in d4.columns else np.full(len(d4), np.nan)
    Xall = d4[exps].to_numpy(float)
    xc = [bool(E.XCOV.get(e)) for e in exps]
    grp = {}
    for j in range(len(exps)):
        m = base & np.isfinite(Xall[:, j]) & (np.isfinite(ct) if xc[j] else True)
        grp.setdefault((m.tobytes(), xc[j]), []).append(j)
    groups = []
    for (key, x1), js in grp.items():
        idx = np.flatnonzero(np.frombuffer(key, dtype=bool))
        n = len(idx)
        if n < 200:
            continue
        st = strata_of(d4["age"].to_numpy(float)[idx], d4["sex_male"].to_numpy(float)[idx])
        Dm = (st[:, None] == np.unique(st)[None, 1:]).astype(float)
        Cm = np.column_stack([Cv[idx]] + ([ct[idx][:, None]] if x1 else []) + [Dm])
        Cm = Cm[:, Cm.std(0) > 0]
        Cm = (Cm - Cm.mean(0)) / Cm.std(0)  # scale-free projection (DXA masses are in grams; pinv is relative to the largest value)
        C = np.column_stack([np.ones(n), Cm])
        Pi = np.linalg.pinv(C.T @ C, rcond=1e-10, hermitian=True)
        X = Xall[idx][:, js]
        keep = [t for t in range(len(js)) if (min(X[:, t].sum(), n - X[:, t].sum()) >= MIN_BIN if isbin(X[:, t]) else X[:, t].std() > 0)]
        if not keep:
            continue
        X, js = X[:, keep], [js[t] for t in keep]
        RX = X - C @ (Pi @ (C.T @ X))
        live = (RX ** 2).sum(0) > 1e-8 * ((X - X.mean(0)) ** 2).sum(0)  # drop exposures (nearly) collinear with the covariates
        if not live.any():
            continue
        X, RX, js = X[:, live], RX[:, live], [j for j, k in zip(js, live) if k]
        Y = (Y0[idx] - Y0[idx].mean(0)) / Y0[idx].std(0)
        RY = Y - C @ (Pi @ (C.T @ Y))
        # Freedman-Lane outcomes are re-residualised on the same design by fl_resid: strata centring plus an orthonormal
        # basis of the other covariates residualised on [1, indicators] (columns in the indicator span, e.g. sex, drop)
        D1 = np.column_stack([np.ones(n), Dm])
        Cc = np.column_stack([Cv[idx]] + ([ct[idx][:, None]] if x1 else []))
        Cc = Cc[:, Cc.std(0) > 0] if Cc.shape[1] else Cc
        Cc = (Cc - Cc.mean(0)) / Cc.std(0) if Cc.shape[1] else Cc
        Cr = Cc - D1 @ np.linalg.lstsq(D1, Cc, rcond=None)[0]
        if Cr.shape[1]:
            Qc, sv, _ = np.linalg.svd(Cr, full_matrices=False)
            Qc = Qc[:, sv > 1e-8 * np.sqrt(n)]
        else:
            Qc = np.zeros((n, 0))
        sc = pd.factorize(st, sort=True)[0]
        ns = np.bincount(sc)
        g = {"idx": idx, "js": np.array(js), "Qc": Qc, "RY": RY, "RX": RX, "n": n, "k": Cm.shape[1] + 2,
             "sc": sc, "so": np.argsort(sc, kind="stable"), "ns": ns, "starts": np.r_[0, np.cumsum(ns)[:-1]],
             "scale": np.array([1.0 if isbin(X[:, t]) else X[:, t].std() for t in range(X.shape[1])])}
        g["fl_check"] = float(np.abs(fl_resid(g, Y[None])[0] - RY).max())  # the two residualisations must agree
        groups.append(g)
    return groups


def het_stats(RX, RY, n, k):
    """Oriented betas, robust joint covariance (HC1, as atlas multi_fit) and the 3-df Wald test of equal betas."""
    YY = (RY[:, :, None] * RY[:, None, :]).reshape(len(RY), 16)
    sxx = (RX ** 2).sum(0)
    B = (RX.T @ RY) / sxx[:, None]
    M2 = ((RX ** 2).T @ YY).reshape(-1, 4, 4)
    M3 = (RX ** 3).T @ RY
    M4 = (RX ** 4).sum(0)
    meat = M2 - M3[:, :, None] * B[:, None, :] - B[:, :, None] * M3[:, None, :] + M4[:, None, None] * B[:, :, None] * B[:, None, :]
    V = meat / (sxx ** 2)[:, None, None] * (n / max(n - k, 1))
    cb = B @ CC.T
    S = np.einsum("ab,pbc,dc->pad", CC, V, CC)
    W = np.einsum("pa,pa->p", cb, np.linalg.solve(S, cb[:, :, None])[:, :, 0])
    return B, V, W, stats.chi2.sf(W, 3)


def het_stats_k(RX, RYk, n, k):
    """het_stats Wald statistics for K outcome sets at once: RX (rows, p) fixed, RYk (K, rows, 4) -> W (K, p)."""
    K, nr, _ = RYk.shape
    p = RX.shape[1]
    RYn = RYk.transpose(1, 0, 2).reshape(nr, K * 4)
    sxx = (RX ** 2).sum(0)
    B = (RX.T @ RYn).reshape(p, K, 4) / sxx[:, None, None]
    YY = (RYn.reshape(nr, K, 4, 1) * RYn.reshape(nr, K, 1, 4)).reshape(nr, K * 16)
    M2 = ((RX ** 2).T @ YY).reshape(p, K, 4, 4)
    M3 = ((RX ** 3).T @ RYn).reshape(p, K, 4)
    M4 = (RX ** 4).sum(0)[:, None, None, None]
    meat = M2 - M3[..., :, None] * B[..., None, :] - B[..., :, None] * M3[..., None, :] + M4 * B[..., :, None] * B[..., None, :]
    V = meat / (sxx ** 2)[:, None, None, None] * (n / max(n - k, 1))
    cb = B @ CC.T
    S = np.einsum("ab,pkbc,dc->pkad", CC, V, CC)
    W = np.einsum("pka,pka->pk", cb, np.linalg.solve(S, cb[..., None])[..., 0])
    return W.T


def fl_prepare(d4, covs):
    """Freedman-Lane reduced model: the four oriented beds on [1, covariates, sex x age-decile indicators] in the rows
    with all covariates (no exposure).  Its residual rows (four beds jointly) are permuted within the sex x age-decile
    strata, so the exposures and their mutual dependence are left untouched."""
    Cv = d4[covs].to_numpy(float) if covs else np.zeros((len(d4), 0))
    idxb = np.flatnonzero(np.isfinite(Cv).all(1))
    st = strata_of(d4["age"].to_numpy(float)[idxb], d4["sex_male"].to_numpy(float)[idxb])
    C = np.column_stack([Cv[idxb], (st[:, None] == np.unique(st)[None, 1:]).astype(float)])
    C = C[:, C.std(0) > 0]
    C = np.column_stack([np.ones(len(idxb)), (C - C.mean(0)) / C.std(0)])
    Y = d4[OC].to_numpy(float)[idxb]
    Yh = C @ np.linalg.lstsq(C, Y, rcond=None)[0]
    return {"idxb": idxb, "st": st, "orig": np.argsort(st, kind="stable"), "Yh": Yh, "E": Y - Yh, "rows": len(d4)}


def fl_ystar(ctx, U):
    """Permuted outcomes Y* = fitted + residual rows permuted within strata, one per row of keys U (K x rows of d4);
    NaN outside the model rows."""
    Ys = np.full((len(U), ctx["rows"], 4), np.nan)
    for k in range(len(U)):
        rnd = np.argsort(ctx["st"] * 2.0 + U[k, ctx["idxb"]], kind="stable")
        pm = np.empty_like(rnd)
        pm[ctx["orig"]] = rnd
        Ys[k, ctx["idxb"]] = ctx["Yh"] + ctx["E"][pm]
    return Ys


def het_perm(groups, Ys, nexp):
    """Wald p values of every exposure for K permuted outcome sets Ys (K, rows of d4, 4): per group the beds are
    z-scored in the group's rows and residualised on the group design (as for the observed test)."""
    P = np.full((len(Ys), nexp), np.nan)
    for g in groups:
        Yg = Ys[:, g["idx"], :]
        Yg = (Yg - Yg.mean(axis=1, keepdims=True)) / Yg.std(axis=1, keepdims=True)
        P[:, g["js"]] = stats.chi2.sf(het_stats_k(g["RX"], fl_resid(g, Yg), g["n"], g["k"]), 3)
    return P


def c7_het(fr, root, workers, nperm, seed, budget, pre=None):
    Em = need(E, "exposome")
    t0 = time.time()
    fr2, xs, sx, dropped = Em.assemble(root, fr, workers) if pre is None else pre
    d4 = A.one_row_per_participant(fr2, OC + ["age", "sex_male"]).reset_index(drop=True)
    out = {"n_exposures": len(xs), "n_dropped_by_cap": len(dropped), "n_common": rc(len(d4)),
           "builder_errors": dict(getattr(Em, "DERR", {})), "assemble_s": time.time() - t0, "models": {}}
    if "legs_area" not in d4.columns:
        try:
            la = legs_area(root, d4)
            if la is not None:
                d4["legs_area"] = la
        except Exception as ex:
            out["legs_area_error"] = err(ex)
    body = [c for c in BODY if c in d4.columns and d4[c].notna().sum() >= 200]
    bm = np.isfinite(d4[body].to_numpy(float)).all(1) if body else None
    specs = {m: (list(c), None) for m, c in Em.MODELS.items()}
    if body:
        specs["M2_bodysample"] = (list(Em.MODELS["M2"]), bm)
        specs["M2B"] = (list(Em.MODELS["M2"]) + body, bm)
    out["body_covariates"] = body
    Gs = {}
    for m, (covs, rs) in specs.items():
        gs = het_groups(d4, xs, covs, rs)
        p, B, nn = np.full(len(xs), np.nan), np.full((len(xs), 4), np.nan), np.zeros(len(xs))
        for g in gs:
            Bg, _, _, pg = het_stats(g["RX"], g["RY"], g["n"], g["k"])
            p[g["js"]], B[g["js"]], nn[g["js"]] = pg, Bg * g["scale"][:, None], g["n"]
        qv = A.bh(list(p))
        if m in Em.MODELS:  # permuted models only (memory)
            Gs[m] = gs
        out["models"][m] = {"covariates": covs, "n_rows": rc(int(rs.sum())) if rs is not None else out["n_common"],
                            "tested": int(np.isfinite(p).sum()), "het_q05": int(np.nansum(qv < 0.05)), "het_p05": int(np.nansum(p < 0.05)),
                            "groups": len(gs), "fl_resid_check": max((g["fl_check"] for g in gs), default=None),
                            "per_exposure": {e: {"n": rc(nn[j]), "b": B[j].tolist(), "p": float(p[j]), "q": float(qv[j])}
                                             for j, e in enumerate(xs) if np.isfinite(p[j])}}
    try:  # consistency with the exposome v1 joint4 Wald test (all exposures, M2): p values and the heterogeneity count
        j4 = Em.joint4(d4, list(xs), list(Em.MODELS["M2"]))
        pe = out["models"]["M2"]["per_exposure"]
        dif = [abs(np.log10(max(j4[e]["het"]["p"], 1e-300)) - np.log10(max(pe[e]["p"], 1e-300))) for e in pe if e in j4]
        out["check_vs_joint4"] = {"n": len(dif), "n_joint4": len(j4), "max_abs_dlog10p": float(max(dif)) if dif else None,
                                  "joint4_het_q05": int(sum(v["het"]["q"] < 0.05 for v in j4.values()))}
    except Exception as ex:
        out["check_vs_joint4"] = {"error": err(ex)}
    # permutation null of the count (Freedman-Lane: residual rows of the four beds permuted jointly within sex x
    # age-decile strata; the same keys for M1, M2 and M3; exposures and their dependence unchanged)
    rng = np.random.default_rng([seed, 7])
    ctx = {m: fl_prepare(d4, list(Em.MODELS[m])) for m in Em.MODELS}
    null = {m: {"q05": [], "p05": [], "minp": []} for m in Em.MODELS}
    tp, k = time.time(), 0
    while k < nperm and time.time() - tp < budget:
        U = rng.random((min(FL_BATCH, nperm - k), len(d4)))
        for m in Em.MODELS:
            for pz in het_perm(Gs[m], fl_ystar(ctx[m], U), len(xs)):
                qz = A.bh(list(pz))
                null[m]["q05"].append(int(np.nansum(qz < 0.05)))
                null[m]["p05"].append(float(np.nanmean(pz < 0.05)))
                null[m]["minp"].append(float(np.nanmin(pz)))
        k += len(U)
    out["perm"] = {"done": k, "requested": nperm, "seconds": time.time() - tp, "stopped_by_budget": k < nperm,
                   "scheme": "Freedman-Lane, four beds jointly, within sex x age-decile strata"}
    for m in Em.MODELS:
        c = np.array(null[m]["q05"])
        obs = out["models"][m]["het_q05"]
        out["models"][m]["null"] = {"q05_counts": c.tolist(), "mean": float(c.mean()) if k else None,
                                    "q95": float(np.percentile(c, 95)) if k else None, "max": int(c.max()) if k else None,
                                    "emp_p": float((1 + (c >= obs).sum()) / (1 + k)) if k else None,
                                    "p05_rate": float(np.mean(null[m]["p05"])) if k else None,
                                    "minp_q05": float(np.percentile(null[m]["minp"], 5)) if k else None}
    if body:
        hits = {m: {e for e, v in out["models"][m]["per_exposure"].items() if v["q"] < 0.05} for m in ("M2", "M2_bodysample", "M2B")}
        out["body_size"] = {"lost": sorted(hits["M2_bodysample"] - hits["M2B"]), "gained": sorted(hits["M2B"] - hits["M2_bodysample"]),
                            "kept": len(hits["M2_bodysample"] & hits["M2B"])}
    return out


# ============================================================================ digest

def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def ci(c):
    return "[NA]" if not c or c[0] is None else f"[{f3(c[0])},{f3(c[1])}]"


def bc(e):
    """{'est'/'r'/'b'/'d': x, 'ci': [lo, hi]} or {'b', 'lo', 'hi'} -> 'x [lo,hi]' (the first of these keys present is the estimate)."""
    if not isinstance(e, dict):
        return "NA"
    for k in ("est", "r", "b", "d"):
        if k in e:
            if e[k] is None:
                return f"NA(n{e['n']})" if e.get("n") is not None and k == "r" else "NA"
            if "lo" in e:
                return f"{f3(e[k])} [{f3(e['lo'])},{f3(e['hi'])}]"
            return f"{f3(e[k])} {ci(e.get('ci'))}"
    return f"NA(n{e.get('n')})"


def sh(k):
    return k.replace("art_w", "art").replace("abi_min", "abi").replace("|", "-")


def bcc(e):
    """Compact 'x[lo,hi]' (no spaces) from an estimate dict."""
    return bc(e).replace(" [", "[")


def wrap(ln, width=220):
    """Lines longer than width are split at the last ' | ' or '; ' before the limit; continuation lines start with '~ '."""
    out = []
    while len(ln) > width:
        cut = max(ln.rfind(" | ", 0, width - 2), ln.rfind("; ", 0, width - 2))
        cut = cut if cut > 40 else width - 2
        out.append(ln[:cut])
        ln = "~ " + ln[cut:].lstrip(" |;")
    return out + [ln]


def digest(R):
    hv = R.get("helper_versions", {})
    L = [f"F1 ROBUST DIGEST {R['version']} | atlas {R.get('atlas_version')} | seed {R.get('seed')} boot {R.get('boot')} perm {R.get('perm')} | "
         f"runtime {f3(R.get('runtime_s'))}s",
         f"helpers: exposome {hv.get('exposome')}; omics {hv.get('omics')}; family {hv.get('family')} | frame visits {g(R, 'frame', 'visits')} "
         f"participants {g(R, 'frame', 'participants')} | load errors: {R.get('load_errors') or 'none'}",
         "conventions: beds adverse-oriented, atlas v1.5-corrected; b = SD of bed per SD (binary: per unit); x[lo,hi] = estimate [95% CI]"]
    C7 = R.get("C7")
    if C7 is not None:
        L.append("## C7 exposome bed heterogeneity (3-df Wald, BH q<0.05, common four-bed sample): permutation null of the COUNT; body size")
    if C7 and "models" in C7:
        pm = C7.get("perm", {})
        flc = max((fl(v.get("fl_resid_check")) for v in C7["models"].values() if v.get("fl_resid_check") is not None), default=np.nan)
        L.append(f"C7 exposures {C7.get('n_exposures')} (cap-dropped {C7.get('n_dropped_by_cap')}), common n {C7.get('n_common')}; Freedman-Lane perms "
                 f"{pm.get('done')}/{pm.get('requested')} in {f3(pm.get('seconds'))}s (resid check {f3(flc)}); vs exposome joint4: max|dlog10 p| "
                 f"{f3(g(C7, 'check_vs_joint4', 'max_abs_dlog10p'))} over {g(C7, 'check_vs_joint4', 'n')}, joint4 M2 count "
                 f"{g(C7, 'check_vs_joint4', 'joint4_het_q05')}; builder errors: {', '.join(C7.get('builder_errors') or {}) or 'none'}"
                 + (f"; legs area error: {C7['legs_area_error']}" if C7.get("legs_area_error") else ""))
        for m in ("M1", "M2", "M3"):
            v = C7["models"].get(m)
            if v:
                nl = v.get("null") or {}
                L.append(f"C7 {m}: het q<0.05 {v['het_q05']}/{v['tested']} (p<0.05 {v['het_p05']}) | null count mean {f3(nl.get('mean'))} q95 "
                         f"{f3(nl.get('q95'))} max {nl.get('max')} | emp p {f3(nl.get('emp_p'))} | null p<0.05 rate {f3(nl.get('p05_rate'))} | null min-p q05 {f3(nl.get('minp_q05'))}")
        bs, mb, m2 = C7["models"].get("M2_bodysample"), C7["models"].get("M2B"), C7["models"].get("M2")
        if bs and mb and m2:
            b_ = C7.get("body_size", {})
            L.append(f"C7 body size: M2 hits {m2['het_q05']} (n {m2['n_rows']}) | M2 same rows {bs['het_q05']} (n {bs['n_rows']}) | M2B +"
                     f"{'+'.join(C7.get('body_covariates', []))} {mb['het_q05']} | kept {b_.get('kept')} lost {len(b_.get('lost', []))} gained "
                     f"{len(b_.get('gained', []))}")
            L.append("C7 lost: " + ", ".join(b_.get("lost", [])[:12]) + " | gained: " + ", ".join(b_.get("gained", [])[:8]))
            L.append("C7 top M2 hits: exposure | M2 q, b pwv/imt/art/abi | same-rows M2 p | M2B p q, b pwv/imt/art/abi")
            top = sorted(((fl(v["p"]), e) for e, v in m2["per_exposure"].items() if fl(v["q"]) < 0.05))[:8]
            bb = lambda v: "/".join(f3(x) for x in (v.get("b") or [None] * 4))
            for _, e in top:
                a1, a2, a3 = m2["per_exposure"][e], bs["per_exposure"].get(e, {}), mb["per_exposure"].get(e, {})
                L.append(f"{e} | {f3(a1['q'])}, {bb(a1)} | {f3(a2.get('p'))} | {f3(a3.get('p'))} {f3(a3.get('q'))}, {bb(a3)}")
    elif C7:
        L.append(f"C7 error: {C7.get('error')}")
    C8 = R.get("C8")
    if C8 is not None:
        L.append("## C8 proportionality of the exposure x bed coefficient matrix (min-distance rank test; W from joint influence functions)")
    for m, v in ((C8 or {}).get("models") or {}).items():
        r1, r2 = v["rank1"], v["rank2"]
        L.append(f"C8 {m} (n {C8.get('n_common')}; {v['p']} exposures x 4 beds): Q0 {f3(v['Q0'])} | rank-1 Q {f3(r1['Q'])} df {r1['df']} p {f3(r1['p_chi2'])} "
                 f"boot p {f3(r1['p_boot'])} (boot null mean {f3(r1['null_boot_mean'])}) | rank-2 Q {f3(r2['Q'])} df {r2['df']} p {f3(r2['p_chi2'])} "
                 f"boot p {f3(r2['p_boot'])} | rank-1 share {bcc(r1['share'])}")
        L.append(f"C8 {m} rank-1 bed gains (PWV=1): " + "; ".join(f"{b_} {bcc(r1['gains'][b_])}" for b_ in CORE[1:]))
        L.append(f"C8 {m} two-bed rank-1, all exposures but age (C13 ratio homogeneity): "
                 + "; ".join(f"PWV-{k} Q {f3(t['Q'])} df {t['df']} p {f3(t['p_chi2'])} boot p {f3(t['p_boot'])} ratio {bcc(t['ratio_common'])}"
                             for k, t in v.get("two_bed", {}).items()))
        na = v.get("rank1_without_age") or {}
        L.append(f"C8 {m} rank-1 without the age row: Q {f3(na.get('Q'))} df {na.get('df')} p {f3(na.get('p_chi2'))} boot p {f3(na.get('p_boot'))}")
        if m == "M1":
            L.append("C8 M1 departures (dQ = drop of rank-1 Q when left out, ~chi2 3 df): exposure: b pwv/imt/art/abi | fit from other rows' gains | dQ p | "
                     "largest residual x[CI]")
            for e, x in sorted(v["departures"].items(), key=lambda t: -fl(t[1]["dQ"]))[:8]:
                k_ = max(CORE, key=lambda b_: abs(fl(g(x, "resid", b_, "est"))) if g(x, "resid", b_) else 0)
                L.append(f"{e}: {'/'.join(f3(t) for t in x['b'])} | {'/'.join(f3(t) for t in x['fit_from_others'])} | {f3(x['dQ'])} {f3(x['p_dQ'])} | "
                         f"{k_} {bcc((x.get('resid') or {}).get(k_))}")
    if C8 and C8.get("error"):
        L.append(f"C8 error: {C8['error']}")
    c13 = (C8 or {}).get("c13") or {}
    if c13:
        L.append("## C13 age equivalents: years per contrast (bed ~ age+sex+exposure; BPc: +SBP+antihtn); ratios PWV:IMT, PWV:retina; ! = denominator CI has 0")
        for e, r in c13.items():
            mg, bp = r.get("marg") or {}, r.get("bp") or {}
            fz = lambda x: "!" if x and x.get("denominator_ci_includes_0") else ""
            ct = mg.get("contrast")
            L.append(f"{e}/{ct if isinstance(ct, str) else f3(ct)} n{mg.get('n')}: " + " ".join(f"{b_} {bcc(g(mg, 'years', b_))}" for b_ in C13_BEDS)
                     + f" | pwv:imt {bcc(mg.get('ratio_pwv_imt'))}{fz(mg.get('ratio_pwv_imt'))} pwv:art {bcc(mg.get('ratio_pwv_art_w'))}{fz(mg.get('ratio_pwv_art_w'))}"
                     + (f" | BPc pwv {bcc(g(bp, 'years', 'pwv'))} pwv:imt {bcc(bp.get('ratio_pwv_imt'))}{fz(bp.get('ratio_pwv_imt'))}" if bp else ""))
    C11 = R.get("C11")
    if C11 is not None:
        L.append(f"## C11 top-decile overlap (age/sex-standardised adverse z; independence Jaccard {f3(J_IND)}, union {f3(U_IND)})")
    if C11 and "observed" in C11:
        ob = C11["observed"]
        items = list(ob["jaccard"].items())
        for i in (0, 3):
            L.append(f"C11 n {C11['n_common']} Jaccard obs[CI] (copula E|r): " + "; ".join(
                f"{sh(k)} {bcc(v)} ({f3(v['copula'])}|{f3(v['r'])})" for k, v in items[i:i + 3]))
        u = ob["union"]
        L.append(f"C11 union {bcc(u)} copula {f3(u['copula'])} | k beds 0-4 obs " + "/".join(f3(x) for x in ob["k_beds"]["est"])
                 + " copula " + "/".join(f3(x) for x in ob["k_beds"]["copula"]))
        for bed, c in (C11.get("ceilings") or {}).items():
            rt, lr = c.get("retest") or {}, c.get("left_right") or {}
            L.append(f"C11 replicate {bed}: retest n {rt.get('n')} r {bcc({'r': rt.get('r'), 'ci': rt.get('r_ci')})} J {bcc({'r': rt.get('jaccard'), 'ci': rt.get('jaccard_ci')})}"
                     f" | left/right n {lr.get('n')} r {bcc({'r': lr.get('r'), 'ci': lr.get('r_ci')})} J {bcc({'r': lr.get('jaccard'), 'ci': lr.get('jaccard_ci')})}")
        for src, s in (C11.get("single_factor") or {}).items():
            L.append(f"C11 one-factor ceiling, rho from {src} (* = left/right fallback) " + "/".join(
                f"{f3(s['rho'][b])}{'*' if s['rho_source'][b] != src else ''}" for b in CORE)
                + f": union {f3(s['union_ceiling'])}, observed position (0 indep, 1 ceiling) {bcc(s.get('union_position'))}; k beds "
                + "/".join(f3(x) for x in s.get("k_beds_ceiling", [])))
            pit = list(s["pairs"].items())
            for i in (0, 3):
                L.append(f"C11 {src} pairs J_ceil, position: " + "; ".join(f"{sh(k)} {f3(v['jaccard_ceiling'])}, {bcc(v['position'])}" for k, v in pit[i:i + 3]))
        cnd = C11.get("conditional") or {}
        for nm, c in cnd.items():
            a0, a1 = c.get("age_sex", {}), c.get("conditional", {})
            L.append(f"C11 given {nm} (n {c.get('n')}): pwv-art J {bcc(g(a0, 'jaccard', 'pwv|art_w'))} -> {bcc(g(a1, 'jaccard', 'pwv|art_w'))} (copula "
                     f"{f3(g(a1, 'jaccard', 'pwv|art_w', 'copula'))}); union {bcc(a0.get('union'))} -> {bcc(a1.get('union'))}")
        if cnd:
            L.append("C11 other pairs J same rows -> given " + " | ".join(nm for nm in cnd) + ": " + "; ".join(
                f"{sh(k)} {f3(g(next(iter(cnd.values())), 'age_sex', 'jaccard', k, 'est'))} -> "
                + "|".join(f3(g(c, 'conditional', 'jaccard', k, 'est')) for c in cnd.values()) for k in (g(next(iter(cnd.values())), 'age_sex', 'jaccard') or {}) if k != "pwv|art_w"))
    elif C11:
        L.append(f"C11: {C11.get('error') or 'n<200'}")
    C5 = R.get("C5")
    if C5 is not None:
        L.append("## C5 pre-stated oral taxa (CLR): M3 x[CI] q | same-sample M3 | +smoking | +HR | +waist+VAT x[CI] | never-smokers x[CI]; q = BH 4 taxa x 4 beds")
    if C5 and "beds" in C5:
        sm = g(C5, "beds", "pwv", "smoking_in_S") or {}
        if C5.get("missing_covariates"):
            L.append("C5 MISSING adjustment columns (dropped from their step): " + ", ".join(C5["missing_covariates"]))
        L.append(f"C5 n (pwv): full {g(C5, 'beds', 'pwv', 'n', 'full')} adjustment sample {g(C5, 'beds', 'pwv', 'n', 'S')} never-smokers {g(C5, 'beds', 'pwv', 'n', 'never')}; "
                 f"adjustment sample never/former/current {sm.get('never')}/{sm.get('former')}/{sm.get('current')}; prevalence "
                 + " ".join(f"{h[:1]}.{h.split('_')[-1][:5]} {f3(p)}" for h, p in (C5.get("prevalence") or {}).items()))
        for h in C5.get("taxa", []):
            for bed in CORE:
                rb = C5["beds"].get(bed, {})
                x = {nm: (rb.get(nm) or {}).get(h) for nm, _, _ in C5_CHAIN}
                if not x.get("M3"):
                    continue
                L.append(f"{h[:1]}.{h.split('_')[-1]} {bed}: {bcc(x['M3'])} q{f3(x['M3'].get('q'))} | {f3(g(x, 'M3s', 'b'))} | {f3(g(x, 'smk', 'b'))} | "
                         f"{f3(g(x, 'smk_hr', 'b'))} | {bcc(x.get('smk_hr_adip'))} | {bcc(x.get('never'))}")
        sc_ = C5.get("smoking_codes") or {}
        if sc_:
            L.append("C5 smoking raw codes (questionnaire rows; UKB 100347/100348): current status 0/1/2/other/NA "
                     + "/".join(str(x) for x in sc_.get("current_status_0_1_2_other_na", [])) + " | past frequency if status 0: 1/2/3/4/other/NA "
                     + "/".join(str(x) for x in sc_.get("past_freq_if_status0_1_2_3_4_other_na", [])))
        sd = C5.get("side") or {}
        for h in C5.get("taxa", []):
            L.append(f"C5 {h[:1]}.{h.split('_')[-1]} -> SBP {bcc(g(sd, 'sbp_sit', h))} HR {bcc(g(sd, 'hr_sit', h))} (age, sex, BMI) | taxon SD in current vs never "
                     f"{bcc(g(sd, 'by_smoking', h, 'current'))} former vs never {bcc(g(sd, 'by_smoking', h, 'former'))}")
    elif C5:
        L.append(f"C5 error: {C5.get('error')}")
    C2 = R.get("C2")
    if C2 is not None:
        L.append("## C2 household: leave-pair-out (LPO) vs atlas leave-one-out (LOO) correction, same-day pairs; r[CI, bootstrap over days]")
    for bed, v in ((C2 or {}).get("lpo") or {}).items():
        if "loo" not in v:
            L.append(f"C2 {bed}: {v.get('status') or v.get('error')}")
            continue
        pr = v.get("params") or {}
        L.append(f"C2 {bed}: mode {v['mode']}" + "".join(f" {k} {f3(x)}" for k, x in pr.items()) + f" | LOO repro max|d| {f3(v.get('loo_reproduction_max_abs_diff'))}"
                 f", LPO engine {f3(v.get('lpo_engine_check_max_abs_diff'))} | rows/day median {f3(v.get('median_rows_per_day'))} q95 {f3(v.get('q95_rows_per_day'))} | "
                 f"same-day pairs cohab {g(v, 'n_pairs', 'cohab')} sib {g(v, 'n_pairs', 'sib')} unrelated {g(v, 'n_pairs', 'stranger')} (of "
                 f"{v.get('n_unrelated_sameday_pairs')})" + (f" | {v['status']}" if v.get("status") else ""))
        for tag in ("loo", "lpo"):
            x = v[tag]
            L.append(f"C2 {bed} {tag.upper()}: cohab {bcc(x.get('cohab'))} sib {bcc(x.get('sib'))} unrelated {bcc(x.get('stranger'))} dt-matched "
                     f"{bcc(x.get('stranger_matched'))} | cohab-matched {bcc(x.get('cohab_minus_stranger_matched'))}")
        L.append(f"C2 {bed} r by |dt| min, cohab/unrelated, LOO; LPO: " + " ".join(
            f"{lab} {f3(g(v, 'loo', 'cohab:' + lab, 'r'))}/{f3(g(v, 'loo', 'stranger:' + lab, 'r'))};{f3(g(v, 'lpo', 'cohab:' + lab, 'r'))}/"
            f"{f3(g(v, 'lpo', 'stranger:' + lab, 'r'))}" for lab in DT_LAB))
    gc = (C2 or {}).get("gap_curve") or {}
    if gc and "error" not in gc:
        L.append("C2 calendar-gap curve: pair r by |days| between visits " + " | ".join(GAP_LAB) + " (r[CI]n)")
        for v_, row in gc.items():
            for t_, cells in row.items():
                L.append(f"G {v_} {t_}: " + " | ".join(
                    (f"{f3(c.get('r'))}{ci([c.get('lo'), c.get('hi')])}n{c.get('n')}" if c and c.get("r") is not None else f"n{(c or {}).get('n')}")
                    for c in (cells.get(lab) for lab in GAP_LAB)))
    elif gc:
        L.append(f"C2 gap curve error: {gc.get('error')}")
    if C2 and C2.get("error"):
        L.append(f"C2 error: {C2['error']}")
    C10 = R.get("C10")
    if C10 is not None:
        L.append("## C10 change on change (exposome X3 design; B = + dSBP): b[90% CI] n | TOST p at 0.05/0.10 SD per SD | smallest equivalence margin")
    if C10 and "est" in C10:
        es = C10["est"]
        fy = C10.get("followup_years") or {}
        L.append(f"C10 pairs {C10.get('n_pairs')} (untreated at both visits {C10.get('n_untreated')}); follow-up y q05/q50/q95 {f3(fy.get('q05'))}/{f3(fy.get('q50'))}/{f3(fy.get('q95'))}")
        tl = lambda x: "NA" if not x else (f"{f3(x['b'])}[{f3(x['lo90'])},{f3(x['hi90'])}] n{x['n']} TOST {f3(x['tost_p'].get('0.05'))}/"
                                           f"{f3(x['tost_p'].get('0.1'))} eq {f3(x['eq_bound'])}")
        for e in ("bmi", "waist", "log_vat"):
            L.append(f"C10 d{e} -> dPWV: B all {tl(es.get(f'{e}|pwv|B|all'))} | B untreated {tl(es.get(f'{e}|pwv|B|untreated'))} | A all {bcc(es.get(f'{e}|pwv|A|all'))}")
        L.append("C10 dBMI B, other beds (b, eq margin): " + "; ".join(f"{b_} {f3(g(es, f'bmi|{b_}|B|all', 'b'))}, {f3(g(es, f'bmi|{b_}|B|all', 'eq_bound'))}"
                                                                       for b_ in C10_BEDS if b_ != "pwv"))
        L.append(f"C10 controls: dBMI->dSokolow A {bcc(es.get('bmi|sokolow|A|all'))} B {bcc(es.get('bmi|sokolow|B|all'))}; dSBP->dPWV A {bcc(es.get('sbp_sit|pwv|A|all'))}")
    elif C10:
        L.append(f"C10 error: {C10.get('error')}")
    errs = {k: v["error"] for k, v in R.items() if isinstance(v, dict) and "error" in v}
    if errs:
        L.append("## errors: " + "; ".join(f"{k}: {v}" for k, v in errs.items()))
    L = [w for ln in L for w in wrap(ln.lstrip())]
    if len(L) > 120:
        L = L[:119] + ["(digest truncated at 120 lines)"]
    return "\n".join(L) + "\n"


# ============================================================================ main

SECTIONS = ("C10", "C11", "C8", "C2", "C5", "C7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--perm", type=int, default=200, help="C7 permutations of the heterogeneity count")
    ap.add_argument("--perm-budget", type=float, default=480.0, help="s; C7 permutations stop after this (count reported)")
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="exposome loaders (CGM files)")
    ap.add_argument("--atlas", default=ATLAS_PATH, help="atlas module (read at import time)")
    ap.add_argument("--exposome", default=EXPO_PATH, help="exposome v1 module (read at import time)")
    ap.add_argument("--omics", default=OMICS_PATH, help="omics v1.2 module (read at import time)")
    ap.add_argument("--family", default=FAMILY_PATH, help="family v1.1 module (read at import time)")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    secs = [s.strip() for s in a.sections.split(",") if s.strip()]
    t0 = time.time()
    load_helpers()
    hv = {k: getattr(m, "VERSION", None) for k, m in (("exposome", E), ("omics", O), ("family", F))}
    log(f"{VERSION}; atlas {A.VERSION}; helpers {hv}; numpy {np.__version__}; pandas {pd.__version__}; python {sys.version.split()[0]}")
    fr, _, _ = A.build_frame(a.root, None)
    fr = fr.reset_index(drop=True)
    pids = set(fr["participant_id"].astype(str))
    R = {"version": VERSION, "atlas_version": A.VERSION, "helper_versions": hv, "load_errors": LOAD_ERR, "seed": f"{a.seed:,}",
         "boot": a.boot, "perm": a.perm, "sections": secs,
         "frame": {"visits": rc(len(fr)), "participants": rc(fr["participant_id"].nunique())}}
    fns = {"C10": lambda: c10_tost(fr), "C11": lambda: c11_overlap(fr, a.boot, a.seed), "C8": lambda: c8_rank(fr, a.boot, a.seed),
           "C2": lambda: c2_household(fr, a.root, a.boot, a.seed), "C5": lambda: c5_oral(fr, a.root),
           "C7": lambda: c7_het(fr, a.root, a.workers, a.perm, a.seed, a.perm_budget)}
    for k in secs:
        guard(R, k, fns[k])
    R["runtime_s"] = time.time() - t0
    pids |= EXTRA_PIDS
    text = json.dumps(A.clean(R), ensure_ascii=False, indent=1, sort_keys=True, default=str)
    A.privacy_check(text, pids)
    dg = digest(json.loads(text))
    A.privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines; total {time.time() - t0:.1f}s")
    logtext = "\n".join(A.LOG) + "\n"
    A.privacy_check(logtext, pids)
    os.makedirs(a.out)
    sums = []
    for fn, tx in (("results.json", text), ("digest.txt", dg), ("run_log.txt", logtext)):
        with open(os.path.join(a.out, fn), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s[:16] + "  " + s.split()[-1] for s in sums))


if __name__ == "__main__":
    main()
