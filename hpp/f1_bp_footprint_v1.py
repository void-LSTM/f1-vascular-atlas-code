#!/usr/bin/env python3
"""F1 P5 - blood-pressure footprint by vascular bed in one sample (HPP sandbox; writes aggregates only).

Question.  Which time scale of blood pressure (BP) does each vascular bed integrate, and which BP component (steady =
diastolic, pulsatile = pulse pressure) acts on which bed?  The genetic (lifelong) BP effect from genome-wide BP scores is
compared with the observational effect of regression-calibrated usual BP in the same people.

Inputs.  Beds from the atlas frame (--atlas, default f1_atlas_v1_5.py beside this script; A.build_frame(root, None);
calendar/day/session-adjusted as in atlas v1.5).  Scores SBP_gw, DBP_gw, PP_gw computed with the PRS v1.1 scoring
functions (--prs, default f1_prs_v1_1.py beside this script; only the selected scores are scored) from the PGS weights
folder (--weights; per-file SHA-256 checked against the manifest).  Samples, PCs, KING relatives and full-sib pairs as
in PRS v1.1 (unrelated set: kinship <= 0.0884; full sibs: 0.177 < kinship <= 0.354 and IBS0 >= 0.005).

Outcomes (adverse-oriented; SD units = SD of the bed in the unrelated analysis rows, one scale for the main, strata,
untreated, PC1-20 and within-sib estimates; the core-4 contrasts use the common-sample SD; and native units): leg PWV,
carotid IMT, retinal artery width, ABI (min), retinal vein width, fractal dimension, Sokolow-Lyon, Cornell, CAVI0, and
PWV split into a structural part (PWV at a reference diastolic pressure of 80 mmHg from the participant's CAVI0 =
2 rho PWV^2 / Pd - ln(Pd / 100 mmHg), Pd = sitting DBP; rho = 1050 kg/m3) and a load-dependent part (PWV - structural).

Sections (each guarded: a failure is recorded as an error and the run continues)
  instruments  first stage of each score on its own BP component (mmHg per SD), robust F, partial R2, in every sample
  main         per score x bed (unrelated, age + age^2 + sex + PC1-10): reduced form; one-sample 2SLS per 10 mmHg of
               usual BP (exposure = visit BP: classical error does not bias IV); 2SLS with the regression-calibrated
               exposure (IVrc; calibration of SBP and DBP includes the scores); observational slope on visit BP
               (naive) and on regression-calibrated usual BP of the same component (obs_rc; own-component calibration
               without scores = marginal slope on usual BP); ratio and difference IV vs obs_rc (person-level
               bootstrap: a drawn person brings his analysis row and his calibration row together)
  contrasts    four core beds in their common sample: IV per bed with joint influence-function covariance, six
               pairwise contrasts, 3-df Wald heterogeneity (score DBP+PP joint model also)
  joint        DBP_gw + PP_gw as instruments for DBP and PP (just-identified multivariable 2SLS) per bed; Sanderson-
               Windmeijer conditional F; observational joint slopes on calibrated usual DBP and PP (bivariate
               calibration of SBP and DBP without scores)
  strata       score x age tertile and score x sex: 2SLS per stratum, first stage per stratum, Cochran Q (native)
  untreated    no antihypertensive at the analysis visit
  pcs20        PC1-20 instead of PC1-10
  pwv_split    total, structural and load-dependent PWV in m/s per 10 mmHg (single-score and joint models)
  within_sib   full-sib pair differences (age, age^2, sex differences adjusted), cluster-robust by sibship;
               population vs within-sib difference (z assuming independence: conservative)
  negative     score -> device brachial SBP given calibrated usual SBP (untreated rows; acquisition excess; the
               calibration includes the scores, so the null is exact if the device reads the same usual BP and the
               calibration transports), score -> naive acute deviation
               (atlas definition; expected to leak usual BP), acquisition clock time, calendar time, calendar
               quarter (score on quarter indicators, Wald), atlas PWV correction term (raw - adjusted), height
Regression calibration: other-visit sitting BP of participants untreated at both visits (atlas calibration_data) on
this visit's sitting BP (+ lying BP where available), age, age^2, sex, PC1-10: own component only (obs_rc), SBP+DBP
(joint obs_rc), SBP+DBP+scores (IVrc, device control).  v1.1 = v1.0 after code review (own-component
calibration for obs_rc; one SD scale and native-scale Q in strata; person-level joint bootstrap; nullable dtypes).
Privacy: aggregates only; counts rounded to 10 and suppressed below 20; no estimate from fewer than 200 rows (50 sib
pairs, 100 per stratum); quantiles 5-95 % only; no ids or dates; every output passes the atlas privacy scan; refuses to
overwrite --out.
Run (sandbox pheno env, from <WORK_DIR>):
  OMP_NUM_THREADS=4 python f1_bp_footprint_v1.py --root <HPP_DATA_ROOT> \
      --weights pgs_weights_f1_v1 --atlas f1_atlas_v1_5.py --prs f1_prs_v1_1.py --out f1_bpfoot_out_v1
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

HERE = os.path.dirname(os.path.abspath(__file__))


def _argpath(flag, names):
    if flag in sys.argv[:-1]:
        return os.path.abspath(sys.argv[sys.argv.index(flag) + 1])
    for nm in names:
        p = os.path.join(HERE, nm)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, names[0])


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


ATLAS_PATH = _argpath("--atlas", ["f1_atlas_v1_5.py", "f1_atlas_v1.py"])
PRS_PATH = _argpath("--prs", ["f1_prs_v1_1.py", "f1_prs_v1.py"])
A = _load(ATLAS_PATH, "f1_atlas")  # frame, privacy and formatting helpers
_argv = sys.argv
sys.argv = [PRS_PATH, "--atlas", ATLAS_PATH, "--"]  # the PRS module imports the same atlas file at import time
try:
    P = _load(PRS_PATH, "f1_prs")  # PRS v1.1: samples, relatives, scoring
finally:
    sys.argv = _argv
sys.modules["f1_atlas"] = A
P.A, P.log, P.rc = A, A.log, A.rc  # PRS helpers log into this run's log

VERSION = "f1_bp_footprint_v1.1 2026-10-04"
log, rc, v3 = A.log, A.rc, A.v3
SCORES = ("SBP_gw", "DBP_gw", "PP_gw")
EXPO = {"SBP_gw": "S", "DBP_gw": "D", "PP_gw": "P"}           # score -> BP component it instruments
XCOL = {"S": "sbp_sit", "D": "dbp_sit", "P": "pp_sit"}
BEDS = ["pwv", "imt", "art_w", "abi_min", "vein_w", "fd", "sokolow", "cornell", "cavi0", "pwv_struct", "pwv_load"]
ORIENT = dict(A.ORIENT, pwv_struct=1, pwv_load=1)
CORE = list(A.CORE)
PAIRS = [(a, b) for i, a in enumerate(CORE) for b in CORE[i + 1:]]
PWVF = ["pwv", "pwv_struct", "pwv_load"]
RHO, PREF, PSTAR, MMHG = 1050.0, 100.0, 80.0, 133.322
N_PCS, ALT_PCS = 10, 20
MIN_N, MIN_PAIRS, MIN_STRAT = 200, 50, 100
API_A = ["build_frame", "calibration_data", "acute_deviation", "one_row_per_participant", "calendar_quarter", "rc", "cnt",
         "v3", "pct", "clean", "privacy_check", "contrasts", "LOG", "MINCELL", "MAIN_STAGES", "ORIENT", "CORE"]
API_P = ["read_fam_ids", "load_samples", "king_pairs", "unrelated_set", "score_all", "variant_counts_text", "KIN_FIRST",
         "KIN_DUP", "IBS0_PO_MAX"]


# ============================================================================ small helpers

def err(ex):
    return f"{type(ex).__name__}: {str(ex)[:200]} (line {traceback.extract_tb(ex.__traceback__)[-1].lineno})"


def guard(R, key, fn):
    t = time.time()
    try:
        R[key] = fn()
        log(f"{key} done in {time.time() - t:.1f}s")
    except Exception as ex:
        R[key] = {"error": err(ex)}
        log(f"{key} FAILED: {R[key]['error']}")


def pv(b, se):
    if se is None or not np.isfinite(se) or se <= 0 or not np.isfinite(b):
        return None
    return float(2 * stats.norm.sf(abs(b / se)))


def cd(b, se, extra=None):
    """coefficient dict b, se, 95 % CI, p."""
    if b is None or not np.isfinite(b):
        return None
    o = {"b": float(b), "se": float(se) if se is not None and np.isfinite(se) else None}
    if o["se"] is not None:
        o.update(lo=o["b"] - 1.96 * o["se"], hi=o["b"] + 1.96 * o["se"], p=pv(o["b"], o["se"]))
    if extra:
        o.update(extra)
    return o


def resid(C, M):
    """Residuals of the columns of M on C (pseudo-inverse: rank-deficient C is safe) and rank of C."""
    B, _, rank, _ = np.linalg.lstsq(C, M, rcond=None)
    return M - C @ B, int(rank)


def vif(IF, n, k, cl=None):
    """Robust (HC1) or cluster-robust covariance from influence functions IF (n x m)."""
    IF = IF.reshape(len(IF), -1)
    if cl is None:
        return IF.T @ IF * (n / max(n - k, 1))
    S = np.zeros((int(cl.max()) + 1, IF.shape[1]))
    np.add.at(S, cl, IF)
    G = S.shape[0]
    return S.T @ S * (G / max(G - 1, 1))


def iv1(g, x, y):
    """Just-identified 2SLS on residualized vectors: theta, IF."""
    sgx = g @ x
    if abs(sgx) < 1e-12:
        return np.nan, np.zeros(len(g))
    th = (g @ y) / sgx
    return th, g * (y - th * x) / sgx


def ols1(x, y):
    sxx = x @ x
    if sxx < 1e-12:
        return np.nan, np.zeros(len(x))
    b = (x @ y) / sxx
    return b, x * (y - b * x) / sxx


def olsk(X, y):
    """OLS on residualized columns: b, IF (n x k)."""
    XtX = X.T @ X
    try:
        Gi = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        Gi = np.linalg.pinv(XtX)
    b = Gi @ (X.T @ y)
    e = y - X @ b
    return b, (X * e[:, None]) @ Gi.T


def ivk(Z, X, y):
    """Just-identified multivariable 2SLS (instruments Z, exposures X, residualized): theta, IF (n x k)."""
    ZX = Z.T @ X
    Gi = np.linalg.inv(ZX)
    th = Gi @ (Z.T @ y)
    e = y - X @ th
    return th, (Z * e[:, None]) @ Gi.T


def sw_f(Z, X, j, n, k, cl=None):
    """Sanderson-Windmeijer conditional F for exposure j (two exposures, two instruments; df 1)."""
    o = 1 - j
    xo = Z @ np.linalg.lstsq(Z, X[:, o], rcond=None)[0]
    den = xo @ X[:, o]
    if abs(den) < 1e-12:
        return None
    e = X[:, j] - (xo @ X[:, j]) / den * X[:, o]
    gam, IF = olsk(Z, e)
    V = vif(IF, n, k + 2, cl)
    try:
        return float(gam @ np.linalg.solve(V, gam)) / (Z.shape[1] - X.shape[1] + 1)
    except np.linalg.LinAlgError:
        return None


def q_het(ests):
    """Cochran Q across independent strata: list of (b, se) -> (Q, df, p)."""
    e = [(b, s) for b, s in ests if b is not None and s is not None and np.isfinite(b) and np.isfinite(s) and s > 0]
    if len(e) < 2:
        return None
    b, s = np.array([x[0] for x in e]), np.array([x[1] for x in e])
    w = 1 / s ** 2
    m = (w * b).sum() / w.sum()
    Q = float((w * (b - m) ** 2).sum())
    return {"Q": Q, "df": len(e) - 1, "p": float(stats.chi2.sf(Q, len(e) - 1))}


def unionfind(a, b):
    par = {}

    def f(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for x, y in zip(a, b):
        rx, ry = f(x), f(y)
        if rx != ry:
            par[rx] = ry
    return {k: f(k) for k in list(par)}


# ============================================================================ genetics (PRS v1.1 functions)

def load_samples_pairs(root):
    """Genotyped samples with PCs (PRS v1.1 rules), PC11-20 when present, unrelated set, full-sib pairs."""
    base = os.path.join(root, "human_genetics", "genotypes", "plink_postqc")
    fam = P.read_fam_ids(os.path.join(base, "chr_1"))
    cov, flow = P.load_samples(root, fam)
    try:
        cols = [f"pc{i}" for i in range(N_PCS + 1, ALT_PCS + 1)]
        pca = pd.read_parquet(os.path.join(root, "human_genetics/pca/pca.parquet"), columns=cols)
        pca = pca.set_axis(pca.index.get_level_values("participant_id").astype(str)).astype("float64")
        cov = cov.join(pca[~pca.index.duplicated()], how="left")
    except Exception as ex:
        flow["pc11_20_error"] = err(ex)
    flow["pcs_available"] = int(sum(f"pc{i}" in cov.columns and cov[f"pc{i}"].notna().mean() > 0.9
                                    for i in range(1, ALT_PCS + 1)))
    pairs = P.king_pairs(root, cov.index)
    unrel = P.unrelated_set(cov.index, pairs)
    k, i0 = pairs["Kinship"].to_numpy(), pairs["IBS0"].to_numpy()
    fs = pairs[(k > P.KIN_FIRST) & (k <= P.KIN_DUP) & (i0 >= P.IBS0_PO_MAX)][["ID1", "ID2"]].reset_index(drop=True)
    flow.update({"unrelated": rc(len(unrel)), "full_sib_pairs": rc(len(fs)), "related_pairs": rc(len(pairs))})
    log(f"samples {A.cnt(len(cov))}; unrelated {A.cnt(len(unrel))}; full-sib pairs {A.cnt(len(fs))}")
    return fam, cov, flow, unrel, fs


def load_weights(wdir, scores):
    mpath = os.path.join(wdir, "manifest.tsv")
    if not os.path.exists(mpath):
        cands = sorted(f for f in os.listdir(wdir) if f.endswith("_manifest.tsv"))
        if len(cands) != 1:
            raise SystemExit(f"need exactly one *_manifest.tsv in {wdir}, found {len(cands)}")
        mpath = os.path.join(wdir, cands[0])
    man = pd.read_csv(mpath, sep="\t")
    W, chk = {}, {}
    for r in man.itertuples(index=False):
        if r.score not in scores:
            continue
        p = os.path.join(wdir, r.file)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        chk[r.score] = {"sha256_ok": h == r.sha256, "pgs_id": r.pgs_id, "trait": r.trait}
        if h != r.sha256:
            log(f"weights {r.score}: SHA-256 mismatch, skipped")
            continue
        W[r.score] = pd.read_parquet(p)
    for s in scores:
        if s not in chk:
            chk[s] = {"error": "not in manifest"}
    log(f"weights loaded: {', '.join(W)}")
    return W, chk


def score_samples(root, W, fam, ids, block):
    S, coverage = P.score_all(root, W, len(fam), block)
    S = pd.DataFrame(S, index=fam, columns=list(W))
    S = S[~S.index.duplicated()].reindex(ids)
    return S, P.variant_counts_text(coverage)


# ============================================================================ frame and samples

def derive(fr):
    """Analysis columns on the atlas frame: naive acute deviation, CAVI0, structural / load-dependent PWV, calendar."""
    fr = fr.copy()
    info = {}
    try:
        fr, info = A.acute_deviation(fr)
    except Exception as ex:
        fr["acute_dev"] = np.nan
        info = {"error": err(ex)}
    fr["pp_sit"] = fr["sbp_sit"] - fr["dbp_sit"]
    pd_pa = fr["dbp_sit"].astype(float) * MMHG
    cav = 2 * RHO * fr["pwv"].astype(float) ** 2 / pd_pa - np.log(fr["dbp_sit"].astype(float) / PREF)
    fr["cavi0"] = cav.where(np.isfinite(cav))
    inner = fr["cavi0"] + np.log(PSTAR / PREF)
    fr["pwv_struct"] = np.sqrt((PSTAR * MMHG * inner / (2 * RHO)).where(inner > 0))
    fr["pwv_load"] = fr["pwv"] - fr["pwv_struct"]
    q, t = A.calendar_quarter(fr)
    fr["cal_q"] = q.where(~q.str.contains("NA"), None)
    t = pd.to_datetime(t, errors="coerce")
    if isinstance(t.dtype, pd.DatetimeTZDtype):
        t = t.dt.tz_convert(A.TZ).dt.tz_localize(None)
    try:
        t = t.dt.as_unit("ns")
    except Exception:
        pass
    fr["cal_years"] = (t - t.min()).dt.total_seconds() / (365.25 * 86400)
    fr["corr_pwv"] = fr["pwv_raw"] - fr["pwv"] if "pwv_raw" in fr.columns else np.nan
    fr["acq_min"] = fr["acq_hour"] * 60.0 if "acq_hour" in fr.columns else np.nan
    for c in ("sbp_lie", "dbp_lie", "dev_bsbp", "height", "antihtn"):
        if c not in fr.columns:
            fr[c] = np.nan
    fr["pp_lie"] = fr["sbp_lie"] - fr["dbp_lie"]
    return fr, info


def build_samples(fr2, S, cov):
    """One row per genotyped participant (first main visit with sitting SBP and DBP); scores per SD of the unrelated
    analysis rows; calibration rows (atlas calibration_data) with the same scaling."""
    scores = list(S.columns)
    d = A.one_row_per_participant(fr2, ["age", "sex_male", "sbp_sit", "dbp_sit"])
    d = d[d["participant_id"].isin(S.index.dropna())].copy()
    pcs = [c for c in cov.columns if c.startswith("pc")]
    d = d.join(S, on="participant_id").join(cov[pcs], on="participant_id")
    d["age2"] = (d["age"] - 50.0) ** 2
    cal = A.calibration_data(fr2)
    if cal is not None:
        cal = cal[cal["participant_id"].isin(S.index.dropna())].copy()
        cal = cal.join(S, on="participant_id").join(cov[pcs], on="participant_id")
        cal["age2"] = (cal["age"] - 50.0) ** 2
        cal["pp_rep"] = cal["sbp_rep"] - cal["dbp_rep"]
    return d, cal, scores, pcs


def standardize_scores(d, cal, scores, unrel):
    u = d["participant_id"].isin(unrel)
    sc = {}
    for s in scores:
        mu, sd = float(d.loc[u, s].mean()), float(d.loc[u, s].std())
        sc[s] = (mu, sd)
        d["g_" + s] = (d[s] - mu) / sd if sd > 0 else np.nan
        if cal is not None:
            cal["g_" + s] = (cal[s] - mu) / sd if sd > 0 else np.nan
    return sc


# ============================================================================ regression calibration

CAL_T = {"S": "sbp_rep", "D": "dbp_rep", "P": "pp_rep"}
OWN = {"S": ("sbp_sit", "sbp_lie"), "D": ("dbp_sit", "dbp_lie"), "P": ("pp_sit", "pp_lie")}


def cal_specs(pcs, gcols, lying):
    """Calibration models: name -> (targets, predictor sets; a later set overrides an earlier one where available).
    own_S/D/P  one component from its own sitting (+ lying) reading and the covariates: the slope of a bed on this
               calibrated value is the MARGINAL slope on usual BP of that component (single-component obs_rc).
               (v1.1 review: predicting one component from both SBP and DBP readings does not give the marginal
               slope when the bed depends on the other component; bias 20-40 % in check_f1_bpfoot_estimands.py.)
    biv        SBP and DBP from both readings: multivariable regression calibration for the joint DBP + PP slopes.
    biv_g      biv + the scores: IVrc and the device-SBP control (scores among the predictors keep the first stage
               on the calibrated exposure unbiased and make the device null exact)."""
    cov = ["age", "age2", "sex_male"] + pcs

    def sets(comps, extra=()):
        s0 = [OWN[c][0] for c in comps] + cov + list(extra)
        return [s0] + ([s0 + [OWN[c][1] for c in comps]] if lying else [])
    sp = {f"own_{c}": ([CAL_T[c]], sets(c)) for c in "SDP"}
    sp["biv"] = (["sbp_rep", "dbp_rep"], sets("SD"))
    sp["biv_g"] = (["sbp_rep", "dbp_rep"], sets("SD", gcols))
    return sp


def cal_fit(Carr, ci, targets, psets):
    """Fit one calibration model (targets = other-visit sitting BP) for each predictor set on complete rows."""
    Tg = Carr[:, [ci[t] for t in targets]]
    fits = []
    for preds in psets:
        X = Carr[:, [ci[p] for p in preds]]
        ok = np.isfinite(X).all(axis=1) & np.isfinite(Tg).all(axis=1)
        if ok.sum() < 100:
            fits.append(None)
            continue
        Xc = np.column_stack([np.ones(ok.sum()), X[ok]])
        B = np.linalg.lstsq(Xc, Tg[ok], rcond=None)[0]
        e = Tg[ok] - Xc @ B
        r2 = 1 - e.var(axis=0) / Tg[ok].var(axis=0)
        fits.append((B, int(ok.sum()), r2))
    return fits


def cal_predict(fits, Xarr, xi, psets, k):
    """Predictions (n x k targets); later (richer) predictor sets override earlier ones where available."""
    out = np.full((len(Xarr), k), np.nan)
    for f, preds in zip(fits, psets):
        if f is None:
            continue
        X = Xarr[:, [xi[p] for p in preds]]
        ok = np.isfinite(X).all(axis=1)
        out[ok] = np.column_stack([np.ones(ok.sum()), X[ok]]) @ f[0]
    return out


def calibrate(Carr, ci, Xarr, xi, specs, only=None):
    """Calibrated usual BP (mmHg) for the rows of Xarr.  Returns ({"own"|"biv"|"biv_g": n x 3 (S, D, P)}, fits);
    None for a block whose first (sitting-only) model could not be fitted."""
    pr, fits = {}, {}
    for name, (targets, psets) in specs.items():
        if only and name not in only:
            continue
        fits[name] = f = cal_fit(Carr, ci, targets, psets)
        pr[name] = cal_predict(f, Xarr, xi, psets, len(targets)) if f[0] is not None else None
    U = {}
    if not only or "own" in only:
        own = [pr.get(f"own_{c}") for c in "SDP"]
        U["own"] = None if any(x is None for x in own) else np.column_stack([x[:, 0] for x in own])
    for nm in ("biv", "biv_g"):
        if nm in pr:
            U[nm] = None if pr[nm] is None else np.column_stack([pr[nm], pr[nm][:, 0] - pr[nm][:, 1]])
    return U, fits


def ustack(U):
    """U columns: 0-2 own-component (obs_rc), 3-5 biv_g (IVrc, device control), 6-8 biv (joint obs_rc); S, D, P."""
    if any(U.get(k) is None for k in ("own", "biv_g", "biv")):
        return None
    return np.column_stack([U["own"], U["biv_g"], U["biv"]])


def cal_info(fits, targets, psets):
    out = []
    for f, preds in zip(fits, psets):
        if f is None:
            out.append(None)
            continue
        lam = {}
        for j, t in enumerate(targets):
            c = {"sbp_rep": "sbp_sit", "dbp_rep": "dbp_sit", "pp_rep": "pp_sit"}[t]
            lam[t] = float(f[0][1 + preds.index(c), j]) if c in preds else None
        out.append({"n": rc(f[1]), "r2": {t: float(f[2][j]) for j, t in enumerate(targets)}, "lambda": lam})
    return out


# ============================================================================ estimation blocks

class Tab:
    """numpy view of a DataFrame (float columns) for fast resampling."""

    def __init__(self, df, cols):
        cols = list(dict.fromkeys(c for c in cols if c in df.columns))
        self.cols = cols
        self.ix = {c: i for i, c in enumerate(cols)}
        self.X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)  # nullable dtypes
        self.X[~np.isfinite(self.X)] = np.nan

    def col(self, c, rows=None):
        v = self.X[:, self.ix[c]] if c in self.ix else np.full(len(self.X), np.nan)
        return v if rows is None else v[rows]


def bed_block(T, rows, bed, scores, covc, U=None, min_n=MIN_N, sd_ref=None):
    """Residualize (on covc) the scores, the BP exposures (per 10 mmHg), the calibrated exposures and the bed (native)
    for the rows with complete data.  Returns dict with residualized arrays, n, k, SD of the bed (sd_ref when given:
    one SD scale across samples and strata), ok-mask."""
    if U is not None and rows is not None:
        U = U[rows]
    cols = [T.col(c, rows) for c in covc]
    G = np.column_stack([T.col("g_" + s, rows) for s in scores])
    X = np.column_stack([T.col(XCOL[c], rows) / 10.0 for c in "SDP"])
    Y = T.col(bed, rows) * ORIENT.get(bed, 1)
    parts = [np.column_stack(cols), G, X, Y[:, None]]
    if U is not None:
        parts.append(U / 10.0)
    ok = np.isfinite(np.column_stack(parts)).all(axis=1)
    n = int(ok.sum())
    if n < min_n:
        return {"n": n}
    C = np.column_stack([np.ones(n)] + [c[ok] for c in cols])
    keep = np.r_[True, C[:, 1:].std(axis=0) > 0]
    C = C[:, keep]
    M = np.column_stack([G[ok], X[ok], Y[ok][:, None]] + ([U[ok] / 10.0] if U is not None else []))
    R, k = resid(C, M)
    ns = len(scores)
    sd = float(sd_ref) if sd_ref else float(np.std(Y[ok], ddof=1))
    o = {"n": n, "k": k, "G": R[:, :ns], "X": R[:, ns:ns + 3], "y": R[:, ns + 3], "sd": sd, "ok": ok}
    if U is not None:  # see ustack(): own-component | with scores | bivariate
        o["Uo"], o["Ui"], o["Uj"] = R[:, ns + 4:ns + 7], R[:, ns + 7:ns + 10], R[:, ns + 10:ns + 13]
    return o


def iv_entry(b, s, j, e, cl=None):
    """2SLS (measured exposure) of score j on exposure e: SD units, native units, reduced form, first stage."""
    n, k = b["n"], b["k"] + 1
    g, x, y, sd = b["G"][:, j], b["X"][:, e], b["y"], b["sd"]
    th, IF = iv1(g, x, y)
    se = float(np.sqrt(vif(IF, n, k, cl)[0, 0]))
    rf, IFr = ols1(g, y)
    fs, IFf = ols1(g, x)
    sef = float(np.sqrt(vif(IFf, n, k, cl)[0, 0]))
    ser = float(np.sqrt(vif(IFr, n, k, cl)[0, 0]))
    F = (fs / sef) ** 2 if sef > 0 else None
    r2p = float((g @ x) ** 2 / ((g @ g) * (x @ x))) if (g @ g) > 0 and (x @ x) > 0 else None
    return {"n": rc(n), "sd_bed": sd, "iv": cd(th / sd, se / sd), "iv_native": cd(th, se), "rf": cd(rf / sd, ser / sd),
            "fs_mmHg": cd(fs * 10, sef * 10), "F": F, "r2_partial": r2p}, IF / sd


def fs_entry(T, rows, s, e, covc, cl=None):
    """First stage of score s on BP component e (mmHg per SD) in the given rows."""
    b = bed_block(T, rows, "sbp_sit", [s], covc)
    if "G" not in b:
        return {"n": rc(b["n"])}
    g, x = b["G"][:, 0], b["X"][:, "SDP".index(e)]
    fs, IF = ols1(g, x)
    se = float(np.sqrt(vif(IF, b["n"], b["k"] + 1, cl)[0, 0]))
    return {"n": rc(b["n"]), "b_mmHg": cd(fs * 10, se * 10), "F": (fs / se) ** 2 if se > 0 else None,
            "r2_partial": float((g @ x) ** 2 / ((g @ g) * (x @ x)))}


def joint_entry(b, jd, jp, cl=None):
    """DBP_gw + PP_gw -> DBP, PP (per 10 mmHg) -> bed: theta (SD and native), conditional F, IF (SD units)."""
    n, k = b["n"], b["k"] + 2
    Z, X, y, sd = b["G"][:, [jd, jp]], b["X"][:, [1, 2]], b["y"], b["sd"]
    try:
        th, IF = ivk(Z, X, y)
    except np.linalg.LinAlgError:
        return None, None
    V = vif(IF, n, k, cl)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    bn, IFn = olsk(X, y)
    sen = np.sqrt(np.clip(np.diag(vif(IFn, n, k, cl)), 0, None))
    o = {"n": rc(n), "dbp": cd(th[0] / sd, se[0] / sd), "pp": cd(th[1] / sd, se[1] / sd),
         "dbp_native": cd(th[0], se[0]), "pp_native": cd(th[1], se[1]),
         "dbp_minus_pp": cd((th[0] - th[1]) / sd, float(np.sqrt(max(V[0, 0] + V[1, 1] - 2 * V[0, 1], 0))) / sd),
         "obs_dbp": cd(bn[0] / sd, sen[0] / sd), "obs_pp": cd(bn[1] / sd, sen[1] / sd),
         "F_cond_dbp": sw_f(Z, X, 0, n, b["k"], cl), "F_cond_pp": sw_f(Z, X, 1, n, b["k"], cl)}
    return o, IF / sd


def ladder_point(b, scores, joint):
    """Point estimates that need the calibration (bootstrap target), SD units: per score IV, IVrc (calibration with
    scores), obs, obs_rc (own-component calibration: marginal slope); joint: IV dbp, pp and obs_rc dbp, pp on the
    bivariate calibration (multivariable regression calibration)."""
    out = {}
    y, sd = b["y"], b["sd"]
    for j, s in enumerate(scores):
        e = "SDP".index(EXPO[s])
        g, x, uo, ui = b["G"][:, j], b["X"][:, e], b["Uo"][:, e], b["Ui"][:, e]
        iv = (g @ y) / (g @ x) if abs(g @ x) > 1e-12 else np.nan
        ivrc = (g @ y) / (g @ ui) if abs(g @ ui) > 1e-12 else np.nan
        obs = (x @ y) / (x @ x)
        orc = (uo @ y) / (uo @ uo) if (uo @ uo) > 1e-12 else np.nan
        out[s] = np.array([iv, ivrc, obs, orc]) / sd
    if joint:
        jd, jp = joint
        try:
            th = np.linalg.solve(b["G"][:, [jd, jp]].T @ b["X"][:, [1, 2]], b["G"][:, [jd, jp]].T @ y)
            bo = np.linalg.solve(b["Uj"][:, [1, 2]].T @ b["Uj"][:, [1, 2]], b["Uj"][:, [1, 2]].T @ y)
            out["joint"] = np.r_[th, bo] / sd
        except np.linalg.LinAlgError:
            out["joint"] = np.full(4, np.nan)
    return out


# ============================================================================ sections

def boot_draw(ctx, rng):
    """Person-level bootstrap over the union of unrelated analysis participants and calibration participants: each
    drawn person brings his analysis row and/or his calibration row (the two samples share people)."""
    a_map, c_map = ctx["bmap"]
    k = rng.integers(0, len(a_map), len(a_map))
    ia, ic = a_map[k], c_map[k]
    return ia[ia >= 0], ic[ic >= 0]


def sec_main(ctx, boot, rng):
    """Instruments, main per score x bed (with calibrated ladder and bootstrap), joint model, PWV split."""
    T, rows, scores, covc, U = ctx["T"], ctx["u_rows"], ctx["scores"], ctx["cov10"], ctx["U"]
    joint = ctx["joint"]
    beds = [b for b in BEDS if b in T.ix]
    res = {"beds": beds, "per_score": {s: {} for s in scores}, "joint": {}, "boot_ok": 0}
    pts = {}
    for bed in beds:
        b = bed_block(T, rows, bed, scores, covc, U, sd_ref=ctx["sd_pop"].get(bed))
        if "G" not in b:
            for s in scores:
                res["per_score"][s][bed] = {"n": rc(b["n"])}
            continue
        for j, s in enumerate(scores):
            e = "SDP".index(EXPO[s])
            ent, _ = iv_entry(b, s, j, e)
            ob, IFo = ols1(b["X"][:, e], b["y"])
            seo = float(np.sqrt(vif(IFo, b["n"], b["k"] + 1)[0, 0]))
            ent["obs"] = cd(ob / b["sd"], seo / b["sd"])
            ent["obs_native"] = cd(ob, seo)
            res["per_score"][s][bed] = ent
        if joint:
            res["joint"][bed], _ = joint_entry(b, *joint)
        if U is not None:
            pts[bed] = ladder_point(b, scores, joint)
            for s in scores:
                iv, ivrc, obs, orc = pts[bed][s]
                res["per_score"][s][bed].update({"ivrc": {"b": ivrc}, "obs_rc": {"b": orc},
                                                 "ratio_iv_obs_rc": {"b": iv / orc if orc else None},
                                                 "diff_iv_obs_rc": {"b": iv - orc}})
            if joint and res["joint"].get(bed):
                res["joint"][bed]["obs_rc_dbp"] = {"b": pts[bed]["joint"][2]}
                res["joint"][bed]["obs_rc_pp"] = {"b": pts[bed]["joint"][3]}
    # bootstrap: persons resampled; analysis and calibration rows of a drawn person enter together (boot_draw)
    if U is not None and boot > 0 and pts:
        C = ctx["C"]
        reps = {bed: [] for bed in pts}
        for _ in range(boot):
            ib, ic = boot_draw(ctx, rng)
            sub = T.X[ib]
            Ub = ustack(calibrate(C.X[ic], C.ix, sub, T.ix, ctx["specs"])[0])
            if Ub is None:
                continue
            Tb = Tab.__new__(Tab)
            Tb.cols, Tb.ix, Tb.X = T.cols, T.ix, sub
            for bed in pts:
                bb = bed_block(Tb, None, bed, scores, covc, Ub)
                if "G" in bb:
                    reps[bed].append(ladder_point(bb, scores, joint))
        res["boot_ok"] = min((len(v) for v in reps.values()), default=0)
        for bed, rr in reps.items():
            if len(rr) < 20:
                continue
            for s in scores:
                a = np.array([r[s] for r in rr])
                ent = res["per_score"][s][bed]
                if ent.get("iv"):
                    ent["iv"]["boot_ci"] = A.pct(a[:, 0])
                ent["ivrc"]["ci"] = A.pct(a[:, 1])
                ent["obs_rc"]["ci"] = A.pct(a[:, 3])
                ent["obs_rc"]["boot_se"] = float(np.nanstd(a[:, 3]))
                ent["ratio_iv_obs_rc"]["ci"] = A.pct(a[:, 0] / a[:, 3])
                ent["diff_iv_obs_rc"]["ci"] = A.pct(a[:, 0] - a[:, 3])
                lo, hi = ent["obs_rc"]["ci"]
                ent["ratio_iv_obs_rc"]["denominator_ci_includes_0"] = bool(lo is None or lo <= 0 <= hi)
            if joint and res["joint"].get(bed):
                a = np.array([r["joint"] for r in rr])
                res["joint"][bed]["obs_rc_dbp"]["ci"] = A.pct(a[:, 2])
                res["joint"][bed]["obs_rc_pp"]["ci"] = A.pct(a[:, 3])
    return res


def sec_instruments(ctx):
    T, scores = ctx["T"], ctx["scores"]
    out = {"score_corr_unrelated": None, "per_score": {}}
    rows = ctx["u_rows"]
    G = np.column_stack([T.col("g_" + s, rows) for s in scores])
    ok = np.isfinite(G).all(axis=1)
    if ok.sum() >= MIN_N:
        out["score_corr_unrelated"] = {s: {t: float(np.corrcoef(G[ok, i], G[ok, j])[0, 1]) for j, t in enumerate(scores)}
                                       for i, s in enumerate(scores)}
    for s in scores:
        e = EXPO[s]
        o = {"exposure": XCOL[e], "main": fs_entry(T, rows, s, e, ctx["cov10"]),
             "cross": {XCOL[f]: fs_entry(T, rows, s, f, ctx["cov10"]).get("b_mmHg") for f in "SDP" if f != e}}
        if ctx.get("cov20"):
            o["pcs20"] = fs_entry(T, rows, s, e, ctx["cov20"])
        o["untreated"] = fs_entry(T, rows & (T.col("antihtn") == 0), s, e, ctx["cov10"])
        o["common_core"] = fs_entry(T, rows & ctx["core_rows"], s, e, ctx["cov10"])
        out["per_score"][s] = o
    return out


def sec_contrasts(ctx):
    """Four core beds in their common sample (unrelated): IV per bed (SD units), joint IF covariance, contrasts."""
    T, scores, covc = ctx["T"], ctx["scores"], ctx["cov10"]
    rows = ctx["u_rows"] & ctx["core_rows"]
    out = {"n": rc(int(rows.sum())), "per_score": {}}
    blocks = {bed: bed_block(T, rows, bed, scores, covc) for bed in CORE}
    if any("G" not in b for b in blocks.values()):
        return out
    n0 = min(b["n"] for b in blocks.values())
    if len({b["n"] for b in blocks.values()}) != 1:
        raise ValueError("common sample differs between beds")
    for j, s in enumerate(scores):
        e = "SDP".index(EXPO[s])
        th, IFs = [], []
        for bed in CORE:
            ent, IF = iv_entry(blocks[bed], s, j, e)
            th.append((ent.get("iv") or {}).get("b", np.nan))
            IFs.append(IF)
        if not np.isfinite(th).all():
            out["per_score"][s] = {"error": "undefined IV (zero first stage)"}
            continue
        IF = np.column_stack(IFs)
        V = vif(IF, n0, blocks[CORE[0]]["k"] + 1)
        con, het = A.contrasts(np.array(th), V, CORE, PAIRS)
        out["per_score"][s] = {"theta": dict(zip(CORE, th)), "se": dict(zip(CORE, np.sqrt(np.diag(V)))),
                               "sd_bed": {bed: blocks[bed]["sd"] for bed in CORE}, "diff": con, "het": het}
    if ctx["joint"]:
        th, IFs = {"dbp": [], "pp": []}, {"dbp": [], "pp": []}
        for bed in CORE:
            o, IF = joint_entry(blocks[bed], *ctx["joint"])
            if o is None:
                return out
            for c_, i_ in (("dbp", 0), ("pp", 1)):
                th[c_].append(o[c_]["b"])
                IFs[c_].append(IF[:, i_])
        out["joint"] = {}
        for c_ in ("dbp", "pp"):
            V = vif(np.column_stack(IFs[c_]), n0, blocks[CORE[0]]["k"] + 2)
            con, het = A.contrasts(np.array(th[c_]), V, CORE, PAIRS)
            out["joint"][c_] = {"theta": dict(zip(CORE, th[c_])), "diff": con, "het": het}
    return out


def iv_table(ctx, rows, covc, beds=None):
    """Measured-exposure 2SLS per score x bed in the given rows (sensitivity samples, strata).  SD units use the SD
    of the bed in the unrelated analysis rows (ctx sd_pop), so estimates are on one scale across samples and strata
    (v1.1 review: stratum-specific SDs made the Cochran Q test SD heterogeneity, not effect modification)."""
    T, scores = ctx["T"], ctx["scores"]
    out = {s: {} for s in scores}
    for bed in beds or [b for b in BEDS if b in T.ix]:
        b = bed_block(T, rows, bed, scores, covc, min_n=MIN_STRAT, sd_ref=ctx["sd_pop"].get(bed))
        if "G" not in b:
            for s in scores:
                out[s][bed] = {"n": rc(b["n"])}
            continue
        for j, s in enumerate(scores):
            out[s][bed], _ = iv_entry(b, s, j, "SDP".index(EXPO[s]))
        if ctx["joint"]:
            out.setdefault("joint", {})[bed], _ = joint_entry(b, *ctx["joint"])
    return out


def sec_strata(ctx):
    T, rows, scores = ctx["T"], ctx["u_rows"], ctx["scores"]
    age = T.col("age")
    c1, c2 = np.nanquantile(age[rows], [1 / 3, 2 / 3])
    strata = {"age_T1": rows & (age < c1), "age_T2": rows & (age >= c1) & (age < c2), "age_T3": rows & (age >= c2),
              "women": rows & (T.col("sex_male") == 0), "men": rows & (T.col("sex_male") == 1)}
    out = {"age_cuts": [round(float(c1), 1), round(float(c2), 1)], "strata": {}, "first_stage": {}, "het": {}}
    for k, r in strata.items():
        covc = [c for c in ctx["cov10"] if not (k in ("women", "men") and c == "sex_male")]
        out["strata"][k] = iv_table(ctx, r, covc)
        out["first_stage"][k] = {s: fs_entry(T, r, s, EXPO[s], covc) for s in scores}
    for s in scores:
        out["het"][s] = {}
        for bed in out["strata"]["age_T1"][s]:
            g = lambda k: (out["strata"][k][s].get(bed) or {}).get("iv_native") or {}  # Q on the native scale
            out["het"][s][bed] = {"age": q_het([(g(k).get("b"), g(k).get("se")) for k in ("age_T1", "age_T2", "age_T3")]),
                                  "sex": q_het([(g(k).get("b"), g(k).get("se")) for k in ("women", "men")])}
    return out


def sec_within_sib(ctx):
    """Full-sib pair differences; cluster-robust by sibship; population vs within-sib."""
    T, scores, fs = ctx["Tall"], ctx["scores"], ctx["fs"]
    pid = ctx["pid_all"]
    pos = pd.Series(np.arange(len(pid)), index=pid)
    pos = pos[~pos.index.duplicated()]
    a, b = fs["ID1"].astype(str).to_numpy(), fs["ID2"].astype(str).to_numpy()
    have = np.isin(a, pos.index) & np.isin(b, pos.index)
    a, b = a[have], b[have]
    out = {"pairs_with_bp_row": rc(len(a)), "per_score": {s: {} for s in scores}, "first_stage": {}}
    if len(a) < MIN_PAIRS:
        return out
    fam = unionfind(a, b)
    lab = pd.factorize(np.array([fam[x] for x in a], dtype=object))[0]
    out["sibships"] = rc(int(lab.max()) + 1)
    ia, ib = pos[a].to_numpy(), pos[b].to_numpy()
    D = Tab.__new__(Tab)
    D.cols, D.ix = T.cols, T.ix
    D.X = T.X[ia] - T.X[ib]
    covc = ["age", "age2", "sex_male"]
    sd_pop = ctx["sd_pop"]
    for bed in [x for x in BEDS if x in T.ix]:
        blk = bed_block(D, None, bed, scores, covc, min_n=MIN_PAIRS)
        if "G" not in blk:
            for s in scores:
                out["per_score"][s][bed] = {"n_pairs": rc(blk["n"])}
            continue
        # no intercept for pair differences (pair order is arbitrary): drop the constant column effect by re-residualizing
        ok = blk["ok"]
        C = np.column_stack([D.col(c)[ok] for c in covc])
        M = np.column_stack([D.col("g_" + s)[ok] for s in scores] + [D.col(XCOL[c])[ok] / 10 for c in "SDP"] +
                            [D.col(bed)[ok] * ORIENT.get(bed, 1)])
        R, k = resid(C[:, C.std(axis=0) > 0], M)
        ns = len(scores)
        sdb = sd_pop.get(bed) or float(np.nanstd(T.col(bed)))
        blk2 = {"n": blk["n"], "k": k, "G": R[:, :ns], "X": R[:, ns:ns + 3], "y": R[:, ns + 3], "sd": sdb}
        cl = pd.factorize(lab[ok])[0]
        for j, s in enumerate(scores):
            ent, _ = iv_entry(blk2, s, j, "SDP".index(EXPO[s]), cl)
            ent["n_pairs"] = ent.pop("n")
            pop = ((ctx.get("main") or {}).get("per_score", {}).get(s, {}).get(bed) or {}).get("iv")
            if pop and ent["iv"] and ent["iv"].get("se"):
                dlt = pop["b"] - ent["iv"]["b"]
                sdd = float(np.sqrt(pop["se"] ** 2 + ent["iv"]["se"] ** 2))
                ent["pop_minus_sib"] = cd(dlt, sdd)
            out["per_score"][s][bed] = ent
            if bed == "pwv":
                out["first_stage"][s] = {"b_mmHg": ent["fs_mmHg"], "F": ent["F"], "n_pairs": ent["n_pairs"]}
        if ctx["joint"]:
            out.setdefault("joint", {})[bed], _ = joint_entry(blk2, *ctx["joint"], cl)
    return out


def lin_g(covs, g, y, extra, ok):
    """Coefficient of g in y ~ g (+ extra) + covs on rows ok: (b, se, n) with HC1 SE."""
    n = int(ok.sum())
    C = np.column_stack([np.ones(n)] + [c[ok] for c in covs])
    C = C[:, np.r_[True, C[:, 1:].std(axis=0) > 0]]
    M = np.column_stack([g[ok]] + ([extra[ok]] if extra is not None else []) + [y[ok]])
    R, k = resid(C, M)
    bb, IF = olsk(R[:, :-1], R[:, -1])
    return float(bb[0]), float(np.sqrt(vif(IF[:, :1], n, k + R.shape[1] - 1)[0, 0])), n


def sec_negative(ctx, boot=0, rng=None):
    """Scores against acquisition variables that should not depend on genotype.  The device-SBP test conditions on
    usual SBP calibrated with the scores; its SE and CI come from the person-level bootstrap of the analysis and
    calibration rows (the calibration coefficients are estimated), else (boot 0) the analytic SE that ignores the
    calibration error."""
    T, rows, scores, covc = ctx["T"], ctx["u_rows"], ctx["scores"], ctx["cov10"]
    out = {s: {} for s in scores}
    Ui_S = ctx["U"][:, 3] if ctx["U"] is not None else None
    covs = [T.col(c) for c in covc]
    untr = T.col("antihtn") == 0
    for s in scores:
        g = T.col("g_" + s)

        def lin(y, extra=None, mask=None):
            parts = covs + [g, y] + ([extra] if extra is not None else [])
            ok = rows & np.isfinite(np.column_stack(parts)).all(axis=1) & (True if mask is None else mask)
            if ok.sum() < MIN_N:
                return {"n": rc(int(ok.sum()))}
            b, se, n = lin_g(covs, g, y, extra, ok)
            return {"n": rc(n), **cd(b, se)}
        if Ui_S is not None:
            # untreated analysis rows only: the calibration is fitted in people untreated at both visits, and the null
            # is exact only where the calibration transports (review v1.1)
            e = lin(T.col("dev_bsbp"), Ui_S, untr)
            if boot > 0 and "b" in e:
                C = ctx["C"]
                bs = []
                for _ in range(boot):
                    ib, ic = boot_draw(ctx, rng)
                    ubd = calibrate(C.X[ic], C.ix, T.X[ib], T.ix, ctx["specs"], only=("biv_g",))[0].get("biv_g")
                    if ubd is None:
                        continue
                    ub = ubd[:, 0]
                    cv = [c[ib] for c in covs]
                    gb, yb = g[ib], T.col("dev_bsbp")[ib]
                    ok = np.isfinite(np.column_stack(cv + [gb, yb, ub])).all(axis=1) & untr[ib]
                    if ok.sum() >= MIN_N:
                        bs.append(lin_g(cv, gb, yb, ub, ok)[0])
                if len(bs) >= 20:
                    se_b = float(np.std(bs, ddof=1))
                    e.update({"se_analytic": e["se"], "se": se_b, "lo": A.pct(bs)[0], "hi": A.pct(bs)[1],
                              "p": pv(e["b"], se_b), "boot_n": len(bs)})
            out[s]["device_sbp_given_usual_mmHg"] = e
        out[s]["acute_dev_naive_mmHg"] = lin(T.col("acute_dev"))
        out[s]["acq_time_min"] = lin(T.col("acq_min"))
        out[s]["calendar_years"] = lin(T.col("cal_years"))
        out[s]["pwv_correction_ms"] = lin(T.col("corr_pwv"))
        out[s]["height_cm"] = lin(T.col("height"))
        # calendar quarter: score on quarter indicators (quarters with >= MINCELL rows)
        q = ctx["quarter"]
        ok = rows & np.isfinite(np.column_stack(covs + [g])).all(axis=1) & pd.notna(q)
        vc = pd.Series(q[ok]).value_counts()
        keep = vc[vc >= A.MINCELL].index
        ok &= pd.Series(q).isin(keep).to_numpy()
        n = int(ok.sum())
        if n >= MIN_N and len(keep) >= 2:
            ref = vc.index[0]
            Dm = pd.get_dummies(pd.Categorical(q[ok], categories=list(keep)), dtype=float).drop(columns=[ref]).to_numpy()
            C = np.column_stack([np.ones(n)] + [c[ok] for c in covs])
            C = C[:, np.r_[True, C[:, 1:].std(axis=0) > 0]]
            R, k = resid(C, np.column_stack([Dm, g[ok]]))
            Xd, y = R[:, :-1], R[:, -1]
            Xd = Xd[:, Xd.std(axis=0) > 1e-9]
            bq, IF = olsk(Xd, y)
            V = vif(IF, n, k + Xd.shape[1])
            try:
                W = float(bq @ np.linalg.solve(V, bq))
                out[s]["quarter"] = {"n": rc(n), "quarters": len(keep), "chi2": W, "df": Xd.shape[1],
                                     "p": float(stats.chi2.sf(W, Xd.shape[1]))}
            except np.linalg.LinAlgError as ex:
                out[s]["quarter"] = {"error": err(ex)}
        else:
            out[s]["quarter"] = {"n": rc(n)}
    return out


# ============================================================================ driver

def prepare(fr, S, cov, unrel, fs):
    """Context for the sections: analysis table, calibration, residualization covariates, flags."""
    fr2, acute_info = derive(fr)
    d, cal, scores, pcs = build_samples(fr2, S, cov)
    sc = standardize_scores(d, cal, scores, unrel)
    gcols = ["g_" + s for s in scores]
    pc10 = [f"pc{i}" for i in range(1, N_PCS + 1)]
    pc20 = [f"pc{i}" for i in range(1, ALT_PCS + 1)]
    has20 = all(c in d.columns and d[c].notna().mean() > 0.9 for c in pc20)
    lying = bool(cal is not None and len(cal) and cal[["sbp_lie", "dbp_lie"]].notna().all(axis=1).mean() >= 0.5)
    specs = cal_specs(pc10, gcols, lying)
    cols = (["age", "age2", "sex_male", "sbp_sit", "dbp_sit", "pp_sit", "sbp_lie", "dbp_lie", "pp_lie", "antihtn", "dev_bsbp",
             "acute_dev", "acq_min", "cal_years", "corr_pwv", "height"] + pc20 + gcols + BEDS)
    T = Tab(d, cols)
    ctx = {"scores": scores, "T": T, "Tall": T, "pid_all": d["participant_id"].astype(str).to_numpy(), "fs": fs,
           "u_rows": d["participant_id"].isin(unrel).to_numpy(), "cov10": ["age", "age2", "sex_male"] + pc10,
           "cov20": ["age", "age2", "sex_male"] + pc20 if has20 else None, "quarter": d["cal_q"].to_numpy(dtype=object),
           "specs": specs, "U": None, "C": None, "bmap": None,
           "joint": (scores.index("DBP_gw"), scores.index("PP_gw")) if {"DBP_gw", "PP_gw"} <= set(scores) else None}
    ctx["core_rows"] = np.isfinite(np.column_stack([T.col(b) for b in CORE])).all(axis=1)
    info = {"acute_dev_model": acute_info, "score_scaling_unrelated": {s: {"mean": m, "sd": v} for s, (m, v) in sc.items()},
            "analysis_rows": rc(len(d)), "analysis_rows_unrelated": rc(int(ctx["u_rows"].sum())),
            "untreated_unrelated": rc(int((ctx["u_rows"] & (T.col("antihtn") == 0)).sum())),
            "common_core_unrelated": rc(int((ctx["u_rows"] & ctx["core_rows"]).sum())),
            "visit_of_row": {k: rc(v) for k, v in d["research_stage"].value_counts().items()},
            "lying_bp_in_calibration": lying, "pcs20_used": has20}
    if cal is not None and len(cal) >= 100:
        try:
            need = ["sbp_rep", "dbp_rep", "pp_rep"] + [p for _, ps in specs.values() for st in ps for p in st]
            C = Tab(cal, need)
            miss = [c for c in need if c not in C.ix]
            if miss:
                raise KeyError(f"calibration columns missing: {miss}")
            Ud, fits = calibrate(C.X, C.ix, T.X, T.ix, specs)
            U = ustack(Ud)
            if U is not None:
                ctx["C"], ctx["U"] = C, U
                # person-level bootstrap map: union of unrelated analysis participants and calibration participants
                pid = ctx["pid_all"]
                ua = np.flatnonzero(ctx["u_rows"])
                cpid = cal["participant_id"].astype(str).to_numpy()
                people = pd.Index(pd.unique(np.concatenate([pid[ua], cpid])))
                a_map, c_map = np.full(len(people), -1), np.full(len(people), -1)
                a_map[people.get_indexer(pid[ua])] = ua
                c_map[people.get_indexer(cpid)] = np.arange(len(cpid))
                ctx["bmap"] = (a_map, c_map)
                info["calibration"] = {nm: cal_info(f, specs[nm][0], specs[nm][1]) for nm, f in fits.items()}
                info["calibration_rows"] = rc(len(cal))
                info["calibration_rows_also_analysis_rows"] = rc(int((c_map[a_map >= 0] >= 0).sum()))
                info["usual_sbp_rc_unrelated"] = A.qtiles(pd.Series(U[ctx["u_rows"], 0]))
        except Exception as ex:  # the IV sections do not need the calibration
            ctx["C"], ctx["U"], ctx["bmap"] = None, None, None
            info["calibration"] = {"error": err(ex)}
    if "calibration" not in info:
        info["calibration"] = {"error": "calibration unavailable (fewer than 100 complete calibration rows)"}
    sdp = {}
    for bed in [b for b in BEDS if b in T.ix]:
        v = T.col(bed, ctx["u_rows"]) * ORIENT.get(bed, 1)
        sdp[bed] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > MIN_N else None
    ctx["sd_pop"] = sdp
    info["bed_n_unrelated"] = {bed: rc(int(np.isfinite(T.col(bed, ctx["u_rows"])).sum())) for bed in BEDS if bed in T.ix}
    return ctx, info


def analyse(fr, S, cov, unrel, fs, boot, seed, sections=None):
    rng = np.random.default_rng(seed)
    R = {}
    try:
        ctx, info = prepare(fr, S, cov, unrel, fs)
    except Exception as ex:
        R["samples"] = {"error": err(ex)}
        log(f"prepare FAILED: {R['samples']['error']}")
        return R
    R["samples"] = info
    secs = sections or ["instruments", "main", "contrasts", "strata", "untreated", "pcs20", "within_sib", "negative"]
    fns = {"instruments": lambda: sec_instruments(ctx), "main": lambda: sec_main(ctx, boot, rng),
           "contrasts": lambda: sec_contrasts(ctx), "strata": lambda: sec_strata(ctx),
           "untreated": lambda: iv_table(ctx, ctx["u_rows"] & (ctx["T"].col("antihtn") == 0), ctx["cov10"]),
           "pcs20": lambda: iv_table(ctx, ctx["u_rows"], ctx["cov20"]) if ctx["cov20"] else {"error": "PC11-20 unavailable"},
           "within_sib": lambda: sec_within_sib(ctx), "negative": lambda: sec_negative(ctx, boot, rng)}
    for k in secs:
        guard(R, k, fns[k])
        if k == "main" and "error" not in R[k]:
            ctx["main"] = R[k]
    return R


# ============================================================================ digest

def f3(x):
    return v3(x)


def ci(c, key="ci"):
    if not isinstance(c, dict) or c.get("b", c.get("d")) is None:
        return "NA"
    b = c.get("b", c.get("d"))
    if "lo" in c and key == "ci":
        return f"{f3(b)} [{f3(c['lo'])},{f3(c['hi'])}]"
    cc = c.get(key) or c.get("ci") or [None, None]
    return f"{f3(b)} [{f3(cc[0])},{f3(cc[1])}]"


def pf(c):
    return f3((c or {}).get("p")) if isinstance(c, dict) else "NA"


def digest(R):
    g = A.g
    sm = R.get("samples", {})
    L = [f"F1 BP FOOTPRINT {R.get('version')} | atlas {R.get('atlas_version')} | prs {R.get('prs_version')} | seed {R.get('seed')}"
         f" boot {R.get('boot')} | runtime {f3(R.get('runtime_s'))}s",
         "units: SD of bed (unrelated rows; core-4: common sample), adverse, per 10 mmHg; IV 2SLS on visit BP; IVrc on calibrated "
         "usual BP; obs on visit BP; obs_rc on own-component calibrated usual BP; adj age age2 sex PC1-10",
         f"flow {R.get('flow')}",
         f"samples rows {sm.get('analysis_rows')} unrelated {sm.get('analysis_rows_unrelated')} untreated {sm.get('untreated_unrelated')} "
         f"core4 {sm.get('common_core_unrelated')} cal rows {sm.get('calibration_rows')} (in analysis {sm.get('calibration_rows_also_analysis_rows')}) "
         f"lying {sm.get('lying_bp_in_calibration')} "
         f"pcs20 {sm.get('pcs20_used')} visits {sm.get('visit_of_row')}",
         "bed n (unrelated): " + " ".join(f"{k} {v}" for k, v in (sm.get("bed_n_unrelated") or {}).items())]
    cal = sm.get("calibration") or {}
    if "error" in cal:
        L.append(f"calibration ERROR {cal['error']}")
    else:
        def calset(fits, keys):
            return "; ".join(f"set{i} n {f['n']} r2 " + "/".join(f3(g(f, "r2", k)) for k in keys) + " lambda " +
                             "/".join(f3(g(f, "lambda", k)) for k in keys) for i, f in enumerate(fits or []) if f)
        L.append("calibration own-component (obs_rc) n r2/lambda, sitting; + lying: " + " | ".join(
            f"{c} " + "; ".join(f"{f['n']} {f3(g(f, 'r2', CAL_T[c]))}/{f3(g(f, 'lambda', CAL_T[c]))}"
                                for f in (cal.get("own_" + c) or []) if f) for c in "SDP"))
        for nm, lbl in (("biv", "SBP+DBP (joint obs_rc)"), ("biv_g", "SBP+DBP+scores (IVrc, device)")):
            L.append(f"calibration {lbl} r2 sbp/dbp, lambda sbp/dbp: {calset(cal.get(nm), ['sbp_rep', 'dbp_rep'])}")
    L.append("# coverage: score | rows in | matched | coverage | direct swap flip pal kept/dropped | weights SHA ok")
    for s, c in (R.get("coverage") or {}).items():
        L.append(f"{s} | {c.get('rows_in')} | {c.get('matched')} | {f3(c.get('coverage'))} | {c.get('direct')} {c.get('swap')} "
                 f"{c.get('flip')} {c.get('pal_kept')}/{c.get('pal_dropped')} | {g(R, 'weights', s, 'sha256_ok')}")
    ins = R.get("instruments") or {}
    L.append("# instruments: score>BP | first stage mmHg/SD [CI] | F | partial R2 | n || F pcs20 | F untreated | F core4 | F sib | "
             "cross mmHg/SD")
    if "error" in ins:
        L.append(f"instruments ERROR {ins['error']}")
    sib_fs = g(R, "within_sib", "first_stage") or {}
    SC = [s for s in (R.get("scores") or []) if s in (ins.get("per_score") or {})]
    for s, o in ((s, ins["per_score"][s]) for s in SC):
        m = o.get("main") or {}
        L.append(f"{s}>{o.get('exposure')} | {ci(m.get('b_mmHg'))} | {f3(m.get('F'))} | {f3(m.get('r2_partial'))} | {m.get('n')} || "
                 f"{f3(g(o, 'pcs20', 'F'))} | {f3(g(o, 'untreated', 'F'))} | {f3(g(o, 'common_core', 'F'))} | "
                 f"{f3(g(sib_fs, s, 'F'))} | " + " ".join(f"{k} {f3((v or {}).get('b'))}" for k, v in (o.get("cross") or {}).items()))
    sc = ins.get("score_corr_unrelated")
    if sc:
        ks = list(sc)
        L.append("score corr: " + "; ".join(f"{a}-{b} {f3(sc[a][b])}" for i, a in enumerate(ks) for b in ks[i + 1:]))
    mn = R.get("main") or {}
    L.append(f"# main (unrelated; boot ok {mn.get('boot_ok')}): score bed | n | F | IV [CI] p | IVrc [boot CI] | obs [CI] | obs_rc "
             "[boot CI] | IV/obs_rc [boot CI] (~ = obs_rc CI includes 0)")
    if "error" in mn:
        L.append(f"main ERROR {mn['error']}")
    order = [b for b in BEDS if b in (mn.get("beds") or BEDS)]
    for s in (R.get("scores") or []):
        beds = g(mn, "per_score", s) or {}
        for bed, e in ((b, beds[b]) for b in order if b in beds):
            if "iv" not in e:
                L.append(f"{s} {bed} | n {e.get('n')} | too few")
                continue
            rt = e.get("ratio_iv_obs_rc") or {}
            L.append(f"{s} {bed} | {e.get('n')} | {f3(e.get('F'))} | {ci(e['iv'])} p {pf(e['iv'])} | {ci(e.get('ivrc'))} | "
                     f"{ci(e.get('obs'))} | {ci(e.get('obs_rc'))} | {ci(rt)}{'~' if rt.get('denominator_ci_includes_0') else ''}")
    ct = R.get("contrasts") or {}
    L.append(f"# core-4 common sample (n {ct.get('n')}): score | IV pwv imt art_w abi | pwv-imt [CI] | pwv-art_w | pwv-abi | "
             "imt-abi | Wald p")
    if "error" in ct:
        L.append(f"contrasts ERROR {ct['error']}")

    def conline(lbl, c):
        th, df_ = c.get("theta", {}), c.get("diff", {})
        return (f"{lbl} | " + " ".join(f3(th.get(b)) for b in CORE) + " | " +
                " | ".join(ci(df_.get(k)) for k in ("pwv_minus_imt", "pwv_minus_art_w", "pwv_minus_abi_min", "imt_minus_abi_min"))
                + f" | {f3(g(c, 'het', 'p'))}")
    for s in (R.get("scores") or []):
        if g(ct, "per_score", s):
            L.append(conline(s, ct["per_score"][s]))
    for c_, c in (ct.get("joint") or {}).items():
        L.append(conline(f"joint_{c_}", c))
    L.append("# joint DBP_gw+PP_gw -> DBP, PP: bed | n | F_cond dbp/pp | IV dbp [CI] | IV pp [CI] | dbp-pp [CI] | obs_rc dbp | obs_rc pp")
    for bed, o in ((b, (mn.get("joint") or {}).get(b)) for b in order):
        if not o:
            continue
        L.append(f"{bed} | {o.get('n')} | {f3(o.get('F_cond_dbp'))}/{f3(o.get('F_cond_pp'))} | {ci(o.get('dbp'))} | {ci(o.get('pp'))} | "
                 f"{ci(o.get('dbp_minus_pp'))} | {ci(o.get('obs_rc_dbp'))} | {ci(o.get('obs_rc_pp'))}")
    L.append("# PWV split, m/s per 10 mmHg: score | total IV [CI] | structural (Pd* 80) | load-dependent | joint dbp tot/str/load | "
             "joint pp tot/str/load")
    for s in (R.get("scores") or []):
        beds = g(mn, "per_score", s) or {}
        L.append(f"{s} | " + " | ".join(ci((beds.get(b) or {}).get("iv_native")) for b in PWVF))
    jo = mn.get("joint") or {}
    if jo:
        L.append("joint | dbp " + " / ".join(ci((jo.get(b) or {}).get("dbp_native")) for b in PWVF) + " | pp " +
                 " / ".join(ci((jo.get(b) or {}).get("pp_native")) for b in PWVF))
    st = R.get("strata") or {}
    L.append(f"# strata (age tertile cuts {' '.join(f3(x) for x in st.get('age_cuts') or [])} y; SD of all unrelated rows; Q on native): "
             "score bed | IV T1 | T2 | T3 | Q p age | women | men | Q p sex")
    if "error" in st:
        L.append(f"strata ERROR {st['error']}")
    fsl = st.get("first_stage") or {}
    for s in (R.get("scores") or []):
        L.append(f"{s} first stage mmHg/SD (F): " + " ".join(
            f"{k} {f3(g(fsl, k, s, 'b_mmHg', 'b'))} ({f3(g(fsl, k, s, 'F'))})" for k in ("age_T1", "age_T2", "age_T3", "women", "men")))
    show = {"SBP_gw": ["pwv", "imt", "art_w", "abi_min", "sokolow"], "DBP_gw": ["pwv", "imt"], "PP_gw": ["pwv", "imt"]}
    for s, beds in show.items():
        for bed in beds:
            ss = st.get("strata") or {}
            gi = lambda k: ci(g(ss, k, s, bed, "iv"))
            L.append(f"{s} {bed} | {gi('age_T1')} | {gi('age_T2')} | {gi('age_T3')} | {f3(g(st, 'het', s, bed, 'age', 'p'))} | "
                     f"{gi('women')} | {gi('men')} | {f3(g(st, 'het', s, bed, 'sex', 'p'))}")
    for key, title in (("untreated", "untreated only"), ("pcs20", "PC1-20")):
        blk = R.get(key) or {}
        L.append(f"# {title}: score | F | IV pwv | imt | art_w | abi_min | cavi0 | pwv_struct (SD/10 mmHg)")
        if "error" in blk:
            L.append(f"{key} ERROR {blk['error']}")
            continue
        for s in (R.get("scores") or []):
            e = blk.get(s) or {}
            L.append(f"{s} | {f3(g(e, 'pwv', 'F'))} | " + " | ".join(ci(g(e, b, 'iv')) for b in ("pwv", "imt", "art_w", "abi_min", "cavi0",
                                                                                                  "pwv_struct")))
        jb = blk.get("joint") or {}
        if jb:
            L.append("joint dbp/pp | " + " | ".join(f"{b} {f3(g(jb, b, 'dbp', 'b'))}/{f3(g(jb, b, 'pp', 'b'))}"
                                                    for b in ("pwv", "imt", "art_w", "abi_min")))
    ws = R.get("within_sib") or {}
    L.append(f"# within full-sib pairs (pairs {ws.get('pairs_with_bp_row')}, sibships {ws.get('sibships')}; cluster-robust): "
             "score bed | n pairs | F | IV [CI] | pop-sib [CI] p")
    if "error" in ws:
        L.append(f"within_sib ERROR {ws['error']}")
    for s in (R.get("scores") or []):
        beds = g(ws, "per_score", s) or {}
        for bed in CORE:
            e = beds.get(bed) or {}
            L.append(f"{s} {bed} | {e.get('n_pairs')} | {f3(e.get('F'))} | {ci(e.get('iv'))} | {ci(e.get('pop_minus_sib'))} "
                     f"p {pf(e.get('pop_minus_sib'))}")
    jw = ws.get("joint") or {}
    if jw:
        L.append("sib joint dbp/pp (F_cond): " + "; ".join(
            f"{b} {ci(g(jw, b, 'dbp'))}/{ci(g(jw, b, 'pp'))} ({f3(g(jw, b, 'F_cond_dbp'))}/{f3(g(jw, b, 'F_cond_pp'))})"
            for b in ("pwv", "imt") if jw.get(b)))
    ng = R.get("negative") or {}
    L.append("# negative controls per SD score: device SBP | usual (untreated; mmHg) | naive acute dev (mmHg) | acq time (min) | calendar (y) | "
             "quarter chi2/df p | PWV correction (m/s) | height (cm)")
    if "error" in ng:
        L.append(f"negative ERROR {ng['error']}")
    for s, o in ((s, ng.get(s)) for s in (R.get("scores") or [])):
        if not isinstance(o, dict):
            continue
        qq = o.get("quarter") or {}
        L.append(f"{s} | {ci(o.get('device_sbp_given_usual_mmHg'))} p {pf(o.get('device_sbp_given_usual_mmHg'))} | "
                 f"{ci(o.get('acute_dev_naive_mmHg'))} | {ci(o.get('acq_time_min'))} p {pf(o.get('acq_time_min'))} | "
                 f"{ci(o.get('calendar_years'))} | {f3(qq.get('chi2'))}/{qq.get('df')} p {f3(qq.get('p'))} | "
                 f"{ci(o.get('pwv_correction_ms'))} | {ci(o.get('height_cm'))}")
    errs = {k: v["error"] for k, v in R.items() if isinstance(v, dict) and "error" in v}
    if errs:
        L.append("## errors: " + "; ".join(f"{k}: {v}" for k, v in errs.items()))
    long_ = [i for i, ln in enumerate(L) if len(ln) > 220]
    if long_:
        log(f"digest: {len(long_)} lines truncated at 220 characters")
    L = [ln.lstrip()[:220] for ln in L]
    if len(L) > 120:
        L = L[:119] + ["(digest truncated at 120 lines)"]
    return "\n".join(L) + "\n"


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", required=True, help="PGS weights folder (f1pgs_*.parquet + *_manifest.tsv)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", default=",".join(SCORES))
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--block", type=int, default=20000, help="max variant rows per genotype read (PRS v1.1)")
    ap.add_argument("--atlas", default=ATLAS_PATH, help="atlas module (read at import time)")
    ap.add_argument("--prs", default=PRS_PATH, help="PRS v1.1 module (read at import time)")
    ap.add_argument("--sections", default="")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    miss = [x for x in API_A if not hasattr(A, x)] + [x for x in API_P if not hasattr(P, x)]
    if miss:
        raise SystemExit(f"atlas/PRS module lacks {miss}")
    scores = [s.strip() for s in a.scores.split(",") if s.strip()]
    bad = [s for s in scores if s not in EXPO]
    if bad:
        raise SystemExit(f"unsupported scores {bad}; supported {list(EXPO)}")
    t0 = time.time()
    log(f"{VERSION}; atlas {A.VERSION} ({os.path.basename(ATLAS_PATH)}); prs {P.VERSION} ({os.path.basename(PRS_PATH)}); "
        f"numpy {np.__version__}; pandas {pd.__version__}; threads {os.environ.get('OMP_NUM_THREADS')}")
    if not str(A.VERSION).startswith("f1_atlas_v1.5"):
        log(f"WARNING: atlas module is {A.VERSION}, not v1.5")
    W, wchk = load_weights(a.weights, scores)
    fam, cov, flow, unrel, fs = load_samples_pairs(a.root)
    t1 = time.time()
    S, coverage = score_samples(a.root, W, fam, cov.index, a.block)
    del W
    log(f"scoring {time.time() - t1:.0f}s")
    fr = A.build_frame(a.root, None)[0].reset_index(drop=True)
    pids = set(fr["participant_id"].astype(str)) | set(map(str, fam))
    R = {"version": VERSION, "atlas_version": A.VERSION, "prs_version": P.VERSION, "seed": f"{a.seed:,}", "boot": a.boot,
         "scores": list(S.columns), "flow": flow, "weights": wchk, "coverage": coverage,
         "settings": {"rho": RHO, "p_ref_mmHg": PREF, "pd_star_mmHg": PSTAR, "n_pcs": N_PCS, "alt_pcs": ALT_PCS,
                      "min_n": MIN_N, "min_pairs": MIN_PAIRS, "min_stratum": MIN_STRAT}}
    secs = [s.strip() for s in a.sections.split(",") if s.strip()] or None
    R.update(analyse(fr, S, cov, unrel, fs, a.boot, a.seed, secs))
    R["runtime_s"] = time.time() - t0
    text = json.dumps(A.clean(R), ensure_ascii=False, indent=1, sort_keys=True)
    A.privacy_check(text, pids)
    dg = digest(json.loads(text))
    A.privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines; total {time.time() - t0:.1f}s")
    logtext = "\n".join(A.LOG) + "\n"
    A.privacy_check(logtext, pids)
    os.makedirs(a.out)
    sums = []
    for fn_, tx in (("results.json", text), ("digest.txt", dg), ("run_log.txt", logtext)):
        with open(os.path.join(a.out, fn_), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn_}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s_[:16] + "  " + s_.split()[-1] for s_ in sums))


if __name__ == "__main__":
    main()
