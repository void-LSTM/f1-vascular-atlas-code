#!/usr/bin/env python3
"""F1 P3 + P4 sandbox module (HPP; writes aggregates only): is the NK/cytotoxic PBMC -> leg PWV signal acute
sympathetic state or immune trait, and how does heart rate (HR) at acquisition split into usual and acute parts by bed?

Specification: docs/NOVELTY_AND_NEW_FINDINGS_20261004.md section 6.2 items 3 (P3) and 4 (P4); section 5.1 #5, #6, #7.
Beds come from the atlas frame (--atlas; build_frame(root, None)), i.e. calendar/day/session-adjusted and
adverse-oriented (o_<bed>, higher = worse).  PBMC modules come from the omics v1.2 loader (--omics; load_rna with the
draw-hour adjustment: batch + draw hour + hour^2 in every RNA model).

Shared heart-rate set (one definition used by every section)
  hr_sit      sitting pulse of the BP device (atlas); ecg_hr = resting ECG hr_bpm (atlas); both kept in 30-200 bpm
  hr_visit    mean of hr_sit and ecg_hr (whichever exist) = visit HR
  hr_noct     median over the nights of the visit's research_stage of sleep heart_rate_mean_during_sleep (WatchPAT;
              30-200 bpm) = usual HR; hr_noct_min the same for heart_rate_min_during_sleep; nights of stage
              'undefined' are not used
  hr_excess   acute HR excess (PRE-SPECIFIED STATE INDEX): hr_visit - E[hr_visit | hr_noct, age, sex], fitted on visits
              without rate-lowering drugs (both visits pooled), applied to all; hr_diff = hr_visit - hr_noct (simple)
  acute_dev   acute SBP deviation (atlas S2: device brachial SBP beyond the visit BP readings)
  rate_drug   beta-blocker (ATC C07), non-dihydropyridine CCB (C08D) or ivabradine (C01EB17) at the visit stage;
              bb = C07; aht_other = C02, C03, C09 or C08 other than C08D
Sections
  H0 flow, HR variable distributions, correlations, night-to-night and between-visit ICCs, timing (acquisition vs BP,
     ECG; nights vs visit date; prior-night gate = visits with a night starting <= 72 h before acquisition)
  H1 HR decomposition per bed (P4 / #7).  Primary sample: no rate-lowering drug; one row per participant (visit 1
     first).  bed_z ~ hr_visit/10 + hr_noct/10 + covariates (Mh1 = M3: age, sex, BMI, sitting SBP, antihypertensives;
     Mh2 = Mh1 + acute_dev).  acute = coefficient of visit HR given nocturnal HR; usual_total = coefficient of nocturnal
     HR when visit HR moves with it (acute + beyond); beyond = usual_total - acute = nocturnal HR given visit HR.
     Pure acquisition physics: beyond = 0; pure usual-HR biology: acute = 0.  TOST (+-0.05 SD per 10 bpm) for non-PWV
     beds.  Four-bed joint fit (robust joint covariance) with pairwise contrasts, 3-df Wald test and reliability-
     disattenuated contrasts (between-visit ICC, bootstrap).  Raw PWV in m/s per 10 bpm.  Within-person change.
     Sensitivities: all participants + rate-drug indicator; hr_sit only; ECG HR only; nocturnal minimum HR;
     regression-calibrated usual HR (nights as replicates; every bed, and raw PWV with bootstrap CI): U_cal =
     E[usual HR | night mean, k nights, visit HR, covariates] with the pooled within-visit night variance of the frame
     visits.  Positive control: hr_excess -> acute_dev.
  H2 beta-blocker positive control (#17, descriptive): users (C07) vs participants without rate-lowering drugs;
     differences in visit, nocturnal and excess HR (bpm) and PWV (m/s); mediation via visit + nocturnal HR (product of
     coefficients, bootstrap CI).  Covariates: age, sex, BMI, sitting SBP, other antihypertensives.
  H3 PBMC modules (P3 / #5 hosting #6).  Pre-specified gene modules (MODULES below; sources in SOURCES).
     replication: omics M3 sample (module -> PWV).  Chain A (visit HR sample): M3 vs + hr_visit vs + hr_visit +
     acute_dev.  Chain B (nocturnal-HR sample; PRIMARY): base M3 | excess (= M3 + hr_excess + acute_dev, primary) |
     visit_hr | decomposed (hr_noct + hr_visit) | all_state (+ acute_dev) | excess_only | acute_dev_only; bootstrap
     delta-beta and retention (excess, all_state).
     Module -> each bed (base and primary), BH q over modules x beds; TOST (+-0.05 SD) for non-PWV beds; common
     four-bed contrasts with Wald test and disattenuated contrasts; state loadings (module -> hr_excess, acute_dev,
     hr_noct; the 'acute-deviation mirror'); beta-blocker difference and module x beta-blocker interaction; module x
     |acquisition - blood draw| (h, same day, <= 6 h); module x draw hour; + CBC lymphocytes; + alerting reaction (first -
     second sitting SBP); + PBMC cell yield (libprep__cells_per_100ul of the matched sample's stage); + smoking, statin,
     systemic steroid (H02), immunosuppressant (L04); + age^2 + age^3 (residual age confounding of age-related modules);
     excluding rate-drug users.
  v1.0.1 (code review, 2026-10-04): usual_given_excess (bed ~ hr_noct + hr_excess) per bed; cell yield keyed by
     the matched RNA sample; L04 in M4; age-cubic sensitivity; the night-before share suppressed when a cell is < 20;
     nocturnal HR kept when the night timestamps fail; tz-robust lab dates; error notes mask digit runs; unknown
     --sections refused.
Privacy (as atlas/omics): aggregates only; counts rounded to 10 (None < 20); quantiles 5-95 % only; any estimate whose
smallest cell (e.g. drug users) is < 20 is suppressed; no participant ids; every output passes the 7+ digit and id scan;
refuses to overwrite --out.
Run (sandbox pheno env, from <WORK_DIR>):
  OMP_NUM_THREADS=4 python f1_nk_hr_v1.py --root <HPP_DATA_ROOT> \
      --atlas f1_atlas_v1_5.py --omics f1_omics_v1_2.py --out f1_nk_hr_out_v1
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

VERSION = "f1_nk_hr_v1.0.1 2026-10-04"
HERE = os.path.dirname(os.path.abspath(__file__))
A = None  # atlas module
O = None  # omics v1.2 module
M3 = ["age", "sex_male", "bmi", "sbp_sit", "antihtn"]
M3_NOAHT = ["age", "sex_male", "bmi", "sbp_sit", "aht_other"]
HR_LIM = (30.0, 200.0)
BEDS_H = ["pwv", "imt", "art_w", "abi_min", "cavi0", "vein_w", "sokolow"]
BEDS_M = ["pwv", "imt", "art_w", "abi_min", "cavi0", "vein_w"]
NONPWV = ("imt", "art_w", "abi_min", "vein_w", "sokolow")
PAIRS4 = [("pwv", "imt"), ("pwv", "art_w"), ("pwv", "abi_min"), ("imt", "art_w"), ("imt", "abi_min"), ("art_w", "abi_min")]
TOST = 0.05        # equivalence margin (bed SD per module SD, or per 10 bpm)
MIN_N = 100
GAP_MAX_H = 6.0    # blood draw vs vascular acquisition, same day
GATE_H = 72.0      # prior-night gate
LYM = "bt__lymphocytes_abs_float_value"
RATE = ("C07", "C08D", "C01EB17")
MODULES = {
    "nk_cytotoxic": ["GNLY", "NKG7", "GZMB", "PRF1", "KLRD1", "GZMA"],
    "nk_cd16": ["KLRF1", "NCR1", "SH2D1B", "FCGR3A"],
    "mobilisable_effector": ["CX3CR1", "GZMH", "FGFBP2", "KLRG1", "ADRB2"],
    "cd8_lineage": ["CD8A", "CD8B"],
    "memory_imprint": ["KLRC2", "ZNF683", "B3GAT1"],
    "naive_t": ["CCR7", "SELL", "LEF1", "TCF7"],
    "gc_response": ["FKBP5", "TSC22D3", "DDIT4", "PER1", "ZBTB16", "KLF9"],
    "camp_ieg": ["CREM", "DUSP1", "FOS"],
}
SOURCES = {
    "nk_cytotoxic": "omics v1 module (unchanged; the C4 lead signal)",
    "nk_cd16": "NK-restricted CD56dim/CD16+ markers, novelty audit 4.2 C4 compositional split (FCGR3A also on CD16+ monocytes)",
    "mobilisable_effector": "CX3CR1+ cytotoxic effector (CD56dim NK + CD8/CD4 TEMRA), audit 4.2 C4 TEMRA list without CD8A, plus "
                            "ADRB2 (audit 5.1 #5/#6); beta2-adrenergic mobilisation: Dimitrov 2010 J Immunol (audit-verified; PMID in PROTOCOL.md)",
    "cd8_lineage": "CD8 lineage control (separates TEMRA from NK in bulk RNA)",
    "memory_imprint": "CMV/adaptive NK-T memory imprint, audit 4.2 C4 (KLRC2/B3GAT1 3'-coverage risk)",
    "naive_t": "naive/central-memory T markers (standard immunology; chosen here, not from the audit); trait, not mobilised",
    "gc_response": "glucocorticoid-response set, audit 4.2 C3 (FKBP5, TSC22D3, DDIT4, PER1, ZBTB16, KLF9)",
    "camp_ieg": "cAMP/immediate-early part of the audit 4.2 C4 stress sub-score (FOS, DUSP1 also ex vivo delay genes)",
}
MODS = list(MODULES)
KEYM = ["nk_cytotoxic", "mobilisable_effector", "memory_imprint", "naive_t"]  # modules shown in sensitivity lines
SHORT = {"nk_cytotoxic": "nk", "nk_cd16": "cd16", "mobilisable_effector": "mobil", "cd8_lineage": "cd8",
         "memory_imprint": "memory", "naive_t": "naive", "gc_response": "gc", "camp_ieg": "camp"}
ATLAS_API = ["build_frame", "rd", "rc", "cnt", "qtiles", "pct", "bh", "zs", "fnum", "log", "LOG", "v3", "icc1", "wide",
             "one_row_per_participant", "acute_deviation", "multi_fit", "contrasts", "to_local", "num", "rowmean",
             "MINCELL", "MAIN_STAGES", "CORE", "TZ", "VERSION"]
OMICS_API = ["load_rna", "match", "rows_for", "covmat", "mols", "MODULES", "DRAW_LAYERS", "privacy_check"]
LONG = re.compile(r"\d{7,}")


# ============================================================================ setup and small helpers

def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _default(names):
    for n in names:
        p = os.path.normpath(os.path.join(HERE, n))
        if os.path.exists(p):
            return p
    return os.path.join(HERE, names[0])


def setup(atlas_path, omics_path):
    """Import the atlas and omics modules by path; extend the omics module dictionary with the new modules."""
    global A, O
    A = _load(atlas_path, "f1_atlas")
    miss = [x for x in ATLAS_API if not hasattr(A, x)]
    if miss:
        raise SystemExit(f"atlas module lacks {miss}; use atlas v1.5")
    O = _load(omics_path, "f1_omics_v1_2")
    O.A = A
    miss = [x for x in OMICS_API if not hasattr(O, x)]
    if miss:
        raise SystemExit(f"omics module lacks {miss}; use omics v1.2")
    if list(O.MODULES.get("nk_cytotoxic", [])) != MODULES["nk_cytotoxic"]:
        raise SystemExit("omics nk_cytotoxic module differs from the one this module expects")
    O.DRAW_LAYERS = {"rna"}  # batch + draw hour + hour^2 in every RNA model (omics v1.1/v1.2 primary)
    O.MODULES = {**O.MODULES, **{k: v for k, v in MODULES.items() if k not in O.MODULES}}
    return A, O


def log(m):
    A.log(m)


def rc(n):
    return A.rc(n)


def err(ex):
    """Error note for results/digest.  Digit runs of 5+ are masked so that an exception text quoting a value or an id
    cannot trip the final privacy scan (which would abort the whole run without writing any output)."""
    tbl = traceback.extract_tb(ex.__traceback__)
    tb = ([f for f in tbl if f.filename == __file__] or tbl or [None])[-1]
    msg = re.sub(r"\d{5,}", "#", str(ex))[:160]
    return f"{type(ex).__name__}: {msg} (line {tb.lineno if tb is not None else 'NA'})"


def guard(out, key, fn):
    t = time.time()
    try:
        out[key] = fn()
        log(f"{key} done in {time.time() - t:.1f}s")
    except Exception as ex:
        out[key] = {"error": err(ex)}
        log(f"{key} FAILED: {out[key]['error']}")
    out.setdefault("timings", {})[key] = time.time() - t


def utc_ns(s):
    """Naive UTC datetime64[ns] (tz-aware input converted; naive input taken as UTC)."""
    t = pd.to_datetime(pd.Series(s), errors="coerce", utc=True)
    return t.dt.tz_convert(None).dt.as_unit("ns")


def local_day(s):
    """Naive local (Asia/Jerusalem) calendar day, datetime64[ns]."""
    s = pd.Series(s)
    if isinstance(s.dtype, pd.DatetimeTZDtype) or s.dtype == object:
        t = A.to_local(s)
        if isinstance(t.dtype, pd.DatetimeTZDtype):
            t = t.dt.tz_localize(None)
    else:
        t = pd.to_datetime(s, errors="coerce")
    return t.dt.as_unit("ns").dt.normalize()


def naive_local(s):
    """Naive local (Asia/Jerusalem) datetime64[ns]: tz-aware input converted first, naive input kept as is."""
    s = pd.Series(s)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert(A.TZ).dt.tz_localize(None)
    t = pd.to_datetime(s, errors="coerce")
    if isinstance(t.dtype, pd.DatetimeTZDtype):  # object column holding tz-aware values
        t = t.dt.tz_convert(A.TZ).dt.tz_localize(None)
    return t.dt.as_unit("ns")


def hrclip(x):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return x.where((x >= HR_LIM[0]) & (x <= HR_LIM[1]))


def zs(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else x * np.nan


def ols_v(y, X):
    """OLS with HC1 covariance."""
    n, k = X.shape
    G = np.linalg.pinv(X.T @ X)
    b = G @ (X.T @ y)
    e = y - X @ b
    V = G @ ((X * (e ** 2)[:, None]).T @ X) @ G * (n / max(n - k, 1))
    return b, V


def cd(b, se, extra=None):
    b, se = float(b), float(se)
    p = float(2 * stats.norm.sf(abs(b / se))) if se > 0 and np.isfinite(se) else None
    out = {"b": b, "se": se, "lo": b - 1.96 * se, "hi": b + 1.96 * se, "p": p}
    if extra:
        out.update(extra)
    return out


def tost(b, se, delta=TOST):
    """Two one-sided tests of |beta| < delta: p = max of the two one-sided p values; 90 % CI."""
    if not (np.isfinite(b) and np.isfinite(se)) or se <= 0:
        return {}
    p = max(stats.norm.sf((b + delta) / se), stats.norm.cdf((b - delta) / se))
    return {"p_tost": float(p), "equivalent": bool(p < 0.05), "ci90": [b - 1.645 * se, b + 1.645 * se]}


def design(dd, xs, covs, scale=None):
    """[1, xs (divided by scale), covs standardized (constant columns dropped)]."""
    cols = [np.ones(len(dd))]
    for x in xs:
        cols.append(dd[x].astype(float).to_numpy() / (scale or {}).get(x, 1.0))
    for c in covs:
        v = dd[c].astype(float).to_numpy()
        s = v.std()
        if s > 0:
            cols.append((v - v.mean()) / s)
    return np.column_stack(cols)


def fitx(d, y, xs, covs, scale=None, ystd=True, min_n=MIN_N):
    """Outcome y (z-scored in the sample when ystd) on xs + standardized covs; returns b, V of the xs and n."""
    cols = list(dict.fromkeys([y] + list(xs) + list(covs)))
    dd = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(dd) < min_n:
        return None
    yy = zs(dd[y]) if ystd else dd[y].astype(float).to_numpy()
    X = design(dd, xs, covs, scale)
    b, V = ols_v(yy, X)
    k = len(xs)
    return {"n": len(dd), "b": b[1:1 + k], "V": V[1:1 + k, 1:1 + k]}


def coefs(f, names):
    return {nm: cd(f["b"][i], np.sqrt(max(f["V"][i, i], 0))) for i, nm in enumerate(names)}


def decomp(f, with_tost=False, gamma=None):
    """f from fitx(..., ['hr_visit', 'hr_noct'], ...): acute, usual_total and beyond.  With gamma (slope of visit HR on
    nocturnal HR in the excess model) also usual_given_excess = coefficient of nocturnal HR in the equivalent
    parametrisation bed ~ hr_noct + hr_excess (+ covariates incl. age and sex) = beyond + gamma * acute."""
    b, V = f["b"], f["V"]
    out = {"n": rc(f["n"]), "acute": cd(b[0], np.sqrt(V[0, 0])),
           "usual_total": cd(b[0] + b[1], np.sqrt(max(V[0, 0] + V[1, 1] + 2 * V[0, 1], 0))),
           "beyond": cd(b[1], np.sqrt(V[1, 1]))}
    if gamma is not None and np.isfinite(gamma):
        out["usual_given_excess"] = cd(b[1] + gamma * b[0],
                                       np.sqrt(max(V[1, 1] + gamma ** 2 * V[0, 0] + 2 * gamma * V[0, 1], 0)))
    if with_tost:
        out["acute"].update(tost(b[0], np.sqrt(V[0, 0])))
    return out


def multi_lin(Y, X, w):
    """Several outcomes on one design; returns w'B (one value per outcome) and its robust joint covariance."""
    n, k = X.shape
    G = np.linalg.pinv(X.T @ X)
    B = G @ (X.T @ Y)
    E = Y - X @ B
    a = np.asarray(w, float) @ (G @ X.T)
    AE = a[:, None] * E
    return np.asarray(w, float) @ B, AE.T @ AE * (n / max(n - k, 1))


def pctl(x):
    return A.pct(np.asarray(x, float))


def clean(o):
    if isinstance(o, np.ndarray):
        o = o.tolist()
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating, int, np.integer, bool, np.bool_)):
        return A.fnum(o)
    return o


# ============================================================================ shared HR set and extra inputs

def med_flags(root):
    med = A.rd(root, "medications/medications.parquet", columns=["atc3", "atc4", "atc5"])
    if med is None:
        return None
    keys = med[["participant_id", "research_stage"]].drop_duplicates()
    parts = [med[["participant_id", "research_stage", c]].explode(c).rename(columns={c: "code"}).dropna(subset=["code"])
             for c in ("atc3", "atc4", "atc5") if c in med.columns]
    if not parts:
        return None
    e = pd.concat(parts, ignore_index=True)
    c = e["code"].astype(str).str.upper().str.strip()
    e["bb"] = c.str.startswith("C07")
    e["bb_nonsel"] = c.str.startswith(("C07AA", "C07AG"))
    e["bb_sel"] = c.str.startswith("C07AB")
    e["ccb_rate"] = c.str.startswith("C08D")
    e["ivab"] = c.str.startswith("C01EB17")
    e["aht_other"] = c.str.startswith(("C02", "C03", "C09")) | (c.str.startswith("C08") & ~c.str.startswith("C08D"))
    e["steroid_sys"] = c.str.startswith("H02")
    e["immsup"] = c.str.startswith("L04")  # immunosuppressants (C4 extra confounders)
    fl = ["bb", "bb_nonsel", "bb_sel", "ccb_rate", "ivab", "aht_other", "steroid_sys", "immsup"]
    f = e.groupby(["participant_id", "research_stage"])[fl].max().astype(float).reset_index()
    out = keys.merge(f, how="left")
    out[fl] = out[fl].fillna(0.0)
    out["rate_drug"] = out[["bb", "ccb_rate", "ivab"]].max(axis=1)
    return out


def night_icc(s):
    """One-way ICC of single-night HR within a visit and the pooled within-visit variance (visits with >= 2 nights)."""
    g = s.dropna(subset=["hm"]).groupby(["participant_id", "research_stage"])["hm"]
    st = g.agg(["mean", "var", "count"])
    st = st[st["count"] >= 2]
    if len(st) < 20:
        return None, None
    k = st["count"].to_numpy(float)
    N, D = k.sum(), len(k)
    msw = float(((k - 1) * st["var"].to_numpy()).sum() / (N - D))
    gm = (st["mean"] * k).sum() / N
    msb = float((k * (st["mean"].to_numpy() - gm) ** 2).sum() / (D - 1))
    n0 = (N - (k ** 2).sum() / N) / (D - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return {"icc_single_night": icc, "within_sd": np.sqrt(msw), "visits": rc(D)}, msw


def add_hr(root, fr):
    """Attach the shared HR set, medication flags, lymphocytes, acute SBP deviation and timing to the atlas frame.
    Each block is independent; a failing block is recorded in info['errors'] and leaves its columns missing."""
    info, errs = {}, {}
    fr = fr.copy()
    stg = set(A.MAIN_STAGES)
    for c in ("hr_sit", "ecg_hr"):
        fr[c] = hrclip(fr[c]) if c in fr.columns else np.nan
    fr["hr_visit"] = A.rowmean(fr, ["hr_sit", "ecg_hr"])
    # --- sleep: nocturnal HR per visit (nights of the same research_stage)
    nights = None
    try:
        s = A.rd(root, "sleep/sleep.parquet", columns=["heart_rate_mean_during_sleep", "heart_rate_min_during_sleep",
                                                       "study_start_time", "collection_timestamp"])
        info["nights_stage_undefined"] = rc((s["research_stage"] == "undefined").sum())
        s = s[s["research_stage"].isin(stg)].copy()
        if "heart_rate_mean_during_sleep" not in s.columns:
            raise KeyError("heart_rate_mean_during_sleep missing")
        s["hm"] = hrclip(s["heart_rate_mean_during_sleep"])
        s["hn"] = hrclip(s["heart_rate_min_during_sleep"]) if "heart_rate_min_during_sleep" in s.columns else np.nan
        g = s.groupby(["participant_id", "research_stage"])
        n = g.agg(hr_noct=("hm", "median"), hr_noct_min=("hn", "median"), hr_noct_mean=("hm", "mean"),
                  n_nights_hr=("hm", "count")).reset_index()
        n.loc[n["n_nights_hr"] == 0, ["hr_noct", "hr_noct_min", "hr_noct_mean"]] = np.nan
        fr = fr.merge(n, on=["participant_id", "research_stage"], how="left")
        ni, msw = night_icc(s.merge(fr[["participant_id", "research_stage"]].drop_duplicates(), how="inner"))
        info["nights"] = ni
        info["_s2w"] = msw
        try:  # night timestamps are needed for the timing block only; their failure must not drop nocturnal HR
            tcol = next((c for c in ("study_start_time", "collection_timestamp") if c in s.columns), None)
            if tcol is None:
                raise KeyError("no night timestamp column")
            s["start"] = utc_ns(s[tcol]).to_numpy()
            s["day"] = local_day(s[tcol]).to_numpy()
            nights = s[["participant_id", "research_stage", "start", "day"]]
        except Exception as ex:
            errs["sleep_time"] = err(ex)
    except Exception as ex:
        errs["sleep"] = err(ex)
        for c in ("hr_noct", "hr_noct_min", "hr_noct_mean", "n_nights_hr"):
            fr[c] = np.nan
    # --- medications
    try:
        mf = med_flags(root)
        if mf is None:
            raise ValueError("medications table missing")
        fr = fr.merge(mf[mf["research_stage"].isin(stg)], on=["participant_id", "research_stage"], how="left")
    except Exception as ex:
        errs["medications"] = err(ex)
    for c in ("bb", "bb_nonsel", "bb_sel", "ccb_rate", "ivab", "aht_other", "rate_drug", "steroid_sys", "immsup"):
        fr[c] = fr[c].fillna(0.0) if c in fr.columns else 0.0
    # --- acute SBP deviation (atlas S2 definition)
    try:
        fr, adi = A.acute_deviation(fr)
        info["acute_dev_model"] = adi
    except Exception as ex:
        errs["acute_dev"] = err(ex)
        fr["acute_dev"] = np.nan
    # --- acute HR excess
    try:
        ok = fr[["hr_visit", "hr_noct", "age", "sex_male"]].notna().all(axis=1)
        fit_ok = ok & (fr["rate_drug"] == 0)
        if fit_ok.sum() < MIN_N:
            raise ValueError("fewer than 100 visits with visit and nocturnal HR")
        X = np.column_stack([np.ones(int(fit_ok.sum()))] + [fr.loc[fit_ok, c].astype(float).to_numpy()
                                                            for c in ("hr_noct", "age", "sex_male")])
        y = fr.loc[fit_ok, "hr_visit"].astype(float).to_numpy()
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        Xa = np.column_stack([np.ones(int(ok.sum()))] + [fr.loc[ok, c].astype(float).to_numpy()
                                                         for c in ("hr_noct", "age", "sex_male")])
        fr["hr_excess"] = np.nan
        fr.loc[ok, "hr_excess"] = fr.loc[ok, "hr_visit"].astype(float).to_numpy() - Xa @ b
        info["excess_model"] = {"n": rc(fit_ok.sum()), "slope_noct": b[1], "slope_age": b[2], "slope_male": b[3],
                                "r2": 1 - np.var(y - X @ b) / np.var(y)}
        info["_gamma"] = float(b[1])
    except Exception as ex:
        errs["hr_excess"] = err(ex)
        fr["hr_excess"] = np.nan
    fr["hr_diff"] = fr["hr_visit"] - fr["hr_noct"]
    fr["alert"] = fr["sbp_sit"] - fr["sbp_sit2"] if "sbp_sit2" in fr.columns else np.nan
    # --- CBC lymphocytes: nearest draw within 180 days of the visit date
    try:
        bt = A.rd(root, "blood_tests/blood_tests.parquet", columns=["collection_date", LYM])
        bt = A.num(bt, [LYM])
        bt["lab_date"] = naive_local(bt["collection_date"]).to_numpy()
        bt = bt.dropna(subset=["lab_date", LYM])[["participant_id", "lab_date", LYM]].sort_values("lab_date")
        left = fr[["participant_id", "research_stage", "visit_date"]].copy()
        left["visit_date"] = naive_local(left["visit_date"]).to_numpy()
        left = left.dropna(subset=["visit_date"])
        m = pd.merge_asof(left.sort_values("visit_date"), bt, left_on="visit_date", right_on="lab_date",
                          by="participant_id", direction="nearest", tolerance=pd.Timedelta(days=180))
        m["lymph_log"] = np.log(m[LYM].where(m[LYM] > 0))
        m["lymph_gap_d"] = (m["lab_date"] - m["visit_date"]).dt.days.abs()
        fr = fr.merge(m[["participant_id", "research_stage", "lymph_log", "lymph_gap_d"]],
                      on=["participant_id", "research_stage"], how="left")
    except Exception as ex:
        errs["lymphocytes"] = err(ex)
        fr["lymph_log"], fr["lymph_gap_d"] = np.nan, np.nan
    # --- timing: acquisition vs BP / ECG; nights vs visit; prior-night gate
    try:
        tm = {}
        vas = utc_ns(fr["vas_ts"]) if "vas_ts" in fr.columns else pd.Series(pd.NaT, index=fr.index)
        if "bp_ts" in fr.columns:
            tm["acq_minus_bp_min"] = A.qtiles((vas - utc_ns(fr["bp_ts"])).dt.total_seconds() / 60.0)
        e = A.rd(root, "ecg/ecg.parquet", columns=["collection_timestamp"])
        if e is not None and "collection_timestamp" in e.columns:
            e["_t"] = utc_ns(e["collection_timestamp"]).to_numpy()
            et = e.groupby(["participant_id", "research_stage"])["_t"].min().rename("ecg_t").reset_index()
            x = fr[["participant_id", "research_stage"]].merge(et, how="left")
            tm["acq_minus_ecg_min"] = A.qtiles(pd.Series((vas.to_numpy() - x["ecg_t"].to_numpy()) / np.timedelta64(1, "m")))
        if nights is not None:
            v = fr[["participant_id", "research_stage", "visit_date"]].copy()
            v["_vday"] = naive_local(v["visit_date"]).dt.normalize().to_numpy()
            v["_vas"] = vas.to_numpy()
            x = nights.merge(v, on=["participant_id", "research_stage"], how="inner")
            dd = (x["day"] - x["_vday"]).dt.days.dropna()
            tm["night_minus_visit_days"] = A.qtiles(dd)
            nb = int((dd < 0).sum())  # a share is reported only when both of its cells are >= 20 (privacy)
            tm["share_nights_before_visit"] = (float(nb / len(dd)) if min(nb, len(dd) - nb) >= A.MINCELL else None)
            hb = (x["_vas"] - x["start"]).dt.total_seconds() / 3600.0
            x["_gate"] = (hb > 0) & (hb <= GATE_H)
            gate = x.groupby(["participant_id", "research_stage"])["_gate"].any()
            tm["prior_night_gate_visits"] = rc(int(gate.sum()))
        info["timing"] = tm
    except Exception as ex:
        errs["timing"] = err(ex)
    info["errors"] = errs
    return fr.reset_index(drop=True), info


def reliability(fr, boot, rng):
    """Between-visit ICC(1) of each core bed with bootstrap replicates (atlas S1 definition)."""
    w = A.wide(fr, A.CORE)
    rel, rb = {}, {}
    s0, s1 = A.MAIN_STAGES
    for c in A.CORE:
        if (c, s0) in w.columns and (c, s1) in w.columns:
            p = w[[(c, s0), (c, s1)]].dropna().to_numpy()
            if len(p) >= 50:
                rel[c] = A.icc1(p[:, 0], p[:, 1])
                rb[c] = np.array([A.icc1(*p[rng.integers(0, len(p), len(p))].T) for _ in range(boot)])
    return rel, rb


def disatt(bfull, bboot, rel, rb, beds, pairs):
    """Disattenuated pairwise contrasts b_i / sqrt(rel_i) - b_j / sqrt(rel_j), bootstrap percentile CI."""
    out = {}
    for a_, c_ in pairs:
        if a_ not in rel or c_ not in rel or rel[a_] <= 0.05 or rel[c_] <= 0.05:
            continue
        i, j = beds.index(a_), beds.index(c_)
        ra = np.where(rb[a_] > 0.05, rb[a_], np.nan)[:len(bboot)]
        rcc = np.where(rb[c_] > 0.05, rb[c_], np.nan)[:len(bboot)]
        out[f"{a_}-{c_}"] = {"d": float(bfull[i] / np.sqrt(rel[a_]) - bfull[j] / np.sqrt(rel[c_])),
                             "ci": pctl(bboot[:, i] / np.sqrt(ra) - bboot[:, j] / np.sqrt(rcc))}
    return out


# ============================================================================ H0 flow

def h0(fr, info):
    out = {k: v for k, v in info.items() if not k.startswith("_")}
    cnts = {}
    for st in A.MAIN_STAGES:
        f = fr[fr["research_stage"] == st]
        cnts[st] = {k: rc(f[c].notna().sum()) for k, c in (("sit", "hr_sit"), ("ecg", "ecg_hr"), ("visit", "hr_visit"),
                                                            ("noct", "hr_noct"), ("excess", "hr_excess"),
                                                            ("adev", "acute_dev"), ("pwv", "pwv"), ("lymph", "lymph_log"),
                                                            ("alert", "alert"))}
        cnts[st]["visit+noct+pwv"] = rc(f[["hr_visit", "hr_noct", "pwv"]].notna().all(axis=1).sum())
        for k, c in (("rate", "rate_drug"), ("bb", "bb"), ("bb_ns", "bb_nonsel"), ("bb_s", "bb_sel"),
                     ("ccb_rate", "ccb_rate"), ("ivab", "ivab"), ("ster", "steroid_sys"), ("immsup", "immsup")):
            cnts[st][k] = rc(f[c].sum())
    out["counts"] = cnts
    out["quantiles"] = {c: A.qtiles(fr[c]) for c in ("hr_sit", "ecg_hr", "hr_visit", "hr_noct", "hr_noct_min", "hr_diff",
                                                      "hr_excess", "acute_dev", "n_nights_hr", "alert", "lymph_gap_d")
                        if c in fr.columns}

    def r(a, b, f=fr):
        x = f[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
        return {"r": float(np.corrcoef(x[a], x[b])[0, 1]), "n": rc(len(x))} if len(x) >= A.MINCELL else None
    out["corr"] = {f"{a}~{b}": r(a, b) for a, b in (("hr_sit", "ecg_hr"), ("hr_visit", "hr_noct"),
                                                   ("hr_excess", "acute_dev"), ("hr_excess", "acq_hour"),
                                                   ("hr_noct", "hr_noct_min"), ("hr_excess", "hr_diff"))
                   if a in fr.columns and b in fr.columns}
    w = A.wide(fr, ["hr_visit", "hr_noct", "hr_excess", "pwv"])
    s0, s1 = A.MAIN_STAGES
    out["icc_between_visits"] = {}
    for c in ("hr_visit", "hr_noct", "hr_excess", "pwv"):
        if (c, s0) in w.columns and (c, s1) in w.columns:
            p = w[[(c, s0), (c, s1)]].dropna().to_numpy()
            out["icc_between_visits"][c] = {"icc": A.icc1(p[:, 0], p[:, 1]), "n": rc(len(p))} if len(p) >= 50 else None
    return out


# ============================================================================ H1 HR decomposition by bed

def ycol(fr, bed):
    return "o_" + bed if "o_" + bed in fr.columns else bed


def rc_usual(d, covs, s2w, ystd):
    """Regression-calibrated usual HR: U_cal = Zb + lambda_i (m_i - Zb), Z = [1, hr_visit, covs], m = mean night HR,
    lambda_i = s2u / (s2u + s2w / k_i).  Then y ~ hr_visit + U_cal + covs (decomposition as in H1)."""
    Z = design(d, ["hr_visit"], covs)
    m = d["hr_noct_mean"].astype(float).to_numpy()
    k = d["n_nights_hr"].astype(float).to_numpy()
    b = np.linalg.lstsq(Z, m, rcond=None)[0]
    r = m - Z @ b
    s2u = np.var(r) - np.mean(s2w / k)
    if s2u <= 0:
        return None, None
    lam = s2u / (s2u + s2w / k)
    dd = d.copy()
    dd["_ucal"] = Z @ b + lam * r
    f = fitx(dd, "_y", ["hr_visit", "_ucal"], covs, {"hr_visit": 10.0, "_ucal": 10.0}, ystd=ystd)
    return f, float(np.mean(lam))


def h1(fr, rel, rb, s2w, boot, rng, gamma=None):
    out = {"per_bed": {}, "raw_pwv_m_s": {}, "contrasts": {}, "sens_pwv": {}}
    prim = fr[fr["rate_drug"] == 0]
    sc = {"hr_visit": 10.0, "hr_noct": 10.0}
    covsets = {"Mh1": M3, "Mh2": M3 + ["acute_dev"]}
    for mn, covs in covsets.items():
        r = {}
        for bed in BEDS_H + ["acute_dev"]:
            y = ycol(fr, bed)
            if y not in fr.columns or (bed == "acute_dev" and "acute_dev" in covs):
                continue
            d = A.one_row_per_participant(prim, [y, "hr_visit", "hr_noct"] + covs)
            f = fitx(d, y, ["hr_visit", "hr_noct"], covs, sc)
            if f:
                r[bed] = decomp(f, with_tost=bed in NONPWV, gamma=gamma)
                if s2w and mn == "Mh1":  # regression-calibrated usual HR (analytic SE given the calibration)
                    try:
                        dr = d[d["n_nights_hr"] >= 1].reset_index(drop=True).assign(_y=lambda x: x[y].astype(float))
                        fr_, lam = rc_usual(dr, covs, s2w, True)
                        if fr_:
                            r[bed]["rc"] = {**decomp(fr_), "mean_lambda": lam}
                    except Exception as ex:
                        r[bed]["rc"] = {"error": err(ex)}
        out["per_bed"][mn] = r
        d = A.one_row_per_participant(prim, ["pwv", "hr_visit", "hr_noct"] + covs)
        f = fitx(d, "pwv", ["hr_visit", "hr_noct"], covs, sc, ystd=False)
        if f:
            out["raw_pwv_m_s"][mn] = decomp(f, gamma=gamma)
    # four-bed joint fit (Mh1): acute, usual_total, beyond with contrasts and disattenuation
    oc = ["o_" + b for b in A.CORE]
    d = A.one_row_per_participant(prim, oc + ["hr_visit", "hr_noct"] + M3)
    if len(d) >= 200:
        X = design(d, ["hr_visit", "hr_noct"], M3, sc)
        Y = np.column_stack([zs(d[c]) for c in oc])
        n = len(d)
        Bb = np.empty((boot, X.shape[1], len(oc)))
        for k in range(boot):
            ix = rng.integers(0, n, n)
            Bb[k] = np.linalg.lstsq(X[ix], Y[ix], rcond=None)[0]
        for term, w in (("acute", [0, 1, 0]), ("usual_total", [0, 1, 1]), ("beyond", [0, 0, 1])):
            ww = np.zeros(X.shape[1])
            ww[:3] = w
            bvec, V = multi_lin(Y, X, ww)
            con, het = A.contrasts(bvec, V, A.CORE, PAIRS4)
            bb = np.einsum("k,bkq->bq", ww, Bb)
            out["contrasts"][term] = {"n": rc(n), "b": {c: cd(bvec[i], np.sqrt(V[i, i])) for i, c in enumerate(A.CORE)},
                                      "pairs": con, "wald": het, "disattenuated": disatt(bvec, bb, rel, rb, A.CORE, PAIRS4)}
    # within-person change (both visits without rate-lowering drugs)
    try:
        w = A.wide(fr, ["pwv", "hr_visit", "hr_noct", "sbp_sit", "rate_drug", "age", "sex_male"])
        s0, s1 = A.MAIN_STAGES
        vd = A.wide(fr.assign(_vd=pd.to_datetime(fr["visit_date"], errors="coerce").dt.as_unit("ns")), ["_vd"])
        dd = pd.DataFrame({"dpwv": w[("pwv", s1)] - w[("pwv", s0)], "dhrv": w[("hr_visit", s1)] - w[("hr_visit", s0)],
                           "dhrn": w[("hr_noct", s1)] - w[("hr_noct", s0)],
                           "dsbp": w[("sbp_sit", s1)] - w[("sbp_sit", s0)], "age": w[("age", s0)],
                           "sex_male": w[("sex_male", s0)],
                           "fu": (vd[("_vd", s1)] - vd[("_vd", s0)]).dt.days / 365.25,
                           "norate": ((w[("rate_drug", s0)] == 0) & (w[("rate_drug", s1)] == 0)).astype(float)})
        dd = dd[dd["norate"] == 1]
        f = fitx(dd, "dpwv", ["dhrv", "dhrn", "dsbp"], ["age", "sex_male", "fu"],
                 {"dhrv": 10.0, "dhrn": 10.0, "dsbp": 10.0}, ystd=False, min_n=50)
        out["within_person_m_s"] = ({"n": rc(f["n"]), **coefs(f, ["d_hr_visit_per10", "d_hr_noct_per10", "d_sbp_per10"])}
                                    if f else {"status": "n<50"})
    except Exception as ex:
        out["within_person_m_s"] = {"error": err(ex)}
    # sensitivities (raw PWV m/s, Mh1)
    sens = {}
    d = A.one_row_per_participant(fr, ["pwv", "hr_visit", "hr_noct"] + M3)
    f = fitx(d, "pwv", ["hr_visit", "hr_noct"], M3 + ["rate_drug"], sc, ystd=False)
    sens["all_plus_rate_cov"] = decomp(f) if f else None
    for nm, vcol, ncol in (("hr_sit_only", "hr_sit", "hr_noct"), ("ecg_only", "ecg_hr", "hr_noct"),
                           ("noct_min", "hr_visit", "hr_noct_min")):
        d = A.one_row_per_participant(prim, ["pwv", vcol, ncol] + M3).rename(columns={vcol: "_v", ncol: "_n"})
        f = fitx(d, "pwv", ["_v", "_n"], M3, {"_v": 10.0, "_n": 10.0}, ystd=False)
        sens[nm] = decomp(f) if f else None
    if s2w:
        d = A.one_row_per_participant(prim, ["pwv", "hr_visit", "hr_noct_mean", "n_nights_hr"] + M3).copy()
        d = d[d["n_nights_hr"] >= 1].reset_index(drop=True)
        d["_y"] = d["pwv"].astype(float)
        f, lam = rc_usual(d, M3, s2w, False)
        if f:
            bs = []
            for _ in range(min(boot, 100)):
                fb, _ = rc_usual(d.iloc[rng.integers(0, len(d), len(d))].reset_index(drop=True), M3, s2w, False)
                if fb:
                    bs.append([fb["b"][0], fb["b"][0] + fb["b"][1], fb["b"][1]])
            bs = np.array(bs)
            sens["rc_usual"] = {"n": rc(f["n"]), "mean_lambda": lam,
                                **{k: {"b": float(v), "ci": pctl(bs[:, i]) if len(bs) else None}
                                   for i, (k, v) in enumerate((("acute", f["b"][0]), ("usual_total", f["b"][0] + f["b"][1]),
                                                               ("beyond", f["b"][1])))}}
    out["sens_pwv"] = sens
    return out


# ============================================================================ H2 beta-blocker positive control

def h2(fr, boot, rng):
    d = A.one_row_per_participant(fr[(fr["bb"] == 1) | (fr["rate_drug"] == 0)],
                                  ["pwv", "hr_visit", "hr_noct", "hr_excess", "hr_diff"] + M3_NOAHT)
    nu = int(d["bb"].sum())
    out = {"n_users": rc(nu), "n_nonusers": rc(len(d) - nu)}
    if nu < A.MINCELL or len(d) - nu < A.MINCELL:
        out["status"] = "suppressed (users or non-users < 20)"
        return out
    if nu < MIN_N:
        out["note"] = "fewer than 100 users"
    covs = M3_NOAHT

    def est(dd):
        a = {y: fitx(dd, y, ["bb"], covs, ystd=False, min_n=50) for y in ("hr_visit", "hr_noct", "hr_excess", "hr_diff")}
        tot = fitx(dd, "pwv", ["bb"], covs, ystd=False, min_n=50)
        med = fitx(dd, "pwv", ["bb", "hr_visit", "hr_noct"], covs, ystd=False, min_n=50)
        if any(v is None for v in a.values()) or tot is None or med is None:
            return None
        ind = a["hr_visit"]["b"][0] * med["b"][1] + a["hr_noct"]["b"][0] * med["b"][2]
        return a, tot, med, ind
    r = est(d)
    if r is None:
        out["status"] = "fit failed"
        return out
    a, tot, med, ind = r
    bs = []
    for _ in range(boot):
        x = est(d.iloc[rng.integers(0, len(d), len(d))])
        if x is not None and x[1]["b"][0] != 0:
            bs.append([x[3], x[1]["b"][0], x[2]["b"][0]])
    bs = np.array(bs)
    out.update({k + "_bpm": cd(v["b"][0], np.sqrt(v["V"][0, 0])) for k, v in a.items()})
    out["pwv_total_m_s"] = cd(tot["b"][0], np.sqrt(tot["V"][0, 0]))
    out["pwv_direct_m_s"] = cd(med["b"][0], np.sqrt(med["V"][0, 0]))
    out["pwv_per_bpm"] = {"visit": cd(med["b"][1], np.sqrt(med["V"][1, 1])), "noct": cd(med["b"][2], np.sqrt(med["V"][2, 2]))}
    out["indirect_via_hr_m_s"] = {"b": ind, "ci": pctl(bs[:, 0]) if len(bs) else None}
    out["proportion_mediated"] = {"b": ind / tot["b"][0] if tot["b"][0] != 0 else None,
                                  "ci": pctl(bs[:, 0] / bs[:, 1]) if len(bs) else None}
    out["n"] = rc(len(d))
    return out


# ============================================================================ H3 PBMC modules

def h3(fr, root, rel, rb, max_genes, boot, rng, pids):
    out = {"modules": {m: {"genes": MODULES[m], "source": SOURCES[m]} for m in MODS}}
    t0 = time.time()
    rna, mod = O.load_rna(root, max_genes)
    mod["info"]["draw_hour"] = rna["info"].get("draw_hour")
    mod["info"]["batches"] = rna["info"].get("batches")
    del rna
    pids |= set(mod["pid"])
    M, mi = O.match(mod, fr)
    out["flow"] = {**{k: v for k, v in mod["info"].items()}, **mi, "load_s": time.time() - t0}
    names = list(mod["names"])
    mods = [m for m in MODS if m in names]
    out["modules_missing"] = [m for m in MODS if m not in names]
    if "nk_cytotoxic" not in mods:
        raise ValueError("nk_cytotoxic module not available")
    Xall = mod["X"][:, [names.index(m) for m in mods]]
    dh_all = np.asarray(mod.get("draw_hour"), float) if mod.get("draw_hour") is not None else None
    out["modules_used"] = mods
    nm = len(mods)

    def need(*cols):
        return [c for c in dict.fromkeys(cols)]

    def mfit(d, covs, beds=("pwv",)):
        Y = np.column_stack([zs(d[ycol(fr, b)]) for b in beds])
        b, se, nj, cov = O.mols(Y, Xall[d["_si"].to_numpy()], O.covmat(d, covs, mod))
        return b, se, cov

    def pack(b, se, n):
        return {"n": rc(n), **{m: cd(b[i], se[i]) for i, m in enumerate(mods)}}

    # replication of the omics M3 result (module -> PWV, full sample)
    d = O.rows_for(fr, M, need("o_pwv", *M3))
    b, se, _ = mfit(d, M3)
    out["replication_M3"] = pack(b[:, 0], se[:, 0], len(d))
    # chain A: visit-HR sample
    chainA = {"base": M3, "visit_hr": M3 + ["hr_visit"], "visit_hr_acute_dev": M3 + ["hr_visit", "acute_dev"]}
    d = O.rows_for(fr, M, need("o_pwv", "hr_visit", "acute_dev", *M3))
    out["chainA"] = {k: pack(*[x[:, 0] for x in mfit(d, c)[:2]], len(d)) for k, c in chainA.items()} if len(d) >= MIN_N else {
        "status": "n<100"}
    # chain B (PRIMARY): nocturnal-HR sample
    chainB = {"base": M3, "excess": M3 + ["hr_excess", "acute_dev"], "visit_hr": M3 + ["hr_visit"],
              "decomposed": M3 + ["hr_noct", "hr_visit"], "all_state": M3 + ["hr_noct", "hr_visit", "acute_dev"],
              "excess_only": M3 + ["hr_excess"], "acute_dev_only": M3 + ["acute_dev"]}
    cb_need = need("o_pwv", "hr_visit", "hr_noct", "hr_excess", "acute_dev", *M3)
    d = O.rows_for(fr, M, cb_need)
    out["chainB_n"] = rc(len(d))
    if len(d) >= MIN_N:
        res = {k: mfit(d, c) for k, c in chainB.items()}
        out["chainB"] = {k: pack(v[0][:, 0], v[1][:, 0], len(d)) for k, v in res.items()}
        n = len(d)
        bs = {k: np.full((boot, nm), np.nan) for k in ("base", "excess", "all_state")}
        for r_ in range(boot):
            db = d.iloc[rng.integers(0, n, n)]
            for k in bs:
                bs[k][r_] = mfit(db, chainB[k])[0][:, 0]
        ret = {}
        for k in ("excess", "all_state"):
            ret[k] = {m: {"delta": float(res[k][0][i, 0] - res["base"][0][i, 0]),
                          "delta_ci": pctl(bs[k][:, i] - bs["base"][:, i]),
                          "retention": float(res[k][0][i, 0] / res["base"][0][i, 0]) if res["base"][0][i, 0] != 0 else None,
                          "retention_ci": pctl(bs[k][:, i] / bs["base"][:, i])} for i, m in enumerate(mods)}
        out["chainB_retention"] = ret
    else:
        out["chainB"] = {"status": "n<100"}
    # module -> each bed (base and primary), BH over modules x beds; TOST for non-PWV beds
    beds = [b for b in BEDS_M if ycol(fr, b) in fr.columns]
    per = {}
    for mn in ("base", "excess"):
        rows, flat = {}, []
        for bed in beds:
            d = O.rows_for(fr, M, need(ycol(fr, bed), *cb_need[1:]))
            if len(d) < MIN_N:
                continue
            b, se, _ = mfit(d, chainB[mn], (bed,))
            rows[bed] = {"n": rc(len(d))}
            for i, m in enumerate(mods):
                e = cd(b[i, 0], se[i, 0])
                if bed in NONPWV:
                    e.update(tost(b[i, 0], se[i, 0]))
                rows[bed][m] = e
                flat.append((bed, m, e["p"]))
        q = A.bh([x[2] for x in flat])
        for (bed, m, _), qq in zip(flat, q):
            rows[bed][m]["q"] = qq
        per[mn] = rows
    out["beds"] = per
    # common four-bed sample: contrasts, Wald, disattenuated (primary model)
    oc = ["o_" + b for b in A.CORE]
    d = O.rows_for(fr, M, need(*oc, *cb_need[1:]))
    if len(d) >= 200:
        Cm = np.array([[1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1]], float)
        b, se, cov = mfit(d, chainB["excess"], tuple(A.CORE))
        n = len(d)
        bb = np.full((boot, nm, 4), np.nan)
        for r_ in range(boot):
            bb[r_] = mfit(d.iloc[rng.integers(0, n, n)], chainB["excess"], tuple(A.CORE))[0]
        con = {"n": rc(n)}
        for i, m in enumerate(mods):
            pr = {}
            for a_, c_ in PAIRS4[:3]:
                ia, ic = A.CORE.index(a_), A.CORE.index(c_)
                dd = b[i, ia] - b[i, ic]
                s = np.sqrt(max(cov[i, ia, ia] + cov[i, ic, ic] - 2 * cov[i, ia, ic], 0))
                pr[f"{a_}-{c_}"] = cd(dd, s)
            cbv = Cm @ b[i]
            S = Cm @ cov[i] @ Cm.T
            try:
                W = float(cbv @ np.linalg.solve(S, cbv))
                pw = float(stats.chi2.sf(W, 3))
            except np.linalg.LinAlgError:
                W, pw = None, None
            con[m] = {"b": {c: float(b[i, j]) for j, c in enumerate(A.CORE)}, "pairs": pr, "wald": W, "wald_p": pw,
                      "disattenuated": disatt(b[i], bb[:, i, :], rel, rb, A.CORE, PAIRS4[:3])}
        out["contrasts_excess"] = con
    # state loadings: module -> hr_excess, acute_dev, hr_noct (the acute-deviation mirror), M3 + RNA technical
    d = O.rows_for(fr, M, need("hr_excess", "acute_dev", "hr_noct", *M3))
    if len(d) >= MIN_N:
        Y = np.column_stack([zs(d[c]) for c in ("hr_excess", "acute_dev", "hr_noct")])
        b, se, _, _ = O.mols(Y, Xall[d["_si"].to_numpy()], O.covmat(d, M3, mod))
        out["state_loadings"] = {"n": rc(len(d)), **{m: {s: cd(b[i, j], se[i, j]) for j, s in
                                                          enumerate(("hr_excess", "acute_dev", "hr_noct"))}
                                                      for i, m in enumerate(mods)}}
    # beta-blocker: module difference (users vs no rate-lowering drug) and module x beta-blocker on PWV
    try:
        fr2 = fr.assign(_bbok=np.where((fr["bb"] == 1) | (fr["rate_drug"] == 0), 1.0, np.nan))
        d = O.rows_for(fr2, M, need("o_pwv", "_bbok", "bb", *M3_NOAHT))
        nu = int(d["bb"].sum())
        r = {"n_users": rc(nu), "n": rc(len(d))}
        if nu >= A.MINCELL and len(d) - nu >= A.MINCELL:
            C = O.covmat(d, M3_NOAHT, mod)
            X = Xall[d["_si"].to_numpy()]
            bbv = d["bb"].to_numpy(float)
            y = zs(d["o_pwv"])
            for i, m in enumerate(mods):
                ok = np.isfinite(X[:, i])
                if ok.sum() < MIN_N:
                    continue
                zm = zs(X[ok, i])
                b1, V1 = ols_v(zm, np.column_stack([C[ok], bbv[ok]]))
                b2, V2 = ols_v(y[ok], np.column_stack([C[ok], zm, bbv[ok], zm * bbv[ok]]))
                r[m] = {"bb_diff_sd": cd(b1[-1], np.sqrt(V1[-1, -1])), "x_bb_on_pwv": cd(b2[-1], np.sqrt(V2[-1, -1]))}
        else:
            r["status"] = "suppressed (users < 20)"
        out["beta_blocker"] = r
    except Exception as ex:
        out["beta_blocker"] = {"error": err(ex)}
    # minutes between blood draw and vascular acquisition; draw hour x module (base M3 incl. hour + hour^2)
    try:
        d = O.rows_for(fr, M, need("o_pwv", "acq_hour", *M3))
        d = d[d["_same"] & (d["_gap"] == 0)]
        si = d["_si"].to_numpy()
        dh = dh_all[si] if dh_all is not None else np.full(len(d), np.nan)
        gs = d["acq_hour"].astype(float).to_numpy() - dh  # signed: acquisition - draw (h)
        ok = np.isfinite(gs) & (np.abs(gs) <= GAP_MAX_H)
        d, gs, dh = d[ok], gs[ok], dh[ok]
        gap = np.abs(gs)  # shared acute state should decay with |acquisition - draw|
        r = {"n": rc(len(d)), "gap_h": A.qtiles(pd.Series(gs)), "abs_gap_h": A.qtiles(pd.Series(gap))}
        if len(d) >= MIN_N:
            C = O.covmat(d, M3, mod)
            X = Xall[d["_si"].to_numpy()]
            y = zs(d["o_pwv"])
            gc_, hc = gap - gap.mean(), dh - dh.mean()
            med = np.median(gap)
            for i, m in enumerate(mods):
                ok = np.isfinite(X[:, i])
                if ok.sum() < MIN_N:
                    continue
                zm = zs(X[ok, i])
                b1, V1 = ols_v(y[ok], np.column_stack([C[ok], zm, gc_[ok], zm * gc_[ok]]))
                b2, V2 = ols_v(y[ok], np.column_stack([C[ok], zm, zm * hc[ok]]))
                e = {"x_gap_per_h": cd(b1[-1], np.sqrt(V1[-1, -1])), "x_drawhour_per_h": cd(b2[-1], np.sqrt(V2[-1, -1]))}
                for lab, sel in (("gap_le_median", gap[ok] <= med), ("gap_gt_median", gap[ok] > med)):
                    if sel.sum() >= MIN_N:
                        b3, V3 = ols_v(y[ok][sel], np.column_stack([C[ok][sel], zm[sel]]))
                        e[lab] = cd(b3[-1], np.sqrt(V3[-1, -1]))
                r[m] = e
        out["timing_interactions"] = r
    except Exception as ex:
        out["timing_interactions"] = {"error": err(ex)}
    # sensitivities: + lymphocytes; + alerting reaction; + PBMC cell yield; + smoking, statin, systemic steroid (M4);
    # excluding rate-lowering drug users
    sens = {}
    try:
        cy = A.rd(root, "rna_seq/rna_seq.parquet", columns=["libprep__cells_per_100ul"])
        cy = A.num(cy, ["libprep__cells_per_100ul"])
        cy["cells_log"] = np.log(cy["libprep__cells_per_100ul"].where(cy["libprep__cells_per_100ul"] > 0))
        cy = cy.groupby(["participant_id", "research_stage"])["cells_log"].mean().reset_index()
        # keyed by the research_stage of the matched RNA sample (a date-matched sample may come from another stage);
        # a left merge on unique keys keeps the row order that M (positional) relies on
        si_ = M["si"].to_numpy()
        sst = np.where(si_ >= 0, np.asarray(mod["stage"], object)[np.clip(si_, 0, None)], "")
        frc = fr.drop(columns=[c for c in ("cells_log",) if c in fr.columns]).assign(_sst=sst).merge(
            cy.rename(columns={"research_stage": "_sst"}), on=["participant_id", "_sst"], how="left")
        if len(frc) != len(fr):
            raise ValueError("cell-yield merge changed the row count")
    except Exception as ex:
        sens["cell_yield_load"] = {"error": err(ex)}
        frc = fr.assign(cells_log=np.nan)
    fr3 = fr.assign(_norate=np.where(fr["rate_drug"] == 0, 1.0, np.nan),
                    _age2=(fr["age"].astype(float) - 50.0) ** 2, _age3=(fr["age"].astype(float) - 50.0) ** 3)
    m4 = [c for c in ("smk_current", "statin", "steroid_sys", "immsup") if c in fr.columns]
    for nm_, extra, data, covp in (("lymphocytes", "lymph_log", fr, chainB["excess"]),
                                   ("alerting", "alert", fr, chainB["excess"]),
                                   ("cell_yield", "cells_log", frc, chainB["excess"]),
                                   ("m4_smk_statin_ster_immsup", m4, fr, chainB["excess"]),
                                   # age-related modules (naive T, memory) vs beds non-linear in age
                                   ("age_cubic", ["_age2", "_age3"], fr3, chainB["excess"]),
                                   ("no_rate_drug", "_norate", fr3, chainB["excess"])):
        try:
            extra_l = extra if isinstance(extra, list) else [extra]
            d = O.rows_for(data, M, need(*cb_need, *extra_l))
            if len(d) < MIN_N:
                sens[nm_] = {"n": rc(len(d)), "status": "n<100"}
                continue
            b0, s0_, _ = mfit(d, M3)
            b1, s1_, _ = mfit(d, covp)
            r = {"n": rc(len(d)), "base": pack(b0[:, 0], s0_[:, 0], len(d)), "excess": pack(b1[:, 0], s1_[:, 0], len(d))}
            if nm_ != "no_rate_drug":
                b2, s2, _ = mfit(d, covp + extra_l)
                r["excess_plus"] = pack(b2[:, 0], s2[:, 0], len(d))
            sens[nm_] = r
        except Exception as ex:
            sens[nm_] = {"error": err(ex)}
    out["sensitivity"] = sens
    return out


# ============================================================================ digest

def v3(x):
    return A.v3(x)


def ci(c):
    if not isinstance(c, dict):
        return "NA"
    if "lo" in c:
        return f"{v3(c.get('b', c.get('d')))} [{v3(c.get('lo'))},{v3(c.get('hi'))}]"
    lo_hi = c.get("ci") or [None, None]
    return f"{v3(c.get('b', c.get('d')))} [{v3(lo_hi[0])},{v3(lo_hi[1])}]"


def star(p):
    return "*" if fnf(p) < 0.05 else ""


def fnf(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def qs(q):
    if not isinstance(q, dict) or "q05" not in q:
        return f"n {g(q, 'n')}"
    return f"n {q.get('n')} " + "/".join(v3(q.get(k)) for k in ("q05", "q25", "q50", "q75", "q95"))


def digest(R):
    L = [f"F1 NK-HR DIGEST {R.get('version')} | atlas {R.get('atlas_version')} | omics {R.get('omics_version')} | "
         f"runtime {v3(R.get('runtime_s'))} s"]
    st = R.get("settings") or {}
    L.append("settings: " + ", ".join(f"{k} {v}" for k, v in st.items()))
    L.append("units: beds adverse-oriented z; HR slopes per 10 bpm; acute = visit HR | nocturnal HR; usual = nocturnal HR "
             "with visit HR moving with it; beyond = usual - acute; module b = bed SD per module SD; ci 95 %")
    h = R.get("H0") or {}
    L.append("## H0 flow")
    if "error" in h:
        L.append(f"ERROR H0: {h['error']}")
    for stg, c in (h.get("counts") or {}).items():
        L.append(f"{stg}: " + " ".join(f"{k} {v}" for k, v in c.items()))
    for k in ("hr_visit", "hr_noct", "hr_diff", "hr_excess", "acute_dev", "n_nights_hr"):
        L.append(f"q {k}: {qs(g(h, 'quantiles', k))}")
    L.append("corr: " + "; ".join(f"{k} {v3(g(v, 'r'))} n {g(v, 'n')}" for k, v in (h.get("corr") or {}).items()))
    ni = h.get("nights") or {}
    L.append("ICC between visits: " + "; ".join(f"{k} {v3(g(v, 'icc'))} n {g(v, 'n')}"
                                                for k, v in (h.get("icc_between_visits") or {}).items())
             + f" | nights: single-night ICC {v3(ni.get('icc_single_night'))} within-visit SD {v3(ni.get('within_sd'))}"
             f" visits {ni.get('visits')} | undefined-stage nights {h.get('nights_stage_undefined')}")
    em = h.get("excess_model") or {}
    L.append(f"excess model (no rate drugs): n {em.get('n')} slope noct {v3(em.get('slope_noct'))} age {v3(em.get('slope_age'))} "
             f"male {v3(em.get('slope_male'))} r2 {v3(em.get('r2'))}; acute_dev r2 {v3(g(h, 'acute_dev_model', 'r2_device_on_visit_bp'))}")
    tm = h.get("timing") or {}
    L.append(f"timing: acq-bp min {qs(tm.get('acq_minus_bp_min'))} | acq-ecg min {qs(tm.get('acq_minus_ecg_min'))}")
    L.append(f"timing: night-visit days {qs(tm.get('night_minus_visit_days'))} | share before {v3(tm.get('share_nights_before_visit'))}"
             f" | prior-night gate (<=72 h) visits {tm.get('prior_night_gate_visits')}")
    L.append(f"lymph gap d {qs(g(h, 'quantiles', 'lymph_gap_d'))} | alert {qs(g(h, 'quantiles', 'alert'))}")
    for k, v in (h.get("errors") or {}).items():
        L.append(f"ERROR block {k}: {v}")
    rl = R.get("reliability_icc") or {}
    L.append("bed reliability (between-visit ICC [boot CI]): " + "; ".join(f"{k} {ci({'b': g(v, 'icc'), 'ci': g(v, 'ci')})}"
                                                                          for k, v in rl.items() if isinstance(v, dict)))
    # H1
    h = R.get("H1") or {}
    L.append("## H1 HR decomposition (no rate-lowering drugs; one row/participant): bed n | acute | usual | beyond | TOST acute | "
             "ux = nocturnal HR at fixed acute excess (beyond + gamma acute; gamma = excess-model slope noct)")
    if "error" in h:
        L.append(f"ERROR H1: {h['error']}")
    for bed, e in (g(h, "per_bed", "Mh1") or {}).items():
        t = e["acute"].get("p_tost")
        L.append(f"Mh1 {bed} n {e.get('n')} | {ci(e['acute'])} | {ci(e['usual_total'])} | {ci(e['beyond'])} p {v3(e['beyond'].get('p'))}"
                 + (f" | tost p {v3(t)} eq {e['acute'].get('equivalent')}" if t is not None else "")
                 + (f" | ux {ci(e['usual_given_excess'])}" if isinstance(e.get("usual_given_excess"), dict) else ""))
    m1 = g(h, "per_bed", "Mh1") or {}
    L.append("Mh1 RC usual HR acute/beyond (*p<0.05): " + "; ".join(
        f"{b} {v3(g(e, 'rc', 'acute', 'b'))}{star(g(e, 'rc', 'acute', 'p'))}/{v3(g(e, 'rc', 'beyond', 'b'))}"
        f"{star(g(e, 'rc', 'beyond', 'p'))}" for b, e in m1.items() if isinstance(e.get("rc"), dict) and b != "acute_dev"))
    m2 = g(h, "per_bed", "Mh2") or {}
    L.append("Mh2 (+acute_dev) acute/beyond (* p<0.05): " + "; ".join(
        f"{b} {v3(g(e, 'acute', 'b'))}{star(g(e, 'acute', 'p'))}/{v3(g(e, 'beyond', 'b'))}{star(g(e, 'beyond', 'p'))}"
        for b, e in m2.items()))
    for mn, e in (h.get("raw_pwv_m_s") or {}).items():
        L.append(f"raw PWV m/s per 10 bpm {mn} n {e.get('n')}: acute {ci(e['acute'])} usual {ci(e['usual_total'])} "
                 f"beyond {ci(e['beyond'])} ux {ci(e.get('usual_given_excess'))}")
    for term, c in (h.get("contrasts") or {}).items():
        L.append(f"4-bed {term} n {c.get('n')}: " + " ".join(f"{b} {v3(g(e, 'b'))}" for b, e in (c.get("b") or {}).items())
                 + f" | wald p {v3(g(c, 'wald', 'p'))} | " + "; ".join(
                     f"{k} {ci(g(c, 'pairs', k.replace('-', '_minus_')))}" for k in ("pwv-imt", "pwv-art_w", "pwv-abi_min")))
        L.append(f"4-bed {term} TOST(pwv-imt) eq {g(c, 'pairs', 'pwv_minus_imt', 'tost_equivalent')} | disattenuated: " + "; ".join(
            f"{k} {ci(g(c, 'disattenuated', k))}" for k in ("pwv-imt", "pwv-art_w", "pwv-abi_min")))
    wp = h.get("within_person_m_s") or {}
    L.append(f"within-person dPWV m/s n {wp.get('n')}: " + "; ".join(
        f"{k} {ci(wp.get(k))}" for k in ("d_hr_visit_per10", "d_hr_noct_per10", "d_sbp_per10")) + (
        f" {wp.get('status', '')}{wp.get('error', '')}"))
    for k, e in (h.get("sens_pwv") or {}).items():
        if not isinstance(e, dict):
            L.append(f"sens {k}: NA")
            continue
        L.append(f"sens {k} n {e.get('n')}: acute {ci(e.get('acute'))} usual {ci(e.get('usual_total'))} beyond "
                 f"{ci(e.get('beyond'))}" + (f" lambda {v3(e.get('mean_lambda'))}" if "mean_lambda" in e else ""))
    # H2
    h = R.get("H2") or {}
    L.append("## H2 beta-blocker positive control (users vs no rate-lowering drug; age, sex, BMI, SBP, other AHT)")
    if "error" in h:
        L.append(f"ERROR H2: {h['error']}")
    L.append(f"users {h.get('n_users')} non-users {h.get('n_nonusers')} {h.get('status', '')} {h.get('note', '')}".strip())
    if "pwv_total_m_s" in h:
        L.append("HR bpm: " + "; ".join(f"{k} {ci(h.get(k + '_bpm'))}" for k in ("hr_visit", "hr_noct", "hr_excess", "hr_diff")))
        L.append(f"PWV m/s: total {ci(h['pwv_total_m_s'])} direct {ci(h['pwv_direct_m_s'])} indirect via HR "
                 f"{ci(h['indirect_via_hr_m_s'])} prop {ci(h['proportion_mediated'])}; per bpm visit "
                 f"{ci(g(h, 'pwv_per_bpm', 'visit'))} noct {ci(g(h, 'pwv_per_bpm', 'noct'))}")
    # H3
    h = R.get("H3") or {}
    L.append("## H3 PBMC modules (RNA: batch + draw hour + hour^2 in every model; M3 = age, sex, BMI, SBP, AHT)")
    if "error" in h:
        L.append(f"ERROR H3: {h['error']}")
    fl = h.get("flow") or {}
    L.append(f"rna samples {fl.get('samples')} matched stage {fl.get('visits_matched_stage')} date {fl.get('visits_matched_date')}"
             f" | draw hour {qs(fl.get('draw_hour'))} | draw vs acq {fl.get('draw_vs_acq_hour', {}).get('r') if isinstance(fl.get('draw_vs_acq_hour'), dict) else 'NA'}")
    L.append("genes used: " + " ".join(f"{SHORT.get(k, k)} {v}/{len(MODULES[k])}" for k, v in (fl.get("genes_used") or {}).items()
                                       if k in MODULES) + f" | missing {h.get('modules_missing')}")
    mods = h.get("modules_used") or []
    rp = h.get("replication_M3") or {}
    L.append(f"replication M3 pwv n {rp.get('n')}: " + "; ".join(f"{SHORT[m]} {ci(rp.get(m))}" for m in mods))
    ca = h.get("chainA") or {}
    if "base" in ca:
        L.append(f"chain A (visit HR) pwv n {g(ca, 'base', 'n')} base/+hr_visit/+hr_visit+acute_dev: " + "; ".join(
            f"{SHORT[m]} {v3(g(ca, 'base', m, 'b'))}/{v3(g(ca, 'visit_hr', m, 'b'))}/{v3(g(ca, 'visit_hr_acute_dev', m, 'b'))}"
            for m in mods))
    cbb = h.get("chainB") or {}
    L.append(f"chain B (nocturnal HR; PRIMARY = excess) pwv n {h.get('chainB_n')}: module | base | excess | excess_only/"
             "acute_dev_only/visit_hr/decomposed/all_state | retention excess [CI] | all_state [CI]")
    for m in mods:
        if "base" not in cbb:
            break
        rt = h.get("chainB_retention") or {}
        L.append(f"{SHORT[m]} | {ci(g(cbb, 'base', m))} | {ci(g(cbb, 'excess', m))} | " + "/".join(
            v3(g(cbb, k, m, 'b')) for k in ("excess_only", "acute_dev_only", "visit_hr", "decomposed", "all_state")) + " | "
                 f"{v3(g(rt, 'excess', m, 'retention'))} {A.ci3(g(rt, 'excess', m, 'retention_ci'))} | "
                 f"{v3(g(rt, 'all_state', m, 'retention'))} {A.ci3(g(rt, 'all_state', m, 'retention_ci'))}")
    be = g(h, "beds", "excess") or {}
    L.append("module -> beds (excess model): b [CI] q (TOST p for non-PWV beds); n " + " ".join(
        f"{b} {g(e, 'n')}" for b, e in be.items()))
    for m in mods:
        for lab, bl in (("pwv/cavi0", ("pwv", "cavi0")), ("non-PWV", ("imt", "art_w", "abi_min", "vein_w"))):
            parts = []
            for b in bl:
                x = g(be, b, m)
                if x:
                    parts.append(f"{b} {v3(x['b'])} [{v3(x['lo'])},{v3(x['hi'])}] q{v3(x.get('q'))}"
                                 + (f" t{v3(x.get('p_tost'))}" if "p_tost" in x else ""))
            L.append(f"{SHORT[m]} {lab}: " + "; ".join(parts))
    bb0 = g(h, "beds", "base") or {}
    L.append("base model q<0.05: " + "; ".join(f"{b} " + ",".join(SHORT[m] for m in mods if fnf(g(e, m, "q")) < 0.05)
                                               for b, e in bb0.items()))
    cx = h.get("contrasts_excess") or {}
    L.append(f"4-bed contrasts (excess model) n {cx.get('n')}: pwv-imt | pwv-art_w | pwv-abi_min | Wald p | disattenuated "
             "pwv-imt | pwv-art_w")
    for m in mods:
        e = cx.get(m)
        if not e:
            continue
        L.append(f"{SHORT[m]}: " + " | ".join(ci(g(e, 'pairs', k)) for k in ("pwv-imt", "pwv-art_w", "pwv-abi_min"))
                 + f" | {v3(e.get('wald_p'))} | " + " | ".join(ci(g(e, 'disattenuated', k)) for k in ("pwv-imt", "pwv-art_w")))
    sl = h.get("state_loadings") or {}
    L.append(f"state loadings (state SD per module SD; M3 + RNA technical) n {sl.get('n')}: module | hr_excess | acute_dev | hr_noct")
    for m in mods:
        if m in sl:
            L.append(f"{SHORT[m]} | " + " | ".join(ci(sl[m].get(s)) for s in ("hr_excess", "acute_dev", "hr_noct")))
    bbk = h.get("beta_blocker") or {}
    L.append(f"beta-blocker (users {bbk.get('n_users')}, n {bbk.get('n')}{' ' + bbk['status'] if 'status' in bbk else ''}"
             f"{' ' + bbk['error'] if 'error' in bbk else ''}): module diff SD | x bb on PWV")
    L.append("; ".join(f"{SHORT[m]} {ci(g(bbk, m, 'bb_diff_sd'))} | {ci(g(bbk, m, 'x_bb_on_pwv'))}" for m in mods if m in bbk)
             or "beta-blocker module estimates: NA")
    ti = h.get("timing_interactions") or {}
    L.append(f"acquisition-draw gap (same day, |gap|<=6 h) n {ti.get('n')} gap h {qs(ti.get('gap_h'))} | abs {qs(ti.get('abs_gap_h'))}"
             f"{' ' + ti['error'] if 'error' in ti else ''}")
    for m in KEYM:
        if m in ti:
            e = ti[m]
            L.append(f"{SHORT[m]}: x |gap|/h {ci(e.get('x_gap_per_h'))} | |gap|<=med {ci(e.get('gap_le_median'))} | >med "
                     f"{ci(e.get('gap_gt_median'))} | x draw hour/h {ci(e.get('x_drawhour_per_h'))}")
    for k, e in (h.get("sensitivity") or {}).items():
        if "base" not in e:
            L.append(f"sens {k}: {e}")
            continue
        L.append(f"sens {k} n {e.get('n')}: " + "; ".join(
            f"{SHORT[m]} {v3(g(e, 'base', m, 'b'))}/{v3(g(e, 'excess', m, 'b'))}"
            + (f"/{v3(g(e, 'excess_plus', m, 'b'))}" if "excess_plus" in e else "") for m in KEYM if m in mods)
            + (" (base/excess/+extra)" if "excess_plus" in e else " (base/excess)"))
    L.append("## timings s: " + ", ".join(f"{k} {v3(v)}" for k, v in (R.get("timings") or {}).items()))
    for k, v in R.items():
        if isinstance(v, dict) and "error" in v and k not in ("H0", "H1", "H2", "H3"):
            L.append(f"ERROR {k}: {v['error']}")
    W = []
    for x in L:  # no leading/trailing or double spaces (transcription); wrap > 220 characters at '; ' boundaries
        x = re.sub(r" {2,}", " ", x.strip())
        while len(x) > 220:
            cut = x.rfind("; ", 0, 218)
            if cut > 40:
                cut += 1
            else:  # no '; ' boundary: wrap at the last space so that no number or key is split
                cut = x.rfind(" ", 0, 218)
                cut = cut if cut > 40 else 218
            W.append(x[:cut].rstrip())
            x = "+ " + x[cut:].strip()
        W.append(x)
    L = W
    if len(L) > 120:  # hard cap (transcription budget); everything is also in results.json
        L = L[:119] + [f"TRUNCATED {len(L) - 119} lines (see results.json)"]
    return "\n".join(L) + "\n"


# ============================================================================ main

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", default=None, help="atlas v1.5 script (default f1_atlas_v1_5.py beside this file)")
    ap.add_argument("--omics", default=None, help="omics v1.2 script (default f1_omics_v1_2.py beside this file)")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--max-genes", type=int, default=100, help="gene-level columns kept by load_rna (modules unaffected)")
    ap.add_argument("--sections", default="H0,H1,H2,H3")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    atlas = os.path.abspath(a.atlas) if a.atlas else _default(["f1_atlas_v1_5.py", "f1_atlas_v1.py"])
    omics = os.path.abspath(a.omics) if a.omics else _default(
        ["f1_omics_v1_2.py", "../../sandbox_packages/f1_omics_v12_20261004/f1_omics_v1_2.py"])
    setup(atlas, omics)
    secs = [s.strip() for s in a.sections.split(",") if s.strip()]
    bad = [s for s in secs if s not in ("H0", "H1", "H2", "H3")]
    if bad:
        raise SystemExit(f"unknown sections {bad}; use a subset of H0,H1,H2,H3")
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    log(f"{VERSION}; atlas {A.VERSION}; omics {O.VERSION}; numpy {np.__version__}; pandas {pd.__version__}; "
        f"python {sys.version.split()[0]}; boot {a.boot}")
    R = {"version": VERSION, "atlas_version": A.VERSION, "omics_version": O.VERSION,
         "settings": {"boot": a.boot, "seed": f"{a.seed:,}", "max_genes": a.max_genes, "rna_draw_time": "hour+hour^2",
                      "tost_delta": TOST, "hr_range": f"{HR_LIM[0]:g}-{HR_LIM[1]:g}", "gap_max_h": GAP_MAX_H,
                      "sections": ",".join(secs)}, "timings": {}}
    fr = A.build_frame(a.root, None)[0].reset_index(drop=True)
    R["timings"]["frame"] = time.time() - t0
    pids = set(fr["participant_id"].astype(str))
    t1 = time.time()
    fr, info = add_hr(a.root, fr)
    R["timings"]["hr_set"] = time.time() - t1
    log(f"HR set: hr_visit {A.cnt(fr['hr_visit'].notna().sum())}, hr_noct {A.cnt(fr['hr_noct'].notna().sum())}, "
        f"excess {A.cnt(fr['hr_excess'].notna().sum())} visits")
    try:
        rel, rb = reliability(fr, a.boot, rng)
        R["reliability_icc"] = {k: {"icc": v, "ci": pctl(rb[k])} for k, v in rel.items()}
    except Exception as ex:
        rel, rb = {}, {}
        R["reliability_icc"] = {"error": err(ex)}
    fns = {"H0": lambda: h0(fr, info), "H1": lambda: h1(fr, rel, rb, info.get("_s2w"), a.boot, rng, info.get("_gamma")),
           "H2": lambda: h2(fr, a.boot, rng), "H3": lambda: h3(fr, a.root, rel, rb, a.max_genes, a.boot, rng, pids)}
    for k in secs:
        guard(R, k, fns[k])
    R["runtime_s"] = time.time() - t0
    text = json.dumps(clean(R), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    O.privacy_check(text, pids)
    dg = digest(json.loads(text))
    O.privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines; total {time.time() - t0:.1f}s")
    logtext = "\n".join(A.LOG) + "\n"
    O.privacy_check(logtext, pids)
    os.makedirs(a.out)
    sums = []
    for fn_, tx in (("results.json", text), ("digest.txt", dg), ("run_log.txt", logtext)):
        with open(os.path.join(a.out, fn_), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn_}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s[:16] + "  " + s.split()[-1] for s in sums))
    return R


if __name__ == "__main__":
    main()
