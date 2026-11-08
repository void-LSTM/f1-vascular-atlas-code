#!/usr/bin/env python3
"""F1 outcomes module P1 v1.0 (HPP sandbox; writes aggregates only): incident hypertension by vascular bed, with leg
PWV split into a structural part and an acute (acquisition-state) part.

Beds come from the atlas frame (--atlas, default f1_atlas_v1_5.py beside this file; build_frame(root, None)), so they
are calendar/day/session-corrected exactly as in atlas v1.5 (raw kept as <bed>_raw). HR = hazard ratio per SD of the
adverse-oriented bed (atlas ORIENT: retinal artery width and ABI per SD lower).

Cohort (baseline 00_00_visit): untreated (no ATC C02/C03/C07/C08/C09), sitting BP < 140/90, no curated or self-reported
hypertension (atlas definition), no ICD-11 BA00-BA04/BA0Z at baseline or with a start year before baseline.
Follow-up contacts: 01_00_call, 02_00_visit, 03_00_call (date from events; else the earliest record date at that
stage; else baseline + the median contact time). A contact counts when it carries outcome data (curated/self-report
row, medication row, condition row, or visit-2 sitting BP). Medication and condition rows with research_stage
'undefined' are assigned by date to the next attended contact.
Composite event at a contact: curated or self-reported hypertension, antihypertensive drug, ICD-11 BA00-BA04/BA0Z, or (visit
2) sitting BP >= 140/90.  Measured outcome (co-primary): visit-2 exam, sitting BP >= 140/90 or antihypertensive drug.
Sections
  O0 flow       cohort flow, contacts attended/due by kind, events by contact and component, follow-up, IPW summary
  O1 pressure   usual BP by regression calibration from baseline readings only (target = visit-2 sitting BP of
                participants untreated at both visits; 5-fold cross-fitted; predictor sets full/dev/nodev/min); acute
                deviation = device brachial SBP - calibrated usual SBP; PWV = usual-BP part + acute part (acute
                deviation, sitting HR) + structural residual; CAVI0-based PWV at 100 mmHg (structural) and load part
  O2 composite  discrete-time hazard: complementary log-log over contact intervals, contact-kind intercepts,
                log(interval) offset, stabilised inverse-probability-of-follow-up weights per contact kind, robust
                SE clustered by participant.  Per bed: covariates A (age, sex, BMI, sitting SBP/DBP), B (age, sex,
                BMI, calibrated usual SBP/DBP), C (A + all baseline readings); unweighted B
  O3 struct     structural vs acute PWV (and acute deviation, HR; PWV at 100 mmHg vs load part), Wald difference
  O4 measured   co-primary measured outcome (one interval to visit 2, weighted for being in the measured v2 sample); BP-only
  O5 joint      three/four-bed joint models where events allow (n >= 300, events >= 20; EPV flagged), Firth-
                penalised sensitivity, Wald test of equal log-HR; common-sample separate models with stacked
                influence functions (paired contrasts)
  O6 topq       BP-adjusted top quintile (age/sex-standardised, adverse) per bed; every 2x2 person cell >= 20
  O7 predict    event by visit 2 (attendees): recalibrated Framingham-type logistic (age, sex, SBP, DBP, BMI, smoking,
                family history, age x DBP) +/- bed: apparent / optimism-corrected (bootstrap) / 10-fold CV AUC,
                delta AUC, calibration slope, E/O, Brier, decision-curve net benefit (guarded cells)
  O8 sens       ACC/AHA 130/80; objective-only and strict-diagnosis events; ABI 0.9-1.4; + height; right/left leg;
                PWV x baseline SBP >= 120; birth-month negative control; corrected vs raw PWV; within-visit
                calibration; cross-lagged PWV <-> SBP (untreated both visits); E-value
Privacy (as the atlas): aggregates only; counts rounded to 10, cells < 20 suppressed; every estimate needs >= 20
events (2x2 and decision-curve cells >= 20 each); quantiles 5-95 % only; no participant ids or dates; aborts if a 7+
digit run or a participant id appears in any output; refuses to overwrite --out.
Run: python f1_outcomes_v1.py --root <HPP_DATA_ROOT> --atlas f1_atlas_v1_5.py --out f1_outcomes_out_v1
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # small problems: avoid oversubscription
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

VERSION = "f1_outcomes_v1.0.1 2026-10-04"
A = None  # atlas module (load_atlas)
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "00_00_visit"
FU = {"01_00_call": ("call1", 1.0), "02_00_visit": ("v2", 2.1), "03_00_call": ("call3", 3.0)}
KINDS = ["call1", "v2", "call3"]
KEY = ["participant_id", "research_stage"]
ICD_HTN = ("BA00", "BA01", "BA02", "BA03", "BA04", "BA0Z")  # v1.0.1: + BA0Z (unspecified)
BEDS = ["pwv", "pwv_raw", "imt", "art_w", "abi_min", "cavi0", "sokolow"]
CORE4 = ["pwv", "imt", "art_w", "abi_min"]
JOINT_SETS = [("pwv", "imt", "abi_min"), ("pwv", "imt", "art_w"), ("pwv", "art_w", "abi_min"),
              ("pwv", "imt", "art_w", "abi_min")]
C3 = ["age", "sex_male", "bmi"]
COV_A = C3 + ["sbp_sit", "dbp_sit"]
COV_B = C3 + ["u_sbp", "u_dbp"]
OPT_READ = ["sbp_sit2", "dbp_sit2", "sbp_lie", "dbp_lie", "sbp_st1", "sbp_st3"]
ALL_READ = OPT_READ + ["dev_bsbp", "hr_sit"]
MIN_N, EPV, GAP0, KFOLD = 200, 5, 30, 5
THRESH = (0.05, 0.10, 0.20)
RHO, PREF = 1050.0, 100 * 133.322
LONG = re.compile(r"\d{7,}")
ATLAS_API = ["build_frame", "rd", "rc", "cnt", "qtiles", "pct", "zs", "ols", "logit", "auc", "fnum", "log", "LOG", "v3",
             "calibration_data", "wide", "fit", "coef", "cf", "ci3", "abi_clin", "MINCELL", "MAIN_STAGES", "ORIENT",
             "ATC_CLASSES", "TZ", "VERSION"]


# ============================================================================ helpers

def load_atlas(path=None):
    """Import the atlas by path (default f1_atlas_v1_5.py beside this file, else f1_atlas_v1.py there)."""
    global A
    if path is None:
        path = os.path.join(HERE, "f1_atlas_v1_5.py")
        if not os.path.exists(path):
            path = os.path.join(HERE, "f1_atlas_v1.py")
    spec = importlib.util.spec_from_file_location("f1_atlas", path)
    A = importlib.util.module_from_spec(spec)
    sys.modules["f1_atlas"] = A
    spec.loader.exec_module(A)
    miss = [x for x in ATLAS_API if not hasattr(A, x)]
    if miss:
        raise SystemExit(f"atlas module lacks {miss}; use atlas v1.5")
    return A


def log(m):
    A.log(m)


def rc(n):
    return A.rc(n)


def v3(x):
    return A.v3(x)


def err(ex):
    tb = ([f for f in traceback.extract_tb(ex.__traceback__) if f.filename == __file__] or
          traceback.extract_tb(ex.__traceback__))[-1]
    return f"{type(ex).__name__}: {str(ex)[:160]} (line {tb.lineno})"


def privacy_check(text, pids):
    m = LONG.search(text)
    if m:
        raise SystemExit(f"PRIVACY ABORT: digit run '{m.group(0)[:3]}...' in output")
    for p in pids:
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


def naive(s):
    """Naive local timestamps (datetime64[ns]); tz-aware input is converted to Asia/Jerusalem first."""
    s = pd.Series(s).reset_index(drop=True)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert(A.TZ).dt.tz_localize(None)
    s = pd.to_datetime(s, errors="coerce")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert(A.TZ).dt.tz_localize(None)
    return s.dt.as_unit("ns")


def years(a, b):
    return (pd.to_datetime(b) - pd.to_datetime(a)).dt.days / 365.25


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def codes(v):
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        out = set()
        for x in v:
            try:
                out.add(int(float(x)))
            except (TypeError, ValueError):
                pass
        return out
    return {int(t) for t in re.findall(r"-?\d+", str(v))}


def mflag(s, pos, empty=np.nan):
    """Multi-select -> 1 if code pos present, 0 for another valid answer, else missing (exposome v1 rule)."""
    def f(v):
        c = codes(v)
        return empty if not c else 1.0 if pos in c else 0.0 if any(x >= 0 or x in (-7, -17, -27) for x in c) else np.nan
    return s.map(f).astype(float)


def hrd(b, se):
    if b is None or se is None or not (np.isfinite(b) and np.isfinite(se)):
        return None
    return {"hr": float(np.exp(b)), "lo": float(np.exp(b - 1.96 * se)), "hi": float(np.exp(b + 1.96 * se)),
            "b": float(b), "se": float(se), "p": float(2 * stats.norm.sf(abs(b / se))) if se > 0 else None}


# ============================================================================ GLM (binary; cloglog or logit)

def _parts(X, b, y, off, link):
    """mu, score factor g (score_i = w (y - mu) g x), Fisher weight W, log-lik terms, d log W / d eta."""
    if link == "logit":
        eta = np.clip(X @ b + off, -30.0, 30.0)
        mu = expit(eta)
        g = np.ones_like(mu)
        W = mu * (1 - mu)
        ll = y * eta - np.logaddexp(0, eta)
        dlw = 1 - 2 * mu
    else:  # complementary log-log: mu = 1 - exp(-exp(eta)); eta capped at 4 (mu = 1 to double precision)
        eta = np.clip(X @ b + off, -30.0, 4.0)
        e = np.exp(eta)
        mu = np.clip(-np.expm1(-e), 1e-300, 1.0)
        g = e / mu                      # (d mu/d eta) / (mu (1 - mu))
        W = e * e * np.exp(-e) / mu     # (d mu/d eta)^2 / (mu (1 - mu))
        ll = y * np.log(mu) - (1 - y) * e
        dlw = 2 - e - e * np.exp(-e) / mu
    return mu, g, W, ll, dlw


def glm(y, X, off=None, w=None, cl=None, link="cloglog", firth=False, iters=100):
    """Weighted binary GLM by Fisher scoring with step halving. Robust (cluster) sandwich covariance from per-cluster
    influence functions IF = S_c I^-1 (V = G/(G-1) IF'IF). firth=True: Jeffreys-penalised likelihood l + 0.5 log|I|,
    maximised with the modified score U + 0.5 X'(h dlogW/deta) (h = leverages); same sandwich at the estimate."""
    y = np.asarray(y, float)
    n, p = X.shape
    off = np.zeros(n) if off is None else np.asarray(off, float)
    w = np.ones(n) if w is None else np.asarray(w, float)

    def objective(bb):
        mu, g, W, ll, dlw = _parts(X, bb, y, off, link)
        info = (X * (w * W)[:, None]).T @ X
        L = float(np.sum(w * ll))
        U = X.T @ (w * (y - mu) * g)
        if firth:
            sg, ld = np.linalg.slogdet(info)
            if sg <= 0:
                return -np.inf, U, info
            L += 0.5 * ld
            h = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(info), X) * w * W
            U = U + 0.5 * X.T @ (h * dlw)
        return L, U, info

    b = np.zeros(p)
    conv = False
    L0, U, info = objective(b)
    for _ in range(iters):
        try:
            step = np.linalg.solve(info + 1e-9 * np.eye(p), U)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(info) @ U
        t = 1.0
        while True:
            bn = b + t * step
            Ln, Un, In = objective(bn)
            if (np.isfinite(Ln) and Ln >= L0 - 1e-9) or t < 1e-4:
                break
            t /= 2
        b, L0, U, info = bn, Ln, Un, In
        if np.max(np.abs(t * step)) < 1e-7:
            conv = True
            break
    mu, g, W, ll, _ = _parts(X, b, y, off, link)
    info = (X * (w * W)[:, None]).T @ X
    bread = np.linalg.pinv(info)
    s = X * (w * (y - mu) * g)[:, None]
    codes_, groups = pd.factorize(pd.Series(np.arange(n) if cl is None else cl))
    G = len(groups)
    S = np.column_stack([np.bincount(codes_, weights=s[:, j], minlength=G) for j in range(p)])
    IF = S @ bread
    V = IF.T @ IF * G / max(G - 1, 1)
    return {"b": b, "V": V, "IF": IF, "groups": np.asarray(groups), "conv": conv, "ll": float(np.sum(w * ll))}


# ============================================================================ loading

def med_rows(root):
    m = A.rd(root, "medications/medications.parquet", columns=["collection_date", "atc3", "atc4", "atc5"])
    if m is None:
        return None
    m = m.reset_index(drop=True)
    m["_row"] = np.arange(len(m))
    hit = set()
    pre = tuple(A.ATC_CLASSES["antihtn"])
    for c in ("atc3", "atc4", "atc5"):
        if c in m.columns:
            e = m[["_row", c]].explode(c).dropna(subset=[c])
            hit |= set(e.loc[e[c].astype(str).str.upper().str.startswith(pre), "_row"])
    m["med"] = m["_row"].isin(hit).astype(float)
    m["date"] = naive(m["collection_date"]).values if "collection_date" in m.columns else pd.NaT
    return m[["participant_id", "research_stage", "date", "med"]]


def cond_rows(root):
    c = A.rd(root, "medical_conditions/medical_conditions.parquet", columns=["icd11_code", "collection_date", "start_year"])
    if c is None:
        return None
    code = c["icd11_code"].astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
    c["icd"] = code.str.startswith(ICD_HTN).astype(float)
    c["date"] = naive(c["collection_date"]).values if "collection_date" in c.columns else pd.NaT
    c["start_year"] = pd.to_numeric(c["start_year"], errors="coerce") if "start_year" in c.columns else np.nan
    return c[["participant_id", "research_stage", "date", "icd", "start_year"]]


def htn_rows(root):
    cur, slf = "hypertension__curated_phenotype", "hypertension__self_reported_hypertension"
    h = A.rd(root, "curated_phenotypes/hypertension.parquet", columns=[cur, slf])
    if h is None:
        return None
    cat = h[cur].astype(str).str.strip().str.lower() if cur in h.columns else pd.Series("nan", index=h.index)
    valid = ~cat.isin(["nan", "<na>", "none", ""])
    h["dx_any"] = np.where(valid, (cat.str.contains("hypertens") & ~cat.str.startswith("non")).astype(float), np.nan)
    h["dx_strict"] = np.where(valid, (cat == "hypertension").astype(float), np.nan)
    sr = h[slf].astype(object) if slf in h.columns else pd.Series(np.nan, index=h.index, dtype=object)
    h["self"] = pd.to_numeric(sr.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0,
                                      1: 1.0, 0: 0.0}), errors="coerce")
    h["h_has"] = (valid | h["self"].notna()).astype(float)
    h["cat"] = np.where(valid, cat, "missing")
    return h[KEY + ["dx_any", "dx_strict", "self", "h_has", "cat"]]


def load_fam(root):
    parts = []
    d = A.rd(root, "family_history/ukbb.parquet")
    if d is not None:
        ill = [c for c in d.columns if re.match(r"family_history_past_(father|mother|siblings)_illness_\d$", c)]
        if ill:
            parts.append(pd.DataFrame({"participant_id": d["participant_id"],
                                       "fam_htn": pd.concat([mflag(d[c], 8) for c in ill], axis=1).max(axis=1)}))
    d = A.rd(root, "family_history/initial_medical.parquet", columns=["family_history"])
    if d is not None and "family_history" in d.columns:
        parts.append(pd.DataFrame({"participant_id": d["participant_id"], "fam_htn": mflag(d["family_history"], 1, 0.0)}))
    if not parts:
        return None
    return pd.concat(parts, ignore_index=True).groupby("participant_id")["fam_htn"].max().reset_index()


def load_contacts(root, fr):
    """Follow-up contacts (one row per participant x follow-up stage) with outcome components, baseline dates and the
    set of participants with a prevalent hypertension ICD code."""
    info = {}
    st_all = [BASE] + list(FU)
    ev = A.rd(root, "events/events.parquet", columns=["research_stage_date"])
    ev["date"] = naive(ev["research_stage_date"]).values
    evd = ev[ev["research_stage"].isin(st_all)].groupby(KEY)["date"].min().rename("edate")
    h, m, c = htn_rows(root), med_rows(root), cond_rows(root)
    parts = [evd]
    if h is not None:
        parts.append(h[h["research_stage"].isin(st_all)].groupby(KEY)[["dx_any", "dx_strict", "self", "h_has"]].max())
    if m is not None:
        parts.append(m[m["research_stage"].isin(st_all)].groupby(KEY).agg(med=("med", "max"), mdate=("date", "min"),
                                                                          m_has=("med", "size")))
    if c is not None:
        parts.append(c[c["research_stage"].isin(st_all)].groupby(KEY).agg(icd=("icd", "max"), cdate=("date", "min"),
                                                                          c_has=("icd", "size")))
    v2 = fr[fr["research_stage"] == "02_00_visit"].drop_duplicates("participant_id")
    bp = v2.set_index(KEY)[["sbp_sit", "dbp_sit"]].rename(columns={"sbp_sit": "sbp2", "dbp_sit": "dbp2"})
    parts.append(bp)
    T = pd.concat(parts, axis=1).reset_index()
    for col in ("dx_any", "dx_strict", "self", "h_has", "med", "icd", "m_has", "c_has"):
        if col not in T.columns:
            T[col] = np.nan
    for col in ("mdate", "cdate", "edate"):
        if col not in T.columns:
            T[col] = pd.NaT
    # baseline dates: events, else the frame visit date
    b0 = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id").set_index("participant_id")["visit_date"]
    b0 = pd.Series(naive(b0).values, index=b0.index)  # v1.0.1: the frame date may be tz-aware (fillna would fail)
    bd = T[T["research_stage"] == BASE].set_index("participant_id")["edate"]
    base_dates = pd.Series(naive(bd.reindex(b0.index)).values, index=b0.index).fillna(b0)
    base_dates = base_dates.astype("datetime64[ns]")
    C = T[T["research_stage"].isin(list(FU))].copy()
    C["kind"] = C["research_stage"].map({k: v[0] for k, v in FU.items()})
    C["base_date"] = C["participant_id"].map(base_dates)
    C["date"] = C["edate"].fillna(C[["mdate", "cdate"]].min(axis=1))
    C["bp_has"] = ((C["kind"] == "v2") & C["sbp2"].notna() & C["dbp2"].notna()).astype(float)
    C["has_data"] = ((C["h_has"].fillna(0) > 0) | (C["m_has"].fillna(0) > 0) | (C["c_has"].fillna(0) > 0) |
                     (C["bp_has"] > 0))
    info["stage_rows_without_outcome_data"] = {k: rc(((C["kind"] == k) & ~C["has_data"] & C["edate"].notna()).sum())
                                               for k in KINDS}
    # nominal contact time: median observed gap (events dates) per kind
    t = years(C["base_date"], C["date"])
    nom = {}
    for k, (_, dflt) in zip(KINDS, FU.values()):
        tk = t[(C["kind"] == k) & C["has_data"] & (t > GAP0 / 365.25)]
        nom[k] = float(tk.median()) if len(tk) >= 20 else dflt
    info["nominal_years"] = nom
    imp = C["date"].isna() & C["base_date"].notna()
    C.loc[imp, "date"] = C.loc[imp, "base_date"] + pd.to_timedelta(C.loc[imp, "kind"].map(nom) * 365.25, unit="D")
    C["date_imputed"] = imp
    info["dates_imputed"] = {k: rc((imp & C["has_data"] & (C["kind"] == k)).sum()) for k in KINDS}
    C["t"] = years(C["base_date"], C["date"])
    # 'undefined'-stage medication / condition rows: to the next attended contact after the row date
    mapped = {}
    att = C[C["has_data"] & (C["t"] > GAP0 / 365.25)][["participant_id", "kind", "date"]].sort_values("date")
    for name, R, col in (("med", m, "med"), ("icd", c, "icd")):
        if R is None or not len(att):
            continue
        u = R[~R["research_stage"].isin(st_all) & (R[col] == 1) & R["date"].notna()][["participant_id", "date"]]
        u = u[u["date"] > u["participant_id"].map(base_dates) + pd.Timedelta(days=GAP0)].sort_values("date")
        if not len(u):
            mapped[name] = None  # v1.0.1: an exact zero is a cell < 20
            continue
        j = pd.merge_asof(u, att.rename(columns={"date": "cdate_"}), left_on="date", right_on="cdate_",
                          by="participant_id", direction="forward").dropna(subset=["kind"])
        hitk = set(zip(j["participant_id"], j["kind"]))
        sel = pd.Series([(p_, k_) in hitk for p_, k_ in zip(C["participant_id"], C["kind"])], index=C.index)
        C.loc[sel, col] = 1.0
        mapped[name] = rc(len(j))
    info["undefined_rows_mapped"] = mapped
    # prevalent ICD-11 hypertension
    prev = set()
    if c is not None:
        ci = c[c["icd"] == 1].copy()
        by = ci["participant_id"].map(base_dates)
        prev |= set(ci.loc[ci["research_stage"] == BASE, "participant_id"])
        prev |= set(ci.loc[ci["start_year"].notna() & (ci["start_year"] < by.dt.year), "participant_id"])
        und = ~ci["research_stage"].isin(st_all) & ci["date"].notna() & (ci["date"] <= by + pd.Timedelta(days=GAP0))
        prev |= set(ci.loc[und, "participant_id"])
    if h is not None:  # curated category distribution at follow-up stages (for the flow line)
        info["_hcat"] = h[h["research_stage"].isin(list(FU))]
    cols = ["participant_id", "kind", "date", "t", "date_imputed", "has_data", "dx_any", "dx_strict", "self", "med", "icd",
            "sbp2", "dbp2"]
    return C[cols].reset_index(drop=True), base_dates, prev, info


# ============================================================================ cohort, calibration, decomposition

def make_cohort(fr, base_dates, prev, fam, thr=(140, 90)):
    b = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id").copy()
    flow = [["baseline", rc(len(b))]]  # ordered steps
    b = b[b["antihtn"].fillna(0) == 0]
    flow.append(["untreated", rc(len(b))])
    b = b[(b["sbp_sit"] < thr[0]) & (b["dbp_sit"] < thr[1])]
    flow.append([f"bp<{thr[0]}/{thr[1]}", rc(len(b))])
    for col in ("htn_curated", "htn_self"):
        if col in b.columns:
            b = b[b[col].fillna(0) == 0]
    flow.append(["no curated/self-reported htn", rc(len(b))])
    b = b[~b["participant_id"].isin(list(prev))]
    flow.append(["no ICD BA00-BA04/BA0Z", rc(len(b))])
    b["base_date"] = b["participant_id"].map(base_dates)
    b = b[b["base_date"].notna()]
    flow.append(["with baseline date (at risk)", rc(len(b))])
    by = pd.to_datetime(b["base_date"])
    b["base_year"] = by.dt.year + (by.dt.dayofyear - 1) / 365.25
    if fam is not None:
        b = b.merge(fam, on="participant_id", how="left")
    else:
        b["fam_htn"] = np.nan
    for bed in BEDS:
        if bed in b.columns and "o_" + bed not in b.columns:
            b["o_" + bed] = A.ORIENT.get(bed, 1) * b[bed]
    return b.reset_index(drop=True), flow


def rc_fit(train, apply_, target, preds, folds):
    """OLS calibration of `target` on `preds` in `train`; predictions for `apply_` (cross-fitted for training rows)."""
    def X(d):
        return np.column_stack([np.ones(len(d))] + [d[p].astype(float).values for p in preds])
    yt = train[target].astype(float).values
    bfull = np.linalg.lstsq(X(train), yt, rcond=None)[0]
    pred = pd.Series(X(apply_) @ bfull, index=apply_.index)
    oof = np.full(len(train), np.nan)
    fid = folds.reindex(train["participant_id"]).values
    for f in range(KFOLD):
        tr, te = fid != f, fid == f
        if te.sum() == 0 or tr.sum() < 50:
            continue
        bf = np.linalg.lstsq(X(train[tr]), yt[tr], rcond=None)[0]
        oof[te] = X(train[te]) @ bf
    o = pd.Series(oof, index=train["participant_id"].values).dropna()
    pid = apply_["participant_id"]
    hit = pid.isin(o.index).values
    pred[hit] = o.reindex(pid[hit]).values
    ok = np.isfinite(oof)
    r2 = float(1 - np.var(yt[ok] - oof[ok]) / np.var(yt[ok])) if ok.sum() > 20 else None
    return pred, bfull, r2


def usual_bp(fr, seed, within=False):
    """Usual SBP/DBP from baseline readings only. Default target = visit-2 sitting BP of participants untreated at both
    visits (atlas calibration_data); within=True: target = second sitting reading of the baseline visit (untreated),
    predictors exclude it. Predictor sets full -> dev -> nodev -> min; each participant gets the first complete set."""
    base = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id").reset_index(drop=True)
    if within:
        cal = base[(base["antihtn"].fillna(0) == 0)].rename(columns={"sbp_sit2": "sbp_rep", "dbp_sit2": "dbp_rep"})
        opt = [c for c in OPT_READ if c not in ("sbp_sit2", "dbp_sit2")]
    else:
        cal = A.calibration_data(fr)
        if cal is None or not len(cal):
            return None, {"error": "no calibration sample"}
        opt = list(OPT_READ)
    opt = [c for c in opt if c in cal.columns and cal[c].notna().mean() >= 0.5]
    # v1.0.1: 'dev' set (device SBP without the optional readings) before 'nodev'. Without it, everyone missing a
    # lying/standing reading fell to 'min' (no device SBP), so their acute deviation (device SBP - usual SBP) carried the
    # residual usual BP and the 'acute' HR was biased upward under a planted null (check_f1_outcomes_review.py).
    sets = {"full": ["sbp_sit", "dbp_sit"] + opt + ["dev_bsbp", "hr_sit"] + C3,
            "dev": ["sbp_sit", "dbp_sit", "dev_bsbp", "hr_sit"] + C3,
            "nodev": ["sbp_sit", "dbp_sit"] + opt + ["hr_sit"] + C3,
            "min": ["sbp_sit", "dbp_sit"] + C3}
    seen = []
    for name in list(sets):  # drop a set identical to an earlier one (e.g. 'dev' when no optional reading qualifies)
        if sets[name] in seen:
            del sets[name]
        else:
            seen.append(sets[name])
    rng = np.random.default_rng(seed)
    ids = cal["participant_id"].drop_duplicates()
    folds = pd.Series(rng.integers(0, KFOLD, len(ids)), index=ids.values)
    out = pd.DataFrame({"participant_id": base["participant_id"], "u_sbp": np.nan, "u_dbp": np.nan, "rc_set": None})
    info = {"target": "baseline second sitting reading" if within else "visit-2 sitting BP, untreated both visits",
            "optional_readings": opt, "sets": {}}
    for name, preds in sets.items():
        preds = [p for p in preds if p in base.columns and p in cal.columns]
        c = cal.replace([np.inf, -np.inf], np.nan).dropna(subset=preds + ["sbp_rep", "dbp_rep"])
        if len(c) < 100:
            info["sets"][name] = {"n": rc(len(c)), "status": "too few"}
            continue
        okr = out["rc_set"].isna().values & base[preds].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).values
        ap = base[okr]
        rec = {"n_calibration": rc(len(c)), "predictors": preds}
        for tgt, col in (("sbp_rep", "u_sbp"), ("dbp_rep", "u_dbp")):
            pred, bfull, r2 = rc_fit(c, ap, tgt, preds, folds)
            out.loc[okr, col] = pred.values
            rec[f"{col}_r2_cv"] = r2
            if tgt == "sbp_rep":
                rec["coef_sbp_sit"] = float(bfull[1 + preds.index("sbp_sit")])
                if "dev_bsbp" in preds:
                    rec["coef_dev_bsbp"] = float(bfull[1 + preds.index("dev_bsbp")])
        out.loc[okr, "rc_set"] = name
        rec["n_assigned_baseline"] = rc(okr.sum())
        info["sets"][name] = rec
    return out, info


def decompose(B):
    """PWV = usual-BP part + acute part (acute deviation = device SBP - usual SBP; sitting HR) + structural part."""
    need = ["pwv", "u_sbp", "u_dbp", "dev_bsbp", "hr_sit"] + C3
    for col in ("acute_dev", "pwv_acute", "pwv_usual", "pwv_struct"):
        B[col] = np.nan
    miss = [c for c in need if c not in B.columns]
    if miss:  # v1.0.1: a missing column skips the decomposition (models report it) instead of a fatal KeyError
        return {"status": f"missing columns {miss}"}
    d = B.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
    B["acute_dev"] = B["dev_bsbp"] - B["u_sbp"]
    if len(d) < MIN_N:
        return {"n": rc(len(d)), "status": "too few"}
    ac = (d["dev_bsbp"] - d["u_sbp"]).values
    hr = d["hr_sit"].values
    m = {"u_sbp": d["u_sbp"].mean(), "u_dbp": d["u_dbp"].mean(), "ac": ac.mean(), "hr": hr.mean()}
    X = np.column_stack([np.ones(len(d)), (d["u_sbp"] - m["u_sbp"]) / 10, (d["u_dbp"] - m["u_dbp"]) / 10,
                         (ac - m["ac"]) / 10, (hr - m["hr"]) / 10] + [d[c].astype(float).values for c in C3])
    b, se, r2 = A.ols(d["pwv"].astype(float).values, X)
    names = ["usual_sbp_per10", "usual_dbp_per10", "acute_dev_per10", "hr_per10bpm"]
    coefs = {nm: {"b": float(b[i + 1]), "lo": float(b[i + 1] - 1.96 * se[i + 1]), "hi": float(b[i + 1] + 1.96 * se[i + 1])}
             for i, nm in enumerate(names)}
    acute = b[3] * (B["acute_dev"] - m["ac"]) / 10 + b[4] * (B["hr_sit"] - m["hr"]) / 10
    usual = b[1] * (B["u_sbp"] - m["u_sbp"]) / 10 + b[2] * (B["u_dbp"] - m["u_dbp"]) / 10
    B["pwv_acute"], B["pwv_usual"] = acute, usual
    B["pwv_struct"] = B["pwv"] - acute - usual
    cav = B["cavi0"].where(B["cavi0"] > 0) if "cavi0" in B.columns else pd.Series(np.nan, index=B.index)
    B["pwv_ref"] = np.sqrt(cav * PREF / (2 * RHO))
    B["pwv_load"] = B["pwv"] - B["pwv_ref"]
    e = B.loc[d.index]
    vp = float(e["pwv"].var())
    return {"n": rc(len(d)), "r2": float(r2), "coef_m_s": coefs, "acute_dev": A.qtiles(e["acute_dev"]),
            "var_share": {k: float(e[k].var() / vp) for k in ("pwv_usual", "pwv_acute", "pwv_struct")},
            "corr_with_pwv": {k: float(e[[k, "pwv"]].corr().iloc[0, 1]) for k in ("pwv_usual", "pwv_acute", "pwv_struct")},
            "corr_struct_usual_sbp": float(e[["pwv_struct", "u_sbp"]].corr().iloc[0, 1]),
            "pwv_ref": A.qtiles(B["pwv_ref"]), "corr_pwv_ref_pwv": float(B[["pwv_ref", "pwv"]].corr().iloc[0, 1])}


# ============================================================================ person-period data and weights

def ev_composite(c, thr=(140, 90)):
    bpv = (c["kind"] == "v2") & ((c["sbp2"] >= thr[0]) | (c["dbp2"] >= thr[1]))
    return ((c["dx_any"] == 1) | (c["self"] == 1) | (c["med"] == 1) | (c["icd"] == 1) | bpv).astype(float)


def ev_objective(c):
    bpv = (c["kind"] == "v2") & ((c["sbp2"] >= 140) | (c["dbp2"] >= 90))
    return ((c["med"] == 1) | (c["icd"] == 1) | bpv).astype(float)


def ev_strict(c):
    bpv = (c["kind"] == "v2") & ((c["sbp2"] >= 140) | (c["dbp2"] >= 90))
    return ((c["dx_strict"] == 1) | (c["self"] == 1) | (c["med"] == 1) | (c["icd"] == 1) | bpv).astype(float)


def person_period(B, C, evfn):
    """One row per attended contact interval until the first event. Columns participant_id, kind, t0, t1, dt, ev."""
    c = C[C["has_data"] & C["participant_id"].isin(B["participant_id"]) & (C["t"] > GAP0 / 365.25)].copy()
    c["ev"] = evfn(c).values
    c = c.sort_values(["participant_id", "t"])
    # contacts within 0.02 y (~1 week) of the previous one are merged into one (event = any; kind = the later)
    gap = c.groupby("participant_id")["t"].diff().fillna(np.inf).values
    c["blk"] = np.cumsum(gap > 0.02)
    c = c.groupby("blk", sort=False).agg(participant_id=("participant_id", "first"), kind=("kind", "last"),
                                         t1=("t", "max"), ev=("ev", "max")).reset_index(drop=True)
    c["t0"] = c.groupby("participant_id")["t1"].shift(1).fillna(0.0)
    c["dt"] = c["t1"] - c["t0"]
    c["cum"] = c.groupby("participant_id")["ev"].cumsum()
    c = c[(c["cum"] - c["ev"]) == 0]
    return c[["participant_id", "kind", "t0", "t1", "dt", "ev"]].reset_index(drop=True)


def wmatrix(B):
    cols = ["age", "sex_male", "bmi", "sbp_sit", "dbp_sit", "hr_sit", "base_year", "smk_current", "fam_htn",
            "o_pwv", "o_imt", "o_abi_min", "o_art_w"]
    parts, names = [], []
    for c in cols:
        if c not in B.columns:
            continue
        x = B[c].astype(float).replace([np.inf, -np.inf], np.nan)
        if x.notna().sum() < 20 or x.std() == 0:
            continue
        miss = x.isna()
        x = (x - x.mean()) / x.std()
        parts.append(x.fillna(0.0).values)
        names.append(c)
        if miss.any() and miss.mean() < 0.99:
            parts.append(miss.astype(float).values)
            names.append(c + "_miss")
    by = (B["base_year"] - B["base_year"].mean()) / max(B["base_year"].std(), 1e-6)
    parts.append((by ** 2).fillna(0).values)
    names.append("base_year_sq")
    return np.column_stack([np.ones(len(B))] + parts), names


def _attend_w(X, a, s, rec):
    """Stabilised weights 1/P(attend) from a logistic model of a on X among s; truncated at the 1st/99th percentile of
    the attendees' weights. Weights 1 (with a status) when either cell is < 20 or the model fails."""
    y = a[s].astype(float)
    wk = np.ones(len(a))
    if y.sum() < MIN_EV() or (1 - y).sum() < MIN_EV():
        rec["status"] = "weights 1 (too few attended or not attended)"
        return wk
    b, _, ps = A.logit(y, X[s])
    if b is None or not np.all(np.isfinite(b)):
        rec["status"] = "weights 1 (attendance model failed)"
        return wk
    sw = y.mean() / np.clip(expit(X @ b), 0.01, 1.0)
    lo, hi = np.percentile(sw[s & a], [1, 99])
    wk = np.clip(sw, lo, hi)
    rec.update({"auc_attendance": A.auc(y, ps), "weights_attended": A.qtiles(pd.Series(wk[s & a]))})
    return wk


def ipw(B, C, PP, PPm=None):
    """Stabilised inverse probability of attending each contact kind (logistic on baseline covariates and beds) among
    participants due by the data cut-off for that kind and event-free at earlier kinds; truncated at 1st/99th pct.
    v1.0.1: with PPm (measured visit-2 sample), also w_v2m = 1/P(in the measured sample) among all participants due at
    visit 2 (no event restriction: the measured outcome is cross-sectional at visit 2), on the same baseline covariates
    plus the observed history before visit 2 (call 1 attended; event observed at call 1)."""
    X, _ = wmatrix(B)
    pid = B["participant_id"].values
    att = C[C["has_data"] & (C["t"] > GAP0 / 365.25)]
    evk = PP.groupby(["participant_id", "kind"])["ev"].max()
    out = pd.DataFrame({"participant_id": pid})
    info = {}
    due_k = {}
    for i, k in enumerate(KINDS):
        ak = att[att["kind"] == k]
        a = np.isin(pid, ak["participant_id"].values)
        nomk = float(ak["t"].median()) if len(ak) >= 20 else list(FU.values())[i][1]
        cut = ak["date"].max() if len(ak) else pd.NaT
        due = (B["base_date"] + pd.to_timedelta((nomk - 0.5) * 365.25, unit="D") <= cut).values if pd.notna(cut) \
            else np.zeros(len(B), bool)
        due_k[k] = due
        prior = np.zeros(len(B), bool)
        for kk in KINDS[:i]:
            e = evk.xs(kk, level="kind") if kk in evk.index.get_level_values("kind") else pd.Series(dtype=float)
            prior |= np.isin(pid, e[e == 1].index.values)
        s = due & ~prior
        rec = {"due_event_free": rc(s.sum()), "attended": rc((a & s).sum()), "attended_all": rc(a.sum())}
        out["w_" + k] = _attend_w(X, a, s, rec)
        info[k] = rec
    if PPm is not None:
        a = np.isin(pid, PPm["participant_id"].values)
        s = due_k["v2"]
        a1 = np.isin(pid, att.loc[att["kind"] == "call1", "participant_id"].values).astype(float)
        e1 = np.isin(pid, PP.loc[(PP["kind"] == "call1") & (PP["ev"] == 1), "participant_id"].values).astype(float)
        extra = [h for h in (a1, e1) if min(h[s].sum(), (1 - h[s]).sum()) >= MIN_EV()]  # history terms that vary
        Xm = np.column_stack([X] + extra)
        rec = {"due": rc(s.sum()), "attended": rc((a & s).sum()), "attended_all": rc(a.sum()),
               "history_terms": len(extra)}
        out["w_v2m"] = _attend_w(Xm, a, s, rec)
        info["v2_measured"] = rec
    return out, info


def attach_w(PP, W, col=None):
    """Weight of each interval = weight of the contact kind that closes it (col: one kind for all rows)."""
    PP = PP.drop(columns=[c for c in ("w",) if c in PP.columns]).merge(W, on="participant_id", how="left")
    if col:
        PP["w"] = PP[col].fillna(1.0).values
    else:
        PP["w"] = np.select([PP["kind"].values == k for k in KINDS], [PP["w_" + k].fillna(1.0).values for k in KINDS],
                            default=1.0)
    return PP.drop(columns=[c for c in PP.columns if c.startswith("w_")])


def MIN_EV():
    return A.MINCELL


# ============================================================================ model wrappers

def hazard(PP, B, xs, covs, weighted=True, firth=False, std=True, min_n=MIN_N, min_ev=None):
    """Discrete-time cloglog hazard: contact-kind intercepts, log(dt) offset, IPW (column w), cluster-robust SE.
    xs are standardised at person level (std=True). Returns (summary dict, aux with b/V/IF for the xs)."""
    min_ev = MIN_EV() if min_ev is None else min_ev
    need = list(dict.fromkeys(list(xs) + list(covs)))
    miss = [c for c in need if c not in B.columns]
    if miss:
        return {"status": f"missing columns {miss}"}, None
    d = PP.merge(B[["participant_id"] + need], on="participant_id", how="inner")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=need + ["dt", "ev"])
    pids = pd.unique(d["participant_id"])
    n, ev = len(pids), int(d["ev"].sum())
    res = {"n": rc(n), "events": rc(ev), "person_years": rc(int(round(d["dt"].sum())))}
    if n < min_n or ev < min_ev:
        res["status"] = "too few"
        return res, None
    pb = B.drop_duplicates("participant_id").set_index("participant_id").loc[pids]
    for x in xs:
        sd = float(pb[x].astype(float).std()) if std else 1.0
        if not np.isfinite(sd) or sd <= 0:
            res["status"] = f"no variation in {x}"
            return res, None
        if std:
            d[x] = (d[x].astype(float) - float(pb[x].astype(float).mean())) / sd
    kind = d["kind"].astype(str).copy()
    evk = d.groupby(kind)["ev"].sum()
    if "call3" in evk.index and evk["call3"] < 5 and "call1" in evk.index:
        kind = kind.replace({"call3": "call1"})
        evk = d.groupby(kind)["ev"].sum()
    keep = kind.isin(evk[evk > 0].index).values
    d, kind = d[keep], kind[keep]
    D = pd.get_dummies(kind, dtype=float)
    Xc = []
    for c in covs:
        x = d[c].astype(float)
        s = x.std()
        if s > 0:
            Xc.append(((x - x.mean()) / s).values)
    X = np.column_stack([D.values] + [d[x].astype(float).values for x in xs] + Xc)
    w = d["w"].values if (weighted and "w" in d.columns) else None
    f = glm(d["ev"].values, X, off=np.log(d["dt"].values), w=w, cl=d["participant_id"].values, firth=firth)
    k0 = D.shape[1]
    idx = list(range(k0, k0 + len(xs)))
    res["status"] = "ok" if f["conv"] else "not converged"
    res["hr"] = {x: hrd(f["b"][i], np.sqrt(f["V"][i, i])) for x, i in zip(xs, idx)}
    res["events_by_kind"] = {str(k): rc(v) for k, v in evk.items()}
    aux = {"b": f["b"][idx], "V": f["V"][np.ix_(idx, idx)],
           "IF": pd.DataFrame(f["IF"][:, idx], index=f["groups"], columns=list(xs))}
    return res, aux


def wald(b, V, L=None):
    b = np.asarray(b, float)
    if L is None:
        k = len(b)
        L = np.column_stack([np.ones(k - 1), -np.eye(k - 1)])
    Lb = L @ b
    S = L @ V @ L.T
    try:
        stat = float(Lb @ np.linalg.solve(S, Lb))
    except np.linalg.LinAlgError:
        return None
    return {"chi2": stat, "df": int(L.shape[0]), "p": float(stats.chi2.sf(stat, L.shape[0]))}


def stack(auxes):
    """Joint covariance of coefficients from separate fits on (overlapping) samples via stacked influence functions."""
    idx = pd.Index([])
    for a in auxes:
        idx = idx.union(a["IF"].index)
    M = np.column_stack([a["IF"].reindex(idx).fillna(0.0).values for a in auxes])
    G = len(idx)
    b = np.concatenate([a["b"] for a in auxes])
    return b, M.T @ M * G / max(G - 1, 1)


def diff_line(b, V, i, j):
    d = b[i] - b[j]
    se = float(np.sqrt(max(V[i, i] + V[j, j] - 2 * V[i, j], 0)))
    return {"ratio_hr": float(np.exp(d)), "lo": float(np.exp(d - 1.96 * se)), "hi": float(np.exp(d + 1.96 * se)),
            "p": float(2 * stats.norm.sf(abs(d / se))) if se > 0 else None}


def evalue(hr):
    """VanderWeele-Ding E-value for an HR with a common outcome (HR -> RR conversion)."""
    if hr is None or not np.isfinite(hr) or hr <= 0:
        return None
    if hr < 1:
        hr = 1 / hr
    rr = (1 - 0.5 ** np.sqrt(hr)) / (1 - 0.5 ** np.sqrt(1 / hr))
    return float(rr + np.sqrt(rr * (rr - 1))) if rr > 1 else 1.0


def measured_pp(B, C, kind="or_treated"):
    """One interval baseline -> visit 2 for attendees with an exam outcome (BP measured or treated)."""
    v = C[(C["kind"] == "v2") & C["has_data"] & (C["t"] > GAP0 / 365.25) & C["participant_id"].isin(B["participant_id"])]
    v = v.copy()
    bp = (v["sbp2"] >= 140) | (v["dbp2"] >= 90)
    hasbp = v["sbp2"].notna() & v["dbp2"].notna()
    tr = v["med"] == 1
    if kind == "or_treated":
        v = v[hasbp | tr]
        v["ev"] = (bp | tr).astype(float).loc[v.index]
    else:  # BP only, untreated at visit 2
        v = v[hasbp & ~tr]
        v["ev"] = bp.astype(float).loc[v.index]
    v["t0"], v["t1"], v["dt"] = 0.0, v["t"], v["t"]
    return v[["participant_id", "kind", "t0", "t1", "dt", "ev"]].reset_index(drop=True)


# ============================================================================ sections

def o0_flow(ctx):
    B, C, PP, info = ctx["B"], ctx["C"], ctx["PP"], ctx["cinfo"]
    out = {"flow": ctx["flow"], "contacts": info}
    att = C[C["has_data"] & C["participant_id"].isin(B["participant_id"])]
    out["contacts_dated_on_or_before_baseline"] = rc((att["t"] <= GAP0 / 365.25).sum())
    att = att[att["t"] > GAP0 / 365.25]
    out["attended"] = {k: rc((att["kind"] == k).sum()) for k in KINDS}
    out["any_followup"] = rc(att["participant_id"].nunique())
    out["contact_years"] = {k: A.qtiles(att.loc[att["kind"] == k, "t"]) for k in KINDS}
    pe = PP.groupby("participant_id").agg(fu=("t1", "max"), ev=("ev", "max"))
    out["events"] = {"total": rc(PP["ev"].sum()), "by_kind": {k: rc(PP.loc[PP["kind"] == k, "ev"].sum()) for k in KINDS},
                     "person_years": rc(int(round(PP["dt"].sum()))), "followup_years": A.qtiles(pe["fu"])}
    c = att.merge(PP[PP["ev"] == 1][["participant_id", "kind"]], on=["participant_id", "kind"])
    comp = {"curated_any": (c["dx_any"] == 1), "self_report": (c["self"] == 1), "antihtn_drug": (c["med"] == 1),
            "icd_ba00_04": (c["icd"] == 1),
            "bp_ge_140_90_v2": (c["kind"] == "v2") & ((c["sbp2"] >= 140) | (c["dbp2"] >= 90))}
    out["event_components"] = {k: rc(v.sum()) for k, v in comp.items()}
    out["event_only_self_or_curated"] = rc(((comp["curated_any"] | comp["self_report"]) & ~comp["antihtn_drug"] &
                                            ~comp["icd_ba00_04"] & ~comp["bp_ge_140_90_v2"]).sum())
    hc = info.get("_hcat")
    if hc is not None:
        hc = hc[hc["participant_id"].isin(B["participant_id"])]
        out["curated_category_followup"] = {str(k): rc(v) for k, v in hc["cat"].value_counts().items()}
    out["ipw"] = ctx["winfo"]
    out["measured"] = {"n": rc(len(ctx["PPm"])), "events": rc(ctx["PPm"]["ev"].sum()),
                       "bp_only_n": rc(len(ctx["PPb"])), "bp_only_events": rc(ctx["PPb"]["ev"].sum()),
                       "interval_years": A.qtiles(ctx["PPm"]["dt"])}
    return out


def o1_pressure(ctx):
    return {"calibration": ctx["rcinfo"], "decomposition": ctx["dec"],
            "rc_set_used_at_risk": {str(k): rc(v) for k, v in ctx["B"]["rc_set"].value_counts().items()},
            "readings_allbp": ctx["readC"]}


def bedcol(bed):
    return "o_" + bed if bed != "pwv_raw" else "o_pwv_raw"


def o2_beds(ctx):
    B, PP = ctx["B"], ctx["PP"]
    out = {}
    for bed in BEDS:
        x = bedcol(bed)
        if x not in B.columns:
            continue
        r = {}
        for tag, cov in (("A_sitting", COV_A), ("B_usual_rc", COV_B), ("C_all_readings", COV_A + ctx["readC"])):
            r[tag] = hazard(PP, B, [x], cov)[0]
        r["B_unweighted"] = hazard(PP, B, [x], COV_B, weighted=False)[0]
        out[bed] = r
    return out


def struct_models(PP, B):
    out = {}
    out["struct_acute"] = hazard(PP, B, ["pwv_struct", "pwv_acute"], COV_B)
    out["struct_dev_hr"] = hazard(PP, B, ["pwv_struct", "acute_dev", "hr_sit"], COV_B)
    out["ref_load"] = hazard(PP, B, ["pwv_ref", "pwv_load"], COV_B)
    out["pwv_adj_acute_hr"] = hazard(PP, B, ["o_pwv"], COV_B + ["acute_dev", "hr_sit"])
    out["struct_alone"] = hazard(PP, B, ["pwv_struct"], COV_B)
    out["acute_alone"] = hazard(PP, B, ["pwv_acute"], COV_B)
    res = {}
    for k, (r, aux) in out.items():
        if aux is not None and len(aux["b"]) == 2:
            r["diff_first_minus_second"] = diff_line(aux["b"], aux["V"], 0, 1)
        res[k] = r
    return res


def o3_struct(ctx):
    return {"composite": struct_models(ctx["PP"], ctx["B"]), "measured": struct_models(ctx["PPm"], ctx["B"])}


def o4_measured(ctx):
    B, PPm, PPb = ctx["B"], ctx["PPm"], ctx["PPb"]
    out = {}
    for bed in BEDS:
        x = bedcol(bed)
        if x not in B.columns:
            continue
        out[bed] = {"A_sitting": hazard(PPm, B, [x], COV_A)[0], "B_usual_rc": hazard(PPm, B, [x], COV_B)[0],
                    "B_unweighted": hazard(PPm, B, [x], COV_B, weighted=False)[0]}
    out["bp_only_untreated"] = {bed: hazard(PPb, B, [bedcol(bed)], COV_B)[0] for bed in CORE4}
    return out


def joint_block(PP, B, bset, covs):
    xs = [bedcol(b) for b in bset]
    need = xs + covs
    e = B.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
    ppe = PP[PP["participant_id"].isin(e["participant_id"])]
    n, ev = e["participant_id"].isin(ppe["participant_id"]).sum(), int(ppe["ev"].sum())
    # v1.0.1: EPV from the rounded event count (ev / k with the exact count would reveal it)
    info = {"n": rc(n), "events": rc(ev), "epv": float(rc(ev) / (len(xs) + len(covs))) if ev >= MIN_EV() else None}
    if n < 300 or ev < MIN_EV():
        info["status"] = "too few (needs n>=300 and >=20 events)"
        return info
    info["epv_ok"] = bool(rc(ev) / (len(xs) + len(covs)) >= EPV)
    r, aux = hazard(ppe, e, xs, covs)
    info["joint"] = r
    if aux is not None:
        info["joint_wald_equal"] = wald(aux["b"], aux["V"])
    rf, _ = hazard(ppe, e, xs, covs, firth=True)
    info["joint_firth"] = rf.get("hr")
    sep = [hazard(ppe, e, [x], covs) for x in xs]
    if all(a is not None for _, a in sep):
        b, V = stack([a for _, a in sep])
        info["separate"] = {x: r_["hr"][x] for x, (r_, _) in zip(xs, sep)}
        info["separate_wald_equal"] = wald(b, V)
        info["separate_pwv_minus"] = {xs[j]: diff_line(b, V, 0, j) for j in range(1, len(xs))}
    return info


def o5_joint(ctx):
    out = {}
    for bset in JOINT_SETS:
        out["+".join(bset)] = joint_block(ctx["PP"], ctx["B"], bset, COV_B)
    out["measured_pwv+imt+abi_min"] = joint_block(ctx["PPm"], ctx["B"], JOINT_SETS[0], COV_B)
    out["sitting_pwv+imt+abi_min"] = joint_block(ctx["PP"], ctx["B"], JOINT_SETS[0], COV_A)
    return out


def topq_block(PP, B, bed, covs_list):
    x = bedcol(bed)
    e = B.replace([np.inf, -np.inf], np.nan).dropna(subset=[x, "age", "sex_male"])
    e = e[e["participant_id"].isin(PP["participant_id"])].copy()
    if len(e) < MIN_N:
        return {"n": rc(len(e)), "status": "too few"}
    a = e["age"].astype(float).values
    X = np.column_stack([np.ones(len(e)), a, (a - 50) ** 2, e["sex_male"].astype(float).values])
    y = e[x].astype(float).values
    r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    e["topq"] = (r >= np.quantile(r, 0.8)).astype(float)
    evp = PP.groupby("participant_id")["ev"].max().reindex(e["participant_id"]).fillna(0).values
    f = e["topq"].values == 1

    def min_cell(ff, ee):
        return min(int((ff & (ee == 1)).sum()), int((ff & (ee == 0)).sum()), int((~ff & (ee == 1)).sum()),
                   int((~ff & (ee == 0)).sum()))

    out = {"n": rc(len(e)), "flagged_fraction": round(float(f.mean()), 2)}  # v1.0.1: 2 decimals (no exact n)
    if min_cell(f, evp) < MIN_EV():
        out["status"] = "suppressed: a 2x2 cell (flag x event) < 20"
        return out
    out["status"] = "ok"
    for tag, cov in covs_list:
        # v1.0.1: the guard is re-applied in each model's own complete-case sample (covariate missingness can push a
        # cell below 20 after the guard on the bed/age/sex sample passed)
        miss = [c for c in cov if c not in e.columns]
        ok = e[cov].replace([np.inf, -np.inf], np.nan).notna().all(axis=1).values if not miss else np.zeros(len(e), bool)
        if min_cell(f[ok], evp[ok]) < MIN_EV():
            out[tag] = {"n": rc(ok.sum()), "status": "suppressed: 2x2 cell < 20 in model sample"}
            continue
        out[tag] = hazard(PP, e[ok], ["topq"], cov, std=False)[0]
    return out


def o6_topq(ctx):
    out = {}
    for bed in CORE4:
        out[bed] = {"composite": topq_block(ctx["PP"], ctx["B"], bed, (("A_sitting", COV_A), ("B_usual_rc", COV_B))),
                    "measured": topq_block(ctx["PPm"], ctx["B"], bed, (("B_usual_rc", COV_B),))}
    return out


def base_pred(B):
    d = pd.DataFrame({"participant_id": B["participant_id"].values}, index=B.index)
    for c in ("age", "sex_male", "sbp_sit", "dbp_sit", "bmi"):
        d[c] = B[c].astype(float)
    d["agexdbp"] = (B["age"] - B["age"].mean()) * (B["dbp_sit"] - B["dbp_sit"].mean()) / 100
    for c in ("smk_current", "fam_htn"):
        x = B[c].astype(float) if c in B.columns else pd.Series(np.nan, index=B.index)
        d[c] = x.fillna(0.0)
        if x.isna().any() and x.notna().any():
            d[c + "_miss"] = x.isna().astype(float)
    return d


def _fitp(y, X):
    b, _, p = A.logit(y, X)
    return b, p


def _metrics(y, p):
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1))
    bs, _, _ = A.logit(y, np.column_stack([np.ones(len(lp)), lp]))
    return {"auc": A.auc(y, p), "slope": float(bs[1]) if bs is not None else np.nan, "brier": float(np.mean((y - p) ** 2))}


def nb_line(y, p, n):
    out = {}
    for pt in THRESH:
        f = p >= pt
        cells = [int((f & (y == 1)).sum()), int((f & (y == 0)).sum()), int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())]
        out[f"{pt:.2f}"] = (cells[0] - cells[1] * pt / (1 - pt)) / n if min(cells) >= MIN_EV() else None
    return out


def pred_compare(d, ycol, base_cols, add_cols, nboot, seed):
    e = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[ycol] + base_cols + add_cols)
    y = e[ycol].values.astype(float)
    n, ev = len(y), int(y.sum())
    out = {"n": rc(n), "events": rc(ev), "added": add_cols}
    if n < MIN_N or ev < MIN_EV() or n - ev < MIN_EV():
        out["status"] = "too few"
        return out
    Z = e[base_cols + add_cols].astype(float)
    sd = Z.std()
    bc = [c for c in base_cols if sd[c] > 0]
    if len([c for c in add_cols if sd[c] > 0]) < len(add_cols):
        out["status"] = "no variation in added predictor"
        return out
    Z = (Z - Z.mean()) / sd.replace(0, 1)
    X0 = np.column_stack([np.ones(n), Z[bc].values])
    X1 = np.column_stack([np.ones(n), Z[bc + add_cols].values])
    rng = np.random.default_rng(seed)
    app, opt, dapp = {}, {}, []
    for tag, X in (("base", X0), ("plus", X1)):
        b, p = _fitp(y, X)
        if b is None:
            out["status"] = "logit failed"
            return out
        app[tag] = _metrics(y, p)
        app[tag]["eo"] = float(p.mean() / y.mean())
    optb = {"base": [], "plus": []}
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        if y[i].sum() < 5:
            continue
        row = {}
        for tag, X in (("base", X0), ("plus", X1)):
            b, pb = _fitp(y[i], X[i])
            if b is None:
                row = None
                break
            po = expit(X @ b)
            ab, ao = A.auc(y[i], pb), A.auc(y, po)
            mo = _metrics(y, po)
            row[tag] = (ab - ao, mo["slope"], np.mean((y[i] - pb) ** 2) - mo["brier"], ab)
        if row:
            for tag in row:
                optb[tag].append(row[tag])
            dapp.append(row["plus"][3] - row["base"][3])
    for tag in ("base", "plus"):
        o = np.array(optb[tag])
        opt[tag] = {"auc_corrected": app[tag]["auc"] - o[:, 0].mean(), "slope_corrected": float(np.nanmean(o[:, 1])),
                    "brier_corrected": app[tag]["brier"] - o[:, 2].mean()}
    out["apparent"] = app
    out["optimism_corrected"] = opt
    out["delta_auc_apparent"] = app["plus"]["auc"] - app["base"]["auc"]
    out["delta_auc_boot_ci"] = A.pct(dapp)
    out["delta_auc_corrected"] = opt["plus"]["auc_corrected"] - opt["base"]["auc_corrected"]
    # 10-fold cross-validated predictions (stratified by outcome)
    fid = np.empty(n, int)
    for cls in (0, 1):
        ix = np.flatnonzero(y == cls)
        fid[ix] = rng.permutation(len(ix)) % 10
    cv = {}
    for tag, X in (("base", X0), ("plus", X1)):
        p = np.full(n, np.nan)
        for f in range(10):
            b, _ = _fitp(y[fid != f], X[fid != f])
            if b is not None:
                p[fid == f] = expit(X[fid == f] @ b)
        ok = np.isfinite(p)
        m = _metrics(y[ok], p[ok])
        cv[tag] = {"auc": m["auc"], "slope": m["slope"], "eo": float(p[ok].mean() / y[ok].mean()),
                   "net_benefit": nb_line(y[ok], p[ok], ok.sum())}
    pr = rc(ev) / rc(n)  # v1.0.1: prevalence from the rounded counts (the exact prevalence would give exact n/events)
    cv["treat_all_net_benefit"] = {f"{pt:.2f}": float(pr - (1 - pr) * pt / (1 - pt)) for pt in THRESH}
    out["cv10"] = cv
    out["status"] = "ok"
    return out


def o7_predict(ctx, nboot, seed):
    B, PP, PPm = ctx["B"], ctx["PP"], ctx["PPm"]
    d = base_pred(B)
    for bed in CORE4:
        d[bedcol(bed)] = B[bedcol(bed)].astype(float)
    v2p = set(PPm["participant_id"]) | set(PP.loc[PP["kind"] == "v2", "participant_id"])
    ycomp = PP[PP["kind"].isin(["call1", "v2"])].groupby("participant_id")["ev"].max()
    d["y_comp"] = d["participant_id"].map(ycomp)
    d.loc[~d["participant_id"].isin(v2p), "y_comp"] = np.nan
    d["y_meas"] = d["participant_id"].map(PPm.set_index("participant_id")["ev"])
    base_cols = [c for c in d.columns if c not in ("participant_id", "y_comp", "y_meas") and not c.startswith("o_")]
    out = {"base_predictors": base_cols}
    for yc in ("y_comp", "y_meas"):
        r = {"pwv": pred_compare(d, yc, base_cols, ["o_pwv"], nboot, seed)}
        for bed in ("imt", "abi_min", "art_w"):
            r[bed] = pred_compare(d, yc, base_cols, [bedcol(bed)], nboot, seed)
        r["pwv_to_pwv_imt_abi"] = pred_compare(d, yc, base_cols + ["o_pwv"], ["o_imt", "o_abi_min"], nboot, seed)
        out[yc] = r
    return out


def cross_lagged(fr):
    w = A.wide(fr, ["pwv", "sbp_sit", "dbp_sit", "antihtn", "age", "bmi", "visit_date"])
    s0, s1 = A.MAIN_STAGES
    need = [(v, s) for v in ("pwv", "sbp_sit", "antihtn") for s in (s0, s1)]
    if not all(k in w.columns for k in need):
        return {"status": "missing columns"}
    keep = (w[("antihtn", s0)] == 0) & (w[("antihtn", s1)] == 0)
    d = pd.DataFrame({"pwv0": w[("pwv", s0)], "pwv1": w[("pwv", s1)], "sbp0": w[("sbp_sit", s0)],
                      "sbp1": w[("sbp_sit", s1)], "dbp0": w[("dbp_sit", s0)], "age0": w[("age", s0)],
                      "bmi0": w[("bmi", s0)],
                      "interval": (pd.to_datetime(w[("visit_date", s1)]) - pd.to_datetime(w[("visit_date", s0)])).dt.days / 365.25
                      })[keep.values].reset_index()
    d = d.merge(fr[["participant_id", "sex_male"]].drop_duplicates("participant_id"), how="left").dropna()
    if len(d) < MIN_N:
        return {"n": rc(len(d)), "status": "too few"}
    d["z_pwv0"], d["z_sbp0"] = A.zs(d["pwv0"]), A.zs(d["sbp0"])
    cv = ["age0", "sex_male", "bmi0", "interval"]
    f1 = A.fit(d, "sbp1", ["z_pwv0"], ["sbp0", "dbp0"] + cv, std=False)
    f2 = A.fit(d, "pwv1", ["z_sbp0"], ["pwv0"] + cv, std=False)
    return {"n": rc(len(d)), "sbp_v2_mmHg_per_sd_pwv_v1": A.coef(f1, "z_pwv0"),
            "pwv_v2_m_s_per_sd_sbp_v1": A.coef(f2, "z_sbp0"),
            "sd_pwv_v1": float(d["pwv0"].std()), "sd_sbp_v1": float(d["sbp0"].std())}


def o8_sens(ctx):
    B, PP, C, W = ctx["B"], ctx["PP"], ctx["C"], ctx["W"]
    out = {}
    # ACC/AHA 130/80 cohort (own weights)
    B130, fl = make_cohort(ctx["fr"], ctx["base_dates"], ctx["prev"], ctx["fam"], thr=(130, 80))
    B130 = B130.merge(ctx["U"], on="participant_id", how="left")
    PP130 = person_period(B130, C, lambda c: ev_composite(c, (130, 80)))
    W130, _ = ipw(B130, C, PP130)
    PP130 = attach_w(PP130, W130)
    out["acc_aha_130_80"] = {"flow": fl, "A_sitting": hazard(PP130, B130, ["o_pwv"], COV_A)[0],
                             "B_usual_rc": hazard(PP130, B130, ["o_pwv"], COV_B)[0]}
    for tag, fn in (("objective_only", ev_objective), ("strict_diagnosis", ev_strict)):
        P2 = attach_w(person_period(B, C, fn), W)
        out[tag] = hazard(P2, B, ["o_pwv"], COV_B)[0]
    abi_ok = (A.abi_clin(B) >= 0.9) & (B["abi_max"] <= 1.4) if "abi_max" in B.columns else pd.Series(True, index=B.index)
    out["abi_0.9_1.4"] = hazard(PP, B[abi_ok.values], ["o_pwv"], COV_B)[0]
    out["plus_height"] = hazard(PP, B, ["o_pwv"], COV_B + ["height"])[0]
    for leg in ("pwv_r", "pwv_l"):
        if leg in B.columns:
            out[leg] = hazard(PP, B, [leg], COV_B)[0]
    # PWV x baseline SBP >= 120
    Bi = B.copy()
    Bi["hi_sbp"] = (Bi["sbp_sit"] >= 120).astype(float)
    zp = A.zs(Bi["o_pwv"])
    Bi["zp"], Bi["zp_x_hi"] = zp, zp * Bi["hi_sbp"]
    # v1.0.1: the interaction needs >= 20 events (and non-events) in each SBP group of the model sample
    cvi = [c for c in COV_B if c in Bi.columns]
    di = PP.merge(Bi[["participant_id", "zp", "hi_sbp"] + cvi], on="participant_id").dropna(subset=["zp", "hi_sbp"] + cvi)
    pe = di.groupby("participant_id").agg(ev=("ev", "max"), hi=("hi_sbp", "first"))
    gmin = min([int(((pe["hi"] == h) & (pe["ev"] == e)).sum()) for h in (0, 1) for e in (0, 1)])
    r = (hazard(PP, Bi, ["zp", "zp_x_hi"], COV_B + ["hi_sbp"], std=False)[0] if gmin >= MIN_EV()
         else {"status": "suppressed: SBP group x event cell < 20"})
    out["pwv_x_sbp_ge_120"] = {"model": r, "hr_low": hazard(PP, Bi[Bi["hi_sbp"] == 0], ["o_pwv"], COV_B)[0],
                               "hr_high": hazard(PP, Bi[Bi["hi_sbp"] == 1], ["o_pwv"], COV_B)[0]}
    # negative-control exposure: birth month (sin, cos)
    if "bm_sin" in B.columns:
        r, aux = hazard(PP, B, ["bm_sin", "bm_cos"], COV_B, std=False)
        out["neg_control_birth_month"] = {"n": r.get("n"), "events": r.get("events"),
                                          "wald": wald(aux["b"], aux["V"], np.eye(2)) if aux is not None else None}
    # corrected vs raw PWV (same sample, stacked)
    e = B.dropna(subset=["o_pwv", "o_pwv_raw"]) if "o_pwv_raw" in B.columns else None
    if e is not None and len(e) >= MIN_N:
        r1, a1 = hazard(PP, e, ["o_pwv"], COV_B)
        r2, a2 = hazard(PP, e, ["o_pwv_raw"], COV_B)
        if a1 is not None and a2 is not None:
            b, V = stack([a1, a2])
            out["corrected_vs_raw"] = {"corrected": r1["hr"]["o_pwv"], "raw": r2["hr"]["o_pwv_raw"], "n": r1["n"],
                                       "events": r1["events"], "ratio": diff_line(b, V, 0, 1)}
    # within-visit calibration (target = second sitting reading at baseline)
    Uw = ctx.get("Uw")
    if Uw is not None:
        Bw = B.drop(columns=["u_sbp", "u_dbp"]).merge(Uw[["participant_id", "u_sbp", "u_dbp"]], on="participant_id",
                                                       how="left")
        out["within_visit_calibration"] = hazard(PP, Bw, ["o_pwv"], COV_B)[0]
        out["within_visit_calibration_info"] = ctx.get("Uwinfo")
    out["cross_lagged_untreated"] = cross_lagged(ctx["fr"])
    prim = g(ctx["res"], "O2_composite", "pwv", "B_usual_rc", "hr", "o_pwv")
    if prim:
        lo_, hi_ = prim["lo"], prim["hi"]  # v1.0.1: CI limit closer to 1 on either side
        out["evalue_pwv_B"] = {"point": evalue(prim["hr"]),
                               "ci_limit": evalue(lo_) if lo_ > 1 else evalue(hi_) if hi_ < 1 else 1.0}
    return out


# ============================================================================ pipeline

def _guard(label, fn, fallback):
    """v1.0.1: a failing preparation step records its error and returns a neutral fallback instead of a fatal run."""
    try:
        return fn()
    except Exception as ex:
        e_ = err(ex)
        log(f"{label} FAILED: {e_}")
        return fallback(e_)


def prepare(fr, C, base_dates, prev, fam, cinfo, seed):
    def no_u(e_):
        base = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id")
        return (pd.DataFrame({"participant_id": base["participant_id"].values, "u_sbp": np.nan, "u_dbp": np.nan,
                              "rc_set": None}), {"error": e_, "sets": {}})

    U, rcinfo = _guard("usual BP calibration", lambda: usual_bp(fr, seed), no_u)
    if U is None:
        U, rcinfo = no_u(rcinfo.get("error", "no calibration"))
    Uw, Uwinfo = _guard("within-visit calibration", lambda: usual_bp(fr, seed + 1, within=True), lambda e_: (None, {"error": e_}))
    B, flow = make_cohort(fr, base_dates, prev, fam)
    B = B.merge(U, on="participant_id", how="left")
    dec = _guard("PWV decomposition", lambda: decompose(B), lambda e_: {"error": e_})
    PP = person_period(B, C, ev_composite)
    PPm = measured_pp(B, C, "or_treated")
    W, winfo = _guard("follow-up weights", lambda: ipw(B, C, PP, PPm),
                      lambda e_: (pd.DataFrame({"participant_id": B["participant_id"].values,
                                                **{"w_" + k: 1.0 for k in KINDS + ["v2m"]}}), {"error": e_}))
    PP = attach_w(PP, W)
    PPm = attach_w(PPm, W, "w_v2m")  # v1.0.1: own attendance model for the measured exam (was w_v2)
    PPb = attach_w(measured_pp(B, C, "bp_only"), W, "w_v2m")
    readC = [c for c in ALL_READ if c in B.columns and B[c].notna().mean() >= 0.8]
    return {"fr": fr, "C": C, "B": B, "PP": PP, "PPm": PPm, "PPb": PPb, "W": W, "winfo": winfo, "U": U, "Uw": Uw,
            "Uwinfo": Uwinfo, "rcinfo": rcinfo, "dec": dec, "flow": flow, "cinfo": cinfo, "base_dates": base_dates,
            "prev": prev, "fam": fam, "readC": readC}


SECTIONS = ("O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8")


def run_all(ctx, sections, boot, seed):
    res = ctx.setdefault("res", {})
    fns = {"O0": ("O0_flow", lambda: o0_flow(ctx)), "O1": ("O1_pressure", lambda: o1_pressure(ctx)),
           "O2": ("O2_composite", lambda: o2_beds(ctx)), "O3": ("O3_structural", lambda: o3_struct(ctx)),
           "O4": ("O4_measured", lambda: o4_measured(ctx)), "O5": ("O5_joint", lambda: o5_joint(ctx)),
           "O6": ("O6_top_quintile", lambda: o6_topq(ctx)), "O7": ("O7_prediction", lambda: o7_predict(ctx, boot, seed)),
           "O8": ("O8_sensitivity", lambda: o8_sens(ctx))}
    for key in sections:
        name, fn = fns[key]
        t1 = time.time()
        try:
            res[name] = fn()
            log(f"{name} done in {time.time() - t1:.1f}s")
        except Exception as ex:  # keep the other sections
            res[name] = {"error": err(ex)}
            log(f"{name} FAILED: {res[name]['error']}")
        res.setdefault("timings", {})[name] = time.time() - t1
    return res


# ============================================================================ digest

def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def hs(r, x=None):
    """'HR [lo,hi]' from a hazard summary (first x when x is None)."""
    if not isinstance(r, dict):
        return "NA"
    if r.get("status") not in ("ok", "not converged"):
        return f"NA({r.get('status', 'NA')})"
    h = r.get("hr") or {}
    if x is None:
        x = next(iter(h), None)
    c = h.get(x)
    if not c:
        return "NA"
    s = f"{v3(c['hr'])} [{v3(c['lo'])},{v3(c['hi'])}]"
    return s + ("*nc" if r.get("status") == "not converged" else "")


SHORT = {"o_pwv": "pwv", "o_imt": "imt", "o_art_w": "ret", "o_abi_min": "abi"}


def ne(r):
    return f"n {g(r, 'n')} ev {g(r, 'events')}" if isinstance(r, dict) else "n NA"


def hc(c):
    return "NA" if not c else f"{v3(c.get('hr', c.get('ratio_hr')))} [{v3(c.get('lo'))},{v3(c.get('hi'))}]"


def pv(w):
    return "NA" if not isinstance(w, dict) else f"p {v3(w.get('p'))}"


def qs(q):
    return "NA" if not isinstance(q, dict) else f"{v3(q.get('q05'))}/{v3(q.get('q50'))}/{v3(q.get('q95'))}"


def kv(d, f=str):
    it = d.items() if isinstance(d, dict) else (d or [])
    return ", ".join(f"{k} {f(v)}" for k, v in it)


def digest(R):
    L = [f"F1 OUTCOMES DIGEST {R.get('version')} | atlas {R.get('atlas_version')} | seed {R.get('seed')} boot "
         f"{R.get('boot')} | runtime {v3(R.get('runtime_s'))}s",
         "HR per SD adverse-oriented bed (ABI, retinal artery width: per SD lower); cloglog discrete-time hazard, contact "
         "intercepts, log(interval) offset, follow-up IPW, robust SE",
         "covariates A = age sex BMI sitting SBP/DBP; B = age sex BMI calibrated usual SBP/DBP; C = A + all baseline "
         "readings; counts rounded to 10, None = <20; [CI] 95 %"]
    if R.get("fatal"):
        L.append(f"FATAL {R['fatal']}")
    o0 = R.get("O0_flow") or {}
    L.append("## O0 flow")
    if "error" in o0:
        L.append(f"error {o0['error']}")
    L.append("cohort: " + kv(o0.get("flow")))
    ct = o0.get("contacts") or {}
    L.append(f"attended contacts {kv(o0.get('attended'))}; participants with any follow-up {o0.get('any_followup')}; "
             f"contacts dated <= baseline {o0.get('contacts_dated_on_or_before_baseline')}")
    L.append(f"stage rows without outcome data {kv(ct.get('stage_rows_without_outcome_data'))}; dates imputed "
             f"{kv(ct.get('dates_imputed'))}; undefined-stage rows mapped {kv(ct.get('undefined_rows_mapped'))}")
    L.append("contact years q05/q50/q95: " + kv(o0.get("contact_years"), qs) +
             f"; nominal {kv(ct.get('nominal_years'), v3)}")
    e = o0.get("events") or {}
    L.append(f"composite events {e.get('total')} by contact {kv(e.get('by_kind'))}; person-years {e.get('person_years')}; "
             f"follow-up y {qs(e.get('followup_years'))}")
    L.append(f"event components (not exclusive): {kv(o0.get('event_components'))}; self/curated only "
             f"{o0.get('event_only_self_or_curated')}")
    L.append(f"curated category at follow-up (at risk): {kv(o0.get('curated_category_followup'))}")
    for k, v in (o0.get("ipw") or {}).items():
        if not isinstance(v, dict):
            L.append(f"ipw {k}: {v}")
            continue
        due = (f"due {v.get('due')} (+{v.get('history_terms')} history terms)" if k == "v2_measured"
               else f"due&event-free {v.get('due_event_free')}")
        L.append(f"ipw {k}: {due} attended {v.get('attended')} (all {v.get('attended_all')})"
                 f" auc {v3(v.get('auc_attendance'))} w q05/q50/q95 {qs(v.get('weights_attended'))} {v.get('status', '')}".rstrip())
    m = o0.get("measured") or {}
    L.append(f"measured v2 exam (BP>=140/90 or drug): n {m.get('n')} ev {m.get('events')}; BP-only untreated n "
             f"{m.get('bp_only_n')} ev {m.get('bp_only_events')}; interval y {qs(m.get('interval_years'))}")
    o1 = R.get("O1_pressure") or {}
    L.append("## O1 usual BP and PWV decomposition")
    if "error" in o1:
        L.append(f"error {o1['error']}")
    cal = o1.get("calibration") or {}
    if cal.get("error"):
        L.append(f"calibration error {cal['error']}")
    sets_ = cal.get("sets") or {}
    for k, v in sorted(sets_.items(), key=lambda kv_: ["full", "dev", "nodev", "min"].index(kv_[0])
                       if kv_[0] in ("full", "dev", "nodev", "min") else 9):
        L.append(f"calibration {k}: n {v.get('n_calibration', v.get('n'))} cv r2 sbp {v3(v.get('u_sbp_r2_cv'))} dbp "
                 f"{v3(v.get('u_dbp_r2_cv'))} coef sit {v3(v.get('coef_sbp_sit'))} dev {v3(v.get('coef_dev_bsbp'))} "
                 f"assigned {v.get('n_assigned_baseline')} {v.get('status', '')}".rstrip())
    L.append(f"optional readings {','.join(cal.get('optional_readings') or [])}; set used at risk "
             f"{kv(o1.get('rc_set_used_at_risk'))}; model C adds {','.join(o1.get('readings_allbp') or [])}")
    dc = o1.get("decomposition") or {}
    cm = dc.get("coef_m_s") or {}
    if dc.get("error") or dc.get("status"):
        L.append(f"pwv decomposition {dc.get('error') or dc.get('status')}")
    L.append(f"pwv decomposition n {dc.get('n')} r2 {v3(dc.get('r2'))}: " +
             "; ".join(f"{k} {v3(c.get('b'))} [{v3(c.get('lo'))},{v3(c.get('hi'))}]" for k, c in cm.items()))
    L.append(f"var share {kv(dc.get('var_share'), v3)}; corr with pwv {kv(dc.get('corr_with_pwv'), v3)}; corr struct-usual "
             f"{v3(dc.get('corr_struct_usual_sbp'))}")
    L.append(f"acute dev mmHg q05/q50/q95 {qs(dc.get('acute_dev'))}; pwv@100mmHg q05/q50/q95 {qs(dc.get('pwv_ref'))} r with "
             f"pwv {v3(dc.get('corr_pwv_ref_pwv'))}")
    o2 = R.get("O2_composite") or {}
    L.append("## O2 composite incident hypertension per bed: n ev | A [CI] | B [CI] | C [CI] | B unweighted")
    if "error" in o2:
        L.append(f"error {o2['error']}")
    for bed, r in o2.items():
        if not isinstance(r, dict) or "B_usual_rc" not in r:
            continue
        L.append(f"{bed}: {ne(r['B_usual_rc'])} py {g(r, 'B_usual_rc', 'person_years')} | {hs(r['A_sitting'])} | "
                 f"{hs(r['B_usual_rc'])} | {hs(r['C_all_readings'])} ({ne(r['C_all_readings'])}) | {hs(r['B_unweighted'])}"
                 f" | by contact {kv(g(r, 'B_usual_rc', 'events_by_kind'))}")
    o3 = R.get("O3_structural") or {}
    L.append("## O3 structural vs acute PWV (covariates B): HR per SD; diff = HR ratio first/second")
    if "error" in o3:
        L.append(f"error {o3['error']}")
    for out_ in ("composite", "measured"):
        s = o3.get(out_) or {}
        a = s.get("struct_acute") or {}
        L.append(f"{out_}: {ne(a)} struct {hs(a, 'pwv_struct')} acute {hs(a, 'pwv_acute')} diff {hc(a.get('diff_first_minus_second'))} "
                 f"{pv(a.get('diff_first_minus_second'))}; alone struct {hs(s.get('struct_alone'))} acute {hs(s.get('acute_alone'))}")
        b = s.get("struct_dev_hr") or {}
        r_ = s.get("ref_load") or {}
        L.append(f"{out_}: struct {hs(b, 'pwv_struct')} acute dev {hs(b, 'acute_dev')} hr {hs(b, 'hr_sit')} | pwv@100mmHg "
                 f"{hs(r_, 'pwv_ref')} load {hs(r_, 'pwv_load')} diff {hc(r_.get('diff_first_minus_second'))} | pwv | B+acute+HR "
                 f"{hs(s.get('pwv_adj_acute_hr'))}")
    o4 = R.get("O4_measured") or {}
    L.append("## O4 measured v2 exam (co-primary; IPW w_v2m for the measured v2 sample): n ev | A | B | B unweighted; BP-only (untreated at v2) B")
    if "error" in o4:
        L.append(f"error {o4['error']}")
    for bed in BEDS:
        r = o4.get(bed)
        if not isinstance(r, dict):
            continue
        bo = g(o4, "bp_only_untreated", bed)
        L.append(f"{bed}: {ne(r['B_usual_rc'])} | {hs(r['A_sitting'])} | {hs(r['B_usual_rc'])} | {hs(r['B_unweighted'])}"
                 + (f" | bp-only {ne(bo)} {hs(bo)}" if bo else ""))
    o5 = R.get("O5_joint") or {}
    L.append("## O5 joint models (B; 'sitting' = A): joint = mutually adjusted, Wald equal log-HR; firth = penalised; separate = "
             "one bed per model in the common sample, stacked IF; pwv/x = HR ratio; ret = retinal artery, abi = ABI")
    if "error" in o5:
        L.append(f"error {o5['error']}")
    for k, v in o5.items():
        if not isinstance(v, dict):
            continue
        k = k.replace("abi_min", "abi").replace("art_w", "ret")
        if v.get("status", "").startswith("too few"):
            L.append(f"{k}: n {v.get('n')} ev {v.get('events')} too few")
            continue
        j = v.get("joint") or {}
        xs = list((j.get("hr") or {}).keys())
        L.append(f"{k} joint: n {v.get('n')} ev {v.get('events')} epv {v3(v.get('epv'))} | " +
                 " ".join(f"{SHORT.get(x, x)} {hs(j, x)}" for x in xs) + f" | equal {pv(v.get('joint_wald_equal'))}")
        L.append(f"{k} firth: " + " ".join(f"{SHORT.get(x, x)} {hc((v.get('joint_firth') or {}).get(x))}" for x in xs))
        if v.get("separate"):
            L.append(f"{k} separate: " + " ".join(f"{SHORT.get(x, x)} {hc(c)}" for x, c in v["separate"].items()) +
                     f" | equal {pv(v.get('separate_wald_equal'))}")
            L.append(f"{k} paired HR ratio: " + "; ".join(f"pwv/{SHORT.get(x, x)} {hc(c)} {pv(c)}"
                                                         for x, c in (v.get("separate_pwv_minus") or {}).items()))
    o6 = R.get("O6_top_quintile") or {}
    L.append("## O6 BP-adjusted top quintile (age/sex-standardised): bed | composite A | B | measured B")
    if "error" in o6:
        L.append(f"error {o6['error']}")
    for bed, v in o6.items():
        if not isinstance(v, dict):
            continue
        c, m_ = v.get("composite") or {}, v.get("measured") or {}
        L.append(f"{bed}: n {c.get('n')} frac {v3(c.get('flagged_fraction'))} {c.get('status')} | {hs(c.get('A_sitting'))} | "
                 f"{hs(c.get('B_usual_rc'))} | measured {m_.get('status')} {hs(m_.get('B_usual_rc'))}")
    o7 = R.get("O7_prediction") or {}
    L.append("## O7 event by v2 (v2 attendees; logistic, Framingham-type base): n ev | AUC app/opt/cv base -> +bed | dAUC opt "
             "[boot CI app] | slope opt | E/O cv | NB cv @.05/.10/.20 base/+bed/all")
    if "error" in o7:
        L.append(f"error {o7['error']}")
    for yc in ("y_comp", "y_meas"):
        for k, v in (o7.get(yc) or {}).items():
            if not isinstance(v, dict):
                continue
            if v.get("status") != "ok":
                L.append(f"{yc} {k}: n {v.get('n')} ev {v.get('events')} {v.get('status')}")
                continue
            ap, op, cv = v["apparent"], v["optimism_corrected"], v["cv10"]
            nb = "; ".join("/".join(v3(g(cv, t, "net_benefit", th)) if t != "all" else v3(g(cv, "treat_all_net_benefit", th))
                                    for t in ("base", "plus", "all")) for th in (f"{p:.2f}" for p in THRESH))
            L.append(f"{yc} {k}: n {v['n']} ev {v['events']} | {v3(ap['base']['auc'])}/{v3(op['base']['auc_corrected'])}/"
                     f"{v3(cv['base']['auc'])} -> {v3(ap['plus']['auc'])}/{v3(op['plus']['auc_corrected'])}/{v3(cv['plus']['auc'])}"
                     f" | {v3(v['delta_auc_corrected'])} {A.ci3(v['delta_auc_boot_ci'])} | {v3(op['base']['slope_corrected'])}/"
                     f"{v3(op['plus']['slope_corrected'])} | {v3(cv['base']['eo'])}/{v3(cv['plus']['eo'])} | {nb}")
    o8 = R.get("O8_sensitivity") or {}
    L.append("## O8 sensitivity (PWV, covariates B unless named)")
    if "error" in o8:
        L.append(f"error {o8['error']}")
    a = o8.get("acc_aha_130_80") or {}
    L.append(f"ACC/AHA 130/80: at risk {kv(a.get('flow'))}")
    L.append(f"ACC/AHA 130/80: {ne(a.get('B_usual_rc'))} A {hs(a.get('A_sitting'))} B {hs(a.get('B_usual_rc'))}")
    for ks in (("objective_only", "strict_diagnosis", "abi_0.9_1.4"), ("plus_height", "pwv_r", "pwv_l")):
        L.append("; ".join(f"{k} {ne(o8.get(k))} {hs(o8.get(k))}" for k in ks if k in o8))
    ix = o8.get("pwv_x_sbp_ge_120") or {}
    L.append(f"pwv x SBP>=120: interaction {hs(ix.get('model'), 'zp_x_hi')}; SBP<120 {ne(ix.get('hr_low'))} "
             f"{hs(ix.get('hr_low'))}; SBP>=120 {ne(ix.get('hr_high'))} {hs(ix.get('hr_high'))}")
    nc = o8.get("neg_control_birth_month") or {}
    cr = o8.get("corrected_vs_raw") or {}
    L.append(f"neg control birth month: n {nc.get('n')} ev {nc.get('events')} 2-df {pv(nc.get('wald'))} | corrected "
             f"{hc(cr.get('corrected'))} raw {hc(cr.get('raw'))} ratio {hc(cr.get('ratio'))} {pv(cr.get('ratio'))}")
    wv = o8.get("within_visit_calibration_info") or {}
    L.append(f"within-visit calibration: {ne(o8.get('within_visit_calibration'))} {hs(o8.get('within_visit_calibration'))}; "
             + "; ".join(f"{k} n {s.get('n_calibration', s.get('n'))} r2 {v3(s.get('u_sbp_r2_cv'))}"
                         for k, s in (wv.get("sets") or {}).items()))
    cl = o8.get("cross_lagged_untreated") or {}
    L.append(f"cross-lagged untreated both visits n {cl.get('n')}: SBP v2 per SD PWV v1 {A.cf(cl.get('sbp_v2_mmHg_per_sd_pwv_v1'))} "
             f"mmHg; PWV v2 per SD SBP v1 {A.cf(cl.get('pwv_v2_m_s_per_sd_sbp_v1'))} m/s {cl.get('status', '')}".rstrip())
    ev_ = o8.get("evalue_pwv_B") or {}
    L.append(f"E-value composite pwv B: point {v3(ev_.get('point'))} CI limit {v3(ev_.get('ci_limit'))}")
    L.append(f"timings s: {kv(R.get('timings'), v3)}")
    out = []
    for ln in L:  # no leading spaces; lines over 220 characters continue on the next line after '+ ' (break at a space)
        ln = ln.strip()
        while len(ln) > 220:
            cut = ln.rfind(" ", 100, 220)
            cut = 220 if cut < 0 else cut
            out.append(ln[:cut].rstrip())
            ln = "+ " + ln[cut:].strip()
        out.append(ln)
    if len(out) > 120:
        out = out[:119] + [f"DIGEST TRUNCATED: {len(out)} lines; see results.json"]
    return "\n".join(out) + "\n"


# ============================================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", default=None, help="atlas script (default f1_atlas_v1_5.py beside this file)")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--boot", type=int, default=200, help="bootstrap replicates for prediction optimism")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    load_atlas(a.atlas)
    t0 = time.time()
    log(f"{VERSION}; atlas {A.VERSION}; numpy {np.__version__}; pandas {pd.__version__}; python {sys.version.split()[0]}")
    secs = [s.strip() for s in a.sections.split(",") if s.strip()]
    fr, _, _ = A.build_frame(a.root, None)
    fr = fr.reset_index(drop=True)
    pids = set(fr["participant_id"].astype(str))
    res = {"version": VERSION, "atlas_version": A.VERSION, "seed": f"{a.seed:,}", "boot": a.boot, "sections": secs,
           "atlas_is_v1_5": "v1.5" in A.VERSION}
    if not res["atlas_is_v1_5"]:
        log(f"WARNING atlas version {A.VERSION} is not v1.5")
    try:
        C, base_dates, prev, cinfo = load_contacts(a.root, fr)
        pids |= set(C["participant_id"].astype(str))
        fam = load_fam(a.root)
        ctx = prepare(fr, C, base_dates, prev, fam, cinfo, a.seed)
        ctx["res"] = res
        log(f"cohort {A.cnt(len(ctx['B']))} at risk; composite events {A.cnt(ctx['PP']['ev'].sum())}; measured "
            f"{A.cnt(len(ctx['PPm']))}")
        run_all(ctx, secs, a.boot, a.seed)
    except Exception as ex:
        res["fatal"] = err(ex)
        log(f"FATAL {res['fatal']}")
    res["runtime_s"] = time.time() - t0
    text = json.dumps(clean(res), ensure_ascii=False, sort_keys=True, indent=1)
    privacy_check(text, pids)
    try:
        dg = digest(json.loads(text))
    except Exception as ex:  # v1.0.1: a digest failure must not lose results.json
        dg = f"F1 OUTCOMES DIGEST {VERSION} | atlas {A.VERSION}\nDIGEST FAILED: {err(ex)}; see results.json\n"
        log(f"digest FAILED: {err(ex)}")
    privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines; total {time.time() - t0:.1f}s")
    logtext = "\n".join(A.LOG) + "\n"
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
