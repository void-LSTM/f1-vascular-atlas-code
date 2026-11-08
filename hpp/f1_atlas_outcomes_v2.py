#!/usr/bin/env python3
"""F1 atlas outcomes module v2 (package P10 'f1_atlas_outcomes_v2'; HPP sandbox; writes aggregates only).

Outcome blocks per bed that the
HPP outcome scout (results/derived/HPP_OUTCOME_SCOUT_20261005.md) found feasible.  Beds are the atlas v1.5 corrected beds
(calendar quarter, clinic day, within-day session; raw kept as <bed>_raw), built by f1_atlas_v1_5.py through
f1_outcomes_v1.py (P1 v1.0.1); both are imported by path and never modified.  Liver tables are never read.

M0  inventory (counts only): stages in events; curated diabetes / CKD / IHD / hyperlipidaemia categories by stage; ICD-11
    target groups and ATC classes (participants) by stage; follow-up medical survey rows and hospitalisation by stage;
    lab pairs over the two visits.
E   incident events per bed, objective definitions.  P1 discrete-time hazard (complementary log-log over follow-up
    contacts, contact-kind intercepts, log(interval) offset, stabilised inverse-probability-of-follow-up weights per
    contact kind refitted for each outcome's own event history, robust SE clustered by participant).
    Outcomes (cohort = baseline participants free of the outcome):
      htn_obj  P1 at-risk cohort; antihypertensive drug, ICD-11 BA00-BA04/BA0Z, visit-2 sitting BP >= 140/90
      ahtn     antihypertensive start (C02/C03/C07/C08/C09); untreated at baseline
      statin   statin start (C10AA); no C10AA at baseline
      dm       strict diabetes: ICD-11 5A11/5A14, glucose-lowering drug other than metformin (A10BA) and GLP-1 RA (A10BJ),
               visit-2 HbA1c >= 6.5 % or glucose >= 126 mg/dL; no A10, no ICD, no curated diabetes, labs below at baseline
      ascvd    ICD-11 BA40-BA6Z, 8B10-8B2Z, BD40, or nitrate (C01DA) start; none at baseline, no curated IHD
      ckd      ICD-11 GB61 or visit-2 eGFR < 60; baseline eGFR >= 60 and no GB61
      nc       negative-control outcome: ICD-11 CA08 (vasomotor/allergic rhinitis); none at baseline
    Per bed (pwv, abi_min, imt, art_w, vein_w, cavi0, sokolow; 3-bed composite c3 = mean age/sex-standardised adverse z of
    pwv, imt, abi_min; sensitivities raw PWV and IMT without the clinic-day step): HR per SD of the adverse-oriented bed,
    covariates A (age, sex), B (A + BMI + calibrated usual SBP/DBP = P1 set B), X (B + outcome extras), B unweighted;
    adverse-bed count (0 / 1 / >=2 of the three core beds in the age/sex-standardised top decile).
    Multi-bed comparison: per outcome, common sample, separate fits stacked by influence functions (P1 O5): ratio of
    HRs per bed pair and Wald test of equal log-HR (pwv, imt, abi_min); pairs with retina, Sokolow and CAVI0.
    Vascular-age gaps (G) as predictors: HR per reference SD of the gap, and gap >= 1 reference SD vs < 1.
P   progression over the two visits (00 -> 02): pairs, mean change (corrected and raw), retest ICC, season-matched
    change, stage effect within calendar quarter, Sokolow calendar profile; ANCOVA (visit-2 value on baseline value,
    interval, age, sex) and change-score models of baseline predictors (visit-2 attendance IPW), baseline-SD units, BH q.
G   bed-specific vascular-age gaps: reference subset (healthy at baseline), per sex restricted cubic spline in age;
    gap z = adverse (observed - expected) / reference SD; gap years for PWV and IMT; reference SD, slope, centiles.
    Gaps as predictors of organ decline (visit-2 SBP untreated at both visits, Sokolow, eGFR, PWV, IMT; ANCOVA, organ
    baseline-SD units per reference SD of the gap) with paired contrasts between gap beds; lagged cross-bed design
    (baseline vs concurrent visit-2 gap for retina and IMT first measured at visit 2).
Privacy: aggregates only; counts rounded to 10, cells < 20 suppressed (None; events too); every HR needs n >= 200 and
>= 20 events (group x event person cells >= 20 in the model sample for binary predictors and the adverse-bed count;
v2.0.1), every regression n >= 200; quantiles 5-95 % only; no
participant ids; no dates (calendar years only, on the Sokolow line); exception text scrubbed; every output is scanned for
7+ digit runs, participant ids and dates before it is written; refuses to overwrite --out.
Run: python f1_atlas_outcomes_v2.py --root <HPP_DATA_ROOT> --out f1_atlas_outcomes_out_v2
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

VERSION = "f1_atlas_outcomes_v2.0.1 (2026Q4 build)"
HERE = os.path.dirname(os.path.abspath(__file__))
O = None  # f1_outcomes_v1 module (load_modules)
A = None  # atlas module (O.A)
EXPECT = {"atlas": "62344144b83db3f2", "outcomes": "9bee14ddca223c57"}
BASE, V2 = "00_00_visit", "02_00_visit"
STAGES = ["00_00_visit", "01_00_call", "02_00_visit", "03_00_call"]
SLAB = {"00_00_visit": "b", "01_00_call": "c1", "02_00_visit": "v2", "03_00_call": "c3"}
BEDS = ["pwv", "abi_min", "imt", "art_w", "vein_w", "cavi0", "sokolow"]
CORE3 = ["pwv", "imt", "abi_min"]
XBEDS = BEDS + ["c3", "pwv_raw", "imt_nod"]
XPAIRS = [("pwv", "art_w"), ("imt", "art_w"), ("pwv", "vein_w"), ("pwv", "sokolow"), ("pwv", "cavi0")]
SH = {"pwv": "pwv", "abi_min": "abi", "imt": "imt", "art_w": "ret", "vein_w": "ven", "cavi0": "cav", "sokolow": "sok",
      "c3": "c3", "pwv_raw": "pwvR", "imt_nod": "imtD", "sbp_sit": "sbp", "egfr": "egfr"}
GAP_BEDS = ["pwv", "imt", "abi_min", "art_w", "vein_w", "cavi0", "sokolow"]
GAP_YEARS = ("pwv", "imt")
COV_A = ["age", "sex_male"]
MED = {"antihtn": ("C02", "C03", "C07", "C08", "C09"), "statin": ("C10AA",), "lipid": ("C10",),
       "dmany": ("A10",), "dmdrug": ("A10A", "A10B"), "nitrate": ("C01DA",), "metformin": ("A10BA",), "glp1": ("A10BJ",)}
DM_EXCL = ("A10BA", "A10BJ")  # metformin (prediabetes) and GLP-1 RA (weight) are not diabetes-defining
ICD = {"htn": ("BA00", "BA01", "BA02", "BA03", "BA04", "BA0Z"), "dm": ("5A11", "5A14"), "ihd": ("BA4", "BA5", "BA6"),
       "cereb": ("8B1", "8B2"), "pad": ("BD40",), "ckd": ("GB61",), "svt_af": ("BC81",), "hf": ("BD1",), "nc": ("CA08",)}
OUTCOMES = {
    "htn_obj": {"label": "HTN objective (drug, ICD BA00-BA04/BA0Z, v2 BP>=140/90; P1 cohort)", "extras": ["fam_htn", "smk_current"]},
    "ahtn": {"label": "antihypertensive start (untreated)", "extras": ["fam_htn", "smk_current"]},
    "statin": {"label": "statin C10AA start", "extras": ["ldl", "hdl", "log_tg"]},
    "dm": {"label": "strict diabetes (ICD 5A11/5A14, drug excl. metformin/GLP-1 RA, v2 HbA1c/glucose)", "extras": ["hba1c", "glucose"]},
    "ascvd": {"label": "ASCVD (ICD BA40-BA6Z, 8B1-8B2, BD40, nitrate start)", "extras": ["ldl", "smk_current", "diabetes_any"]},
    "ckd": {"label": "CKD (ICD GB61 or v2 eGFR<60; baseline eGFR>=60)", "extras": ["egfr", "diabetes_any"]},
    "nc": {"label": "negative-control outcome ICD CA08", "extras": ["smk_current"]},
}
PRIMARY_OC = ("htn_obj", "statin", "ahtn")
PRED = [("age", "age", False), ("sex", "sex_male", True), ("bmi", "bmi", False), ("waist", "waist", False),
        ("sbp", "sbp_sit", False), ("dbp", "dbp_sit", False), ("hr", "hr_sit", False), ("ldl", "ldl", False),
        ("hdl", "hdl", False), ("tg", "log_tg", False), ("a1c", "hba1c", False), ("egfr", "egfr", False),
        ("smk", "smk_current", True), ("ahtn", "antihtn", True), ("stat", "statin", True), ("ahi", "log1p_ahi", False),
        ("vat", "log_vat", False)]
PROG = [("pwv", "pwv_raw"), ("imt", "imt_raw"), ("abi_min", "abi_min_raw"), ("cavi0", "cavi0_raw"), ("sokolow", "sok_cal"),
        ("art_w", "art_w_raw"), ("vein_w", None), ("sbp_sit", None)]
PROG_PRIO = {"pwv": 1, "imt": 1, "sokolow": 1, "sbp_sit": 1, "abi_min": 2, "cavi0": 2, "art_w": 2, "vein_w": 2}
QSTAGE = ["pwv", "pwv_raw", "imt", "imt_raw", "abi_min", "sokolow", "sbp_sit"]
ORGANS = [("sbp", "sbp_sit"), ("sok", "sokolow"), ("egfr", "egfr"), ("pwv", "pwv"), ("imt", "imt")]
LAGGED = [("art_w", ["pwv", "abi_min", "imt"]), ("vein_w", ["pwv", "abi_min", "imt"]), ("imt", ["pwv", "abi_min"])]
REF_AGES = (40, 45, 50, 55, 60, 65, 70)
MIN_N = 200
WIDTH, MAXLINES = 220, 120
LONG = re.compile(r"\d{7,}")
LIVER = re.compile(r"liver|hepat|mafld|nafld|steato", re.I)
DATE_RX = re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?([+-]\d{2}:?\d{2}|Z)?")
BLOCKED = []  # liver paths whose read was refused


# ============================================================================ helpers (guard copied from f1_followup_v1)

def hfmt(h):
    """First 16 hex characters of a SHA-256 in groups of 6/6/4 (an unbroken hash can hold 7+ digit runs)."""
    h = str(h)[:16]
    return ".".join(h[i:i + 6] for i in range(0, len(h), 6))


def sha16_file(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def load_modules(atlas_path, outcomes_path):
    """Import f1_outcomes_v1 by path; it imports the atlas by path (O.load_atlas).  A.rd is wrapped so that no liver
    table is ever read (F1 partition; the atlas frame then has no liver attenuation, which no model here uses)."""
    global O, A
    spec = importlib.util.spec_from_file_location("f1_outcomes_v1", outcomes_path)
    O = importlib.util.module_from_spec(spec)
    sys.modules["f1_outcomes_v1"] = O
    spec.loader.exec_module(O)
    A = O.load_atlas(atlas_path)
    orig = A.rd

    def rd(root, rel, columns=None):
        if LIVER.search(str(rel)):
            BLOCKED.append(str(rel))
            return None
        return orig(root, rel, columns)
    A.rd = rd
    return O, A


def log(m):
    A.log(m)


def rc(n):
    return A.rc(n)


def v3(x):
    return A.v3(x)


def nrc(mask):
    return rc(int(np.asarray(mask, bool).sum()))


def scrub(s):
    """Exception text can quote data values: dates, clock times and digit runs >= 4 are masked."""
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
    """Module version strings carry build dates ('f1_atlas_v1.5 2026-10-03'); outputs show the calendar quarter only."""
    return DATE_RX.sub(lambda m: f"{m.group(0)[:4]}Q{(int(m.group(0)[5:7]) - 1) // 3 + 1}", str(s))


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def label(s, n=22):
    """Category label for output: lower case letters, digits < 4 in a row, spaces; at most n characters."""
    return re.sub(r"\d{4,}", "#", re.sub(r"[^a-z0-9_ <>=+/-]", "", str(s).strip().lower()))[:n] or "blank"


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * len(pv) / np.arange(1, len(pv) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty(len(pv))
        qq[o] = np.minimum(r, 1.0)
        q[ok] = qq
    return q


def wols(y, X, w=None):
    """Weighted least squares; sandwich covariance from per-row influence functions (one row per participant)."""
    y = np.asarray(y, float)
    n, k = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    Xw = X * w[:, None]
    bread = np.linalg.pinv(X.T @ Xw)
    b = bread @ (Xw.T @ y)
    e = y - X @ b
    IF = (X * (w * e)[:, None]) @ bread
    V = IF.T @ IF * n / max(n - k, 1)
    return b, V, IF


def coef_d(b, V, i, n):
    se = float(np.sqrt(max(V[i, i], 0)))
    return {"b": float(b[i]), "lo": float(b[i] - 1.96 * se), "hi": float(b[i] + 1.96 * se), "se": se,
            "p": float(2 * stats.norm.sf(abs(b[i] / se))) if se > 0 else None, "n": rc(n)}


def stack_diff(auxes, names):
    """Paired contrasts of coefficients from separate fits on one sample (stacked influence functions, P1 O.stack)."""
    b, V = O.stack(auxes)
    out = {"pairs": {}}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = b[i] - b[j]
            se = float(np.sqrt(max(V[i, i] + V[j, j] - 2 * V[i, j], 0)))
            out["pairs"][f"{names[i]}-{names[j]}"] = {"d": float(d), "lo": float(d - 1.96 * se), "hi": float(d + 1.96 * se),
                                                     "p": float(2 * stats.norm.sf(abs(d / se))) if se > 0 else None}
    if len(names) > 2:
        out["wald"] = O.wald(b, V)
    return out


# ============================================================================ data: medications, conditions, contacts

def med_class_rows(root):
    """One row per medication record with class flags m_<class> from the finest ATC level present in the record."""
    m = A.rd(root, "medications/medications.parquet", columns=["collection_date", "atc3", "atc4", "atc5"])
    if m is None:
        return None
    m = m.reset_index(drop=True)
    m["_row"] = np.arange(len(m))
    parts = []
    for lev, c in ((3, "atc3"), (4, "atc4"), (5, "atc5")):
        if c not in m.columns:
            continue
        e = m[["_row", c]].explode(c).dropna(subset=[c]).rename(columns={c: "code"})
        e["code"] = e["code"].astype(str).str.upper().str.strip()
        e = e[~e["code"].isin(["", "NAN", "NONE", "<NA>"])]
        e["lev"] = lev
        parts.append(e)
    out = pd.DataFrame({"participant_id": m["participant_id"].values, "research_stage": m["research_stage"].values,
                        "date": O.naive(m["collection_date"]).values if "collection_date" in m.columns else pd.NaT})
    for k in MED:
        out["m_" + k] = 0.0
    if parts:
        e = pd.concat(parts, ignore_index=True)
        e = e[e["lev"] == e.groupby("_row")["lev"].transform("max")]
        for k, pre in MED.items():
            hit = e["code"].str.startswith(pre)
            if k == "dmdrug":
                hit &= ~e["code"].str.startswith(DM_EXCL)
            out.loc[pd.unique(e.loc[hit, "_row"]), "m_" + k] = 1.0
    return out


def cond_group_rows(root):
    c = A.rd(root, "medical_conditions/medical_conditions.parquet", columns=["icd11_code", "collection_date", "start_year"])
    if c is None:
        return None
    code = c["icd11_code"].astype(str).str.upper().str.replace(".", "", regex=False).str.strip()
    out = pd.DataFrame({"participant_id": c["participant_id"].values, "research_stage": c["research_stage"].values,
                        "date": O.naive(c["collection_date"]).values if "collection_date" in c.columns else pd.NaT,
                        "start_year": pd.to_numeric(c["start_year"], errors="coerce").values
                        if "start_year" in c.columns else np.nan})
    for k, pre in ICD.items():
        out["i_" + k] = code.str.startswith(pre).astype(float).values
    return out[out[["i_" + k for k in ICD]].max(axis=1) > 0].reset_index(drop=True)


def add_flags(C, base_dates, R, cols):
    """Contact flags from rows R at the follow-up stages; rows of other stages ('undefined') dated > 30 days after
    baseline go to the next attended contact (P1 load_contacts rule)."""
    C = C.copy()
    info = {}
    for col in cols:
        C[col] = 0.0
    if R is None or not len(R):
        return C, info
    kmap = {k: v[0] for k, v in O.FU.items()}
    fu = R[R["research_stage"].isin(list(O.FU))]
    agg = fu.groupby(["participant_id", "research_stage"])[cols].max().reset_index()
    agg["kind"] = agg["research_stage"].map(kmap)
    C = C.drop(columns=cols).merge(agg.drop(columns=["research_stage"]), on=["participant_id", "kind"], how="left")
    C[cols] = C[cols].fillna(0.0)
    st_all = [O.BASE] + list(O.FU)
    att = C[C["has_data"] & (C["t"] > O.GAP0 / 365.25)][["participant_id", "kind", "date"]].sort_values("date")
    u0 = R[~R["research_stage"].isin(st_all) & R["date"].notna()]
    u0 = u0[u0["date"] > u0["participant_id"].map(base_dates) + pd.Timedelta(days=O.GAP0)]
    for col in cols:
        u = u0[u0[col] == 1][["participant_id", "date"]].sort_values("date")
        if not len(u) or not len(att):
            info[col] = None
            continue
        j = pd.merge_asof(u, att.rename(columns={"date": "cdate_"}), left_on="date", right_on="cdate_",
                          by="participant_id", direction="forward").dropna(subset=["kind"])
        hit = set(zip(j["participant_id"], j["kind"]))
        sel = np.array([(p_, k_) in hit for p_, k_ in zip(C["participant_id"], C["kind"])], bool)
        C.loc[sel, col] = 1.0
        info[col] = rc(len(j))
    return C, info


def prevalent(R, base_dates, col):
    """Participants with a flagged row at baseline, a start year before the baseline year, or an 'undefined' row dated
    up to 30 days after baseline (P1 rule for prevalent ICD hypertension)."""
    if R is None or not len(R):
        return set()
    r = R[R[col] == 1]
    by = pd.to_datetime(r["participant_id"].map(base_dates))
    ids = set(r.loc[r["research_stage"] == O.BASE, "participant_id"])
    if "start_year" in r.columns:
        sy = pd.to_numeric(r["start_year"], errors="coerce")
        ids |= set(r.loc[sy.notna() & (sy < by.dt.year), "participant_id"])
    st_all = [O.BASE] + list(O.FU)
    und = ~r["research_stage"].isin(st_all) & r["date"].notna() & (r["date"] <= by + pd.Timedelta(days=O.GAP0))
    ids |= set(r.loc[und, "participant_id"])
    return ids


def lab_thresholds(fr):
    hb = pd.to_numeric(fr.get("hba1c"), errors="coerce").median() if "hba1c" in fr.columns else np.nan
    gl = pd.to_numeric(fr.get("glucose"), errors="coerce").median() if "glucose" in fr.columns else np.nan
    return {"hba1c": 48.0 if np.isfinite(hb) and hb > 20 else 6.5, "glucose": 7.0 if np.isfinite(gl) and gl < 30 else 126.0,
            "egfr": 60.0}


def curated_flags(root):
    """Baseline curated diabetes / IHD flags (participant sets) for the cohort exclusions."""
    out = {}
    for name, col, rule in (("dm", "diabetes__curated_phenotype",
                             lambda s: s.str.contains("diabet") & ~s.str.startswith(("non", "pre"))),
                            ("ihd", "ischemic_heart_disease__curated_phenotype",
                             lambda s: s.str.contains("ischemic heart") & ~s.str.startswith("no"))):
        tab = "diabetes" if name == "dm" else "ischemic_heart_disease"
        d = A.rd(root, f"curated_phenotypes/{tab}.parquet", columns=[col])
        if d is None or col not in d.columns:
            out[name] = set()
            continue
        s = d[col].astype(str).str.strip().str.lower()
        out[name] = set(d.loc[(d["research_stage"] == BASE) & rule(s), "participant_id"])
    return out



# ============================================================================ frame additions: IMT without the clinic-day step, gaps

def cal_only(fr, col):
    """The atlas v1.5 calendar-quarter step alone (A.calendar_adjust on a copy holding `col` in the 'imt' slot)."""
    keep = [c for c in ("participant_id", "research_stage", "visit_date", "vas_ts", "age", "sex_male") if c in fr.columns]
    sub = fr[keep].copy()
    sub["imt"] = fr[col].values
    out = A.calendar_adjust(sub)
    return out["imt"].values


def rcs(x, k):
    """Restricted cubic spline basis (Harrell): x and len(k)-2 non-linear terms."""
    x = np.asarray(x, float)
    p3 = lambda u: np.clip(u, 0, None) ** 3  # noqa: E731
    K = len(k)
    cols = [x]
    nrm = (k[-1] - k[0]) ** 2
    for j in range(K - 2):
        cols.append((p3(x - k[j]) - p3(x - k[K - 2]) * (k[K - 1] - k[j]) / (k[K - 1] - k[K - 2])
                     + p3(x - k[K - 1]) * (k[K - 2] - k[j]) / (k[K - 1] - k[K - 2])) / nrm)
    return np.column_stack(cols)


def reference_ids(B0, prev_ascvd, cur_ihd, thr):
    """Healthy reference subset at baseline: untreated, no hypertension diagnosis, BP < 140/90, no glucose-lowering
    drug, HbA1c and glucose below the diabetes thresholds (or missing), no ASCVD (ICD or curated IHD), not a current
    smoker, BMI 18.5-30."""
    b = B0

    def c(name):
        return b[name].astype(float) if name in b.columns else pd.Series(np.nan, index=b.index)
    m = (c("antihtn").fillna(0) == 0) & (c("htn_curated").fillna(0) == 0) & (c("htn_self").fillna(0) == 0)
    m &= (c("sbp_sit") < 140) & (c("dbp_sit") < 90) & (c("diabetes_any").fillna(0) == 0)
    m &= ~(c("hba1c") >= thr["hba1c"]) & ~(c("glucose") >= thr["glucose"])
    m &= ~b["participant_id"].isin(prev_ascvd | cur_ihd) & (c("smk_current").fillna(0) != 1)
    m &= (c("bmi") >= 18.5) & (c("bmi") <= 30)
    return set(b.loc[m, "participant_id"])


def gap_reference(fr, ref):
    """Per bed and sex: bed ~ RCS(age, knots at reference age 5/35/65/95 %) in the reference rows at baseline; gap
    columns gz_<bed> (adverse z per reference SD) and gy_<bed> (years, PWV and IMT) on every frame row."""
    out = {}
    b0 = (fr["research_stage"] == BASE).values & fr["participant_id"].isin(ref).values
    for bed in GAP_BEDS:
        fr["gz_" + bed] = np.nan
        if bed in GAP_YEARS:
            fr["gy_" + bed] = np.nan
        if bed not in fr.columns:
            out[bed] = {"status": "missing"}
            continue
        sign = A.ORIENT.get(bed, 1)
        rec = {}
        for sx, lab_ in ((0, "F"), (1, "M")):
            r = fr[b0 & (fr["sex_male"] == sx).values].dropna(subset=[bed, "age"])
            if len(r) < MIN_N:
                rec[lab_] = {"n_ref": rc(len(r)), "status": "too few"}
                continue
            a = r["age"].astype(float).values
            k = np.unique(np.quantile(a, [0.05, 0.35, 0.65, 0.95]))
            if len(k) < 3:
                rec[lab_] = {"n_ref": rc(len(r)), "status": "age range too narrow"}
                continue
            lo, hi = float(np.quantile(a, 0.01)), float(np.quantile(a, 0.99))
            X = np.column_stack([np.ones(len(a)), rcs(a, k)])
            y = r[bed].astype(float).values
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            res = y - X @ beta
            sd = float(np.sqrt((res ** 2).sum() / max(len(y) - X.shape[1], 1)))

            def f(age, beta=beta, k=k, lo=lo, hi=hi):
                aa = np.clip(np.asarray(age, float), lo, hi)
                return np.column_stack([np.ones(len(aa)), rcs(aa, k)]) @ beta
            slope = float(np.mean(f(a + 0.5) - f(a - 0.5)))
            sel = (fr["sex_male"] == sx).values & fr[bed].notna().values & fr["age"].notna().values
            pred = f(fr.loc[sel, "age"].values)
            dev = fr.loc[sel, bed].astype(float).values - pred
            fr.loc[sel, "gz_" + bed] = sign * dev / sd
            if bed in GAP_YEARS and sign * slope > 0:
                fr.loc[sel, "gy_" + bed] = dev / slope
            curve = {}
            for ag in REF_AGES:
                if ((a >= ag - 2.5) & (a < ag + 2.5)).sum() >= A.MINCELL and lo <= ag <= hi:
                    curve[str(ag)] = float(f([ag])[0])
            cent = {}
            for ag in REF_AGES[:-1]:
                cent[f"{ag}-{ag + 4}"] = A.qtiles(pd.Series(y[(a >= ag) & (a < ag + 5)]))
            rec[lab_] = {"n_ref": rc(len(r)), "sd_ref": sd, "slope_per_year": slope,  # prediction ages clipped to the
                         "years_per_sd": sd / abs(slope) if abs(slope) > 0 else None,  # reference 1-99 % range (not exported)
                         "curve": curve, "centiles": cent}
        gz = fr.loc[fr["research_stage"] == BASE, "gz_" + bed]
        rec["gap_z_baseline"] = A.qtiles(gz)
        rec["gap_ge1_baseline"] = {"n": nrc(gz.notna()), "ge1": nrc(gz >= 1)}
        if bed in GAP_YEARS:
            rec["gap_years_baseline"] = A.qtiles(fr.loc[fr["research_stage"] == BASE, "gy_" + bed])
        out[bed] = rec
    out["reference_n"] = rc(len(ref))
    return out


# ============================================================================ baseline table and cohorts

def baseline_table(fr, ctx, extra):
    """One row per baseline participant with a baseline date: frame columns, P1 base_date/base_year/fam_htn, usual BP,
    oriented beds o_<bed>, gaps, 3-bed composite c3 and adverse-bed count, baseline drug and ICD flags."""
    b = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id").copy()
    b["base_date"] = b["participant_id"].map(ctx["base_dates"])
    b = b[b["base_date"].notna()].reset_index(drop=True)
    by = pd.to_datetime(b["base_date"])
    b["base_year"] = by.dt.year + (by.dt.dayofyear - 1) / 365.25
    if ctx["fam"] is not None:
        b = b.merge(ctx["fam"], on="participant_id", how="left")
    else:
        b["fam_htn"] = np.nan
    b = b.merge(ctx["U"][["participant_id", "u_sbp", "u_dbp"]], on="participant_id", how="left")
    for bed in BEDS + ["pwv_raw", "imt_nod"]:
        if bed in b.columns:
            b["o_" + bed] = A.ORIENT.get(bed, 1) * b[bed].astype(float)
    # 3-bed composite: mean of age/sex-standardised adverse z (residual on age, (age-50)^2, sex in baseline rows with all 3)
    cols = ["o_" + c for c in CORE3]
    b["c3"], b["o_c3"], b["cnt3"] = np.nan, np.nan, np.nan
    ok = b[cols + ["age", "sex_male"]].notna().all(axis=1).values
    info = {"n_all3": nrc(ok)}
    if ok.sum() >= MIN_N:
        a = b.loc[ok, "age"].astype(float).values
        X = np.column_stack([np.ones(ok.sum()), a, (a - 50) ** 2, b.loc[ok, "sex_male"].astype(float).values])
        Z, top = [], []
        for c in cols:
            y = b.loc[ok, c].astype(float).values
            r = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            z = r / r.std()
            Z.append(z)
            top.append(z >= np.quantile(z, 0.9))
        b.loc[ok, "c3"] = np.mean(Z, axis=0)
        b.loc[ok, "o_c3"] = b.loc[ok, "c3"]
        cnt = np.sum(top, axis=0)
        b.loc[ok, "cnt3"] = cnt
        info.update({"count_0": nrc(cnt == 0), "count_1": nrc(cnt == 1), "count_ge2": nrc(cnt >= 2),
                     "corr_z": {f"{SH[CORE3[i]]}-{SH[CORE3[j]]}": float(np.corrcoef(Z[i], Z[j])[0, 1])
                                for i in range(3) for j in range(i + 1, 3)}})
    b["cnt_1"] = np.where(b["cnt3"].notna(), (b["cnt3"] == 1).astype(float), np.nan)
    b["cnt_2"] = np.where(b["cnt3"].notna(), (b["cnt3"] >= 2).astype(float), np.nan)
    for bed in GAP_BEDS:
        c = "gz_" + bed
        if c in b.columns:
            b["acc_" + bed] = np.where(b[c].notna(), (b[c] >= 1).astype(float), np.nan)
    m0, prev = extra["med0"], extra["prev"]
    for k in MED:
        b["b_" + k] = b["participant_id"].map(m0.get(k, {})).fillna(0.0) if m0 else 0.0
    for k, col in (("antihtn", "antihtn"), ("statin", "statin"), ("dmany", "diabetes_any")):  # frame flags as well
        if col in b.columns:
            b["b_" + k] = np.maximum(b["b_" + k], b[col].astype(float).fillna(0))
    for k in ICD:
        b["p_" + k] = b["participant_id"].isin(prev.get(k, set())).astype(float)
    return b, info


def cohort_mask(oc, b, ctx, extra, thr):
    if oc == "htn_obj":
        return b["participant_id"].isin(set(ctx["B"]["participant_id"])).values
    if oc == "ahtn":
        return (b["b_antihtn"] == 0).values
    if oc == "statin":
        return (b["b_statin"] == 0).values
    if oc == "dm":
        return ((b["b_dmany"] == 0) & (b["p_dm"] == 0) & ~(b["hba1c"] >= thr["hba1c"]) & ~(b["glucose"] >= thr["glucose"])
                & ~b["participant_id"].isin(extra["cur"].get("dm", set()))).values
    if oc == "ascvd":
        return ((b["p_ihd"] == 0) & (b["p_cereb"] == 0) & (b["p_pad"] == 0) & (b["b_nitrate"] == 0)
                & ~b["participant_id"].isin(extra["cur"].get("ihd", set()))).values
    if oc == "ckd":
        return ((b["p_ckd"] == 0) & (b["egfr"] >= thr["egfr"])).values
    if oc == "nc":
        return (b["p_nc"] == 0).values
    raise KeyError(oc)


def ev_fn(oc, thr):
    v2 = lambda c: (c["kind"] == "v2").values  # noqa: E731
    if oc == "htn_obj":
        return O.ev_objective
    if oc == "ahtn":
        return lambda c: (c["med"] == 1).astype(float)
    if oc == "statin":
        return lambda c: (c["m_statin"] == 1).astype(float)
    if oc == "dm":
        return lambda c: ((c["m_dmdrug"] == 1) | (c["i_dm"] == 1) | (v2(c) & ((c["hba1c2"] >= thr["hba1c"]) |
                                                                         (c["glu2"] >= thr["glucose"])))).astype(float)
    if oc == "ascvd":
        return lambda c: ((c["i_ihd"] == 1) | (c["i_cereb"] == 1) | (c["i_pad"] == 1) | (c["m_nitrate"] == 1)).astype(float)
    if oc == "ckd":
        return lambda c: ((c["i_ckd"] == 1) | (v2(c) & (c["egfr2"] < thr["egfr"]))).astype(float)
    if oc == "nc":
        return lambda c: (c["i_nc"] == 1).astype(float)
    raise KeyError(oc)


COMPONENTS = {
    "htn_obj": {"drug": lambda c: c["med"] == 1, "icd": lambda c: c["icd"] == 1,
                "bp_v2": lambda c: (c["kind"] == "v2") & ((c["sbp2"] >= 140) | (c["dbp2"] >= 90))},
    "dm": {"drug": lambda c: c["m_dmdrug"] == 1, "icd": lambda c: c["i_dm"] == 1, "lab_v2": lambda c: c["_lab"]},
    "ascvd": {"ihd": lambda c: c["i_ihd"] == 1, "cereb": lambda c: c["i_cereb"] == 1, "pad": lambda c: c["i_pad"] == 1,
              "nitrate": lambda c: c["m_nitrate"] == 1},
    "ckd": {"icd": lambda c: c["i_ckd"] == 1, "egfr_v2": lambda c: (c["kind"] == "v2") & (c["egfr2"] < 60)},
}


# ============================================================================ E: incident events, paired contrasts, gaps

def hz_fit(P, B, x, covs, std=True, weighted=True):
    r, aux = O.hazard(P, B, [x], covs, weighted=weighted, std=std)
    return r, aux


def model_people(P, B, need):
    """v2.0.1: one row per person (event indicator + `need`) in the sample O.hazard fits: complete cases of `need`, contact
    kinds without events dropped, call 3 folded into call 1 when it has < 5 events (P1 hazard rules)."""
    need = list(dict.fromkeys(c for c in need if c in B.columns))
    d = P.merge(B[["participant_id"] + need], on="participant_id", how="inner")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=need + ["dt", "ev"])
    kind = d["kind"].astype(str)
    evk = d.groupby(kind)["ev"].sum()
    if "call3" in evk.index and evk["call3"] < 5 and "call1" in evk.index:
        kind = kind.replace({"call3": "call1"})
        evk = d.groupby(kind)["ev"].sum()
    d = d[kind.isin(evk[evk > 0].index).values]
    return d.groupby("participant_id").agg(ev=("ev", "max"), **{c: (c, "first") for c in need})


def binary_cells_ok(P, B, x, covs=()):
    """2 x 2 (group x event) cells >= 20 in the model sample (v2.0.1: complete cases of x and covs, as fitted)."""
    d = model_people(P, B, [x] + list(covs))
    if x not in d.columns:
        return False
    return min(int(((d[x] == h) & (d["ev"] == e)).sum()) for h in (0, 1) for e in (0, 1)) >= A.MINCELL


def count_cells_ok(P, B, covs=()):
    """3 x 2 (adverse-bed count 0/1/>=2 x event) cells >= 20 in the model sample (v2.0.1: complete cases, as fitted)."""
    d = model_people(P, B, ["cnt3"] + list(covs))
    if "cnt3" not in d.columns:
        return False
    cat = np.minimum(d["cnt3"].values, 2)
    return min(int(((cat == h) & (d["ev"].values == e)).sum()) for h in (0, 1, 2) for e in (0, 1)) >= A.MINCELL


def paired_block(P, B, beds, covs):
    cols = ["o_" + b for b in beds]
    need = [c for c in covs if c in B.columns]
    if any(c not in B.columns for c in cols):
        return {"status": "missing beds"}
    Bc = B.dropna(subset=cols + need)
    res, aux = {}, []
    for bed in beds:
        r, a = hz_fit(P, Bc, "o_" + bed, covs)
        res[bed] = r
        aux.append(a)
    out = {"n": g(res, beds[0], "n"), "events": g(res, beds[0], "events"),
           "hr": {b_: g(res[b_], "hr", "o_" + b_) for b_ in beds}}
    if any(a is None for a in aux):
        out["status"] = "too few"
        return out
    b, V = O.stack(aux)
    out["status"] = "ok"
    out["ratio"] = {f"{SH[beds[i]]}/{SH[beds[j]]}": O.diff_line(b, V, i, j)
                    for i in range(len(beds)) for j in range(i + 1, len(beds))}
    if len(beds) > 2:
        out["wald"] = O.wald(b, V)
    return out


def first_components(oc, PP, C2):
    comps = COMPONENTS.get(oc)
    if not comps:
        return None
    fe = PP[PP["ev"] == 1][["participant_id", "kind"]].merge(C2, on=["participant_id", "kind"], how="left")
    return {k: nrc(f(fe).fillna(False).astype(bool)) for k, f in comps.items()}


def outcome_block(oc, b, C2, ctx, extra, thr):
    spec = OUTCOMES[oc]
    B = b[cohort_mask(oc, b, ctx, extra, thr)].reset_index(drop=True)
    out = {"label": spec["label"], "cohort": rc(len(B))}
    PP = O.person_period(B, C2, ev_fn(oc, thr))
    out["first_event_components"] = first_components(oc, PP, C2)
    W, wi = O.ipw(B, C2, PP)
    P = O.attach_w(PP, W)
    pe = P.groupby("participant_id").agg(fu=("t1", "max"))
    out["flow"] = {"followed": rc(len(pe)), "events": nrc(P["ev"] == 1),
                   "by_kind": {k: nrc((P["kind"] == k) & (P["ev"] == 1)) for k in O.KINDS},
                   "person_years": rc(int(round(P["dt"].sum()))), "followup_years": A.qtiles(pe["fu"]),
                   "ipw_auc": {k: g(wi, k, "auc_attendance") for k in O.KINDS},
                   "ipw_status": {k: g(wi, k, "status") for k in O.KINDS if g(wi, k, "status")}}
    covX = O.COV_B + [c for c in spec["extras"] if c in B.columns and B[c].notna().mean() >= 0.3]
    out["covX"] = covX
    beds = {}
    for bed in XBEDS:
        x = "o_" + bed
        if x not in B.columns or B[x].notna().sum() < MIN_N:
            beds[bed] = {"status": "too few with bed"}
            continue
        beds[bed] = {"A": hz_fit(P, B, x, COV_A)[0], "B": hz_fit(P, B, x, O.COV_B)[0], "X": hz_fit(P, B, x, covX)[0],
                     "Bu": hz_fit(P, B, x, O.COV_B, weighted=False)[0]}
    out["beds"] = beds
    if count_cells_ok(P, B, O.COV_B):
        r, _ = O.hazard(P, B, ["cnt_1", "cnt_2"], O.COV_B, std=False)
        out["count3"] = r
    else:
        out["count3"] = {"status": "suppressed: count x event cell < 20"}
    out["paired"] = {"S3_B": paired_block(P, B, CORE3, O.COV_B), "S3_A": paired_block(P, B, CORE3, COV_A)}
    for p_ in XPAIRS:
        out["paired"][f"{SH[p_[0]]}/{SH[p_[1]]}_B"] = paired_block(P, B, list(p_), O.COV_B)
    gaps = {}
    for bed in GAP_BEDS:
        x = "gz_" + bed
        if x not in B.columns or B[x].notna().sum() < MIN_N:
            gaps[bed] = {"status": "too few with gap"}
            continue
        rec = {"A": hz_fit(P, B, x, COV_A, std=False)[0], "B": hz_fit(P, B, x, O.COV_B, std=False)[0]}
        xa = "acc_" + bed
        rec["ge1_B"] = (hz_fit(P, B, xa, O.COV_B, std=False)[0] if binary_cells_ok(P, B, xa, O.COV_B)
                        else {"status": "suppressed: gap group x event cell < 20"})
        gaps[bed] = rec
    out["gaps"] = gaps
    gc = ["gz_" + c for c in CORE3]
    if all(c in B.columns for c in gc):
        Bg = B.dropna(subset=gc + O.COV_B)
        aux = [hz_fit(P, Bg, c, O.COV_B, std=False)[1] for c in gc]
        out["gaps_paired"] = (stack_diff(aux, [SH[c] for c in CORE3]) if all(a is not None for a in aux)
                              else {"status": "too few"})
    return out


def repro_check(ctx):
    """P1 O8 objective-only (composite follow-up weights) PWV B: must equal the P1 digest line OUT L82."""
    P = O.attach_w(O.person_period(ctx["B"], ctx["C"], O.ev_objective), ctx["W"])
    r = O.hazard(P, ctx["B"], ["o_pwv"], O.COV_B)[0]
    h = g(r, "hr", "o_pwv")
    ref = (1.04, 0.812, 1.34)
    ok = bool(h) and all(f"{x:.3g}" == f"{y:.3g}" for x, y in zip((h["hr"], h["lo"], h["hi"]), ref))
    return {"objective_composite_weights_pwv_B": h, "n": g(r, "n"), "events": g(r, "events"), "p1_digest": list(ref),
            "match_3sd": ok}


# ============================================================================ P: progression

def pair_table(fr, b, col, untreated=False):
    """Participants with `col` at both main visits: y1, y2, interval (y), v2 attendance weight among those due."""
    v1 = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id").set_index("participant_id")
    v2 = fr[fr["research_stage"] == V2].drop_duplicates("participant_id").set_index("participant_id")
    if col not in v1.columns:
        return None, {"status": "missing"}
    base = b.set_index("participant_id")
    y1 = v1[col].reindex(base.index)
    y2 = v2[col].reindex(base.index)
    vd2 = pd.Series(O.naive(v2["visit_date"]).values, index=v2.index).reindex(base.index)
    bd = pd.Series(O.naive(base["base_date"]).values, index=base.index)
    t = (vd2 - bd).dt.days / 365.25
    has1 = y1.notna()
    pair = has1 & y2.notna() & (t > 0.5)
    if untreated:
        tr2 = v2["antihtn"].reindex(base.index).fillna(0) if "antihtn" in v2.columns else 0
        keep = (base["antihtn"].fillna(0) == 0) & (tr2 == 0)
        pair &= keep
        has1 &= keep
    info = {"pairs": nrc(pair)}
    if pair.sum() < MIN_N:
        info["status"] = "too few"
        return None, info
    nom = float(t[pair].median())
    cut = vd2.max()
    due = (bd + pd.to_timedelta((nom - 0.5) * 365.25, unit="D") <= cut).values
    X, _ = O.wmatrix(b)
    rec = {}
    w = O._attend_w(X, pair.values, (has1.values & due), rec)
    info.update({"interval_years": A.qtiles(t[pair]), "ipw_auc": rec.get("auc_attendance"), "ipw_status": rec.get("status")})
    d = pd.DataFrame({"participant_id": base.index[pair.values], "y1": y1[pair].values.astype(float),
                      "y2": y2[pair].values.astype(float), "interval": t[pair].values, "w": w[pair.values],
                      "q1": bd[pair].dt.quarter.values, "q2": vd2[pair].dt.quarter.values})
    return d, info


def change_summary(d, draw=None):
    dl = d["y2"] - d["y1"]
    sd1 = float(d["y1"].std())
    out = {"n": rc(len(d)), "mean_change": float(dl.mean()), "se": float(dl.std() / np.sqrt(len(d))),
           "sd_change": float(dl.std()), "change_q": A.qtiles(dl), "per_year": float((dl / d["interval"]).mean()),
           "mean_change_sd_units": float(dl.mean() / sd1) if sd1 > 0 else None, "sd_baseline": sd1,
           "icc": float(A.icc1(d["y1"], d["y2"]))}
    sm = d["q1"] == d["q2"]
    out["season_matched"] = {"n": nrc(sm), "mean_change": float(dl[sm].mean()) if sm.sum() >= A.MINCELL else None}
    if draw is not None:
        rr = draw.reindex(d["participant_id"])
        ok = rr["y1"].notna().values & rr["y2"].notna().values
        if ok.sum() >= A.MINCELL:
            dr = rr["y2"].values[ok] - rr["y1"].values[ok]
            out["raw"] = {"n": nrc(ok), "mean_change": float(dr.mean()), "icc": float(A.icc1(rr["y1"].values[ok], rr["y2"].values[ok])),
                          "season_matched_mean_change": float(dr[sm.values[ok]].mean()) if (sm.values[ok]).sum() >= A.MINCELL else None}
    return out


def predictor_models(d, b, bed):
    """ANCOVA (y2 on y1, interval, age, sex, predictor) and change-score (y2 - y1 on interval, age, sex, predictor);
    outcome in baseline-SD units; continuous predictors per SD, binary per unit; weighted by v2 attendance IPW."""
    sd1 = float(d["y1"].std())
    base = b.set_index("participant_id")
    out = {"ancova": {}, "change": {}}
    for nm, col, binary in PRED:
        if col == bed or col not in base.columns:
            continue
        e = d.copy()
        e["x"] = base[col].reindex(e["participant_id"]).astype(float).values
        for c in ("age", "sex_male"):
            e[c] = base[c].reindex(e["participant_id"]).astype(float).values
        e = e.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "age", "sex_male", "y1", "y2", "interval", "w"])
        if len(e) < MIN_N:
            continue
        if binary:
            if min((e["x"] == 1).sum(), (e["x"] == 0).sum()) < A.MINCELL:
                continue
            x = e["x"].values
        else:
            s = e["x"].std()
            if not s > 0:
                continue
            x = ((e["x"] - e["x"].mean()) / s).values
        cov = ([] if col == "age" else [(e["age"].values - 50) / 10]) + ([] if col == "sex_male" else [e["sex_male"].values])
        y1s, y2s = e["y1"].values / sd1, e["y2"].values / sd1
        X1 = np.column_stack([np.ones(len(e)), x, y1s, e["interval"].values] + cov)
        b1, V1, _ = wols(y2s, X1, e["w"].values)
        out["ancova"][nm] = coef_d(b1, V1, 1, len(e))
        X2 = np.column_stack([np.ones(len(e)), x, e["interval"].values] + cov)
        b2, V2_, _ = wols(y2s - y1s, X2, e["w"].values)
        out["change"][nm] = coef_d(b2, V2_, 1, len(e))
    for k in ("ancova", "change"):
        names = list(out[k])
        q = bh([out[k][n_]["p"] if out[k][n_]["p"] is not None else np.nan for n_ in names])
        for n_, qq in zip(names, q):
            out[k][n_]["q"] = float(qq) if np.isfinite(qq) else None
    return out


def stage_within_quarter(fr, col):
    """Visit-2 minus baseline difference among rows measured in the same calendar quarter (quarter dummies, age,
    (age-50)^2, sex): the stage effect that calendar drift cannot explain."""
    q, _ = A.calendar_quarter(fr)
    d = pd.DataFrame({"y": fr[col].astype(float).values, "q": q.values, "v2": (fr["research_stage"] == V2).astype(float).values,
                      "age": fr["age"].astype(float).values, "sex": fr["sex_male"].astype(float).values})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[~d["q"].astype(str).str.contains("NA")]
    ct = d.groupby("q")["v2"].agg(["sum", "count"])
    both = ct[(ct["sum"] >= A.MINCELL) & (ct["count"] - ct["sum"] >= A.MINCELL)].index
    d = d[d["q"].isin(both)]
    if len(both) < 1 or len(d) < MIN_N:
        return {"status": "too few shared quarters", "quarters_with_both": len(both)}
    D = pd.get_dummies(pd.Categorical(d["q"]), dtype=float).values
    a = d["age"].values
    X = np.column_stack([d["v2"].values, a, (a - 50) ** 2, d["sex"].values, D])
    b, V, _ = wols(d["y"].values, X)
    return {"stage_effect": coef_d(b, V, 0, len(d)), "quarters_with_both": len(both)}


def sok_by_year(fr):
    out = {}
    for st in (BASE, V2):
        d = fr[(fr["research_stage"] == st)].dropna(subset=["sokolow", "visit_date"])
        yr = pd.to_datetime(d["visit_date"]).dt.year
        out[SLAB[st]] = {str(int(y)): {"mean": float(d.loc[yr == y, "sokolow"].mean()), "n": rc((yr == y).sum())}
                         for y in sorted(yr.unique()) if (yr == y).sum() >= A.MINCELL}
    return out


def run_progression(fr, b):
    out = {}
    for bed, raw in PROG:
        rec = {}
        d, info = pair_table(fr, b, bed, untreated=(bed == "sbp_sit"))
        rec["pairs"] = info
        if d is None:
            out[bed] = rec
            continue
        draw = None
        if raw and raw in fr.columns:
            dr_, _ = pair_table(fr, b, raw, untreated=False)
            if dr_ is not None:
                draw = dr_.set_index("participant_id")[["y1", "y2"]]
        rec["summary"] = change_summary(d, draw)
        rec["models"] = predictor_models(d, b, bed)
        out[bed] = rec
    out["stage_within_quarter"] = {c: stage_within_quarter(fr, c) for c in QSTAGE if c in fr.columns}
    out["sokolow_by_year"] = sok_by_year(fr)
    return out


# ============================================================================ G: organ decline and lagged cross-bed

def organ_block(fr, b, oname, col):
    d, info = pair_table(fr, b, col, untreated=(col == "sbp_sit"))
    out = {"pairs": info}
    if d is None:
        return out
    sd1 = float(d["y1"].std())
    base = b.set_index("participant_id")
    for c in ["age", "sex_male"] + ["gz_" + x for x in GAP_BEDS]:
        d[c] = base[c].reindex(d["participant_id"]).astype(float).values if c in base.columns else np.nan
    d["y1s"], d["y2s"] = d["y1"] / sd1, d["y2"] / sd1
    d["a10"] = (d["age"] - 50) / 10
    out["sd_baseline"] = sd1
    res, auxd = {}, {}
    for gb in GAP_BEDS:
        if gb == col:
            continue
        x = "gz_" + gb
        e = d.dropna(subset=[x, "y1s", "y2s", "interval", "a10", "sex_male", "w"])
        if len(e) < MIN_N:
            res[gb] = {"n": rc(len(e)), "status": "too few"}
            continue
        X = np.column_stack([np.ones(len(e)), e[x].values, e["y1s"].values, e["interval"].values, e["a10"].values,
                             e["sex_male"].values])
        bb, V, IF = wols(e["y2s"].values, X, e["w"].values)
        res[gb] = coef_d(bb, V, 1, len(e))
        auxd[gb] = (bb, V, IF, e["participant_id"].values)
    out["gap"] = res
    core = [c for c in CORE3 if c != col]
    e = d.dropna(subset=["gz_" + c for c in core] + ["y1s", "y2s", "interval", "a10", "sex_male", "w"])
    if len(e) >= MIN_N and len(core) >= 2:
        auxes = []
        for c in core:
            X = np.column_stack([np.ones(len(e)), e["gz_" + c].values, e["y1s"].values, e["interval"].values,
                                 e["a10"].values, e["sex_male"].values])
            bb, V, IF = wols(e["y2s"].values, X, e["w"].values)
            auxes.append({"b": bb[[1]], "V": V[np.ix_([1], [1])],
                          "IF": pd.DataFrame(IF[:, [1]], index=e["participant_id"].values, columns=[c])})
        out["paired"] = {"n": rc(len(e)), **stack_diff(auxes, [SH[c] for c in core])}
    return out


def lagged_block(fr, outbed, expos):
    """Outcome bed first measured at visit 2 (no baseline value): its visit-2 value (adverse, z in the sample) on the
    baseline gap (lagged) and on the visit-2 gap (concurrent) of each exposure bed, same people; age at v2, sex."""
    v1 = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id").set_index("participant_id")
    v2 = fr[fr["research_stage"] == V2].drop_duplicates("participant_id").set_index("participant_id")
    if outbed not in v2.columns:
        return {"status": "missing"}
    ids = v2.index[v2[outbed].notna() & ~v1[outbed].reindex(v2.index).notna()]
    out = {"n_new_at_v2": rc(len(ids))}
    y = A.ORIENT.get(outbed, 1) * v2.loc[ids, outbed].astype(float)
    for ex in expos:
        gx = "gz_" + ex
        d = pd.DataFrame({"y": y.values, "lag": v1[gx].reindex(ids).values, "con": v2[gx].reindex(ids).values,
                          "age": v2.loc[ids, "age"].astype(float).values, "sex": v2.loc[ids, "sex_male"].astype(float).values},
                         index=ids).dropna()
        if len(d) < MIN_N:
            out[ex] = {"n": rc(len(d)), "status": "too few"}
            continue
        ys = (d["y"] - d["y"].mean()) / d["y"].std()
        auxes, rec = [], {"n": rc(len(d))}
        for k in ("lag", "con"):
            X = np.column_stack([np.ones(len(d)), d[k].values, (d["age"].values - 50) / 10, d["sex"].values])
            bb, V, IF = wols(ys.values, X)
            rec[k] = coef_d(bb, V, 1, len(d))
            auxes.append({"b": bb[[1]], "V": V[np.ix_([1], [1])], "IF": pd.DataFrame(IF[:, [1]], index=d.index, columns=[k])})
        rec["lag_minus_con"] = stack_diff(auxes, ["lag", "con"])["pairs"]["lag-con"]
        out[ex] = rec
    return out


# ============================================================================ M0: inventory

def stage_lab(s):
    return SLAB.get(s, "und" if str(s).lower() in ("undefined", "nan", "none", "") else "oth")


def by_stage_participants(R, col):
    if R is None or col not in R.columns:
        return None
    r = R[R[col] == 1]
    s = r.groupby(r["research_stage"].map(stage_lab))["participant_id"].nunique()
    return {k: rc(s.get(k, 0)) for k in ("b", "c1", "v2", "c3", "und", "oth")}


def inventory(root, fr, mrows, crows):
    out = {}
    ev = A.rd(root, "events/events.parquet", columns=["research_stage_date"])
    if ev is not None:
        s = ev.groupby("research_stage")["participant_id"].nunique()
        out["event_stages"] = {label(k, 16): rc(v) for k, v in s.items()}
    cur = {}
    for tab in ("diabetes", "ckd", "ischemic_heart_disease", "hyperlipidemia"):
        col = f"{tab}__curated_phenotype"
        d = A.rd(root, f"curated_phenotypes/{tab}.parquet", columns=[col])
        if d is None or col not in d.columns:
            cur[tab] = {"status": "missing"}
            continue
        d["st"] = d["research_stage"].map(stage_lab)
        d["cat"] = d[col].astype(str).map(lambda x: label(x, 18))
        t = d.groupby(["st", "cat"])["participant_id"].nunique()
        cur[tab] = {st: {c: rc(n) for (s2, c), n in t.items() if s2 == st} for st in ("b", "c1", "v2", "c3", "und", "oth")
                    if st in t.index.get_level_values(0)}
    out["curated"] = cur
    out["icd"] = {k: by_stage_participants(crows, "i_" + k) for k in ICD}
    out["atc"] = {k: by_stage_participants(mrows, "m_" + k) for k in MED}
    fu = A.rd(root, "medical_procedures/follow_up_medical.parquet", columns=["hospitalization_since_recruitment"])
    if fu is not None:
        fu["st"] = fu["research_stage"].map(stage_lab)
        hv = fu["hospitalization_since_recruitment"].astype(str).str.strip().str.lower() \
            if "hospitalization_since_recruitment" in fu.columns else pd.Series("nan", index=fu.index)
        yes = hv.isin(["true", "1", "1.0", "yes", "y"])
        out["follow_up_medical"] = {"participants": {k: rc(v) for k, v in fu.groupby("st")["participant_id"].nunique().items()},
                                    "hospitalised_yes": {k: rc(v) for k, v in fu[yes].groupby("st")["participant_id"].nunique().items()},
                                    "answered": {k: rc(v) for k, v in fu[~hv.isin(["nan", "none", "<na>", ""])].groupby("st")["participant_id"].nunique().items()}}
    else:
        out["follow_up_medical"] = {"status": "missing"}
    v1 = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id").set_index("participant_id")
    v2 = fr[fr["research_stage"] == V2].drop_duplicates("participant_id").set_index("participant_id")
    out["lab_pairs"] = {c: nrc(v1[c].reindex(v2.index).notna() & v2[c].notna()) for c in ("hba1c", "glucose", "egfr", "ldl", "acr")
                        if c in v1.columns}
    return out


# ============================================================================ pipeline

def run(root, fr, seed, secs):
    R = {}
    C, base_dates, prev_htn, cinfo = O.load_contacts(root, fr)
    fam = O.load_fam(root)
    ctx = O.prepare(fr, C, base_dates, prev_htn, fam, cinfo, seed)
    ids = set(C["participant_id"].astype(str))
    R["p1_cohort"] = {"at_risk": rc(len(ctx["B"])), "contacts": cinfo.get("nominal_years")}
    guard(R, "repro", lambda: repro_check(ctx))
    thr = lab_thresholds(fr)
    R["lab_thresholds"] = thr
    mrows, crows = med_class_rows(root), cond_group_rows(root)
    if mrows is not None:
        ids |= set(mrows["participant_id"].astype(str))
    if crows is not None:
        ids |= set(crows["participant_id"].astype(str))
    C2, mi = add_flags(ctx["C"], base_dates, mrows, ["m_" + k for k in MED])
    C2, ci = add_flags(C2, base_dates, crows, ["i_" + k for k in ICD])
    R["undefined_rows_mapped"] = {**mi, **ci}
    v2 = fr[fr["research_stage"] == V2].drop_duplicates("participant_id").set_index("participant_id")
    isv2 = C2["kind"] == "v2"
    for src, dst in (("hba1c", "hba1c2"), ("glucose", "glu2"), ("egfr", "egfr2")):
        C2[dst] = np.where(isv2, C2["participant_id"].map(v2[src]) if src in v2.columns else np.nan, np.nan)
    C2["_lab"] = isv2 & ((C2["hba1c2"] >= thr["hba1c"]) | (C2["glu2"] >= thr["glucose"]))
    med0 = {}
    if mrows is not None:
        m0 = mrows[mrows["research_stage"] == BASE].groupby("participant_id")[["m_" + k for k in MED]].max()
        med0 = {k: m0["m_" + k] for k in MED}
    prev = {k: prevalent(crows, base_dates, "i_" + k) for k in ICD}
    extra = {"med0": med0, "prev": prev, "cur": curated_flags(root)}
    # frame additions: IMT without the clinic-day step, calendar-adjusted Sokolow (QC), gaps
    fr = fr.copy()
    fr["imt_nod"] = cal_only(fr, "imt_raw") if "imt_raw" in fr.columns else np.nan
    fr["sok_cal"] = cal_only(fr, "sokolow") if "sokolow" in fr.columns else np.nan
    b0 = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id")
    ref = reference_ids(b0, prev["ihd"] | prev["cereb"] | prev["pad"], extra["cur"].get("ihd", set()), thr)
    try:
        R["G_ref"] = gap_reference(fr, ref)
    except Exception as ex:
        R["G_ref"] = {"error": err(ex)}
        log(f"G_ref FAILED: {R['G_ref']['error']}")
        for bed in GAP_BEDS:
            fr["gz_" + bed] = np.nan
    b, binfo = baseline_table(fr, ctx, extra)
    R["composite"] = binfo
    R["imt_nod_vs_imt_corr"] = float(fr[["imt", "imt_nod"]].corr().iloc[0, 1]) if fr["imt_nod"].notna().sum() > 20 else None
    if "M0" in secs:
        guard(R, "M0", lambda: inventory(root, fr, mrows, crows))
    if "E" in secs:
        E = {}
        for oc in OUTCOMES:
            t1 = time.time()
            guard(E, oc, lambda oc=oc: outcome_block(oc, b, C2, ctx, extra, thr))
            log(f"E {oc} done in {time.time() - t1:.1f}s")
        R["E"] = E
    if "P" in secs:
        guard(R, "P", lambda: run_progression(fr, b))
        log("P done")
    if "G" in secs:
        G = {}
        for nm, col in ORGANS:
            guard(G, nm, lambda col=col, nm=nm: organ_block(fr, b, nm, col))
        R["G_organ"] = G
        L = {}
        for outbed, expos in LAGGED:
            guard(L, outbed, lambda outbed=outbed, expos=expos: lagged_block(fr, outbed, expos))
        R["G_lagged"] = L
        log("G done")
    return R, ids


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
    needed (one line of room is kept for the note); truncate as a last resort."""
    groups = []
    for p, ln, first in wl:
        if first or not groups:
            groups.append([p, [ln]])
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
    if dropped:  # after the three header lines (header continuation lines included)
        pos = sum(len(ls) for _, ls in groups[:3])
        lines.insert(min(pos, len(lines)), f"SHORTENED: {dropped} lower-priority lines dropped; see results.json")
    if len(lines) > MAXLINES:
        lines = lines[:MAXLINES - 1] + [f"DIGEST TRUNCATED: {len(lines)} lines; see results.json"]
    return lines


def ci(c, key="hr"):
    """'v[lo,hi]' from a dict with hr/ratio_hr/b/d and lo/hi."""
    if not isinstance(c, dict):
        return "NA"
    v = c.get(key, c.get("ratio_hr", c.get("b", c.get("d"))))
    if v is None:
        return "NA"
    return f"{v3(v)}[{v3(c.get('lo'))},{v3(c.get('hi'))}]"


def pstr(p):
    return "NA" if p is None else f"{float(p):.2g}"


def hzs(r, x):
    """Bed HR from a hazard result: 'hr[lo,hi]', or the reason it is missing."""
    if not isinstance(r, dict):
        return "NA"
    if r.get("error"):
        return "ERR"
    st = r.get("status")
    if st not in ("ok", "not converged"):
        if str(st).startswith("suppressed"):  # v2.0.1: a group x event cell < 20 in the model sample
            return "NA(cell<20)"
        return "NA(<20)" if st == "too few" else f"NA({str(st)[:14]})"
    return ci(g(r, "hr", x)) + ("*nc" if st == "not converged" else "")


def kvs(d):
    return " ".join(f"{k} {v}" for k, v in (d or {}).items()) if isinstance(d, dict) else "NA"


def dg_m0(M):
    L = []
    if not isinstance(M, dict):
        return L
    if M.get("error"):
        return [(1, f"ERROR M0 {M['error']}")]
    L.append((2, "M0 events stages (participants): " + kvs(M.get("event_stages"))))
    for tab, d in (M.get("curated") or {}).items():
        if isinstance(d, dict) and d.get("status"):
            L.append((3, f"M0 curated {tab}: {d['status']}"))
            continue
        L.append((2, f"M0 curated {tab}: " + " | ".join(f"{st}: {kvs(v)}" for st, v in (d or {}).items())))
    L.append((1, "M0 ICD participants b/c1/v2/c3/und/oth: " + " ".join(
        f"{k} {'/'.join(str(v.get(s)) for s in ('b', 'c1', 'v2', 'c3', 'und', 'oth')) if isinstance(v, dict) else 'NA'}"
        for k, v in (M.get("icd") or {}).items())))
    L.append((1, "M0 ATC participants b/c1/v2/c3/und/oth: " + " ".join(
        f"{k} {'/'.join(str(v.get(s)) for s in ('b', 'c1', 'v2', 'c3', 'und', 'oth')) if isinstance(v, dict) else 'NA'}"
        for k, v in (M.get("atc") or {}).items())))
    fu = M.get("follow_up_medical") or {}
    L.append((1, f"M0 follow-up medical survey participants {kvs(fu.get('participants'))}; answered hospitalisation "
                 f"{kvs(fu.get('answered'))}; yes {kvs(fu.get('hospitalised_yes'))}{' ' + fu['status'] if fu.get('status') else ''}"
                 f" | lab pairs b-v2: {kvs(M.get('lab_pairs'))}"))
    return L


def dg_e(E, prim_only=False):
    L = []
    if not isinstance(E, dict):
        return L
    for oc, r in E.items():
        if oc not in OUTCOMES or not isinstance(r, dict):
            continue
        p1 = 1 if oc in PRIMARY_OC else 2
        if r.get("error"):
            L.append((1, f"E {oc} ERROR {r['error']}"))
            continue
        fl = r.get("flow") or {}
        comp = r.get("first_event_components")
        L.append((1, f"E {oc} cohort {r.get('cohort')} followed {fl.get('followed')} events {fl.get('events')} "
                     f"(c1 {g(fl, 'by_kind', 'call1')} v2 {g(fl, 'by_kind', 'v2')} c3 {g(fl, 'by_kind', 'call3')}) PY "
                     f"{fl.get('person_years')} fu q05/50/95 {v3(g(fl, 'followup_years', 'q05'))}/"
                     f"{v3(g(fl, 'followup_years', 'q50'))}/{v3(g(fl, 'followup_years', 'q95'))} IPW auc "
                     f"{'/'.join(v3(v) for v in (fl.get('ipw_auc') or {}).values())}"
                     + (f" | first-event components {kvs(comp)}" if comp else "")))
        beds = r.get("beds") or {}

        def bl(model, bl_=None):
            parts = []
            for bed in (bl_ or XBEDS[:8]):
                rb = beds.get(bed) or {}
                if rb.get("status"):
                    parts.append(f"{SH[bed]} NA(no bed)")
                    continue
                m = rb.get(model) or {}
                parts.append(f"{SH[bed]} {hzs(m, 'o_' + bed)}" + (f" {m.get('n')}/{m.get('events')}" if model == "B" else ""))
            return " ".join(parts)
        L.append((1, f"E {oc} B: {bl('B')}"))
        L.append((3, f"E {oc} A: {bl('A')}"))
        c3 = r.get("count3") or {}
        cnt = (f"{hzs(c3, 'cnt_1')} / {hzs(c3, 'cnt_2')}" if c3.get("status") in ("ok", "not converged")
               else f"NA({str(c3.get('status'))[:30]})")
        L.append((p1 + 1, f"E {oc} sens B: pwvR {hzs(g(beds, 'pwv_raw', 'B'), 'o_pwv_raw')} imtD "
                          f"{hzs(g(beds, 'imt_nod', 'B'), 'o_imt_nod')} | unweighted pwv {hzs(g(beds, 'pwv', 'Bu'), 'o_pwv')} "
                          f"imt {hzs(g(beds, 'imt', 'Bu'), 'o_imt')} abi {hzs(g(beds, 'abi_min', 'Bu'), 'o_abi_min')} | X(+"
                          f"{','.join(c for c in (r.get('covX') or [])[5:])}) pwv {hzs(g(beds, 'pwv', 'X'), 'o_pwv')} imt "
                          f"{hzs(g(beds, 'imt', 'X'), 'o_imt')} abi {hzs(g(beds, 'abi_min', 'X'), 'o_abi_min')} ret "
                          f"{hzs(g(beds, 'art_w', 'X'), 'o_art_w')} | adverse-bed count 1 / >=2 vs 0: {cnt}"))
        pr = r.get("paired") or {}

        def pline(key):
            x = pr.get(key) or {}
            if x.get("error"):
                return f"{key[:-2]} ERR"
            head = f"{key[:-2]}[{x.get('n')}/{x.get('events')}]"
            if x.get("status") != "ok":
                return f"{head} NA({str(x.get('status'))[:12]})"
            rr = x.get("ratio") or {}
            rat = (" ".join(f"{k} {ci(v)} p{pstr(v.get('p'))}" for k, v in rr.items()) if len(rr) > 1 else
                   " ".join(f"{ci(v)} p{pstr(v.get('p'))}" for v in rr.values()))
            w = x.get("wald")
            return f"{head} " + (f"Wald p{pstr(w.get('p'))} " if isinstance(w, dict) else "") + rat
        L.append((1, f"C {oc} HR ratios B: {pline('S3_B')}"))
        L.append((3, f"C {oc} HR ratios A: {pline('S3_A')}"))
        L.append((p1 + 1, f"C {oc} pairs B: " + " | ".join(pline(f"{SH[a]}/{SH[b_]}_B") for a, b_ in XPAIRS)))
        gp = r.get("gaps") or {}
        gtxt = " ".join(f"{SH[bed]} {hzs(g(gp, bed, 'B'), 'gz_' + bed)}" for bed in GAP_BEDS)
        gpair = r.get("gaps_paired") or {}
        gpt = (f" | paired dlogHR Wald p{pstr(g(gpair, 'wald', 'p'))} " + " ".join(
            f"{k} {ci(v, 'd')}" for k, v in (gpair.get("pairs") or {}).items()) if gpair.get("pairs") else "")
        L.append((p1, f"G {oc} gap HR per ref-SD (B): {gtxt}{gpt}"))
        L.append((3, f"G {oc} gap>=1 vs <1 (B): " + " ".join(f"{SH[bed]} {hzs(g(gp, bed, 'ge1_B'), 'acc_' + bed)}"
                                                              for bed in GAP_BEDS)))
    return L


def coefs_line(models, kind):
    m = (models or {}).get(kind) or {}
    if not m:
        return "NA"
    ns = [v.get("n") for v in m.values() if v.get("n") is not None]
    rng = f"n {min(ns)}-{max(ns)}" if ns else "n NA"
    return rng + ": " + " ".join(f"{k} {v3(v.get('b'))}{'*' if (v.get('q') is not None and float(v['q']) < 0.05) else ''}"
                                 for k, v in m.items())


def dg_p(P):
    L = []
    if not isinstance(P, dict):
        return L
    if P.get("error"):
        return [(1, f"ERROR P {P['error']}")]
    L.append((1, "P = progression b->v2 (atlas v1.5 beds; sbp untreated both visits): pairs, interval q50 y, mean change "
                 "(se) [per y] {SD units} sd, change q05/50/95, ICC, raw (calendar/day/session uncorrected; sok: calendar-adjusted) "
                 "change and ICC, season-matched change corr/raw; models: v2 in baseline-SD units per SD predictor (binary per "
                 "unit), v2-attendance IPW, * BH q<.05"))
    for bed, _ in PROG:
        r = P.get(bed) or {}
        if r.get("error"):
            L.append((1, f"P {SH.get(bed, bed)} ERROR {r['error']}"))
            continue
        pi = r.get("pairs") or {}
        s = r.get("summary")
        pr = PROG_PRIO.get(bed, 2)
        if not s:
            L.append((pr, f"P {SH.get(bed, bed)} pairs {pi.get('pairs')} {pi.get('status', '')}"))
            continue
        raw = s.get("raw") or {}
        sm = s.get("season_matched") or {}
        L.append((1 if pr == 1 else 2,
                  f"P {SH.get(bed, bed)} pairs {s.get('n')} int {v3(g(pi, 'interval_years', 'q50'))} chg {v3(s.get('mean_change'))} "
                  f"({v3(s.get('se'))}) [{v3(s.get('per_year'))}/y] {{{v3(s.get('mean_change_sd_units'))}}} sd {v3(s.get('sd_change'))} "
                  f"q {v3(g(s, 'change_q', 'q05'))}/{v3(g(s, 'change_q', 'q50'))}/{v3(g(s, 'change_q', 'q95'))} ICC {v3(s.get('icc'))}"
                  f" | raw chg {v3(raw.get('mean_change'))} ICC {v3(raw.get('icc'))} | season-matched n {sm.get('n')} "
                  f"{v3(sm.get('mean_change'))}/{v3(raw.get('season_matched_mean_change'))} | IPW auc {v3(pi.get('ipw_auc'))}"))
        L.append((pr, f"P {SH.get(bed, bed)} ANCOVA {coefs_line(r.get('models'), 'ancova')}"))
        L.append((3, f"P {SH.get(bed, bed)} change-score {coefs_line(r.get('models'), 'change')}"))
    sq = P.get("stage_within_quarter") or {}
    L.append((1, "P stage effect v2-b within calendar quarter (quarters with both): " + " ".join(
        f"{SH.get(k, k)} {ci(g(v, 'stage_effect'), 'b')} ({v.get('quarters_with_both')})" if isinstance(v, dict) and v.get("stage_effect")
        else f"{SH.get(k, k)} NA" for k, v in sq.items())))
    sy = P.get("sokolow_by_year") or {}
    L.append((1, "P sok mean by year: " + " | ".join(
        f"{st}: " + " ".join(f"{y} {v3(v.get('mean'))}" for y, v in (d or {}).items()) for st, d in sy.items())))
    return L


def dg_g(R):
    L = []
    ref = R.get("G_ref") or {}
    if isinstance(ref, dict) and ref.get("error"):
        L.append((1, f"ERROR G_ref {ref['error']}"))
    else:
        L.append((1, f"G reference subset n {ref.get('reference_n')} (untreated, no HTN dx, BP<140/90, no glucose-lowering drug, "
                     "HbA1c/glucose below, no ASCVD, non-smoker, BMI 18.5-30); per sex F/M: n, ref SD, age slope/y, years per "
                     "ref-SD; baseline gap z q05/50/95; share gap>=1"))
        for bed in GAP_BEDS:
            r = ref.get(bed) or {}
            f_, m_ = r.get("F") or {}, r.get("M") or {}
            gz = r.get("gap_z_baseline") or {}
            ge = r.get("gap_ge1_baseline") or {}
            gy = r.get("gap_years_baseline")
            L.append((2, f"G ref {SH[bed]}: n {f_.get('n_ref')}/{m_.get('n_ref')} sd {v3(f_.get('sd_ref'))}/{v3(m_.get('sd_ref'))} "
                         f"slope {v3(f_.get('slope_per_year'))}/{v3(m_.get('slope_per_year'))} yrs/SD "
                         f"{v3(f_.get('years_per_sd'))}/{v3(m_.get('years_per_sd'))} | gap z {v3(gz.get('q05'))}/"
                         f"{v3(gz.get('q50'))}/{v3(gz.get('q95'))} >=1 {ge.get('ge1')}/{ge.get('n')}"
                         + (f" | gap years q05/50/95 {v3(gy.get('q05'))}/{v3(gy.get('q50'))}/{v3(gy.get('q95'))}"
                            if isinstance(gy, dict) and gy.get("q50") is not None else "")))
    G = R.get("G_organ") or {}
    if G:
        L.append((1, "O = organ decline: v2 organ (baseline-SD units) per ref-SD gap, adjusted for baseline organ, interval, "
                     "age, sex; v2-attendance IPW; sbp untreated both visits; paired = stacked contrasts of core gaps"))
    for nm, r in G.items():
        if not isinstance(r, dict) or nm == "timings":
            continue
        if r.get("error"):
            L.append((1, f"O {nm} ERROR {r['error']}"))
            continue
        gg = r.get("gap") or {}
        if not gg:
            L.append((1, f"O {nm} pairs {g(r, 'pairs', 'pairs')}: {g(r, 'pairs', 'status')}"))
            continue
        L.append((1, f"O {nm} pairs {g(r, 'pairs', 'pairs')}: " + " ".join(
            f"{SH[k]} {ci(v, 'b') if 'b' in v else 'NA(<' + str(MIN_N) + ')'}" for k, v in gg.items())))
        pp = r.get("paired")
        if isinstance(pp, dict):
            L.append((2, f"O {nm} paired n {pp.get('n')} " + (f"Wald p{pstr(g(pp, 'wald', 'p'))} " if pp.get("wald") else "")
                         + " ".join(f"{k} {ci(v, 'd')} p{pstr(v.get('p'))}" for k, v in (pp.get("pairs") or {}).items())))
    LG = R.get("G_lagged") or {}
    for ob, r in LG.items():
        if not isinstance(r, dict) or ob == "timings":
            continue
        if r.get("error"):
            L.append((2, f"L {ob} ERROR {r['error']}"))
            continue
        L.append((2, f"L {SH.get(ob, ob)} first at v2 n {r.get('n_new_at_v2')} (v2 adverse z per ref-SD gap; lag = baseline "
                     "gap, con = v2 gap): " + " | ".join(
                         f"{SH[ex]} n {v.get('n')} lag {ci(v.get('lag'), 'b')} con {ci(v.get('con'), 'b')} diff p"
                         f"{pstr(g(v, 'lag_minus_con', 'p'))}" if isinstance(v, dict) and v.get("lag") else f"{SH[ex]} NA"
                         for ex, v in r.items() if ex in SH)))
    return L


def digest(R):
    mods = R.get("modules") or {}
    rp = R.get("repro") or {}
    um = {k: v for k, v in (R.get("undefined_rows_mapped") or {}).items() if v}
    L = [(1, f"F1 ATLAS OUTCOMES v2 DIGEST {R.get('version')} | atlas {R.get('atlas_version')} {mods.get('atlas_sha16')} "
             f"{'ok' if mods.get('atlas_ok') else 'DIFF'} | outcomes {mods.get('outcomes_version')} {mods.get('outcomes_sha16')} "
             f"{'ok' if mods.get('outcomes_ok') else 'DIFF'} | seed {R.get('seed')} | sections {','.join(R.get('sections') or [])} "
             f"| runtime {v3(R.get('runtime_s'))}s"),
         (1, "counts rounded to 10, None = <20; HR per SD of the adverse-oriented bed (ret = artery width and abi per SD "
             "lower), P1 hazard: cloglog over contacts, contact intercepts, log(interval) offset, IPW refitted per outcome, "
             "robust SE; A = age sex; B = A BMI usual SBP/DBP; X = B + extras; n/events of B after the CI; NA(<20) = "
             "fewer than 20 events or n < 200"),
         (1, f"check: liver tables read 0 (refused {len(R.get('liver_blocked') or [])}); frame rows {R.get('frame_rows')}; "
             f"P1 cohort {g(R, 'p1_cohort', 'at_risk')}; repro P1 O8 objective-only pwv B {ci(rp.get('objective_composite_weights_pwv_B'))} "
             f"vs OUT L82 1.04[0.812,1.34] {'ok' if rp.get('match_3sd') else 'DIFF'}; labs thr {kvs(R.get('lab_thresholds'))}; "
             f"3-bed n {g(R, 'composite', 'n_all3')} count 0/1/>=2 {g(R, 'composite', 'count_0')}/{g(R, 'composite', 'count_1')}/"
             f"{g(R, 'composite', 'count_ge2')}; imtD-imt r {v3(R.get('imt_nod_vs_imt_corr'))}"
             + (f"; undefined-stage rows mapped {kvs(um)}" if um else ""))]
    if R.get("fatal"):
        L.append((1, f"FATAL {R['fatal']}"))
    L += dg_m0(R.get("M0"))
    L += dg_e(R.get("E"))
    L += dg_g(R)
    L += dg_p(R.get("P"))
    L.append((3, "timings s: " + " ".join(f"{k} {v3(v)}" for k, v in (R.get("timings") or {}).items())))
    return "\n".join(shrink(wrap(L))) + "\n"


# ============================================================================ main

SECTIONS = ("M0", "E", "P", "G")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="HPP dataset root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", default=os.path.join(HERE, "f1_atlas_v1_5.py"))
    ap.add_argument("--outcomes", default=os.path.join(HERE, "f1_outcomes_v1.py"))
    ap.add_argument("--seed", type=int, default=20261004, help="P1 seed (usual-BP cross-fitting folds); keep 20261004")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    for p in (a.atlas, a.outcomes):
        if not os.path.exists(p):
            raise SystemExit(f"module not found: {p}")
    load_modules(a.atlas, a.outcomes)
    t0 = time.time()
    secs = [s.strip().upper() for s in a.sections.split(",") if s.strip()]
    sa, so = sha16_file(a.atlas), sha16_file(a.outcomes)
    mods = {"atlas_sha16": hfmt(sa), "outcomes_sha16": hfmt(so), "outcomes_version": vq(O.VERSION),
            "atlas_ok": sa == EXPECT["atlas"], "outcomes_ok": so == EXPECT["outcomes"]}
    log(f"{VERSION}; atlas {vq(A.VERSION)} {mods['atlas_sha16']}; outcomes {vq(O.VERSION)} {mods['outcomes_sha16']}; numpy "
        f"{np.__version__}; pandas {pd.__version__}; python {sys.version.split()[0]}")
    if not (mods["atlas_ok"] and mods["outcomes_ok"]):
        log("WARNING atlas or outcomes module is not the expected version (digest header shows DIFF)")
    res = {"version": VERSION, "atlas_version": vq(A.VERSION), "modules": mods, "seed": f"{a.seed:,}", "sections": secs}
    pids = set()
    try:
        fr = A.build_frame(a.root, None)[0].reset_index(drop=True)
        res["frame_rows"] = rc(len(fr))
        pids |= set(fr["participant_id"].astype(str))
        R, ids = run(a.root, fr, a.seed, secs)
        pids |= ids
        res.update(R)
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
        dg = f"F1 ATLAS OUTCOMES v2 DIGEST {VERSION}\nDIGEST FAILED: {err(ex)}; see results.json\n"
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
