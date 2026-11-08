#!/usr/bin/env python3
"""F1 phenome-wide acquisition audit v1 (package P8; HPP sandbox; writes aggregates only).

For every timestamped, non-liver HPP measurement table (found from the dataset folders, their data dictionaries and the
parquet schemas) and every continuous variable in it (one unit per participant x main visit; dated draws for tables
without visits), this module measures the hidden acquisition structure that the atlas found in leg PWV, with the
atlas's own machinery (atlas v1.5 imported by path, --atlas; it is not changed):
  calendar    R2 of calendar-quarter fixed effects given age, (age-50)^2 and sex (the A.calendar_adjust estimator), the
              R2 of a one-harmonic season term, and sub-quarter drift c17 = covariance of different, unrelated units
              measured 1-7 calendar days apart (after the quarter adjustment), / V;
  day         clinic-day variance share (A.day_adjust ANOVA moments) and the independent clinic-day component
              dind = c0 - c17 (c0 = same-day covariance of different, unrelated units, / V);
  within-day  the same-day unrelated-pair time-lag kernel b0 + s exp(-dt/tau) (A.kernel_fit replica on binned sufficient
              statistics: share s/V, day b0/V, tau 5-180 min, jackknife z over 20 day groups) and a decay statistic
              dec = (mean product of pairs <= 30 min apart - pairs >= 120 min apart) / V with its jackknife SE.
  hour curve  (annotation H, not a class) population clock-hour curve: share of the residual variance explained by two
              harmonics of the local clock hour (24 h, 12 h) given age, (age-50)^2, sex and quarter, Wald test with a
              leave-one-day-out jackknife (physiology or booking by time of day; the K placebo keeps it, so K needs a
              day-specific process).  Visit-mix diagnostic for C: the quarter R2 among first-visit units only.
Placebos (--placebo replicates of each): (i) whole days permuted across the calendar (null of the quarter R2 under day
clustering); (ii) day labels shuffled within quarter, units keep their clock time (null of drift, day and within-day
statistics; a population clock-hour curve survives it).  Flags, BH within the phenotype and the technical family:
  C  quarter R2 (scaled-chi2 fit to placebo i) or drift c17 (jackknife z);  D  dind (jackknife z);
  K  dec (jackknife z) and the atlas kernel rule (s > 0, z >= 3, kernel reaching 0.1 in a fitted bin);
each also needs q < 0.05, the statistic above every placebo replicate and an excess over the placebo mean >= 0.01 (C) /
0.02 (D, K).  Class = the finest flagged scale (none / calendar / day / within-day).  Rep 0 of the placebos is also
classified against the others ("placebo-as-real" false flags).
Corrections (atlas order and estimators): quarter effects -> leave-one-out day BLUP (A.day_adjust) -> A.krige_day kernel
kriging (A.session_adjust).  "Flag-guided" applies only the flagged components (drift is removed by the day BLUP);
"universal" applies the atlas pipeline to every variable.  Per variable and correction: removed variance share,
sample-size inflation factor of the uncorrected endpoint (parallel: variance ratio of age/sex residuals; change score:
variance ratio of the 2-visit change), retest ICC before/after, and the change of the standardized reference
associations with age, sex, BMI and sitting SBP (joint model, one unit per participant) and of their SEs.
Clinical block (atlas frame, raw vs v1.5): leg PWV, IMT, retinal artery width, ABI: reclassification across the adverse
top quintile/decile (raw and age/sex-standardized ranks), sample-size inflation (parallel, change, model-based),
participant bootstrap.  Control block: vascular-device pressures vs the separate sitting/lying/standing cuff, height,
BMI, ECG heart rate; same-day stranger cross-correlations of PWV, device SBP, cuff SBP and cuff HR.
Checks: the P8 machinery rebuilds atlas v1.5 PWV and ABI from raw; the binned kernel fit equals A.kernel_fit.
Exclusions: every liver table and field (liver ultrasound; liver enzymes and bilirubin; A.rd refuses liver paths, so the
atlas frame is built without liver attenuation); questionnaires, registers, derived phenotypes, logs and genotypes.
Privacy (as the atlas): aggregates only; counts rounded to 10, cells < 20 suppressed, pair counts >= 1e6 as text;
quantiles 5-95 % only; no participant ids, dates or per-day values; every output is scanned for 7+ digit runs and ids.
Run (sandbox pheno env, from <WORK_DIR>):
  python f1_acq_audit_v1.py --root <HPP_DATA_ROOT> \
      --atlas f1_atlas_v1_5.py --out f1_acq_audit_out_v1 --workers 4
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # many small problems + table workers
    os.environ.setdefault(_v, "2")
import argparse
import glob
import hashlib
import importlib.util
import json
import multiprocessing as mp
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse, stats

VERSION = "f1_acq_audit_v1.1 2026-10-05"
A = None  # atlas module (set_atlas)
CTX = {}  # shared context for the table workers (set before forking)
LONG = re.compile(r"\d{7,}")
ATLAS_API = ["build_frame", "rd", "rc", "rcp", "cnt", "qtiles", "fnum", "log", "LOG", "MINCELL", "MAIN_STAGES", "TZ",
             "VERSION", "CAL_BEDS", "calendar_quarter", "kernel_fit", "krige_day", "relation_components", "icc1",
             "one_row_per_participant", "ORIENT", "bh", "DIAG", "SESSION_EDGES", "SESSION_TAUS", "SESSION_JK",
             "SESSION_MIN_BIN", "SESSION_MIN_Z", "SESSION_MAX_NB", "beta_only", "CAL_ADJUST", "DAY_ADJUST",
             "SESSION_ADJUST"]
LIVER_TABLE_RX = re.compile(r"liver|hepat|mafld|nafld|masld|steato", re.I)
LIVER_FIELD_RX = re.compile(r"liver|hepat|mafld|nafld|masld|steato|fib4|(^|_)(alt|ast|ggt|alp)(_|$)|alkaline_phosph|"
                            r"bilirubin|alanine_amino|aspartate_amino|gamma_glut", re.I)
CATEGORY = {"vascular_health": "clinic", "blood_pressure": "clinic", "anthropometrics": "clinic", "hand_grip": "clinic",
            "ecg": "clinic", "carotid_ultrasound": "clinic", "fundus": "clinic", "body_composition": "clinic",
            "bone_density": "clinic", "sleep": "home", "cgm": "home", "blood_tests": "lab",
            "nightingale_metabolomics": "lab", "serum_lipidomics": "lab", "rna_seq": "tech", "gut_microbiome": "tech",
            "oral_microbiome": "tech"}
EXCLUDED = {"liver_ultrasound": "liver", "curated_phenotypes": "derived", "events": "register", "population": "register",
            "participant_relation": "register", "medications": "register", "medical_conditions": "questionnaire",
            "medical_procedures": "questionnaire", "medical_symptoms": "questionnaire",
            "health_and_medical_history": "questionnaire", "family_history": "questionnaire",
            "lifestyle_and_environment": "questionnaire", "psychological_and_social_health": "questionnaire",
            "sociodemographics": "questionnaire", "sex_specific_factors": "questionnaire", "children": "questionnaire",
            "covid": "questionnaire", "diet_logging": "self-log", "human_genetics": "genotype", "truth": "not-hpp",
            "metadata": "metadata"}
TS_DONOR = {"cgm/iglu.parquet": "cgm/cgm.parquet", "fundus/microvasculature.parquet": "fundus/fundus.parquet",
            "sleep/hrv.parquet": "sleep/sleep.parquet", "ecg/ecg_qc.parquet": "ecg/ecg.parquet"}
SKIP = {"cgm/iglu_daily.parquet": "per-wear-day", "nightingale_metabolomics/tags_per_biomarker.parquet": "qc-tags",
        "oral_microbiome/oral_microbiome_2025_04_UUID.parquet": "duplicate", "human_genetics/variants_qc.parquet": "variants"}
ABBR = {"vascular_health/vascular_health.parquet": "vas", "blood_pressure/blood_pressure.parquet": "bp",
        "anthropometrics/anthropometrics.parquet": "anth", "hand_grip/hand_grip.parquet": "grip", "ecg/ecg.parquet": "ecg",
        "ecg/ecg_qc.parquet": "ecgqc", "carotid_ultrasound/carotid_ultrasound.parquet": "car",
        "fundus/fundus.parquet": "fund", "fundus/microvasculature.parquet": "ret",
        "body_composition/body_composition.parquet": "dxa", "bone_density/bone_density.parquet": "bmd",
        "sleep/sleep.parquet": "slp", "sleep/hrv.parquet": "hrv", "cgm/cgm.parquet": "cgmc", "cgm/iglu.parquet": "cgm",
        "blood_tests/blood_tests.parquet": "lab", "nightingale_metabolomics/nightingale_metabolomics.parquet": "nmr",
        "serum_lipidomics/serum_lipidomics.parquet": "lip", "rna_seq/rna_seq.parquet": "rna",
        "gut_microbiome/gut_microbiome.parquet": "gut", "oral_microbiome/oral_microbiome.parquet": "oral"}
DERIVED = {"vascular_health/vascular_health.parquet": [
    ("pwv", ("from_r_thigh_to_r_ankle_pwv", "from_l_thigh_to_l_ankle_pwv"), "mean"),
    ("abi_min", ("r_abi", "l_abi"), "min"), ("dev_bsbp", ("r_brachial_pressure", "l_brachial_pressure"), "mean"),
    ("dev_asbp", ("r_ankle_pressure", "l_ankle_pressure"), "mean")],
    "carotid_ultrasound/carotid_ultrasound.parquet": [("imt", ("imt_left", "imt_right"), "mean")],
    "fundus/microvasculature.parquet": [("art_w", ("artery_average_width_left", "artery_average_width_right"), "mean"),
                                        ("vein_w", ("vein_average_width_left", "vein_average_width_right"), "mean")]}
MEDIAN_FOLDERS = ("sleep",)  # several nights per visit: median (atlas)
KEYS = ("participant_id", "cohort", "research_stage", "array_index", "connection_id")
NONVAR_RX = re.compile(r"^(participant_id|cohort|research_stage|array_index|connection_id|timezone|night|study_id|"
                       r"year_of_birth|month_of_birth|sample_id|sample_name|batch|pool|plate|well|hmo)$|timestamp|"
                       r"_date$|^date|_sn$|serial|_id$", re.I)  # numeric keys, codes and clocks (paths are not numeric)
TECH_RX = re.compile(r"(^|_)qc|quality|snr|_check|read_count|reads|fastqc|cutadapt|ercc|preprocessing|duplicat|mapped|"
                     r"wgs_compare|datapoints_lost|warning|_dose|scale_resolution|centre_|window_width|imt_fit|detected_",
                     re.I)
MIN_UNITS = 300
MIN_LEVELS = 10
MAX_VARS = 400
MAX_PAIRS = 600_000
Q_ALPHA = 0.05
MIN_EX = {"C": 0.01, "D": 0.02, "K": 0.02}  # minimum excess over the placebo mean (variance shares) for a flag
MIN_H = 0.01  # hour-curve annotation H: share of the residual variance (and BH q < 0.05, day-jackknife Wald test)
TS_LO, TS_HI = pd.Timestamp("1900-01-01"), pd.Timestamp("2200-01-01")  # outside: NaT (sentinels such as 9999-12-31)
DEC_SHORT, DEC_LONG = 30.0, 120.0  # decay test: pairs <= 30 min apart vs >= 120 min apart (same day)
NB_LAG = 7  # neighbouring-day covariance: different units 1-7 calendar days apart
IDENT_MIN = 0.1  # kernel identified only if exp(-dt/tau) reaches 0.1 in a used bin (else s is an extrapolation)
T0 = pd.Timestamp("2018-01-01", tz="UTC")
REF = ("age", "sex", "bmi", "sbp")
CLIN = [("pwv", "vas:pwv"), ("imt", "car:imt"), ("art_w", "ret:art_w"), ("abi_min", "vas:abi_min")]
CTRL = ["vas:dev_bsbp", "vas:r_brachial_pressure", "vas:l_brachial_pressure", "vas:dev_asbp", "vas:pwv", "vas:abi_min",
        "bp:sitting_blood_pressure_systolic", "bp:sitting_blood_pressure_diastolic", "bp:sitting_blood_pressure_pulse_rate",
        "bp:sitting_second_blood_pressure_systolic", "bp:lying_blood_pressure_systolic",
        "bp:standing_one_min_blood_pressure_systolic", "anth:height", "anth:bmi", "ecg:hr_bpm"]
SECTIONS = ("AUDIT", "CHECK", "CLIN", "XR")
CLASSES = ("none", "calendar", "day", "within-day")


# ============================================================================ atlas, logging, privacy

def set_atlas(path):
    """Import the atlas by path; A.rd refuses liver tables (so the atlas frame is built without liver attenuation)."""
    global A
    spec = importlib.util.spec_from_file_location("f1_atlas", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["f1_atlas"] = m
    spec.loader.exec_module(m)
    miss = [x for x in ATLAS_API if not hasattr(m, x)]
    if miss:
        raise SystemExit(f"atlas module lacks {miss}; use atlas v1.5")
    orig = m.rd

    def rd_noliver(root, rel, columns=None):
        if LIVER_TABLE_RX.search(rel):
            m.log(f"liver table not read (F1 excludes liver): {rel.split('/')[0]}")
            return None
        return orig(root, rel, columns)
    m.rd = rd_noliver
    A = m
    return m


def log(msg, buf=None):
    if buf is not None:
        buf.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
    else:
        A.log(msg)


def privacy_check(text, pids):
    m = LONG.search(text)
    if m:
        raise SystemExit(f"PRIVACY ABORT: digit run '{m.group(0)[:3]}...' in output")
    for p in pids:  # ids of 7+ digits are caught above
        if not (p.isdigit() and len(p) >= 7) and len(p) >= 4 and p in text:
            raise SystemExit("PRIVACY ABORT: participant id in output")


DATE_RX = re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?([+-]\d{2}:?\d{2}|Z)?")


def scrub(s):
    """Free text from exceptions (it can quote data values): dates, clock times and digit runs >= 4 are masked."""
    s = DATE_RX.sub("<date>", str(s))
    s = re.sub(r"(?<!\d)\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?", "<time>", s)
    return re.sub(r"\d{4,}", "#", s)


def err(ex):
    tb = traceback.extract_tb(ex.__traceback__)
    ln = tb[-1].lineno if tb else -1
    return f"{type(ex).__name__}: {scrub(str(ex)[:240])[:160]} (line {ln})"


def rc(n):
    return A.rc(n)


def fl(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def q595(x):
    """median [q05, q95] of a list of statistics across variables (not participant data)."""
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(a) < 5:
        return {"nv": len(a)}
    return {"nv": len(a), "q50": float(np.median(a)), "q05": float(np.quantile(a, 0.05)),
            "q95": float(np.quantile(a, 0.95))}


def mask_name(s):
    return re.sub(r"\d{5,}", "#", str(s))


# ============================================================================ discovery

def read_dicts(folder_path):
    out = []
    for f in sorted(glob.glob(os.path.join(folder_path, "metadata", "*_data_dictionary.csv"))):
        try:
            d = pd.read_csv(f, usecols=lambda c: c in ("tabular_field_name", "field_type", "relative_location",
                                                       "pandas_dtype"))
            out.append(d)
        except Exception:
            pass
    return pd.concat(out, ignore_index=True) if out else None


def numeric_field(f):
    t = f.type
    return (pa.types.is_integer(t) or pa.types.is_floating(t)) and not pa.types.is_boolean(t)


def discover(root, only=None):
    """Included tables (with their candidate variables) and excluded tables (with reasons)."""
    inc, exc, liver_fields = [], [], 0
    for folder in sorted(os.listdir(root)):
        fp = os.path.join(root, folder)
        if not os.path.isdir(fp) or folder.startswith("."):
            continue
        rels = sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(fp, "*.parquet")))
        if not rels:
            continue
        if LIVER_TABLE_RX.search(folder):
            exc += [(r, "liver") for r in rels]
            continue
        if folder in EXCLUDED:
            exc += [(r, EXCLUDED[folder]) for r in rels]
            continue
        dd = read_dicts(fp)
        for rel in rels:
            if only and rel not in only and ABBR.get(rel) not in only:
                continue
            if LIVER_TABLE_RX.search(rel):
                exc.append((rel, "liver"))
                continue
            if rel in SKIP:
                exc.append((rel, SKIP[rel]))
                continue
            try:
                sch = pq.read_schema(os.path.join(root, rel))
            except Exception as ex:
                exc.append((rel, f"unreadable {type(ex).__name__}"))
                continue
            names = sch.names
            own = [c for c in ("collection_timestamp", "collection_date") if c in names]
            donor = TS_DONOR.get(rel) if not own else None
            if not own and not (donor and os.path.exists(os.path.join(root, donor))):
                exc.append((rel, "no-acq-time"))
                continue
            ft = {}
            if dd is not None and "relative_location" in dd.columns:
                sub = dd[dd["relative_location"].astype(str) == rel]
                ft = dict(zip(sub["tabular_field_name"].astype(str), sub["field_type"].astype(str)))
            cand = []
            for f in sch:
                c = f.name
                if c in KEYS or NONVAR_RX.search(c) or not numeric_field(f):
                    continue
                if c in ft and ft[c] not in ("Continuous", "Integer"):
                    continue
                if LIVER_FIELD_RX.search(c):
                    liver_fields += 1
                    continue
                cand.append(c)
            der = [d_ for d_ in DERIVED.get(rel, []) if all(s in names for s in d_[1])]
            if not cand and not der:
                exc.append((rel, "no-numeric"))
                continue
            inc.append({"rel": rel, "folder": folder,
                        "tid": ABBR.get(rel, mask_name(f"{folder[:8]}.{os.path.basename(rel)[:-8][:8]}")),
                        "cat": CATEGORY.get(folder, "other"), "donor": donor, "vars": cand, "derived": der,
                        "own_time": own})
    return inc, exc, liver_fields


# ============================================================================ units (one per participant x visit)

def parse_ts(s):
    """-> UTC tz-aware datetime64[ns]; naive timestamps are local (Asia/Jerusalem); numeric = epoch (unit guessed)."""
    if s is None:
        return None
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        x = pd.to_numeric(s, errors="coerce").astype(float)
        med = float(np.nanmedian(np.abs(x))) if x.notna().any() else np.nan
        unit = ("s" if 1e8 < med < 1e10 else "ms" if 1e11 < med < 1e13 else "us" if 1e14 < med < 1e16
                else "ns" if 1e17 < med < 1e19 else None)
        if unit is None:
            return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
        return bounded(pd.to_datetime(x, unit=unit, utc=True, errors="coerce")).dt.as_unit("ns")
    t = bounded(pd.to_datetime(s, errors="coerce"))
    if t.dtype == object:
        t = bounded(pd.to_datetime(s, errors="coerce", utc=True))
    if getattr(t.dt, "tz", None) is None:
        t = t.dt.tz_localize(A.TZ, ambiguous="NaT", nonexistent="NaT")
    return t.dt.tz_convert("UTC").dt.as_unit("ns")


def bounded(t):
    """NaT outside 1900-2200 (any unit; a sentinel such as 9999-12-31 in datetime64[us] cannot be held in ns)."""
    if not pd.api.types.is_datetime64_any_dtype(t):
        return t
    tz = getattr(t.dt, "tz", None)
    lo, hi = (TS_LO.tz_localize("UTC"), TS_HI.tz_localize("UTC")) if tz is not None else (TS_LO, TS_HI)
    return t.where((t >= lo) & (t < hi))


def local_date(s):
    t = bounded(pd.to_datetime(s, errors="coerce"))
    if t.dtype == object:
        t = bounded(pd.to_datetime(s, errors="coerce", utc=True))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert(A.TZ).dt.tz_localize(None)
    return t.dt.as_unit("ns").dt.normalize()


def load_units(root, spec, buf):
    rel = spec["rel"]
    names = pq.read_schema(os.path.join(root, rel)).names
    dsrc = [s for d_ in spec["derived"] for s in d_[1]]
    read = list(dict.fromkeys(spec["vars"] + [c for c in dsrc if c in names] + spec["own_time"]))
    d = A.rd(root, rel, columns=read)
    if d is None:
        raise RuntimeError("table not readable")
    tsrc = "own"
    if spec.get("donor"):
        dn_names = pq.read_schema(os.path.join(root, spec["donor"])).names
        tc = [c for c in ("collection_timestamp", "collection_date") if c in dn_names]
        dn = A.rd(root, spec["donor"], columns=tc)
        # visit-level match (participant, cohort, stage): the earliest donor time of the visit, so that sibling rows
        # need not share the donor's array or connection indices (a unit is a visit and takes its earliest time anyway)
        kk = [k for k in ("participant_id", "cohort", "research_stage") if k in d.columns and k in dn.columns]
        for k in kk:
            d[k] = d[k].astype(str)
            dn[k] = dn[k].astype(str)
        if "collection_timestamp" in tc:
            dn["collection_timestamp"] = parse_ts(dn["collection_timestamp"])
        if "collection_date" in tc:
            dn["collection_date"] = local_date(dn["collection_date"])
        dn = dn[kk + tc].groupby(kk, sort=False, dropna=False)[tc].min().reset_index()
        d = d.drop(columns=[c for c in tc if c in d.columns]).merge(dn, on=kk, how="left")
        tsrc = "donor"
    st = d["research_stage"].astype(str)
    main = st.isin(A.MAIN_STAGES)
    mode = "visit"
    if main.sum() < MIN_UNITS and (st == "undefined").sum() >= MIN_UNITS:
        mode = "dated"
        d = d[(st == "undefined").to_numpy()].copy()
    else:
        d = d[main.to_numpy()].copy()
    if not len(d):
        raise RuntimeError("no rows at the main visits")
    ts = parse_ts(d["collection_timestamp"]) if "collection_timestamp" in d.columns else \
        pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")
    ld = local_date(ts)
    if "collection_date" in d.columns:
        ld = ld.fillna(local_date(d["collection_date"]))
    if ld.isna().all():
        raise RuntimeError("no acquisition dates")
    for c in set(spec["vars"]) | set(dsrc):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    for nm, srcs, how in spec["derived"]:
        X = d[list(srcs)]
        d[nm] = X.mean(axis=1, skipna=True) if how == "mean" else X.min(axis=1, skipna=True)
    vcols = list(dict.fromkeys([nm for nm, _, _ in spec["derived"]] + [c for c in spec["vars"] if c in d.columns]))
    w = pd.DataFrame({"participant_id": d["participant_id"].astype(str).to_numpy(),
                      "research_stage": d["research_stage"].astype(str).to_numpy(),
                      "_ts": ts.reset_index(drop=True), "_ld": ld.reset_index(drop=True)})
    w = pd.concat([w, d[vcols].reset_index(drop=True)], axis=1)
    keys = ["participant_id", "research_stage"] + (["_ld"] if mode == "dated" else [])
    if mode == "dated":
        w = w[w["_ld"].notna()]
    how = "median" if spec["folder"] in MEDIAN_FOLDERS else "mean"
    g = w.groupby(keys, sort=True)
    u = g[vcols].agg(how)
    u["_ts"] = g["_ts"].min()
    if mode == "visit":
        u["_ld"] = g["_ld"].min()
    u = u.reset_index()
    u = u[u["_ld"].notna()].reset_index(drop=True)
    u["_ts"] = pd.to_datetime(u["_ts"], utc=True)
    return u, vcols, mode, tsrc


def build_context(u, vcols, mode, buf):
    """Unit arrays (time, day, quarter, covariates) and the transformed variable matrix."""
    n = len(u)
    T = {"n": n, "mode": mode}
    pid = u["participant_id"].astype(str).to_numpy()
    T["pid"] = pid
    T["pcode"] = pd.factorize(pid)[0]
    comp = CTX.get("comp")
    T["cid"] = pd.factorize(np.array([comp.get(p, p) for p in pid] if comp else pid, dtype=object))[0]
    T["stage"] = u["research_stage"].astype(str).to_numpy()
    ts = u["_ts"]
    lt = ts.dt.tz_convert(A.TZ)
    clk = (lt.dt.hour * 60.0 + lt.dt.minute + lt.dt.second / 60.0 + lt.dt.microsecond / 6e7).to_numpy(float)
    has = ts.notna().to_numpy() & np.isfinite(clk) & (clk > 0)
    timed = bool(n and has.mean() >= 0.5 and len(np.unique(np.round(clk[has]))) >= 20)
    tm = ((ts - T0).dt.total_seconds() / 60.0).to_numpy(float)
    tm[~has] = np.nan
    if not timed:
        tm[:] = np.nan
    T["timed"], T["tm"] = timed, tm
    T["clk"] = np.where(np.isfinite(tm), clk, np.nan)
    ldt = pd.to_datetime(u["_ld"])
    day = ldt.dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
    T["dcode"] = pd.factorize(day)[0]
    T["dnum"] = ((ldt - pd.Timestamp("2018-01-01")).dt.days).to_numpy(np.int64)
    dday = np.zeros(T["dcode"].max() + 1, np.int64)
    dday[T["dcode"]] = T["dnum"]
    T["dday"] = dday
    qs = (ldt.dt.year.astype(int).astype(str) + "Q" + ldt.dt.quarter.astype(int).astype(str)).to_numpy(dtype=object)
    T["qcode"] = pd.factorize(qs)[0]
    nd = T["dcode"].max() + 1
    qd = np.zeros(nd, int)
    qd[T["dcode"]] = T["qcode"]
    T["qday"] = qd
    doy = ldt.dt.dayofyear.to_numpy(float)
    T["season"] = np.column_stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
    # covariates: age (events via the atlas frame; dated units: nearest main visit within 180 d, else birth year),
    # sex (population), BMI and sitting SBP at the anchor visit (reference associations)
    an = CTX["anchor"]
    if mode == "visit":
        m = u[["participant_id", "research_stage"]].merge(an, how="left", on=["participant_id", "research_stage"])
        age = m["age"].to_numpy(float)
        bmi, sbp = m["bmi"].to_numpy(float), m["sbp_sit"].to_numpy(float)
    else:
        left = pd.DataFrame({"participant_id": pid, "_ld": ldt.to_numpy(), "_i": np.arange(n)}).sort_values("_ld")
        right = an.dropna(subset=["visit_date"]).sort_values("visit_date")
        m = pd.merge_asof(left, right, left_on="_ld", right_on="visit_date", by="participant_id", direction="nearest",
                          tolerance=pd.Timedelta(days=180)).sort_values("_i")
        age = (m["age"] + (m["_ld"] - m["visit_date"]).dt.days / 365.25).to_numpy(float)
        bmi, sbp = m["bmi"].to_numpy(float), m["sbp_sit"].to_numpy(float)
    yob = pd.Series(pid).map(CTX["yob"]).to_numpy(float)
    fb = ~np.isfinite(age) & np.isfinite(yob)
    age[fb] = (ldt.dt.year.to_numpy(float) + (ldt.dt.dayofyear.to_numpy(float) - 0.5) / 365.25 - yob - 0.5)[fb]
    sex = pd.Series(pid).map(CTX["sex"]).to_numpy(float)
    T["age"], T["sex"], T["bmi"], T["sbp"] = age, sex, bmi, sbp
    T["X0"] = np.column_stack([np.ones(n), age, (age - 50.0) ** 2, sex])
    T["okc"] = np.isfinite(age) & np.isfinite(sex) & (T["dcode"] >= 0)
    # one unit per participant for the reference associations: per variable, the first eligible unit (value, age, sex,
    # BMI and SBP present) in the order first main visit, then date (dated draws: the earliest draw near a visit)
    T["order"] = np.lexsort((ldt.to_numpy(), np.where(T["stage"] == A.MAIN_STAGES[0], 0, 1), T["pcode"]))
    if mode == "visit":
        i0 = pd.Series(np.flatnonzero(T["stage"] == A.MAIN_STAGES[0]), index=pid[T["stage"] == A.MAIN_STAGES[0]])
        i1 = pd.Series(np.flatnonzero(T["stage"] == A.MAIN_STAGES[1]), index=pid[T["stage"] == A.MAIN_STAGES[1]])
        both = i0.index.intersection(i1.index)
        T["ret0"], T["ret1"] = i0[both].to_numpy(int), i1[both].to_numpy(int)
    else:
        T["ret0"] = T["ret1"] = np.zeros(0, int)
    # variables: >= MIN_UNITS values, >= MIN_LEVELS levels; log if positive and skew > 2; winsorised 0.5/99.5 %
    Y, names, tf, drop = [], [], [], {"few": 0, "levels": 0, "const": 0, "cap": 0}
    for c in vcols:
        x = u[c].to_numpy(float).copy()
        f = np.isfinite(x)
        if f.sum() < MIN_UNITS:
            drop["few"] += 1
            continue
        if len(np.unique(x[f])) < MIN_LEVELS:
            drop["levels"] += 1
            continue
        t_ = "id"
        if x[f].min() > 0 and stats.skew(x[f]) > 2:
            x = np.log(x)
            t_ = "log"
        lo, hi = np.quantile(x[f], [0.005, 0.995])
        x = np.where(f, np.clip(x, lo, hi), np.nan)
        if not np.nanstd(x) > 0:
            drop["const"] += 1
            continue
        if len(names) >= MAX_VARS:
            drop["cap"] += 1
            continue
        Y.append(x)
        names.append(c)
        tf.append(t_)
    T["Y"] = np.column_stack(Y) if Y else np.zeros((n, 0))
    T["names"], T["tf"], T["drop"] = names, tf, drop
    return T


# ============================================================================ core statistics (atlas machinery)

def _solve(M, v):
    try:
        return np.linalg.solve(M, v)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, v, rcond=None)[0]


def cal_fe(yy, X0, q, adj=True, emap=None):
    """Quarter fixed effects given age, (age-50)^2, sex on rows in quarters with >= MINCELL rows (the A.calendar_adjust
    estimator, solved by within-quarter demeaning). Returns (R2 increment, centred adjustment for every row); with a
    dict emap, also fills quarter code -> centred effect (weights = rows per quarter, as the atlas)."""
    ok = q >= 0
    a = np.zeros(len(yy))
    if not ok.any():
        return None, a
    Q = int(q.max()) + 1
    nq = np.bincount(q[ok], minlength=Q)
    keep = nq >= A.MINCELL
    if keep.sum() < 2:
        return None, a
    e = ok & keep[np.where(ok, q, 0)]
    if e.sum() < 200:
        return None, a
    k = q[e]
    ye, X = yy[e], X0[e]
    cz = [j for j in range(1, X.shape[1]) if np.ptp(X[:, j]) > 0]  # constant covariates (e.g. one sex) dropped
    Z = X[:, cz]
    X = X[:, [0] + cz]
    nk = np.maximum(nq, 1).astype(float)
    my = np.bincount(k, ye, minlength=Q) / nk
    mZ = np.column_stack([np.bincount(k, Z[:, j], minlength=Q) / nk for j in range(Z.shape[1])]) if cz else \
        np.zeros((Q, 0))
    yw, Zw = ye - my[k], Z - mZ[k]
    bw = _solve(Zw.T @ Zw, Zw.T @ yw) if cz else np.zeros(0)
    sse1 = float(((yw - Zw @ bw) ** 2).sum())
    b0 = _solve(X.T @ X, X.T @ ye)
    sse0 = float(((ye - X @ b0) ** 2).sum())
    r2 = 1.0 - sse1 / sse0 if sse0 > 0 else None
    if adj:
        al = my - mZ @ bw
        w = np.where(keep, nq, 0).astype(float)
        eff = al - (al * w).sum() / w.sum()
        a[e] = eff[k]
        if emap is not None:
            emap.update({int(c): float(eff[c]) for c in np.flatnonzero(keep)})
    return r2, a


def resid(y, X):
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def season_r2(yy, X0, S):
    s0 = float((resid(yy, X0) ** 2).sum())
    s1 = float((resid(yy, np.column_stack([X0, S])) ** 2).sum())
    return 1.0 - s1 / s0 if s0 > 0 else None


def day_anova(r, d, correct=True, want_p=True):
    """A.day_adjust: one-way ANOVA moments over days with >= 2 rows (>= 20 such days); leave-one-out shrunken day BLUP
    u = sum_{j != i} r_j / (n_d - 1 + k), k = var_within / var_day; F test of the day effect."""
    S = np.bincount(d, r)
    C = np.bincount(d).astype(float)
    multi = C >= 2
    D = int(multi.sum())
    if D < 20:
        return None, None
    rm = multi[d]
    nd = C[multi]
    N = nd.sum()
    gm = r[rm].mean()
    ss_tot = float(((r[rm] - gm) ** 2).sum())
    ss_b = float(((S[multi] / nd - gm) ** 2 * nd).sum())
    msw = (ss_tot - ss_b) / (N - D)
    msb = ss_b / (D - 1)
    n0 = (N - (nd ** 2).sum() / N) / (D - 1)
    vday = max(0.0, (msb - msw) / n0)
    out = {"share": float(vday / (vday + msw)) if (vday + msw) > 0 else 0.0, "n_days": A.rc(D)}
    if want_p:
        out["F"] = float(msb / msw) if msw > 0 else None
        out["p"] = float(stats.f.sf(msb / msw, D - 1, N - D)) if msw > 0 else None
    if not correct or vday <= 0:
        return out, None
    k = msw / vday
    n = C[d]
    return out, np.where(n >= 2, (S[d] - r) / (n - 1 + k), 0.0)


_TRIU = {}


def triu(n):
    if n not in _TRIU:
        _TRIU[n] = np.triu_indices(n, 1)
    return _TRIU[n]


def pair_set(dcode, t, pcode, cid, max_pairs=MAX_PAIRS, rng=None):
    """Unordered same-day pairs of different, unrelated units with finite t (atlas session_adjust order: days, then
    time). Above max_pairs, whole days are subsampled (reported as 'sub')."""
    r_ = np.flatnonzero((dcode >= 0) & np.isfinite(t))
    o = r_[np.lexsort((t[r_], dcode[r_]))]
    groups = [g_ for g_ in np.split(o, np.flatnonzero(np.diff(dcode[o])) + 1) if len(g_) >= 2]
    tot = sum(len(g_) * (len(g_) - 1) // 2 for g_ in groups)
    sub = 1.0
    if tot > max_pairs and rng is not None:
        sub = max_pairs / tot
        groups = [g_ for g_ in groups if rng.random() < sub]
    I, J = [np.zeros(0, int)], [np.zeros(0, int)]
    for g_ in groups:
        iu, ju = triu(len(g_))
        a, b = g_[iu], g_[ju]
        k = (cid[a] != cid[b]) & (pcode[a] != pcode[b])
        I.append(a[k])
        J.append(b[k])
    I, J = np.concatenate(I), np.concatenate(J)
    return {"I": I, "J": J, "dt": np.abs(t[I] - t[J]), "sub": sub}


def kcells(R0, MF, ps, grp_code):
    """Binned sufficient statistics of the kernel for all variables at once: pair counts, residual-product sums and
    exp(-dt/tau) sums per (jackknife group = day code mod 20, |dt| bin) cell. R0 = residuals (0 where missing),
    MF = 1/0 presence. Returns NB, SP (JK, nbin, p) and KS (ntau, JK, nbin, p)."""
    edges = np.asarray(A.SESSION_EDGES, float)
    nbin, JK, ntau, p = len(edges), A.SESSION_JK, len(A.SESSION_TAUS), R0.shape[1]
    NB, SP, KS = np.zeros((JK * nbin, p)), np.zeros((JK * nbin, p)), np.zeros((ntau, JK * nbin, p))
    I, J, dt = ps["I"], ps["J"], ps["dt"]
    if len(I) and p:
        b = np.searchsorted(edges, dt, side="right") - 1
        ci = (grp_code[I] % JK) * nbin + b
        col = np.arange(len(I))
        shp = (JK * nbin, len(I))
        G = sparse.csr_matrix((np.ones(len(I)), (ci, col)), shape=shp)
        Gt = [sparse.csr_matrix((np.exp(-dt / tau), (ci, col)), shape=shp) for tau in A.SESSION_TAUS]
        ch = int(max(1, min(64, 4e6 // len(I))))
        for c0 in range(0, p, ch):
            cs = slice(c0, min(p, c0 + ch))
            W = MF[I, cs] * MF[J, cs]
            NB[:, cs] = G @ W
            SP[:, cs] = G @ (R0[I, cs] * R0[J, cs])
            for t in range(ntau):
                KS[t][:, cs] = Gt[t] @ W
    return NB.reshape(JK, nbin, p), SP.reshape(JK, nbin, p), KS.reshape(ntau, JK, nbin, p)


def _wls2(w, cv, E):
    """WLS of the bin covariances cv on [1, E_tau] for every tau row of E (2x2 normal equations), with the atlas
    fallback when a coefficient is negative (the better of the day-only and kernel-only fits)."""
    S0, T0 = w.sum(), (w * cv).sum()
    S1, S2, T1 = E @ w, (E * E) @ w, (E * w) @ cv
    det = S0 * S2 - S1 * S1
    with np.errstate(all="ignore"):
        b0 = (S2 * T0 - S1 * T1) / det
        s = (S0 * T1 - S1 * T0) / det
    bad = ~(np.isfinite(b0) & np.isfinite(s)) | (np.abs(det) <= 1e-12 * np.abs(S0 * S2))
    for i in np.flatnonzero(bad):
        Xk = np.column_stack([np.ones(len(w)), E[i]])
        b0[i], s[i] = np.linalg.lstsq((Xk.T * w) @ Xk, (Xk.T * w) @ cv, rcond=None)[0]
    neg = (b0 < 0) | (s < 0)
    if neg.any():
        c1 = max(T0 / S0, 0.0)
        with np.errstate(all="ignore"):
            c2 = np.maximum(T1 / S2, 0.0)
        e1 = (w * (cv - c1) ** 2).sum()
        e2 = (w[None, :] * (cv[None, :] - c2[:, None] * E) ** 2).sum(1)
        p2 = e2 < e1  # atlas: min(candidates) keeps the first (day-only) on ties
        b0 = np.where(neg, np.where(p2, 0.0, c1), b0)
        s = np.where(neg, np.where(p2, c2, 0.0), s)
    sse = (w[None, :] * (cv[None, :] - b0[:, None] - s[:, None] * E) ** 2).sum(1)
    return b0, s, sse


def _kf(nb, sp, ks, tau_idx):
    """The fit closure of A.kernel_fit on bin totals (nb counts, sp product sums, ks[tau] kernel sums), vectorised over
    tau with the same selection rule. Returns ((sse, tau index, b0, s), max kernel value over the used bins)."""
    use = nb >= A.SESSION_MIN_BIN
    if use.sum() < 3:
        return None, None
    cov = sp / np.maximum(nb, 1)
    w, cv = nb[use], cov[use]
    ti = np.asarray(list(tau_idx), int)
    E = (ks[ti] / np.maximum(nb, 1))[:, use]
    b0, s, sse = _wls2(w, cv, E)
    best, bi = None, 0
    for i, it in enumerate(ti):
        if best is None or sse[i] < best[0] - 1e-15:
            best, bi = (float(sse[i]), int(it), float(b0[i]), float(s[i])), i
    return best, float(E[bi].max())


def kfit_cells(NBj, SPj, KSj, want_z=True):
    """A.kernel_fit replica from the cell statistics of one variable: (tau, b0, s, z, emax) or None. z = s / jackknife
    SE over the day groups with tau fixed (None when not wanted); emax = max kernel value over the used bins."""
    nb, sp, ks = NBj.sum(0), SPj.sum(0), KSj.sum(1)
    best, emax = _kf(nb, sp, ks, range(len(A.SESSION_TAUS)))
    if best is None:
        return None
    _, it, b0, s = best
    z = 0.0
    if s > 0 and want_z:
        sj = []
        for g_ in range(A.SESSION_JK):
            q, _ = _kf(nb - NBj[g_], sp - SPj[g_], ks - KSj[:, g_], (it,))
            if q is not None:
                sj.append(q[3])
        sj = np.array(sj)
        se = np.sqrt((len(sj) - 1) / len(sj) * ((sj - sj.mean()) ** 2).sum()) if len(sj) > 2 else np.inf
        z = float(s / se) if se > 0 else 0.0
    elif s > 0:
        z = None
    return float(A.SESSION_TAUS[it]), b0, s, z, emax


def dec_bins():
    e = np.asarray(A.SESSION_EDGES, float)
    up = np.r_[e[1:], np.inf]
    return np.flatnonzero(up <= DEC_SHORT), np.flatnonzero(e >= DEC_LONG)


def decay(NBj, SPj, V):
    """Within-day decay test statistic: (mean residual product of same-day pairs <= 30 min apart - that of pairs
    >= 120 min apart) / V, with its jackknife SE over the day groups. Linear (no tau search), so its z is honest."""
    S_, L_ = dec_bins()
    nb, sp = NBj.sum(0), SPj.sum(0)
    ns, nl = nb[S_].sum(), nb[L_].sum()
    if ns < A.SESSION_MIN_BIN or nl < A.SESSION_MIN_BIN or not V > 0:
        return None, None
    d = (sp[S_].sum() / ns - sp[L_].sum() / nl) / V
    nbg, spg = nb[None, :] - NBj, sp[None, :] - SPj
    nsg, nlg = nbg[:, S_].sum(1), nbg[:, L_].sum(1)
    okg = (nsg > 0) & (nlg > 0)
    dg = (spg[okg][:, S_].sum(1) / nsg[okg] - spg[okg][:, L_].sum(1) / nlg[okg]) / V
    G = len(dg)
    se = float(np.sqrt((G - 1) / G * ((dg - dg.mean()) ** 2).sum())) if G > 2 else None
    return float(d), se


def daycov_index(dn, cid):
    """Day and day x relation-component indices of the units, and the 1-7-day neighbour pairs of both (computed once
    per table and per placebo replicate; the statistics of every variable then cost a few bincounts)."""
    ud, inv = np.unique(dn, return_inverse=True)
    M = int(cid.max()) + 1 if len(cid) else 1
    uk, kinv = np.unique(dn.astype(np.int64) * M + cid, return_inverse=True)
    kday = np.searchsorted(ud, uk // M)
    nd, nk_ = [], []
    for L in range(1, NB_LAG + 1):
        j = np.searchsorted(ud, ud + L)
        jc = np.minimum(j, len(ud) - 1)
        ok = (j < len(ud)) & (ud[jc] == ud + L)
        nd.append((np.flatnonzero(ok), jc[ok]))
        jk = np.searchsorted(uk, uk + L * M)
        jkc = np.minimum(jk, max(len(uk) - 1, 0))
        okk = (jk < len(uk)) & (uk[jkc] == uk + L * M)
        nk_.append((np.flatnonzero(okk), jkc[okk]))
    return {"ud": ud, "inv": inv, "kinv": kinv, "kday": kday, "nk": len(uk), "nbd": nd, "nbk": nk_,
            "gk": (ud.astype(np.int64) % A.SESSION_JK).astype(int)}


def daycov_stats(r, m, X):
    """Same-day (c0) and neighbouring-day (c17: 1-7 calendar days apart) covariance of the residuals of different,
    unrelated units (pairs within one relation component, i.e. the same person or relatives, are removed, as in the
    atlas kernel), / V, from day and day x component sums. r = residuals (0 where missing), m = 1/0 presence. The
    independent clinic-day component is dind = c0 - c17 (drift within the quarter shows as c17 ~ c0). Jackknife SEs
    over 20 day groups (day number mod 20; a neighbouring pair belongs to its first day)."""
    nD = len(X["ud"])
    S, n, Q = np.bincount(X["inv"], r, nD), np.bincount(X["inv"], m, nD), np.bincount(X["inv"], r * r, nD)
    N = n.sum()
    if (n > 0).sum() < 20 or N < 2:
        return None
    V = float(Q.sum() / N - (S.sum() / N) ** 2)
    if not V > 0:
        return None
    Sk, nk, Qk = (np.bincount(X["kinv"], r, X["nk"]), np.bincount(X["kinv"], m, X["nk"]),
                  np.bincount(X["kinv"], r * r, X["nk"]))
    kday = X["kday"]
    num0 = S * S - Q - np.bincount(kday, Sk * Sk - Qk, nD)
    den0 = n * (n - 1) - np.bincount(kday, nk * (nk - 1), nD)
    num1, den1 = np.zeros(nD), np.zeros(nD)
    for (i_, j_), (ik, jk) in zip(X["nbd"], X["nbk"]):
        num1[i_] += S[i_] * S[j_]
        den1[i_] += n[i_] * n[j_]
        num1 -= np.bincount(kday[ik], Sk[ik] * Sk[jk], nD)
        den1 -= np.bincount(kday[ik], nk[ik] * nk[jk], nD)
    if den0.sum() < A.SESSION_MIN_BIN or den1.sum() < A.SESSION_MIN_BIN:
        return None
    gk, G = X["gk"], A.SESSION_JK
    N0, D0 = np.bincount(gk, num0, G), np.bincount(gk, den0, G)
    N1, D1 = np.bincount(gk, num1, G), np.bincount(gk, den1, G)
    c0, c1 = num0.sum() / den0.sum() / V, num1.sum() / den1.sum() / V
    okg = ((den0.sum() - D0) > 0) & ((den1.sum() - D1) > 0)
    j0 = (num0.sum() - N0[okg]) / (den0.sum() - D0[okg]) / V
    j1 = (num1.sum() - N1[okg]) / (den1.sum() - D1[okg]) / V
    mm = okg.sum()

    def se(x):
        return float(np.sqrt((mm - 1) / mm * ((x - x.mean()) ** 2).sum())) if mm > 2 else None
    return {"c0": float(c0), "c17": float(c1), "dind": float(c0 - c1), "c17_se": se(j1), "dind_se": se(j0 - j1)}


def daycov(r, dn, cid):
    """daycov_stats for the rows given (all present)."""
    return daycov_stats(r, np.ones(len(r)), daycov_index(dn, cid))


def krige_var(r, tt, dk, s, tau, V, b0):
    """A.session_adjust kriging: days (sorted by day, then time; untimed rows last) with >= 2 rows, A.krige_day."""
    o = np.lexsort((tt, dk))
    ah = np.zeros(len(r))
    f = min(1.0, 0.95 * V / (b0 + s))
    for g_ in np.split(o, np.flatnonzero(np.diff(dk[o])) + 1):
        if len(g_) >= 2:
            ah[g_] = A.krige_day(r[g_], tt[g_], s * f, tau, V, b0 * f)
    return ah


def ols_hc1(y, X):
    n, k = X.shape
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ (X.T @ y)
    e = y - X @ b
    meat = (X * (e ** 2)[:, None]).T @ X * (n / max(n - k, 1))
    V = XtX @ meat @ XtX
    return b, np.sqrt(np.clip(np.diag(V), 0, None))


def zsd(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else x * np.nan


def ref_fit(y, X, names):
    """Joint standardized model z(y) ~ z(age) + sex + z(BMI) + z(SBP): {name: (b, se)}."""
    b, se = ols_hc1(zsd(y), X)
    return {nm: [float(b[k + 1]), float(se[k + 1])] for k, nm in enumerate(names)}


def hour_test(r, h, dk, q, X):
    """Population clock-hour curve (annotation H): two harmonics of the local clock hour h (24 h and 12 h periods),
    residualised on age, (age-50)^2, sex and quarter (within-quarter demeaning; r is already adjusted for them, so this
    is the Frisch-Waugh partial fit); share of var(r) they explain and a Wald test with the leave-one-clinic-day-out
    jackknife covariance (days are the clusters, so clinic-day and within-day processes do not inflate it, and the
    jackknife is not fooled by high-leverage clock times).  Clock times outside the 1-99 % range of the variable's
    units (circular, so a window across midnight is handled) are left out."""
    f = np.isfinite(h)
    if f.sum() < MIN_UNITS:
        return None
    w = 2.0 * np.pi * h[f] / 24.0
    mu = np.arctan2(np.sin(w).mean(), np.cos(w).mean()) * 24.0 / (2.0 * np.pi)
    hc = (h[f] - mu + 12.0) % 24.0 - 12.0  # hours from the circular mean
    lo, hi = np.quantile(hc, [0.01, 0.99])
    keep = (hc >= lo) & (hc <= hi)
    ix = np.flatnonzero(f)[keep]
    if len(ix) < MIN_UNITS:
        return None
    r, q, X, w = r[ix], q[ix], X[ix], w[keep]
    d = pd.factorize(dk[ix])[0]
    G = int(d.max()) + 1
    if G < 50:
        return None
    Z = np.column_stack([np.sin(w), np.cos(w), np.sin(2 * w), np.cos(2 * w)])
    Q = int(q.max()) + 1
    nq = np.maximum(np.bincount(q, minlength=Q), 1).astype(float)

    def wq(M):
        return M - (np.column_stack([np.bincount(q, M[:, j], Q) for j in range(M.shape[1])]) / nq[:, None])[q]
    Zw, Xw = wq(Z), wq(X[:, 1:])
    Xw = Xw[:, np.ptp(Xw, axis=0) > 1e-12]
    if Xw.shape[1]:
        Zw = Zw - Xw @ np.linalg.lstsq(Xw, Zw, rcond=None)[0]
    U, sv, _ = np.linalg.svd(Zw, full_matrices=False)
    k = int((sv > 1e-6 * sv[0]).sum()) if sv.size and sv[0] > 0 else 0
    if k == 0:
        return None
    U = U[:, :k]
    rc_ = r - r.mean()
    ss = float(rc_ @ rc_)
    if not ss > 0:
        return None
    b = U.T @ rc_  # U'U = I
    # leave-one-day-out: b(-g) = (I - U_g'U_g)^-1 (b - U_g' r_g)
    Ag = np.stack([np.stack([np.bincount(d, U[:, i] * U[:, j], G) for j in range(k)], -1) for i in range(k)], 1)
    cg = np.column_stack([np.bincount(d, U[:, i] * rc_, G) for i in range(k)])
    Mg = np.eye(k)[None, :, :] - Ag
    if np.linalg.eigvalsh(Mg).min() < 1e-6:
        return None
    bj = np.linalg.solve(Mg, (b[None, :] - cg)[:, :, None])[:, :, 0]
    D_ = bj - bj.mean(0)
    Vb = D_.T @ D_ * (G - 1.0) / G
    W = float(b @ np.linalg.pinv(Vb) @ b)
    return {"hour_r2": float(b @ b / ss), "hour_df": k, "hour_p": float(stats.chi2.sf(W, k)), "hour_n": rc(len(ix))}


def satt_p(x, null):
    """p of x under a scaled chi-square moment-matched to the placebo values (Satterthwaite)."""
    null = np.asarray([v for v in null if v is not None and np.isfinite(v)], float)
    if x is None or len(null) < 5:
        return None
    m, v = null.mean(), null.var(ddof=1)
    if m <= 0 or v <= 0:
        return 0.0 if x > null.max() else 1.0
    c, k = v / (2 * m), 2 * m * m / v
    return float(stats.chi2.sf(x / c, k))


# ============================================================================ placebo labels

def placebo_labels(T, rng):
    """(day labels shuffled within quarter, units keep their clock time; quarter labels of whole days permuted
    across the calendar)."""
    dc, qc = T["dcode"], T["qcode"]
    db = dc.copy()
    idx = np.flatnonzero((dc >= 0) & (qc >= 0))
    o = idx[np.argsort(qc[idx], kind="stable")]
    for g_ in np.split(o, np.flatnonzero(np.diff(qc[o])) + 1):
        db[g_] = dc[rng.permutation(g_)]
    perm = rng.permutation(len(T["qday"]))
    qb = np.where(dc >= 0, T["qday"][perm][np.maximum(dc, 0)], -1)
    return db, qb


# ============================================================================ one table

COMBOS = ("0", "C", "D", "CD", "K", "CK")


def consequences(T, j, idx, yraw, yv, rraw_var, ref_s, ref_X, ref_names):
    """Removed variance share, sample-size inflation (parallel, change score), retest ICC and reference associations
    of one corrected version yv (over rows idx) of variable j."""
    out = {}
    Xm = T["X0"][idx]
    vf = float(resid(yv, Xm).var())
    out["vr"] = 1.0 - vf / rraw_var if rraw_var > 0 else None
    out["ssif"] = rraw_var / vf if vf > 0 else None
    full = np.full(T["n"], np.nan)
    full[idx] = yv
    if len(T["ret0"]):
        raw = np.full(T["n"], np.nan)
        raw[idx] = yraw
        a0, a1 = T["ret0"], T["ret1"]
        okr = np.isfinite(raw[a0]) & np.isfinite(raw[a1])
        if okr.sum() >= A.MINCELL:
            dr = raw[a1][okr] - raw[a0][okr]
            dc = full[a1][okr] - full[a0][okr]
            out["ssif_chg"] = float(dr.var() / dc.var()) if dc.var() > 0 and dr.var() > 0 else None
            out["icc"] = fl(A.icc1(full[a0][okr], full[a1][okr]))
    if ref_s is not None:
        try:
            out["ref"] = ref_fit(full[ref_s], ref_X, ref_names)
        except Exception as ex:
            out["ref_error"] = err(ex)
    return out


def kernel_one(j, w_, rec, NB, SP, KS, ps, R0, MF, T, dcode, chk):
    """Within-day statistics of variable j from the cell statistics: decay test, A.kernel_fit replica, identification,
    atlas application rule and kriging (A.krige_day); the first three fits are also checked against A.kernel_fit."""
    V = w_["V"]
    rec["n_pairs"] = A.rcp(NB[:, :, j].sum())
    dd, dse = decay(NB[:, :, j], SP[:, :, j], V)
    rec["dec"], rec["dec_se"] = dd, dse
    rec["dec_z"] = dd / dse if dd is not None and dse else None
    kf = kfit_cells(NB[:, :, j], SP[:, :, j], KS[:, :, :, j])
    if kf is None:
        return
    tau, b0, s, z, emax = kf
    rec.update({"k_tau": tau, "k_day": b0 / V, "k_share": s / V, "k_z": z, "l0": (b0 + s) / V,
                "k_ident": bool(emax >= IDENT_MIN), "k_emax": emax})
    rec["k_applied"] = bool(s > 0 and z >= A.SESSION_MIN_Z and emax >= IDENT_MIN)
    if rec["k_applied"]:
        idx = w_["idx"]
        w_["aK"] = krige_var(R0[idx, j], T["tm"][idx], dcode[idx], s, tau, V, b0)
    if len(chk) < 3:  # replica check against A.kernel_fit on the same pairs and day groups
        m_ = (MF[ps["I"], j] * MF[ps["J"], j]) > 0
        I_, J_ = ps["I"][m_], ps["J"][m_]
        _, _, bt = A.kernel_fit(ps["dt"][m_], R0[I_, j] * R0[J_, j], V, dcode[I_] % A.SESSION_JK)
        if bt is not None:
            chk.append(max(abs(bt[0] - tau), abs(bt[1] - b0) / V, abs(bt[2] - s) / V, abs(bt[3] - z) / max(1.0, abs(z))))


def first_units(order, pcode, el):
    """Mask of each participant's first eligible unit (el) in the given order (participant, first main visit, date)."""
    o = order[el[order]]
    s_ = np.zeros(len(el), bool)
    if len(o):
        s_[o[np.r_[True, np.diff(pcode[o]) != 0]]] = True
    return s_


def consequences_all(j, w_, rec, Y, T, n):
    """Raw reference associations and retest ICC, then the consequences of every correction set of variable j."""
    idx = w_["idx"]
    yraw = Y[idx, j]
    s_ = first_units(T["order"], T["pcode"], np.isfinite(Y[:, j]) & T["okc"] & np.isfinite(T["bmi"]) &
                     np.isfinite(T["sbp"]))
    ref_s = ref_X = ref_names = None
    if s_.sum() >= 200:
        C = {"age": zsd(T["age"][s_]), "sex": T["sex"][s_], "bmi": zsd(T["bmi"][s_]), "sbp": zsd(T["sbp"][s_])}
        nm = rec["var"].lower()
        yr_ = Y[s_, j]
        ref_names = [k for k in REF if np.ptp(C[k]) > 0 and not (k == "bmi" and nm == "bmi")
                     and not (k == "sbp" and nm == "sitting_blood_pressure_systolic")
                     and not (k in ("bmi", "sbp") and abs(np.corrcoef(yr_, C[k])[0, 1]) > 0.98)]
        ref_s = s_
        ref_X = np.column_stack([np.ones(s_.sum())] + [C[k] for k in ref_names])
        rec["n_ref"] = rc(s_.sum())
        try:
            rec["ref_raw"] = ref_fit(yr_, ref_X, ref_names)
        except Exception as ex:
            rec["ref_error"] = err(ex)
            ref_s = None
    raw_full = np.full(n, np.nan)
    raw_full[idx] = yraw
    if len(T["ret0"]):
        a0, a1 = T["ret0"], T["ret1"]
        okr = np.isfinite(raw_full[a0]) & np.isfinite(raw_full[a1])
        if okr.sum() >= A.MINCELL:
            rec["n_ret"] = rc(okr.sum())
            rec["icc_raw"] = fl(A.icc1(raw_full[a0][okr], raw_full[a1][okr]))
    cons = {}
    for cb in COMBOS:
        if "K" in cb and w_["aK"] is None:
            continue
        if cb == "0":
            cons[cb] = {"vr": 0.0, "ssif": 1.0, "ssif_chg": 1.0 if rec.get("n_ret") else None,
                        "icc": rec.get("icc_raw"), "ref": rec.get("ref_raw")}
            continue
        yv = yraw - (w_["aC"] if "C" in cb else 0.0) - (w_["aK"] if "K" in cb else w_["aD"] if "D" in cb else 0.0)
        cons[cb] = consequences(T, j, idx, yraw, yv, w_["rraw_var"], ref_s, ref_X, ref_names)
    rec["cons"] = cons


def audit_table(job):
    """One table; any failure outside the per-variable isolation becomes an error note for this table only."""
    try:
        return _audit_table(job)
    except Exception as ex:
        spec = job[0]
        return {"rel": mask_name(spec["rel"]), "tid": spec["tid"], "cat": spec["cat"], "info": {}, "vars": [],
                "error": err(ex), "log": [f"[{time.strftime('%H:%M:%S')}] {spec['tid']} FAILED: {err(ex)}"]}


def _audit_table(job):
    spec, ti = job
    a = CTX["args"]
    buf = []
    t0 = time.time()
    out = {"rel": mask_name(spec["rel"]), "tid": spec["tid"], "cat": spec["cat"], "info": {}, "vars": [], "log": buf}
    try:
        u, vcols, mode, tsrc = load_units(CTX["root"], spec, buf)
        T = build_context(u, vcols, mode, buf)
        del u
    except Exception as ex:
        out["error"] = err(ex)
        log(f"{spec['tid']} FAILED to load: {out['error']}", buf)
        return out
    n, Y = T["n"], T["Y"]
    p = Y.shape[1]
    info = out["info"]
    nd = np.bincount(T["dcode"][T["dcode"] >= 0])
    info.update({"n_units": rc(n), "n_vars": p, "dropped": T["drop"], "mode": mode, "time_source": tsrc,
                 "timed": T["timed"], "n_days": rc(len(nd)), "per_day": A.qtiles(pd.Series(nd)) if len(nd) >= 20 else None,
                 "n_participants": rc(len(np.unique(T["pid"]))), "retest_pairs": rc(len(T["ret0"]))})
    if p == 0:
        info["status"] = "no analysable variable"
        return out
    rng = np.random.default_rng([a.seed, ti])
    PL = [placebo_labels(T, rng) for _ in range(a.placebo)]
    X0, qcode, dcode = T["X0"], T["qcode"], T["dcode"]
    R0, MF = np.zeros((n, p)), np.zeros((n, p))
    recs, W = [], []
    verr = {}
    DCI = daycov_index(T["dnum"], T["cid"])

    def fail(j, ex, stage):
        """A variable that fails is dropped from every later step (error noted per table)."""
        verr.setdefault(stage, [0, err(ex)])[0] += 1
        W[j] = None
        R0[:, j], MF[:, j] = 0.0, 0.0
    # ---- real: calendar, residual, day (atlas order: quarter -> day)
    for j in range(p):
        y = Y[:, j]
        idx = np.flatnonzero(np.isfinite(y) & T["okc"])
        rec = {"tid": spec["tid"], "var": mask_name(T["names"][j]), "cat": spec["cat"], "tf": T["tf"][j], "n": rc(len(idx)),
               "tech": bool(spec["cat"] == "tech" or TECH_RX.search(T["names"][j])), "timed": T["timed"]}
        recs.append(rec)
        W.append(None)
        if len(idx) < MIN_UNITS:
            continue
        try:
            yy, Xm = y[idx], X0[idx]
            r2, aC = cal_fe(yy, Xm, qcode[idx])
            rec["cal_r2"] = r2
            if mode == "visit":  # visit-mix diagnostic: the quarter R2 among first-visit units only (no 02_00 units)
                b0 = T["stage"][idx] == A.MAIN_STAGES[0]
                if b0.sum() >= MIN_UNITS and (~b0).sum() >= A.MINCELL:
                    rec["cal_r2_base"] = cal_fe(yy[b0], Xm[b0], qcode[idx][b0], adj=False)[0]
            rec["seas_r2"] = season_r2(yy, Xm, T["season"][idx])
            yc = yy - aC
            r = resid(yc, Xm)
            da, aD = day_anova(r, dcode[idx])
            rec["day"] = da
            if T["timed"]:
                ht = hour_test(r, T["clk"][idx] / 60.0, dcode[idx], qcode[idx], Xm)
                if ht:
                    rec.update(ht)
            dc_ = daycov_stats(np.bincount(idx, r, n), np.bincount(idx, None, n).astype(float), DCI)
            if dc_:
                rec.update(dc_)
                rec["dind_z"] = dc_["dind"] / dc_["dind_se"] if dc_["dind_se"] else None
                rec["c17_z"] = dc_["c17"] / dc_["c17_se"] if dc_["c17_se"] else None
            R0[idx, j], MF[idx, j] = r, 1.0
            W[j] = {"idx": idx, "aC": aC, "aD": aD if aD is not None else np.zeros(len(idx)), "aK": None,
                    "V": float(r.var()), "rraw_var": float(resid(yy, Xm).var())}
        except Exception as ex:
            fail(j, ex, "core")
    # ---- real: within-day kernel (binned sufficient statistics for all variables at once)
    if T["timed"]:
        ps = pair_set(dcode, T["tm"], T["pcode"], T["cid"], rng=rng)
        info["n_pairs"] = A.rcp(len(ps["I"]))
        info["pair_subsample"] = ps["sub"] if ps["sub"] < 1 else None
        NB, SP, KS = kcells(R0, MF, ps, dcode)
        chk = []
        for j in range(p):
            w_, rec = W[j], recs[j]
            if w_ is None:
                continue
            try:
                kernel_one(j, w_, rec, NB, SP, KS, ps, R0, MF, T, dcode, chk)
            except Exception as ex:
                fail(j, ex, "kernel")
        info["kfit_replica_maxdiff"] = max(chk) if chk else None
        info["kfit_replica_n"] = len(chk)
        del NB, SP, KS
    # ---- consequences of each correction set (the flag-guided and the universal one are kept after classification)
    for j in range(p):
        if W[j] is None:
            continue
        try:
            consequences_all(j, W[j], recs[j], Y, T, n)
        except Exception as ex:
            fail(j, ex, "consequences")
    # ---- placebo replicates
    for rec in recs:
        rec["pl"] = {"cal": [], "day": [], "dind": [], "c17": [], "dec": [], "k": []}
    S_, L_ = dec_bins()
    for b, (db, qb) in enumerate(PL):
        DCB = daycov_index(T["dday"][db], T["cid"])
        for j in range(p):
            if W[j] is None:
                continue
            try:
                idx, pl = W[j]["idx"], recs[j]["pl"]
                pl["cal"].append(cal_fe(Y[idx, j], X0[idx], qb[idx], adj=False)[0])
                dab, _ = day_anova(R0[idx, j], db[idx], correct=False, want_p=False)
                pl["day"].append(dab["share"] if dab else None)
                dcb = daycov_stats(R0[:, j], MF[:, j], DCB)
                pl["dind"].append(dcb["dind"] if dcb else None)
                pl["c17"].append(dcb["c17"] if dcb else None)
                if b == 0:
                    pl["dind_z0"] = dcb["dind"] / dcb["dind_se"] if dcb and dcb["dind_se"] else None
                    pl["c17_z0"] = dcb["c17"] / dcb["c17_se"] if dcb and dcb["c17_se"] else None
            except Exception as ex:
                fail(j, ex, "placebo")
        if T["timed"]:
            psb = pair_set(db, T["clk"], T["pcode"], T["cid"], rng=rng)
            NB, SP, KS = kcells(R0, MF, psb, db)
            for j in range(p):
                if W[j] is None:
                    continue
                try:
                    V, pl = W[j]["V"], recs[j]["pl"]
                    dd, dse = decay(NB[:, :, j], SP[:, :, j], V)
                    pl["dec"].append(dd)
                    kf = kfit_cells(NB[:, :, j], SP[:, :, j], KS[:, :, :, j], want_z=(b == 0))
                    pl["k"].append(kf[2] / V if kf is not None else None)
                    if b == 0:
                        pl["dec_z0"] = dd / dse if dd is not None and dse else None
                        pl["k_applied0"] = bool(kf is not None and kf[2] > 0 and (kf[3] or 0) >= A.SESSION_MIN_Z
                                                and kf[4] >= IDENT_MIN)
                except Exception as ex:
                    fail(j, ex, "placebo-kernel")
            if b == 0:
                info["n_pairs_placebo"] = A.rcp(len(psb["I"]))
            del NB, SP, KS
    recs = [rec for j, rec in enumerate(recs) if W[j] is not None]
    if verr:
        info["var_errors"] = {k: {"n": v[0], "first": v[1]} for k, v in verr.items()}
        log(f"{spec['tid']}: variable errors {info['var_errors']}", buf)
    for rec in recs:
        pl = rec["pl"]
        for k in ("cal", "day", "dind", "c17", "dec", "k"):
            v = [x for x in pl[k] if x is not None]
            v1 = [x for x in pl[k][1:] if x is not None]
            pl[k + "0"] = pl[k][0] if pl[k] else None
            pl[k + "_mean"], pl[k + "_max"] = (float(np.mean(v)), float(np.max(v))) if v else (None, None)
            pl[k + "_mean1"], pl[k + "_max1"] = (float(np.mean(v1)), float(np.max(v1))) if v1 else (None, None)
        rec["cal_p"] = satt_p(rec["cal_r2"], pl["cal"])
        pl["cal_p0"] = satt_p(pl["cal0"], pl["cal"][1:])
        for k in ("cal", "day", "dind", "c17", "dec", "k"):
            del pl[k]
    out["vars"] = recs
    info["runtime_s"] = time.time() - t0
    log(f"{spec['tid']}: {A.cnt(n)} units, {p} variables, timed {T['timed']}, pairs {info.get('n_pairs')}; "
        f"{info['runtime_s']:.1f}s", buf)
    return out


# ============================================================================ classification and summaries

def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def classify(recs):
    """Flags with BH within the phenotype and the technical family, for the real data and for placebo rep 0
    ('placebo-as-real', compared with reps 1..B-1):
      C  calendar: quarter R2 (scaled-chi2 placebo p) or sub-quarter drift (neighbouring-day covariance c17, jackknife z);
      D  independent clinic-day component dind = c0 - c17 (jackknife z);
      K  within-day decay (jackknife z) and the atlas kernel rule (s > 0, z >= 3, identified).
    Each also needs the statistic above every placebo replicate and an excess over the placebo mean >= MIN_EX.
    Class = finest flagged scale; acquisition share; the flag-guided and the universal correction sets are kept."""
    fam = np.array([bool(r["tech"]) for r in recs])

    def bhf(pv):
        q = np.full(len(pv), np.nan)
        for f_ in (False, True):
            ix = np.flatnonzero(fam == f_)
            if len(ix):
                q[ix] = A.bh([pv[i] for i in ix])
        return q

    def nsf(z):
        return float(stats.norm.sf(z)) if z is not None else None

    def run(real):
        if real:
            P = [[r.get("cal_p") for r in recs], [nsf(r.get("c17_z")) for r in recs],
                 [nsf(r.get("dind_z")) for r in recs], [nsf(r.get("dec_z")) for r in recs]]
        else:
            P = [[g(r, "pl", "cal_p0") for r in recs], [nsf(g(r, "pl", "c17_z0")) for r in recs],
                 [nsf(g(r, "pl", "dind_z0")) for r in recs], [nsf(g(r, "pl", "dec_z0")) for r in recs]]
        Qv = [bhf(p_) for p_ in P]
        res = []
        for i, r in enumerate(recs):
            pl = r["pl"]
            sfx = "" if real else "1"
            if real:
                v = (r.get("cal_r2"), r.get("c17"), r.get("dind"), r.get("dec"))
                app = r.get("k_applied")
            else:
                v = (pl.get("cal0"), pl.get("c170"), pl.get("dind0"), pl.get("dec0"))
                app = pl.get("k_applied0")

            def ok(val, key, mn, q):
                mx, me = pl.get(f"{key}_max{sfx}"), pl.get(f"{key}_mean{sfx}")
                return bool(np.isfinite(q) and q < Q_ALPHA and val is not None and mx is not None and val > mx
                            and val - (me or 0.0) >= mn)
            sig = [ok(v[0], "cal", 0.0, Qv[0][i]), ok(v[1], "c17", 0.0, Qv[1][i]), ok(v[2], "dind", 0.0, Qv[2][i]),
                   bool(app) and ok(v[3], "dec", 0.0, Qv[3][i])]
            res.append((ok(v[0], "cal", MIN_EX["C"], Qv[0][i]), ok(v[1], "c17", MIN_EX["C"], Qv[1][i]),
                        ok(v[2], "dind", MIN_EX["D"], Qv[2][i]), bool(app) and ok(v[3], "dec", MIN_EX["K"], Qv[3][i]),
                        [Qv[k][i] for k in range(4)], sig))
        return res
    for r, (Cq, Cd, D, K, qv, sig) in zip(recs, run(True)):
        r["flags"] = ("C" if (Cq or Cd) else "") + ("D" if D else "") + ("K" if K else "")
        r["sig"] = ("C" if (sig[0] or sig[1]) else "") + ("D" if sig[2] else "") + ("K" if sig[3] else "")
        r["c_kind"] = ("q" if Cq else "") + ("d" if Cd else "")
        r["cls"] = "within-day" if K else "day" if D else "calendar" if (Cq or Cd) else "none"
        r["q"] = {"Cq": fl(qv[0]), "Cd": fl(qv[1]), "D": fl(qv[2]), "K": fl(qv[3])}
        c = r.get("cal_r2") or 0.0
        d_ = g(r, "day", "share") or 0.0
        w = min(1.0, max(d_, r.get("l0") or 0.0)) if K else d_
        r["acq"] = c + (1 - c) * w
        cons = r.pop("cons", {})
        k_ = K and "K" in cons
        r["fix"] = (("C" if Cq else "") + ("K" if k_ else "D" if (D or Cd) else "")) or "0"
        r["univ"] = "CK" if "CK" in cons else "CD"
        r["cons_flag"] = cons.get(r["fix"])
        r["cons_univ"] = cons.get(r["univ"])
    for r, (Cq, Cd, D, K, _q, _s) in zip(recs, run(False)):
        r["pl_flags"] = ("C" if (Cq or Cd) else "") + ("D" if D else "") + ("K" if K else "")
    # hour-curve annotation H (not a class and not corrected): BH within family, share >= MIN_H
    qh = bhf([r.get("hour_p") for r in recs])
    for r, q_ in zip(recs, qh):
        r["q"]["H"] = fl(q_)
        r["H"] = bool(np.isfinite(q_) and q_ < Q_ALPHA and (r.get("hour_r2") or 0.0) >= MIN_H)


def ref_summary(recs, which):
    """Change of the standardized reference associations (b, SE) after correction `which`, per association: over all
    variables with an evaluable raw association (|z| >= 3) and over the flagged ones; 'any' = a variable with at least
    one evaluable association changing by more than 10 %."""
    out = {}
    by_var = {}
    for nm in REF:
        rel, ser, dabs, frel, fser = [], [], [], [], []
        for i, r in enumerate(recs):
            x0, x1 = g(r, "ref_raw", nm), g(r, which, "ref", nm)
            if not x0 or not x1:
                continue
            (br, sr), (bc, sc) = x0, x1
            dabs.append(bc - br)
            if sr > 0:
                ser.append(sc / sr)
                if r.get("flags"):
                    fser.append(sc / sr)
            if sr > 0 and abs(br / sr) >= 3:
                rv = (bc - br) / abs(br)
                rel.append(rv)
                if r.get("flags"):
                    frel.append(rv)
                by_var.setdefault(i, []).append(abs(rv))
        out[nm] = {"rel": q595(rel), "se_ratio": q595(ser), "dabs": q595(dabs), "n_eval": len(rel),
                   "frac_gt10": float(np.mean(np.abs(rel) > 0.10)) if rel else None,
                   "f_rel": q595(frel), "f_se_ratio": q595(fser), "f_n_eval": len(frel),
                   "f_frac_gt10": float(np.mean(np.abs(frel) > 0.10)) if frel else None}
    vals = list(by_var.values())
    out["any_gt10"] = {"n_vars": len(vals), "frac": float(np.mean([max(v) > 0.10 for v in vals])) if vals else None}
    for cl in ("none", "flagged"):
        sel = [i for i, r in enumerate(recs) if (r.get("cls") == "none") == (cl == "none") and i in by_var]
        out["any_gt10_" + cl] = {"n_vars": len(sel), "frac": float(np.mean([max(by_var[i]) > 0.10 for i in sel]))
                                 if sel else None}
    return out


def cons_dist(rr, which):
    return {"vr": q595([g(r, which, "vr") for r in rr]), "ssif": q595([g(r, which, "ssif") for r in rr]),
            "ssif_chg": q595([g(r, which, "ssif_chg") for r in rr]),
            "dicc": q595([g(r, which, "icc") - r["icc_raw"] for r in rr if g(r, which, "icc") is not None
                          and r.get("icc_raw") is not None])}


def summaries(recs):
    S = {}
    for grp_name, sel in (("pheno", lambda r: not r["tech"]), ("clinic", lambda r: r["cat"] == "clinic" and not r["tech"]),
                          ("home", lambda r: r["cat"] == "home" and not r["tech"]),
                          ("lab", lambda r: r["cat"] == "lab" and not r["tech"]),
                          ("other", lambda r: r["cat"] == "other" and not r["tech"]), ("tech", lambda r: r["tech"])):
        rr = [r for r in recs if sel(r)]
        if not rr:
            continue
        fl_ = [r for r in rr if r["flags"]]
        S[grp_name] = {"n_vars": len(rr), "n_timed": sum(bool(r["timed"]) for r in rr),
                       "classes": {c: sum(r["cls"] == c for r in rr) for c in CLASSES},
                       "flags": {f: sum(f in r["flags"] for r in rr) for f in "CDK"},
                       "pl_flags": {f: sum(f in r["pl_flags"] for r in rr) for f in "CDK"},
                       "pl_any": sum(bool(r["pl_flags"]) for r in rr), "n_flagged": len(fl_),
                       "sig": {f: sum(f in r["sig"] for r in rr) for f in "CDK"},
                       "acq": q595([r.get("acq") for r in rr]), "acq_flagged": q595([r.get("acq") for r in fl_]),
                       "cal": q595([r.get("cal_r2") for r in rr]), "seas": q595([r.get("seas_r2") for r in rr]),
                       "day": q595([g(r, "day", "share") for r in rr]), "c17": q595([r.get("c17") for r in rr]),
                       "dind": q595([r.get("dind") for r in rr]),
                       "ks": q595([r.get("k_share") for r in rr if r.get("k_ident")]),
                       "dec": q595([r.get("dec") for r in rr]),
                       "cq_base": {"n_cq": sum(1 for r in rr if "q" in (r.get("c_kind") or "") and
                                            r.get("cal_r2_base") is not None),
                                   "half": sum(1 for r in rr if "q" in (r.get("c_kind") or "") and
                                               r.get("cal_r2_base") is not None and
                                               r["cal_r2_base"] >= 0.5 * (r.get("cal_r2") or 0.0))},
                       "hour": {"n_tested": sum(r.get("hour_p") is not None for r in rr),
                                "n_H": sum(bool(r.get("H")) for r in rr),
                                "n_H_none": sum(bool(r.get("H")) and r["cls"] == "none" for r in rr),
                                "r2": q595([r.get("hour_r2") for r in rr])},
                       "fix_flagged": cons_dist(fl_, "cons_flag"), "univ_all": cons_dist(rr, "cons_univ")}
    return S


# ============================================================================ frame-level blocks (atlas frame)

def frame_arrays(fr, comp):
    """Atlas conventions: UTC day/quarter of the vascular timestamp, absolute minutes, relation components."""
    q, t = A.calendar_quarter(fr)
    quarter = q.where(~q.astype(str).str.contains("NA"), None).to_numpy(dtype=object)
    tn = pd.to_datetime(t, errors="coerce")
    day = tn.dt.strftime("%Y-%m-%d").where(tn.notna(), None).to_numpy(dtype=object)
    ts = pd.to_datetime(fr["vas_ts"], errors="coerce", utc=True) if "vas_ts" in fr.columns else \
        pd.Series(pd.NaT, index=fr.index, dtype="datetime64[ns, UTC]")
    tm = ((ts - ts.min()).dt.total_seconds() / 60.0).to_numpy(float)
    pid = fr["participant_id"].astype(str).to_numpy()
    cid = pd.factorize(np.array([comp.get(p_, p_) for p_ in pid] if comp else pid, dtype=object))[0]
    return {"quarter": quarter, "day": day, "tm": tm, "pcode": pd.factorize(pid)[0], "cid": cid,
            "age": pd.to_numeric(fr["age"], errors="coerce").to_numpy(float),
            "sex": pd.to_numeric(fr["sex_male"], errors="coerce").to_numpy(float)}


def rebuild_bed(fr, F, bed):
    """P8 machinery on the atlas frame with atlas conventions: raw bed -> quarter -> day -> kernel (A.session_adjust
    rule). Returns the rebuilt values, the day share and the kernel (replica and A.kernel_fit)."""
    y = pd.to_numeric(fr[bed + "_raw"], errors="coerce").to_numpy(float)
    age, sex = F["age"], F["sex"]
    okq = np.isfinite(y) & np.isfinite(age) & np.isfinite(sex) & np.array([q_ is not None for q_ in F["quarter"]])
    out = y.copy()
    qc = pd.factorize(F["quarter"])[0]
    X0 = np.column_stack([np.ones(len(y)), age, (age - 50.0) ** 2, sex])
    iq = np.flatnonzero(okq)
    emap = {}
    cal_fe(y[iq], X0[iq], qc[iq], emap=emap)
    adj = pd.Series(qc).map(emap).fillna(0.0).to_numpy(float)  # atlas: every row of a kept quarter, age or not
    out = np.where(np.isfinite(y), y - adj, y)
    ok = np.isfinite(out) & np.isfinite(age) & np.isfinite(sex) & np.array([d_ is not None for d_ in F["day"]])
    idx = np.flatnonzero(ok)
    Xm = X0[idx]
    yc = out[idx]
    r = yc - Xm @ A.beta_only(yc, Xm)
    V = float(r.var())
    dk = pd.factorize(F["day"][idx])[0]
    da, uu = day_anova(r, dk)
    yd = yc - uu if uu is not None else yc
    res = {"day_share": da["share"] if da else None, "V": V}
    tt = F["tm"][idx]
    pcode, cid = F["pcode"][idx], F["cid"][idx]
    ps = pair_set(dk, tt, pcode, cid, max_pairs=10 ** 9)
    final = yd
    if len(ps["I"]):
        R0 = r[:, None]
        MF = np.ones((len(r), 1))
        NB, SP, KS = kcells(R0, MF, ps, dk)
        kf = kfit_cells(NB[:, :, 0], SP[:, :, 0], KS[:, :, :, 0])
        _, _, bt = A.kernel_fit(ps["dt"], r[ps["I"]] * r[ps["J"]], V, dk[ps["I"]] % A.SESSION_JK)
        res["kernel"] = list(kf[:4]) if kf is not None else None
        res["kernel_atlas"] = bt
        if kf is not None:
            tau, b0, s, z, _ = kf
            if A.SESSION_ADJUST and s > 0 and z >= A.SESSION_MIN_Z:  # the atlas rule (no identification guard)
                final = yc - krige_var(r, tt, dk, s, tau, V, b0)
    out[idx] = final
    res["values"] = out
    return res


def block_check(fr, comp):
    F = frame_arrays(fr, comp)
    R = {}
    for bed in ("pwv", "abi_min"):
        if bed + "_raw" not in fr.columns:
            R[bed] = {"status": "no raw column"}
            continue
        rb = rebuild_bed(fr, F, bed)
        a_ = rb["values"]
        b_ = pd.to_numeric(fr[bed], errors="coerce").to_numpy(float)
        x = {"maxdiff": float(np.nanmax(np.abs(a_ - b_))) if np.isfinite(a_ - b_).any() else None,
             "same_missing": bool((np.isnan(a_) == np.isnan(b_)).all()),
             "day_share": rb["day_share"], "atlas_day_share": g(A.DIAG, "day_effects", bed, "day_share")}
        sk = g(A.DIAG, "session_kernel", bed) or {}
        x["atlas_kernel"] = [sk.get("tau"), sk.get("day_share"), sk.get("share"), sk.get("z"), sk.get("applied")]
        kf, ka = rb.get("kernel"), rb.get("kernel_atlas")
        if kf is not None and ka is not None:
            x["kernel_replica_maxdiff"] = float(max(abs(kf[0] - ka[0]), abs(kf[1] - ka[1]), abs(kf[2] - ka[2]),
                                                    abs(kf[3] - ka[3])))
            x["kernel"] = [kf[0], kf[1] / rb["V"], kf[2] / rb["V"], kf[3]]  # tau, day share, kernel share, z
        R[bed] = x
    return R


def block_clin(fr, recmap, rng, boot):
    """Clinical quantification, atlas raw vs v1.5 values: reclassification across the adverse top quintile/decile,
    sample-size inflation factors (parallel and change score), retest ICC; participant bootstrap (correction fixed)."""
    out = {}
    for bed, key in CLIN:
        if bed not in fr.columns or bed + "_raw" not in fr.columns:
            out[bed] = {"status": "missing"}
            continue
        sgn = A.ORIENT.get(bed, 1)
        d = A.one_row_per_participant(fr, [bed, bed + "_raw", "age", "sex_male"])
        orw, oco = sgn * d[bed + "_raw"].to_numpy(float), sgn * d[bed].to_numpy(float)
        a_ = d["age"].to_numpy(float)
        X = np.column_stack([np.ones(len(d)), a_, (a_ - 50) ** 2, d["sex_male"].to_numpy(float)])
        rr = orw - X @ A.beta_only(orw, X)
        rcz = oco - X @ A.beta_only(oco, X)
        o = {"n": rc(len(d))}

        def recl(xr, xc, frac):
            fr_ = xr >= np.quantile(xr, 1 - frac)
            fc_ = xc >= np.quantile(xc, 1 - frac)
            mv = int((fr_ & ~fc_).sum())
            return mv, fr_.sum(), float((fr_ != fc_).mean())
        for sc, (xr, xc) in (("abs", (orw, oco)), ("as", (rr, rcz))):
            for frac in (0.2, 0.1):
                mv, nf, ch = recl(xr, xc, frac)
                o[f"{sc}_top{int(frac * 100)}"] = {"moved": (mv / nf) if mv >= A.MINCELL else "<20",
                                                    "changed_all": ch if mv >= A.MINCELL else "<20"}
        o["ssif1"] = float(rr.var() / rcz.var()) if rcz.var() > 0 else None
        w0 = fr[fr["research_stage"] == A.MAIN_STAGES[0]].drop_duplicates("participant_id").set_index("participant_id")
        w1 = fr[fr["research_stage"] == A.MAIN_STAGES[1]].drop_duplicates("participant_id").set_index("participant_id")
        both = w0.index.intersection(w1.index)
        r0r, r1r = w0.loc[both, bed + "_raw"].to_numpy(float), w1.loc[both, bed + "_raw"].to_numpy(float)
        r0c, r1c = w0.loc[both, bed].to_numpy(float), w1.loc[both, bed].to_numpy(float)
        okr = np.isfinite(r0r) & np.isfinite(r1r) & np.isfinite(r0c) & np.isfinite(r1c)
        dr, dc = (r1r - r0r)[okr], (r1c - r0c)[okr]
        if okr.sum() >= A.MINCELL:
            o["n_ret"] = rc(okr.sum())
            o["ssif_chg"] = float(dr.var() / dc.var()) if dc.var() > 0 else None
            o["icc_r"], o["icc_c"] = fl(A.icc1(r0r[okr], r1r[okr])), fl(A.icc1(r0c[okr], r1c[okr]))
        rec = recmap.get(key)
        if rec and rec.get("acq") is not None and rec["acq"] < 1:
            o["ssif_model"] = 1.0 / (1.0 - rec["acq"])
            o["p8_cls"] = rec.get("cls")
        bt = {"as_top20": [], "as_top10": [], "ssif1": [], "ssif_chg": []}
        nn = len(rr)
        for _ in range(boot):
            i = rng.integers(0, nn, nn)
            for frac in (0.2, 0.1):
                mv, nf, _c = recl(rr[i], rcz[i], frac)
                bt[f"as_top{int(frac * 100)}"].append(mv / nf if nf else np.nan)
            bt["ssif1"].append(rr[i].var() / rcz[i].var())
            if okr.sum() >= A.MINCELL:
                k = rng.integers(0, len(dr), len(dr))
                bt["ssif_chg"].append(dr[k].var() / dc[k].var())
        o["boot"] = {k: (pctl(v) if len(v) >= 20 else None) for k, v in bt.items()}
        for k in ("as_top20", "as_top10"):  # no interval around a suppressed (< 20 participants moved) share
            if o[k]["moved"] == "<20":
                o["boot"][k] = None
        out[bed] = o
    return out


def pctl(v):
    a = np.asarray(v, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) >= 20 else None


def block_xr(fr, comp, rng):
    """Same-day stranger cross-correlations (composition-free day processes) between PWV, device SBP, cuff SBP and
    cuff HR on the atlas frame (raw values, residuals on age, (age-50)^2, sex and quarter); all same-day pairs and
    pairs <= 60 min apart; jackknife SE over 20 day groups."""
    F = frame_arrays(fr, comp)
    vars_ = [v for v in ("pwv_raw", "dev_bsbp", "sbp_sit", "hr_sit") if v in fr.columns]
    Y = np.column_stack([pd.to_numeric(fr[v], errors="coerce").to_numpy(float) for v in vars_])
    ok = np.isfinite(Y).all(1) & np.isfinite(F["age"]) & np.isfinite(F["sex"]) & \
        np.array([d_ is not None for d_ in F["day"]]) & np.array([q_ is not None for q_ in F["quarter"]])
    idx = np.flatnonzero(ok)
    if len(idx) < 200:
        return {"status": "too few rows"}
    qc = pd.factorize(F["quarter"][idx])[0]
    cnt = np.bincount(qc)
    keep = np.flatnonzero(cnt >= A.MINCELL)
    a_ = F["age"][idx]
    cols = [np.ones(len(idx)), a_, (a_ - 50) ** 2, F["sex"][idx]] + [(qc == k).astype(float) for k in keep[1:]]
    X = np.column_stack(cols)
    R = Y[idx] - X @ np.linalg.lstsq(X, Y[idx], rcond=None)[0]
    dk = pd.factorize(F["day"][idx])[0]
    ps = pair_set(dk, np.where(np.isfinite(F["tm"][idx]), F["tm"][idx], 0.0), F["pcode"][idx], F["cid"][idx],
                  max_pairs=10 ** 9)
    I, J, dt = ps["I"], ps["J"], ps["dt"]
    timed = np.isfinite(F["tm"][idx])
    grp = dk[I] % 20
    out = {"n_rows": rc(len(idx)), "n_pairs": A.rcp(len(I)), "vars": vars_}
    for ia in range(len(vars_)):
        for ib in range(ia, len(vars_)):
            ra, rb = R[:, ia], R[:, ib]
            den = np.sqrt(ra.var() * rb.var())
            pr = (ra[I] * rb[J] + ra[J] * rb[I]) / 2.0 / den
            res = {}
            for lab, sel in (("all", np.ones(len(I), bool)), ("le60", (dt <= 60) & timed[I] & timed[J])):
                if sel.sum() < 100:
                    continue
                x = pr[sel]
                gg = grp[sel]
                tot, n_ = x.sum(), len(x)
                sg = np.bincount(gg, x, minlength=20)
                ng = np.bincount(gg, minlength=20)
                jk = np.array([(tot - sg[k]) / (n_ - ng[k]) for k in range(20) if n_ - ng[k] > 0])
                se = float(np.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean()) ** 2).sum())) if len(jk) > 2 else None
                res[lab] = {"r": float(x.mean()), "se": se, "n": A.rcp(n_)}
            out[f"{vars_[ia]}|{vars_[ib]}"] = res
    return out


# ============================================================================ digest

def v3(x):
    if x is None:
        return "NA"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "NA"
    s = f"{x:.3g}"
    return s.replace("e-0", "e-").replace("e+0", "e")


def qs(d):
    if not isinstance(d, dict) or d.get("q50") is None:
        return "NA"
    return f"{v3(d['q50'])}[{v3(d['q05'])},{v3(d['q95'])}]"


def short(v):
    s = str(v)
    for a_, b_ in (("_float_value", ""), ("bt__", ""), ("sitting_second_blood_pressure_", "sit2_"),
                   ("sitting_blood_pressure_", "sit_"), ("lying_blood_pressure_", "lie_"),
                   ("standing_one_min_blood_pressure_", "st1_"), ("standing_three_min_blood_pressure_", "st3_"),
                   ("systolic", "sys"), ("diastolic", "dia"), ("pulse_rate", "pr"),
                   ("from_r_thigh_to_r_ankle_pwv", "pwv_r"), ("from_l_thigh_to_l_ankle_pwv", "pwv_l"),
                   ("_brachial_pressure", "_brach"), ("_ankle_pressure", "_ankle"), ("body_comp_", ""),
                   ("_during_", "_"), ("hrv_frequency_", "hrv_"), ("hrv_nonlinear_", "hrv_"), ("MDC_ECG_LEAD_", "")):
        s = s.replace(a_, b_)
    return mask_name(s)[:34]


def vline(r, rank=None):
    """One variable: class[flags] n, acquisition share, calendar R2, season R2, day share, kernel share/z/tau (? = not
    identified), decay (se); consequences of the flag-guided (F) or, if unflagged, the universal (U) correction."""
    pre = f"{rank}. " if rank is not None else ""
    hc = f" h={v3(r.get('hour_r2'))}{'*' if r.get('H') else ''}"
    if r.get("timed") and r.get("k_share") is not None:
        ki = (f"k={v3(r.get('k_share')) if r.get('k_ident') else 'unid'} z={v3(r.get('k_z'))} t={v3(r.get('k_tau'))} "
              f"dec={v3(r.get('dec'))}({v3(r.get('dec_se'))}){hc}")
    elif r.get("timed"):
        ki = f"k=NA dec={v3(r.get('dec'))}({v3(r.get('dec_se'))}){hc}"
    else:
        ki = "untimed"
    tag, c = ("F", r.get("cons_flag")) if r.get("flags") else ("U", r.get("cons_univ"))
    c = c or {}
    ic = f" icc={v3(r.get('icc_raw'))}>{v3(c.get('icc'))}" if r.get("icc_raw") is not None else ""
    b0, b1 = g(r, "ref_raw", "age"), g(c, "ref", "age")
    ra = f" bAge={v3(b0[0])}>{v3(b1[0])}" if b0 and b1 else ""
    fg = (r.get("flags") or "-").replace("C", "C" + (r.get("c_kind") or ""))
    cb = f"/{v3(r.get('cal_r2_base'))}" if "q" in (r.get("c_kind") or "") else ""  # Cq: pooled / first-visit-only R2
    return (f"{pre}{r['tid']}:{short(r['var'])} {r['cls']}[{fg}] n={v3(r.get('n'))} acq={v3(r.get('acq'))} "
            f"cal={v3(r.get('cal_r2'))}{cb} sea={v3(r.get('seas_r2'))} d={v3(g(r, 'day', 'share'))} dr={v3(r.get('c17'))} "
            f"di={v3(r.get('dind'))} {ki} {tag}:vr={v3(c.get('vr'))} ssif={v3(c.get('ssif'))}/{v3(c.get('ssif_chg'))}"
            f"{ic}{ra}")


def digest(R):
    """Screen-transcription digest: <= 120 lines, <= 220 characters, no leading spaces, 3 significant digits."""
    L = [f"F1_ACQ_AUDIT {R.get('version')} | atlas {R.get('atlas_version')} | seed {R.get('seed')} | placebo "
         f"{g(R, 'settings', 'placebo')} | boot {g(R, 'settings', 'boot')} | workers {g(R, 'settings', 'workers')}"]
    I = R.get("info", {})
    ck = R.get("CHECK") or {}
    cp, ca = ck.get("pwv") or {}, ck.get("abi_min") or {}
    L.append(f"check: atlas v1.5 rebuilt by the P8 machinery: pwv max|d| {v3(cp.get('maxdiff'))} same-missing "
             f"{cp.get('same_missing')} day share {v3(cp.get('day_share'))}/atlas {v3(cp.get('atlas_day_share'))} kernel "
             f"t/d/s/z " + "/".join(v3(x) for x in (cp.get("kernel") or [])[:4]) + " replica-vs-A.kernel_fit "
             f"{v3(cp.get('kernel_replica_maxdiff'))} atlas t/d/s/z/applied " + "/".join(v3(x) for x in (cp.get("atlas_kernel") or []))
             + f"; abi max|d| {v3(ca.get('maxdiff'))}")
    L.append(f"check: table kfit replica max|d| {v3(I.get('kfit_replica_maxdiff'))} ({I.get('kfit_replica_n')} vars); "
             f"relations excluded {I.get('relations_excluded')}; liver tables excluded {I.get('liver_tables')}, liver "
             f"fields excluded {I.get('liver_fields')}, atlas frame has liver att {I.get('frame_has_att')}")
    ex = I.get("excluded") or {}
    L.append("excluded: " + "; ".join(f"{k} {len(v)} ({','.join(sorted(set(x.split('/')[0][:10] for x in v)))[:90]})"
                                      for k, v in sorted(ex.items())))
    for k, v in sorted((R.get("errors") or {}).items()):
        L.append(f"ERROR {k}: {v}")
    L.append("## TAB per table: cat mode (t=timed, donor=time from a sibling table) units days per-day q05/50/95 pairs | "
             "vars [dropped few/levels/const/cap] | classes none/cal/day/within-day | placebo-as-real flagged")
    for tb in R.get("tables", []):
        i = tb.get("info", {})
        if tb.get("error"):
            L.append(f"TAB {tb['tid']}: ERROR {tb['error']}")
            continue
        pdy = i.get("per_day") or {}
        dr = i.get("dropped") or {}
        cl = tb.get("classes") or {}
        L.append(f"TAB {tb['tid']} {tb['cat']} {i.get('mode')}{'/t' if i.get('timed') else ''}"
                 f"{'/donor' if i.get('time_source') == 'donor' else ''} u{i.get('n_units')} d{i.get('n_days')} "
                 f"{v3(pdy.get('q05'))}/{v3(pdy.get('q50'))}/{v3(pdy.get('q95'))} pr{v3(i.get('n_pairs'))}"
                 f"{'(sub' + v3(i.get('pair_subsample')) + ')' if i.get('pair_subsample') else ''} | v{i.get('n_vars')} "
                 f"[{dr.get('few', 0)}/{dr.get('levels', 0)}/{dr.get('const', 0)}/{dr.get('cap', 0)}] | "
                 f"{'/'.join(str(cl.get(c, 0)) for c in CLASSES)} | pl {tb.get('pl_any', 0)} | {v3(i.get('runtime_s'))}s"
                 + (f" | top {short(g(tb, 'top', 'var'))} {g(tb, 'top', 'cls')} {v3(g(tb, 'top', 'acq'))}" if tb.get("top")
                    else "") + (f" | var errors {sum(int(v['n']) for v in i['var_errors'].values())}"
                                if i.get("var_errors") else ""))
    S = R.get("SUM") or {}
    L.append("## SUM classes none/cal/day/within-day; flags C/D/K (excess >= .01/.02/.02); sig-any-size; "
             "placebo-as-real C/D/K (any); medians [q05,q95]; acq = cal + (1-cal) * same-occasion share; "
             "H hour curve n/tested (none) h")
    for k in ("pheno", "clinic", "home", "lab", "other", "tech"):
        s = S.get(k)
        if not s:
            continue
        c, f_, pf = s["classes"], s["flags"], s["pl_flags"]
        sg = s.get("sig") or {}
        hr = s.get("hour") or {}
        L.append(f"SUM {k} n{s['n_vars']} (timed {s['n_timed']}) cls {'/'.join(str(c[x]) for x in CLASSES)} flags "
                 f"{f_['C']}/{f_['D']}/{f_['K']} sig-any-size {sg.get('C')}/{sg.get('D')}/{sg.get('K')} pl {pf['C']}/"
                 f"{pf['D']}/{pf['K']} ({s['pl_any']}) | acq all {qs(s['acq'])} flagged({s['n_flagged']}) {qs(s['acq_flagged'])}"
                 f" | H {hr.get('n_H')}/{hr.get('n_tested')} (none {hr.get('n_H_none')}) h {qs(hr.get('r2'))}")
        cb = s.get("cq_base") or {}
        L.append(f"SUM {k} cal {qs(s['cal'])} sea {qs(s['seas'])} day {qs(s['day'])} drift {qs(s.get('c17'))} "
                 f"dind {qs(s.get('dind'))} dec {qs(s['dec'])} ks(identified) {qs(s['ks'])} | Cq base>=half "
                 f"{cb.get('half')}/{cb.get('n_cq')}")
        fx, un = s.get("fix_flagged") or {}, s.get("univ_all") or {}
        L.append(f"SUM {k} flag-guided, flagged vars: vr {qs(fx.get('vr'))} ssif {qs(fx.get('ssif'))}/"
                 f"{qs(fx.get('ssif_chg'))} dICC {qs(fx.get('dicc'))} | universal, all vars: vr {qs(un.get('vr'))} ssif "
                 f"{qs(un.get('ssif'))}/{qs(un.get('ssif_chg'))} dICC {qs(un.get('dicc'))}")
    for which, lab in (("REF_FLAG", "flag-guided"), ("REF_UNIV", "universal")):
        RS = R.get(which) or {}
        if not RS:
            continue
        L.append(f"## {which} phenotype reference associations (joint z(y)~z(age)+sex+z(BMI)+z(SBP), one unit/participant), "
                 f"{lab} correction vs raw: n=|z_raw|>=3, rel=(b_c-b_r)/|b_r|, >10%=share |rel|>.1, se=SE ratio, dabs=b_c-b_r")
        for nm in REF:
            x = RS.get(nm) or {}
            fx = (f"flagged n{x.get('f_n_eval')} rel {qs(x.get('f_rel'))} >10% {v3(x.get('f_frac_gt10'))} se "
                  f"{qs(x.get('f_se_ratio'))}")
            if which == "REF_FLAG":
                L.append(f"{which} {nm}: all n{x.get('n_eval')} >10% {v3(x.get('frac_gt10'))} | {fx}")
            else:
                L.append(f"{which} {nm}: all n{x.get('n_eval')} rel {qs(x.get('rel'))} >10% {v3(x.get('frac_gt10'))} se "
                         f"{qs(x.get('se_ratio'))} dabs {qs(x.get('dabs'))} | {fx}")
        L.append(f"{which} any association >10%: {v3(g(RS, 'any_gt10', 'frac'))} of {g(RS, 'any_gt10', 'n_vars')} vars | "
                 f"class none {v3(g(RS, 'any_gt10_none', 'frac'))} of {g(RS, 'any_gt10_none', 'n_vars')} | flagged "
                 f"{v3(g(RS, 'any_gt10_flagged', 'frac'))} of {g(RS, 'any_gt10_flagged', 'n_vars')}")
    CL = R.get("CLIN") or {}
    if CL and not CL.get("error"):
        L.append("## CLIN atlas raw vs v1.5, one row/participant: moved = share of the adverse top quintile (q5) / decile "
                 "(d10) that leaves it, abs or age-sex ranks [boot 95%] / share of all changing class; ssif parallel, change")
        for bed, _ in CLIN:
            o = CL.get(bed) or {}
            if o.get("status"):
                L.append(f"CLIN {bed}: {o['status']}")
                continue
            bt = o.get("boot") or {}

            def ci(k):
                c = bt.get(k)
                return f"[{v3(c[0])},{v3(c[1])}]" if c else ""
            L.append(f"CLIN {bed} n{o.get('n')} q5 abs {v3(g(o, 'abs_top20', 'moved'))}/{v3(g(o, 'abs_top20', 'changed_all'))}"
                     f" as {v3(g(o, 'as_top20', 'moved'))}{ci('as_top20')}/{v3(g(o, 'as_top20', 'changed_all'))} | d10 abs "
                     f"{v3(g(o, 'abs_top10', 'moved'))} as {v3(g(o, 'as_top10', 'moved'))}{ci('as_top10')} | ssif "
                     f"{v3(o.get('ssif1'))}{ci('ssif1')} chg {v3(o.get('ssif_chg'))}{ci('ssif_chg')} n{o.get('n_ret') or 'NA'} "
                     f"model {v3(o.get('ssif_model'))} ({o.get('p8_cls')}) | icc {v3(o.get('icc_r'))}>{v3(o.get('icc_c'))}")
    elif CL:
        L.append(f"CLIN: {CL.get('error')}")
    allr = R.get("_recs") or []

    def rk(r):
        return (0 if r.get("flags") else 1, -(fl(r.get("acq")) or 0.0))
    ph = sorted([r for r in allr if not r["tech"] and fl(r.get("acq")) is not None], key=rk)
    L.append("## TOP non-technical vars, flagged first, by acq. flags Cq/Cd=calendar quarter/drift; cal(/first-visit "
             "only if Cq) d=day dr=drift di=c0-c17 k z t dec(se) h(*=H) | F/U=flag-guided/universal: vr ssif par/chg icc bAge")
    for i, r in enumerate(ph[:R.get("_ntop", 25)]):
        L.append("TOP " + vline(r, i + 1))
    th = sorted([r for r in allr if r["tech"] and fl(r.get("acq")) is not None], key=rk)
    for i, r in enumerate(th[:R.get("_ntech", 4)]):
        L.append("TECH " + vline(r, i + 1))
    L.append("## CTRL separate devices: vascular-device pressures/PWV/ABI (vas) vs the sitting/lying/standing cuff (bp); "
             "height, BMI, ECG heart rate")
    rm = {f"{r['tid']}:{r['var']}": r for r in allr}
    for k in CTRL:
        r = rm.get(k)
        L.append("CTRL " + (vline(r) if r else f"{k}: not analysed"))
    XR = R.get("XR") or {}
    if XR and not XR.get("error") and not XR.get("status"):
        L.append("CTRL xr same-day stranger cross-correlation r(se) all pairs | <=60 min: " + " ; ".join(
            f"{k.replace('_raw', '')} {v3(g(v, 'all', 'r'))}({v3(g(v, 'all', 'se'))})|{v3(g(v, 'le60', 'r'))}"
            f"({v3(g(v, 'le60', 'se'))})" for k, v in XR.items() if isinstance(v, dict) and "|" in k))
    elif XR:
        L.append(f"CTRL xr: {XR.get('error') or XR.get('status')}")
    L.append(f"runtime {v3(R.get('runtime_s'))}s; blocks " + " ".join(f"{k} {v3(v)}" for k, v in (R.get("timings") or {}).items()))

    def wrap(ln):
        ln = re.sub(r"(?<![\w.])(-?)0\.(\d)", r"\1.\2", ln.strip())
        w = []
        while len(ln) > 220:
            w.append(ln[:220].rstrip())
            ln = "+" + ln[220:]
        return w + [ln]
    return [x for ln in L for x in wrap(ln)]


def make_digest(R):
    """Fit the digest into 120 lines: shorten the TOP list, then the TECH lines, then drop the TAB lines of technical
    tables; truncate only as a last resort."""
    for ntop, ntech in ((25, 4), (20, 3), (15, 2), (12, 1), (10, 0)):
        R["_ntop"], R["_ntech"] = ntop, ntech
        out = digest(R)
        if len(out) <= 120:
            return "\n".join(out) + "\n"
    out = [x for x in out if not (x.startswith("TAB ") and " tech " in x)]
    if len(out) > 120:
        out = out[:119] + [f"TRUNCATED {len(out) - 119} lines (see results.json)"]
    return "\n".join(out) + "\n"


# ============================================================================ main

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


def run_tables(specs, workers):
    jobs = sorted([(s, i) for i, s in enumerate(specs)], key=lambda j: -len(j[0]["vars"]))
    if workers > 1 and len(jobs) > 1:
        try:
            with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
                return list(pool.imap_unordered(audit_table, jobs))
        except Exception as ex:
            log(f"worker pool failed ({err(ex)}); running sequentially")
    return [audit_table(j) for j in jobs]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", default=None, help="atlas v1.5 script (default f1_atlas_v1_5.py, else f1_atlas_v1.py, "
                                                  "beside this file)")
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--placebo", type=int, default=20, help="placebo replicates of each kind (>= 6)")
    ap.add_argument("--boot", type=int, default=200, help="participant bootstrap of the clinical block")
    ap.add_argument("--workers", type=int, default=4, help="table workers (fork)")
    ap.add_argument("--tables", default="", help="debugging: comma-separated table abbreviations or relative paths")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    if a.placebo < 6:
        raise SystemExit("--placebo must be >= 6")
    here = os.path.dirname(os.path.abspath(__file__))
    path = a.atlas or next((p for p in (os.path.join(here, "f1_atlas_v1_5.py"), os.path.join(here, "f1_atlas_v1.py"))
                            if os.path.exists(p)), None)
    if path is None:
        raise SystemExit("atlas not found: pass --atlas f1_atlas_v1_5.py")
    set_atlas(path)
    t0 = time.time()
    import scipy
    log(f"{VERSION}; atlas {A.VERSION} ({os.path.basename(path)}); numpy {np.__version__}; pandas {pd.__version__}; "
        f"scipy {scipy.__version__}; pyarrow {pa.__version__}; python {sys.version.split()[0]}; "
        f"threads {os.environ.get('OMP_NUM_THREADS')}; workers {a.workers}")
    if not str(A.VERSION).startswith("f1_atlas_v1.5"):
        log(f"WARNING: atlas version {A.VERSION} is not v1.5")
    secs = [s.strip() for s in a.sections.split(",") if s.strip()]
    R = {"version": VERSION, "atlas_version": A.VERSION, "seed": f"{a.seed:,}", "sections": secs, "errors": {},
         "timings": {}, "settings": {"placebo": a.placebo, "boot": a.boot, "workers": a.workers, "min_units": MIN_UNITS,
                                     "min_levels": MIN_LEVELS, "max_vars": MAX_VARS, "q_alpha": Q_ALPHA,
                                     "min_excess": MIN_EX, "edges": list(A.SESSION_EDGES), "taus": list(A.SESSION_TAUS),
                                     "jk": A.SESSION_JK, "min_bin": A.SESSION_MIN_BIN, "min_z": A.SESSION_MIN_Z}}
    t1 = time.time()
    fr, medflags, htn = A.build_frame(a.root, None)
    R["timings"]["frame"] = time.time() - t1
    pids = set(fr["participant_id"].astype(str))
    pop = A.rd(a.root, "population/population.parquet")
    pids |= set(pop["participant_id"].astype(str))
    sex = pd.to_numeric(pop["sex"].astype(str), errors="coerce")
    CTX["sex"] = dict(zip(pop["participant_id"].astype(str), sex))
    CTX["yob"] = dict(zip(pop["participant_id"].astype(str), pd.to_numeric(pop["year_of_birth"], errors="coerce")))
    an = fr[["participant_id", "research_stage", "age", "visit_date"] +
            [c for c in ("bmi", "sbp_sit") if c in fr.columns]].copy()
    for c in ("bmi", "sbp_sit"):
        if c not in an.columns:
            an[c] = np.nan
    an["participant_id"] = an["participant_id"].astype(str)
    an["research_stage"] = an["research_stage"].astype(str)
    an["visit_date"] = pd.to_datetime(an["visit_date"], errors="coerce").dt.as_unit("ns")
    CTX["anchor"] = an.drop_duplicates(["participant_id", "research_stage"])
    comp = A.relation_components(a.root)
    CTX["comp"], CTX["root"], CTX["args"] = comp, a.root, a
    only = {x.strip() for x in a.tables.split(",") if x.strip()} or None
    inc, exc, nliv = discover(a.root, only)
    exd = {}
    for rel, why in exc:
        exd.setdefault(why, []).append(mask_name(rel))
    R["info"] = {"relations_excluded": comp is not None, "liver_fields": nliv,
                 "liver_tables": len(exd.get("liver", [])), "frame_has_att": "att" in fr.columns,
                 "excluded": exd, "included": [mask_name(s["rel"]) for s in inc]}
    log(f"discovery: {len(inc)} tables included, {len(exc)} excluded; liver fields excluded {nliv}")
    recs = []
    if "AUDIT" in secs:
        t1 = time.time()
        try:
            tabs = run_tables(inc, a.workers)
            for tb in tabs:
                for ln in tb.pop("log", []):
                    A.LOG.append(ln)
                    print(ln, flush=True)
            tabs = sorted(tabs, key=lambda tb: list(ABBR.values()).index(tb["tid"]) if tb["tid"] in ABBR.values() else 99)
            for tb in tabs:
                recs += tb.get("vars", [])
            classify(recs)
            for tb in tabs:
                vs = tb.get("vars", [])
                tb["classes"] = {c: sum(r["cls"] == c for r in vs) for c in CLASSES}
                tb["pl_any"] = sum(bool(r["pl_flags"]) for r in vs)
                top = sorted([r for r in vs if r["flags"] and not r["tech"]] or [r for r in vs if r["flags"]],
                             key=lambda r: -(r.get("acq") or 0.0))
                if top:
                    tb["top"] = {"var": top[0]["var"], "acq": top[0].get("acq"), "cls": top[0]["cls"]}
            chk = [tb["info"].get("kfit_replica_maxdiff") for tb in tabs if tb.get("info", {}).get("kfit_replica_maxdiff")
                   is not None]
            R["info"]["kfit_replica_maxdiff"] = max(chk) if chk else None
            R["info"]["kfit_replica_n"] = sum(tb.get("info", {}).get("kfit_replica_n") or 0 for tb in tabs)
            R["tables"] = [{k: v for k, v in tb.items() if k != "vars"} for tb in tabs]
            for tb in tabs:
                if tb.get("error"):
                    R["errors"][f"table {tb['tid']}"] = tb["error"]
            R["SUM"] = summaries(recs)
            ph = [r for r in recs if not r["tech"]]  # reference-association summaries: phenotype variables only
            R["REF_FLAG"] = ref_summary(ph, "cons_flag")
            R["REF_UNIV"] = ref_summary(ph, "cons_univ")
            R["variables"] = recs
            R["_recs"] = recs
        except Exception as ex:
            R["errors"]["AUDIT"] = err(ex)
            log(f"AUDIT FAILED: {R['errors']['AUDIT']}")
        R["timings"]["audit"] = time.time() - t1
    recmap = {f"{r['tid']}:{r['var']}": r for r in recs}
    blocks = (("CHECK", lambda: block_check(fr, comp)),
              ("CLIN", lambda: block_clin(fr, recmap, np.random.default_rng([a.seed, 999]), a.boot)),
              ("XR", lambda: block_xr(fr, comp, np.random.default_rng([a.seed, 998]))))
    for key, fn in blocks:
        if key not in secs:
            continue
        t1 = time.time()
        try:
            R[key] = fn()
        except Exception as ex:
            R[key] = {"error": err(ex)}
            R["errors"][key] = R[key]["error"]
            log(f"{key} FAILED: {R[key]['error']}")
        R["timings"][key] = time.time() - t1
        log(f"block {key} done in {R['timings'][key]:.1f}s")
    if isinstance(R.get("CHECK"), dict):
        for bed in ("pwv", "abi_min"):
            x = R["CHECK"].get(bed)
            if isinstance(x, dict):
                x.pop("values", None)
    R["runtime_s"] = time.time() - t0
    text = json.dumps(clean(R), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    privacy_check(text, pids)
    R2 = json.loads(text)
    R2["_recs"] = R2.get("variables") or []
    dg = make_digest(R2)
    privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines (max {max(len(x) for x in dg.splitlines())} chars); total {time.time() - t0:.1f}s")
    logtext = LONG.sub("#", "\n".join(A.LOG)) + "\n"  # free text (e.g. a table path quoted by A.rd): mask, then check
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
