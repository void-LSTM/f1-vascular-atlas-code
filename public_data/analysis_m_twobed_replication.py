#!/usr/bin/env python3
"""F1 Stage 0, task m (PLAN 9.3a L3): external replication of bed heterogeneity of drivers in open two-bed cohorts.

GATED: refuses to run on real data without --real. --simulate
runs a wiring test on fake tables with the real column names.

Pre-specified analysis
----------------------
Question: in open cohorts that measured two vascular beds in the same people, do upstream drivers relate differently
to the two beds, and do the between-bed differences point the same way as the HPP atlas rows?
Cohorts and bed pairs (complete cases on both beds and on every variable used for that cohort):
  Keio 890   baPWV > 1400 vs IMT+ (binary, logistic; log-OR per SD). Cross-modality pair (oscillometric vs carotid
             ultrasound). Both rows of the duplicated serial number excluded.
  Japan 912  baPWV max vs ABI max (higher leg; values > 3 already missing; ABI >= 0.9 required). OLS.
  China 36k  baPWV mean (L/R) vs ABI min (lower leg), ABI min >= 0.9 required (baPWV invalid below). OLS.
  IMPACT     baPWV mean vs ABI min, ABI min >= 0.9, age >= 20, haemodialysis excluded (patients). OLS.
  baPWV + ABI pairs come from one four-limb oscillometric acquisition: same-device pairs = ONE modality (flagged).
Orientation and scaling: continuous beds standardised per SD in the analysis sample, adverse-oriented (ABI
  sign-flipped). Continuous drivers per SD (re-standardised in every bootstrap sample); binary drivers (male,
  smoker, diabetes, hypertension/hyperlipidaemia history) exposed vs not.
Drivers (as available):
  Keio   MA = age, male, DBP, PP, BMI, HbA1c, smoker (1 - non_smoker), LDL, HDL, log TG, eGFR; extra rows in the
         one-driver models: SBP, diabetes (DM+), waist, log VAT.
  Japan  MA = age, male, DBP, PP, BMI, fasting glucose, current smoker, LDL, HDL, log TG, eGFR; extra: SBP.
  China  MA = age, male. Sensitivity: + same-acquisition implied brachial SBP (= ankle SBP / ABI: mathematically
         coupled to ABI; never used for concordance).
  IMPACT MA = age, male, BMI, diabetes, smoking, eGFR (recomputed from creatinine with the Japanese equation
         194 x Cr^-1.094 x age^-0.287 x 0.739 if female; recorded eGFR 36 % missing), hypertension history,
         hyperlipidaemia history. No BP, heart rate or lipid values in IMPACT.
  No cohort has heart rate. No liver phenotypes (fatty liver, MAFLD, GGT, ALT, AST) are used anywhere.
Models (each bed fitted in the same people):
  MA  (primary for heterogeneity) all MA drivers mutually adjusted.
  MIN (primary for HPP concordance; = HPP atlas M1) one driver + age + male.
  M3  (Keio, Japan; ~ HPP atlas M3 without antihypertensive drugs) one driver + age + male + BMI + SBP (the driver
      itself removed from the covariates; DBP and PP rows are therefore DBP | SBP and PP | SBP).
  BPJ (Keio, Japan) DBP + PP + age + male (analogue of the HPP joint DBP/PP footprint model).
Between-bed difference per driver: Delta = beta(baPWV) - beta(second bed), both adverse-oriented, from a paired
  person-level bootstrap (persons resampled with replacement; every model refitted on the same resample;
  B = --n-boot, default 2000; seed --seed, default 20261003): bootstrap SE, percentile 95 % CI, z-test p.
  Non-converged logistic refits (or |coef| >= 15) are dropped and counted.
Heterogeneity: joint Wald chi2 = Delta' S^-1 Delta with S = bootstrap covariance of the Delta vector (MA model, all
  drivers; df = number of drivers). Secondary: excluding age; excluding age and sex; the MIN-model Delta vector;
  and a gain-only null beta(baPWV) = c x beta(second bed) (minimised over c incl. +-inf; df = K - 1), absorbing a common
  scale difference between beds (latent-scale or reliability differences).
Keio scale: the log-OR difference is primary for inference. A latent-scale version (log-OR x sqrt(3)/pi) is
  reported for magnitude comparison with HPP SD-unit rows (constant rescaling: identical tests). Sensitivity:
  McKelvey-Zavoina fully standardised coefficients (log-OR / sqrt(var(linear predictor) + pi^2/3), per bed).
BP component lean: L(bed) = beta(DBP) - beta(PP) per 10 mmHg (also per SD) in BPJ (primary) and MA; Delta L =
  L(baPWV) - L(second bed), compared with HPP BPF joint DBP - PP per bed (IV 2SLS and obs_rc columns).
HPP concordance: MIN rows vs HPP atlas v1.5 M1 rows, M3 rows vs atlas M3 rows (ATL; parsed with line numbers), lean
  rows vs BPF. PWV - IMT rows use the printed HPP pwv-imt difference [CI]; baPWV - ABI rows use the HPP difference
  of the printed oriented betas pwv - abi (point only). Sign agreement per row; exact binomial sign test across
  rows (rows are not independent: correlated drivers, shared cohorts). Mapping: glycaemia -> HPP hba1c (Keio
  HbA1c, IMPACT diabetes = proxy) or glucose (Japan FPG); smoker -> smk_current; sex, hypertension and
  hyperlipidaemia history and implied SBP have no comparable HPP row.
Caveats: baPWV (brachial-ankle) is not HPP femoral-ankle PWV; Japan ABI max is the higher leg; Keio thresholds/
  definitions of PWV > 1400 and IMT+ undocumented; codings inferred (China sex, IMPACT abbreviations, smoking
  current vs ever); BP acquisition relative to PWV not stated (Keio, Japan); exploratory.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import re  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import optimize, stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_survival as ls  # noqa: E402
import s0an_common as C  # noqa: E402

TASK = "m_twobed_replication"
N_WORKERS = 4
LATENT = np.sqrt(3) / np.pi
AGG = C.PAPER / "results" / "real_aggregates"
HPP_FILES = {"ATL": AGG / "hpp_run_20261003" / "f1_atlas_v1_5_digest_transcribed.txt",
             "EXP": AGG / "hpp_run_20261004" / "f1_exposome_v1_a15_digest_transcribed.txt",
             "BPF": AGG / "hpp_run_20261005" / "f1_bp_footprint_v1_digest_transcribed.txt"}
CONT = {"age", "sbp", "dbp", "pp", "bmi", "waist", "log_vat", "glyc", "ldl", "hdl", "log_tg", "egfr", "sbp_impl"}
LABEL = {"age": "age", "male": "male sex", "dbp": "DBP", "pp": "PP", "sbp": "SBP", "bmi": "BMI", "waist": "waist",
         "log_vat": "log VAT", "glyc": "glycaemia", "diabetes": "diabetes", "smoker": "smoking", "ldl": "LDL-C",
         "hdl": "HDL-C", "log_tg": "log TG", "egfr": "eGFR", "htn_hx": "hypertension history",
         "hl_hx": "hyperlipidaemia history", "sbp_impl": "implied brachial SBP (coupled)"}

SPECS = {
    "keio": dict(label="Keio 890", table="keio_bapwv_imt", family="binomial",
                 used=["age_y", "male", "sbp_mmhg", "dbp_mmhg", "bmi_kgm2", "waist_cm", "vat_cm2", "hba1c_pct",
                       "diabetes", "non_smoker", "ldl_mgdl", "hdl_mgdl", "tg_mgdl", "egfr", "bapwv_gt1400",
                       "imt_positive", "qc_duplicate_serial"],
                 beds=[("pwv", "baPWV>1400"), ("imt", "IMT+")], hpp_pair="pwv-imt", same_device=False,
                 ma=["age", "male", "dbp", "pp", "bmi", "glyc", "smoker", "ldl", "hdl", "log_tg", "egfr"],
                 extra=["sbp", "diabetes", "waist", "log_vat"], m3=True, bpj=True, glyc="HbA1c",
                 glyc_hpp="hba1c", sens=[]),
    "japan": dict(label="Japan 912", table="japan_ggt_bapwv", family="gaussian",
                  used=["age_y", "male", "sbp_mmhg", "dbp_mmhg", "bmi_kgm2", "fpg_mgdl", "current_smoker",
                        "ldl_mgdl", "hdl_mgdl", "tg_mgdl", "egfr", "bapwv_max_cms", "abi_max"],
                  beds=[("pwv", "baPWV max"), ("abi", "ABI max (sign-flipped)")], hpp_pair="pwv-abi",
                  same_device=True,
                  ma=["age", "male", "dbp", "pp", "bmi", "glyc", "smoker", "ldl", "hdl", "log_tg", "egfr"],
                  extra=["sbp"], m3=True, bpj=True, glyc="fasting glucose", glyc_hpp="glucose", sens=[]),
    "china": dict(label="China 36k", table="china_bapwv_36k", family="gaussian",
                  used=["age_y", "sex_male_probable", "bapwv_mean_cms", "abi_min", "brachial_sbp_implied_mmhg"],
                  beds=[("pwv", "baPWV mean"), ("abi", "ABI min (sign-flipped)")], hpp_pair="pwv-abi",
                  same_device=True, ma=["age", "male"], extra=[], m3=False, bpj=False, glyc=None, glyc_hpp=None,
                  sens=[("SENS_sbpimpl", ["age", "male", "sbp_impl"])]),
    "impact": dict(label="IMPACT", table="impact_abi", family="gaussian",
                   used=["age_y", "female", "bmi_kgm2_calc", "diabetes", "smoking", "creatinine_mgdl", "htn",
                         "hyperlipidaemia", "hemodialysis", "abi_min", "bapwv_mean_cms"],
                   beds=[("pwv", "baPWV mean"), ("abi", "ABI min (sign-flipped)")], hpp_pair="pwv-abi",
                   same_device=True, ma=["age", "male", "bmi", "diabetes", "smoker", "egfr", "htn_hx", "hl_hx"],
                   extra=[], m3=False, bpj=False, glyc=None, glyc_hpp="hba1c", sens=[]),
}
ORDER = ["keio", "japan", "china", "impact"]
CHUNKS = {"keio": 4, "japan": 2, "china": 2, "impact": 1}


# ---------------------------------------------------------------------------------------------- preparation
def prep(key, raw):
    if key == "keio":
        r = raw[raw["qc_duplicate_serial"] != 1]
        d = pd.DataFrame({"age": r["age_y"], "male": r["male"], "sbp": r["sbp_mmhg"], "dbp": r["dbp_mmhg"],
                          "pp": r["sbp_mmhg"] - r["dbp_mmhg"], "bmi": r["bmi_kgm2"], "waist": r["waist_cm"],
                          "log_vat": np.log(r["vat_cm2"]), "glyc": r["hba1c_pct"], "diabetes": r["diabetes"],
                          "smoker": 1 - r["non_smoker"], "ldl": r["ldl_mgdl"], "hdl": r["hdl_mgdl"],
                          "log_tg": np.log(r["tg_mgdl"]), "egfr": r["egfr"], "pwv": r["bapwv_gt1400"],
                          "imt": r["imt_positive"]})
        rule = "duplicated serial excluded; complete cases"
    elif key == "japan":
        r = raw[raw["abi_max"] >= 0.9]
        d = pd.DataFrame({"age": r["age_y"], "male": r["male"], "sbp": r["sbp_mmhg"], "dbp": r["dbp_mmhg"],
                          "pp": r["sbp_mmhg"] - r["dbp_mmhg"], "bmi": r["bmi_kgm2"], "glyc": r["fpg_mgdl"],
                          "smoker": r["current_smoker"], "ldl": r["ldl_mgdl"], "hdl": r["hdl_mgdl"],
                          "log_tg": np.log(r["tg_mgdl"]), "egfr": r["egfr"], "pwv": r["bapwv_max_cms"],
                          "abi": -r["abi_max"]})
        rule = "ABI max valid (>= 0.9; values > 3 missing); complete cases"
    elif key == "china":
        r = raw[raw["abi_min"] >= 0.9]
        d = pd.DataFrame({"age": r["age_y"], "male": r["sex_male_probable"], "sbp_impl":
                          r["brachial_sbp_implied_mmhg"], "pwv": r["bapwv_mean_cms"], "abi": -r["abi_min"]})
        rule = "ABI min >= 0.9 (both legs); baPWV sentinels missing; complete cases"
    elif key == "impact":
        r = raw[(raw["abi_min"] >= 0.9) & (raw["age_y"] >= 20) & (raw["hemodialysis"] == 0)]
        egfr = 194 * r["creatinine_mgdl"] ** -1.094 * r["age_y"] ** -0.287 * np.where(r["female"] == 1, 0.739, 1.0)
        d = pd.DataFrame({"age": r["age_y"], "male": 1 - r["female"], "bmi": r["bmi_kgm2_calc"],
                          "diabetes": r["diabetes"], "smoker": r["smoking"], "egfr": egfr, "htn_hx": r["htn"],
                          "hl_hx": r["hyperlipidaemia"], "pwv": r["bapwv_mean_cms"], "abi": -r["abi_min"]})
        rule = "ABI min >= 0.9, age >= 20, haemodialysis excluded; eGFR from creatinine; complete cases"
    else:
        raise KeyError(key)
    return d.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True), rule


def build_models(spec):
    """(name, design columns, focus columns)."""
    models = [("MA", spec["ma"], spec["ma"])]
    for k in spec["ma"] + spec["extra"]:
        models.append((f"MIN:{k}", [k] + [c for c in ("age", "male") if c != k], [k]))
    if spec["bpj"]:
        models.append(("BPJ", ["dbp", "pp", "age", "male"], ["dbp", "pp"]))
    if spec["m3"]:
        for k in spec["ma"] + spec["extra"]:
            models.append((f"M3:{k}", [k] + [c for c in ("age", "male", "bmi", "sbp") if c != k], [k]))
    for name, cols in spec["sens"]:
        models.append((name, cols, cols))
    return models


# ---------------------------------------------------------------------------------------------- fitting
def design(arr, cols):
    n = len(next(iter(arr.values())))
    X = np.empty((n, len(cols) + 1))
    X[:, 0] = 1.0
    for j, c in enumerate(cols):
        x = arr[c]
        X[:, j + 1] = (x - x.mean()) / x.std(ddof=1) if c in CONT else x
    return X


def logit_newton(X, y, b0, max_iter=50, tol=1e-9):
    b = b0.copy()
    for _ in range(max_iter):
        mu = 1.0 / (1.0 + np.exp(-(X @ b)))
        w = mu * (1.0 - mu)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        b = b + step
        if not np.all(np.isfinite(b)):
            return b, False
        if np.max(np.abs(step)) < tol:
            return b, True
    return b, False


def sample_coefs(arr, spec, models, warm=None):
    """All coefficients of interest in one sample. Returns (dict key -> value, ok, fitted coefs for warm start)."""
    out, fits, ok = {}, {}, True
    beds = [b for b, _ in spec["beds"]]
    for mname, cols, focus in models:
        X = design(arr, cols)
        if spec["family"] == "gaussian":
            Y = np.column_stack([(arr[b] - arr[b].mean()) / arr[b].std(ddof=1) for b in beds])
            B = np.linalg.lstsq(X, Y, rcond=None)[0]
            coefs = {b: B[:, i] for i, b in enumerate(beds)}
        else:
            coefs = {}
            for b in beds:
                y = arr[b]
                if warm is None:
                    f = ls.glm_fit(y, X, family="binomial", vcov="model")
                    cb, conv = f.coef, f.converged
                else:
                    cb, conv = logit_newton(X, y, warm[(mname, b)])
                ok = ok and conv and bool(np.all(np.abs(cb) < 15))
                coefs[b] = cb
        for b in beds:
            cb = coefs[b]
            fits[(mname, b)] = cb
            for f_ in focus:
                out[f"{mname}|{b}|{f_}"] = cb[1 + cols.index(f_)]
            if mname == "MA" and spec["family"] == "binomial":
                lp = X[:, 1:] @ cb[1:]
                out[f"MA|{b}|__mzsd"] = np.sqrt(lp.var(ddof=1) + np.pi ** 2 / 3)
    for c in ("dbp", "pp"):
        if c in arr:
            out[f"SD|{c}"] = arr[c].std(ddof=1)
    return out, ok, fits


def _boot_worker(task):
    key, d, n_rep, seed_seq, warm = task
    spec, models = SPECS[key], build_models(SPECS[key])
    rng = np.random.default_rng(seed_seq)
    arr = {c: d[c].to_numpy(float) for c in d.columns}
    n = len(d)
    rows, fails = [], 0
    for _ in range(n_rep):
        idx = rng.integers(0, n, n)
        sa = {c: a[idx] for c, a in arr.items()}
        try:
            o, ok, _ = sample_coefs(sa, spec, models, warm)
        except np.linalg.LinAlgError:
            ok = False
        if not ok:
            fails += 1
            continue
        rows.append(o)
    return key, rows, fails


# ---------------------------------------------------------------------------------------------- summaries
def summ(full, boot):
    boot = np.asarray(boot, float)
    se = boot.std(ddof=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2 * stats.norm.sf(abs(full / se)) if se > 0 else np.nan
    return {"est": float(full), "boot_se": float(se), "lower": float(lo), "upper": float(hi), "p": float(p)}


def wald(dfull, dboot):
    S = np.cov(np.asarray(dboot, float), rowvar=False)
    S = np.atleast_2d(S)
    d = np.asarray(dfull, float)
    chi2 = float(d @ np.linalg.solve(S, d))
    return chi2, len(d), float(stats.chi2.sf(chi2, len(d)))


def gain_only(bA, bB, BA, BB):
    """Rank-1 (collinearity) null cos(t) bA - sin(t) bB = 0, i.e. bA = c bB with c = tan(t) (c = inf: bB = 0).
    Wald statistic with bootstrap covariance minimised over t in [0, pi) (grid + local refinement); df K - 1."""
    def q(th):
        a, b = np.cos(th), np.sin(th)
        S = np.atleast_2d(np.cov(a * BA - b * BB, rowvar=False))
        r = a * bA - b * bB
        return float(r @ np.linalg.solve(S, r))
    grid = np.linspace(0, np.pi, 720, endpoint=False)
    i = int(np.argmin([q(t) for t in grid]))
    step = np.pi / 720
    res = optimize.minimize_scalar(q, bounds=(grid[i] - step, grid[i] + step), method="bounded")
    th = res.x % np.pi
    c = float(np.tan(th)) if abs(np.cos(th)) > 1e-9 else np.inf
    df = len(bA) - 1
    return c, float(res.fun), df, float(stats.chi2.sf(res.fun, df)) if df > 0 else np.nan


def ci_str(e, lo=None, hi=None, nd=3):
    if e is None or (isinstance(e, float) and not np.isfinite(e)):
        return "NA"
    s = f"{e:.{nd}g}"
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        s += f" [{lo:.{nd}g}, {hi:.{nd}g}]"
    return s


# ---------------------------------------------------------------------------------------------- HPP rows
_NUMCI = re.compile(r"(-?[\d.]+(?:e[-+]?\d+)?)\s*\[(-?[\d.]+(?:e[-+]?\d+)?),\s*(-?[\d.]+(?:e[-+]?\d+)?)\]")


def _numci(s):
    m = _NUMCI.search(s)
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (np.nan, np.nan, np.nan)


def parse_hpp():
    atl, model = {}, None
    for i, ln in enumerate(HPP_FILES["ATL"].read_text(encoding="utf-8").splitlines(), 1):
        if ln.startswith("# M1_age_sex"):
            model = "M1"
            continue
        if ln.startswith("# M3_plus_bp_meds"):
            model = "M3"
            continue
        if ln.startswith("#"):
            model = None
            continue
        if model and " | " in ln:
            f = [x.strip() for x in ln.split("|")]
            b = [float(x) for x in f[2].split()]
            d, lo, hi = _numci(f[3])
            atl[(model, f[0])] = {"line": i, "n": f[1], "pwv": b[0], "imt": b[1], "art_w": b[2], "abi": b[3],
                                  "pwv_imt": d, "pwv_imt_lo": lo, "pwv_imt_hi": hi, "wald_p": f[5]}
    bpf, on = {}, False
    for i, ln in enumerate(HPP_FILES["BPF"].read_text(encoding="utf-8").splitlines(), 1):
        if ln.startswith("# joint DBP_gw+PP_gw"):
            on = True
            continue
        if on and ln.startswith("#"):
            break
        if on and " | " in ln:
            f = [x.strip() for x in ln.split("|")]
            bpf[f[0]] = {"line": i, "iv_dbp": _numci(f[3]), "iv_pp": _numci(f[4]), "iv_lean": _numci(f[5]),
                         "obs_dbp": _numci(f[6])[0], "obs_pp": _numci(f[7])[0]}
    exp = {}
    for i, ln in enumerate(HPP_FILES["EXP"].read_text(encoding="utf-8").splitlines(), 1):
        if ln.startswith("ecg.hr_bpm |") and "hr" not in exp:
            f = [x.strip() for x in ln.split("|")]
            exp["hr"] = {"line": i, "pwv_imt": _numci(f[3]), "betas": f[2]}
    return atl, bpf, exp


def hpp_key(spec, k):
    """External driver -> (HPP exposure, note). None = no comparable HPP row."""
    if k == "glyc":
        return spec["glyc_hpp"], ""
    if k == "diabetes":
        return "hba1c", "proxy (binary diabetes vs HPP HbA1c per SD)"
    m = {"age": "age", "bmi": "bmi", "dbp": "dbp_sit", "pp": "pp_sit", "sbp": "sbp_sit", "egfr": "egfr",
         "smoker": "smk_current", "ldl": "ldl", "hdl": "hdl", "log_tg": "log_tg", "waist": "waist",
         "log_vat": "log_vat"}
    if k in m:
        return m[k], ""
    why = {"male": "sex is an adjuster in HPP atlas/EXP (no sex main-effect row)",
           "htn_hx": "no hypertension-history row in ATL/EXP", "hl_hx": "no hyperlipidaemia-history row",
           "sbp_impl": "coupled to ABI (not comparable)"}
    return None, why.get(k, "no row")


# ---------------------------------------------------------------------------------------------- simulation
SIM_KEIO = {"pwv": {"age": 1.1, "dbp": 0.7, "pp": 0.1}, "imt": {"age": 0.9, "dbp": 0.0, "pp": 0.5}}
SIM_JP = {"pwv": {"age": 0.5, "dbp": 0.4, "pp": 0.1}, "abi": {"age": 0.1, "dbp": -0.1, "pp": 0.2}}  # adverse units
SIM_CN = {"pwv": 2.0, "abi": -0.03}   # age effect: m/s per SD age; ABI units per SD age (raw orientation)


def _z(x):
    return (x - x.mean()) / x.std(ddof=1)


def sim_tables(rng):
    out = {}
    n = 890
    age = np.clip(rng.normal(60, 13, n), 23, 94).round()
    sbp = (121 + 0.5 * (age - 60) + rng.normal(0, 15, n)).round()
    dbp = (0.45 * sbp + 22 + rng.normal(0, 7, n)).round()
    bmi = rng.normal(23.6, 3.4, n).round(1)
    hba1c = rng.normal(5.8, 0.6, n).round(1)
    ldl = rng.normal(116, 29, n).round()
    lp = {b: (-0.1 if b == "pwv" else -1.2) + P["age"] * _z(age) + P["dbp"] * _z(dbp) + P["pp"] * _z(sbp - dbp)
          for b, P in SIM_KEIO.items()}
    y = {b: rng.binomial(1, 1 / (1 + np.exp(-v))).astype(float) for b, v in lp.items()}
    y["pwv"][rng.choice(n, 8, replace=False)] = np.nan
    dup = np.zeros(n, dtype="int64")
    dup[[258, 259]] = 1
    out["keio"] = pd.DataFrame({
        "age_y": age.astype("int64"), "male": rng.binomial(1, 0.67, n).astype("int64"), "sbp_mmhg": sbp,
        "dbp_mmhg": dbp, "bmi_kgm2": bmi, "waist_cm": (3 * bmi + 13 + rng.normal(0, 4, n)).round(1),
        "vat_cm2": np.exp(np.log(94) + 0.08 * (bmi - 23.6) + rng.normal(0, 0.4, n)).round(1), "hba1c_pct": hba1c,
        "diabetes": (hba1c >= 6.5).astype("int64"), "non_smoker": rng.binomial(1, 0.5, n).astype("int64"),
        "ldl_mgdl": ldl, "hdl_mgdl": rng.normal(58, 15, n).round(),
        "tg_mgdl": np.exp(rng.normal(np.log(100), 0.5, n)).round(), "egfr": rng.normal(70, 12, n).round()
        .astype("int64"), "bapwv_gt1400": y["pwv"], "imt_positive": y["imt"], "qc_duplicate_serial": dup})
    n = 912
    age = np.clip(rng.normal(51, 10, n), 24, 84).round()
    sbp = (120 + 0.4 * (age - 51) + rng.normal(0, 14, n)).round(1)
    dbp = (0.45 * sbp + 22 + rng.normal(0, 7, n)).round(1)
    zz = {"age": _z(age), "dbp": _z(dbp), "pp": _z(sbp - dbp)}
    pwv = 1400 + 150 * sum(SIM_JP["pwv"][k] * zz[k] for k in zz) + rng.normal(0, 150, n)
    abi = 1.18 - 0.06 * sum(SIM_JP["abi"][k] * zz[k] for k in zz) + rng.normal(0, 0.06, n)
    abi[rng.choice(n, 36, replace=False)] = np.nan
    out["japan"] = pd.DataFrame({
        "age_y": age, "male": rng.binomial(1, 0.65, n).astype("int64"), "sbp_mmhg": sbp, "dbp_mmhg": dbp,
        "bmi_kgm2": rng.normal(23, 3.2, n).round(2), "fpg_mgdl": rng.normal(98, 12, n).round(),
        "current_smoker": rng.binomial(1, 0.22, n).astype(float), "ldl_mgdl": rng.normal(128, 30, n).round(),
        "hdl_mgdl": rng.normal(53, 13, n).round(1), "tg_mgdl": np.exp(rng.normal(np.log(81), 0.5, n)).round(),
        "egfr": rng.normal(70, 12, n).round(1), "bapwv_max_cms": pwv.round(), "abi_max": abi.round(2)})
    n = 36223
    age = rng.integers(30, 81, n).astype(float)
    pwv = 14 + SIM_CN["pwv"] * _z(age) + rng.normal(0, 2, n)
    abi = 1.12 + SIM_CN["abi"] * _z(age) + rng.normal(0, 0.08, n)
    out["china"] = pd.DataFrame({"age_y": age, "sex_male_probable": rng.binomial(1, 0.63, n).astype(float),
                                 "bapwv_mean_cms": (100 * pwv).round(1), "abi_min": abi.round(2),
                                 "brachial_sbp_implied_mmhg": rng.normal(128, 18, n)})
    n = 3056
    age = np.clip(rng.normal(67, 13, n), 13, 99).round()
    out["impact"] = pd.DataFrame({
        "age_y": age, "female": rng.binomial(1, 0.28, n).astype(float), "bmi_kgm2_calc": rng.normal(23.4, 3.5, n),
        "diabetes": rng.binomial(1, 0.28, n).astype(float), "smoking": rng.binomial(1, 0.46, n).astype(float),
        "creatinine_mgdl": np.exp(rng.normal(np.log(0.9), 0.3, n)), "htn": rng.binomial(1, 0.6, n).astype(float),
        "hyperlipidaemia": rng.binomial(1, 0.43, n).astype(float),
        "hemodialysis": rng.binomial(1, 0.05, n).astype(float), "abi_min": rng.normal(1.05, 0.15, n).round(2),
        "bapwv_mean_cms": (1600 + 15 * (age - 67) + rng.normal(0, 300, n)).round()})
    return out


# ---------------------------------------------------------------------------------------------- main
def main():
    args = C.parse_args(__doc__.splitlines()[0])
    mode, outdir, token = C.gate(args, TASK)
    rep = C.Report(TASK, "F1 Stage 0 task m - external replication of bed heterogeneity of drivers (PLAN 9.3a L3)",
                   mode, args)
    if mode == "simulate":
        raws = sim_tables(np.random.default_rng(args.seed))
        rep.h("Wiring check against the real schemas (column names/types only)")
        for k in ORDER:
            rep.table(f"wiring_check_{k}", C.check_columns(raws[k], SPECS[k]["table"], SPECS[k]["used"]),
                      max_rows=None)
    else:
        raws = {}
        for k in ORDER:
            raws[k], info = C.load_real(SPECS[k]["table"], token, SPECS[k]["used"])
            rep.inputs.append(info)
    for nm, p in HPP_FILES.items():
        rep.inputs.append({"table": f"HPP aggregate digest {nm}", "path": str(p), "sha256": C.sha256(p)})
    atl, bpf, exp = parse_hpp()

    data, rules, models_by, full, warm = {}, {}, {}, {}, {}
    for k in ORDER:
        data[k], rules[k] = prep(k, raws[k])
        models_by[k] = build_models(SPECS[k])
        arr = {c: data[k][c].to_numpy(float) for c in data[k].columns}
        o, ok, fits = sample_coefs(arr, SPECS[k], models_by[k], None)
        if not ok:
            raise RuntimeError(f"full-sample fit failed for {k}")
        full[k], warm[k] = pd.Series(o), fits

    # ---- paired bootstrap (parallel, deterministic seeds per chunk)
    tasks, seeds = [], np.random.SeedSequence(args.seed).spawn(sum(CHUNKS.values()))
    si = 0
    for k in ORDER:
        base, rem = divmod(args.n_boot, CHUNKS[k])
        for c in range(CHUNKS[k]):
            tasks.append((k, data[k], base + (1 if c < rem else 0), seeds[si], warm[k]))
            si += 1
    boot_rows = {k: [] for k in ORDER}
    fails = {k: 0 for k in ORDER}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for k, rows, nf in ex.map(_boot_worker, tasks):
            boot_rows[k] += rows
            fails[k] += nf
    boot = {k: pd.DataFrame(boot_rows[k])[full[k].index] for k in ORDER}

    # ---- analysis sets
    sets = []
    for k in ORDER:
        s, d = SPECS[k], data[k]
        prev = {b: (f"{d[b].mean():.3f}" if s["family"] == "binomial" else "continuous") for b, _ in s["beds"]}
        sets.append({"cohort": s["label"], "pair": " vs ".join(lbl for _, lbl in s["beds"]),
                     "modality": "same device (one modality)" if s["same_device"] else "two modalities",
                     "model": "logistic" if s["family"] == "binomial" else "OLS (beds per SD)", "n": len(d),
                     "prevalence_bed1": prev[s["beds"][0][0]], "prevalence_bed2": prev[s["beds"][1][0]],
                     "MA drivers": ", ".join(LABEL[c] if c != "glyc" else s["glyc"] for c in s["ma"]),
                     "extra one-driver rows": ", ".join(LABEL[c] for c in s["extra"]) or "-",
                     "rule": rules[k], "B_used": len(boot[k]), "B_failed": fails[k]})
    sets = pd.DataFrame(sets)

    # ---- per-bed estimates and contrasts
    est_rows, con_rows, het_rows = [], [], []
    for k in ORDER:
        s, F, Bt = SPECS[k], full[k], boot[k]
        b1, b2 = s["beds"][0][0], s["beds"][1][0]
        lab = dict(s["beds"])
        for mname, cols, focus in models_by[k]:
            for f_ in focus:
                drv = LABEL[f_] if f_ != "glyc" else s["glyc"]
                for b in (b1, b2):
                    key = f"{mname}|{b}|{f_}"
                    r = summ(F[key], Bt[key])
                    est_rows.append({"cohort": s["label"], "model": mname.split(":")[0], "bed": lab[b],
                                     "driver": f_, "driver_label": drv, **r,
                                     **({"latent_est": r["est"] * LATENT} if s["family"] == "binomial" else {})})
                kA, kB = f"{mname}|{b1}|{f_}", f"{mname}|{b2}|{f_}"
                r = summ(F[kA] - F[kB], Bt[kA] - Bt[kB])
                row = {"cohort": s["label"], "model": mname.split(":")[0], "contrast": f"{lab[b1]} - {lab[b2]}",
                       "driver": f_, "driver_label": drv, **r, "n": len(data[k]), "B_used": len(Bt)}
                if s["family"] == "binomial":
                    row.update({"latent_est": r["est"] * LATENT, "latent_lower": r["lower"] * LATENT,
                                "latent_upper": r["upper"] * LATENT})
                    if mname == "MA":
                        mA, mB = F[f"MA|{b1}|__mzsd"], F[f"MA|{b2}|__mzsd"]
                        BmA, BmB = Bt[f"MA|{b1}|__mzsd"], Bt[f"MA|{b2}|__mzsd"]
                        rz = summ(F[kA] / mA - F[kB] / mB, Bt[kA] / BmA - Bt[kB] / BmB)
                        row.update({"mz_est": rz["est"], "mz_lower": rz["lower"], "mz_upper": rz["upper"],
                                    "mz_p": rz["p"]})
                con_rows.append(row)
        # heterogeneity tests
        ma = s["ma"]
        A = np.array([F[f"MA|{b1}|{c}"] for c in ma])
        Bv = np.array([F[f"MA|{b2}|{c}"] for c in ma])
        BA = np.column_stack([Bt[f"MA|{b1}|{c}"] for c in ma])
        BB = np.column_stack([Bt[f"MA|{b2}|{c}"] for c in ma])
        scale = "log-OR (primary; latent x sqrt(3)/pi identical)" if s["family"] == "binomial" else "SD of bed"
        for tname, keep in (("MA: all drivers", ma), ("MA: excluding age", [c for c in ma if c != "age"]),
                            ("MA: excluding age and sex", [c for c in ma if c not in ("age", "male")])):
            if not keep:
                continue
            ix = [ma.index(c) for c in keep]
            chi2, df, p = wald(A[ix] - Bv[ix], BA[:, ix] - BB[:, ix])
            het_rows.append({"cohort": s["label"], "test": tname, "scale": scale, "chi2": chi2, "df": df, "p": p,
                             "drivers": ", ".join(keep), "n": len(data[k]), "B_used": len(Bt)})
        if s["family"] == "binomial":
            mA, mB = F[f"MA|{b1}|__mzsd"], F[f"MA|{b2}|__mzsd"]
            BmA = Bt[f"MA|{b1}|__mzsd"].to_numpy()[:, None]
            BmB = Bt[f"MA|{b2}|__mzsd"].to_numpy()[:, None]
            chi2, df, p = wald(A / mA - Bv / mB, BA / BmA - BB / BmB)
            het_rows.append({"cohort": s["label"], "test": "MA: all drivers", "scale":
                             "McKelvey-Zavoina fully standardised (sensitivity)", "chi2": chi2, "df": df, "p": p,
                             "drivers": ", ".join(ma), "n": len(data[k]), "B_used": len(Bt)})
        c_hat, chi2, df, p = gain_only(A, Bv, BA, BB)
        het_rows.append({"cohort": s["label"], "test": f"MA: gain-only null beta(bed1) = c x beta(bed2), c = "
                         f"{c_hat:.3g}", "scale": scale, "chi2": chi2, "df": df, "p": p, "drivers": ", ".join(ma),
                         "n": len(data[k]), "B_used": len(Bt)})
        mins = [c for c in ma]
        Am = np.array([F[f"MIN:{c}|{b1}|{c}"] - F[f"MIN:{c}|{b2}|{c}"] for c in mins])
        Bm = np.column_stack([Bt[f"MIN:{c}|{b1}|{c}"] - Bt[f"MIN:{c}|{b2}|{c}"] for c in mins])
        chi2, df, p = wald(Am, Bm)
        het_rows.append({"cohort": s["label"], "test": "MIN (driver + age + sex): joint over MA drivers",
                         "scale": scale, "chi2": chi2, "df": df, "p": p, "drivers": ", ".join(mins),
                         "n": len(data[k]), "B_used": len(Bt)})
        for name, cols in s["sens"]:
            A2 = np.array([F[f"{name}|{b1}|{c}"] - F[f"{name}|{b2}|{c}"] for c in cols])
            B2 = np.column_stack([Bt[f"{name}|{b1}|{c}"] - Bt[f"{name}|{b2}|{c}"] for c in cols])
            chi2, df, p = wald(A2, B2)
            het_rows.append({"cohort": s["label"], "test": f"{name}: + implied brachial SBP (coupled; sensitivity)",
                             "scale": scale, "chi2": chi2, "df": df, "p": p, "drivers": ", ".join(cols),
                             "n": len(data[k]), "B_used": len(Bt)})
    est, con, het = pd.DataFrame(est_rows), pd.DataFrame(con_rows), pd.DataFrame(het_rows)

    # ---- BP component lean
    lean_rows = []
    for k in ("keio", "japan"):
        s, F, Bt = SPECS[k], full[k], boot[k]
        b1, b2 = s["beds"][0][0], s["beds"][1][0]
        lab = dict(s["beds"])
        for mname in ("BPJ", "MA"):
            def lean(S, b, per10):
                if per10:
                    return 10 * S[f"{mname}|{b}|dbp"] / S["SD|dbp"] - 10 * S[f"{mname}|{b}|pp"] / S["SD|pp"]
                return S[f"{mname}|{b}|dbp"] - S[f"{mname}|{b}|pp"]
            for per10 in (True, False):
                unit = "per 10 mmHg" if per10 else "per SD"
                r1, r2 = summ(lean(F, b1, per10), lean(Bt, b1, per10)), summ(lean(F, b2, per10), lean(Bt, b2, per10))
                rd = summ(lean(F, b1, per10) - lean(F, b2, per10), lean(Bt, b1, per10) - lean(Bt, b2, per10))
                lean_rows.append({
                    "cohort": s["label"], "model": mname, "unit": unit,
                    "scale": "log-OR" if s["family"] == "binomial" else "SD of bed",
                    f"L_bed1": ci_str(r1["est"], r1["lower"], r1["upper"]), "L_bed2":
                    ci_str(r2["est"], r2["lower"], r2["upper"]), "bed1": lab[b1], "bed2": lab[b2],
                    "delta_L": rd["est"], "lower": rd["lower"], "upper": rd["upper"], "p": rd["p"],
                    "delta_L_latent": rd["est"] * LATENT if s["family"] == "binomial" else np.nan,
                    "latent_lower": rd["lower"] * LATENT if s["family"] == "binomial" else np.nan,
                    "latent_upper": rd["upper"] * LATENT if s["family"] == "binomial" else np.nan})
    lean = pd.DataFrame(lean_rows)

    # ---- HPP lean references
    Lp = bpf["pwv"]["iv_lean"][0]
    hpp_lean = {}
    for bed in ("imt", "abi_min"):
        Lb = bpf[bed]["iv_lean"][0]
        ob_p = bpf["pwv"]["obs_dbp"] - bpf["pwv"]["obs_pp"]
        ob_b = bpf[bed]["obs_dbp"] - bpf[bed]["obs_pp"]
        hpp_lean[bed] = {"iv": Lp - Lb, "obs": ob_p - ob_b, "ptr": f"BPF:{bpf['pwv']['line']},{bpf[bed]['line']}",
                         "iv_pwv": bpf["pwv"]["iv_lean"], "iv_bed": bpf[bed]["iv_lean"]}

    # ---- concordance vs HPP
    conc = []
    for k in ORDER:
        s = SPECS[k]
        csub = con[con["cohort"] == s["label"]].set_index(["model", "driver"])
        for ext_m, hpp_m in (("MIN", "M1"), ("M3", "M3")):
            if ext_m == "M3" and not s["m3"]:
                continue
            for c in s["ma"] + s["extra"]:
                r = csub.loc[(ext_m, c)]
                hk, note = hpp_key(s, c)
                row = {"cohort": s["label"], "pair": s["hpp_pair"], "same_device": s["same_device"],
                       "pairing": f"{ext_m} vs ATL {hpp_m}", "driver": c,
                       "driver_label": LABEL[c] if c != "glyc" else s["glyc"], "ext_delta": r["est"],
                       "ext_lower": r["lower"], "ext_upper": r["upper"], "ext_p": r["p"],
                       "ext_ci_excl0": bool(r["lower"] > 0 or r["upper"] < 0),
                       "ext_latent": r["est"] * LATENT if s["family"] == "binomial" else np.nan,
                       "hpp_row": hk or "", "note": note}
                if hk is None or (hpp_m, hk) not in atl:
                    row.update({"hpp_ptr": "", "hpp_delta": np.nan, "hpp_lower": np.nan, "hpp_upper": np.nan,
                                "sign_agree": None, "note": note or "no HPP row"})
                else:
                    a = atl[(hpp_m, hk)]
                    if s["hpp_pair"] == "pwv-imt":
                        hd, hl, hh = a["pwv_imt"], a["pwv_imt_lo"], a["pwv_imt_hi"]
                    else:
                        hd, hl, hh = a["pwv"] - a["abi"], np.nan, np.nan
                    row.update({"hpp_ptr": f"ATL:{a['line']}", "hpp_delta": hd, "hpp_lower": hl, "hpp_upper": hh,
                                "sign_agree": bool(np.sign(hd) == np.sign(r["est"]))})
                conc.append(row)
        if s["bpj"]:
            bed = "imt" if s["hpp_pair"] == "pwv-imt" else "abi_min"
            for mname in ("BPJ", "MA"):
                lr = lean[(lean["cohort"] == s["label"]) & (lean["model"] == mname) &
                          (lean["unit"] == "per 10 mmHg")].iloc[0]
                for ref in ("iv", "obs"):
                    hd = hpp_lean[bed][ref]
                    conc.append({"cohort": s["label"], "pair": s["hpp_pair"], "same_device": s["same_device"],
                                 "pairing": f"{mname} lean vs BPF {'IV 2SLS' if ref == 'iv' else 'obs_rc'}",
                                 "driver": "dbp_minus_pp", "driver_label": "DBP - PP lean (per 10 mmHg)",
                                 "ext_delta": lr["delta_L"], "ext_lower": lr["lower"], "ext_upper": lr["upper"],
                                 "ext_p": lr["p"], "ext_ci_excl0": bool(lr["lower"] > 0 or lr["upper"] < 0),
                                 "ext_latent": lr["delta_L_latent"], "hpp_row": f"joint DBP-PP pwv - {bed}",
                                 "note": "HPP value = difference of per-bed values (no CI printed)",
                                 "hpp_ptr": hpp_lean[bed]["ptr"], "hpp_delta": hd, "hpp_lower": np.nan,
                                 "hpp_upper": np.nan, "sign_agree": bool(np.sign(hd) == np.sign(lr["delta_L"]))})
    conc = pd.DataFrame(conc)

    # per-bed direction concordance (MIN vs ATL M1)
    pb = []
    hmap_bed = {"pwv": "pwv", "imt": "imt", "abi": "abi"}
    for k in ORDER:
        s = SPECS[k]
        e = est[(est["cohort"] == s["label"]) & (est["model"] == "MIN")]
        lab = dict(s["beds"])
        inv = {v: kk for kk, v in lab.items()}
        for _, r in e.iterrows():
            hk, note = hpp_key(s, r["driver"])
            bk = inv[r["bed"]]
            row = {"cohort": s["label"], "bed": r["bed"], "hpp_bed": hmap_bed[bk], "driver": r["driver"],
                   "driver_label": r["driver_label"], "ext_beta": r["est"], "ext_lower": r["lower"],
                   "ext_upper": r["upper"], "hpp_row": hk or "", "note": note}
            if hk is None or ("M1", hk) not in atl:
                row.update({"hpp_ptr": "", "hpp_beta": np.nan, "sign_agree": None})
            else:
                a = atl[("M1", hk)]
                row.update({"hpp_ptr": f"ATL:{a['line']}", "hpp_beta": a[hmap_bed[bk]],
                            "sign_agree": bool(np.sign(a[hmap_bed[bk]]) == np.sign(r["est"]))})
            pb.append(row)
    perbed = pd.DataFrame(pb)

    # sign tests
    def stest(df, label):
        d_ = df[df["sign_agree"].notna()]
        n_, k_ = len(d_), int(d_["sign_agree"].astype(bool).sum())
        p_ = stats.binomtest(k_, n_, 0.5).pvalue if n_ else np.nan
        return {"row set": label, "rows": n_, "agree": k_, "fraction": k_ / n_ if n_ else np.nan,
                "binomial p (two-sided)": p_}
    noproxy = conc["note"].fillna("").str.contains("proxy") == False  # noqa: E712
    M1rows = conc[(conc["pairing"] == "MIN vs ATL M1") & noproxy]
    M3rows = conc[(conc["pairing"] == "M3 vs ATL M3") & noproxy]
    leanrows = conc[conc["driver"] == "dbp_minus_pp"]
    signs = pd.DataFrame([
        stest(M1rows, "PRIMARY: MIN vs ATL M1, all cohorts (proxy rows excluded)"),
        stest(M1rows[M1rows["ext_ci_excl0"]], "MIN vs ATL M1, external CI excludes 0"),
        stest(M1rows[M1rows["pair"] == "pwv-imt"], "MIN vs ATL M1, Keio only (cross-modality PWV - IMT)"),
        stest(M1rows[M1rows["pair"] == "pwv-abi"], "MIN vs ATL M1, baPWV - ABI cohorts (same device)"),
        stest(M3rows, "M3 vs ATL M3 (Keio, Japan; proxy rows excluded)"),
        stest(M3rows[M3rows["ext_ci_excl0"]], "M3 vs ATL M3, external CI excludes 0"),
        stest(leanrows, "DBP - PP lean rows vs BPF (2 cohorts x BPJ/MA x IV/obs_rc; 2 independent comparisons)"),
        stest(perbed[perbed["note"].fillna("").str.contains("proxy") == False],  # noqa: E712
              "per-bed directions, MIN vs ATL M1 betas (proxy rows excluded)")])

    # ---- key replication rows
    def crow(k, model, drv):
        r = con[(con["cohort"] == SPECS[k]["label"]) & (con["model"] == model) & (con["driver"] == drv)]
        return None if r.empty else r.iloc[0]
    key_rows = []

    def add_key(target, k, model, drv, hpp_txt, hpp_ptr, hpp_val):
        r = crow(k, model, drv)
        if r is None:
            return
        s = SPECS[k]
        ext = ci_str(r["est"], r["lower"], r["upper"])
        if s["family"] == "binomial":
            ext += f" (latent {ci_str(r['est'] * LATENT, r['lower'] * LATENT, r['upper'] * LATENT)})"
        key_rows.append({"target": target, "cohort": s["label"], "pair": s["hpp_pair"] +
                         (" (same device)" if s["same_device"] else ""), "model": model,
                         "driver": LABEL[drv] if drv != "glyc" else s["glyc"], "external delta [CI]": ext,
                         "p": r["p"], "HPP row": hpp_txt, "HPP pointer": hpp_ptr,
                         "sign agree": ("yes" if np.sign(hpp_val) == np.sign(r["est"]) else "no")
                         if hpp_val is not None and np.isfinite(hpp_val) else "n/a"})

    def hpp_txt(model, ex, pair):
        a = atl.get((model, ex))
        if a is None:
            return "no row", "", None
        if pair == "pwv-imt":
            return (f"{model} {ex} pwv-imt {ci_str(a['pwv_imt'], a['pwv_imt_lo'], a['pwv_imt_hi'])}",
                    f"ATL:{a['line']}", a["pwv_imt"])
        return (f"{model} {ex} pwv {a['pwv']:.3g} - abi {a['abi']:.3g} = {a['pwv'] - a['abi']:.3g}",
                f"ATL:{a['line']}", a["pwv"] - a["abi"])
    # (a) lean
    for k in ("keio", "japan"):
        s = SPECS[k]
        bed = "imt" if s["hpp_pair"] == "pwv-imt" else "abi_min"
        h = hpp_lean[bed]
        for mname in ("BPJ", "MA"):
            lr = lean[(lean["cohort"] == s["label"]) & (lean["model"] == mname) & (lean["unit"] == "per 10 mmHg")]
            lr = lr.iloc[0]
            ext = ci_str(lr["delta_L"], lr["lower"], lr["upper"])
            if s["family"] == "binomial":
                ext += f" (latent {ci_str(lr['delta_L_latent'], lr['latent_lower'], lr['latent_upper'])})"
            key_rows.append({"target": "(a) DBP - PP lean", "cohort": s["label"], "pair": s["hpp_pair"] +
                             (" (same device)" if s["same_device"] else ""), "model": mname,
                             "driver": "DBP - PP per 10 mmHg", "external delta [CI]": ext, "p": lr["p"],
                             "HPP row": f"BPF joint DBP-PP: pwv {ci_str(*h['iv_pwv'])} - {bed} "
                             f"{ci_str(*h['iv_bed'])} = IV {h['iv']:.3g}; obs_rc {h['obs']:.3g}",
                             "HPP pointer": h["ptr"],
                             "sign agree": "yes" if np.sign(h["iv"]) == np.sign(lr["delta_L"]) else "no"})
        for m_ext, m_h in (("M3", "M3"),):
            t, p_, v = hpp_txt(m_h, "dbp_sit", s["hpp_pair"])
            add_key("(a) DBP given SBP (M3)", k, m_ext, "dbp", t, p_, v)
    # (b) adiposity
    for k in ORDER:
        s = SPECS[k]
        for drv, ex in (("bmi", "bmi"), ("waist", "waist"), ("log_vat", "log_vat")):
            if drv not in s["ma"] + s["extra"]:
                continue
            for m_ext, m_h in (("MIN", "M1"), ("M3", "M3"), ("MA", "M3")):
                if m_ext == "M3" and not s["m3"]:
                    continue
                if m_ext == "MA" and drv not in s["ma"]:
                    continue
                t, p_, v = hpp_txt(m_h, ex, s["hpp_pair"])
                if m_ext == "MA":
                    t = "(closest) " + t
                add_key("(b) adiposity", k, m_ext, drv, t, p_, v)
    a3 = atl.get(("M3", "fat_total"))
    key_rows.append({"target": "(b) adiposity", "cohort": "-", "pair": "pwv-imt", "model": "-",
                     "driver": "total fat (DXA)", "external delta [CI]": "no external DXA fat", "p": np.nan,
                     "HPP row": f"M3 fat_total pwv {a3['pwv']:.3g} imt {a3['imt']:.3g}; pwv-imt "
                     f"{ci_str(a3['pwv_imt'], a3['pwv_imt_lo'], a3['pwv_imt_hi'])}" if a3 else "no row",
                     "HPP pointer": f"ATL:{a3['line']}" if a3 else "", "sign agree": "n/a"})
    # (c) heart rate
    hr1 = atl.get(("M1", "hr_sit"))
    key_rows.append({"target": "(c) heart rate", "cohort": "all four", "pair": "-", "model": "-",
                     "driver": "heart rate", "external delta [CI]": "NOT TESTABLE: no heart-rate column in Keio, "
                     "Japan, China or IMPACT", "p": np.nan,
                     "HPP row": f"EXP M2 ecg.hr_bpm pwv-imt {ci_str(*exp['hr']['pwv_imt'])}; ATL M1 hr_sit pwv-imt "
                     f"{ci_str(hr1['pwv_imt'], hr1['pwv_imt_lo'], hr1['pwv_imt_hi'])}",
                     "HPP pointer": f"EXP:{exp['hr']['line']}; ATL:{hr1['line']}", "sign agree": "n/a"})
    # (d) age, sex, diabetes/glycaemia
    for k in ORDER:
        s = SPECS[k]
        for drv in ("age", "male", "glyc", "diabetes"):
            if drv not in s["ma"] + s["extra"]:
                continue
            for m_ext, m_h in (("MIN", "M1"), ("MA", None)):
                hk, note = hpp_key(s, drv)
                if hk is None:
                    t, p_, v = f"no comparable HPP row ({note})", "", None
                elif m_h is None:
                    t, p_, v = hpp_txt("M1", hk, s["hpp_pair"])
                    t = "(M1 shown) " + t
                else:
                    t, p_, v = hpp_txt(m_h, hk, s["hpp_pair"])
                if note and hk is not None:
                    t += f" [{note}]"
                add_key("(d) age / sex / glycaemia", k, m_ext, drv, t, p_, v)
    keyt = pd.DataFrame(key_rows)

    # ---- report
    rep.h("Pre-specified analysis")
    rep.p("```text\nPre-specified analysis" + __doc__.split("Pre-specified analysis", 1)[1].rstrip() + "\n```")
    rep.h("Analysis sets")
    rep.table("analysis_sets", sets, "Cohorts, bed pairs and drivers", max_rows=None)
    rep.h("Joint heterogeneity tests (bootstrap covariance of the between-bed difference vector)")
    rep.table("heterogeneity", het, "Wald tests of beta(bed 1) = beta(bed 2)", max_rows=None)
    rep.h("Summary bullets (auto-generated from the tables in this report)")
    bl = []
    for k in ORDER:
        s = SPECS[k]
        h = het[het["cohort"] == s["label"]]
        r0 = h[(h["test"] == "MA: all drivers") & ~h["scale"].str.startswith("McKelvey")].iloc[0]
        r1 = h[h["test"] == "MA: excluding age"].iloc[0]
        rg = h[h["test"].str.startswith("MA: gain-only")].iloc[0]
        bl.append(f"{s['label']} ({'same-device' if s['same_device'] else 'cross-modality'} "
                  f"{s['beds'][0][1]} vs {s['beds'][1][1]}; n {len(data[k])}): MA joint heterogeneity chi2 "
                  f"{r0['chi2']:.4g}, df {r0['df']}, p {r0['p']:.3g}; excluding age chi2 {r1['chi2']:.4g}, df "
                  f"{r1['df']}, p {r1['p']:.3g}; gain-only null ({rg['test'].split(', ')[-1]}) chi2 "
                  f"{rg['chi2']:.4g}, df {rg['df']}, p {rg['p']:.3g}.")
    for k in ("keio", "japan"):
        s = SPECS[k]
        bed = "imt" if s["hpp_pair"] == "pwv-imt" else "abi_min"
        for mname in ("BPJ", "MA"):
            lr = lean[(lean["cohort"] == s["label"]) & (lean["model"] == mname) &
                      (lean["unit"] == "per 10 mmHg")].iloc[0]
            bl.append(f"{s['label']} DBP - PP lean, {mname}, per 10 mmHg ({lr['scale']}): delta L "
                      f"{ci_str(lr['delta_L'], lr['lower'], lr['upper'])}, p {lr['p']:.3g} (L {lr['bed1']} "
                      f"{lr['L_bed1']}; L {lr['bed2']} {lr['L_bed2']}); HPP pwv - {bed} IV "
                      f"{hpp_lean[bed]['iv']:.3g}, obs_rc {hpp_lean[bed]['obs']:.3g} ({hpp_lean[bed]['ptr']}).")
    for _, r in signs.iterrows():
        bl.append(f"Sign test, {r['row set']}: {r['agree']}/{r['rows']} agree, p {r['binomial p (two-sided)']:.3g}.")
    rep.bullets(bl)
    rep.h("Key replication rows vs HPP (targets a-d)")
    rep.p("External delta = beta(baPWV) - beta(second bed), adverse-oriented; Keio in log-OR per SD (latent = log-OR x "
          "sqrt(3)/pi); other cohorts in SD of bed per SD of driver; lean rows per 10 mmHg. HPP PWV is femoral-ankle "
          "PWV. Sign agreement compares the external point estimate with the HPP value.")
    rep.table("key_rows", keyt, "Key rows", max_rows=None)
    rep.h("BP component lean (DBP - PP)")
    rep.table("bp_lean", lean, "L(bed) = beta(DBP) - beta(PP); delta L = L(baPWV) - L(second bed)", max_rows=None)
    rep.h("Sign concordance with HPP")
    rep.table("sign_tests", signs, "Exact binomial sign tests (rows not independent)", max_rows=None)
    show = ["cohort", "pair", "pairing", "driver_label", "ext_delta", "ext_lower", "ext_upper", "ext_p",
            "hpp_ptr", "hpp_delta", "hpp_lower", "hpp_upper", "sign_agree", "note"]
    rep.tables["concordance_hpp"] = conc
    rep.p("")
    rep.p("**Driver x bed-pair concordance** (`concordance_hpp.csv`; more columns there)")
    rep.p("")
    rep.p(C.md_table(conc[show], 4, None))
    rep.p("")
    rep.table("concordance_per_bed", perbed, "Per-bed directions, MIN vs ATL M1 betas", max_rows=None)
    rep.h("Bed contrasts, mutually adjusted model (MA)")
    cshow = ["cohort", "contrast", "driver_label", "est", "lower", "upper", "p", "latent_est", "mz_est", "mz_lower",
             "mz_upper", "mz_p"]
    rep.p(C.md_table(con[con["model"] == "MA"][[c for c in cshow if c in con.columns]], 4, None))
    rep.p("")
    rep.p("_(All models in `bed_contrasts.csv`; per-bed estimates in `bed_estimates.csv`.)_")
    rep.tables["bed_contrasts"] = con
    rep.tables["bed_estimates"] = est
    rep.h("Caveats")
    rep.bullets([
        "Exploratory. A CI that includes 0 bounds the between-bed difference; it does not show equality.",
        "baPWV (brachial-ankle, includes the aortic segment) is not HPP femoral-ankle PWV; magnitudes are not "
        "comparable, so concordance is judged by sign. The baPWV path includes the aorta, the HPP path only the leg; "
        "this may matter most for age (the Keio age row disagrees in sign with HPP PWV - IMT); not tested here.",
        "Keio binary outcomes: a driver that shifts a common-prevalence outcome (baPWV > 1400, 49 %) and a rarer one "
        "(IMT+, 25 %) need not have equal log-ORs even with equal latent effects; the gain-only and "
        "McKelvey-Zavoina rows address a common scale difference only.",
        "baPWV + ABI (Japan, China, IMPACT) come from one four-limb oscillometric acquisition: one modality, shared "
        "ankle waveforms and pressures. Only Keio (baPWV vs carotid ultrasound) is a cross-modality pair. HPP PWV and "
        "ABI also share a device.",
        "Keio beds are binary (thresholds/definitions undocumented); log-OR differences are the primary inference; "
        "latent and McKelvey-Zavoina scales are for magnitude only. HPP comparison for Keio is sign-only.",
        "Japan ABI max is the higher leg (insensitive to unilateral PAD). China and IMPACT use ABI min >= 0.9.",
        "Inferred codings: China sex (1 = male), IMPACT abbreviations and smoking (current vs ever), Keio smoker "
        "(1 - non-smoker), Japan sex. IMPACT eGFR recomputed from creatinine (Japanese equation).",
        "China 36k drivers are age and sex only; the implied brachial SBP is ABI's denominator (coupled) and is a "
        "sensitivity, never a concordance row. IMPACT has no BP, heart rate or lipid values; patients, not a "
        "population sample.",
        "No open cohort has heart rate: the HR positive control (EXP) cannot be replicated here.",
        "HPP baPWV - ABI comparators are differences of printed per-bed betas (no CI); HPP M3 also adjusts for "
        "antihypertensive drugs (not available externally). Sign tests treat correlated rows as independent.",
        "No liver phenotypes were used (Keio MAFLD/NAFLD, Japan fatty liver, GGT, ALT, AST excluded)."])
    planted = None
    if mode == "simulate":
        ch = []
        ck = con[(con["cohort"] == "Keio 890") & (con["model"] == "MA")].set_index("driver")
        for drv in ("dbp", "pp", "age"):
            ch.append((f"Keio MA delta log-OR/SD {drv} (approx.)", SIM_KEIO["pwv"][drv] - SIM_KEIO["imt"][drv],
                       ck.loc[drv, "est"], ck.loc[drv, "boot_se"]))
        dj = data["japan"]
        sdp = dj["pwv"].std(ddof=1)
        sda = dj["abi"].std(ddof=1)
        cj = con[(con["cohort"] == "Japan 912") & (con["model"] == "MA")].set_index("driver")
        for drv in ("dbp", "pp"):
            tru = 150 * SIM_JP["pwv"][drv] / sdp - 0.06 * SIM_JP["abi"][drv] / sda
            ch.append((f"Japan MA delta std beta {drv} (approx.)", tru, cj.loc[drv, "est"], cj.loc[drv, "boot_se"]))
        dc = data["china"]
        cc = con[(con["cohort"] == "China 36k") & (con["model"] == "MA")].set_index("driver")
        tru = SIM_CN["pwv"] * 100 / dc["pwv"].std(ddof=1) + SIM_CN["abi"] / dc["abi"].std(ddof=1)
        ch.append(("China MA delta std beta age (approx.)", tru, cc.loc["age", "est"], cc.loc["age", "boot_se"]))
        planted = C.planted_rows(ch)
    C.finish(rep, outdir, __file__, planted)


if __name__ == "__main__":
    main()
