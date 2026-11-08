#!/usr/bin/env python3
"""F1 Stage 0, task m (PLAN 9.3d L12): robustness of the NHANES retina-mortality findings (L4) and of the ABI-vs-AVN
heart/cerebrovascular death profile contrast.

GATED: refuses to run on real data without --real. --simulate runs a planted-answer check of the new estimators (Fine-Gray by weighted
counting-process Cox, Aalen-Johansen differences) and the wiring on simulated tables (no real file opened).
Label: PUBLIC-DATA SUPPORTING EVIDENCE for F1. Exploratory. No HPP data. No liver phenotypes.

Pre-specified analysis
----------------------
Data: exactly those of L4 (analysis_m_nhanes_retina_mortality: NHANES 2005-2008 OPXRET + covariates + 2019 LMF; load,
derivation, retina sample, domain D, design and M1/M2 imported unchanged) and of task f (analysis_f_nhanes9904_abi:
NHANES 1999-2004 ABI clean table; prepare/design imported unchanged; heart/cerebrovascular deaths from UCOD_LEADING as
in L2). Design-based (Taylor) variance, t(df) intervals; df = #PSU - #strata of the strata that hold domain records
(31 for 2005-2008, 44 for 1999-2004; per-cycle subsets use that cycle's strata only).
Heart vs cerebrovascular contrast: two cause-specific Cox models (UCOD 1 heart; UCOD 5 cerebrovascular; other deaths
censored) in the same domain; joint design-based variance from the stacked dfbeta influence values (identical to the
Lunn-McNeil stacked model of L4/L2; reproduction checked against their CSVs).
A1 Competing risks (domain D, AVN and any retinopathy each vs absent):
  (a) survey-weighted Aalen-Johansen cumulative incidence of heart and cerebrovascular death (all other deaths
      competing) at 5 and 10 years by exposure; differences and risk ratios with design-based SE from the influence
      functions (exposed minus unexposed influence, one design); ratio of risk ratios heart/cerebrovascular (log scale,
      delta method). Crude (no covariate adjustment).
  (b) Fine-Gray subdistribution hazard ratios (M1, M2 of L4; and exposure-only M0 for QC) by a weighted
      counting-process Cox model: people with a competing death stay in the risk set with weight G(t-)/G(T-), G =
      survey-weighted Kaplan-Meier of the censoring distribution estimated within cycle (administrative censoring
      depends on exam date); rows split at the censoring jump times (as R survival::finegray). Design-based variance
      from the influence values aggregated per person (variability of G ignored, as in finegray + robust variance).
      sHR ratio heart/cerebrovascular from the joint influence of the two fits. QC: M0 sHR vs the complementary
      log-log contrast of the 10-year cumulative incidences.
A2 Harmonised cross-sample contrast. Ratio of ratios = [HR heart / HR cerebrovascular, ABI < 0.9 vs 1.00-1.40,
   NHANES 1999-2004, 3-indicator ABI class model] / [same for AVN, NHANES 2005-2008]; independent samples, var =
   se_A^2 + se_B^2, t(min df). Covariate set H (both samples): age, sex, race/ethnicity, seated SBP, BP-lowering
   prescription, diabetes, smoking, total and HDL cholesterol, BMI, eGFR (L4 M2 without HbA1c; task f M2 without
   prevalent CVD); H + prevalent CVD as a variant. Domain: complete H + prevalent CVD in each sample. Follow-up
   horizons: untruncated; 15 y (the 2005-2008 maximum); 13 y; 10 y (every survivor in both samples followed >= 10.8 y).
   Variant: age 40-79 in both (age top-coded at 85 in 1999-2006 and at 80 in 2007-2008). Original (as-published)
   specifications reproduced first. Secondary pairs: ABI 0.90-0.99 vs AVN (same class model); ABI < 0.9 vs any
   retinopathy.
A3 AVN robustness (L4 M2 unless stated): deaths in the first 2 years excluded (follow-up restarted at 2 y); age 40-79;
   age as 5 bands (40-49 ref, 50-59, 60-69, 70-79, >= 80 incl. top-coded) instead of linear age; M2 + prevalent stroke
   (MCQ160F), CHD (MCQ160C/D/E) and heart failure (MCQ160B) as separate terms; prevalent stroke excluded; any
   prevalent CVD excluded; AVN graded in both eyes (OPDDAVN and OPDSAVN present) vs one eye only; 2005-2006 only;
   2007-2008 only (cycle heterogeneity: difference of the two cycle-specific log estimates, independent strata,
   t(min df)); all restrictions combined (age 40-79, no prevalent CVD, first 2 years excluded). Per row: AVN -> heart
   and cerebrovascular death HR and their ratio.
A4 Multiplicity over the L4 p-values (CSV inputs): family A = M2 heart/cerebrovascular contrasts of the main sign
   terms; family B = A + all M2 main sign x outcome HRs; family C = every fitted L4 p-value (all models, sensitivity
   codings; worst case). Benjamini-Hochberg q and Bonferroni within family.
Rows with < 3 exposed events in a cause are not fitted (n.f.); < 20 flagged (*).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")     # <= 4 workers (session rule)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_survival as ls  # noqa: E402
import s0an_common as C  # noqa: E402
import analysis_m_nhanes_retina_mortality as M  # noqa: E402  (L4: load, derivation, domains, design, M1/M2)
import analysis_f_nhanes9904_abi as F  # noqa: E402  (task f: prepare, design, M1/M2)
import analysis_j_nhanes_acq as J  # noqa: E402  (input registry of raw NHANES files)

TASK = "m_retina_robustness"
L4_DIR = C.ANALYSIS / "m_nhanes_retina_mortality"
L2_DIR = C.ANALYSIS / "m_outcome_contrasts"
XB_DIR = C.ANALYSIS / "m_crossbed_profile_nhanes"
MIN_EV, FLAG_EV = 3, 20
HORIZONS = [None, 15.0, 13.0, 10.0]
CIF_TIMES = [5.0, 10.0]
H_RET = M.M1 + ["sbp10", "bp_treat", "diabetes_any", "smk_former", "smk_current", "tc10", "hdl10", "bmi5", "egfr10"]
H_ABI = F.M1 + ["sbp10", "rx_any_antihypertensive", "diabetes_any", "smk_former", "smk_current", "tc10", "hdl10",
                "bmi5", "egfr10"]
ABI_EXPO = ["abi_lt09", "abi_0909", "abi_gt14"]
AGE_BANDS = ["age_50_59", "age_60_69", "age_70_79", "age_80p"]
HX = ["hx_stroke", "hx_chd", "hx_chf"]
SIGNS = [("avn", "AVN (definite)"), ("ret_any", "Any retinopathy")]
FG_PLANT = {"beta": 0.7, "p": 0.15}

HARMONISATION = [
    ("Age", "RIDAGEYR, top-coded at 85", "RIDAGEYR, top-coded at 85 (2005-2006) and 80 (2007-2008)",
     "Differs at the top: variant restricted to age 40-79 in both"),
    ("Sex; race/ethnicity", "RIAGENDR; RIDRETH1 (ref NH white)", "Same", "Same"),
    ("Seated SBP", "Mean of available BPXSY1-4 (auscultatory)", "Same", "Same"),
    ("BP-lowering prescription", "RXQ_RX (RXD030 1999-2002, RXDUSE 2003-2004) x Multum classes 42, 43, 44, 47, 48, 49,"
     " 53, 55, 56, 340, 342; 0 if no prescription medicine", "RXQ_RX (RXDUSE) x the same Multum classes",
     "Same codes; question name differs only"),
    ("Diabetes", "DIQ010 = 1, insulin, pills, HbA1c >= 6.5 % or FPG >= 126 (FPG in fasting subsample)", "Same rule",
     "Same rule; HbA1c and FPG laboratory methods change across cycles (not recalibrated in either driver)"),
    ("Smoking", "SMQ020 / SMQ040 never, former, current", "Same", "Same"),
    ("Total cholesterol", "LBXTC", "LBXTC", "Same variable"),
    ("HDL cholesterol", "LBDHDL (1999-2002), LBXHDD (2003-2004)", "LBDHDD (2005-2008)",
     "Assay changed between 1999-2002 and 2003 onward; not recalibrated"),
    ("BMI", "BMXBMI", "BMXBMI", "Same"),
    ("eGFR", "CKD-EPI 2021; creatinine 1999-2000 recalibrated 1.013 x + 0.147, 2001-2004 as is",
     "CKD-EPI 2021; creatinine 2005-2006 recalibrated -0.016 + 0.978 x, 2007-2008 as is",
     "Cycle-specific calibration equations differ (each per NCHS guidance)"),
    ("Prevalent CVD (variant only)", "MCQ160B-F any yes", "MCQ160B-F any yes", "Same"),
    ("Not in H", "task f M2 has prevalent CVD", "L4 M2 has HbA1c", "Dropped to harmonise (prevalent CVD = variant)"),
    ("Weights", "wtmec6yr (6-year MEC)", "WTMEC2YR / 2 (4-year MEC)", "Cycle pooling rule of each sample"),
    ("Outcome", "UCOD_LEADING 1 vs 5, 2019 LMF, time from MEC exam", "Same", "Same; follow-up length differs "
     "(1999-2004 up to 20.75 y; 2005-2008 up to 15 y) -> horizons"),
]


# =================================================================================================================
# Small helpers
# =================================================================================================================
def fmt_ci(e, lo, hi, nd=3):
    if not np.all(np.isfinite([e, lo, hi])):
        return "n.f."
    return f"{e:.{nd}g} [{lo:.{nd}g}, {hi:.{nd}g}]"


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return ""
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def dom_df(des: ls.SurveyDesign, m) -> int:
    """Design df of the strata that hold domain records: #PSU - #strata over those strata."""
    m = np.asarray(m, bool)
    if not m.any():
        return 0
    st = np.unique(des.psu_stratum[np.unique(des.psu_code[m])])
    return int(np.isin(des.psu_stratum, st).sum() - len(st))


def tci(est, se, df):
    q = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(est / se), df) if se > 0 else np.nan
    return est - q * se, est + q * se, float(p)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


# =================================================================================================================
# Cause-specific heart vs cerebrovascular pair (joint design-based variance)
# =================================================================================================================
def cs_pair(d, des, dom, expo, covs, ev_h="death_heart", ev_c="death_cereb", tcol="t", horizon=None, label=""):
    cols = list(expo) + list(covs)
    m = np.asarray(dom, bool) & d[cols + [tcol, ev_h, ev_c]].notna().all(axis=1).values
    t = d[tcol].fillna(1).values.astype(float)
    eh = d[ev_h].fillna(0).values.astype(float)
    ec = d[ev_c].fillna(0).values.astype(float)
    if horizon is not None:
        eh = eh * (t <= horizon)
        ec = ec * (t <= horizon)
        t = np.minimum(t, horizon)
    X = d[cols].fillna(0).values.astype(float)
    df = dom_df(des, m)
    k = len(expo)
    rows = []
    base = {"spec": label, "n": int(m.sum()), "heart_events": int(eh[m].sum()), "cereb_events": int(ec[m].sum()),
            "design_df": df, "horizon_y": horizon if horizon is not None else np.nan}
    exp_ev = {c: (int(eh[m & (d[c].values == 1)].sum()), int(ec[m & (d[c].values == 1)].sum())) for c in expo}
    if base["heart_events"] < MIN_EV or base["cereb_events"] < MIN_EV or max(min(v) for v in exp_ev.values()) < MIN_EV:
        return [{**base, "term": c, "n_exposed": int((m & (d[c].values == 1)).sum()), "heart_events_exposed":
                 exp_ev[c][0], "cereb_events_exposed": exp_ev[c][1], "note": "not fitted"} for c in expo]
    dd = des.domain(m)
    fh = ls.coxph(t, eh, X, design=dd, names=cols)
    fc = ls.coxph(t, ec, X, design=dd, names=cols)
    infl = np.column_stack([fh.dfbeta()[:, :k], fc.dfbeta()[:, :k]])
    V = ls.design_vcov(infl, dd)
    for j, c in enumerate(expo):
        hev, cev = exp_ev[c]
        r = {**base, "term": c, "n_exposed": int((m & (d[c].values == 1)).sum()), "heart_events_exposed": hev,
             "cereb_events_exposed": cev, "flag_lt20": min(hev, cev) < FLAG_EV,
             "note": "; ".join(sorted(set(fh.warnings + fc.warnings)))}
        if min(hev, cev) < MIN_EV:
            r["note"] = (r["note"] + "; " if r["note"] else "") + f"term not reported (< {MIN_EV} exposed events)"
            rows.append(r)
            continue
        for lab, b, v in (("heart", fh.coef[j], V[j, j]), ("cereb", fc.coef[j], V[k + j, k + j])):
            se = float(np.sqrt(v))
            lo, hi, p = tci(b, se, df)
            r.update({f"log_hr_{lab}": float(b), f"se_{lab}": se, f"hr_{lab}": float(np.exp(b)),
                      f"hr_{lab}_lower": float(np.exp(lo)), f"hr_{lab}_upper": float(np.exp(hi)), f"p_{lab}": p})
        lr = float(fh.coef[j] - fc.coef[j])
        se = float(np.sqrt(V[j, j] + V[k + j, k + j] - 2 * V[j, k + j]))
        lo, hi, p = tci(lr, se, df)
        r.update({"log_ratio": lr, "se_ratio": se, "ratio": float(np.exp(lr)), "ratio_lower": float(np.exp(lo)),
                  "ratio_upper": float(np.exp(hi)), "p_ratio": p})
        rows.append(r)
    return rows


def ror(a: dict, b: dict) -> dict:
    """Ratio of ratios of two independent samples (A / B)."""
    if not (np.isfinite(a.get("log_ratio", np.nan)) and np.isfinite(b.get("log_ratio", np.nan))):
        return {"ror": np.nan, "ror_lower": np.nan, "ror_upper": np.nan, "p_ror": np.nan, "df_ror": np.nan}
    dlr = a["log_ratio"] - b["log_ratio"]
    se = float(np.sqrt(a["se_ratio"] ** 2 + b["se_ratio"] ** 2))
    df = int(min(a["design_df"], b["design_df"]))
    lo, hi, p = tci(dlr, se, df)
    return {"log_ror": dlr, "se_log_ror": se, "ror": float(np.exp(dlr)), "ror_lower": float(np.exp(lo)),
            "ror_upper": float(np.exp(hi)), "p_ror": p, "df_ror": df}


# =================================================================================================================
# A1: Aalen-Johansen cumulative incidence by exposure; Fine-Gray
# =================================================================================================================
def cif_by_exposure(d, des, dom, expo, times):
    t = d["t"].fillna(1).values.astype(float)
    st = d["cr_status"].fillna(0).values.astype(float)
    df = dom_df(des, dom)
    est, inf, rows = {}, {}, []
    for cause, clab in ((1, "heart"), (2, "cereb")):
        for g in (0, 1):
            gm = np.asarray(dom, bool) & (d[expo].values == g)
            r, IF = ls.cuminc(t, st, times, cause=cause, design=des.domain(gm), return_influence=True)
            est[(clab, g)], inf[(clab, g)] = r["cuminc"].values, IF
            for j, tau in enumerate(times):
                rows.append({"sign": expo, "cause": clab, "exposed": g, "time_y": tau, "n": int(gm.sum()),
                             "events_total": int(((st == cause) & gm).sum()),
                             "events_by_time": int(((st == cause) & gm & (t <= tau)).sum()),
                             "cuminc": float(r["cuminc"].iloc[j]), "se": float(r["se"].iloc[j])})
    per = pd.DataFrame(rows)
    out = []
    for j, tau in enumerate(times):
        row = {"sign": expo, "time_y": tau, "design_df": df}
        lrr_if = {}
        for clab in ("heart", "cereb"):
            f1, f0 = est[(clab, 1)][j], est[(clab, 0)][j]
            dif = inf[(clab, 1)][:, j] - inf[(clab, 0)][:, j]
            se = float(np.sqrt(ls.design_vcov(dif, des)))
            lo, hi, p = tci(f1 - f0, se, df)
            lrr_if[clab] = inf[(clab, 1)][:, j] / f1 - inf[(clab, 0)][:, j] / f0
            se_l = float(np.sqrt(ls.design_vcov(lrr_if[clab], des)))
            llo, lhi, lp = tci(np.log(f1 / f0), se_l, df)
            row.update({f"cif_{clab}_exposed": f1, f"cif_{clab}_unexposed": f0, f"diff_{clab}": f1 - f0,
                        f"diff_{clab}_lower": lo, f"diff_{clab}_upper": hi, f"p_diff_{clab}": p,
                        f"rr_{clab}": f1 / f0, f"rr_{clab}_lower": np.exp(llo), f"rr_{clab}_upper": np.exp(lhi),
                        f"p_rr_{clab}": lp})
        lr = np.log(row["rr_heart"]) - np.log(row["rr_cereb"])
        se = float(np.sqrt(ls.design_vcov(lrr_if["heart"] - lrr_if["cereb"], des)))
        lo, hi, p = tci(lr, se, df)
        row.update({"rr_ratio_heart_over_cereb": np.exp(lr), "rr_ratio_lower": np.exp(lo), "rr_ratio_upper": np.exp(hi),
                    "p_rr_ratio": p})
        out.append(row)
    return per, pd.DataFrame(out)


def fg_expand(t, status, cause, w, dom, gvar):
    """Fine-Gray counting-process expansion with censoring weights G(t-)/G(T-) (G by survey-weighted KM per gvar)."""
    n = len(t)
    m = np.asarray(dom, bool) & (w > 0)
    stop0 = t.copy()
    pid, start, stop, wf = [np.arange(n)], [np.zeros(n)], [stop0], [np.where(m, 1.0, 0.0)]
    comp = m & (status != 0) & (status != cause)
    for g in np.unique(gvar[m]):
        x = m & (gvar == g)
        km = ls.km(t[x], (status[x] == 0).astype(float), weights=w[x])
        u, G = km["time"].values, km["surv"].values
        for i in np.flatnonzero(comp & (gvar == g)):
            jl = int(np.searchsorted(u, t[i], side="left"))     # first censoring jump >= T_i
            if jl >= len(u):
                continue
            Gi = G[jl - 1] if jl > 0 else 1.0                   # G(T_i -)
            stop0[i] = u[jl]
            if jl < len(u) - 1 and Gi > 0:
                js = np.arange(jl, len(u) - 1)
                pid.append(np.full(len(js), i))
                start.append(u[js])
                stop.append(u[js + 1])
                wf.append(G[js] / Gi)
    pid = np.concatenate(pid)
    ev = np.where(np.arange(len(pid)) < n, ((status == cause) & m).astype(float)[pid], 0.0)
    return pid, np.concatenate(start), np.concatenate(stop), ev, np.concatenate(wf)


def fg_pair(d, des, strata, psu, dom, expo, covs, gvar, cache, label):
    """Fine-Gray sHR for heart and cerebrovascular death; joint design-based variance; sHR ratio heart/cereb."""
    cols = list(expo) + list(covs)
    m = np.asarray(dom, bool) & d[cols].notna().all(axis=1).values
    n, k = len(d), len(expo)
    t = d["t"].fillna(1).values.astype(float)
    st = d["cr_status"].fillna(0).values.astype(float)
    w = des.weights * m
    X = d[cols].fillna(0).values.astype(float)
    df = dom_df(des, m)
    aggs, fits, qc = {}, {}, {}
    for cause, clab in ((1, "heart"), (2, "cereb")):
        key = (cause, m.tobytes())
        if key not in cache:
            cache[key] = fg_expand(t, st, cause, w, m, gvar)
        pid, sta, sto, ev, wf = cache[key]
        sdes = ls.SurveyDesign(w[pid] * wf, strata[pid], psu[pid], nest=True, lonely_psu="adjust")
        f = ls.coxph(sto, ev, X[pid], entry=sta, design=sdes, names=cols)
        agg = np.zeros((n, len(cols)))
        np.add.at(agg, pid, f.dfbeta())
        aggs[clab], fits[clab] = agg, f
        qc[clab] = (len(pid), f.var_design)
    V = ls.design_vcov(np.column_stack([aggs["heart"][:, :k], aggs["cereb"][:, :k]]), des.domain(m))
    rows = []
    for j, c in enumerate(expo):
        r = {"spec": label, "term": c, "n": int(m.sum()), "n_exposed": int((m & (d[c].values == 1)).sum()),
             "heart_events_exposed": int((m & (st == 1) & (d[c].values == 1)).sum()),
             "cereb_events_exposed": int((m & (st == 2) & (d[c].values == 1)).sum()),
             "heart_events": int((m & (st == 1)).sum()), "cereb_events": int((m & (st == 2)).sum()), "design_df": df,
             "rows_expanded_heart": qc["heart"][0], "rows_expanded_cereb": qc["cereb"][0],
             "qc_se_heart_rowdesign_vs_person": abs(np.sqrt(qc["heart"][1][j, j]) - np.sqrt(V[j, j])),
             "note": "; ".join(sorted(set(fits["heart"].warnings + fits["cereb"].warnings)))}
        for lab, b, v in (("heart", fits["heart"].coef[j], V[j, j]), ("cereb", fits["cereb"].coef[j], V[k + j, k + j])):
            se = float(np.sqrt(v))
            lo, hi, p = tci(b, se, df)
            r.update({f"log_shr_{lab}": float(b), f"se_{lab}": se, f"shr_{lab}": float(np.exp(b)),
                      f"shr_{lab}_lower": float(np.exp(lo)), f"shr_{lab}_upper": float(np.exp(hi)), f"p_{lab}": p})
        lr = float(fits["heart"].coef[j] - fits["cereb"].coef[j])
        se = float(np.sqrt(V[j, j] + V[k + j, k + j] - 2 * V[j, k + j]))
        lo, hi, p = tci(lr, se, df)
        r.update({"log_ratio": lr, "se_ratio": se, "ratio": float(np.exp(lr)), "ratio_lower": float(np.exp(lo)),
                  "ratio_upper": float(np.exp(hi)), "p_ratio": p})
        rows.append(r)
    return rows


# =================================================================================================================
# Data
# =================================================================================================================
def add_retina_columns(d):
    d = d.copy()
    yn = lambda s: s.map({1: 1.0, 2: 0.0})   # noqa: E731  (7/9 = refused/don't know -> NaN)
    d["hx_stroke"] = yn(d["MCQ160F"])
    d["hx_chd"] = pd.concat([yn(d[c]) for c in ("MCQ160C", "MCQ160D", "MCQ160E")], axis=1).max(axis=1)
    d["hx_chf"] = yn(d["MCQ160B"])
    d["avn_both_eyes"] = (d["OPDDAVN"].notna() & d["OPDSAVN"].notna()).astype(int)
    a = d["age_y"]
    for c, lo, hi in (("age_50_59", 50, 60), ("age_60_69", 60, 70), ("age_70_79", 70, 80), ("age_80p", 80, 999)):
        d[c] = ((a >= lo) & (a < hi)).astype(float).where(a.notna())
    d["cr_status"] = np.where(d["death_all"] == 1, np.where(d["death_heart"] == 1, 1,
                                                            np.where(d["death_cereb"] == 1, 2, 3)), 0)
    d["cr_status"] = d["cr_status"].where(d["death_all"].notna())
    return d


def abi_frame(raw):
    d, _ = F.prepare(raw)
    uc, dead, el = raw["UCOD_LEADING"], raw["death_all"], raw["mort_eligible"] == 1
    d["death_heart"] = np.where(el, ((dead == 1) & (uc == 1)).astype(float), np.nan)
    d["death_cereb"] = np.where(el, ((dead == 1) & (uc == 5)).astype(float), np.nan)
    return d


# =================================================================================================================
# Simulation (planted-answer check of the new estimators + wiring; no real file opened)
# =================================================================================================================
def sim_fg(rng, n=24000):
    """Fine & Gray (1999) generating model: F1(t|x) = 1 - [1 - p (1 - exp(-t/10))]^exp(beta x); cause 2 otherwise."""
    beta, p = FG_PLANT["beta"], FG_PLANT["p"]
    x = rng.binomial(1, 0.2, n).astype(float)
    e = np.exp(beta * x)
    p1 = 1 - (1 - p) ** e
    c1 = rng.random(n) < p1
    u = rng.random(n)
    inner = (1 - (1 - u * p1) ** (1 / e)) / p
    T1 = -10 * np.log(np.clip(1 - inner, 1e-12, None))
    T2 = rng.exponential(25 / np.exp(0.3 * x))
    T = np.where(c1, T1, T2)
    cens = rng.uniform(11, 15, n)
    obs = np.minimum(T, cens)
    status = np.where(T <= cens, np.where(c1, 1, 2), 0)
    stv = rng.integers(0, 30, n)
    psu = rng.integers(1, 3, n)
    w = np.exp(rng.normal(9, 0.5, n))
    d = pd.DataFrame({"t": np.ceil(obs * 12) / 12, "cr_status": status.astype(float), "x": x, "cyc": stv // 15})
    truth = {tau: 1 - (1 - p * (1 - np.exp(-tau / 10))) ** np.exp(beta * g) for tau in (10.0,) for g in (0,)}
    des = ls.SurveyDesign(w, stv, psu, nest=True, lonely_psu="adjust")
    return d, des, stv, psu, truth


def sim_retina(rng):
    d = M.simulate(rng)
    n = len(d)
    d["t"] = np.ceil(d["t"] * 12) / 12          # whole months, as PERMTH_EXM (keeps the censoring grid small)
    for c in ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"):
        d[c] = np.where(rng.random(n) < 0.06, 1.0, 2.0)
    d["OPDDAVN"] = np.where(d["avn"] == 1, 2.0, 0.0)
    d["OPDSAVN"] = np.where(rng.random(n) < 0.03, np.nan, d["OPDDAVN"])
    return d


def sim_abi(rng):
    raw = F.simulate(rng)
    n = len(raw)
    dead = raw["death_all"] == 1
    cvd = raw["death_cvd"] == 1
    raw["UCOD_LEADING"] = np.where(cvd, np.where(rng.random(n) < 0.8, 1.0, 5.0),
                                   np.where(dead, rng.choice([2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0], n), np.nan))
    return raw


# =================================================================================================================
# Analyses
# =================================================================================================================
def run_a1(d, des, R):
    strata, psu = d["SDMVSTRA"].values, d["SDMVPSU"].values
    gvar = d["k"].values
    cifs, cifd = [], []
    for s, _ in SIGNS:
        per, diff = cif_by_exposure(d, des, R["base"], s, CIF_TIMES)
        cifs.append(per)
        cifd.append(diff)
    cache, fg = {}, []
    for s, _ in SIGNS:
        for mlab, covs in (("M0 (exposure only)", []), ("M1", M.M1), ("M2", M.M2)):
            for r in fg_pair(d, des, strata, psu, R["base"], [s], covs, gvar, cache, mlab):
                fg.append(r)
    fg = pd.DataFrame(fg)
    # QC: M0 sHR vs complementary log-log contrast of the 10-y cumulative incidences
    qc = []
    cd = pd.concat(cifd, ignore_index=True)
    for s, _ in SIGNS:
        x = cd[(cd["sign"] == s) & (cd["time_y"] == 10.0)].iloc[0]
        for clab in ("heart", "cereb"):
            f1, f0 = x[f"cif_{clab}_exposed"], x[f"cif_{clab}_unexposed"]
            cll = np.log(-np.log(1 - f1)) - np.log(-np.log(1 - f0))
            y = fg[(fg["term"] == s) & (fg["spec"] == "M0 (exposure only)")].iloc[0]
            qc.append({"sign": s, "cause": clab, "log_shr_M0": y[f"log_shr_{clab}"], "cloglog_contrast_10y": cll,
                       "abs_diff": abs(y[f"log_shr_{clab}"] - cll)})
    return pd.concat(cifs, ignore_index=True), cd, fg, pd.DataFrame(qc)


def run_a2(d, des, R, d2, des2):
    ret_ok = R["retina"] & (d["mort_eligible"] == 1).values
    abi_ok = ((d2["age_y"] >= 40) & (d2["abi_class"] != "missing") & (d2["mort_eligible"] == 1)).values
    dom_ret = ret_ok & d[H_RET + ["hx_cvd_any"]].notna().all(axis=1).values
    dom_abi = abi_ok & d2[H_ABI + ["hx_cvd_any"]].notna().all(axis=1).values
    abi_base = abi_ok & d2[F.M2].notna().all(axis=1).values
    y_ret = (d["age_y"] < 80).values
    y_abi = (d2["age_y"] < 80).values
    specs = [("Original (task f M2 / L4 M2)", abi_base, F.M2, R["base"], M.M2, [None]),
             ("Harmonised H", dom_abi, H_ABI, dom_ret, H_RET, HORIZONS),
             ("Harmonised H + prevalent CVD", dom_abi, H_ABI + ["hx_cvd_any"], dom_ret, H_RET + ["hx_cvd_any"],
              HORIZONS),
             ("Harmonised H, age 40-79 both", dom_abi & y_abi, H_ABI, dom_ret & y_ret, H_RET, [None, 15.0, 10.0])]
    fits, rr = [], []
    for lab, da, ca, dr, cr, hs in specs:
        for h in hs:
            hl = "untruncated" if h is None else f"{h:g} y"
            a = {r["term"]: r for r in cs_pair(d2, des2, da, ABI_EXPO, ca, horizon=h, label=f"{lab} | {hl}")}
            b = {}
            for s, _ in SIGNS:
                b.update({r["term"]: r for r in cs_pair(d, des, dr, [s], cr, horizon=h, label=f"{lab} | {hl}")})
            for side, dct in (("ABI 1999-2004", a), ("retina 2005-2008", b)):
                for r in dct.values():
                    fits.append({"sample": side, **r})
            for pa, pb in (("abi_lt09", "avn"), ("abi_0909", "avn"), ("abi_lt09", "ret_any")):
                rr.append({"spec": lab, "horizon": hl, "bed_A": pa, "bed_B": pb,
                           "ratio_A": a[pa].get("ratio", np.nan), "ratio_A_lower": a[pa].get("ratio_lower", np.nan),
                           "ratio_A_upper": a[pa].get("ratio_upper", np.nan),
                           "ratio_B": b[pb].get("ratio", np.nan), "ratio_B_lower": b[pb].get("ratio_lower", np.nan),
                           "ratio_B_upper": b[pb].get("ratio_upper", np.nan),
                           "events_A_heart_cereb_exposed": f"{a[pa]['heart_events_exposed']}/{a[pa]['cereb_events_exposed']}",
                           "events_B_heart_cereb_exposed": f"{b[pb]['heart_events_exposed']}/{b[pb]['cereb_events_exposed']}",
                           "n_A": a[pa]["n"], "n_B": b[pb]["n"],
                           "flag_lt20": bool(min(a[pa]["cereb_events_exposed"], b[pb]["cereb_events_exposed"]) < FLAG_EV),
                           **ror(a[pa], b[pb])})
    return pd.DataFrame(fits), pd.DataFrame(rr)


def run_a3(d, des, R):
    base = R["base"]
    noCVD = (d["hx_cvd_any"] == 0).values
    age79 = (d["age_y"] < 80).values
    lm = d.copy()
    lm["t"] = d["t"] - 2
    gt2 = (d["t"] > 2).values
    m2_bands = [c for c in M.M2 if c != "age10"] + AGE_BANDS
    scen = [
        ("L4 M2 (reproduction)", d, base, M.M2),
        ("Deaths in first 2 y excluded (follow-up from 2 y)", lm, base & gt2, M.M2),
        ("Age 40-79", d, base & age79, M.M2),
        ("Age in 5 bands (>= 80 incl. top-coded) instead of linear", d, base, m2_bands),
        ("M2 + prevalent stroke, CHD, heart failure (separate terms)", d, base, M.M2 + HX),
        ("Prevalent stroke excluded", d, base & (d["hx_stroke"] == 0).values, M.M2),
        ("Any prevalent CVD excluded", d, base & noCVD, M.M2),
        ("AVN graded in both eyes", d, base & (d["avn_both_eyes"] == 1).values, M.M2),
        ("AVN graded in one eye only", d, base & (d["avn_both_eyes"] == 0).values, M.M2),
        ("2005-2006 cycle only", d, base & (d["k"] == 4).values, M.M2),
        ("2007-2008 cycle only (without 2005-2006)", d, base & (d["k"] == 5).values, M.M2),
        ("Combined: age 40-79, no prevalent CVD, first 2 y excluded", lm, base & age79 & noCVD & gt2, M.M2),
    ]
    rows = []
    for lab, dd, dom, covs in scen:
        for r in cs_pair(dd, des, dom, ["avn"], covs, label=lab):
            rows.append(r)
    res = pd.DataFrame(rows)
    het = []
    a = res[res["spec"] == "2005-2006 cycle only"].iloc[0]
    b = res[res["spec"] == "2007-2008 cycle only (without 2005-2006)"].iloc[0]
    for q, se in (("log_hr_cereb", "se_cereb"), ("log_hr_heart", "se_heart"), ("log_ratio", "se_ratio")):
        dlt = b[q] - a[q]
        s = float(np.sqrt(a[se] ** 2 + b[se] ** 2))
        df = int(min(a["design_df"], b["design_df"]))
        lo, hi, p = tci(dlt, s, df)
        het.append({"quantity": q.replace("log_", ""), "est_2005_2006": float(np.exp(a[q])),
                    "est_2007_2008": float(np.exp(b[q])), "ratio_2007_08_over_2005_06": float(np.exp(dlt)),
                    "lower": float(np.exp(lo)), "upper": float(np.exp(hi)), "p_heterogeneity": p, "df": df})
    return res, pd.DataFrame(het)


def run_a4():
    cox = pd.read_csv(L4_DIR / "svy_cox_all.csv")
    con = pd.read_csv(L4_DIR / "heart_vs_cereb_contrast.csv")
    cox["src"] = [f"results/stage0/analysis/m_nhanes_retina_mortality/svy_cox_all.csv:{i + 2}" for i in range(len(cox))]
    con["src"] = [f"results/stage0/analysis/m_nhanes_retina_mortality/heart_vs_cereb_contrast.csv:{i + 2}"
                  for i in range(len(con))]
    main_terms = ["ret_any", "fan", "avn", "nsign_1", "nsign_2p"]
    main_expos = [e for e, _ in M.EXPOS]
    c_rows = con[con["p"].notna() & ~con["term"].astype(str).str.startswith("joint")].copy()
    c_rows["test"] = ("contrast heart/cereb | " + c_rows["exposure"] + " | " + c_rows["term"].astype(str) + " | "
                      + c_rows["model"])
    c_rows["estimate"] = c_rows["ratio_heart_over_cereb"]
    k_rows = cox[cox["p"].notna()].copy()
    k_rows["test"] = ("HR | " + k_rows["exposure"] + " | " + k_rows["term"].astype(str) + " | " + k_rows["outcome"]
                      + " | " + k_rows["model"] + " | " + k_rows["analysis"])
    k_rows["estimate"] = k_rows["hr"]
    fam_a = c_rows[(c_rows["model"] == "M2") & c_rows["exposure"].isin(main_expos) & c_rows["term"].isin(main_terms)]
    fam_b_hr = k_rows[(k_rows["model"] == "M2") & (k_rows["analysis"] == "main") & k_rows["term"].isin(main_terms)]
    fam_b = pd.concat([fam_a, fam_b_hr])
    joint = con[con["term"].astype(str).str.startswith("joint") & con["p"].notna()].copy()
    joint["test"] = "joint F | " + joint["exposure"] + " | " + joint["model"]
    joint["estimate"] = np.nan
    fam_c = pd.concat([c_rows, k_rows, joint])
    out, summ = [], []
    n_fan_nf = int(((con["model"] == "M2") & (con["term"] == "fan") & con["p"].isna()).sum())
    for fam, df_, note in (("A: M2 heart/cereb contrasts, main sign terms", fam_a,
                            f"FAN not fitted (< 3 exposed cerebrovascular deaths; {n_fan_nf} untested member)"),
                           ("B: A + M2 main sign x outcome HRs", fam_b, ""),
                           ("C: every fitted L4 p-value (all models, sensitivity codings; worst case)", fam_c, "")):
        x = df_[["test", "estimate", "p", "src"]].copy().reset_index(drop=True)
        x["bh_q"] = bh(x["p"].values)
        x["bonferroni_p"] = np.minimum(x["p"] * len(x), 1.0)
        x["rank"] = x["p"].rank(method="min").astype(int)
        x.insert(0, "family", fam)
        x["m"] = len(x)
        out.append(x)
        for key, lab in (("contrast heart/cereb | AV nicking (definite) | avn | M2", "AVN heart/cereb contrast (M2)"),
                         ("HR | AV nicking (definite) | avn | Cerebrovascular | M2 | main", "AVN -> cerebrovascular HR (M2)"),
                         ("HR | Any retinopathy | ret_any | Heart | M2 | main", "Retinopathy -> heart HR (M2)")):
            y = x[x["test"] == key]
            if len(y):
                r = y.iloc[0]
                summ.append({"family": fam, "m": len(x), "test": lab, "estimate": r["estimate"], "p": r["p"],
                             "rank": int(r["rank"]), "bh_q": r["bh_q"], "bonferroni_p": r["bonferroni_p"],
                             "src": r["src"], "note": note})
    return pd.concat(out, ignore_index=True), pd.DataFrame(summ)


# =================================================================================================================
# Report tables
# =================================================================================================================
def key_a3(res):
    rows = []
    for _, r in res.iterrows():
        nf = pd.isna(r.get("ratio", np.nan))
        fl = "*" if bool(r.get("flag_lt20") == True) else ""  # noqa: E712
        rows.append({"check": r["spec"], "n": int(r["n"]), "AVN n": int(r["n_exposed"]),
                     "exp. events heart/cereb": f"{int(r['heart_events_exposed'])}/{int(r['cereb_events_exposed'])}",
                     "HR cerebrovascular [95% CI]": "n.f." if nf else fmt_ci(r["hr_cereb"], r["hr_cereb_lower"],
                                                                            r["hr_cereb_upper"]) + fl,
                     "p cereb": "" if nf else fmt_p(r["p_cereb"]),
                     "HR heart [95% CI]": "n.f." if nf else fmt_ci(r["hr_heart"], r["hr_heart_lower"], r["hr_heart_upper"]),
                     "ratio heart/cereb [95% CI]": "n.f." if nf else fmt_ci(r["ratio"], r["ratio_lower"],
                                                                           r["ratio_upper"]) + fl,
                     "p ratio": "" if nf else fmt_p(r["p_ratio"]), "df": int(r["design_df"]),
                     "lean survives (ratio CI < 1)": "" if nf else ("yes" if r["ratio_upper"] < 1 else "no")})
    return pd.DataFrame(rows)


def key_a2(rr, pair=("abi_lt09", "avn")):
    x = rr[(rr["bed_A"] == pair[0]) & (rr["bed_B"] == pair[1])]
    rows = []
    for _, r in x.iterrows():
        fl = "*" if r["flag_lt20"] else ""
        rows.append({"spec": r["spec"], "horizon": r["horizon"],
                     "ratio A heart/cereb [95% CI]": fmt_ci(r["ratio_A"], r["ratio_A_lower"], r["ratio_A_upper"]),
                     "A exp. ev h/c": r["events_A_heart_cereb_exposed"],
                     "ratio B heart/cereb [95% CI]": fmt_ci(r["ratio_B"], r["ratio_B_lower"], r["ratio_B_upper"]),
                     "B exp. ev h/c": r["events_B_heart_cereb_exposed"],
                     "ratio of ratios [95% CI]": fmt_ci(r["ror"], r["ror_lower"], r["ror_upper"]) + fl,
                     "p": fmt_p(r["p_ror"]),
                     "survives (CI > 1)": "" if not np.isfinite(r["ror"]) else ("yes" if r["ror_lower"] > 1 else "no")})
    return pd.DataFrame(rows)


def key_a1_cif(cd):
    rows = []
    for _, r in cd.iterrows():
        rows.append({"sign": r["sign"], "t (y)": f"{r['time_y']:g}",
                     "CIF heart exp/unexp (%)": f"{100 * r['cif_heart_exposed']:.2f} / {100 * r['cif_heart_unexposed']:.2f}",
                     "diff heart (pp) [95% CI]": fmt_ci(100 * r["diff_heart"], 100 * r["diff_heart_lower"],
                                                        100 * r["diff_heart_upper"]),
                     "CIF cereb exp/unexp (%)": f"{100 * r['cif_cereb_exposed']:.2f} / {100 * r['cif_cereb_unexposed']:.2f}",
                     "diff cereb (pp) [95% CI]": fmt_ci(100 * r["diff_cereb"], 100 * r["diff_cereb_lower"],
                                                        100 * r["diff_cereb_upper"]),
                     "p diff cereb": fmt_p(r["p_diff_cereb"]),
                     "RR heart [95% CI]": fmt_ci(r["rr_heart"], r["rr_heart_lower"], r["rr_heart_upper"]),
                     "RR cereb [95% CI]": fmt_ci(r["rr_cereb"], r["rr_cereb_lower"], r["rr_cereb_upper"]),
                     "RR heart / RR cereb [95% CI]": fmt_ci(r["rr_ratio_heart_over_cereb"], r["rr_ratio_lower"],
                                                            r["rr_ratio_upper"]),
                     "p": fmt_p(r["p_rr_ratio"])})
    return pd.DataFrame(rows)


def key_a1_fg(fg):
    rows = []
    for _, r in fg.iterrows():
        rows.append({"sign": r["term"], "model": r["spec"], "n": int(r["n"]),
                     "exp. events heart/cereb": f"{int(r['heart_events_exposed'])}/{int(r['cereb_events_exposed'])}",
                     "sHR heart [95% CI]": fmt_ci(r["shr_heart"], r["shr_heart_lower"], r["shr_heart_upper"]),
                     "sHR cereb [95% CI]": fmt_ci(r["shr_cereb"], r["shr_cereb_lower"], r["shr_cereb_upper"]),
                     "p cereb": fmt_p(r["p_cereb"]),
                     "sHR ratio heart/cereb [95% CI]": fmt_ci(r["ratio"], r["ratio_lower"], r["ratio_upper"]),
                     "p ratio": fmt_p(r["p_ratio"])})
    return pd.DataFrame(rows)


# =================================================================================================================
def main():
    import warnings
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    args = C.parse_args(__doc__.splitlines()[0])
    mode, outdir, token = C.gate(args, TASK)
    rng = np.random.default_rng(args.seed)
    rep = C.Report(TASK, "F1 Stage 0 task m (L12) - robustness of the NHANES retina-mortality findings and of the "
                   "ABI-vs-AVN heart/cerebrovascular contrast (public data; exploratory)", mode, args)
    if mode == "simulate":
        d = sim_retina(rng)
        raw2 = sim_abi(rng)
    else:
        d = M.prepare(M.load_raw())
        raw2, meta = C.load_real(F.TABLE, token)
        rep.inputs.append({"id": meta["table"], "path": str(Path(meta["path"]).relative_to(C.PAPER)),
                           "sha256": meta["sha256"], "rows": meta["rows"]})
    d = add_retina_columns(d)
    R = M.domains(d)
    des = M.design(d)
    d2 = abi_frame(raw2)
    des2 = F.design(d2)

    cif_per, cif_diff, fg, fg_qc = run_a1(d, des, R)
    a2_fits, a2_rr = run_a2(d, des, R, d2, des2)
    a3, a3_het = run_a3(d, des, R)
    a4_all, a4_sum = (run_a4() if mode == "real" else (pd.DataFrame(), pd.DataFrame()))

    # ------------- QC: reproduction of L4 / L2 / derived contrast -------------
    qc = []
    orig = a2_rr[(a2_rr["spec"].str.startswith("Original")) & (a2_rr["bed_A"] == "abi_lt09") & (a2_rr["bed_B"] == "avn")]
    if mode == "real":
        l4 = pd.read_csv(L4_DIR / "heart_vs_cereb_contrast.csv")
        l4r = l4[(l4["model"] == "M2") & (l4["term"] == "avn") & (l4["exposure"] == "AV nicking (definite)")].iloc[0]
        l2 = pd.read_csv(L2_DIR / "contrasts_primary.csv")
        l2r = l2[l2["id"] == "C1-NH9904-abi_lt09-M2"].iloc[0]
        xb = pd.read_csv(XB_DIR / "crossbed_profile.csv").iloc[0]
        r3 = a3.iloc[0]
        a_o = a2_fits[(a2_fits["sample"] == "ABI 1999-2004") & a2_fits["spec"].str.startswith("Original")
                      & (a2_fits["term"] == "abi_lt09")].iloc[0]
        qc += [{"check": "AVN M2 log ratio heart/cereb vs L4", "this": r3["log_ratio"], "reference": l4r["log_ratio"],
                "abs_diff": abs(r3["log_ratio"] - l4r["log_ratio"]),
                "source": "results/stage0/analysis/m_nhanes_retina_mortality/heart_vs_cereb_contrast.csv:9"},
               {"check": "AVN M2 se(log ratio) vs L4", "this": r3["se_ratio"], "reference": l4r["se_design"],
                "abs_diff": abs(r3["se_ratio"] - l4r["se_design"]),
                "source": "results/stage0/analysis/m_nhanes_retina_mortality/heart_vs_cereb_contrast.csv:9"},
               {"check": "ABI < 0.9 M2 log ratio vs L2", "this": a_o["log_ratio"], "reference": l2r["log_ratio"],
                "abs_diff": abs(a_o["log_ratio"] - l2r["log_ratio"]),
                "source": "results/stage0/analysis/m_outcome_contrasts/contrasts_primary.csv:6"},
               {"check": "ABI < 0.9 M2 se(log ratio) vs L2", "this": a_o["se_ratio"], "reference": l2r["se"],
                "abs_diff": abs(a_o["se_ratio"] - l2r["se"]),
                "source": "results/stage0/analysis/m_outcome_contrasts/contrasts_primary.csv:6"},
               {"check": "ABI < 0.9 vs AVN ratio of ratios vs derived report", "this": orig["ror"].iloc[0],
                "reference": xb["ratio_of_ratios"], "abs_diff": abs(orig["ror"].iloc[0] - xb["ratio_of_ratios"]),
                "source": "results/stage0/analysis/m_crossbed_profile_nhanes/crossbed_profile.csv:2"}]
        for p in (L4_DIR / "heart_vs_cereb_contrast.csv", L4_DIR / "svy_cox_all.csv", L2_DIR / "contrasts_primary.csv",
                  XB_DIR / "crossbed_profile.csv"):
            rep.inputs.append({"id": p.name, "path": str(p.relative_to(C.PAPER)), "sha256": C.sha256(p)})
        rep.inputs += list(J.INPUTS.values())
    qc += [{"check": f"FG person-aggregated vs row-design SE ({r['term']}, {r['spec']})", "this": np.nan,
            "reference": np.nan, "abs_diff": r["qc_se_heart_rowdesign_vs_person"], "source": "internal"}
           for _, r in fg.iterrows()]
    qc += [{"check": f"FG M0 log sHR vs cloglog 10-y CIF contrast ({r['sign']}, {r['cause']})",
            "this": r["log_shr_M0"], "reference": r["cloglog_contrast_10y"], "abs_diff": r["abs_diff"],
            "source": "internal (approximate: equal only under proportional subdistribution hazards)"}
           for _, r in fg_qc.iterrows()]
    qc = pd.DataFrame(qc)

    # ------------- summary -------------
    def s_row(analysis, check, est, lo, hi, p, ev, surv):
        return {"analysis": analysis, "check": check, "estimate [95% CI]": fmt_ci(est, lo, hi), "p": fmt_p(p),
                "events (exp. heart/cereb) or note": ev, "survives": surv}
    summ = []
    for s, slab in SIGNS:
        for tau in CIF_TIMES:
            r = cif_diff[(cif_diff["sign"] == s) & (cif_diff["time_y"] == tau)].iloc[0]
            summ.append(s_row("A1 CIF", f"{slab}: RR heart / RR cereb at {tau:g} y (crude)",
                              r["rr_ratio_heart_over_cereb"], r["rr_ratio_lower"], r["rr_ratio_upper"],
                              r["p_rr_ratio"], f"cereb events in window, exposed: "
                              f"{int(cif_per[(cif_per['sign'] == s) & (cif_per['cause'] == 'cereb') & (cif_per['exposed'] == 1) & (cif_per['time_y'] == tau)]['events_by_time'].iloc[0])}",
                              ("yes" if r["rr_ratio_upper"] < 1 else "no") if s == "avn" else ""))
        r = fg[(fg["term"] == s) & (fg["spec"] == "M2")].iloc[0]
        ev = f"{int(r['heart_events_exposed'])}/{int(r['cereb_events_exposed'])}"
        summ.append(s_row("A1 Fine-Gray", f"{slab}: sHR cerebrovascular (M2)", r["shr_cereb"], r["shr_cereb_lower"],
                          r["shr_cereb_upper"], r["p_cereb"], ev,
                          ("yes" if r["shr_cereb_lower"] > 1 else "no") if s == "avn" else ""))
        summ.append(s_row("A1 Fine-Gray", f"{slab}: sHR ratio heart/cereb (M2)", r["ratio"], r["ratio_lower"],
                          r["ratio_upper"], r["p_ratio"], ev,
                          ("yes" if r["ratio_upper"] < 1 else "no") if s == "avn" else ""))
    for _, r in a2_rr[(a2_rr["bed_A"] == "abi_lt09") & (a2_rr["bed_B"] == "avn")].iterrows():
        summ.append(s_row("A2 ABI<0.9 vs AVN", f"{r['spec']} | {r['horizon']}", r["ror"], r["ror_lower"],
                          r["ror_upper"], r["p_ror"], f"A {r['events_A_heart_cereb_exposed']}; "
                          f"B {r['events_B_heart_cereb_exposed']}",
                          "" if not np.isfinite(r["ror"]) else ("yes" if r["ror_lower"] > 1 else "no")))
    for _, r in a3.iterrows():
        if pd.isna(r.get("ratio", np.nan)):
            summ.append(s_row("A3 AVN", r["spec"], np.nan, np.nan, np.nan, np.nan,
                              f"{int(r['heart_events_exposed'])}/{int(r['cereb_events_exposed'])}", "n.f."))
            continue
        summ.append(s_row("A3 AVN", f"{r['spec']}: HR cereb", r["hr_cereb"], r["hr_cereb_lower"],
                          r["hr_cereb_upper"], r["p_cereb"],
                          f"{int(r['heart_events_exposed'])}/{int(r['cereb_events_exposed'])}",
                          "yes" if r["hr_cereb_lower"] > 1 else "no"))
        summ.append(s_row("A3 AVN", f"{r['spec']}: ratio heart/cereb", r["ratio"], r["ratio_lower"],
                          r["ratio_upper"], r["p_ratio"], "", "yes" if r["ratio_upper"] < 1 else "no"))
    for _, r in a3_het.iterrows():
        summ.append(s_row("A3 AVN cycle heterogeneity", f"{r['quantity']}: 2007-2008 / 2005-2006",
                          r["ratio_2007_08_over_2005_06"], r["lower"], r["upper"], r["p_heterogeneity"],
                          f"df {int(r['df'])}", ""))
    for _, r in a4_sum.iterrows():
        summ.append({"analysis": "A4 multiplicity", "check": f"{r['test']} | family {r['family'].split(':')[0]} "
                     f"(m = {r['m']})", "estimate [95% CI]": f"{r['estimate']:.3g}", "p": fmt_p(r["p"]),
                     "events (exp. heart/cereb) or note": f"BH q {fmt_p(r['bh_q'])}; Bonferroni {fmt_p(r['bonferroni_p'])}",
                     "survives": "yes" if r["bh_q"] < 0.05 else "no"})
    summ = pd.DataFrame(summ)

    # ------------- report -------------
    rep.p("- **Label:** public-data supporting evidence for F1 (PLAN 9.3d L12). Exploratory. NHANES 2005-2008 "
          "retinal signs (L4) and NHANES 1999-2004 ABI (task f / L2); 2019 linked mortality. No HPP data; no liver "
          "phenotypes. Design-based t(df) intervals. `*` = < 20 exposed events in a cause; n.f. = not fitted.")
    rep.p("- Inputs: raw public "
          "NHANES XPORT/LMF files via the L4 reader; `results/stage0/clean/nhanes9904_abi.parquet`; L4/L2/derived CSVs "
          "(provenance.json).")
    rep.p("- 'survives': A1/A3 AVN rows = CI of the HR (cerebrovascular) above 1, or CI of the heart/cerebrovascular "
          "ratio below 1; A2 = CI of the ratio of ratios above 1; A4 = BH q < 0.05.")
    rep.h("Summary of checks")
    rep.table("summary", summ, None, max_rows=None)
    rep.h("A2 Harmonised cross-sample contrast: ABI < 0.9 (1999-2004) vs AVN (2005-2008)")
    rep.table("key_a2_abi_lt09_vs_avn", key_a2(a2_rr), None, max_rows=None)
    rep.h("A3 AVN robustness (M2 unless stated)")
    rep.table("key_a3_avn", key_a3(a3), None, max_rows=None)
    rep.table("a3_cycle_heterogeneity", a3_het, "Cycle heterogeneity (2007-2008 vs 2005-2006; independent strata)",
              max_rows=None)
    rep.h("A1 Competing risks")
    rep.table("key_a1_cif", key_a1_cif(cif_diff), "Survey-weighted Aalen-Johansen cumulative incidence (domain D, "
              "crude): exposed vs unexposed; other deaths competing", max_rows=None)
    rep.table("key_a1_finegray", key_a1_fg(fg), "Fine-Gray subdistribution HR (weighted counting-process Cox, "
              "censoring KM within cycle, design-based t(df))", max_rows=None)
    rep.h("A4 Multiplicity over the L4 p-values")
    if len(a4_sum):
        rep.table("a4_summary", a4_sum, "Standing of the key L4 tests within each family", max_rows=None)
    rep.h("A2 secondary pairs: ABI 0.90-0.99 vs AVN; ABI < 0.9 vs any retinopathy")
    rep.table("key_a2_abi_0909_vs_avn", key_a2(a2_rr, ("abi_0909", "avn")), "ABI 0.90-0.99 (1999-2004) vs AVN "
              "(2005-2008); same 3-indicator ABI class model", max_rows=None)
    rep.table("key_a2_abi_lt09_vs_ret", key_a2(a2_rr, ("abi_lt09", "ret_any")), "ABI < 0.9 (1999-2004) vs any "
              "retinopathy (2005-2008)", max_rows=None)
    rep.h("A2 variable definitions by sample")
    rep.table("harmonisation_definitions", pd.DataFrame(HARMONISATION, columns=[
        "variable", "NHANES 1999-2004 (task f)", "NHANES 2005-2008 (L4)", "difference"]), None, max_rows=None)
    rep.h("Null rows phrased as bounds")
    bnd = []
    for _, r in a3.iterrows():
        if np.isfinite(r.get("ratio", np.nan)) and r["ratio_upper"] >= 1:
            bnd.append(f"A3 {r['spec']}: AVN heart/cereb ratio {r['ratio']:.2f}; compatible with {r['ratio_lower']:.2f} "
                       f"to {r['ratio_upper']:.2f} (excludes ratios below {r['ratio_lower']:.2f})")
        if np.isfinite(r.get("hr_cereb", np.nan)) and r["hr_cereb_lower"] <= 1:
            bnd.append(f"A3 {r['spec']}: AVN -> cerebrovascular HR {r['hr_cereb']:.2f}; compatible with "
                       f"{r['hr_cereb_lower']:.2f} to {r['hr_cereb_upper']:.2f} (excludes HR > {r['hr_cereb_upper']:.2f})")
    for _, r in cif_diff.iterrows():
        if r["rr_ratio_lower"] <= 1 <= r["rr_ratio_upper"]:
            bnd.append(f"A1 CIF {r['sign']} at {r['time_y']:g} y: RR heart / RR cereb {r['rr_ratio_heart_over_cereb']:.2f}; "
                       f"compatible with {r['rr_ratio_lower']:.2f} to {r['rr_ratio_upper']:.2f}")
    for _, r in a2_rr[(a2_rr["bed_A"] == "abi_lt09") & (a2_rr["bed_B"] == "avn")].iterrows():
        if np.isfinite(r["ror"]) and r["ror_lower"] <= 1:
            bnd.append(f"A2 {r['spec']} | {r['horizon']}: ratio of ratios {r['ror']:.2f}; compatible with "
                       f"{r['ror_lower']:.2f} to {r['ror_upper']:.2f}")
    rep.bullets(bnd or ["(none)"])
    rep.h("QC: reproduction and estimator checks")
    rep.table("qc", qc, None, max_rows=None)
    rep.h("Full tables")
    rep.table("a1_cif_by_group", cif_per, "Cumulative incidence by group (numeric)", max_rows=None)
    rep.table("a1_cif_contrasts", cif_diff, "CIF differences, risk ratios, ratio of risk ratios (numeric)",
              max_rows=None)
    rep.table("a1_finegray", fg, "Fine-Gray (numeric)", max_rows=None)
    rep.table("a2_fits", a2_fits, "Cause-specific pairs per sample, spec and horizon (numeric)", max_rows=None)
    rep.table("a2_ratio_of_ratios", a2_rr, "Ratio of ratios (numeric)", max_rows=None)
    rep.table("a3_avn_robustness", a3, "AVN robustness (numeric)", max_rows=None)
    if len(a4_all):
        rep.table("a4_multiplicity_all", a4_all, "All tests by family with BH q and Bonferroni", max_rows=None)
    rep.h("Caveats")
    rep.bullets([
        "Cross-sample contrast: different people, cycles (1999-2004 vs 2005-2008), ages at exam and follow-up lengths; "
        "harmonising covariates and horizons removes two known differences, not all.",
        "CIF rows are crude (unadjusted): AVN and retinopathy carriers are older, so absolute differences mix age and "
        "sign; the ratio of risk ratios and the adjusted Fine-Gray rows address this only partly.",
        "Fine-Gray variance ignores the estimation of the censoring distribution (as R finegray with robust variance); "
        "censoring in the LMF is administrative, estimated within cycle.",
        "Age is top-coded (80 in 2007-2008; 85 in 1999-2006); age as the time scale is not used for that reason; age "
        "bands and the 40-79 restriction are the checks.",
        "Prevalent CVD and stroke are self-reported (MCQ160B-F); 'don't know' answers are excluded where used.",
        "Cerebrovascular deaths are few (83 in domain D; 26 among AVN carriers); subgroup rows are imprecise and "
        "flagged. Multiplicity families are defined post hoc over L4; exploratory.",
    ])
    planted = None
    if mode == "simulate":
        sd, sdes, stv, psu, truth = sim_fg(np.random.default_rng(args.seed + 1))
        R_ = np.ones(len(sd), bool)
        r = fg_pair(sd, sdes, stv, psu, R_, ["x"], [], sd["cyc"].values, {}, "sim")[0]
        ci0 = ls.cuminc(sd["t"].values, sd["cr_status"].values, [10.0], cause=1,
                        design=sdes.domain(sd["x"].values == 0))
        a3r = a3.iloc[0]
        planted = C.planted_rows([
            ("Fine-Gray log sHR (cause 1, x)", FG_PLANT["beta"], r["log_shr_heart"], r["se_heart"]),
            ("Aalen-Johansen F1(10 y | x = 0)", truth[10.0], float(ci0["cuminc"].iloc[0]), float(ci0["se"].iloc[0])),
            ("AVN log ratio heart/cereb (L4 simulate plant 0.6; M2 approx.)", M.PLANT["avn_heart"] - M.PLANT["avn_cereb"],
             a3r["log_ratio"], a3r["se_ratio"])])
    C.finish(rep, outdir, __file__, planted)


if __name__ == "__main__":
    main()
