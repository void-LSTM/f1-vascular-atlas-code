#!/usr/bin/env python3
"""F1 Stage 0, task f: NHANES 1999-2004 ABI + 2019 linked mortality - survey-weighted Cox.

GATED: refuses to run on real data without --real. Use --simulate for a wiring test.

Pre-specified analysis
----------------------
Design: all 9,145 MEC-examined records aged >= 40 kept; 6-year MEC weight `wtmec6yr`, strata SDMVSTRA, PSUs
SDMVPSU nested in strata; Taylor linearisation (PSUs with replacement); lonely PSU 'adjust'. Subpopulation
(domain) analysis by setting weights to 0 outside the domain - never by dropping records.
Domain D: age >= 40, ABI class known, mortality-linkage eligible, complete covariates of the model's M2 set
(the same domain is used for M1 and M2).
Time: follow-up from the MEC exam (`fu_exam_years`); time 0 set to 0.5 month. Outcomes: all-cause death; CVD
death (UCOD heart or cerebrovascular), cause-specific (other deaths censored). Efron ties.
ABI classes (from per-leg ABI): < 0.9 = lowest leg < 0.9; > 1.4 = highest leg > 1.4 with no leg < 0.9;
0.90-0.99 = lowest leg 0.90-0.99 (no leg > 1.4); 1.00-1.40 = reference. Continuous: lowest-leg ABI per 0.1 lower,
restricted to highest leg <= 1.4.
Adjustment: M1 age (per 10 y), sex, race/ethnicity (ref non-Hispanic white); M2 (primary) = M1 + smoking
(former, current vs never), diabetes, seated SBP (per 10 mmHg), antihypertensive prescription, total and HDL
cholesterol (per 10 mg/dL), BMI (per 5 kg/m2), eGFR (per 10), prevalent CVD.
Sensitivity: (a) non-compressible ankle (> 255 mmHg, 2001-2004; no ABI value) added to the > 1.4 class;
(b) excluding prevalent CVD; (c) age as the time scale with delayed entry (entry = age at exam, exit = entry +
follow-up), M2 without age.
Also: survey-weighted prevalence of the ABI classes; survey-weighted cumulative incidence at 5, 10 and 15 years by
ABI class (1 - KM for death; Aalen-Johansen with competing non-CVD death for CVD death); unweighted n and events
per class (cells < 100 persons flagged: small-domain design SEs are unstable, see lib validation V9); PH
diagnostics (unweighted Schoenfeld test on the weighted fit, diagnostic only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_survival as ls  # noqa: E402
import s0an_common as C  # noqa: E402

TASK, TABLE = "f_nhanes9904_abi", "nhanes9904_abi"
USED = ["SEQN", "SDMVSTRA", "SDMVPSU", "wtmec6yr", "age_y", "age_exam_y", "male", "race_eth", "abi_min", "abi_max",
        "abi_cat_nchs", "noncompressible_ankle_any", "mort_eligible", "death_all", "death_cvd", "fu_exam_years",
        "smoking_status", "diabetes_any", "sbp_seated_mean_mmhg", "rx_any_antihypertensive", "tc_mgdl", "hdl_mgdl",
        "bmi_kgm2", "egfr_ckdepi2021", "hx_cvd_any"]
CLASSES = ["1.00-1.40 (ref)", "<0.90", "0.90-0.99", ">1.40"]
RACES = ["Mexican American", "Other Hispanic", "Non-Hispanic black", "Other/multiracial"]
M1 = ["age10", "male", "race_mexam", "race_othhisp", "race_black", "race_other"]
M2 = M1 + ["smk_former", "smk_current", "diabetes_any", "sbp10", "rx_any_antihypertensive", "tc10", "hdl10",
           "bmi5", "egfr10", "hx_cvd_any"]
PLANT = {"<0.90": 0.7, "0.90-0.99": 0.3, ">1.40": 0.4}


def simulate(rng) -> pd.DataFrame:
    rows = []
    n_per = 9145
    strata = np.concatenate([np.arange(1, 14), np.arange(14, 29), np.arange(29, 44)])
    st = rng.choice(strata, n_per)
    psu = rng.choice([1, 2], n_per, p=[0.5, 0.5]).astype(float)
    psu[(st == 5) & (rng.random(n_per) < 0.1)] = 3
    cyc = np.where(st <= 13, "1999-2000", np.where(st <= 28, "2001-2002", "2003-2004"))
    w = np.exp(rng.normal(9.2, 0.8, n_per))
    age = np.clip(rng.normal(60, 13, n_per), 40, 85).round()
    male = rng.binomial(1, 0.49, n_per)
    race = rng.choice(["Non-Hispanic white"] + RACES, n_per, p=[0.53, 0.21, 0.04, 0.19, 0.03])
    u = rng.random(n_per)
    amin = np.where(u < 0.07, rng.uniform(0.3, 0.89, n_per), np.where(u < 0.17, rng.uniform(0.9, 0.99, n_per),
                                                                         rng.uniform(1.0, 1.35, n_per))).round(2)
    amax = np.where(rng.random(n_per) < 0.02, rng.uniform(1.41, 1.9, n_per), np.minimum(amin + np.abs(rng.normal(0, 0.05, n_per)), 1.4)).round(2)
    amax = np.maximum(amax, amin)
    noabi = rng.random(n_per) < 0.17
    amin[noabi] = np.nan
    amax[noabi] = np.nan
    nc = np.where(cyc == "1999-2000", np.nan, (noabi & (rng.random(n_per) < 0.05)).astype(float))
    cls = np.where(np.isnan(amin), "missing", np.where(amin < 0.9, "<0.90", np.where(amax > 1.4, ">1.40",
                                                                                     np.where(amin < 1.0, "0.90-0.99", "1.00-1.40 (ref)"))))
    smk = rng.choice(["never", "former", "current"], n_per, p=[0.47, 0.34, 0.19])
    dm = rng.binomial(1, 0.19, n_per).astype(float)
    sbp = rng.normal(133, 20, n_per).round(1)
    rx = rng.binomial(1, 0.41, n_per).astype(float)
    tc = rng.normal(209, 40, n_per).round()
    hdl = rng.normal(52, 15, n_per).round()
    bmi = rng.normal(28.6, 5.5, n_per).round(2)
    eg = np.clip(rng.normal(84, 18, n_per), 5, 130).round(1)
    cvd = rng.binomial(1, 0.18, n_per).astype(float)
    lp = (0.08 * (age - 60) + 0.3 * male + 0.4 * (smk == "current") + 0.4 * dm + 0.08 * (sbp - 133) / 10
          + 0.4 * cvd - 0.05 * (eg - 84) / 10
          + sum(PLANT[k] * (cls == k) for k in PLANT))
    T = -np.log(rng.random(n_per)) / (0.012 * np.exp(lp))
    cens = rng.uniform(15, 20.75, n_per)
    death = (T <= cens).astype(float)
    fu = np.floor(np.minimum(T, cens) * 12) / 12
    cvdth = death * rng.binomial(1, 0.32, n_per)
    elig = np.ones(n_per, dtype="int64")
    elig[rng.choice(n_per, 13, replace=False)] = 0
    df = pd.DataFrame({"SEQN": np.arange(1, n_per + 1), "SDMVSTRA": st.astype(float), "SDMVPSU": psu, "wtmec6yr": w,
                       "age_y": age, "age_exam_y": np.where(rng.random(n_per) < 0.05, np.nan, age + rng.uniform(0, 1, n_per)),
                       "male": male.astype("int64"), "race_eth": race, "abi_min": amin, "abi_max": amax,
                       "abi_cat_nchs": np.where(cls == "missing", "missing", cls), "noncompressible_ankle_any": nc,
                       "mort_eligible": elig, "death_all": np.where(elig == 1, death, np.nan),
                       "death_cvd": np.where(elig == 1, cvdth, np.nan), "fu_exam_years": np.where(elig == 1, fu, np.nan),
                       "smoking_status": smk, "diabetes_any": dm, "sbp_seated_mean_mmhg": sbp,
                       "rx_any_antihypertensive": rx, "tc_mgdl": tc, "hdl_mgdl": hdl, "bmi_kgm2": bmi,
                       "egfr_ckdepi2021": eg, "hx_cvd_any": cvd})
    for c in ("sbp_seated_mean_mmhg", "tc_mgdl", "bmi_kgm2"):
        df.loc[rng.choice(n_per, 400, replace=False), c] = np.nan
    df["race_eth"] = df["race_eth"].astype(object)
    df["smoking_status"] = df["smoking_status"].astype(object)
    df["abi_cat_nchs"] = df["abi_cat_nchs"].astype(object)
    return df


def prepare(raw):
    d = raw[USED].copy()
    amin, amax = d["abi_min"], d["abi_max"]
    cls = np.select([amin < 0.9, amax > 1.4, amin < 1.0, amin >= 1.0], ["<0.90", ">1.40", "0.90-0.99", "1.00-1.40 (ref)"],
                    default="missing")
    cls = np.where(amin.isna(), "missing", cls)
    d["abi_class"] = cls
    nc = (d["noncompressible_ankle_any"] == 1) & (d["abi_class"] != "<0.90")
    d["abi_class_nc"] = np.where(nc, ">1.40", d["abi_class"])
    for k, lab in [("abi_lt09", "<0.90"), ("abi_0909", "0.90-0.99"), ("abi_gt14", ">1.40")]:
        d[k] = (d["abi_class"] == lab).astype(float)
        d[k + "_nc"] = (d["abi_class_nc"] == lab).astype(float)
    d["abi_per01_lower"] = np.where(amax <= 1.4, -amin / 0.1, np.nan)
    d["age10"] = d["age_y"] / 10
    d["race_mexam"], d["race_othhisp"], d["race_black"], d["race_other"] = [
        (d["race_eth"] == r).astype(float).where(d["race_eth"].notna()) for r in RACES]
    d["smk_former"] = (d["smoking_status"] == "former").astype(float).where(d["smoking_status"].notna())
    d["smk_current"] = (d["smoking_status"] == "current").astype(float).where(d["smoking_status"].notna())
    d["sbp10"] = d["sbp_seated_mean_mmhg"] / 10
    d["tc10"], d["hdl10"] = d["tc_mgdl"] / 10, d["hdl_mgdl"] / 10
    d["bmi5"], d["egfr10"] = d["bmi_kgm2"] / 5, d["egfr_ckdepi2021"] / 10
    d["t"] = np.maximum(d["fu_exam_years"], 0.5 / 12)
    d["entry_age"] = d["age_exam_y"].fillna(d["age_y"])
    d["exit_age"] = d["entry_age"] + d["t"]
    d["cvd_aj"] = np.where(d["death_cvd"] == 1, 1, np.where(d["death_all"] == 1, 2, 0))
    qc = pd.crosstab(d["abi_class"], d["abi_cat_nchs"]).reset_index()
    return d, qc


def design(d):
    return ls.SurveyDesign(d["wtmec6yr"].values, d["SDMVSTRA"].values, d["SDMVPSU"].values, nest=True,
                           lonely_psu="adjust")


def svy_cox(d, des, dom, outcome, expo, covs, entry=None, time="t"):
    cols = expo + covs
    m = dom & d[cols + [time, outcome]].notna().all(axis=1).values
    if entry is not None:
        m &= d[entry].notna().values
    X = d[cols].fillna(0).values
    tt = d[time].fillna(1).values
    ev = d[outcome].fillna(0).values
    en = None if entry is None else d[entry].fillna(0).values
    if entry is not None:
        tt = np.where(m, tt, en + 1)
    f = ls.coxph(tt, ev, X, entry=en, design=des.domain(m), names=cols)
    t = f.table("design", use_t=True)   # t(design df) intervals: few PSUs per stratum, small domains
    try:
        z = ls.cox_zph(f, "km")
        ph = dict(zip(z["term"], z["p"]))
    except ValueError:
        ph = {}
    rows = []
    for j, c in enumerate(expo):
        r = t.iloc[j]
        rows.append({"term": c, "n_unweighted": int(m.sum()), "events_unweighted": int(ev[m].sum()),
                     "log_hr": r["coef"], "se_design": r["se"], "hr": r["exp_coef"], "lower": r["exp_lower"],
                     "upper": r["exp_upper"], "p": r["p"],
                     "ph_p_term": ph.get(c, np.nan), "ph_p_global": ph.get("GLOBAL", np.nan),
                     "design_df": f.design_df})
    return rows


def analyse(d, des):
    base = ((d["age_y"] >= 40) & (d["abi_class"] != "missing") & (d["mort_eligible"] == 1)
            & d[M2].notna().all(axis=1)).values
    expos = {"ABI class (ref 1.00-1.40)": ["abi_lt09", "abi_0909", "abi_gt14"],
             "ABI per 0.1 lower (highest leg <= 1.4)": ["abi_per01_lower"]}
    rows = []
    for oc, olab in [("death_all", "All-cause death"), ("death_cvd", "CVD death")]:
        for elab, ecols in expos.items():
            for alab, covs in [("M1", M1), ("M2 primary", M2)]:
                for r in svy_cox(d, des, base, oc, ecols, covs):
                    rows.append({"outcome": olab, "exposure": elab, "adjustment": alab, "analysis": "main", **r})
            # sensitivity analyses (M2)
            if elab.startswith("ABI class"):
                for r in svy_cox(d, des, base | ((d["abi_class_nc"] != "missing") & (d["mort_eligible"] == 1)
                                                  & d[M2].notna().all(axis=1)).values,
                                 oc, ["abi_lt09_nc", "abi_0909_nc", "abi_gt14_nc"], M2):
                    rows.append({"outcome": olab, "exposure": elab, "adjustment": "M2 primary",
                                 "analysis": "(a) non-compressible ankle in >1.40", **r})
            for r in svy_cox(d, des, base & (d["hx_cvd_any"] == 0).values, oc, ecols,
                             [c for c in M2 if c != "hx_cvd_any"]):
                rows.append({"outcome": olab, "exposure": elab, "adjustment": "M2 primary",
                             "analysis": "(b) excluding prevalent CVD", **r})
            for r in svy_cox(d, des, base, oc, ecols, [c for c in M2 if c != "age10"], entry="entry_age",
                             time="exit_age"):
                rows.append({"outcome": olab, "exposure": elab, "adjustment": "M2 primary (age = time scale)",
                             "analysis": "(c) age as time scale, delayed entry", **r})
    return pd.DataFrame(rows), base


def descriptives(d, des, base):
    rows = []
    known = ((d["age_y"] >= 40) & (d["abi_class"] != "missing")).values
    for lab in CLASSES:
        r = ls.svy_mean((d["abi_class"] == lab).astype(float), des, domain=known, proportion=True)
        ev = d.loc[base & (d["abi_class"] == lab).values]
        rows.append({"abi_class": lab, "n_unweighted_known_abi": int((known & (d["abi_class"] == lab).values).sum()),
                     "weighted_prevalence": r["estimate"], "se": r["se"], "lower": r["lower"], "upper": r["upper"],
                     "n_analysis_domain": len(ev), "deaths": int(ev["death_all"].sum()),
                     "cvd_deaths": int(ev["death_cvd"].sum()), "flag_lt100": len(ev) < 100})
    prev = pd.DataFrame(rows)
    ci = []
    for oc, olab, cause in [("death_all", "All-cause death", None), ("cvd_aj", "CVD death (Aalen-Johansen)", 1)]:
        r = ls.svy_cuminc_by_group(d["t"].fillna(1).values, d[oc].fillna(0).values, d["abi_class"].values,
                                   [5, 10, 15], des, cause=cause, domain=base)
        r = r[r["group"] != "missing"]
        r.insert(0, "outcome", olab)
        ci.append(r)
    return prev, pd.concat(ci, ignore_index=True)


def main():
    args = C.parse_args(__doc__.splitlines()[0])
    mode, outdir, token = C.gate(args, TASK)
    rng = np.random.default_rng(args.seed)
    rep = C.Report(TASK, "F1 Stage 0 task f - NHANES 1999-2004: ABI and mortality (survey-weighted Cox)", mode, args)
    if mode == "simulate":
        raw = simulate(rng)
        rep.h("Wiring check against the real schema (column names/types only)")
        rep.table("wiring_check", C.check_columns(raw, TABLE, USED), max_rows=None)
    else:
        raw, info = C.load_real(TABLE, token, USED)
        rep.inputs.append(info)
    rep.h("Pre-specified analysis")
    rep.p("```text\nPre-specified analysis" + __doc__.split("Pre-specified analysis", 1)[1].rstrip() + "\n```")
    d, qc = prepare(raw)
    des = design(d)
    rep.table("design_summary", pd.DataFrame([{"records": len(d), "strata": des.n_strata, "psus": des.n_psu,
                                               "design_df": des.df}]), "Design", max_rows=None)
    rep.table("abi_class_vs_nchs_qc", qc, "QC: derived ABI class vs parser's abi_cat_nchs (counts)", max_rows=None)
    res, base = analyse(d, des)
    rep.table("svy_cox", res, "Survey-weighted Cox (design-based SE)", max_rows=None)
    prev, ci = descriptives(d, des, base)
    rep.table("abi_class_prevalence", prev, "Survey-weighted ABI class prevalence and events", max_rows=None)
    rep.table("svy_cumulative_incidence", ci, "Survey-weighted cumulative incidence by ABI class", max_rows=None)
    planted = None
    if mode == "simulate":
        sel = res[(res["outcome"] == "All-cause death") & (res["adjustment"] == "M2 primary") &
                  (res["analysis"] == "main") & (res["exposure"].str.startswith("ABI class"))].set_index("term")
        planted = C.planted_rows([(f"death log-HR {k}", v, sel.loc[t, "log_hr"], sel.loc[t, "se_design"])
                                  for k, v, t in [("<0.90", PLANT["<0.90"], "abi_lt09"),
                                                  ("0.90-0.99", PLANT["0.90-0.99"], "abi_0909"),
                                                  (">1.40", PLANT[">1.40"], "abi_gt14")]])
    C.finish(rep, outdir, __file__, planted)


if __name__ == "__main__":
    main()
