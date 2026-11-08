#!/usr/bin/env python3
"""F1 Stage 0, task g: NHANES III ECG LV voltage + SBP + 2019 linked mortality - survey-weighted Cox.

GATED: refuses to run on real data without --real. Use --simulate for a wiring test.

Pre-specified analysis
----------------------
Design: all 9,737 MEC-examined records aged >= 40 kept; MEC exam weight WTPFEX6 (the ECG is an MEC component),
pseudo-strata SDPSTRA6, pseudo-PSUs SDPPSU6 nested in strata; Taylor linearisation; domain analysis by zero
weights (never by dropping records).
Domain D: age >= 40, NH3ECG record present and voltage criteria applicable (`ecg_voltage_valid` = 1: QRS < 120
ms, no complete BBB/IVCD, no pacing), mortality-linkage eligible, SBP and all M2 covariates present.
Time: follow-up from the MEC exam (`fu_exam_years`), time 0 set to 0.5 month; outcomes all-cause death and CVD
death (UCOD heart or cerebrovascular; cause-specific). Efron ties.
Exposures (separate models): Sokolow-Lyon voltage (per SD), Cornell voltage (per SD), Cornell product (per SD,
secondary); LVH flags (Sokolow-Lyon >= 3.5 mV; Cornell voltage > 2.8 mV men / > 2.0 mV women; Cornell product
> 244 mV.ms secondary); SBP (mean of up to 6 readings) per 10 mmHg. SDs are survey-weighted SDs in domain D.
Adjustment: M1 age (per 10 y), sex, race/ethnicity (ref non-Hispanic white); M2 (primary) = M1 + smoking
(former, current), diabetes, BMI (per 5), total and HDL cholesterol (per 10 mg/dL), BP medication, eGFR (per 10),
prior MI, CHF, stroke; M3 = M2 + SBP per 10 mmHg for the voltage exposures, M2 + Sokolow-Lyon and Cornell
voltage (per SD) for SBP.
Voltage within SBP strata (< 120, 120-139, 140-159, >= 160 mmHg): Sokolow-Lyon and Cornell voltage per SD (M2)
fitted in each SBP stratum (domain), plus a pooled model M2 + SBP-stratum indicators + voltage x stratum
interactions, design-based Wald test (3 df).
Also: unweighted n/events per LVH flag and SBP stratum (cells < 100 flagged); PH diagnostics (diagnostic only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_survival as ls  # noqa: E402
import s0an_common as C  # noqa: E402

TASK, TABLE = "g_nhanes3_ecg", "nhanes3_ecg_lvh"
USED = ["SEQN", "SDPSTRA6", "SDPPSU6", "WTPFEX6", "age_y", "male", "race_eth", "has_ecg", "ecg_voltage_valid",
        "sokolow_lyon_mv", "cornell_voltage_mv", "cornell_product_mvms", "lvh_sokolow", "lvh_cornell_voltage",
        "lvh_cornell_product", "sbp_mmhg", "mort_eligible", "death_all", "death_cvd", "fu_exam_years",
        "smoking_status", "diabetes_any", "bmi_kgm2", "tc_mgdl", "hdl_mgdl", "bp_med_selfreport", "egfr_ckdepi2021",
        "hx_mi", "hx_chf", "hx_stroke"]
RACES = ["Non-Hispanic black", "Mexican-American", "Other"]
M1 = ["age10", "male", "race_black", "race_mexam", "race_other"]
M2 = M1 + ["smk_former", "smk_current", "diabetes_any", "bmi5", "tc10", "hdl10", "bp_med_selfreport", "egfr10",
           "hx_mi", "hx_chf", "hx_stroke"]
SBP_STRATA = [(-np.inf, 120, "<120"), (120, 140, "120-139"), (140, 160, "140-159"), (160, np.inf, ">=160")]
PLANT = {"sokolow_sd": 0.15, "sbp10": 0.10}


def simulate(rng) -> pd.DataFrame:
    n = 9737
    st = rng.integers(1, 50, n)
    psu = rng.integers(1, 3, n)
    w = np.exp(rng.normal(8.0, 0.9, n))
    age = np.clip(rng.normal(61, 13, n), 40, 90).round()
    male = rng.binomial(1, 0.48, n)
    race = rng.choice(["Non-Hispanic white"] + RACES, n, p=[0.5, 0.24, 0.22, 0.04])
    sbp = np.clip(rng.normal(134, 21, n) + 0.4 * (age - 61), 80, 244).round()
    sok = np.clip(rng.normal(2.2, 0.75, n) + 0.006 * (sbp - 134), 0, 6.3).round(3)
    cor = np.clip(rng.normal(1.3, 0.5, n) + 0.004 * (sbp - 134), 0, 4.6).round(3)
    qrs = rng.normal(97, 9, n)
    corp = (cor + 0.6 * (male == 0)) * qrs
    has_ecg = (rng.random(n) < 0.88).astype("int64")
    valid = (has_ecg & (rng.random(n) < 0.92)).astype("int64")
    for a in (sok, cor, corp):
        a[has_ecg == 0] = np.nan
    smk = rng.choice(["never", "former", "current"], n, p=[0.45, 0.33, 0.22])
    dm = rng.binomial(1, 0.18, n).astype(float)
    lp = (0.085 * (age - 61) + 0.35 * male + 0.45 * (smk == "current") + 0.5 * dm
          + PLANT["sbp10"] * (sbp - 134) / 10 + PLANT["sokolow_sd"] * (np.nan_to_num(sok, nan=2.2) - 2.2) / 0.75)
    T = -np.log(rng.random(n)) / (0.02 * np.exp(lp))
    cens = rng.uniform(25, 31.08, n)
    death = (T <= cens).astype(float)
    fu = np.floor(np.minimum(T, cens) * 12) / 12
    elig = np.ones(n, dtype="int64")
    elig[rng.choice(n, 4, replace=False)] = 0
    df = pd.DataFrame({
        "SEQN": np.arange(1, n + 1), "SDPSTRA6": st.astype("int64"), "SDPPSU6": psu.astype("int64"), "WTPFEX6": w,
        "age_y": age, "male": male.astype("int64"), "race_eth": race.astype(object), "has_ecg": has_ecg,
        "ecg_voltage_valid": valid, "sokolow_lyon_mv": sok, "cornell_voltage_mv": cor, "cornell_product_mvms": corp,
        "lvh_sokolow": np.where(np.isnan(sok), np.nan, (sok >= 3.5).astype(float)),
        "lvh_cornell_voltage": np.where(np.isnan(cor), np.nan, (cor > np.where(male == 1, 2.8, 2.0)).astype(float)),
        "lvh_cornell_product": np.where(np.isnan(corp), np.nan, (corp > 244).astype(float)),
        "sbp_mmhg": sbp, "mort_eligible": elig, "death_all": np.where(elig == 1, death, np.nan),
        "death_cvd": np.where(elig == 1, death * rng.binomial(1, 0.37, n), np.nan),
        "fu_exam_years": np.where(elig == 1, fu, np.nan), "smoking_status": smk.astype(object), "diabetes_any": dm,
        "bmi_kgm2": rng.normal(27.6, 5.5, n).round(1), "tc_mgdl": rng.normal(218, 43, n).round(),
        "hdl_mgdl": rng.normal(51, 16, n).round(), "bp_med_selfreport": rng.binomial(1, 0.25, n).astype(float),
        "egfr_ckdepi2021": np.clip(rng.normal(87, 18, n), 4, 176).round(1),
        "hx_mi": rng.binomial(1, 0.08, n).astype(float), "hx_chf": rng.binomial(1, 0.06, n).astype(float),
        "hx_stroke": rng.binomial(1, 0.05, n).astype(float)})
    for c in ("tc_mgdl", "hdl_mgdl", "egfr_ckdepi2021"):
        df.loc[rng.choice(n, 500, replace=False), c] = np.nan
    return df


def prepare(raw):
    d = raw[USED].copy()
    d["age10"] = d["age_y"] / 10
    d["race_black"], d["race_mexam"], d["race_other"] = [
        (d["race_eth"] == r).astype(float).where(d["race_eth"].notna()) for r in RACES]
    d["smk_former"] = (d["smoking_status"] == "former").astype(float).where(d["smoking_status"].notna())
    d["smk_current"] = (d["smoking_status"] == "current").astype(float).where(d["smoking_status"].notna())
    d["bmi5"], d["tc10"], d["hdl10"], d["egfr10"] = d["bmi_kgm2"] / 5, d["tc_mgdl"] / 10, d["hdl_mgdl"] / 10, \
        d["egfr_ckdepi2021"] / 10
    d["sbp10"] = d["sbp_mmhg"] / 10
    d["t"] = np.maximum(d["fu_exam_years"], 0.5 / 12)
    d["sbp_stratum"] = pd.cut(d["sbp_mmhg"], [s[0] for s in SBP_STRATA] + [np.inf], right=False,
                              labels=[s[2] for s in SBP_STRATA]).astype(str)
    return d


def design(d):
    return ls.SurveyDesign(d["WTPFEX6"].values, d["SDPSTRA6"].values, d["SDPPSU6"].values, nest=True)


def svy_cox(d, des, dom, outcome, cols, report_cols):
    m = dom & d[cols + ["t", outcome]].notna().all(axis=1).values
    f = ls.coxph(d["t"].fillna(1).values, d[outcome].fillna(0).values, d[cols].fillna(0).values,
                 design=des.domain(m), names=cols)
    t = f.table("design", use_t=True)   # t(design df) intervals: few PSUs per stratum, small domains
    try:
        z = ls.cox_zph(f, "km")
        ph = dict(zip(z["term"], z["p"]))
    except ValueError:
        ph = {}
    out = []
    for c in report_cols:
        j = cols.index(c)
        r = t.iloc[j]
        out.append({"term": c, "n_unweighted": int(m.sum()), "events_unweighted": int(d.loc[m, outcome].sum()),
                    "log_hr": r["coef"], "se_design": r["se"], "hr": r["exp_coef"], "lower": r["exp_lower"],
                    "upper": r["exp_upper"], "p": r["p"], "ph_p_term": ph.get(c, np.nan),
                    "ph_p_global": ph.get("GLOBAL", np.nan)})
    return out, f


def analyse(d, des):
    dom = ((d["age_y"] >= 40) & (d["has_ecg"] == 1) & (d["ecg_voltage_valid"] == 1) & (d["mort_eligible"] == 1)
           & d[M2 + ["sbp_mmhg", "sokolow_lyon_mv", "cornell_voltage_mv"]].notna().all(axis=1)).values
    scal = []
    for c in ("sokolow_lyon_mv", "cornell_voltage_mv", "cornell_product_mvms"):
        sd = ls.weighted_sd(d.loc[dom, c].values, des.weights[dom])
        d[c + "_sd"] = d[c] / sd
        scal.append({"exposure": c, "weighted_SD_in_domain": sd})
    volts = ["sokolow_lyon_mv_sd", "cornell_voltage_mv_sd", "cornell_product_mvms_sd", "lvh_sokolow",
             "lvh_cornell_voltage", "lvh_cornell_product"]
    rows = []
    for oc, olab in [("death_all", "All-cause death"), ("death_cvd", "CVD death")]:
        for v in volts:
            for alab, covs in [("M1", M1), ("M2 primary", M2), ("M3 (+ SBP)", M2 + ["sbp10"])]:
                r, _ = svy_cox(d, des, dom, oc, [v] + covs, [v])
                rows += [{"outcome": olab, "exposure": v, "adjustment": alab, **x} for x in r]
        for alab, covs in [("M1", M1), ("M2 primary", M2),
                           ("M3 (+ Sokolow, Cornell per SD)", M2 + ["sokolow_lyon_mv_sd", "cornell_voltage_mv_sd"])]:
            r, _ = svy_cox(d, des, dom, oc, ["sbp10"] + covs, ["sbp10"])
            rows += [{"outcome": olab, "exposure": "sbp10", "adjustment": alab, **x} for x in r]
    main = pd.DataFrame(rows)
    # voltage within SBP strata + interaction test
    srows, trows = [], []
    labs = [s[2] for s in SBP_STRATA]
    ind = {lab: (d["sbp_stratum"] == lab).astype(float) for lab in labs}
    for oc, olab in [("death_all", "All-cause death"), ("death_cvd", "CVD death")]:
        for v in ("sokolow_lyon_mv_sd", "cornell_voltage_mv_sd"):
            for lab in labs:
                r, _ = svy_cox(d, des, dom & (d["sbp_stratum"] == lab).values, oc, [v] + M2, [v])
                srows += [{"outcome": olab, "exposure": v, "sbp_stratum": lab, **x} for x in r]
            dd = d.copy()
            inter = []
            for lab in labs[1:]:
                dd[f"s_{lab}"] = ind[lab]
                dd[f"{v}_x_{lab}"] = dd[v] * ind[lab]
                inter.append(f"{v}_x_{lab}")
            cols = [v] + inter + [f"s_{lab}" for lab in labs[1:]] + M2
            _, f = svy_cox(dd, des, dom, oc, cols, [v])
            w = ls.wald_test(f.coef, f.var_design, idx=[1, 2, 3])
            trows.append({"outcome": olab, "exposure": v, "test": "voltage x SBP stratum (3 df, design-based)", **w,
                          "design_df": f.design_df})
    counts = []
    for c in ("lvh_sokolow", "lvh_cornell_voltage", "lvh_cornell_product", "sbp_stratum"):
        g = d[dom].groupby(c)
        for k, s in g:
            counts.append({"variable": c, "level": k, "n_unweighted": len(s), "deaths": int(s["death_all"].sum()),
                           "cvd_deaths": int(s["death_cvd"].sum()), "flag_lt100": len(s) < 100})
    return main, pd.DataFrame(srows), pd.DataFrame(trows), pd.DataFrame(scal), pd.DataFrame(counts)


def main():
    args = C.parse_args(__doc__.splitlines()[0])
    mode, outdir, token = C.gate(args, TASK)
    rng = np.random.default_rng(args.seed)
    rep = C.Report(TASK, "F1 Stage 0 task g - NHANES III: ECG LV voltage, SBP and mortality", mode, args)
    if mode == "simulate":
        raw = simulate(rng)
        rep.h("Wiring check against the real schema (column names/types only)")
        rep.table("wiring_check", C.check_columns(raw, TABLE, USED), max_rows=None)
    else:
        raw, info = C.load_real(TABLE, token, USED)
        rep.inputs.append(info)
    rep.h("Pre-specified analysis")
    rep.p("```text\nPre-specified analysis" + __doc__.split("Pre-specified analysis", 1)[1].rstrip() + "\n```")
    d = prepare(raw)
    des = design(d)
    rep.table("design_summary", pd.DataFrame([{"records": len(d), "strata": des.n_strata, "psus": des.n_psu,
                                               "design_df": des.df}]), "Design", max_rows=None)
    main_, strat, tests, scal, counts = analyse(d, des)
    rep.table("exposure_scaling", scal, "Survey-weighted SD in the analysis domain", max_rows=None)
    rep.table("cell_counts", counts, "Unweighted n and deaths per category (domain D)", max_rows=None)
    rep.table("svy_cox", main_, "Survey-weighted Cox (design-based SE)", max_rows=None)
    rep.table("voltage_within_sbp_strata", strat, "Voltage (per SD) within SBP strata, M2", max_rows=None)
    rep.table("voltage_x_sbp_tests", tests, "Interaction tests", max_rows=None)
    planted = None
    if mode == "simulate":
        sok_sd = float(scal.loc[scal["exposure"] == "sokolow_lyon_mv", "weighted_SD_in_domain"].iloc[0])
        r1 = main_[(main_["outcome"] == "All-cause death") & (main_["exposure"] == "sokolow_lyon_mv_sd") &
                   (main_["adjustment"] == "M3 (+ SBP)")].iloc[0]
        r2 = main_[(main_["outcome"] == "All-cause death") & (main_["exposure"] == "sbp10") &
                   (main_["adjustment"].str.startswith("M3"))].iloc[0]
        planted = C.planted_rows([
            ("death log-HR per SD Sokolow (M3: + SBP)", PLANT["sokolow_sd"] / 0.75 * sok_sd, r1["log_hr"], r1["se_design"]),
            ("death log-HR per 10 mmHg SBP (M3: + voltages; Cornell has no true effect)", PLANT["sbp10"], r2["log_hr"],
             r2["se_design"])])
    C.finish(rep, outdir, __file__, planted)


if __name__ == "__main__":
    main()
