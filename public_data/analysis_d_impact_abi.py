#!/usr/bin/env python3
"""F1 Stage 0, task d: IMPACT-ABI (PLoS One e0177609 S2) - baPWV and ABI vs death, CV death and MACE (Cox).

GATED: refuses to run on real data without --real. Use --simulate for a wiring test.

Pre-specified analysis
----------------------
Population: age >= 20 years (21 participants aged 13-19 excluded); follow-up in days converted to years;
time 0 set to 0.5 day. Cox models, Efron ties, model-based SE (robust SE also reported), PH tests (scaled
Schoenfeld residuals, KM time transform).
Outcomes (primary): all-cause death; CV death (cause-specific; non-CV deaths censored); MACE = CV death, MI or
stroke. Secondary: MACE with all-cause death; HF hospitalisation.
Follow-up rule (pre-specified): non-fatal event follow-up is truncated at the death follow-up:
  t_k* = min(t_k, t_death); event_k* = event_k and t_k <= t_death (events dated after death are censored at death).
  Composites are rebuilt from components: event = any component event*; time = earliest component event time*,
  else the shortest component follow-up* (end of complete observation). Provided MACE columns are only compared
  (QC agreement table).
Exposures (separate models):
  baPWV per SD - max of sides (primary) and mean of sides; SD fixed from the primary baPWV analysis set;
  ABI min per 0.1 lower; ABI min < 0.9 (flag).
  baPWV models are restricted to ABI min >= 0.9 (baPWV invalid with ABI < 0.9; pre-specified), all participants
  as sensitivity.
Adjustment: primary = age (per 10 y), sex, haemodialysis, diabetes, eGFR (per 10 mL/min/1.73 m2) computed from
creatinine with the Japanese equation 194 x Cr^-1.094 x age^-0.287 (x 0.739 women) ("eGFR as available": the
recorded eGFR is missing in 36 %); sensitivity: (i) without eGFR; (ii) recorded eGFR, complete cases.
Sensitivity: (a) excluding participants with an HF hospitalisation on day 0 (possible index admission);
(b) excluding baPWV > 4000 cm/s (baPWV models).
Descriptive: cumulative incidence at 1, 3, 5 years by ABI < 0.9 vs >= 0.9 (1 - KM for death; Aalen-Johansen with
competing death for the other outcomes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_survival as ls  # noqa: E402
import s0an_common as C  # noqa: E402

TASK, TABLE = "d_impact_abi", "impact_abi"
USED = ["age_y", "female", "hemodialysis", "diabetes", "egfr", "creatinine_mgdl", "bapwv_mean_cms", "bapwv_max_cms",
        "abi_min", "qc_bapwv_gt_4000", "death", "death_fu_days", "cv_death", "mi", "mi_fu_days", "stroke",
        "stroke_fu_days", "hf_hosp", "hf_hosp_fu_days", "mace_death_mi_stroke", "mace_death_mi_stroke_fu_days",
        "mace_cvdeath_mi_stroke", "mace_cvdeath_mi_stroke_fu_days"]
PLANT = {"beta_pwv_per100": 0.06, "beta_abi_lt09": 0.6}


# ----------------------------------------------------------------------------------------------------------------
def egfr_jsn(cr, age, female):
    with np.errstate(divide="ignore", invalid="ignore"):
        e = 194 * np.power(cr, -1.094) * np.power(age, -0.287) * np.where(female == 1, 0.739, 1.0)
    return np.where((cr > 0) & (age > 0), e, np.nan)


def simulate(rng) -> pd.DataFrame:
    n = 3056
    age = np.clip(rng.normal(68, 12, n), 21, 99).round()
    age[rng.choice(n, 21, replace=False)] = rng.integers(13, 20, 21)
    female = rng.binomial(1, 0.28, n).astype(float)
    hd = rng.binomial(1, 0.05, n).astype(float)
    dm = rng.binomial(1, 0.28, n).astype(float)
    cr = np.where(hd == 1, rng.uniform(4, 10, n), np.exp(rng.normal(np.log(0.86), 0.3, n))).round(2)
    eg = egfr_jsn(cr, age, female)
    eg_rec = np.where(rng.random(n) < 0.36, np.nan, np.round(eg + rng.normal(0, 2, n)))
    pad = rng.random(n) < 0.15
    abi_min = np.where(pad, rng.uniform(0.3, 0.89, n), rng.normal(1.12, 0.07, n)).round(2)
    pwv_mean = np.round(1000 + 9 * age + 150 * hd + rng.normal(0, 300, n))
    pwv_max = pwv_mean + np.abs(rng.normal(0, 60, n)).round()
    lp = (0.04 * (age - 68) - 0.3 * female + 1.2 * hd + 0.3 * dm - 0.1 * (np.nan_to_num(eg, nan=60) - 60) / 10
          + PLANT["beta_pwv_per100"] * (pwv_max - 1650) / 100 + PLANT["beta_abi_lt09"] * (abi_min < 0.9))
    def tt(rate):
        return -np.log(rng.random(n)) / (rate * np.exp(lp))
    T_death, T_mi, T_st, T_hf = tt(0.025), tt(0.003), tt(0.006), tt(0.012)
    cens = rng.uniform(0.05, 9.8, n)
    death = (T_death <= cens).astype(float)
    t_death = np.minimum(T_death, cens)
    cvd = death * rng.binomial(1, 0.4, n)

    def nonfatal(T):
        ev = (T <= t_death).astype(float)
        t = np.minimum(T, t_death)
        return ev, t
    mi, t_mi = nonfatal(T_mi)
    st, t_st = nonfatal(T_st)
    hf, t_hf = nonfatal(T_hf)
    days = lambda t: np.round(t * 365.25)
    d_death, d_mi, d_st, d_hf = days(t_death), days(t_mi), days(t_st), days(t_hf)
    # quirks of the real file: non-fatal FU exceeding death FU, HF events at day 0, a death at day 0
    q = rng.choice(np.flatnonzero(death == 1), 200, replace=False)
    d_mi[q] += rng.integers(1, 30, 200)
    d_st[q] += rng.integers(1, 30, 200)
    late = q[:5]
    mi[late] = 1                                    # MI dated after death -> must be censored by the rule
    hf0 = rng.choice(np.flatnonzero(hf == 0), 19, replace=False)
    hf[hf0], d_hf[hf0] = 1, 0
    d0 = rng.choice(np.flatnonzero(death == 1), 1)
    d_death[d0] = 0
    d_mi[d0] = d_st[d0] = d_hf[d0] = 0
    mace_cv = np.maximum.reduce([cvd, mi, st])
    mace_all = np.maximum.reduce([death, mi, st])
    t_mace = np.minimum.reduce([np.where(mi == 1, d_mi, np.inf), np.where(st == 1, d_st, np.inf), d_death])
    df = pd.DataFrame({
        "age_y": age, "female": female, "hemodialysis": hd, "diabetes": dm, "egfr": eg_rec, "creatinine_mgdl": cr,
        "bapwv_mean_cms": pwv_mean, "bapwv_max_cms": pwv_max, "abi_min": abi_min,
        "qc_bapwv_gt_4000": (pwv_max > 4000).astype("int64"), "death": death, "death_fu_days": d_death,
        "cv_death": cvd, "mi": mi, "mi_fu_days": d_mi, "stroke": st, "stroke_fu_days": d_st, "hf_hosp": hf,
        "hf_hosp_fu_days": d_hf, "mace_death_mi_stroke": mace_all, "mace_death_mi_stroke_fu_days": t_mace,
        "mace_cvdeath_mi_stroke": mace_cv, "mace_cvdeath_mi_stroke_fu_days": t_mace})
    miss = rng.choice(n, 120, replace=False)
    df.loc[miss[:60], ["bapwv_mean_cms", "bapwv_max_cms"]] = np.nan
    df.loc[miss[60:], "abi_min"] = np.nan
    return df


# ----------------------------------------------------------------------------------------------------------------
def prepare(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = raw[USED].copy()
    td = d["death_fu_days"].astype(float)
    comp = {}
    for k, ev, tm in [("mi", "mi", "mi_fu_days"), ("stroke", "stroke", "stroke_fu_days"),
                      ("hf", "hf_hosp", "hf_hosp_fu_days")]:
        t = d[tm].astype(float)
        e = (d[ev] == 1) & (t <= td)
        comp[k] = (e.astype(float), np.minimum(t, td))
        d[f"{k}_event"] = comp[k][0]
        d[f"{k}_time_days"] = comp[k][1]
        d[f"{k}_event_after_death"] = ((d[ev] == 1) & (t > td)).astype(int)
    comp["cvdeath"] = (d["cv_death"].astype(float), td)
    comp["death"] = (d["death"].astype(float), td)
    for name, parts in [("mace_cv", ["cvdeath", "mi", "stroke"]), ("mace_all", ["death", "mi", "stroke"])]:
        E = np.column_stack([comp[p][0] for p in parts])
        T = np.column_stack([comp[p][1] for p in parts])
        ev = E.max(axis=1)
        t_ev = np.where(E == 1, T, np.inf).min(axis=1)
        t_cens = T.min(axis=1)
        d[f"{name}_event"] = ev
        d[f"{name}_time_days"] = np.where(ev == 1, t_ev, t_cens)
    d["death_event"], d["death_time_days"] = comp["death"]
    d["cvdeath_event"], d["cvdeath_time_days"] = comp["cvdeath"]
    # competing death indicator for Aalen-Johansen (death observed at the end of follow-up without the event)
    for k in ("cvdeath", "mace_cv", "mi", "stroke", "hf"):
        d[f"{k}_ajcode"] = np.where(d[f"{k}_event"] == 1, 1,
                                    np.where((d["death"] == 1) & (d[f"{k}_time_days"] >= td), 2, 0))
    d["mace_all_ajcode"] = d["mace_all_event"].astype(int)
    for k in ("death", "cvdeath", "mace_cv", "mace_all", "hf"):
        d[f"{k}_time_y"] = np.maximum(d[f"{k}_time_days"], 0.5) / 365.25
    d["egfr_jsn"] = egfr_jsn(d["creatinine_mgdl"].values, d["age_y"].values, d["female"].values)
    d["age10"] = d["age_y"] / 10
    d["egfr_jsn10"] = d["egfr_jsn"] / 10
    d["egfr_rec10"] = d["egfr"] / 10
    d["abi_lt09"] = (d["abi_min"] < 0.9).astype(float).where(d["abi_min"].notna())
    d["abi_per01_lower"] = -d["abi_min"] / 0.1
    d["hf_day0"] = ((d["hf_hosp"] == 1) & (d["hf_hosp_fu_days"] <= 0)).astype(int)
    qc = []
    for name, prov in [("mace_cv", "mace_cvdeath_mi_stroke"), ("mace_all", "mace_death_mi_stroke")]:
        a, b = d[f"{name}_event"], d[prov]
        qc.append({"composite": name, "provided_column": prov, "events_rebuilt": int(a.sum()),
                   "events_provided": int(b.sum()), "both": int(((a == 1) & (b == 1)).sum()),
                   "rebuilt_only": int(((a == 1) & (b == 0)).sum()), "provided_only": int(((a == 0) & (b == 1)).sum()),
                   "median_abs_time_diff_days": float((d[f"{name}_time_days"] - d[prov + "_fu_days"]).abs().median())})
    for k in ("mi", "stroke", "hf"):
        qc.append({"composite": f"{k}: events dated after death (censored at death)",
                   "events_rebuilt": int(d[f"{k}_event"].sum()), "provided_only": int(d[f"{k}_event_after_death"].sum())})
    qc.append({"composite": "HF hospitalisation on day 0", "events_provided": int(d["hf_day0"].sum())})
    return d, pd.DataFrame(qc)


OUTCOMES = [("death", "All-cause death", "primary"), ("cvdeath", "CV death", "primary"),
            ("mace_cv", "MACE (CV death, MI, stroke)", "primary"),
            ("mace_all", "MACE (all-cause death, MI, stroke)", "secondary"),
            ("hf", "HF hospitalisation", "secondary")]
ADJ = {"primary (age, sex, HD, DM, eGFR-JSN)": ["age10", "female", "hemodialysis", "diabetes", "egfr_jsn10"],
       "no eGFR": ["age10", "female", "hemodialysis", "diabetes"],
       "recorded eGFR (complete cases)": ["age10", "female", "hemodialysis", "diabetes", "egfr_rec10"]}


def fit(d, mask, outcome, expo, covs):
    cols = [expo] + covs
    m = mask & d[cols + [f"{outcome}_time_y", f"{outcome}_event"]].notna().all(axis=1).values
    s = d[m]
    f = ls.coxph(s[f"{outcome}_time_y"].values, s[f"{outcome}_event"].values, s[cols].values, names=cols,
                 robust=True)
    tm = f.table("model").iloc[0]
    tr = f.table("robust").iloc[0]
    try:
        z = ls.cox_zph(f, "km")
        ph_x, ph_g = float(z["p"].iloc[0]), float(z["p"].iloc[-1])
    except ValueError:
        ph_x = ph_g = np.nan
    return {"n": int(m.sum()), "events": int(s[f"{outcome}_event"].sum()), "log_hr": tm["coef"], "se": tm["se"],
            "hr": tm["exp_coef"], "lower": tm["exp_lower"], "upper": tm["exp_upper"], "p": tm["p"],
            "robust_se": tr["se"], "hr_lower_robust": tr["exp_lower"], "hr_upper_robust": tr["exp_upper"],
            "ph_p_exposure": ph_x, "ph_p_global": ph_g, "converged": f.converged}


def analyse(d):
    base = (d["age_y"] >= 20).values
    abi_ok = (d["abi_min"] >= 0.9).values
    prim_pwv = base & abi_ok & d[["bapwv_max_cms"] + ADJ["primary (age, sex, HD, DM, eGFR-JSN)"]].notna().all(axis=1).values
    sd_max = float(d.loc[prim_pwv, "bapwv_max_cms"].std(ddof=1))
    sd_mean = float(d.loc[prim_pwv, "bapwv_mean_cms"].std(ddof=1))
    d = d.copy()
    d["bapwv_max_perSD"] = (d["bapwv_max_cms"] - d.loc[prim_pwv, "bapwv_max_cms"].mean()) / sd_max
    d["bapwv_mean_perSD"] = (d["bapwv_mean_cms"] - d.loc[prim_pwv, "bapwv_mean_cms"].mean()) / sd_mean
    expos = [("bapwv_max_perSD", "baPWV max (per SD)"), ("bapwv_mean_perSD", "baPWV mean (per SD)"),
             ("abi_per01_lower", "ABI min (per 0.1 lower)"), ("abi_lt09", "ABI min < 0.9")]
    sets = {"main": base, "excl HF day 0": base & (d["hf_day0"] == 0).values}
    rows = []
    for oc, olab, otype in OUTCOMES:
        for ex, elab in expos:
            is_pwv = ex.startswith("bapwv")
            for adj, covs in ADJ.items():
                for sname, smask in sets.items():
                    variants = [("ABI>=0.9" if is_pwv else "all", smask & (abi_ok if is_pwv else True))]
                    if is_pwv and adj.startswith("primary") and sname == "main":
                        variants += [("all incl ABI<0.9", smask),
                                     ("ABI>=0.9, excl baPWV>4000", smask & abi_ok & (d["qc_bapwv_gt_4000"] == 0).values)]
                    if not (adj.startswith("primary") or sname == "main"):
                        continue
                    for vlab, vm in variants:
                        r = fit(d, vm, oc, ex, covs)
                        rows.append({"outcome": olab, "outcome_type": otype, "exposure": elab, "adjustment": adj,
                                     "analysis_set": f"{sname}; {vlab}", **r})
    res = pd.DataFrame(rows)
    scal = pd.DataFrame([{"exposure": "baPWV max", "SD_cms": sd_max}, {"exposure": "baPWV mean", "SD_cms": sd_mean}])
    return res, scal, d


def cuminc_tables(d):
    rows = []
    base = (d["age_y"] >= 20) & d["abi_lt09"].notna()
    for oc, olab, _ in OUTCOMES:
        for g, glab in [(0, "ABI >= 0.9"), (1, "ABI < 0.9")]:
            s = d[base & (d["abi_lt09"] == g)]
            t = s[f"{oc}_time_y"].values
            if oc == "death":
                r = ls.cuminc(t, s["death_event"].values, [1, 3, 5])
            elif oc == "mace_all":
                r = ls.cuminc(t, s["mace_all_event"].values, [1, 3, 5])
            else:
                r = ls.cuminc(t, s[f"{oc}_ajcode"].values, [1, 3, 5], cause=1)
            for _, x in r.iterrows():
                rows.append({"outcome": olab, "group": glab, "n": len(s), "year": x["time"], "cuminc": x["cuminc"],
                             "se": x["se"], "lower": x["lower"], "upper": x["upper"]})
    return pd.DataFrame(rows)


def main():
    args = C.parse_args(__doc__.splitlines()[0])
    mode, outdir, token = C.gate(args, TASK)
    rng = np.random.default_rng(args.seed)
    rep = C.Report(TASK, "F1 Stage 0 task d - IMPACT-ABI: baPWV and ABI vs death, CV death, MACE", mode, args)
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
    rep.table("followup_rule_qc", qc, "Follow-up rule and composite reconstruction (QC; no exposure involved)",
              max_rows=None)
    res, scal, d2 = analyse(d)
    rep.table("exposure_scaling", scal, "SD used for per-SD scaling (primary baPWV analysis set)", max_rows=None)
    prim = res[(res["adjustment"].str.startswith("primary")) & (res["analysis_set"].str.startswith("main"))]
    rep.table("cox_primary_adjustment", prim, "Cox models, primary adjustment, main analysis sets", max_rows=None)
    rep.table("cox_all_models", res, "All Cox models (incl. sensitivity analyses)", max_rows=40)
    ci = cuminc_tables(d2)
    rep.table("cumulative_incidence_by_abi", ci, "Cumulative incidence by ABI group (descriptive)", max_rows=None)
    planted = None
    if mode == "simulate":
        sd_max = float(scal.iloc[0]["SD_cms"])
        pick = lambda o, e, a: res[(res["outcome"] == o) & (res["exposure"] == e) & (res["adjustment"].str.startswith("primary"))
                                   & (res["analysis_set"] == a)].iloc[0]
        ch = []
        for o in ("All-cause death", "CV death", "MACE (CV death, MI, stroke)"):
            r = pick(o, "baPWV max (per SD)", "main; ABI>=0.9")
            ch.append((f"{o}: log-HR per SD baPWV max (ABI>=0.9)", PLANT["beta_pwv_per100"] * sd_max / 100, r["log_hr"], r["se"]))
        r = pick("All-cause death", "ABI min < 0.9", "main; all")
        ch.append(("All-cause death: log-HR ABI<0.9 (approx.: baPWV omitted)", PLANT["beta_abi_lt09"], r["log_hr"], r["se"]))
        planted = C.planted_rows(ch)
    C.finish(rep, outdir, __file__, planted)


if __name__ == "__main__":
    main()
