#!/usr/bin/env python3
"""F1 Stage 0, task e: ARTPER (PLoS One e0209163 S4) - ABI vs MACE and ICE (Cox).

GATED: refuses to run on real data without --real. Use --simulate for a wiring test.

Pre-specified analysis
----------------------
Data: incidence file (`artper_incidence`), one Doppler ABI per person (lowest-leg convention presumed, not stated;
ABI >= 1.4 absent by design), follow-up in years. Cox models, Efron ties, model-based SE (robust also reported),
PH tests (scaled Schoenfeld residuals, KM transform). Time 0 (none expected) set to 0.5 day.
Outcomes: MACE (`mace`, `mace_time_y`; = MI, stroke or vascular death, verified) and ICE (`ice`, `ice_time_y`;
composite definition only partly reconstructible - see data notes).
Exposures (separate models): ABI category (`abi_cat3`: 0 normal 1.0-1.39 = reference, 1 borderline 0.90-0.99,
2 low < 0.90); continuous ABI per 0.1 lower; ABI < 0.9 vs >= 0.9.
Adjustment ("as available"): M1 age (per 10 y) + sex; M2 (primary) = M1 + ever smoking, hypertension, diabetes,
hypercholesterolaemia; M3 (sensitivity) = M2 + general obesity category + lipid-lowering and antihypertensive
treatment. Complete cases per model. Trend test across the 3 ordered categories (1 df).
Descriptive: cumulative incidence (1 - KM) at 5 and 10 years by ABI category.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_survival as ls  # noqa: E402
import s0an_common as C  # noqa: E402

TASK, TABLE = "e_artper", "artper_incidence"
USED = ["age_y", "female", "ever_smoker", "hypertension", "diabetes", "hypercholesterolaemia", "obesity_cat3",
        "lipid_lowering_use", "antihypertensive_use", "abi", "abi_cat3", "mace", "mace_time_y", "ice", "ice_time_y"]
PLANT = {"borderline": 0.3, "low": 0.7}
ADJ = {"M1 age + sex": ["age10", "female"],
       "M2 primary": ["age10", "female", "ever_smoker", "hypertension", "diabetes", "hypercholesterolaemia"],
       "M3 sensitivity": ["age10", "female", "ever_smoker", "hypertension", "diabetes", "hypercholesterolaemia",
                          "obese_overweight", "obese", "lipid_lowering_use", "antihypertensive_use"]}


def simulate(rng) -> pd.DataFrame:
    n = 3146
    age = np.round(rng.uniform(49.1, 90, n), 2)
    female = rng.binomial(1, 0.57, n)
    smk = rng.binomial(1, 0.42, n)
    htn = rng.binomial(1, 0.43, n)
    dm = rng.binomial(1, 0.14, n)
    hc = rng.binomial(1, 0.45, n)
    ob = rng.choice([0, 1, 2], n, p=[0.18, 0.46, 0.36]).astype(float)
    ob[rng.choice(n, 4, replace=False)] = np.nan
    u = rng.random(n)
    abi = np.where(u < 0.064, rng.uniform(0.37, 0.899, n), np.where(u < 0.23, rng.uniform(0.9, 0.999, n),
                                                                     rng.uniform(1.0, 1.38, n))).round(3)
    cat = np.where(abi < 0.9, 2, np.where(abi < 1.0, 1, 0)).astype(float)
    lp = (0.06 * (age - 64) - 0.4 * female + 0.3 * smk + 0.3 * htn + 0.4 * dm + 0.1 * hc
          + PLANT["borderline"] * (cat == 1) + PLANT["low"] * (cat == 2))
    T1 = -np.log(rng.random(n)) / (0.006 * np.exp(lp))
    T2 = -np.log(rng.random(n)) / (0.004 * np.exp(lp))
    cens = rng.uniform(0.14, 10.29, n)
    mace = (T1 <= cens).astype(float)
    tm = np.minimum(T1, cens)
    ice_T = np.minimum(T1, T2)
    ice = (ice_T <= cens).astype(float)
    ti = np.minimum(ice_T, cens)
    return pd.DataFrame({"age_y": age, "female": female.astype(float), "ever_smoker": smk.astype(float),
                         "hypertension": htn.astype(float), "diabetes": dm.astype(float),
                         "hypercholesterolaemia": hc.astype(float), "obesity_cat3": ob,
                         "lipid_lowering_use": rng.binomial(1, 0.27, n).astype(float),
                         "antihypertensive_use": rng.binomial(1, 0.38, n).astype(float), "abi": abi,
                         "abi_cat3": cat, "mace": mace, "mace_time_y": np.round(tm, 3), "ice": ice,
                         "ice_time_y": np.round(ti, 3)})


def prepare(raw):
    d = raw[USED].copy()
    d["age10"] = d["age_y"] / 10
    d["abi_borderline"] = (d["abi_cat3"] == 1).astype(float).where(d["abi_cat3"].notna())
    d["abi_low"] = (d["abi_cat3"] == 2).astype(float).where(d["abi_cat3"].notna())
    d["abi_lt09"] = (d["abi"] < 0.9).astype(float).where(d["abi"].notna())
    d["abi_per01_lower"] = -d["abi"] / 0.1
    d["abi_cat_trend"] = d["abi_cat3"]
    d["obese_overweight"] = (d["obesity_cat3"] == 1).astype(float).where(d["obesity_cat3"].notna())
    d["obese"] = (d["obesity_cat3"] == 2).astype(float).where(d["obesity_cat3"].notna())
    for o in ("mace", "ice"):
        d[f"{o}_t"] = np.maximum(d[f"{o}_time_y"], 0.5 / 365.25)
    return d


def fit(d, outcome, expo_cols, covs):
    cols = expo_cols + covs
    m = d[cols + [f"{outcome}_t", outcome]].notna().all(axis=1).values
    s = d[m]
    f = ls.coxph(s[f"{outcome}_t"].values, s[outcome].values, s[cols].values, names=cols, robust=True)
    tm, tr = f.table("model"), f.table("robust")
    try:
        z = ls.cox_zph(f, "km")
    except ValueError:
        z = None
    rows = []
    for j, c in enumerate(expo_cols):
        rows.append({"term": c, "n": int(m.sum()), "events": int(s[outcome].sum()), "log_hr": tm["coef"].iloc[j],
                     "se": tm["se"].iloc[j], "hr": tm["exp_coef"].iloc[j], "lower": tm["exp_lower"].iloc[j],
                     "upper": tm["exp_upper"].iloc[j], "p": tm["p"].iloc[j], "robust_se": tr["se"].iloc[j],
                     "ph_p_term": np.nan if z is None else z["p"].iloc[j],
                     "ph_p_global": np.nan if z is None else z["p"].iloc[-1]})
    return rows


def analyse(d):
    expos = [("ABI category (ref normal)", ["abi_borderline", "abi_low"]), ("ABI per 0.1 lower", ["abi_per01_lower"]),
             ("ABI < 0.9", ["abi_lt09"]), ("ABI category, linear trend (0/1/2)", ["abi_cat_trend"])]
    rows = []
    for o, olab in [("mace", "MACE"), ("ice", "ICE")]:
        for elab, ecols in expos:
            for alab, covs in ADJ.items():
                for r in fit(d, o, ecols, covs):
                    rows.append({"outcome": olab, "exposure": elab, "adjustment": alab, **r})
    return pd.DataFrame(rows)


def cuminc(d):
    rows = []
    for o, olab in [("mace", "MACE"), ("ice", "ICE")]:
        for g, glab in [(0, "normal 1.0-1.39"), (1, "borderline 0.90-0.99"), (2, "low <0.90")]:
            s = d[d["abi_cat3"] == g]
            r = ls.cuminc(s[f"{o}_t"].values, s[o].values, [5, 10])
            for _, x in r.iterrows():
                rows.append({"outcome": olab, "abi_category": glab, "n": len(s), "events": int(s[o].sum()),
                             "year": x["time"], "cuminc": x["cuminc"], "se": x["se"], "lower": x["lower"],
                             "upper": x["upper"]})
    return pd.DataFrame(rows)


def main():
    args = C.parse_args(__doc__.splitlines()[0])
    mode, outdir, token = C.gate(args, TASK)
    rng = np.random.default_rng(args.seed)
    rep = C.Report(TASK, "F1 Stage 0 task e - ARTPER: ABI vs MACE and ICE", mode, args)
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
    res = analyse(d)
    rep.table("cox_models", res, "Cox models (HR per exposure unit)", max_rows=None)
    rep.table("cumulative_incidence_by_abi", cuminc(d), "Cumulative incidence (1 - KM) by ABI category", max_rows=None)
    planted = None
    if mode == "simulate":
        sel = res[(res["outcome"] == "MACE") & (res["adjustment"] == "M2 primary") &
                  (res["exposure"] == "ABI category (ref normal)")].set_index("term")
        planted = C.planted_rows([("MACE log-HR borderline vs normal (M2)", PLANT["borderline"],
                                   sel.loc["abi_borderline", "log_hr"], sel.loc["abi_borderline", "se"]),
                                  ("MACE log-HR low vs normal (M2)", PLANT["low"], sel.loc["abi_low", "log_hr"],
                                   sel.loc["abi_low", "se"])])
    C.finish(rep, outdir, __file__, planted)


if __name__ == "__main__":
    main()
