#!/usr/bin/env python3
"""F1 Stage 0, task m: NHANES 2005-2008 retinal photograph signs and 2019 linked mortality (survey-weighted Cox).

GATED: refuses to run on real data without --real. Use --simulate for a planted-answer wiring test (no real file is opened). Like task j,
this driver reads the raw public files under external/nhanes/ directly (read-only) and writes only
results/stage0/analysis/m_nhanes_retina_mortality/ (simulate mode: results/stage0/validation/drivers/<task>/).

Label: PUBLIC-DATA SUPPORTING EVIDENCE for F1 (PLAN 9.3a L4: retina observational outcome gap). Exploratory. The
exposures are qualitative grader-assessed retinal signs, not vessel calibre. No HPP data. No liver phenotypes.

Pre-specified analysis
----------------------
Data: NHANES 2005-2006 (D) and 2007-2008 (E). DEMO (all MEC-examined records, RIDSTATR = 2, kept in the design),
OPXRET (retinal imaging, target age >= 40; two 45-degree non-mydriatic fields per eye, graded at the University of
Wisconsin), BPX, BPQ, BMX, BIOPRO, GHB, GLU, TCHOL, HDL, DIQ, SMQ, MCQ, RXQ_RX + RXQ_DRUG (Multum categories), and the
2019 public-use linked mortality files. XPORT code 0 is read by pandas as 5.4e-79 and is reset to 0 (task j reader).
Design: pooled 4-year MEC weight = WTMEC2YR / 2; strata SDMVSTRA (asserted unique across the two cycles), PSUs
SDMVPSU nested in strata; Taylor linearisation; lonely PSU 'adjust'; domains by zero weights, never by dropping
records. Coefficient intervals: t(design df), df = #PSU - #strata.
Retina sample: age >= 40, >= 1 eye with gradable images (OPXDGRBL or OPXSGRBL in 0-5; 6 = all fields ungradable), and
all three signs graded. Eye rule: the NHANES derived worse-eye variables (OPDU*); for every sign used here the worse-eye
value equals the maximum of the right- and left-eye grades (checked), i.e. the sign is present in EITHER eye.
Exposures (each in its own model): any retinopathy (OPDUARMA = 1, ETDRS level >= 14); focal arteriolar narrowing
(OPDUFAN = 2, definite; 0 no and 1 questionable = absent, per codebook); AV nicking (OPDUAVN = 2, definite); count of
these three signs 0 / 1 / >= 2 (ref 0). Sensitivity exposures: FAN and AVN with 'questionable' counted as present;
the three signs mutually adjusted.
Outcomes (2019 LMF; ELIGSTAT = 1): all-cause death; CVD death (UCOD_LEADING 1 heart or 5 cerebrovascular); heart (1);
cerebrovascular (5); kidney (9, nephritis/nephrosis). Cause-specific Cox models (other deaths censored); time =
PERMTH_EXM / 12 years from the MEC exam (0 set to 0.5 month); Efron ties.
Covariates: M1 = age (per 10 y), sex, race/ethnicity (RIDRETH1; ref non-Hispanic white). M2 = M1 + seated SBP (mean
of available readings, per 10 mmHg), BP-lowering prescription (Multum antihypertensive classes in RXQ_RX, as task f),
diabetes (self-report, insulin, pills, HbA1c >= 6.5 % or fasting glucose >= 126 mg/dL), HbA1c (per 1 %), smoking
(former, current vs never), total and HDL cholesterol (per 10 mg/dL), BMI (per 5 kg/m2), eGFR CKD-EPI 2021 (per 10;
2005-2006 creatinine calibrated -0.016 + 0.978 x, BIOPRO_D.htm). M3 = M2 without the diabetes term, in participants
without diabetes. Analysis domain D: retina sample + mortality-eligible + complete M2 covariates (the same domain
for M1 and M2; M3 = D and diabetes = 0). Sensitivity: M2 + prevalent CVD (MCQ160B-F).
Counts: unweighted n, events and events among the exposed per row; rows with < 20 exposed events are flagged; rows
with < 3 exposed events are not fitted.
Heart vs cerebrovascular contrast (per sign, M1/M2/M3): Lunn-McNeil data duplication (one record per person and cause
type), baseline hazard stratified by cause type, every covariate interacted with type (equivalent to the two
cause-specific models); contrast = log HR(heart) - log HR(cerebrovascular), design-based variance from the stacked
influence values (both records of a person sit in the same PSU), t(df) interval. QC: the same contrast computed from
the two separate fits' influence values.
Descriptives: survey-weighted prevalence of each sign (retina sample; overall, by diabetes); deaths by underlying
cause; sanity check against a published-scale figure (retinopathy prevalence among adults >= 40 with diabetes,
NHANES 2005-2008, Zhang et al. JAMA 2010: 28.5 % [24.9, 32.5]; their diabetes = self-reported diagnosis or HbA1c >=
6.5 %). PH diagnostics (Grambsch-Therneau on the weighted fit; diagnostic only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_survival as ls  # noqa: E402
import s0an_common as C  # noqa: E402
import s0lib as L  # noqa: E402
import analysis_j_nhanes_acq as J  # noqa: E402  (raw XPORT / LMF readers, CKD-EPI, PH tests, Wald F)

TASK = "m_nhanes_retina_mortality"
NH = L.EXT / "nhanes"
CYCLES = [(4, "2005-2006", "_D"), (5, "2007-2008", "_E")]
DEMO_V = ["SDDSRVYR", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "WTMEC2YR", "SDMVPSU", "SDMVSTRA"]
OPX_V = ["OPASCST2", "OPXDGRBL", "OPXSGRBL", "OPDURL4", "OPDUARMA", "OPDUFAN", "OPDUAVN", "OPDUHOLL", "OPDUBVO",
         "OPDUCVO", "OPDUSE", "OPDDARMA", "OPDSARMA", "OPDDFAN", "OPDSFAN", "OPDDAVN", "OPDSAVN"]
FILE_V = {"BPX": ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"], "BPQ": ["BPQ020", "BPQ040A", "BPQ050A"],
          "BMX": ["BMXBMI"], "BIOPRO": ["LBXSCR"], "GHB": ["LBXGH"], "GLU": ["LBXGLU"], "TCHOL": ["LBXTC"],
          "HDL": ["LBDHDD"], "DIQ": ["DIQ010", "DIQ050", "DIQ070"], "SMQ": ["SMQ020", "SMQ040"],
          "MCQ": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]}
RENAME_V = {"DIQ": {"DID070": "DIQ070"}}     # 'take diabetic pills' is DID070 in DIQ_D / DIQ_E
# Multum second-level categories counted as BP-lowering (as task f / parse_nhanes9904_abi.py): ACEi 42, peripheral
# antiadrenergic 43, central antiadrenergic 44, beta-blockers 47, CCB 48, diuretics 49, vasodilators 53,
# antihypertensive combinations 55, angiotensin II inhibitors 56, aldosterone antagonists 340, renin inhibitors 342
AHT_CODES = {42, 43, 44, 47, 48, 49, 53, 55, 56, 340, 342}
UCOD = {1: "Diseases of heart", 2: "Malignant neoplasms", 3: "Chronic lower respiratory diseases", 4: "Accidents",
        5: "Cerebrovascular diseases", 6: "Alzheimer's disease", 7: "Diabetes mellitus", 8: "Influenza and pneumonia",
        9: "Nephritis/nephrosis", 10: "All other causes"}

M1 = ["age10", "male", "race_mexam", "race_othhisp", "race_black", "race_other"]
M2 = M1 + ["sbp10", "bp_treat", "diabetes_any", "hba1c", "smk_former", "smk_current", "tc10", "hdl10", "bmi5",
           "egfr10"]
M3 = [c for c in M2 if c != "diabetes_any"]
OUTCOMES = [("death_all", "All-cause"), ("death_cvd", "CVD"), ("death_heart", "Heart"),
            ("death_cereb", "Cerebrovascular"), ("death_kidney", "Kidney")]
EXPOS = [("Any retinopathy", ["ret_any"]), ("Focal arteriolar narrowing (definite)", ["fan"]),
         ("AV nicking (definite)", ["avn"]), ("Sign count (ref 0)", ["nsign_1", "nsign_2p"])]
EXPOS_SENS = [("FAN (questionable or definite)", ["fan_q"]), ("AVN (questionable or definite)", ["avn_q"])]
MUTUAL = ("Three signs mutually adjusted", ["ret_any", "fan", "avn"])
MUTUAL2 = ("Retinopathy + AVN mutually adjusted", ["ret_any", "avn"])   # FAN too rare for cerebrovascular deaths
TERM_LAB = {"ret_any": "retinopathy", "fan": "FAN", "avn": "AVN", "nsign_1": "1 sign", "nsign_2p": ">=2 signs",
            "fan_q": "FAN (q+def)", "avn_q": "AVN (q+def)"}
MIN_EXPOSED_EVENTS_FIT, FLAG_EVENTS = 3, 20
PUBLISHED = {"label": "Retinopathy (ETDRS >= 14, worse eye) among adults >= 40 with diabetes, NHANES 2005-2008",
             "estimate": 0.285, "lower": 0.249, "upper": 0.325,
             "source": "Zhang X et al. JAMA 2010;304(6):649-656 (diabetes = self-reported diagnosis or HbA1c >= 6.5 %)"}
PLANT = {"avn_heart": 0.6, "avn_cereb": 0.0, "ret_all": 0.4}


# =================================================================================================================
# Reading (real mode)
# =================================================================================================================
def read_multi_xpt(p: Path) -> pd.DataFrame:
    """XPORT reader for files with several rows per SEQN (RXQ_RX); zeros fixed as in task j's reader."""
    d = pd.read_sas(p, format="xport", encoding="latin-1")
    num = d.select_dtypes(include=[np.floating]).columns
    if len(num):
        v = d[num].to_numpy(copy=True)
        v[(v != 0) & (np.abs(v) < 1e-70)] = 0.0
        d[num] = v
    d["SEQN"] = d["SEQN"].astype(np.int64)
    J._register(p, len(d))
    return d


def rx_bp_lowering() -> pd.DataFrame:
    p = NH / "shared_drug_dictionary" / "RXQ_DRUG.xpt"
    drug = pd.read_sas(p, format="xport", encoding="latin-1")      # drug dictionary: no SEQN
    J._register(p, len(drug))
    cat = pd.concat([drug[["RXDDRGID", f"RXDDCI{i}B"]].rename(columns={f"RXDDCI{i}B": "c"}) for i in range(1, 5)])
    aht_ids = set(cat.loc[cat["c"].round().isin(AHT_CODES), "RXDDRGID"])
    rows = []
    for k, cyc, sfx in CYCLES:
        r = read_multi_xpt(NH / cyc / f"RXQ_RX{sfx}.xpt")
        r["aht"] = r["RXDDRGID"].isin(aht_ids).astype(float)
        g = r.groupby("SEQN").agg(rx_use=("RXDUSE", "first"), rx_aht=("aht", "max"))
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


def load_raw() -> pd.DataFrame:
    parts = []
    for k, cyc, sfx in CYCLES:
        dem = J.read_xpt(NH / cyc / f"DEMO{sfx}.xpt")
        df = dem.loc[dem["RIDSTATR"] == 2, ["SEQN"] + DEMO_V].copy()
        o = J.read_xpt(NH / cyc / f"OPXRET{sfx}.xpt")
        df = df.merge(o[["SEQN"] + OPX_V], on="SEQN", how="left")
        df["in_opxret"] = df["SEQN"].isin(o["SEQN"]).astype(int)
        for key, keep in FILE_V.items():
            x = J.read_xpt(NH / cyc / f"{key}{sfx}.xpt")
            x = x.rename(columns={a: b for a, b in RENAME_V.get(key, {}).items() if a in x.columns})
            miss = [v for v in keep if v not in x.columns]
            assert not miss, (key, cyc, miss)
            df = df.merge(x[["SEQN"] + keep], on="SEQN", how="left")
        df = df.merge(J.read_mort(J.mort_file(cyc)), on="SEQN", how="left")
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True, sort=False)
    raw = raw.merge(rx_bp_lowering(), on="SEQN", how="left")
    for cyc, sfx in (("2005-2006", "_D"), ("2007-2008", "_E")):    # codebooks used for the definitions
        for stem in ("OPXRET", "BIOPRO"):
            J._register(NH / cyc / f"{stem}{sfx}.htm")
    J._register(NH / "linked_mortality_2019" / "R_ReadInProgramAllSurveys.R")
    return raw


# =================================================================================================================
# Derivation
# =================================================================================================================
def worse_eye_qc(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in ("ARMA", "FAN", "AVN"):
        u, r, l_ = raw[f"OPDU{s}"], raw[f"OPDD{s}"], raw[f"OPDS{s}"]
        mx = pd.concat([r, l_], axis=1).max(axis=1)
        both = u.notna() | mx.notna()
        rows.append({"sign": s, "records_graded": int(u.notna().sum()),
                     "worse_eye_equals_max_of_eyes": int(((u == mx) & both).sum()),
                     "disagree": int((both & ~((u == mx) | (u.isna() & mx.isna()))).sum()),
                     "one_eye_only_graded": int((u.notna() & (r.isna() | l_.isna())).sum())})
    return pd.DataFrame(rows)


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.copy()
    d["k"] = d["SDDSRVYR"].round().astype(int)
    d["cycle"] = d["k"].map({4: "2005-2006", 5: "2007-2008"})
    d["w4"] = np.nan_to_num((d["WTMEC2YR"] / 2).astype(float), nan=0.0)
    d["age_y"] = d["RIDAGEYR"].astype(float)
    d["age10"] = d["age_y"] / 10
    d["male"] = (d["RIAGENDR"] == 1).astype(float)
    female = 1 - d["male"]
    for c, code in (("race_mexam", 1), ("race_othhisp", 2), ("race_black", 4), ("race_other", 5)):
        d[c] = (d["RIDRETH1"] == code).astype(float).where(d["RIDRETH1"].notna())
    d["sbp"] = d[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].mean(axis=1)
    d["sbp10"] = d["sbp"] / 10
    d["bp_treat"] = np.where(d["rx_use"] == 2, 0.0, np.where(d["rx_use"] == 1, d["rx_aht"], np.nan))
    d["bp_med_selfreport"] = np.where(d["BPQ020"] == 2, 0.0, d["BPQ050A"].map({1: 1.0, 2: 0.0}))
    dm_yes = ((d["DIQ010"] == 1) | (d["DIQ050"] == 1) | (d["DIQ070"] == 1) | (d["LBXGH"] >= 6.5)
              | (d["LBXGLU"] >= 126))
    dm_no = d["DIQ010"].isin([2, 3]) & (d["LBXGH"] < 6.5)
    d["diabetes_any"] = np.where(dm_yes, 1.0, np.where(dm_no, 0.0, np.nan))
    d["dm_zhang"] = np.where((d["DIQ010"] == 1) | (d["LBXGH"] >= 6.5), 1.0,
                             np.where(d["DIQ010"].isin([2, 3]) & d["LBXGH"].notna(), 0.0, np.nan))
    d["hba1c"] = d["LBXGH"]
    smk = np.select([d["SMQ020"] == 2, (d["SMQ020"] == 1) & (d["SMQ040"] == 3),
                     (d["SMQ020"] == 1) & d["SMQ040"].isin([1, 2])], ["never", "former", "current"], default="")
    d["smk_former"] = np.where(smk == "", np.nan, (smk == "former") * 1.0)
    d["smk_current"] = np.where(smk == "", np.nan, (smk == "current") * 1.0)
    d["tc10"], d["hdl10"], d["bmi5"] = d["LBXTC"] / 10, d["LBDHDD"] / 10, d["BMXBMI"] / 5
    scr = np.where(d["k"] == 4, 0.978 * d["LBXSCR"] - 0.016, d["LBXSCR"])
    d["egfr"] = J.ckd_epi_2021(scr, d["age_y"].values, female.values)
    d["egfr10"] = d["egfr"] / 10
    d["hx_cvd_any"] = pd.concat([d[f"MCQ160{c}"].map({1: 1.0, 2: 0.0}) for c in "BCDEF"], axis=1).max(axis=1)
    # retina
    grad = lambda x: x.round().isin([0, 1, 2, 3, 4, 5])    # noqa: E731
    d["grad_any_eye"] = (grad(d["OPXDGRBL"]) | grad(d["OPXSGRBL"])).astype(int)
    d["ret_any"] = d["OPDUARMA"].round()
    d["fan"] = (d["OPDUFAN"].round() == 2).astype(float).where(d["OPDUFAN"].notna())
    d["avn"] = (d["OPDUAVN"].round() == 2).astype(float).where(d["OPDUAVN"].notna())
    d["fan_q"] = (d["OPDUFAN"].round() >= 1).astype(float).where(d["OPDUFAN"].notna())
    d["avn_q"] = (d["OPDUAVN"].round() >= 1).astype(float).where(d["OPDUAVN"].notna())
    d["nsign"] = d["ret_any"] + d["fan"] + d["avn"]
    d["nsign_1"] = (d["nsign"] == 1).astype(float).where(d["nsign"].notna())
    d["nsign_2p"] = (d["nsign"] >= 2).astype(float).where(d["nsign"].notna())
    d["nsign_0"] = (d["nsign"] == 0).astype(float).where(d["nsign"].notna())
    rl4 = d["OPDURL4"].round()
    d["ret_mild"] = (rl4 == 2).astype(float).where(rl4.notna())
    d["ret_modsev"] = (rl4 == 3).astype(float).where(rl4.notna())
    d["ret_prolif"] = (rl4 == 4).astype(float).where(rl4.notna())
    d["hollenhorst"] = (d["OPDUHOLL"].round() == 2).astype(float).where(d["OPDUHOLL"].notna())
    rvo = pd.concat([d["OPDUBVO"].round() >= 2, d["OPDUCVO"].round() >= 2], axis=1).any(axis=1)
    d["rvo"] = rvo.astype(float).where(d["OPDUBVO"].notna() | d["OPDUCVO"].notna())
    d["soft_exudate"] = (d["OPDUSE"].round() == 2).astype(float).where(d["OPDUSE"].notna())
    # mortality
    elig = d["ELIGSTAT"] == 1
    d["mort_eligible"] = elig.astype(int)
    uc = d["UCOD_LEADING"]
    dead = elig & (d["MORTSTAT"] == 1)
    d["death_all"] = np.where(elig, (d["MORTSTAT"] == 1) * 1.0, np.nan)
    for c, codes in (("death_cvd", [1, 5]), ("death_heart", [1]), ("death_cereb", [5]), ("death_kidney", [9])):
        d[c] = np.where(elig, (dead & uc.isin(codes)) * 1.0, np.nan)
    d["t"] = np.where(elig, np.maximum(d["PERMTH_EXM"] / 12, 0.5 / 12), np.nan)
    assert d.groupby("SDMVSTRA")["k"].nunique().max() == 1, "SDMVSTRA not unique across cycles"
    return d


def domains(d: pd.DataFrame) -> dict:
    retina = ((d["age_y"] >= 40) & (d["grad_any_eye"] == 1) & d[["ret_any", "fan", "avn"]].notna().all(axis=1)
              & (d["w4"] > 0))
    base = retina & (d["mort_eligible"] == 1) & d[M2 + ["t", "death_all"]].notna().all(axis=1)
    return {"retina": retina.values, "base": base.values, "nodm": (base & (d["diabetes_any"] == 0)).values}


def design(d):
    return ls.SurveyDesign(d["w4"].values, d["SDMVSTRA"].values, d["SDMVPSU"].values, nest=True, lonely_psu="adjust")


# =================================================================================================================
# Models
# =================================================================================================================
def fit_cox(d, des, m, outcome, cols):
    X = d[cols].fillna(0).values
    return ls.coxph(d["t"].fillna(1).values, d[outcome].fillna(0).values, X, design=des.domain(m), names=cols)


def cox_rows(d, des, dom, outcome, expo, covs):
    cols = expo + covs
    m = dom & d[cols + ["t", outcome]].notna().all(axis=1).values
    ev = d[outcome].fillna(0).values
    exp_ev = {c: int(ev[m & (d[c].values == 1)].sum()) for c in expo}
    base = {"n_unweighted": int(m.sum()), "events_unweighted": int(ev[m].sum())}
    if min(exp_ev.values()) < MIN_EXPOSED_EVENTS_FIT:
        return [{"term": c, **base, "n_exposed": int((m & (d[c].values == 1)).sum()), "events_exposed": exp_ev[c],
                 "note": f"not fitted (< {MIN_EXPOSED_EVENTS_FIT} exposed events in a term)"} for c in expo]
    f = fit_cox(d, des, m, outcome, cols)
    tab = f.table("design", use_t=True)
    try:
        ph = J.ph_tests(f)
    except Exception:  # noqa: BLE001 - diagnostic only
        ph = {}
    rows = []
    for j, c in enumerate(expo):
        r = tab.iloc[j]
        rows.append({"term": c, **base, "n_exposed": int((m & (d[c].values == 1)).sum()),
                     "events_exposed": exp_ev[c], "log_hr": r["coef"], "se_design": r["se"], "hr": r["exp_coef"],
                     "lower": r["exp_lower"], "upper": r["exp_upper"], "p": r["p"], "design_df": f.design_df,
                     "ph_p_term": ph.get(c, np.nan), "ph_p_global": ph.get("GLOBAL", np.nan),
                     "flag_lt20_exposed_events": exp_ev[c] < FLAG_EVENTS,
                     "note": "; ".join(f.warnings)})
    return rows


def model_sets():
    return [("M1", "base", M1), ("M2", "base", M2), ("M3 (no diabetes)", "nodm", M3)]


def analyse(d, des, R):
    rows = []
    for oc, olab in OUTCOMES:
        for elab, ecols in EXPOS + EXPOS_SENS:
            for mlab, dom, covs in model_sets():
                for r in cox_rows(d, des, R[dom], oc, ecols, covs):
                    rows.append({"outcome": olab, "exposure": elab, "model": mlab,
                                 "analysis": "main" if (elab, ecols) in EXPOS else "sensitivity: questionable = present",
                                 **r})
        for mut in (MUTUAL, MUTUAL2):
            for mlab, dom, covs in model_sets()[1:]:
                for r in cox_rows(d, des, R[dom], oc, mut[1], covs):
                    rows.append({"outcome": olab, "exposure": mut[0], "model": mlab,
                                 "analysis": "sensitivity: mutual adjustment", **r})
        for elab, ecols in EXPOS:
            for r in cox_rows(d, des, R["base"], oc, ecols, M2 + ["hx_cvd_any"]):
                rows.append({"outcome": olab, "exposure": elab, "model": "M2 + prevalent CVD",
                             "analysis": "sensitivity: + prevalent CVD", **r})
    return pd.DataFrame(rows)


def lunn_mcneil(d, des, dom, expo, covs, df_design):
    """Heart vs cerebrovascular death: stacked cause-specific Cox, strata = cause type, all covariates x type."""
    cols = expo + covs
    m = dom & d[cols + ["t", "death_heart", "death_cereb"]].notna().all(axis=1).values
    n, p = len(d), len(cols)
    X = d[cols].fillna(0).values
    Z = np.zeros_like(X)
    Xs = np.vstack([np.hstack([X, Z]), np.hstack([Z, X])])
    tt = np.r_[d["t"].fillna(1).values, d["t"].fillna(1).values]
    ev = np.r_[d["death_heart"].fillna(0).values, d["death_cereb"].fillna(0).values]
    typ = np.r_[np.zeros(n), np.ones(n)]
    sdes = ls.SurveyDesign(np.r_[des.weights, des.weights], np.r_[d["SDMVSTRA"].values, d["SDMVSTRA"].values],
                           np.r_[d["SDMVPSU"].values, d["SDMVPSU"].values], nest=True, lonely_psu="adjust")
    names = [f"{c}|heart" for c in cols] + [f"{c}|cereb" for c in cols]
    f = ls.coxph(tt, ev, Xs, strata=typ, design=sdes.domain(np.r_[m, m]), names=names)
    V = f.var_design
    q = stats.t.ppf(0.975, df_design)
    out, Ls = [], []
    # QC: separate cause-specific fits, joint design variance from their influence values
    fh = fit_cox(d, des, m, "death_heart", cols)
    fc = fit_cox(d, des, m, "death_cereb", cols)
    infl = np.column_stack([fh.dfbeta()[:, :len(expo)], fc.dfbeta()[:, :len(expo)]])
    Vsep = ls.design_vcov(infl, des)
    ev_h, ev_c = d["death_heart"].fillna(0).values, d["death_cereb"].fillna(0).values
    for j, c in enumerate(expo):
        Lv = np.zeros(2 * p)
        Lv[j], Lv[p + j] = 1, -1
        Ls.append(Lv)
        est = float(Lv @ f.coef)
        se = float(np.sqrt(Lv @ V @ Lv))
        k = len(expo)
        se_sep = float(np.sqrt(Vsep[j, j] + Vsep[k + j, k + j] - 2 * Vsep[j, k + j]))
        tval = est / se
        out.append({"term": c, "n_unweighted": int(m.sum()), "heart_events": int(ev_h[m].sum()),
                    "cereb_events": int(ev_c[m].sum()),
                    "heart_events_exposed": int(ev_h[m & (d[c].values == 1)].sum()),
                    "cereb_events_exposed": int(ev_c[m & (d[c].values == 1)].sum()),
                    "hr_heart": float(np.exp(f.coef[j])), "hr_cereb": float(np.exp(f.coef[p + j])),
                    "log_ratio": est, "se_design": se, "ratio_heart_over_cereb": float(np.exp(est)),
                    "lower": float(np.exp(est - q * se)), "upper": float(np.exp(est + q * se)),
                    "p": float(2 * stats.t.sf(abs(tval), df_design)), "design_df": df_design,
                    "qc_abs_diff_logratio_vs_separate_fits": abs(est - (fh.coef[j] - fc.coef[j])),
                    "qc_se_separate_fits": se_sep,
                    "flag_lt20_exposed_events": min(int(ev_h[m & (d[c].values == 1)].sum()),
                                                    int(ev_c[m & (d[c].values == 1)].sum())) < FLAG_EVENTS})
    if len(expo) > 1:
        w = ls.wald_test(f.coef, V, L=np.vstack(Ls))
        wf = J.wald_f(w, df_design)
        out.append({"term": "joint (all count terms)", "n_unweighted": int(m.sum()), "chi2": w["chi2"],
                    "wald_df": w["df"], "p_chi2": w["p"], "F": wf["F"], "p": wf["p_F"], "design_df": df_design})
    return out


def contrasts(d, des, R):
    rows = []
    for elab, ecols in EXPOS + EXPOS_SENS + [MUTUAL2]:
        for mlab, dom, covs in model_sets():
            ok = all(d.loc[R[dom] & (d[c].values == 1), oc].sum() >= MIN_EXPOSED_EVENTS_FIT
                     for c in ecols for oc in ("death_heart", "death_cereb"))
            if not ok:
                rows.append({"exposure": elab, "model": mlab, "term": ",".join(ecols),
                             "note": f"not fitted (< {MIN_EXPOSED_EVENTS_FIT} exposed events)"})
                continue
            for r in lunn_mcneil(d, des, R[dom], ecols, covs, des.df):
                rows.append({"exposure": elab, "model": mlab, **r})
    return pd.DataFrame(rows)


# =================================================================================================================
# Descriptives
# =================================================================================================================
def flow(d, R) -> pd.DataFrame:
    a = d["age_y"] >= 40
    steps = [("MEC-examined, age >= 40 (2005-2008)", a),
             ("in OPXRET file", a & (d["in_opxret"] == 1)),
             ("retinal exam complete or partial (OPASCST2 1-2)", a & d["OPASCST2"].round().isin([1, 2])),
             (">= 1 eye with gradable images (GRBL 0-5)", a & (d["grad_any_eye"] == 1)),
             ("retina sample: + retinopathy, FAN and AVN all graded", pd.Series(R["retina"], index=d.index)),
             ("+ mortality-eligible (ELIGSTAT 1)", pd.Series(R["retina"], index=d.index) & (d["mort_eligible"] == 1)),
             ("analysis domain D: + complete M2 covariates", pd.Series(R["base"], index=d.index)),
             ("M3 domain: D and no diabetes", pd.Series(R["nodm"], index=d.index))]
    rows = []
    for lab, m in steps:
        m = m.values
        rows.append({"step": lab, "n_2005_2006": int((m & (d["k"] == 4).values).sum()),
                     "n_2007_2008": int((m & (d["k"] == 5).values).sum()), "n_total": int(m.sum())})
    return pd.DataFrame(rows)


def deaths_by_cause(d, R) -> pd.DataFrame:
    m = R["base"]
    x = d.loc[m]
    rows = [{"cause": "Alive (assumed) at 31 Dec 2019", "ucod_leading": "", "deaths_unweighted":
             int((x["death_all"] == 0).sum())}]
    for code, lab in UCOD.items():
        rows.append({"cause": lab, "ucod_leading": code,
                     "deaths_unweighted": int(((x["death_all"] == 1) & (x["UCOD_LEADING"] == code)).sum())})
    rows.append({"cause": "All deaths", "ucod_leading": "", "deaths_unweighted": int(x["death_all"].sum())})
    rows.append({"cause": "CVD (heart + cerebrovascular)", "ucod_leading": "1, 5",
                 "deaths_unweighted": int(x["death_cvd"].sum())})
    out = pd.DataFrame(rows)
    out["followup_median_y"] = float(np.median(x["t"]))
    out["followup_max_y"] = float(x["t"].max())
    return out


def prevalence(d, des, R) -> pd.DataFrame:
    signs = [("ret_any", "Any retinopathy (ETDRS >= 14)"), ("ret_mild", "Mild NPR (RL4 = 2)"),
             ("ret_modsev", "Moderate/severe NPR (RL4 = 3)"), ("ret_prolif", "Proliferative (RL4 = 4)"),
             ("fan", "Focal arteriolar narrowing (definite)"), ("fan_q", "FAN (questionable or definite)"),
             ("avn", "AV nicking (definite)"), ("avn_q", "AVN (questionable or definite)"),
             ("nsign_0", "Sign count 0"), ("nsign_1", "Sign count 1"), ("nsign_2p", "Sign count >= 2"),
             ("hollenhorst", "Hollenhorst plaque (definite)"), ("rvo", "Retinal vein occlusion"),
             ("soft_exudate", "Soft exudate (definite)")]
    groups = [("Retina sample", R["retina"]),
              ("Diabetes (M2 definition)", R["retina"] & (d["diabetes_any"] == 1).values),
              ("No diabetes", R["retina"] & (d["diabetes_any"] == 0).values),
              ("Analysis domain D", R["base"])]
    rows = []
    for v, lab in signs:
        for g, gm in groups:
            r = ls.svy_mean(d[v].values, des, domain=gm, proportion=True)
            rows.append({"sign": lab, "variable": v, "group": g, "n_unweighted": r["n"],
                         "n_positive": int(np.nansum(d.loc[gm, v].values)), "weighted_prevalence": r["estimate"],
                         "se": r["se"], "lower": r["lower"], "upper": r["upper"]})
    return pd.DataFrame(rows)


def sanity(d, des, R) -> pd.DataFrame:
    rows = []
    for g, gm in [("Diabetes, Zhang 2010 definition (DIQ010 = 1 or HbA1c >= 6.5 %)",
                   R["retina"] & (d["dm_zhang"] == 1).values),
                  ("No diabetes, Zhang 2010 definition", R["retina"] & (d["dm_zhang"] == 0).values),
                  ("Diabetes, M2 definition", R["retina"] & (d["diabetes_any"] == 1).values)]:
        r = ls.svy_mean(d["ret_any"].values, des, domain=gm, proportion=True)
        rows.append({"group": g, "n_unweighted": r["n"], "weighted_prevalence": r["estimate"], "lower": r["lower"],
                     "upper": r["upper"],
                     "published": PUBLISHED["estimate"] if g.startswith("Diabetes, Zhang") else np.nan,
                     "published_lower": PUBLISHED["lower"] if g.startswith("Diabetes, Zhang") else np.nan,
                     "published_upper": PUBLISHED["upper"] if g.startswith("Diabetes, Zhang") else np.nan})
    return pd.DataFrame(rows)


def covariate_qc(d, R) -> pd.DataFrame:
    m = R["retina"] & (d["mort_eligible"] == 1).values
    x = d.loc[m]
    rows = [{"variable": c, "missing_in_retina_eligible": int(x[c].isna().sum())} for c in M2 + ["hx_cvd_any"]]
    both = x[["bp_treat", "bp_med_selfreport"]].dropna()
    rows.append({"variable": "agreement bp_treat (Rx Multum) vs BPQ050A self-report",
                 "missing_in_retina_eligible": int(len(both)),
                 "value": float((both["bp_treat"] == both["bp_med_selfreport"]).mean())})
    return pd.DataFrame(rows)


def sign_crosstab(d, R) -> pd.DataFrame:
    x = d.loc[R["retina"]]
    return (x.groupby(["ret_any", "fan", "avn"]).size().rename("n_unweighted").reset_index()
            .rename(columns={"ret_any": "retinopathy", "fan": "FAN_definite", "avn": "AVN_definite"}))


VAR_DICT = [
    ("w4", "WTMEC2YR / 2", "Pooled 4-year MEC exam weight (2 cycles)"),
    ("SDMVSTRA, SDMVPSU", "DEMO", "Design strata (unique across cycles, asserted) and PSUs nested in strata"),
    ("grad_any_eye", "OPXDGRBL, OPXSGRBL", "1 if >= 1 eye has GRBL 0-5 (6 = all available fields ungradable)"),
    ("ret_any", "OPDUARMA", "Any retinopathy, worse eye (ETDRS level >= 14; OPDURET); graded masked to diabetes"),
    ("ret_mild / ret_modsev / ret_prolif", "OPDURL4", "Worse-eye severity 2 mild NPR / 3 moderate-severe NPR / 4 PR"),
    ("fan", "OPDUFAN", "Focal arteriolar narrowing definite (2); 0 no, 1 questionable = absent (codebook)"),
    ("avn", "OPDUAVN", "Arteriovenous nicking definite (2); 0 no, 1 questionable = absent (codebook)"),
    ("fan_q, avn_q", "OPDUFAN, OPDUAVN", "Sensitivity: questionable (1) or definite (2) = present"),
    ("nsign, nsign_1, nsign_2p", "ret_any + fan + avn", "Count of the three signs; indicators 1 and >= 2 (ref 0)"),
    ("hollenhorst, rvo, soft_exudate", "OPDUHOLL; OPDUBVO/OPDUCVO; OPDUSE", "Descriptive only (definite = 2; RVO 2-3)"),
    ("death_all", "ELIGSTAT, MORTSTAT", "Death by 31 Dec 2019 (ELIGSTAT = 1 only)"),
    ("death_cvd / death_heart / death_cereb / death_kidney", "UCOD_LEADING", "1 or 5 / 1 / 5 / 9"),
    ("t", "PERMTH_EXM", "Years from MEC exam to death or 31 Dec 2019; minimum 0.5 month"),
    ("age10, male, race_*", "RIDAGEYR, RIAGENDR, RIDRETH1", "Age per 10 y; ref non-Hispanic white"),
    ("sbp10", "BPXSY1-4", "Mean of available seated SBP readings, per 10 mmHg"),
    ("bp_treat", "RXQ_RX (RXDUSE, RXDDRGID) + RXQ_DRUG (RXDDCI1B-4B)",
     "Any prescription in Multum antihypertensive classes 42, 43, 44, 47, 48, 49, 53, 55, 56, 340, 342; 0 if RXDUSE = 2"),
    ("bp_med_selfreport", "BPQ020, BPQ050A", "QC only (agreement with bp_treat)"),
    ("diabetes_any", "DIQ010, DIQ050, DID070, LBXGH, LBXGLU",
     "1 if self-report, insulin, pills, HbA1c >= 6.5 % or FPG >= 126; 0 if DIQ010 2/3 and HbA1c < 6.5"),
    ("dm_zhang", "DIQ010, LBXGH", "Sanity-check definition: self-reported diagnosis or HbA1c >= 6.5 %"),
    ("hba1c", "LBXGH", "HbA1c, per 1 %"),
    ("smk_former, smk_current", "SMQ020, SMQ040", "vs never smokers"),
    ("tc10, hdl10", "LBXTC, LBDHDD", "per 10 mg/dL"),
    ("bmi5", "BMXBMI", "per 5 kg/m2"),
    ("egfr10", "LBXSCR (2005-2006: -0.016 + 0.978 x)", "CKD-EPI 2021 race-free eGFR, per 10 mL/min/1.73 m2"),
    ("hx_cvd_any", "MCQ160B-F", "Self-reported CHF, CHD, angina, MI or stroke (sensitivity only)"),
]


# =================================================================================================================
# Simulation (wiring test; no real file opened)
# =================================================================================================================
def simulate(rng) -> pd.DataFrame:
    n = 12000
    st = rng.integers(44, 74, n)
    psu = rng.integers(1, 3, n).astype(float)
    k = np.where(st < 59, 4, 5)
    age = np.clip(rng.normal(52, 16, n), 20, 85).round()
    male = rng.binomial(1, 0.48, n).astype(float)
    race = rng.choice([1, 2, 3, 4, 5], n, p=[0.17, 0.08, 0.5, 0.2, 0.05])
    dm = rng.binomial(1, 0.17, n).astype(float)
    sbp = rng.normal(127, 19, n)
    ret = rng.binomial(1, np.where(dm == 1, 0.3, 0.08)).astype(float)
    fan = rng.binomial(1, 0.02, n).astype(float)
    avn = rng.binomial(1, 0.12, n).astype(float)
    lp_common = 0.08 * (age - 60) + 0.3 * male + 0.4 * dm + 0.1 * (sbp - 127) / 10
    lam_h = 0.004 * np.exp(lp_common + PLANT["avn_heart"] * avn + PLANT["ret_all"] * ret)
    lam_c = 0.002 * np.exp(lp_common + PLANT["avn_cereb"] * avn + PLANT["ret_all"] * ret)
    lam_o = 0.010 * np.exp(lp_common + PLANT["ret_all"] * ret)
    T = -np.log(rng.random(n)) / (lam_h + lam_c + lam_o)
    u = rng.random(n) * (lam_h + lam_c + lam_o)
    cause = np.where(u < lam_h, 1, np.where(u < lam_h + lam_c, 5, 2))
    cens = rng.uniform(11, 14.75, n)
    dead = T <= cens
    d = pd.DataFrame({"SEQN": np.arange(n), "SDMVSTRA": st.astype(float), "SDMVPSU": psu, "k": k,
                      "cycle": np.where(k == 4, "2005-2006", "2007-2008"), "w4": np.exp(rng.normal(9, 0.7, n)),
                      "age_y": age, "age10": age / 10, "male": male, "race_mexam": (race == 1) * 1.0,
                      "race_othhisp": (race == 2) * 1.0, "race_black": (race == 4) * 1.0,
                      "race_other": (race == 5) * 1.0, "sbp10": sbp / 10,
                      "bp_treat": rng.binomial(1, 0.35, n) * 1.0, "diabetes_any": dm, "dm_zhang": dm,
                      "hba1c": rng.normal(5.7 + dm, 0.6), "smk_former": rng.binomial(1, 0.3, n) * 1.0,
                      "smk_current": rng.binomial(1, 0.18, n) * 1.0, "tc10": rng.normal(20, 4, n),
                      "hdl10": rng.normal(5.3, 1.5, n), "bmi5": rng.normal(5.7, 1.2, n),
                      "egfr10": rng.normal(8.5, 2, n), "hx_cvd_any": rng.binomial(1, 0.12, n) * 1.0,
                      "grad_any_eye": (rng.random(n) < 0.85) * 1, "ret_any": ret, "fan": fan, "avn": avn,
                      "mort_eligible": 1, "death_all": dead * 1.0, "t": np.maximum(np.minimum(T, cens), 0.5 / 12),
                      "UCOD_LEADING": np.where(dead, cause, np.nan)})
    d.loc[d["age_y"] < 40, ["ret_any", "fan", "avn"]] = np.nan
    d["fan_q"] = np.maximum(d["fan"], rng.binomial(1, 0.05, n))
    d["avn_q"] = np.maximum(d["avn"], rng.binomial(1, 0.2, n))
    d["nsign"] = d["ret_any"] + d["fan"] + d["avn"]
    d["nsign_0"] = (d["nsign"] == 0).astype(float).where(d["nsign"].notna())
    d["nsign_1"] = (d["nsign"] == 1).astype(float).where(d["nsign"].notna())
    d["nsign_2p"] = (d["nsign"] >= 2).astype(float).where(d["nsign"].notna())
    for c in ("ret_mild", "ret_modsev", "ret_prolif", "hollenhorst", "rvo", "soft_exudate"):
        d[c] = rng.binomial(1, 0.02, n).astype(float)
    d["death_cvd"] = (dead & np.isin(cause, [1, 5])) * 1.0
    d["death_heart"] = (dead & (cause == 1)) * 1.0
    d["death_cereb"] = (dead & (cause == 5)) * 1.0
    d["death_kidney"] = 0.0
    d["in_opxret"] = (d["age_y"] >= 40) * 1
    d["OPASCST2"] = np.where(d["age_y"] >= 40, 1.0, np.nan)
    d["bp_med_selfreport"] = d["bp_treat"]
    return d


# =================================================================================================================
# Report helpers
# =================================================================================================================
def ci_txt(r, k="hr"):
    if pd.isna(r.get(k, np.nan)):
        return "n.f."
    return f"{r[k]:.2f} [{r['lower']:.2f}, {r['upper']:.2f}]"


def key_table(res: pd.DataFrame) -> pd.DataFrame:
    main = res[res["analysis"] == "main"]
    rows = []
    for (elab, ecols) in EXPOS:
        for c in ecols:
            for _, olab in OUTCOMES:
                row = {"sign": TERM_LAB[c], "outcome": olab}
                for mlab, short in (("M1", "M1"), ("M2", "M2"), ("M3 (no diabetes)", "M3")):
                    x = main[(main["exposure"] == elab) & (main["term"] == c) & (main["outcome"] == olab)
                             & (main["model"] == mlab)]
                    if not len(x):
                        continue
                    r = x.iloc[0]
                    flag = "*" if r.get("flag_lt20_exposed_events") == True else ""  # noqa: E712 (NaN-safe)
                    row[f"{short} HR [95% CI]"] = ci_txt(r) + flag
                    row[f"{short} ev exp/tot"] = f"{int(r['events_exposed'])}/{int(r['events_unweighted'])}"
                rows.append(row)
    return pd.DataFrame(rows)


def key_contrast(con: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in con.iterrows():
        if str(r.get("term", "")).startswith("joint") or pd.isna(r.get("log_ratio", np.nan)):
            continue
        sign = TERM_LAB.get(r["term"], r["term"]) + (" (ret + AVN mutual)" if r["exposure"] == MUTUAL2[0] else "")
        rows.append({"sign": sign, "model": r["model"],
                     "HR heart": f"{r['hr_heart']:.2f}", "HR cerebrovascular": f"{r['hr_cereb']:.2f}",
                     "ratio heart/cereb [95% CI]": f"{r['ratio_heart_over_cereb']:.2f} [{r['lower']:.2f}, "
                                                   f"{r['upper']:.2f}]" + ("*" if r["flag_lt20_exposed_events"] else ""),
                     "p": f"{r['p']:.3g}",
                     "exposed events heart/cereb": f"{int(r['heart_events_exposed'])}/{int(r['cereb_events_exposed'])}"})
    return pd.DataFrame(rows)


def key_sens(res: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity rows for retinopathy and AVN (M2 family)."""
    sel = [("M2", "main", None), ("M2 + prevalent CVD", "sensitivity: + prevalent CVD", None),
           ("M2", "sensitivity: mutual adjustment", MUTUAL2[0]),
           ("M2", "sensitivity: questionable = present", None)]
    rows = []
    for term, qterm in (("ret_any", None), ("avn", "avn_q")):
        for _, olab in OUTCOMES[:4]:
            row = {"sign": TERM_LAB[term], "outcome": olab}
            for mlab, ana, expo in sel:
                t = qterm if ana.endswith("present") else term
                if t is None:
                    continue
                x = res[(res["model"] == mlab) & (res["analysis"] == ana) & (res["term"] == t)
                        & (res["outcome"] == olab)]
                if expo is not None:
                    x = x[x["exposure"] == expo]
                if len(x):
                    r = x.iloc[0]
                    lab = {"main": "M2", "sensitivity: + prevalent CVD": "M2 + prev CVD",
                           "sensitivity: mutual adjustment": "M2 ret + AVN mutual",
                           "sensitivity: questionable = present": "M2 questionable = present"}[ana]
                    row[lab] = ci_txt(r) + ("*" if r.get("flag_lt20_exposed_events") == True else "")  # noqa: E712
            rows.append(row)
    return pd.DataFrame(rows)


def bounds(res: pd.DataFrame) -> list[str]:
    """Null main rows (CI includes 1) for M2, phrased as bounds."""
    out = []
    x = res[(res["analysis"] == "main") & (res["model"] == "M2") & res["hr"].notna()]
    for _, r in x.iterrows():
        if r["lower"] <= 1 <= r["upper"]:
            out.append(f"{TERM_LAB[r['term']]} -> {r['outcome']} death (M2): HR {r['hr']:.2f}; the data are compatible "
                       f"with HR from {r['lower']:.2f} to {r['upper']:.2f} (excludes HR > {r['upper']:.2f})"
                       + (" [< 20 exposed events]" if r["flag_lt20_exposed_events"] else ""))
    return out


# =================================================================================================================
def main():
    import warnings
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
    args = C.parse_args(__doc__.splitlines()[0])
    mode, outdir, token = C.gate(args, TASK)
    rng = np.random.default_rng(args.seed)
    rep = C.Report(TASK, "F1 Stage 0 task m - NHANES 2005-2008 retinal signs and mortality (survey-weighted Cox; "
                   "public-data supporting evidence, exploratory)", mode, args)
    if mode == "simulate":
        d = simulate(rng)
        eqc = None
    else:
        raw = load_raw()
        eqc = worse_eye_qc(raw)
        d = prepare(raw)
    R = domains(d)
    des = design(d)

    res = analyse(d, des, R)
    con = contrasts(d, des, R)
    keyt = key_table(res)
    keyc = key_contrast(con)
    dth = deaths_by_cause(d, R) if mode == "real" else pd.DataFrame()
    prev = prevalence(d, des, R)
    san = sanity(d, des, R)

    # ---------------- report ----------------
    rep.p("- **Label:** public-data supporting evidence for F1 (PLAN 9.3a L4). Exploratory. Qualitative retinal "
          "signs graded on non-mydriatic photographs, not vessel calibre. No HPP data; no liver phenotypes.")
    rep.p("- Rows marked `*` have < 20 deaths among the exposed. `n.f.` = not fitted (< 3 exposed events). "
          "HR [95% CI] with design-based t(df) intervals; M3 = M2 without the diabetes term, in people without "
          "diabetes. In the sign-count model, `n.f.` applies to the whole model when either count term has < 3 "
          "exposed events.")
    rep.p("- Source: raw public NHANES 2005-2008 XPORT files and 2019 LMF read directly (no clean table).")
    rep.h("Key rows: sign x outcome, M1 / M2 / M3 (ev exp/tot = deaths among exposed / all deaths)")
    rep.table("key_hr_table", keyt, None, max_rows=None)
    rep.h("Key rows: heart vs cerebrovascular death contrast (Lunn-McNeil, design-based)")
    rep.table("key_contrast_table", keyc, None, max_rows=None)
    rep.h("Key sensitivity rows: retinopathy and AVN (M2 family)")
    rep.table("key_sensitivity_table", key_sens(res), None, max_rows=None)
    rep.h("Null M2 rows phrased as bounds")
    rep.bullets(bounds(res) or ["(none)"])
    rep.h("Design and sample")
    rep.table("design_summary", pd.DataFrame([{"records_in_design": len(d), "strata": des.n_strata,
                                               "psus": des.n_psu, "design_df": des.df,
                                               "strata_unique_across_cycles": True}]), "Design", max_rows=None)
    if mode == "real":
        rep.table("sample_flow", flow(d, R), "Sample flow (unweighted)", max_rows=None)
        rep.table("worse_eye_qc", eqc, "QC: NHANES worse-eye variable vs max(right, left)", max_rows=None)
        rep.table("deaths_by_cause", dth, "Deaths by underlying cause in domain D (unweighted)", max_rows=None)
        rep.table("covariate_qc", covariate_qc(d, R), "Covariate missingness (retina sample, mortality-eligible)",
                  max_rows=None)
        rep.table("sign_crosstab", sign_crosstab(d, R), "Co-occurrence of the three signs (retina sample, unweighted)",
                  max_rows=None)
    rep.h("Weighted prevalence of retinal signs")
    rep.table("sign_prevalence", prev, "Survey-weighted prevalence (logit-Wald 95% CI)", max_rows=None)
    rep.h("Sanity check against a published-scale figure")
    rep.p(f"- Reference: {PUBLISHED['label']}: {PUBLISHED['estimate']:.3f} [{PUBLISHED['lower']:.3f}, "
          f"{PUBLISHED['upper']:.3f}]; {PUBLISHED['source']}.")
    rep.table("sanity_check", san, None, max_rows=None)
    rep.h("All survey-weighted Cox rows")
    rep.table("svy_cox_all", res, "Survey-weighted cause-specific Cox (design-based SE, t(df) CI)", max_rows=None)
    rep.h("All heart vs cerebrovascular contrast rows")
    rep.table("heart_vs_cereb_contrast", con, "Lunn-McNeil contrast; QC columns compare with separate fits",
              max_rows=None)
    rep.h("Variable dictionary")
    rep.table("variable_dictionary", pd.DataFrame(VAR_DICT, columns=["variable", "source", "definition"]), None,
              max_rows=None)
    rep.h("Caveats")
    rep.bullets([
        "Exposures are qualitative, grader-assessed signs (present / questionable / absent) on two 45-degree "
        "non-mydriatic fields per eye; no vessel calibre (CRAE/CRVE) or AVR is available in NHANES 2005-2008, so these "
        "rows are not the same quantity as HPP retinal calibre.",
        "Worse-eye = either-eye rule; one-eye-only grading is counted as graded (see worse_eye_qc).",
        "Weights: MEC exam weights (WTMEC2YR / 2) without adjustment for retinal-imaging nonresponse or ungradable "
        "images (about 15-20 % of the age >= 40 examined sample have no usable grade).",
        "Definite focal arteriolar narrowing is rare (see prevalence); its rows have few events and wide intervals.",
        "Cause of death = underlying cause recode on the death certificate (UCOD_LEADING); kidney deaths are few.",
        "Retinopathy is graded masked to diabetes; in people without diabetes it mostly reflects mild, non-specific "
        "lesions (hypertensive or other).",
        "Many rows (5 outcomes x 4 exposures x 3 models + sensitivities); no multiplicity correction; exploratory.",
        "PH diagnostics (ph_p_term, ph_p_global) are diagnostic only.",
    ])
    rep.inputs = list(J.INPUTS.values())
    planted = None
    if mode == "simulate":
        x = con[(con["model"] == "M2") & (con["term"] == "avn")].iloc[0]
        y = res[(res["model"] == "M2") & (res["term"] == "ret_any") & (res["outcome"] == "All-cause")
                & (res["analysis"] == "main")].iloc[0]
        planted = C.planted_rows([
            ("log ratio HR heart / HR cereb, AVN (M2)", PLANT["avn_heart"] - PLANT["avn_cereb"], x["log_ratio"],
             x["se_design"]),
            ("all-cause log HR, retinopathy (M2)", PLANT["ret_all"], y["log_hr"], y["se_design"])])
    C.finish(rep, outdir, __file__, planted)


if __name__ == "__main__":
    main()
