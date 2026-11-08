"""Stage 0 parser: IMPACT-ABI cohort (PLoS One 2017, e0177609, S2 data), n = 3,056.

Parsing, cleaning and description only (no models). Output: results/stage0/clean/impact_abi.parquet and
results/stage0/dictionaries/impact_abi.md.
"""
from __future__ import annotations

import re

import numpy as np
import openpyxl
import pandas as pd

import s0lib as L

XLSX = L.EXT / "open_cohorts" / "oc02_IMPACT_ABI_pone0177609" / "pone.0177609.s002.xlsx"
NAME = "impact_abi"

# (original column, harmonised name, meaning, unit, flag)
COLMAP = [
    ("female", "female", "Sex, 1 = female (header 'female'; heights confirm)", "", ""),
    ("Age", "age_y", "Age at ABI/baPWV examination", "years", "21 participants aged 13-19"),
    ("HIght(m)", "height_m", "Height; values > 3 treated as cm and divided by 100", "m", ""),
    ("Weight(kg)", "weight_kg", "Weight; < 25 kg set missing", "kg", ""),
    ("BMI", "bmi_orig", "BMI as in sheet (Excel formula D/C^2)", "kg/m2", "formula; wrong where height in cm"),
    ("rt ABI", "abi_r", "Right ankle-brachial index; '-' and > 3 set missing", "ratio", ""),
    ("lt ABI", "abi_l", "Left ankle-brachial index; '-' and > 3 set missing", "ratio", ""),
    ("ABI 0.9,0.9-1.0,1.0-1.4,1.4", "abi_cat", "ABI category: 1 < 0.9, 2 0.9-<1.0, 3 1.0-<1.4 (4 >= 1.4 absent)", "",
     "codes inferred from header + data"),
    ("ABI<0.9", "abi_lt_0_9", "ABI < 0.9 (either leg)", "", ""),
    ("右PWV", "bapwv_r_cms", "Right baPWV ('右' = right); '-' and < 400 set missing", "cm/s",
     "PWV type inferred (baPWV)"),
    ("左PWV", "bapwv_l_cms", "Left baPWV ('左' = left); '-' and < 400 set missing", "cm/s",
     "PWV type inferred (baPWV)"),
    ("HTN", "htn", "Hypertension (history)", "", ""),
    ("HL", "hyperlipidaemia", "Hyperlipidaemia (history; 'HL' inferred)", "", "abbreviation inferred"),
    ("DM", "diabetes", "Diabetes mellitus", "", ""),
    ("insulin", "insulin", "Insulin use", "", ""),
    ("stroke", "hx_stroke", "History of stroke (first 'stroke' column, baseline block)", "", ""),
    ("intracranial bleeding", "hx_intracranial_bleeding", "History of intracranial bleeding", "", ""),
    ("OMI", "hx_old_mi", "Old myocardial infarction (history)", "", ""),
    ("HD", "hemodialysis", "Haemodialysis (inferred: median creatinine 5.4 mg/dL, eGFR 8 when HD = 1)", "",
     "meaning inferred"),
    ("Previous HF", "hx_hf", "Previous heart failure", "", ""),
    ("smoking", "smoking", "Smoking (current vs ever not stated); '0\\1' set missing", "", "current/ever unclear"),
    ("AF", "af", "Atrial fibrillation", "", ""),
    ("PAD", "pad", "Peripheral artery disease (history); free-text entry set missing", "", ""),
    ("LVEF", "lvef_pct", "Left ventricular ejection fraction", "%", ""),
    ("aspirin", "aspirin", "Aspirin; '不明' (unknown) set missing", "", ""),
    ("thienopyridine", "thienopyridine", "Thienopyridine", "", ""),
    ("cilostazol", "cilostazol", "Cilostazol", "", ""),
    ("Statin", "statin", "Statin", "", ""),
    ("ACEIs", "acei", "ACE inhibitor", "", ""),
    ("ARBs", "arb", "Angiotensin receptor blocker", "", ""),
    ("Beta blocker", "beta_blocker", "Beta blocker", "", ""),
    ("Ca-antagonist", "ccb", "Calcium-channel blocker", "", ""),
    ("Wf", "warfarin", "Warfarin ('Wf' inferred); 'NOAC' entry set missing (see noac_text)", "",
     "abbreviation inferred"),
    ("Hb (g/dl)", "hb_gdl", "Haemoglobin; outside 3-25 set missing", "g/dL", ""),
    ("UA", "uric_acid_mgdl", "Uric acid; '9,4' -> 9.4, '6..2' -> 6.2", "mg/dL", "unit inferred"),
    ("Cre", "creatinine_mgdl", "Serum creatinine; > 20 set missing", "mg/dL", "unit inferred"),
    ("eGFR", "egfr", "Estimated GFR (equation not stated; probably Japanese equation)", "mL/min/1.73m2",
     "36 % missing"),
    ("BNP", "bnp_pgml", "B-type natriuretic peptide; '<5' censored (see bnp_censored_lt5); '.42.2' -> 42.2",
     "pg/mL", "unit inferred"),
    ("death", "death", "All-cause death", "", ""),
    ("death FU", "death_fu_days", "Follow-up time for death", "days", "unit inferred (max 3,576)"),
    ("cardiovascular death", "cv_death", "Cardiovascular death", "", ""),
    ("CV death FU", "cv_death_fu_days", "Follow-up time for CV death (identical to death FU)", "days", ""),
    ("MI", "mi", "Incident myocardial infarction", "", ""),
    ("MI FU", "mi_fu_days", "Time to MI or censoring", "days", ""),
    ("stroke.1", "stroke", "Incident stroke (second 'stroke' column, outcome block)", "", ""),
    ("Stroke FU", "stroke_fu_days", "Time to stroke or censoring", "days", ""),
    ("major bleeding", "major_bleeding", "Major bleeding", "", ""),
    ("major bleeding FU", "major_bleeding_fu_days", "Time to major bleeding or censoring", "days", ""),
    ("HF hospitalization", "hf_hosp", "Hospitalisation for heart failure", "", ""),
    ("HF FU", "hf_hosp_fu_days", "Time to HF hospitalisation or censoring", "days", ""),
    ("MACE(death,MI,stroke)", "mace_death_mi_stroke", "MACE = all-cause death, MI or stroke", "", ""),
    ("MACE(death,MI,stroke) FU", "mace_death_mi_stroke_fu_days", "Time to MACE (all-cause death version)", "days",
     ""),
    ("MACE(CVdeath,MI,stroke)", "mace_cvdeath_mi_stroke", "MACE = CV death, MI or stroke", "", ""),
    ("MACE(CVdeath,MI,stroke) FU", "mace_cvdeath_mi_stroke_fu_days",
     "Time to MACE (CV-death version; identical to all-cause MACE FU)", "days", ""),
]
EVENTS = [("All-cause death", "death", "death_fu_days"), ("CV death", "cv_death", "cv_death_fu_days"),
          ("MI", "mi", "mi_fu_days"), ("Stroke", "stroke", "stroke_fu_days"),
          ("Major bleeding", "major_bleeding", "major_bleeding_fu_days"),
          ("HF hospitalisation", "hf_hosp", "hf_hosp_fu_days"),
          ("MACE (death, MI, stroke)", "mace_death_mi_stroke", "mace_death_mi_stroke_fu_days"),
          ("MACE (CV death, MI, stroke)", "mace_cvdeath_mi_stroke", "mace_cvdeath_mi_stroke_fu_days")]


def to_float_clean(v):
    """Return (float or NaN, note) for messy cells."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan, None
    if isinstance(v, (int, float)):
        return float(v), None
    s = str(v).strip()
    if s in ("-", "記載なし", "記録なし", "未", "不明", "0\\1"):
        return np.nan, f"text '{s}'"
    if re.fullmatch(r"\d+,\d+", s):
        return float(s.replace(",", ".")), f"'{s}' read as decimal comma"
    if re.fullmatch(r"\d+\.\.\d+", s):
        return float(s.replace("..", ".")), f"'{s}' read with single decimal point"
    if re.fullmatch(r"\.\d+\.\d+", s):
        return float(s[1:]), f"'{s}' read without leading point"
    if re.fullmatch(r"<\s*\d+(\.\d+)?", s):
        return np.nan, f"censored '{s}'"
    try:
        return float(s), None
    except ValueError:
        return np.nan, f"text '{s}'"


def main() -> None:
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    sheet = wb.sheetnames[0]
    hdr = next(wb[sheet].iter_rows(min_row=1, max_row=1, values_only=True))
    wb.close()
    raw = pd.read_excel(XLSX, sheet_name=sheet, header=0, engine="openpyxl")
    assert [c for c, *_ in COLMAP] == list(raw.columns), list(raw.columns)
    n = len(raw)
    df = raw.copy()
    df.insert(0, "row_id", np.arange(1, n + 1))
    qc = {i: [] for i in df.index}

    for orig, new, *_ in COLMAP:
        vals, notes_ = zip(*[to_float_clean(v) for v in raw[orig]])
        df[new] = np.array(vals, dtype=float)
        for i, nt in zip(df.index, notes_):
            if nt and not nt.startswith("censored"):
                qc[i].append(f"{orig}: {nt}")

    def setna(mask, col, why):
        for i in df.index[mask.fillna(False)]:
            qc[i].append(f"{col}={df.at[i, col]:g} {why}")
        df.loc[mask.fillna(False), col] = np.nan

    cm = df["height_m"] > 3
    for i in df.index[cm]:
        qc[i].append(f"height {df.at[i, 'height_m']:g} read as cm")
    df.loc[cm, "height_m"] = df.loc[cm, "height_m"] / 100
    setna(df["weight_kg"] < 25, "weight_kg", "implausible, set missing")
    for c in ("abi_r", "abi_l"):
        setna(df[c] > 3, c, "implausible (probably missing decimal point), set missing")
    for c in ("bapwv_r_cms", "bapwv_l_cms"):
        setna(df[c] < 400, c, "implausible low, set missing")
    setna((df["hb_gdl"] < 3) | (df["hb_gdl"] > 25), "hb_gdl", "implausible, set missing")
    setna(df["creatinine_mgdl"] > 20, "creatinine_mgdl", "implausible for mg/dL, set missing")
    df["noac_text"] = (raw["Wf"].astype(str) == "NOAC").astype(int)
    df["bnp_censored_lt5"] = raw["BNP"].astype(str).str.strip().eq("<5").astype(int)
    pad_text = raw["PAD"].map(lambda v: isinstance(v, str))
    for i in df.index[pad_text]:
        qc[i].append(f"PAD free text '{raw.at[i, 'PAD']}' (right brachiocephalic artery / renal occlusion?) set missing")

    # derived
    df["bmi_kgm2_calc"] = df["weight_kg"] / df["height_m"] ** 2
    df["qc_height_lt_1_3"] = (df["height_m"] < 1.3).astype(int)
    df["abi_min"] = df[["abi_r", "abi_l"]].min(axis=1)
    df["abi_min_lt_0_9"] = np.where(df["abi_min"].notna(), (df["abi_min"] < 0.9).astype(float), np.nan)
    df["qc_abi_cat_vs_values_mismatch"] = (df["abi_min"].notna() & (
        ((df["abi_min"] < 0.9) & (df["abi_cat"] != 1)) | ((df["abi_min"] >= 0.9) & (df["abi_min"] < 1.0)
                                                         & (df["abi_cat"] != 2)) |
        ((df["abi_min"] >= 1.0) & (df["abi_cat"] != 3)))).astype(int)
    df["bapwv_mean_cms"] = df[["bapwv_r_cms", "bapwv_l_cms"]].mean(axis=1, skipna=False)
    df["bapwv_max_cms"] = df[["bapwv_r_cms", "bapwv_l_cms"]].max(axis=1, skipna=False)
    df["qc_bapwv_gt_4000"] = ((df["bapwv_r_cms"] > 4000) | (df["bapwv_l_cms"] > 4000)).astype(int)
    for _, ev, tm in EVENTS:
        df[tm.replace("_days", "_years")] = df[tm] / 365.25
    df["qc_fu_exceeds_death_fu"] = ((df["mi_fu_days"] > df["death_fu_days"]) | (df["stroke_fu_days"] > df[
        "death_fu_days"]) | (df["hf_hosp_fu_days"] > df["death_fu_days"])).astype(int)
    df["qc_notes"] = [("; ".join(qc[i]) if qc[i] else np.nan) for i in df.index]

    # ---------------- descriptives ----------------
    ev = L.events_table(df, EVENTS, "days")
    ev_y = L.events_table(df, [(a, b, c.replace("_days", "_years")) for a, b, c in EVENTS], "years")
    mace_chk = {
        "MACE(death,MI,stroke) == max(death, MI, stroke)":
            bool((df["mace_death_mi_stroke"] == df[["death", "mi", "stroke"]].max(axis=1)).all()),
        "MACE(CVdeath,MI,stroke) == max(CV death, MI, stroke)":
            bool((df["mace_cvdeath_mi_stroke"] == df[["cv_death", "mi", "stroke"]].max(axis=1)).all()),
        "CV death only among deaths": bool(((df["cv_death"] == 1) <= (df["death"] == 1)).all()),
        "death FU == CV death FU": bool((df["death_fu_days"] == df["cv_death_fu_days"]).all()),
        "MACE FU (all-cause) == MACE FU (CV)": bool(
            (df["mace_death_mi_stroke_fu_days"] == df["mace_cvdeath_mi_stroke_fu_days"]).all()),
        "rows where MI/stroke/HF FU exceeds death FU": int(df["qc_fu_exceeds_death_fu"].sum()),
        "max excess of MI FU over death FU (days)": float((df["mi_fu_days"] - df["death_fu_days"]).max()),
        "rows with death FU = 0": int((df["death_fu_days"] == 0).sum()),
    }
    excess = (df["mi_fu_days"] - df["death_fu_days"])
    exc_desc = excess[excess > 0].describe()
    qc_rows = df[df["qc_notes"].notna()][["row_id", "qc_notes"]]
    abi_avail = pd.DataFrame([{
        "N": n, "ABI both legs": int(df[["abi_r", "abi_l"]].notna().all(axis=1).sum()),
        "baPWV both legs": int(df[["bapwv_r_cms", "bapwv_l_cms"]].notna().all(axis=1).sum()),
        "ABI min < 0.9": int((df["abi_min"] < 0.9).sum()), "ABI min 0.9-<1.0": int(((df["abi_min"] >= 0.9) & (
            df["abi_min"] < 1.0)).sum()), "ABI > 1.4 either leg": int((df[["abi_r", "abi_l"]].max(axis=1) > 1.4).sum()),
        "baPWV > 4000 either leg": int(df["qc_bapwv_gt_4000"].sum()), "haemodialysis": int(df["hemodialysis"].sum()),
    }])

    notes = [
        f"One sheet (`{sheet}`, Japanese 'data entry'), header in row 1, {n} rows, no participant ID (`row_id` = row "
        "order). Two empty trailing columns are ignored. The column name `stroke` occurs twice: the first is a "
        "baseline history variable (in the comorbidity block), the second (`stroke.1` after reading) is the incident "
        "stroke outcome (it is followed by `Stroke FU`).",
        "No codebook in the file. Decoding: `female` 1 = female (median height 1.51 vs 1.65 m); `HD` = haemodialysis "
        "(creatinine/eGFR pattern); `OMI` = old MI; `HL` = hyperlipidaemia; `Wf` = warfarin; `ABI 0.9,0.9-1.0,1.0-1.4,"
        "1.4` = category of the lower-leg ABI (1 < 0.9, 2 0.9-0.99, 3 1.0-1.39; the >= 1.4 class is empty, so "
        "ABI >= 1.4 was presumably excluded). `右PWV`/`左PWV` = right/left PWV in cm/s; with simultaneous ABI this "
        "is **baPWV** (oscillometric device; inferred). These decodings are consistent with the data but should be "
        "checked against the article text (Miura et al., PLoS One 2017;12:e0177609).",
        "Follow-up columns are in **days** (inferred: max 3,585 = 9.8 years). `death FU` and `CV death FU` are "
        "identical; the two MACE definitions share one FU column. In "
        f"{mace_chk['rows where MI/stroke/HF FU exceeds death FU']} rows the MI/stroke/HF FU exceeds the death FU "
        f"(excess median {exc_desc['50%']:.0f} days, max {exc_desc['max']:.0f}): the event-specific FU columns were "
        "not truncated at death or were updated separately. Analyses of non-fatal events must decide on censoring "
        "at death.",
        "Event indicator checks: " + "; ".join(f"{k}: {v}" for k, v in mace_chk.items()) + ".",
        "Rows are sorted by decreasing death FU (file order). Short follow-up: death FU <= 30 days in "
        f"{int((df['death_fu_days'] <= 30).sum())} rows, <= 365 days in {int((df['death_fu_days'] <= 365).sum())}. "
        f"HF hospitalisation has {int(((df['hf_hosp'] == 1) & (df['hf_hosp_fu_days'] == 0)).sum())} events at day 0 "
        "(probably the index admission); an at-risk start rule (e.g. exclude day-0 events) must be pre-specified.",
        "Messy cells (all originals kept as text in the original-name columns): heights entered in cm (162, 165), "
        "weight 6 and 1.02 kg, ABI 112, PWV 111-380 cm/s, Hb 1.4/41.5/45.2/135, creatinine 25-392, "
        "'記載なし'/'記録なし'/'未' (not recorded), '-' (not measured), '0\\1', '不明' (unknown), 'NOAC' in the warfarin "
        "column, '9,4'/'6..2' (uric acid), '<5' and '.42.2' (BNP), one PAD free-text entry. Rules applied are in "
        "`qc_notes` per row. baPWV > 4000 cm/s (n = "
        f"{int(df['qc_bapwv_gt_4000'].sum())} participants) is kept but flagged (`qc_bapwv_gt_4000`).",
        "baPWV is unreliable when ABI < 0.9 (pressure-wave damping); the paper population includes many patients "
        "with PAD and haemodialysis. ABI and baPWV come from the same four-limb acquisition (shared ankle "
        "waveforms/pressures). No BP at the acquisition is provided, so acquisition-pressure coupling cannot be "
        "studied here.",
        "eGFR is missing in 36 % of rows (creatinine is nearly complete): eGFR could be recomputed from creatinine, "
        "age and sex with the Japanese equation if the analysis plan needs it.",
    ]
    extra = [
        ("Availability and ABI/baPWV classes (counts)", L.md_table(abi_avail)),
        ("Events and follow-up (days)", L.md_table(ev)),
        ("Events and follow-up (years)", L.md_table(ev_y)),
        ("Rows with QC notes", L.md_table(qc_rows)),
    ]
    varmeta = {"row_id": {"kind": "derived", "meaning": "Row order in sheet (no ID in file)"}}
    for orig, new, meaning, unit, flag in COLMAP:
        varmeta[orig] = {"kind": "original", "meaning": f"Original column (raw cell values as text if mixed): {meaning}",
                         "unit": unit}
        varmeta[new] = {"kind": "harmonised", "meaning": meaning, "unit": unit, "flag": flag}
    varmeta.update({
        "noac_text": {"kind": "derived", "meaning": "'NOAC' written in the warfarin column"},
        "bnp_censored_lt5": {"kind": "derived", "meaning": "BNP recorded as '<5' (numeric set missing)"},
        "bmi_kgm2_calc": {"kind": "derived", "meaning": "weight_kg / height_m^2 after cleaning", "unit": "kg/m2"},
        "qc_height_lt_1_3": {"kind": "qc", "meaning": "Height < 1.3 m"},
        "abi_min": {"kind": "derived", "meaning": "Lower of right/left ABI", "unit": "ratio"},
        "abi_min_lt_0_9": {"kind": "derived", "meaning": "abi_min < 0.9 (NA if both missing)"},
        "qc_abi_cat_vs_values_mismatch": {"kind": "qc", "meaning": "ABI category disagrees with recorded ABI values"},
        "bapwv_mean_cms": {"kind": "derived", "meaning": "Mean of right/left baPWV (both required)", "unit": "cm/s"},
        "bapwv_max_cms": {"kind": "derived", "meaning": "Max of right/left baPWV (both required)", "unit": "cm/s"},
        "qc_bapwv_gt_4000": {"kind": "qc", "meaning": "baPWV > 4000 cm/s in either leg (kept)"},
        "qc_fu_exceeds_death_fu": {"kind": "qc", "meaning": "MI, stroke or HF FU longer than death FU"},
        "qc_notes": {"kind": "qc", "meaning": "Cleaning notes for the row"},
    })
    for lab, ev_, tm in EVENTS:
        varmeta[tm.replace("_days", "_years")] = {"kind": "derived", "meaning": f"{tm} / 365.25", "unit": "years"}
    L.write_parquet(df, NAME)
    L.write_dictionary(df, NAME, "IMPACT-ABI cohort (PLoS One 2017; e0177609 S2 data), n = 3,056", [XLSX],
                       "parse_impact_abi.py", varmeta, notes=notes, extra_sections=extra)
    print(NAME, df.shape)
    print(ev.to_string())


if __name__ == "__main__":
    main()
