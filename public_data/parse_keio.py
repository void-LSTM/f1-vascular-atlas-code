"""Stage 0 parser: Keio check-up dataset (PLoS One 2022, e0269265, S1 data), n = 890 rows.

Binary baPWV (> 1400 cm/s), carotid IMT (IMT+), coronary calcium (CACS), MAFLD/NAFLD, metabolic labs.
Parsing, cleaning and description only (no models). Output: results/stage0/clean/keio_bapwv_imt.parquet + dictionary.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import openpyxl
import pandas as pd

import s0lib as L

XLSX = L.EXT / "open_cohorts" / "oc14_Keio_baPWV_cIMT_NAFLD_pone0269265" / "pone.0269265.s001.xlsx"
NAME = "keio_bapwv_imt"

COLS = [  # NFKC-normalised original, harmonised, meaning, unit, flag
    ("Serial No.", "serial_no", "Serial number (1-889; 259 appears twice)", "", "duplicate serial"),
    ("age", "age_y", "Age", "years", ""),
    ("Male=1", "male", "Sex, 1 = male", "", ""),
    ("Height", "height_cm", "Height; one value 15.7 set missing", "cm", ""),
    ("BW", "weight_kg", "Body weight", "kg", "BW/Height^2 disagrees with BMI in 91 rows"),
    ("BMI", "bmi_kgm2", "Body mass index (as recorded)", "kg/m2", ""),
    ("WC", "waist_cm", "Waist circumference", "cm", ""),
    ("VAT", "vat_cm2", "Visceral adipose tissue area (method not stated; probably CT/BIA)", "cm2", "unit inferred"),
    ("BP(systolic)", "sbp_mmhg", "Systolic BP (check-up; acquisition vs baPWV not stated)", "mmHg", ""),
    ("BP(diastolic)", "dbp_mmhg", "Diastolic BP", "mmHg", ""),
    ("AST", "ast_ul", "AST", "U/L", ""), ("ALT", "alt_ul", "ALT", "U/L", ""), ("LDH", "ldh_ul", "LDH", "U/L", ""),
    ("ALP", "alp_ul", "ALP", "U/L", ""), ("γGTP", "ggt_ul", "GGT", "U/L", ""),
    ("CHE", "che_ul", "Cholinesterase", "U/L", ""), ("TB", "tbil_mgdl", "Total bilirubin", "mg/dL", ""),
    ("DB", "dbil_mgdl", "Direct bilirubin", "mg/dL", ""), ("TP", "total_protein_gdl", "Total protein", "g/dL", ""),
    ("Alb", "albumin_gdl", "Albumin", "g/dL", ""), ("TC", "tc_mgdl", "Total cholesterol", "mg/dL", ""),
    ("HDL-C", "hdl_mgdl", "HDL cholesterol", "mg/dL", ""), ("LDL-C", "ldl_mgdl", "LDL cholesterol", "mg/dL", ""),
    ("TG", "tg_mgdl", "Triglycerides", "mg/dL", ""), ("FBS", "fpg_mgdl", "Fasting blood sugar", "mg/dL", ""),
    ("HbA1c", "hba1c_pct", "HbA1c (NGSP presumed)", "%", ""), ("HOMA-R", "homa_ir", "HOMA-IR", "", ""),
    ("HOMA-β", "homa_beta", "HOMA-beta", "%", ""), ("UA", "uric_acid_mgdl", "Uric acid", "mg/dL", ""),
    ("BUN", "bun_mgdl", "Blood urea nitrogen", "mg/dL", ""), ("Cr", "creatinine_mgdl", "Creatinine", "mg/dL", ""),
    ("eGFR", "egfr", "Estimated GFR", "mL/min/1.73m2", ""),
    ("elevated CRP", "crp_elevated", "Elevated CRP (threshold not stated)", "", "threshold unknown"),
    ("CACS>0", "cacs_gt0", "Coronary artery calcium score > 0", "", ""),
    ("CACS>100", "cacs_gt100", "Coronary artery calcium score > 100", "", ""),
    ("PWV>1400", "bapwv_gt1400", "baPWV > 1400 cm/s (binary only; side rule not stated)", "", "binary only"),
    ("IMT+", "imt_positive", "Carotid IMT positive (threshold/plaque definition not stated)", "",
     "definition unknown"),
    ("MAFLD", "mafld", "MAFLD", "", ""), ("Obesity-MAFLD", "mafld_obesity", "MAFLD, overweight/obesity criterion",
                                         "", ""),
    ("Lean-MAFLD", "mafld_lean", "MAFLD, lean with metabolic dysregulation", "", ""),
    ("DM-MAFLD", "mafld_dm", "MAFLD, diabetes criterion", "", ""), ("NAFLD", "nafld", "NAFLD", "", ""),
    ("HTN+", "hypertension", "Hypertension", "", ""), ("DM+", "diabetes", "Diabetes", "", ""),
    ("HL+", "hyperlipidaemia", "Hyperlipidaemia", "", ""), ("Non-smoker", "non_smoker", "Non-smoker", "", ""),
    ("Non-drinker", "non_drinker", "Non-drinker", "", ""), ("Exercise", "exercise", "Exercise habit", "", ""),
]


def main() -> None:
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    sheet = wb.sheetnames[0]
    rows = [list(r) for r in wb[sheet].iter_rows(values_only=True)]
    wb.close()
    hdr_raw = rows[0]
    keep = [i for i, h in enumerate(hdr_raw) if h is not None]
    extra_nonempty = sum(1 for r in rows[1:] for i, v in enumerate(r) if i not in keep and v is not None)
    hdr = [unicodedata.normalize("NFKC", hdr_raw[i]).strip() for i in keep]
    data = [[r[i] for i in keep] for r in rows[1:] if any(v is not None for v in r)]
    raw = pd.DataFrame(data, columns=hdr)
    assert [c for c, *_ in COLS] == hdr, hdr
    df = raw.copy()
    df.insert(0, "row_id", np.arange(1, len(df) + 1))
    for o, h, *_ in COLS:
        df[h] = pd.to_numeric(raw[o], errors="coerce")
    dup_serial = df["serial_no"].duplicated(keep=False)
    df["qc_duplicate_serial"] = dup_serial.astype(int)
    dd = df[dup_serial]
    diff_cols = [c for c in hdr if dd.groupby("serial_no")[c].nunique(dropna=False).max() > 1]
    bad_h = df["height_cm"] < 100
    df["qc_height_implausible"] = bad_h.astype(int)
    df.loc[bad_h, "height_cm"] = np.nan
    df["bmi_kgm2_calc"] = df["weight_kg"] / (df["height_cm"] / 100) ** 2
    df["qc_bmi_inconsistent"] = ((df["bmi_kgm2_calc"] - df["bmi_kgm2"]).abs() > 1).astype(int)
    df["map_mmhg_formula"] = df["dbp_mmhg"] + (df["sbp_mmhg"] - df["dbp_mmhg"]) / 3

    avail = pd.DataFrame([{
        "rows": len(df), "unique serial numbers": int(df["serial_no"].nunique()),
        "PWV>1400 known": int(df.bapwv_gt1400.notna().sum()), "PWV>1400 = 1": int(df.bapwv_gt1400.sum()),
        "IMT+ known": int(df.imt_positive.notna().sum()), "IMT+ = 1": int(df.imt_positive.sum()),
        "CACS>0 = 1": int(df.cacs_gt0.sum()), "MAFLD = 1": int(df.mafld.sum()),
        "all of PWV/IMT/CACS known": int(df[["bapwv_gt1400", "imt_positive", "cacs_gt0"]].notna().all(axis=1).sum()),
    }])
    sub = pd.DataFrame([{"MAFLD": int(df.mafld.sum()),
                         "any MAFLD subtype": int(df[["mafld_obesity", "mafld_lean", "mafld_dm"]].max(axis=1).sum()),
                         "MAFLD=1 without subtype": int(((df.mafld == 1) & (df[["mafld_obesity", "mafld_lean",
                                                                              "mafld_dm"]].max(axis=1) == 0)).sum()),
                         "CACS>100 but CACS>0 = 0": int(((df.cacs_gt100 == 1) & (df.cacs_gt0 == 0)).sum())}])
    notes = [
        f"One sheet (`{sheet}`), header row 1, {len(df)} data rows, 48 named columns (full-width characters in some "
        f"headers normalised with NFKC, e.g. 'ＢＭＩ' -> 'BMI'); 12 unnamed trailing columns are empty "
        f"(non-empty cells outside named columns: {extra_nonempty}).",
        f"Serial No(s). {sorted(dd['serial_no'].unique().astype(int).tolist())} appear twice: the two rows are "
        f"identical except {diff_cols} (conflicting liver classification). Both rows flagged `qc_duplicate_serial`; "
        f"unique serial numbers = {df['serial_no'].nunique()}. Decide in the analysis plan which row (if any) to keep.",
        "**Vascular measures are binary only**: `PWV>1400` (baPWV > 1400 cm/s; whether max or mean of the two sides "
        "is not stated), `IMT+` (carotid IMT abnormal; threshold/plaque rule not stated), `CACS>0`, `CACS>100`. No "
        "continuous baPWV, IMT, ABI or ankle pressures. Same-person PWV/IMT driver comparisons are therefore limited "
        "to binary outcomes.",
        "BW/Height^2 differs from the recorded BMI by > 1 kg/m2 in "
        f"{int(df.qc_bmi_inconsistent.sum())} rows (flag `qc_bmi_inconsistent`); the recorded BMI agrees with WC/VAT "
        "patterns, so BW (or Height) may be misaligned in those rows. Use recorded BMI; treat BW with caution. "
        "Height 15.7 cm set missing (probably 157).",
        "Definitions of HTN+, DM+, HL+, elevated CRP, Exercise, Non-drinker are not in the file (from the article). "
        "Units are standard Japanese check-up units (mg/dL, U/L, g/dL); VAT unit cm2 and method inferred.",
        "BP is a check-up BP; whether it was taken at the baPWV acquisition is not stated.",
    ]
    vm = {"row_id": {"kind": "derived", "meaning": "Row order in sheet"}}
    for o, h, m, u, f in COLS:
        vm[o] = {"kind": "original", "meaning": m, "unit": u}
        vm[h] = {"kind": "harmonised", "meaning": m, "unit": u, "flag": f}
    vm.update({"qc_duplicate_serial": {"kind": "qc", "meaning": "Serial number occurs twice (both rows flagged)"},
               "qc_height_implausible": {"kind": "qc", "meaning": "Height < 100 cm (set missing)"},
               "bmi_kgm2_calc": {"kind": "derived", "meaning": "weight_kg / (height_cm/100)^2", "unit": "kg/m2"},
               "qc_bmi_inconsistent": {"kind": "qc", "meaning": "|BW/Height^2 - BMI| > 1 kg/m2"},
               "map_mmhg_formula": {"kind": "derived", "meaning": "DBP + (SBP - DBP)/3", "unit": "mmHg"}})
    L.write_parquet(df, NAME)
    L.write_dictionary(df, NAME, "Keio check-up dataset (PLoS One e0269265 S1), 890 rows / 889 persons", [XLSX],
                       "parse_keio.py", vm, notes=notes,
                       extra_sections=[("Availability (counts)", L.md_table(avail)),
                                       ("Internal consistency (counts)", L.md_table(sub))])
    print(NAME, df.shape)


if __name__ == "__main__":
    main()
