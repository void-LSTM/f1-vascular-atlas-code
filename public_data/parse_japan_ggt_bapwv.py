"""Stage 0 parser: Japanese health check-up GGT-baPWV dataset (Dryad doi:10.5061/dryad.m484p; BMJ Open 2014), n = 912.

Parsing, cleaning and description only (no models). Reads the legacy .xls via xlrd (.venv-readers subprocess).
Output: results/stage0/clean/japan_ggt_bapwv.parquet and results/stage0/dictionaries/japan_ggt_bapwv.md.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import s0lib as L

SRC = L.EXT / "open_cohorts" / "dryad_m484p_ggt_bapwv_fatty_liver"
XLS = SRC / "data bmjopen dryad.xls"
META = SRC / "dryad_m484p_ggt_bapwv_fatty_liver_public_manifest.source_metadata.json"
NAME = "japan_ggt_bapwv"

COLS = [  # original, harmonised, meaning, unit, flag
    ("ID", "participant_id", "Participant ID (1-912)", "", ""),
    ("Sex", "sex_raw", "Sex: 1 = male, 2 = female (menopause recorded only for 2)", "", "coding inferred"),
    ("Age", "age_y", "Age", "years", ""),
    ("BMI", "bmi_kgm2", "Body mass index", "kg/m2", ""),
    ("sBP", "sbp_mmhg", "Systolic BP (x.5 values: mean of 2 readings); device/timing vs baPWV not stated", "mmHg",
     "acquisition not stated"),
    ("dBP", "dbp_mmhg", "Diastolic BP (mean of 2 readings)", "mmHg", "acquisition not stated"),
    ("AST", "ast_ul", "AST", "U/L", ""),
    ("ALT", "alt_ul", "ALT", "U/L", ""),
    ("γGTP", "ggt_ul", "Gamma-glutamyl transpeptidase", "U/L", ""),
    ("log2ggt", "log2_ggt", "log2(GGT) (verified = log2 of γGTP)", "log2 U/L", ""),
    ("FBS", "fpg_mgdl", "Fasting blood sugar", "mg/dL", ""),
    ("Uric acid", "uric_acid_mgdl", "Uric acid", "mg/dL", ""),
    ("TC", "tc_mgdl", "Total cholesterol", "mg/dL", ""),
    ("TG", "tg_mgdl", "Triglycerides", "mg/dL", ""),
    ("HDL", "hdl_mgdl", "HDL cholesterol", "mg/dL", ""),
    ("LDL3", "ldl_mgdl", "LDL cholesterol ('LDL3'; not Friedewald - median 8 mg/dL below TC-HDL-TG/5; probably "
     "direct assay)", "mg/dL", "method unknown"),
    ("current smoker", "current_smoker", "Current smoker", "", ""),
    ("exsmoker", "ever_smoker", "'exsmoker' column = EVER smoker (all current smokers have 1)", "", "name misleading"),
    ("Alcohol", "alcohol_g_week_probable", "Alcohol intake (values 0, 1, 12, 22, 36, 126, 154 ...: probably g "
     "ethanol/week, 154 = 7 x 22 g 'go' of sake)", "g/week (?)", "unit inferred"),
    ("exercise", "exercise", "Regular exercise (definition not stated)", "", ""),
    ("fatty liver", "fatty_liver_us", "Fatty liver by abdominal ultrasonography", "", ""),
    ("eGFR", "egfr", "Estimated GFR (equation not stated; probably Japanese equation)", "mL/min/1.73m2", ""),
    ("posr menopausal state", "postmenopausal_raw", "Menopausal state, women only: 1 (mean age 45) = pre, 2 (mean "
     "age 58) = post", "", "coding inferred"),
    ("ABImax", "abi_max", "Higher of left/right ABI; values > 3 (PWV-like numbers) set missing", "ratio",
     "36 entries > 3 (35 PWV-like, one 13.05)"),
    ("PWVmax", "bapwv_max_cms", "Higher of left/right baPWV", "cm/s", ""),
]


def main() -> None:
    meta = json.loads(META.read_text())
    wb = L.read_xls(XLS)
    assert list(wb) == ["Sheet1"], list(wb)
    raw = L.xls_sheet_to_frame(wb["Sheet1"])
    assert [c for c, *_ in COLS] == list(raw.columns), list(raw.columns)
    df = raw.copy()
    for o, h, *_ in COLS:
        df[h] = pd.to_numeric(raw[o].replace("", np.nan), errors="coerce")
    df["participant_id"] = df["participant_id"].astype(int)
    df["male"] = (df["sex_raw"] == 1).astype(int)
    df["former_smoker"] = ((df["ever_smoker"] == 1) & (df["current_smoker"] == 0)).astype(int)
    df["postmenopausal"] = df["postmenopausal_raw"].map({1: 0, 2: 1})
    bad_abi = df["abi_max"] > 3
    df["qc_abi_max_implausible"] = bad_abi.astype(int)
    df["abi_max_orig_value"] = df["abi_max"]
    df.loc[bad_abi, "abi_max"] = np.nan
    df["map_mmhg_formula"] = df["dbp_mmhg"] + (df["sbp_mmhg"] - df["dbp_mmhg"]) / 3
    df["bapwv_max_ms"] = df["bapwv_max_cms"] / 100
    df["ldl_friedewald_mgdl"] = np.where(df["tg_mgdl"] < 400, df["tc_mgdl"] - df["hdl_mgdl"] - df["tg_mgdl"] / 5,
                                         np.nan)

    smk = pd.crosstab(df["current_smoker"], df["ever_smoker"]).reset_index()
    smk.columns = ["current_smoker \\ exsmoker"] + [str(c) for c in smk.columns[1:]]
    meno = df.groupby("postmenopausal_raw").agg(N=("age_y", "size"), age_mean=("age_y", "mean"),
                                                age_min=("age_y", "min"), age_max=("age_y", "max")).reset_index()
    avail = pd.DataFrame([{"N": len(df), "men": int(df.male.sum()), "women": int((1 - df.male).sum()),
                           "baPWV": int(df.bapwv_max_cms.notna().sum()), "ABImax valid": int(df.abi_max.notna().sum()),
                           "ABImax < 0.9": int((df.abi_max < 0.9).sum()), "fatty liver": int(df.fatty_liver_us.sum()),
                           "SBP": int(df.sbp_mmhg.notna().sum())}])
    notes = [
        "One sheet, header row 1, 912 rows x 25 columns, one row per participant; no codebook in the file. The BMJ "
        "Open abstract (metadata JSON) reports 912 participants aged 24-84 (the Dryad file description says 846).",
        "Sex: 1 = male (n = 592), 2 = female (n = 320); inferred because the menopause variable is filled only for "
        "Sex = 2. Menopause: 1 = premenopausal, 2 = postmenopausal (inferred from mean age 45 vs 58; the abstract "
        "defines postmenopause as > 1 year after the last menses).",
        "`exsmoker` is 1 for every current smoker, so it means **ever smoker**; `former_smoker` = exsmoker & not "
        "current.",
        "**`ABImax` is the higher of the two legs**, not the lower: it is insensitive to unilateral PAD and cannot "
        f"define ABI < 0.9 the usual way. {int(bad_abi.sum())} entries are > 3 (IDs "
        f"{sorted(df.loc[bad_abi, 'participant_id'].tolist())}): all but one (13.05) are PWV-like numbers close to "
        "`PWVmax` (probably the other-side baPWV pasted into the wrong column); set missing, original kept in "
        "`abi_max_orig_value`.",
        "`PWVmax` = higher of left/right baPWV (cm/s), from an automatic waveform analyser (form PWV/ABI type "
        "device; inferred from the abstract). Only the max is given: no side-specific values, no ankle pressures.",
        "BP: `sBP`/`dBP` with .5 values are means of two readings; whether they come from the PWV device (same "
        "acquisition) or a separate check-up measurement is not stated (BMJ Open methods should be checked).",
        "Units follow Japanese check-up conventions (mg/dL for glucose, lipids, uric acid; U/L for enzymes). "
        "`Alcohol` unit is not stated (probably g ethanol/week). `LDL3` method not stated.",
        "Overlap: Dryad metadata states one study population; cross-source identity cannot be resolved.",
    ]
    extra = [("Availability (counts)", L.md_table(avail)), ("Smoking columns (decoding check)", L.md_table(smk)),
             ("Menopause code vs age (decoding check)", L.md_table(meno)),
             ("Source metadata", f"Title: {meta.get('title')}; DOI {meta.get('doi')}; licence {meta.get('license')}; "
                                 f"design: {meta.get('design')}.")]
    vm = {}
    for o, h, m, u, f in COLS:
        vm[o] = {"kind": "original", "meaning": m, "unit": u}
        vm[h] = {"kind": "harmonised", "meaning": m, "unit": u, "flag": f}
    vm.update({"male": {"kind": "derived", "meaning": "1 = male (Sex = 1)"},
               "former_smoker": {"kind": "derived", "meaning": "exsmoker = 1 and current smoker = 0"},
               "postmenopausal": {"kind": "derived", "meaning": "1 = postmenopausal (code 2), 0 = code 1; NA men"},
               "qc_abi_max_implausible": {"kind": "qc", "meaning": "ABImax > 3 (set missing)"},
               "abi_max_orig_value": {"kind": "derived", "meaning": "ABImax before cleaning"},
               "map_mmhg_formula": {"kind": "derived", "meaning": "DBP + (SBP - DBP)/3", "unit": "mmHg"},
               "bapwv_max_ms": {"kind": "derived", "meaning": "PWVmax / 100", "unit": "m/s"},
               "ldl_friedewald_mgdl": {"kind": "derived", "meaning": "TC - HDL - TG/5 (TG < 400)", "unit": "mg/dL"}})
    L.write_parquet(df, NAME)
    L.write_dictionary(df, NAME, "Japan GGT-baPWV check-up dataset (Dryad m484p), n = 912", [XLS, META],
                       "parse_japan_ggt_bapwv.py", vm, notes=notes, extra_sections=extra)
    print(NAME, df.shape)


if __name__ == "__main__":
    main()
