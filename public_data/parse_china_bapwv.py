"""Stage 0 parser: China (Xiangyang) arterial stiffness dataset, figshare 30225667 (n = 36,223).

Parsing, cleaning and description only (no models). Output: results/stage0/clean/china_bapwv_36k.parquet and
results/stage0/dictionaries/china_bapwv_36k.md.
"""
from __future__ import annotations

import json

import numpy as np
import openpyxl
import pandas as pd

import s0lib as L

SRC_DIR = L.EXT / "open_cohorts" / "china_arterial_stiffness_figshare_30225667"
XLSX = SRC_DIR / "Atherosclerosis_data_SD.xlsx"
META = SRC_DIR / "china_arterial_stiffness_figshare_30225667_public_manifest.source_metadata.json"
NAME = "china_bapwv_36k"
ORIG = ["Sex", "Age", "LbaPWV", "RbaPWV", "LbPMap", "RbPMap", "LaSYS", "RaSYS", "LaMAP", "RaMAP", "Labi", "Rabi",
        "Label"]
PWV_SENTINEL_MS = 50.0  # values >= 50 m/s (87.38, 99.99) are treated as device/entry sentinels


def main() -> None:
    meta = json.loads(META.read_text())
    # Sheet structure check (one sheet, header in row 1, 13 named columns + empty trailing columns)
    wb = openpyxl.load_workbook(XLSX, read_only=False)
    sheets = wb.sheetnames
    ws = wb[sheets[0]]
    stray = []
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=14, max_col=ws.max_column):
        for c in row:
            if c.value is not None:
                stray.append(f"{c.coordinate}={c.value!r}")
    wb.close()

    raw = pd.read_excel(XLSX, sheet_name=sheets[0], header=0, engine="openpyxl")
    assert list(raw.columns[:13]) == ORIG, raw.columns
    extra = raw.iloc[:, 13:]
    df = raw[ORIG].copy()
    df.insert(0, "row_id", np.arange(1, len(df) + 1))

    # ---------------- harmonised variables ----------------
    df["sex_male_probable"] = df["Sex"].astype(float)  # 1 = male (inferred, see notes)
    df["age_y"] = df["Age"].astype(float)
    for side, s in (("l", "L"), ("r", "R")):
        v = df[f"{s}baPWV"].astype(float)
        df[f"qc_bapwv_{side}_sentinel"] = (v >= PWV_SENTINEL_MS).astype(int)
        df[f"bapwv_{side}_ms"] = v.where(v < PWV_SENTINEL_MS)
        df[f"ankle_sbp_{side}_mmhg"] = df[f"{s}aSYS"].astype(float)
        df[f"ankle_map_{side}_mmhg"] = df[f"{s}aMAP"].astype(float)
        df[f"abi_{side}"] = df[f"{s}abi"].astype(float)
        df[f"brachial_pctmap_{side}_uncertain"] = df[f"{s}bPMap"].astype(float)
        # ABI = ankle SBP / brachial SBP  ->  brachial SBP (ABI denominator) = ankle SBP / ABI
        df[f"brachial_sbp_implied_from_{side}_mmhg"] = df[f"ankle_sbp_{side}_mmhg"] / df[f"abi_{side}"]
    df["bapwv_mean_ms"] = df[["bapwv_l_ms", "bapwv_r_ms"]].mean(axis=1, skipna=False)
    df["bapwv_max_ms"] = df[["bapwv_l_ms", "bapwv_r_ms"]].max(axis=1, skipna=False)
    df["bapwv_mean_cms"] = df["bapwv_mean_ms"] * 100.0
    df["ankle_sbp_mean_mmhg"] = df[["ankle_sbp_l_mmhg", "ankle_sbp_r_mmhg"]].mean(axis=1)
    df["ankle_map_mean_mmhg"] = df[["ankle_map_l_mmhg", "ankle_map_r_mmhg"]].mean(axis=1)
    df["brachial_sbp_implied_mmhg"] = df[["brachial_sbp_implied_from_l_mmhg",
                                          "brachial_sbp_implied_from_r_mmhg"]].mean(axis=1)
    df["qc_brachial_sbp_implied_lr_diff_mmhg"] = (df["brachial_sbp_implied_from_l_mmhg"]
                                                  - df["brachial_sbp_implied_from_r_mmhg"])
    df["abi_min"] = df[["abi_l", "abi_r"]].min(axis=1)
    df["abi_lt_0_9_any"] = (df["abi_min"] < 0.9).astype(int)
    df["abi_gt_1_4_any"] = (df[["abi_l", "abi_r"]].max(axis=1) > 1.4).astype(int)
    df["label_undocumented"] = df["Label"].astype(int)

    # ---------------- descriptives ----------------
    n = len(df)
    lr = df["qc_brachial_sbp_implied_lr_diff_mmhg"]
    lab = df.groupby(pd.cut(df["bapwv_max_ms"], [0, 13, 14, 14.5, 15, 15.2, 15.5, 16, 100]), observed=False)[
        "Label"].agg(["count", "sum"]).reset_index()
    lab.columns = ["max baPWV (m/s) bin", "N", "Label = 1"]
    lab["max baPWV (m/s) bin"] = lab["max baPWV (m/s) bin"].astype(str)
    # Decoding evidence for Sex only (no PWV-BP or PWV-age contrasts are tabulated in Stage 0)
    by_sex = df.groupby("Sex").agg(N=("row_id", "size"), age_mean=("age_y", "mean"),
                                   ankle_sbp_mean=("ankle_sbp_mean_mmhg", "mean"),
                                   brachial_sbp_implied_mean=("brachial_sbp_implied_mmhg", "mean")).reset_index()
    age_bins = df.groupby(pd.cut(df["age_y"], [29, 39, 49, 59, 69, 80]), observed=False).agg(
        N=("row_id", "size"), n_abi_lt_0_9=("abi_lt_0_9_any", "sum")).reset_index()
    age_bins["age_y"] = age_bins["age_y"].astype(str)

    extra_sections = [
        ("Sheet structure", "\n".join([
            f"- Workbook sheets: {sheets}. One sheet, header in row 1, one row per participant, N = {n}.",
            f"- 13 named columns {ORIG}; columns N-T are unnamed and empty except stray cells: {stray}. "
            "They are dropped (no information).",
            f"- Source metadata (figshare JSON): population = '{meta.get('population')}'; licence "
            f"{meta.get('license', {}).get('name')}; listed measurements = {meta.get('additional_measurements')}.",
        ])),
        ("Which BP variables were recorded at the PWV acquisition", "\n".join([
            "- **Ankle SBP** (`LaSYS`, `RaSYS`, mmHg) and **ankle MAP** (`LaMAP`, `RaMAP`, mmHg): recorded by the "
            "same four-limb oscillometric acquisition that produced baPWV and ABI (implied by the device design and "
            "by the figshare description 'left/right ankle systolic pressure, ankle mean arterial pressure'). "
            "No ankle DBP and no ankle pulse pressure are provided.",
            "- **Brachial SBP is not a column**, but it is recoverable exactly up to rounding: ABI = ankle SBP / "
            "brachial SBP, so `LaSYS/Labi` and `RaSYS/Rabi` must both equal the ABI denominator (the higher-arm "
            "brachial SBP of the same acquisition). They agree: L-R difference mean "
            f"{lr.mean():.3f} mmHg, SD {lr.std():.2f}, 1st-99th percentile {lr.quantile(0.01):.2f} to "
            f"{lr.quantile(0.99):.2f} mmHg, which is the size expected from ABI rounded to 0.01. "
            "`brachial_sbp_implied_mmhg` (mean of the two) is therefore a same-acquisition brachial SBP "
            "(precision about +/-0.5 mmHg).",
            "- **`LbPMap` / `RbPMap` are NOT brachial MAP in mmHg**: values are "
            f"{df['LbPMap'].min()}-{df['LbPMap'].max()} (mean {df['LbPMap'].mean():.1f}, SD "
            f"{df['LbPMap'].std():.1f}), impossible for MAP in mmHg in adults with ankle SBP of about 144 mmHg. "
            "The name and range fit the device's brachial %MAP (percentage mean arterial pressure = (MAP - DBP) / "
            "(SBP - DBP) x 100, a waveform-shape index, typically 40-55 %). Treated as **brachial %MAP (unit %), "
            "uncertain**; it cannot be converted to mmHg without brachial DBP. Consequently there is **no "
            "same-acquisition brachial MAP or DBP** in this file.",
            "- Implication for F1 (beta_acute): baPWV can be related descriptively to same-acquisition ankle SBP, "
            "ankle MAP and implied brachial SBP; a brachial-MAP-based coupling coefficient is not available.",
        ])),
        ("Decoding check for the Sex code (descriptive)", L.md_table(by_sex)),
        ("N by age band", L.md_table(age_bins)),
        ("`Label` versus max baPWV (decoding check)", L.md_table(lab) + "\n\nThe highest max baPWV among Label = 0 "
         f"is {df.loc[df['Label'] == 0, 'bapwv_max_ms'].max():.2f} m/s, so every participant above that has "
         "Label = 1, but Label = 1 also occurs at low baPWV ("
         f"{int(((df['Label'] == 1) & (df['bapwv_max_ms'] <= 13)).sum())} with max baPWV <= 13 m/s), so "
         "Label is not a simple baPWV threshold. Its definition is not in the file (probably the curated "
         "'arterial stiffness / atherosclerosis' class of the data descriptor). Do not use until documented."),
    ]

    notes = [
        "`Sex` is coded 0/1 with no codebook in the file. Sex = 1 (n = "
        f"{int((df['Sex'] == 1).sum())}, {100 * df['Sex'].mean():.1f} %) has mean ankle SBP about 11 mmHg and mean "
        "implied brachial SBP about 7 mmHg higher than Sex = 0 at the same mean age, the pattern expected for men, so **1 = male is inferred "
        "(probable, not documented)**. Verify against the data descriptor (Scientific Data) before use.",
        "baPWV is in **m/s** (median about 14 m/s); `bapwv_*_cms` = x100. Values >= 50 m/s (LbaPWV 87.38, "
        f"99.99, 99.99; n = {int(df['qc_bapwv_l_sentinel'].sum() + df['qc_bapwv_r_sentinel'].sum())}) are treated "
        "as sentinels and set to missing in the harmonised variables (originals kept). Values 35-45 m/s are kept "
        "but are extreme.",
        "ABI is reported to 2 decimals; ABI < 0.9 in either leg: "
        f"{int(df['abi_lt_0_9_any'].sum())}; ABI > 1.4 in either leg: {int(df['abi_gt_1_4_any'].sum())}. baPWV is "
        "not valid with ABI < 0.9 (device manuals and guidelines exclude these); flagged via `abi_lt_0_9_any`.",
        "**Mathematical coupling**: ABI = ankle SBP / brachial SBP and baPWV is measured from the same ankle "
        "waveforms; ankle SBP, ankle MAP, implied brachial SBP, ABI and baPWV are all outputs of one acquisition. "
        "Any analysis of baPWV on ankle SBP and on ABI shares the ankle SBP numerator; implied brachial SBP is "
        "computed from ankle SBP / ABI and is therefore not independent of ABI.",
        "No height (path length), heart rate, medication, diagnosis or brachial DBP columns: baPWV path-length "
        "formula and HR adjustment cannot be checked.",
        "Cross-sectional, one row per person; no participant identifier in the file (`row_id` = file row order). "
        "The same Xiangyang cohort may underlie other publications; do not add counts to other sources.",
    ]
    varmeta = {
        "row_id": {"kind": "derived", "meaning": "Row order in the source sheet (no ID in file)"},
        "Sex": {"kind": "original", "meaning": "Sex code 0/1 (undocumented; 1 = male inferred)", "flag": "uncertain"},
        "Age": {"kind": "original", "meaning": "Age", "unit": "years"},
        "LbaPWV": {"kind": "original", "meaning": "Left brachial-ankle PWV", "unit": "m/s"},
        "RbaPWV": {"kind": "original", "meaning": "Right brachial-ankle PWV", "unit": "m/s"},
        "LbPMap": {"kind": "original", "meaning": "Left 'bPMap' (NOT mmHg; probably brachial %MAP)", "unit": "% (?)",
                   "flag": "uncertain"},
        "RbPMap": {"kind": "original", "meaning": "Right 'bPMap' (NOT mmHg; probably brachial %MAP)",
                   "unit": "% (?)", "flag": "uncertain"},
        "LaSYS": {"kind": "original", "meaning": "Left ankle SBP (same acquisition as baPWV)", "unit": "mmHg"},
        "RaSYS": {"kind": "original", "meaning": "Right ankle SBP (same acquisition as baPWV)", "unit": "mmHg"},
        "LaMAP": {"kind": "original", "meaning": "Left ankle MAP (same acquisition)", "unit": "mmHg"},
        "RaMAP": {"kind": "original", "meaning": "Right ankle MAP (same acquisition)", "unit": "mmHg"},
        "Labi": {"kind": "original", "meaning": "Left ankle-brachial index", "unit": "ratio"},
        "Rabi": {"kind": "original", "meaning": "Right ankle-brachial index", "unit": "ratio"},
        "Label": {"kind": "original", "meaning": "Binary label, definition not documented in file",
                  "flag": "undocumented"},
        "sex_male_probable": {"kind": "harmonised", "meaning": "1 = male (inferred from Sex), 0 = female",
                              "flag": "inferred coding"},
        "age_y": {"kind": "harmonised", "meaning": "Age", "unit": "years"},
        "bapwv_l_ms": {"kind": "harmonised", "meaning": "Left baPWV, sentinels (>=50) set missing", "unit": "m/s"},
        "bapwv_r_ms": {"kind": "harmonised", "meaning": "Right baPWV, sentinels (>=50) set missing", "unit": "m/s"},
        "qc_bapwv_l_sentinel": {"kind": "qc", "meaning": "Left baPWV >= 50 m/s (set missing)"},
        "qc_bapwv_r_sentinel": {"kind": "qc", "meaning": "Right baPWV >= 50 m/s (set missing)"},
        "ankle_sbp_l_mmhg": {"kind": "harmonised", "meaning": "Left ankle SBP", "unit": "mmHg"},
        "ankle_sbp_r_mmhg": {"kind": "harmonised", "meaning": "Right ankle SBP", "unit": "mmHg"},
        "ankle_map_l_mmhg": {"kind": "harmonised", "meaning": "Left ankle MAP", "unit": "mmHg"},
        "ankle_map_r_mmhg": {"kind": "harmonised", "meaning": "Right ankle MAP", "unit": "mmHg"},
        "abi_l": {"kind": "harmonised", "meaning": "Left ABI", "unit": "ratio"},
        "abi_r": {"kind": "harmonised", "meaning": "Right ABI", "unit": "ratio"},
        "brachial_pctmap_l_uncertain": {"kind": "harmonised", "meaning": "Copy of LbPMap, interpreted as brachial %MAP",
                                        "unit": "% (?)", "flag": "uncertain"},
        "brachial_pctmap_r_uncertain": {"kind": "harmonised", "meaning": "Copy of RbPMap, interpreted as brachial %MAP",
                                        "unit": "% (?)", "flag": "uncertain"},
        "brachial_sbp_implied_from_l_mmhg": {"kind": "derived", "meaning": "LaSYS / Labi (ABI denominator)",
                                             "unit": "mmHg", "flag": "coupled to ABI"},
        "brachial_sbp_implied_from_r_mmhg": {"kind": "derived", "meaning": "RaSYS / Rabi (ABI denominator)",
                                             "unit": "mmHg", "flag": "coupled to ABI"},
        "bapwv_mean_ms": {"kind": "derived", "meaning": "Mean of left and right baPWV (both required)", "unit": "m/s"},
        "bapwv_max_ms": {"kind": "derived", "meaning": "Max of left and right baPWV (both required)", "unit": "m/s"},
        "bapwv_mean_cms": {"kind": "derived", "meaning": "bapwv_mean_ms x 100", "unit": "cm/s"},
        "ankle_sbp_mean_mmhg": {"kind": "derived", "meaning": "Mean of left/right ankle SBP", "unit": "mmHg"},
        "ankle_map_mean_mmhg": {"kind": "derived", "meaning": "Mean of left/right ankle MAP", "unit": "mmHg"},
        "brachial_sbp_implied_mmhg": {"kind": "derived",
                                      "meaning": "Same-acquisition brachial SBP = mean(LaSYS/Labi, RaSYS/Rabi)",
                                      "unit": "mmHg", "flag": "coupled to ABI; +/-0.5 mmHg rounding"},
        "qc_brachial_sbp_implied_lr_diff_mmhg": {"kind": "qc", "meaning": "Left minus right implied brachial SBP "
                                                 "(should be ~0)", "unit": "mmHg"},
        "abi_min": {"kind": "derived", "meaning": "Lower of left/right ABI", "unit": "ratio"},
        "abi_lt_0_9_any": {"kind": "derived", "meaning": "ABI < 0.9 in either leg"},
        "abi_gt_1_4_any": {"kind": "derived", "meaning": "ABI > 1.4 in either leg"},
        "label_undocumented": {"kind": "harmonised", "meaning": "Copy of Label (undocumented)", "flag": "undocumented"},
    }
    L.write_parquet(df, NAME)
    L.write_dictionary(df, NAME, "China arterial stiffness dataset (figshare 30225667), n = 36,223",
                       [XLSX, META], "parse_china_bapwv.py", varmeta, notes=notes, extra_sections=extra_sections)
    print(f"{NAME}: {df.shape}")


if __name__ == "__main__":
    main()
