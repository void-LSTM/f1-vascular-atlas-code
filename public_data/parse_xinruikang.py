"""Stage 0 parser: Xinruikang hypertension RCT (Mendeley Data vz39k7myss), about 122 participants.

Workbook with five sheets and three-row merged Chinese headers. Outputs (no models):
  results/stage0/clean/xinruikang_long.parquet          - one row per participant x visit (weeks 0, 4, 8, 12)
  results/stage0/clean/xinruikang_participants.parquet  - one row per participant (questionnaire + arm + presence)
"""
from __future__ import annotations

import re

import numpy as np
import openpyxl
import pandas as pd

import s0lib as L

XLSX = L.EXT / "open_cohorts" / "oc05_Xinruikang_RCT_PWV_mendeley_vz39k7myss" / "Xinruikang-data.xlsx"

SHEET_Q = "入组调查问卷数值"   # enrolment questionnaire values
SHEET_W0 = "入组体检数值"      # enrolment (week 0) examination values
SHEET_W4 = "四周体检高血压"    # week-4 examination (BP)
SHEET_W8 = "八周体检高血压"    # week-8 examination (BP)
SHEET_W12 = "12周体检数据"     # week-12 examination data

# Positional column maps (0-based) with the original merged header path, verified against rows 1-3 of each sheet.
LAB_COLS = [  # (english name, unit, original header path)
    ("ast_ul", "U/L", "生化 > 谷草转氨酶 (AST)"),
    ("alt_ul", "U/L", "生化 > 谷丙转氨酶 (ALT)"),
    ("bun_mmoll", "mmol/L", "生化 > 尿素氮 (urea nitrogen)"),
    ("creatinine_umoll", "umol/L", "生化 > 肌酐 (creatinine)"),
    ("uric_acid_umoll", "umol/L", "生化 > 尿酸 (uric acid)"),
    ("tc_mmoll", "mmol/L", "生化 > 总胆固醇 (total cholesterol)"),
    ("tg_mmoll", "mmol/L", "生化 > 甘油三脂 (triglycerides)"),
    ("hdl_mmoll", "mmol/L", "生化 > HDL"),
    ("ldl_mmoll", "mmol/L", "生化 > LDL"),
    ("lpa_mgl", "mg/L", "生化 > 脂蛋白 (lipoprotein; probably Lp(a))"),
    ("glucose_mmoll", "mmol/L", "生化 > 血糖 (blood glucose)"),
    ("apoa1_gl", "g/L", "生化 > 载脂蛋白A1 (APOA1)"),
    ("apob_gl", "g/L", "生化 > 载脂蛋白B (APOB)"),
    ("apoa1_apob_ratio", "ratio", "生化 > 载脂蛋白A1/载脂蛋白B"),
    ("crp_mgl", "mg/L", "炎性 > CRP (inflammation)"),
]
EXAM_COMMON = [
    ("visit_date_raw", "", "检测时间 (examination date)"),
    ("bp_status_raw", "", "血压 > 状态 (BP status)"),
    ("sbp_mmhg", "mmHg", "血压 > 高 (systolic)"),
    ("dbp_mmhg", "mmHg", "血压 > 低 (diastolic)"),
]
PWV_BLOCK = [
    ("pwv_l_cms", "cm/s", "PWV > 左侧脉搏波传导速度（cm/s) (left PWV)"),
    ("pwv_r_cms", "cm/s", "PWV > 右侧脉搏波传导速度（cm/s) (right PWV)"),
    ("abi_l", "ratio", "PWV > 左侧踝臂指数 (left ankle-brachial index)"),
    ("abi_r", "ratio", "PWV > 右侧踝臂指数 (right ankle-brachial index)"),
    ("endopat_rhi", "ratio", "Endopat (no sub-header)"),
    ("fmd_col_rhi", "ratio", "FMD > RHI"),
]
W0_MAP = ([("visit_date_raw", "", "入组期时间 (enrolment date)"), ("arm_code_raw", "", "分组 (group)")]
          + EXAM_COMMON[1:] + PWV_BLOCK + LAB_COLS + [("trial_status_raw", "", "状态 (status)")])
W12_MAP = EXAM_COMMON + PWV_BLOCK + LAB_COLS + [("trial_status_raw", "", "状态 (status)")]
W48_MAP = EXAM_COMMON + [("trial_status_raw", "", "状态 (status)")]
Q_MAP = [("sex_raw", "", "调查问卷 > 性别 (sex)"), ("age_y", "years", "年龄 (age)"), ("height_cm", "cm", "身高 (height)"),
         ("weight_kg", "kg", "体重 (weight)"), ("office_sbp_mmhg", "mmHg", "诊室血压 > 高 (office SBP)"),
         ("office_dbp_mmhg", "mmHg", "诊室血压 > 低 (office DBP)"), ("medication_raw", "", "服药情况 (medication)")]

DRUGS = [  # (Chinese token, English, class)
    ("酒石酸美托洛尔", "metoprolol tartrate", "bb"), ("美托洛尔", "metoprolol", "bb"),
    ("盐酸地尔硫卓", "diltiazem hydrochloride", "ccb"), ("硝苯地平", "nifedipine", "ccb"),
    ("氨氯地平", "amlodipine", "ccb"), ("缬沙坦", "valsartan", "arb"), ("坎地沙坦", "candesartan", "arb"),
    ("替米沙坦", "telmisartan", "arb"), ("卡托普利", "captopril", "acei"), ("吲达帕胺", "indapamide", "diuretic"),
    ("利血平", "reserpine", "reserpine"), ("瑞舒伐他汀", "rosuvastatin", "statin"),
    ("阿托伐他汀", "atorvastatin", "statin"), ("辛伐他汀", "simvastatin", "statin"),
    ("辛伐他丁", "simvastatin (typo in source)", "statin"), ("阿阿司匹林", "aspirin (typo in source)", "aspirin"),
    ("阿司匹林", "aspirin", "aspirin"), ("尼可地尔", "nicorandil", "antianginal"),
    ("曲米沙嗪", "'qu-mi-sha-qin' (unrecognised; probably trimetazidine)", "antianginal"),
    ("丹七片", "Danqi tablet (TCM)", "tcm"), ("丹参片", "danshen tablet (TCM)", "tcm"), ("丹参", "danshen (TCM)", "tcm"),
    ("心血康", "Xinxuekang (TCM)", "tcm"), ("稳心颗粒", "Wenxin granule (TCM)", "tcm"),
    ("甲钴胺", "mecobalamin", "other"), ("吗丁啉", "domperidone", "other"),
    ("降压药忘记药名", "antihypertensive, name forgotten", "antihypertensive_unnamed"),
    ("降压药", "antihypertensive (unnamed)", "antihypertensive_unnamed"),
    ("忘记药名", "drug name forgotten", "name_forgotten"), ("未服药", "not taking medication", "none"),
    ("他汀", "statin (unnamed)", "statin"),
]
MISC_TOKENS = [("服用", "taking "), ("个月", " months"), ("年", " years"), ("月", " months"), ("日", " days"),
               ("、", "; "), ("，", "; "), ("忘记药名", "name forgotten")]


def read_rows(ws) -> list[list]:
    return [list(r) for r in ws.iter_rows(values_only=True)]


def body(rows: list[list], colmap: list, first_data_col: int) -> pd.DataFrame:
    """Data rows start at sheet row 4; column A = site (merged, forward filled), B = participant id."""
    recs = []
    site = None
    for i, r in enumerate(rows[3:], start=4):
        if r[0] is not None:
            site = str(r[0]).strip()
        if r[1] is None:
            continue
        rec = {"sheet_row": i, "site": site, "participant_id": str(r[1]).strip()}
        for k, (name, _, _) in enumerate(colmap):
            rec[name] = r[first_data_col + k] if first_data_col + k < len(r) else None
        trailing = [v for v in r[first_data_col + len(colmap):] if v is not None]
        rec["_trailing_nonempty"] = len(trailing)
        recs.append(rec)
    return pd.DataFrame(recs)


def parse_censored(v):
    """Return (numeric value or NaN, censoring direction '<'/'>' or None, limit)."""
    if v is None:
        return np.nan, None, np.nan
    if isinstance(v, (int, float)):
        return float(v), None, np.nan
    s = str(v).strip().replace("＜", "<").replace("＞", ">")
    m = re.match(r"^([<>])\s*([0-9.]+)$", s)
    if m:
        return np.nan, m.group(1), float(m.group(2))
    try:
        return float(s), None, np.nan
    except ValueError:
        return np.nan, "unparsed", np.nan


def translate_med(s) -> str | None:
    if s is None:
        return None
    s = str(s).strip()
    if s == "/":
        return "none recorded ('/')"
    out = s
    for zh, en, _ in DRUGS:
        out = out.replace(zh, f" {en} ")
    for zh, en in MISC_TOKENS:
        out = out.replace(zh, en)
    return re.sub(r"\s+", " ", out).strip()


def med_flags(s) -> dict:
    s = "" if s is None else str(s)
    flags = {c: 0 for c in ["ccb", "arb", "acei", "bb", "diuretic", "reserpine", "antihypertensive_unnamed",
                            "statin", "aspirin", "antianginal", "tcm", "name_forgotten", "none"]}
    rest = s
    for zh, _, cls in DRUGS:
        if zh in rest:
            if cls in flags:
                flags[cls] = 1
            rest = rest.replace(zh, "")
    flags["none"] = int(s.strip() in ("/", "未服药"))
    flags["any_antihypertensive"] = int(any(flags[c] for c in ["ccb", "arb", "acei", "bb", "diuretic", "reserpine",
                                                               "antihypertensive_unnamed"]))
    return {f"med_{k}": v for k, v in flags.items()}


def main() -> None:
    wb = openpyxl.load_workbook(XLSX, data_only=True)
    assert wb.sheetnames == [SHEET_Q, SHEET_W0, SHEET_W4, SHEET_W8, SHEET_W12], wb.sheetnames
    rows = {s: read_rows(wb[s]) for s in wb.sheetnames}
    # Header verification (row 3 sub-headers)
    assert rows[SHEET_W0][2][7].startswith("左侧脉搏波传导速度") and rows[SHEET_W0][2][12] == "RHI"
    assert rows[SHEET_W12][2][6].startswith("左侧脉搏波传导速度") and rows[SHEET_W12][2][11] == "RHI"
    assert rows[SHEET_W0][1][11] == "Endopat" and rows[SHEET_W0][1][12] == "FMD"
    assert rows[SHEET_W4][2][3] == "状态" and rows[SHEET_W4][2][4] == "高"

    q = body(rows[SHEET_Q], Q_MAP, 2)
    w0 = body(rows[SHEET_W0], W0_MAP, 2)
    w4 = body(rows[SHEET_W4], W48_MAP, 2)
    w8 = body(rows[SHEET_W8], W48_MAP, 2)
    w12 = body(rows[SHEET_W12], W12_MAP, 2)
    for d, wk, sh in ((w0, 0, SHEET_W0), (w4, 4, SHEET_W4), (w8, 8, SHEET_W8), (w12, 12, SHEET_W12)):
        d.insert(0, "week", wk)
        d.insert(1, "source_sheet", sh)
    long = pd.concat([w0, w4, w8, w12], ignore_index=True, sort=False)
    trailing = int(long["_trailing_nonempty"].sum())
    long = long.drop(columns="_trailing_nonempty")

    # arm from the week-0 sheet (only sheet with the group column)
    w0["arm_code"] = w0["arm_code_raw"].astype(str).str.strip()
    arm = dict(zip(w0["participant_id"], w0["arm_code"]))
    long["arm_code"] = long["participant_id"].map(arm)
    long = long.drop(columns=["arm_code_raw"]).merge(w0[["participant_id", "arm_code_raw"]], on="participant_id",
                                                     how="left")

    # ---- keep the original text of every source cell, then harmonise ----
    value_cols = [c for c, _, _ in EXAM_COMMON[2:] + PWV_BLOCK + LAB_COLS]
    for c in value_cols:
        long[f"orig_{c}"] = long[c].map(L.raw_to_text)
    for c in value_cols:
        parsed = long[c].map(parse_censored)
        long[c] = parsed.map(lambda t: t[0]).astype(float)
        cens = parsed.map(lambda t: t[1])
        if cens.notna().any():
            long[f"{c}_censor_dir"] = cens
            long[f"{c}_censor_limit"] = parsed.map(lambda t: t[2]).astype(float)
    long["visit_date"] = pd.to_datetime(long["visit_date_raw"].astype(str).str.strip(), format="%Y.%m.%d",
                                        errors="coerce")
    long["bp_status"] = long["bp_status_raw"].map({"药后": "post-dose (measured after taking medication)"})
    long["trial_status"] = long["trial_status_raw"].map({"在组": "in study", "结题": "study completed"})

    # ---- QC rules (flags; harmonised value set to missing where clearly erroneous) ----
    qc = pd.Series([""] * len(long), index=long.index, dtype=object)

    def add(mask, msg):
        nonlocal qc
        mask = mask.fillna(False).astype(bool)
        qc = qc.where(~mask, qc + "; " + msg)

    for side in ("l", "r"):
        c = f"pwv_{side}_cms"
        bad = long[c].notna() & ((long[c] >= 4000) | (long[c] < 500))
        for idx in long.index[bad]:
            one = pd.Series(False, index=long.index)
            one[idx] = True
            add(one, f"{c} = {long.at[idx, c]:.0f} implausible, set missing")
        long.loc[bad, c] = np.nan
        c = f"abi_{side}"
        bad = long[c].notna() & (long[c] > 3)
        add(bad, f"{c} > 3 (missing decimal point, e.g. 1019 = 1.019?) set missing")
        long.loc[bad, c] = np.nan
    # row-specific transcription errors identified by inspection (see dictionary notes)
    fixes = [
        (0, "L2", ["ast_ul", "alt_ul", "bun_mmoll"], "AST/ALT/BUN cells garbled (AST=2721, i.e. '27'+'21'; ALT and "
         "BUN shifted) set missing"),
        (0, "L3", ["apoa1_gl", "apob_gl", "apoa1_apob_ratio"], "APOA1/APOB/ratio are copies of BUN/creatinine/UA; "
         "set missing"),
        (0, "O2", ["bun_mmoll"], "BUN = 302 is a copy of uric acid; set missing"),
        (12, "L4", ["bun_mmoll"], "BUN = 49 implausible (probably 4.9); set missing"),
        (12, "K2", ["apob_gl", "crp_mgl"], "APOB duplicates APOA1 and CRP equals the probable APOB (1.46; ratio "
         "1.81/1.46 = 1.24); set missing"),
    ]
    for wk, pid, cols, msg in fixes:
        m = (long["week"] == wk) & (long["participant_id"] == pid)
        assert m.sum() == 1, (wk, pid)
        long.loc[m, cols] = np.nan
        add(m, msg)
    ratio_calc = long["apoa1_gl"] / long["apob_gl"]
    bad = (ratio_calc - long["apoa1_apob_ratio"]).abs() > 0.15
    add(bad, "APOA1/APOB ratio inconsistent with APOA1 and APOB (>0.15)")
    add(long["endopat_rhi"] > 5, "RHI > 5 (implausible for EndoPAT)")
    diff = long["endopat_rhi"].notna() & long["fmd_col_rhi"].notna() & (long["endopat_rhi"] != long["fmd_col_rhi"])
    add(diff, "Endopat and FMD/RHI columns differ")
    long["qc_notes"] = qc.str.strip("; ").replace("", np.nan)

    # date order check
    d = long.pivot_table(index="participant_id", columns="week", values="visit_date", aggfunc="first")
    bad_order = d.index[(d[0] > d[4]) | (d[4] > d[8]) | (d[8] > d[12])].tolist()
    m = long["participant_id"].isin(bad_order) & (long["week"] == 0)
    long.loc[m, "qc_notes"] = long.loc[m, "qc_notes"].fillna("").astype(str).str.cat(
        ["visit dates out of order (week-0 date after week-4 date; probable year typo)"] * int(m.sum()),
        sep="; ").str.strip("; ")

    # derived haemodynamics
    long["map_mmhg_formula"] = long["dbp_mmhg"] + (long["sbp_mmhg"] - long["dbp_mmhg"]) / 3.0
    long["pp_mmhg"] = long["sbp_mmhg"] - long["dbp_mmhg"]
    long["pwv_mean_cms"] = long[["pwv_l_cms", "pwv_r_cms"]].mean(axis=1, skipna=False)
    long["pwv_l_ms"] = long["pwv_l_cms"] / 100
    long["pwv_r_ms"] = long["pwv_r_cms"] / 100
    long["pwv_mean_ms"] = long["pwv_mean_cms"] / 100
    long["_k1"] = long["participant_id"].str[0]
    long["_k2"] = long["participant_id"].str.extract(r"(\d+)$")[0].astype(int)
    long = long.sort_values(["_k1", "_k2", "week"]).drop(columns=["_k1", "_k2"]).reset_index(drop=True)
    first = ["participant_id", "site", "arm_code", "arm_code_raw", "week", "visit_date", "visit_date_raw",
             "source_sheet", "sheet_row", "bp_status", "bp_status_raw", "sbp_mmhg", "dbp_mmhg", "map_mmhg_formula",
             "pp_mmhg", "pwv_l_cms", "pwv_r_cms", "pwv_mean_cms", "pwv_l_ms", "pwv_r_ms", "pwv_mean_ms", "abi_l",
             "abi_r", "endopat_rhi", "fmd_col_rhi"]
    long = long[first + [c for c in long.columns if c not in first]]

    # ---- participant table ----
    q["sex_male"] = q["sex_raw"].map({"男": 1, "女": 0})
    for c in ("age_y", "height_cm", "weight_kg", "office_sbp_mmhg", "office_dbp_mmhg"):
        q[c] = pd.to_numeric(q[c], errors="coerce")
    q["bmi_kgm2_calc"] = q["weight_kg"] / (q["height_cm"] / 100) ** 2
    q["medication_en"] = q["medication_raw"].map(translate_med)
    q = pd.concat([q, pd.DataFrame([med_flags(s) for s in q["medication_raw"]])], axis=1)
    ids = sorted(set(q.participant_id) | set(long.participant_id),
                 key=lambda x: (x[0], int(re.sub(r"\D", "", x))))
    part = pd.DataFrame({"participant_id": ids})
    part = part.merge(q.drop(columns=["sheet_row"]).rename(columns={"site": "site_questionnaire"}),
                      on="participant_id", how="left")
    part["arm_code"] = part["participant_id"].map(arm)
    part["in_questionnaire"] = part["participant_id"].isin(q["participant_id"]).astype(int)
    for wk in (0, 4, 8, 12):
        part[f"in_week{wk}"] = part["participant_id"].isin(long.loc[long.week == wk, "participant_id"]).astype(int)
        part[f"has_pwv_week{wk}"] = part["participant_id"].isin(
            long.loc[(long.week == wk) & long.pwv_mean_cms.notna(), "participant_id"]).astype(int)
    part["site"] = part["participant_id"].str[0]

    # ---- descriptives ----
    n_arm = part.groupby("arm_code", dropna=False).size().to_dict()
    # Availability only: BP/PWV change over time and by arm are analyses, not tabulated in Stage 0
    vis = long.groupby(["week"]).agg(N_rows=("participant_id", "size"), n_sbp=("sbp_mmhg", "count"),
                                     n_dbp=("dbp_mmhg", "count"), n_pwv_both=("pwv_mean_cms", "count"),
                                     n_abi_both=("abi_l", "count")).reset_index()
    vis_arm = long.groupby(["arm_code", "week"]).agg(N=("participant_id", "size"), n_sbp=("sbp_mmhg", "count"),
                                                     n_pwv_both=("pwv_mean_cms", "count")).reset_index()
    both = long[long.week.isin([0, 12])].pivot_table(index="participant_id", columns="week",
                                                      values=["sbp_mmhg", "pwv_mean_cms"])
    n_pair_pwv = int(both["pwv_mean_cms"].dropna().shape[0])
    n_pair_sbp = int(both["sbp_mmhg"].dropna().shape[0])
    qc_rows = long[long.qc_notes.notna()][["participant_id", "week", "qc_notes"]]
    only_q = part.loc[(part.in_questionnaire == 1) & (part.in_week0 == 0), "participant_id"].tolist()
    only_exam = part.loc[(part.in_questionnaire == 0) & (part.in_week0 == 1), "participant_id"].tolist()
    med_tab = part[[c for c in part.columns if c.startswith("med_") and c != "medication_raw"]].sum().reset_index()
    med_tab.columns = ["medication flag (questionnaire)", "n participants"]

    notes = [
        "Sheets: `入组调查问卷数值` (enrolment questionnaire: sex, age, height, weight, office BP, medication text), "
        "`入组体检数值` (week-0 examination: date, group, BP, L/R PWV, L/R ABI, EndoPAT, labs), `四周体检高血压` and "
        "`八周体检高血压` (weeks 4 and 8: BP only), `12周体检数据` (week-12 examination: as week 0 without group). "
        "Headers span rows 1-3 with merged cells; data start in row 4; column A (site letter A-O) is merged and "
        f"forward-filled. Trailing non-empty cells outside the mapped columns: {trailing}.",
        "**PWV type**: the column headers are '左/右侧脉搏波传导速度 (cm/s)' (left/right pulse wave velocity) paired with "
        "left/right ankle-brachial index, values 1,000-3,000 cm/s in hypertensive adults aged about 60-70: this is "
        "the bilateral **brachial-ankle PWV (baPWV)** of an oscillometric ABI/PWV device (device not named in the "
        "file). Inferred, not stated.",
        f"**Arms**: column `分组` holds 'g' or 'G' (one 'g ' with a trailing space, stripped). Counts: {n_arm}. "
        "**Which code is Xinruikang and which is control is not stated in the file** (blocks arm-specific effect "
        "estimates until the Mendeley description/publication is checked; does not block pooled within-person "
        "delta-PWV / delta-BP description).",
        "BP 'status' is '药后' (= measured after taking the usual antihypertensive dose) at every visit. Whether the BP "
        "at weeks 0 and 12 was taken by the PWV device (same acquisition) or as a separate office measurement is "
        "not stated; the questionnaire has a separate 'office BP' (诊室血压) that differs from the week-0 exam BP. "
        "MAP is computed as DBP + PP/3 (formula, not measured). Treat exam BP as **not proven same-acquisition**.",
        "PWV and ABI exist only at weeks 0 and 12; weeks 4 and 8 have BP only. Within-person delta-PWV vs "
        f"delta-BP is possible for {n_pair_pwv} participants with PWV at both week 0 and 12 ({n_pair_sbp} with SBP "
        "at both).",
        f"ID mismatches between sheets: in the questionnaire but not in the week-0 exam: {only_q}; in the week-0 exam "
        f"but not in the questionnaire: {only_exam}. Sheet presence of these IDs (Q, W0, W4, W8, W12): "
        + "; ".join(f"{r.participant_id}: {r.in_questionnaire}{r.in_week0}{r.in_week4}{r.in_week8}{r.in_week12}"
                    for r in part[part.participant_id.isin(only_q + only_exam)].itertuples())
        + ". A relabelling (e.g. B2 = B13) cannot be verified; linkage is by ID as written, so B13 has no "
        "questionnaire data and B2/C2 have no examination PWV.",
        "Units are inferred from Chinese clinical-lab conventions (not stated except PWV cm/s): AST/ALT U/L, BUN "
        "mmol/L, creatinine and uric acid umol/L, lipids and glucose mmol/L, '脂蛋白' (lipoprotein, probably Lp(a)) "
        "mg/L, apoA1/apoB g/L, CRP mg/L. Censored values ('<2.0', '＜2', '<30', '＞2.5') are set missing in the "
        "numeric column and kept in `*_censor_dir` / `*_censor_limit` and `orig_*`.",
        "`Endopat` (no sub-header) and `FMD` (sub-header 'RHI') hold identical values in all rows but one, i.e. a "
        "single EndoPAT reactive-hyperaemia index; **no independent FMD** is available.",
        "Errors set to missing (originals kept in `orig_*`; reason in `qc_notes`): PWV 4000/4000 (M6 week 0) and "
        "7545 (N1 week 12, right); ABI 1019/1016 (N7 week 12); garbled lab rows L2/L3/O2 (week 0) and L4/K2 "
        "(week 12). G25 week-0 date '2024.11.1' is after its week-4 date (probably 2023.11.1).",
        "Medication text (questionnaire) translated to English with drug-class flags (CCB, ARB, ACEi, BB, diuretic, "
        "reserpine, unnamed antihypertensive, statin, aspirin, TCM). '/' = no medication recorded (not necessarily "
        "none); '忘记药名' = name forgotten. '曲米沙嗪' is not a recognised drug name (probably trimetazidine).",
    ]
    extra = [
        ("Visits", L.md_table(vis)),
        ("Visits by arm code (counts only)", L.md_table(vis_arm)),
        ("Rows with QC notes", L.md_table(qc_rows) if len(qc_rows) else "None."),
    ]
    vm = {
        "participant_id": {"kind": "original", "meaning": "编号 participant code (site letter + number)"},
        "site": {"kind": "original", "meaning": "地区 site letter (merged cells forward-filled)"},
        "arm_code": {"kind": "harmonised", "meaning": "分组 group code from week-0 sheet, stripped ('g'/'G'); "
                     "allocation meaning unknown", "flag": "arm meaning undocumented"},
        "arm_code_raw": {"kind": "original", "meaning": "分组 group code as written in week-0 sheet"},
        "week": {"kind": "derived", "meaning": "Visit week (0, 4, 8, 12) from sheet", "unit": "weeks"},
        "visit_date": {"kind": "harmonised", "meaning": "Examination date (parsed YYYY.M.D)"},
        "visit_date_raw": {"kind": "original", "meaning": "检测时间 examination date text"},
        "source_sheet": {"kind": "derived", "meaning": "Source sheet name"},
        "sheet_row": {"kind": "derived", "meaning": "1-based row in source sheet"},
        "bp_status": {"kind": "harmonised", "meaning": "BP status translated"},
        "bp_status_raw": {"kind": "original", "meaning": "血压 > 状态 BP status ('药后')"},
        "map_mmhg_formula": {"kind": "derived", "meaning": "DBP + (SBP - DBP)/3", "unit": "mmHg"},
        "pp_mmhg": {"kind": "derived", "meaning": "SBP - DBP", "unit": "mmHg"},
        "pwv_mean_cms": {"kind": "derived", "meaning": "Mean of left/right baPWV (both required)", "unit": "cm/s"},
        "pwv_l_ms": {"kind": "derived", "meaning": "Left baPWV", "unit": "m/s"},
        "pwv_r_ms": {"kind": "derived", "meaning": "Right baPWV", "unit": "m/s"},
        "pwv_mean_ms": {"kind": "derived", "meaning": "Mean baPWV", "unit": "m/s"},
        "trial_status_raw": {"kind": "original", "meaning": "状态 trial status ('在组' in study / '结题' completed)"},
        "trial_status": {"kind": "harmonised", "meaning": "Trial status translated"},
        "qc_notes": {"kind": "qc", "meaning": "Reasons for values set missing / flags"},
    }
    for name, unit, hdr in EXAM_COMMON[2:] + PWV_BLOCK + LAB_COLS:
        flag = "PWV type inferred (baPWV)" if name.startswith("pwv") else (
            "unit inferred" if unit not in ("mmHg", "cm/s", "ratio") else "")
        if name == "fmd_col_rhi":
            flag = "duplicate of endopat_rhi"
        vm[name] = {"kind": "harmonised", "meaning": f"{hdr}; numeric, errors/censored set missing", "unit": unit,
                    "flag": flag}
        vm[f"orig_{name}"] = {"kind": "original", "meaning": f"{hdr}; original cell text"}
        vm[f"{name}_censor_dir"] = {"kind": "derived", "meaning": f"Censoring direction of {name} ('<' or '>')"}
        vm[f"{name}_censor_limit"] = {"kind": "derived", "meaning": f"Censoring limit of {name}", "unit": unit}
    L.write_parquet(long, "xinruikang_long")
    L.write_dictionary(long, "xinruikang_long", "Xinruikang hypertension RCT (Mendeley vz39k7myss): visit-level long "
                       "table", [XLSX], "parse_xinruikang.py", vm, notes=notes, extra_sections=extra)
    vmp = {"participant_id": {"kind": "original", "meaning": "编号 participant code"},
           "site_questionnaire": {"kind": "original", "meaning": "地区 site letter in questionnaire"},
           "site": {"kind": "derived", "meaning": "Site letter from ID"},
           "sex_male": {"kind": "harmonised", "meaning": "1 = male (男), 0 = female (女)"},
           "bmi_kgm2_calc": {"kind": "derived", "meaning": "weight / height^2", "unit": "kg/m2"},
           "medication_en": {"kind": "derived", "meaning": "English translation of medication text"},
           "arm_code": {"kind": "harmonised", "meaning": "Group code from week-0 sheet ('g'/'G'); meaning unknown",
                        "flag": "arm meaning undocumented"},
           "in_questionnaire": {"kind": "derived", "meaning": "Present in questionnaire sheet"}}
    for name, unit, hdr in Q_MAP:
        vmp[name] = {"kind": "original", "meaning": hdr, "unit": unit}
    for c in [c for c in part.columns if c.startswith("med_")]:
        vmp[c] = {"kind": "derived", "meaning": f"Medication text mentions class '{c[4:]}' (questionnaire)"}
    for wk in (0, 4, 8, 12):
        vmp[f"in_week{wk}"] = {"kind": "derived", "meaning": f"Has a row in the week-{wk} sheet"}
        vmp[f"has_pwv_week{wk}"] = {"kind": "derived", "meaning": f"Has valid L and R PWV at week {wk}"}
    L.write_parquet(part, "xinruikang_participants")
    L.write_dictionary(part, "xinruikang_participants", "Xinruikang hypertension RCT: participant table", [XLSX],
                       "parse_xinruikang.py", vmp, notes=notes, extra_sections=[("Medication classes", L.md_table(med_tab))])
    print("xinruikang_long", long.shape, "participants", part.shape, "arms", n_arm)


if __name__ == "__main__":
    main()
