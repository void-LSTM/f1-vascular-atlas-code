"""Stage 0 parser: ARTPER cohort (Barcelona primary care), two PLoS One supplementary files.

  pone.0209163.s004.xls  - 'ARTPER incidence' (n = 3,146): ABI (itb) + incident events and times (years)
  pone.0191283.s001.xlsx - 'ARTPER reclassification' (n = 2,716): PAD status, risk scores, cardiac and
                           cerebrovascular events (no ABI value)

Both files carry a 'Labels' codebook sheet, used verbatim. No shared ID: the files are linked here by exact match
on (baseline age, sex, follow-up time), which are stored to float precision in both files.
Outputs (no models): results/stage0/clean/artper_incidence.parquet, artper_reclass.parquet + dictionaries.
"""
from __future__ import annotations

import numpy as np
import openpyxl
import pandas as pd

import s0lib as L

XLS_INC = L.EXT / "open_cohorts" / "oc03_ARTPER_pone0209163" / "pone.0209163.s004.xls"
XLSX_REC = L.EXT / "open_cohorts" / "oc03_ARTPER_pone0191283" / "pone.0191283.s001.xlsx"

INC_NAMES = {  # original -> harmonised
    "iam": "mi", "ictus": "stroke", "cor": "coronary_disease", "cer": "cerebrovascular_event",
    "inter": "vascular_intervention", "mortv": "vascular_death", "mortnv": "nonvascular_death",
    "cort": "coronary_disease_time_y", "cert": "cerebrovascular_event_time_y", "intert": "vascular_intervention_time_y",
    "ts": "followup_y", "sexe": "female", "edat": "age_y", "edat3": "age_cat3", "obesitat": "obesity_cat3",
    "obesab": "abdominal_obesity", "fuma2": "ever_smoker", "htahc": "hypertension", "diabhc": "diabetes",
    "hipercolhc": "hypercholesterolaemia", "antihtahc": "antihypertensive_use", "hipoliphc": "lipid_lowering_use",
    "hipoglichc": "hypoglycaemic_use", "antiaghc": "antiplatelet_anticoagulant_use", "regicor": "regicor_score",
    "score": "score_index", "itb": "abi", "itb3": "abi_cat3", "coltot": "total_chol_mgdl", "hdl": "hdl_mgdl",
    "ldl": "ldl_mgdl", "event": "morbimortality", "eventt": "morbimortality_time_y", "iamt": "mi_time_y",
    "ictust": "stroke_time_y", "mace": "mace", "macet": "mace_time_y", "ecv": "ice", "ecvt": "ice_time_y",
    "morbi": "morbidity", "morbit": "morbidity_time_y",
}
REC_NAMES = {
    "id": "artper_id", "edat": "age_y", "sexe": "female", "fuma3": "smoking_status3", "imc": "bmi_kgm2",
    "htahc": "hypertension", "hipercolhc": "hypercholesterolaemia", "diabhc": "diabetes",
    "framingham": "framingham_score", "regicor": "regicor_score", "ts": "followup_y", "chd": "cardiac_event",
    "chdt": "cardiac_event_time_y", "cer": "cerebrovascular_event", "cert": "cerebrovascular_event_time_y",
    "ap": "pad_baseline", "claudi": "claudication",
}


def labels_from_rows(rows) -> dict:
    out = {}
    for r in rows[1:]:
        if r and r[0]:
            out[str(r[0]).strip()] = (str(r[1] or "").strip(), str(r[2] or "").strip())
    return out


def main() -> None:
    # ---- incidence file (.xls via xlrd in .venv-readers) ----
    wb = L.read_xls(XLS_INC)
    inc_sheet = [s for s in wb if s != "Labels"][0]
    inc = L.xls_sheet_to_frame(wb[inc_sheet]).apply(pd.to_numeric, errors="coerce")
    inc_lab = labels_from_rows(wb["Labels"])
    # ---- reclassification file (.xlsx) ----
    wbx = openpyxl.load_workbook(XLSX_REC, read_only=True)
    rec_sheet = [s for s in wbx.sheetnames if s != "Labels"][0]
    rows = [list(r) for r in wbx[rec_sheet].iter_rows(values_only=True)]
    rec = pd.DataFrame(rows[1:], columns=rows[0]).apply(pd.to_numeric, errors="coerce")
    rec_lab = labels_from_rows([list(r) for r in wbx["Labels"].iter_rows(values_only=True)])
    wbx.close()
    assert set(INC_NAMES) == set(inc.columns), set(inc.columns) ^ set(INC_NAMES)
    assert set(REC_NAMES) == set(rec.columns), set(rec.columns) ^ set(REC_NAMES)

    inc.insert(0, "inc_row_id", np.arange(1, len(inc) + 1))
    for o, h in INC_NAMES.items():
        inc[h] = inc[o]
    inc["death_any"] = ((inc["mortv"] == 1) | (inc["mortnv"] == 1)).astype(int)
    rec.insert(0, "rec_row_id", np.arange(1, len(rec) + 1))
    for o, h in REC_NAMES.items():
        rec[h] = rec[o]

    # ---- linkage on (age, sex, follow-up) ----
    key = lambda d: list(zip(d["edat"].round(6), d["sexe"], d["ts"].round(6)))
    inc["_k"] = key(inc)
    rec["_k"] = key(rec)
    kc_inc = inc["_k"].value_counts()
    kc_rec = rec["_k"].value_counts()
    uniq_inc = inc[inc["_k"].map(kc_inc) == 1]
    m = rec[rec["_k"].map(kc_rec) == 1].merge(uniq_inc[["_k", "inc_row_id"]], on="_k", how="inner")
    rec["linked_inc_row_id"] = rec["rec_row_id"].map(dict(zip(m["rec_row_id"], m["inc_row_id"])))
    inc["linked_artper_id"] = inc["inc_row_id"].map(dict(zip(m["inc_row_id"], m["id"])))
    n_dup_inc = int((inc["_k"].map(kc_inc) > 1).sum())
    n_dup_rec = int((rec["_k"].map(kc_rec) > 1).sum())
    link_ok = rec["linked_inc_row_id"].notna()
    lk = rec[link_ok].merge(inc, left_on="linked_inc_row_id", right_on="inc_row_id", suffixes=("", "_inc"))
    agree = {
        "hypertension": float((lk["htahc"] == lk["htahc_inc"]).mean()),
        "diabetes": float((lk["diabhc"] == lk["diabhc_inc"]).mean()),
        "hypercholesterolaemia": float((lk["hipercolhc"] == lk["hipercolhc_inc"]).mean()),
        "REGICOR": float((lk["regicor"].round(3) == lk["regicor_inc"].round(3)).mean()),
        "cerebrovascular event": float((lk["cer"] == lk["cer_inc"]).mean()),
        "PAD (ap) vs ABI < 0.9": float((lk["ap"] == (lk["itb"] < 0.9 - 1e-6).astype(int)).mean()),
    }
    # attach ABI to the reclassification table (linked rows only)
    rec = rec.merge(inc[["inc_row_id", "itb", "itb3"]].rename(columns={"inc_row_id": "linked_inc_row_id",
                                                                       "itb": "abi_linked",
                                                                       "itb3": "abi_cat3_linked"}),
                    on="linked_inc_row_id", how="left")
    inc = inc.drop(columns="_k")
    rec = rec.drop(columns="_k")

    # ---- checks and descriptives ----
    abi_tab = pd.crosstab(pd.cut(inc["itb"], [0, 0.9 - 1e-6, 1.0 - 1e-6, 1.4, 9], right=False,
                                 labels=["<0.9", "0.9-<1.0", "1.0-<1.4", ">=1.4"]), inc["itb3"]).reset_index()
    abi_tab.columns = ["ABI (itb) range"] + [f"itb3 = {c}" for c in abi_tab.columns[1:]]
    comp = []
    for nm, parts in [("mace", ["iam", "ictus", "mortv"]), ("ecv (ICE)", ["cor", "cer", "inter", "mortv"]),
                      ("event (morbimortality)", ["cor", "cer", "inter", "mortv", "mortnv"]),
                      ("morbi (morbidity)", ["cor", "cer", "inter"])]:
        col = nm.split()[0]
        u = inc[parts].max(axis=1)
        comp.append({"composite": nm, "candidate components": "+".join(parts), "n composite = 1": int(inc[col].sum()),
                     "n any component = 1": int(u.sum()),
                     "composite 1, no component": int(((inc[col] == 1) & (u == 0)).sum()),
                     "component 1, composite 0": int(((inc[col] == 0) & (u == 1)).sum())})
    comp = pd.DataFrame(comp)
    same_time = {"eventt == ecvt": bool((inc["eventt"] == inc["ecvt"]).all()),
                 "iamt == cert": bool((inc["iamt"] == inc["cert"]).all())}
    ev_inc = L.events_table(inc, [
        ("Myocardial infarction", "iam", "iamt"), ("Stroke", "ictus", "ictust"), ("Coronary disease", "cor", "cort"),
        ("Cerebrovascular event", "cer", "cert"), ("Vascular intervention", "inter", "intert"),
        ("MACE", "mace", "macet"), ("ICE (ecv)", "ecv", "ecvt"), ("Morbimortality (event)", "event", "eventt"),
        ("Morbidity (morbi)", "morbi", "morbit"), ("Vascular death (time = ts, inferred)", "mortv", "ts"),
        ("Non-vascular death (time = ts, inferred)", "mortnv", "ts"),
        ("Any death (time = ts, inferred)", "death_any", "ts")], "years")
    ev_rec = L.events_table(rec, [("Cardiac event (chd)", "chd", "chdt"), ("Cerebrovascular event", "cer", "cert")],
                            "years")
    abi_desc = pd.DataFrame([{"N": len(inc), "ABI non-missing": int(inc["itb"].notna().sum()),
                              "ABI < 0.9": int((inc["itb3"] == 2).sum()), "ABI 0.9-0.99": int((inc["itb3"] == 1).sum()),
                              "ABI 1.0-1.39": int((inc["itb3"] == 0).sum()),
                              "ABI >= 1.4": int((inc["itb"] >= 1.4).sum())}])

    common = [
        "ARTPER (from the literature, not stated in the files): population-based cohort of adults aged > 49 years "
        "from primary-care centres in Barcelona (Spain), baseline 2006-2008, Doppler ABI. Both files contain a "
        "`Labels` sheet; variable meanings and codes below are taken from it verbatim (Catalan/Spanish variable "
        "names).",
        "Linkage: no common ID. Exact match on (age `edat`, sex `sexe`, follow-up `ts`) links "
        f"{int(link_ok.sum())} of {len(rec)} reclassification rows uniquely to the incidence file "
        f"({n_dup_rec} reclassification rows and {n_dup_inc} incidence rows share a duplicated key and are left "
        "unlinked). Agreement of shared variables among linked rows: "
        + "; ".join(f"{k} {100 * v:.1f} %" for k, v in agree.items())
        + ". The reclassification file is a subset (people free of baseline CVD in that paper, per its design); "
        "the link is a reconstruction, not an official key.",
    ]
    notes_inc = common + [
        "ABI (`itb`) is a single value per person, max 1.379: people with ABI >= 1.4 are absent (excluded by design). "
        "`itb3` codes (Labels: 0 normal, 1 borderline, 2 low) correspond to 1.0-1.39, 0.90-0.99 and < 0.90; five "
        "people with itb stored as 0.89999998 (float rendering of 0.90) are coded borderline. Which leg/arm "
        "convention defines the single ABI (lower of two legs is the ARTPER convention) is not stated in the file.",
        "Composite outcome definitions are not given beyond the label. Reconstruction (table below): `mace` = MI or "
        "stroke or vascular death exactly; `ecv` (ICE) = coronary/cerebrovascular/intervention/vascular death "
        "except a few; `event` and `morbi` include events not in the listed components. Check the article before "
        "using composites. " + "; ".join(f"{k}: {v}" for k, v in same_time.items()) + ". `eventt` equals `ecvt` "
        "although `event` has more events than `ecv`; among the "
        f"{int(((inc['event'] == 1) & (inc['ecv'] == 0)).sum())} rows with event = 1 and ecv = 0, eventt = ts in "
        f"{100 * (inc.loc[(inc['event'] == 1) & (inc['ecv'] == 0), 'eventt'] == inc.loc[(inc['event'] == 1) & (inc['ecv'] == 0), 'ts']).mean():.0f} % "
        f"({int(inc.loc[(inc['event'] == 1) & (inc['ecv'] == 0), 'mortnv'].sum())} of them non-vascular deaths), "
        "i.e. the extra events end follow-up and ICE is censored there, so the shared time is consistent.",
        "Death: `mortv` (vascular) and `mortnv` (non-vascular) have no own time variable; `ts` (overall follow-up, "
        "years) is assumed to end at death for decedents (inferred). `death_any` = mortv or mortnv.",
        "No BP values, no PWV and no IMT are in either file (only diagnoses/drug flags and risk scores). ABI enters "
        "only as a single index (no ankle/brachial pressures), so ABI-pressure coupling cannot be examined.",
    ]
    notes_rec = common + [
        "No ABI value in this file; `ap` = baseline PAD (0 healthy, 1 PAD) and `claudi` = claudication. "
        "`abi_linked` / `abi_cat3_linked` are attached from the incidence file through the reconstructed link.",
        "Events: `chd` (incident cardiac event) and `cer` (incident cerebrovascular event) with times in years.",
    ]
    vm_inc = {"inc_row_id": {"kind": "derived", "meaning": "Row order in the incidence sheet"},
              "linked_artper_id": {"kind": "derived", "meaning": "`id` of the uniquely matched reclassification row"},
              "death_any": {"kind": "derived", "meaning": "mortv or mortnv"}}
    for o, h in INC_NAMES.items():
        lab, val = inc_lab.get(o, ("", ""))
        unit = "years" if ("Time" in lab or "years" in lab or o == "ts") else (
            "mg/dL" if "mg/dl" in lab else ("ratio" if o == "itb" else ""))
        vm_inc[o] = {"kind": "original", "meaning": f"{lab} {('[' + val + ']') if val else ''}".strip(), "unit": unit}
        vm_inc[h] = {"kind": "harmonised", "meaning": f"= `{o}`: {lab}", "unit": unit}
    vm_inc["female"]["meaning"] = "= `sexe`: 1 women, 0 men"
    vm_rec = {"rec_row_id": {"kind": "derived", "meaning": "Row order in the reclassification sheet"},
              "linked_inc_row_id": {"kind": "derived", "meaning": "Matched row of the incidence file (unique key)"},
              "abi_linked": {"kind": "derived", "meaning": "ABI (itb) from the linked incidence row", "unit": "ratio"},
              "abi_cat3_linked": {"kind": "derived", "meaning": "itb3 from the linked incidence row"}}
    for o, h in REC_NAMES.items():
        lab, val = rec_lab.get(o, ("", ""))
        unit = "years" if ("years" in lab or "Time" in lab) else ("kg/m2" if o == "imc" else "")
        vm_rec[o] = {"kind": "original", "meaning": f"{lab} {('[' + val + ']') if val else ''}".strip(), "unit": unit}
        vm_rec[h] = {"kind": "harmonised", "meaning": f"= `{o}`: {lab}", "unit": unit}
    vm_rec["female"]["meaning"] = "= `sexe`: 1 women, 0 men"

    L.write_parquet(inc, "artper_incidence")
    L.write_dictionary(inc, "artper_incidence", "ARTPER incidence file (PLoS One e0209163 S4), n = 3,146", [XLS_INC],
                       "parse_artper.py", vm_inc, notes=notes_inc, extra_sections=[
                           ("ABI availability (counts)", L.md_table(abi_desc)),
                           ("itb vs itb3 (decoding check)", L.md_table(abi_tab)),
                           ("Composite outcomes vs candidate components", L.md_table(comp)),
                           ("Events and follow-up (years)", L.md_table(ev_inc))])
    L.write_parquet(rec, "artper_reclass")
    L.write_dictionary(rec, "artper_reclass", "ARTPER reclassification file (PLoS One e0191283 S1), n = 2,716",
                       [XLSX_REC, XLS_INC], "parse_artper.py", vm_rec, notes=notes_rec,
                       extra_sections=[("Events and follow-up (years)", L.md_table(ev_rec))])
    print("artper_incidence", inc.shape, "artper_reclass", rec.shape, "linked", int(link_ok.sum()))
    print(ev_inc.to_string())


if __name__ == "__main__":
    main()
