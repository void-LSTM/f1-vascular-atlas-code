#!/usr/bin/env python3
"""analysis_l_bed_outcomes.py - F1 Stage 0 task l: vascular bed -> clinical outcome evidence from open data.

Paper F1 (multi-bed vascular atlas). ALL RESULTS ARE EXPLORATORY.
Open public data only (no HPP data, no liver phenotypes). Fills bed x outcome cells that the earlier Stage 0
drivers (a-k, i, i2, i3) left empty, and assembles every bed x outcome estimate (old and new) into one long table
with source path:line.

Part MR (two-sample MR, LD-clumped bed instruments)
---------------------------------------------------
- Exposures: the eight bed GWAS of analysis_i_mr (ASI, cIMT, carotid plaque, CRAE, CRVE, AVR, tortuosity, density)
  with the instrument sets written by analysis_i3_mr_ldclump.py (results/stage0/analysis/i3_mr_ldclump/
  instruments_ld.csv: LD r2 <= 0.001 within 10 Mb, 1000 Genomes EUR; sens_r2_instruments.csv: r2 <= 0.01). The
  instruments are read, not recomputed, so the univariable part does not need the LD panel.
- Outcomes not yet used for the beds (family L_beds_to_new_outcomes): CKD (CKDGen 2019 EA), log eGFRcrea (Stanzick
  2021), log UACR (Teumer 2019), atrial fibrillation (Nielsen 2018), abdominal aortic aneurysm (MVP EUR), large-artery
  and small-vessel ischaemic stroke (MEGASTROKE EUR, no UKB), PAD (Sakaue 2021, BBJ+UKB+FinnGen; i_mr used it only as
  a risk-factor outcome). Family L_beds_to_bp: SBP, DBP and PP (Keaton 2024) as continuous outcomes, the only
  genetic proxy for incident hypertension on disk (no hypertension GWAS is on disk).
- Positive controls of the new outcome files (family L_pc_sbp_to_new_outcomes): SBP (Keaton 2024, LD-clumped
  instruments of i3) -> each new outcome; checks allele orientation and units (expected: SBP raises CKD, UACR, AF,
  large-artery and small-vessel stroke and PAD, lowers eGFR; aneurysm direction not fixed: Burgess 2022 found
  DBP up and PP down).
- Sensitivity: the same bed pairs with the r2 <= 0.01 instrument sets (family L_beds_to_new_outcomes_r2sens).
- Multivariable MR bed + SBP -> new outcomes (family L_MV_bed_plus_sbp_to_new_outcomes) with joint LD clumping
  (analysis_i3_mr_ldclump.mvmr_ld, unchanged). Needs the LD panel external/ld_ref/1000g_eur; skipped with a note
  if the panel is missing.
- Estimators, harmonisation, retinal allele-label correction ('swap_alleles'), Steiger filtering, FDR (BH within
  family) and evidence classes: analysis_i_mr (imported, unchanged). analysis_i_mr.py and
  analysis_i3_mr_ldclump.py are imported and not modified.

Part OBS (open cohorts and NHANES; Cox models)
----------------------------------------------
- NHANES 1999-2004 (clean table of task f): ABI class and ABI per 0.1 lower -> heart-disease, cerebrovascular,
  kidney-disease (UCOD_LEADING 9, nephritis/nephrosis) and non-cardiovascular death; survey-weighted Cox, M1 and M2
  of task f (analysis_f_nhanes9904_abi.svy_cox, unchanged; cause-specific, other deaths censored).
- NHANES III (clean table of task g): Sokolow-Lyon and Cornell voltage per SD, LVH flags -> heart-disease and
  cerebrovascular death; M2 and M3 (+ SBP) of task g (analysis_g_nhanes3_ecg.svy_cox, unchanged).
- IMPACT-ABI (clean table of task d): baPWV (per SD, ABI >= 0.9) and ABI -> MI and stroke separately (task d
  modelled them only inside MACE); follow-up rule of task d (events after death censored at death).
- ARTPER (clean table of task e): ABI (< 0.9; per 0.1 lower; categories) -> MI, stroke, coronary disease,
  cerebrovascular event, vascular death, any death (task e modelled MACE and ICE only). Death times = overall
  follow-up `followup_y` (inferred in the parser).
- Small open outcome cohorts read from external/open_cohorts (raw supplementary files; read-only):
  oc22 retinal arteriolar / venular calibre (fluorescein angiography) and macrovascular death, 1:1 age/sex-matched
  case-control (conditional logistic regression on within-pair differences); oc20 carotid IMT and plaque -> death
  in non-dialysis CKD (Cox); oc06 ABI -> death and CVD death in haemodialysis (Cox); oc19 ABI -> death and CV death
  after acute MI (Cox); oc08 cfPWV -> death, CV death and return to dialysis (graft loss) after kidney
  transplantation (Cox); oc21 baPWV -> death in dialysis (Cox; n 61). oc06 (.xls) and oc21 (.sav) need the readers
  environment; they are read from CSV copies made with .venv-readers (command and SHA-256 in the report).

Part ASSEMBLE
-------------
One long table (bed_outcome_evidence_long.csv) of every bed x outcome estimate: earlier outputs (tasks d, e, f, g,
i3) and this task's outputs, each with source path:line (CSV line numbers; header = line 1; the files have no
embedded line breaks, checked).

Modes: --real (open public data); --part all|mr|obs|assemble.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_d_impact_abi as D  # noqa: E402
import analysis_e_artper as E  # noqa: E402
import analysis_f_nhanes9904_abi as F  # noqa: E402
import analysis_g_nhanes3_ecg as G  # noqa: E402
import analysis_i3_mr_ldclump as L3  # noqa: E402
import analysis_i_mr as I  # noqa: E402
import lib_mr as M  # noqa: E402
import lib_survival as ls  # noqa: E402
import s0an_common as C  # noqa: E402

TASK = "l_bed_outcomes"
HUB, GWAS = I.HUB, I.GWAS
I3_DIR = C.ANALYSIS / "i3_mr_ldclump"
OPEN = HUB / "external" / "open_cohorts"
SCRATCH = C.PAPER / "scratch" / "public-outcomes"
STATUS = "exploratory"
T0 = time.time()


def log(msg: str):
    print(f"[{_dt.datetime.now():%H:%M:%S} +{time.time() - T0:7.1f}s] {msg}", flush=True)


def fmt_ci(e, lo, hi, nd=3):
    if not all(np.isfinite([e, lo, hi])):
        return ""
    return f"{e:.{nd}g} [{lo:.{nd}g}, {hi:.{nd}g}]"


# ================================================================================================================
# GWAS specs of the new outcome files (schema of analysis_i_mr.SPECS)
# ================================================================================================================
_KID = dict(chr="Chr", pos="Pos_b37", snp="RSID", ea="Allele1", oa="Allele2", eaf="Freq1", beta="Effect",
            se="StdErr", p="P-value", n="n_total_sum")
_EGFR_NAMES = ["vid1", "rsid", "effect_allele", "other_allele", "n", "effect_allele_frequency", "beta",
               "standard_error", "p_value", "chromosome", "base_pair_location", "vid2", "rsid2", "odds_ratio",
               "ci_lower", "ci_upper"]          # the file header repeats 'variant_id' four times
NEW_SPECS = [
    dict(id="ckd", label="CKD (eGFRcrea < 60)", group="l_out", kind="bin",
         path="18a_CKD/CKD_overall_EA_JW_20180223_nstud23.dbgap.txt.gz", sep=" ", build="37", cols=_KID,
         whitespace=True, n_default=480_698, ancestry="EUR", ukb=None, units="log-odds", unit_label="per log-odds", adverse=1,
         source="Wuttke 2019 Nat Genet (CKDGen, European ancestry; CKD = eGFRcrea < 60), GCST008065 "
                "[CHECK: UK Biobank inclusion in this file]"),
    dict(id="uacr", label="UACR (log)", group="l_out", kind="cont",
         path="18b_UACR/formatted_20180517-UACR_overall-EA-nstud_18-SumMac_400.tbl.rsid.gz", sep=" ", build="37",
         cols=_KID, n_default=547_361, ancestry="EUR", ukb=True, units="log UACR (as published)",
         unit_label="per log unit", adverse=1,
         source="Teumer 2019 Nat Commun (CKDGen + UK Biobank, European ancestry), GCST008794"),
    dict(id="egfr", label="eGFRcrea (log)", group="l_out", kind="cont",
         path="10c_eGFR_creatinine/34272381-GCST90103634-EFO_0005208-Build37.f.tsv.gz", sep="\t", build="37",
         colnames=_EGFR_NAMES,
         cols=dict(chr="chromosome", pos="base_pair_location", snp="rsid", ea="effect_allele", oa="other_allele",
                   eaf="effect_allele_frequency", beta="beta", se="standard_error", p="p_value", n="n"),
         n_default=1_004_040, ancestry="EUR", ukb=True, units="log eGFRcrea (as published)",
         unit_label="per log unit", adverse=-1,
         source="Stanzick 2021 Nat Commun (CKDGen + UK Biobank, European ancestry), GCST90103634"),
    dict(id="af", label="Atrial fibrillation", group="l_out", kind="bin",
         path="20a_Atrial_fibrillation/nielsen-thorolfsdottir-willer-NG2018-AFib-gwas-summary-statistics.tbl.gz",
         sep="\t", build="37",
         cols=dict(chr="CHR", pos="POS_GRCh37", snp="rs_dbSNP147", ea="A2", oa="A1", eaf="Freq_A2",
                   beta="Effect_A2", se="StdErr", p="Pvalue"),
         n_default=1_030_836, ancestry="EUR", ukb=True, units="log-odds", unit_label="per log-odds", adverse=1,
         source="Nielsen 2018 Nat Genet (HUNT, deCODE, MGI, DiscovEHR, UK Biobank, AFGen), GCST006414; effect and "
                "frequency refer to A2 (checked: PITX2 rs2200733 T allele raises risk)"),
]
BEDS = I.BEDS
NEW_OUT = ["ckd", "egfr", "uacr", "af", "aaa_mvp", "las_mega", "svs_mega", "pad_sakaue"]
BP_OUT = ["sbp", "dbp", "pp"]


def _register():
    L3._register_ext_specs()
    for s in NEW_SPECS:
        I.SPEC[s["id"]] = s


def SPEC(i):
    return I.SPEC[i]


# ================================================================================================================
# File scans at instrument keys (analysis_i_mr.scan_worker; a variant with explicit column names for the eGFR file)
# ================================================================================================================
def scan_task(task: dict) -> dict:
    spec = task["spec"]
    if spec.get("whitespace"):
        # CKDGen CKD file: fixed-width padding inside space-separated rows (e.g. Freq1 '   1'), so a single-space
        # delimiter splits padded fields; read with runs of whitespace as the separator (pandas C engine, chunks)
        t0 = time.time()
        c = spec["cols"]
        include = list(dict.fromkeys(c.values()))
        rs, pos, pairs = set(task["rs"]), set(task["pos"]), set(task["pairs"])
        parts, nrows = [], 0
        for ch in pd.read_csv(task["path"], sep=r"\s+", usecols=include, dtype=str, chunksize=2_000_000,
                              na_values=I.NA_VALUES, keep_default_na=False):
            nrows += len(ch)
            pv = pd.to_numeric(ch[c["pos"]], errors="coerce")
            m = ch[c["snp"]].isin(rs) | pv.isin(pos)
            if m.any():
                parts.append(ch[m])
        raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=include)
        df = I.standardize(spec, raw)
        k = I.cpkey(df["chr"], df["pos"])
        keep = df["snp"].isin(rs).fillna(False) | k.isin(pairs).fillna(False)
        df = df[keep].reset_index(drop=True)
        return {"id": spec["id"], "mode": "keys", "df": df, "rows": nrows, "kept": len(df),
                "seconds": time.time() - t0, "maxrss_mb": I.maxrss_mb()}
    if not spec.get("colnames"):
        return I.scan_worker(task)
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    pa.set_cpu_count(task.get("threads", 2))
    t0 = time.time()
    c = spec["cols"]
    include = list(dict.fromkeys(c.values()))
    reader = pcsv.open_csv(
        pa.input_stream(str(task["path"]), compression="gzip"),
        read_options=pcsv.ReadOptions(block_size=32 << 20, column_names=spec["colnames"], skip_rows=1),
        parse_options=pcsv.ParseOptions(delimiter=spec["sep"]),
        convert_options=pcsv.ConvertOptions(include_columns=include, column_types=I._pa_types(spec),
                                            null_values=I.NA_VALUES, strings_can_be_null=True))
    rs_arr = pa.array(sorted(task["rs"]), pa.string())
    pos_arr = pa.array(sorted(task["pos"]), pa.float64())
    parts, nrows = [], 0
    for batch in reader:
        nrows += batch.num_rows
        m = pc.or_(pc.fill_null(pc.is_in(batch.column(c["snp"]), value_set=rs_arr), False),
                   pc.fill_null(pc.is_in(batch.column(c["pos"]), value_set=pos_arr), False))
        sub = batch.filter(m)
        if sub.num_rows:
            parts.append(sub)
    raw = pa.Table.from_batches(parts).to_pandas() if parts else pd.DataFrame({k: pd.Series(dtype="object")
                                                                                for k in include})
    df = I.standardize(spec, raw)
    k = I.cpkey(df["chr"], df["pos"])
    keep = df["snp"].isin(set(task["rs"])).fillna(False) | k.isin(set(task["pairs"])).fillna(False)
    df = df[keep].reset_index(drop=True)
    return {"id": spec["id"], "mode": "keys", "df": df, "rows": nrows, "kept": len(df),
            "seconds": time.time() - t0, "maxrss_mb": I.maxrss_mb()}


def scan_all(ids: list, u_rs: set, u_pairs: set, cache: Path, workers: int, stats_rows: list) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    u_pos = sorted({float(x.split(":")[1]) for x in u_pairs})
    hk = hashlib.sha1(("\n".join(sorted(u_rs)) + "|" + "\n".join(sorted(u_pairs))).encode()).hexdigest()[:12]
    out, todo = {}, []
    for i in ids:
        s = SPEC(i)
        p = GWAS / s["path"]
        st = p.stat()
        key = hashlib.sha1(json.dumps([i, str(p), st.st_size, int(st.st_mtime), hk, s.get("cols"),
                                       s.get("special"), s.get("colnames"), s.get("whitespace")], default=str).encode()).hexdigest()[:16]
        f = cache / f"{i}_keys_{key}.parquet"
        if f.exists():
            out[i] = pd.read_parquet(f)
            stats_rows.append({"id": i, "cached": True, "rows_kept": len(out[i])})
            continue
        if s["build"] == "38":
            t = dict(spec=s, path=str(p), mode="keys", threads=2, rs=sorted(u_rs))
        else:
            t = dict(spec=s, path=str(p), mode="keys", threads=2, rs=sorted(u_rs), pos=u_pos, pairs=sorted(u_pairs))
        todo.append((i, f, t))
    todo.sort(key=lambda x: -(GWAS / SPEC(x[0])["path"]).stat().st_size)
    if todo:
        log(f"scanning {len(todo)} GWAS files at {len(u_rs):,} rsIDs / {len(u_pairs):,} positions")
    res = I.run_pool(scan_task, [t for _, _, t in todo], workers)
    for (i, f, _), r in zip(todo, res):
        r["df"].to_parquet(f, index=False)
        out[i] = r["df"]
        stats_rows.append({"id": i, "cached": False, "rows_scanned": r["rows"], "rows_kept": r["kept"],
                           "seconds": round(r["seconds"], 1)})
        log(f"  {i}: {r['rows']:,} rows scanned, {r['kept']:,} kept, {r['seconds']:.1f}s")
    return out


def read_instruments(path: Path) -> dict:
    d = pd.read_csv(path, dtype={"chr": str, "snp": str, "ea": str, "oa": str, "ref_snp": str})
    d["pos"] = pd.to_numeric(d["pos"], errors="coerce").round().astype("Int64")
    for c in ("snp", "chr", "ea", "oa"):
        d[c] = d[c].astype("string")
    return {k: g.drop(columns=["exposure"]).reset_index(drop=True) for k, g in d.groupby("exposure")}


# ================================================================================================================
# Part MR
# ================================================================================================================
def run_mr(args, outdir: Path, rep, notes: list) -> dict:
    _register()
    t_mr = time.time()
    f_main, f_sens = I3_DIR / "instruments_ld.csv", I3_DIR / "sens_r2_instruments.csv"
    for f in (f_main, f_sens):
        rep.inputs.append({"id": f.name, "path": str(f.relative_to(HUB)), "bytes": f.stat().st_size,
                           "sha256": C.sha256(f)})
    inst = {k: v for k, v in read_instruments(f_main).items() if k in BEDS + ["sbp"]}
    inst_s = {k: v for k, v in read_instruments(f_sens).items() if k in BEDS + ["sbp"]}
    ninst = pd.DataFrame([{"exposure": k, "instruments_r2_0.001": len(inst.get(k, [])),
                           "instruments_r2_0.01": len(inst_s.get(k, []))} for k in BEDS + ["sbp"]])
    beds_ok = [b for b in BEDS if len(inst.get(b, [])) >= 3]
    beds_ok_s = [b for b in BEDS if len(inst_s.get(b, [])) >= 3]

    out_ids = NEW_OUT + BP_OUT
    avail = []
    for i in out_ids + BEDS:
        p = GWAS / SPEC(i)["path"]
        if p.exists():
            avail.append(i)
        else:
            notes.append(f"GWAS file missing, analyses with {i} skipped: {p.relative_to(HUB)}")
    u_rs, u_pairs = set(), set()
    for d in list(inst.values()) + list(inst_s.values()):
        u_rs |= set(d["snp"].dropna())
        ok = d["pos"].notna()
        u_pairs |= set(I.cpkey(d.loc[ok, "chr"], d.loc[ok, "pos"]))
    scan_stats = []
    ext = scan_all([i for i in dict.fromkeys(out_ids + BEDS) if i in avail], u_rs, u_pairs,
                   outdir / "_cache", args.workers, scan_stats)
    vmap = I.VariantMap()
    for d in list(inst.values()):
        vmap.add(d)
    for i, d in ext.items():
        if SPEC(i)["build"] == "37":
            vmap.add(d)
    for i, d in ext.items():
        if SPEC(i)["build"] == "38":
            vmap.fill_pos37(d)

    # registry with SHA-256 (cached by size and mtime)
    reg = []
    shaf = outdir / "_cache" / "sha256.json"
    shac = json.loads(shaf.read_text()) if shaf.exists() else {}
    for i in sorted(ext):
        s, p = SPEC(i), GWAS / SPEC(i)["path"]
        st = p.stat()
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        if key not in shac and not args.no_hash:
            shac[key] = C.sha256(p)
        reg.append({"id": i, "label": s["label"], "source": s["source"], "file": str(p.relative_to(HUB)),
                    "build": s["build"], "ancestry": s["ancestry"], "kind": s["kind"], "units": s["units"],
                    "UKB_in_sample": s.get("ukb"), "adverse_direction": s.get("adverse", 0),
                    "bytes": st.st_size, "sha256": shac.get(key)})
        rep.inputs.append({"id": i, "path": str(p.relative_to(HUB)), "bytes": st.st_size, "sha256": shac.get(key)})
    shaf.write_text(json.dumps(shac, indent=1))
    registry = pd.DataFrame(reg)
    msha = I.manifest_sha()
    registry["manifest_sha256_match"] = [None if r["sha256"] is None else msha.get(r["file"]) == r["sha256"]
                                         for _, r in registry.iterrows()]

    # EAF concordance of the new files against Keaton SBP (allele-label check)
    eafqc = I.eaf_qc(ext["sbp"], {k: ext[k] for k in ["ckd", "uacr", "egfr", "af"] if k in ext}) \
        if "sbp" in ext else pd.DataFrame()

    # ---------------- univariable MR ----------------
    def ujob(eid, oid, fam, src):
        e, o = SPEC(eid), SPEC(oid)
        return dict(e={k: e[k] for k in ("id", "label", "kind", "unit_label", "units", "ukb") if k in e} |
                    {"scale": e.get("scale", 1.0)},
                    o={k: o[k] for k in ("id", "label", "kind", "unit_label", "units", "ukb") if k in o},
                    inst=I.with_ids(src[eid]), inst10=None, out=ext[oid], family=fam, seed=args.seed,
                    n_boot=args.n_boot, n_boot_mode=min(args.n_boot, 1000), drop_pal=not (I.eur(e) and I.eur(o)))

    fams = [("L_beds_to_new_outcomes", b, o, inst) for b in beds_ok for o in NEW_OUT]
    fams += [("L_beds_to_bp", b, o, inst) for b in beds_ok for o in BP_OUT]
    fams += [("L_pc_sbp_to_new_outcomes", "sbp", o, inst) for o in NEW_OUT]
    fams += [("L_beds_to_new_outcomes_r2sens", b, o, inst_s) for b in beds_ok_s for o in NEW_OUT]
    fams = [x for x in fams if x[2] in ext and x[1] in x[3] and len(x[3][x[1]])]
    log(f"univariable MR: {len(fams)} pairs on {args.mr_workers} workers")
    res = I.run_pool(I.mr_pair_worker, [ujob(e, o, f, src) for f, e, o, src in fams], args.mr_workers)
    summ = pd.DataFrame([r["summary"] for r in res])
    longd = pd.DataFrame([x for r in res for x in r["long"]])
    snps = pd.concat([r["snps"] for r in res if r.get("snps") is not None], ignore_index=True)
    for r in res:
        if r.get("trace"):
            notes.append(f"MR error {r['summary']['exposure']}->{r['summary']['outcome']}: "
                         f"{r['summary'].get('error')}")
    summ["ivw_q"] = np.nan
    for fam, g in summ.groupby("family"):
        summ.loc[g.index, "ivw_q"] = I.bh(g["ivw_p"].values)
    summ["evidence"] = summ.apply(I.classify, axis=1)
    adv = {k: SPEC(k).get("adverse", 0) for k in set(summ["outcome"])}
    summ["outcome_adverse_direction"] = summ["outcome"].map(adv)
    summ["ivw_adverse_oriented"] = summ["ivw"] * summ["outcome_adverse_direction"].replace(0, np.nan)
    binm = summ["outcome"].map(lambda k: SPEC(k)["kind"] == "bin")
    summ["OR_ivw"] = np.where(binm, np.exp(summ["ivw"]), np.nan)
    summ["OR_lci"] = np.where(binm, np.exp(summ["ivw_lci"]), np.nan)
    summ["OR_uci"] = np.where(binm, np.exp(summ["ivw_uci"]), np.nan)
    summ["IVW [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(summ["ivw"], summ["ivw_lci"], summ["ivw_uci"])]
    summ["clumping"] = np.where(summ["family"].str.endswith("r2sens"),
                                "LD r2<=0.01 within 10 Mb (1000G EUR; i3 instrument set)",
                                "LD r2<=0.001 within 10 Mb (1000G EUR; i3 instrument set)")
    summ["status"] = STATUS
    log(f"univariable MR done ({time.time() - t_mr:.0f}s since start of part MR)")

    # ---------------- MVMR bed + SBP (joint LD clumping) ----------------
    mv = pd.DataFrame()
    ld_ok = all((L3.LDREF_DIR / f).exists() for f in ("g1000_eur.bed", "g1000_eur.bim", "g1000_eur.fam"))
    if not ld_ok:
        notes.append("LD panel external/ld_ref/1000g_eur not on disk: MVMR bed + SBP not run")
    elif "sbp" not in ext:
        notes.append("Keaton SBP file missing: MVMR bed + SBP not run")
    else:
        log("loading the LD panel for MVMR")
        ldref = L3.LDRef(L3.LDREF_PREFIX)
        inst_mv = {}
        for k in beds_ok + ["sbp"]:
            d = inst[k].copy()
            mt = ldref.match(d[["snp", "chr", "pos", "ea", "oa"]])
            d["ref_idx"] = mt["ref_idx"].values
            if (d["ref_idx"] < 0).any():
                notes.append(f"MVMR: {int((d['ref_idx'] < 0).sum())} {k} instruments not re-matched to the panel")
            inst_mv[k] = d
        keys = L3.refkeys(inst_mv)
        rows = []
        for b in beds_ok:
            for o in NEW_OUT:
                if o not in ext or b not in ext:
                    continue
                dp = not (I.eur(SPEC(b)) and I.eur(SPEC("sbp")) and I.eur(SPEC(o)))
                try:
                    rows += L3.mvmr_ld([b, "sbp"], o, "L_MV_bed_plus_sbp_to_new_outcomes", ext, inst_mv, keys,
                                       ldref, dp)
                except Exception as e:                                # pragma: no cover - reported in the table
                    rows.append({"family": "L_MV_bed_plus_sbp_to_new_outcomes", "exposures": f"{b}+sbp",
                                 "outcome": o, "error": f"{type(e).__name__}: {e}"})
        mv = pd.DataFrame(rows)
        if "p" in mv.columns:
            mv["q_within_family_term"] = np.nan
            for term, g in mv.dropna(subset=["p"]).groupby("term"):
                mv.loc[g.index, "q_within_family_term"] = I.bh(g["p"].values)
        mv["status"] = STATUS
        log(f"MVMR: {len(mv)} rows")

    # ---------------- report ----------------
    show = ["family", "exposure", "outcome", "nsnp", "ivw", "ivw_lci", "ivw_uci", "ivw_p", "ivw_q", "OR_ivw",
            "OR_lci", "OR_uci", "egger_intercept_p", "wmedian", "wmedian_p", "wmode_p", "Q_p",
            "steiger_correct_direction", "ivw_steiger", "ivw_steiger_p", "ukb_overlap", "evidence"]
    rep.h("Part MR: bed -> outcomes not used before (LD-clumped instruments of i3)")
    rep.table("mr_instrument_counts", ninst, "Bed and SBP instruments read from i3")
    rep.table("mr_gwas_registry", registry, "Outcome and exposure GWAS (SHA-256; manifest match)", max_rows=None)
    if len(eafqc):
        rep.table("mr_eaf_concordance_new_files", eafqc, "EAF concordance of the new outcome files with Keaton SBP")
    for fam, cap in [("L_pc_sbp_to_new_outcomes", "Positive controls: SBP (per 10 mmHg) -> new outcome files"),
                     ("L_beds_to_new_outcomes", "Beds -> new outcomes (IVW random effects; LD r2 <= 0.001)"),
                     ("L_beds_to_bp", "Beds -> SBP, DBP, PP (mmHg; proxy for incident hypertension)"),
                     ("L_beds_to_new_outcomes_r2sens", "Sensitivity: beds -> new outcomes, LD r2 <= 0.01")]:
        rep.table(f"mr_{fam}", summ.loc[summ["family"] == fam, [c for c in show if c in summ.columns]],
                  cap, max_rows=None)
    if len(mv):
        mshow = [c for c in ["exposures", "outcome", "term", "nsnp", "estimate", "lower", "upper", "p",
                             "q_within_family_term", "conditional_F", "Q_A_p", "ivw_same_snps", "error"]
                 if c in mv.columns]
        rep.table("mr_mvmr_bed_plus_sbp", mv[mshow], "MVMR bed + SBP (joint LD clumping) -> new outcomes",
                  max_rows=None)
    summ.to_csv(outdir / "mr_all_pairs_summary.csv", index=False)
    longd.to_csv(outdir / "mr_all_methods_long.csv", index=False)
    snps.to_csv(outdir / "mr_harmonised_snps.csv.gz", index=False, compression="gzip")
    if len(mv):
        mv.to_csv(outdir / "mr_mvmr.csv", index=False)
    pd.DataFrame(scan_stats).to_csv(outdir / "mr_scan_stats.csv", index=False)
    return {"summ": summ, "mv": mv}


# ================================================================================================================
# Part OBS
# ================================================================================================================
def cox_row(d: pd.DataFrame, time_col: str, event_col: str, expo: str, covs: list) -> dict:
    cols = [expo] + covs
    s = d.dropna(subset=cols + [time_col, event_col])
    f = ls.coxph(s[time_col].values.astype(float), s[event_col].values.astype(float), s[cols].values.astype(float),
                 names=cols, robust=True)
    tm, tr = f.table("model").iloc[0], f.table("robust").iloc[0]
    ev = int(s[event_col].sum())
    return {"n": len(s), "events": ev, "epv": ev / len(cols), "log_hr": tm["coef"], "se": tm["se"],
            "hr": tm["exp_coef"], "lower": tm["exp_lower"], "upper": tm["exp_upper"], "p": tm["p"],
            "robust_se": tr["se"], "hr_lower_robust": tr["exp_lower"], "hr_upper_robust": tr["exp_upper"],
            "converged": f.converged}


def clogit_pairs(dx: np.ndarray) -> dict:
    """Conditional logistic regression for 1:1 matched pairs: maximise sum log expit(dx @ b), dx = case - control."""
    dx = np.asarray(dx, float)
    if dx.ndim == 1:
        dx = dx[:, None]
    b = np.zeros(dx.shape[1])
    for _ in range(100):
        eta = dx @ b
        pr = 1 / (1 + np.exp(-eta))
        g = dx.T @ (1 - pr)
        H = (dx * (pr * (1 - pr))[:, None]).T @ dx
        step = np.linalg.solve(H, g)
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    eta = dx @ b
    pr = 1 / (1 + np.exp(-eta))
    H = (dx * (pr * (1 - pr))[:, None]).T @ dx
    V = np.linalg.inv(H)
    se = np.sqrt(np.diag(V))
    z = b / se
    return {"coef": b, "se": se, "p": 2 * stats.norm.sf(np.abs(z))}


def obs_nhanes9904(token, rep) -> pd.DataFrame:
    raw, meta = C.load_real(F.TABLE, token)
    rep.inputs.append({"id": F.TABLE, "path": str(Path(meta["path"]).relative_to(HUB)), "sha256": meta["sha256"]})
    d, _ = F.prepare(raw)
    uc = raw["UCOD_LEADING"]
    dead = raw["death_all"]
    for k, codes in [("death_heart_l", [1]), ("death_cerebro_l", [5]), ("death_kidney_l", [9])]:
        d[k] = np.where(raw["mort_eligible"] == 1, ((dead == 1) & uc.isin(codes)).astype(float), np.nan)
    d["death_noncvd_l"] = np.where(raw["mort_eligible"] == 1, ((dead == 1) & ~uc.isin([1, 5])).astype(float), np.nan)
    des = F.design(d)
    base = ((d["age_y"] >= 40) & (d["abi_class"] != "missing") & (d["mort_eligible"] == 1)
            & d[F.M2].notna().all(axis=1)).values
    expos = {"ABI class (ref 1.00-1.40)": ["abi_lt09", "abi_0909", "abi_gt14"],
             "ABI per 0.1 lower (highest leg <= 1.4)": ["abi_per01_lower"]}
    rows = []
    for oc, olab in [("death_heart_l", "Heart-disease death (UCOD 1)"),
                     ("death_cerebro_l", "Cerebrovascular death (UCOD 5)"),
                     ("death_kidney_l", "Kidney-disease death (UCOD 9, nephritis/nephrosis)"),
                     ("death_noncvd_l", "Non-cardiovascular death (not UCOD 1 or 5)")]:
        for elab, ecols in expos.items():
            for alab, covs in [("M1", F.M1), ("M2 primary", F.M2)]:
                for r in F.svy_cox(d, des, base, oc, ecols, covs):
                    rows.append({"cohort": "NHANES 1999-2004", "outcome": olab, "exposure": elab,
                                 "adjustment": alab, **r})
    return pd.DataFrame(rows)


def obs_nhanes3(token, rep) -> pd.DataFrame:
    raw, meta = C.load_real(G.TABLE, token)
    rep.inputs.append({"id": G.TABLE, "path": str(Path(meta["path"]).relative_to(HUB)), "sha256": meta["sha256"]})
    d = G.prepare(raw)
    uc, dead = raw["UCOD_LEADING"], raw["death_all"]
    elig = raw["mort_eligible"] == 1
    d["death_heart_l"] = np.where(elig, ((dead == 1) & (uc == 1)).astype(float), np.nan)
    d["death_cerebro_l"] = np.where(elig, ((dead == 1) & (uc == 5)).astype(float), np.nan)
    des = G.design(d)
    dom = ((d["age_y"] >= 40) & (d["has_ecg"] == 1) & (d["ecg_voltage_valid"] == 1) & (d["mort_eligible"] == 1)
           & d[G.M2 + ["sbp_mmhg", "sokolow_lyon_mv", "cornell_voltage_mv"]].notna().all(axis=1)).values
    for c in ("sokolow_lyon_mv", "cornell_voltage_mv"):
        sd = ls.weighted_sd(d.loc[dom, c].values, des.weights[dom])
        d[c + "_sd"] = d[c] / sd
    rows = []
    for oc, olab in [("death_heart_l", "Heart-disease death (UCOD 1)"),
                     ("death_cerebro_l", "Cerebrovascular death (UCOD 5)")]:
        for v in ["sokolow_lyon_mv_sd", "cornell_voltage_mv_sd", "lvh_sokolow", "lvh_cornell_voltage"]:
            for alab, covs in [("M2 primary", G.M2), ("M3 (+ SBP)", G.M2 + ["sbp10"])]:
                r, _ = G.svy_cox(d, des, dom, oc, [v] + covs, [v])
                rows += [{"cohort": "NHANES III", "outcome": olab, "exposure": v, "adjustment": alab, **x} for x in r]
    return pd.DataFrame(rows)


def obs_impact(token, rep) -> pd.DataFrame:
    raw, meta = C.load_real(D.TABLE, token)
    rep.inputs.append({"id": D.TABLE, "path": str(Path(meta["path"]).relative_to(HUB)), "sha256": meta["sha256"]})
    d, _ = D.prepare(raw)
    for k in ("mi", "stroke"):
        d[f"{k}_time_y"] = np.maximum(d[f"{k}_time_days"], 0.5) / 365.25
    base = (d["age_y"] >= 20).values
    abi_ok = (d["abi_min"] >= 0.9).values
    covs = D.ADJ["primary (age, sex, HD, DM, eGFR-JSN)"]
    prim = base & abi_ok & d[["bapwv_max_cms"] + covs].notna().all(axis=1).values
    for c, nm in [("bapwv_max_cms", "bapwv_max_perSD"), ("bapwv_mean_cms", "bapwv_mean_perSD")]:
        d[nm] = (d[c] - d.loc[prim, c].mean()) / d.loc[prim, c].std(ddof=1)
    rows = []
    for oc, olab in [("mi", "MI (events after death censored)"), ("stroke", "Stroke (events after death censored)")]:
        for ex, elab, m in [("bapwv_max_perSD", "baPWV max (per SD)", base & abi_ok),
                            ("bapwv_mean_perSD", "baPWV mean (per SD)", base & abi_ok),
                            ("abi_per01_lower", "ABI min (per 0.1 lower)", base),
                            ("abi_lt09", "ABI min < 0.9", base)]:
            for alab, cv in [("primary (age, sex, HD, DM, eGFR-JSN)", covs), ("age + sex", ["age10", "female"])]:
                r = D.fit(d, m, oc, ex, cv)
                rows.append({"cohort": "IMPACT-ABI", "outcome": olab, "exposure": elab, "adjustment": alab,
                             "analysis_set": "ABI>=0.9" if ex.startswith("bapwv") else "all", **r})
    return pd.DataFrame(rows)


def obs_artper(token, rep) -> pd.DataFrame:
    extra = ["mi", "mi_time_y", "stroke", "stroke_time_y", "coronary_disease", "coronary_disease_time_y",
             "cerebrovascular_event", "cerebrovascular_event_time_y", "vascular_death", "death_any", "followup_y"]
    raw, meta = C.load_real(E.TABLE, token)
    rep.inputs.append({"id": E.TABLE, "path": str(Path(meta["path"]).relative_to(HUB)), "sha256": meta["sha256"]})
    d = E.prepare(raw)
    for c in extra:
        d[c] = raw[c].values
    for o, t in [("mi", "mi_time_y"), ("stroke", "stroke_time_y"), ("coronary_disease", "coronary_disease_time_y"),
                 ("cerebrovascular_event", "cerebrovascular_event_time_y"), ("vascular_death", "followup_y"),
                 ("death_any", "followup_y")]:
        d[f"{o}_t"] = np.maximum(d[t].astype(float), 0.5 / 365.25)
    rows = []
    labs = {"mi": "MI", "stroke": "Stroke", "coronary_disease": "Coronary disease",
            "cerebrovascular_event": "Cerebrovascular event", "vascular_death": "Vascular death (time = ts, inferred)",
            "death_any": "Any death (time = ts, inferred)"}
    for o, olab in labs.items():
        for elab, ecols in [("ABI < 0.9", ["abi_lt09"]), ("ABI per 0.1 lower", ["abi_per01_lower"]),
                            ("ABI category (ref normal)", ["abi_borderline", "abi_low"])]:
            for alab in ("M1 age + sex", "M2 primary"):
                for r in E.fit(d, o, ecols, E.ADJ[alab]):
                    rows.append({"cohort": "ARTPER", "outcome": olab, "exposure": elab, "adjustment": alab, **r})
    return pd.DataFrame(rows)


def obs_small(rep, scratch: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, desc = [], []

    def add_input(p: Path, note=""):
        rel = str(p.relative_to(HUB)) if str(p).startswith(str(HUB)) else str(p)
        rep.inputs.append({"id": p.name, "path": rel, "bytes": p.stat().st_size, "sha256": C.sha256(p), "note": note})

    # --- oc22 retinal calibre (fluorescein angiography), 1:1 matched case-control of macrovascular death
    p = OPEN / "oc22_Retinal_calibre_mortality_pone0266423" / "pone.0266423.s001.xlsx"
    if p.exists():
        add_input(p)
        r = pd.read_excel(p, sheet_name="data")
        r.columns = [str(c).strip() for c in r.columns]
        g = r.groupby("group-id")
        ok = g["status"].agg(lambda s: sorted(s.tolist()) == [0, 1])
        r = r[r["group-id"].isin(ok[ok].index)]
        desc.append({"cohort": "oc22 retinal FA calibre (Vienna)", "n": len(r), "pairs": r["group-id"].nunique(),
                     "cases_deceased": int(r["status"].sum()),
                     "note": "1:1 matched by age and sex (paper design); calibre from fluorescein angiography, not "
                             "fundus CRAE/CRVE; units of d_artery / d_vein as in the file [CHECK: um]"})
        for v, lab in [("d_artery", "Retinal arteriolar diameter (FA)"), ("d_vein", "Retinal venular diameter (FA)"),
                       ("av_ratio", "Retinal AV ratio (FA)")]:
            sd = float(r[v].std(ddof=1))
            cs = r[r["status"] == 1].set_index("group-id")
            co = r[r["status"] == 0].set_index("group-id")
            for alab, covs in [("matched pairs (age, sex by design)", []),
                               ("matched pairs + diabetes + age difference", ["diabetes", "age_at_examination"])]:
                cols = [v] + covs
                dx = (cs[cols] - co.loc[cs.index, cols]).dropna()
                dx[v] = dx[v] / sd
                fit = clogit_pairs(dx.values)
                b, se = fit["coef"][0], fit["se"][0]
                rows.append({"cohort": "oc22 retinal FA calibre (Vienna)", "outcome": "Macrovascular death "
                             "(case-control)", "exposure": f"{lab} per SD ({sd:.3g})", "adjustment": alab,
                             "model": "conditional logistic (1:1)", "n": 2 * len(dx), "events": len(dx),
                             "estimate_type": "OR", "estimate": np.exp(b), "lower": np.exp(b - 1.96 * se),
                             "upper": np.exp(b + 1.96 * se), "p": fit["p"][0]})
    # --- oc20 fetuin-A / carotid IMT / plaque -> death (non-dialysis CKD + controls), 36 months
    p = OPEN / "oc20_FetuinA_CAC_carotid_mortality_dataverse" / "fetuinA_data.tab"
    if p.exists():
        add_input(p)
        r = pd.read_csv(p, sep="\t")
        r["cimt01"] = r["CIMT"] / 0.1
        r["plaque_yes"] = (r["Plaques"] == 2).astype(float)
        r["age10"] = r["age"] / 10
        r["male"] = (r["sex"] == 1).astype(float)
        r["gfr10"] = r["GFR"] / 10
        r["t_m"] = r["Time.to.death"].astype(float)
        desc.append({"cohort": "oc20 fetuin-A CKD (Minia)", "n": len(r), "deaths": int(r["Overall.Mortality"].sum()),
                     "cv_deaths": int(r["CV.mortality"].sum()), "note": "Time.to.death in months (max 36); "
                     "'Plaques' 2 = present, sex 1 = male assumed [CHECK: codebook in the .spv]; CV.mortality "
                     "missing for the 25 controls"})
        for ev, olab in [("Overall.Mortality", "All-cause death (36 months)"), ("CV.mortality", "CV death")]:
            for ex, elab in [("cimt01", "Carotid IMT per 0.1 mm"), ("plaque_yes", "Carotid plaque (yes vs no)")]:
                for alab, covs in [("age + sex", ["age10", "male"]), ("age + sex + GFR (CKD only)",
                                                                       ["age10", "male", "gfr10"])]:
                    if ev == "CV.mortality" and len(covs) > 2:
                        continue
                    rr = cox_row(r, "t_m", ev, ex, covs)
                    rows.append({"cohort": "oc20 fetuin-A CKD (Minia)", "outcome": olab, "exposure": elab,
                                 "adjustment": alab, "model": "Cox", "estimate_type": "HR", **rr,
                                 "estimate": rr["hr"]})
    # --- oc06 haemodialysis ABI -> death (CSV copy of the .xls, made with .venv-readers)
    p = scratch / "oc06_hd_abi.csv"
    src = OPEN / "oc06_HD_ABI_mortality_pone0128968" / "pone.0128968.s001.xls"
    if p.exists() and src.exists():
        add_input(src)
        add_input(p, "CSV copy of the .xls (sheet 1) made with .venv-readers pandas.read_excel")
        r = pd.read_csv(p)
        r["abi_min"] = r[["RABI", "LABI"]].min(axis=1)
        r["abi_lt09"] = (r["abi_min"] < 0.9).astype(float)
        r["abi_per01_lower"] = -r["abi_min"] / 0.1
        r["age10"] = r["age"] / 10
        r["t_y"] = r["time_survival"] / 12
        desc.append({"cohort": "oc06 haemodialysis ABI (Taipei)", "n": len(r), "deaths": int(r["death_all"].sum()),
                     "cv_deaths": int(r["death_CVD"].sum()), "abi_lt09": int(r["abi_lt09"].sum()),
                     "note": "time_survival in months (max 80); SEX coding not documented in the file"})
        for ev, olab in [("death_all", "All-cause death"), ("death_CVD", "CVD death")]:
            for ex, elab in [("abi_lt09", "ABI min < 0.9"), ("abi_per01_lower", "ABI min per 0.1 lower")]:
                for alab, covs in [("age + sex", ["age10", "SEX"]),
                                   ("age + sex + DM + prior CVD + dialysis vintage", ["age10", "SEX", "DM", "CVD",
                                                                                      "HD_time"])]:
                    rr = cox_row(r, "t_y", ev, ex, covs)
                    rows.append({"cohort": "oc06 haemodialysis ABI (Taipei)", "outcome": olab, "exposure": elab,
                                 "adjustment": alab, "model": "Cox", "estimate_type": "HR", **rr,
                                 "estimate": rr["hr"]})
    # --- oc19 acute MI, ABI -> death
    p = OPEN / "oc19_AMI_ABIdevice_mortality_pone0245860" / "pone.0245860.s001.csv"
    if p.exists():
        add_input(p)
        r = pd.read_csv(p)
        r["abi_lt09"] = (r["lowabi"] < 0.9).astype(float)
        r["abi_per01_lower"] = -r["lowabi"] / 0.1
        r["age10"] = r["age"] / 10
        r["t_y"] = np.maximum(r["followperiod"], 0.5) / 12
        desc.append({"cohort": "oc19 acute MI ABI (Kaohsiung)", "n": len(r), "deaths": int(r["death"].sum()),
                     "cv_deaths": int(r["cvdeath"].sum()), "abi_lt09": int(r["abi_lt09"].sum()),
                     "note": "followperiod in months (median 98 per catalogue); lowabi = lower-leg ABI (assumed)"})
        for ev, olab in [("death", "All-cause death"), ("cvdeath", "CV death")]:
            for ex, elab in [("abi_lt09", "ABI < 0.9"), ("abi_per01_lower", "ABI per 0.1 lower")]:
                for alab, covs in [("age + sex", ["age10", "sex"]), ("age + sex + DM + HT", ["age10", "sex", "dm",
                                                                                            "ht"])]:
                    rr = cox_row(r, "t_y", ev, ex, covs)
                    rows.append({"cohort": "oc19 acute MI ABI (Kaohsiung)", "outcome": olab, "exposure": elab,
                                 "adjustment": alab, "model": "Cox", "estimate_type": "HR", **rr,
                                 "estimate": rr["hr"]})
    # --- oc08 kidney transplant cfPWV -> death, CV death, return to dialysis
    p = OPEN / "oc08_KTx_cfPWV_pone0195928" / "pone.0195928.s001.xlsx"
    if p.exists():
        add_input(p)
        r = pd.read_excel(p, sheet_name="Sheet1")
        r["pwv_sd"] = (r["VOPCor1"] - r["VOPCor1"].mean()) / r["VOPCor1"].std(ddof=1)
        sd = float(r["VOPCor1"].std(ddof=1))
        r["age10"] = r["AgeR"] / 10
        r["male"] = (r["Sexe.cl"] == 0).astype(float)            # [CHECK] coding of Sexe.cl (SexeD = donor)
        r["sbp10"] = r["PAS1"] / 10
        r["t_surv"] = np.maximum(r["Durée.Suivi.years.Survie"], 1 / 365.25)
        r["t_dial"] = np.maximum(r["Durée.Suivi.years.DialyseEtGlobale"], 1 / 365.25)
        desc.append({"cohort": "oc08 kidney transplant cfPWV (France)", "n": len(r), "deaths": int(r["Décès"].sum()),
                     "cv_deaths": int(r["decesCV"].sum()), "return_to_dialysis": int(r["Dialyse"].sum()),
                     "note": f"VOPCor1 = cfPWV at month 3 (m/s; SD {sd:.3g}) [CHECK: French labels]; "
                             "Sexe.cl coding not documented (0 assumed male)"})
        for ev, t, olab in [("Décès", "t_surv", "All-cause death"), ("decesCV", "t_surv", "CV death"),
                            ("Dialyse", "t_dial", "Return to dialysis (graft loss)")]:
            for alab, covs in [("age + sex", ["age10", "male"]), ("age + sex + SBP", ["age10", "male", "sbp10"])]:
                rr = cox_row(r, t, ev, "pwv_sd", covs)
                rows.append({"cohort": "oc08 kidney transplant cfPWV (France)", "outcome": olab,
                             "exposure": f"cfPWV per SD ({sd:.3g} m/s)", "adjustment": alab, "model": "Cox",
                             "estimate_type": "HR", **rr, "estimate": rr["hr"]})
    # --- oc21 dialysis baPWV -> death (CSV copy of selected .sav columns, made with .venv-readers)
    p = scratch / "oc21_dialysis_pwv.csv"
    src = OPEN / "oc21_Dialysis_centralBP_PWV_pone0250653" / "pone.0250653.s001.sav"
    if p.exists() and src.exists():
        add_input(src)
        add_input(p, "CSV copy of 13 columns of the .sav made with .venv-readers pandas.read_spss")
        r = pd.read_csv(p)
        r["death"] = (r["Death"] != "Live").astype(float)
        r["cvd_death"] = (r["CVDDEATH"] == "Yes").astype(float)
        r["bapwv"] = r[["PREBASERPWV1", "PREBASELPWV1"]].mean(axis=1)
        sd = float(r["bapwv"].std(ddof=1))
        r["bapwv_sd"] = (r["bapwv"] - r["bapwv"].mean()) / sd
        r["age10"] = r["age"] / 10
        r["male"] = (r["SEx"] == "M").astype(float)
        r["t_y"] = r["FU"] / 12
        desc.append({"cohort": "oc21 dialysis baPWV (Korea)", "n": len(r), "deaths": int(r["death"].sum()),
                     "cv_deaths": int(r["cvd_death"].sum()), "note": f"FU in months; baPWV = mean of sides, "
                     f"pre-dialysis (cm/s; SD {sd:.4g})"})
        rr = cox_row(r, "t_y", "death", "bapwv_sd", ["age10", "male"])
        rows.append({"cohort": "oc21 dialysis baPWV (Korea)", "outcome": "All-cause death",
                     "exposure": f"baPWV per SD ({sd:.4g} cm/s)", "adjustment": "age + sex", "model": "Cox",
                     "estimate_type": "HR", **rr, "estimate": rr["hr"]})
    return pd.DataFrame(rows), pd.DataFrame(desc)


def run_obs(args, outdir: Path, rep, token, notes: list) -> dict:
    out = {}
    rep.h("Part OBS: open-cohort outcome models not run before")
    for name, fn, cap in [("obs_nhanes9904_abi_cause", obs_nhanes9904,
                           "NHANES 1999-2004: ABI -> cause-specific death (survey-weighted Cox, design-based t CIs)"),
                          ("obs_nhanes3_ecg_cause", obs_nhanes3,
                           "NHANES III: ECG LV voltage -> heart-disease and cerebrovascular death"),
                          ("obs_impact_mi_stroke", obs_impact, "IMPACT-ABI: baPWV and ABI -> MI and stroke"),
                          ("obs_artper_components", obs_artper, "ARTPER: ABI -> MACE components and deaths")]:
        try:
            t = fn(token, rep)
        except Exception as e:                                        # pragma: no cover - reported
            notes.append(f"{name} failed: {type(e).__name__}: {e}")
            continue
        out[name] = t
        t.to_csv(outdir / f"{name}.csv", index=False)       # needed by assemble() before the report is written
        rep.table(name, t, cap, max_rows=None)
    t, desc = obs_small(rep, Path(args.scratch))
    t.to_csv(outdir / "obs_small_cohorts.csv", index=False)
    out["obs_small_cohorts"] = t
    rep.table("obs_small_cohorts_description", desc, "Small open outcome cohorts: size and events", max_rows=None)
    rep.table("obs_small_cohorts", t, "Small open outcome cohorts: models (Cox HR or conditional OR)", max_rows=None)
    return out


# ================================================================================================================
# Part ASSEMBLE: one long table of every bed x outcome estimate with source path:line
# ================================================================================================================
def read_lines(path: Path) -> pd.DataFrame:
    """CSV with its physical line numbers (header = line 1); refuses files with embedded line breaks."""
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        hdr = next(rd)
        recs, lines = [], []
        for row in rd:
            recs.append(row)
            lines.append(rd.line_num)
    nphys = sum(1 for _ in open(path))
    if nphys != len(recs) + 1:
        raise ValueError(f"{path}: embedded line breaks ({nphys} lines, {len(recs) + 1} records)")
    d = pd.DataFrame(recs, columns=hdr)
    d["_line"] = lines
    for c in d.columns:
        if c != "_line":
            d[c] = pd.to_numeric(d[c], errors="ignore")
    return d


BED_OF = {"asi": "PWV (ASI proxy, MR)", "cimt": "Carotid IMT", "plaque": "Carotid plaque", "crae": "Retina CRAE",
          "crve": "Retina CRVE", "avr": "Retina AVR", "tort": "Retina tortuosity", "dens": "Retina density"}
OUT_STD = {"cad": "CAD", "cad_nikpay": "CAD", "stroke": "Any stroke", "stroke_mega": "Any stroke",
           "istroke": "Ischaemic stroke", "istroke_mega": "Ischaemic stroke", "las_mega": "Large-artery stroke",
           "svs_mega": "Small-vessel stroke", "hf": "Heart failure", "hf_mvp": "Heart failure",
           "pad_mvp": "PAD", "pad_sakaue": "PAD", "aaa_mvp": "AAA", "ckd": "CKD", "egfr": "eGFR (kidney function)",
           "uacr": "UACR (kidney damage)", "af": "AF", "life": "Lifespan (parental)", "sbp": "BP level (SBP)",
           "dbp": "BP level (DBP)", "pp": "BP level (PP)"}


def assemble(outdir: Path) -> pd.DataFrame:
    rows = []

    def rel(p):
        return str(p.relative_to(C.PAPER))

    def mr_rows(path: Path, fams: dict):
        if not path.exists():
            return
        d = read_lines(path)
        d = d[d["family"].isin(list(fams))]
        for _, r in d.iterrows():
            if r["exposure"] not in BED_OF or r["outcome"] not in OUT_STD:
                continue
            kind = SPEC(r["outcome"])["kind"] if r["outcome"] in I.SPEC else "bin"
            est, lo, hi = float(r["ivw"]), float(r["ivw_lci"]), float(r["ivw_uci"])
            scale = "OR per exposure unit" if kind == "bin" else (
                "HR of parental death per exposure unit" if r["outcome"] == "life" else
                f"beta ({SPEC(r['outcome'])['units']}) per exposure unit")
            if kind == "bin" or r["outcome"] == "life":
                est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
            rows.append({"bed": BED_OF[r["exposure"]], "outcome": OUT_STD[r["outcome"]],
                         "evidence_type": fams[r["family"]], "cohort_or_gwas": r["outcome"],
                         "exposure_detail": f"{r['exposure']} ({r.get('exposure_unit', '')})", "adjustment": "",
                         "scale": scale, "estimate": est, "lower": lo, "upper": hi, "p": r["ivw_p"],
                         "q": r.get("ivw_q", np.nan), "n_or_nsnp": f"{int(r['nsnp'])} SNPs",
                         "events": "", "class": r.get("evidence", ""), "ukb_overlap": r.get("ukb_overlap", ""),
                         "steiger_correct_direction": r.get("steiger_correct_direction", ""),
                         "source": f"{rel(path)}:{r['_line']}"})

    def mv_rows(path: Path, fam: str, label: str):
        if not path.exists():
            return
        d = read_lines(path)
        d = d[(d["family"] == fam) & (d["term"].isin(list(BED_OF)))]
        for _, r in d.iterrows():
            if r["outcome"] not in OUT_STD or not np.isfinite(pd.to_numeric(r["estimate"], errors="coerce")):
                continue
            kind = SPEC(r["outcome"])["kind"] if r["outcome"] in I.SPEC else "bin"
            est, lo, hi = float(r["estimate"]), float(r["lower"]), float(r["upper"])
            if kind == "bin" or r["outcome"] == "life":
                est, lo, hi = np.exp(est), np.exp(lo), np.exp(hi)
            rows.append({"bed": BED_OF[r["term"]], "outcome": OUT_STD[r["outcome"]], "evidence_type": label,
                         "cohort_or_gwas": r["outcome"], "exposure_detail": f"{r['term']} | SBP",
                         "adjustment": "SBP (MVMR)", "scale": "OR" if kind == "bin" else (
                             "HR of parental death" if r["outcome"] == "life" else "beta"),
                         "estimate": est, "lower": lo, "upper": hi, "p": r["p"],
                         "q": r.get("q_within_family_term", np.nan), "n_or_nsnp": f"{int(r['nsnp'])} SNPs",
                         "events": "", "class": "", "ukb_overlap": "", "steiger_correct_direction": "",
                         "source": f"{rel(path)}:{r['_line']}"})

    mr_rows(I3_DIR / "mr_all_pairs_summary.csv",
            {"E_beds_to_outcomes": "MR LD-clumped (outcome GWAS incl. UKB where marked)",
             "X_E_beds_to_nonUKB_outcomes": "MR LD-clumped, non-UKB outcome"})
    mr_rows(I3_DIR / "sens_r2_mr_all_pairs_summary.csv", {"E_beds_to_outcomes": "MR LD-clumped r2<=0.01 (sens)"})
    mv_rows(I3_DIR / "mvmr.csv", "MV_bed_plus_sbp_to_outcomes", "MVMR bed + SBP (LD-clumped)")
    mr_rows(outdir / "mr_all_pairs_summary.csv",
            {"L_beds_to_new_outcomes": "MR LD-clumped (new outcome; this task)",
             "L_beds_to_bp": "MR LD-clumped (BP outcome; this task)",
             "L_beds_to_new_outcomes_r2sens": "MR LD-clumped r2<=0.01 (sens; this task)"})
    mv_rows(outdir / "mr_mvmr.csv", "L_MV_bed_plus_sbp_to_new_outcomes", "MVMR bed + SBP (LD-clumped; this task)")

    # observational, earlier tasks
    def obs(path: Path, bed_fn, out_col, expo_col, adj_col, keep, est="hr", lo="lower", hi="upper",
            n="n", ev="events", label="observational open cohort", cohort=""):
        if not path.exists():
            return
        d = read_lines(path)
        for _, r in d.iterrows():
            if not keep(r):
                continue
            rows.append({"bed": bed_fn(r), "outcome": r[out_col], "evidence_type": label, "cohort_or_gwas": cohort
                         or r.get("cohort", ""), "exposure_detail": f"{r[expo_col]}" + (f" [{r['term']}]" if "term"
                                                                                            in r else ""),
                         "adjustment": r[adj_col],
                         "scale": r["estimate_type"] if isinstance(r.get("estimate_type"), str) else "HR",
                         "estimate": r[est], "lower": r[lo],
                         "upper": r[hi], "p": r["p"], "q": np.nan, "n_or_nsnp": r[n], "events": r[ev],
                         "class": "", "ukb_overlap": "", "steiger_correct_direction": "",
                         "source": f"{rel(path)}:{r['_line']}"})

    A = C.ANALYSIS
    obs(A / "d_impact_abi" / "cox_all_models.csv", lambda r: "baPWV" if "baPWV" in r["exposure"] else "ABI",
        "outcome", "exposure", "adjustment",
        lambda r: str(r["adjustment"]).startswith("primary") and str(r["analysis_set"]) in
        ("main; ABI>=0.9", "main; all"), cohort="IMPACT-ABI (Japan, CVD inpatients)")
    obs(A / "e_artper" / "cox_models.csv", lambda r: "ABI", "outcome", "exposure", "adjustment",
        lambda r: r["adjustment"] == "M2 primary", cohort="ARTPER (Spain, population)")
    obs(A / "f_nhanes9904_abi" / "svy_cox.csv", lambda r: "ABI", "outcome", "exposure", "adjustment",
        lambda r: r["adjustment"] == "M2 primary" and r["analysis"] == "main", n="n_unweighted",
        ev="events_unweighted", cohort="NHANES 1999-2004 (US, survey-weighted)")
    obs(A / "g_nhanes3_ecg" / "svy_cox.csv", lambda r: "ECG voltage" if r["exposure"] != "sbp10" else "SBP (ref.)",
        "outcome", "exposure", "adjustment", lambda r: r["exposure"] != "sbp10" and r["adjustment"] != "M1",
        n="n_unweighted", ev="events_unweighted", cohort="NHANES III (US, survey-weighted)")
    # this task
    obs(outdir / "obs_nhanes9904_abi_cause.csv", lambda r: "ABI", "outcome", "exposure", "adjustment",
        lambda r: r["adjustment"] == "M2 primary", n="n_unweighted", ev="events_unweighted",
        label="observational open cohort (this task)", cohort="NHANES 1999-2004 (US, survey-weighted)")
    obs(outdir / "obs_nhanes3_ecg_cause.csv", lambda r: "ECG voltage", "outcome", "exposure", "adjustment",
        lambda r: True, n="n_unweighted", ev="events_unweighted", label="observational open cohort (this task)",
        cohort="NHANES III (US, survey-weighted)")
    obs(outdir / "obs_impact_mi_stroke.csv", lambda r: "baPWV" if "baPWV" in r["exposure"] else "ABI", "outcome",
        "exposure", "adjustment", lambda r: str(r["adjustment"]).startswith("primary"),
        label="observational open cohort (this task)", cohort="IMPACT-ABI (Japan, CVD inpatients)")
    obs(outdir / "obs_artper_components.csv", lambda r: "ABI", "outcome", "exposure", "adjustment",
        lambda r: r["adjustment"] == "M2 primary", label="observational open cohort (this task)",
        cohort="ARTPER (Spain, population)")

    def small_bed(r):
        x = str(r["exposure"])
        if "Retinal" in x:
            return "Retina (FA calibre)"
        if "IMT" in x:
            return "Carotid IMT"
        if "plaque" in x:
            return "Carotid plaque"
        if "cfPWV" in x:
            return "cfPWV"
        if "baPWV" in x:
            return "baPWV"
        return "ABI"
    obs(outdir / "obs_small_cohorts.csv", small_bed, "outcome", "exposure", "adjustment", lambda r: True,
        est="estimate", label="observational open cohort (this task)")
    long = pd.DataFrame(rows)
    long["estimate [95% CI]"] = [fmt_ci(float(a), float(b), float(c)) if all(
        np.isfinite(pd.to_numeric([a, b, c], errors="coerce"))) else "" for a, b, c in
        zip(long["estimate"], long["lower"], long["upper"])]
    return long


# ================================================================================================================
def main():
    def extra(ap):
        ap.add_argument("--part", default="all", choices=["all", "mr", "obs", "assemble"])
        ap.add_argument("--workers", type=int, default=6)
        ap.add_argument("--mr-workers", type=int, default=8)
        ap.add_argument("--no-hash", action="store_true")
        ap.add_argument("--scratch", default=str(SCRATCH), help="folder with the CSV copies of the .xls/.sav files")
    args = C.parse_args(__doc__.splitlines()[0], extra)
    if args.simulate:
        raise SystemExit("--simulate is not implemented for this driver (it reuses validated drivers d-g, i, i3)")
    mode, outdir, token = C.gate(args, TASK)
    if mode != "real" or token is None:
        raise SystemExit("real mode needs the gate token")
    _register()
    rep = C.Report(TASK, "F1 Stage 0 (l): vascular bed -> clinical outcomes from open data (MR + open cohorts)",
                   mode, args)
    notes = []
    rep.p("**Exploratory.** Open public data only; no HPP data; no liver phenotypes. Not paper numbers until "
          "checked by the F1 session.")
    if args.part in ("all", "mr"):
        run_mr(args, outdir, rep, notes)
    if args.part in ("all", "obs"):
        run_obs(args, outdir, rep, token, notes)
    long = assemble(outdir)
    rep.table("bed_outcome_evidence_long", long, "All bed x outcome estimates with source path:line "
              "(first 40 rows; full table in the CSV)", max_rows=40)
    rep.h("Notes")
    rep.bullets(notes or ["none"])
    if args.part in ("mr", "obs", "assemble") and args.part != "all":
        rep.p(f"Run with --part {args.part}: tables of the other parts are not regenerated in this report.")
    C.finish(rep, outdir, __file__)
    log("finished")


if __name__ == "__main__":
    main()
