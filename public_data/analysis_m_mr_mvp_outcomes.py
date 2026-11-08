#!/usr/bin/env python3
"""analysis_m_mr_mvp_outcomes.py - F1 Stage 0 task m (PLAN 9.3a L5): vascular beds -> MVP phecode outcomes (MR).

Paper F1 (multi-bed vascular atlas). ALL RESULTS ARE EXPLORATORY. Open public data only (no HPP data, no liver
phenotypes). Two-sample MR of each vascular bed on 14 Million Veteran Program (MVP) European-ancestry phecode GWAS
without UK Biobank (Verma 2024 Science; GWAS Catalog harmonised files, GRCh38, matched by rsID as the MVP AAA / HF /
PAD specs of analysis_i_mr / analysis_i3_mr_ldclump). Purpose: a non-UKB replication of the earlier bed -> outcome
MR signals (OUTCOMES_PUBLIC_SYNTHESIS_20261005.md section 3, R4-R7) and a genetic bed -> hypertension estimate (the
bed -> incident-hypertension gap).

Families (BH FDR within family; analysis_i_mr.bh):
- M_pc: positive controls, SBP (Keaton 2024; LD-clumped i3 instruments) -> each MVP outcome.
- M_beds_to_mvp: 8 beds (ASI, cIMT, plaque, CRAE, CRVE, AVR, tortuosity, density) x 14 MVP outcomes.
- M_beds_to_mvp_r2sens: same pairs with the r2 <= 0.01 instrument sets (sens_r2_instruments.csv of i3).
- M_MV_bed_plus_sbp: MVMR bed + SBP with joint LD clumping (analysis_i3_mr_ldclump.mvmr_ld, unchanged) for every
  bed -> MVP pair with univariable p < 0.05 plus the pre-specified replication pairs (table REPL below).
- Replication table: earlier estimates (path:line of the l / i3 CSVs) vs the MVP estimate, direction agreement and
  verdict (replicated / direction-consistent / not replicated).

Conventions: random-effects (multiplicative) IVW primary (analysis_i_mr.IVW_PRIMARY, picked by exact method name);
MR-Egger, weighted median and weighted mode alongside; Steiger filtering; evidence classes analysis_i_mr.classify.
Instruments are read from i3 (instruments_ld.csv: r2 <= 0.001 within 10 Mb, 1000G EUR; sens_r2_instruments.csv:
r2 <= 0.01), not recomputed. Retinal GWAS 4a-4e: allele labels swapped back on reading ('swap_alleles' special of
analysis_i_mr; the i3 instrument files already carry the corrected alleles). Units: cIMT per 0.1 mm (spec copy with
scale 0.1; the source GWAS is per mm), retinal traits and ASI per SD, SBP per 10 mmHg, outcomes log-odds (OR).
Exposure rows at the instrument keys are read from the analysis_l_bed_outcomes parquet cache when the cache key
matches (same files, same instrument union), else scanned into this task's _cache. MVP files are streamed (never
decompressed to disk).

Imported unchanged: analysis_i_mr, analysis_i3_mr_ldclump, analysis_l_bed_outcomes, lib_mr, s0an_common.
Mode: --real.
"""
from __future__ import annotations

import datetime as _dt
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_i3_mr_ldclump as L3  # noqa: E402
import analysis_i_mr as I  # noqa: E402
import analysis_l_bed_outcomes as L  # noqa: E402
import s0an_common as C  # noqa: E402

TASK = "m_mr_mvp_outcomes"
HUB, GWAS = I.HUB, I.GWAS
I3_DIR = C.ANALYSIS / "i3_mr_ldclump"
L_DIR = C.ANALYSIS / "l_bed_outcomes"
STATUS = "exploratory"
T0 = time.time()


def log(msg: str):
    print(f"[{_dt.datetime.now():%H:%M:%S} +{time.time() - T0:7.1f}s] {msg}", flush=True)


def fmt_ci(e, lo, hi, nd=3):
    if not all(np.isfinite([e, lo, hi])):
        return ""
    return f"{e:.{nd}g} [{lo:.{nd}g}, {hi:.{nd}g}]"


def rel(p: Path) -> str:
    p = Path(p)
    try:
        return str(p.relative_to(C.PAPER))
    except ValueError:
        return str(p.relative_to(HUB))


# ================================================================================================================
# MVP outcome specs (schema of analysis_i_mr.SPECS; reader = analysis_i_mr._MVP with OR / CI -> log-OR / SE)
# ================================================================================================================
# id, folder, GCST, label, SBP positive-control expectation (+1 raise, 0 not fixed), expectation pre-specified
MVP_LIST = [
    ("htn_ess_mvp", "23a_EssentialHTN_MVP_EUR_noUKB", "GCST90475922", "Essential hypertension", 1, True),
    ("htn_mvp", "23b_Hypertension_MVP_EUR_noUKB", "GCST90475919", "Hypertension", 1, True),
    ("af_mvp", "20b_AF_MVP_EUR_noUKB", "GCST90475970", "Atrial fibrillation", 1, True),
    ("ckd_mvp", "18c_CKD_MVP_EUR_noUKB", "GCST90476123", "Chronic kidney disease", 1, True),
    ("htnckd_mvp", "18d_HypertensiveCKD_MVP_EUR_noUKB", "GCST90475927", "Hypertensive CKD", 1, True),
    ("mi_mvp", "14c_MI_MVP_EUR_noUKB", "GCST90475932", "Myocardial infarction", 1, True),
    ("ihd_mvp", "14d_IHD_MVP_EUR_noUKB", "GCST90475929", "Ischaemic heart disease", 1, True),
    ("cinf_mvp", "15g_CerebralArteryOcclusionInfarct_MVP_EUR_noUKB", "GCST90477991",
     "Cerebral artery occlusion with cerebral infarction", 1, True),
    ("cisch_mvp", "15h_CerebralIschaemia_MVP_EUR_noUKB", "GCST90475988", "Cerebral ischaemia", 1, True),
    ("precereb_mvp", "3c_PrecerebralOcclusionStenosis_MVP_EUR_noUKB", "GCST90475986",
     "Occlusion / stenosis of precerebral arteries", 1, False),
    ("hfpef_mvp", "16c_HFpEF_MVP_EUR_noUKB", "GCST90475984", "Heart failure with preserved EF", 1, True),
    ("hfref_mvp", "16d_HFrEF_MVP_EUR_noUKB", "GCST90475983", "Heart failure with reduced EF", 1, True),
    ("aneur_mvp", "21b_AorticAneurysm_MVP_EUR_noUKB", "GCST90475993", "Aortic aneurysm", 0, False),
    ("retino_mvp", "4f_NonDiabeticRetinopathy_MVP_EUR_noUKB", "GCST90477636", "Other non-diabetic retinopathy", 1,
     False),
]
MVP_OUT = [x[0] for x in MVP_LIST]
SHORT = {"htn_ess_mvp": "EssHTN", "htn_mvp": "HTN", "af_mvp": "AF", "ckd_mvp": "CKD", "htnckd_mvp": "HTN-CKD",
         "mi_mvp": "MI", "ihd_mvp": "IHD", "cinf_mvp": "CerebInfarct", "cisch_mvp": "CerebIsch",
         "precereb_mvp": "PrecerebSten", "hfpef_mvp": "HFpEF", "hfref_mvp": "HFrEF", "aneur_mvp": "AorticAneur",
         "retino_mvp": "Retinopathy"}
BEDS = I.BEDS
RETINA = ["crae", "crve", "avr", "tort", "dens"]
BED_LABEL = {"asi": "ASI", "cimt": "cIMT", "plaque": "Plaque", "crae": "CRAE", "crve": "CRVE", "avr": "AVR",
             "tort": "Tortuosity", "dens": "Density", "sbp": "SBP"}


def first_row_counts(path: Path) -> dict:
    """Case / control / total n from the first data row of an MVP file (streamed; two lines read)."""
    with gzip.open(path, "rt") as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        row = fh.readline().rstrip("\n").split("\t")
    d = dict(zip(hdr, row))

    def num(k):
        try:
            return float(d.get(k))
        except (TypeError, ValueError):
            return np.nan
    return {"n_cases": num("num_cases"), "n_controls": num("num_controls"), "n_total": num("n")}


MVP_COUNTS: dict = {}


def mvp_specs() -> list[dict]:
    out = []
    for oid, folder, gcst, label, _, _ in MVP_LIST:
        p = GWAS / folder / f"{gcst}.tsv.gz"
        cnt = first_row_counts(p) if p.exists() else {"n_cases": np.nan, "n_controls": np.nan, "n_total": np.nan}
        MVP_COUNTS[oid] = cnt
        nd = cnt["n_total"] if np.isfinite(cnt["n_total"]) else (cnt["n_cases"] + cnt["n_controls"])
        out.append(dict(id=oid, label=f"{label} (MVP EUR phecode)", group="m_out", kind="bin",
                        path=f"{folder}/{gcst}.tsv.gz", sep="\t", build="38", cols=I._MVP, special=("or_ci",),
                        n_default=float(nd) if np.isfinite(nd) else 430_000.0, ancestry="EUR (MVP)", ukb=False,
                        units="log-odds", unit_label="per log-odds", adverse=1,
                        source=f"Verma 2024 Science (MVP EUR phecode GWAS, no UK Biobank), {gcst}; cases / controls "
                               f"from the file's first data row"))
    return out


def _register():
    L._register()                                   # i3 extension specs + l specs (unchanged)
    for s in mvp_specs():
        I.SPEC[s["id"]] = s
    # cIMT per 0.1 mm: a copy of the cIMT spec with scale 0.1 (the source GWAS is per mm); the scan keys do not
    # depend on 'scale' / 'unit_label', so the cached rows of the l task stay valid
    I.SPEC["cimt"] = dict(I.SPEC["cimt"], scale=0.1, unit_label="per 0.1 mm", units="mm (rescaled to 0.1 mm)")


def SPEC(i):
    return I.SPEC[i]


# ================================================================================================================
# Pre-specified replication rows (sources: OUTCOMES_PUBLIC_SYNTHESIS_20261005.md section 3, R4-R7 and section 4)
# Earlier = (task, family, outcome id, label). Kind: signal (earlier MR signal), proxy (bed -> SBP, continuous),
# null (earlier null), new (no earlier estimate).
# ================================================================================================================
_SVS = [("l", "L_beds_to_new_outcomes", "svs_mega", "Small-vessel stroke, MEGASTROKE EUR (no UKB)")]
_ISTR = [("i3", "E_beds_to_outcomes", "istroke", "Ischaemic stroke, GIGASTROKE (UKB overlap)"),
         ("i3", "X_E_beds_to_nonUKB_outcomes", "istroke_mega", "Ischaemic stroke, MEGASTROKE EUR (no UKB)")]
_AVR_STR = [("i3", "E_beds_to_outcomes", "istroke", "Ischaemic stroke, GIGASTROKE (UKB overlap)"),
            ("i3", "X_E_beds_to_nonUKB_outcomes", "istroke_mega", "Ischaemic stroke, MEGASTROKE EUR (no UKB)"),
            ("i3", "E_beds_to_outcomes", "stroke", "Any stroke, GIGASTROKE (UKB overlap)")]
_CAD = [("i3", "E_beds_to_outcomes", "cad", "CAD, Aragam 2022 (UKB)"),
        ("i3", "X_E_beds_to_nonUKB_outcomes", "cad_nikpay", "CAD, CARDIoGRAMplusC4D 2015 (no UKB)")]
_CIMT_STR = [("i3", "X_E_beds_to_nonUKB_outcomes", "istroke_mega", "Ischaemic stroke, MEGASTROKE EUR (no UKB)"),
             ("i3", "X_E_beds_to_nonUKB_outcomes", "stroke_mega", "Any stroke, MEGASTROKE EUR (no UKB)"),
             ("i3", "E_beds_to_outcomes", "stroke", "Any stroke, GIGASTROKE (UKB overlap)")]
_LAS = [("l", "L_beds_to_new_outcomes", "las_mega", "Large-artery stroke, MEGASTROKE EUR (no UKB; nearest "
                                                    "carotid-relevant earlier outcome)")]
_SBP = [("l", "L_beds_to_bp", "sbp", "SBP, Keaton 2024 (continuous proxy for hypertension; mmHg)")]
_NOTE_STROKE = ("MVP has no lacunar / small-vessel phecode: cerebral artery occlusion with infarction and cerebral "
                "ischaemia are the nearest outcomes")
REPL = []
for _b in ["crve", "avr", "tort"]:
    for _o in ["cinf_mvp", "cisch_mvp"]:
        REPL.append(("R4", "signal", _b, _o, _SVS, _NOTE_STROKE))
for _b in ["avr", "crve"]:
    for _o in ["cinf_mvp", "cisch_mvp"]:
        REPL.append(("R5", "signal", _b, _o, _AVR_STR if _b == "avr" else _ISTR,
                     "MVP ischaemic-stroke proxies: cerebral infarction (primary), cerebral ischaemia"))
for _o in ["mi_mvp", "ihd_mvp"]:
    REPL.append(("R6", "signal", "tort", _o, _CAD, "MVP MI and IHD as CAD outcomes"))
for _o in ["cinf_mvp", "cisch_mvp"]:
    REPL.append(("R7", "signal", "cimt", _o, _CIMT_STR, "cIMT per 0.1 mm (earlier rows rescaled from per mm)"))
for _b in ["cimt", "plaque"]:
    REPL.append(("R7", "signal", _b, "precereb_mvp", _LAS, "carotid-relevant outcome; earlier = large-artery stroke"))
for _o in ["htn_ess_mvp", "htn_mvp"]:
    for _b in BEDS:
        REPL.append(("HTN", "proxy", _b, _o, _SBP, "bed -> hypertension gap; earlier = bed -> SBP (mmHg)"))
for _b in RETINA:
    REPL.append(("RET", "new", _b, "retino_mvp", [], "retina -> other non-diabetic retinopathy (no earlier estimate)"))
for _c, _o, _src, _nt in [
        ("N2", "af_mvp", ("l", "L_beds_to_new_outcomes", "af", "AF, Nielsen 2018 (UKB)"),
         "earlier null (synthesis N2): bed -> AF"),
        ("CKD", "ckd_mvp", ("l", "L_beds_to_new_outcomes", "ckd", "CKD, CKDGen 2019 EA"),
         "earlier bed -> CKD (l); kind set per row: signal if earlier p < 0.05, else null")]:
    for _b in BEDS:
        REPL.append((_c, "auto", _b, _o, [_src], _nt))
EARLIER_FILES = {"l": L_DIR / "mr_all_pairs_summary.csv", "i3": I3_DIR / "mr_all_pairs_summary.csv"}
EARLIER_MV = {"l": (L_DIR / "mr_mvmr.csv", "L_MV_bed_plus_sbp_to_new_outcomes"),
              "i3": (I3_DIR / "mvmr.csv", "MV_bed_plus_sbp_to_outcomes")}
# earlier MVMR rows to show next to the MVP MVMR (claim, bed, earlier task, earlier outcome)
EARLIER_MV_ROWS = [("R4", "crve", "l", "svs_mega"), ("R4", "avr", "l", "svs_mega"), ("R5", "crve", "i3", "istroke"),
                   ("R5", "avr", "i3", "istroke"), ("R6", "tort", "i3", "cad"), ("R7", "cimt", "l", "las_mega")]


# ================================================================================================================
# Scans
# ================================================================================================================
def exposure_rows(ids: list, u_rs: set, u_pairs: set, cache: Path, workers: int, stats: list) -> dict:
    """Exposure rows at the instrument keys: the l task's parquet cache when the key matches (read-only), else
    scanned into this task's cache with analysis_l_bed_outcomes.scan_all (same key formula)."""
    hk = hashlib.sha1(("\n".join(sorted(u_rs)) + "|" + "\n".join(sorted(u_pairs))).encode()).hexdigest()[:12]
    out, missing = {}, []
    for i in ids:
        s = SPEC(i)
        p = GWAS / s["path"]
        st = p.stat()
        key = hashlib.sha1(json.dumps([i, str(p), st.st_size, int(st.st_mtime), hk, s.get("cols"),
                                       s.get("special"), s.get("colnames"), s.get("whitespace")],
                                      default=str).encode()).hexdigest()[:16]
        f = L_DIR / "_cache" / f"{i}_keys_{key}.parquet"
        if f.exists():
            out[i] = pd.read_parquet(f)
            stats.append({"id": i, "cached": True, "cache": rel(f), "rows_kept": len(out[i])})
        else:
            missing.append(i)
    if missing:
        log(f"exposure rows not in the l cache, scanning: {missing}")
        out.update(L.scan_all(missing, u_rs, u_pairs, cache, workers, stats))
    return out


def hash_files(paths: list[Path], cache_file: Path, seeds: list[Path], workers: int) -> dict:
    """SHA-256 by path|size|mtime; seeded read-only from earlier tasks' caches; new hashes cached here."""
    shac = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    for sf in seeds:
        if sf.exists():
            for k, v in json.loads(sf.read_text()).items():
                shac.setdefault(k, v)
    keys = {p: f"{p}|{p.stat().st_size}|{int(p.stat().st_mtime)}" for p in paths}
    todo = [str(p) for p, k in keys.items() if k not in shac]
    if todo:
        log(f"SHA-256 of {len(todo)} files")
        for path, h in I.run_pool(I.hash_worker, todo, workers):
            shac[keys[Path(path)]] = h
    cache_file.write_text(json.dumps(shac, indent=1))
    return {p: shac[k] for p, k in keys.items()}


# ================================================================================================================
# Earlier estimates (path:line; CSV header = line 1, no embedded line breaks)
# ================================================================================================================
_EARLIER_CACHE: dict = {}


def earlier_table(task: str) -> pd.DataFrame:
    if task not in _EARLIER_CACHE:
        d = pd.read_csv(EARLIER_FILES[task])
        d["_line"] = d.index + 2
        _EARLIER_CACHE[task] = d
    return _EARLIER_CACHE[task]


def earlier_row(task, fam, expo, out) -> dict | None:
    d = earlier_table(task)
    m = d[(d["family"] == fam) & (d["exposure"] == expo) & (d["outcome"] == out)]
    if len(m) != 1:
        return None
    r = m.iloc[0]
    k = 0.1 if expo == "cimt" else 1.0          # earlier cIMT rows are per mm -> per 0.1 mm
    est, lo, hi = r["ivw"] * k, r["ivw_lci"] * k, r["ivw_uci"] * k
    return {"est": est, "lo": lo, "hi": hi, "p": r["ivw_p"], "q": r.get("ivw_q", np.nan), "nsnp": r["nsnp"],
            "evidence": r.get("evidence", ""), "source": f"{rel(EARLIER_FILES[task])}:{int(r['_line'])}",
            "rescaled": k != 1.0}


def earlier_mv_row(task, expo, out) -> dict | None:
    f, fam = EARLIER_MV[task]
    d = pd.read_csv(f)
    d["_line"] = d.index + 2
    m = d[(d["family"] == fam) & (d["exposures"] == f"{expo}+sbp") & (d["outcome"] == out) & (d["term"] == expo)]
    if len(m) != 1:
        return None
    r = m.iloc[0]
    k = 0.1 if expo == "cimt" else 1.0
    return {"est": r["estimate"] * k, "lo": r["lower"] * k, "hi": r["upper"] * k, "p": r["p"],
            "cF": r.get("conditional_F", np.nan), "source": f"{rel(f)}:{int(r['_line'])}"}


# ================================================================================================================
def run(args, outdir: Path, rep, notes: list):
    _register()
    cache = outdir / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    f_main, f_sens = I3_DIR / "instruments_ld.csv", I3_DIR / "sens_r2_instruments.csv"
    inst = {k: v for k, v in L.read_instruments(f_main).items() if k in BEDS + ["sbp"]}
    inst_s = {k: v for k, v in L.read_instruments(f_sens).items() if k in BEDS + ["sbp"]}
    ninst = pd.DataFrame([{"exposure": k, "label": BED_LABEL[k], "unit": SPEC(k).get("unit_label", ""),
                           "instruments_r2_0.001": len(inst.get(k, [])),
                           "instruments_r2_0.01": len(inst_s.get(k, [])),
                           "exposure_gwas": SPEC(k)["source"]} for k in BEDS + ["sbp"]])
    beds_ok = [b for b in BEDS if len(inst.get(b, [])) >= 3]
    beds_ok_s = [b for b in BEDS if len(inst_s.get(b, [])) >= 3]
    same_sets = [b for b in BEDS if len(inst.get(b, [])) == len(inst_s.get(b, []))]

    avail = []
    for i in MVP_OUT:
        p = GWAS / SPEC(i)["path"]
        if p.exists():
            avail.append(i)
        else:
            notes.append(f"MVP file missing, analyses with {i} skipped: {rel(p)}")
    u_rs, u_pairs = set(), set()
    for d in list(inst.values()) + list(inst_s.values()):
        u_rs |= set(d["snp"].dropna())
        ok = d["pos"].notna()
        u_pairs |= set(I.cpkey(d.loc[ok, "chr"], d.loc[ok, "pos"]))
    scan_stats = []
    t_scan = time.time()
    ext = exposure_rows(BEDS + ["sbp"], u_rs, u_pairs, cache, args.workers, scan_stats)
    ext.update(L.scan_all(avail, u_rs, u_pairs, cache, args.workers, scan_stats))
    scan_s = time.time() - t_scan
    vmap = I.VariantMap()
    for d in inst.values():
        vmap.add(d)
    for i, d in ext.items():
        if SPEC(i)["build"] == "37":
            vmap.add(d)
    for i, d in ext.items():
        if SPEC(i)["build"] == "38":
            vmap.fill_pos37(d)

    # ---------------- registry and SHA-256 ----------------
    for f in (f_main, f_sens):
        rep.inputs.append({"id": f.name, "path": rel(f), "bytes": f.stat().st_size, "sha256": C.sha256(f)})
    gpaths = [GWAS / SPEC(i)["path"] for i in sorted(ext)]
    sha = {} if args.no_hash else hash_files(gpaths, cache / "sha256.json", [L_DIR / "_cache" / "sha256.json"],
                                              min(args.workers, 4))
    msha = I.manifest_sha()
    reg = []
    for i in sorted(ext, key=lambda x: (x not in MVP_OUT, MVP_OUT.index(x) if x in MVP_OUT else 0, x)):
        s, p = SPEC(i), GWAS / SPEC(i)["path"]
        cnt = MVP_COUNTS.get(i, {})
        h = sha.get(p)
        reg.append({"id": i, "label": s["label"], "role": "outcome" if i in MVP_OUT else "exposure",
                    "file": rel(p), "build": s["build"], "ancestry": s["ancestry"], "UKB_in_sample": s.get("ukb"),
                    "n_cases": cnt.get("n_cases"), "n_controls": cnt.get("n_controls"),
                    "n_total_first_row": cnt.get("n_total"), "rows_at_instrument_keys": len(ext[i]),
                    "bytes": p.stat().st_size, "sha256": h,
                    "manifest_sha256_match": None if h is None else msha.get(rel(p)) == h, "source": s["source"]})
        rep.inputs.append({"id": i, "path": rel(p), "bytes": p.stat().st_size, "sha256": h})
    registry = pd.DataFrame(reg)
    for task, f in list(EARLIER_FILES.items()) + [(k, v[0]) for k, v in EARLIER_MV.items()]:
        rep.inputs.append({"id": f"earlier_{task}_{f.name}", "path": rel(f), "bytes": f.stat().st_size,
                           "sha256": C.sha256(f)})

    # ---------------- allele orientation / EAF concordance ----------------
    eafqc = I.eaf_qc(ext["sbp"], {k: ext[k] for k in MVP_OUT if k in ext}) if "sbp" in ext else pd.DataFrame()

    # ---------------- univariable MR ----------------
    def ujob(eid, oid, fam, src):
        e, o = SPEC(eid), SPEC(oid)
        return dict(e={k: e[k] for k in ("id", "label", "kind", "unit_label", "units", "ukb") if k in e} |
                    {"scale": e.get("scale", 1.0)},
                    o={k: o[k] for k in ("id", "label", "kind", "unit_label", "units", "ukb") if k in o},
                    inst=I.with_ids(src[eid]), inst10=None, out=ext[oid], family=fam, seed=args.seed,
                    n_boot=args.n_boot, n_boot_mode=min(args.n_boot, 1000), drop_pal=not (I.eur(e) and I.eur(o)))

    fams = [("M_pc", "sbp", o, inst) for o in MVP_OUT]
    fams += [("M_beds_to_mvp", b, o, inst) for b in beds_ok for o in MVP_OUT]
    fams += [("M_beds_to_mvp_r2sens", b, o, inst_s) for b in beds_ok_s for o in MVP_OUT]
    fams = [x for x in fams if x[2] in ext and x[1] in x[3] and len(x[3][x[1]])]
    t_mr = time.time()
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
        ok = g["ivw_p"].notna()
        summ.loc[g.index[ok], "ivw_q"] = I.bh(g.loc[ok, "ivw_p"].values)
    summ["evidence"] = summ.apply(I.classify, axis=1)
    summ["OR_ivw"], summ["OR_lci"], summ["OR_uci"] = np.exp(summ["ivw"]), np.exp(summ["ivw_lci"]), \
        np.exp(summ["ivw_uci"])
    summ["OR [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(summ["OR_ivw"], summ["OR_lci"], summ["OR_uci"])]
    for m, c in [("wmedian", "OR_wmedian"), ("egger", "OR_egger"), ("wmode", "OR_wmode")]:
        summ[c] = np.exp(summ[m]) if m in summ.columns else np.nan
    summ["mvp_sample_overlap"] = summ["exposure"].eq("sbp")       # Keaton 2024 SBP includes MVP
    summ["clumping"] = np.where(summ["family"].str.endswith("r2sens"),
                                "LD r2<=0.01 within 10 Mb (1000G EUR; i3 instrument set)",
                                "LD r2<=0.001 within 10 Mb (1000G EUR; i3 instrument set)")
    summ["status"] = STATUS
    mr_s = time.time() - t_mr
    log(f"univariable MR done ({mr_s:.0f}s)")

    # EAF concordance exposure vs outcome on the harmonised instruments (per exposure, across MVP outcomes)
    pe = []
    hs = snps[snps["family"].isin(["M_pc", "M_beds_to_mvp"])]
    for (e, o), g in hs.groupby(["exposure", "outcome"]):
        g = g.dropna(subset=["eaf_x", "eaf_y"])
        pe.append({"exposure": e, "outcome": o, "n": len(g),
                   "eaf_corr": float(np.corrcoef(g["eaf_x"], g["eaf_y"])[0, 1]) if len(g) >= 3 else np.nan,
                   "mean_abs_diff": float((g["eaf_x"] - g["eaf_y"]).abs().mean()) if len(g) else np.nan})
    pe = pd.DataFrame(pe)
    eaf_expo = pe.groupby("exposure").agg(n_outcomes=("outcome", "size"), snps_median=("n", "median"),
                                          eaf_corr_min=("eaf_corr", "min"), eaf_corr_median=("eaf_corr", "median"),
                                          mean_abs_diff_max=("mean_abs_diff", "max")).reset_index()
    disc = summ[summ["family"].isin(["M_pc", "M_beds_to_mvp"])].groupby("exposure")["n_eaf_discordant"].max()
    eaf_expo["n_eaf_discordant_max"] = eaf_expo["exposure"].map(disc)
    eaf_expo["flag"] = eaf_expo["eaf_corr_min"] < 0.5

    # ---------------- MVMR bed + SBP ----------------
    mainf = summ[summ["family"] == "M_beds_to_mvp"]
    nominal = {(r["exposure"], r["outcome"]) for _, r in mainf.iterrows() if r["ivw_p"] < 0.05}
    presp = {(b, o) for c, kind, b, o, _, _ in REPL if c not in ("N2", "CKD")}
    mv_pairs = sorted((nominal | presp) & {(b, o) for b in beds_ok for o in MVP_OUT if o in ext},
                      key=lambda x: (BEDS.index(x[0]), MVP_OUT.index(x[1])))
    mv = pd.DataFrame()
    t_mv = time.time()
    ld_ok = all((L3.LDREF_DIR / f).exists() for f in ("g1000_eur.bed", "g1000_eur.bim", "g1000_eur.fam"))
    if args.no_mvmr:
        notes.append("MVMR skipped (--no-mvmr)")
    elif not ld_ok:
        notes.append("LD panel external/ld_ref/1000g_eur not on disk: MVMR bed + SBP not run")
    else:
        log(f"loading the LD panel for MVMR ({len(mv_pairs)} bed -> outcome pairs)")
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
        for b, o in mv_pairs:
            dp = not (I.eur(SPEC(b)) and I.eur(SPEC("sbp")) and I.eur(SPEC(o)))
            try:
                rr = L3.mvmr_ld([b, "sbp"], o, "M_MV_bed_plus_sbp", ext, inst_mv, keys, ldref, dp)
            except Exception as e:                                    # pragma: no cover - reported in the table
                rr = [{"family": "M_MV_bed_plus_sbp", "exposures": f"{b}+sbp", "outcome": o,
                       "error": f"{type(e).__name__}: {e}"}]
            for r in rr:
                r["selected_because"] = "+".join(x for x, c in [("univariable p<0.05", (b, o) in nominal),
                                                                ("pre-specified", (b, o) in presp)] if c)
            rows += rr
        mv = pd.DataFrame(rows)
        if "p" in mv.columns:
            mv["q_within_family_term"] = np.nan
            for term, g in mv.dropna(subset=["p"]).groupby("term"):
                mv.loc[g.index, "q_within_family_term"] = I.bh(g["p"].values)
            mv["OR"], mv["OR_lci"], mv["OR_uci"] = np.exp(mv["estimate"]), np.exp(mv["lower"]), np.exp(mv["upper"])
            mv["OR [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(mv["OR"], mv["OR_lci"], mv["OR_uci"])]
        mv["mvp_sample_overlap"] = "SBP (Keaton 2024 includes MVP)"
        mv["status"] = STATUS
        log(f"MVMR: {len(mv)} rows ({time.time() - t_mv:.0f}s)")
    mv_s = time.time() - t_mv

    # ---------------- replication table ----------------
    S = mainf.set_index(["exposure", "outcome"])
    Ss = summ[summ["family"] == "M_beds_to_mvp_r2sens"].set_index(["exposure", "outcome"])
    presp_q = {}
    pp = sorted(presp & set(S.index))
    pv = np.array([S.loc[k, "ivw_p"] for k in pp], float)
    okp = np.isfinite(pv)
    if okp.any():
        qv = I.bh(pv[okp])
        for k, q in zip([k for k, o in zip(pp, okp) if o], qv):
            presp_q[k] = q
    mvd = mv.set_index(["exposures", "outcome", "term"]) if len(mv) and "term" in mv.columns else None
    rrows = []
    for claim, kind, b, o, srcs, note in REPL:
        if (b, o) not in S.index:
            continue
        r = S.loc[(b, o)]
        er = [x for x in ((earlier_row(t, f, b, eo), lab) for t, f, eo, lab in srcs) if x[0] is not None]
        prim = er[0] if er else (None, "")
        e0 = prim[0]
        if kind == "auto":
            kind = "signal" if (e0 is not None and e0["p"] < 0.05) else "null"
        mvp_sign = np.sign(r["ivw"]) if np.isfinite(r["ivw"]) else 0
        agree = None if e0 is None else bool(np.sign(e0["est"]) == mvp_sign and mvp_sign != 0)
        sig = np.isfinite(r["ivw_p"]) and r["ivw_p"] < 0.05
        cons = bool(r.get("consistent_direction")) if pd.notna(r.get("consistent_direction")) else False
        if r["nsnp"] < 3:
            verdict = "not testable (<3 SNPs)"
        elif kind == "new":
            verdict = ("new: nominal, directions consistent" if sig and cons else
                       "new: nominal, directions inconsistent" if sig else "new: null")
        elif kind == "null":
            verdict = "earlier null confirmed" if not sig else (
                "earlier null not confirmed (MVP p<0.05" + (", directions consistent)" if cons else
                                                            ", directions inconsistent)"))
        elif kind == "proxy" and e0 is not None and e0["p"] >= 0.05:
            verdict = ("vs SBP proxy: both null" if not sig else
                       "vs SBP proxy: proxy null, MVP p<0.05 (" + ("direction agrees" if agree else
                                                                   "direction differs") +
                       (", estimators consistent)" if cons else ", estimators inconsistent)"))
        else:
            if agree and sig and cons:
                verdict = "replicated"
            elif agree:
                verdict = "direction-consistent"
            else:
                verdict = "not replicated" + (" (opposite, p<0.05)" if sig else "")
            if kind == "proxy":
                verdict = "vs SBP proxy: " + verdict
        rs = Ss.loc[(b, o)] if (b, o) in Ss.index else None
        mvt = None
        if mvd is not None and (f"{b}+sbp", o, b) in mvd.index:
            mvt = mvd.loc[(f"{b}+sbp", o, b)]
        unit = "per 0.1 mm" if b == "cimt" else ("per log-odds" if b == "plaque" else "per SD")
        rrows.append({
            "claim": claim, "kind": kind, "exposure": b, "exposure_label": BED_LABEL[b], "unit": unit,
            "mvp_outcome": o, "mvp_outcome_label": SPEC(o)["label"],
            "earlier_outcome": prim[1],
            "earlier_estimate [95% CI]": ("" if e0 is None else
                                          (fmt_ci(e0["est"], e0["lo"], e0["hi"]) + (" mmHg" if kind == "proxy" else
                                                                                    " log-OR"))),
            "earlier_OR [95% CI]": ("" if e0 is None or kind == "proxy" else
                                    fmt_ci(np.exp(e0["est"]), np.exp(e0["lo"]), np.exp(e0["hi"]))),
            "earlier_p": np.nan if e0 is None else e0["p"], "earlier_nsnp": np.nan if e0 is None else e0["nsnp"],
            "earlier_class": "" if e0 is None else e0["evidence"],
            "earlier_source": "" if e0 is None else e0["source"],
            "earlier_rescaled_per_0.1mm": False if e0 is None else e0["rescaled"],
            "earlier_other": "; ".join(f"{lab}: OR {fmt_ci(np.exp(x['est']), np.exp(x['lo']), np.exp(x['hi']))} "
                                       f"({x['source']})" for x, lab in er[1:]),
            "mvp_nsnp": r["nsnp"], "mvp_OR [95% CI]": r["OR [95% CI]"], "mvp_log_or": r["ivw"],
            "mvp_lci": r["ivw_lci"], "mvp_uci": r["ivw_uci"], "mvp_p": r["ivw_p"], "mvp_q_main_family": r["ivw_q"],
            "mvp_q_prespecified": presp_q.get((b, o), np.nan),
            "mvp_wmedian_OR": r.get("OR_wmedian"), "mvp_wmedian_p": r.get("wmedian_p"),
            "mvp_egger_OR": r.get("OR_egger"), "mvp_egger_intercept_p": r.get("egger_intercept_p"),
            "mvp_wmode_OR": r.get("OR_wmode"), "mvp_consistent_direction": r.get("consistent_direction"),
            "mvp_class": r["evidence"],
            "mvp_r2sens_OR [95% CI]": "" if rs is None else rs["OR [95% CI]"],
            "mvp_r2sens_p": np.nan if rs is None else rs["ivw_p"],
            "mvp_mvmr_bed_OR [95% CI]": "" if mvt is None else mvt["OR [95% CI]"],
            "mvp_mvmr_bed_p": np.nan if mvt is None else mvt["p"],
            "mvp_mvmr_conditional_F": np.nan if mvt is None else mvt["conditional_F"],
            "direction_agrees": agree,
            "earlier_in_mvp_ci": None if (e0 is None or kind == "proxy") else
            bool(r["ivw_lci"] <= e0["est"] <= r["ivw_uci"]),
            "verdict": verdict, "note": note})
    repl = pd.DataFrame(rrows)

    # MVMR key pairs vs earlier MVMR
    mvk = []
    if mvd is not None:
        seen = set()
        extra = [("nominal", "", b, o, [], "") for b, o in mv_pairs if (b, o) not in presp]
        for claim, _, b, o, _, _ in REPL + extra:
            if (b, o) in seen or claim in ("N2", "CKD") or (f"{b}+sbp", o, b) not in mvd.index:
                continue
            seen.add((b, o))
            claims = {c for c, k2, b2, o2, _, _ in REPL if b2 == b and o2 == o and c not in ("N2", "CKD")} or \
                {"univariable p<0.05 (not pre-specified)"}
            mb, ms = mvd.loc[(f"{b}+sbp", o, b)], mvd.loc[(f"{b}+sbp", o, "sbp")]
            em = [earlier_mv_row(t, b, eo) for c2, bb, t, eo in EARLIER_MV_ROWS if bb == b and c2 in claims]
            em = [x for x in em if x is not None]
            u = S.loc[(b, o)]
            mvk.append({"claim": "+".join(sorted(claims)), "exposure": b, "mvp_outcome": o, "nsnp": mb["nsnp"],
                        "univariable_OR [95% CI]": u["OR [95% CI]"], "univariable_p": u["ivw_p"],
                        "mvmr_bed_OR [95% CI]": mb["OR [95% CI]"], "mvmr_bed_p": mb["p"],
                        "mvmr_bed_q": mb["q_within_family_term"], "conditional_F_bed": mb["conditional_F"],
                        "mvmr_sbp_OR [95% CI] (per 10 mmHg)": ms["OR [95% CI]"], "mvmr_sbp_p": ms["p"],
                        "conditional_F_sbp": ms["conditional_F"], "Q_A_p": mb.get("Q_A_p"),
                        "earlier_mvmr_bed_OR [95% CI]": "; ".join(
                            f"{fmt_ci(np.exp(x['est']), np.exp(x['lo']), np.exp(x['hi']))} (cF "
                            f"{x['cF']:.2g}; {x['source']})" for x in em)})
    mvk = pd.DataFrame(mvk)

    # ---------------- report ----------------
    pc = summ[summ["family"] == "M_pc"].copy()
    exp_map = {x[0]: x[4] for x in MVP_LIST}
    presp_map = {x[0]: x[5] for x in MVP_LIST}
    pc["expected_sign"] = pc["outcome"].map(exp_map)
    pc["expectation_prespecified"] = pc["outcome"].map(presp_map)
    pc["matches_expectation"] = [None if e == 0 else bool(np.sign(v) == e and p < 0.05)
                                 for v, e, p in zip(pc["ivw"], pc["expected_sign"], pc["ivw_p"])]
    pcs = pc[["outcome", "nsnp", "OR [95% CI]", "ivw_p", "ivw_q", "OR_wmedian", "wmedian_p", "OR_egger",
              "egger_intercept_p", "OR_wmode", "Q_p", "steiger_correct_direction", "evidence", "expected_sign",
              "expectation_prespecified", "matches_expectation"]].copy()
    pcs.insert(1, "outcome_label", pcs["outcome"].map(lambda k: SPEC(k)["label"]))
    pcs.insert(2, "cases / controls", pcs["outcome"].map(
        lambda k: f"{MVP_COUNTS[k]['n_cases']:,.0f} / {MVP_COUNTS[k]['n_controls']:,.0f}"))
    for c in ("ivw_p", "ivw_q", "wmedian_p"):            # p underflows to 0 for SBP -> hypertension
        pcs[c] = [("<1e-300" if (np.isfinite(v) and v == 0) else v) for v in pcs[c].astype(float)]
    pl = summ[(summ["family"] == "M_beds_to_mvp") & (summ["exposure"] == "plaque")]
    if len(pl):
        ps = snps[(snps["family"] == "M_beds_to_mvp") & (snps["exposure"] == "plaque")].drop_duplicates("snp")
        notes.append(f"Plaque: {int(pl['n_instruments'].max())} instruments, {int(pl['nsnp'].max())} after "
                     f"harmonisation ({int(pl['n_palindromic_dropped'].max())} palindromic with EAF in [0.42, 0.58] "
                     f"dropped); plaque rows are single-variant Wald ratios at "
                     + ", ".join(f"{r.snp} (chr{r.chr}:{int(r.pos)} GRCh37)" for r in ps.itertuples())
                     + ", a variant with a strong SBP association (l: plaque -> SBP 5.38 mmHg per log-odds, "
                     "results/stage0/analysis/l_bed_outcomes/mr_all_pairs_summary.csv:72); not interpretable as a "
                     "plaque effect, and its MVMR term is weakly instrumented.")

    def grid(fam):
        g = summ[summ["family"] == fam]
        tab = []
        for o in MVP_OUT:
            row = {"outcome": SHORT[o]}
            for b in BEDS:
                m = g[(g["exposure"] == b) & (g["outcome"] == o)]
                if not len(m):
                    row[BED_LABEL[b]] = ""
                    continue
                r = m.iloc[0]
                mark = "**" if r["ivw_q"] < 0.05 else ("*" if r["ivw_p"] < 0.05 else "")
                row[BED_LABEL[b]] = (f"{r['OR [95% CI]']}{mark}" if r["nsnp"] >= 1 else "")
            tab.append(row)
        return pd.DataFrame(tab)

    nom = mainf[mainf["ivw_p"] < 0.05].sort_values("ivw_p")
    nshow = ["exposure", "outcome", "nsnp", "OR [95% CI]", "ivw_p", "ivw_q", "OR_wmedian", "wmedian_p", "OR_egger",
             "egger_intercept_p", "OR_wmode", "wmode_p", "Q_p", "steiger_correct_direction", "ivw_steiger_p",
             "evidence"]
    sens_nom = Ss.reset_index()
    sens_nom = sens_nom[sens_nom.set_index(["exposure", "outcome"]).index.isin(list(zip(nom["exposure"],
                                                                                       nom["outcome"])))]
    fdr_main = mainf[mainf["ivw_q"] < 0.05]
    fdr_main3 = fdr_main[(fdr_main["nsnp"] >= 3) & fdr_main["evidence"].isin(
        ["robust", "supported (FDR) - sensitivity weaker"])]

    rep.h("Scope and methods")
    rep.bullets([
        "Exploratory two-sample MR (PLAN 9.3a L5). Exposures: the eight bed GWAS of analysis_i_mr and Keaton 2024 SBP "
        "with the LD-clumped instrument sets of i3 (`results/stage0/analysis/i3_mr_ldclump/instruments_ld.csv`, "
        "r2 <= 0.001 within 10 Mb; sensitivity `sens_r2_instruments.csv`, r2 <= 0.01), read, not recomputed.",
        "Outcomes: 14 MVP European-ancestry phecode GWAS without UK Biobank (Verma 2024 Science; GWAS Catalog "
        "harmonised files, GRCh38). Matched by rsID; log-OR from odds_ratio, SE from the 95% CI (from log-OR and p "
        "when p <= 1e-5), as for the MVP AAA / HF / PAD files (analysis_i_mr._MVP, special 'or_ci').",
        "Primary estimator: IVW multiplicative random effects (selected by exact method name); MR-Egger, weighted "
        "median, weighted mode, Steiger filtering and evidence classes of analysis_i_mr. BH FDR within family "
        "(M_pc, M_beds_to_mvp, M_beds_to_mvp_r2sens, MVMR per term).",
        "Units: cIMT per 0.1 mm (source GWAS per mm; spec copy with scale 0.1; earlier cIMT rows rescaled by 1/10), "
        "plaque per log-odds, ASI and retinal traits per SD, SBP per 10 mmHg; outcomes as OR.",
        "Retinal GWAS 4a-4e: allele labels swapped back on reading (analysis_i_mr 'swap_alleles'); the i3 "
        "instrument files carry the corrected alleles.",
        f"r2 <= 0.01 instrument sets are identical to the primary sets for {', '.join(BED_LABEL[b] for b in same_sets)}"
        f"; the sensitivity family differs from the main family only for the other beds.",
        "Sample overlap: no bed GWAS includes MVP (ASI and retina: UK Biobank; cIMT and plaque: CHARGE). Keaton 2024 "
        "SBP includes MVP, so the positive controls and the SBP term of the MVMR share samples with the outcomes.",
        "MVMR bed + SBP: analysis_i3_mr_ldclump.mvmr_ld (joint LD clumping of the bed and SBP instruments, r2 <= 0.001 "
        "within 10 Mb, 1000G EUR) for every bed -> MVP pair with univariable p < 0.05 plus the pre-specified "
        "replication pairs (claims R4-R7, HTN, RET).",
    ])
    rep.table("mvp_gwas_registry", registry, "GWAS registry (MVP outcomes and exposures; SHA-256; manifest match)",
              max_rows=None)
    rep.table("instrument_counts", ninst, "Instruments read from i3")
    if len(eafqc):
        rep.table("eaf_concordance_mvp_vs_keaton_sbp", eafqc, "Allele orientation: EAF concordance of each MVP file "
                  "with Keaton SBP at the instrument keys (label alignment only; palindromic excluded)")
    rep.table("eaf_concordance_by_exposure", eaf_expo, "Allele orientation: EAF correlation exposure vs MVP outcome "
              "on the harmonised instruments (minimum and median across the 14 outcomes)")
    rep.h("Positive controls (family M_pc): SBP per 10 mmHg -> MVP outcomes")
    rep.table("pc_sbp_to_mvp", pcs, "SBP -> MVP outcome (OR per 10 mmHg; expected +1 = SBP raises risk)", max_rows=None)
    rep.h("Replication table (earlier bed -> outcome signals vs MVP)")
    rep.p("Verdict rules: replicated = same direction as the primary earlier estimate, MVP IVW p < 0.05 and the four "
          "estimators agree in direction; direction-consistent = same direction otherwise; not replicated = opposite "
          "direction. HTN rows compare with the earlier bed -> SBP estimate (continuous proxy; direction only). N2 "
          "rows test the earlier null bed -> AF; CKD rows compare with the earlier bed -> CKD (CKDGen; treated as a "
          "signal where the earlier p < 0.05, else as a null). HTN 'both null' = earlier bed -> SBP p >= 0.05 and MVP "
          "p >= 0.05. RET rows have no earlier estimate. q_prespecified = BH over the "
          f"{len(presp_q)} unique pre-specified bed -> MVP pairs (R4-R7, HTN, RET).")
    rcols = ["claim", "exposure_label", "mvp_outcome", "earlier_outcome", "earlier_estimate [95% CI]",
             "earlier_source", "mvp_nsnp", "mvp_OR [95% CI]", "mvp_p", "mvp_q_main_family", "mvp_q_prespecified",
             "mvp_wmedian_p", "mvp_consistent_direction", "mvp_r2sens_p", "mvp_mvmr_bed_OR [95% CI]",
             "mvp_mvmr_bed_p", "direction_agrees", "verdict"]
    rep.table("replication_table", repl[rcols], "Replication table (full columns in replication_table_full.csv)",
              max_rows=None)
    rep.h("Main family M_beds_to_mvp: beds -> 14 MVP outcomes")
    rep.table("grid_beds_to_mvp", grid("M_beds_to_mvp"), "OR [95% CI] per bed unit (IVW random effects; LD r2 <= "
              "0.001); * p < 0.05, ** q < 0.05 within family", max_rows=None)
    rep.table("fdr_supported_main_ge3snps", fdr_main3[nshow], "New FDR-supported bed -> MVP pairs (q < 0.05, >= 3 "
              "SNPs, class robust or supported)", max_rows=None)
    rep.table("fdr_supported_main", fdr_main[nshow], "All pairs with q < 0.05 in M_beds_to_mvp (single-SNP plaque "
              "rows are Wald ratios at one variant; class 'insufficient')",
              max_rows=None)
    rep.table("nominal_main", nom[nshow], "All pairs with IVW p < 0.05 in M_beds_to_mvp", max_rows=None)
    if len(sens_nom):
        rep.table("r2sens_for_nominal", sens_nom[[c for c in nshow if c in sens_nom.columns]],
                  "r2 <= 0.01 sensitivity for the nominal pairs", max_rows=None)
    rep.table("grid_beds_to_mvp_r2sens", grid("M_beds_to_mvp_r2sens"), "Sensitivity r2 <= 0.01: OR [95% CI]; * p < "
              "0.05, ** q < 0.05 within family", max_rows=None)
    rep.h("MVMR bed + SBP (joint LD clumping)")
    if len(mvk):
        rep.table("mvmr_key_pairs", mvk, "MVMR bed + SBP for the pre-specified pairs (and earlier MVMR where one "
                  "exists)", max_rows=None)
    if len(mv):
        mshow = [c for c in ["exposures", "outcome", "term", "nsnp", "OR [95% CI]", "p", "q_within_family_term",
                             "conditional_F", "Q_A_p", "ivw_same_snps", "selected_because", "error"]
                 if c in mv.columns]
        rep.table("mvmr_all", mv[mshow], "All MVMR rows (bed and SBP terms)", max_rows=None)
    summ.to_csv(outdir / "mr_all_pairs_summary.csv", index=False)
    longd.to_csv(outdir / "mr_all_methods_long.csv", index=False)
    snps.to_csv(outdir / "mr_harmonised_snps.csv.gz", index=False, compression="gzip")
    pe.to_csv(outdir / "eaf_concordance_by_pair.csv", index=False)
    repl.to_csv(outdir / "replication_table_full.csv", index=False)
    if len(mv):
        mv.to_csv(outdir / "mr_mvmr.csv", index=False)
    pd.DataFrame(scan_stats).to_csv(outdir / "scan_stats.csv", index=False)
    ss = pd.DataFrame(scan_stats)
    n_cached = int(ss["cached"].sum()) if len(ss) else 0
    return {"summ": summ, "mv": mv, "repl": repl, "times": {"scan_s": scan_s, "mr_s": mr_s, "mvmr_s": mv_s,
                                                            "n_cached": n_cached, "n_files": len(ss)}}


def main():
    def extra(ap):
        ap.add_argument("--workers", type=int, default=4, help="parallel file scans (each up to 2 pyarrow threads)")
        ap.add_argument("--mr-workers", type=int, default=6)
        ap.add_argument("--no-hash", action="store_true")
        ap.add_argument("--no-mvmr", action="store_true")
    args = C.parse_args(__doc__.splitlines()[0], extra)
    if args.simulate:
        raise SystemExit("--simulate is not implemented for this driver (it reuses validated drivers i, i3, l)")
    mode, outdir, token = C.gate(args, TASK)
    if mode != "real" or token is None:
        raise SystemExit("real mode needs the gate token")
    rep = C.Report(TASK, "F1 Stage 0 (m): vascular beds -> MVP phecode outcomes without UK Biobank (two-sample MR)",
                   mode, args)
    notes = [
        "MVP is mostly male US veterans; phecodes are EHR code groupings (not adjudicated events); hypertension "
        "phecodes have about 75% cases (320,429 / 107,275), so log-ORs are not risk ratios.",
        "Essential hypertension and hypertension share almost all cases; MI is nested in IHD; the two cerebral "
        "outcomes overlap; HFpEF / HFrEF overlap the MVP heart-failure file used in i3. The 14 tests are not "
        "independent.",
        "MVP has no lacunar / small-vessel stroke phecode: R4 is tested against cerebral artery occlusion with "
        "infarction and cerebral ischaemia (nearest outcomes, not the same endpoint).",
        "Exposure GWAS for ASI and retina are UK Biobank-based (the European UKB population), cIMT and plaque CHARGE; "
        "instrument counts are small for ASI, plaque, CRAE, AVR and cIMT.",
        "Keaton 2024 SBP includes MVP: sample overlap in the positive controls and in the SBP term of the MVMR.",
    ]
    rep.p("**Exploratory.** Open public data only; no HPP data; no liver phenotypes. Not paper numbers until checked "
          "by the F1 session. MR is exploratory.")
    out = run(args, outdir, rep, notes)
    rep.h("Notes and caveats")
    rep.bullets(notes)
    tm = out["times"]
    rep.h("Runtime")
    rep.p(f"File scans {tm['scan_s']:.0f} s ({tm['n_cached']} of {tm['n_files']} files read from parquet caches: the "
          f"exposure rows from the l task's cache, the MVP rows from this task's cache when it exists; scan_stats.csv); "
          f"univariable MR {tm['mr_s']:.0f} s; MVMR {tm['mvmr_s']:.0f} s; total "
          f"{time.time() - T0:.0f} s (workers: scans {args.workers}, MR {args.mr_workers}).")
    C.finish(rep, outdir, __file__)
    log("finished")


if __name__ == "__main__":
    main()
