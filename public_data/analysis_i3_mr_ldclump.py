#!/usr/bin/env python3
"""analysis_i3_mr_ldclump.py - F1 Stage 0 item 4: the analysis_i_mr MR panel rerun with LD clumping (1000G EUR).

Paper F1 (multi-bed vascular atlas). ALL RESULTS ARE EXPLORATORY.

Why
---
analysis_i_mr.py selects instruments by greedy distance clumping (+/-1 Mb; +/-10 Mb sensitivity) because no LD
reference was on disk. The 1000 Genomes phase 3 EUR reference distributed with MAGMA (CTG lab, VU Amsterdam;
503 samples, GRCh37, PLINK 1 binary, bi-allelic SNPs) is now in external/ld_ref/1000g_eur/ (downloaded 2026-10-05
with scripts/fetch_one.sh; SHA-256 in that folder's README.md and checked below). This driver reruns the main panel
with LD clumping and compares every evidence class with the distance-clumped panel of results/stage0/analysis/i_mr/
(random-effects rerun of 2026-10-05).

Unchanged (imported from analysis_i_mr.py; analysis_i_mr.py itself is not modified)
-----------------------------------------------------------------------------------
GWAS registry and streaming readers, including the retinal 'swap_alleles' correction of the Ortin Vela files
(4a-4e: effect/other allele labels swapped as deposited); candidate filters (p <= 5e-8, autosomes, no MHC,
MAF >= 0.01, valid alleles, finite beta/SE); GRCh38 cross-walks; harmonisation (lib_mr.harmonise; palindromic SNPs
by EAF outside [0.42, 0.58], all dropped when either GWAS is not European); the univariable MR worker (IVW with
multiplicative random effects as the primary estimator, as corrected on 2026-10-05; IVW fixed, MR-Egger, weighted
median, weighted mode, Q, Steiger filtering); FDR (BH) within family; evidence classes (analysis_i_mr.classify).

New
---
- LD clumping without plink. Greedy: index variant = smallest p (ties: largest |z|, then file order); every other
  remaining candidate on the same chromosome within +/-10 Mb whose r^2 with the index exceeds 0.001 is removed, so
  the retained index variants have pairwise r^2 <= 0.001 within 10 Mb (PLINK --clump-r2 0.001 --clump-kb 10000, the
  TwoSampleMR / ieugwasr default). r^2 is the squared Pearson correlation of allele counts in the 503 samples (no
  missing genotypes in this panel; missing would be mean-imputed). The .bed is memory-mapped and decoded with numpy
  (lib_mr._bed_lut).
- Matching to the panel: rsID (chromosome must agree when known; the allele pair must agree, strand complements
  accepted), then the MAGMA dbSNP-151 synonym file for merged rsIDs, then GRCh37 chr:pos + alleles. Candidates
  without a match are dropped before clumping (as PLINK and ieugwasr do); counts are reported per exposure. GRCh38
  instruments take their GRCh37 position from the panel (by rsID); build-37 instruments without rsID take the
  panel rsID.
- EAF rules: as in analysis_i_mr, MAF >= 0.01 where the GWAS reports EAF. Rows without EAF in a file that reports
  EAF for its other rows (only the MAGIC fasting-glucose and HbA1c files) are excluded before clumping (review fix
  of 2026-10-05). These rows come from a small subset of the samples (N 283 to 17,302), and their reported p is
  often not supported by beta/SE; in the distance-clumped panel they entered the FG set unfiltered (implausible
  effects of several mmol/L per allele) and a diagnostic table recomputes the i_mr estimates without them. Before the
  fix the LD run removed them only by the clumping order (absent from the panel, or clumped away by a stronger
  neighbour); the run now checks that the instrument sets are identical with and without the exclusion and lists
  the fate of every excluded candidate. In a file without any EAF the variant needs MAF >= 0.01 in the panel.
- Sensitivity (--r2-sens, default 0.01): the main univariable families and the two main MVMR families rerun with
  r2 <= 0.01 (same +/-10 Mb window and rules), to separate the loss of independent instruments under r2 <= 0.001
  with 503 samples from the removal of correlated instruments.
- Multivariable MR: union of the exposures' LD-clumped instruments, re-clumped jointly by LD on the smallest p across
  exposures (replaces the joint +/-1 Mb distance re-clumping of analysis_i_mr._mvmr; everything else identical).
- Chami 2025 uncoupled-adiposity variants: the 266 published lead variants are LD-clumped too, with p threshold 1
  (169 of them have body-fat p > 5e-8); analysis_i_mr used all 266 without re-clumping, so for family C the
  comparison is 'unclumped published leads -> LD-clumped' (column i_mr_instrument_set), not 'distance -> LD'.
- Comparison with the distance-clumped panel: class-by-class transition tables and every pair whose evidence class
  changed; MVMR terms whose p < 0.05 status or sign changed.
- Extensions on open GWAS downloaded on 2026-10-05 (first look, exploratory; families prefixed X_):
  X_mv_whradjbmi_bmi[_height] (Yengo 2022 height; WHRadjBMI + BMI + height on the beds);
  X_B_nonUKB / X_E_nonUKB (outcomes without UK Biobank: CARDIoGRAMplusC4D 2015 CAD, MEGASTROKE 2018 EUR any and
  ischaemic stroke, MVP EUR heart failure (phecode 428; added in the review fix)); X_mv_dbp_pp (Keaton DBP + PP on the beds and on pipeline positive controls: MEGASTROKE EUR
  small-vessel and large-artery stroke, MVP abdominal aortic aneurysm); X_mv_gera_dbp_pp (GERA DBP + PP, exposure
  GWAS without UK Biobank or MVP); X_mv_map_pp (Pan-UK Biobank MAP + Keaton PP); X_osa_bmiadj and
  X_mv_osa_bmiadj_bmi (Campos 2023 BMI-adjusted sleep apnoea, European). GERA (no SE column) and Pan-UKB MAP
  (-log10 p, no rsID) are read by a small reader in this file (analysis_i_mr.standardize cannot read them); their
  units are checked against Keaton effects at Keaton-selected instruments (approximate: winner's curse in Keaton
  biases the ratios towards zero; a column restricted to Keaton p <= 1e-20 bounds the bias).

Not rerun
---------
Family D (BioBank Japan / East Asian exposures): a European panel is not an appropriate LD reference for East Asian
instruments. Those results stay distance-clumped (results/stage0/analysis/i_mr/mr_D_bbj_eas.csv).

Caveat
------
With 503 reference samples, n*r^2 between independent SNPs is roughly chi-square(1), so r^2 between independent SNPs
exceeds 0.001 with probability about 0.48 (P(chi2_1 > 0.503)). The r^2 <= 0.001 / 10 Mb rule therefore also removes
many truly independent signals within 10 Mb. It is the field's default (results comparable with published MR) but
conservative: instrument counts fall, and imprecise associations can lose significance for that reason alone.

Modes
-----
--real  real public GWAS. Output: results/stage0/analysis/i3_mr_ldclump/.
--selftest       planted-answer checks of the new code on a synthetic PLINK panel written to a temporary folder inside
                 the output folder (removed afterwards); also run at the start of every real run.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analysis_i_mr as I
import lib_mr as M
import s0an_common as C

TASK = "i3_mr_ldclump"
HUB, GWAS, MANIFEST = I.HUB, I.GWAS, I.MANIFEST
P_THRESH = I.P_THRESH
R2_THRESH = 0.001
R2_SENS = 0.01          # sensitivity threshold (--r2-sens): separates pruning of independent signals from LD
LD_WINDOW_KB = 10000.0
LDREF_DIR = HUB / "external" / "ld_ref" / "1000g_eur"
LDREF_PREFIX = LDREF_DIR / "g1000_eur"
LDREF_URL = "https://vu.data.surfsara.nl/index.php/s/VZNByNwpD8qqINe/download"
LDREF_SHA256 = {   # verified 2026-10-05 (zip CRC test passed; zip SHA-256 = external/DOWNLOAD_MANIFEST.tsv row)
    "g1000_eur.zip": "83a48fd9dcaa0b9a874b18c63143a4ede93f05505b215b0bd8790130a0d7a954",
    "g1000_eur.bed": "fadce7321e622ed20e3cb7798b3e54699c7f342752c642a14ddb982d93d8e08c",
    "g1000_eur.bim": "b7dfeb30cef0573059ae43dd85a6924e46e418d6fe03df6b0ff1166c79cfa131",
    "g1000_eur.fam": "79dee21226fd04b5a889cbbeacd08079e1740e7d63b0a4dea115527e8fea4dae",
    "g1000_eur.synonyms": "33cf56ff20d1ca6129dc5dbddf35ebebcfedf1f4ffd13afacbc21bf805505b3a",
}
I_MR_DIR = C.ANALYSIS / "i_mr"
I2_DIR = C.ANALYSIS / "i2_mr_ext"
STATUS = "exploratory"

# ---------------------------------------------------------------------------------------------------------------
# GWAS used only by the extensions (same schema as analysis_i_mr.SPECS); registered in analysis_i_mr.SPEC because
# analysis_i_mr.Scanner looks specs up there. dbp / pp are copied unchanged from analysis_i2_mr_ext.NEW_SPECS.
# ---------------------------------------------------------------------------------------------------------------
_SSF = I._SSF
EXT_SPECS = [
    dict(id="height", label="Height", group="ext_risk", kind="cont",
         path="22a_Height_Yengo2022_EUR/GCST90245992_buildGRCh37.tsv", sep="\t", build="37",
         cols=dict(_SSF, snp="variant_id", n="n"), n_default=1_597_374, ancestry="EUR", ukb=True,
         units="SD (as published)", unit_label="per SD",
         source="Yengo 2022 Nature (GIANT EUR incl. UKB, without 23andMe; HapMap3 SNPs), GCST90245992"),
    dict(id="dbp", label="DBP", group="ext_risk", kind="cont", path="1e_DBP/GCST90310295.tsv.gz", sep="\t",
         build="37", cols=dict(_SSF, snp="rs_id", n="n"), n_default=1_028_920, ancestry="EUR", ukb=True,
         units="mmHg", scale=10.0, unit_label="per 10 mmHg",
         source="Keaton 2024 Nat Genet (ICBP+UKB+MVP+BioVU), GCST90310295"),
    dict(id="pp", label="PP", group="ext_risk", kind="cont", path="1f_PP/GCST90310296.tsv.gz", sep="\t", build="37",
         cols=dict(_SSF, snp="rs_id", n="n"), n_default=1_009_410, ancestry="EUR", ukb=True, units="mmHg",
         scale=10.0, unit_label="per 10 mmHg", source="Keaton 2024 Nat Genet (ICBP+UKB+MVP+BioVU), GCST90310296"),
    dict(id="dbp_gera", label="DBP (GERA)", group="ext_risk", kind="cont",
         path="1g_DBP_GERA_noUKB/gera-dbp.tsv.gz", sep="\t", build="37", cols=dict(_SSF), custom="gera",
         n_default=99_785, ancestry="multi (81% EUR; GERA)", ukb=False, units="mmHg (checked against Keaton)",
         scale=10.0, unit_label="per 10 mmHg",
         source="Hoffmann 2017 Nat Genet (GERA discovery only; no UKB, no MVP), GCST007098; no SE column: SE from "
                "beta and p"),
    dict(id="pp_gera", label="PP (GERA)", group="ext_risk", kind="cont",
         path="1h_PP_GERA_noUKB/gera-pp.tsv.gz", sep="\t", build="37", cols=dict(_SSF), custom="gera",
         n_default=99_785, ancestry="multi (81% EUR; GERA)", ukb=False, units="mmHg (checked against Keaton)",
         scale=10.0, unit_label="per 10 mmHg",
         source="Hoffmann 2017 Nat Genet (GERA discovery only; no UKB, no MVP), GCST007097; no SE column: SE from "
                "beta and p"),
    dict(id="map_panukb", label="MAP (Pan-UKB EUR+CSA)", group="ext_risk", kind="cont",
         path="1i_MAP_PanUKB_EUR_CSA/GCST90692977.tsv.gz", sep="\t", build="37", cols=dict(_SSF), custom="panukb",
         n_default=428_976, ancestry="EUR (98%) + CSA (Pan-UKB meta)", ukb=True,
         units="SD (Pan-UKB continuous traits are rank-inverse-normal transformed; checked against Keaton)",
         scale=1.0, unit_label="per SD",
         source="Karczewski 2025 Nat Genet (Pan-UK Biobank; MAP, automated + manual reading), GCST90692977; "
                "p from -log10 p; no rsID"),
    dict(id="osa_campos_bmiadj", label="Sleep apnoea, BMI-adjusted (UKB+FinnGen)", group="ext_risk", kind="bin",
         path="5h_Sleep_apnoea_BMIadj_UKB_FinnGen/GCST90267363.tsv", sep="\t", build="37",
         cols=dict(_SSF, snp="rs_id"), n_default=362_638, ancestry="EUR", ukb=True, units="log-odds",
         unit_label="per log-odds", source="Campos 2023 Sleep (UKB+FinnGen, BMI-adjusted), GCST90267363"),
    dict(id="cad_nikpay", label="CAD (CARDIoGRAMplusC4D 2015)", group="ext_out", kind="bin",
         path="14b_CAD_Nikpay2015_noUKB/26343387-GCST003116-EFO_0000378-build37.f.tsv.gz", sep="\t", build="37",
         cols=dict(_SSF, snp="variant_id"), n_default=184_305, ancestry="multi (77% EUR)", ukb=False,
         units="log-odds", unit_label="per log-odds", adverse=1,
         source="Nikpay 2015 Nat Genet (CARDIoGRAMplusC4D 1000G; no UKB), GCST003116"),
    dict(id="stroke_mega", label="Any stroke (MEGASTROKE EUR)", group="ext_out", kind="bin",
         path="15c_Any_stroke_MEGASTROKE_EUR_noUKB/29531354-GCST006906-EFO_0000712-build37.f.tsv.gz", sep="\t",
         build="37", cols=dict(_SSF, snp="variant_id"), n_default=446_696, ancestry="EUR", ukb=False,
         units="log-odds", unit_label="per log-odds", adverse=1,
         source="Malik 2018 Nat Genet (MEGASTROKE EUR; no UKB), GCST006906"),
    dict(id="istroke_mega", label="Ischaemic stroke (MEGASTROKE EUR)", group="ext_out", kind="bin",
         path="15d_Ischaemic_stroke_MEGASTROKE_EUR_noUKB/29531354-GCST006908-HP_0002140-build37.f.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF, snp="variant_id"), n_default=440_328, ancestry="EUR", ukb=False,
         units="log-odds", unit_label="per log-odds", adverse=1,
         source="Malik 2018 Nat Genet (MEGASTROKE EUR; no UKB), GCST006908"),
    dict(id="svs_mega", label="Small-vessel stroke (MEGASTROKE EUR)", group="ext_out", kind="bin",
         path="15e_Small_vessel_stroke_MEGASTROKE_EUR_noUKB/29531354-GCST006909-EFO_1001504-build37.f.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF, snp="variant_id"), n_default=348_946, ancestry="EUR", ukb=False,
         units="log-odds", unit_label="per log-odds", adverse=1,
         source="Malik 2018 Nat Genet (MEGASTROKE EUR; no UKB), GCST006909"),
    dict(id="las_mega", label="Large-artery stroke (MEGASTROKE EUR)", group="ext_out", kind="bin",
         path="15f_Large_artery_stroke_MEGASTROKE_EUR_noUKB/29531354-GCST006907-EFO_0005524-build37.f.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF, snp="variant_id"), n_default=301_663, ancestry="EUR", ukb=False,
         units="log-odds", unit_label="per log-odds", adverse=1,
         source="Malik 2018 Nat Genet (MEGASTROKE EUR; no UKB), GCST006907"),
    dict(id="aaa_mvp", label="Abdominal aortic aneurysm (MVP EUR phecode)", group="ext_out", kind="bin",
         path="21a_AAA_MVP_EUR/GCST90475994.tsv.gz", sep="\t", build="38", cols=I._MVP, special=("or_ci",),
         n_default=443_400, ancestry="EUR (MVP)", ukb=False, units="log-odds", unit_label="per log-odds", adverse=1,
         source="Verma 2024 Science (MVP EUR, phecode 442.11), GCST90475994"),
    # added 2026-10-05 (review fix): the only open European heart-failure GWAS without UK Biobank found in the GWAS
    # Catalog (FinnGen needs a form; HERMES includes UKB). MVP: overlaps the MVP sleep-apnoea exposure (family A/B 'osa')
    dict(id="hf_mvp", label="Heart failure (MVP EUR phecode 428)", group="ext_out", kind="bin",
         path="16b_Heart_failure_MVP_EUR_noUKB/GCST90475981.tsv.gz", sep="\t", build="38", cols=I._MVP,
         special=("or_ci",), n_default=429_672, ancestry="EUR (MVP)", ukb=False, units="log-odds",
         unit_label="per log-odds", adverse=1,
         source="Verma 2024 Science (MVP EUR, phecode 428 congestive heart failure, nonhypertensive; 58,749 cases), "
                "GCST90475981"),
]
NONUKB_OUT = ["cad_nikpay", "stroke_mega", "istroke_mega", "hf_mvp"]
NONUKB_PAIR = {"cad_nikpay": "cad", "stroke_mega": "stroke", "istroke_mega": "istroke", "hf_mvp": "hf"}
PC_OUT = ["svs_mega", "las_mega", "aaa_mvp"]


def _register_ext_specs():
    """Register the extension specs in analysis_i_mr.SPEC (analysis_i_mr.Scanner looks specs up there)."""
    for s in EXT_SPECS:
        I.SPEC[s["id"]] = s


# ---------------------------------------------------------------------------------------------------------------
# Readers for two files whose layout analysis_i_mr.standardize cannot read (it is not modified):
#  - GERA (Hoffmann 2017): no SE column (SE from beta and p); effect allele in 'Effect allele (EA)', the pair in
#    'Allele 1'/'Allele 2' (other allele = the member of the pair that is not EA).
#  - Pan-UKB MAP: p given as -log10 p; no rsID column.
# Streaming with pyarrow (as analysis_i_mr.scan_worker), cached as parquet in <outdir>/_cache.
# ---------------------------------------------------------------------------------------------------------------
_CUSTOM_COLS = {
    "gera": ["SNP_ID", "chromosome", "position", "Allele 1", "Allele 2", "Effect allele (EA)",
             "Effect allele frequency (EAF)", "Sample size", "Estimate Effect", "P value"],
    "panukb": ["chromosome", "base_pair_location", "effect_allele", "other_allele", "beta", "standard_error",
               "effect_allele_frequency", "neg_log_10_p_value"],
}


def _custom_std(kind: str, raw: pd.DataFrame, n_default: float) -> pd.DataFrame:
    d = pd.DataFrame(index=raw.index)
    if kind == "gera":
        d["chr"] = raw["chromosome"].astype("string").str.strip().str.replace(r"^chr", "", regex=True, case=False)
        d["pos"] = pd.to_numeric(raw["position"], errors="coerce").round().astype("Int64")
        s = raw["SNP_ID"].astype("string").str.strip()
        d["snp"] = s.where(s.str.match(r"^rs\d+$").fillna(False), pd.NA)
        a1 = raw["Allele 1"].astype("string").str.strip().str.upper()
        a2 = raw["Allele 2"].astype("string").str.strip().str.upper()
        ea = raw["Effect allele (EA)"].astype("string").str.strip().str.upper()
        d["ea"] = ea
        d["oa"] = a2.where(ea == a1, a1.where(ea == a2, pd.NA))
        beta = pd.to_numeric(raw["Estimate Effect"], errors="coerce").astype(float)
        p = pd.to_numeric(raw["P value"], errors="coerce").astype(float)
        d["beta"], d["p"] = beta, p
        d["se"] = M.se_from_beta_p(beta, p)
        d["eaf"] = pd.to_numeric(raw["Effect allele frequency (EAF)"], errors="coerce")
        d["n"] = pd.to_numeric(raw["Sample size"], errors="coerce").fillna(n_default).astype(float)
        d["se_source"] = "beta_p"
    else:
        d["chr"] = raw["chromosome"].astype("string").str.strip().str.replace(r"^chr", "", regex=True, case=False)
        d["pos"] = pd.to_numeric(raw["base_pair_location"], errors="coerce").round().astype("Int64")
        d["snp"] = pd.Series(pd.NA, index=raw.index, dtype="string")
        d["ea"] = raw["effect_allele"].astype("string").str.strip().str.upper()
        d["oa"] = raw["other_allele"].astype("string").str.strip().str.upper()
        d["beta"] = pd.to_numeric(raw["beta"], errors="coerce").astype(float)
        d["se"] = pd.to_numeric(raw["standard_error"], errors="coerce").astype(float)
        d["p"] = np.power(10.0, -pd.to_numeric(raw["neg_log_10_p_value"], errors="coerce").astype(float))
        d["eaf"] = pd.to_numeric(raw["effect_allele_frequency"], errors="coerce")
        d["n"] = float(n_default)
        d["se_source"] = "se"
    d["pos38"] = pd.Series(pd.NA, index=raw.index, dtype="Int64")
    return d.reset_index(drop=True)


def scan_custom(spec: dict, path: Path, cache: Path, mode: str, p_thresh: float | None = None,
                rs: set | None = None, pairs: set | None = None) -> pd.DataFrame:
    import hashlib
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    kind = spec["custom"]
    st = path.stat()
    key = hashlib.sha1(json.dumps([spec["id"], str(path), st.st_size, int(st.st_mtime), mode, p_thresh,
                                   hashlib.sha1("\n".join(sorted(rs or [])).encode()).hexdigest(),
                                   hashlib.sha1("\n".join(sorted(pairs or [])).encode()).hexdigest()]).encode()
                       ).hexdigest()[:16]
    f = cache / f"custom_{spec['id']}_{mode}_{key}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    cols = _CUSTOM_COLS[kind]
    types = {c: pa.string() for c in cols}
    for c in ("position", "base_pair_location", "Effect allele frequency (EAF)", "Sample size", "Estimate Effect",
              "P value", "beta", "standard_error", "effect_allele_frequency", "neg_log_10_p_value"):
        if c in cols:
            types[c] = pa.float64()
    reader = pcsv.open_csv(pa.input_stream(str(path), compression="gzip"),
                           read_options=pcsv.ReadOptions(block_size=32 << 20),
                           parse_options=pcsv.ParseOptions(delimiter="\t"),
                           convert_options=pcsv.ConvertOptions(include_columns=cols, column_types=types,
                                                               null_values=I.NA_VALUES, strings_can_be_null=True))
    poscol = "position" if kind == "gera" else "base_pair_location"
    pos_arr = pa.array(sorted({float(x.split(":")[1]) for x in (pairs or set())}), pa.float64())
    rs_arr = pa.array(sorted(rs or []), pa.string())
    parts = []
    t0 = time.time()
    nrows = 0
    for batch in reader:
        nrows += batch.num_rows
        if mode == "sig":
            if kind == "gera":
                m = pc.less_equal(batch.column("P value"), p_thresh)
            else:
                m = pc.greater_equal(batch.column("neg_log_10_p_value"), -np.log10(p_thresh))
        else:
            m = pc.is_in(batch.column(poscol), value_set=pos_arr)
            if kind == "gera" and len(rs_arr):
                m = pc.or_(pc.fill_null(m, False), pc.fill_null(pc.is_in(batch.column("SNP_ID"), value_set=rs_arr),
                                                                False))
        sub = batch.filter(pc.fill_null(m, False))
        if sub.num_rows:
            parts.append(sub)
    raw = pa.Table.from_batches(parts).to_pandas() if parts else pd.DataFrame(columns=cols)
    d = _custom_std(kind, raw, spec["n_default"])
    if mode == "keys":
        k = I.cpkey(d["chr"], d["pos"])
        keep = k.isin(pairs or set()).fillna(False)
        if rs:
            keep |= d["snp"].isin(rs).fillna(False)
        d = d[keep].reset_index(drop=True)
    d.to_parquet(f, index=False)
    log(f"  custom {mode} {spec['id']}: {nrows:,} rows scanned, {len(d):,} kept, {time.time() - t0:.1f}s")
    return d


def SPEC(i):
    return I.SPEC[i]


# ================================================================================================================
# Small utilities
# ================================================================================================================
T0 = time.time()


def log(msg: str):
    print(f"[{_dt.datetime.now():%H:%M:%S} +{time.time() - T0:7.1f}s] {msg}", flush=True)


def fmt_ci(e, lo, hi, nd=3):
    try:
        if not np.isfinite(e):
            return ""
    except TypeError:
        return ""
    return f"{e:.{nd}g} [{lo:.{nd}g}, {hi:.{nd}g}]"


def sha_cached(paths: list[Path], cache_file: Path, seed_files: list[Path], workers: int) -> dict:
    """SHA-256 per file with a cache keyed by path|size|mtime (seeded read-only from other drivers' caches)."""
    hc = {}
    for f in seed_files:
        if f.exists():
            try:
                hc.update(json.loads(f.read_text()))
            except Exception:
                pass
    own = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    hc.update(own)
    out, todo = {}, []
    for p in paths:
        st = p.stat()
        key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        if key in hc:
            out[str(p)] = hc[key]
            own[key] = hc[key]
        else:
            todo.append((p, key))
    if todo:
        log(f"SHA-256 of {len(todo)} files")
        res = I.run_pool(I.hash_worker, [str(p) for p, _ in todo], workers)
        for (p, key), (_, h) in zip(todo, res):
            out[str(p)] = own[key] = h
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(own, indent=1))
    return out


# ================================================================================================================
# LD reference: light .bim index (pyarrow) + memory-mapped .bed
# ================================================================================================================
class LDRef:
    """PLINK 1 SNP-major reference (prefix.bed/.bim/.fam [+ .synonyms]) for LD look-ups.

    The .bim is held as a pyarrow table (22.7 M rows for g1000_eur) and searched with pyarrow.compute.is_in, which
    keeps memory near 1 GB; genotypes are read from a memory map only for the variants being clumped.
    """

    COLS = ["chr", "snp", "cm", "pos", "a1", "a2"]

    def __init__(self, prefix, use_synonyms: bool = True):
        import pyarrow as pa
        import pyarrow.csv as pcsv
        self.prefix = str(prefix)
        self.bim = pcsv.read_csv(
            self.prefix + ".bim",
            read_options=pcsv.ReadOptions(column_names=self.COLS, block_size=64 << 20),
            parse_options=pcsv.ParseOptions(delimiter="\t"),
            convert_options=pcsv.ConvertOptions(
                include_columns=["chr", "snp", "pos", "a1", "a2"],
                column_types={"chr": pa.string(), "snp": pa.string(), "pos": pa.int64(), "a1": pa.string(),
                              "a2": pa.string()}))
        self.n_snps = self.bim.num_rows
        fam = pd.read_csv(self.prefix + ".fam", sep=r"\s+", header=None)
        self.n = len(fam)
        self.bps = (self.n + 3) // 4
        bed = self.prefix + ".bed"
        with open(bed, "rb") as fh:
            magic = fh.read(3)
        if magic != M.PlinkBed.MAGIC:
            raise ValueError("not a SNP-major PLINK 1 .bed file")
        size = os.path.getsize(bed)
        if size != 3 + self.n_snps * self.bps:
            raise ValueError(f".bed size {size} does not match {self.n_snps} SNPs x {self.bps} bytes + 3")
        self._mm = np.memmap(bed, dtype=np.uint8, mode="r", offset=3, shape=(self.n_snps, self.bps))
        syn = self.prefix + ".synonyms"
        self.syn_path = syn if (use_synonyms and os.path.exists(syn)) else None

    # ---------------- look-ups ----------------
    def _take(self, mask) -> pd.DataFrame:
        import pyarrow.compute as pc
        m = pc.fill_null(mask, False)
        idx = np.flatnonzero(np.asarray(m.to_numpy(zero_copy_only=False), dtype=bool))
        d = self.bim.take(idx).to_pandas()
        d.insert(0, "ref_idx", idx.astype(np.int64))
        d["chr"] = d["chr"].astype("string").str.replace(r"^chr", "", regex=True, case=False)
        for c in ("snp", "a1", "a2"):
            d[c] = d[c].astype("string").str.upper() if c != "snp" else d[c].astype("string")
        return d

    def rows_by_rsid(self, rsids) -> pd.DataFrame:
        import pyarrow as pa
        import pyarrow.compute as pc
        rs = sorted({str(x) for x in rsids if isinstance(x, str) or pd.notna(x)})
        if not rs:
            return self._take(pc.is_in(self.bim["snp"], value_set=pa.array([], pa.string())))
        return self._take(pc.is_in(self.bim["snp"], value_set=pa.array(rs, pa.string())))

    def rows_by_pos(self, positions) -> pd.DataFrame:
        import pyarrow as pa
        import pyarrow.compute as pc
        ps = sorted({int(x) for x in positions if pd.notna(x)})
        return self._take(pc.is_in(self.bim["pos"], value_set=pa.array(ps, pa.int64())))

    def synonyms_for(self, rsids: set) -> dict:
        """rsID -> list of its dbSNP-151 synonyms (MAGMA .synonyms file: one line per synonym set, numbers only)."""
        if not self.syn_path or not rsids:
            return {}
        want = {s[2:] for s in rsids if isinstance(s, str) and s.startswith("rs") and s[2:].isdigit()}
        out = {}
        with open(self.syn_path) as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                toks = line.split()
                if want.isdisjoint(toks):
                    continue
                for t in toks:
                    if t in want:
                        out["rs" + t] = ["rs" + u for u in toks if u != t]
        return out

    def match(self, d: pd.DataFrame) -> pd.DataFrame:
        """Match variants to the panel. d needs snp, chr, pos (GRCh37; may be NA), ea, oa.

        Returns a frame on d.index: ref_idx (-1 = unmatched), ref_chr, ref_pos, ref_snp, how ('rsid', 'synonym',
        'chrpos', 'none') and reason (for unmatched rows)."""
        res = pd.DataFrame(index=d.index)
        res["ref_idx"] = np.int64(-1)
        res["ref_chr"] = pd.Series(pd.NA, index=d.index, dtype="string")
        res["ref_pos"] = pd.Series(pd.NA, index=d.index, dtype="Int64")
        res["ref_snp"] = pd.Series(pd.NA, index=d.index, dtype="string")
        res["how"] = "none"
        res["reason"] = "not in panel"
        if len(d) == 0:
            return res
        ak = I.akey(d["ea"].astype("string"), d["oa"].astype("string"))
        akc = I.akey_comp(d["ea"].astype("string"), d["oa"].astype("string"))
        dchr = d["chr"].astype("string")

        def accept(m: pd.DataFrame, how: str):
            """m: rows with column 'row' (index label in d) and panel columns; first acceptable panel row wins."""
            if m.empty:
                return
            rk = I.akey(m["a1"], m["a2"])
            a_ok = np.asarray(((rk == ak.loc[m["row"]].values) | (rk == akc.loc[m["row"]].values))
                              .fillna(False), dtype=bool)
            dc = dchr.loc[m["row"]].values
            c_ok = np.array([(x is pd.NA) or (str(x) == str(y)) for x, y in zip(dc, m["chr"].values)], dtype=bool)
            ok = a_ok & c_ok
            free = np.asarray(res.loc[m["row"], "ref_idx"].values < 0, dtype=bool)
            bad = m[~ok & free]
            for r, why in zip(bad["row"], np.where(~c_ok[~ok & free], "chromosome mismatch", "allele mismatch")):
                if res.at[r, "ref_idx"] < 0:
                    res.at[r, "reason"] = why
            good = m[ok & free].drop_duplicates("row")
            if good.empty:
                return
            res.loc[good["row"], "ref_idx"] = good["ref_idx"].values
            res.loc[good["row"], "ref_chr"] = good["chr"].values
            res.loc[good["row"], "ref_pos"] = good["pos"].astype("Int64").values
            res.loc[good["row"], "ref_snp"] = good["snp"].values
            res.loc[good["row"], "how"] = how
            res.loc[good["row"], "reason"] = ""

        # 1) rsID
        has = d["snp"].notna()
        R = None
        if has.any():
            R = self.rows_by_rsid(d.loc[has, "snp"].unique())
            m = pd.DataFrame({"row": d.index[has], "snp": d.loc[has, "snp"].astype("string").values}).merge(
                R, on="snp", how="inner")
            accept(m, "rsid")
        # 2) synonyms for rsIDs absent from the panel
        if self.syn_path is not None:
            present = set(R["snp"]) if R is not None else set()
            miss = has & (res["ref_idx"] < 0) & ~d["snp"].isin(present).fillna(False)
            if miss.any():
                syn = self.synonyms_for(set(d.loc[miss, "snp"].astype(str)))
                pairs = [(r, s2) for r, s in zip(d.index[miss], d.loc[miss, "snp"].astype(str))
                         for s2 in syn.get(s, [])]
                if pairs:
                    pm = pd.DataFrame(pairs, columns=["row", "snp"])
                    R2 = self.rows_by_rsid(pm["snp"].unique())
                    accept(pm.merge(R2, on="snp", how="inner"), "synonym")
        # 3) GRCh37 chr:pos + alleles
        rest = (res["ref_idx"] < 0) & d["pos"].notna() & d["chr"].notna()
        if rest.any():
            R3 = self.rows_by_pos(d.loc[rest, "pos"].astype("Int64").unique())
            q = pd.DataFrame({"row": d.index[rest], "chr": dchr[rest].values,
                              "pos": d.loc[rest, "pos"].astype("Int64").astype("int64").values})
            m3 = q.merge(R3.rename(columns={"chr": "chr_ref"}), on="pos", how="inner")
            m3 = m3[m3["chr"].astype(str) == m3["chr_ref"].astype(str)].drop(columns=["chr"]).rename(
                columns={"chr_ref": "chr"})
            accept(m3, "chrpos")
        nores = (res["ref_idx"] < 0) & d["snp"].isna() & d["pos"].isna()
        res.loc[nores, "reason"] = "no rsID and no GRCh37 position"
        return res

    # ---------------- genotypes ----------------
    def genotypes(self, idx) -> np.ndarray:
        """int8 (n_samples, k) A1 allele counts, missing = -1."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        o = np.argsort(idx, kind="mergesort")
        raw = np.asarray(self._mm[idx[o]])
        g = M._bed_lut()[raw].reshape(len(idx), self.bps * 4)[:, : self.n]
        out = np.empty_like(g)
        out[o] = g
        return out.T

    def a1_freq(self, idx, chunk: int = 200_000) -> np.ndarray:
        """Frequency of the .bim A1 allele (column 5) among non-missing genotypes, per variant."""
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        out = np.full(len(idx), np.nan)
        for s in range(0, len(idx), chunk):
            G = self.genotypes(idx[s:s + chunk]).astype(float)
            G[G < 0] = np.nan
            with np.errstate(invalid="ignore"):
                out[s:s + chunk] = np.nanmean(G, axis=0) / 2.0
        return out

    def std_rows(self, idx) -> np.ndarray:
        """float64 (k, n_samples): centred, unit-norm allele counts per variant (monomorphic rows = 0), so that
        std_rows(a) @ std_rows(b).T is the Pearson correlation matrix."""
        G = self.genotypes(idx).T.astype(float)
        G[G < 0] = np.nan
        mu = np.nanmean(G, axis=1, keepdims=True) if G.size else G
        G = np.where(np.isnan(G), mu, G) - mu
        nrm = np.sqrt((G ** 2).sum(axis=1, keepdims=True))
        with np.errstate(invalid="ignore", divide="ignore"):
            X = np.where(nrm > 0, G / nrm, 0.0)
        return X


def clump_ld_greedy(ref_chr, ref_pos, ref_idx, p, z, ldref: LDRef, r2_thresh: float = R2_THRESH,
                    window_kb: float = LD_WINDOW_KB, p_thresh: float = P_THRESH) -> tuple[np.ndarray, np.ndarray]:
    """Greedy LD clumping. Eligible: p <= p_thresh and ref_idx >= 0. Index order: p ascending, |z| descending, then
    input order (the tie rule of analysis_i_mr.clump_fast). For each index variant, every remaining eligible variant
    on the same chromosome within +/- window with r^2 > r2_thresh is removed.

    Returns keep (bool array) and clumped_by (input position of the index variant that removed a row; -1 otherwise).
    """
    p = np.asarray(p, float)
    n = len(p)
    ref_idx = np.asarray(ref_idx, np.int64)
    ref_chr = np.asarray(pd.Series(ref_chr).astype("string").fillna("").astype(str))
    ref_pos = np.asarray(pd.Series(ref_pos).astype("Float64").fillna(-1).astype(float))
    z = np.nan_to_num(np.abs(np.asarray(z, float)))
    keep = np.zeros(n, bool)
    by = np.full(n, -1, np.int64)
    elig = (p <= p_thresh) & (ref_idx >= 0) & (ref_chr != "")
    order = np.lexsort((np.arange(n), -z, p))
    w = window_kb * 1000.0
    for c in np.unique(ref_chr[elig]):
        rows = np.flatnonzero(elig & (ref_chr == c))
        srt = rows[np.argsort(ref_pos[rows], kind="mergesort")]
        P = ref_pos[srt]
        X = ldref.std_rows(ref_idx[srt])
        loc = np.full(n, -1, np.int64)
        loc[srt] = np.arange(len(srt))
        alive = np.ones(len(srt), bool)
        for i in order[loc[order] >= 0]:
            j = loc[i]
            if not alive[j]:
                continue
            keep[i] = True
            alive[j] = False
            lo = np.searchsorted(P, P[j] - w, "left")
            hi = np.searchsorted(P, P[j] + w, "right")
            cand = lo + np.flatnonzero(alive[lo:hi])
            if cand.size == 0:
                continue
            r = X[cand] @ X[j]
            drop = cand[r * r > r2_thresh]
            alive[drop] = False
            by[srt[drop]] = i
    return keep, by


# ================================================================================================================
# Self-test (planted answers on a synthetic panel)
# ================================================================================================================
def _synthetic_panel(rng, n=503, m_per_chr=300, chroms=("1", "2"), step=100_000):
    """Haplotypes from a Markov copying process (LD decays with distance); returns G (n x m) and the bim frame."""
    G_parts, bims = [], []
    k = 0
    for c in chroms:
        f = rng.uniform(0.05, 0.5, m_per_chr)
        H = np.zeros((2 * n, m_per_chr), np.int8)
        H[:, 0] = rng.random(2 * n) < f[0]
        for j in range(1, m_per_chr):
            copy = rng.random(2 * n) < 0.93
            H[:, j] = np.where(copy, H[:, j - 1], rng.random(2 * n) < f[j])
        G_parts.append(H[:n] + H[n:])
        pos = (np.arange(m_per_chr) + 1) * step + rng.integers(0, 1000, m_per_chr)
        alle = rng.choice([("A", "G"), ("C", "T"), ("A", "C"), ("G", "T")], m_per_chr)
        bims.append(pd.DataFrame({"chr": c, "snp": [f"rs{k + j + 1}" for j in range(m_per_chr)], "cm": 0,
                                  "pos": pos, "a1": [a[0] for a in alle], "a2": [a[1] for a in alle]}))
        k += m_per_chr
    return np.concatenate(G_parts, axis=1), pd.concat(bims, ignore_index=True)


def selftest(seed: int, workdir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tmp = workdir / "_selftest_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    rows = []
    try:
        G, bim = _synthetic_panel(rng)
        prefix = str(tmp / "panel")
        M.write_plink_bed(prefix, G, bim)
        with open(prefix + ".synonyms", "w") as fh:
            fh.write("# PREFIX = rs\n999001 7\n999002 888888\n")
        ref = LDRef(prefix)
        # 1) correlations from the memory map equal numpy's
        ii = rng.integers(0, len(bim), (60, 2))
        X = ref.std_rows(np.unique(ii.ravel()))
        pos_of = {v: k for k, v in enumerate(np.unique(ii.ravel()))}
        diffs = [abs(float(X[pos_of[a]] @ X[pos_of[b]]) - float(np.corrcoef(G[:, a], G[:, b])[0, 1]))
                 for a, b in ii]
        rows.append(("r from memory-mapped .bed equals numpy corrcoef (max abs diff)", max(diffs), 1e-10,
                     max(diffs) < 1e-10))
        jj = rng.integers(0, len(bim), 50)
        fd = float(np.max(np.abs(ref.a1_freq(jj) - G[:, jj].mean(axis=0) / 2)))
        rows.append(("A1 frequency from the .bed equals numpy (max abs diff)", fd, 1e-12, fd < 1e-12))
        # 2) clumping equals lib_mr.clump_ld (PlinkBed) for two parameter sets
        m = len(bim)
        p = np.where(rng.random(m) < 0.6, 10 ** -rng.uniform(8.0, 30.0, m), rng.uniform(1e-6, 1, m))
        df = bim[["snp", "chr", "pos"]].copy()
        df["p"] = p
        df["ea"], df["oa"] = bim["a1"].values, bim["a2"].values
        mt = ref.match(df)
        for r2t, wkb in ((0.001, 10000.0), (0.1, 1000.0)):
            mine, by = clump_ld_greedy(mt["ref_chr"], mt["ref_pos"], mt["ref_idx"], df["p"].values,
                                       np.zeros(m), ref, r2_thresh=r2t, window_kb=wkb)
            lib = M.clump_ld(df.assign(chr=df["chr"].astype(str)), M.PlinkBed(prefix), p_thresh=P_THRESH,
                             r2_thresh=r2t, window_kb=wkb)
            same = set(df["snp"][mine]) == set(lib.loc[lib["kept"], "snp"])
            rows.append((f"kept set equals lib_mr.clump_ld (r2 {r2t}, {wkb:g} kb): n kept", int(mine.sum()),
                         int(lib["kept"].sum()), bool(same)))
            # properties: kept pairs within window have r2 <= t; each removed SNP has r2 > t with its index
            k = np.flatnonzero(mine)
            Xk = ref.std_rows(k)
            R2 = (Xk @ Xk.T) ** 2
            same_chr = df["chr"].values[k][:, None] == df["chr"].values[k][None, :]
            near = np.abs(df["pos"].values[k][:, None] - df["pos"].values[k][None, :]) <= wkb * 1000
            off = same_chr & near & ~np.eye(len(k), dtype=bool)
            mx = float(R2[off].max()) if off.any() else 0.0
            rows.append((f"max r2 between kept variants within window (r2 {r2t})", mx, r2t, mx <= r2t))
            rem = np.flatnonzero(by >= 0)
            r2r = np.array([float(ref.std_rows([a])[0] @ ref.std_rows([b])[0]) ** 2 for a, b in zip(rem, by[rem])])
            rows.append((f"min r2 of removed variant with its index (r2 {r2t})", float(r2r.min()) if len(r2r) else
                         np.nan, r2t, bool(len(r2r) == 0 or r2r.min() > r2t)))
        # 3) matching rules
        b = bim.iloc[:6]
        comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
        q = pd.DataFrame({
            "snp": [b.snp[0], b.snp[1], b.snp[2], pd.NA, b.snp[4], "rs999001", pd.NA],
            "chr": [b.chr[0], b.chr[1], b.chr[2], b.chr[3], "9", pd.NA, pd.NA],
            "pos": [b.pos[0], b.pos[1], b.pos[2], b.pos[3], pd.NA, pd.NA, pd.NA],
            "ea": [b.a2[0], comp[b.a1[1]], "A", b.a1[3], b.a1[4], bim.a1[6], "A"],
            "oa": [b.a1[0], comp[b.a2[1]], "A", b.a2[3], b.a2[4], bim.a2[6], "G"]})
        # row 2: alleles A/A never match a real pair -> allele mismatch
        q.loc[2, ["ea", "oa"]] = ["T", "T"] if set(b[["a1", "a2"]].iloc[2]) != {"T"} else ["A", "A"]
        q["pos"] = q["pos"].astype("Int64")
        q["chr"] = q["chr"].astype("string")
        q["snp"] = q["snp"].astype("string")
        mm = ref.match(q)
        exp_how = ["rsid", "rsid", "none", "chrpos", "none", "synonym", "none"]
        exp_idx = [0, 1, -1, 3, -1, 6, -1]
        ok = (list(mm["how"]) == exp_how) and (list(mm["ref_idx"]) == exp_idx)
        rows.append(("match: rsID, strand complement, allele mismatch, chr:pos, wrong chromosome, synonym, none",
                     "|".join(mm["how"]), "|".join(exp_how), bool(ok)))
        rows.append(("match: reasons for unmatched rows", "|".join(mm.loc[mm["ref_idx"] < 0, "reason"]),
                     "allele mismatch|chromosome mismatch|no rsID and no GRCh37 position",
                     list(mm.loc[mm["ref_idx"] < 0, "reason"]) ==
                     ["allele mismatch", "chromosome mismatch", "no rsID and no GRCh37 position"]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return pd.DataFrame(rows, columns=["check", "value", "expected", "pass"])


# ================================================================================================================
# Instrument selection with LD clumping
# ================================================================================================================
def ld_match_candidates(c2: dict, ldref: LDRef) -> pd.DataFrame:
    """Per-candidate panel match plus the EAF rules (see ld_select)."""
    allc = pd.concat([d.assign(_exp=k, _row=np.arange(len(d))) for k, d in c2.items()], ignore_index=True)
    log(f"LD panel matching: {len(allc):,} candidate rows from {len(c2)} exposures")
    t0 = time.time()
    mt = ldref.match(allc[["snp", "chr", "pos", "ea", "oa"]])
    log(f"  matched in {time.time() - t0:.1f}s: " + ", ".join(f"{k}={v:,}" for k, v in mt["how"].value_counts().items()))
    allc = pd.concat([allc, mt], axis=1)
    # Panel MAF of every matched row without GWAS EAF (reported; used by the panel-MAF rule below).
    allc["maf_panel"] = np.nan
    need = allc["eaf"].isna() & (allc["ref_idx"] >= 0)
    if need.any():
        f = ldref.a1_freq(allc.loc[need, "ref_idx"].values)
        allc.loc[need, "maf_panel"] = np.minimum(f, 1 - f)
    # Rule 1 (review fix, 2026-10-05): rows without EAF in a file that reports EAF for its other rows are excluded
    # before clumping. Only the MAGIC FG and HbA1c files have such rows; they come from a small subset of the
    # contributing samples (N 283 to 17,302 of about 150,000-200,000) and their reported p is often not supported
    # by beta/SE (up to 18 orders of magnitude in either direction), so p-ranked selection cannot rely on them.
    # Before this rule, the run removed them only by luck of the clumping order (absent from the panel, or clumped
    # away by a stronger neighbour); ld_select checks that the instrument sets are unchanged.
    file_has_eaf = allc.groupby("_exp")["eaf"].transform(lambda s: bool(s.notna().any())).astype(bool)
    allc["no_eaf_row_excluded"] = allc["eaf"].isna() & file_has_eaf
    # Rule 2: in files without any EAF the variant needs MAF >= 0.01 in the panel (i_mr requires MAF >= 0.01 only
    # where the GWAS reports EAF; rare panel variants also give unstable r^2 with 503 samples). No file of the
    # current panel lacks EAF entirely; the flag is still reported for every no-EAF row.
    allc["rare_in_panel_no_eaf"] = need & (allc["maf_panel"] < I.MAF_MIN)
    return allc


def _p_implied(beta, se) -> np.ndarray:
    z = np.abs(np.asarray(beta, float) / np.asarray(se, float))
    return 2 * stats.norm.sf(z)


def ld_select(c2: dict, ldref: LDRef, p_thresh_of: dict, r2_thresh: float = R2_THRESH,
              allc: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """c2: exposure -> filtered candidates (analysis_i_mr.candidate_filter output). Returns LD-clumped instruments
    (with ref columns; GRCh37 chr/pos and rsID filled from the panel), per-exposure match/clump statistics, the
    per-candidate match table and the disposition of every excluded no-EAF candidate. `allc` (output of
    ld_match_candidates) may be passed to reuse the panel match."""
    if allc is None:
        allc = ld_match_candidates(c2, ldref)
    inst, srows, disp = {}, [], []
    for k, d in c2.items():
        a = allc[allc["_exp"] == k].reset_index(drop=True)
        pth = p_thresh_of.get(k, P_THRESH)
        t1 = time.time()
        z = (a["beta"] / a["se"]).values
        excl = (a["no_eaf_row_excluded"] | a["rare_in_panel_no_eaf"]).values
        p_eff = np.where(excl, np.inf, a["p"].values)
        keep, by = clump_ld_greedy(a["ref_chr"], a["ref_pos"], a["ref_idx"], p_eff, z, ldref,
                                   r2_thresh=r2_thresh, p_thresh=pth)
        same_prev = np.nan
        nx = (a["no_eaf_row_excluded"] & (a["p"] <= pth)).values
        if nx.any():
            # the previous rule (panel MAF only): would any excluded row have been an instrument?
            p_old = np.where(a["rare_in_panel_no_eaf"].values, np.inf, a["p"].values)
            keep_o, by_o = clump_ld_greedy(a["ref_chr"], a["ref_pos"], a["ref_idx"], p_old, z, ldref,
                                           r2_thresh=r2_thresh, p_thresh=pth)
            same_prev = bool(np.array_equal(keep, keep_o))
            pimp = _p_implied(a["beta"], a["se"])
            for i in np.flatnonzero(nx):
                if a.at[i, "ref_idx"] < 0:
                    st_ = "absent from the panel"
                elif a.at[i, "rare_in_panel_no_eaf"]:
                    st_ = "panel MAF < 0.01 (panel-MAF rule)"
                elif keep_o[i]:
                    st_ = "index variant (would have been an instrument)"
                elif by_o[i] >= 0:
                    j = by_o[i]
                    st_ = (f"clumped away by a stronger neighbour ({a.at[j, 'ref_snp']}, p {a.at[j, 'p']:.2g}, "
                           f"GWAS EAF {'missing' if pd.isna(a.at[j, 'eaf']) else 'present'})")
                else:
                    st_ = "not eligible"
                disp.append({"r2_thresh": r2_thresh, "exposure": k, "snp": a.at[i, "snp"], "chr": a.at[i, "chr"],
                             "pos": a.at[i, "pos"], "ea": a.at[i, "ea"], "oa": a.at[i, "oa"],
                             "beta": a.at[i, "beta"], "se": a.at[i, "se"], "n": a.at[i, "n"],
                             "p_reported": a.at[i, "p"], "p_from_beta_se": pimp[i],
                             "log10_p_reported_over_implied": float(np.log10(a.at[i, "p"]) - np.log10(pimp[i]))
                             if pimp[i] > 0 else np.nan,
                             "panel_maf": a.at[i, "maf_panel"], "fate_under_previous_rule": st_})
        x = a[keep].copy()
        # GRCh37 coordinates and rsID from the panel where the GWAS lacks them
        fill_pos = x["pos"].isna()
        x.loc[fill_pos, "pos"] = x.loc[fill_pos, "ref_pos"].astype("Int64")
        x.loc[fill_pos, "chr"] = x.loc[fill_pos, "ref_chr"]
        fill_rs = x["snp"].isna()
        x.loc[fill_rs, "snp"] = x.loc[fill_rs, "ref_snp"]
        x["pos"] = x["pos"].astype("Int64")
        inst[k] = x.drop(columns=["_exp", "_row"]).reset_index(drop=True)
        el = a["p"] <= pth
        srows.append({"exposure": k, "candidates": int(el.sum()),
                      "matched_rsid": int((el & (a["how"] == "rsid")).sum()),
                      "matched_synonym": int((el & (a["how"] == "synonym")).sum()),
                      "matched_chrpos": int((el & (a["how"] == "chrpos")).sum()),
                      "unmatched_not_in_panel": int((el & (a["reason"] == "not in panel")).sum()),
                      "unmatched_allele_mismatch": int((el & (a["reason"] == "allele mismatch")).sum()),
                      "unmatched_chromosome_mismatch": int((el & (a["reason"] == "chromosome mismatch")).sum()),
                      "unmatched_no_id": int((el & (a["reason"] == "no rsID and no GRCh37 position")).sum()),
                      "no_eaf_in_gwas": int((el & a["eaf"].isna()).sum()),
                      "excluded_no_eaf_row": int((el & a["no_eaf_row_excluded"]).sum()),
                      "no_eaf_row_in_panel_maf_ge_0.01": int((el & a["no_eaf_row_excluded"] & (a["ref_idx"] >= 0)
                                                              & ~a["rare_in_panel_no_eaf"]).sum()),
                      "instruments_ld": int(keep.sum()),
                      "instruments_same_without_no_eaf_rule": same_prev,
                      "rsid_filled_from_panel": int(fill_rs.sum()),
                      "pos37_filled_from_panel": int(fill_pos.sum()), "seconds": round(time.time() - t1, 1)})
    return inst, pd.DataFrame(srows), allc, pd.DataFrame(disp)


def refkeys(inst: dict) -> tuple[dict, dict, dict]:
    """rsID -> panel row, GRCh37 chr:pos -> panel row and GRCh37 chr:pos -> rsID, from all LD-matched instruments."""
    by_rs, by_cp, cp2rs = {}, {}, {}
    for d in inst.values():
        for s, c, p_, r in zip(d["snp"], d["chr"], d["pos"], d["ref_idx"]):
            if pd.notna(s):
                by_rs.setdefault(str(s), int(r))
            if pd.notna(p_) and pd.notna(c):
                by_cp.setdefault(f"{c}:{int(p_)}", int(r))
                if pd.notna(s) and str(s).startswith("rs"):
                    cp2rs.setdefault(f"{c}:{int(p_)}", str(s))
    return by_rs, by_cp, cp2rs


def mvmr_ld(xids: list, oid: str, family: str, ext: dict, inst: dict, keys: tuple, ldref: LDRef,
            drop_pal: bool, r2_thresh: float = R2_THRESH) -> list[dict]:
    """analysis_i_mr._mvmr with joint LD clumping instead of joint +/-1 Mb distance clumping."""
    by_rs, by_cp, cp2rs = keys
    xs = [dict(spec=SPEC(x), inst=inst[x], ext=ext[x]) for x in xids]
    base = xs[0]["ext"].copy()
    # first exposure without rsIDs (e.g. Pan-UKB MAP, MAGIC FG): take the instrument rsID by GRCh37 chr:pos so that
    # outcome files matched by rsID only (MEGASTROKE formatted files carry no positions) can be harmonised
    miss = base["snp"].isna() & base["pos"].notna()
    if miss.any():
        base.loc[miss, "snp"] = pd.Series([cp2rs.get(k, pd.NA) for k in I.cpkey(base.loc[miss, "chr"],
                                                                                 base.loc[miss, "pos"])],
                                          index=base.index[miss], dtype="string")
    base = I.with_ids(base)
    rs_c, pos_c = set(), set()
    for x in xs:
        d = x["inst"]
        rs_c |= set(d["snp"].dropna())
        ok = d["pos"].notna()
        pos_c |= set(I.cpkey(d.loc[ok, "chr"], d.loc[ok, "pos"]))
    isc = base["snp"].isin(rs_c).fillna(False) | I.cpkey(base["chr"], base["pos"]).isin(pos_c).fillna(False)
    cand = base[isc.values].copy()
    names = list(xids)
    tab = cand[["snp", "chr", "pos", "ea", "oa", "beta", "se", "p", "eaf", "n"]].rename(
        columns={"beta": f"b_{names[0]}", "se": f"s_{names[0]}", "p": f"p_{names[0]}"})
    for x in xs[1:]:
        h, _ = I._harm(cand, x["ext"])
        nm = x["spec"]["id"]
        tab = tab.merge(h[["snp", "beta_y", "se_y", "p_y"]].rename(
            columns={"beta_y": f"b_{nm}", "se_y": f"s_{nm}", "p_y": f"p_{nm}"}), on="snp", how="inner")
    ho, _ = I._harm(cand, ext[oid], drop_pal)
    tab = tab.merge(ho[["snp", "beta_y", "se_y"]].rename(columns={"beta_y": "by", "se_y": "sy"}), on="snp",
                    how="inner")
    tab = tab[tab["pos"].notna()].reset_index(drop=True)
    base_row = {"family": family, "exposures": "+".join(names), "outcome": oid}
    if len(tab) == 0:
        return [{**base_row, "nsnp": 0, "error": "no SNPs"}]
    ridx = np.array([by_rs.get(str(s), by_cp.get(f"{c}:{int(p_)}", -1)) if pd.notna(s) else
                     by_cp.get(f"{c}:{int(p_)}", -1) for s, c, p_ in zip(tab["snp"], tab["chr"], tab["pos"])],
                    np.int64)
    rchr = np.where(ridx >= 0, tab["chr"].astype(str).values, "")
    pmin = tab[[f"p_{nm}" for nm in names]].min(axis=1).values
    zmax = np.max(np.abs(np.column_stack([tab[f"b_{nm}"] / tab[f"s_{nm}"] for nm in names])), axis=1)
    keep, _ = clump_ld_greedy(rchr, tab["pos"].astype(float).values, ridx, pmin, zmax, ldref, r2_thresh=r2_thresh,
                              p_thresh=P_THRESH)
    n_noref = int(((pmin <= P_THRESH) & (ridx < 0)).sum())
    tab = tab[keep].reset_index(drop=True)
    J, K = len(tab), len(names)
    base_row.update({"nsnp": J, "n_union_not_in_panel": n_noref})
    if J <= K + 1:
        return [{**base_row, "error": f"too few SNPs ({J})"}]
    scales = np.array([float(x["spec"].get("scale", 1.0)) for x in xs])
    BX = np.column_stack([tab[f"b_{nm}"].values for nm in names]) / scales
    SX = np.column_stack([tab[f"s_{nm}"].values for nm in names]) / scales
    r = M.mvmr_ivw(BX, tab["by"].values, SX, tab["sy"].values, names=names)
    rows = []
    for k, rr in enumerate(r["results"]):
        uni = M.mr_ivw(BX[:, k], tab["by"].values, tab["sy"].values, "random")
        rows.append({**base_row, "term": names[k], "term_unit": xs[k]["spec"].get("unit_label", ""),
                     "estimate": rr["estimate"], "se": rr["se"], "lower": rr["lower"], "upper": rr["upper"],
                     "p": rr["p"], "conditional_F": rr["conditional_F"], "Q_A": r["Q_A"], "Q_A_df": r["Q_A_df"],
                     "Q_A_p": r["Q_A_p"], "ivw_same_snps": uni["estimate"], "ivw_same_snps_se": uni["se"],
                     "n_term_instruments": int((tab[f"p_{names[k]}"] <= P_THRESH).sum())})
    return rows


# ================================================================================================================
# Comparison helpers
# ================================================================================================================
COARSE = {"robust": "robust", "supported (FDR) - sensitivity weaker": "supported",
          "suggestive (p<0.05, consistent)": "suggestive", "inconsistent direction": "inconsistent direction",
          "no evidence (p>=0.05)": "no evidence", "no evidence (<3 SNPs)": "no evidence",
          "insufficient (<3 SNPs)": "insufficient (<3 SNPs)", "no estimate": "no estimate"}
CLASS_ORDER = ["robust", "supported", "suggestive", "inconsistent direction", "no evidence",
               "insufficient (<3 SNPs)", "no estimate", "pair not fitted"]
SUPPORTED = {"robust", "supported"}
MAIN_FAMS = ["A_risk_to_beds", "A2_osa_alternatives", "B_risk_to_outcomes_posctrl", "C_chami_uncoupled",
             "E_beds_to_outcomes"]
KEY_PAIRS = [("A_risk_to_beds", "sbp", o) for o in ["asi", "cimt", "plaque", "crae", "crve", "avr", "tort", "dens",
                                                     "pad_mvp"]] + [
    ("A_risk_to_beds", "whradjbmi", "asi"), ("A_risk_to_beds", "bmi", "asi"), ("A_risk_to_beds", "bmi", "cimt"),
    ("A_risk_to_beds", "bmi", "crve"), ("A_risk_to_beds", "tg", "asi"), ("A_risk_to_beds", "tg", "tort"),
    ("A_risk_to_beds", "osa", "cimt"), ("A_risk_to_beds", "hba1c", "asi"), ("A_risk_to_beds", "fg", "cimt"),
    ("A_risk_to_beds", "t2d", "pad_mvp"), ("C_chami_uncoupled", "chami_bfp", "asi"),
    ("C_chami_uncoupled", "chami_bfp", "crve"), ("E_beds_to_outcomes", "crve", "istroke"),
    ("E_beds_to_outcomes", "crve", "stroke"), ("E_beds_to_outcomes", "crve", "cad"),
    ("E_beds_to_outcomes", "crve", "hf")]


IMR_SET = {f: "distance-clumped +/-1 Mb (i_mr)" for f in MAIN_FAMS}
IMR_SET["C_chami_uncoupled"] = "unclumped: all 266 published lead variants (i_mr did not clump family C)"


def coarse(x) -> str:
    if not isinstance(x, str) or x == "":
        return "pair not fitted"
    return COARSE.get(x, x)


def transition_tables(cmp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam, g in list(cmp.groupby("family")) + [("all main families", cmp)]:
        ct = pd.crosstab(g["class_distance"], g["class_ld"])
        for a in CLASS_ORDER:
            if a not in ct.index:
                continue
            r = {"family": fam, "class_distance (rows) -> class_ld (columns)": a}
            for b in CLASS_ORDER:
                r[b] = int(ct.loc[a, b]) if b in ct.columns else 0
            r["total"] = int(ct.loc[a].sum())
            rows.append(r)
    return pd.DataFrame(rows)


def class_counts(cmp: pd.DataFrame, r2_sens: float | None = None) -> pd.DataFrame:
    rows = []
    sets = [("distance (i_mr)", "class_distance"), ("LD (this run)", "class_ld")]
    if "class_ld_r2sens" in cmp.columns:
        sets.append((f"LD r2 <= {r2_sens:g} (sensitivity)", "class_ld_r2sens"))
    for fam, g in list(cmp.groupby("family")) + [("all main families", cmp)]:
        for lab, col in sets:
            if fam == "C_chami_uncoupled" and col == "class_distance":
                lab = "distance (i_mr) [family C: unclumped published leads]"
            vc = g[col].value_counts()
            rows.append({"family": fam, "clumping": lab, **{c: int(vc.get(c, 0)) for c in CLASS_ORDER},
                         "n_pairs": len(g)})
    return pd.DataFrame(rows)


# ================================================================================================================
# Main run
# ================================================================================================================
def run(args, outdir: Path):
    rep = C.Report(TASK, "F1 Stage 0 (i3): the MR panel of analysis_i_mr rerun with LD clumping (1000 Genomes EUR)",
                   "real", args)
    t_start = time.time()
    notes, scan_stats = [], []
    _register_ext_specs()

    # ---------------- self-test of the new code ----------------
    st = selftest(args.seed, outdir)
    log(f"selftest: {int(st['pass'].sum())}/{len(st)} pass")
    if not st["pass"].all():
        print(C.md_table(st))
        raise SystemExit("selftest failed; real run not started")

    # ---------------- LD reference provenance ----------------
    ld_files = [LDREF_DIR / f for f in LDREF_SHA256]
    ld_sha = sha_cached([f for f in ld_files if f.exists()], outdir / "_cache" / "sha256_ldref.json", [], 4)
    ld_tab = pd.DataFrame([{"file": str(f.relative_to(HUB)), "bytes": f.stat().st_size if f.exists() else np.nan,
                            "sha256": ld_sha.get(str(f)), "expected_sha256": LDREF_SHA256[f.name],
                            "match": ld_sha.get(str(f)) == LDREF_SHA256[f.name]} for f in ld_files])
    if not ld_tab.loc[ld_tab["file"].str.endswith((".bed", ".bim", ".fam")), "match"].all():
        raise SystemExit("LD reference SHA-256 mismatch")
    log("loading the LD panel index")
    t0 = time.time()
    ldref = LDRef(LDREF_PREFIX)
    log(f"LD panel: {ldref.n_snps:,} variants x {ldref.n} samples ({time.time() - t0:.1f}s)")

    # ---------------- inputs ----------------
    specs_all = I.SPECS + [I.CHAMI] + EXT_SPECS
    paths = {s["id"]: GWAS / s["path"] for s in specs_all}
    avail = {k for k, v in paths.items() if v.exists()}
    for k in sorted(set(paths) - avail):
        notes.append(f"Input missing, skipped: {k} ({paths[k].relative_to(HUB)})")
    scanner = I.Scanner(paths, outdir / "_cache", args.workers, not args.no_cache, scan_stats)

    # ---------------- pass 1: genome-wide significant rows of every exposure ----------------
    ext_exp = ["height", "dbp", "pp", "dbp_gera", "pp_gera", "map_panukb", "osa_campos_bmiadj"]
    p1_ids = [i for i in I.RISK + I.RISK_SENS + I.BEDS + ext_exp if i in avail]
    custom_ids = [i for i in p1_ids if SPEC(i).get("custom")]
    cand = scanner.run([(i, "sig", {"p_thresh": P_THRESH}) for i in p1_ids if i not in custom_ids], "pass1")
    for i in custom_ids:
        cand[i] = scan_custom(SPEC(i), paths[i], outdir / "_cache", "sig", p_thresh=P_THRESH)
    c2, cinfo = {}, {}
    for i in p1_ids:
        native = cand[i]["pos38"] if SPEC(i)["build"] == "38" else cand[i]["pos"]
        c2[i], cinfo[i] = I.candidate_filter(cand[i], native)
    if I.CHAMI["id"] in avail:
        c2[I.CHAMI["id"]] = I.load_chami(paths[I.CHAMI["id"]])
        cinfo[I.CHAMI["id"]] = {"p_le_5e8_rows": int((c2[I.CHAMI["id"]]["p"] <= P_THRESH).sum()),
                                "candidates": len(c2[I.CHAMI["id"]])}
    # GRCh38 files without rsID (Kurniansyah): cross-walk ALL candidates to rsIDs through the MVP file
    xw_rows = []
    for i in [i for i in p1_ids if SPEC(i).get("xwalk_via")]:
        via = SPEC(i)["xwalk_via"]
        if via not in avail:
            notes.append(f"{i}: cross-walk source {via} missing; candidates cannot be matched to the LD panel")
            continue
        prs = set(I.cpkey(c2[i]["chr"], c2[i]["pos38"]))
        refx = scanner.run([(via, "keys", {"pos": sorted({float(x.split(":")[1]) for x in prs}),
                                           "pairs": sorted(prs)})], f"xwalk_{i}")[via]
        s, nf = I.crosswalk_38(c2[i], refx)
        c2[i]["snp"] = s
        xw_rows.append({"exposure": i, "via": via, "candidates": len(c2[i]), "rsid_found": nf})

    # ---------------- LD clumping ----------------
    inst, ldstats, allc, noeaf_disp = ld_select(c2, ldref, {I.CHAMI["id"]: 1.0})
    for _, r in ldstats[ldstats["instruments_same_without_no_eaf_rule"].notna()].iterrows():
        if not bool(r["instruments_same_without_no_eaf_rule"]):
            notes.append(f"{r['exposure']}: the no-EAF exclusion changed the LD instrument set (an excluded row "
                         "would otherwise have been an index variant)")
    # sensitivity: the main-panel exposures clumped at r2 <= args.r2_sens (same window, same rules)
    main_exp = [k for k in dict.fromkeys(I.RISK + I.RISK_SENS + I.BEDS + [I.CHAMI["id"]]) if k in c2]
    inst_s, ldstats_s = {}, pd.DataFrame()
    if args.r2_sens and args.r2_sens > 0:
        inst_s, ldstats_s, _, _ = ld_select({k: c2[k] for k in main_exp}, ldref, {I.CHAMI["id"]: 1.0},
                                            r2_thresh=args.r2_sens, allc=allc[allc["_exp"].isin(main_exp)])
        log(f"instruments (LD r2 <= {args.r2_sens:g} / 10 Mb, sensitivity): "
            + ", ".join(f"{k}={len(v)}" for k, v in inst_s.items()))
    for i, r in cinfo.items():
        ldstats.loc[ldstats["exposure"] == i, "p_le_5e8_rows"] = r.get("p_le_5e8_rows")
    # distance-clumped instruments of i_mr (+/-1 Mb) for the overlap columns
    dist_inst = pd.read_csv(I_MR_DIR / "instruments_1mb.csv", dtype={"chr": str})
    ov = []
    for k, d in inst.items():
        g = dist_inst[dist_inst["exposure"] == k]
        if k in [s["id"] for s in EXT_SPECS] or g.empty:
            ov.append({"exposure": k, "instruments_distance_1mb": np.nan, "in_both": np.nan})
            continue
        ks = set(d["snp"].dropna()) | set(I.cpkey(d["chr"], d["pos"]).dropna())
        gk = set(g["snp"].dropna()) | set(g["chr"].astype(str) + ":" + pd.to_numeric(g["pos"], errors="coerce")
                                          .astype("Int64").astype("string").fillna("NA"))
        hit = [((s in gk) if pd.notna(s) else False) or (f"{c}:{int(p_)}" in gk if pd.notna(p_) else False)
               for s, c, p_ in zip(d["snp"], d["chr"], d["pos"])]
        ov.append({"exposure": k, "instruments_distance_1mb": len(g), "in_both": int(np.sum(hit))})
    ldstats = ldstats.merge(pd.DataFrame(ov), on="exposure", how="left")
    log("instruments (LD r2 <= 0.001 / 10 Mb): " + ", ".join(f"{k}={len(v)}" for k, v in inst.items()))

    # ---------------- pass 2: union of LD instruments in every outcome / MVMR file ----------------
    u_rs, u_pairs = set(), set()
    for d in list(inst.values()) + list(inst_s.values()):
        u_rs |= set(d["snp"].dropna())
        ok = d["pos"].notna()
        u_pairs |= set(I.cpkey(d.loc[ok, "chr"], d.loc[ok, "pos"]))
    u_pos = sorted({float(x.split(":")[1]) for x in u_pairs})
    log(f"union of LD instruments: {len(u_rs):,} rsIDs, {len(u_pairs):,} GRCh37 positions")
    p2_ids = [i for i in (I.RISK + I.BEDS + ["pad_mvp", "pad_sakaue"] + I.CLIN_PC + ext_exp + NONUKB_OUT + PC_OUT)
              if i in avail]
    jobs = []
    for i in dict.fromkeys(p2_ids):
        if SPEC(i).get("custom"):
            continue
        if SPEC(i)["build"] == "38":
            jobs.append((i, "keys", {"rs": sorted(u_rs)}))
        else:
            jobs.append((i, "keys", {"rs": sorted(u_rs), "pos": u_pos, "pairs": sorted(u_pairs)}))
    ext = scanner.run(jobs, "pass2")
    for i in [i for i in dict.fromkeys(p2_ids) if SPEC(i).get("custom")]:
        ext[i] = scan_custom(SPEC(i), paths[i], outdir / "_cache", "keys", rs=u_rs, pairs=u_pairs)
    vmap = I.VariantMap()
    for d in list(inst.values()) + list(inst_s.values()):
        vmap.add(d)
    for i, d in ext.items():
        if SPEC(i)["build"] == "37":
            vmap.add(d)
    for i, d in ext.items():
        if SPEC(i)["build"] == "38":
            vmap.fill_pos37(d)

    # ---------------- EAF QC of the newly downloaded files (allele-label check against Keaton SBP) ----------------
    eafqc = pd.DataFrame()
    if "sbp" in ext:
        qc_sets = {k: ext[k] for k in [s["id"] for s in EXT_SPECS] if k in ext}
        eafqc = I.eaf_qc(ext["sbp"], qc_sets)
        for _, r in eafqc.iterrows():
            if r.get("flag"):
                notes.append(f"EAF concordance with the SBP reference is low for {r['id']} "
                             f"(r = {r['eaf_corr']:.2f}): check allele labels before using it")

    # ---------------- unit check of GERA and Pan-UKB MAP against Keaton (instruments selected in Keaton) ----------
    unit_rows = []
    for gid, kids, wts in (("dbp_gera", ["dbp"], [1.0]), ("pp_gera", ["pp"], [1.0]),
                           ("map_panukb", ["dbp", "pp"], [1.0, 1.0 / 3.0])):
        if gid not in ext or not all(k in ext and k in inst for k in kids):
            continue
        base = I.with_ids(inst[kids[0]])
        tab = base[["snp", "chr", "pos", "ea", "oa", "beta", "se", "p", "eaf", "n"]].copy()
        pred = wts[0] * tab["beta"].values
        ok = np.ones(len(tab), bool)
        if len(kids) > 1:
            h2, _ = I._harm(base, ext[kids[1]])
            m2 = tab[["snp"]].merge(h2[["snp", "beta_y"]], on="snp", how="left")
            pred = pred + wts[1] * m2["beta_y"].values
            ok &= np.isfinite(m2["beta_y"].values)
        hg, _ = I._harm(base, ext[gid])
        mg = tab[["snp"]].merge(hg[["snp", "beta_y"]], on="snp", how="left")
        y = mg["beta_y"].values
        ok &= np.isfinite(y) & np.isfinite(pred)
        slope = float(np.sum(pred[ok] * y[ok]) / np.sum(pred[ok] ** 2)) if ok.sum() > 10 else np.nan
        cor = float(np.corrcoef(pred[ok], y[ok])[0, 1]) if ok.sum() > 10 else np.nan
        # winner's curse inflates Keaton effects at Keaton-selected instruments (slope biased towards zero); the
        # strongest instruments are least affected
        strong = ok & (tab["p"].values <= 1e-20)
        slope_s = float(np.sum(pred[strong] * y[strong]) / np.sum(pred[strong] ** 2)) if strong.sum() > 10 else np.nan
        declared = "SD" if SPEC(gid)["unit_label"] == "per SD" else "mmHg"
        mm = 1.0 / slope if np.isfinite(slope) and slope != 0 else np.nan
        consistent = bool(np.isfinite(mm) and ((declared == "mmHg" and 0.5 <= mm <= 2) or
                                                (declared == "SD" and 5 <= mm <= 25)))
        unit_rows.append({"file": gid, "reference": " + ".join(f"{w:g}*{k}" for k, w in zip(kids, wts)) +
                          " (Keaton, mmHg)", "n_keaton_instruments": int(ok.sum()), "slope_through_origin": slope,
                          "correlation": cor, "declared_unit": declared, "mmHg_per_file_unit": mm,
                          "declared_unit_consistent": consistent,
                          "n_keaton_p_le_1e-20": int(strong.sum()), "slope_keaton_p_le_1e-20": slope_s,
                          "mmHg_per_file_unit_keaton_p_le_1e-20": (1.0 / slope_s if np.isfinite(slope_s) and slope_s
                                                                   else np.nan),
                          "caveat": "approximate: instruments selected in Keaton, so winner's curse inflates the "
                                    "Keaton effects and biases the slope towards zero (ratio understated, mmHg per "
                                    "unit overstated)" + ("; Keaton and Pan-UKB share UK Biobank samples, which "
                                                          "partly offsets this" if gid == "map_panukb" else "")})
        if not consistent:
            notes.append(f"Unit check failed for {gid} (slope {slope:.3g} against Keaton): its declared unit "
                         f"({declared}) is not supported; do not interpret its MVMR estimates on that scale")
        elif declared == "mmHg" and not (0.8 <= mm <= 1.25):
            notes.append(f"{gid}: per-allele effects at Keaton instruments are {slope:.2f} times Keaton's (correlation "
                         f"{cor:.2f}; {slope_s:.2f} at Keaton p <= 1e-20); the file is labelled mmHg, but its scale or "
                         "effect sizes differ from Keaton's, so its 'per 10 mmHg' MVMR estimates are not directly "
                         "comparable with the Keaton-based ones. Approximate: winner's curse in Keaton (the selection "
                         "GWAS) biases these ratios towards zero")
    unit_check = pd.DataFrame(unit_rows)

    # ---------------- i_mr data-quality diagnostic: variants without EAF in the distance-clumped sets -----------
    diag = pd.DataFrame()
    hs = I_MR_DIR / "harmonised_snps.csv.gz"
    if hs.exists():
        H = pd.read_csv(hs, low_memory=False)
        H = H[H["exposure"].isin(["fg", "hba1c"]) & H["family"].isin(["A_risk_to_beds", "B_risk_to_outcomes_posctrl"])]
        drow = []
        for (fam, e, o), g in H.groupby(["family", "exposure", "outcome"]):
            sc = float(SPEC(e).get("scale", 1.0))
            full = M.mr_ivw(g["beta_x"].values / sc, g["beta_y"].values, g["se_y"].values, "random")
            gg = g[g["eaf_x"].notna()]
            red = M.mr_ivw(gg["beta_x"].values / sc, gg["beta_y"].values, gg["se_y"].values, "random") \
                if len(gg) >= 3 else {"estimate": np.nan, "lower": np.nan, "upper": np.nan, "p": np.nan}
            drow.append({"family": fam, "exposure": e, "outcome": o, "nsnp_distance": len(g),
                         "n_without_eaf": int(g["eaf_x"].isna().sum()),
                         "max_abs_beta_without_eaf": float(g.loc[g["eaf_x"].isna(), "beta_x"].abs().max())
                         if g["eaf_x"].isna().any() else np.nan,
                         "IVW distance, all [95% CI] (recomputed)": fmt_ci(full["estimate"], full["lower"],
                                                                           full["upper"]),
                         "p_all": full["p"],
                         "IVW distance, variants with EAF only [95% CI]": fmt_ci(red["estimate"], red["lower"],
                                                                                 red["upper"]),
                         "p_with_eaf_only": red["p"]})
        diag = pd.DataFrame(drow)
        diag = diag[diag["n_without_eaf"] > 0].reset_index(drop=True)

    # ---------------- registry with SHA-256 ----------------
    used = sorted(set(p1_ids) | set(p2_ids) | ({I.CHAMI["id"]} & avail))
    hashes = {} if args.no_hash else sha_cached([paths[k] for k in used], outdir / "_cache" / "sha256.json",
                                                [I_MR_DIR / "_cache" / "sha256.json"], args.workers)
    msha = I.manifest_sha()
    reg = []
    for k in used:
        s = SPEC(k)
        rel = str(paths[k].relative_to(HUB))
        sha = hashes.get(str(paths[k]))
        reg.append({"id": k, "label": s["label"], "group": s["group"], "source": s["source"], "file": rel,
                    "build": s["build"], "ancestry": s["ancestry"], "N": s["n_default"], "kind": s["kind"],
                    "units": s["units"], "UKB_in_sample": s.get("ukb"),
                    "implied_sd": I.implied_sd(ext[k]) if (k in ext and s["kind"] == "cont") else np.nan,
                    "bytes": paths[k].stat().st_size, "sha256": sha,
                    "manifest_sha256_match": (None if sha is None else msha.get(rel) == sha)})
        rep.inputs.append({"id": k, "path": rel, "bytes": paths[k].stat().st_size, "sha256": sha})
    for f in ld_files:
        if f.exists():
            rep.inputs.append({"id": "ld_ref", "path": str(f.relative_to(HUB)), "bytes": f.stat().st_size,
                               "sha256": ld_sha.get(str(f))})
    registry = pd.DataFrame(reg)

    # ---------------- univariable MR ----------------
    n_boot, n_boot_mode = args.n_boot, min(args.n_boot, 1000)

    def ujob(eid, oid, fam, src=None):
        src = inst if src is None else src
        e, o = SPEC(eid), SPEC(oid)
        return dict(e={k: e[k] for k in ("id", "label", "kind", "unit_label", "units", "ukb") if k in e} |
                    {"scale": e.get("scale", 1.0)},
                    o={k: o[k] for k in ("id", "label", "kind", "unit_label", "units", "ukb") if k in o},
                    inst=I.with_ids(src[eid]), inst10=None, out=ext[oid], family=fam, seed=args.seed,
                    n_boot=n_boot, n_boot_mode=n_boot_mode, drop_pal=not (I.eur(e) and I.eur(o)))
    uj = []
    for eid in I.RISK:
        uj += [("A_risk_to_beds", eid, o) for o in I.BED_OUT]
        uj += [("B_risk_to_outcomes_posctrl", eid, o) for o in I.CLIN_PC]
    for eid in I.RISK_SENS:
        uj += [("A2_osa_alternatives", eid, o) for o in I.BED_OUT + I.CLIN_PC]
    uj += [("C_chami_uncoupled", I.CHAMI["id"], o) for o in I.BED_OUT + I.CLIN_PC]
    beds_ok = [b for b in I.BEDS if b in inst and len(inst[b]) >= 3]
    beds_ok_s = [b for b in I.BEDS if b in inst_s and len(inst_s[b]) >= 3]
    for b in I.BEDS:
        if b in inst and len(inst[b]) < 3:
            notes.append(f"bed {b} has {len(inst[b])} LD-clumped instruments (< 3): not used as an exposure "
                         "(family E and MV_bed_plus_sbp_to_outcomes)")
    uj += [("E_beds_to_outcomes", b, o) for b in beds_ok for o in I.CLIN]
    # extensions
    uj += [("X_B_nonUKB_outcomes", eid, o) for eid in I.RISK for o in NONUKB_OUT]
    uj += [("X_E_beds_to_nonUKB_outcomes", b, o) for b in beds_ok for o in NONUKB_OUT]
    uj += [("X_osa_bmiadj", "osa_campos_bmiadj", o) for o in I.BED_OUT + I.CLIN_PC]
    uj += [("X_height_univariable", "height", o) for o in I.BED_OUT]
    uj = [(f, e, o) for f, e, o in uj if e in inst and o in ext and len(inst[e])]
    log(f"univariable MR: {len(uj)} pairs on {args.mr_workers} workers")
    t_mr = time.time()
    res = I.run_pool(I.mr_pair_worker, [ujob(e, o, f) for f, e, o in uj], args.mr_workers)
    log(f"univariable MR done in {time.time() - t_mr:.1f}s")
    summ = pd.DataFrame([r["summary"] for r in res])
    longd = pd.DataFrame([x for r in res for x in r["long"]])
    snps = pd.concat([r["snps"] for r in res if r.get("snps") is not None], ignore_index=True)
    harm = pd.DataFrame([{"family": r["summary"]["family"], "exposure": r["summary"]["exposure"],
                          "outcome": r["summary"]["outcome"], **r["harm"]} for r in res]).fillna(0)
    for r in res:
        if r.get("trace"):
            notes.append(f"MR error {r['summary']['exposure']}->{r['summary']['outcome']}: {r['summary'].get('error')}")
    summ["ivw_q"] = np.nan
    for fam, g in summ.groupby("family"):
        summ.loc[g.index, "ivw_q"] = I.bh(g["ivw_p"].values)
    sd_map = dict(zip(registry["id"], registry["implied_sd"]))
    dist_reg = pd.read_csv(I_MR_DIR / "gwas_registry.csv")
    for k, v in zip(dist_reg["id"], dist_reg["implied_sd"]):
        if not np.isfinite(sd_map.get(k, np.nan)):
            sd_map[k] = v
    adv = {s["id"]: s.get("adverse", 0) for s in specs_all}
    summ["ivw_adverse_oriented"] = summ["ivw"] * summ["outcome"].map(adv).replace(0, np.nan)
    summ["outcome_implied_sd"] = summ["outcome"].map(sd_map)
    summ["ivw_per_sd_outcome"] = summ["ivw"] / summ["outcome_implied_sd"]
    summ["evidence"] = summ.apply(I.classify, axis=1)
    summ["clumping"] = "LD r2<=0.001 within 10 Mb (1000G EUR)"
    summ["status"] = STATUS

    # ---------------- sensitivity: the main univariable families with instruments clumped at r2 <= r2_sens -------
    summ_s = pd.DataFrame()
    if inst_s:
        ujs = []
        for eid in I.RISK:
            ujs += [("A_risk_to_beds", eid, o) for o in I.BED_OUT]
            ujs += [("B_risk_to_outcomes_posctrl", eid, o) for o in I.CLIN_PC]
        for eid in I.RISK_SENS:
            ujs += [("A2_osa_alternatives", eid, o) for o in I.BED_OUT + I.CLIN_PC]
        ujs += [("C_chami_uncoupled", I.CHAMI["id"], o) for o in I.BED_OUT + I.CLIN_PC]
        ujs += [("E_beds_to_outcomes", b, o) for b in beds_ok_s for o in I.CLIN]
        ujs = [(f, e, o) for f, e, o in ujs if e in inst_s and o in ext and len(inst_s[e])]
        log(f"sensitivity univariable MR (r2 <= {args.r2_sens:g}): {len(ujs)} pairs")
        t_s = time.time()
        res_s = I.run_pool(I.mr_pair_worker, [ujob(e, o, f, inst_s) for f, e, o in ujs], args.mr_workers)
        log(f"sensitivity univariable MR done in {time.time() - t_s:.1f}s")
        summ_s = pd.DataFrame([r["summary"] for r in res_s])
        summ_s["ivw_q"] = np.nan
        for fam, g in summ_s.groupby("family"):
            summ_s.loc[g.index, "ivw_q"] = I.bh(g["ivw_p"].values)
        summ_s["evidence"] = summ_s.apply(I.classify, axis=1)
        summ_s["clumping"] = f"LD r2<={args.r2_sens:g} within 10 Mb (1000G EUR; sensitivity)"
        summ_s["status"] = STATUS

    # ---------------- MVMR (main families + extensions), joint LD clumping, in this process ----------------
    keys = refkeys(inst)
    mj = []
    if "sbp" in ext:
        for x in I.MVMR_X:
            if x in ext and x in inst:
                mj += [((x, "sbp"), o, "MV_risk_plus_sbp_to_beds") for o in I.BED_OUT if o in ext]
        for b in beds_ok:
            if b in ext:
                mj += [((b, "sbp"), o, "MV_bed_plus_sbp_to_outcomes") for o in I.CLIN if o in ext]
    if all(x in ext and x in inst for x in ("whradjbmi", "bmi")):
        mj += [(("whradjbmi", "bmi"), o, "X_mv_whradjbmi_bmi") for o in I.BED_OUT if o in ext]
        if "height" in ext and "height" in inst:
            mj += [(("whradjbmi", "bmi", "height"), o, "X_mv_whradjbmi_bmi_height") for o in I.BED_OUT if o in ext]
    if all(x in ext and x in inst for x in ("dbp", "pp")):
        mj += [(("dbp", "pp"), o, "X_mv_dbp_pp") for o in I.BED_OUT + PC_OUT if o in ext]
    if all(x in ext and x in inst for x in ("map_panukb", "pp")):
        mj += [(("map_panukb", "pp"), o, "X_mv_map_pp") for o in I.BED_OUT + PC_OUT if o in ext]
    if all(x in ext and x in inst for x in ("dbp_gera", "pp_gera")):
        mj += [(("dbp_gera", "pp_gera"), o, "X_mv_gera_dbp_pp") for o in I.BED_OUT + PC_OUT if o in ext]
    if all(x in ext and x in inst for x in ("osa_campos_bmiadj", "bmi")):
        mj += [(("osa_campos_bmiadj", "bmi"), o, "X_mv_osa_bmiadj_bmi") for o in ["cimt", "plaque"] if o in ext]
    log(f"MVMR: {len(mj)} fits (joint LD clumping)")
    t_mv = time.time()
    mrows = []
    for x in sorted({x for xids, _, _ in mj for x in xids}):
        if len(inst[x]) < 3:
            notes.append(f"{x} has {len(inst[x])} LD-clumped instruments at p <= {P_THRESH:g} (< 3): its MVMR fits "
                         "are not run")
    for x in [s_["id"] for s_ in EXT_SPECS if s_["id"] in inst and len(inst[s_["id"]]) < 3]:
        notes.append(f"{x}: {len(inst[x])} variants at p <= {P_THRESH:g} after the filters; no instrument set "
                     "(its univariable and multivariable analyses are not run)")
    for xids, oid, fam in mj:
        short = [x for x in xids if len(inst[x]) < 3]
        if short:
            mrows.append({"family": fam, "exposures": "+".join(xids), "outcome": oid,
                          "error": f"fewer than 3 instruments for {', '.join(short)}"})
            continue
        dp = not (all(I.eur(SPEC(x)) for x in xids) and I.eur(SPEC(oid)))
        try:
            mrows += mvmr_ld(list(xids), oid, fam, ext, inst, keys, ldref, dp)
        except Exception as e:                                        # pragma: no cover - reported in the table
            mrows.append({"family": fam, "exposures": "+".join(xids), "outcome": oid,
                          "error": f"{type(e).__name__}: {e}"})
    mv = pd.DataFrame(mrows)
    if "p" in mv.columns:
        mv["q_within_family_term"] = np.nan
        for (fam, term), g in mv.dropna(subset=["p"]).groupby(["family", "term"]):
            mv.loc[g.index, "q_within_family_term"] = I.bh(g["p"].values)
    mv["status"] = STATUS
    log(f"MVMR done in {time.time() - t_mv:.1f}s")
    # sensitivity: the two main MVMR families with instruments (and joint clumping) at r2 <= r2_sens
    mv_s = pd.DataFrame()
    if inst_s and "sbp" in ext and "sbp" in inst_s:
        keys_s = refkeys(inst_s)
        mjs = [((x, "sbp"), o, "MV_risk_plus_sbp_to_beds") for x in I.MVMR_X if x in ext and x in inst_s
               for o in I.BED_OUT if o in ext]
        mjs += [((b, "sbp"), o, "MV_bed_plus_sbp_to_outcomes") for b in beds_ok_s if b in ext
                for o in I.CLIN if o in ext]
        mrows_s = []
        for xids, oid, fam in mjs:
            short = [x for x in xids if len(inst_s[x]) < 3]
            if short:
                mrows_s.append({"family": fam, "exposures": "+".join(xids), "outcome": oid,
                                "error": f"fewer than 3 instruments for {', '.join(short)}"})
                continue
            dp = not (all(I.eur(SPEC(x)) for x in xids) and I.eur(SPEC(oid)))
            try:
                mrows_s += mvmr_ld(list(xids), oid, fam, ext, inst_s, keys_s, ldref, dp, r2_thresh=args.r2_sens)
            except Exception as e:                                    # pragma: no cover - reported in the table
                mrows_s.append({"family": fam, "exposures": "+".join(xids), "outcome": oid,
                                "error": f"{type(e).__name__}: {e}"})
        mv_s = pd.DataFrame(mrows_s)
        log(f"sensitivity MVMR (r2 <= {args.r2_sens:g}): {len(mjs)} fits")

    # ---------------- comparison with the distance-clumped panel ----------------
    dist = pd.read_csv(I_MR_DIR / "mr_all_pairs_summary.csv")
    dist = dist[dist["family"].isin(MAIN_FAMS)]
    ld_main = summ[summ["family"].isin(MAIN_FAMS)]
    kcols = ["family", "exposure", "outcome"]
    vcols = ["nsnp", "ivw", "ivw_lci", "ivw_uci", "ivw_p", "ivw_q", "egger", "egger_intercept_p", "wmedian",
             "wmedian_p", "wmode", "wmode_p", "Q_p", "I2", "F_mean", "consistent_direction", "evidence"]
    cmp = dist[kcols + vcols].merge(ld_main[kcols + vcols], on=kcols, how="outer", suffixes=("_dist", "_ld"))
    cmp["class_distance"] = cmp["evidence_dist"].map(coarse)
    cmp["class_ld"] = cmp["evidence_ld"].map(coarse)
    cmp["class_changed"] = cmp["class_distance"] != cmp["class_ld"]
    cmp["fdr_support_change"] = np.select(
        [cmp["class_distance"].isin(SUPPORTED) & ~cmp["class_ld"].isin(SUPPORTED),
         ~cmp["class_distance"].isin(SUPPORTED) & cmp["class_ld"].isin(SUPPORTED)],
        ["lost (robust/supported -> other)", "gained (other -> robust/supported)"], "")
    cmp["sign_flip"] = np.sign(cmp["ivw_dist"]) * np.sign(cmp["ivw_ld"]) < 0
    cmp["ratio_ld_to_distance"] = cmp["ivw_ld"] / cmp["ivw_dist"]
    if len(diag):
        cmp = cmp.merge(diag[["family", "exposure", "outcome", "n_without_eaf"]].rename(
            columns={"n_without_eaf": "imr_variants_without_eaf"}), on=kcols, how="left")
    else:
        cmp["imr_variants_without_eaf"] = np.nan
    cmp["imr_variants_without_eaf"] = cmp["imr_variants_without_eaf"].fillna(0).astype(int)
    cmp["IVW distance [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(cmp["ivw_dist"], cmp["ivw_lci_dist"],
                                                                         cmp["ivw_uci_dist"])]
    cmp["IVW LD [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(cmp["ivw_ld"], cmp["ivw_lci_ld"], cmp["ivw_uci_ld"])]
    # what the 'distance' side of the comparison is (family C: i_mr used the 266 published leads without clumping)
    cmp["i_mr_instrument_set"] = cmp["family"].map(IMR_SET)
    # sensitivity at r2 <= r2_sens: does a changed conclusion recover with the looser threshold?
    if len(summ_s):
        s_ = summ_s[summ_s["family"].isin(MAIN_FAMS)][kcols + ["nsnp", "ivw", "ivw_lci", "ivw_uci", "ivw_p", "ivw_q",
                                                                "evidence"]]
        s_ = s_.rename(columns={c: f"{c}_r2sens" for c in s_.columns if c not in kcols})
        cmp = cmp.merge(s_, on=kcols, how="left")
        cmp["class_ld_r2sens"] = cmp["evidence_r2sens"].map(coarse)
        cmp["IVW LD r2sens [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in
                                         zip(cmp["ivw_r2sens"], cmp["ivw_lci_r2sens"], cmp["ivw_uci_r2sens"])]
        cmp["r2sens_reading"] = np.select(
            [~cmp["class_changed"], cmp["class_ld_r2sens"] == cmp["class_distance"],
             cmp["class_ld_r2sens"] == cmp["class_ld"]],
            ["", "recovers at the looser threshold (class as with distance clumping)",
             "persists at the looser threshold (class as at r2 0.001)"], "differs from both")
    cmp["family"] = pd.Categorical(cmp["family"], MAIN_FAMS, ordered=True)
    cmp = cmp.sort_values(kcols).reset_index(drop=True)
    cmp["family"] = cmp["family"].astype(str)
    trans = transition_tables(cmp)
    counts = class_counts(cmp, args.r2_sens)
    changed = cmp[cmp["class_changed"]].copy()
    show_ch = ["family", "exposure", "outcome", "class_distance", "class_ld", "fdr_support_change",
               "IVW distance [95% CI]", "IVW LD [95% CI]", "ivw_p_dist", "ivw_p_ld", "ivw_q_dist", "ivw_q_ld",
               "nsnp_dist", "nsnp_ld", "sign_flip", "imr_variants_without_eaf", "i_mr_instrument_set"]
    if "class_ld_r2sens" in cmp.columns:
        show_ch += ["class_ld_r2sens", "IVW LD r2sens [95% CI]", "ivw_p_r2sens", "ivw_q_r2sens", "nsnp_r2sens",
                    "r2sens_reading"]
    keyp = pd.DataFrame(KEY_PAIRS, columns=kcols).merge(cmp, on=kcols, how="left")

    # MVMR comparison (main MVMR families)
    dmv = pd.read_csv(I_MR_DIR / "mvmr.csv")
    mkey = ["family", "exposures", "outcome", "term"]
    mvc = pd.DataFrame()
    if "term" in mv.columns:
        mvc = dmv[mkey + ["nsnp", "estimate", "lower", "upper", "p", "conditional_F"]].merge(
            mv[mv["family"].isin(dmv["family"].unique())][mkey + ["nsnp", "estimate", "lower", "upper", "p",
                                                                   "conditional_F"]],
            on=mkey, how="outer", suffixes=("_dist", "_ld"))
        mvc["p05_dist"] = mvc["p_dist"] < 0.05
        mvc["p05_ld"] = mvc["p_ld"] < 0.05
        mvc["p05_changed"] = mvc["p05_dist"] != mvc["p05_ld"]
        mvc["sign_flip"] = np.sign(mvc["estimate_dist"]) * np.sign(mvc["estimate_ld"]) < 0
        mvc["MVMR distance [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(mvc["estimate_dist"], mvc["lower_dist"],
                                                                             mvc["upper_dist"])]
        mvc["MVMR LD [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(mvc["estimate_ld"], mvc["lower_ld"],
                                                                       mvc["upper_ld"])]
        if "term" in mv_s.columns:
            ms = mv_s[mkey + ["nsnp", "estimate", "lower", "upper", "p", "conditional_F"]].rename(
                columns={c: f"{c}_r2sens" for c in ["nsnp", "estimate", "lower", "upper", "p", "conditional_F"]})
            mvc = mvc.merge(ms, on=mkey, how="left")
            mvc["p05_r2sens"] = mvc["p_r2sens"] < 0.05
            mvc["MVMR LD r2sens [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in
                                              zip(mvc["estimate_r2sens"], mvc["lower_r2sens"], mvc["upper_r2sens"])]

    # Extensions: comparisons with i2 (distance-clumped) and with UKB-including outcomes
    ext_tabs = {}
    if "term" in mv.columns:
        for fam_ld, f_i2 in (("X_mv_whradjbmi_bmi", "mv_whradjbmi_bmi.csv"), ("X_mv_dbp_pp", "mv_dbp_pp.csv"),
                             ("X_mv_osa_bmiadj_bmi", None)):
            g = mv[mv["family"] == fam_ld].copy()
            if g.empty:
                continue
            g["MVMR LD [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(g["estimate"], g["lower"], g["upper"])]
            if f_i2 and (I2_DIR / f_i2).exists():
                d2 = pd.read_csv(I2_DIR / f_i2)
                d2 = d2[["outcome", "term", "nsnp", "estimate", "lower", "upper", "p"]].rename(
                    columns={"nsnp": "nsnp_i2_dist", "estimate": "est_i2_dist", "p": "p_i2_dist"})
                d2["MVMR i2 distance [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in
                                                   zip(d2["est_i2_dist"], d2["lower"], d2["upper"])]
                g = g.merge(d2.drop(columns=["lower", "upper"]), on=["outcome", "term"], how="left")
            ext_tabs[fam_ld] = g
        for fam_ld in ("X_mv_whradjbmi_bmi_height", "X_mv_gera_dbp_pp", "X_mv_map_pp"):
            g = mv[mv["family"] == fam_ld].copy()
            if not g.empty:
                g["MVMR LD [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(g["estimate"], g["lower"], g["upper"])]
                ext_tabs[fam_ld] = g
    nonukb = pd.DataFrame()
    xb = summ[summ["family"].isin(["X_B_nonUKB_outcomes", "X_E_beds_to_nonUKB_outcomes"])].copy()
    if len(xb):
        xb["outcome_ukb_version"] = xb["outcome"].map(NONUKB_PAIR)
        xb["family_ukb_version"] = np.where(xb["family"] == "X_B_nonUKB_outcomes", "B_risk_to_outcomes_posctrl",
                                            "E_beds_to_outcomes")
        u = summ[["family", "exposure", "outcome", "nsnp", "ivw", "ivw_lci", "ivw_uci", "ivw_p", "evidence"]].rename(
            columns={"family": "family_ukb_version", "outcome": "outcome_ukb_version", "nsnp": "nsnp_ukb",
                     "ivw": "ivw_ukb", "ivw_lci": "lci_ukb", "ivw_uci": "uci_ukb", "ivw_p": "p_ukb",
                     "evidence": "evidence_ukb_outcome"})
        nonukb = xb.merge(u, on=["family_ukb_version", "exposure", "outcome_ukb_version"], how="left")
        nonukb["IVW non-UKB outcome [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in
                                                  zip(nonukb["ivw"], nonukb["ivw_lci"], nonukb["ivw_uci"])]
        nonukb["IVW UKB-including outcome [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in
                                                        zip(nonukb["ivw_ukb"], nonukb["lci_ukb"], nonukb["uci_ukb"])]

    # ---------------- report ----------------
    write_report(rep, outdir, st, ld_tab, ldref, ldstats, xw_rows, registry, eafqc, summ, cmp, trans, counts,
                 changed[show_ch], keyp, mvc, mv, ext_tabs, nonukb, notes, scan_stats, args, t_start, unit_check, diag,
                 noeaf_disp=noeaf_disp, ldstats_s=ldstats_s, summ_s=summ_s)
    rep.tables["mr_all_pairs_summary"] = summ
    rep.tables["mr_all_methods_long"] = longd
    rep.tables["harmonisation_actions"] = harm
    rep.tables["mvmr"] = mv
    if len(summ_s):
        rep.tables["sens_r2_mr_all_pairs_summary"] = summ_s
    if len(mv_s):
        rep.tables["sens_r2_mvmr"] = mv_s
    if inst_s:
        inst_tab_s = pd.concat([d.assign(exposure=k, F=(d["beta"] / d["se"]) ** 2) for k, d in inst_s.items()],
                               ignore_index=True)
        rep.tables["sens_r2_instruments"] = inst_tab_s[["exposure", "snp", "chr", "pos", "ea", "oa", "beta", "se",
                                                        "p", "eaf", "n", "F", "ref_snp", "ref_pos", "how"]]
    rep.tables["comparison_pairs"] = cmp
    rep.tables["ld_match_by_candidate"] = allc[["_exp", "snp", "chr", "pos", "ea", "oa", "p", "eaf", "ref_idx",
                                                "ref_snp", "ref_chr", "ref_pos", "how", "reason", "maf_panel",
                                                "rare_in_panel_no_eaf", "no_eaf_row_excluded"]].rename(
        columns={"_exp": "exposure"})
    inst_tab = pd.concat([d.assign(exposure=k, F=(d["beta"] / d["se"]) ** 2) for k, d in inst.items()],
                         ignore_index=True)
    rep.tables["instruments_ld"] = inst_tab[["exposure", "snp", "chr", "pos", "pos38", "ea", "oa", "beta", "se", "p",
                                             "eaf", "n", "F", "ref_snp", "ref_pos", "how"]]
    snps.to_csv(outdir / "harmonised_snps.csv.gz", index=False, compression="gzip")
    rep.write(outdir, __file__)
    # the per-candidate match table is large: keep it compressed
    big = outdir / "ld_match_by_candidate.csv"
    if big.exists():
        pd.read_csv(big, low_memory=False).to_csv(outdir / "ld_match_by_candidate.csv.gz", index=False,
                                                  compression="gzip")
        big.unlink()
    print(f"[{TASK}] real: wrote {outdir}")


def write_report(rep, outdir, st, ld_tab, ldref, ldstats, xw_rows, registry, eafqc, summ, cmp, trans, counts,
                 changed, keyp, mvc, mv, ext_tabs, nonukb, notes, scan_stats, args, t_start, unit_check=None,
                 diag=None, noeaf_disp=None, ldstats_s=None, summ_s=None):
    n_main = len(cmp)
    has_s = "class_ld_r2sens" in cmp.columns
    rs = args.r2_sens
    n_ch = int(cmp["class_changed"].sum())
    lost = int((cmp["fdr_support_change"] == "lost (robust/supported -> other)").sum())
    gained = int((cmp["fdr_support_change"] == "gained (other -> robust/supported)").sum())
    allc = counts[counts["family"] == "all main families"].set_index("clumping")
    rep.h("Summary (auto-generated; exploratory)")
    rep.bullets([
        f"Main panel rerun with LD clumping (r2 <= {R2_THRESH} within {LD_WINDOW_KB / 1000:g} Mb; 1000 Genomes "
        f"phase 3 EUR, {ldref.n} samples, {ldref.n_snps:,} variants): {n_main} univariable pairs in families "
        f"{', '.join(MAIN_FAMS)} (family D, East Asian exposures, not rerun).",
        f"Evidence class changed for {n_ch} of {n_main} pairs; {lost} lost and {gained} gained FDR support "
        "(robust or supported).",
        "Class counts, distance -> LD: " + "; ".join(
            f"{c} {int(allc.loc['distance (i_mr)', c])} -> {int(allc.loc['LD (this run)', c])}"
            for c in CLASS_ORDER if (allc.loc['distance (i_mr)', c] + allc.loc['LD (this run)', c]) > 0) + ".",
        "Every changed pair is listed below and in `comparison_pairs.csv` (`class_changed`).",
        "Family C (Chami uncoupled adiposity): i_mr used all 266 published lead variants without clumping, so for "
        "this family the comparison is 'unclumped published leads -> LD-clumped', not 'distance -> LD'.",
    ] + ([
        f"Sensitivity at r2 <= {rs:g} (same window and rules): of the {n_ch} changed pairs, "
        + "; ".join(f"{k}: {int(v)}" for k, v in cmp.loc[cmp["class_changed"], "r2sens_reading"].value_counts()
                    .items()) + ". Class counts at r2 <= " + f"{rs:g}: " + "; ".join(
            f"{c} {int(allc.loc[f'LD r2 <= {rs:g} (sensitivity)', c])}" for c in CLASS_ORDER
            if allc.loc[f'LD r2 <= {rs:g} (sensitivity)', c] > 0) + "."] if has_s else []) + [
        "Status: exploratory; not paper numbers until checked by the paper session.",
    ])
    hw = outdir / "results_summary_handwritten.md"
    if hw.exists():
        rep.h("Results summary (hand-written after inspection of this run)")
        rep.p(hw.read_text(encoding="utf-8").strip())
    rep.h("Methods (as run)")
    rep.bullets([
        "Everything as in `analysis_i_mr.py` (imported, unchanged) except instrument clumping; see the module "
        "docstring. Instruments: p <= 5e-8, autosomes, MHC excluded, MAF >= 0.01, valid alleles, finite beta/SE.",
        f"LD clumping: greedy, index = smallest p (ties largest |z|); remove remaining candidates on the same "
        f"chromosome within +/-{LD_WINDOW_KB / 1000:g} Mb with r2 > {R2_THRESH} (squared Pearson correlation of "
        "allele counts in the reference samples). Candidates not found in the panel are dropped. Chami 2025 lead "
        "variants are clumped with p threshold 1.",
        "Panel matching order: rsID (chromosome and allele pair must agree; strand complements accepted), MAGMA "
        "dbSNP-151 synonyms, GRCh37 chr:pos + alleles.",
        "MVMR: union of each exposure's LD-clumped instruments, jointly LD-clumped on the smallest p across exposures.",
        "EAF rules: as in analysis_i_mr, MAF >= 0.01 where the GWAS reports EAF. Rows without EAF in a file that "
        "reports EAF for its other rows (only the MAGIC fasting-glucose and HbA1c files) are excluded before "
        "clumping (review fix of 2026-10-05; see the data-quality section). In a file without any EAF the variant "
        "would need MAF >= 0.01 in the panel (no such file in this run).",
        f"Sensitivity: the main univariable families and the two main MVMR families rerun with r2 <= {rs:g} "
        "(same +/-10 Mb window and rules). With 503 samples, r2 between independent SNPs exceeds 0.01 with "
        "probability about 0.025, against about 0.48 for 0.001, so the looser threshold separates loss of "
        "independent instruments from removal of correlated ones." if has_s else
        "Sensitivity threshold not run.",
        "Retinal GWAS 4a-4e are read with the effect/other allele labels swapped back (analysis_i_mr 'swap_alleles').",
        f"Estimators and evidence classes as in analysis_i_mr (IVW multiplicative random effects primary; FDR within "
        f"family). Bootstrap: weighted median {args.n_boot}, weighted mode {min(args.n_boot, 1000)} replicates; seed "
        f"{args.seed}.",
        "Caveat: with 503 reference samples r2 between independent SNPs exceeds 0.001 with probability about 0.48, "
        "so the rule also prunes independent signals within 10 Mb (conservative; field default).",
    ])
    rep.h("Self-test of the new code (synthetic panel, planted answers)")
    rep.table("selftest", st, None, max_rows=None)
    rep.h("LD reference")
    rep.p(f"MAGMA 1000 Genomes phase 3 EUR reference (CTG lab), {LDREF_URL}; unzipped next to the zip. "
          "See `external/ld_ref/1000g_eur/README.md`.")
    rep.table("ld_reference_files", ld_tab, None, max_rows=None)
    rep.h("Instruments: panel matching and LD clumping per exposure")
    rep.p("`candidates` = rows eligible for clumping (after the analysis_i_mr filters; Chami: all 266 lead variants). "
          "`instruments_distance_1mb` and `in_both` compare with `i_mr/instruments_1mb.csv`.")
    rep.table("ld_instruments_by_exposure", ldstats, None, max_rows=None)
    if xw_rows:
        rep.table("grch38_crosswalk_candidates", pd.DataFrame(xw_rows), "GRCh38 cross-walk of all candidates "
                  "(Kurniansyah OSA, via MVP sleep apnoea)", max_rows=None)
    rep.h("Class-by-class comparison with the distance-clumped panel (i_mr, random effects)")
    ch = ldstats[ldstats["exposure"] == I.CHAMI["id"]]
    if len(ch):
        ch = ch.iloc[0]
        rep.p(f"For families A, A2, B and E the i_mr side is +/-1 Mb distance clumping. For family C (Chami uncoupled "
              f"adiposity) i_mr used all {int(ch['candidates'])} published lead variants without any clumping; this "
              f"run keeps {int(ch['candidates'] - ch['unmatched_not_in_panel'] - ch['unmatched_allele_mismatch'] - ch['unmatched_chromosome_mismatch'] - ch['unmatched_no_id'])} "
              f"that are in the panel and LD-clumps them (p threshold 1) to {int(ch['instruments_ld'])}. Family C "
              "changes therefore compare unclumped published leads with LD-clumped leads (column "
              "`i_mr_instrument_set`), and the r2 <= 0.001 rule with 503 samples also removes many independent leads.")
    rep.table("class_counts", counts, "Evidence-class counts by family and clumping", max_rows=None)
    rep.table("class_transitions", trans, "Transitions: rows = class with distance clumping, columns = class with "
              "LD clumping", max_rows=None)
    rep.h("Every pair whose evidence class changed")
    rep.table("changed_conclusions", changed, None, max_rows=None)
    rep.h("Pairs cited in the Results draft or the novelty audit (C15)")
    kp = keyp[["family", "exposure", "outcome", "class_distance", "class_ld", "IVW distance [95% CI]",
               "IVW LD [95% CI]", "ivw_p_dist", "ivw_p_ld", "nsnp_dist", "nsnp_ld"]]
    rep.table("key_pairs", kp, None, max_rows=None)
    if len(mvc):
        rep.h("MVMR terms: distance (i_mr) vs LD clumping")
        chg = mvc[mvc["p05_changed"] | mvc["sign_flip"]]
        rep.p(f"{len(mvc)} MVMR terms; p < 0.05 status changed for {int(mvc['p05_changed'].sum())}, sign flipped for "
              f"{int(mvc['sign_flip'].sum())}. All terms in `mvmr_comparison.csv`.")
        rep.table("mvmr_comparison", mvc, None, max_rows=0)
        mcols = ["family", "exposures", "outcome", "term", "MVMR distance [95% CI]", "MVMR LD [95% CI]", "p_dist",
                 "p_ld", "nsnp_dist", "nsnp_ld"]
        if "MVMR LD r2sens [95% CI]" in chg.columns:
            mcols += ["MVMR LD r2sens [95% CI]", "p_r2sens", "nsnp_r2sens"]
        rep.parts[-2] = C.md_table(chg[mcols])
    if has_s:
        rep.h(f"Sensitivity: LD clumping at r2 <= {rs:g} (same window and rules)")
        rep.p(f"Purpose: with 503 reference samples the r2 <= 0.001 rule prunes about half of truly independent "
              f"pairs within 10 Mb; at r2 <= {rs:g} about 2.5 % are. A changed conclusion that recovers at r2 <= "
              f"{rs:g} (class as with distance clumping) is attributable to the loss of instruments; one that persists "
              "is not explained by pruning alone. Columns with suffix `r2sens` in `comparison_pairs.csv`, "
              "`changed_conclusions.csv` and `mvmr_comparison.csv`; full results in `sens_r2_mr_all_pairs_summary.csv`, "
              "`sens_r2_mvmr.csv` and `sens_r2_instruments.csv`.")
        if ldstats_s is not None and len(ldstats_s):
            ic = ldstats[["exposure", "instruments_distance_1mb", "instruments_ld"]].merge(
                ldstats_s[["exposure", "instruments_ld"]].rename(columns={"instruments_ld": f"instruments_r2_{rs:g}"}),
                on="exposure", how="inner").rename(columns={"instruments_ld": "instruments_r2_0.001"})
            rep.table("sens_r2_instrument_counts", ic, "Instrument counts by clumping rule", max_rows=None)
        sc = cmp[cmp["class_changed"]][["family", "exposure", "outcome", "class_distance", "class_ld",
                                        "class_ld_r2sens", "IVW distance [95% CI]", "IVW LD [95% CI]",
                                        "IVW LD r2sens [95% CI]", "ivw_p_ld", "ivw_p_r2sens", "nsnp_dist", "nsnp_ld",
                                        "nsnp_r2sens", "r2sens_reading"]]
        rep.table("sens_r2_changed_pairs", sc, "Every pair whose class changed between distance and r2 <= 0.001, "
                  f"with its class at r2 <= {rs:g}", max_rows=None)
    if diag is not None and len(diag):
        rep.h("Data-quality finding in the distance-clumped panel (i_mr): FG and HbA1c variants without EAF")
        rep.p("The MAGIC FG and HbA1c files report no EAF for some rows, so analysis_i_mr could not apply its MAF "
              "filter to them. These rows come from a small subset of the contributing samples (N 283 to 17,302 "
              "against about 150,000-200,000 for complete rows), and many have implausibly large effects (several "
              "mmol/L per allele) with a reported p that beta/SE does not support (for example chr10:15468802, beta "
              "-3.05, SE 0.707, N 1,820: z about 4.3, reported p 5.2e-19). In the distance-clumped FG set they "
              "dominate the IVW weights. The table recomputes the i_mr distance-clumped IVW (random effects) from "
              "`i_mr/harmonised_snps.csv.gz` with and without these variants, which separates their effect from the "
              "effect of LD clumping.")
        rep.table("diag_imr_variants_without_eaf", diag, None, max_rows=None)
    if noeaf_disp is not None and len(noeaf_disp):
        d0 = noeaf_disp[noeaf_disp["r2_thresh"] == R2_THRESH]
        fate = d0["fate_under_previous_rule"].str.replace(r" \(.*", "", regex=True)
        cnt = d0.assign(fate=fate).groupby(["exposure", "fate"]).size().reset_index(name="n")
        txt = "; ".join(f"{e}: " + ", ".join(f"{int(r['n'])} {r['fate']}" for _, r in g.iterrows())
                        for e, g in cnt.groupby("exposure"))
        same = ldstats.loc[ldstats["instruments_same_without_no_eaf_rule"].notna(),
                           ["exposure", "instruments_same_without_no_eaf_rule"]]
        elig = d0[fate == "clumped away by a stronger neighbour"]
        rest = d0[fate != "clumped away by a stronger neighbour"]
        rep.p("How this run handles them (review fix, 2026-10-05). Rows without EAF in these files are now excluded "
              "before clumping. Before the fix they were removed only by the clumping order. Fate of the excluded "
              f"candidates at p <= 5e-8 under the previous rule (panel MAF only): {txt}. The implausible rows (|beta| "
              f"up to {rest['beta'].abs().max():.3g}, N {rest['n'].min():,.0f} to {rest['n'].max():,.0f}) were all "
              f"absent from the panel or rare in it; the {len(elig)} rows that were eligible had plausible effects "
              f"(|beta| <= {elig['beta'].abs().max():.2g}, N {elig['n'].min():,.0f} to {elig['n'].max():,.0f}) and "
              "were clumped away by complete-row neighbours at the same loci. None of them was an index "
              "variant, so the instrument sets are identical with and without the exclusion ("
              + ", ".join(f"{r['exposure']}: {'identical' if r['instruments_same_without_no_eaf_rule'] else 'DIFFERENT'}"
                          for _, r in same.iterrows())
              + "). p-consistency (reported p against beta/SE) is not used as a filter: complete rows of these MAGIC "
              "files also differ (5th-95th percentile of the log10 difference about -4 to +3.5).")
        rep.table("noeaf_candidates_disposition", noeaf_disp, None, max_rows=0)
        rep.parts[-2] = "_(every excluded no-EAF candidate, with reported and beta/SE-implied p and its fate under the previous rule: `noeaf_candidates_disposition.csv`)_"
    rep.h("Extensions on the open GWAS downloaded on 2026-10-05 (first look; exploratory)")
    if unit_check is not None and len(unit_check):
        rep.table("unit_check_gera_map", unit_check, "Unit check: effects of GERA DBP/PP and Pan-UKB MAP regressed "
                  "on Keaton effects at Keaton-selected instruments (slope near 1 = mmHg, as labelled). Approximate: "
                  "the instruments are selected in Keaton, so winner's curse inflates the Keaton effects and biases "
                  "the slopes towards zero; Keaton and Pan-UKB also share UK Biobank samples. The columns restricted "
                  "to Keaton p <= 1e-20 are less affected by winner's curse", max_rows=None)
    if len(nonukb):
        rep.table("ext_nonukb_outcomes", nonukb[["family", "exposure", "outcome", "nsnp",
                                                 "IVW non-UKB outcome [95% CI]", "ivw_p", "evidence",
                                                 "outcome_ukb_version", "IVW UKB-including outcome [95% CI]", "p_ukb",
                                                 "evidence_ukb_outcome"]],
                  "Outcomes without UK Biobank (CARDIoGRAMplusC4D 2015 CAD; MEGASTROKE 2018 EUR any and ischaemic "
                  "stroke; MVP EUR heart failure, phecode 428) next to the UKB-including outcomes of families B and E (both LD-clumped; FDR within the "
                  "X_ family)", max_rows=None)
    for fam, g in ext_tabs.items():
        cols = [c for c in ["exposures", "outcome", "term", "nsnp", "MVMR LD [95% CI]", "p", "conditional_F",
                            "q_within_family_term", "nsnp_i2_dist", "MVMR i2 distance [95% CI]", "p_i2_dist",
                            "error"] if c in g.columns]
        rep.table(f"ext_{fam}", g[cols], f"{fam}", max_rows=None)
    xo = summ[summ["family"].isin(["X_osa_bmiadj", "X_height_univariable"])]
    if len(xo):
        xo = xo.assign(**{"IVW [95% CI]": [fmt_ci(a, b, c) for a, b, c in zip(xo["ivw"], xo["ivw_lci"],
                                                                               xo["ivw_uci"])]})
        rep.table("ext_univariable", xo[["family", "exposure", "outcome", "nsnp", "IVW [95% CI]", "ivw_p", "ivw_q",
                                         "evidence", "F_mean"]], "Univariable extensions", max_rows=None)
    if len(eafqc):
        rep.table("eaf_concordance_new_files", eafqc, "Allele-label QC of the new files: EAF concordance with Keaton "
                  "SBP after alignment (non-palindromic)", max_rows=None)
    rv = outdir / "public_data_requests_review.csv"
    if rv.exists():
        rep.h("Review of the F1 public-data requests (2026-10-05)")
        rep.p("Hand-curated table (`public_data_requests_review.csv`): what was requested, whether it is openly "
              "downloadable, where, how large, which F1 MR item it serves, and what was done.")
        rv_df = pd.read_csv(rv)
        if "bytes" in rv_df.columns:
            rv_df["bytes"] = [f"{int(x):,}" if pd.notna(x) else "" for x in rv_df["bytes"]]
        rep.p(C.md_table(rv_df))
    rep.h("GWAS registry")
    rep.table("gwas_registry", registry, None, max_rows=None)
    rep.parts[-2] = C.md_table(registry[["id", "label", "build", "ancestry", "N", "UKB_in_sample", "implied_sd",
                                         "manifest_sha256_match"]])
    if notes:
        rep.h("Notes")
        rep.bullets(notes)
    rep.h("Run")
    rep.table("scan_stats", pd.DataFrame(scan_stats), None, max_rows=0)
    rep.parts[-2] = "_(per-file scan statistics in `scan_stats.csv`)_"
    rep.p(f"Run time {time.time() - t_start:.0f} s; workers {args.workers} (scans), {args.mr_workers} (MR).")


def main():
    def extra(ap):
        ap.add_argument("--workers", type=int, default=6, help="parallel file scans")
        ap.add_argument("--mr-workers", type=int, default=10, help="parallel MR fits")
        ap.add_argument("--no-cache", action="store_true", help="ignore cached per-file extracts")
        ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 of the input files")
        ap.add_argument("--selftest", action="store_true", help="planted-answer checks only (synthetic panel)")
        ap.add_argument("--r2-sens", type=float, default=R2_SENS,
                        help="sensitivity r2 threshold for the main families (0 = skip)")
    args = C.parse_args(__doc__.splitlines()[0], extra)
    if args.selftest:
        outdir = C.ANALYSIS / TASK
        outdir.mkdir(parents=True, exist_ok=True)
        t = selftest(args.seed, outdir)
        print(C.md_table(t))
        print(f"selftest: {int(t['pass'].sum())}/{len(t)} pass")
        sys.exit(0 if t["pass"].all() else 1)
    if args.simulate:
        raise SystemExit("--simulate is not implemented for this driver; use --selftest")
    mode, outdir, token = C.gate(args, TASK)
    if mode != "real" or token is None:
        raise SystemExit("real mode needs the gate token")
    log(f"mode {mode}; output {outdir}")
    run(args, outdir)
    log("finished")


if __name__ == "__main__":
    main()
