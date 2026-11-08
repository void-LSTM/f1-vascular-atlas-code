#!/usr/bin/env python3
"""analysis_i_mr.py - F1 Stage 0 item 4: two-sample Mendelian randomisation panel on open GWAS summary statistics.

Paper F1 (multi-bed vascular atlas), PLAN §3 Stage 0 item 4; Fig 5 (genetic atlas) and Fig 6 (outcomes).

Questions
---------
A  Do genetically proxied upstream risk factors (SBP, BMI, WHRadjBMI, TG, OSA, fasting glucose, HbA1c, T2D) affect
   each vascular bed: arterial stiffness index (ASI), carotid IMT, carotid plaque, retinal CRAE, CRVE, AVR,
   arteriolar tortuosity, arterial vascular density, and PAD?
B  Are the effects independent of SBP (multivariable MR, MVMR, risk factor + SBP)?
C  Do adiposity variants uncoupled from cardiometabolic risk (Chami 2025, 266 variants) affect the beds?
D  Do BioBank Japan / East Asian instruments (SBP, BMI, TG) replicate the European estimates?
E  Do genetically proxied beds (exposures with >= 3 instruments) affect CAD, stroke, ischaemic stroke, HF, PAD and
   parental lifespan (with MVMR bed + SBP)?
Positive controls: risk factors -> CAD, stroke, HF, parental lifespan.

Method
------
- Streaming: every GWAS file is read with the pyarrow CSV streaming reader (pyarrow 15; 32 MB blocks; only the
  needed columns), filtered block by block, so memory stays at a few hundred MB per worker. Pass 1 keeps rows with
  p <= 5e-8 in exposure files; pass 2 keeps the union of all instrument variants (by rsID and GRCh37 chr:pos) in
  every outcome / MVMR file. Per-file extracts are cached as parquet in `<outdir>/_cache/`.
- Builds: GRCh37 throughout. GRCh38 files (MVP sleep apnoea and PAD, Kurniansyah OSA) are matched by rsID; their
  instruments get GRCh37 positions from rsID look-ups in two GRCh37 reference GWAS (Keaton SBP, Ortin Vela CRAE). The
  Kurniansyah files have no rsID: their instruments are cross-walked to rsIDs through the MVP sleep-apnoea file
  (same GRCh38 coordinates, allele-checked). GRCh37 files without rsID get rsIDs the same way (chr:pos + alleles).
- Instruments: p <= 5e-8; autosomes; MHC (chr6 25-34 Mb) excluded; MAF >= 0.01 where EAF is given; valid alleles;
  finite beta/SE. Greedy distance clumping (no LD reference available): index variant = smallest p (ties: largest
  |z|), all other candidates within +/- 1 Mb removed. Sensitivity: +/- 10 Mb.
- Harmonisation (lib_mr.harmonise): rsID first, GRCh37 chr:pos for the rest; direct / swapped / strand-flipped;
  palindromic SNPs kept only when both EAFs are outside [0.42, 0.58] (strand inferred from EAF).
- Estimators (lib_mr): IVW multiplicative random effects (primary), IVW fixed, MR-Egger (slope, intercept, I2_GX),
  weighted median, weighted mode, Cochran's Q / I2; mean F; Steiger directionality and per-SNP Steiger filtering
  (sensitivity); MVMR-IVW with conditional F (exposure covariance assumed 0).
- Conclusions require directional consistency of IVW, Egger, weighted median and weighted mode.

Modes
-----
--simulate      synthetic GWAS files written in the exact formats of the real files (headers read from the real
                files; nothing else is read), with planted causal effects, GRCh38 files, allele swaps, strand flips,
                palindromic SNPs and LD-like neighbours; the whole pipeline runs on them and the planted effects are
                checked (within 3 SE). Output: results/stage0/validation/drivers/i_mr/.
--real real GWAS. Output: results/stage0/analysis/i_mr/.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import resource
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import lib_mr as M
import s0an_common as C

TASK = "i_mr"
HUB = C.PAPER
GWAS = HUB / "external" / "gwas"
MANIFEST = HUB / "external" / "DOWNLOAD_MANIFEST.tsv"
P_THRESH = 5e-8
WINDOW_KB = 1000.0           # +/- 1 Mb (primary)
WINDOW_KB_SENS = 10000.0     # +/- 10 Mb (sensitivity)
MHC = ("6", 25_000_000, 34_000_000)
MAF_MIN = 0.01
NA_VALUES = ["NA", "#NA", "", "NaN", "nan", "NULL", "null", "na", "N/A"]
CACHE_VERSION = "v1"


def _bin_s(phi: float) -> float:
    """Residual SD of a log-odds GWAS beta (per allele) for case fraction phi (simulation only)."""
    return 1.0 / np.sqrt(phi * (1 - phi))


_SSF = dict(chr="chromosome", pos="base_pair_location", ea="effect_allele", oa="other_allele", beta="beta",
            se="standard_error", eaf="effect_allele_frequency", p="p_value")
_PULIT = dict(chr="CHR", pos="POS", snp="SNP", ea="Tested_Allele", oa="Other_Allele", eaf="Freq_Tested_Allele",
              beta="BETA", se="SE", p="P", n="N")
_GLGC = dict(snp="rsID", chr="CHROM", pos="POS_b37", oa="REF", ea="ALT", n="N", eaf="POOLED_ALT_AF",
             beta="EFFECT_SIZE", se="SE", p="pvalue")
_MVP = dict(chr="chromosome", pos="base_pair_location", ea="effect_allele", oa="other_allele", eaf="effect_allele_frequency",
            p="p_value", snp="rsid", n="n", **{"or": "odds_ratio", "or_lci": "ci_lower", "or_uci": "ci_upper"})
_RET = dict(_SSF, snp="rsid", n="n")
# Ortin Vela 2024 GWAS Catalog files: the effect_allele / other_allele labels are swapped relative to beta and
# effect_allele_frequency (after label alignment their EAF correlates -0.66 with the Keaton SBP EAF, every other
# European file > 0.96; see eaf_concordance.csv). Beta and EAF refer to the allele in the other_allele column, so
# the labels are swapped back on reading ('swap_alleles'); this also restores the expected sign of SBP -> CRAE.
_RET_SPECIAL = ("swap_alleles",)

# Registry of the GWAS used. Fields: kind cont/bin; build 37/38; scale = exposure rescaling (estimate per `scale`
# source units); adverse = sign of the adverse direction of the outcome (+1 higher is worse, -1 lower is worse,
# 0 unclear); ukb = UK Biobank among the GWAS samples (True/False/None unknown); sim = simulation parameters.
SPECS = [
    # ---- primary European risk-factor exposures ----
    dict(id="sbp", label="SBP", group="risk", kind="cont", path="1d_SBP/GCST90310294.tsv.gz", sep="\t", build="37",
         cols=dict(_SSF, snp="rs_id", n="n"), n_default=1_028_980, ancestry="EUR", ukb=True, units="mmHg",
         scale=10.0, unit_label="per 10 mmHg", ref_map=True,
         source="Keaton 2024 Nat Genet (ICBP+UKB+MVP+BioVU), GCST90310294",
         sim=dict(trait="sbp", c=1.0, s=18.0, N=1_000_000)),
    dict(id="bmi", label="BMI", group="risk", kind="cont",
         path="6a_BMI/bmi.giant-ukbb.meta-analysis.combined.23May2018.txt.gz", sep=" ", build="37", cols=_PULIT,
         special=("pulit_snp",), n_default=806_834, ancestry="EUR", ukb=True, units="SD", unit_label="per SD",
         source="Pulit 2019 Hum Mol Genet (GIANT+UKB), Zenodo 1251813", sim=dict(trait="bmi", c=1, s=1, N=800_000)),
    dict(id="whradjbmi", label="WHRadjBMI", group="risk", kind="cont",
         path="6b_WHRadjBMI/whradjbmi.giant-ukbb.meta-analysis.combined.23May2018.txt.gz", sep=" ", build="37",
         cols=_PULIT, special=("pulit_snp",), n_default=694_649, ancestry="EUR", ukb=True, units="SD",
         unit_label="per SD", source="Pulit 2019 Hum Mol Genet (GIANT+UKB), Zenodo 1251813",
         sim=dict(trait="whr", c=1, s=1, N=690_000)),
    dict(id="tg", label="TG", group="risk", kind="cont",
         path="8-EUR-logTG_TG_log/logTG_INV_EUR_HRC_1KGP3_others_ALL.meta.singlevar.results.gz", sep="\t",
         build="37", cols=_GLGC, special=("pos_str",), n_default=1_320_000, ancestry="EUR", ukb=True, ref_map=True,
         units="SD of log TG (inverse-normal)", unit_label="per SD",
         source="Graham 2021 Nature (GLGC 2021), EUR logTG INV", sim=dict(trait="tg", c=1, s=1, N=1_300_000)),
    dict(id="osa", label="Sleep apnoea (MVP)", group="risk", kind="bin",
         path="5g_Sleep_apnoea_MVP_phecode/GCST90475824.tsv.gz", sep="\t", build="38", cols=_MVP,
         special=("or_ci",), n_default=430_058, ancestry="EUR", ukb=False, units="log-odds",
         unit_label="per log-odds", source="Verma 2024 Science (MVP EUR, phecode sleep apnoea), GCST90475824",
         sim=dict(trait="osa", c=1, s=_bin_s(0.35), N=430_000)),
    dict(id="fg", label="Fasting glucose", group="risk", kind="cont",
         path="9a_Fasting_glucose/GCST90002232_buildGRCh37.tsv.gz", sep="\t", build="37",
         cols=dict(_SSF, n="sample_size"), n_default=200_622, ancestry="EUR", ukb=None, units="mmol/L",
         unit_label="per mmol/L", source="Chen 2021 Nat Genet (MAGIC EUR), GCST90002232",
         sim=dict(trait="fg", c=1, s=0.7, N=200_000, eaf=False)),
    dict(id="hba1c", label="HbA1c", group="risk", kind="cont", path="9c_HbA1c/GCST90002244_buildGRCh37.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF, n="sample_size"), n_default=146_806, ancestry="EUR", ukb=None,
         units="% (DCCT)", unit_label="per 1 % HbA1c", source="Chen 2021 Nat Genet (MAGIC EUR), GCST90002244",
         sim=dict(trait="hba1c", c=1, s=0.5, N=146_000, eaf=False)),
    dict(id="t2d", label="T2D", group="risk", kind="bin", path="9d_T2D/Xue_et_al_T2D_META_Nat_Commun_2018.gz",
         sep=" ", build="37", cols=dict(chr="CHR", pos="BP", snp="SNP", ea="A1", oa="A2", eaf="frq_A1", beta="b",
                                        se="se", p="P", n="N"),
         n_default=659_316, ancestry="EUR", ukb=True, units="log-odds", unit_label="per log-odds",
         source="Xue 2018 Nat Commun (DIAGRAM+GERA+UKB), GCST006867",
         sim=dict(trait="t2d", c=1, s=_bin_s(0.095), N=659_000)),
    # ---- OSA sensitivity exposures ----
    dict(id="osa_campos", label="Sleep apnoea (UKB+FinnGen)", group="risk_sens", kind="bin",
         path="5f_Sleep_apnoea_UKB_FinnGen/GCST90267361.tsv", sep="\t", build="37", cols=dict(_SSF, snp="rs_id"),
         n_default=362_638, ancestry="EUR", ukb=True, units="log-odds", unit_label="per log-odds",
         source="Campos 2023 Sleep (UKB+FinnGen), GCST90267361",
         sim=dict(trait="osa", c=1, s=_bin_s(0.07), N=362_000)),
    dict(id="osa_kurn", label="OSA (MVP+FinnGen, multi-ancestry)", group="risk_sens", kind="bin",
         path="5d_OSA_BMI_unadjusted/GCST90693193.tsv", sep="\t", build="38", cols=dict(_SSF),
         n_default=980_757, ancestry="multi (EUR, AFR, HIS, EAS)", ukb=False, units="log-odds",
         unit_label="per log-odds", xwalk_via="osa",
         source="Kurniansyah 2025 EBioMedicine, BMI-unadjusted, GCST90693193",
         sim=dict(trait="osa", c=1, s=_bin_s(0.3), N=980_000)),
    dict(id="osa_kurn_bmiadj", label="OSA BMI-adjusted (MVP+FinnGen, multi-ancestry)", group="risk_sens",
         kind="bin", path="5e_OSA_BMI_adjusted/GCST90693190.tsv", sep="\t", build="38", cols=dict(_SSF),
         n_default=749_890, ancestry="multi (EUR, AFR, HIS, EAS)", ukb=False, units="log-odds",
         unit_label="per log-odds", xwalk_via="osa",
         source="Kurniansyah 2025 EBioMedicine, BMI-adjusted, GCST90693190",
         sim=dict(trait="osa", c=1, s=_bin_s(0.3), N=750_000)),
    # ---- BioBank Japan / East Asian exposures ----
    dict(id="sbp_bbj", label="SBP (BBJ)", group="bbj", kind="cont", path="11a_SBP/GCST90018752_buildGRCh37.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF), n_default=145_505, ancestry="EAS (JPN)", ukb=False,
         units="SD (inverse-normal)", unit_label="per SD", source="Sakaue 2021 Nat Genet (BBJ), GCST90018752",
         sim=dict(trait="sbp", c=1 / 18.0, s=1, N=145_000, eas=True)),
    dict(id="bmi_bbj", label="BMI (BBJ)", group="bbj", kind="cont", path="11d_BMI/GCST90018727_buildGRCh37.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF), n_default=163_835, ancestry="EAS (JPN)", ukb=False,
         units="SD (inverse-normal)", unit_label="per SD", source="Sakaue 2021 Nat Genet (BBJ), GCST90018727",
         sim=dict(trait="bmi", c=1, s=1, N=163_000, eas=True)),
    dict(id="tg_bbj", label="TG (BBJ)", group="bbj", kind="cont", path="11c_TG/GCST90018755_buildGRCh37.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF), n_default=111_667, ancestry="EAS (JPN)", ukb=False,
         units="SD (inverse-normal)", unit_label="per SD", source="Sakaue 2021 Nat Genet (BBJ), GCST90018755",
         sim=dict(trait="tg", c=1, s=1, N=111_000, eas=True)),
    dict(id="tg_eas", label="TG (GLGC EAS)", group="bbj", kind="cont",
         path="8-EAS-logTG_TG_log/logTG_INV_EAS_1KGP3_ALL.meta.singlevar.results.gz", sep="\t", build="37",
         cols=_GLGC, special=("pos_str",), n_default=146_492, ancestry="EAS (incl. BBJ)", ukb=False,
         units="SD of log TG (inverse-normal)", unit_label="per SD",
         source="Graham 2021 Nature (GLGC 2021), EAS logTG INV", sim=dict(trait="tg", c=1, s=1, N=146_000, eas=True)),
    # ---- vascular beds (outcomes of A-D, exposures of E) ----
    dict(id="asi", label="Arterial stiffness index", group="bed", kind="cont",
         path="2a_Arterial_stiffness_index_UKB_21021/31235810-GCST008403-EFO_0004517-Build37.f.tsv.gz", sep="\t",
         build="37", cols=dict(_SSF), n_default=127_121, ancestry="EUR (UKB)", ukb=True,
         units="SD (GWAS-implied SD 1.0)", unit_label="per SD", adverse=1,
         source="Fung 2019 Sci Rep (UKB field 21021), GCST008403", sim=dict(trait="asi", c=1, s=1.5, N=127_000)),
    dict(id="cimt", label="Carotid IMT", group="bed", kind="cont", path="3a_Carotid_IMT/GCST007436.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF, snp="rs_id", n="n"), n_default=71_128, ancestry="EUR", ukb=False,
         units="IMT source units (mm)", unit_label="per source unit", adverse=1,
         source="Franceschini 2018 Nat Commun (CHARGE), GCST007436", sim=dict(trait="cimt", c=1, s=0.13, N=71_000)),
    dict(id="plaque", label="Carotid plaque", group="bed", kind="bin", path="3b_Carotid_plaque/GCST007435.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF, snp="rs_id", n="n"), n_default=48_434, ancestry="EUR", ukb=False,
         units="log-odds", unit_label="per log-odds", adverse=1,
         source="Franceschini 2018 Nat Commun (CHARGE), GCST007435",
         sim=dict(trait="plaque", c=1, s=_bin_s(0.45), N=48_000)),
    dict(id="crae", label="Retinal CRAE", group="bed", kind="cont",
         path="4a_Retinal_CRAE_arteriolar_calibre/GCST90446763.tsv.gz", sep="\t", build="37", cols=_RET, special=_RET_SPECIAL,
         n_default=65_015, ancestry="EUR (UKB)", ukb=True, units="SD (source scale)", unit_label="per SD",
         adverse=-1, ref_map=True, source="Ortin Vela 2024 Nat Commun (UKB), GCST90446763",
         sim=dict(trait="crae", c=1, s=1, N=65_000)),
    dict(id="crve", label="Retinal CRVE", group="bed", kind="cont",
         path="4b_Retinal_CRVE_venular_calibre/GCST90446764.tsv.gz", sep="\t", build="37", cols=_RET, special=_RET_SPECIAL,
         n_default=65_374, ancestry="EUR (UKB)", ukb=True, units="SD (source scale)", unit_label="per SD",
         adverse=1, source="Ortin Vela 2024 Nat Commun (UKB), GCST90446764",
         sim=dict(trait="crve", c=1, s=1, N=65_000)),
    dict(id="avr", label="Retinal AVR", group="bed", kind="cont",
         path="4c_Retinal_AVR_CRAE_CRVE_ratio/GCST90446765.tsv.gz", sep="\t", build="37", cols=_RET, special=_RET_SPECIAL,
         n_default=64_431, ancestry="EUR (UKB)", ukb=True, units="SD (source scale)", unit_label="per SD",
         adverse=-1, source="Ortin Vela 2024 Nat Commun (UKB), GCST90446765",
         sim=dict(trait="avr", c=1, s=1, N=64_000)),
    dict(id="tort", label="Retinal arteriolar tortuosity", group="bed", kind="cont",
         path="4d_Retinal_arteriolar_tortuosity/GCST90446760.tsv.gz", sep="\t", build="37", cols=_RET, special=_RET_SPECIAL,
         n_default=68_088, ancestry="EUR (UKB)", ukb=True, units="SD (source scale)", unit_label="per SD",
         adverse=1, source="Ortin Vela 2024 Nat Commun (UKB), GCST90446760",
         sim=dict(trait="tort", c=1, s=1, N=68_000)),
    dict(id="dens", label="Retinal arterial vascular density", group="bed", kind="cont",
         path="4e_Retinal_arterial_vascular_density/GCST90446768.tsv.gz", sep="\t", build="37", cols=_RET, special=_RET_SPECIAL,
         n_default=68_249, ancestry="EUR (UKB)", ukb=True, units="SD (source scale)", unit_label="per SD",
         adverse=-1, source="Ortin Vela 2024 Nat Commun (UKB), GCST90446768",
         sim=dict(trait="dens", c=1, s=1, N=68_000)),
    # ---- PAD (bed outcome and clinical outcome) ----
    dict(id="pad_mvp", label="PAD (MVP phecode)", group="pad", kind="bin",
         path="17a_Peripheral_vascular_disease_MVP_phecode/GCST90475996.tsv.gz", sep="\t", build="38", cols=_MVP,
         special=("or_ci",), n_default=426_178, ancestry="EUR (MVP)", ukb=False, units="log-odds",
         unit_label="per log-odds", adverse=1, source="Verma 2024 Science (MVP EUR), GCST90475996",
         sim=dict(trait="pad", c=1, s=_bin_s(0.1), N=426_000)),
    dict(id="pad_sakaue", label="PAD (BBJ+UKB+FinnGen)", group="pad", kind="bin",
         path="17b_Peripheral_artery_disease/GCST90018890_buildGRCh37.tsv.gz", sep="\t", build="37",
         cols=dict(_SSF), n_default=660_791, ancestry="EUR+EAS", ukb=True, units="log-odds",
         unit_label="per log-odds", adverse=1, source="Sakaue 2021 Nat Genet (BBJ+UKB+FinnGen), GCST90018890",
         sim=dict(trait="pad", c=1, s=_bin_s(0.017), N=660_000)),
    # ---- clinical outcomes ----
    dict(id="cad", label="CAD", group="clin", kind="bin", path="14a_CAD/GCST90132314_buildGRCh37.tsv", sep="\t",
         build="37", cols=dict(_SSF, n="n"), n_default=1_165_690, ancestry="EUR", ukb=True, units="log-odds",
         unit_label="per log-odds", adverse=1, special=("lower_alleles",),
         source="Aragam 2022 Nat Genet (EUR), GCST90132314", sim=dict(trait="cad", c=1, s=_bin_s(0.156), N=1_165_000)),
    dict(id="stroke", label="Any stroke", group="clin", kind="bin",
         path="15a_Any_stroke/GCST90104539_buildGRCh37.tsv.gz", sep="\t", build="37", cols=dict(_SSF),
         n_default=1_308_460, ancestry="EUR", ukb=True, units="log-odds", unit_label="per log-odds", adverse=1,
         source="Mishra 2022 Nature (GIGASTROKE EUR), GCST90104539",
         sim=dict(trait="stroke", c=1, s=_bin_s(0.056), N=1_300_000)),
    dict(id="istroke", label="Ischaemic stroke", group="clin", kind="bin",
         path="15b_Ischaemic_stroke/GCST90104540_buildGRCh37.tsv.gz", sep="\t", build="37", cols=dict(_SSF),
         n_default=1_296_908, ancestry="EUR", ukb=True, units="log-odds", unit_label="per log-odds", adverse=1,
         source="Mishra 2022 Nature (GIGASTROKE EUR), GCST90104540",
         sim=dict(trait="istroke", c=1, s=_bin_s(0.048), N=1_296_000)),
    dict(id="hf", label="Heart failure", group="clin", kind="bin", path="16a_Heart_failure/GCST90728695.tsv.gz",
         sep="\t", build="37", cols=dict(_SSF, snp="rsid", n="N_total"), n_default=1_708_342, ancestry="EUR",
         ukb=True, units="log-odds", unit_label="per log-odds", adverse=1,
         source="Henry 2025 Nat Genet (HERMES 2.0 EUR), GCST90728695",
         sim=dict(trait="hf", c=1, s=_bin_s(0.082), N=1_700_000)),
    dict(id="life", label="Parental lifespan (log HR of death)", group="clin", kind="cont",
         path="19a_Parental_lifespan/lifegen_phase2_bothpl_alldr_2017_09_18.tsv.gz", sep="\t", build="37",
         cols=dict(snp="rsid", chr="chr", pos="pos", ea="a1", oa="a0", n="n", eaf="freq1", beta="beta1", se="se",
                   p="p"),
         special=("flip_sign",), n_default=1_012_240, ancestry="EUR", ukb=True,
         units="log HR of parental death (Timmers beta1 sign-flipped)", unit_label="per unit", adverse=1,
         source="Timmers 2019 eLife (UKB+LifeGen), GCST009890", sim=dict(trait="life", c=1, s=1.0, N=1_000_000)),
]
CHAMI = dict(id="chami_bfp", label="Uncoupled adiposity (Chami 2025; body-fat %)", group="chami", kind="cont",
             path="12a_Adiposity_variants_uncoupled_from_cardio/41591_2025_3931_MOESM2_ESM.xlsx", build="37",
             n_default=444_572, ancestry="EUR (UKB)", ukb=True, units="SD body-fat %", unit_label="per SD BFP",
             source="Chami 2025 Nat Med, Supplementary Table 3 (266 lead variants, BFP association)",
             sim=dict(trait="bfp", c=1, s=1, N=444_000))
SPEC = {s["id"]: s for s in SPECS + [CHAMI]}
RISK = ["sbp", "bmi", "whradjbmi", "tg", "osa", "fg", "hba1c", "t2d"]
RISK_SENS = ["osa_campos", "osa_kurn", "osa_kurn_bmiadj"]
BBJ = ["sbp_bbj", "bmi_bbj", "tg_bbj", "tg_eas"]
BEDS = ["asi", "cimt", "plaque", "crae", "crve", "avr", "tort", "dens"]
BED_OUT = BEDS + ["pad_mvp", "pad_sakaue"]
CLIN_PC = ["cad", "stroke", "istroke", "hf", "life"]
CLIN = ["cad", "stroke", "istroke", "hf", "pad_mvp", "life"]
MVMR_X = ["bmi", "whradjbmi", "tg", "osa", "fg", "hba1c", "t2d"]
PASS1 = RISK + RISK_SENS + BBJ + BEDS
PASS2 = RISK + BEDS + ["pad_mvp", "pad_sakaue"] + CLIN_PC
REF_MAP = [s["id"] for s in SPECS if s.get("ref_map")]
NOT_USED = [
    ("1a-1c Evangelou 2018 SBP/DBP/PP", "superseded by Keaton 2024 SBP (1d); no rsID"),
    ("1e/1f Keaton DBP/PP", "not in the F1 MR exposure list"),
    ("6c WHR", "WHRadjBMI (6b) used as the fat-distribution exposure"),
    ("9b fasting insulin", "BMI-adjusted GWAS; glycaemia represented by FG, HbA1c, T2D"),
    ("8-EUR LDL/HDL", "not in the F1 MR exposure list"),
    ("10a/10c/18a/18b urate, eGFR, CKD, UACR", "not in the F1 MR exposure list"),
    ("20a atrial fibrillation", "not in the F1 outcome list"),
    ("7a-7q liver fat / cT1 / NAFLD / liver enzymes", "out of scope for F1"),
    ("12b-12d", "R code / liver-fat partitioned scores / figures; not GWAS"),
    ("Retinal fractal dimension, cfPWV/baPWV, aortic distensibility", "no open full summary statistics"),
]

# ================================================================================================================
# Small utilities
# ================================================================================================================
T0 = time.time()


def log(msg: str):
    print(f"[{_dt.datetime.now():%H:%M:%S} +{time.time() - T0:7.1f}s] {msg}", flush=True)


def maxrss_mb(who=resource.RUSAGE_SELF) -> float:
    r = resource.getrusage(who).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3       # macOS: bytes; Linux: kB


def bh(p) -> np.ndarray:
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return q
    o = ok[np.argsort(p[ok], kind="mergesort")]
    n = len(o)
    qq = p[o] * n / np.arange(1, n + 1)
    qq = np.minimum.accumulate(qq[::-1])[::-1]
    q[o] = np.minimum(qq, 1.0)
    return q


def read_header(path: Path, sep: str) -> list[str]:
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        line = fh.readline().rstrip("\n").rstrip("\r")
    return line.split(sep)


def akey(a: pd.Series, b: pd.Series) -> pd.Series:
    """Unordered allele-pair key 'A|G'."""
    a, b = a.astype("string"), b.astype("string")
    lo = a.where(a <= b, b)
    hi = b.where(a <= b, a)
    return lo + "|" + hi


def akey_comp(a: pd.Series, b: pd.Series) -> pd.Series:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    return akey(a.map(comp).astype("string"), b.map(comp).astype("string"))


def cpkey(chr_: pd.Series, pos: pd.Series) -> pd.Series:
    return chr_.astype("string") + ":" + pos.astype("Int64").astype("string")


def with_ids(d: pd.DataFrame) -> pd.DataFrame:
    """Give rows without rsID a synthetic, stable id chr:pos:ea:oa (kept through lib_mr.harmonise)."""
    d = d.copy()
    miss = d["snp"].isna()
    if miss.any():
        d.loc[miss, "snp"] = (d.loc[miss, "chr"].astype("string") + ":" + d.loc[miss, "pos"].astype("string") + ":"
                              + d.loc[miss, "ea"] + ":" + d.loc[miss, "oa"])
    return d


# ================================================================================================================
# Streaming reader (pyarrow) and standardisation
# ================================================================================================================
def _pa_types(spec):
    import pyarrow as pa
    sp = set(spec.get("special", ()))
    t = {}
    for k, col in spec["cols"].items():
        if k in ("chr", "ea", "oa", "snp"):
            t[col] = pa.string()
        elif k == "pos":
            t[col] = pa.string() if "pos_str" in sp else pa.float64()
        else:
            t[col] = pa.float64()
    return t


def standardize(spec: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """Raw extracted rows -> internal schema snp, chr, pos, ea, oa, beta, se, p, eaf, n (+ pos38 for GRCh38)."""
    c, sp = spec["cols"], set(spec.get("special", ()))
    d = pd.DataFrame(index=raw.index)
    ch = raw[c["chr"]].astype("string").str.strip().str.replace(r"^chr", "", regex=True, case=False)
    d["chr"] = ch.replace({"X": "23", "x": "23", "Y": "24", "MT": "26"})
    pos = raw[c["pos"]]
    if "pos_str" in sp:
        pos = pos.astype("string").str.strip()
    d["pos"] = pd.to_numeric(pos, errors="coerce").round().astype("Int64")
    if "snp" in c:
        s = raw[c["snp"]].astype("string").str.strip()
        if "pulit_snp" in sp:
            s = s.str.split(":").str[0].astype("string")
        d["snp"] = s.where(s.str.match(r"^rs\d+$").fillna(False), pd.NA)
    else:
        d["snp"] = pd.Series(pd.NA, index=raw.index, dtype="string")
    ea_col, oa_col = (c["oa"], c["ea"]) if "swap_alleles" in sp else (c["ea"], c["oa"])
    d["ea"] = raw[ea_col].astype("string").str.strip().str.upper()
    d["oa"] = raw[oa_col].astype("string").str.strip().str.upper()
    p = pd.to_numeric(raw[c["p"]], errors="coerce")
    if "or" in c:
        orv = pd.to_numeric(raw[c["or"]], errors="coerce")
        beta = np.log(orv.where(orv > 0))
        se_ci = pd.Series(M.se_from_ci(raw[c["or_lci"]], raw[c["or_uci"]]), index=raw.index)
        se_bp = pd.Series(M.se_from_beta_p(beta, p), index=raw.index)
        # OR and CI are printed with 4 significant digits: for strong signals beta + p is more precise
        use_bp = np.isfinite(se_bp) & (p <= 1e-5)
        se = se_bp.where(use_bp, se_ci)
        se = se.where(np.isfinite(se), se_bp)
        d["se_source"] = np.where(use_bp, "beta_p", "or_ci")
    else:
        beta = pd.to_numeric(raw[c["beta"]], errors="coerce")
        se = pd.to_numeric(raw[c["se"]], errors="coerce")
        d["se_source"] = "se"
    if "flip_sign" in sp:
        beta = -beta
    d["beta"], d["se"], d["p"] = beta.astype(float), se.astype(float), p.astype(float)
    d["eaf"] = pd.to_numeric(raw[c["eaf"]], errors="coerce") if "eaf" in c else np.nan
    n = pd.to_numeric(raw[c["n"]], errors="coerce") if "n" in c else pd.Series(np.nan, index=raw.index)
    d["n"] = n.fillna(spec["n_default"]).astype(float)
    if spec["build"] == "38":
        d["pos38"] = d["pos"]
        d["pos"] = pd.Series(pd.NA, index=raw.index, dtype="Int64")
    else:
        d["pos38"] = pd.Series(pd.NA, index=raw.index, dtype="Int64")
    return d.reset_index(drop=True)


def scan_worker(task: dict) -> dict:
    """Stream one GWAS file and return standardised rows that pass the filter.

    task: spec, path, mode ('sig' | 'keys'), p_thresh, rs (list of rsIDs), pos (list of native positions),
    pairs (set of 'chr:pos' native keys for refinement), threads.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    pa.set_cpu_count(task.get("threads", 2))
    t0 = time.time()
    spec, path, mode = task["spec"], str(task["path"]), task["mode"]
    c, sp = spec["cols"], set(spec.get("special", ()))
    include = list(dict.fromkeys(c.values()))
    reader = pcsv.open_csv(
        pa.input_stream(path, compression="gzip" if path.endswith(".gz") else None),
        read_options=pcsv.ReadOptions(block_size=32 << 20),
        parse_options=pcsv.ParseOptions(delimiter=spec["sep"]),
        convert_options=pcsv.ConvertOptions(include_columns=include, column_types=_pa_types(spec),
                                            null_values=NA_VALUES, strings_can_be_null=True))
    snpcol = c.get("snp") if "pulit_snp" not in sp else None
    rs_arr = pa.array(sorted(task["rs"]), pa.string()) if task.get("rs") else None
    pos_arr = pa.array(sorted(task["pos"]), pa.float64()) if task.get("pos") else None
    parts, nrows, nsig = [], 0, 0
    for batch in reader:
        nrows += batch.num_rows
        if mode == "sig":
            m = pc.less_equal(batch.column(c["p"]), task["p_thresh"])
        else:
            m = None
            if rs_arr is not None and snpcol:
                m = pc.is_in(batch.column(snpcol), value_set=rs_arr)
            if pos_arr is not None:
                pv = batch.column(c["pos"])
                if "pos_str" in sp:
                    pv = pc.cast(pc.utf8_trim_whitespace(pv), pa.float64())
                m2 = pc.is_in(pv, value_set=pos_arr)
                m = m2 if m is None else pc.or_(m, m2)
            if m is None:
                continue
        m = pc.fill_null(m, False)
        sub = batch.filter(m)
        if sub.num_rows:
            parts.append(sub)
            nsig += sub.num_rows
    if parts:
        raw = pa.Table.from_batches(parts).to_pandas()
    else:
        raw = pd.DataFrame({k: pd.Series(dtype="object") for k in include})
    df = standardize(spec, raw)
    if mode == "keys":
        native_pos = df["pos38"] if spec["build"] == "38" else df["pos"]
        k = cpkey(df["chr"], native_pos)
        keep = pd.Series(False, index=df.index)
        if task.get("rs"):
            keep |= df["snp"].isin(set(task["rs"])).fillna(False)
        if task.get("pairs"):
            keep |= k.isin(task["pairs"]).fillna(False)
        df = df[keep].reset_index(drop=True)
    return {"id": spec["id"], "mode": mode, "df": df, "rows": nrows, "kept": len(df), "seconds": time.time() - t0,
            "maxrss_mb": maxrss_mb()}


def hash_worker(path: str) -> tuple[str, str]:
    return path, C.sha256(Path(path))


def run_pool(func, tasks, workers):
    if workers <= 1 or len(tasks) <= 1:
        return [func(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
        return list(ex.map(func, tasks))


class Scanner:
    """Runs scan tasks in parallel with a parquet cache keyed by file identity and filter."""

    def __init__(self, paths: dict, cache: Path, workers: int, use_cache: bool, stats_rows: list):
        self.paths, self.cache, self.workers, self.use_cache = paths, cache, workers, use_cache
        self.stats = stats_rows
        cache.mkdir(parents=True, exist_ok=True)

    def _key(self, sid, mode, extra) -> str:
        p = Path(self.paths[sid])
        st = p.stat()
        sp = SPEC[sid]
        spec_key = [sp.get("cols"), sp.get("special"), sp.get("build"), sp.get("n_default"), sp.get("sep")]
        h = hashlib.sha1(json.dumps([CACHE_VERSION, sid, str(p), st.st_size, int(st.st_mtime), mode, extra, spec_key],
                                    sort_keys=True, default=str).encode()).hexdigest()[:16]
        return f"{sid}_{mode}_{h}"

    def run(self, jobs: list[tuple[str, str, dict]], label: str) -> dict:
        """jobs: (spec_id, mode, filter dict). Returns id -> DataFrame."""
        out, todo = {}, []
        for sid, mode, flt in jobs:
            extra = {"p": flt.get("p_thresh"), "rs": hashlib.sha1("\n".join(sorted(flt.get("rs") or [])).encode()).hexdigest(),
                     "pos": hashlib.sha1("\n".join(sorted(flt.get("pairs") or [])).encode()).hexdigest()}
            key = self._key(sid, mode, extra)
            f = self.cache / f"{key}.parquet"
            if self.use_cache and f.exists():
                out[sid] = pd.read_parquet(f)
                meta = json.loads((self.cache / f"{key}.json").read_text())
                self.stats.append({"step": label, "id": sid, **meta, "cached": True})
                continue
            todo.append((sid, mode, flt, f, key))
        # biggest files first
        todo.sort(key=lambda t: -Path(self.paths[t[0]]).stat().st_size)
        tasks = [dict(spec=SPEC[sid], path=str(self.paths[sid]), mode=mode, threads=2, **flt)
                 for sid, mode, flt, _, _ in todo]
        if tasks:
            log(f"{label}: scanning {len(tasks)} files with {min(self.workers, len(tasks))} workers")
        t0 = time.time()
        res = run_pool(scan_worker, tasks, self.workers)
        for (sid, mode, flt, f, key), r in zip(todo, res):
            df = r["df"]
            df.to_parquet(f, index=False)
            meta = {"rows_scanned": r["rows"], "rows_kept": r["kept"], "seconds": round(r["seconds"], 1),
                    "worker_maxrss_mb": round(r["maxrss_mb"], 1)}
            (self.cache / f"{key}.json").write_text(json.dumps(meta))
            self.stats.append({"step": label, "id": sid, **meta, "cached": False})
            out[sid] = df
            log(f"  {label} {sid}: {r['rows']:,} rows scanned, {r['kept']:,} kept, {r['seconds']:.1f}s")
        if tasks:
            log(f"{label}: done in {time.time() - t0:.1f}s")
        return out


# ================================================================================================================
# Instrument selection: candidate filters and fast greedy distance clumping
# ================================================================================================================
def clump_fast(chr_, pos, p, z, window_kb: float) -> np.ndarray:
    """Greedy distance clumping, same rule as lib_mr.clump_distance (index = smallest p; ties by largest |z|;
    remove all candidates on the same chromosome within +/- window). Returns a boolean keep mask."""
    chr_ = np.asarray(pd.Series(chr_).astype(str))
    pos = np.asarray(pd.Series(pos).astype(float))
    p = np.asarray(p, float)
    z = np.nan_to_num(np.abs(np.asarray(z, float)))
    n = len(p)
    keep = np.zeros(n, bool)
    removed = np.zeros(n, bool)
    order = np.lexsort((np.arange(n), -z, p))
    groups = {}
    for ch in np.unique(chr_):
        idx = np.where(chr_ == ch)[0]
        o = idx[np.argsort(pos[idx], kind="mergesort")]
        groups[ch] = (pos[o], o)
    w = window_kb * 1000
    for i in order:
        if removed[i]:
            continue
        keep[i] = True
        sp, so = groups[chr_[i]]
        lo, hi = np.searchsorted(sp, pos[i] - w, "left"), np.searchsorted(sp, pos[i] + w, "right")
        removed[so[lo:hi]] = True
    return keep


def candidate_filter(d: pd.DataFrame, native_pos: pd.Series) -> tuple[pd.DataFrame, dict]:
    """p <= 5e-8 rows -> eligible candidates (autosomes, no MHC, MAF, valid alleles, finite beta/se)."""
    n0 = len(d)
    def b(x):
        return np.asarray(pd.Series(x).fillna(False).astype(bool))
    chrn = pd.to_numeric(d["chr"], errors="coerce")
    ok_chr = b(chrn.between(1, 22))
    npos = native_pos.astype("Float64")
    mhc = b(d["chr"] == MHC[0]) & b(npos.between(MHC[1], MHC[2]))
    maf_ok = ~b(d["eaf"].notna() & ((d["eaf"] < MAF_MIN) | (d["eaf"] > 1 - MAF_MIN)))
    allele_ok = b(d["ea"].str.fullmatch(r"[ACGT]+")) & b(d["oa"].str.fullmatch(r"[ACGT]+")) & b(d["ea"] != d["oa"])
    num_ok = b(np.isfinite(d["beta"]) & np.isfinite(d["se"]) & (d["se"] > 0)) & b(npos.notna())
    keep = ok_chr & ~mhc & maf_ok & allele_ok & num_ok & b(d["p"] <= P_THRESH)
    info = {"p_le_5e8_rows": n0, "drop_non_autosome": int((~ok_chr).sum()), "drop_mhc": int((ok_chr & mhc).sum()),
            "drop_maf": int((~maf_ok).sum()), "drop_alleles": int((~allele_ok).sum()),
            "candidates": int(keep.sum())}
    return d[keep].reset_index(drop=True), info


def select_instruments(cand: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    native = cand["pos38"] if spec["build"] == "38" else cand["pos"]
    c2, info = candidate_filter(cand, native)
    native = c2["pos38"] if spec["build"] == "38" else c2["pos"]
    z = c2["beta"] / c2["se"]
    k1 = clump_fast(c2["chr"], native, c2["p"], z, WINDOW_KB)
    k10 = clump_fast(c2["chr"], native, c2["p"], z, WINDOW_KB_SENS)
    info["instruments_1mb"] = int(k1.sum())
    info["instruments_10mb"] = int(k10.sum())
    return c2[k1].reset_index(drop=True), c2[k10].reset_index(drop=True), info


# ================================================================================================================
# Variant maps (GRCh38 -> rsID cross-walk; rsID <-> GRCh37 chr:pos)
# ================================================================================================================
def crosswalk_38(inst: pd.DataFrame, ref38: pd.DataFrame) -> tuple[pd.Series, int]:
    """rsIDs for GRCh38 instruments without rsID from a GRCh38 file with rsIDs (same coordinates), allele-checked."""
    if inst.empty or ref38.empty:
        return pd.Series(pd.NA, index=inst.index, dtype="string"), 0
    r = ref38.dropna(subset=["snp", "pos38"]).copy()
    r["k"] = cpkey(r["chr"], r["pos38"])
    r["a"] = akey(r["ea"], r["oa"])
    look = {}
    for k, a, s in zip(r["k"], r["a"], r["snp"]):
        look.setdefault((k, a), s)
    k = cpkey(inst["chr"], inst["pos38"])
    a, ac = akey(inst["ea"], inst["oa"]), akey_comp(inst["ea"], inst["oa"])
    out = [look.get((kk, aa), look.get((kk, cc), pd.NA)) for kk, aa, cc in zip(k, a, ac)]
    s = pd.Series(out, index=inst.index, dtype="string")
    return s, int(s.notna().sum())


class VariantMap:
    """rsID -> GRCh37 (chr, pos) and GRCh37 (chr, pos, alleles) -> rsID from GRCh37 files with rsIDs."""

    def __init__(self):
        self.rs2pos: dict = {}
        self.pos2rs: dict = {}

    def add(self, d: pd.DataFrame):
        d = d.dropna(subset=["snp", "pos"])
        if d.empty:
            return
        for s, ch, ps, a in zip(d["snp"], d["chr"], d["pos"].astype(int), akey(d["ea"], d["oa"])):
            self.rs2pos.setdefault(s, (ch, int(ps)))
            self.pos2rs.setdefault((f"{ch}:{ps}", a), s)

    def fill_rsid(self, d: pd.DataFrame) -> int:
        miss = d["snp"].isna() & d["pos"].notna()
        if not miss.any():
            return 0
        k = cpkey(d.loc[miss, "chr"], d.loc[miss, "pos"])
        a = akey(d.loc[miss, "ea"], d.loc[miss, "oa"])
        ac = akey_comp(d.loc[miss, "ea"], d.loc[miss, "oa"])
        vals = [self.pos2rs.get((kk, aa), self.pos2rs.get((kk, cc), pd.NA)) for kk, aa, cc in zip(k, a, ac)]
        d.loc[miss, "snp"] = pd.Series(vals, index=d.index[miss], dtype="string")
        return int(pd.Series(vals).notna().sum())

    def fill_pos37(self, d: pd.DataFrame) -> int:
        miss = d["pos"].isna() & d["snp"].notna()
        if not miss.any():
            return 0
        got = [self.rs2pos.get(s) for s in d.loc[miss, "snp"]]
        ch = [g[0] if g else pd.NA for g in got]
        ps = [g[1] if g else pd.NA for g in got]
        chr_ok = [(g is None) or (g[0] == c0) for g, c0 in zip(got, d.loc[miss, "chr"])]
        ps = [p if ok else pd.NA for p, ok in zip(ps, chr_ok)]
        d.loc[miss, "pos"] = pd.array(ps, dtype="Int64")
        d.loc[miss, "chr"] = [c if (c is not pd.NA and ok) else c0 for c, ok, c0 in zip(ch, chr_ok, d.loc[miss, "chr"])]
        return int(pd.Series(ps).notna().sum())


# ================================================================================================================
# MR per pair and MVMR (run in worker processes)
# ================================================================================================================
def _r2(beta, se, n, eaf, kind):
    beta, se, n = np.asarray(beta, float), np.asarray(se, float), np.asarray(n, float)
    if kind == "bin":
        f = np.where(np.isfinite(eaf), eaf, 0.5)
        return M.r2_from_lor(beta, f)
    return M.r2_from_beta_se(beta, se, n)


def _harm(exp: pd.DataFrame, out: pd.DataFrame, drop_pal: bool = False) -> tuple[pd.DataFrame, dict]:
    """lib_mr.harmonise; drop_pal=True drops every palindromic SNP (used when either GWAS is not European, because
    EAF-based strand inference is unreliable across ancestries)."""
    h = M.harmonise(exp, out, match_on="auto", pal_band=(0.0, 1.0) if drop_pal else (0.42, 0.58))
    acts = h["action"].value_counts().to_dict() if len(h) else {}
    hk = h[h["keep"]].drop_duplicates(subset=["snp"]).reset_index(drop=True) if len(h) else h
    return hk, acts


IVW_PRIMARY = "IVW (multiplicative random effects)"     # documented primary estimator (lib_mr.mr_ivw "random")
IVW_FIXED = "IVW (fixed effects)"


def _methods(bx, by, sx, sy, n_boot, n_boot_mode, seed, full=True) -> list[dict]:
    rows = [M.mr_ivw(bx, by, sy, "random"), M.mr_ivw(bx, by, sy, "fixed")]
    if len(bx) >= 3:
        rows.append(M.mr_egger(bx, by, sx, sy))
        rows.append(M.mr_weighted_median(bx, by, sx, sy, n_boot=n_boot, seed=seed))
        if full:
            rows.append(M.mr_weighted_mode(bx, by, sx, sy, n_boot=n_boot_mode, seed=seed))
    return rows


def _pick(rows: list[dict], method: str) -> dict:
    """The one estimator row whose method name is exactly `method` (fails loudly if absent or duplicated).

    2026-10-05 fix: estimators used to be keyed by method.split(' (')[0], which maps both IVW rows to 'IVW' so the
    fixed-effect row (appended second) overwrote the random-effects primary."""
    hit = [r for r in rows if r["method"] == method]
    if len(hit) != 1:
        raise KeyError(f"expected exactly one '{method}' row, found {len(hit)}")
    return hit[0]


def mr_pair_worker(job: dict) -> dict:
    """Univariable two-sample MR for one exposure-outcome pair (primary + sensitivity analyses)."""
    try:
        return _mr_pair(job)
    except Exception as e:                                            # pragma: no cover - reported in the table
        return {"summary": {"exposure": job["e"]["id"], "outcome": job["o"]["id"], "family": job["family"],
                            "error": f"{type(e).__name__}: {e}"}, "long": [], "snps": None,
                "harm": {}, "trace": traceback.format_exc()}


def _mr_pair(job: dict) -> dict:
    e, o = job["e"], job["o"]
    sc = float(e.get("scale", 1.0))
    seed = job["seed"]
    hk, acts = _harm(job["inst"], job["out"], job.get("drop_pal", False))
    S = {"family": job["family"], "exposure": e["id"], "outcome": o["id"], "exposure_label": e["label"],
         "outcome_label": o["label"], "exposure_unit": e.get("unit_label", ""), "outcome_units": o.get("units", ""),
         "n_instruments": len(job["inst"]), "nsnp": len(hk), "palindromic_rule": "all dropped (non-EUR pair)"
         if job.get("drop_pal") else "EAF-inferred outside [0.42, 0.58]",
         "n_palindromic_dropped": int(sum(v for k, v in acts.items() if k.startswith("drop_palindromic"))),
         "n_incompatible_dropped": int(acts.get("drop_incompatible", 0)),
         "n_eaf_discordant": int(hk["eaf_discordant"].sum()) if len(hk) else 0,
         "ukb_overlap": (None if (e.get("ukb") is None or o.get("ukb") is None) else bool(e["ukb"] and o["ukb"]))}
    long = []
    if len(hk) == 0:
        return {"summary": S, "long": long, "snps": None, "harm": acts}
    bx, sx = hk["beta_x"].values / sc, hk["se_x"].values / sc
    by, sy = hk["beta_y"].values, hk["se_y"].values
    res = _methods(bx, by, sx, sy, job["n_boot"], job["n_boot_mode"], seed)
    for r in res:
        long.append({"analysis": "primary", **{k: v for k, v in r.items() if not isinstance(v, np.ndarray)}})
    by_m = {r["method"]: r for r in res}         # exact method names (all distinct within one _methods call)
    ivw = _pick(res, IVW_PRIMARY)
    S.update({"ivw": ivw["estimate"], "ivw_se": ivw["se"], "ivw_lci": ivw["lower"], "ivw_uci": ivw["upper"],
              "ivw_p": ivw["p"], "Q": ivw["Q"], "Q_df": ivw["Q_df"], "Q_p": ivw["Q_p"], "I2": ivw["I2"]})
    fx = _pick(res, IVW_FIXED)
    S.update({"ivw_fixed": fx["estimate"], "ivw_fixed_se": fx["se"]})
    if "MR-Egger" in by_m:
        g = by_m["MR-Egger"]
        S.update({"egger": g["estimate"], "egger_se": g["se"], "egger_p": g["p"], "egger_intercept": g["intercept"],
                  "egger_intercept_se": g["intercept_se"], "egger_intercept_p": g["intercept_p"],
                  "I2_GX": g["I2_GX"]})
    if "Weighted median" in by_m:
        g = by_m["Weighted median"]
        S.update({"wmedian": g["estimate"], "wmedian_se": g["se"], "wmedian_p": g["p"]})
    if "Weighted mode" in by_m:
        g = by_m["Weighted mode"]
        S.update({"wmode": g["estimate"], "wmode_se": g["se"], "wmode_p": g["p"]})
    fs = M.f_statistics(hk["beta_x"].values, hk["se_x"].values, n=hk["n_x"].values)
    S.update({"F_mean": fs["F_mean"], "F_min": fs["F_min"], "n_F_lt_10": fs["n_F_lt_10"], "R2_x": fs.get("R2")})
    # consistency of direction across the four estimators
    if len(hk) >= 3:
        signs = np.sign([S["ivw"], S.get("egger", np.nan), S.get("wmedian", np.nan), S.get("wmode", np.nan)])
        S["consistent_direction"] = bool(np.all(signs == signs[0]) and signs[0] != 0)
    else:
        S["consistent_direction"] = None
    # Steiger (directionality) and Steiger-filtered sensitivity
    r2x = _r2(hk["beta_x"], hk["se_x"], hk["n_x"], hk["eaf_x"].where(hk["eaf_x"].notna(), hk["eaf_y"]).values, e["kind"])
    r2y = _r2(hk["beta_y"], hk["se_y"], hk["n_y"], hk["eaf_y"].where(hk["eaf_y"].notna(), hk["eaf_x"]).values, o["kind"])
    st = M.steiger(r2x, r2y, hk["n_x"].values, hk["n_y"].values)
    S.update({"steiger_r2_exposure": st["r2_exposure"], "steiger_r2_outcome": st["r2_outcome"],
              "steiger_correct_direction": st["correct_direction"], "steiger_p": st["steiger_p"]})
    keep = M.steiger_snp_filter(r2x, r2y, hk["n_x"].values, hk["n_y"].values)
    S["steiger_n_removed"] = int((~keep).sum())
    if keep.sum() >= 1 and (~keep).sum() > 0:
        rs = _methods(bx[keep], by[keep], sx[keep], sy[keep], job["n_boot"], job["n_boot_mode"], seed, full=False)
        for r in rs:
            long.append({"analysis": "steiger_filtered", **r})
        rsv = _pick(rs, IVW_PRIMARY)
        S.update({"ivw_steiger": rsv["estimate"], "ivw_steiger_se": rsv["se"], "ivw_steiger_p": rsv["p"],
                  "nsnp_steiger": int(keep.sum())})
    elif keep.sum() == len(keep):
        # nothing removed: copy the primary (random-effects) IVW summary, set from _pick(res, IVW_PRIMARY) above
        S.update({"ivw_steiger": S["ivw"], "ivw_steiger_se": S["ivw_se"], "ivw_steiger_p": S["ivw_p"],
                  "nsnp_steiger": int(keep.sum())})
    # +/- 10 Mb clumping sensitivity
    if job.get("inst10") is not None and len(job["inst10"]):
        h10, _ = _harm(job["inst10"], job["out"], job.get("drop_pal", False))
        if len(h10):
            b10x, s10x = h10["beta_x"].values / sc, h10["se_x"].values / sc
            r10 = _methods(b10x, h10["beta_y"].values, s10x, h10["se_y"].values, job["n_boot"], 0, seed, full=False)
            for r in r10:
                long.append({"analysis": "clump_10mb", **r})
            r10v = _pick(r10, IVW_PRIMARY)
            S.update({"ivw_10mb": r10v["estimate"], "ivw_10mb_se": r10v["se"], "ivw_10mb_p": r10v["p"],
                      "nsnp_10mb": len(h10)})
            eg = [r for r in r10 if r["method"] == "MR-Egger"]
            if eg:
                S.update({"egger_10mb": eg[0]["estimate"], "egger_10mb_intercept_p": eg[0]["intercept_p"]})
    for r in long:
        r.update({"family": job["family"], "exposure": e["id"], "outcome": o["id"]})
    snps = hk.assign(exposure=e["id"], outcome=o["id"], family=job["family"], steiger_keep=keep,
                     ratio=by / bx)[["family", "exposure", "outcome", "snp", "chr", "pos", "ea", "oa", "beta_x", "se_x",
                                     "p_x", "eaf_x", "beta_y", "se_y", "p_y", "eaf_y", "action", "matched_by",
                                     "steiger_keep", "ratio"]]
    return {"summary": S, "long": long, "snps": snps, "harm": acts}


def mvmr_worker(job: dict) -> dict:
    try:
        return _mvmr(job)
    except Exception as e:                                            # pragma: no cover
        return {"rows": [{"family": job["family"], "exposures": "+".join(x["spec"]["id"] for x in job["xs"]),
                          "outcome": job["o"]["id"], "error": f"{type(e).__name__}: {e}"}],
                "trace": traceback.format_exc()}


def _mvmr(job: dict) -> dict:
    """MVMR-IVW: union of the exposures' +/-1 Mb instruments, re-clumped jointly (+/-1 Mb, min p across exposures)."""
    xs, o = job["xs"], job["o"]
    base = with_ids(xs[0]["ext"])
    rs_c, pos_c = set(), set()
    for x in xs:
        inst = x["inst"]
        rs_c |= set(inst["snp"].dropna())
        ok = inst["pos"].notna()
        pos_c |= set(cpkey(inst.loc[ok, "chr"], inst.loc[ok, "pos"]))
    isc = base["snp"].isin(rs_c).fillna(False) | cpkey(base["chr"], base["pos"]).isin(pos_c).fillna(False)
    cand = base[isc.values].copy()
    names = [x["spec"]["id"] for x in xs]
    tab = cand[["snp", "chr", "pos", "ea", "oa", "beta", "se", "p", "eaf", "n"]].rename(
        columns={"beta": f"b_{names[0]}", "se": f"s_{names[0]}", "p": f"p_{names[0]}"})
    for x in xs[1:]:
        h, _ = _harm(cand, x["ext"])
        nm = x["spec"]["id"]
        tab = tab.merge(h[["snp", "beta_y", "se_y", "p_y"]].rename(
            columns={"beta_y": f"b_{nm}", "se_y": f"s_{nm}", "p_y": f"p_{nm}"}), on="snp", how="inner")
    ho, _ = _harm(cand, job["out"], job.get("drop_pal", False))
    tab = tab.merge(ho[["snp", "beta_y", "se_y"]].rename(columns={"beta_y": "by", "se_y": "sy"}), on="snp",
                    how="inner")
    tab = tab[tab["pos"].notna()].reset_index(drop=True)
    pmin = tab[[f"p_{nm}" for nm in names]].min(axis=1).values
    zmax = np.max(np.abs(np.column_stack([tab[f"b_{nm}"] / tab[f"s_{nm}"] for nm in names])), axis=1) if len(tab) else []
    keep = clump_fast(tab["chr"], tab["pos"], pmin, zmax, WINDOW_KB) if len(tab) else np.zeros(0, bool)
    tab = tab[keep & (pmin <= P_THRESH)].reset_index(drop=True)
    J, K = len(tab), len(names)
    base_row = {"family": job["family"], "exposures": "+".join(names), "outcome": o["id"], "nsnp": J}
    if J <= K + 1:
        return {"rows": [{**base_row, "error": f"too few SNPs ({J})"}]}
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
    return {"rows": rows}


# ================================================================================================================
# Chami 2025 uncoupled-adiposity variants
# ================================================================================================================
def load_chami(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Supplementary Table 3", header=None, engine="openpyxl")
    hdr = [str(x).strip() for x in raw.iloc[2, :11]]
    want = ["SNP", "Variant ID", "Chr", "Position", "Reference Allele", "Effect Allele",
            "Effect allele frequency (%)", "Beta", "SE", "Pvalue", "N"]
    if hdr != want:
        raise RuntimeError(f"Chami ST3 header changed: {hdr}")
    grp = str(raw.iloc[1, 7]).strip()
    if grp != "BFP":
        raise RuntimeError(f"Chami ST3: first trait block is {grp!r}, expected BFP")
    d = raw.iloc[3:, :11].copy()
    d.columns = want
    d = d[d["SNP"].notna() & pd.to_numeric(d["Position"], errors="coerce").notna()]
    out = pd.DataFrame({
        "snp": d["SNP"].astype("string").str.strip(),
        "chr": d["Chr"].astype(int).astype("string"),
        "pos": pd.to_numeric(d["Position"]).astype("Int64"),
        "ea": d["Effect Allele"].astype("string").str.strip().str.upper(),
        "oa": d["Reference Allele"].astype("string").str.strip().str.upper(),
        "beta": pd.to_numeric(d["Beta"], errors="coerce").astype(float),
        "se": pd.to_numeric(d["SE"], errors="coerce").astype(float),
        "p": pd.to_numeric(d["Pvalue"], errors="coerce").astype(float),
        "eaf": pd.to_numeric(d["Effect allele frequency (%)"], errors="coerce").astype(float),
        "n": pd.to_numeric(d["N"], errors="coerce").astype(float)}).reset_index(drop=True)
    out["snp"] = out["snp"].where(out["snp"].str.match(r"^rs\d+$").fillna(False), pd.NA)
    out["pos38"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["se_source"] = "se"
    return out


# ================================================================================================================
# Simulation (wiring test with planted answers)
# ================================================================================================================
SIM_TRAITS = ["sbp", "bmi", "whr", "tg", "osa", "fg", "hba1c", "t2d", "bfp", "asi", "cimt", "plaque", "crae", "crve",
              "avr", "tort", "dens", "pad", "cad", "stroke", "istroke", "hf", "life"]
SIM_LOCI = dict(sbp=(110, 0.35), bmi=(110, 0.02), whr=(60, 0.02), tg=(80, 0.025), osa=(40, 0.035), fg=(40, 0.015),
                hba1c=(40, 0.015), t2d=(50, 0.06), bfp=(50, 0.015), asi=(18, 0.05), cimt=(15, 0.006),
                plaque=(10, 0.08), crae=(25, 0.04), crve=(25, 0.04), avr=(20, 0.04), tort=(20, 0.04),
                dens=(20, 0.04), pad=(10, 0.05), cad=(40, 0.04), stroke=(10, 0.03), istroke=(10, 0.03),
                hf=(10, 0.03), life=(10, 0.02))
SIM_EDGES = {("sbp", "bmi"): 1.5, ("asi", "sbp"): 0.03, ("cimt", "sbp"): 0.002, ("cimt", "bmi"): 0.01,
             ("cimt", "bfp"): -0.01, ("crae", "tg"): -0.05, ("avr", "whr"): -0.03, ("pad", "osa"): 0.1,
             ("crve", "fg"): 0.05, ("plaque", "t2d"): 0.1, ("tort", "hba1c"): 0.04, ("cad", "asi"): 0.05,
             ("stroke", "cimt"): 2.0, ("istroke", "plaque"): 0.3, ("hf", "crae"): -0.1, ("life", "crve"): 0.05}
SIM_CURATED = [("sbp", "asi"), ("bmi", "asi"), ("bmi", "cimt"), ("tg", "crae"), ("whradjbmi", "avr"),
               ("osa", "pad_mvp"), ("osa_kurn", "pad_mvp"), ("osa_kurn_bmiadj", "pad_sakaue"),
               ("osa_campos", "pad_sakaue"), ("fg", "crve"), ("t2d", "plaque"), ("hba1c", "tort"), ("tg", "asi"),
               ("sbp", "cad"), ("asi", "cad"), ("cimt", "stroke"), ("plaque", "istroke"), ("crae", "hf"),
               ("crve", "life"), ("sbp_bbj", "asi"), ("bmi_bbj", "cimt"), ("tg_eas", "crae"),
               ("chami_bfp", "cimt")]
SIM_CURATED_MV = [(("bmi", "sbp"), "cimt"), (("bmi", "sbp"), "asi"), (("asi", "sbp"), "cad")]


def sim_truth():
    idx = {t: i for i, t in enumerate(SIM_TRAITS)}
    B = np.zeros((len(SIM_TRAITS), len(SIM_TRAITS)))
    for (t, s), v in SIM_EDGES.items():
        B[idx[t], idx[s]] = v
    T = np.linalg.inv(np.eye(len(B)) - B)
    return idx, B, T


def planted_uni(e: dict, o: dict):
    idx, B, T = sim_truth()
    s, t = idx[e["sim"]["trait"]], idx[o["sim"]["trait"]]
    for u in range(len(T)):
        if u != s and abs(T[s, u]) > 1e-12 and abs(T[t, u] - T[t, s] * T[s, u]) > 1e-9:
            return np.nan, False
    return T[t, s] * o["sim"]["c"] / e["sim"]["c"] * float(e.get("scale", 1.0)), True


def planted_mv(es: list, o: dict):
    idx, B, T = sim_truth()
    S = [idx[e["sim"]["trait"]] for e in es]
    t = idx[o["sim"]["trait"]]
    th = []
    for k, s in enumerate(S):
        Bk = B.copy()
        for j, s2 in enumerate(S):
            if j != k:
                Bk[s2, :] = 0.0
        Tk = np.linalg.inv(np.eye(len(B)) - Bk)
        th.append(Tk[t, s])
    valid = True
    for u in range(len(T)):
        if any(abs(T[s, u]) > 1e-12 for s in S):
            if abs(T[t, u] - sum(th[k] * T[S[k], u] for k in range(len(S)))) > 1e-9:
                valid = False
    return [th[k] * o["sim"]["c"] / es[k]["sim"]["c"] * float(es[k].get("scale", 1.0)) for k in range(len(S))], valid


def simulate_inputs(simdir: Path, seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Write synthetic GWAS files with the real headers and formats. Returns (paths, wiring table, variant truth)."""
    rng = np.random.default_rng(seed)
    idx, B, T = sim_truth()
    n_chr, per_chr, step = 22, 2000, 50_000
    chr_ = np.repeat(np.arange(1, n_chr + 1), per_chr)
    pos37 = np.tile(np.arange(1, per_chr + 1) * step, n_chr) + rng.integers(-2000, 2000, n_chr * per_chr)
    Mv = len(chr_)
    pos38 = pos37 + 100_000 + 37 * chr_
    rsid = np.array([f"rs{1_000_000 + i}" for i in range(Mv)])
    pairs = np.array([("A", "G"), ("C", "T"), ("A", "C"), ("G", "T"), ("A", "T"), ("C", "G"), ("G", "A"),
                      ("T", "C")])
    pk = rng.choice(len(pairs), Mv, p=[.17, .17, .12, .12, .06, .06, .15, .15])
    a1, a2 = pairs[pk, 0].astype(object), pairs[pk, 1].astype(object)
    indel = rng.random(Mv) < 0.03
    a2[indel] = a1[indel] + "T"
    f = rng.uniform(0.03, 0.97, Mv)
    # loci: one lead per 2.2 Mb slot; LD-like neighbours +/- 3 variants (r = 0.8^k)
    slots = [(c, 1_500_000 + k * 2_200_000) for c in range(1, n_chr + 1) for k in range(45)]
    order = rng.permutation(len(slots))
    G = np.zeros((Mv, len(SIM_TRAITS)))
    used = 0
    lead_of = {}
    for tr, (nl, sd) in SIM_LOCI.items():
        for _ in range(nl):
            c, centre = slots[order[used]]
            used += 1
            lead = (c - 1) * per_chr + int(round(centre / step)) - 1
            g = rng.choice([-1, 1]) * (abs(rng.normal(0, sd)) + 0.25 * sd)
            lead_of[lead] = tr
            for k in range(-3, 4):
                if 0 <= lead + k < Mv and chr_[lead + k] == c:
                    G[lead + k, idx[tr]] += g * 0.8 ** abs(k)
    Gam = G @ T.T                                        # marginal effects on every latent trait
    truth = pd.DataFrame({"snp": rsid, "chr": chr_.astype(str), "pos37": pos37, "pos38": pos38, "a1": a1, "a2": a2,
                          "f": f})
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    paths, wiring = {}, []
    for spec in SPECS:
        real = GWAS / spec["path"]
        hdr = read_header(real, spec["sep"])
        miss = [v for v in spec["cols"].values() if v not in hdr]
        wiring.append({"id": spec["id"], "real_file": spec["path"], "header_columns": len(hdr),
                       "mapped_columns": len(set(spec["cols"].values())), "missing_in_real_header": ",".join(miss),
                       "ok": not miss})
        if miss:
            raise RuntimeError(f"wiring: {spec['id']} columns missing in real header: {miss}")
        sm = spec["sim"]
        present = rng.random(Mv) < 0.97
        ff = np.clip(f + (rng.normal(0, 0.12, Mv) if sm.get("eas") else rng.normal(0, 0.005, Mv)), 0.01, 0.99)
        se = sm["s"] / np.sqrt(2 * ff * (1 - ff) * sm["N"])
        b = sm["c"] * Gam[:, idx[sm["trait"]]] + rng.standard_normal(Mv) * se
        ea, oa, eaf, bb = a1.copy(), a2.copy(), ff.copy(), b.copy()
        sw = rng.random(Mv) < 0.5
        ea[sw], oa[sw], eaf[sw], bb[sw] = a2[sw], a1[sw], 1 - ff[sw], -b[sw]
        snv = ~indel
        pal = snv & np.array([comp.get(x) == y for x, y in zip(a1, a2)])
        flip = snv & (((~pal) & (rng.random(Mv) < 0.05)) | (pal & (rng.random(Mv) < 0.3)))
        ea[flip] = [comp[x] for x in ea[flip]]
        oa[flip] = [comp[x] for x in oa[flip]]
        z = bb / se
        p = 2 * stats.norm.sf(np.abs(z))
        sp = set(spec.get("special", ()))
        cols = spec["cols"]
        df = pd.DataFrame({h: ["NA"] * Mv for h in hdr})
        df[cols["chr"]] = chr_
        df[cols["pos"]] = pos38 if spec["build"] == "38" else pos37
        if "pos_str" in sp:
            df[cols["pos"]] = [f"{x:>9d}" for x in pos37]
        if "snp" in cols:
            df[cols["snp"]] = [f"{r}:{x}:{y}" for r, x, y in zip(rsid, a1, a2)] if "pulit_snp" in sp else rsid
        lower = "lower_alleles" in sp
        wea, woa = (oa, ea) if "swap_alleles" in sp else (ea, oa)       # reproduce the Ortin Vela label swap
        df[cols["ea"]] = [x.lower() for x in wea] if lower else wea
        df[cols["oa"]] = [x.lower() for x in woa] if lower else woa
        if "eaf" in cols:
            df[cols["eaf"]] = np.round(eaf, 4) if sm.get("eaf", True) else "NA"
        df[cols["p"]] = [f"{x:.4g}" for x in p]
        if "or" in cols:
            df[cols["or"]] = [f"{x:.4g}" for x in np.exp(bb)]
            df[cols["or_lci"]] = [f"{x:.4g}" for x in np.exp(bb - 1.959964 * se)]
            df[cols["or_uci"]] = [f"{x:.4g}" for x in np.exp(bb + 1.959964 * se)]
            if "standard_error" in hdr:
                df["standard_error"] = "#NA"
        else:
            df[cols["beta"]] = np.round(-bb if "flip_sign" in sp else bb, 7)
            df[cols["se"]] = np.round(se, 7)
        if "n" in cols:
            df[cols["n"]] = sm["N"]
        df = df[present].reset_index(drop=True)
        out = simdir / spec["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, sep=spec["sep"], index=False, compression="gzip" if str(out).endswith(".gz") else None)
        paths[spec["id"]] = out
    # Chami-like workbook (ST3 layout: title, trait-block header, column names, data)
    import openpyxl
    leads = [v for v, tr in lead_of.items() if tr == "bfp"]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Supplementary Table 3"
    ws.append(["Supplementary Table 3. (simulated)"])
    ws.append(["Lead SNPs", None, None, None, None, None, None, "BFP", None, None, None, "BMI"])
    ws.append(["SNP", "Variant ID", "Chr", "Position", "Reference Allele", "Effect Allele",
               "Effect allele frequency (%)", "Beta", "SE", "Pvalue", "N", "Beta"])
    sm = CHAMI["sim"]
    for v in leads:
        se = sm["s"] / np.sqrt(2 * f[v] * (1 - f[v]) * sm["N"])
        b = Gam[v, idx["bfp"]] + rng.standard_normal() * se
        ws.append([rsid[v], f"{chr_[v]}:{pos37[v]}:{a2[v]}:{a1[v]}", int(chr_[v]), int(pos37[v]), a2[v], a1[v],
                   round(float(f[v]), 4), round(float(b), 5), round(float(se), 5),
                   float(2 * stats.norm.sf(abs(b / se))), sm["N"], 0.0])
    out = simdir / CHAMI["path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    paths[CHAMI["id"]] = out
    return paths, pd.DataFrame(wiring), truth


# ================================================================================================================
# Main pipeline
# ================================================================================================================
def implied_sd(d: pd.DataFrame) -> float:
    ok = d["eaf"].between(0.05, 0.95) & np.isfinite(d["se"]) & (d["n"] > 0)
    if ok.sum() < 20:
        return np.nan
    x = d.loc[ok]
    return float(np.median(x["se"] * np.sqrt(2 * x["eaf"] * (1 - x["eaf"]) * x["n"])))


def eaf_qc(ref: pd.DataFrame, sets: dict) -> pd.DataFrame:
    """EAF concordance of every file with the reference (Keaton SBP) after allele alignment by label only
    (palindromic SNPs excluded). A negative correlation means mislabelled effect/other alleles."""
    rows = []
    r = ref.dropna(subset=["eaf"])
    pal = M.is_palindromic(r["ea"], r["oa"])
    r = r[~pal]
    for k, d in sets.items():
        if d is None or len(d) == 0:
            continue
        x = with_ids(d)
        h = M.harmonise(r, x, match_on="auto")
        if len(h) == 0:
            rows.append({"id": k, "n": 0})
            continue
        h = h[h["keep"] & h["eaf_x"].notna() & h["eaf_y"].notna()]
        cor = float(np.corrcoef(h["eaf_x"], h["eaf_y"])[0, 1]) if len(h) > 10 else np.nan
        rows.append({"id": k, "ancestry": SPEC[k]["ancestry"], "n": len(h), "eaf_corr": cor,
                     "mean_abs_diff": float((h["eaf_x"] - h["eaf_y"]).abs().mean()) if len(h) else np.nan,
                     "flag": bool(np.isfinite(cor) and cor < 0.5)})
    return pd.DataFrame(rows)


def manifest_sha() -> dict:
    if not MANIFEST.exists():
        return {}
    m = pd.read_csv(MANIFEST, sep="\t", dtype=str)
    m = m[m["status"] == "OK"]
    return dict(zip(m["local_path"], m["sha256"]))


def eur(spec: dict) -> bool:
    return str(spec.get("ancestry", "")).startswith("EUR") and "+" not in str(spec.get("ancestry", ""))


def run(args, mode: str, outdir: Path):
    rep = C.Report(TASK, "F1 Stage 0 (i): two-sample MR panel - risk factors -> vascular beds -> outcomes", mode, args)
    t_start = time.time()
    scan_stats, notes = [], []
    simdir = None
    if mode == "simulate":
        simdir = outdir / "_sim_inputs"
        if simdir.exists():
            shutil.rmtree(simdir)
        log("simulate: writing synthetic GWAS files with the real headers")
        paths, wiring, truth = simulate_inputs(simdir, args.seed)
        n_boot, n_boot_mode = 300, 100
    else:
        paths = {s["id"]: GWAS / s["path"] for s in SPECS + [CHAMI]}
        missing = [k for k, v in paths.items() if not v.exists()]
        for k in missing:
            notes.append(f"Input missing, skipped: {k} ({paths[k]})")
        wiring, truth = None, None
        n_boot, n_boot_mode = args.n_boot, min(args.n_boot, 1000)
    avail = {k for k, v in paths.items() if v.exists()}
    scanner = Scanner(paths, outdir / "_cache", args.workers, not args.no_cache, scan_stats)

    # ---------------- pass 1: genome-wide significant rows of every exposure file ----------------
    p1_ids = [i for i in PASS1 if i in avail]
    cand = scanner.run([(i, "sig", {"p_thresh": P_THRESH}) for i in p1_ids], "pass1")
    inst1, inst10, sel_info = {}, {}, {}
    for i in p1_ids:
        inst1[i], inst10[i], sel_info[i] = select_instruments(cand[i], SPEC[i])
    if CHAMI["id"] in avail:
        ch = load_chami(paths[CHAMI["id"]])
        inst1[CHAMI["id"]], inst10[CHAMI["id"]] = ch, None
        dist = clump_fast(ch["chr"], ch["pos"], ch["p"], ch["beta"] / ch["se"], WINDOW_KB)
        sel_info[CHAMI["id"]] = {"p_le_5e8_rows": int((ch["p"] <= P_THRESH).sum()), "candidates": len(ch),
                                 "instruments_1mb": len(ch), "instruments_10mb": np.nan,
                                 "note": f"all {len(ch)} published lead variants used (not re-clumped); "
                                         f"{int(len(ch) - dist.sum())} lie within 1 Mb of a stronger one"}
    log("instruments (+/-1 Mb): " + ", ".join(f"{k}={len(v)}" for k, v in inst1.items()))

    # ---------------- variant maps ----------------
    xw_rows = []
    for i in [i for i in p1_ids if SPEC[i].get("xwalk_via")]:
        via = SPEC[i]["xwalk_via"]
        if via not in avail:
            notes.append(f"{i}: cross-walk source {via} missing; instruments cannot be matched")
            continue
        both = pd.concat([inst1[i], inst10[i]], ignore_index=True)
        prs = set(cpkey(both["chr"], both["pos38"]))
        ref = scanner.run([(via, "keys", {"pos": sorted({float(x.split(":")[1]) for x in prs}), "pairs": sorted(prs)})],
                          f"xwalk_{i}")[via]
        for d in (inst1[i], inst10[i]):
            s, nm = crosswalk_38(d, ref)
            d["snp"] = s
        xw_rows.append({"exposure": i, "via": via, "instruments_1mb": len(inst1[i]),
                        "rsid_found_1mb": int(inst1[i]["snp"].notna().sum())})
    need_rs, need_pairs = set(), set()
    for i, d in list(inst1.items()) + list(inst10.items()):
        if d is None:
            continue
        need_rs |= set(d.loc[d["pos"].isna() & d["snp"].notna(), "snp"])
        m = d["snp"].isna() & d["pos"].notna()
        need_pairs |= set(cpkey(d.loc[m, "chr"], d.loc[m, "pos"]))
    vmap = VariantMap()
    for i, d in list(inst1.items()) + list(inst10.items()):
        if d is not None and SPEC[i]["build"] == "37":
            vmap.add(d)
    ref_ids = [r for r in REF_MAP if r in avail]
    if need_rs or need_pairs:
        refs = scanner.run([(r, "keys", {"rs": sorted(need_rs), "pos": sorted({float(x.split(":")[1]) for x in need_pairs}),
                                         "pairs": sorted(need_pairs)}) for r in ref_ids], "refmap")
        for r in ref_ids:
            vmap.add(refs[r])
    map_rows = []
    for i in inst1:
        for lab, d in (("1mb", inst1[i]), ("10mb", inst10[i])):
            if d is None:
                continue
            n_rs = vmap.fill_rsid(d)
            n_pos = vmap.fill_pos37(d)
            map_rows.append({"exposure": i, "set": lab, "n": len(d), "rsid_filled": n_rs, "pos37_filled": n_pos,
                             "with_rsid": int(d["snp"].notna().sum()), "with_pos37": int(d["pos"].notna().sum())})

    # ---------------- pass 2: union of instrument variants in every outcome / MVMR file ----------------
    u_rs, u_pairs = set(), set()
    for d in list(inst1.values()) + [x for x in inst10.values() if x is not None]:
        u_rs |= set(d["snp"].dropna())
        ok = d["pos"].notna()
        u_pairs |= set(cpkey(d.loc[ok, "chr"], d.loc[ok, "pos"]))
    u_pos = sorted({float(x.split(":")[1]) for x in u_pairs})
    log(f"union of instrument variants: {len(u_rs):,} rsIDs, {len(u_pairs):,} GRCh37 positions")
    p2_ids = [i for i in PASS2 if i in avail]
    jobs = []
    for i in p2_ids:
        if SPEC[i]["build"] == "38":
            jobs.append((i, "keys", {"rs": sorted(u_rs)}))
        else:
            jobs.append((i, "keys", {"rs": sorted(u_rs), "pos": u_pos, "pairs": sorted(u_pairs)}))
    ext = scanner.run(jobs, "pass2")
    for i, d in ext.items():
        if SPEC[i]["build"] == "37":
            vmap.add(d)
    for i, d in ext.items():
        if SPEC[i]["build"] == "38":
            vmap.fill_pos37(d)

    # ---------------- EAF concordance QC (allele-label check) ----------------
    qc_sets = {k: v for k, v in ext.items() if k != "sbp"}
    for k in RISK_SENS + BBJ + [CHAMI["id"]]:
        if k in inst1:
            qc_sets[k] = inst1[k]
    eafqc = eaf_qc(ext["sbp"], qc_sets) if "sbp" in ext else pd.DataFrame()
    for _, r in eafqc.iterrows():
        if r.get("flag"):
            notes.append(f"EAF concordance with the SBP reference is low for {r['id']} (r = {r['eaf_corr']:.2f}): "
                         "check allele labels")

    # ---------------- GWAS registry ----------------
    msha = manifest_sha()
    reg = []
    hashes = {}
    if mode == "real" and not args.no_hash:
        hfile = outdir / "_cache" / "sha256.json"
        hc = json.loads(hfile.read_text()) if hfile.exists() else {}
        todo = []
        for k in sorted(avail):
            st = paths[k].stat()
            key = f"{paths[k]}|{st.st_size}|{int(st.st_mtime)}"
            if key in hc:
                hashes[k] = hc[key]
            else:
                todo.append((k, key))
        if todo:
            log(f"SHA-256 of {len(todo)} input files")
            res = run_pool(hash_worker, [str(paths[k]) for k, _ in todo], args.workers)
            for (k, key), (_, h) in zip(todo, res):
                hashes[k] = hc[key] = h
            hfile.write_text(json.dumps(hc, indent=1))
    for s in SPECS + [CHAMI]:
        i = s["id"]
        if i not in avail:
            continue
        st = paths[i].stat()
        rel = str(paths[i].relative_to(HUB)) if mode == "real" else str(paths[i].relative_to(outdir))
        sha = hashes.get(i)
        reg.append({"id": i, "label": s["label"], "group": s["group"], "source": s["source"], "file": rel,
                    "build": s["build"], "ancestry": s["ancestry"], "N": s["n_default"], "kind": s["kind"],
                    "units": s["units"], "UKB_in_sample": s.get("ukb"),
                    "implied_sd": implied_sd(ext[i]) if (i in ext and s["kind"] == "cont") else np.nan,
                    "rows_scanned": next((r["rows_scanned"] for r in scan_stats if r["id"] == i), np.nan),
                    **{k: v for k, v in sel_info.get(i, {}).items()},
                    "bytes": st.st_size, "sha256": sha,
                    "manifest_sha256_match": (None if sha is None else msha.get(rel) == sha)})
        rep.inputs.append({"id": i, "path": rel, "bytes": st.st_size,
                           "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(), "sha256": sha})
    registry = pd.DataFrame(reg)

    # ---------------- univariable MR jobs ----------------
    def ujob(eid, oid, fam):
        e, o = SPEC[eid], SPEC[oid]
        inst = with_ids(inst1[eid])
        i10 = with_ids(inst10[eid]) if inst10.get(eid) is not None else None
        return dict(e={k: e[k] for k in ("id", "label", "kind", "unit_label", "units", "ukb") if k in e} |
                    {"scale": e.get("scale", 1.0)},
                    o={k: o[k] for k in ("id", "label", "kind", "unit_label", "units", "ukb") if k in o},
                    inst=inst, inst10=i10, out=ext[oid], family=fam, seed=args.seed, n_boot=n_boot,
                    n_boot_mode=n_boot_mode, drop_pal=not (eur(e) and eur(o)))
    ujobs = []
    for eid in RISK:
        for oid in BED_OUT:
            ujobs.append(("A_risk_to_beds", eid, oid))
        for oid in CLIN_PC:
            ujobs.append(("B_risk_to_outcomes_posctrl", eid, oid))
    for eid in RISK_SENS:
        for oid in BED_OUT + CLIN_PC:
            ujobs.append(("A2_osa_alternatives", eid, oid))
    for eid in BBJ:
        for oid in BED_OUT + CLIN_PC:
            ujobs.append(("D_bbj_eas", eid, oid))
    for oid in BED_OUT + CLIN_PC:
        ujobs.append(("C_chami_uncoupled", CHAMI["id"], oid))
    beds_ok = [b for b in BEDS if b in inst1 and len(inst1[b]) >= 3]
    for b in BEDS:
        if b in inst1 and len(inst1[b]) < 3:
            notes.append(f"bed {b} has {len(inst1[b])} instruments (< 3): not used as an exposure")
    for eid in beds_ok:
        for oid in CLIN:
            ujobs.append(("E_beds_to_outcomes", eid, oid))
    ujobs = [(f, e, o) for f, e, o in ujobs if e in inst1 and o in ext and len(inst1[e])]
    log(f"univariable MR: {len(ujobs)} pairs on {args.mr_workers} workers")
    t_mr = time.time()
    res = run_pool(mr_pair_worker, [ujob(e, o, f) for f, e, o in ujobs], args.mr_workers)
    log(f"univariable MR done in {time.time() - t_mr:.1f}s")
    summ = pd.DataFrame([r["summary"] for r in res])
    longd = pd.DataFrame([x for r in res for x in r["long"]])
    snps = pd.concat([r["snps"] for r in res if r.get("snps") is not None], ignore_index=True)
    harm = pd.DataFrame([{"family": r["summary"]["family"], "exposure": r["summary"]["exposure"],
                          "outcome": r["summary"]["outcome"], **r["harm"]} for r in res]).fillna(0)
    for r in res:
        if r.get("trace"):
            notes.append(f"MR error {r['summary']['exposure']}->{r['summary']['outcome']}: {r['summary'].get('error')}")
    # FDR within family; evidence class
    summ["ivw_q"] = np.nan
    for fam, g in summ.groupby("family"):
        summ.loc[g.index, "ivw_q"] = bh(g["ivw_p"].values)
    sd_map = dict(zip(registry["id"], registry["implied_sd"]))
    adv = {s["id"]: s.get("adverse", 0) for s in SPECS}
    summ["ivw_adverse_oriented"] = summ["ivw"] * summ["outcome"].map(adv).replace(0, np.nan)
    summ["outcome_implied_sd"] = summ["outcome"].map(sd_map)
    summ["ivw_per_sd_outcome"] = summ["ivw"] / summ["outcome_implied_sd"]
    summ["evidence"] = summ.apply(classify, axis=1)

    # ---------------- MVMR jobs ----------------
    def mjob(xids, oid, fam):
        xs = [dict(spec={k: SPEC[x][k] for k in ("id", "label", "unit_label") if k in SPEC[x]} |
                   {"scale": SPEC[x].get("scale", 1.0)}, inst=inst1[x], ext=ext[x]) for x in xids]
        return dict(xs=xs, o={"id": oid}, out=ext[oid], family=fam,
                    drop_pal=not (all(eur(SPEC[x]) for x in xids) and eur(SPEC[oid])))
    mjobs = []
    if "sbp" in ext:
        for x in MVMR_X:
            if x in ext and x in inst1:
                for oid in BED_OUT:
                    if oid in ext:
                        mjobs.append(((x, "sbp"), oid, "MV_risk_plus_sbp_to_beds"))
        for b in beds_ok:
            if b in ext:
                for oid in CLIN:
                    if oid in ext:
                        mjobs.append(((b, "sbp"), oid, "MV_bed_plus_sbp_to_outcomes"))
    log(f"MVMR: {len(mjobs)} fits")
    mres = run_pool(mvmr_worker, [mjob(x, o, f) for x, o, f in mjobs], args.mr_workers)
    mv = pd.DataFrame([r for m in mres for r in m["rows"]])
    if "p" in mv.columns:
        mv["q_within_family_term"] = np.nan
        for (fam, term), g in mv.dropna(subset=["p"]).groupby(["family", "term"]):
            mv.loc[g.index, "q_within_family_term"] = bh(g["p"].values)

    # ---------------- simulate: planted checks ----------------
    planted = None
    sim_tables = {}
    if mode == "simulate":
        planted, sim_tables = sim_checks(summ, mv, inst1, cand, truth, xw_rows, map_rows)

    # ---------------- report ----------------
    write_report(rep, mode, registry, summ, longd, snps, harm, mv, scan_stats, xw_rows, map_rows, notes,
                 wiring, sim_tables, t_start, args, eafqc, outdir)
    snps.to_csv(outdir / "harmonised_snps.csv.gz", index=False, compression="gzip")
    inst_tab = pd.concat([d.assign(exposure=k, F=(d["beta"] / d["se"]) ** 2) for k, d in inst1.items()],
                         ignore_index=True)
    rep.tables["instruments_1mb"] = inst_tab[["exposure", "snp", "chr", "pos", "pos38", "ea", "oa", "beta", "se", "p",
                                              "eaf", "n", "F"]]
    C.finish(rep, outdir, __file__, planted)
    if simdir is not None and not args.keep_sim:
        shutil.rmtree(simdir, ignore_errors=True)
        shutil.rmtree(outdir / "_cache", ignore_errors=True)


def classify(r) -> str:
    if not np.isfinite(r.get("ivw_p", np.nan)):
        return "no estimate"
    if r.get("nsnp", 0) < 3:
        return "insufficient (<3 SNPs)" if r["ivw_p"] < 0.05 else "no evidence (<3 SNPs)"
    cons = r.get("consistent_direction")
    if r["ivw_p"] >= 0.05:
        return "no evidence (p>=0.05)"
    if not cons:
        return "inconsistent direction"
    robust_support = (r.get("wmedian_p", 1) < 0.05) or (r.get("wmode_p", 1) < 0.05)
    icpt_ok = not (r.get("egger_intercept_p", 1) < 0.05)
    if r.get("ivw_q", 1) < 0.05:
        return "robust" if (robust_support and icpt_ok) else "supported (FDR) - sensitivity weaker"
    return "suggestive (p<0.05, consistent)"


def sim_checks(summ, mv, inst1, cand, truth, xw_rows, map_rows):
    tabs = {}
    # planted vs estimated, all valid pairs
    rows = []
    for _, r in summ.iterrows():
        e, o = SPEC[r["exposure"]], SPEC[r["outcome"]]
        th, valid = planted_uni(e, o)
        if not valid or not np.isfinite(r.get("ivw", np.nan)):
            continue
        rows.append({"family": r["family"], "exposure": r["exposure"], "outcome": r["outcome"], "nsnp": r["nsnp"],
                     "planted": th, "ivw": r["ivw"], "ivw_se": r["ivw_se"],
                     "z": (r["ivw"] - th) / r["ivw_se"], "within_3se": abs(r["ivw"] - th) <= 3 * r["ivw_se"]})
    allp = pd.DataFrame(rows)
    tabs["sim_planted_all_pairs"] = allp
    checks = []
    for e, o in SIM_CURATED:
        m = summ[(summ["exposure"] == e) & (summ["outcome"] == o)]
        if len(m):
            th, _ = planted_uni(SPEC[e], SPEC[o])
            checks.append((f"IVW {e} -> {o} ({SPEC[e].get('unit_label', '')})", th, float(m["ivw"].iloc[0]),
                           float(m["ivw_se"].iloc[0])))
    for xs, o in SIM_CURATED_MV:
        th, _ = planted_mv([SPEC[x] for x in xs], SPEC[o])
        for k, x in enumerate(xs):
            m = mv[(mv["exposures"] == "+".join(xs)) & (mv["outcome"] == o) & (mv["term"] == x)]
            if len(m):
                checks.append((f"MVMR {x} | {'+'.join(xs)} -> {o}", th[k], float(m["estimate"].iloc[0]),
                               float(m["se"].iloc[0])))
    planted = C.planted_rows(checks)
    # clumping equivalence with lib_mr.clump_distance
    eq = []
    for i, d in cand.items():
        native = d["pos38"] if SPEC[i]["build"] == "38" else d["pos"]
        c2, _ = candidate_filter(d, native)
        if len(c2) == 0:
            continue
        nat = (c2["pos38"] if SPEC[i]["build"] == "38" else c2["pos"]).astype(float)
        ref = M.clump_distance(pd.DataFrame({"chr": c2["chr"].astype(str), "pos": nat, "p": c2["p"]}),
                               p_thresh=P_THRESH, window_kb=WINDOW_KB)
        mine = clump_fast(c2["chr"], nat, c2["p"], np.zeros(len(c2)), WINDOW_KB)    # no |z| tie-break here
        eq.append({"exposure": i, "candidates": len(c2), "lib_mr_kept": int(ref.sum()), "fast_kept": int(mine.sum()),
                   "identical": bool((ref.values == mine).all())})
    tabs["sim_clumping_equivalence"] = pd.DataFrame(eq)
    # GRCh38 handling: cross-walked rsIDs and mapped GRCh37 positions vs truth
    tr = truth.set_index("snp")
    mp = []
    for i in inst1:
        if SPEC.get(i, {}).get("build") != "38":
            continue
        d = inst1[i]
        ok = d["snp"].notna()
        got = d[ok]
        right_rs = (tr.reindex(got["snp"])["pos38"].values == got["pos38"].astype(float).values)
        right_pos = (tr.reindex(got["snp"])["pos37"].values == got["pos"].astype(float).values)
        mp.append({"exposure": i, "instruments": len(d), "with_rsid": int(ok.sum()),
                   "rsid_consistent_with_grch38_pos": int(right_rs.sum()), "pos37_correct": int(right_pos.sum())})
    tabs["sim_grch38_mapping"] = pd.DataFrame(mp)
    return planted, tabs


def fmt_ci(e, lo, hi, nd=3):
    if not np.isfinite(e):
        return ""
    return f"{e:.{nd}g} [{lo:.{nd}g}, {hi:.{nd}g}]"


def write_report(rep, mode, registry, summ, longd, snps, harm, mv, scan_stats, xw_rows, map_rows, notes, wiring,
                 sim_tables, t_start, args, eafqc=None, rep_outdir: Path | None = None):
    rep.h("Results summary")
    hw = rep_outdir / "results_summary_handwritten.md" if mode == "real" else None
    if hw is not None and hw.exists():
        rep.p(hw.read_text(encoding="utf-8").strip())
    else:
        rep.p("_(Auto-generated tables below. A hand-written interpretation is added after inspection of the real "
              "run, in `results_summary_handwritten.md`, and is inserted here on every rerun.)_")
    rep.h("Methods (as run)")
    rep.bullets([
        "Two-sample MR on open GWAS summary statistics (registry below; GRCh37; GRCh38 files matched by rsID).",
        f"Instruments: p <= {P_THRESH:g}; autosomes; MHC (chr6:25-34 Mb) excluded; MAF >= {MAF_MIN} where EAF "
        "is reported; greedy distance clumping, index = smallest p, window +/- 1 Mb (sensitivity +/- 10 Mb). "
        "**No LD reference panel was available**, so residual LD between index variants > 1 Mb apart (long-range "
        "LD regions) cannot be excluded; this is a limitation (pending hub request for a shared 1000G EUR panel).",
        "Harmonisation (lib_mr.harmonise): rsID, then GRCh37 chr:pos; direct, swapped and strand-flipped alleles; "
        "palindromic variants kept only when both EAFs lie outside [0.42, 0.58]; incompatible alleles dropped. When "
        "either GWAS is not European-only (BBJ/EAS exposures, Kurniansyah multi-ancestry OSA, BBJ+UKB+FinnGen PAD), "
        "all palindromic variants are dropped.",
        "Allele-label QC: EAF of every file vs Keaton SBP after alignment. The five Ortin Vela retinal files have "
        "their effect_allele / other_allele labels swapped relative to beta and EAF (EAF correlation -0.66 as "
        "labelled); they are read with the labels swapped back. With the labels as deposited, SBP would appear to "
        "widen retinal arterioles, contrary to established epidemiology; after the correction it narrows them.",
        "GRCh38 inputs: MVP sleep apnoea and PAD carry rsIDs (GRCh37 positions taken from GRCh37 reference GWAS by "
        "rsID); Kurniansyah OSA (no rsID) cross-walked to rsIDs through the MVP sleep-apnoea file (same GRCh38 "
        "coordinates, allele-checked).",
        "Estimators: IVW multiplicative random effects (primary), IVW fixed, MR-Egger (intercept, I2_GX), weighted "
        "median (parametric bootstrap SE), weighted mode (bootstrap SE); Cochran's Q and I2; mean F. Sensitivity: "
        "per-SNP Steiger filtering (SNPs explaining significantly more variance in the outcome removed; "
        "binary traits via the log-OR liability approximation), +/- 10 Mb clumping. MVMR-IVW with conditional F "
        "(exposure covariance set to 0; joint re-clumping +/- 1 Mb by the smaller p).",
        "Evidence classes (univariable): robust = IVW FDR q < 0.05 within family, same sign for IVW, Egger, weighted "
        "median and weighted mode, weighted median or mode p < 0.05, Egger intercept p >= 0.05; 'supported' = q < "
        "0.05 and consistent but weaker sensitivity support; 'suggestive' = p < 0.05 and consistent; 'inconsistent "
        "direction' = p < 0.05 with disagreeing signs.",
        "Units: estimates are per exposure unit shown (SBP per 10 mmHg; BMI, WHRadjBMI, TG, BBJ traits per SD; FG per "
        "mmol/L; HbA1c per 1 %; OSA, T2D per log-odds of liability; beds as exposures per SD (ASI, retina), per mm "
        "(cIMT) or per log-odds (plaque)) on the outcome's source scale (ASI and retina SD units, cIMT mm, log-odds "
        "for binary outcomes; parental lifespan as log HR of death, sign of Timmers beta1 flipped). "
        "`ivw_adverse_oriented` multiplies by the adverse sign (CRAE, AVR, density: lower is adverse; CRVE, "
        "tortuosity, ASI, IMT, plaque, PAD: higher is adverse; same orientation as the HPP atlas). "
        "`ivw_per_sd_outcome` divides by the outcome SD implied by the GWAS (median se*sqrt(2f(1-f)N)).",
        f"Bootstrap: weighted median {args.n_boot if mode == 'real' else 300}, weighted mode "
        f"{min(args.n_boot, 1000) if mode == 'real' else 100} replicates; seed {args.seed}.",
    ])
    rep.h("GWAS registry and instruments")
    cols = [c for c in ["id", "label", "build", "ancestry", "N", "units", "UKB_in_sample", "implied_sd",
                        "rows_scanned", "p_le_5e8_rows", "candidates", "instruments_1mb", "instruments_10mb",
                        "manifest_sha256_match"] if c in registry.columns]
    rep.table("gwas_registry", registry, "GWAS used (full table with sources, files and SHA-256 in the CSV)",
              max_rows=None)
    rep.parts[-2] = C.md_table(registry[cols])
    rep.p("Not used: " + "; ".join(f"{a} ({b})" for a, b in NOT_USED) + ".")
    if eafqc is not None and len(eafqc):
        rep.table("eaf_concordance", eafqc, "Allele-label QC: EAF concordance with Keaton SBP after alignment "
                  "(non-palindromic; Ortin Vela retina files are read with swapped labels, see Methods)", max_rows=None)
    if xw_rows:
        rep.table("grch38_crosswalk", pd.DataFrame(xw_rows), "GRCh38 cross-walk (rsID found for instruments)")
    if map_rows:
        rep.table("variant_mapping", pd.DataFrame(map_rows), "rsID / GRCh37 position filling of instrument sets",
                  max_rows=80)

    def headline(fam, title, name):
        g = summ[summ["family"] == fam].copy()
        if g.empty:
            return
        g["IVW [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(g["ivw"], g["ivw_lci"], g["ivw_uci"])]
        show = g[["exposure", "outcome", "nsnp", "IVW [95% CI]", "ivw_p", "ivw_q", "egger", "egger_intercept_p",
                  "wmedian", "wmode", "I2", "F_mean", "steiger_n_removed", "ivw_steiger", "ivw_10mb", "evidence"]
                 if "egger" in g.columns else ["exposure", "outcome", "nsnp", "IVW [95% CI]", "ivw_p", "evidence"]]
        show = show[[c for c in show.columns if c in g.columns or c == "IVW [95% CI]"]]
        rep.table(name, g, title, max_rows=None)
        rep.parts[-2] = C.md_table(show)
    rep.h("A. Risk factors -> vascular beds (primary, European)")
    headline("A_risk_to_beds", "IVW (multiplicative random effects) with sensitivity estimators", "mr_A_risk_to_beds")
    rep.h("A2. Sleep-apnoea alternatives -> beds and outcomes")
    headline("A2_osa_alternatives", "OSA sensitivity exposures", "mr_A2_osa_alternatives")
    rep.h("B. Positive controls: risk factors -> clinical outcomes")
    headline("B_risk_to_outcomes_posctrl", "Risk factors -> CAD, stroke, HF, lifespan", "mr_B_positive_controls")
    rep.h("C. Uncoupled adiposity (Chami 2025) -> beds and outcomes")
    headline("C_chami_uncoupled", "Chami uncoupled-adiposity variants (per SD body-fat %)", "mr_C_chami")
    rep.h("D. BioBank Japan / East Asian instruments -> beds and outcomes")
    headline("D_bbj_eas", "EAS exposures (outcomes are European or mixed): trans-ancestry MR", "mr_D_bbj_eas")
    cmp_rows = []
    sd_sbp = registry.set_index("id")["implied_sd"].get("sbp", np.nan) if len(registry) else np.nan
    for eas, eur_id, conv in (("sbp_bbj", "sbp", sd_sbp / 10.0), ("bmi_bbj", "bmi", 1.0), ("tg_bbj", "tg", 1.0),
                              ("tg_eas", "tg", 1.0)):
        a = summ[summ["exposure"] == eas].set_index("outcome")
        b = summ[summ["exposure"] == eur_id].set_index("outcome")
        for oc in a.index.intersection(b.index):
            cmp_rows.append({"eas_exposure": eas, "outcome": oc, "eas_nsnp": a.loc[oc, "nsnp"],
                             "eas_ivw_per_sd": a.loc[oc, "ivw"], "eas_lci": a.loc[oc, "ivw_lci"],
                             "eas_uci": a.loc[oc, "ivw_uci"], "eas_p": a.loc[oc, "ivw_p"],
                             "eur_ivw_per_sd": b.loc[oc, "ivw"] * conv, "eur_p": b.loc[oc, "ivw_p"],
                             "same_sign": bool(np.sign(a.loc[oc, "ivw"]) == np.sign(b.loc[oc, "ivw"])),
                             "both_p_lt_0.05": bool(a.loc[oc, "ivw_p"] < 0.05 and b.loc[oc, "ivw_p"] < 0.05)})
    if cmp_rows:
        rep.table("bbj_vs_eur", pd.DataFrame(cmp_rows),
                  "EAS vs EUR estimates on a per-SD exposure scale (EUR SBP per 10 mmHg x implied SD / 10)",
                  max_rows=None)
    rep.h("E. Beds -> clinical outcomes")
    headline("E_beds_to_outcomes", "Genetically proxied beds (>= 3 instruments) -> outcomes", "mr_E_beds_to_outcomes")
    rep.h("MVMR (with SBP)")
    if len(mv):
        keep = [c for c in ["family", "exposures", "outcome", "term", "nsnp", "n_term_instruments", "estimate", "lower",
                            "upper", "p", "q_within_family_term", "conditional_F", "Q_A_p", "ivw_same_snps", "error"]
                if c in mv.columns]
        rep.table("mvmr", mv, "MVMR-IVW (risk factor or bed + SBP)", max_rows=None)
        rep.parts[-2] = C.md_table(mv[keep])
    rep.tables["mr_all_pairs_summary"] = summ
    rep.tables["mr_all_methods_long"] = longd
    rep.tables["harmonisation_actions"] = harm
    for k, v in sim_tables.items():
        rep.table(k, v, k.replace("_", " "), max_rows=40)
    if wiring is not None:
        rep.table("sim_wiring_real_headers", wiring, "Wiring: mapped columns present in the real file headers",
                  max_rows=None)
    rep.h("Run statistics")
    ss = pd.DataFrame(scan_stats)
    rep.table("scan_stats", ss, "File scans (pyarrow streaming; worker peak RSS)", max_rows=None)
    rep.bullets([f"Total wall time {time.time() - t_start:.0f} s; main-process peak RSS {maxrss_mb():.0f} MB; "
                 f"largest worker peak RSS {maxrss_mb(resource.RUSAGE_CHILDREN):.0f} MB; scan workers {args.workers}, "
                 f"MR workers {args.mr_workers}."])
    if notes:
        rep.h("Problems and notes")
        rep.bullets(notes)


def main():
    def extra(ap):
        ap.add_argument("--workers", type=int, default=6, help="parallel file scans")
        ap.add_argument("--mr-workers", type=int, default=10, help="parallel MR fits")
        ap.add_argument("--no-cache", action="store_true", help="ignore cached per-file extracts")
        ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 of the input files")
        ap.add_argument("--keep-sim", action="store_true", help="keep simulated input files")
    args = C.parse_args(__doc__.splitlines()[0], extra)
    mode, outdir, token = C.gate(args, TASK)
    if mode == "real" and token is None:
        raise SystemExit("real mode needs the gate token")
    log(f"mode {mode}; output {outdir}")
    run(args, mode, outdir)
    log("finished")


if __name__ == "__main__":
    main()
