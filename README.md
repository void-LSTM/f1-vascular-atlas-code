# F1 vascular atlas code

Source code for the analyses in the paper "Within-person comparison of four vascular measurements reveals distinct exposure profiles and weakly coupled change".

Zenodo DOI: https://doi.org/10.5281/zenodo.24049325 (v1.0.0)

## Overview

The study relates four vascular measurements in the Human Phenotype Project (HPP) to 250 exposures, molecular layers, genetics, change over time and later clinical measurements. The four measurements are femoral–ankle pulse wave velocity (faPWV), carotid intima–media thickness (cIMT), retinal artery width and the ankle–brachial index (ABI). External public cohorts and GWAS summary statistics are used for replication, Mendelian randomization and clinical endpoints.

- `hpp/`: analyses run inside the HPP research environment. Keep all files in one folder: scripts load each other from the same folder.
- `public_data/`: parsers and analyses for the external cohorts, Mendelian randomization, colocalization and polygenic-score weights.
- `public_data/impact_nonfatal/`: first myocardial infarction or stroke in IMPACT (Supplementary Table 6).

Details of every analysis are given in the Methods and Supplementary Note of the paper.

## Figures and tables

| Paper item | Code |
|---|---|
| Measurement correction and retest reliability (Methods; Extended Data Fig. 2a) | `hpp/f1_atlas_v1_5.py`, `hpp/f1_acq_anatomy_v1.py`, `hpp/f1_acq_audit_v1.py` |
| Table 1 | `hpp/f1_common_four_characteristics_v1.py` |
| Fig. 1 (overview) | Values from the analyses below (Source Data Fig. 1) |
| Fig. 2a,d; Extended Data Fig. 1; Supplementary Fig. 7; Supplementary Table 2 (250-exposure atlas, models M1–M3, heterogeneity tests) | `hpp/f1_exposome_v1.py` |
| Fig. 2b,c; Extended Data Figs. 2b, 5c and 7; Supplementary Fig. 1; Supplementary Table 1 (shared pattern, permutation tests, adverse-decile overlap, age equivalents, body-size matching, oral taxa) | `hpp/f1_robust_v1.py` (uses `f1_exposome_v1.py`, `f1_family_v1_1.py`, `f1_omics_v1_2.py`) |
| Fig. 3a,b; Extended Data Fig. 5a,b (cardiorespiratory and skeletal exposures) | `hpp/f1_exposome_v1.py` |
| Fig. 3c; Extended Data Fig. 3 (usual, acute and beyond-acute heart rate) | `hpp/f1_nk_hr_v1.py` |
| Fig. 3d (matched body-size adjustment) | `hpp/f1_robust_v1.py` |
| Fig. 4a; Extended Data Fig. 6 (genetically instrumented pressure components) | `hpp/f1_bp_footprint_v1.py`, `hpp/f1_prs_v1_1.py` |
| Fig. 4b; Supplementary Figs. 2a and 3a; Supplementary Table 3 (molecular screens, equal-n subsamples) | `hpp/f1_omics_v1_3.py` (run `ew`) |
| Fig. 4c; Supplementary Figs. 2b and 3b (usual- and acute-pressure conditioning) | `hpp/f1_omics_v1_3.py` (run `ps`) |
| PBMC RNA, composition and modules (Supplementary Table 3) | `hpp/f1_omics_v1_1.py`, `hpp/f1_molecular_followup_v1.py`, `hpp/f1_longitudinal_feasibility_v1.py`, `hpp/prepare_p14_reference.py` |
| Fig. 4d; Extended Data Fig. 8a–c; Supplementary Fig. 4 (disease polygenic scores) | `hpp/f1_outcome_liability_v1.py`, `hpp/f1_followup2_v1.py` |
| Extended Data Fig. 8d–f (family and cohabitant resemblance) | `hpp/f1_family_v1_1.py` |
| Fig. 4e; Supplementary Fig. 6a; Supplementary Table 4 (coupling of observed changes) | `hpp/f1_change_synchrony_v1.py`, `hpp/f1_longitudinal_extension_v1.py`, `hpp/f1_selection_weighting_v1.py`, `hpp/f1_correction_jackknife_v1.py`, `hpp/f1_baseline_molecular_longitudinal_v1.py`, `hpp/f1_lipid_classes_longitudinal_v1.py` |
| Fig. 5a–d; Extended Data Fig. 10; Supplementary Table 5 (prediction framework) | `hpp/run_joint_tuned_v3.py`, `hpp/recorded_stage_outcome_v3.py`, `hpp/audit_clinical_timing_v1.py`, with `hpp/atlas_joint_network_v1.py`, `hpp/network_data_clinical_v2.py`, `hpp/f1_clinical_utility_v1.py`, `hpp/f1_clinical_task_feasibility_v1.py` |
| Fig. 5e (earlier input-set comparisons) | `hpp/run_joint_network_v1.py`, `hpp/network_data_v1.py`, `hpp/run_joint_clinical_v2.py`, `hpp/network_data_clinical_v2.py` |
| Extended Data Fig. 9a; Supplementary Table 7 (incident hypertension, treatment initiation, reference centiles in HPP) | `hpp/f1_followup2_v1.py` (uses `f1_atlas_outcomes_v2.py`, `f1_outcomes_v1.py`) |
| Extended Data Fig. 4a (MARK) | `public_data/prepare_mark.py`, `public_data/analysis_o_mark_external.py` |
| Extended Data Fig. 4b–f (Keio, Japanese health-check, Chinese baPWV and IMPACT profiles) | `public_data/parse_keio.py`, `public_data/parse_japan_ggt_bapwv.py`, `public_data/parse_china_bapwv.py`, `public_data/parse_impact_abi.py`, `public_data/analysis_m_twobed_replication.py` |
| Extended Data Fig. 9b; Supplementary Table 7 (IMPACT major adverse cardiovascular events) | `public_data/analysis_d_impact_abi.py`, `public_data/analysis_n_impact_profiles.py` |
| Supplementary Table 6 (IMPACT first myocardial infarction or stroke) | `public_data/impact_nonfatal/run.py`, `public_data/impact_nonfatal/feasibility.py` |
| Supplementary Table 7 (NHANES 2005–2008 retinal signs and mortality; ARTPER ABI) | `public_data/analysis_m_nhanes_retina_mortality.py`, `public_data/analysis_m_retina_robustness.py`, `public_data/derive_m_crossbed_profile_nhanes.py`, `public_data/analysis_e_artper.py`, `public_data/parse_artper.py` |
| Supplementary Fig. 6b; Supplementary Table 4 (post-PCI and Xinruikang changes) | `public_data/analysis_p_public_longitudinal.py`, `public_data/parse_xinruikang.py` |
| Supplementary Fig. 5; Supplementary Table 8 (Mendelian randomization of systolic pressure on cIMT) | `public_data/analysis_i3_mr_ldclump.py`, `public_data/analysis_i_mr.py`, `public_data/check_l6_mr_reconcile.py`, `public_data/lib_mr.py` |
| Supplementary Table 8 (MVP outcomes, subtype contrasts, ApoB colocalization) | `public_data/analysis_m_mr_mvp_outcomes.py`, `public_data/analysis_m_mr_subtype_interaction.py`, `public_data/analysis_r_apob_coloc.py` |
| Polygenic-score weights from the PGS Catalog | `public_data/prepare_pgs_weights.py`, `public_data/prepare_pgs_weights_outcome.py` |

Shared modules: `hpp/f1_atlas_v1_5.py` (measurement frame and correction, loaded by most HPP scripts), `public_data/s0an_common.py`, `public_data/s0lib.py`, `public_data/lib_survival.py`, `public_data/lib_mr.py`.

## Data

HPP data are available to researchers at universities and other research institutions through https://humanphenotypeproject.org/data-access (contact info@pheno.ai). The HPP scripts run in the HPP secure research environment and write aggregate results only.

The external datasets are public. Their sources and DOIs are listed in the Data availability statement of the paper. GWAS summary statistics come from the original publications, polygenic scores from the PGS Catalog (https://www.pgscatalog.org) and the LD reference from the 1000 Genomes Project European panel.

## Installation

```bash
git clone https://github.com/void-LSTM/f1-vascular-atlas-code.git
cd f1-vascular-atlas-code
pip install -r requirements.txt
```

The HPP analyses used Python 3.11.5 with the versions pinned in `requirements.txt`. The public-data parsers also need `openpyxl`, `pyreadstat` and, for legacy `.xls` files, `xlrd`. `public_data/impact_nonfatal` used Python 3.13.5 with numpy 2.1.3, pandas 2.2.3, scipy 1.15.3 and statsmodels 0.14.4.

## Usage

**HPP analyses.** Copy `hpp/` into the HPP research environment and run each script on the HPP data root, for example

```bash
python f1_atlas_v1_5.py --root <HPP_DATA_ROOT> --out f1_atlas_out_v1_5
python f1_exposome_v1.py --root <HPP_DATA_ROOT> --out f1_exposome_out_v1
```

Each script lists its arguments in its header and with `--help`. Counts are rounded to tens and cells with fewer than 20 participants are suppressed before results are written.

**Public data.** Set a working directory, place the downloaded files under `external/` as named in each parser, then run the parsers and the analyses:

```bash
export F1_WORKDIR=/path/to/workdir
python public_data/parse_impact_abi.py
python public_data/analysis_d_impact_abi.py --real
```

Parsers write cleaned tables to `results/stage0/clean/`; analyses write to `results/stage0/analysis/<task>/`. `--simulate` runs an analysis end to end on a simulated table with the same columns.

## License

[![License: BSD-2](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)

## Contact

Imran Razzak (imran.razzak@mbzuai.ac.ae)
