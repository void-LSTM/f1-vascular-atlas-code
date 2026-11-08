#!/usr/bin/env python3
"""Compact the outcome PGS (package P11, outcome-liability atlas) into sandbox weight files.

A minimal wrapper around prepare_pgs_weights.py (same folder): it reuses that script's header(), compact() and sha()
unchanged and only replaces the score list.  Reads the harmonised GRCh37 scoring files downloaded on 2026-10-06 into
`external/pgs/14-*` (read-only) and writes one parquet per score (chr int8, pos int32, ea, oa, w float32, af float32)
plus `<out name>_manifest.tsv` and `<out name>_SHA256SUMS`, in the format f1_prs_v1_1.py and f1_outcome_liability_v1.py
read.  T2D is not re-derived: the PRS v1.1 weight file `f1pgs_T2D_gw.parquet` (PGS000036, `pgs_weights_f1_v1`) is
copied byte for byte as `f1pgs_T2D.parquet` so that the P11 run can reproduce the PRS v1.1 T2D_gw lines.

Usage: prepare_pgs_weights_outcome.py --out <dir>   (refuses to overwrite)
       prepare_pgs_weights_outcome.py --set v2 --out <dir>
--set v1 (default, unchanged since P11): the ten P11 scores above.  --set v2 (package P12 f1_followup2_v1, added
2026-10-06): the better-covered replacement scores downloaded on 2026-10-06 into `external/pgs/15-*` -- CAD2 (PGS004696,
PRS-CSx, Smith 2024), CADm (PGS003725 GPS_Mult, Patel 2023; multi-trait, sensitivity only) and T2D2 (PGS002771, PRS-CS,
Mars 2022); no byte copy.  Same compact format and file naming (f1pgs_<score>.parquet; names differ from v1).
Only public PGS Catalog data are processed; nothing from HPP.
"""
import argparse
import importlib.util
import os
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(os.environ.get("F1_WORKDIR", "."), "external", "pgs")
T2D_SRC = os.path.join(F1, "sandbox_packages", "f1_atlas_v1_20261003", "pgs_weights_f1_v1", "f1pgs_T2D_gw.parquet")
T2D_SHA = None  # filled from the pgs_weights_f1_v1 manifest at run time (must match)
# score -> (folder under external/pgs, trait)
SCORES = {
    "CAD": ("14-PGS000018_CAD_PGS", "coronary artery disease (metaGRS, Inouye 2018)"),
    "IS": ("14-PGS002724_IschStroke_PGS", "ischaemic stroke (GIGASTROKE iPGS EUR, Mishra 2022)"),
    "PCS": ("14-PGS002052_PrecerebralStenosis_PGS", "precerebral artery occlusion/stenosis (UKB phecode 433.1; "
                                                    "large-artery proxy, Prive 2022)"),
    "PAD": ("14-PGS005217_PAD_PGS", "peripheral artery disease (Flores 2025)"),
    "HF": ("14-PGS005097_HF_PGS", "heart failure (PRS-CSx meta, Lee 2025)"),
    "AF": ("14-PGS004878_AF_PGS", "atrial fibrillation (INTERVENE MegaPRS, Jermy 2024)"),
    "CKD": ("14-PGS002237_CKD_PGS", "chronic kidney disease stage 3+ (GPS, Khan 2022)"),
    "HTN": ("14-PGS002047_HTN_PGS", "hypertension (UKB phecode 401, Prive 2022)"),
    "AAA": ("14-PGS003972_AAA_PGS", "abdominal aortic aneurysm (Roychowdhury 2023)"),
}
# --set v2 (P12): replacement CAD and T2D scores built on HapMap3 variant sets (P11 coverage 0.32 / 0.30 for v1 CAD / T2D)
SCORES_V2 = {
    "CAD2": ("15-PGS004696_CAD_PGS", "coronary heart disease (PRS-CSx multi-ancestry, Smith 2024)"),
    "CADm": ("15-PGS003725_CADmulti_PGS", "coronary artery disease (GPS_Mult, CAD + 10 risk factors, Patel 2023)"),
    "T2D2": ("15-PGS002771_T2D_PGS", "type 2 diabetes (PRS-CS, Mars 2022)"),
}


def load_base():
    spec = importlib.util.spec_from_file_location("prepare_pgs_weights", os.path.join(HERE, "prepare_pgs_weights.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="v1", choices=["v1", "v2"])
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("out exists")
    base = load_base()
    if a.set == "v2":
        return build_v2(base, a.out)
    man0 = pd.read_csv(os.path.join(os.path.dirname(T2D_SRC), "pgs_weights_f1_v1_manifest.tsv"), sep="\t")
    t2d = man0[man0["score"] == "T2D_gw"].iloc[0]
    if base.sha(T2D_SRC) != t2d["sha256"]:
        raise SystemExit("T2D_gw source file does not match its PRS v1.1 manifest")
    os.makedirs(a.out)
    rows = []
    for name, (folder, trait) in SCORES.items():
        src = [f for f in os.listdir(os.path.join(EXT, folder)) if f.endswith("_hmPOS_GRCh37.txt.gz")][0]
        path = os.path.join(EXT, folder, src)
        meta = base.header(path)
        d, n_in = base.compact(path)
        fn = f"f1pgs_{name}.parquet"
        d.to_parquet(os.path.join(a.out, fn), index=False, compression="zstd")
        rows.append({"score": name, "pgs_id": meta.get("pgs_id"), "trait": trait,
                     "weight_type": meta.get("weight_type"), "citation": meta.get("citation"),
                     "source_file": src, "source_sha256": base.sha(path), "rows_in": n_in, "rows_kept": len(d),
                     "file": fn, "sha256": base.sha(os.path.join(a.out, fn))})
        print(f"{name}: {n_in} -> {len(d)} rows, {os.path.getsize(os.path.join(a.out, fn)) / 1e6:.1f} MB", flush=True)
    fn = "f1pgs_T2D.parquet"
    shutil.copyfile(T2D_SRC, os.path.join(a.out, fn))
    rows.append({"score": "T2D", "pgs_id": t2d["pgs_id"], "trait": "type 2 diabetes (PRS v1.1 T2D_gw, byte copy)",
                 "weight_type": t2d.get("weight_type"), "citation": t2d.get("citation"),
                 "source_file": "pgs_weights_f1_v1/f1pgs_T2D_gw.parquet", "source_sha256": t2d["sha256"],
                 "rows_in": t2d["rows_in"], "rows_kept": t2d["rows_kept"], "file": fn,
                 "sha256": base.sha(os.path.join(a.out, fn))})
    m = pd.DataFrame(rows)
    tag = os.path.basename(os.path.normpath(a.out))
    m.to_csv(os.path.join(a.out, f"{tag}_manifest.tsv"), sep="\t", index=False)
    with open(os.path.join(a.out, f"{tag}_SHA256SUMS"), "w") as fh:
        for f in sorted(os.listdir(a.out)):
            if not f.endswith("_SHA256SUMS"):
                fh.write(f"{base.sha(os.path.join(a.out, f))}  {f}\n")
    print(m[["score", "pgs_id", "rows_in", "rows_kept", "sha256"]].to_string())


def build_v2(base, out):
    """P12 replacement scores only (no T2D byte copy); same manifest and SHA256SUMS layout as v1."""
    os.makedirs(out)
    rows = []
    for name, (folder, trait) in SCORES_V2.items():
        src = [f for f in os.listdir(os.path.join(EXT, folder)) if f.endswith("_hmPOS_GRCh37.txt.gz")][0]
        path = os.path.join(EXT, folder, src)
        meta = base.header(path)
        d, n_in = base.compact(path)
        fn = f"f1pgs_{name}.parquet"
        d.to_parquet(os.path.join(out, fn), index=False, compression="zstd")
        rows.append({"score": name, "pgs_id": meta.get("pgs_id"), "trait": trait,
                     "weight_type": meta.get("weight_type"), "citation": meta.get("citation"),
                     "source_file": src, "source_sha256": base.sha(path), "rows_in": n_in, "rows_kept": len(d),
                     "file": fn, "sha256": base.sha(os.path.join(out, fn))})
        print(f"{name}: {n_in} -> {len(d)} rows, {os.path.getsize(os.path.join(out, fn)) / 1e6:.1f} MB", flush=True)
    m = pd.DataFrame(rows)
    tag = os.path.basename(os.path.normpath(out))
    m.to_csv(os.path.join(out, f"{tag}_manifest.tsv"), sep="\t", index=False)
    with open(os.path.join(out, f"{tag}_SHA256SUMS"), "w") as fh:
        for f in sorted(os.listdir(out)):
            if not f.endswith("_SHA256SUMS"):
                fh.write(f"{base.sha(os.path.join(out, f))}  {f}\n")
    print(m[["score", "pgs_id", "rows_in", "rows_kept", "sha256"]].to_string())


if __name__ == "__main__":
    main()
