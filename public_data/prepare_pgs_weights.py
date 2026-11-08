#!/usr/bin/env python3
"""Compact PGS Catalog GRCh37-harmonised scoring files into small parquet files for upload to the HPP sandbox.

Reads the selected `external/pgs/*/PGS*_hmPOS_GRCh37.txt.gz` files (read-only), keeps autosomal SNV/indel rows
with a harmonised position and both alleles, and writes one parquet per score with columns
  chr (int8), pos (int32), ea (str), oa (str), w (float32), af (float32, effect-allele frequency or NaN)
plus `<out name>_manifest.tsv` (score, PGS ID, trait, citation, rows in, rows kept, SHA-256 of the parquet) and
`<out name>_SHA256SUMS`; parquet files are named `f1pgs_<score>.parquet` (unique names survive a flattened upload).

Usage: prepare_pgs_weights.py --out <dir>   (refuses to overwrite)
Only public PGS Catalog data are processed; nothing from HPP.
"""
import argparse
import gzip
import hashlib
import os

import numpy as np
import pandas as pd

EXT = os.path.join(os.environ.get("F1_WORKDIR", "."), "external", "pgs")
# score name -> (folder, trait).  F1 uses non-liver exposures only.
SCORES = {
    "SBP_gw": ("13-PGS004603_SBP_PGS", "systolic BP (genome-wide)"),
    "DBP_gw": ("13-PGS004604_DBP_PGS", "diastolic BP (genome-wide)"),
    "PP_gw": ("13-PGS004605_PP_PGS", "pulse pressure (genome-wide)"),
    "SBP_grs": ("13-PGS002257_SBP_PGS", "systolic BP (885-variant GRS, Evangelou 2018)"),
    "DBP_grs": ("13-PGS002258_DBP_PGS", "diastolic BP (886-variant GRS, Evangelou 2018)"),
    "BMI_gw": ("13-PGS000027_BMI_PGS", "BMI (genome-wide)"),
    "BMI_grs": ("13-PGS000298_BMI_PGS", "BMI (942-variant GRS)"),
    "WHRadjBMI_gw": ("13-PGS003485_WHRadjBMI_PGS", "WHR adjusted for BMI (genome-wide)"),
    "WHRadjBMI_grs": ("13-PGS000299_WHRadjBMI_PGS", "WHR adjusted for BMI (463-variant GRS)"),
    "WHR_gw": ("13-PGS003484_WHR_PGS", "WHR (genome-wide)"),
    "TG_gw": ("13-PGS002197_TG_PGS", "triglycerides (genome-wide)"),
    "TG_grs": ("13-PGS000699_TG_PGS", "triglycerides (16k variants)"),
    "LDL_gw": ("13-PGS000892_LDL_C_PGS", "LDL cholesterol (genome-wide)"),
    "T2D_gw": ("13-PGS000036_T2D_PGS", "type 2 diabetes (171k variants)"),
    "OSA_a": ("13-PGS003479_OSA_PGS", "obstructive sleep apnoea"),
    "OSA_b": ("13-PGS005219_OSA_PGS", "obstructive sleep apnoea"),
    "OSA_c": ("13-PGS005220_OSA_PGS", "obstructive sleep apnoea"),
}


def header(path):
    meta = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if "=" in line:
                k, v = line.lstrip("#").strip().split("=", 1)
                meta[k] = v
    return meta


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def compact(path):
    cols = None
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                cols = line.rstrip("\n").split("\t")
                break
    use = [c for c in ("hm_chr", "hm_pos", "effect_allele", "other_allele", "hm_inferOtherAllele", "effect_weight",
                       "allelefrequency_effect") if c in cols]
    d = pd.read_csv(path, sep="\t", comment="#", usecols=use, dtype=str, low_memory=False)
    n_in = len(d)
    oa = d["other_allele"] if "other_allele" in d.columns else pd.Series(np.nan, index=d.index)
    if "hm_inferOtherAllele" in d.columns:
        inf = d["hm_inferOtherAllele"]
        single = inf.notna() & ~inf.astype(str).str.contains("/")
        oa = oa.where(oa.notna(), inf.where(single))
    out = pd.DataFrame({
        "chr": pd.to_numeric(d["hm_chr"], errors="coerce"),
        "pos": pd.to_numeric(d["hm_pos"], errors="coerce"),
        "ea": d["effect_allele"].astype(str).str.upper(),
        "oa": oa.astype(str).str.upper(),
        "w": pd.to_numeric(d["effect_weight"], errors="coerce"),
        "af": pd.to_numeric(d["allelefrequency_effect"], errors="coerce") if "allelefrequency_effect" in d.columns
        else np.nan})
    ok = out["chr"].between(1, 22) & out["pos"].notna() & out["w"].notna() & out["ea"].str.match(r"^[ACGT]+$") & \
        out["oa"].str.match(r"^[ACGT]+$") & (out["ea"] != out["oa"])
    out = out[ok].copy()
    out["chr"] = out["chr"].astype("int8")
    out["pos"] = out["pos"].astype("int32")
    out["w"] = out["w"].astype("float32")
    out["af"] = out["af"].astype("float32")
    out = out.drop_duplicates(["chr", "pos", "ea", "oa"]).sort_values(["chr", "pos"]).reset_index(drop=True)
    return out, n_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("out exists")
    os.makedirs(a.out)
    rows = []
    for name, (folder, trait) in SCORES.items():
        src = [f for f in os.listdir(os.path.join(EXT, folder)) if f.endswith(".txt.gz")][0]
        path = os.path.join(EXT, folder, src)
        meta = header(path)
        d, n_in = compact(path)
        fn = f"f1pgs_{name}.parquet"
        d.to_parquet(os.path.join(a.out, fn), index=False, compression="zstd")
        rows.append({"score": name, "pgs_id": meta.get("pgs_id"), "trait": trait,
                     "weight_type": meta.get("weight_type"), "citation": meta.get("citation"),
                     "source_file": src, "source_sha256": sha(path), "rows_in": n_in, "rows_kept": len(d),
                     "file": fn, "sha256": sha(os.path.join(a.out, fn))})
        print(f"{name}: {n_in} -> {len(d)} rows, {os.path.getsize(os.path.join(a.out, fn)) / 1e6:.1f} MB", flush=True)
    m = pd.DataFrame(rows)
    tag = os.path.basename(os.path.normpath(a.out))   # unique names: the portal may flatten folders on upload
    m.to_csv(os.path.join(a.out, f"{tag}_manifest.tsv"), sep="\t", index=False)
    with open(os.path.join(a.out, f"{tag}_SHA256SUMS"), "w") as fh:
        for f in sorted(os.listdir(a.out)):
            if not f.endswith("_SHA256SUMS"):
                fh.write(f"{sha(os.path.join(a.out, f))}  {f}\n")


if __name__ == "__main__":
    main()
