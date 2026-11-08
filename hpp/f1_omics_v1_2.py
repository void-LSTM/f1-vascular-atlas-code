#!/usr/bin/env python3
"""F1 multi-bed vascular atlas - omics module v1 (HPP sandbox; writes aggregates only).

Every omics layer is run against every vascular bed.  Beds come from the atlas frame: the atlas script given by
--atlas is imported and its build_frame(root, hb=None) is called, so beds are calendar-adjusted exactly as in the
atlas (nothing about bed construction is re-implemented here).  Beds are adverse-oriented (atlas ORIENT: higher =
worse; retinal artery width, ABI and fractal dimension are sign-flipped).

Layers (PLAN section 6, B1-B5)
  metab   Nightingale NMR (~250 measures); non-negative right-skewed measures (skew > 1) log-transformed
  lipid   serum lipidomics (~190 species); same rule (the stored values are already log-scale intensities)
  gut     MetaPhlAn4 species (aggregated species table); prevalence >= 10 %; CLR with pseudocount (half the 1st
          percentile of non-zero abundances); Shannon and richness from all species; samples failing the stool-vs-WGS
          validation dropped; technical covariate log10 read count
  oral    oral MetaPhlAn4 species (aggregated table), same treatment; the pre-stated periodontal taxa
          (P. gingivalis, T. forsythia, T. denticola, F. nucleatum) are kept regardless of prevalence (>= 20 carriers)
  rna     PBMC 3' RNA-seq counts (long per-batch tables): genes detected in >= 50 % of samples (top --max-genes by
          mean logCPM), logCPM, centred within batch; batch fixed effects in every model
  rnamod  pre-specified PBMC modules = mean z-score of batch-centred logCPM of the module genes
Matching: each atlas visit (00_00 / 02_00) takes the omics sample with the same research_stage, else the sample
nearest in date within 365 days.  Per bed: one row per participant (stage match, then smallest gap, then visit 1).
Analyses
  A   per feature x bed x model: standardized beta (feature SD -> bed SD), HC1 SE, vectorised masked OLS; BH q
      within layer x bed x model.  M1 age + sex; M2 + BMI; M3 + BMI + sitting SBP + antihypertensives; technical
      covariates (batch / read depth) in every model.
  B   common four-bed sample: the four oriented betas with their joint robust covariance; pairwise contrasts (BH q
      within layer x model x pair) and the 3-df Wald heterogeneity test.  Bed-specific feature (strict): q < 0.05
      in exactly one core bed and contrast q < 0.05 against every other core bed; lenient: against at least one.
  C1  molecular vascular ages: nested 5-fold CV elastic net (inner 3-fold CV over 20 alphas x l1 ratio 0.1/0.5/0.9;
      gene-level RNA: the 500 genes with the highest mean CPM, an outcome-blind choice) of the age/sex-residualised
      oriented bed on each layer and on all layers combined (rna modules instead of genes);
      out-of-fold R2 with fold SD and bootstrap CI; top weights; correlation of bed-specific predicted gaps.
  C2  vascular-ageing subtypes: Gaussian mixture (full covariance, k = 2..6, BIC) on the four age/sex-standardised
      oriented beds; bootstrap ARI; cluster sizes, bed and exposure means; per-cluster omics differences.
Privacy (as the atlas): aggregates only; counts rounded to 10, cells < 20 suppressed; quantiles 5-95 % only; no
participant ids; aborts if any 7+ digit run or participant id appears in any output; refuses to overwrite --out.
Run (sandbox pheno env): python f1_omics_v1.py --root <HPP_DATA_ROOT> \
    --atlas f1_atlas_v1_3.py --out f1_omics_out_v1
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # small problems: avoid BLAS oversubscription
    os.environ.setdefault(_v, "4")
import argparse
import glob
import hashlib
import importlib.util
import json
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse, stats

VERSION = "f1_omics_v1.2 2026-10-04"
DRAW_LAYERS = set()  # v1.1/v1.2: layers whose sample clock hour (centred, + square) is added to the technical covariates
A = None  # atlas module (imported in main)
MAXGAP = 365
PREV = 0.10
MINOBS = 0.5
MIN_N = 100
EXTRA = ["vein_w", "fd", "art_tort", "cavi0", "sokolow"]
MODELS = {"M1": ["age", "sex_male"], "M2": ["age", "sex_male", "bmi"],
          "M3": ["age", "sex_male", "bmi", "sbp_sit", "antihtn"]}
PAIRS = [("pwv", "imt"), ("pwv", "art_w"), ("imt", "art_w"), ("pwv", "abi_min"), ("imt", "abi_min"),
         ("art_w", "abi_min")]
HYP = ["Porphyromonas_gingivalis", "Tannerella_forsythia", "Treponema_denticola", "Fusobacterium_nucleatum"]
# PBMC modules: inflammasome/alarmin and interferon programmes (hypotheses), plus cell-type proxies (bulk PBMC
# expression is dominated by composition) and platelet / erythroid contamination (technical controls)
MODULES = {"inflam_alarmin": ["IL1B", "TNF", "IL6", "CXCL8", "NLRP3", "S100A8", "S100A9", "S100A12"],
           "interferon": ["IFI27", "IFI44L", "IFIT1", "ISG15", "RSAD2", "SIGLEC1", "MX1", "OAS1", "IFIT3", "IFI44"],
           "monocyte": ["CD14", "LYZ", "FCN1", "VCAN", "CSF1R", "CD68"],
           "t_cell": ["CD3D", "CD3E", "CD2", "IL7R", "CD28", "LCK"],
           "nk_cytotoxic": ["GNLY", "NKG7", "GZMB", "PRF1", "KLRD1", "GZMA"],
           "b_cell": ["MS4A1", "CD79A", "CD79B", "CD19", "BANK1"],
           "platelet": ["PPBP", "PF4", "GP9", "ITGA2B", "TUBB1"],
           "erythroid": ["HBB", "HBA1", "HBA2", "ALAS2"]}
EXPO = ["age", "sex_male", "bmi", "waist", "log_vat", "sbp_sit", "dbp_sit", "hr_sit", "log_tg", "ldl", "hdl", "hba1c",
        "cgm_mean", "log1p_ahi", "log1p_odi", "egfr", "smk_current", "antihtn", "statin", "att"]
MAXP_RNA = 500
ATLAS_API = ["build_frame", "rd", "rc", "cnt", "qtiles", "pct", "bh", "zs", "fnum", "log", "LOG", "g", "v3", "ci3",
             "one_row_per_participant", "MINCELL", "MAIN_STAGES", "CORE", "ORIENT", "TZ", "VERSION"]
LONG = re.compile(r"\d{7,}")


def log(m):
    A.log(m)


def privacy_check(text, pids):
    m = LONG.search(text)
    if m:
        raise SystemExit(f"PRIVACY ABORT: digit run '{m.group(0)[:3]}...' in output")
    for p in pids:  # ids of 7+ digits are already caught above
        if not (p.isdigit() and len(p) >= 7) and p in text:
            raise SystemExit("PRIVACY ABORT: participant id in output")


def day(s):
    """Naive local calendar day (datetime64[ns]); tz-aware input is converted to Asia/Jerusalem first."""
    s = pd.Series(s).reset_index(drop=True)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_convert(A.TZ).dt.tz_localize(None)
    s = pd.to_datetime(s, errors="coerce")
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_localize(None)
    return s.dt.as_unit("ns").dt.normalize()


def keys(d):
    d = d.reset_index()
    d["participant_id"] = d["participant_id"].astype(str)
    d["research_stage"] = d["research_stage"].astype(str) if "research_stage" in d else "undefined"
    return d


def short(s, n=30):
    s = re.sub(r"^s__", "", str(s))
    return s if len(s) <= n else s[:n - 1] + "~"


def qt(x):
    return A.qtiles(pd.Series(np.asarray(x, float)))


def bh(p):
    return A.bh(list(np.asarray(p, float)))


def pnorm(b, se):
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * stats.norm.sf(np.abs(b / se))


# ============================================================================ layers

def draw_hour(ts):
    """v1.1: local clock hour of a sample timestamp (NaN when missing)."""
    ts = pd.Series(ts).reset_index(drop=True)
    if isinstance(ts.dtype, pd.DatetimeTZDtype):
        ts = ts.dt.tz_convert(A.TZ)
    ts = pd.to_datetime(ts, errors="coerce")
    return (ts.dt.hour + ts.dt.minute / 60.0).to_numpy(float)


def draw_tech(tech, hr, info, layer):
    """v1.2: report the sample-hour quantiles; for layers in DRAW_LAYERS append hour and hour^2 to `tech`."""
    info["draw_hour"] = qt(hr)
    if layer in DRAW_LAYERS and np.isfinite(hr).sum() >= A.MINCELL:
        hc = np.where(np.isfinite(hr), hr, np.nanmedian(hr)) - np.nanmean(hr)
        extra = np.column_stack([hc, hc ** 2])
        tech = extra if tech is None else np.column_stack([tech, extra])
        info["draw_time_adjusted"] = "hour + hour^2"
    return tech


def finish(name, pid, stage, date, X, names, info, tech=None):
    """Drop features with long digit runs in their names (privacy scan) and package a layer."""
    ok = np.array([not LONG.search(str(n)) for n in names], bool)
    info["features_dropped_name"] = int((~ok).sum())
    info["features"] = int(ok.sum())
    info["samples"] = A.rc(len(pid))
    info["participants"] = A.rc(pd.Series(pid).nunique())
    return {"name": name, "pid": np.asarray(pid, str), "stage": np.asarray(stage, str),
            "date": day(date).values, "X": np.ascontiguousarray(X[:, ok]), "names": [str(n) for n, k in zip(names, ok) if k],
            "tech": tech, "info": info}


def tf_cont(X, names, info):
    """Keep features observed in >= MINOBS of samples; log non-negative right-skewed ones; clip at mean +/- 5 SD."""
    X[~np.isfinite(X)] = np.nan
    keep = np.isfinite(X).mean(0) >= MINOBS
    info["features_in"] = len(names)
    info["dropped_missing"] = int((~keep).sum())
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]
    nlog = 0
    for j in range(X.shape[1]):
        x = X[:, j]
        f = np.isfinite(x)
        v = x[f]
        if len(v) > 20 and v.min() >= 0 and stats.skew(v) > 1:
            c = 0.0 if v.min() > 0 else (0.5 * v[v > 0].min() if (v > 0).any() else 1.0)
            x[f] = np.log(v + c)
            nlog += 1
        m, s = np.nanmean(x), np.nanstd(x)
        if s > 0:
            X[:, j] = np.clip(x, m - 5 * s, m + 5 * s)
    sd = np.nanstd(X, 0)
    ok = sd > 0
    info["logged"] = nlog
    return X[:, ok], [n for n, k in zip(names, ok) if k]


def load_wide(root, rel, name):
    d = A.rd(root, rel)
    if d is None:
        return None
    meta = {"participant_id", "cohort", "research_stage", "array_index", "index", "collection_date",
            "collection_timestamp", "timezone"}
    cols = [c for c in d.columns if c not in meta and not str(c).startswith("synthetic_pad")
            and pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])]
    date = day(d["collection_date"]) if "collection_date" in d else pd.Series(pd.NaT, index=range(len(d)))
    if "collection_timestamp" in d:
        date = date.fillna(day(d["collection_timestamp"]))
    info = {}
    X, names = tf_cont(d[cols].to_numpy(dtype=float, na_value=np.nan), cols, info)
    hr = draw_hour(d["collection_timestamp"]) if "collection_timestamp" in d else None  # v1.2
    tech = draw_tech(None, hr, info, name) if hr is not None else None
    L = finish(name, d["participant_id"], d["research_stage"], date, X, names, info, tech)
    L["draw_hour"] = hr
    return L


def mpa_layer(name, s, date, depth, info, force=()):
    """MetaPhlAn species table (rows = samples, columns = taxonomy strings) -> CLR features + alpha diversity."""
    cols = [c for c in s.columns if "s__" in str(c) and "|t__" not in str(c)]
    P = s[cols].to_numpy(dtype=float, na_value=np.nan)
    P[~np.isfinite(P)] = 0.0
    P[P < 0] = 0.0
    tot = P.sum(1)
    ok = tot > 0
    info["samples_empty"] = A.rc((~ok).sum()) if (~ok).sum() >= A.MINCELL else "<20"
    P, s, date, depth = P[ok], s.loc[ok], date[ok], (None if depth is None else depth[ok])
    P /= P.sum(1, keepdims=True)
    pos = P > 0
    rich = pos.sum(1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        shan = -np.where(pos, P * np.log(np.where(pos, P, 1)), 0).sum(1)
    nm = [re.sub(r"^(.*\|)?s__", "", str(c)) for c in cols]
    prev = pos.mean(0)
    carriers = pos.sum(0)
    base = prev >= PREV
    hyp = np.array([any(h in n for h in force) for n in nm]) & (carriers >= A.MINCELL) & ~base
    info["species_total"] = len(cols)
    info["species_prev10"] = int(base.sum())
    keep = base | hyp
    pc = 0.5 * np.percentile(P[:, keep][pos[:, keep]], 1)  # half the 1st percentile of non-zero abundances
    L = np.log(P[:, keep] + pc)
    gm = L[:, base[keep]].mean(1, keepdims=True)
    X = np.column_stack([L - gm, shan, rich])
    names = [n for n, k in zip(nm, keep) if k] + ["alpha_shannon", "alpha_richness"]
    seen = {}
    for i, n in enumerate(names):  # unique names
        seen[n] = seen.get(n, 0) + 1
        if seen[n] > 1:
            names[i] = f"{n}_{seen[n]}"
    info["pseudocount"] = float(pc)
    info["richness"] = qt(rich)
    info["shannon"] = qt(shan)
    info["hyp_prevalence"] = {h: (float(prev[j]) if carriers[j] >= A.MINCELL else "<20")
                              for h in force for j in [i for i, n in enumerate(nm) if h in n][:1]}
    tech = None
    if depth is not None:
        dl = np.log10(pd.to_numeric(pd.Series(depth), errors="coerce").astype(float).clip(lower=1)).to_numpy()
        if np.isfinite(dl).sum() > 0.5 * len(dl):
            tech = np.where(np.isfinite(dl), dl, np.nanmedian(dl))[:, None]
    return finish(name, s["participant_id"], s["research_stage"], date, X, names, info, tech)


def load_gut(root):
    m = A.rd(root, "gut_microbiome/gut_microbiome.parquet",
             columns=["collection_date", "collection_timestamp", "metaphlan_abundance_species_parquet",
                      "trimmed_read_count", "wgs_compare_pass"])
    rel = m["metaphlan_abundance_species_parquet"].dropna().astype(str) if "metaphlan_abundance_species_parquet" in m else []
    path = os.path.join(root, "gut_microbiome", rel.iloc[0].lstrip("./")) if len(rel) else ""
    if not os.path.exists(path):
        path = (glob.glob(os.path.join(root, "gut_microbiome/abundance/metaphlan4/aggregated/*species*.parquet")) + [""])[0]
    s = keys(pd.read_parquet(path))
    k = ["participant_id", "research_stage", "array_index"]
    for d in (s, m):
        d["array_index"] = pd.to_numeric(d["array_index"], errors="coerce")
    mm = m.drop_duplicates(k)
    s = s.merge(mm[k + [c for c in ("collection_date", "collection_timestamp", "trimmed_read_count", "wgs_compare_pass")
                        if c in mm]].rename(columns={"collection_date": "_cd", "collection_timestamp": "_ct"}),
                on=k, how="left")
    info = {"source": os.path.relpath(path, root)}
    if "wgs_compare_pass" in s:
        bad = s["wgs_compare_pass"].astype(str).str.lower().str.contains("fail").to_numpy()
        info["excluded_failed_wgs_validation"] = A.rc(bad.sum()) if bad.sum() >= A.MINCELL else "<20"
        s = s.loc[~bad].reset_index(drop=True)
    date = day(s["_cd"]) if "_cd" in s else pd.Series(pd.NaT, index=s.index)
    if "_ct" in s:
        date = date.fillna(day(s["_ct"]))
    depth = s["trimmed_read_count"].to_numpy() if "trimmed_read_count" in s else None
    return mpa_layer("gut", s, date.values, depth, info)


def load_oral(root):
    path = os.path.join(root, "oral_microbiome/aggregated/metaphlan_abundances_aggregated/species.parquet")
    if not os.path.exists(path):
        path = (glob.glob(os.path.join(root, "oral_microbiome", "**", "species*.parquet"), recursive=True) + [""])[0]
    s = keys(pd.read_parquet(path))
    s["_d"] = day(s["collection_date"]).values
    info = {"source": os.path.relpath(path, root)}
    depth = None
    m = A.rd(root, "oral_microbiome/oral_microbiome.parquet", columns=["collection_date", "metaphlan4_aligned_read_count"])
    if m is not None and "metaphlan4_aligned_read_count" in m:
        m["_d"] = day(m["collection_date"]).values
        m = m.drop_duplicates(["participant_id", "_d"])[["participant_id", "_d", "metaphlan4_aligned_read_count"]]
        s = s.merge(m, on=["participant_id", "_d"], how="left")
        depth = s["metaphlan4_aligned_read_count"].to_numpy()
    return mpa_layer("oral", s, s["_d"].values, depth, info, force=HYP)


def enc(col):
    col = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    if not pa.types.is_dictionary(col.type):
        col = col.dictionary_encode()
    return col.indices.to_numpy(zero_copy_only=False), np.array([str(x) for x in col.dictionary.to_pylist()], object)


def load_rna(root, max_genes):
    m = A.rd(root, "rna_seq/rna_seq.parquet", columns=["collection_date", "collection_timestamp", "batch",
                                                      "full_counts_raw_parquet"])
    m["array_index"] = pd.to_numeric(m["array_index"], errors="coerce")
    files = sorted(set(m["full_counts_raw_parquet"].dropna().astype(str)))
    gid, R, Cc, V, K = {}, [], [], [], []
    off = 0
    for rel in files:
        p = os.path.join(root, "rna_seq", rel.lstrip("./"))
        if not os.path.exists(p):
            continue
        t = pq.read_table(p, columns=["participant_id", "research_stage", "array_index", "gene", "count"])
        gi, gd = enc(t.column("gene"))
        gmap = np.array([gid.setdefault(g, len(gid)) for g in gd], np.int64)
        pi, pdict = enc(t.column("participant_id"))
        si, sdict = enc(t.column("research_stage"))
        ai = t.column("array_index").combine_chunks().to_numpy(zero_copy_only=False)
        kd = pd.DataFrame({"p": pi, "s": si, "a": ai})
        code = kd.groupby(["p", "s", "a"], sort=False).ngroup().to_numpy()
        u = kd.drop_duplicates()
        K.append(pd.DataFrame({"participant_id": pdict[u["p"].to_numpy()], "research_stage": sdict[u["s"].to_numpy()],
                               "array_index": u["a"].to_numpy(), "file": os.path.basename(rel)}))
        R.append(code + off)
        Cc.append(gmap[gi])
        V.append(t.column("count").combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32))
        off += len(u)
    genes = np.array(sorted(gid, key=gid.get), object)
    M = sparse.coo_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cc))),
                          shape=(off, len(genes))).tocsr()
    del R, Cc, V
    S = pd.concat(K, ignore_index=True)
    info = {"batch_files": len(files), "genes_total": len(genes), "count_entries_per_sample":
            qt(np.diff(M.indptr))}
    gok = np.array([not (g.startswith("ERCC-") or LONG.search(g)) for g in genes])
    M, genes = M[:, np.flatnonzero(gok)], genes[gok]
    lib = np.asarray(M.sum(1)).ravel()
    S["lib"] = lib
    S = S.merge(m.drop_duplicates(["participant_id", "research_stage", "array_index"])
                [["participant_id", "research_stage", "array_index", "collection_date", "collection_timestamp", "batch"]],
                on=["participant_id", "research_stage", "array_index"], how="left")
    S["batch"] = S["batch"].astype(str).where(S["batch"].notna(), S["file"].str.replace(".parquet", "", regex=False))
    ll = np.log10(np.clip(lib, 1, None))
    med, mad = np.median(ll), np.median(np.abs(ll - np.median(ll))) * 1.4826
    S["_keep"] = (ll >= med - 3 * mad) & (lib > 0)
    S["_r"] = np.arange(len(S))
    S = S[S["_keep"]].sort_values("lib", ascending=False).drop_duplicates(
        ["participant_id", "research_stage", "array_index"]).sort_values("_r")
    nf = int((ll < med - 3 * mad).sum())
    info["samples_failed_libsize"] = A.rc(nf) if nf >= A.MINCELL else "<20"
    info["library_size"] = qt(lib)
    M = M[S["_r"].to_numpy()]
    lib = S["lib"].to_numpy(float)
    det = np.asarray((M > 0).mean(0)).ravel()
    mean_cpm = np.asarray(M.multiply(1e6 / lib[:, None]).mean(0)).ravel()
    thr = 0.5 if (det >= 0.5).sum() >= 500 else 0.2  # 0.2: fallback for sparse (synthetic) count tables
    cand = np.flatnonzero(det >= thr)
    cand = cand[np.argsort(-mean_cpm[cand])][:max_genes]
    modg = {g for v in MODULES.values() for g in v}
    cs = set(cand.tolist())
    extra = [j for j in np.flatnonzero(det >= 0.1) if genes[j] in modg and j not in cs]
    cols = np.r_[cand, np.array(extra, int)].astype(int)
    D = np.asarray(M[:, cols].todense(), dtype=float)
    D = np.log2((D + 0.5) / (lib[:, None] + 1.0) * 1e6)
    bt = S["batch"].astype(str).to_numpy()
    for b in np.unique(bt):  # centre within batch
        w = bt == b
        D[w] -= D[w].mean(0)
    gnames = list(genes[cols])
    info.update(genes_detected50=int((det >= 0.5).sum()), detection_threshold=thr, genes_used=len(cand),
                batches=int(len(np.unique(bt))))
    tech = pd.get_dummies(pd.Categorical(bt), dtype=float).to_numpy()[:, 1:]
    hr = draw_hour(S["collection_timestamp"])  # v1.1: local blood-draw hour
    tech = draw_tech(tech, hr, info, "rna")
    date = day(S["collection_date"]).fillna(day(S["collection_timestamp"])).values
    rna = finish("rna", S["participant_id"], S["research_stage"], date, D[:, :len(cand)], gnames[:len(cand)], info, tech)
    rna["draw_hour"] = hr
    Z = D / np.where(D.std(0) > 0, D.std(0), np.nan)
    mods, used = [], {}
    for mname, gl in MODULES.items():
        js = [gnames.index(g) for g in gl if g in gnames]
        used[mname] = len(js)
        mods.append(np.nanmean(Z[:, js], 1) if len(js) >= 2 else np.full(len(Z), np.nan))
    mi = {"genes_used": used}
    X = np.column_stack(mods)
    okm = np.isfinite(X).any(0)
    if "draw_time_adjusted" in info:
        mi["draw_time_adjusted"] = info["draw_time_adjusted"]
    mod = finish("rnamod", S["participant_id"], S["research_stage"], date, X[:, okm],
                 [n for n, k in zip(MODULES, okm) if k], mi, tech)
    mod["draw_hour"] = hr
    return rna, mod


# ============================================================================ matching

def match(L, fr):
    """Per atlas visit row: sample index (-1 = none), stage-match flag, |gap| days."""
    s = pd.DataFrame({"participant_id": L["pid"], "s_stage": L["stage"], "s_date": L["date"], "si": np.arange(len(L["pid"]))})
    v = fr[["participant_id", "research_stage", "visit_date"]].copy()
    v["fi"] = np.arange(len(fr))
    v["visit_date"] = day(v["visit_date"]).values
    c = v.merge(s, on="participant_id")
    c["gap"] = (c["s_date"] - c["visit_date"]).dt.days.abs()
    c["same"] = c["s_stage"] == c["research_stage"]
    c = c[c["same"] | (c["gap"] <= MAXGAP)].copy()
    c["g2"] = c["gap"].fillna(1e6)
    c = c.sort_values(["fi", "same", "g2"], ascending=[True, False, True]).drop_duplicates("fi")
    n, fi = len(fr), c["fi"].to_numpy()
    si, same, gap = np.full(n, -1), np.zeros(n, bool), np.full(n, np.nan)
    si[fi], same[fi], gap[fi] = c["si"].to_numpy(), c["same"].to_numpy(bool), c["gap"].to_numpy(float)
    M = pd.DataFrame({"si": si, "same": same, "gap": gap})
    used = np.unique(M.loc[M.si >= 0, "si"])
    info = {"visits_matched_stage": A.rc(M["same"].sum()),
            "visits_matched_date": A.rc(((M.si >= 0) & ~M["same"]).sum()),
            "participants_matched": A.rc(fr.loc[M.si.to_numpy() >= 0, "participant_id"].nunique()),
            "samples_unmatched": A.rc(len(L["pid"]) - len(used)),
            "gap_days_stage_matched": A.qtiles(M.loc[M["same"], "gap"]),
            "gap_days_date_matched": A.qtiles(M.loc[(M.si >= 0) & ~M["same"], "gap"])}
    if L.get("draw_hour") is not None and "acq_hour" in fr.columns:  # v1.1: draw hour vs vascular acquisition hour
        w = M["si"].to_numpy() >= 0
        dh = np.full(len(fr), np.nan)
        dh[w] = np.asarray(L["draw_hour"], float)[M["si"].to_numpy()[w]]
        ah = pd.to_numeric(fr["acq_hour"], errors="coerce").to_numpy(float)
        ok = np.isfinite(dh) & np.isfinite(ah) & M["same"].to_numpy()
        if ok.sum() >= A.MINCELL:
            info["draw_vs_acq_hour"] = {"n": A.rc(ok.sum()), "r": float(np.corrcoef(dh[ok], ah[ok])[0, 1]),
                                        "acq_minus_draw_h": A.qtiles(pd.Series(ah[ok] - dh[ok]))}
    return M, info


def rows_for(fr, M, need):
    """One atlas row per participant with a matched sample and all `need` columns (stage match, gap, visit 1)."""
    d = fr[need].replace([np.inf, -np.inf], np.nan)
    ok = (M["si"].to_numpy() >= 0) & d.notna().all(1).to_numpy()
    d = fr.loc[ok, ["participant_id", "research_stage"] + [c for c in dict.fromkeys(need)]].copy()
    d["_si"] = M["si"].to_numpy()[ok]
    d["_same"] = M["same"].to_numpy()[ok]
    d["_gap"] = M["gap"].to_numpy()[ok]
    d["_ord"] = (d["research_stage"] != A.MAIN_STAGES[0]).astype(int)
    d = d.sort_values(["_same", "_gap", "_ord"], ascending=[False, True, True], na_position="last")
    return d.drop_duplicates("participant_id")


def covmat(d, covs, L):
    C = [d[c].astype(float).to_numpy() for c in covs]
    if L.get("tech") is not None:
        C += list(L["tech"][d["_si"].to_numpy()].T)
    C = np.column_stack(C) if C else np.zeros((len(d), 0))
    sd = C.std(0)
    C = (C[:, sd > 0] - C[:, sd > 0].mean(0)) / sd[sd > 0]
    return np.column_stack([np.ones(len(d)), C])


# ============================================================================ vectorised masked OLS

def mols(Y, X, C, min_n=MIN_N):
    """OLS of each outcome (columns of Y, fully observed) on each feature (columns of X; NaN = missing) plus the
    covariates C (first column = intercept), using each feature's observed rows.  Features are standardized within
    their rows.  Returns b (p, q), se (p, q) [HC1], nobs (p,), cov (p, q, q) of the q betas of one feature."""
    n, p = X.shape
    k, q = C.shape[1], Y.shape[1]
    Mk = np.isfinite(X)
    Mf = Mk.astype(float)
    nj = Mf.sum(0)
    X0 = np.where(Mk, X, 0.0)
    mu = X0.sum(0) / np.maximum(nj, 1)
    Xs = np.where(Mk, X0 - mu, 0.0)
    del X0
    sd = np.sqrt((Xs ** 2).sum(0) / np.maximum(nj - 1, 1))
    Xs /= np.where(sd > 0, sd, 1.0)
    G = (Mf.T @ (C[:, :, None] * C[:, None, :]).reshape(n, k * k)).reshape(p, k, k)
    Am = np.empty((p, k + 1, k + 1))
    Am[:, 0, 0] = (Xs ** 2).sum(0)
    Cx = Xs.T @ C
    Am[:, 0, 1:] = Cx
    Am[:, 1:, 0] = Cx
    Am[:, 1:, 1:] = G
    Ai = np.linalg.pinv(Am, rcond=1e-10)
    Rh = np.concatenate([(Xs.T @ Y)[:, None, :],
                         (Mf.T @ (C[:, :, None] * Y[:, None, :]).reshape(n, k * q)).reshape(p, k, q)], 1)
    Bt = Ai @ Rh
    W = (Xs * Ai[:, 0, 0] + C @ Ai[:, 0, 1:].T) * Mf
    adj = nj / np.maximum(nj - k - 1, 1)
    WE = [W * (Y[:, [j]] - Xs * Bt[:, 0, j] - C @ Bt[:, 1:, j].T) for j in range(q)]
    del W
    cov = np.empty((p, q, q))
    for a in range(q):
        for c in range(a, q):
            cov[:, a, c] = cov[:, c, a] = (WE[a] * WE[c]).sum(0) * adj
    b = Bt[:, 0, :].copy()
    se = np.sqrt(np.clip(np.einsum("pjj->pj", cov), 0, None))
    bad = (nj < min_n) | ~(sd > 0)
    b[bad], se[bad], cov[bad] = np.nan, np.nan, np.nan
    return b, se, nj, cov


def pack(b, se, nj, n):
    p = pnorm(b, se)
    q = bh(p)
    out = {"n": A.rc(n), "b": b, "se": se, "q": q, "n_q05": int(np.nansum(q < 0.05)), "n_q10": int(np.nansum(q < 0.10))}
    if np.nanmin(nj) != np.nanmax(nj):
        out["nobs"] = [A.rc(x) for x in nj]
    return out


# ============================================================================ A: per-bed associations

def sec_assoc(fr, layers, Ms, beds):
    out = {}
    for ln, L in layers.items():
        t0 = time.time()
        r = {"features": L["names"], "beds": {}}
        try:
            for bed in beds:
                rb = {}
                for mn, covs in MODELS.items():
                    covs = [c for c in covs if c in fr.columns]
                    d = rows_for(fr, Ms[ln], ["o_" + bed] + covs)
                    if len(d) < MIN_N:
                        continue
                    y = A.zs(d["o_" + bed]).to_numpy()[:, None]
                    b, se, nj, _ = mols(y, L["X"][d["_si"].to_numpy()], covmat(d, covs, L))
                    rb[mn] = pack(b[:, 0], se[:, 0], nj, len(d))
                r["beds"][bed] = rb
        except Exception as ex:
            r["error"] = err(ex)
            log(f"A {ln} FAILED: {r['error']}")
        out[ln] = r
        log(f"A {ln}: {len(L['names'])} features x {len(r['beds'])} beds in {time.time() - t0:.1f}s")
    return out


def hyp_oral(assoc):
    """Pre-stated periodontal taxa x core beds; BH q over the 4 taxa x 4 beds within each model."""
    r = assoc.get("oral") or {}
    names = r.get("features") or []
    out = {}
    for mn in MODELS:
        rows = []
        for h in HYP:
            js = [i for i, n in enumerate(names) if h in n]
            for bed in A.CORE:
                e = (r.get("beds", {}).get(bed) or {}).get(mn)
                if js and e:
                    j = js[0]
                    rows.append((h, bed, e["b"][j], e["se"][j]))
        if not rows:
            continue
        p = pnorm(np.array([x[2] for x in rows], float), np.array([x[3] for x in rows], float))
        q = bh(p)
        for (h, bed, b_, s_), pi, qi in zip(rows, p, q):
            out.setdefault(h, {}).setdefault(bed, {})[mn] = {"b": b_, "lo": b_ - 1.96 * s_, "hi": b_ + 1.96 * s_,
                                                             "p": pi, "q_hyp": qi}
    return out


# ============================================================================ B: bed contrasts

def sec_contrasts(fr, layers, Ms, assoc):
    out = {}
    core = A.CORE
    Cm = np.array([[1, -1, 0, 0], [1, 0, -1, 0], [1, 0, 0, -1]], float)
    for ln, L in layers.items():
        r = {}
        try:
            for mn, covs in MODELS.items():
                covs = [c for c in covs if c in fr.columns]
                d = rows_for(fr, Ms[ln], ["o_" + b for b in core] + covs)
                if len(d) < 200:
                    r[mn] = {"n": A.rc(len(d)), "status": "n<200"}
                    continue
                Y = np.column_stack([A.zs(d["o_" + b]).to_numpy() for b in core])
                B, SE, nj, V = mols(Y, L["X"][d["_si"].to_numpy()], covmat(d, covs, L))
                rm = {"n": A.rc(len(d)), "b": {b: B[:, i] for i, b in enumerate(core)}, "pairs": {}}
                for a_, c_ in PAIRS:
                    i, j = core.index(a_), core.index(c_)
                    dd = B[:, i] - B[:, j]
                    s = np.sqrt(np.clip(V[:, i, i] + V[:, j, j] - 2 * V[:, i, j], 0, None))
                    rm["pairs"][f"{a_}-{c_}"] = {"d": dd, "se": s, "q": bh(pnorm(dd, s))}
                cb = B @ Cm.T
                S = np.einsum("ab,pbc,dc->pad", Cm, V, Cm)
                ok = np.isfinite(cb).all(1)
                w = np.full(len(cb), np.nan)
                if ok.any():
                    w[ok] = np.einsum("pa,pa->p", cb[ok], (np.linalg.pinv(S[ok]) @ cb[ok][:, :, None])[:, :, 0])
                ph = stats.chi2.sf(w, 3)
                rm["het_q"] = bh(ph)
                rm["n_het_q05"] = int(np.nansum(rm["het_q"] < 0.05))
                rm["bed_specific"] = flags(L["names"], assoc.get(ln, {}), mn, rm)
                r[mn] = rm
        except Exception as ex:
            r["error"] = err(ex)
            log(f"B {ln} FAILED: {r['error']}")
        out[ln] = r
    return out


def flags(names, ra, mn, rm):
    core = A.CORE
    qb = {}
    for b in core:
        e = (ra.get("beds", {}).get(b) or {}).get(mn)
        if not e:
            return {"status": f"per-bed q missing for {b}"}
        qb[b] = np.nan_to_num(np.asarray(e["q"], float), nan=1.0)
    sig = np.column_stack([qb[b] < 0.05 for b in core])
    one = sig.sum(1) == 1
    res = []
    for i, b in enumerate(core):
        pq_ = []
        for k_, v in rm["pairs"].items():
            a_, c_ = k_.split("-")
            if b in (a_, c_):
                pq_.append(np.nan_to_num(np.asarray(v["q"], float), nan=1.0))
        pq_ = np.column_stack(pq_)
        strict = sig[:, i] & one & (pq_ < 0.05).all(1)
        lenient = sig[:, i] & one & (pq_ < 0.05).any(1)
        for j in np.flatnonzero(lenient):
            res.append({"feature": names[j], "bed": b, "strict": bool(strict[j]),
                        "b_common": {c: rm["b"][c][j] for c in core}, "q_bed": {c: qb[c][j] for c in core},
                        "q_pairs": {k_: v["q"][j] for k_, v in rm["pairs"].items() if b in k_.split("-")}})
    res.sort(key=lambda x: (not x["strict"], min(x["q_pairs"].values())))
    return {"n_strict": sum(x["strict"] for x in res), "n_lenient": len(res), "features": res[:60]}


# ============================================================================ C1: molecular vascular ages

def resid_age_sex(d, col):
    a = d["age"].astype(float).to_numpy()
    X = np.column_stack([np.ones(len(d)), a, (a - 50) ** 2, d["sex_male"].astype(float).to_numpy()])
    y = d[col].astype(float).to_numpy()
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    return (e - e.mean()) / e.std()


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)


def stdz(X, mu=None, sd=None):
    if mu is None:
        with np.errstate(invalid="ignore"):
            mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
        mu = np.nan_to_num(mu)
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return np.nan_to_num((X - mu) / sd), mu, sd


def enet_fit(X, y, seed):
    """Elastic net with inner 3-fold CV over 20 alphas x l1 ratio 0.1/0.5/0.9 on training-standardized features
    (missing -> training mean)."""
    from sklearn.linear_model import ElasticNetCV
    Z, mu, sd = stdz(X)
    m = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], n_alphas=20, cv=3, max_iter=2000, tol=1e-3, selection="random",
                     random_state=seed).fit(Z, y)
    return lambda Xn: m.predict(stdz(Xn, mu, sd)[0]), m


def enet_cv(X, y, seed, folds=5):
    """Nested CV: outer 5 folds (out-of-fold predictions); all tuning and screening inside each training fold."""
    from sklearn.model_selection import KFold
    oof, fr2 = np.full(len(y), np.nan), []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        pred = enet_fit(X[tr], y[tr], seed)[0]
        oof[te] = pred(X[te])
        fr2.append(r2(y[te], oof[te]))
    pred, m = enet_fit(X, y, seed)
    return oof, fr2, pred, m, m.coef_


def sec_clocks(fr, layers, Ms, boot, seed, budget):
    rng = np.random.default_rng(seed)
    out = {}
    pool = {k: v for k, v in layers.items() if k != "rna"}
    comb = combined_layer(fr, layers, Ms)
    if comb is not None:
        pool["combined"] = comb[0]
        Ms = dict(Ms, combined=comb[1])
    if "rna" in layers:
        pool["rna"] = layers["rna"]  # slowest last
    tc = time.time()
    for ln, L in pool.items():
        t0 = time.time()
        r = {}
        if t0 - tc > budget:
            out[ln] = {"status": "skipped: C1 time budget"}
            log(f"C1 {ln} skipped (time budget)")
            continue
        try:
            nf = MAXP_RNA if ln == "rna" else L["X"].shape[1]  # rna genes are ordered by mean CPM
            anyrow = rows_for(fr, Ms[ln], ["age", "sex_male"])
            Xany = L["X"][anyrow["_si"].to_numpy()][:, :nf]
            preds = {}
            for bed in A.CORE:
                d = rows_for(fr, Ms[ln], ["o_" + bed, "age", "sex_male"])
                if len(d) < 200:
                    r[bed] = {"n": A.rc(len(d)), "status": "n<200"}
                    continue
                y = resid_age_sex(d, "o_" + bed)
                X = L["X"][d["_si"].to_numpy()]
                oof, fr2, pred, m, coef = enet_cv(X[:, :nf], y, seed)
                bs = [r2(y[i], oof[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(boot))]
                top = [j for j in np.argsort(-np.abs(coef))[:10] if coef[j] != 0]
                r[bed] = {"n": A.rc(len(d)), "r2": r2(y, oof), "r2_ci": A.pct(bs), "fold_sd": float(np.std(fr2)),
                          "alpha": float(m.alpha_), "l1_ratio": float(m.l1_ratio_), "n_nonzero": int((coef != 0).sum()),
                          "top": [[L["names"][j], round(float(coef[j]), 4)] for j in top]}
                pa_ = pred(Xany)
                r[bed]["n_features"] = nf
                o = pd.Series(oof, index=d["participant_id"].to_numpy())
                hit = anyrow["participant_id"].isin(o.index).to_numpy()
                pa_[hit] = o.reindex(anyrow["participant_id"].to_numpy()[hit]).to_numpy()
                preds[bed] = pa_
            if len(preds) >= 2:
                P = pd.DataFrame(preds)
                r["discordance"] = {"n": A.rc(len(P)), "r": {f"{a}-{b}": float(P[a].corr(P[b]))
                                                            for i, a in enumerate(P) for b in list(P)[i + 1:]}}
        except Exception as ex:
            r["error"] = err(ex)
            log(f"C1 {ln} FAILED: {r['error']}")
        out[ln] = r
        log(f"C1 {ln} in {time.time() - t0:.1f}s")
    if comb is not None:
        out["combined_layers"] = comb[2]
    return out


def combined_layer(fr, layers, Ms):
    """All layers except gene-level RNA (modules instead): per atlas visit, the matched samples' features (NaN when
    a layer is absent) plus layer-presence indicators; visits with >= 2 layers."""
    use = [ln for ln in layers if ln != "rna"]
    if len(use) < 2:
        return None
    blocks, names, nl = [], [], np.zeros(len(fr))
    for ln in use:
        si = Ms[ln]["si"].to_numpy()
        X = np.full((len(fr), layers[ln]["X"].shape[1]), np.nan)
        X[si >= 0] = layers[ln]["X"][si[si >= 0]]
        blocks += [X, (si >= 0).astype(float)[:, None]]
        names += [f"{ln}:{n}" for n in layers[ln]["names"]] + [f"has_{ln}"]
        nl += si >= 0
    L = {"name": "combined", "X": np.column_stack(blocks), "names": names, "tech": None}
    M = pd.DataFrame({"si": np.where(nl >= 2, np.arange(len(fr)), -1), "same": True, "gap": -nl})
    return L, M, {"layers": use, "visits_2plus_layers": A.rc((nl >= 2).sum())}


# ============================================================================ C2: subtypes

def cellok(s):
    """Mean reportable: >= MINCELL values and, for 0/1 variables, >= MINCELL in each category."""
    s = s.dropna().astype(float)
    if len(s) < A.MINCELL:
        return False
    return not s.isin([0, 1]).all() or min(s.sum(), len(s) - s.sum()) >= A.MINCELL


def sec_subtypes(fr, layers, Ms, nboot, seed):
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    need = ["o_" + b for b in A.CORE] + ["age", "sex_male"]
    d = A.one_row_per_participant(fr, need)
    out = {"n": A.rc(len(d))}
    if len(d) < 200:
        return {**out, "status": "n<200"}
    Z = np.column_stack([resid_age_sex(d, "o_" + b) for b in A.CORE])
    bic, fits = {}, {}
    for k in range(1, 7):
        g = GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(Z)
        bic[k], fits[k] = float(g.bic(Z)), g
    kb = min(range(2, 7), key=lambda k: bic[k])
    lab0 = fits[kb].predict(Z)
    order = np.argsort(-np.bincount(lab0, minlength=kb))
    lab = np.argsort(order)[lab0]
    rng = np.random.default_rng(seed)
    ari = []
    for i in range(nboot):
        idx = rng.integers(0, len(Z), len(Z))
        g = GaussianMixture(kb, covariance_type="full", n_init=2, random_state=seed + i + 1).fit(Z[idx])
        ari.append(adjusted_rand_score(lab, g.predict(Z)))
    out.update(bic=bic, k_best=kb, k1_bic_lower=bool(bic[1] < bic[kb]), ari_median=float(np.median(ari)) if ari else None,
               ari_ci=A.pct(ari))
    cl = {}
    big = []
    for c in range(kb):
        w = lab == c
        e = {"n": A.rc(w.sum())}
        if w.sum() >= A.MINCELL:
            big.append(c)
            e["bed_means_z"] = {b: float(Z[w, i].mean()) for i, b in enumerate(A.CORE)}
            e["exposures"] = {x: float(d.loc[w, x].mean()) for x in EXPO if x in d.columns and cellok(d.loc[w, x])}
        cl[f"C{c + 1}"] = e
    out["clusters"] = cl
    ptest = {}
    for x in EXPO:
        if x in d.columns:
            gs = [d.loc[(lab == c), x].dropna().astype(float) for c in big]
            gs = [g for g in gs if len(g) >= A.MINCELL]
            if len(gs) >= 2:
                ptest[x] = float(stats.f_oneway(*gs).pvalue)
    out["exposure_anova_p"] = ptest
    om = {}
    d = d.assign(_lab=lab)
    for ln, L in layers.items():
        try:
            si = Ms[ln]["si"].reindex(d.index).to_numpy()
            ok = si >= 0
            if ok.sum() < 200:
                om[ln] = {"n": A.rc(ok.sum()), "status": "n<200"}
                continue
            dd = d.loc[ok].assign(_si=si[ok])
            X = L["X"][si[ok]]
            C = covmat(dd, ["age", "sex_male"], L)
            r = {"n": A.rc(len(dd))}
            for c in big:
                y = (dd["_lab"].to_numpy() == c).astype(float)
                if y.sum() < A.MINCELL or (1 - y).sum() < A.MINCELL:
                    continue
                b, se, nj, _ = mols(y[:, None], X, C)
                q = bh(pnorm(b[:, 0], se[:, 0]))
                with np.errstate(invalid="ignore", divide="ignore"):
                    m1, m0 = np.nanmean(X[y == 1], 0), np.nanmean(X[y == 0], 0)
                    smd = (m1 - m0) / np.nanstd(X, 0)
                top = [j for j in np.argsort(np.nan_to_num(q, nan=2.0)) if np.isfinite(q[j])][:10]
                r[f"C{c + 1}"] = {"n_q05": int(np.nansum(q < 0.05)),
                                  "top": [[L["names"][j], smd[j], q[j]] for j in top]}
            om[ln] = r
        except Exception as ex:
            om[ln] = {"error": err(ex)}
    out["omics"] = om
    return out


# ============================================================================ digest

def err(ex):
    tb = ([f for f in traceback.extract_tb(ex.__traceback__) if f.filename == __file__] or
          traceback.extract_tb(ex.__traceback__))[-1]
    return f"{type(ex).__name__}: {str(ex)[:160]} (line {tb.lineno})"


def v3(x):
    return A.v3(x)


def bci(b, se):
    try:
        b, se = float(b), float(se)
    except (TypeError, ValueError):
        return "NA"
    return f"{v3(b)}[{v3(b - 1.96 * se)},{v3(b + 1.96 * se)}]"


def digest(res, dm):
    L = [f"F1 OMICS DIGEST {res.get('version')} | atlas {res.get('atlas_version')} | runtime {v3(res.get('runtime_s'))} s",
         "settings: " + ", ".join(f"{k} {v}" for k, v in (res.get("settings") or {}).items()),
         "beds adverse-oriented (higher = worse); b = SD of bed per SD of feature; q = BH within layer x bed x model",
         "## flow"]
    for ln, f in (res.get("flow") or {}).items():
        if "error" in f:
            L.append(f"{ln}: ERROR {f['error']}")
            continue
        gq = f.get("gap_days_date_matched") or {}
        L.append(f"{ln}: samples {f.get('samples')} participants {f.get('participants')} features {f.get('features')}"
                 f" (in {f.get('features_in', f.get('species_total', f.get('genes_total')))}, logged {f.get('logged', '-')}"
                 f", prev10 {f.get('species_prev10', '-')}, detected50 {f.get('genes_detected50', '-')}); matched visits stage "
                 f"{f.get('visits_matched_stage')} date {f.get('visits_matched_date')} participants "
                 f"{f.get('participants_matched')}; unmatched samples {f.get('samples_unmatched')}; date gap d q05/50/95 "
                 f"{v3(gq.get('q05'))}/{v3(gq.get('q50'))}/{v3(gq.get('q95'))}")
        extra = {k: f[k] for k in ("excluded_failed_wgs_validation", "samples_failed_libsize", "batches", "genes_used",
                                   "detection_threshold", "hyp_prevalence", "pseudocount", "draw_time_adjusted")
                 if f.get(k)}
        for k in ("draw_hour", "draw_vs_acq_hour"):  # v1.1
            v = f.get(k)
            if v:
                q = v.get("acq_minus_draw_h", v)
                extra[k] = (f"r {v3(v['r'])} n {v.get('n')}; " if "r" in v else f"n {v.get('n')}; ") + \
                    "q05/25/50/75/95 " + "/".join(v3(q.get(x)) for x in ("q05", "q25", "q50", "q75", "q95"))
        if extra:
            L.append(f"  {ln} qc: {extra}")
    asc = res.get("A_assoc") or {}
    L.append("## n per bed (M1 | M3) and FDR hits q<0.05 M1/M2/M3")
    for ln, r in asc.items():
        if "error" in r:
            L.append(f"{ln}: ERROR {r['error']}")
        L.append(f"{ln}: " + "; ".join(
            f"{b} n {(e.get('M1') or {}).get('n')}|{(e.get('M3') or {}).get('n')} hits "
            + "/".join(str((e.get(m) or {}).get('n_q05', '-')) for m in MODELS) for b, e in (r.get("beds") or {}).items()))
    L.append(f"## top 10 per core bed by p, model {dm}: feature b[95% CI] q")
    for ln, r in asc.items():
        if ln == "rnamod":
            continue
        nm = r.get("features") or []
        for b in A.CORE:
            e = ((r.get("beds") or {}).get(b) or {}).get(dm)
            if not e:
                continue
            bb, se, q = (np.array(e[k], float) for k in ("b", "se", "q"))
            top = [j for j in np.argsort(np.nan_to_num(pnorm(bb, se), nan=2.0)) if np.isfinite(bb[j])][:10]
            items = [f"{short(nm[j], 24)} {bci(bb[j], se[j])} q{v3(q[j])}" for j in top]
            L.append(f"{ln} {b} (n {e.get('n')}, hits {e.get('n_q05')}): " + "; ".join(items[:5]))
            if len(items) > 5:
                L.append("    " + "; ".join(items[5:]))
    mod = asc.get("rnamod") or {}
    if mod.get("features"):
        L.append("## RNA modules (genes used " + str((res.get("flow", {}).get("rnamod") or {}).get("genes_used")) + ")")
        for j, mname in enumerate(mod["features"]):
            for m in ("M1", "M3"):
                L.append(f"{mname} {m}: " + "; ".join(
                    f"{b} {bci(e[m]['b'][j], e[m]['se'][j])} q{v3(e[m]['q'][j])}"
                    for b, e in (mod.get("beds") or {}).items() if b in A.CORE and e.get(m)))
    hyp = res.get("A_hyp_oral") or {}
    L.append("## oral pre-stated taxa (q over 4 taxa x 4 beds): M1 b[CI] p | M3 b[CI] p")
    for h, e in hyp.items():
        L.append(f"{h}: " + "; ".join(
            f"{b} {v3(g(v, 'M1', 'b'))}[{v3(g(v, 'M1', 'lo'))},{v3(g(v, 'M1', 'hi'))}] p{v3(g(v, 'M1', 'p'))} | "
            f"{v3(g(v, 'M3', 'b'))}[{v3(g(v, 'M3', 'lo'))},{v3(g(v, 'M3', 'hi'))}] p{v3(g(v, 'M3', 'p'))} q{v3(g(v, 'M3', 'q_hyp'))}"
            for b, v in e.items()))
    con = res.get("B_contrasts") or {}
    L.append("## common four-bed sample: n; contrast hits q<0.05 per pair; Wald het hits; bed-specific strict/lenient")
    for ln, r in con.items():
        if "error" in r:
            L.append(f"{ln}: ERROR {r['error']}")
        for m in ("M1", "M3"):
            e = r.get(m) or {}
            if "pairs" not in e:
                L.append(f"{ln} {m}: n {e.get('n')} {e.get('status', '')}")
                continue
            fl = e.get("bed_specific") or {}
            L.append(f"{ln} {m}: n {e.get('n')} | " + " ".join(
                f"{k} {int(np.nansum(np.array(v['q'], float) < 0.05))}" for k, v in e["pairs"].items())
                + f" | het {e.get('n_het_q05')} | specific {fl.get('n_strict')}/{fl.get('n_lenient')}")
    L.append(f"## bed-specific features (strict first), model {dm}: feature bed | common-sample b pwv/imt/art_w/abi | min pair q")
    for ln, r in con.items():
        fl = ((r.get(dm) or {}).get("bed_specific") or {}).get("features") or []
        for x in fl[:6]:
            L.append(f"{ln} {short(x['feature'], 26)} {x['bed']}{'' if x['strict'] else ' (lenient)'} | "
                     + "/".join(v3(x["b_common"][c]) for c in A.CORE) + f" | q {v3(min(x['q_pairs'].values()))}")
    clk = res.get("C1_clocks") or {}
    L.append("## C1 molecular vascular ages: out-of-fold R2 [bootstrap CI] fold-sd n | discordance r of predicted gaps")
    for ln, r in clk.items():
        if ln == "combined_layers":
            L.append(f"combined = {r}")
            continue
        if "error" in r:
            L.append(f"{ln}: ERROR {r['error']}")
        L.append(f"{ln}: " + "; ".join(
            f"{b} {v3(e.get('r2'))} {A.ci3(e.get('r2_ci'))} sd {v3(e.get('fold_sd'))} n {e.get('n')}"
            for b, e in r.items() if b in A.CORE) + " | disc " + " ".join(
            f"{k} {v3(v)}" for k, v in ((r.get("discordance") or {}).get("r") or {}).items()))
        L.append(f"  {ln} top: " + "; ".join(
            f"{b}: " + ", ".join(f"{short(t[0], 20)} {v3(t[1])}" for t in (e.get("top") or [])[:3])
            for b, e in r.items() if b in A.CORE and isinstance(e, dict) and e.get("top")))
    st = res.get("C2_subtypes") or {}
    L.append("## C2 subtypes (GMM on age/sex-standardised oriented beds)")
    if "error" in st:
        L.append(f"ERROR {st['error']}")
    L.append(f"n {st.get('n')} k {st.get('k_best')} (BIC k1 lower: {st.get('k1_bic_lower')}) ARI {v3(st.get('ari_median'))} "
             f"{A.ci3(st.get('ari_ci'))}; BIC " + " ".join(f"{k}:{v3(v)}" for k, v in (st.get("bic") or {}).items()))
    for c, e in (st.get("clusters") or {}).items():
        L.append(f"{c} n {e.get('n')} beds z " + " ".join(f"{b} {v3(v)}" for b, v in (e.get("bed_means_z") or {}).items()))
    cls = [c for c, e in (st.get("clusters") or {}).items() if e.get("exposures")]
    for x in EXPO:
        vals = [v3(((st["clusters"][c].get("exposures") or {}).get(x))) for c in cls]
        if any(v != "NA" for v in vals):
            L.append(f"  {x}: " + " ".join(f"{c} {v}" for c, v in zip(cls, vals)) + f" | p {v3((st.get('exposure_anova_p') or {}).get(x))}")
    for ln, r in (st.get("omics") or {}).items():
        for c, e in r.items():
            if isinstance(e, dict) and "top" in e:
                L.append(f"  {ln} {c} (n {r.get('n')}, hits {e['n_q05']}): " + "; ".join(
                    f"{short(t[0], 20)} {v3(t[1])} q{v3(t[2])}" for t in e["top"][:4]))
            elif c in ("error", "status"):
                L.append(f"  {ln}: {c} {e}")
    L.append("## timings s: " + ", ".join(f"{k} {v3(v)}" for k, v in (res.get("timings") or {}).items()))
    for k, v in res.items():
        if isinstance(v, dict) and "error" in v:
            L.append(f"ERROR {k}: {v['error']}")
    return "\n".join(L) + "\n"


def g(d, *ks):
    return A.g(d, *ks)


# ============================================================================ main

def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--atlas", required=True, help="atlas script (e.g. f1_atlas_v1_3.py); its build_frame is used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--boot", type=int, default=200, help="bootstrap replicates for CV R2 CIs")
    ap.add_argument("--gmm-boot", type=int, default=50)
    ap.add_argument("--max-genes", type=int, default=5000)
    ap.add_argument("--c1-budget", type=float, default=720, help="s; later C1 layers are skipped beyond this")
    ap.add_argument("--layers", default="metab,lipid,gut,oral,rna")
    ap.add_argument("--sections", default="A,B,C1,C2")
    ap.add_argument("--digest-model", default="M3", choices=list(MODELS))
    ap.add_argument("--rna-draw-time", action="store_true",
                    help="v1.1: add PBMC blood-draw hour (centred, + square) to the RNA technical covariates")
    ap.add_argument("--draw-time", default="",
                    help="v1.2: comma list of layers (metab,lipid,rna) whose sample hour + hour^2 are added as covariates")
    a = ap.parse_args()
    global DRAW_LAYERS
    DRAW_LAYERS = {x.strip() for x in a.draw_time.split(",") if x.strip()} | ({"rna"} if a.rna_draw_time else set())
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    spec = importlib.util.spec_from_file_location("f1_atlas", a.atlas)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    miss = [x for x in ATLAS_API if not hasattr(A, x)]
    if miss:
        raise SystemExit(f"atlas module lacks {miss}; use atlas v1.2 or later")
    import sklearn
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    t0 = time.time()
    log(f"{VERSION}; atlas {A.VERSION}; numpy {np.__version__}; pandas {pd.__version__}; sklearn {sklearn.__version__}; "
        f"python {sys.version.split()[0]}")
    res = {"version": VERSION, "atlas_version": A.VERSION, "settings": {
        "maxgap_days": MAXGAP, "prevalence": PREV, "min_observed": MINOBS, "max_genes": a.max_genes, "boot": a.boot,
        "gmm_boot": a.gmm_boot, "calendar_adjust": getattr(A, "CAL_ADJUST", None),
        "draw_time_layers": ",".join(sorted(DRAW_LAYERS)) or "none"}, "flow": {}, "timings": {}}
    fr = A.build_frame(a.root, None)[0].reset_index(drop=True)
    res["timings"]["frame"] = time.time() - t0
    pids = set(fr["participant_id"].astype(str))
    beds = A.CORE + [b for b in EXTRA if b in fr.columns and fr[b].notna().sum() >= MIN_N]
    loaders = {"metab": lambda: load_wide(a.root, "nightingale_metabolomics/nightingale_metabolomics.parquet", "metab"),
               "lipid": lambda: load_wide(a.root, "serum_lipidomics/serum_lipidomics.parquet", "lipid"),
               "gut": lambda: load_gut(a.root), "oral": lambda: load_oral(a.root),
               "rna": lambda: load_rna(a.root, a.max_genes)}
    layers, Ms = {}, {}
    for ln in [x.strip() for x in a.layers.split(",") if x.strip()]:
        t1 = time.time()
        try:
            out = loaders[ln]()
            for L in (out if isinstance(out, tuple) else (out,)):
                if L is None:
                    continue
                pids |= set(L["pid"])
                Ms[L["name"]], mi = match(L, fr)
                layers[L["name"]] = L
                res["flow"][L["name"]] = {**L["info"], **mi}
                log(f"layer {L['name']}: {cnt(len(L['pid']))} samples, {L['X'].shape[1]} features; "
                    f"matched visits {cnt((Ms[L['name']].si >= 0).sum())}")
        except Exception as ex:
            res["flow"][ln] = {"error": err(ex)}
            log(f"layer {ln} FAILED: {res['flow'][ln]['error']}")
        res["timings"][f"load_{ln}"] = time.time() - t1
    secs = [s.strip() for s in a.sections.split(",") if s.strip()]
    fns = {"A": ("A_assoc", lambda: sec_assoc(fr, layers, Ms, beds)),
           "B": ("B_contrasts", lambda: sec_contrasts(fr, layers, Ms, res.get("A_assoc") or {})),
           "C1": ("C1_clocks", lambda: sec_clocks(fr, layers, Ms, a.boot, a.seed, a.c1_budget)),
           "C2": ("C2_subtypes", lambda: sec_subtypes(fr, layers, Ms, a.gmm_boot, a.seed))}
    for key in secs:
        name, fn = fns[key]
        t1 = time.time()
        try:
            res[name] = fn()
            if key == "A":
                res["A_hyp_oral"] = hyp_oral(res[name])
            log(f"{name} done in {time.time() - t1:.1f}s")
        except Exception as ex:
            res[name] = {"error": err(ex)}
            log(f"{name} FAILED: {res[name]['error']}")
        res["timings"][name] = time.time() - t1
    res["runtime_s"] = time.time() - t0
    text = json.dumps(clean(res), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    privacy_check(text, pids)
    dg = digest(json.loads(text), a.digest_model)
    privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines; total {time.time() - t0:.1f}s")
    logtext = "\n".join(A.LOG) + "\n"
    privacy_check(logtext, pids)
    os.makedirs(a.out)
    sums = []
    for fn_, tx in {"results.json": text, "digest.txt": dg, "run_log.txt": logtext}.items():
        with open(os.path.join(a.out, fn_), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn_}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s[:16] + "  " + s.split()[-1] for s in sums))


def cnt(n):
    return A.cnt(n)


def clean(o):
    if isinstance(o, np.ndarray):
        o = o.tolist()
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating, int, np.integer, bool, np.bool_)):
        return A.fnum(o)
    return o


if __name__ == "__main__":
    main()
