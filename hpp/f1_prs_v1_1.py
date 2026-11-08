#!/usr/bin/env python3
"""F1 genetic atlas — polygenic scores (BP, adiposity, lipids, glycaemia, OSA) against the vascular beds.

Writes aggregates only.  Needs the atlas module (phenotype frame; `--atlas`, default `f1_atlas_v1.py` in the
same folder; v1.1: pass the calendar-adjusted atlas, e.g. `--atlas f1_atlas_v1_3.py`) and a weights folder
(compact PGS Catalog GRCh37 files: f1pgs_<score>.parquet + *_manifest.tsv, made by
prepare_pgs_weights.py from public data).

Steps
  1. Samples: .fam IID (= participant_id) of the post-QC PLINK files with PCs, minus explicit sample-QC failures.
     Relatives from the KING all-pairs file (streamed); the unrelated set drops one member of every pair with
     kinship > 0.0884; full-sib pairs = first-degree pairs with IBS0 above the parent-offspring cut.
  2. Scoring: weights matched to the post-QC .bim on GRCh37 chr:pos and alleles (direct, swapped, or strand-
     complemented for non-palindromic variants; palindromic variants kept only when the weights carry an
     effect-allele frequency and it agrees with the cohort frequency, |f - 0.5| > 0.1).  Genotypes are read in
     contiguous blocks (count of A1); score = sum of weight x effect-allele dosage.  Coverage per score is reported.
  3. Positive controls: each score against its own trait (SBP, DBP, PP, BMI, WHR, TG, LDL, HbA1c, AHI/ODI/T90).
  4. Atlas: each score against each bed (adverse-oriented, standardized) in the unrelated set, adjusted for age,
     age^2, sex and PC1-10, then also for visit BP and antihypertensives; bed contrasts in the common sample with
     the robust joint covariance and a Wald heterogeneity test.
  5. Within-sibship: pair differences of each bed on the score difference (age and sex differences adjusted).

Run (sandbox `pheno` env):
  python f1_prs_v1.py --root <HPP_DATA_ROOT> \
      --weights <WORK_DIR>/pgs_weights_f1_v1 --out f1_prs_out_v1
Privacy: aggregates only (counts rounded to 10, cells < 20 suppressed); per-person scores never leave memory.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ATLAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "f1_atlas_v1.py")
if "--atlas" in sys.argv[:-1]:
    ATLAS_PATH = os.path.abspath(sys.argv[sys.argv.index("--atlas") + 1])
_spec = importlib.util.spec_from_file_location("f1_atlas", ATLAS_PATH)
A = importlib.util.module_from_spec(_spec)  # phenotype frame, privacy helpers, regression helpers
sys.modules["f1_atlas"] = A
_spec.loader.exec_module(A)

VERSION = "f1_prs_v1.1 2026-10-03"
KIN_RELATED, KIN_FIRST, KIN_DUP = 0.0884, 0.177, 0.354
IBS0_PO_MAX = 0.005
N_PCS = 10
LUT = np.array([[[2, -1, 1, 0][(b >> (2 * j)) & 3] for j in range(4)] for b in range(256)], dtype=np.int8)
COMP = str.maketrans("ACGT", "TGCA")
BIM_COLS = ["chr", "id", "cm", "pos", "a1", "a2"]
log = A.log
rc = A.rc

# score -> own-trait outcome (frame column) for positive controls
OWN = {"SBP_gw": "sbp_sit", "SBP_grs": "sbp_sit", "DBP_gw": "dbp_sit", "DBP_grs": "dbp_sit", "PP_gw": "pp_sit",
       "BMI_gw": "bmi", "BMI_grs": "bmi", "WHR_gw": "whr", "WHRadjBMI_gw": "whr", "WHRadjBMI_grs": "whr",
       "TG_gw": "log_tg", "TG_grs": "log_tg", "LDL_gw": "ldl", "T2D_gw": "hba1c", "OSA_a": "log1p_ahi",
       "OSA_b": "log1p_ahi", "OSA_c": "log1p_ahi"}
BED_OUT = ["pwv", "imt", "art_w", "abi_min", "vein_w", "fd", "sokolow", "cavi0"]
EXTRA_OUT = ["sbp_sit", "dbp_sit", "pp_sit", "bmi", "whr", "waist", "log_vat", "log_tg", "ldl", "hba1c",
             "log1p_ahi", "log1p_odi", "log1p_t90"]


# ============================================================================ PLINK helpers

def read_bim(prefix):
    t = pv.read_csv(f"{prefix}.bim", read_options=pv.ReadOptions(column_names=BIM_COLS, block_size=1 << 26),
                    parse_options=pv.ParseOptions(delimiter="\t"),
                    convert_options=pv.ConvertOptions(column_types={"chr": pa.string(), "id": pa.string(),
                                                                    "cm": pa.float64(), "pos": pa.int64(),
                                                                    "a1": pa.string(), "a2": pa.string()}))
    d = t.select(["pos", "a1", "a2"]).to_pandas()
    d["row"] = np.arange(len(d))
    return d


def read_fam_ids(prefix):
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, usecols=[1], names=["iid"], dtype={"iid": str})
    return fam["iid"].astype(str).to_numpy()


def read_block(fh, start, m, n):
    bpv = (n + 3) // 4
    fh.seek(3 + start * bpv)
    raw = np.frombuffer(fh.read(m * bpv), dtype=np.uint8).reshape(m, bpv)
    return LUT[raw].reshape(m, -1)[:, :n]


# ============================================================================ samples and relatives

def load_samples(root, fam):
    hg = pd.read_parquet(os.path.join(root, "human_genetics/human_genetics.parquet"),
                         columns=["genetic_sex", "pass_het_check", "sample_qc_gp_pass_check"])
    hg = hg.set_axis(hg.index.get_level_values("participant_id").astype(str))
    hg = hg[~hg.index.duplicated()]
    qc_fail = ~(hg["pass_het_check"].astype("boolean").fillna(True).to_numpy(dtype=bool)
                & hg["sample_qc_gp_pass_check"].astype("boolean").fillna(True).to_numpy(dtype=bool))
    pca = pd.read_parquet(os.path.join(root, "human_genetics/pca/pca.parquet"),
                          columns=[f"pc{i}" for i in range(1, N_PCS + 1)])
    pca = pca.set_axis(pca.index.get_level_values("participant_id").astype(str)).astype("float64")
    pca = pca[~pca.index.duplicated()]
    ids = pd.Index(fam).intersection(hg.index[~qc_fail]).intersection(pca.dropna().index)
    cov = pca.reindex(ids)
    cov["gsex"] = pd.to_numeric(hg["genetic_sex"], errors="coerce").reindex(ids).astype(float)
    flow = {"fam": rc(len(fam)), "qc_fail_explicit": rc(int(qc_fail.sum())), "with_pcs_and_qc": rc(len(ids))}
    return cov, flow


def king_pairs(root, ids):
    path = os.path.join(root, "human_genetics/relationship/relationship_king.tsv")
    conv = pv.ConvertOptions(include_columns=["ID1", "ID2", "IBS0", "Kinship"],
                             column_types={"ID1": pa.string(), "ID2": pa.string(), "IBS0": pa.float64(),
                                           "Kinship": pa.float64()})
    reader = pv.open_csv(path, read_options=pv.ReadOptions(block_size=1 << 26),
                         parse_options=pv.ParseOptions(delimiter="\t"), convert_options=conv)
    keep = []
    for batch in reader:
        k = batch.column("Kinship").to_numpy(zero_copy_only=False)
        sel = np.flatnonzero(k > 0.0442)
        if len(sel):
            keep.append(batch.to_pandas().iloc[sel])
    pairs = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=["ID1", "ID2", "IBS0", "Kinship"])
    s = set(ids)
    return pairs[pairs["ID1"].isin(s) & pairs["ID2"].isin(s)].reset_index(drop=True)


def unrelated_set(ids, pairs):
    rel = pairs[pairs["Kinship"] > KIN_RELATED]
    deg = pd.concat([rel["ID1"], rel["ID2"]]).value_counts()
    removed = set()
    for a, b in zip(rel["ID1"].to_numpy(), rel["ID2"].to_numpy()):
        if a in removed or b in removed:
            continue
        removed.add(a if (deg.get(a, 0), a) >= (deg.get(b, 0), b) else b)
    return pd.Index([i for i in ids if i not in removed])


# ============================================================================ scoring

def match_chr(bim, w):
    """Match weights of one chromosome to .bim rows -> DataFrame(row, coef_on_a1, const, kind, af, pal)."""
    m = w.merge(bim, on="pos", how="inner")
    if not len(m):
        return m.assign(coef=[], const=[], kind=[], pal=[])
    ea, oa, a1, a2 = m["ea"], m["oa"], m["a1"].astype(str).str.upper(), m["a2"].astype(str).str.upper()
    ea_c, oa_c = ea.str.translate(COMP), oa.str.translate(COMP)
    pal = (ea == oa_c)
    direct = (ea == a1) & (oa == a2)
    swap = (ea == a2) & (oa == a1)
    fdir = ~pal & (ea_c == a1) & (oa_c == a2)
    fswap = ~pal & (ea_c == a2) & (oa_c == a1)
    kind = np.select([direct, swap, fdir, fswap], ["direct", "swap", "flip_direct", "flip_swap"], "none")
    m = m.assign(kind=kind, pal=pal.values)
    m = m[m["kind"] != "none"].copy()
    eff_is_a1 = m["kind"].isin(["direct", "flip_direct"])
    m["coef"] = np.where(eff_is_a1, m["w"], -m["w"]).astype(np.float64)
    m["const"] = np.where(eff_is_a1, 0.0, 2.0 * m["w"]).astype(np.float64)
    m["eff_is_a1"] = eff_is_a1.values
    # several weight rows can hit one bim row (e.g. multi-allelic); keep the first
    return m.drop_duplicates("row")


def score_all(root, weights, fam_n, block):
    """Return scores (n_samples x n_scores) and coverage per score."""
    base = os.path.join(root, "human_genetics", "genotypes", "plink_postqc")
    names = list(weights)
    S = np.zeros((fam_n, len(names)))
    cov = {s: {"rows_in": int(len(w)), "matched": 0, "direct": 0, "swap": 0, "flip": 0, "pal_kept": 0,
               "pal_dropped": 0} for s, w in weights.items()}
    for ch in range(1, 23):
        prefix = os.path.join(base, f"chr_{ch}")
        if not os.path.exists(prefix + ".bim"):
            continue
        bim = read_bim(prefix)
        per = {}
        for s, w in weights.items():
            wc = w[w["chr"] == ch]
            if len(wc):
                per[s] = match_chr(bim, wc[["pos", "ea", "oa", "w", "af"]])
        rows = np.unique(np.concatenate([p["row"].to_numpy() for p in per.values() if len(p)] or [np.array([], int)]))
        if not len(rows):
            continue
        ridx = {r: i for i, r in enumerate(rows)}
        W = np.zeros((len(rows), len(names)))
        C = np.zeros(len(names))
        pal_rows = {}
        for j, s in enumerate(names):
            p = per.get(s)
            if p is None or not len(p):
                continue
            ii = np.array([ridx[r] for r in p["row"].to_numpy()])
            W[ii, j] = p["coef"].to_numpy()
            C[j] += p.loc[~p["pal"], "const"].sum()
            pm = p[p["pal"]]
            if len(pm):
                pal_rows[s] = (ii[p["pal"].to_numpy()], pm)
            npal = ~p["pal"]
            cov[s]["direct"] += int(((p["kind"] == "direct") & npal).sum())
            cov[s]["swap"] += int(((p["kind"] == "swap") & npal).sum())
            cov[s]["flip"] += int((p["kind"].str.startswith("flip") & npal).sum())
        # stream genotypes for the needed rows (contiguous spans read in blocks)
        freq = np.full(len(rows), np.nan)
        with open(prefix + ".bed", "rb") as fh:
            if fh.read(3) != b"\x6c\x1b\x01":
                raise ValueError("not a SNP-major PLINK .bed")
            i = 0
            while i < len(rows):
                j = i
                while j + 1 < len(rows) and rows[j + 1] - rows[i] < block:
                    j += 1
                start, span = rows[i], rows[j] - rows[i] + 1
                G = read_block(fh, int(start), int(span), fam_n)[rows[i:j + 1] - start].astype(np.float32)
                miss = G < 0
                if miss.any():
                    mu = np.where(miss, np.nan, G)
                    mu = np.nanmean(mu, axis=1)
                    G = np.where(miss, mu[:, None], G)
                freq[i:j + 1] = G.mean(axis=1) / 2
                # palindromic rows: keep only when the weight file's effect-allele frequency agrees with the cohort
                Wb = W[i:j + 1].copy()
                for s, (ii, pm) in pal_rows.items():
                    sel = (ii >= i) & (ii <= j)
                    if not sel.any():
                        continue
                    js = names.index(s)
                    af = pm["af"].to_numpy(dtype=float)[sel]
                    f_a1 = freq[ii[sel]]
                    f_eff = np.where(pm["eff_is_a1"].to_numpy()[sel], f_a1, 1 - f_a1)
                    keep = np.isfinite(af) & (np.abs(af - 0.5) > 0.1) & (np.abs(f_eff - af) < 0.1)
                    Wb[ii[sel][~keep] - i, js] = 0.0
                    C[js] += pm["const"].to_numpy()[sel][keep].sum()
                    cov[s]["pal_kept"] += int(keep.sum())
                    cov[s]["pal_dropped"] += int((~keep).sum())
                S += (G.T @ Wb.astype(np.float32)).astype(np.float64)
                i = j + 1
        log(f"chr{ch}: {A.cnt(len(rows))} variant rows scored")
    S += C[None, :]
    for s in names:
        c = cov[s]
        c["matched"] = c["direct"] + c["swap"] + c["flip"] + c["pal_kept"]
        c["coverage"] = c["matched"] / c["rows_in"] if c["rows_in"] else None
    return S, cov


def variant_counts_text(cov):
    """Variant counts (not person counts) as thousands-separated strings, so no long digit runs are written."""
    return {s: {k: (f"{v:,}" if isinstance(v, (int, np.integer)) else v) for k, v in c.items()} for s, c in cov.items()}


# ============================================================================ analyses

def assoc(d, y, x, covs, min_n=200):
    dd = d[[y, x] + covs].replace([np.inf, -np.inf], np.nan).dropna()
    if len(dd) < min_n or dd[x].std() == 0:
        return None
    X = np.column_stack([np.ones(len(dd)), A.zs(dd[x]).values] + [dd[c].astype(float).values for c in covs])
    yy = A.zs(dd[y]).values
    b, se, r2 = A.ols(yy, X)
    X0 = np.column_stack([np.ones(len(dd))] + [dd[c].astype(float).values for c in covs])
    b0, _, r20 = A.ols(yy, X0)
    return {"n": rc(len(dd)), "b": b[1], "se": se[1], "lo": b[1] - 1.96 * se[1], "hi": b[1] + 1.96 * se[1],
            "p": A.p_norm(b[1], se[1]), "delta_r2": r2 - r20}


def sib_within(d, pairs_fs, y, x):
    a, b = pairs_fs["ID1"].to_numpy(), pairs_fs["ID2"].to_numpy()
    dd = d.set_index("participant_id")
    have = np.isin(a, dd.index) & np.isin(b, dd.index)
    a, b = a[have], b[have]
    if len(a) < 50:
        return {"n_pairs": rc(len(a))}
    ya, yb = dd.loc[a, y].to_numpy(float), dd.loc[b, y].to_numpy(float)
    xa, xb = dd.loc[a, x].to_numpy(float), dd.loc[b, x].to_numpy(float)
    dage = dd.loc[a, "age"].to_numpy(float) - dd.loc[b, "age"].to_numpy(float)
    dsex = dd.loc[a, "sex_male"].to_numpy(float) - dd.loc[b, "sex_male"].to_numpy(float)
    sdy, sdx = np.nanstd(dd[y].to_numpy(float)), np.nanstd(dd[x].to_numpy(float))
    Y, Xd = (ya - yb) / sdy, np.column_stack([(xa - xb) / sdx, dage, dsex])
    ok = np.isfinite(Y) & np.isfinite(Xd).all(axis=1)
    if ok.sum() < 50:
        return {"n_pairs": rc(ok.sum())}
    bb, se, _ = A.ols(Y[ok], Xd[ok])
    return {"n_pairs": rc(ok.sum()), "b": bb[0], "se": se[0], "lo": bb[0] - 1.96 * se[0], "hi": bb[0] + 1.96 * se[0],
            "p": A.p_norm(bb[0], se[0])}


def digest(res):
    g, v3, cf = A.g, A.v3, A.cf
    L = [f"F1 PRS DIGEST  {res.get('version')}  atlas module {res.get('atlas_module')}  calendar_adjust "
         f"{res.get('calendar_adjust')}", f"flow {res.get('flow')}"]
    L.append("# coverage: score | rows in | matched | coverage | direct swap flip pal kept/dropped | weights SHA ok")
    for s, c in (res.get("coverage") or {}).items():
        L.append(f"{s} | {c.get('rows_in')} | {c.get('matched')} | {v3(c.get('coverage'))} | {c.get('direct')} "
                 f"{c.get('swap')} {c.get('flip')} {c.get('pal_kept')}/{c.get('pal_dropped')} | "
                 f"{g(res, 'weights', s, 'sha256_ok')}")
    L.append("# positive controls (unrelated; beta per SD score, outcome SD units) | delta R2")
    for s, c in (res.get("positive_controls") or {}).items():
        L.append(f"{s} -> {c.get('outcome')}: {cf(c)} n {c.get('n')} dR2 {v3(c.get('delta_r2'))}")
    for key, title in (("beds", "beds (age, age2, sex, PC1-10)"), ("beds_bp_adjusted", "beds + visit SBP, DBP, antihtn"),
                       ("within_sib", "within full-sib pairs")):
        L.append(f"# {title}: score | " + " | ".join(A.CORE + ["sokolow"]))
        for s, beds in (res.get(key) or {}).items():
            L.append(f"{s} | " + " | ".join(cf(beds.get(b)) for b in A.CORE + ["sokolow"]) +
                     (f" | n_pairs {g(beds, 'pwv', 'n_pairs')}" if key == "within_sib" else ""))
    L.append("# bed contrasts (oriented, common sample): score | n | pwv-imt | pwv-art_w | Wald p")
    for s, c in (res.get("bed_contrasts") or {}).items():
        L.append(f"{s} | {c.get('n')} | {cf(g(c, 'diff', 'pwv_minus_imt'))} | {cf(g(c, 'diff', 'pwv_minus_art_w'))} | "
                 f"{v3(g(c, 'heterogeneity', 'p'))}")
    L.append("# other outcomes (unrelated): score | sbp_sit | bmi | log_tg | hba1c | log1p_ahi | log_vat")
    for s, beds in (res.get("beds") or {}).items():
        L.append(f"{s} | " + " | ".join(cf(beds.get(b)) for b in ("sbp_sit", "bmi", "log_tg", "hba1c", "log1p_ahi", "log_vat")))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=20000, help="max variant rows per genotype read")
    ap.add_argument("--atlas", default=ATLAS_PATH, help="atlas module file (read at import time)")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    t0 = time.time()
    log(f"{VERSION}; atlas module {A.VERSION}; numpy {np.__version__}; pandas {pd.__version__}")
    mpath = os.path.join(a.weights, "manifest.tsv")
    if not os.path.exists(mpath):
        cands = sorted(f for f in os.listdir(a.weights) if f.endswith("_manifest.tsv"))
        if len(cands) != 1:
            raise SystemExit(f"need exactly one *_manifest.tsv in {a.weights}, found {len(cands)}")
        mpath = os.path.join(a.weights, cands[0])
    man = pd.read_csv(mpath, sep="\t")
    weights, wcheck = {}, {}
    for r in man.itertuples(index=False):
        p = os.path.join(a.weights, r.file)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        wcheck[r.score] = {"sha256_ok": h == r.sha256, "pgs_id": r.pgs_id, "trait": r.trait}
        if h != r.sha256:
            log(f"weights {r.score}: SHA-256 mismatch, skipped")
            continue
        weights[r.score] = pd.read_parquet(p)
    log(f"weights loaded: {len(weights)} scores")

    base = os.path.join(a.root, "human_genetics", "genotypes", "plink_postqc")
    fam = read_fam_ids(os.path.join(base, "chr_1"))
    cov, flow = load_samples(a.root, fam)
    S, coverage = score_all(a.root, weights, len(fam), a.block)
    scores = pd.DataFrame(S, index=fam, columns=list(weights)).loc[cov.index]
    pairs = king_pairs(a.root, cov.index)
    unrel = unrelated_set(cov.index, pairs)
    k, i0 = pairs["Kinship"].to_numpy(), pairs["IBS0"].to_numpy()
    fs = pairs[(k > KIN_FIRST) & (k <= KIN_DUP) & (i0 >= IBS0_PO_MAX)]
    flow.update({"unrelated": rc(len(unrel)), "full_sib_pairs": rc(len(fs)), "related_pairs": rc(len(pairs))})
    log(f"samples: {A.cnt(len(cov))}; unrelated {A.cnt(len(unrel))}; full-sib pairs {A.cnt(len(fs))}")

    fr, _, _ = A.build_frame(a.root, None)
    outs = [c for c in BED_OUT + EXTRA_OUT if c in fr.columns]
    d = A.one_row_per_participant(fr, ["age", "sex_male"])
    d = d[d["participant_id"].isin(scores.index)].copy()
    d = d.join(scores, on="participant_id").join(cov[[f"pc{i}" for i in range(1, N_PCS + 1)]], on="participant_id")
    d["age2"] = (d["age"] - d["age"].mean()) ** 2
    for b_ in outs:
        if b_ in A.ORIENT:
            d["o_" + b_] = A.ORIENT[b_] * d[b_]
    base_cov = ["age", "age2", "sex_male"] + [f"pc{i}" for i in range(1, N_PCS + 1)]
    du = d[d["participant_id"].isin(unrel)]
    pids = set(fr["participant_id"].astype(str))
    res = {"version": VERSION, "atlas_module": A.VERSION, "calendar_adjust": bool(getattr(A, "CAL_ADJUST", False)),
           "flow": flow, "weights": wcheck,
           "coverage": variant_counts_text(coverage),
           "score_corr_unrelated": A.clean(du[list(weights)].corr().round(3).to_dict()),
           "positive_controls": {}, "beds": {}, "beds_bp_adjusted": {}, "bed_contrasts": {}, "within_sib": {}}
    for s in weights:
        y = OWN.get(s)
        if y in du.columns:
            res["positive_controls"][s] = {"outcome": y, **(assoc(du, y, s, base_cov) or {})}
        res["beds"][s], res["beds_bp_adjusted"][s], res["within_sib"][s] = {}, {}, {}
        for b_ in outs:
            yo = "o_" + b_ if b_ in A.ORIENT else b_
            res["beds"][s][b_] = assoc(du, yo, s, base_cov)
            if b_ in BED_OUT:
                res["beds_bp_adjusted"][s][b_] = assoc(du, yo, s, base_cov + ["sbp_sit", "dbp_sit", "antihtn"])
                res["within_sib"][s][b_] = sib_within(d, fs, yo, s)
        for y in ("sbp_sit", "bmi"):
            if y in d.columns:
                res["within_sib"][s][y] = sib_within(d, fs, y, s)
        # bed contrasts in the common sample (four core beds, unrelated)
        oc = ["o_" + b_ for b_ in A.CORE]
        cm = du[oc + [s] + base_cov].replace([np.inf, -np.inf], np.nan).dropna()
        if len(cm) >= 300:
            X = np.column_stack([np.ones(len(cm)), A.zs(cm[s]).values] + [cm[c].astype(float).values for c in base_cov])
            Y = np.column_stack([A.zs(cm[c]).values for c in oc])
            bvec, V = A.multi_fit(Y, X)
            con, het = A.contrasts(bvec, V, A.CORE, [("pwv", "imt"), ("pwv", "art_w"), ("pwv", "abi_min"),
                                                     ("imt", "art_w"), ("imt", "abi_min"), ("art_w", "abi_min")])
            res["bed_contrasts"][s] = {"n": rc(len(cm)), "beta_oriented": dict(zip(A.CORE, bvec)), "diff": con,
                                       "heterogeneity": het}
    res["runtime_s"] = time.time() - t0
    text = json.dumps(A.clean(res), ensure_ascii=False, indent=1, sort_keys=True)
    A.privacy_check(text, pids)
    logtext = "\n".join(A.LOG) + "\n"
    A.privacy_check(logtext, pids)
    dg = digest(json.loads(text))
    A.privacy_check(dg, pids)
    os.makedirs(a.out)
    sums = []
    for fn_, tx in {"results.json": text, "digest.txt": dg, "run_log.txt": logtext}.items():
        with open(os.path.join(a.out, fn_), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn_}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s_[:16] + "  " + s_.split()[-1] for s_ in sums))
    log(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
