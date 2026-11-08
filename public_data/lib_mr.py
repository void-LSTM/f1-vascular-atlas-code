"""lib_mr.py - two-sample Mendelian randomisation on summary statistics, numpy/scipy only.

Paper F1, Stage 0 analysis library (HPP `pheno` environment: numpy 1.26, pandas 2.1, scipy 1.11; no R, no plink).
Validated only on simulated summary statistics with planted answers (`validate_lib_mr.py`).

Contents
--------
Summary statistics
    standardize_sumstats   rename columns to the internal schema; derive beta from OR, SE from OR + 95 % CI or from
                           beta + p; upper-case alleles
    harmonise              match exposure/outcome by rsID or GRCh37 chr:pos; align alleles (direct, swapped,
                           strand-flipped, strand-flipped + swapped); palindromic SNPs dropped when EAF is missing or
                           within [0.42, 0.58], otherwise strand inferred from EAF; incompatible alleles dropped
Clumping
    clump_distance         greedy distance-based clumping (window, p threshold)
    clump_ld               greedy LD clumping using an LD matrix (LDMatrix) or a PLINK .bed reference (PlinkBed)
    PlinkBed               numpy reader of PLINK 1 binary genotypes (SNP-major; 2 bits per genotype; A1 count;
                           missing = -1); write_plink_bed (for tests)
Estimators
    mr_ivw                 inverse-variance weighted, fixed and multiplicative random effects
    mr_egger               MR-Egger (oriented to positive exposure effects), residual-SE scaled, t(J-2); I2_GX
    mr_weighted_median     weighted median (Bowden 2016) with parametric bootstrap SE
    mr_weighted_mode       weighted mode (Hartwig 2017; normal kernel, modified Silverman bandwidth) + bootstrap SE
    cochran_q              Cochran's Q for IVW (and Ruecker's Q' for Egger in mr_egger)
    steiger                Steiger directionality test (needs sample sizes); r2 from beta/se/N or from log-OR
    mvmr_ivw               multivariable IVW, with conditional F statistics (Sanderson-Windmeijer-style
                           approximation of Sanderson, Spiller & Bowden 2021) and Q_A heterogeneity
    f_statistics           per-SNP F = beta^2/se^2, mean F, and R2-based F (with N)
    mr_all                 one-call summary table

Sign conventions: all estimates are per unit of the exposure scale of beta_x (e.g. per SD or per log-odds).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}
INTERNAL = ["snp", "chr", "pos", "ea", "oa", "beta", "se", "p", "eaf", "n"]


# ================================================================================================================
# Summary statistics: standardisation and harmonisation
# ================================================================================================================
def se_from_ci(lower, upper, log_scale_input: bool = True, level: float = 0.95):
    """SE of log(OR) from a CI of the OR (log_scale_input=True means the CI is on the OR scale)."""
    z = stats.norm.ppf(0.5 + level / 2)
    lo, hi = np.asarray(lower, float), np.asarray(upper, float)
    if log_scale_input:
        with np.errstate(divide="ignore", invalid="ignore"):
            lo, hi = np.log(lo), np.log(hi)
    return (hi - lo) / (2 * z)


def se_from_beta_p(beta, p):
    """SE from beta and two-sided p: |beta| / Phi^-1(1 - p/2). NaN when p is 0, 1 or missing, or beta = 0."""
    beta, p = np.asarray(beta, float), np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = stats.norm.isf(p / 2)
        se = np.abs(beta) / z
    bad = ~np.isfinite(se) | (se <= 0) | (p <= 0) | (p >= 1)
    return np.where(bad, np.nan, se)


def standardize_sumstats(df: pd.DataFrame, colmap: dict | None = None) -> pd.DataFrame:
    """Return a frame with the internal columns snp, chr, pos, ea, oa, beta, se, p, eaf, n (+ derived flags).

    colmap maps internal names (and optionally 'or', 'or_lci', 'or_uci') to source column names.
    beta: from 'beta', else log('or'). se: from 'se', else from the OR CI, else from beta and p.
    p: from 'p', else from beta/se. Alleles upper-cased; chr as string without 'chr' prefix; pos as Int64.
    """
    colmap = colmap or {}
    src = {k: colmap.get(k, k) for k in INTERNAL + ["or", "or_lci", "or_uci"]}
    out = pd.DataFrame(index=df.index)
    for k in INTERNAL:
        out[k] = df[src[k]] if src[k] in df.columns else np.nan
    out["beta_source"] = np.where(pd.notna(out["beta"]), "beta", "")
    if src["or"] in df.columns:
        orv = pd.to_numeric(df[src["or"]], errors="coerce")
        need = out["beta"].isna() & orv.notna() & (orv > 0)
        out.loc[need, "beta"] = np.log(orv[need])
        out.loc[need, "beta_source"] = "log_or"
    out["beta"] = pd.to_numeric(out["beta"], errors="coerce")
    out["se"] = pd.to_numeric(out["se"], errors="coerce")
    out["p"] = pd.to_numeric(out["p"], errors="coerce")
    out["se_source"] = np.where(out["se"].notna(), "se", "")
    if src["or_lci"] in df.columns and src["or_uci"] in df.columns:
        se_ci = se_from_ci(pd.to_numeric(df[src["or_lci"]], errors="coerce"),
                           pd.to_numeric(df[src["or_uci"]], errors="coerce"))
        need = out["se"].isna() & np.isfinite(se_ci)
        out.loc[need, "se"] = se_ci[need.values]
        out.loc[need, "se_source"] = "or_ci"
    need = out["se"].isna() & out["beta"].notna() & out["p"].notna()
    if need.any():
        se_bp = se_from_beta_p(out.loc[need, "beta"], out.loc[need, "p"])
        out.loc[need, "se"] = se_bp
        out.loc[need, "se_source"] = np.where(np.isfinite(se_bp), "beta_p", "")
    need = out["p"].isna() & out["beta"].notna() & out["se"].notna()
    out.loc[need, "p"] = 2 * stats.norm.sf(np.abs(out.loc[need, "beta"] / out.loc[need, "se"]))
    for a in ("ea", "oa"):
        out[a] = out[a].astype("string").str.strip().str.upper()
    out["chr"] = out["chr"].astype("string").str.replace("^chr", "", regex=True, case=False)
    out["pos"] = pd.to_numeric(out["pos"], errors="coerce").astype("Int64")
    out["eaf"] = pd.to_numeric(out["eaf"], errors="coerce")
    out["n"] = pd.to_numeric(out["n"], errors="coerce")
    return out


def _is_snv(a: pd.Series) -> pd.Series:
    return a.isin(list(COMP)).fillna(False).astype(bool)


def _comp(a: pd.Series) -> pd.Series:
    return a.map(COMP)


def is_palindromic(ea: pd.Series, oa: pd.Series) -> pd.Series:
    return (_is_snv(ea) & _is_snv(oa) & (_comp(ea) == oa)).fillna(False).astype(bool)


def harmonise(exp: pd.DataFrame, out: pd.DataFrame, match_on: str = "auto",
              pal_band: tuple = (0.42, 0.58)) -> pd.DataFrame:
    """Harmonise standardised exposure and outcome statistics to the exposure effect allele.

    match_on: 'rsid', 'chrpos' (GRCh37) or 'auto' (rsID first, chr:pos for the remainder).
    Returns one row per matched SNP with columns snp, chr, pos, ea, oa, beta_x, se_x, p_x, eaf_x, n_x, beta_y,
    se_y, p_y, eaf_y, n_y, action, keep. Actions: direct, swapped, strand_flip, strand_flip_swapped,
    palindromic_direct, palindromic_flipped (kept); drop_incompatible, drop_palindromic_no_eaf,
    drop_palindromic_ambiguous_eaf, drop_missing (dropped). eaf_y is re-expressed for the exposure effect allele.
    Rule for palindromic SNPs (A/T, C/G): dropped if either EAF is missing or the exposure or outcome EAF lies in
    [pal_band]; otherwise the outcome is aligned so that both EAFs fall on the same side of 0.5.
    """
    def _dedup(d):
        # one row per rsID (smallest p); rows without rsID are kept (matched by chr:pos only)
        has = d["snp"].notna()
        return pd.concat([d[has].sort_values("p", kind="mergesort").drop_duplicates("snp"), d[~has]])

    ex = _dedup(exp.copy())
    ou = _dedup(out.copy())
    sx = {c: c + "_x" for c in ["beta", "se", "p", "eaf", "n", "ea", "oa"]}
    sy = {c: c + "_y" for c in ["beta", "se", "p", "eaf", "n", "ea", "oa"]}
    keep_cols = ["snp", "chr", "pos", "beta", "se", "p", "eaf", "n", "ea", "oa"]
    ex = ex[keep_cols].rename(columns=sx)
    ou = ou[keep_cols].rename(columns=sy)
    parts = []
    if match_on in ("rsid", "auto"):
        m = ex.dropna(subset=["snp"]).merge(ou.dropna(subset=["snp"]).drop(columns=["chr", "pos"]),
                                            on="snp", how="inner")
        m["matched_by"] = "rsid"
        parts.append(m)
    if match_on in ("chrpos", "auto"):
        done = set(parts[0]["snp"]) if parts else set()
        exr = ex[~ex["snp"].isin(done)] if done else ex
        our = ou[~ou["snp"].isin(done)] if done else ou
        exr = exr.dropna(subset=["chr", "pos"])
        our = our.dropna(subset=["chr", "pos"]).rename(columns={"snp": "snp_y"})
        m2 = exr.merge(our, on=["chr", "pos"], how="inner")
        m2["snp"] = m2["snp"].fillna(m2["snp_y"])
        m2 = m2.drop(columns=["snp_y"])
        m2["matched_by"] = "chrpos"
        parts.append(m2)
    if match_on not in ("rsid", "chrpos", "auto"):
        raise ValueError("match_on must be rsid, chrpos or auto")
    h = pd.concat(parts, ignore_index=True)
    if h.empty:
        return h.assign(action=pd.Series(dtype=str), keep=pd.Series(dtype=bool))
    ea_x, oa_x, ea_y, oa_y = h["ea_x"], h["oa_x"], h["ea_y"], h["oa_y"]
    beta_y = h["beta_y"].astype(float).copy()
    eaf_y = h["eaf_y"].astype(float).copy()
    action = pd.Series("drop_incompatible", index=h.index, dtype=object)
    missing = ea_x.isna() | oa_x.isna() | ea_y.isna() | oa_y.isna() | h["beta_x"].isna() | h["se_x"].isna() \
        | h["beta_y"].isna() | h["se_y"].isna()
    pal = is_palindromic(ea_x, oa_x)
    snv = _is_snv(ea_x) & _is_snv(oa_x) & _is_snv(ea_y) & _is_snv(oa_y)
    direct = (ea_y == ea_x) & (oa_y == oa_x)
    swapped = (ea_y == oa_x) & (oa_y == ea_x)
    sflip = snv & (_comp(ea_y) == ea_x) & (_comp(oa_y) == oa_x)
    sflip_sw = snv & (_comp(ea_y) == oa_x) & (_comp(oa_y) == ea_x)
    direct, swapped, sflip, sflip_sw = [s.fillna(False).astype(bool) for s in (direct, swapped, sflip, sflip_sw)]
    npal = ~pal
    action[npal & direct] = "direct"
    action[npal & ~direct & swapped] = "swapped"
    action[npal & ~direct & ~swapped & sflip] = "strand_flip"
    action[npal & ~direct & ~swapped & ~sflip & sflip_sw] = "strand_flip_swapped"
    flip = npal & ~direct & ((swapped) | (~sflip & sflip_sw))
    # palindromic: first express the outcome for the exposure effect allele as written, then check strand by EAF
    pal_ok = pal & (direct | swapped)
    pal_sw = pal_ok & ~direct & swapped
    exf, eyf = h["eaf_x"].astype(float), eaf_y.where(~pal_sw, 1 - eaf_y)
    lo, hi = pal_band
    no_eaf = pal_ok & (exf.isna() | eyf.isna())
    ambig = pal_ok & ~no_eaf & (exf.between(lo, hi) | eyf.between(lo, hi))
    infer = pal_ok & ~no_eaf & ~ambig
    strand_differs = infer & ((exf < 0.5) != (eyf < 0.5))
    action[no_eaf] = "drop_palindromic_no_eaf"
    action[ambig] = "drop_palindromic_ambiguous_eaf"
    action[infer & ~strand_differs] = "palindromic_direct"
    action[infer & strand_differs] = "palindromic_flipped"
    pal_flip = (pal_sw & infer) ^ (infer & strand_differs)       # net sign change for kept palindromic SNPs
    sign = np.where(flip | pal_flip, -1.0, 1.0)
    h["beta_y"] = beta_y * sign
    h["eaf_y"] = np.where(sign < 0, 1 - eaf_y, eaf_y)
    action[missing] = "drop_missing"
    h["action"] = action
    h["keep"] = h["action"].isin(["direct", "swapped", "strand_flip", "strand_flip_swapped",
                                  "palindromic_direct", "palindromic_flipped"])
    h["ea"], h["oa"] = h["ea_x"], h["oa_x"]
    h["eaf_discordant"] = h["keep"] & h["eaf_x"].notna() & h["eaf_y"].notna() & \
        ((h["eaf_x"] - h["eaf_y"]).abs() > 0.2)
    cols = ["snp", "chr", "pos", "ea", "oa", "beta_x", "se_x", "p_x", "eaf_x", "n_x", "beta_y", "se_y", "p_y",
            "eaf_y", "n_y", "ea_y", "oa_y", "matched_by", "action", "keep", "eaf_discordant"]
    return h[cols].reset_index(drop=True)


# ================================================================================================================
# Clumping and PLINK .bed reader
# ================================================================================================================
_LUT = None


def _bed_lut() -> np.ndarray:
    """256 x 4 lookup: byte -> 4 genotypes (A1 count; 2-bit codes 00->2, 01->-1, 10->1, 11->0; low bits first)."""
    global _LUT
    if _LUT is None:
        m = np.array([2, -1, 1, 0], dtype=np.int8)
        b = np.arange(256)
        _LUT = np.stack([m[(b >> (2 * k)) & 3] for k in range(4)], axis=1).astype(np.int8)
    return _LUT


class PlinkBed:
    """PLINK 1 binary fileset reader (prefix.bed/.bim/.fam), SNP-major only.

    read(idx) -> int8 array (n_samples, len(idx)) of A1 allele counts (A1 = .bim column 5), missing = -1.
    """

    MAGIC = bytes([0x6C, 0x1B, 0x01])

    def __init__(self, prefix: str):
        self.prefix = str(prefix)
        self.bim = pd.read_csv(self.prefix + ".bim", sep=r"\s+", header=None,
                               names=["chr", "snp", "cm", "pos", "a1", "a2"],
                               dtype={"chr": str, "snp": str, "a1": str, "a2": str})
        self.fam = pd.read_csv(self.prefix + ".fam", sep=r"\s+", header=None,
                               names=["fid", "iid", "pat", "mat", "sex", "pheno"], dtype=str)
        self.n_snps, self.n_samples = len(self.bim), len(self.fam)
        self.bps = (self.n_samples + 3) // 4
        with open(self.prefix + ".bed", "rb") as fh:
            magic = fh.read(3)
        if magic[:2] != self.MAGIC[:2]:
            raise ValueError("not a PLINK .bed file (bad magic number)")
        if magic[2] != 0x01:
            raise ValueError("individual-major .bed files are not supported (SNP-major required)")
        size = os.path.getsize(self.prefix + ".bed")
        if size != 3 + self.n_snps * self.bps:
            raise ValueError(f".bed size {size} does not match {self.n_snps} SNPs x {self.bps} bytes + 3")
        self._mm = np.memmap(self.prefix + ".bed", dtype=np.uint8, mode="r", offset=3,
                             shape=(self.n_snps, self.bps))
        self.bim["chr"] = self.bim["chr"].str.replace("^chr", "", regex=True, case=False)
        self._by_snp = pd.Series(np.arange(self.n_snps), index=self.bim["snp"])
        self._by_snp = self._by_snp[~self._by_snp.index.duplicated()]
        key = self.bim["chr"].astype(str) + ":" + self.bim["pos"].astype(str)
        self._by_chrpos = pd.Series(np.arange(self.n_snps), index=key)
        self._by_chrpos = self._by_chrpos[~self._by_chrpos.index.duplicated()]

    def read(self, idx) -> np.ndarray:
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        raw = np.asarray(self._mm[idx])                     # (k, bps)
        g = _bed_lut()[raw].reshape(len(idx), self.bps * 4)[:, : self.n_samples]
        return g.T.copy()

    def index_of(self, snp=None, chrom=None, pos=None) -> np.ndarray:
        """Row indices in the .bim (-1 if absent), by rsID or by chr:pos."""
        if snp is not None:
            return self._by_snp.reindex(pd.Index(snp)).fillna(-1).astype(int).values
        key = pd.Index(pd.Series(chrom).astype(str).str.replace("^chr", "", regex=True, case=False) + ":"
                       + pd.Series(pos).astype(str))
        return self._by_chrpos.reindex(key).fillna(-1).astype(int).values


def write_plink_bed(prefix: str, G: np.ndarray, bim: pd.DataFrame, fam: pd.DataFrame | None = None) -> None:
    """Write a SNP-major PLINK 1 .bed/.bim/.fam. G: (n_samples, n_snps) A1 counts with -1 = missing."""
    G = np.asarray(G)
    n, m = G.shape
    code = np.full(G.shape, 1, dtype=np.uint8)                 # missing -> 01
    code[G == 2] = 0
    code[G == 1] = 2
    code[G == 0] = 3
    bps = (n + 3) // 4
    pad = np.full((bps * 4 - n, m), 0, dtype=np.uint8)
    c = np.concatenate([code, pad], axis=0).T.reshape(m, bps, 4)
    byts = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)
    with open(prefix + ".bed", "wb") as fh:
        fh.write(PlinkBed.MAGIC)
        fh.write(byts.tobytes())
    bim[["chr", "snp", "cm", "pos", "a1", "a2"]].to_csv(prefix + ".bim", sep="\t", header=False, index=False)
    if fam is None:
        fam = pd.DataFrame({"fid": [f"F{i}" for i in range(n)], "iid": [f"I{i}" for i in range(n)],
                            "pat": 0, "mat": 0, "sex": 0, "pheno": -9})
    fam.to_csv(prefix + ".fam", sep=" ", header=False, index=False)


def r2_from_genotypes(G: np.ndarray) -> np.ndarray:
    """Pairwise r^2 between columns of a genotype matrix (missing = -1 mean-imputed per SNP)."""
    X = np.asarray(G, float)
    X[X < 0] = np.nan
    mu = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mu, X) - mu
    sd = np.sqrt((X ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (X.T @ X) / np.outer(sd, sd)
    r = np.nan_to_num(r)
    return r ** 2


@dataclass
class LDMatrix:
    """LD reference as a correlation (r) or r^2 matrix with SNP identifiers."""
    snps: list
    r: np.ndarray
    is_r2: bool = False

    def __post_init__(self):
        self._pos = {s: i for i, s in enumerate(self.snps)}

    def r2(self, a, b_list) -> np.ndarray:
        ia = self._pos.get(a)
        out = np.full(len(b_list), np.nan)
        if ia is None:
            return out
        for j, b in enumerate(b_list):
            ib = self._pos.get(b)
            if ib is not None:
                v = self.r[ia, ib]
                out[j] = v if self.is_r2 else v ** 2
        return out


def clump_distance(df: pd.DataFrame, p_thresh: float = 5e-8, window_kb: float = 500.0) -> pd.Series:
    """Greedy distance clumping: index SNP = smallest p; remove SNPs on the same chr within +/- window.

    df needs chr, pos, p. Returns a boolean Series (index of df): True = kept index SNP.
    """
    d = df[["chr", "pos", "p"]].copy()
    keep = pd.Series(False, index=df.index)
    cand = d[(d["p"] <= p_thresh) & d["pos"].notna()].sort_values("p", kind="mergesort")
    removed = set()
    w = window_kb * 1000
    by_chr = {c: g for c, g in cand.groupby("chr")}
    for i, row in cand.iterrows():
        if i in removed:
            continue
        keep[i] = True
        g = by_chr[row["chr"]]
        near = g.index[(g["pos"] - row["pos"]).abs() <= w]
        removed.update(near)
    return keep


def clump_ld(df: pd.DataFrame, ref, p_thresh: float = 5e-8, r2_thresh: float = 0.001, window_kb: float = 10000.0,
             match_on: str = "rsid", missing_in_ref: str = "drop") -> pd.DataFrame:
    """Greedy LD clumping (PLINK --clump logic) using an LDMatrix or a PlinkBed reference.

    df needs snp, chr, pos, p. Candidates: p <= p_thresh. For each index SNP (smallest p first) remove remaining
    candidates on the same chr within window_kb with r^2 > r2_thresh. SNPs absent from the reference are dropped
    (missing_in_ref='drop') or kept as independent ('keep'). Returns df with columns in_ref, kept, clumped_by.
    """
    d = df.copy()
    d["in_ref"] = True
    if isinstance(ref, PlinkBed):
        ridx = ref.index_of(snp=d["snp"]) if match_on == "rsid" else ref.index_of(chrom=d["chr"], pos=d["pos"])
        d["_ref_idx"] = ridx
        d["in_ref"] = ridx >= 0
    elif isinstance(ref, LDMatrix):
        d["in_ref"] = d["snp"].isin(set(ref.snps))
    else:
        raise TypeError("ref must be LDMatrix or PlinkBed")
    d["kept"] = False
    d["clumped_by"] = pd.Series(pd.NA, index=d.index, dtype="string")
    cand = d[(d["p"] <= p_thresh)].sort_values("p", kind="mergesort")
    if missing_in_ref == "keep":
        d.loc[cand.index[~cand["in_ref"]], "kept"] = True
    cand = cand[cand["in_ref"]]
    alive = pd.Series(True, index=cand.index)
    w = window_kb * 1000
    for i in cand.index:
        if not alive[i]:
            continue
        d.loc[i, "kept"] = True
        alive[i] = False
        row = cand.loc[i]
        others = cand.index[alive.values & (cand["chr"] == row["chr"]).values
                            & ((cand["pos"] - row["pos"]).abs() <= w).values]
        if len(others) == 0:
            continue
        if isinstance(ref, PlinkBed):
            G = ref.read(np.concatenate([[int(row["_ref_idx"])], cand.loc[others, "_ref_idx"].astype(int).values]))
            r2 = r2_from_genotypes(G)[0, 1:]
        else:
            r2 = ref.r2(row["snp"], list(cand.loc[others, "snp"]))
        drop = others[np.nan_to_num(r2) > r2_thresh]
        alive[drop] = False
        d.loc[drop, "clumped_by"] = str(row["snp"])
    return d.drop(columns=[c for c in ["_ref_idx"] if c in d.columns])


# ================================================================================================================
# Estimators
# ================================================================================================================
def _arr(*a):
    return [np.asarray(x, float) for x in a]


def _res(method, est, se, nsnp, df=None, **extra) -> dict:
    q = stats.t.ppf(0.975, df) if df else stats.norm.ppf(0.975)
    z = est / se if se and np.isfinite(se) and se > 0 else np.nan
    p = 2 * (stats.t.sf(abs(z), df) if df else stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan
    out = {"method": method, "nsnp": int(nsnp), "estimate": float(est), "se": float(se),
           "lower": float(est - q * se), "upper": float(est + q * se), "p": float(p)}
    out.update(extra)
    return out


def cochran_q(bx, by, sy, theta) -> dict:
    bx, by, sy = _arr(bx, by, sy)
    q = float(np.sum((by - theta * bx) ** 2 / sy ** 2))
    df = len(bx) - 1
    return {"Q": q, "Q_df": df, "Q_p": float(stats.chi2.sf(q, df)) if df > 0 else np.nan,
            "I2": float(max(0.0, (q - df) / q)) if q > 0 else 0.0}


def mr_ivw(bx, by, sy, model: str = "random") -> dict:
    """IVW (first-order weights 1/se_y^2). model 'fixed' or 'random' (multiplicative: SE x max(1, sqrt(Q/(J-1))))."""
    bx, by, sy = _arr(bx, by, sy)
    w = 1 / sy ** 2
    theta = float(np.sum(w * bx * by) / np.sum(w * bx ** 2))
    se_f = float(np.sqrt(1 / np.sum(w * bx ** 2)))
    q = cochran_q(bx, by, sy, theta)
    phi = q["Q"] / q["Q_df"] if q["Q_df"] > 0 else 1.0
    se = se_f if model == "fixed" else se_f * max(1.0, np.sqrt(phi))
    return _res(f"IVW ({'fixed' if model == 'fixed' else 'multiplicative random'} effects)", theta, se,
                len(bx), **q, phi=phi)


def isq_gx(bx, sx, sy=None) -> float:
    """I2_GX (Bowden 2016) on oriented exposure effects; with sy, uses bx/sy and sx/sy (first-order weights)."""
    bx, sx = _arr(bx, sx)
    y, s = np.abs(bx), sx
    if sy is not None:
        sy = np.asarray(sy, float)
        y, s = np.abs(bx) / sy, sx / sy
    w = 1 / s ** 2
    mu = np.sum(w * y) / np.sum(w)
    q = np.sum(w * (y - mu) ** 2)
    k = len(y)
    return float(max(0.0, (q - (k - 1)) / q)) if q > 0 else 0.0


def mr_egger(bx, by, sx, sy) -> dict:
    """MR-Egger: orient SNPs to bx > 0, weighted regression by ~ a + b bx (weights 1/sy^2).

    SEs multiplied by max(1, residual SE); inference t(J - 2). Returns slope result with intercept, Rucker's Q',
    and I2_GX (weighted by 1/sy and unweighted)."""
    bx, by, sx, sy = _arr(bx, by, sx, sy)
    s = np.sign(bx)
    s[s == 0] = 1
    bx, by = bx * s, by * s
    J = len(bx)
    if J < 3:
        raise ValueError("MR-Egger needs >= 3 SNPs")
    w = 1 / sy ** 2
    X = np.column_stack([np.ones(J), bx])
    A = X.T @ (w[:, None] * X)
    coef = np.linalg.solve(A, X.T @ (w * by))
    res = by - X @ coef
    sigma = float(np.sqrt(np.sum(w * res ** 2) / (J - 2)))
    V = np.linalg.inv(A) * max(1.0, sigma) ** 2
    se = np.sqrt(np.diag(V))
    df = J - 2
    qe = float(np.sum(w * res ** 2))
    icpt = _res("MR-Egger intercept", coef[0], se[0], J, df=df)
    return _res("MR-Egger", coef[1], se[1], J, df=df, intercept=float(coef[0]), intercept_se=float(se[0]),
                intercept_p=icpt["p"], intercept_lower=icpt["lower"], intercept_upper=icpt["upper"],
                sigma=sigma, Q_egger=qe, Q_egger_df=df, Q_egger_p=float(stats.chi2.sf(qe, df)),
                I2_GX=isq_gx(bx, sx, sy), I2_GX_unweighted=isq_gx(bx, sx))


def _wmedian(ratio, w):
    o = np.argsort(ratio)
    b, ww = ratio[o], w[o] / np.sum(w)
    cs = np.cumsum(ww) - 0.5 * ww
    j = np.searchsorted(cs, 0.5) - 1
    j = int(np.clip(j, 0, len(b) - 2))
    return float(b[j] + (b[j + 1] - b[j]) * (0.5 - cs[j]) / (cs[j + 1] - cs[j]))


def _wmedian_rows(R, w):
    """Weighted median for every row of R (B x J) with common weights w (J)."""
    o = np.argsort(R, axis=1)
    b = np.take_along_axis(R, o, axis=1)
    ww = (w / w.sum())[o]
    cs = np.cumsum(ww, axis=1) - 0.5 * ww
    j = np.clip((cs < 0.5).sum(axis=1) - 1, 0, R.shape[1] - 2)
    r = np.arange(R.shape[0])
    c0, c1 = cs[r, j], cs[r, j + 1]
    return b[r, j] + (b[r, j + 1] - b[r, j]) * (0.5 - c0) / (c1 - c0)


def mr_weighted_median(bx, by, sx, sy, n_boot: int = 1000, seed: int = 1) -> dict:
    """Weighted median of ratio estimates (weights bx^2/sy^2), parametric bootstrap SE (weights held fixed)."""
    bx, by, sx, sy = _arr(bx, by, sx, sy)
    ratio = by / bx
    w = bx ** 2 / sy ** 2
    est = _wmedian(ratio, w)
    rng = np.random.default_rng(seed)
    BX = bx + rng.standard_normal((n_boot, len(bx))) * sx
    BY = by + rng.standard_normal((n_boot, len(by))) * sy
    boots = _wmedian_rows(BY / BX, w)
    return _res("Weighted median", est, float(np.std(boots, ddof=1)), len(bx), n_boot=n_boot)


def _wmode(ratio, se_ratio, phi=1.0, grid=401):
    """Mode of the weighted normal-kernel density of the ratio estimates.

    Bandwidth h = phi * 0.9 * min(SD, scaled MAD) / J^(1/5) (Hartwig 2017). The density is first evaluated at every
    ratio estimate (the global maximum lies near one of them; this is robust to far outliers that would stretch a
    fixed grid), then refined on a fine grid of +/- 2h around the best three candidates.
    """
    w = se_ratio ** -2
    w = w / w.sum()
    sd = np.std(ratio, ddof=1)
    mad = stats.median_abs_deviation(ratio, scale="normal")
    s = 0.9 * min(sd, mad) / len(ratio) ** 0.2
    h = max(1e-8, s * phi)

    def dens(x):
        return (w[None, :] * np.exp(-0.5 * ((x[:, None] - ratio[None, :]) / h) ** 2)).sum(axis=1)

    d0 = dens(ratio)
    best_x, best_d = None, -np.inf
    for c in ratio[np.argsort(d0)[-3:]]:
        xf = np.linspace(c - 2 * h, c + 2 * h, grid)
        df = dens(xf)
        k = int(np.argmax(df))
        if df[k] > best_d:
            best_x, best_d = float(xf[k]), float(df[k])
    return best_x


def mr_weighted_mode(bx, by, sx, sy, phi: float = 1.0, n_boot: int = 1000, seed: int = 1) -> dict:
    """Weighted mode (Hartwig 2017, NOME: ratio SE = sy/|bx|), parametric bootstrap SE."""
    bx, by, sx, sy = _arr(bx, by, sx, sy)
    ratio = by / bx
    se_r = sy / np.abs(bx)
    est = _wmode(ratio, se_r, phi)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        bxb = bx + rng.standard_normal(len(bx)) * sx
        byb = by + rng.standard_normal(len(by)) * sy
        boots[b] = _wmode(byb / bxb, sy / np.abs(bxb), phi, grid=101)
    return _res("Weighted mode", est, float(np.std(boots, ddof=1)), len(bx), n_boot=n_boot, phi=phi)


def r2_from_beta_se(beta, se, n) -> np.ndarray:
    """Variance explained by a SNP from its t statistic: r2 = t^2 / (t^2 + n - 2)."""
    beta, se, n = _arr(beta, se, n)
    t2 = (beta / se) ** 2
    return t2 / (t2 + n - 2)


def r2_from_lor(lor, eaf, prevalence: float | None = None, model: str = "logit") -> np.ndarray:
    """Liability-scale variance explained by a SNP from its log-OR: vg / (vg + ve), vg = lor^2 * 2 f (1 - f)?

    Implementation: vg = lor^2 * f (1 - f) as in TwoSampleMR get_r_from_lor, with f the reported allele frequency
    (the case-control to population frequency correction is omitted; `prevalence` is accepted for future use);
    ve = pi^2/3 (logit) or 1 (probit).
    """
    lor, f = _arr(lor, eaf)
    ve = np.pi ** 2 / 3 if model == "logit" else 1.0
    vg = lor ** 2 * f * (1 - f)
    return vg / (vg + ve)


def steiger(r2_x, r2_y, n_x, n_y) -> dict:
    """Steiger directionality test from SNP-level r2 (summed over SNPs) and sample sizes.

    z = (atanh(r_x) - atanh(r_y)) / sqrt(1/(n_x - 3) + 1/(n_y - 3)); direction X -> Y supported when r2_x > r2_y.
    """
    rx2, ry2 = float(np.sum(r2_x)), float(np.sum(r2_y))
    nx, ny = float(np.median(np.atleast_1d(n_x))), float(np.median(np.atleast_1d(n_y)))
    z = (np.arctanh(np.sqrt(rx2)) - np.arctanh(np.sqrt(ry2))) / np.sqrt(1 / (nx - 3) + 1 / (ny - 3))
    return {"r2_exposure": rx2, "r2_outcome": ry2, "correct_direction": bool(rx2 > ry2), "steiger_z": float(z),
            "steiger_p": float(2 * stats.norm.sf(abs(z)))}


def steiger_snp_filter(r2_x, r2_y, n_x, n_y, alpha: float = 0.05) -> np.ndarray:
    """Per-SNP Steiger filter: keep SNPs whose r2 with the exposure exceeds that with the outcome (or not sig.)."""
    rx, ry = np.sqrt(np.asarray(r2_x, float)), np.sqrt(np.asarray(r2_y, float))
    nx, ny = np.asarray(n_x, float), np.asarray(n_y, float)
    z = (np.arctanh(rx) - np.arctanh(ry)) / np.sqrt(1 / (nx - 3) + 1 / (ny - 3))
    p = 2 * stats.norm.sf(np.abs(z))
    return (rx > ry) | (p > alpha)


def f_statistics(bx, sx, n=None, eaf=None) -> dict:
    """Per-SNP F (bx^2/sx^2), mean and min F; with n, the R2-based overall F = (n - J - 1)/J * R2/(1 - R2)."""
    bx, sx = _arr(bx, sx)
    f = (bx / sx) ** 2
    out = {"F_mean": float(np.mean(f)), "F_min": float(np.min(f)), "F_median": float(np.median(f)),
           "n_F_lt_10": int(np.sum(f < 10)), "F_snp": f}
    if n is not None:
        r2 = r2_from_beta_se(bx, sx, n)
        R2 = float(np.sum(r2))
        J = len(bx)
        nn = float(np.median(np.atleast_1d(n)))
        out["R2"] = R2
        out["F_overall_R2"] = float((nn - J - 1) / J * R2 / (1 - R2)) if R2 < 1 else np.nan
    return out


def mvmr_ivw(BX, by, SX, sy, names=None, model: str = "random", gencov=None) -> dict:
    """Multivariable IVW: weighted regression of by on the J x K matrix BX (no intercept), weights 1/sy^2.

    SE scaled by max(1, residual SE) (model='random') or not ('fixed'). Q_A = sum w (by - BX theta)^2, df J - K.
    Conditional F per exposure k (Sanderson, Spiller & Bowden 2021):  delta = OLS of BX[:, k] on BX[:, -k];
    Q_k = sum_j (bx_kj - delta' bx_-k,j)^2 / sigma2_kj with sigma2_kj = sx_kj^2 + sum_m delta_m^2 sx_mj^2
    (+ covariance terms if gencov, a K x K phenotypic correlation matrix, is given); F_k = Q_k / (J - K + 1).
    """
    BX, SX = np.asarray(BX, float), np.asarray(SX, float)
    by, sy = _arr(by, sy)
    J, K = BX.shape
    names = list(names) if names is not None else [f"exposure{k + 1}" for k in range(K)]
    w = 1 / sy ** 2
    A = BX.T @ (w[:, None] * BX)
    theta = np.linalg.solve(A, BX.T @ (w * by))
    res = by - BX @ theta
    qa = float(np.sum(w * res ** 2))
    sigma = np.sqrt(qa / (J - K)) if J > K else 1.0
    V = np.linalg.inv(A) * (max(1.0, sigma) ** 2 if model == "random" else 1.0)
    se = np.sqrt(np.diag(V))
    condF = []
    for k in range(K):
        others = [m for m in range(K) if m != k]
        Xo = BX[:, others]
        delta = np.linalg.lstsq(Xo, BX[:, k], rcond=None)[0]
        resid = BX[:, k] - Xo @ delta
        s2 = SX[:, k] ** 2 + (SX[:, others] ** 2) @ (delta ** 2)
        if gencov is not None:
            G = np.asarray(gencov, float)
            for a_i, a in enumerate(others):
                s2 = s2 - 2 * delta[a_i] * G[k, a] * SX[:, k] * SX[:, a]
                for b_i, b in enumerate(others):
                    if b > a:
                        s2 = s2 + 2 * delta[a_i] * delta[b_i] * G[a, b] * SX[:, a] * SX[:, b]
        condF.append(float(np.sum(resid ** 2 / s2) / (J - K + 1)))
    rows = [_res(f"MVMR-IVW: {names[k]}", theta[k], se[k], J, conditional_F=condF[k]) for k in range(K)]
    return {"results": rows, "Q_A": qa, "Q_A_df": J - K, "Q_A_p": float(stats.chi2.sf(qa, J - K)) if J > K else np.nan,
            "sigma": float(sigma), "vcov": V, "conditional_F": dict(zip(names, condF))}


def mr_all(h: pd.DataFrame, n_boot: int = 1000, seed: int = 1, steiger_n=True) -> pd.DataFrame:
    """Run IVW (fixed, random), Egger, weighted median, weighted mode on harmonised rows with keep=True."""
    d = h[h["keep"]] if "keep" in h.columns else h
    bx, by, sx, sy = d["beta_x"].values, d["beta_y"].values, d["se_x"].values, d["se_y"].values
    rows = [mr_ivw(bx, by, sy, "fixed"), mr_ivw(bx, by, sy, "random")]
    if len(d) >= 3:
        rows.append(mr_egger(bx, by, sx, sy))
        rows.append(mr_weighted_median(bx, by, sx, sy, n_boot=n_boot, seed=seed))
        rows.append(mr_weighted_mode(bx, by, sx, sy, n_boot=n_boot, seed=seed))
    out = pd.DataFrame(rows)
    fs = f_statistics(bx, sx)
    out["F_mean"] = fs["F_mean"]
    if steiger_n and {"n_x", "n_y"} <= set(d.columns) and d["n_x"].notna().all() and d["n_y"].notna().all():
        st = steiger(r2_from_beta_se(bx, sx, d["n_x"]), r2_from_beta_se(by, sy, d["n_y"]), d["n_x"], d["n_y"])
        for k, v in st.items():
            out[k] = v
    return out
