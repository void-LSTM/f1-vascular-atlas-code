#!/usr/bin/env python3
"""analysis_m_mr_subtype_interaction.py - F1 Stage 0 local experiment L11 (PLAN 9.3d): MR bed x outcome-subtype tests.

Paper F1 (multi-bed vascular atlas). ALL RESULTS ARE EXPLORATORY. Open public data only (no HPP data, no liver
phenotypes). Formal tests that the causal (MR) effect of a vascular bed differs between outcome subtypes from the same
consortium, i.e. a genetic, tested "bed -> outcome profile".

Contrast = ratio of MR odds ratios, outcome A / outcome B, with the same exposure instruments (SNPs harmonised in both
outcome files). The two outcome GWAS share samples (controls; nested cases), so per-SNP errors are correlated:
- rho (error correlation) per outcome pair is estimated from genome-wide null variants: rsIDs ending in 0 (a fixed
  10 % subsample), EAF 0.05-0.95 in both files, non-palindromic SNVs, |z| < 2 in both files, GRCh37 positions from the
  1000G EUR panel, greedily thinned to >= 100 kb apart; primary rho = Pearson r of the aligned z-scores. Sensitivity:
  rho from a truncated bivariate-normal MLE (corrects the attenuation by the |z| < 2 truncation) and rho = 0.
- Per SNP: d_j = log OR_A,j - log OR_B,j, var(d_j) = se_A^2 + se_B^2 - 2 rho se_A se_B. Primary estimator: IVW of d_j
  on beta_x,j (lib_mr.mr_ivw, multiplicative random effects, selected explicitly), i.e. log(OR_A / OR_B) per exposure
  unit. Alongside: MR-Egger, weighted median and weighted mode of the per-SNP differences; the difference of the two
  separate IVW estimates with the analytic covariance (stacked IVW); nonparametric bootstrap over instruments
  (percentile CI); the same with the r2 <= 0.01 instrument sets.
- Cross-bed interaction ("profile" test): difference of two beds' adverse-oriented log ratios, e.g.
  (cIMT LAS/SVS) / (CRVE LAS/SVS); covariance from shared SNPs and from 1000G EUR LD between the two beds' instruments
  (pairs within 1 Mb). Omnibus per contrast: joint Wald test that all beds' log ratios are 0 (K df) and that they are
  equal (K-1 df; unit-dependent, shown with cIMT per 0.1 mm and per GWAS-implied SD 0.13 mm).

Families (BH within family): C1 MEGASTROKE large-artery vs small-vessel stroke (beds); C2 MVP IHD, MI and precerebral
stenosis vs cerebral infarction (beds); C3 secondary (LAS or SVS vs ischaemic stroke, IHD vs cerebral ischaemia); PC
SBP positive controls; I cross-bed interactions; pre-specified key hypotheses (q over those rows).

Conventions: instruments from i3 (instruments_ld.csv r2 <= 0.001; sens_r2_instruments.csv r2 <= 0.01), not
recomputed; retinal allele labels as corrected by analysis_i_mr ('swap_alleles'; the i3 instrument files already carry
the corrected alleles); cIMT per 0.1 mm (analysis_m_mr_mvp_outcomes spec), retina and ASI per SD (GWAS scale), SBP per
10 mmHg; outcome rows scanned with analysis_l_bed_outcomes.scan_all and harmonised with analysis_i_mr._harm (the same
code path as the l and m drivers). Beds with < 3 SNPs harmonised in both outcomes are not contrasted (plaque).

Imported unchanged: analysis_i_mr, analysis_i3_mr_ldclump, analysis_l_bed_outcomes, analysis_m_mr_mvp_outcomes,
lib_mr, s0an_common. Mode: --real.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis_i3_mr_ldclump as L3  # noqa: E402
import analysis_i_mr as I  # noqa: E402
import analysis_l_bed_outcomes as L  # noqa: E402
import analysis_m_mr_mvp_outcomes as MM  # noqa: E402
import lib_mr as M  # noqa: E402
import s0an_common as C  # noqa: E402

TASK = "m_mr_subtype_interaction"
HUB, GWAS = I.HUB, I.GWAS
I3_DIR = C.ANALYSIS / "i3_mr_ldclump"
L_DIR = C.ANALYSIS / "l_bed_outcomes"
M_DIR = C.ANALYSIS / "m_mr_mvp_outcomes"
I_DIR = C.ANALYSIS / "i_mr"
REFETCH_LOG = HUB / "logs" / "refetch_from_nas_20261006.txt"
STATUS = "exploratory"
T0 = time.time()

MEGA = ["stroke_mega", "istroke_mega", "las_mega", "svs_mega"]
MVP5 = ["mi_mvp", "ihd_mvp", "cinf_mvp", "cisch_mvp", "precereb_mvp"]
OUTS = MEGA + MVP5
CONS = {**{o: "MEGASTROKE" for o in MEGA}, **{o: "MVP" for o in MVP5}}
SHORT = {"stroke_mega": "AS", "istroke_mega": "IS", "las_mega": "LAS", "svs_mega": "SVS", "mi_mvp": "MI",
         "ihd_mvp": "IHD", "cinf_mvp": "CerebInfarct", "cisch_mvp": "CerebIsch", "precereb_mvp": "PrecerebSten"}
# Malik 2018 Nat Genet, MEGASTROKE European analyses: cases (controls = spec n_default - cases)
MEGA_CASES = {"stroke_mega": 40_585, "istroke_mega": 34_217, "las_mega": 4_373, "svs_mega": 5_386}
NESTED = {("las_mega", "istroke_mega"), ("svs_mega", "istroke_mega"), ("las_mega", "stroke_mega"),
          ("svs_mega", "stroke_mega"), ("istroke_mega", "stroke_mega"), ("mi_mvp", "ihd_mvp")}
CONTRASTS = [("C1_mega_LAS_vs_SVS", "las_mega", "svs_mega"),
             ("C2_mvp_vs_cerebral_infarction", "ihd_mvp", "cinf_mvp"),
             ("C2_mvp_vs_cerebral_infarction", "mi_mvp", "cinf_mvp"),
             ("C2_mvp_vs_cerebral_infarction", "precereb_mvp", "cinf_mvp"),
             ("C3_secondary", "las_mega", "istroke_mega"),
             ("C3_secondary", "svs_mega", "istroke_mega"),
             ("C3_secondary", "ihd_mvp", "cisch_mvp")]
PC_CONTRASTS = [("PC_sbp", "las_mega", "svs_mega"), ("PC_sbp", "ihd_mvp", "cinf_mvp"),
                ("PC_sbp", "mi_mvp", "cinf_mvp"), ("PC_sbp", "precereb_mvp", "cinf_mvp")]
# pre-specified key hypotheses: expected sign of the ADVERSE-ORIENTED log(OR_A / OR_B) (+1 = leans A)
HYP = {("cimt", "las_mega", "svs_mega"): 1, ("crve", "las_mega", "svs_mega"): -1, ("avr", "las_mega", "svs_mega"): -1,
       ("tort", "ihd_mvp", "cinf_mvp"): 1, ("cimt", "precereb_mvp", "cinf_mvp"): 1,
       ("avr", "ihd_mvp", "cinf_mvp"): -1, ("crve", "ihd_mvp", "cinf_mvp"): -1}
# cross-bed interactions (adverse-oriented): log ratio(bed1) - log ratio(bed2); expected sign +1
INTERACTIONS = [("I1", "las_mega", "svs_mega", "cimt", "crve"), ("I2", "las_mega", "svs_mega", "cimt", "avr"),
                ("I3", "ihd_mvp", "cinf_mvp", "tort", "crve"), ("I4", "ihd_mvp", "cinf_mvp", "tort", "avr"),
                ("I5", "precereb_mvp", "cinf_mvp", "cimt", "crve")]
BEDS = I.BEDS
BED_LABEL = MM.BED_LABEL
CIMT_SD_MM = 0.13          # GWAS-implied SD of cIMT (results/stage0/analysis/i_mr/report.md:38)
CIMT_SD_SRC = "results/stage0/analysis/i_mr/report.md:38"
Z_NULL, THIN_BP, EAF_MIN = 2.0, 100_000, 0.05
LD_WINDOW_BP = 1_000_000
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def log(msg: str):
    print(f"[{_dt.datetime.now():%H:%M:%S} +{time.time() - T0:7.1f}s] {msg}", flush=True)


def rel(p: Path) -> str:
    return MM.rel(p)


def fmt_ci(e, lo, hi, nd=3):
    return MM.fmt_ci(e, lo, hi, nd)


def adverse(e: str) -> int:
    return int(I.SPEC[e].get("adverse", 1)) if e != "sbp" else 1


# ================================================================================================================
# Null variants (error correlation between outcome files)
# ================================================================================================================
def null_scan_worker(task: dict) -> dict:
    """Stream one outcome file; keep rsIDs ending in '0' (fixed 10 % subsample), standardise with
    analysis_i_mr.standardize, keep non-palindromic SNVs with EAF in [EAF_MIN, 1 - EAF_MIN] and finite beta / SE."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    pa.set_cpu_count(task.get("threads", 2))
    t0 = time.time()
    spec, path = task["spec"], str(task["path"])
    c = spec["cols"]
    include = list(dict.fromkeys(c.values()))
    reader = pcsv.open_csv(
        pa.input_stream(path, compression="gzip"), read_options=pcsv.ReadOptions(block_size=32 << 20),
        parse_options=pcsv.ParseOptions(delimiter=spec["sep"]),
        convert_options=pcsv.ConvertOptions(include_columns=include, column_types=I._pa_types(spec),
                                            null_values=I.NA_VALUES, strings_can_be_null=True))
    parts, nrows = [], 0
    for batch in reader:
        nrows += batch.num_rows
        m = pc.fill_null(pc.ends_with(batch.column(c["snp"]), "0"), False)
        sub = batch.filter(m)
        if sub.num_rows:
            parts.append(sub)
    raw = pa.Table.from_batches(parts).to_pandas()
    d = I.standardize(spec, raw)
    snv = d["ea"].isin(list(COMP)) & d["oa"].isin(list(COMP))
    pal = d["ea"].map(COMP) == d["oa"]
    ok = (d["snp"].notna() & snv.fillna(False) & ~pal.fillna(False) & d["eaf"].between(EAF_MIN, 1 - EAF_MIN)
          & np.isfinite(d["beta"]) & np.isfinite(d["se"]) & (d["se"] > 0))
    d = d.loc[ok.values, ["snp", "ea", "oa", "beta", "se", "eaf"]].drop_duplicates("snp", keep=False)
    d = d.reset_index(drop=True)
    for k in ("snp", "ea", "oa"):
        d[k] = d[k].astype(str)
    return {"id": spec["id"], "df": d, "rows": nrows, "kept": len(d), "seconds": time.time() - t0}


def null_rows(ids: list, cache: Path, workers: int, stats_rows: list) -> dict:
    import hashlib
    out, todo = {}, []
    for i in ids:
        s = I.SPEC[i]
        p = GWAS / s["path"]
        st = p.stat()
        key = hashlib.sha1(json.dumps([i, str(p), st.st_size, int(st.st_mtime), s.get("cols"), s.get("special"),
                                       "rs_endswith_0", EAF_MIN], default=str).encode()).hexdigest()[:16]
        f = cache / f"{i}_null_{key}.parquet"
        if f.exists():
            out[i] = pd.read_parquet(f)
            stats_rows.append({"id": i, "scan": "null", "cached": True, "rows_kept": len(out[i])})
        else:
            todo.append((i, f, dict(spec=s, path=str(p), threads=2)))
    todo.sort(key=lambda x: -(GWAS / I.SPEC[x[0]]["path"]).stat().st_size)
    if todo:
        log(f"null-variant scan of {len(todo)} outcome files")
    for (i, f, _), r in zip(todo, I.run_pool(null_scan_worker, [t for _, _, t in todo], workers)):
        r["df"].to_parquet(f, index=False)
        out[i] = r["df"]
        stats_rows.append({"id": i, "scan": "null", "cached": False, "rows_scanned": r["rows"],
                           "rows_kept": r["kept"], "seconds": round(r["seconds"], 1)})
        log(f"  {i}: {r['rows']:,} rows, {r['kept']:,} null candidates kept, {r['seconds']:.1f}s")
    return out


def align_pair(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Merge two null-variant frames by rsID; z_b re-expressed for the effect allele of a (non-palindromic SNVs:
    direct, swapped, strand-flipped, strand-flipped + swapped); incompatible alleles dropped."""
    m = a.merge(b, on="snp", suffixes=("_a", "_b"))
    ca, cb = m["ea_b"].map(COMP), m["oa_b"].map(COMP)
    same = ((m["ea_a"] == m["ea_b"]) & (m["oa_a"] == m["oa_b"])) | ((m["ea_a"] == ca) & (m["oa_a"] == cb))
    swap = ((m["ea_a"] == m["oa_b"]) & (m["oa_a"] == m["ea_b"])) | ((m["ea_a"] == cb) & (m["oa_a"] == ca))
    m = m[(same | swap).values].copy()
    sgn = np.where(swap[same | swap].values & ~same[same | swap].values, -1.0, 1.0)
    m["z_a"] = m["beta_a"] / m["se_a"]
    m["z_b"] = sgn * m["beta_b"] / m["se_b"]
    return m[["snp", "z_a", "z_b"]].reset_index(drop=True)


def thin(chr_: np.ndarray, pos: np.ndarray, min_bp: int = THIN_BP) -> np.ndarray:
    """Greedy thinning: per chromosome, walk positions in order and keep a variant when it lies >= min_bp after the
    last kept one."""
    keep = np.zeros(len(pos), bool)
    order = np.lexsort((pos, chr_))
    last_c, last_p = None, -np.inf
    for i in order:
        if chr_[i] != last_c:
            last_c, last_p = chr_[i], -np.inf
        if pos[i] - last_p >= min_bp:
            keep[i] = True
            last_p = pos[i]
    return keep


def _box_prob(s1, s2, r, T=Z_NULL, n=2000):
    """P(|Z1| < T, |Z2| < T) for a centred bivariate normal (sd s1, s2; corr r), midpoint rule over Z1."""
    x = np.linspace(-T, T, n + 1)
    xm, dx = (x[:-1] + x[1:]) / 2, x[1] - x[0]
    mu, sd = r * s2 / s1 * xm, s2 * np.sqrt(1 - r * r)
    pc = stats.norm.cdf((T - mu) / sd) - stats.norm.cdf((-T - mu) / sd)
    return float(np.sum(stats.norm.pdf(xm, 0, s1) * pc) * dx)


def rho_mle(z1, z2, T=Z_NULL) -> dict:
    """Truncated bivariate-normal MLE of (sd1, sd2, rho) for z-scores observed only inside |z1|, |z2| < T."""
    def nll(par):
        s1, s2, r = np.exp(par[0]), np.exp(par[1]), np.tanh(par[2])
        q = (z1 / s1) ** 2 - 2 * r * (z1 / s1) * (z2 / s2) + (z2 / s2) ** 2
        ll = -np.log(2 * np.pi * s1 * s2 * np.sqrt(1 - r * r)) - q / (2 * (1 - r * r))
        return -(ll.sum() - len(z1) * np.log(_box_prob(s1, s2, r, T)))
    r0 = float(np.corrcoef(z1, z2)[0, 1])
    res = optimize.minimize(nll, [0.0, 0.0, np.arctanh(r0)], method="Nelder-Mead",
                            options=dict(xatol=1e-7, fatol=1e-7, maxiter=4000))
    return {"rho_mle": float(np.tanh(res.x[2])), "sd1_mle": float(np.exp(res.x[0])),
            "sd2_mle": float(np.exp(res.x[1])), "mle_converged": bool(res.success)}


def expected_rho_mega(a: str, b: str) -> float:
    """Lin & Sullivan 2009 approximation for MEGASTROKE pairs: all controls of the smaller control set shared; cases
    shared only for nested pairs (subtype cases within IS / AS). An approximation (cohort-level sharing unknown)."""
    if a not in MEGA_CASES or b not in MEGA_CASES:
        return np.nan
    n1a, n1b = MEGA_CASES[a], MEGA_CASES[b]
    n0a, n0b = I.SPEC[a]["n_default"] - n1a, I.SPEC[b]["n_default"] - n1b
    ns0 = min(n0a, n0b)
    ns1 = min(n1a, n1b) if ((a, b) in NESTED or (b, a) in NESTED) else 0
    num = ns0 * np.sqrt(n1a * n1b / (n0a * n0b)) + ns1 * np.sqrt(n0a * n0b / (n1a * n1b))
    return float(num / np.sqrt((n1a + n0a) * (n1b + n0b)))


def estimate_rho(nulls: dict, pos37: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grp in (MEGA, MVP5):
        for i, a in enumerate(grp):
            for b in grp[i + 1:]:
                if a not in nulls or b not in nulls:
                    continue
                m = align_pair(nulls[a], nulls[b]).merge(pos37, on="snp", how="inner")
                fin = np.isfinite(m["z_a"]) & np.isfinite(m["z_b"])
                m = m[fin].reset_index(drop=True)
                ch, ps = m["chr"].astype(str).values, m["pos"].astype(np.int64).values
                nul = (m["z_a"].abs() < Z_NULL) & (m["z_b"].abs() < Z_NULL)
                idx = np.flatnonzero(nul.values)
                k = idx[thin(ch[idx], ps[idx])]
                z1, z2 = m.loc[k, "z_a"].values, m.loc[k, "z_b"].values
                r = float(np.corrcoef(z1, z2)[0, 1])
                zf = np.arctanh(r)
                se = 1 / np.sqrt(len(k) - 3)
                kall = thin(ch, ps)
                r_all = float(np.corrcoef(m.loc[kall, "z_a"], m.loc[kall, "z_b"])[0, 1])
                mle = rho_mle(z1, z2)
                rows.append({"consortium": CONS[a], "outcome_a": a, "outcome_b": b,
                             "pair": f"{SHORT[a]} / {SHORT[b]}", "nested_cases": (a, b) in NESTED or (b, a) in NESTED,
                             "n_matched_candidates": len(m), "n_null_thinned": len(k),
                             "rho": r, "rho_lci": float(np.tanh(zf - 1.96 * se)),
                             "rho_uci": float(np.tanh(zf + 1.96 * se)), **mle,
                             "rho_all_thinned_no_z_filter": r_all, "n_all_thinned": int(kall.sum()),
                             "sd_z_a_null": float(np.std(z1, ddof=1)), "sd_z_b_null": float(np.std(z2, ddof=1)),
                             "expected_rho_lin_sullivan": expected_rho_mega(a, b)})
    return pd.DataFrame(rows)


# ================================================================================================================
# Contrast estimators
# ================================================================================================================
def _ivw_point(bx, d, w):
    return np.sum(w * bx * d, axis=-1) / np.sum(w * bx ** 2, axis=-1)


def pair_rows(hA: pd.DataFrame, hB: pd.DataFrame, sc: float) -> pd.DataFrame:
    """SNPs harmonised (to the exposure effect allele) in both outcome files."""
    a = hA[["snp", "chr", "pos", "ea", "oa", "beta_x", "se_x", "eaf_x", "beta_y", "se_y"]]
    b = hB[["snp", "ea", "oa", "beta_y", "se_y"]].rename(columns={"ea": "ea_b", "oa": "oa_b"})
    m = a.merge(b, on="snp", suffixes=("_A", "_B"))
    if not (m["ea"] == m["ea_b"]).all():
        raise RuntimeError("exposure effect allele differs between the two harmonised outcome sets")
    m["bx"], m["sx"] = m["beta_x"] / sc, m["se_x"] / sc
    return m.drop(columns=["ea_b", "oa_b"]).reset_index(drop=True)


def contrast(e: str, A: str, B: str, h: pd.DataFrame, rho: dict, n_boot: int, n_boot_mode: int,
             seed: int) -> tuple[dict, dict]:
    """Per-SNP difference IVW and sensitivities. Returns (summary row, internals for cross-bed covariance)."""
    bx, sx = h["bx"].values, h["sx"].values
    yA, sA, yB, sB = h["beta_y_A"].values, h["se_y_A"].values, h["beta_y_B"].values, h["se_y_B"].values
    J = len(h)
    r0 = rho["rho"]
    d = yA - yB
    sd = np.sqrt(sA ** 2 + sB ** 2 - 2 * r0 * sA * sB)
    S = {"exposure": e, "exposure_label": BED_LABEL.get(e, e), "unit": I.SPEC[e].get("unit_label", ""),
         "adverse_sign": adverse(e), "outcome_A": A, "outcome_B": B, "contrast": f"{SHORT[A]} vs {SHORT[B]}",
         "consortium": CONS[A], "nsnp": J, "rho": r0}
    rA, rB = M.mr_ivw(bx, yA, sA, "random"), M.mr_ivw(bx, yB, sB, "random")
    S.update({"logOR_A": rA["estimate"], "logOR_A_se": rA["se"], "p_A": rA["p"],
              "logOR_B": rB["estimate"], "logOR_B_se": rB["se"], "p_B": rB["p"]})
    res = I._methods(bx, d, sx, sd, n_boot, n_boot_mode, seed, full=True)
    iv = I._pick(res, I.IVW_PRIMARY)
    fx = I._pick(res, I.IVW_FIXED)
    S.update({"theta": iv["estimate"], "theta_se": iv["se"], "theta_lci": iv["lower"], "theta_uci": iv["upper"],
              "p": iv["p"], "theta_fixed_se": fx["se"], "Q": iv["Q"], "Q_df": iv["Q_df"], "Q_p": iv["Q_p"],
              "phi": iv["phi"]})
    by_m = {r["method"]: r for r in res}
    for key, name in [("egger", "MR-Egger"), ("wmedian", "Weighted median"), ("wmode", "Weighted mode")]:
        if name in by_m:
            g = by_m[name]
            S.update({key: g["estimate"], f"{key}_se": g["se"], f"{key}_lci": g["lower"], f"{key}_uci": g["upper"],
                      f"{key}_p": g["p"]})
            if key == "egger":
                S.update({"egger_intercept": g["intercept"], "egger_intercept_p": g["intercept_p"]})
    # rho sensitivities (rho = 0; truncation-corrected MLE)
    for tag, rr in [("rho0", 0.0), ("rhomle", rho["rho_mle"])]:
        sdx = np.sqrt(sA ** 2 + sB ** 2 - 2 * rr * sA * sB)
        g = M.mr_ivw(bx, d, sdx, "random")
        S.update({f"theta_{tag}": g["estimate"], f"se_{tag}": g["se"], f"p_{tag}": g["p"]})
    # stacked IVW: difference of the two separate IVWs with the analytic error covariance (same SNPs)
    wA, wB = 1 / sA ** 2, 1 / sB ** 2
    SA, SB = np.sum(wA * bx ** 2), np.sum(wB * bx ** 2)
    cov_f = np.sum(wA * wB * bx ** 2 * r0 * sA * sB) / (SA * SB)
    pA, pB = max(1.0, rA["phi"]), max(1.0, rB["phi"])
    v_st = rA["se"] ** 2 + rB["se"] ** 2 - 2 * cov_f * np.sqrt(pA * pB)
    t_st = rA["estimate"] - rB["estimate"]
    se_st = float(np.sqrt(max(v_st, 0.0)))
    S.update({"theta_stacked": t_st, "se_stacked": se_st,
              "p_stacked": float(2 * stats.norm.sf(abs(t_st / se_st))) if se_st > 0 else np.nan})
    # nonparametric bootstrap over instruments
    rng = np.random.default_rng(seed)
    w = 1 / sd ** 2
    ix = rng.integers(0, J, size=(n_boot, J))
    boots = _ivw_point(bx[ix], d[ix], w[ix])
    boots = boots[np.isfinite(boots)]
    S.update({"boot_lci": float(np.quantile(boots, 0.025)), "boot_uci": float(np.quantile(boots, 0.975)),
              "boot_se": float(np.std(boots, ddof=1)), "n_boot": len(boots)})
    # internals: per-SNP weights a_j (theta = sum a_j d_j) for cross-bed covariances
    a = w * bx / np.sum(w * bx ** 2)
    inner = {"snp": h["snp"].values, "ea": h["ea"].values, "oa": h["oa"].values, "chr": h["chr"].values,
             "pos": h["pos"].values, "a": a, "sA": sA, "sB": sB, "rho": r0, "phi": max(1.0, iv["phi"]),
             "theta": iv["estimate"], "se": iv["se"], "boots": boots}
    return S, inner


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["OR_A"], df["OR_B"] = np.exp(df["logOR_A"]), np.exp(df["logOR_B"])
    z = stats.norm.ppf(0.975)
    df["OR_A [95% CI]"] = [fmt_ci(np.exp(t), np.exp(t - z * s), np.exp(t + z * s))
                           for t, s in zip(df["logOR_A"], df["logOR_A_se"])]
    df["OR_B [95% CI]"] = [fmt_ci(np.exp(t), np.exp(t - z * s), np.exp(t + z * s))
                           for t, s in zip(df["logOR_B"], df["logOR_B_se"])]
    df["ratio"], df["ratio_lci"], df["ratio_uci"] = np.exp(df["theta"]), np.exp(df["theta_lci"]), \
        np.exp(df["theta_uci"])
    df["ratio [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in zip(df["ratio"], df["ratio_lci"], df["ratio_uci"])]
    df["boot ratio CI"] = [f"[{np.exp(a):.3g}, {np.exp(b):.3g}]" for a, b in zip(df["boot_lci"], df["boot_uci"])]
    for k in ("egger", "wmedian", "wmode"):
        if k in df.columns:
            df[f"{k}_ratio [95% CI]"] = [fmt_ci(np.exp(a), np.exp(b), np.exp(c)) if np.isfinite(a) else ""
                                         for a, b, c in zip(df[k], df[f"{k}_lci"], df[f"{k}_uci"])]
    df["stacked_ratio"] = np.exp(df["theta_stacked"])
    adv = df["adverse_sign"] * df["theta"]
    df["adverse_log_ratio"] = adv
    df["leans"] = np.where(adv > 0, df["outcome_A"].map(SHORT), df["outcome_B"].map(SHORT))
    df["leans"] = [f"{x} (p<0.05)" if p < 0.05 else f"{x} (n.s.)" for x, p in zip(df["leans"], df["p"])]
    signs = np.sign(df[["theta", "egger", "wmedian", "wmode"]].values)
    df["consistent_direction"] = [bool(np.all(s == s[0]) and s[0] != 0) if np.all(np.isfinite(s)) else None
                                  for s in signs]
    df["boot_excludes_1"] = (df["boot_lci"] > 0) | (df["boot_uci"] < 0)
    return df


def bh_by(df: pd.DataFrame, fam_col: str, out_col: str = "q") -> pd.DataFrame:
    df[out_col] = np.nan
    for _, g in df.groupby(fam_col):
        ok = g["p"].notna()
        df.loc[g.index[ok], out_col] = I.bh(g.loc[ok, "p"].values)
    return df


# ================================================================================================================
# LD between instruments of different beds (1000G EUR) and cross-bed covariance
# ================================================================================================================
class LDHelper:
    """1000G EUR panel rows of the bed instruments (matched once) and pairwise LD r (cached)."""

    def __init__(self, ldref, inst_frames: list):
        self.ld = ldref
        self.cache: dict = {}
        d = pd.concat([I.with_ids(x)[["snp", "chr", "pos", "ea", "oa"]] for x in inst_frames], ignore_index=True)
        d = d.drop_duplicates("snp").reset_index(drop=True)
        mt = ldref.match(d)
        a1 = pd.Series(pd.NA, index=d.index, dtype="string")
        ok = mt["ref_idx"] >= 0
        if ok.any():
            t = ldref.bim.take(mt.loc[ok, "ref_idx"].values).to_pandas()
            a1.loc[ok] = t["a1"].astype("string").str.upper().values
        ea = d["ea"].astype("string")
        sg = np.where(ok & ((ea == a1) | (ea.map(COMP) == a1)).fillna(False), 1.0, -1.0)
        self.map = {s: (int(r), str(c) if pd.notna(c) else "", int(p) if pd.notna(p) else -1, str(e), g)
                    for s, r, c, p, e, g in zip(d["snp"], mt["ref_idx"], mt["ref_chr"], mt["ref_pos"], d["ea"], sg)}
        self.n_matched, self.n_total = int(ok.sum()), len(d)

    def arrays(self, snps, eas):
        """ref_idx, ref chr, ref pos and the sign turning panel-A1 coding into the given effect-allele coding."""
        ref, ch, ps, sg = [], [], [], []
        for s, e in zip(snps, eas):
            m = self.map.get(s)
            if m is None or m[0] < 0:
                ref.append(-1); ch.append(""); ps.append(-1); sg.append(np.nan)
                continue
            ref.append(m[0]); ch.append(m[1]); ps.append(m[2])
            sg.append(m[4] if e == m[3] else -m[4])
        return np.array(ref, np.int64), np.array(ch, object), np.array(ps, np.int64), np.array(sg, float)

    def r(self, i1: int, i2: int) -> float:
        key = (min(i1, i2), max(i1, i2))
        if key not in self.cache:
            X = self.ld.std_rows(np.array([i1, i2], np.int64))
            self.cache[key] = float(X[0] @ X[1])
        return self.cache[key]


def cross_cov(c1: dict, c2: dict, ldh: LDHelper | None) -> tuple[float, int, int, float]:
    """Covariance of two beds' difference estimates (same outcome pair, theta = sum_j a_j d_j): shared SNPs (r = +/-1
    by effect allele) and distinct SNPs within 1 Mb with 1000G EUR LD r (oriented to each effect allele). Per-pair
    covariance of d: r (sA_j sA_k + sB_j sB_k - rho (sA_j sB_k + sB_j sA_k)). Random-effects scaling
    sqrt(phi1 phi2)."""
    s1, s2 = np.asarray(c1["snp"], object), np.asarray(c2["snp"], object)
    e1, e2 = np.asarray(c1["ea"], object), np.asarray(c2["ea"], object)
    R = np.zeros((len(s1), len(s2)))
    same = s1[:, None] == s2[None, :]
    R[same] = np.where(e1[:, None] == e2[None, :], 1.0, -1.0)[same]
    n_ld, max_r = 0, 0.0
    if ldh is not None:
        r1, c1h, p1, g1 = ldh.arrays(s1, e1)
        r2, c2h, p2, g2 = ldh.arrays(s2, e2)
        cand = (~same & (r1 >= 0)[:, None] & (r2 >= 0)[None, :] & (c1h[:, None] == c2h[None, :])
                & (np.abs(p1[:, None] - p2[None, :]) <= LD_WINDOW_BP))
        for j, k in np.argwhere(cand):
            R[j, k] = ldh.r(int(r1[j]), int(r2[k])) * g1[j] * g2[k]
            n_ld += 1
            max_r = max(max_r, abs(R[j, k]))
    sA1, sB1 = np.asarray(c1["sA"])[:, None], np.asarray(c1["sB"])[:, None]
    sA2, sB2 = np.asarray(c2["sA"])[None, :], np.asarray(c2["sB"])[None, :]
    rho = c1["rho"]
    Cm = R * (sA1 * sA2 + sB1 * sB2 - rho * (sA1 * sB2 + sB1 * sA2))
    cov = float(np.asarray(c1["a"]) @ Cm @ np.asarray(c2["a"]))
    return cov * np.sqrt(c1["phi"] * c2["phi"]), int(same.sum()), n_ld, max_r


# ================================================================================================================
def run(args, outdir: Path, rep, notes: list):
    MM._register()                    # l + i3 specs, MVP specs, cIMT per 0.1 mm
    cache = outdir / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    f_main, f_sens = I3_DIR / "instruments_ld.csv", I3_DIR / "sens_r2_instruments.csv"
    inst = {k: v for k, v in L.read_instruments(f_main).items() if k in BEDS + ["sbp"]}
    inst_s = {k: v for k, v in L.read_instruments(f_sens).items() if k in BEDS + ["sbp"]}
    missing = [o for o in OUTS if not (GWAS / I.SPEC[o]["path"]).exists()]
    if missing:
        raise SystemExit(f"outcome files missing (re-fetch from the NAS first): {missing}")

    # ---------------- scans ----------------
    u_rs, u_pairs = set(), set()
    for d in list(inst.values()) + list(inst_s.values()):
        u_rs |= set(d["snp"].dropna())
        ok = d["pos"].notna()
        u_pairs |= set(I.cpkey(d.loc[ok, "chr"], d.loc[ok, "pos"]))
    scan_stats = []
    t_scan = time.time()
    ext = L.scan_all(OUTS, u_rs, u_pairs, cache, args.workers, scan_stats)
    nulls = null_rows(OUTS, cache, args.workers, scan_stats)
    scan_s = time.time() - t_scan
    vmap = I.VariantMap()
    for d in list(inst.values()) + list(inst_s.values()):
        vmap.add(d)
    for i, d in ext.items():
        if I.SPEC[i]["build"] == "37":
            vmap.add(d)
    for i, d in ext.items():
        if I.SPEC[i]["build"] == "38":
            vmap.fill_pos37(d)

    # ---------------- LD panel: null-variant positions and instrument LD ----------------
    log("loading the 1000G EUR panel index")
    ldref = L3.LDRef(L3.LDREF_PREFIX)
    all_null_rs = sorted(set().union(*[set(d["snp"]) for d in nulls.values()]))
    pr = ldref.rows_by_rsid(all_null_rs)
    pos37 = pr[pr["chr"].isin([str(c) for c in range(1, 23)])].drop_duplicates("snp", keep=False)[
        ["snp", "chr", "pos"]].astype({"snp": str, "chr": str})
    log(f"null candidates with a unique autosomal 1000G EUR position: {len(pos37):,} of {len(all_null_rs):,}")
    rho_tab = estimate_rho(nulls, pos37)
    rho_map = {}
    for _, r in rho_tab.iterrows():
        v = {"rho": r["rho"], "rho_mle": r["rho_mle"]}
        rho_map[(r["outcome_a"], r["outcome_b"])] = v
        rho_map[(r["outcome_b"], r["outcome_a"])] = v
    log("rho per pair: " + "; ".join(f"{p} {x:.3f} (n={n})" for p, x, n in
                                     zip(rho_tab["pair"], rho_tab["rho"], rho_tab["n_null_thinned"])))

    # ---------------- registry and SHA-256 ----------------
    for f in (f_main, f_sens):
        rep.inputs.append({"id": f.name, "path": rel(f), "bytes": f.stat().st_size, "sha256": C.sha256(f)})
    gpaths = [GWAS / I.SPEC[i]["path"] for i in OUTS]
    ldpaths = [L3.LDREF_DIR / f for f in ("g1000_eur.bed", "g1000_eur.bim", "g1000_eur.fam", "g1000_eur.synonyms")]
    sha = {} if args.no_hash else MM.hash_files(gpaths + ldpaths, cache / "sha256.json",
                                                 [I3_DIR / "_cache" / "sha256_ldref.json"], min(args.workers, 4))
    verified = {}
    if REFETCH_LOG.exists():
        for line in REFETCH_LOG.read_text().splitlines():
            t = line.split()
            if len(t) >= 3 and t[0].startswith("external/") and t[2] == "OK":
                verified[t[0]] = t[1]
    reg = []
    for i in OUTS:
        s, p = I.SPEC[i], GWAS / I.SPEC[i]["path"]
        cnt = MM.MVP_COUNTS.get(i, {})
        h = sha.get(p)
        reg.append({"id": i, "short": SHORT[i], "consortium": CONS[i], "label": s["label"], "file": rel(p),
                    "build": s["build"], "n_total": s["n_default"],
                    "n_cases": MEGA_CASES.get(i, cnt.get("n_cases")),
                    "n_controls": (s["n_default"] - MEGA_CASES[i]) if i in MEGA_CASES else cnt.get("n_controls"),
                    "rows_at_instrument_keys": len(ext[i]), "null_candidates": len(nulls[i]),
                    "lambda_gc_null_candidates": float(np.median((nulls[i]["beta"] / nulls[i]["se"]) ** 2)
                                                       / stats.chi2.ppf(0.5, 1)),
                    "bytes": p.stat().st_size, "sha256": h,
                    "sha256_equals_nas_verified": None if h is None else verified.get(rel(p)) == h,
                    "source": s["source"]})
        rep.inputs.append({"id": i, "path": rel(p), "bytes": p.stat().st_size, "sha256": h})
    for p in ldpaths:
        rep.inputs.append({"id": p.name, "path": rel(p), "bytes": p.stat().st_size, "sha256": sha.get(p),
                           "expected_sha256": L3.LDREF_SHA256.get(p.name)})
    registry = pd.DataFrame(reg)
    for f in (L_DIR / "mr_all_pairs_summary.csv", I3_DIR / "mr_all_pairs_summary.csv",
              M_DIR / "mr_all_pairs_summary.csv"):
        rep.inputs.append({"id": f"earlier_{f.parent.name}", "path": rel(f), "bytes": f.stat().st_size,
                           "sha256": C.sha256(f)})

    # ---------------- harmonisation (analysis_i_mr._harm, as in the l / m drivers) ----------------
    def harm(e, o, src):
        dp = not (I.eur(I.SPEC[e]) and I.eur(I.SPEC[o]))
        hk, _ = I._harm(I.with_ids(src[e]), ext[o], dp)
        return hk

    H = {"main": {}, "r2sens": {}}
    for setname, src in (("main", inst), ("r2sens", inst_s)):
        for e in BEDS + ["sbp"]:
            if e not in src:
                continue
            for o in OUTS:
                H[setname][(e, o)] = harm(e, o, src)

    # check: separate IVW on each full harmonised set vs the earlier drivers (path:line)
    chk = []
    efile = {"l": L_DIR / "mr_all_pairs_summary.csv", "i3": I3_DIR / "mr_all_pairs_summary.csv",
             "m": M_DIR / "mr_all_pairs_summary.csv"}
    earlier = {k: pd.read_csv(f) for k, f in efile.items()}
    for k in earlier:
        earlier[k]["_line"] = earlier[k].index + 2
    src_fam = {"las_mega": ("l", "L_beds_to_new_outcomes", "L_pc_sbp_to_new_outcomes"),
               "svs_mega": ("l", "L_beds_to_new_outcomes", "L_pc_sbp_to_new_outcomes"),
               "istroke_mega": ("i3", "X_E_beds_to_nonUKB_outcomes", "X_B_nonUKB_outcomes"),
               "stroke_mega": ("i3", "X_E_beds_to_nonUKB_outcomes", "X_B_nonUKB_outcomes"),
               **{o: ("m", "M_beds_to_mvp", "M_pc") for o in MVP5}}
    for (e, o), hk in H["main"].items():
        if len(hk) == 0:
            continue
        sc = float(I.SPEC[e].get("scale", 1.0))
        g = M.mr_ivw(hk["beta_x"].values / sc, hk["beta_y"].values, hk["se_y"].values, "random")
        task, fam_b, fam_pc = src_fam[o]
        d = earlier[task]
        m = d[(d["family"] == (fam_pc if e == "sbp" else fam_b)) & (d["exposure"] == e) & (d["outcome"] == o)]
        k = 0.1 if (e == "cimt" and task in ("l", "i3")) else 1.0       # earlier l / i3 cIMT rows are per mm
        row = {"exposure": e, "outcome": o, "nsnp_here": len(hk), "ivw_here": g["estimate"], "se_here": g["se"]}
        if len(m) == 1:
            r = m.iloc[0]
            row.update({"nsnp_earlier": int(r["nsnp"]), "ivw_earlier": r["ivw"] * k, "se_earlier": r["ivw_se"] * k,
                        "source": f"{rel(efile[task])}:{int(r['_line'])}"})
            row["abs_diff"] = abs(row["ivw_here"] - row["ivw_earlier"])
        chk.append(row)
    chk = pd.DataFrame(chk)

    # ---------------- contrasts ----------------
    rows, inner = [], {}
    seed = args.seed
    for setname in ("main", "r2sens"):
        src = inst if setname == "main" else inst_s
        jobs = [(f, A, B, e) for f, A, B in CONTRASTS for e in BEDS] + \
               [(f, A, B, "sbp") for f, A, B in PC_CONTRASTS]
        if setname == "r2sens":
            jobs = [j for j in jobs if j[0].startswith(("C1", "C2", "PC"))]
        for fam, A, B, e in jobs:
            if e not in src:
                continue
            sc = float(I.SPEC[e].get("scale", 1.0))
            h = pair_rows(H[setname][(e, A)], H[setname][(e, B)], sc)
            base = {"family": fam if setname == "main" else f"{fam}_r2sens", "instrument_set": setname,
                    "exposure": e, "outcome_A": A, "outcome_B": B, "nsnp": len(h),
                    "n_instruments": len(src[e]), "same_set_as_main": len(inst.get(e, [])) == len(inst_s.get(e, []))}
            if len(h) < 3:
                rows.append({**base, "exposure_label": BED_LABEL.get(e, e), "contrast": f"{SHORT[A]} vs {SHORT[B]}",
                             "consortium": CONS[A], "note": f"not contrasted: {len(h)} SNP(s) in both outcomes"})
                continue
            seed += 1
            S, inn = contrast(e, A, B, h, rho_map[(A, B)], args.n_boot, min(args.n_boot, 1000), seed)
            rows.append({**base, **S})
            inner[(setname, e, A, B)] = inn
    res = pd.DataFrame(rows)
    done = res[res["theta"].notna()].copy()
    skipped = res[res["theta"].isna()].copy()
    done = add_derived(done)
    done = bh_by(done, "family", "q_family")
    done["hypothesis"] = ""
    done["agrees_with_hypothesis"] = None
    for (e, A, B), sgn in HYP.items():
        m = (done["exposure"] == e) & (done["outcome_A"] == A) & (done["outcome_B"] == B)
        done.loc[m, "hypothesis"] = f"leans {SHORT[A] if sgn > 0 else SHORT[B]}"
        done.loc[m, "agrees_with_hypothesis"] = (np.sign(done.loc[m, "adverse_log_ratio"]) == sgn).values
    pre = done["hypothesis"].ne("") & done["instrument_set"].eq("main")
    done["q_prespecified"] = np.nan
    done.loc[pre, "q_prespecified"] = I.bh(done.loc[pre, "p"].values)
    done["status"] = STATUS

    # ---------------- cross-bed interactions and omnibus ----------------
    ldh = LDHelper(ldref, [inst[e] for e in BEDS if e in inst] + [inst_s[e] for e in BEDS if e in inst_s])
    log(f"bed instruments matched to the 1000G EUR panel: {ldh.n_matched} of {ldh.n_total}")
    irows = []
    for setname in ("main", "r2sens"):
        for iid, A, B, e1, e2 in INTERACTIONS:
            k1, k2 = (setname, e1, A, B), (setname, e2, A, B)
            if k1 not in inner or k2 not in inner:
                continue
            c1, c2 = inner[k1], inner[k2]
            s1, s2 = adverse(e1), adverse(e2)
            cv, n_sh, n_ld, max_r = cross_cov(c1, c2, ldh)
            for scale_tag, f1 in [("cIMT per 0.1 mm", 1.0), (f"cIMT per SD ({CIMT_SD_MM} mm)", CIMT_SD_MM / 0.1)]:
                if scale_tag != "cIMT per 0.1 mm" and "cimt" not in (e1, e2):
                    continue
                g1 = f1 if e1 == "cimt" else 1.0
                g2 = f1 if e2 == "cimt" else 1.0
                t1, t2 = s1 * g1 * c1["theta"], s2 * g2 * c2["theta"]
                v = (g1 * c1["se"]) ** 2 + (g2 * c2["se"]) ** 2 - 2 * s1 * s2 * g1 * g2 * cv
                est, se = t1 - t2, float(np.sqrt(max(v, 0)))
                v0 = (g1 * c1["se"]) ** 2 + (g2 * c2["se"]) ** 2
                bt = s1 * g1 * c1["boots"][: min(len(c1["boots"]), len(c2["boots"]))] - \
                    s2 * g2 * c2["boots"][: min(len(c1["boots"]), len(c2["boots"]))]
                z = est / se
                irows.append({"id": iid, "instrument_set": setname, "scale": scale_tag,
                              "contrast": f"{SHORT[A]} vs {SHORT[B]}", "bed1": e1, "bed2": e2,
                              "description": f"({BED_LABEL[e1]} {SHORT[A]}/{SHORT[B]}) / ({BED_LABEL[e2]} "
                                             f"{SHORT[A]}/{SHORT[B]}), adverse-oriented",
                              "adv_log_ratio_bed1": t1, "adv_log_ratio_bed2": t2, "estimate": est, "se": se,
                              "p": float(2 * stats.norm.sf(abs(z))), "cov_bed1_bed2": s1 * s2 * g1 * g2 * cv,
                              "p_ignoring_cov": float(2 * stats.norm.sf(abs(est / np.sqrt(v0)))),
                              "n_shared_snps": n_sh, "n_ld_pairs_within_1mb": n_ld, "max_abs_ld_r": max_r,
                              "ratio_of_ratios": float(np.exp(est)), "lci": float(np.exp(est - 1.96 * se)),
                              "uci": float(np.exp(est + 1.96 * se)),
                              "boot_lci": float(np.exp(np.quantile(bt, 0.025))),
                              "boot_uci": float(np.exp(np.quantile(bt, 0.975))), "expected": "> 1"})
    inter = pd.DataFrame(irows)
    if len(inter):
        inter["ratio_of_ratios [95% CI]"] = [fmt_ci(a, b, c) for a, b, c in
                                             zip(inter["ratio_of_ratios"], inter["lci"], inter["uci"])]
        inter["boot CI"] = [f"[{a:.3g}, {b:.3g}]" for a, b in zip(inter["boot_lci"], inter["boot_uci"])]
        inter["q_family"] = np.nan
        for (s, sc_), g in inter.groupby(["instrument_set", "scale"]):
            inter.loc[g.index, "q_family"] = I.bh(g["p"].values)
        inter["status"] = STATUS

    orow = []
    for setname in ("main", "r2sens"):
        for fam, A, B in CONTRASTS:
            if setname == "r2sens" and not fam.startswith(("C1", "C2")):
                continue
            ks = [e for e in BEDS if (setname, e, A, B) in inner]
            if len(ks) < 2:
                continue
            K = len(ks)
            V = np.zeros((K, K))
            for i, e in enumerate(ks):
                V[i, i] = inner[(setname, e, A, B)]["se"] ** 2
            for i in range(K):
                for j in range(i + 1, K):
                    cv, _, _, _ = cross_cov(inner[(setname, ks[i], A, B)], inner[(setname, ks[j], A, B)], ldh)
                    V[i, j] = V[j, i] = cv
            sgn = np.array([adverse(e) for e in ks], float)
            for scale_tag, fc in [("cIMT per 0.1 mm", 1.0), (f"cIMT per SD ({CIMT_SD_MM} mm)", CIMT_SD_MM / 0.1)]:
                g = np.array([fc if e == "cimt" else 1.0 for e in ks])
                th = sgn * g * np.array([inner[(setname, e, A, B)]["theta"] for e in ks])
                Va = V * np.outer(sgn * g, sgn * g)
                x2_0 = float(th @ np.linalg.solve(Va, th))
                Cm = np.hstack([-np.ones((K - 1, 1)), np.eye(K - 1)])
                ct = Cm @ th
                x2_eq = float(ct @ np.linalg.solve(Cm @ Va @ Cm.T, ct))
                orow.append({"instrument_set": setname, "contrast": f"{SHORT[A]} vs {SHORT[B]}", "family": fam,
                             "beds": ",".join(ks), "K": K, "scale": scale_tag,
                             "chi2_all_zero": x2_0, "df_all_zero": K, "p_all_zero": float(stats.chi2.sf(x2_0, K)),
                             "chi2_all_equal": x2_eq, "df_all_equal": K - 1,
                             "p_all_equal": float(stats.chi2.sf(x2_eq, K - 1))})
    omni = pd.DataFrame(orow)

    # ---------------- report ----------------
    ref_rows = []
    if REFETCH_LOG.exists():
        for line in REFETCH_LOG.read_text().splitlines():
            t = line.split()
            if len(t) >= 3 and t[0].startswith("external/"):
                ref_rows.append({"file": t[0], "sha256_12": t[1][:12], "status": " ".join(t[2:])})
            elif line.startswith("# done"):
                ref_rows.append({"file": "(free disk after re-fetch)", "sha256_12": "", "status": line[2:]})
    rep.h("Re-fetched outcome files (NAS, SHA-256 checked before rename)")
    rep.p(f"Log: `{rel(REFETCH_LOG)}`.")
    rep.table("refetch_status", pd.DataFrame(ref_rows), "Re-fetch log", max_rows=None)
    rep.table("outcome_registry", registry[["id", "short", "consortium", "build", "n_cases", "n_controls",
                                            "rows_at_instrument_keys", "null_candidates",
                                            "lambda_gc_null_candidates", "sha256",
                                            "sha256_equals_nas_verified"]], "Outcome files", max_rows=None)

    rep.h("Error correlation rho per outcome pair (null variants)")
    rep.p(f"Null variants: rsID ending in 0, EAF {EAF_MIN}-{1 - EAF_MIN} in both files, non-palindromic SNVs, "
          f"|z| < {Z_NULL} in both, autosomal 1000G EUR GRCh37 position, greedily thinned to >= {THIN_BP // 1000} kb. "
          "rho = Pearson r (primary; attenuated by the truncation, so conservative for positive rho); rho_mle = "
          "truncated bivariate-normal MLE; rho_all = all thinned variants without the |z| filter (includes genetic "
          "covariance); expected = Lin & Sullivan approximation from pooled counts (MEGASTROKE only; all controls of "
          "the smaller set shared, cases shared only for nested pairs). Reading: (1) the pooled-count approximation "
          "fits the nested MEGASTROKE pairs (0.34-0.41 vs observed 0.32-0.36) but not LAS / SVS (0.014 vs 0.096): "
          "per-cohort case:control ratios in a meta-analysis raise the overlap correlation above the pooled value; "
          "(2) the MLE assumes one bivariate normal for the truncated z, so with polygenic inflation (MVP IHD lambda "
          "about 2 on null candidates, outcome_registry.csv) it absorbs genetic covariance (sd_mle 1.9-2.1 for IHD) "
          "and is an upper sensitivity, not the error correlation.")
    rshow = rho_tab[["pair", "consortium", "nested_cases", "n_null_thinned", "rho", "rho_lci", "rho_uci", "rho_mle",
                     "sd1_mle", "sd2_mle", "rho_all_thinned_no_z_filter", "n_all_thinned",
                     "expected_rho_lin_sullivan"]]
    rep.tables["rho_by_pair"] = rho_tab.copy()
    rep.p("")
    rep.p("**rho by outcome pair** (`rho_by_pair.csv`)")
    rep.p("")
    rep.p(C.md_table(rshow, 4))
    rep.p("")

    show = ["exposure_label", "contrast", "nsnp", "rho", "OR_A [95% CI]", "OR_B [95% CI]", "ratio [95% CI]", "p",
            "q_family", "boot ratio CI", "wmedian_ratio [95% CI]", "egger_ratio [95% CI]", "p_rho0", "p_rhomle",
            "stacked_ratio", "p_stacked", "Q_p", "leans", "hypothesis", "agrees_with_hypothesis",
            "consistent_direction"]
    unit_note = ("Ratio = OR_A / OR_B per exposure unit (cIMT per 0.1 mm, retina and ASI per SD on the GWAS scale, "
                 "SBP per 10 mmHg); IVW (multiplicative random effects) of per-SNP log-OR differences with "
                 "var = se_A^2 + se_B^2 - 2 rho se_A se_B. 'leans' uses the adverse orientation (CRAE, AVR and "
                 "density: lower is adverse, so a ratio < 1 means the adverse direction acts more on outcome A).")
    main = done[done["instrument_set"] == "main"]
    rep.h("C1 MEGASTROKE: large-artery vs small-vessel stroke (beds)")
    rep.p(unit_note)
    rep.table("contrasts_C1_mega", main[main["family"] == "C1_mega_LAS_vs_SVS"][show], "LAS vs SVS", max_rows=None)
    rep.h("C2 MVP: IHD, MI and precerebral stenosis vs cerebral infarction (beds)")
    rep.table("contrasts_C2_mvp", main[main["family"] == "C2_mvp_vs_cerebral_infarction"][show],
              "MVP contrasts (BH within the 21 rows)", max_rows=None)
    rep.h("Pre-specified key hypotheses")
    hshow = ["exposure_label", "contrast", "nsnp", "ratio [95% CI]", "p", "q_prespecified", "q_family",
             "boot ratio CI", "wmedian_ratio [95% CI]", "p_rho0", "hypothesis", "leans", "agrees_with_hypothesis"]
    rep.table("prespecified_hypotheses", main[main["hypothesis"].ne("")][hshow],
              "Key hypotheses (q_prespecified = BH over these rows)", max_rows=None)
    rep.h("Cross-bed interaction (bed x subtype) and omnibus tests")
    rep.p("Estimate = adverse-oriented log ratio of bed 1 minus that of bed 2; ratio_of_ratios > 1 means bed 1 leans "
          "more to outcome A than bed 2 does. Covariance from shared SNPs and 1000G EUR LD within 1 Mb (random-effects "
          "scaled); bootstrap CI from independent instrument resampling per bed. The per-unit interaction depends on "
          "the exposure units: shown with cIMT per 0.1 mm and per GWAS-implied SD "
          f"({CIMT_SD_MM} mm; {CIMT_SD_SRC}).")
    if len(inter):
        ishow = ["id", "instrument_set", "scale", "description", "ratio_of_ratios [95% CI]", "p", "q_family",
                 "boot CI", "p_ignoring_cov", "n_shared_snps", "n_ld_pairs_within_1mb", "max_abs_ld_r"]
        rep.table("interactions", inter[ishow], "Cross-bed interactions (q within instrument set x scale)",
                  max_rows=None)
    rep.p("Omnibus: chi2_all_zero tests that no bed's log ratio differs from 0 (K df; unit-free); chi2_all_equal tests "
          "that all beds have the same adverse-oriented log ratio (K-1 df; unit-dependent).")
    rep.table("omnibus", omni, "Omnibus Wald tests per contrast", max_rows=None)
    rep.h("Positive controls: SBP (i3 instruments)")
    rep.table("contrasts_pc_sbp", main[main["family"] == "PC_sbp"][show], "SBP per 10 mmHg", max_rows=None)
    rep.h("C3 secondary contrasts (nested outcomes: subtype vs ischaemic stroke; IHD vs cerebral ischaemia)")
    rep.table("contrasts_C3_secondary", main[main["family"] == "C3_secondary"][show], "Secondary", max_rows=None)
    rep.h("Sensitivity: r2 <= 0.01 instrument sets")
    s2 = done[done["instrument_set"] == "r2sens"]
    rep.p("Only tortuosity (83 -> 101 instruments) and SBP (541 -> 1004) change; the other beds have identical sets "
          "(same_set_as_main).")
    rep.table("contrasts_r2sens", s2[["family", "exposure_label", "contrast", "nsnp", "same_set_as_main",
                                      "ratio [95% CI]", "p", "q_family", "boot ratio CI", "leans"]],
              "r2 <= 0.01", max_rows=None)
    if len(skipped):
        rep.table("not_contrasted", skipped[["family", "exposure", "contrast", "nsnp", "note"]],
                  "Pairs not contrasted (< 3 SNPs in both outcomes)", max_rows=None)
    rep.h("Check: separate IVW on the full harmonised sets vs the earlier drivers")
    rep.table("check_vs_earlier", chk, "Same code path; cIMT rows of l / i3 rescaled per 0.1 mm", max_rows=None)
    done.to_csv(outdir / "contrasts_all.csv", index=False)
    pd.DataFrame(scan_stats).to_csv(outdir / "scan_stats.csv", index=False)
    snps = []
    for (setname, e, A, B), inn in inner.items():
        snps.append(pd.DataFrame({"instrument_set": setname, "exposure": e, "outcome_A": A, "outcome_B": B,
                                  "snp": inn["snp"], "ea": inn["ea"], "a_weight": inn["a"], "se_A": inn["sA"],
                                  "se_B": inn["sB"]}))
    pd.concat(snps, ignore_index=True).to_csv(outdir / "contrast_snp_weights.csv.gz", index=False,
                                              compression="gzip")
    return {"done": done, "inter": inter, "omni": omni, "rho": rho_tab, "chk": chk, "scan_s": scan_s}


def main():
    def extra(ap):
        ap.add_argument("--workers", type=int, default=4, help="parallel file scans (each up to 2 pyarrow threads)")
        ap.add_argument("--no-hash", action="store_true")
    args = C.parse_args(__doc__.splitlines()[0], extra)
    if args.simulate:
        raise SystemExit("--simulate is not implemented for this driver (it reuses validated drivers i, i3, l, m)")
    if args.workers > 6:
        raise SystemExit("at most 6 workers")
    mode, outdir, token = C.gate(args, TASK)
    if mode != "real" or token is None:
        raise SystemExit("real mode needs the gate token")
    rep = C.Report(TASK, "F1 Stage 0 L11: MR bed x outcome-subtype interaction (ratio of MR odds ratios)", mode,
                   args)
    notes = [
        "MR is exploratory. Ratios compare causal log-odds per exposure unit between two outcomes of one consortium; "
        "they do not compare absolute risks (different baseline incidences).",
        "MEGASTROKE subtypes (TOAST) come from a subset of cohorts; MVP phecodes are EHR code groupings; MVP has no "
        "lacunar / small-vessel phecode (cerebral artery occlusion with infarction is the nearest stroke outcome).",
        "MI is nested in IHD; LAS and SVS are nested in IS; MVP phecode case sets can overlap (comorbidity). rho "
        "captures the resulting error correlation only approximately (null-variant z correlation).",
        "Multiplicative random effects partly absorb a misspecified rho (Q of the differences inflates the SE); "
        "p with rho = 0 is the conservative bound for positive rho.",
        "Few instruments for ASI (4), CRAE / AVR (7) and cIMT (8): Egger, mode and bootstrap CIs are coarse. Plaque "
        "has 1 SNP in both outcome files and is not contrasted.",
        "Retina and ASI exposure GWAS are UK Biobank-based; outcomes exclude UKB (no exposure-outcome sample overlap). "
        "Keaton 2024 SBP includes MVP (overlap in the SBP positive controls only).",
        "Cross-bed interactions on the per-unit scale depend on the exposure units; a sign reversal between beds is "
        "unit-free, a difference in magnitude is not.",
        "The per-SNP difference IVW is more precise than the difference of the two separate random-effects IVWs "
        "(stacked; p_stacked) when heterogeneity is shared by both outcomes (pleiotropy acting on both cancels in "
        "the difference); e.g. cIMT precerebral stenosis vs cerebral infarction p 0.009 (difference) vs 0.11 "
        "(stacked). Both are reported.",
        "With 4-8 instruments the nonparametric bootstrap CI over instruments is too narrow (small-sample "
        "under-coverage; e.g. AVR LAS vs SVS); read it as a check of the point estimate's stability, not as the "
        "primary interval.",
        "MVP outcome GWAS show genome-wide inflation on null candidates (lambda 1.17-1.99; outcome_registry.csv); "
        "MR SEs are not rescaled for it (as in the m driver).",
        "Cross-bed instruments in LD (1000G EUR; max_abs_ld_r in interactions.csv): chr8 10.61-10.68 Mb (cIMT "
        "rs2277130 with CRVE rs1073913 and AVR rs6601527) and chr14 60.79-60.89 Mb (tortuosity rs2350892 with AVR "
        "rs1272131); their covariance is included in the interaction SEs (p_ignoring_cov shows the difference).",
    ]
    rep.p("**Exploratory.** Open public data only; no HPP data; no liver phenotypes. Not paper numbers until checked "
          "by the F1 session. MR is exploratory.")
    out = run(args, outdir, rep, notes)
    rep.h("Notes and caveats")
    rep.bullets(notes)
    rep.h("Runtime")
    rep.p(f"File scans {out['scan_s']:.0f} s; total {time.time() - T0:.0f} s (scan workers {args.workers}).")
    C.finish(rep, outdir, __file__)
    log("finished")


if __name__ == "__main__":
    main()
