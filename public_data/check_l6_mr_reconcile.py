#!/usr/bin/env python3
"""check_l6_mr_reconcile.py - F1 L6 (PLAN 9.3a): reconcile the LD-clumped MR (I3) with the distance-clumped i_mr and
i2_mr_ext for every risk factor -> bed estimate cited for Fig. 6c-d (PLAN 9.2 row 6: MR:26, :50-58, :69;
MR2:138-177), plus all 80 family-A pairs as a supplement.

EXPLORATORY. Reads existing CSVs only (no GWAS scan). LD-clumped MVMR models that I3 did not fit are read from
check_l6_mvmr_ld_gapfill.py output (l6_mvmr_ld_gapfill.csv) when present.

Verdict rule (distance or i2 reference -> LD r2 <= 0.001):
- changed sign: opposite signs and p < 0.05 in at least one of the two;
- lost: p < 0.05 -> p >= 0.05 (same sign);
- weakened: p < 0.05 in both, but the LD estimate is < 0.67 x the reference, or the univariable class falls from
  robust/supported to suggestive or inconsistent direction;
- gained: p >= 0.05 -> p < 0.05;
- consistent: everything else (incl. null in both). The r2 <= 0.01 reading is reported next to it.

Output: results/stage0/analysis/l6_l8_l9_checks_20261006/l6_cited_pairs.csv, l6_familyA_all.csv,
l6_instrument_counts.csv; markdown tables printed to stdout.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(os.environ.get("F1_WORKDIR", Path(__file__).resolve().parents[1]))
AN = PAPER / "results" / "stage0" / "analysis"
I3, I2, IMR = AN / "i3_mr_ldclump", AN / "i2_mr_ext", AN / "i_mr"
OUT = AN / "l6_l8_l9_checks_20261006"
SUPPORTED = {"robust", "supported"}


def fmt(e, lo, hi, nd=3):
    if pd.isna(e):
        return "—"
    f = lambda x: f"{x:.{nd}g}" if abs(x) >= 1e-3 else f"{x:.2g}"
    return f"{f(e)} [{f(lo)}, {f(hi)}]"


def coarse(c):
    c = str(c)
    for k in ["robust", "supported", "suggestive", "inconsistent direction", "no evidence", "insufficient"]:
        if c.startswith(k):
            return k
    return c


def verdict(e_ref, p_ref, e_ld, p_ld, c_ref=None, c_ld=None):
    if any(pd.isna(x) for x in (e_ref, p_ref, e_ld, p_ld)):
        return "not tested"
    s_ref, s_ld = p_ref < 0.05, p_ld < 0.05
    if np.sign(e_ref) != np.sign(e_ld) and (s_ref or s_ld):
        return "changed sign"
    if s_ref and not s_ld:
        return "lost"
    if not s_ref and s_ld:
        return "gained"
    if s_ref and s_ld:
        if abs(e_ld) < 0.67 * abs(e_ref):
            return "weakened"
        if c_ref is not None and coarse(c_ref) in SUPPORTED and coarse(c_ld) in ("suggestive", "inconsistent direction"):
            return "weakened"
        return "consistent"
    return "consistent (null in both)"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmp = pd.read_csv(I3 / "comparison_pairs.csv")
    imr = pd.read_csv(IMR / "mr_all_pairs_summary.csv")
    mvc = pd.read_csv(I3 / "mvmr_comparison.csv")
    xwb = pd.read_csv(I3 / "mvmr.csv")
    gap = pd.read_csv(OUT / "l6_mvmr_ld_gapfill.csv") if (OUT / "l6_mvmr_ld_gapfill.csv").exists() else pd.DataFrame()
    rows = []

    def uni(src, fam, e, o, label):
        r = cmp[(cmp.family == fam) & (cmp.exposure == e) & (cmp.outcome == o)]
        if r.empty:
            rows.append({"source": src, "pair": label, "type": "univariable", "verdict": "not in I3"})
            return
        r = r.iloc[0]
        q = imr[(imr.family == fam) & (imr.exposure == e) & (imr.outcome == o)]
        n_imr = int(q["nsnp"].iloc[0]) if len(q) else np.nan
        v = verdict(r.ivw_dist, r.ivw_p_dist, r.ivw_ld, r.ivw_p_ld, r.class_distance, r.class_ld)
        v2 = verdict(r.ivw_dist, r.ivw_p_dist, r.ivw_r2sens, r.ivw_p_r2sens, r.class_distance, r.class_ld_r2sens)
        rows.append({"source": src, "pair": label, "type": "univariable",
                     "reference": "i_mr distance ±1 Mb" if fam != "C_chami_uncoupled" else "i_mr unclumped 266 leads",
                     "ref_est_ci": fmt(r.ivw_dist, r.ivw_lci_dist, r.ivw_uci_dist), "ref_p": r.ivw_p_dist,
                     "ref_nsnp": int(r.nsnp_dist), "imr_nsnp_check": n_imr, "ref_class": coarse(r.class_distance),
                     "ld_est_ci": fmt(r.ivw_ld, r.ivw_lci_ld, r.ivw_uci_ld), "ld_p": r.ivw_p_ld,
                     "ld_nsnp": int(r.nsnp_ld), "ld_class": coarse(r.class_ld),
                     "r2sens_est_ci": fmt(r.ivw_r2sens, r.ivw_lci_r2sens, r.ivw_uci_r2sens), "r2sens_p": r.ivw_p_r2sens,
                     "r2sens_nsnp": int(r.nsnp_r2sens), "r2sens_class": coarse(r.class_ld_r2sens),
                     "ratio_ld_ref": r.ivw_ld / r.ivw_dist, "verdict": v, "verdict_r2_0.01": v2,
                     "imr_no_eaf_instruments": int(r.imr_variants_without_eaf)})

    def mv_from_cmp(src, exposures, o, term, label):
        r = mvc[(mvc.exposures == exposures) & (mvc.outcome == o) & (mvc.term == term)].iloc[0]
        v = verdict(r.estimate_dist, r.p_dist, r.estimate_ld, r.p_ld)
        v2 = verdict(r.estimate_dist, r.p_dist, r.estimate_r2sens, r.p_r2sens)
        rows.append({"source": src, "pair": label, "type": "MVMR",
                     "reference": "i_mr distance ±1 Mb (joint)",
                     "ref_est_ci": fmt(r.estimate_dist, r.lower_dist, r.upper_dist), "ref_p": r.p_dist,
                     "ref_nsnp": int(r.nsnp_dist), "ref_condF": r.conditional_F_dist,
                     "ld_est_ci": fmt(r.estimate_ld, r.lower_ld, r.upper_ld), "ld_p": r.p_ld, "ld_nsnp": int(r.nsnp_ld),
                     "ld_condF": r.conditional_F_ld,
                     "r2sens_est_ci": fmt(r.estimate_r2sens, r.lower_r2sens, r.upper_r2sens), "r2sens_p": r.p_r2sens,
                     "r2sens_nsnp": int(r.nsnp_r2sens), "ratio_ld_ref": r.estimate_ld / r.estimate_dist,
                     "verdict": v, "verdict_r2_0.01": v2})

    def mv_i2(src, i2file, exposures, o, term, label, ld=None, ld_s=None, ref_filter=None):
        d = pd.read_csv(I2 / i2file)
        d = d[(d.exposures == exposures) & (d.outcome == o) & (d.term == term)] if ref_filter is None else ref_filter(d)
        r = d.iloc[0]
        out = {"source": src, "pair": label, "type": "MVMR", "reference": f"i2 distance ±1 Mb ({i2file})",
               "ref_est_ci": fmt(r.estimate, r.lower, r.upper), "ref_p": r.p, "ref_nsnp": int(r.nsnp),
               "ref_condF": r.conditional_F}
        if ld is not None and len(ld):
            a = ld.iloc[0]
            out.update({"ld_est_ci": fmt(a.estimate, a.lower, a.upper), "ld_p": a.p, "ld_nsnp": int(a.nsnp),
                        "ld_condF": a.conditional_F, "ratio_ld_ref": a.estimate / r.estimate,
                        "verdict": verdict(r.estimate, r.p, a.estimate, a.p),
                        "ld_source": "I3" if "q_within_family_term" in a.index else "L6 gap-fill"})
        else:
            out.update({"verdict": "not tested (no LD-clumped fit)"})
        if ld_s is not None and len(ld_s):
            b = ld_s.iloc[0]
            out.update({"r2sens_est_ci": fmt(b.estimate, b.lower, b.upper), "r2sens_p": b.p,
                        "r2sens_nsnp": int(b.nsnp), "verdict_r2_0.01": verdict(r.estimate, r.p, b.estimate, b.p)})
        rows.append(out)

    def g(exposures, o, term, r2, model=None):
        if gap.empty:
            return pd.DataFrame()
        q = gap[(gap.exposures == exposures) & (gap.outcome == o) & (gap.term == term) & (gap.r2_thresh == r2)]
        if model is not None:
            q = q[q.model == model]
        return q[q["family"].astype(str).str.startswith(("MV_", "Chami", "ANCHOR"))]

    # MR:50-58 SBP -> beds; MR:69 WHRadjBMI -> ASI (+ SBP)
    beds = [("asi", "ASI", 50), ("cimt", "cIMT", 51), ("plaque", "Plaque", 52), ("crae", "CRAE", 53),
            ("avr", "AVR", 54), ("dens", "Density", 55), ("crve", "CRVE", 56), ("pad_mvp", "PAD (MVP)", 57),
            ("tort", "Tortuosity", 58)]
    for o, lab, ln in beds:
        uni(f"MR:{ln}", "A_risk_to_beds", "sbp", o, f"SBP → {lab}")
    uni("MR:69", "A_risk_to_beds", "whradjbmi", "asi", "WHRadjBMI → ASI")
    mv_from_cmp("MR:69", "whradjbmi+sbp", "asi", "whradjbmi", "WHRadjBMI → ASI | SBP (MVMR)")
    # MR2:138-141 adiposity MVMR on ASI
    xw = xwb[(xwb.exposures == "whradjbmi+bmi") & (xwb.outcome == "asi")]
    mv_i2("MR2:138", "mv_whradjbmi_bmi.csv", "whradjbmi+bmi", "asi", "whradjbmi", "WHRadjBMI → ASI | BMI (MVMR)",
          xw[xw.term == "whradjbmi"], g("whradjbmi+bmi", "asi", "whradjbmi", 0.01))
    mv_i2("MR2:138", "mv_whradjbmi_bmi.csv", "whradjbmi+bmi", "asi", "bmi", "BMI → ASI | WHRadjBMI (MVMR)",
          xw[xw.term == "bmi"], g("whradjbmi+bmi", "asi", "bmi", 0.01))
    mv_i2("MR2:139", "mv_whradjbmi_bmi_sbp.csv", "whradjbmi+bmi+sbp", "asi", "whradjbmi",
          "WHRadjBMI → ASI | BMI + SBP (MVMR)", g("whradjbmi+bmi+sbp", "asi", "whradjbmi", 0.001),
          g("whradjbmi+bmi+sbp", "asi", "whradjbmi", 0.01))
    mv_i2("MR2:140", "mv_whr_bmi.csv", "whr+bmi", "asi", "whr", "WHR → ASI | BMI (MVMR)",
          g("whr+bmi", "asi", "whr", 0.001), g("whr+bmi", "asi", "whr", 0.01))
    mv_i2("MR2:140", "mv_whr_bmi.csv", "whr+bmi", "asi", "bmi", "BMI → ASI | WHR (MVMR)",
          g("whr+bmi", "asi", "bmi", 0.001), g("whr+bmi", "asi", "bmi", 0.01))
    mv_i2("MR2:141", "mv_whr_bmi_sbp.csv", "whr+bmi+sbp", "asi", "whr", "WHR → ASI | BMI + SBP (MVMR)",
          g("whr+bmi+sbp", "asi", "whr", 0.001), g("whr+bmi+sbp", "asi", "whr", 0.01))
    # MR2:146-150 Chami (univariable family C and MVMR)
    for o, lab, src in [("asi", "ASI", "MR2:146"), ("plaque", "Plaque", "MR2:148"), ("crve", "CRVE", "MR2:150")]:
        uni(src, "C_chami_uncoupled", "chami_bfp", o, f"Uncoupled adiposity (BFP) → {lab}")
    ch = pd.read_csv(I2 / "chami_mvmr.csv")
    for o, lab, models, src in [("asi", "ASI", ["BFP + SBP", "BFP + SBP + TG"], "MR2:146-147"),
                                ("plaque", "Plaque", ["BFP + SBP"], "MR2:148"),
                                ("crve", "CRVE", ["BFP + SBP + TG"], "MR2:150")]:
        for mdl in models:
            ref = ch[(ch.variant_set == "all 266 published leads") & (ch.model == mdl) & (ch.outcome == o) &
                     (ch.term == "bfp")]
            r = ref.iloc[0]
            ld = g("chami_bfp+" + "+".join(x.strip().lower() for x in mdl.split("+")[1:]), o, "chami_bfp", 0.001, mdl)
            ld_s = g("chami_bfp+" + "+".join(x.strip().lower() for x in mdl.split("+")[1:]), o, "chami_bfp", 0.01, mdl)
            out = {"source": src, "pair": f"BFP → {lab} | {mdl[6:]} (MVMR)", "type": "MVMR",
                   "reference": "i2 unclumped 266 leads (chami_mvmr.csv)",
                   "ref_est_ci": fmt(r.estimate, r.lower, r.upper), "ref_p": r.p, "ref_nsnp": int(r.nsnp),
                   "ref_condF": r.conditional_F, "ref_same_snps_bfp_alone": r.ivw_same_snps}
            if len(ld):
                a = ld.iloc[0]
                out.update({"ld_est_ci": fmt(a.estimate, a.lower, a.upper), "ld_p": a.p, "ld_nsnp": int(a.nsnp),
                            "ld_condF": a.conditional_F, "ld_same_snps_bfp_alone": a.ivw_same_snps,
                            "ratio_ld_ref": a.estimate / r.estimate, "verdict": verdict(r.estimate, r.p, a.estimate, a.p),
                            "ld_source": "L6 gap-fill (LD-clumped leads)"})
            else:
                out["verdict"] = "not tested (no LD-clumped fit)"
            if len(ld_s):
                b = ld_s.iloc[0]
                out.update({"r2sens_est_ci": fmt(b.estimate, b.lower, b.upper), "r2sens_p": b.p,
                            "r2sens_nsnp": int(b.nsnp), "verdict_r2_0.01": verdict(r.estimate, r.p, b.estimate, b.p)})
            rows.append(out)
    # MR2:163-175 OSA -> cIMT
    uni("MR2:164", "A_risk_to_beds", "osa", "cimt", "OSA (MVP) → cIMT")
    uni("MR2:165", "A2_osa_alternatives", "osa_kurn", "cimt", "OSA (Kurniansyah) → cIMT")
    uni("MR2:166", "A2_osa_alternatives", "osa_campos", "cimt", "OSA (Campos) → cIMT")
    uni("MR2:167", "A2_osa_alternatives", "osa_kurn_bmiadj", "cimt", "OSA BMI-adjusted (Kurniansyah) → cIMT")
    mv_from_cmp("MR2:169", "osa+sbp", "cimt", "osa", "OSA (MVP) → cIMT | SBP (MVMR)")
    lab = {"osa": "MVP", "osa_campos": "Campos", "osa_kurn": "Kurniansyah", "osa_kurn_bmiadj": "Kurniansyah BMI-adj"}
    for x in ["osa", "osa_kurn", "osa_campos", "osa_kurn_bmiadj"]:
        for adj, fam, src in [("bmi", "MV_osa_bmi", "MR2:170-174"), ("bmi+sbp", "MV_osa_bmi_sbp", "MR2:171-172")]:
            ex = f"{x}+{adj}"
            mv_i2(src, "mv_osa.csv", ex, "cimt", x, f"OSA ({lab[x]}) → cIMT | {adj.upper().replace('+', ' + ')} (MVMR)",
                  g(ex, "cimt", x, 0.001), g(ex, "cimt", x, 0.01),
                  ref_filter=lambda d, ex=ex, x=x, fam=fam: d[(d.family == fam) & (d.exposures == ex) &
                                                               (d.outcome == "cimt") & (d.term == x)])
        if x != "osa":
            ex = f"{x}+sbp"
            mv_i2("MR2:172", "mv_osa.csv", ex, "cimt", x, f"OSA ({lab[x]}) → cIMT | SBP (MVMR)",
                  g(ex, "cimt", x, 0.001), g(ex, "cimt", x, 0.01),
                  ref_filter=lambda d, ex=ex, x=x: d[(d.family == "MV_osa_sbp") & (d.exposures == ex) &
                                                     (d.outcome == "cimt") & (d.term == x)])
    cited = pd.DataFrame(rows)
    cited.to_csv(OUT / "l6_cited_pairs.csv", index=False)

    # MR:26 instrument counts (i_mr distance vs I3 LD r2 0.001 / 0.01)
    ic = pd.read_csv(I3 / "sens_r2_instrument_counts.csv")
    ld = pd.read_csv(I3 / "ld_instruments_by_exposure.csv")[["exposure", "in_both", "excluded_no_eaf_row"]]
    ic = ic.merge(ld, on="exposure", how="left")
    ic.to_csv(OUT / "l6_instrument_counts.csv", index=False)

    # all family-A pairs
    A = cmp[cmp.family.isin(["A_risk_to_beds", "A2_osa_alternatives"])].copy()
    A["verdict"] = [verdict(a, b, c, d, e, f) for a, b, c, d, e, f in
                    zip(A.ivw_dist, A.ivw_p_dist, A.ivw_ld, A.ivw_p_ld, A.class_distance, A.class_ld)]
    A["verdict_r2_0.01"] = [verdict(a, b, c, d, e, f) for a, b, c, d, e, f in
                            zip(A.ivw_dist, A.ivw_p_dist, A.ivw_r2sens, A.ivw_p_r2sens, A.class_distance,
                                A.class_ld_r2sens)]
    A["imr_fg_not_citable"] = (A.exposure == "fg")
    keep = ["family", "exposure", "outcome", "nsnp_dist", "IVW distance [95% CI]", "ivw_p_dist", "class_distance",
            "nsnp_ld", "IVW LD [95% CI]", "ivw_p_ld", "class_ld", "nsnp_r2sens", "IVW LD r2sens [95% CI]",
            "ivw_p_r2sens", "class_ld_r2sens", "verdict", "verdict_r2_0.01", "imr_variants_without_eaf",
            "imr_fg_not_citable"]
    A[keep].to_csv(OUT / "l6_familyA_all.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 60)
    cols = ["source", "pair", "ref_est_ci", "ref_p", "ref_nsnp", "ref_class", "ld_est_ci", "ld_p", "ld_nsnp",
            "ld_class", "r2sens_est_ci", "r2sens_p", "r2sens_nsnp", "verdict", "verdict_r2_0.01"]
    print(cited[[c for c in cols if c in cited]].to_string())
    print()
    print(ic.to_string())
    print()
    t = A[A.family == "A_risk_to_beds"]
    print("family A (80 pairs) verdicts r2 0.001:", t["verdict"].value_counts().to_dict())
    print("family A verdicts r2 0.01:", t["verdict_r2_0.01"].value_counts().to_dict())
    tnf = t[t.exposure != "fg"]
    print("family A without FG (70) r2 0.001:", tnf["verdict"].value_counts().to_dict())
    print(A[~A.verdict.str.startswith("consistent")][keep[:17]].to_string())


if __name__ == "__main__":
    main()
