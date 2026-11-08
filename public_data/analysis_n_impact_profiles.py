#!/usr/bin/env python3
"""L13: fixed continuous two-bed profile versus IMPACT hard outcomes; no figures."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import analysis_d_impact_abi as D
import lib_survival as ls
import s0an_common as C

TASK = "n_impact_profiles"
PROTOCOL = C.PAPER / "docs/L13_IMPACT_PROFILE_PROTOCOL_20261006.md"
COVS = ["age10", "female", "hemodialysis", "diabetes", "egfr_jsn10"]


def fit(s, terms, outcome="mace_cv", horizon=None):
    t = s[f"{outcome}_time_y"].to_numpy(float)
    e = s[f"{outcome}_event"].to_numpy(float)
    x = s[terms].to_numpy(float)
    keep = np.isfinite(t) & np.isfinite(e) & np.isfinite(x).all(axis=1) & (t > 0)
    t, e, x = t[keep], e[keep], x[keep]
    if horizon is not None:
        e = e * (t <= horizon)
        t = np.minimum(t, horizon)
    if np.linalg.matrix_rank(x - x.mean(axis=0)) != len(terms):
        raise ValueError("rank deficient Cox design")
    f = ls.coxph(t, e, x, names=terms, robust=True, ties="efron")
    if not f.converged or f.warnings:
        raise RuntimeError(f"Cox fit not clean: {f.warnings}")
    return f


def rows(f, label, role):
    tab = f.table("robust")
    tab.insert(0, "model", label)
    tab.insert(1, "role", role)
    tab.insert(2, "n", f.n)
    tab.insert(3, "events", f.n_events)
    tab["loglik"] = f.loglik
    tab["converged"] = f.converged
    return tab


def analyse(raw):
    d, qc = D.prepare(raw)
    needed = ["bapwv_max_cms", "abi_min"] + COVS + ["mace_cv_time_y", "mace_cv_event"]
    age = d.age_y >= 20
    eligible = age & (d.abi_min >= .9)
    cc = eligible & np.isfinite(d[needed]).all(axis=1) & (d.mace_cv_time_y > 0)
    s = d.loc[cc].copy()
    if len(s) < 100 or s.mace_cv_event.sum() < 20:
        raise ValueError("insufficient primary population")
    flow = pd.DataFrame([
        {"step": "public source", "n": len(d), "mace_cv_events": int(d.mace_cv_event.sum())},
        {"step": "age>=20", "n": int(age.sum()), "mace_cv_events": int(d.loc[age, "mace_cv_event"].sum())},
        {"step": "age>=20 and ABI>=0.9", "n": int(eligible.sum()), "mace_cv_events": int(d.loc[eligible, "mace_cv_event"].sum())},
        {"step": "primary complete cases", "n": len(s), "mace_cv_events": int(s.mace_cv_event.sum())},
    ])
    scales = []
    for col, target, sign in [("bapwv_max_cms", "P", 1), ("abi_min", "A", -1),
                              ("bapwv_mean_cms", "P_mean", 1)]:
        mu, sd = float(s[col].mean()), float(s[col].std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError("invalid primary-population scale")
        s[target] = sign * (s[col] - mu) / sd
        scales.append({"source": col, "oriented": target, "mean": mu, "sd_ddof1": sd, "sign": sign})
    s["B"], s["D"] = (s.P + s.A) / 2, (s.P - s.A) / 2
    base = fit(s, ["B"] + COVS)
    main = fit(s, ["B", "D"] + COVS)
    bed = fit(s, ["P", "A"] + COVS)
    tables = [rows(main, "primary_equal_burden", "PRIMARY"),
              rows(bed, "joint_beds", "same-person context"),
              rows(base, "burden_only", "same-person context"),
              rows(fit(s, ["P"] + COVS), "PWV_only", "same-person context"),
              rows(fit(s, ["A"] + COVS), "ABI_only", "same-person context")]
    # D coefficient must equal beta_P-beta_A, including covariance.
    contrast = np.zeros(len(bed.coef)); contrast[:2] = [1, -1]
    equivalence = pd.DataFrame([{
        "coef_abs_difference": abs(main.coef[1] - contrast @ bed.coef),
        "variance_abs_difference": abs(main.var_robust[1, 1] - contrast @ bed.var_robust @ contrast),
        "loglik_abs_difference": abs(main.loglik - bed.loglik),
    }])
    if (equivalence.to_numpy() > 1e-6).any():
        raise RuntimeError("joint-bed/profile reparametrisation disagrees")
    knots = np.quantile(s.B, [.1, .5, .9])
    s["B_nonlinear"] = ls.rcs_basis(s.B, knots)[:, 1]
    tables.append(rows(fit(s, ["B", "B_nonlinear", "D"] + COVS),
                       "nonlinear_burden", "sensitivity"))
    for col, out, sign in [("bapwv_max_cms", "P_rank", 1), ("abi_min", "A_rank", -1)]:
        z = sign * stats.norm.ppf((stats.rankdata(s[col]) - .5) / len(s))
        s[out] = (z - z.mean()) / z.std(ddof=1)
    s["B_rank"], s["D_rank"] = (s.P_rank + s.A_rank) / 2, (s.P_rank - s.A_rank) / 2
    tables.append(rows(fit(s, ["B_rank", "D_rank"] + COVS), "rank_normal_beds", "sensitivity"))
    s["B_mean"], s["D_mean"] = (s.P_mean + s.A) / 2, (s.P_mean - s.A) / 2
    tables.append(rows(fit(s, ["B_mean", "D_mean"] + COVS), "mean_PWV", "sensitivity"))
    tables.append(rows(fit(s.loc[s.hemodialysis == 0], ["B", "D"] + [c for c in COVS if c != "hemodialysis"]),
                       "exclude_HD", "sensitivity"))
    tables.append(rows(fit(s.loc[s.bapwv_max_cms <= 4000], ["B", "D"] + COVS),
                       "exclude_PWV_gt4000", "sensitivity"))
    tables.append(rows(fit(s, ["B", "D"] + COVS, horizon=5), "censor_5years", "sensitivity"))
    tables.append(rows(fit(s, ["B", "D"] + COVS, outcome="mace_all"),
                       "allcause_composite", "SECONDARY"))
    coefs = pd.concat(tables, ignore_index=True)
    summary = coefs.loc[coefs.term.isin(["B", "D", "D_rank", "D_mean"]),
                       ["model", "role", "term", "n", "events", "coef", "se", "exp_coef", "exp_lower", "exp_upper", "p"]]
    nested = pd.DataFrame([{"comparison": "joint versus equal-burden only", "lr_chi2": 2 * (main.loglik - base.loglik),
                            "df": 1, "model_based_p": stats.chi2.sf(2 * (main.loglik - base.loglik), 1),
                            "primary_inference": "robust Wald for D; not this LRT"}])
    ph = ls.cox_zph(main, "km")
    describe = pd.DataFrame([{"measure": "P versus A Pearson r", "value": s.P.corr(s.A)},
                             {"measure": "median MACE observed follow-up years", "value": s.mace_cv_time_y.median()},
                             {"measure": "burden spline knot10", "value": knots[0]},
                             {"measure": "burden spline knot50", "value": knots[1]},
                             {"measure": "burden spline knot90", "value": knots[2]}])
    return dict(flow=flow, scales=pd.DataFrame(scales), profile_summary=summary,
                model_coefficients=coefs, nested_comparison=nested, ph_diagnostics=ph,
                reconstruction_qc=qc, reparametrisation=equivalence, descriptives=describe)


def known_answer():
    rng = np.random.default_rng(17)
    n = 2400
    p, a = rng.normal(size=(2, n))
    b, dd = (p + a) / 2, (p - a) / 2
    z = rng.normal(size=n)
    x = np.column_stack([b, dd, z])
    truth = np.array([.25, .4, .2])
    event_time = rng.exponential(size=n) / (.13 * np.exp(x @ truth))
    censor_time = rng.uniform(2, 8, n)
    e = (event_time <= censor_time).astype(int)
    t = np.maximum(np.minimum(event_time, censor_time).round(2), .005)
    f = ls.coxph(t, e, x, names=["B", "D", "Z"], robust=True)
    g = ls.coxph(t, e, np.column_stack([p, a, z]), names=["P", "A", "Z"], robust=True)
    l = np.array([1., -1., 0.])
    checks = [
        ("known B slope within3SE", abs(f.coef[0] - truth[0]) <= 3 * np.sqrt(f.var_robust[0, 0])),
        ("known D slope within3SE", abs(f.coef[1] - truth[1]) <= 3 * np.sqrt(f.var_robust[1, 1])),
        ("joint/profile coefficient identity", abs(f.coef[1] - l @ g.coef) < 1e-7),
        ("joint/profile robust variance identity", abs(f.var_robust[1, 1] - l @ g.var_robust @ l) < 1e-7),
        ("joint/profile likelihood and convergence", f.converged and g.converged and abs(f.loglik - g.loglik) < 1e-7),
    ]
    table = pd.DataFrame([{"check": k, "pass": bool(v)} for k, v in checks])
    if not table["pass"].all():
        raise RuntimeError("known-answer/reparametrisation verification failed")
    return table


def main():
    args = C.parse_args(__doc__)
    mode, outdir, token = C.gate(args, TASK)
    if (outdir / "provenance.json").exists():
        raise FileExistsError("L13 completed output already exists; do not overwrite")
    rep = C.Report(TASK, "F1 L13: IMPACT continuous equal-burden bed profile and MACE", mode, args)
    rep.inputs.append({"protocol": str(PROTOCOL), "sha256": C.sha256(PROTOCOL) if PROTOCOL.exists() else None})
    if mode == "simulate":
        raw = D.simulate(np.random.default_rng(args.seed))
        rep.h("Changed-logic checks")
        rep.table("known_answer", known_answer(), max_rows=None)
        wiring = C.check_columns(raw, D.TABLE, D.USED)
        rep.bullets([f"Schema wiring {int(wiring.ok.sum())}/{len(wiring)} passed; no real participant data read."])
    else:
        raw, provenance = C.load_real(D.TABLE, token, columns=D.USED)
        expected = "070d98e5ff905abef4565e108eb8058a87f2c457cbb3b435b760175c7f2ea137"
        if provenance["sha256"] != expected:
            raise ValueError("public clean table differs from manifest")
        rep.inputs.append(provenance)
    result = analyse(raw)
    rep.h("Fixed primary and bounded sensitivities")
    rep.bullets([
        "Primary: robust two-sided Wald for D in rebuilt CV-death/MI/stroke MACE, conditional on B and fixed covariates.",
        "D+1 at fixed B means PWV+1SD and adverse ABI−1SD; exp(beta_D)=exp(beta_P−beta_A). No latent class or synergy test.",
        "One primary test; sensitivities/secondary endpoint do not replace it. The equal-SD burden is a chosen statistical scale.",
        "No BP/HR/lipid adjustment; CVD inpatients and ABI>=0.9. No general-population screening or incremental prediction claim.",
    ])
    for key in ["flow", "profile_summary", "scales", "nested_comparison", "ph_diagnostics", "descriptives", "reconstruction_qc", "reparametrisation"]:
        rep.h(key.replace("_", " ").capitalize())
        rep.table(key, result[key], nd=6, max_rows=None)
    rep.tables["model_coefficients"] = result["model_coefficients"]
    rep.h("Sources and interpretation limits")
    rep.bullets([
        "Protocol: docs/L13_IMPACT_PROFILE_PROTOCOL_20261006.md (frozen before new fit). Existing single-bed outcomes were already seen.",
        "Preprocessing: scripts/stage0/analysis_d_impact_abi.py:124; engine: scripts/stage0/lib_survival.py:383. Original MACE times are not directly used.",
        "Prior combined baPWV/ABI prognosis: Park et al., Coron Artery Dis2020, PMID31821193, https://pubmed.ncbi.nlm.nih.gov/31821193/ (abstract checked2026-10-06).",
        "This extension can add partial external evidence; it cannot by itself close four-bed hard-event replication or clinical-utility gaps.",
    ])
    rep.write(outdir, __file__)
    provfile = outdir / "provenance.json"
    prov = json.loads(provfile.read_text())
    prov["preprocessing_sha256"] = C.sha256(Path(D.__file__))
    provfile.write_text(json.dumps(prov, indent=2) + "\n")
    primary = result["profile_summary"].query("model == 'primary_equal_burden' and term == 'D'").iloc[0]
    print(f"L13 {mode}: n{primary.n} events{primary.events}; profile HR {primary.exp_coef:.6g} [{primary.exp_lower:.6g},{primary.exp_upper:.6g}], robust p{primary.p:.6g}; {outdir}")


if __name__ == "__main__":
    main()
