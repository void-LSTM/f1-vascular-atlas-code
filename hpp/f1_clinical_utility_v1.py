#!/usr/bin/env python3
"""P13: family-grouped cross-validated PWV/ABI prediction increments; aggregates only.

Frozen P1/P10 definitions; composite hypertension primary, objective hypertension
sensitivity, treatment starts exploratory. Baseline measured BP only: no usual-BP
calibration. Five family-grouped folds, training-only imputation/scaling, fixed
ridge logistic models. Cluster bootstrap resamples within fixed folds and refits
the entire prediction pipeline. Corrections remain the frozen atlas transform;
raw baseline beds are a sensitivity, not independently validated clinical use.
"""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold

VERSION = "f1_clinical_utility_v1.0"
EXPECT = {"atlas": "62344144b83db3f2", "outcomes": "9bee14ddca223c57", "p10": "e8cc37eaa90adef6"}
MINCELL, MIN_N, EPV, FOLDS, RIDGE = 20, 200, 5, 5, 1.0
THRESH = (0.05, 0.10, 0.20)
MODELS = {"clinical": [], "pwv": ["pwv"], "abi": ["abi"], "both": ["pwv", "abi"]}
PAIRS = {"abi_given_pwv": ("pwv", "both"), "pwv_given_abi": ("abi", "both"),
         "pwv_given_clinical": ("clinical", "pwv"), "abi_given_clinical": ("clinical", "abi")}
LONG = re.compile(r"\d{7,}")
DATE = re.compile(r"\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b")
A = O = M = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_modules(paths):
    global A, O, M
    for key, path in paths.items():
        if sha(path)[:16] != EXPECT[key]:
            raise SystemExit("DEPENDENCY ABORT: " + key + " hash mismatch")
    spec = importlib.util.spec_from_file_location("f1_clinical_p10", paths["p10"])
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    O, A = M.load_modules(str(paths["atlas"]), str(paths["outcomes"]))


def rc(n):
    return None if n < MINCELL else int(round(int(n) / 10) * 10)


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return None
        return f"{x:.3e}" if x != 0 and (abs(x) < .001 or abs(x) >= 1e5) else f"{x:.4g}"
    return x


def privacy_check(tx, ids):
    if LONG.search(tx) or DATE.search(tx):
        raise SystemExit("PRIVACY ABORT: long digit run or fine date")
    if any(len(str(p)) >= 4 and str(p) in tx for p in ids):
        raise SystemExit("PRIVACY ABORT: identifier")


def baseline_family(root):
    """Reuse the frozen family-history parser through a baseline-only reader view."""
    original = A.rd
    def rd_base(r, rel, columns=None):
        d = original(r, rel, columns)
        if d is not None and str(rel).startswith("family_history/"):
            if "research_stage" not in d:
                return None  # no timing contract: treat the optional predictor as missing
            return d[d.research_stage == O.BASE].copy()
        return d
    A.rd = rd_base
    try:
        return O.load_fam(root)
    finally:
        A.rd = original


def followup(root, fr):
    """Reuse cohort and event definitions, without follow-up-derived BP calibration."""
    C, dates, prev, info = O.load_contacts(root, fr)
    fam = baseline_family(root)
    H, flow = O.make_cohort(fr, dates, prev, fam)
    b = fr[fr.research_stage == O.BASE].drop_duplicates("participant_id").copy()
    b["base_date"] = b.participant_id.map(dates)
    b = b[b.base_date.notna()].copy()
    b = b.merge(fam, on="participant_id", how="left") if fam is not None else b.assign(fam_htn=np.nan)
    med = M.med_class_rows(root)
    C2, _ = M.add_flags(C, dates, med, ["m_" + k for k in M.MED])
    m0 = med[med.research_stage == O.BASE].groupby("participant_id")[["m_" + k for k in M.MED]].max() if med is not None else None
    for key, frame_key in (("antihtn", "antihtn"), ("statin", "statin")):
        v = b.participant_id.map(m0["m_" + key]).fillna(0) if m0 is not None else pd.Series(0, index=b.index)
        b["b_" + key] = np.maximum(v, b.get(frame_key, pd.Series(0, index=b.index)).fillna(0))
    return {"C": C2, "b": b, "H": set(H.participant_id), "flow": flow, "info": info}


def outcome_sample(S, key):
    """Observed visit-2 attendees with outcome data; first event on/before that person's visit-2 contact."""
    b, C = S["b"], S["C"]
    mask = b.participant_id.isin(S["H"]) if key in ("comp", "objective") else b["b_" + ("antihtn" if key == "ahtn" else "statin")] == 0
    B = b[mask].copy()
    v2 = C[(C.kind == "v2") & C.has_data & (C.t > O.GAP0 / 365.25)].drop_duplicates("participant_id").set_index("participant_id")
    B = B[B.participant_id.isin(v2.index)].copy()
    cc = C[C.participant_id.isin(B.participant_id) & C.has_data & (C.t > O.GAP0 / 365.25)].copy()
    cc = cc[(cc.t <= cc.participant_id.map(v2.t)) & cc.kind.isin(["call1", "v2"])]
    fn = {"comp": O.ev_composite, "objective": O.ev_objective,
          "ahtn": lambda c: (c.med == 1).astype(float),
          "statin": lambda c: (c.m_statin == 1).astype(float)}[key]
    cc["target"] = fn(cc).values
    ys = cc.groupby("participant_id").target.max()
    B["y"] = B.participant_id.map(ys)
    B = B[B.y.notna()].reset_index(drop=True)
    return B, {"at_risk": rc(mask.sum()), "v2_observed": rc(len(B)), "events": rc(B.y.sum()),
               "nonevents": rc((1 - B.y).sum()),
               "followup_years_q05_q50_q95": v2.t.reindex(B.participant_id).quantile([.05, .5, .95]).tolist() if len(B) >= MINCELL else None}


def numeric(d, col):
    return pd.to_numeric(d[col], errors="coerce").replace([np.inf, -np.inf], np.nan) if col in d else pd.Series(np.nan, index=d.index)


def feature_table(B, key, raw):
    cols = ["age", "sex_male", "bmi", "sbp_sit", "dbp_sit", "smk_current", "fam_htn"]
    if key == "statin":
        cols += ["ldl", "hdl", "log_tg", "diabetes_any"]
    d = pd.DataFrame({c: numeric(B, c) for c in cols})
    if key == "statin":
        # The atlas lab join permits +/-180 days. Prediction permits known prior draws only.
        gap = numeric(B, "lab_gap_days")
        d.loc[~gap.le(0), ["ldl", "hdl", "log_tg"]] = np.nan
    d["age_dbp"] = (d.age - 50) * (d.dbp_sit - 80) / 100  # fixed centring, no held-out estimate
    for bed, src in (("pwv", "pwv"), ("abi", "abi_min")):
        src = src + "_raw" if raw else src
        d[bed] = (1 if bed == "pwv" else -1) * numeric(B, src)
    d["y"] = B.y.astype(float)
    # The same participants in all four models; require core clinical variables and both beds.
    keep = d[["age", "sex_male", "bmi", "sbp_sit", "dbp_sit", "pwv", "abi"]].notna().all(axis=1)
    return d[keep].reset_index(drop=True), B.loc[keep, "participant_id"].astype(str).to_numpy(), cols + ["age_dbp"]


def weighted_median(x, w):
    ix = np.argsort(x)
    return float(x[ix[np.searchsorted(np.cumsum(w[ix]), w.sum() / 2)]])


def transform(train, test, cols, weights):
    """Fit all imputation and scaling in training rows with bootstrap multiplicity weights."""
    tr, te = [], []
    for c in cols:
        x, z = train[c].to_numpy(float), test[c].to_numpy(float)
        ok = np.isfinite(x) & (weights > 0)
        med = weighted_median(x[ok], weights[ok]) if ok.any() else 0.0
        missing = ~np.isfinite(x)
        xx, zz = np.where(missing, med, x), np.where(np.isfinite(z), z, med)
        variants = [(xx, zz)]
        if c not in ("pwv", "abi", "age", "sex_male", "bmi", "sbp_sit", "dbp_sit", "age_dbp"):
            variants.append((missing.astype(float), (~np.isfinite(z)).astype(float)))
        for xx, zz in variants:
            mu = np.average(xx, weights=weights)
            sd = np.sqrt(np.average((xx - mu) ** 2, weights=weights))
            if sd > 1e-8:
                tr.append((xx - mu) / sd)
                te.append((zz - mu) / sd)
    return np.column_stack([np.ones(len(train))] + tr), np.column_stack([np.ones(len(test))] + te)


def logistic(X, y, w, penalty=RIDGE):
    """Weighted fixed-ridge likelihood, unpenalised intercept; reject non-convergence."""
    b = np.zeros(X.shape[1])
    mean = np.clip(np.average(y, weights=w), 1e-6, 1 - 1e-6)
    b[0] = np.log(mean / (1 - mean))
    pen = np.ones(len(b)) * penalty
    pen[0] = 0
    for _ in range(80):
        p = expit(np.clip(X @ b, -30, 30))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H + np.eye(len(b)) * 1e-10, X.T @ (w * (y - p)) - pen * b)
        # Backtracking prevents separation/extreme-weight Newton overshoot.
        def loss(bb):
            eta = X @ bb
            return float(np.sum(w * (np.logaddexp(0, eta) - y * eta)) + .5 * np.sum(pen * bb ** 2))
        old, alpha = loss(b), 1.0
        while alpha > 1 / 1024 and loss(b + alpha * step) > old + 1e-8:
            alpha /= 2
        b += alpha * step
        if np.max(np.abs(alpha * step)) < 1e-7:
            return b
    raise ValueError("logistic did not converge")


def cv_predict(d, base, folds, weights=None, gate=False):
    w = np.ones(len(d)) if weights is None else weights
    ys = d.y.to_numpy(float)
    pred, dims, fold_counts = {k: np.empty(len(d)) for k in MODELS}, [], []
    for k in range(FOLDS):
        tr, te = folds != k, folds == k
        nt, ev, non = w[tr].sum(), np.dot(w[tr], ys[tr]), np.dot(w[tr], 1 - ys[tr])
        if ev < MINCELL or non < MINCELL or nt < MIN_N:
            raise ValueError("training cell gate")
        fold_counts.append({"train_n": rc(nt), "train_events": rc(ev), "test_n": rc(w[te].sum()),
                            "test_events": rc(np.dot(w[te], ys[te]))})
        for name, beds in MODELS.items():
            X, Z = transform(d[tr], d[te], base + beds, w[tr])
            if name == "both":
                dims.append(X.shape[1] - 1)
                if gate and min(ev, non) < EPV * (X.shape[1] - 1):
                    raise ValueError("training events per parameter gate")
            b = logistic(X, ys[tr], w[tr])
            pred[name][te] = expit(Z @ b)
    return pred, dims, fold_counts


def weighted_auc(y, p, w):
    ix = np.argsort(p)
    _, first = np.unique(p[ix], return_index=True)
    case = np.add.reduceat((w * y)[ix], first)
    control = np.add.reduceat((w * (1 - y))[ix], first)
    return float(np.sum(case * (np.cumsum(control) - .5 * control)) / (case.sum() * control.sum()))


def metrics(y, p, w):
    out = {"auc": weighted_auc(y, p, w), "brier": float(np.average((y - p) ** 2, weights=w)),
           "eo": float(np.dot(w, p) / np.dot(w, y))}
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    try:
        cal = logistic(np.column_stack([np.ones(len(y)), z]), y, w, penalty=0)
        out.update(cal_intercept=cal[0], cal_slope=cal[1])
    except (ValueError, np.linalg.LinAlgError):
        out.update(cal_intercept=None, cal_slope=None)
    return out


def net_benefit(y, p, w, threshold, gate=True):
    hit = p >= threshold
    cells = [np.dot(w, (y == a) & (hit == b)) for a in (0, 1) for b in (0, 1)]
    if gate and min(cells) < MINCELL:
        return None
    return float((np.dot(w, y * hit) - np.dot(w, (1 - y) * hit) * threshold / (1 - threshold)) / w.sum())


def contrast(y, pred, w):
    mm = {k: metrics(y, p, w) for k, p in pred.items()}
    out = {}
    for name, (old, new) in PAIRS.items():
        out[name] = {"delta_auc": mm[new]["auc"] - mm[old]["auc"],
                     "brier_reduction": mm[old]["brier"] - mm[new]["brier"],
                     "nb": {str(t): None if net_benefit(y, pred[old], w, t) is None or net_benefit(y, pred[new], w, t) is None
                            else net_benefit(y, pred[new], w, t) - net_benefit(y, pred[old], w, t) for t in THRESH}}
    return mm, out


def bootstrap_weights(groups, folds, rng):
    w = np.zeros(len(groups))
    for k in range(FOLDS):
        ids, inverse = np.unique(groups[folds == k], return_inverse=True)
        cnt = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        w[folds == k] = cnt[inverse]
    return w


def estimate_ci(value, samples, want):
    vals = [float(x) for x in samples if x is not None and np.isfinite(x)]
    if len(vals) < max(20, .8 * want):
        return {"value": value, "lo": None, "hi": None, "valid_replicates": rc(len(vals)), "status": "CI unavailable"}
    lo, hi = np.quantile(vals, [.025, .975])
    return {"value": value, "lo": lo, "hi": hi, "valid_replicates": rc(len(vals)), "status": "ok"}


def compare(d, ids, base, relations, nboot, seed, fixed_folds=None):
    y = d.y.to_numpy(float)
    rec = {"n": rc(len(d)), "events": rc(y.sum()), "nonevents": rc((1 - y).sum()), "base_predictors": base}
    if len(d) < MIN_N or min(y.sum(), (1 - y).sum()) < MINCELL:
        return {**rec, "status": "pooled cell gate"}
    groups = np.array([relations.get(p, p) for p in ids])
    rec["family_groups"] = rc(len(np.unique(groups)))
    if len(np.unique(groups)) < FOLDS * MINCELL:
        return {**rec, "status": "family group gate"}
    folds = np.full(len(d), -1, dtype=int) if fixed_folds is None else fixed_folds.copy()
    if fixed_folds is None:
        for k, (_, te) in enumerate(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=seed).split(d, y, groups)):
            folds[te] = k
    if np.any(folds < 0) or any(len(np.unique(folds[groups == g])) != 1 for g in np.unique(groups)):
        raise ValueError("family fold contamination")
    try:
        pred, dims, counts = cv_predict(d, base, folds, gate=True)
    except (ValueError, np.linalg.LinAlgError):
        return {**rec, "status": "training size/event/EPV or fit gate"}
    rec.update(status="ok", max_parameters=max(dims), fold_counts=counts)
    mm, delta = contrast(y, pred, np.ones(len(d)))
    samples = []
    rng = np.random.default_rng(seed + 100)
    for _ in range(nboot):
        w = bootstrap_weights(groups, folds, rng)
        try:
            pp, _, _ = cv_predict(d, base, folds, w)
            samples.append(contrast(y, pp, w))
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            pass
    rec["bootstrap_valid"] = rc(len(samples))
    rec["metrics"] = {m: {v: estimate_ci(x, [s[0][m][v] for s in samples], nboot) for v, x in vals.items()}
                      for m, vals in mm.items()}
    rec["contrasts"] = {}
    for name, vals in delta.items():
        rec["contrasts"][name] = {v: estimate_ci(vals[v], [s[1][name][v] for s in samples], nboot)
                                 for v in ("delta_auc", "brier_reduction")}
        rec["contrasts"][name]["nb"] = {t: None if val is None else estimate_ci(val, [s[1][name]["nb"][t] for s in samples], nboot)
                                        for t, val in vals["nb"].items()}
    return rec


def digest(R):
    def fv(x):
        return "NA" if x is None else str(x)
    def ci(r):
        return "NA" if not r else f"{fv(r['value'])}[{fv(r['lo'])},{fv(r['hi'])}]"
    rows = ["F1 CLINICAL UTILITY P13 " + VERSION,
            "Primary: composite HTN; ABI given PWV, OOF Brier reduction. Objective HTN sensitivity; drug starts exploratory.",
            "Same people/models; 5 family-grouped folds; train-only imputation/scaling; fixed ridge1; baseline measured BP.",
            "Bootstrap: family clusters within fixed folds; all prediction models/preprocessing refit; intervals conditional on folds.",
            "Corrected beds use frozen atlas correction outside folds; raw baseline beds sensitivity. Internal validation only.",
            "Counts round10, <20 hidden; pooled and training gates; every decision-curve cell >=20. No deployment-benefit claim.",
            "Modules " + str(R["modules"]) + "; bootstrap setting " + str(R["bootstrap_setting"]),
            "Family metadata " + R["relation_status"] + "; liver reads0; blocked " + str(R["liver_blocked_types"])]
    for key, res in R["outcomes"].items():
        f = res["flow"]
        rows.append(f"{key} risk {fv(f['at_risk'])} v2 observed {fv(f['v2_observed'])} events {fv(f['events'])}; years q05/50/95 {f['followup_years_q05_q50_q95']}")
        for variant in ("corrected", "raw"):
            r = res[variant]
            rows.append(f"{key}/{variant}: {r['status']} n{fv(r['n'])} ev{fv(r['events'])} groups{fv(r.get('family_groups'))} parameters{fv(r.get('max_parameters'))} boot{fv(r.get('bootstrap_valid'))}")
            if r["status"] != "ok":
                continue
            for model, met in r["metrics"].items():
                rows.append(f"{model}: AUC {ci(met['auc'])} Brier {ci(met['brier'])} slope {ci(met['cal_slope'])} E/O {ci(met['eo'])}")
            for name, con in r["contrasts"].items():
                rows.append(f"{name}: dAUC {ci(con['delta_auc'])} Brier reduction {ci(con['brier_reduction'])}")
                rows.append("NB " + name + ": " + " ".join(t + "=" + ci(v) for t, v in con["nb"].items()))
    rows.append("Read NA as not estimable under the fixed gate; drug initiation is not cardiovascular disease incidence.")
    lines = [line for r in rows for line in textwrap.wrap(r, width=220, break_long_words=False, break_on_hyphens=False)]
    if len(lines) > 120 or max(map(len, lines)) > 220:
        raise ValueError("digest size gate")
    return "\n".join(lines) + "\n"


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", default=str(here / "f1_atlas_v1_5.py"))
    ap.add_argument("--outcomes", default=str(here / "f1_outcomes_v1.py"))
    ap.add_argument("--p10", default=str(here / "f1_atlas_outcomes_v2.py"))
    ap.add_argument("--bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--test-mode", action="store_true", help="local scratch only; permits fewer bootstrap replicates")
    args = ap.parse_args()
    out, root = Path(args.out).resolve(), Path(args.root).resolve()
    if out.exists():
        raise SystemExit("OUTPUT ABORT: destination already exists")
    if out == root or root in out.parents:
        raise SystemExit("OUTPUT ABORT: destination is inside source root")
    if args.bootstrap < (20 if args.test_mode else 200):
        raise SystemExit("BOOTSTRAP ABORT: insufficient setting")
    if args.test_mode and "/results/validation/" not in str(root):
        raise SystemExit("TEST ABORT: scratch root must be under results/validation")
    paths = {k: Path(getattr(args, k)) for k in EXPECT}
    load_modules(paths)
    t0 = time.time()
    # A.LOG and stdout use frozen privacy-aware atlas logging; generic exceptions reveal no values.
    fr = A.build_frame(str(root), None)[0].reset_index(drop=True)
    ids = set(fr.participant_id.astype(str))
    if not all(re.fullmatch(r"\d{10}", p) for p in ids):
        raise SystemExit("INPUT ABORT: identifier contract requires ten digits")
    S = followup(str(root), fr)
    relations = A.relation_components(str(root))
    R = {"version": VERSION, "modules": {k: "_".join(EXPECT[k][i:i+4] for i in range(0, 16, 4)) for k in EXPECT},
         "bootstrap_setting": args.bootstrap, "seed_setting": args.seed, "test_mode": args.test_mode,
         "relation_status": "available" if relations is not None else "unavailable; prediction skipped",
         "outcomes": {}}
    for key in ("comp", "objective", "ahtn", "statin"):
        B, flow = outcome_sample(S, key)
        r = {"flow": flow}
        _, corrected_ids, _ = feature_table(B, key, False)
        _, raw_ids, _ = feature_table(B, key, True)
        common_ids = set(corrected_ids) & set(raw_ids)
        B = B[B.participant_id.isin(common_ids)].reset_index(drop=True)
        for raw in (False, True):
            d, did, base = feature_table(B, key, raw)
            r["raw" if raw else "corrected"] = compare(d, did, base, relations, args.bootstrap, args.seed) if relations is not None else {
                "n": rc(len(d)), "events": rc(d.y.sum()), "status": "family metadata unavailable"}
        R["outcomes"][key] = r
        print("P13 " + key + " complete", flush=True)
    R["liver_blocked_types"] = len(set(M.BLOCKED))  # technical count of table types, not participant cell
    R["runtime_s"] = time.time() - t0
    tx = json.dumps(clean(R), indent=1, allow_nan=False)
    dg = digest(json.loads(tx))
    log = "\n".join(A.LOG) + "\n"
    for payload in (tx, dg, log):
        privacy_check(payload, ids)
    out.mkdir(parents=True, exist_ok=False)
    sums = []
    for fn, payload in (("results.json", tx), ("digest.txt", dg), ("run_log.txt", log)):
        (out / fn).write_text(payload)
        sums.append(hashlib.sha256(payload.encode()).hexdigest() + "  " + fn)
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("P13 FINISHED; aggregate files and digest passed privacy scan", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        raise SystemExit("P13 FAILED: " + type(ex).__name__ + "; no participant values exported") from None
