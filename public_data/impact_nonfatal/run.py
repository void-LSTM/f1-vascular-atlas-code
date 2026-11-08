"""Focused exploratory cause-specific association in the held public cohort."""
import argparse
import importlib.util
import json
import platform
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.stats import norm
from statsmodels.duration.hazard_regression import PHReg

from feasibility import (CLINICAL, EXTRA, HORIZON_DAYS, MARKERS, audit, endpoint, fingerprint,
                         prepare, precision_proxy, read_impact)

LIB_SHA = "00001887121b26d6f1aadead4131a3920b9beed72b2f82ddf443e0a525ea94fe"
MODELS = {"C": CLINICAL, "P": CLINICAL + MARKERS[:1],
          "A": CLINICAL + MARKERS[1:], "PA": CLINICAL + MARKERS}
SCALES = {"abi_adverse": 0.10, "bapwv_max_cms": 100.0}
NUMERICAL_ERRORS = (np.linalg.LinAlgError, FloatingPointError, OverflowError, RuntimeWarning)


def load_lib(path):
    if fingerprint(path) != LIB_SHA:
        raise ValueError("F1 survival library hash mismatch")
    spec = importlib.util.spec_from_file_location("f1_original_lib_survival", path)
    mod = importlib.util.module_from_spec(spec)
    # Dataclass internals resolve the module by name.
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def fit(s, terms, lib, labels=None):
    t = s.time_days.to_numpy(float)
    y = s.status.eq(1).to_numpy(int) if labels is None else labels
    context = {"n":len(s), "events":int(y.sum()), "terms":terms}
    raw = s[terms].to_numpy(float)
    sd = raw.std(axis=0)
    if not np.isfinite(sd).all() or np.any(sd <= 0):
        return context | {"status": "constant_or_nonfinite_term"}
    x = (raw - raw.mean(axis=0)) / sd
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            if np.linalg.matrix_rank(x) != len(terms):
                return context | {"status": "rank_deficient"}
            f = lib.coxph(t, y, x, names=terms, robust=True, ties="efron")
    except NUMERICAL_ERRORS as exc:
        return context | {"status":"fit_unavailable", "warnings":[str(exc)]}
    if (not f.converged or f.warnings or not np.isfinite(f.coef).all()
            or not np.isfinite(f.var).all() or not np.isfinite(f.var_model).all()
            or np.any(np.diag(f.var) <= 0)):
        return context | {"status":"fit_unavailable","warnings":f.warnings,"iterations":f.iterations}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ref = PHReg(t, x, status=y, ties="efron").fit(disp=0)
        ref_var = ref.cov_params()
    except NUMERICAL_ERRORS as exc:
        return context | {"status":"independent_numerical_crosscheck_failed", "warnings":[str(exc)]}
    if caught or not np.isfinite(ref.params).all() or not np.isfinite(ref_var).all():
        return context | {"status":"independent_numerical_crosscheck_failed",
                          "warnings":[str(item.message) for item in caught]}
    beta_error = float(np.max(np.abs(f.coef - ref.params)))
    var_error = float(np.max(np.abs(f.var_model - ref_var)))
    if beta_error > 1e-5 or var_error > 1e-5:
        return context | {"status":"independent_numerical_crosscheck_failed",
                "max_beta_error":beta_error,"max_model_covariance_error":var_error}
    diagnostic_warnings = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ph = lib.cox_zph(f).set_index("term")["p"].dropna().to_dict()
    except NUMERICAL_ERRORS as exc:
        ph = {}
        diagnostic_warnings.append(str(exc))
    effects = {}
    for marker, increment in SCALES.items():
        if marker not in terms:
            continue
        i = terms.index(marker)
        b, se = f.coef[i], np.sqrt(f.var[i,i])
        k = increment / sd[i]
        effects[marker] = {"increment":increment,"observed_sd":float(sd[i]),
                           "hr":float(np.exp(k*b)), "ci95":np.exp(k*(b + np.array([-1,1])*norm.ppf(.975)*se)).tolist(),
                           "p_nominal":float(2*norm.sf(abs(b/se))),
                           "hr_per_sd":float(np.exp(b)),
                           "ci95_per_sd":np.exp(b + np.array([-1,1])*norm.ppf(.975)*se).tolist(),
                           "robust_se_log_hr_per_sd":float(se),
                           "model_se_log_hr_per_sd":float(np.sqrt(f.var_model[i,i])),
                           "ph_diagnostic_p":ph.get(marker)}
    return context | {"status":"ok","effects":effects,
            "ph_global_diagnostic_p":ph.get("GLOBAL"),"ph_diagnostic_warnings":diagnostic_warnings,
            "iterations":f.iterations,
            "max_beta_error":beta_error,"max_model_covariance_error":var_error}


def passes(n, widths):
    return n >= 200 and widths is not None and max(widths.values()) <= np.log(1.5)


def event_first_labels(s):
    tie = s.same_day_death & s.status.eq(2)
    return (s.status.eq(1) | tie).to_numpy(int)


def calculate(raw, lib):
    f = audit(raw)
    frame, _, keep = prepare(raw)
    s = frame[keep].copy()
    if not passes(len(s), f["expected_log_hr_halfwidth"]):
        return {"status":"infeasible","feasibility":f}
    rows = {k:fit(s, terms, lib) for k,terms in MODELS.items()}
    sensitivity = fit(s, MODELS["PA"], lib, event_first_labels(s))
    if sensitivity["status"] == "ok":
        sensitivity["effects"] = {k:v for k,v in sensitivity["effects"].items() if k=="abi_adverse"}
    clinical_ext = CLINICAL + EXTRA
    s_ext = s.dropna(subset=EXTRA)
    widths_ext = precision_proxy(s_ext, clinical_ext)
    expanded = {"n":len(s_ext),"events":int(s_ext.status.eq(1).sum()),
                "expected_log_hr_halfwidth":widths_ext,"status":"infeasible"}
    if passes(len(s_ext), widths_ext):
        expanded |= fit(s_ext, clinical_ext + MARKERS, lib)
    aj = lib.cuminc(s.time_days.to_numpy(), s.status.to_numpy(), [HORIZON_DAYS], cause=1)
    return {"status":"completed" if all(r["status"]=="ok" for r in rows.values()) else "partial",
            "scope":"Public IMPACT-ABI. Exploratory cause-specific association, before documented death.",
            "feasibility":f,"models":rows,"event_first_death_tie_sensitivity":sensitivity,
            "hypertension_smoking_sensitivity":expanded,
            "aj_unadjusted_five_year":aj.to_dict(orient="records"),
            "prediction_status":"not_run_insufficient_5y_target_events_existing_gate_100"}


def self_check(lib):
    class Tests(unittest.TestCase):
        def test_event_order(self):
            d = pd.DataFrame(dict(mi=[1,1,1,0,0,1,0,1], stroke=[0,0,0,1,0,0,0,1],
                                  death=[1,1,1,0,1,0,0,0],
                                  mi_fu_days=[8,8,9,5,4,0,10,9],
                                  stroke_fu_days=[12,12,12,7,4,10,10,4],
                                  death_fu_days=[9,8,8,15,3,10,10,12]))
            r = endpoint(d)
            np.testing.assert_array_equal(r.status, [1,2,2,0,2,0,0,1])
            np.testing.assert_array_equal(r.time_days, [8,8,8,5,3,0,10,4])
            self.assertFalse(r.valid.iloc[5])
            self.assertTrue(r.apparent_event_beyond_complete_followup.iloc[3])
            self.assertTrue(r.same_day_death.iloc[1])
            self.assertTrue(r.event_after_death.iloc[2])
            np.testing.assert_array_equal(event_first_labels(r), [1,1,0,0,0,0,0,1])

        def test_competing_incidence(self):
            r = lib.cuminc(np.array([1.,2.,3.,4.]),np.array([2,1,0,0]),[4.],cause=1)
            self.assertAlmostEqual(float(r["cuminc"].iloc[0]),.25)

        def test_fit_known_signal_and_crosscheck(self):
            rng = np.random.default_rng(20261106)
            n = 1600
            x, z = rng.normal(size=(2,n))
            t = np.round(100*rng.exponential(scale=np.exp(-.7*x+.3*z)))+1
            c = rng.uniform(30,200,n)
            s = pd.DataFrame(dict(time_days=np.minimum(t,c),status=(t<=c).astype(int),
                                  abi_adverse=x,bapwv_max_cms=z))
            r = fit(s, MARKERS, lib)
            self.assertEqual(r["status"],"ok")
            self.assertGreater(r["effects"]["abi_adverse"]["hr_per_sd"],1.7)
            self.assertLess(r["effects"]["abi_adverse"]["hr_per_sd"],2.5)
    ran = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    if not ran.wasSuccessful():
        raise SystemExit("synthetic validation failed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
    lib = load_lib(a.library)
    if a.synthetic:
        self_check(lib)
        return
    if a.source is None or a.out is None:
        p.error("--source and a new --out folder are required")
    a.out.mkdir(exist_ok=False)
    result = calculate(read_impact(a.source), lib)
    with (a.out/"results.json").open("x") as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write("\n")
    hashes = {x.name:fingerprint(x) for x in (Path(__file__),Path(__file__).with_name("feasibility.py"),
                                             a.library)}
    with (a.out/"provenance.json").open("x") as f:
        json.dump({"sources":hashes,"runtime":{"python":platform.python_version(),"numpy":np.__version__,
                   "pandas":pd.__version__,"scipy":scipy.__version__,"statsmodels":statsmodels.__version__},
                   "results_sha256":fingerprint(a.out/"results.json")},f,indent=2); f.write("\n")


if __name__ == "__main__":
    main()
