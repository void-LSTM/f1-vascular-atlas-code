#!/usr/bin/env python3
"""F1 follow-up 2 (package P12 'f1_followup2_v1'; HPP sandbox; writes aggregates only).

Three follow-ups of HPP run 6
(F1_RUN6_NOTES.md), built on the code of P10 (f1_atlas_outcomes_v2.py, e8cc37eaa90adef6) and P11
(f1_outcome_liability_v1.py, f027a316a269ccfa), which are imported by path with P1 (f1_outcomes_v1.py, 9bee14ddca223c57),
PRS v1.1 (f1_prs_v1_1.py, 0fb97f1301077fd6) and atlas v1.5 (f1_atlas_v1_5.py, 62344144b83db3f2); none is modified.

T  bed-specific treatment initiation, formal test.  P10 outcome builders and P1 hazard (discrete-time cloglog over the
   follow-up contacts, contact intercepts, log(interval) offset, outcome-specific stabilised IPW, robust SE clustered by
   participant; covariates B = age, sex, BMI, calibrated usual SBP/DBP).  Outcomes: antihypertensive start (ahtn),
   statin start (statin), objective incident hypertension (htn_obj) and the composite incident hypertension of P1
   (comp; primary hypertension outcome).
   (a) per outcome, pairwise common samples of the four core beds (pwv/abi, pwv/imt, imt/abi, pwv/ret, imt/ret, ret/abi):
       separate model-B fits per bed, stacked influence functions, HR ratio per pair (P10 paired_block).
   (b) bed x outcome interaction in the same people: participants at risk for both outcomes (both cohorts) with both
       beds and covariates; beds standardised once in that sample; four fits (2 beds x 2 outcomes), joint covariance of
       the four log-HRs from stacked influence functions (the same participants contribute to both outcomes);
       ratio of ratios RoR = [HR_x/HR_y for outcome 1] / [HR_x/HR_y for outcome 2], Wald p.  Primary: ahtn vs statin;
       secondary: comp vs statin.  Omnibus: (pwv, imt, abi) 2 df and (pwv, imt, ret, abi) 3 df interaction.
G  better-covered CAD and T2D scores for P11.  The P11 weights (pgs_weights_outcome_v1, already in the sandbox) and the
   replacement scores (pgs_weights_outcome_v2: CAD2 = PGS004696, CADm = PGS003725 multi-trait sensitivity, T2D2 =
   PGS002771) are scored together by PRS v1.1's own functions; P11's analysis functions are reused unchanged.
   Coverage; positive controls (CAD -> LDL, non-HDL, ApoB; T2D -> HbA1c, glucose); P11 D1i/D1ii (matrix test with 1000
   permutations), D2 (per-score joint test), D3 (BP part) with CAD2/T2D2 replacing CAD/T2D; the old-score matrix as a
   reproduction gate of the P11 digest; CAD -> IMT old vs new side by side.
C  Fig. 2g reference centiles (q05/q25/q50/q75/q95 per 5-year age band x sex, cells >= 20) for PWV, IMT, retinal artery
   width and ABI: read from P10's f1_atlas_outcomes_out_v2/results.json (aggregate fields) and recomputed with P10's own
   reference function on the same reference subset (stored values are used when the file has the run-6 SHA-256).
Privacy: aggregates only; counts rounded to 10, cells < 20 suppressed (None); hazard models n >= 200 and >= 20 events;
every regression n >= 200 (joint and matrix tests n >= 300, within-sib >= 200 pairs); quantiles 5-95 % only; no
participant ids; no dates; exception text scrubbed; every output scanned for 7+ digit runs, participant ids and ISO
dates before anything is written; refuses to overwrite --out.  Liver tables are never read.
Run: python f1_followup2_v1.py --root <HPP_DATA_ROOT> --out f1_followup2_out_v1
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy import stats

VERSION = "f1_followup2_v1.0 (2026Q4 build)"
HERE = os.path.dirname(os.path.abspath(__file__))
EXPECT = {"atlas": "62344144b83db3f2", "prs": "0fb97f1301077fd6", "p10": "e8cc37eaa90adef6", "p1": "9bee14ddca223c57",
          "p11": "f027a316a269ccfa", "w1": "2156bc74d20682e8", "w2": "9a5521ae6bb27dc3", "p10res": "477378032b32e2cd"}
L = None    # P11 module
P = None    # PRS v1.1 module (L.P)
A = None    # atlas module (L.A)
M10 = None  # P10 module (L.M10)
O = None    # P1 module (M10.O)
BASE = "00_00_visit"
SH = {"pwv": "pwv", "imt": "imt", "art_w": "ret", "abi_min": "abi"}
CORE4 = ["pwv", "imt", "art_w", "abi_min"]
PAIRS6 = [("pwv", "abi_min"), ("art_w", "abi_min"), ("imt", "abi_min"), ("pwv", "imt"), ("pwv", "art_w"),
          ("imt", "art_w")]
PRIMARY_PAIRS = 3  # pwv/abi, ret/abi, imt/abi
TOUT = {"ahtn": "antihypertensive start (C02/C03/C07/C08/C09; untreated at baseline)",
        "statin": "statin start (C10AA; none at baseline)",
        "comp": "composite incident hypertension (P1 ev_composite, P1 at-risk cohort; primary HTN outcome)",
        "htn_obj": "objective incident hypertension (P1 ev_objective, P1 at-risk cohort)"}
CONTRASTS = [("ahtn", "statin"), ("comp", "statin")]  # first = primary (pre-stated)
OMNI = [("pwv", "imt", "abi_min"), ("pwv", "imt", "art_w", "abi_min")]
# reproduction gate for T: P10 digest (hpp_run_20261006 AO:23, :35, :46, :53) and P1 digest OUT L31 (composite, B)
REPRO_T = [("ahtn", "pwv", "1.55[1.2,2]", 4870, 70, "AO:35"), ("statin", "abi_min", "1.28[1.08,1.52]", 4760, 130, "AO:46"),
           ("statin", "art_w", "1.4[1.04,1.9]", 2250, 40, "AO:46"), ("htn_obj", "pwv", "1.04[0.813,1.34]", 4020, 120, "AO:23"),
           ("comp", "pwv", "1.29[1.03,1.61]", 4020, 140, "OUT:31")]
REPRO_TP = ("statin", "imt/ret", "0.519[0.322,0.836]", 2080, 40, "AO:53")
# G: scores
OLD = ["CAD", "IS", "PCS", "PAD", "HF", "AF", "CKD", "HTN", "AAA", "T2D"]
NEW = ["CAD2", "CADm", "T2D2"]
REPL = {"CAD": "CAD2", "T2D": "T2D2"}
PRIM_OLD = ["CAD", "IS", "PAD", "HF", "AF", "CKD", "HTN", "AAA", "T2D"]  # P11 PRIMARY
PRIM_NEW = [REPL.get(s, s) for s in PRIM_OLD]
LIPID = [("ldl", 1), ("nhdl_bt", 1), ("apob_nmr", 1), ("nhdl_nmr", 1)]
GLYC = [("hba1c", 1), ("glucose", 1)]
CONTROLS = {"CAD": LIPID, "CAD2": LIPID, "CADm": LIPID, "T2D": GLYC, "T2D2": GLYC}
ACCEPT = {"CAD2": "ldl", "T2D2": "hba1c"}  # pre-stated acceptance control (p < 0.05, expected sign)
# reproduction gate for G: P11 digest (hpp_run_20261006 OL:8-11 D1i/D1ii, :28 CAD i, :88 CAD cov, :98 controls)
REPRO_G = {"d1i": ("42.2", 24, "0.012", "0.015"), "d1ii": ("28", 24, "0.26", "0.27"), "n": 3180,
           "cad_imt_i": "0.0575[0.0302,0.0849]", "cad_ldl": "0.0211[-0.0124,0.0547]",
           "t2d_hba1c": "0.0293[-0.00424,0.0629]", "cad_cov": ("1,745,179", "564,986")}
# C: beds and bands
CBEDS = ["pwv", "imt", "art_w", "abi_min"]
BANDS = ["40-44", "45-49", "50-54", "55-59", "60-64", "65-69"]
QK = ["q05", "q25", "q50", "q75", "q95"]
MIN_N = 200
LONG = re.compile(r"\d{7,}")
DATE_RX = re.compile(r"\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?([+-]\d{2}:?\d{2}|Z)?")


# ============================================================================ helpers (guard as P9/P10/P11)

def hfmt(h):
    """First 16 hex characters of a SHA-256 in groups of 6/6/4 (an unbroken hash can hold 7+ digit runs)."""
    h = str(h)[:16]
    return ".".join(h[i:i + 6] for i in range(0, len(h), 6))


def sha_file(p, n=16):
    hs = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            hs.update(blk)
    return hs.hexdigest()[:n] if n else hs.hexdigest()


def log(m):
    A.log(m)


def rc(n):
    return A.rc(n)


def v3(x):
    return A.v3(x)


def nrc(mask):
    return rc(int(np.asarray(mask, bool).sum()))


def scrub(s):
    s = DATE_RX.sub("<date>", str(s))
    s = re.sub(r"(?<!\d)\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?", "<time>", s)
    return re.sub(r"\d{4,}", "#", s)


def err(ex):
    tb = ([f for f in traceback.extract_tb(ex.__traceback__) if f.filename == __file__] or
          traceback.extract_tb(ex.__traceback__) or [None])[-1]
    return f"{type(ex).__name__}: {scrub(str(ex))[:160]} (line {tb.lineno if tb else '?'})"


def guard(out, key, fn):
    t1 = time.time()
    try:
        out[key] = fn()
    except Exception as ex:
        out[key] = {"error": err(ex)}
        log(f"{key} FAILED: {out[key]['error']}")
    out.setdefault("timings", {})[key] = time.time() - t1
    return out[key]


def privacy_check(text, pids):
    m = LONG.search(text)
    if m:
        raise SystemExit(f"PRIVACY ABORT: digit run '{m.group(0)[:3]}...' in output")
    if DATE_RX.search(text):
        raise SystemExit("PRIVACY ABORT: a date in output")
    for p in pids:  # ids of 7+ digits are caught above (real HPP ids have 10 digits)
        if not (p.isdigit() and len(p) >= 7) and len(p) >= 4 and p in text:
            raise SystemExit("PRIVACY ABORT: participant id in output")


def clean(o):
    if isinstance(o, np.ndarray):
        o = o.tolist()
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating, int, np.integer, bool, np.bool_)):
        return A.fnum(o)
    return o


def vq(s):
    """Module version strings carry build dates; outputs show the calendar quarter only."""
    return DATE_RX.sub(lambda m: f"{m.group(0)[:4]}Q{(int(m.group(0)[5:7]) - 1) // 3 + 1}", str(s))


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def dfmt(v, lo, hi):
    """Digest form of an estimate with CI, as printed after results.json cleaning (4 s.f.) and v3 (3 s.f.)."""
    return f"{v3(A.fnum(v))}[{v3(A.fnum(lo))},{v3(A.fnum(hi))}]"


def ld(b, se):
    """exp(b) with CI and p from a log-scale estimate."""
    if se is None or not np.isfinite(se):
        return None
    return {"ratio_hr": float(np.exp(b)), "lo": float(np.exp(b - 1.96 * se)), "hi": float(np.exp(b + 1.96 * se)),
            "log": float(b), "se": float(se), "p": float(2 * stats.norm.sf(abs(b / se))) if se > 0 else None}


# ============================================================================ loading

def load_modules(a):
    """P11 by path; P11 loads PRS v1.1 (which loads the atlas) and, at the expected hashes only, P10 and P1, all sharing
    one atlas instance whose reader refuses liver tables (P11 load_modules / load_p10, unchanged)."""
    global L, P, A, M10, O
    spec = importlib.util.spec_from_file_location("f1_outcome_liability_v1", a.p11)
    L = importlib.util.module_from_spec(spec)
    sys.modules["f1_outcome_liability_v1"] = L
    spec.loader.exec_module(L)
    L.load_modules(os.path.abspath(a.atlas), os.path.abspath(a.prs))
    P, A = L.P, L.A
    L.ATLAS_FILE = os.path.abspath(a.atlas)
    try:
        st = L.load_p10(os.path.abspath(a.p10), os.path.abspath(a.outcomes))
    except Exception as ex:
        st = f"load error {err(ex)[:60]}"
    if st == "ok":
        M10, O = L.M10, L.M10.O
    return st


# ============================================================================ shared follow-up build (T and C)

def build_followup(root, fr, p1_seed):
    """P10 run() steps for the outcome data (P1 contacts and usual BP, medication/condition flags, baseline table)."""
    C, base_dates, prev_htn, cinfo = O.load_contacts(root, fr)
    fam = O.load_fam(root)
    ctx = O.prepare(fr, C, base_dates, prev_htn, fam, cinfo, p1_seed)
    thr = M10.lab_thresholds(fr)
    mrows, crows = M10.med_class_rows(root), M10.cond_group_rows(root)
    ids = set(C["participant_id"].astype(str))
    for r_ in (mrows, crows):
        if r_ is not None:
            ids |= set(r_["participant_id"].astype(str))
    C2, mi = M10.add_flags(ctx["C"], base_dates, mrows, ["m_" + k for k in M10.MED])
    C2, ci = M10.add_flags(C2, base_dates, crows, ["i_" + k for k in M10.ICD])
    med0 = {}
    if mrows is not None:
        m0 = mrows[mrows["research_stage"] == BASE].groupby("participant_id")[["m_" + k for k in M10.MED]].max()
        med0 = {k: m0["m_" + k] for k in M10.MED}
    prev = {k: M10.prevalent(crows, base_dates, "i_" + k) for k in M10.ICD}
    extra = {"med0": med0, "prev": prev, "cur": M10.curated_flags(root)}
    b, binfo = M10.baseline_table(fr, ctx, extra)
    S = {"ctx": ctx, "C2": C2, "b": b, "extra": extra, "thr": thr,
         "info": {"p1_cohort": rc(len(ctx["B"])), "baseline": rc(len(b)),
                  "undefined_rows_mapped": {k: v for k, v in {**mi, **ci}.items() if v}}}
    return S, ids


# ============================================================================ T: treatment initiation

def t_outcome(oc, S):
    b, ctx, extra, thr, C2 = S["b"], S["ctx"], S["extra"], S["thr"], S["C2"]
    key = "htn_obj" if oc in ("comp", "htn_obj") else oc
    evfn = O.ev_composite if oc == "comp" else M10.ev_fn(oc, thr)
    B = b[M10.cohort_mask(key, b, ctx, extra, thr)].reset_index(drop=True)
    PP = O.person_period(B, C2, evfn)
    W, wi = O.ipw(B, C2, PP)
    Pw = O.attach_w(PP, W)
    rec = {"label": TOUT[oc], "cohort": rc(len(B)), "followed": rc(Pw["participant_id"].nunique()),
           "events": nrc(Pw["ev"] == 1), "events_by_kind": {k: nrc((Pw["kind"] == k) & (Pw["ev"] == 1)) for k in O.KINDS},
           "ipw_auc": {k: g(wi, k, "auc_attendance") for k in O.KINDS}}
    rec["beds_B"] = {bed: M10.hz_fit(Pw, B, "o_" + bed, O.COV_B)[0] for bed in CORE4}
    rec["pairs_B"] = {f"{SH[x]}/{SH[y]}": M10.paired_block(Pw, B, [x, y], O.COV_B) for x, y in PAIRS6}
    if oc == "comp":
        rec["S3_B"] = M10.paired_block(Pw, B, ["pwv", "imt", "abi_min"], O.COV_B)
    return rec, (B, Pw)


def joint_sample(S1, S2, beds, covs):
    """Participants at risk for both outcomes (in both cohorts) with every bed and covariate; beds standardised once in
    this sample (person level), so both outcomes use the same units."""
    B1, B2 = S1[0], S2[0]
    ids = set(B1["participant_id"]) & set(B2["participant_id"])
    need = ["o_" + x for x in beds] + list(covs)
    J = B1[B1["participant_id"].isin(ids)].replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
    for x in beds:
        v = J["o_" + x].astype(float)
        J["z_" + x] = (v - v.mean()) / v.std()
    return J.reset_index(drop=True)


def stacked_fits(J, outs, beds, covs):
    """One fit per (outcome, bed) on J, std=False (beds already standardised in J).  Returns results and auxes or the
    reason a fit is missing."""
    res, aux = {}, []
    for oc, (_, Pw) in outs:
        Pj = Pw[Pw["participant_id"].isin(J["participant_id"])]
        for x in beds:
            r, a_ = O.hazard(Pj, J, ["z_" + x], covs, std=False)
            res[f"{oc}:{x}"] = {"n": r.get("n"), "events": r.get("events"), "status": r.get("status"),
                                "hr": g(r, "hr", "z_" + x)}
            if a_ is None:
                return res, None
            aux.append(a_)
    return res, aux


def ror_block(oc1, oc2, x, y, S1, S2, covs):
    """Ratio of ratios [HR_x/HR_y | oc1] / [HR_x/HR_y | oc2] in participants at risk for both outcomes; joint covariance
    of the four log-HRs from stacked influence functions (P1 stack)."""
    J = joint_sample(S1, S2, [x, y], covs)
    out = {"n_common": rc(len(J)), "beds": [x, y]}
    if len(J) < MIN_N:
        out["status"] = "too few"
        return out
    res, aux = stacked_fits(J, [(oc1, S1), (oc2, S2)], [x, y], covs)
    out["fits"] = res
    out["n"] = {oc1: g(res, f"{oc1}:{x}", "n"), oc2: g(res, f"{oc2}:{x}", "n")}
    out["events"] = {oc1: g(res, f"{oc1}:{x}", "events"), oc2: g(res, f"{oc2}:{x}", "events")}
    if aux is None:
        out["status"] = "too few"
        return out
    b, V = O.stack(aux)
    out["ratio"] = {oc1: O.diff_line(b, V, 0, 1), oc2: O.diff_line(b, V, 2, 3)}
    Lr = np.array([1.0, -1.0, -1.0, 1.0])
    est = float(Lr @ b)
    se = float(np.sqrt(max(Lr @ V @ Lr, 0)))
    out["ror"] = ld(est, se)
    out["ror"]["chi2"] = float((est / se) ** 2) if se > 0 else None
    out["status"] = "ok"
    return out


def omnibus_block(oc1, oc2, beds, S1, S2, covs):
    """Bed x outcome interaction over several beds (reference = last bed): (k - 1)-df Wald test that the log-HR profile
    over beds differs between the two outcomes, same people, stacked influence functions."""
    J = joint_sample(S1, S2, list(beds), covs)
    out = {"n_common": rc(len(J)), "beds": list(beds)}
    if len(J) < MIN_N:
        out["status"] = "too few"
        return out
    res, aux = stacked_fits(J, [(oc1, S1), (oc2, S2)], list(beds), covs)
    out["events"] = {oc1: g(res, f"{oc1}:{beds[0]}", "events"), oc2: g(res, f"{oc2}:{beds[0]}", "events")}
    if aux is None:
        out["status"] = "too few"
        return out
    b, V = O.stack(aux)
    k = len(beds)
    Lm = np.zeros((k - 1, 2 * k))
    for i in range(k - 1):
        Lm[i, i], Lm[i, k - 1], Lm[i, k + i], Lm[i, 2 * k - 1] = 1, -1, -1, 1
    out["wald"] = O.wald(b, V, Lm)
    out["status"] = "ok"
    return out


def t_repro(T):
    """Per-outcome bed HRs (model B, full cohorts) must equal P10/P1 digest values at the printed precision."""
    out = {}
    for oc, bed, ref, n, ev, src in REPRO_T:
        r = g(T, "outcomes", oc, "beds_B", bed) or {}
        h = g(r, "hr", "o_" + bed) or {}
        got = dfmt(h.get("hr"), h.get("lo"), h.get("hi")) if h else "NA"
        out[f"{oc}:{SH[bed]}"] = {"got": got, "ref": ref, "src": src,
                                  "ok": bool(got == ref and r.get("n") == n and r.get("events") == ev)}
    oc, pr, ref, n, ev, src = REPRO_TP
    x = g(T, "outcomes", oc, "pairs_B", pr) or {}
    rr = g(x, "ratio", pr) or {}
    got = dfmt(rr.get("ratio_hr"), rr.get("lo"), rr.get("hi")) if rr else "NA"
    out[f"{oc}:{pr}"] = {"got": got, "ref": ref, "src": src, "ok": bool(got == ref and x.get("n") == n and x.get("events") == ev)}
    out["all_ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def section_t(S):
    T = {"outcomes": {}}
    data = {}
    for oc in ("ahtn", "statin", "comp", "htn_obj"):
        t1 = time.time()
        try:
            T["outcomes"][oc], data[oc] = t_outcome(oc, S)
        except Exception as ex:
            T["outcomes"][oc] = {"error": err(ex)}
            log(f"T {oc} FAILED: {T['outcomes'][oc]['error']}")
        log(f"T {oc} done in {time.time() - t1:.1f}s")
    T["ror"], T["ror_A"], T["omnibus"] = {}, {}, {}
    for oc1, oc2 in CONTRASTS:
        key = f"{oc1}_vs_{oc2}"
        if oc1 not in data or oc2 not in data:
            T["ror"][key] = {"status": "outcome missing"}
            continue
        T["ror"][key] = {}
        for x, y in PAIRS6:
            guard(T["ror"][key], f"{SH[x]}/{SH[y]}", lambda x=x, y=y: ror_block(oc1, oc2, x, y, data[oc1], data[oc2], O.COV_B))
        T["ror"][key].pop("timings", None)
        T["omnibus"][key] = {}
        for beds in OMNI:
            guard(T["omnibus"][key], ",".join(SH[x] for x in beds),
                  lambda beds=beds: omnibus_block(oc1, oc2, beds, data[oc1], data[oc2], O.COV_B))
        T["omnibus"][key].pop("timings", None)
    oc1, oc2 = CONTRASTS[0]
    if oc1 in data and oc2 in data:  # model A (age, sex) sensitivity of the primary contrast, primary pairs
        key = f"{oc1}_vs_{oc2}"
        T["ror_A"][key] = {}
        for x, y in PAIRS6[:PRIMARY_PAIRS]:
            guard(T["ror_A"][key], f"{SH[x]}/{SH[y]}", lambda x=x, y=y: ror_block(oc1, oc2, x, y, data[oc1], data[oc2], M10.COV_A))
        T["ror_A"][key].pop("timings", None)
    T["repro"] = t_repro(T)
    return T


# ============================================================================ G: replacement CAD and T2D scores

def add_labs(root, d):
    """ApoB and non-HDL cholesterol for the CAD positive controls, attached without reordering d: Nightingale NMR
    (per-visit mean at the stage of d's row) and blood-test non-HDL (nearest draw within 180 days of d's visit; P11
    tc_hdl rule: total - HDL, total from non-HDL + HDL when missing)."""
    info = {}
    nm = A.rd(root, "nightingale_metabolomics/nightingale_metabolomics.parquet", columns=["apob", "non_hdl_c"])
    d["apob_nmr"], d["nhdl_nmr"] = np.nan, np.nan
    if nm is not None:
        cols = [c for c in ("apob", "non_hdl_c") if c in nm.columns]
        nm = A.num(nm, cols)
        pv_ = A.per_visit(nm, cols).set_index(["participant_id", "research_stage"])
        idx = pd.MultiIndex.from_arrays([d["participant_id"].astype(str).values, d["research_stage"].astype(str).values])
        for src, dst in (("apob", "apob_nmr"), ("non_hdl_c", "nhdl_nmr")):
            if src in pv_.columns:
                d[dst] = pv_[src].reindex(idx).values
        info["nmr"] = {k: nrc(d[k].notna()) for k in ("apob_nmr", "nhdl_nmr")}
    else:
        info["nmr"] = "missing"
    d["nhdl_bt"] = np.nan
    if "visit_date" in d.columns:
        lip, li = L.tc_hdl(root, d[["participant_id", "visit_date"]])
        m = lip.dropna(subset=["tc", "hdl_b"]).drop_duplicates("participant_id").set_index("participant_id")
        d["nhdl_bt"] = d["participant_id"].map(m["tc"] - m["hdl_b"]).astype(float).values
        info["blood_tests"] = {**li, "nhdl_bt": nrc(d["nhdl_bt"].notna())}
    return d, info


def g_scores(root, w1dir, w2dir, block):
    w1, i1 = L.load_weights(w1dir)
    w2, i2 = L.load_weights(w2dir)
    weights = dict(w1)
    for k, v in w2.items():
        if k not in weights:
            weights[k] = v
    base = os.path.join(root, "human_genetics", "genotypes", "plink_postqc")
    fam = P.read_fam_ids(os.path.join(base, "chr_1"))
    cov, flow = P.load_samples(root, fam)
    t0 = time.time()
    S_, coverage = P.score_all(root, weights, len(fam), block)
    return weights, {"v1": i1, "v2": i2}, fam, cov, flow, S_, coverage, time.time() - t0


def cad_imt_block(R, du, base_cov):
    """CAD -> IMT old vs new side by side; same-sample difference new - old (model i, paired influence functions)."""
    out = {s: {"i": g(R, "beds_i", s, "imt"), "ii": g(R, "beds_ii", s, "imt"), "bp_part": g(R, "bp_part", s, "imt")}
           for s in ("CAD", "CAD2", "CADm") if s in R.get("beds_i", {})}
    for new in ("CAD2", "CADm"):
        if not all(c in du.columns for c in ("CAD", new)):
            continue
        dd = du[["o_imt", "CAD", new] + base_cov].replace([np.inf, -np.inf], np.nan).dropna()
        if len(dd) < MIN_N:
            out[f"{new}_minus_CAD_i"] = {"n": rc(len(dd)), "status": "too few"}
            continue
        y = L.zcol(dd["o_imt"])
        C0 = [dd[c].astype(float).values for c in base_cov]
        b1, if1, n, k = L.ols_if(y, np.column_stack([np.ones(len(dd)), L.zcol(dd["CAD"])] + C0))
        b2, if2, _, _ = L.ols_if(y, np.column_stack([np.ones(len(dd)), L.zcol(dd[new])] + C0))
        se = float(np.sqrt(((if2 - if1) ** 2).sum() * n / max(n - k, 1)))
        out[f"{new}_minus_CAD_i"] = L.coef(b2 - b1, se, n)
    return out


def g_repro(R):
    """Old-score lines must reproduce the P11 digest (same weights, samples and seeds)."""
    out = {}
    for key, ref in (("old_i", REPRO_G["d1i"]), ("old_ii", REPRO_G["d1ii"])):
        t2 = g(R, "matrix", key, "T2_score_x_bed_interaction") or {}
        got = (v3(A.fnum(t2.get("chi2"))), t2.get("df"), L.pstr(A.fnum(t2.get("p"))), L.pstr(A.fnum(t2.get("perm_p"))))
        out[key] = {"got": list(map(str, got)), "ref": list(map(str, ref)),
                    "ok": bool(t2) and list(map(str, got)) == list(map(str, ref)) and g(R, "matrix", key, "n") == REPRO_G["n"]}
    for key, s, b_, ref in (("cad_imt_i", "CAD", "imt", REPRO_G["cad_imt_i"]),):
        c = g(R, "beds_i", s, b_) or {}
        got = dfmt(c.get("b"), c.get("lo"), c.get("hi")) if c else "NA"
        out[key] = {"got": got, "ref": ref, "ok": got == ref}
    for key, s, y, ref in (("cad_ldl", "CAD", "ldl", REPRO_G["cad_ldl"]), ("t2d_hba1c", "T2D", "hba1c", REPRO_G["t2d_hba1c"])):
        c = g(R, "controls", s, y) or {}
        got = dfmt(c.get("b"), c.get("lo"), c.get("hi")) if c else "NA"
        out[key] = {"got": got, "ref": ref, "ok": got == ref}
    cv = g(R, "coverage", "CAD") or {}
    out["cad_cov"] = {"got": [cv.get("rows_in"), cv.get("matched")], "ref": list(REPRO_G["cad_cov"]),
                      "ok": (cv.get("rows_in"), cv.get("matched")) == REPRO_G["cad_cov"]}
    out["all_ok"] = all(v["ok"] for v in out.values() if isinstance(v, dict))
    return out


def acceptance(R):
    out = {}
    for s, y in ACCEPT.items():
        c = g(R, "controls", s, y) or {}
        ok = c.get("b") is not None and fl(c.get("p")) < 0.05 and fl(c.get("b")) > 0
        out[s] = {"control": y, "accepted": bool(ok), "b": c.get("b"), "p": c.get("p")}
    return out


def section_g(root, fr, w1dir, w2dir, block, perms, seed):
    R = {}
    weights, winfo, fam, cov, flow, S_, coverage, tsec = g_scores(root, w1dir, w2dir, block)
    R["weights"], R["timings_scoring_s"] = winfo, tsec
    names = list(weights)
    R["scores"] = names
    scores = pd.DataFrame(S_, index=fam, columns=names).loc[cov.index]
    pairs = P.king_pairs(root, cov.index)
    unrel = P.unrelated_set(cov.index, pairs)
    k, i0 = pairs["Kinship"].to_numpy(), pairs["IBS0"].to_numpy()
    fs = pairs[(k > P.KIN_FIRST) & (k <= P.KIN_DUP) & (i0 >= P.IBS0_PO_MAX)]
    flow.update({"unrelated": rc(len(unrel)), "full_sib_pairs": rc(len(fs)), "related_pairs": rc(len(pairs))})
    log(f"samples: {A.cnt(len(cov))}; unrelated {A.cnt(len(unrel))}; full-sib pairs {A.cnt(len(fs))}")
    R["flow"] = flow
    R["coverage"] = P.variant_counts_text(coverage)
    # analysis frame exactly as P11 section_a (same rows, same order), then lab columns attached by mapping
    fr2 = L.add_ecg_intervals(root, fr)
    d = A.one_row_per_participant(fr2, ["age", "sex_male"])
    d = d[d["participant_id"].isin(scores.index)].copy()
    d = d.join(scores, on="participant_id").join(cov[[f"pc{i}" for i in range(1, L.N_PCS + 1)]], on="participant_id")
    d["age2"] = (d["age"] - d["age"].mean()) ** 2
    for b_ in L.BEDS:
        if b_ in d.columns:
            d["o_" + b_] = A.ORIENT.get(b_, 1) * d[b_]
    d, R["labs"] = add_labs(root, d)
    base_cov = ["age", "age2", "sex_male"] + [f"pc{i}" for i in range(1, L.N_PCS + 1)]
    bp = ["sbp_sit", "dbp_sit", "antihtn"]
    du = d[d["participant_id"].isin(unrel)]
    corr_pairs = [("CAD", "CAD2"), ("CAD", "CADm"), ("CAD2", "CADm"), ("T2D", "T2D2"), ("CAD2", "HTN"), ("CADm", "HTN")]
    R["score_corr_unrelated"] = {f"{a_}-{b_}": float(du[a_].corr(du[b_])) for a_, b_ in corr_pairs
                                 if a_ in du.columns and b_ in du.columns}
    for key in ("controls", "beds_i", "beds_ii", "within_sib", "joint_i", "joint_ii", "bp_part", "other"):
        R[key] = {}
    for s in names:
        own = CONTROLS.get(s) or [L.OWN.get(s, (None, 0))]
        R["controls"][s] = {y: {"expected_sign": sg, **(P.assoc(du, y, s, base_cov) or {})} for y, sg in own if y in du.columns}
        R["beds_i"][s], R["beds_ii"][s], R["within_sib"][s] = {}, {}, {}
        for b_ in L.BEDS:
            yo = "o_" + b_
            if yo not in du.columns:
                continue
            R["beds_i"][s][b_] = P.assoc(du, yo, s, base_cov)
            R["beds_ii"][s][b_] = P.assoc(du, yo, s, base_cov + bp)
            if s in NEW or b_ in L.CORE:
                R["within_sib"][s][b_] = L.sib_within(d, fs, yo, s)
        R["joint_i"][s] = L.joint_core(du, s, base_cov)
        R["joint_ii"][s] = L.joint_core(du, s, base_cov + bp)
        R["bp_part"][s] = L.bp_part(du, s, base_cov, bp)
        if s in NEW:
            R["other"][s] = {y_: P.assoc(du, y_, s, base_cov) for y_ in L.EXTRA if y_ in du.columns}
    t1 = time.time()
    R["matrix"] = {}
    for key, prim, cv, sd in (("new_i", PRIM_NEW, base_cov, seed), ("new_ii", PRIM_NEW, base_cov + bp, seed + 1),
                              ("old_i", PRIM_OLD, base_cov, seed), ("old_ii", PRIM_OLD, base_cov + bp, seed + 1)):
        if all(s in names for s in prim):
            R["matrix"][key] = L.matrix_test(du, prim, cv, perms, sd)
        else:
            R["matrix"][key] = {"status": "score missing: " + ",".join(s for s in prim if s not in names)}
    R["timings_matrix_s"] = time.time() - t1
    R["cad_imt"] = cad_imt_block(R, du, base_cov)
    R["acceptance"] = acceptance(R)
    R["repro"] = g_repro(R)
    ids = set(d["participant_id"].astype(str)) | set(map(str, fam)) | set(pairs["ID1"].astype(str)) | \
        set(pairs["ID2"].astype(str))
    return R, ids


# ============================================================================ C: reference centiles (Fig. 2g)

def section_c(fr, S, path, want_sha):
    out = {"beds": CBEDS}
    stored, sha_ok = None, False
    if path and os.path.exists(path):
        h = sha_file(path, None)
        sha_ok = h[:16] == want_sha
        out["file"] = {"sha16": hfmt(h), "sha_ok": sha_ok}
        try:
            with open(path, encoding="utf-8") as fh:
                gref = json.load(fh).get("G_ref") or {}
            stored = {bed: {sx: (g(gref, bed, sx) or {}) for sx in ("F", "M")} for bed in CBEDS}
            out["reference_n_stored"] = gref.get("reference_n")
        except Exception as ex:
            out["file"]["error"] = err(ex)
    else:
        out["file"] = {"status": "absent"}
    rec = None
    if S is not None:
        try:
            extra, thr = S["extra"], S["thr"]
            b0 = fr[fr["research_stage"] == BASE].drop_duplicates("participant_id")
            prev = extra["prev"]
            ref = M10.reference_ids(b0, prev["ihd"] | prev["cereb"] | prev["pad"], extra["cur"].get("ihd", set()), thr)
            gr = M10.gap_reference(fr.copy(), ref)
            rec = {bed: {sx: (g(gr, bed, sx) or {}) for sx in ("F", "M")} for bed in CBEDS}
            out["reference_n_recomputed"] = gr.get("reference_n")
        except Exception as ex:
            out["recompute_error"] = err(ex)
    else:
        out["recompute_error"] = "P10/P1 modules not loaded"
    # agreement of recomputed with stored values at the stored precision (4 s.f.)
    if stored is not None and rec is not None:
        nv = ne = 0
        for bed in CBEDS:
            for sx in ("F", "M"):
                for k in ("n_ref", "sd_ref"):
                    nv += 1
                    ne += A.fnum(g(rec, bed, sx, k)) == g(stored, bed, sx, k)
                for band in BANDS:
                    cs, cr = g(stored, bed, sx, "centiles", band) or {}, g(rec, bed, sx, "centiles", band) or {}
                    for q in ["n"] + QK:
                        nv += 1
                        ne += A.fnum(cr.get(q)) == cs.get(q)
        out["agreement"] = {"values": nv, "equal": ne}
    src = "stored" if (stored is not None and sha_ok) else ("recomputed" if rec is not None else
                                                           ("stored (other SHA)" if stored is not None else "none"))
    out["source"] = src
    use = stored if src.startswith("stored") else rec
    out["centiles"] = {}
    for bed in CBEDS:
        out["centiles"][bed] = {}
        for sx in ("F", "M"):
            r = g(use, bed, sx) or {}
            out["centiles"][bed][sx] = {"n_ref": r.get("n_ref"), "sd_ref": r.get("sd_ref"), "status": r.get("status"),
                                        "bands": {band: g(r, "centiles", band) for band in BANDS}}
    return out


# ============================================================================ digest

def ci(c, key="b"):
    return L.ci(c, key)


def pstr(p):
    return L.pstr(p)


def hr_s(r, x):
    """Hazard result -> 'hr[lo,hi]' or the reason it is missing."""
    return L.hzs(r, x)


def ratio_s(rr):
    if not isinstance(rr, dict) or rr.get("ratio_hr") is None:
        return "NA"
    return f"{v3(rr['ratio_hr'])}[{v3(rr.get('lo'))},{v3(rr.get('hi'))}]"


def dg_header(R):
    mods = R.get("modules") or {}
    T, G, C = R.get("T") or {}, R.get("G") or {}, R.get("C") or {}
    tr, gr = T.get("repro") or {}, G.get("repro") or {}
    w = G.get("weights") or {}
    nw1 = sum(1 for v in (w.get("v1") or {}).values() if v.get("sha256_ok"))
    nw2 = sum(1 for v in (w.get("v2") or {}).values() if v.get("sha256_ok"))
    ok = lambda k: "ok" if mods.get(k + "_ok") else "DIFF"  # noqa: E731
    L_ = [(1, f"F1 FOLLOWUP2 v1 DIGEST {R.get('version')} | atlas {mods.get('atlas_sha16')} {ok('atlas')} | prs "
              f"{mods.get('prs_sha16')} {ok('prs')} | P10 {mods.get('p10_sha16')} {ok('p10')} | P1 {mods.get('p1_sha16')} "
              f"{ok('p1')} | P11 {mods.get('p11_sha16')} {ok('p11')} | P10/P1 import {R.get('p10_status')} | "
              + (f"weights v1 {nw1}/{len(w.get('v1') or {})} manifest {mods.get('w1_sha16')} {ok('w1')} v2 {nw2}/"
                 f"{len(w.get('v2') or {})} manifest {mods.get('w2_sha16')} {ok('w2')}" if w else "weights not read")
              + f" | seed {R.get('seed')} P1 seed {R.get('p1_seed')} perms "
              f"{R.get('perms')} | modules {','.join(R.get('modules_run') or [])} | runtime {v3(R.get('runtime_s'))}s"),
          (1, "counts rounded to 10, None = <20; T: HR per SD of the adverse-oriented bed (ret = retinal artery width and "
              "abi per SD lower), P1 cloglog hazard, outcome-specific IPW, robust SE, model B = age sex BMI usual SBP/DBP; "
              "RoR = ratio of HR ratios in people at risk for both outcomes (beds standardised once there); G: beta per SD "
              "of score in SD of the oriented bed, unrelated set, P11 models i/ii; C: P10 reference subset")]
    ft = (lambda k: "ok" if g(tr, k, "ok") else f"DIFF({g(tr, k, 'got')})")
    tline = ("T repro (P10/P1 digests) " + " ".join(f"{k} {ft(k)}" for k in tr if k != "all_ok")) if tr else "T not run"
    fg = (lambda k: "ok" if g(gr, k, "ok") else f"DIFF({g(gr, k, 'got')})")
    gline = ("G repro (P11 digest) " + " ".join(f"{k} {fg(k)}" for k in gr if k != "all_ok")) if gr else "G not run"
    ag = C.get("agreement") or {}
    gfl = (f"; G flow fam {g(G, 'flow', 'fam')} unrelated {g(G, 'flow', 'unrelated')} matrix n {g(G, 'matrix', 'new_i', 'n')}; "
           f"scoring {v3(G.get('timings_scoring_s'))}s matrix {v3(G.get('timings_matrix_s'))}s" if G.get("flow") else "")
    cfl = (f"; C source {C.get('source', 'NA')} file {g(C, 'file', 'sha16') or g(C, 'file', 'status')} "
           f"{'ok' if g(C, 'file', 'sha_ok') else 'DIFF'} recomputed=stored {ag.get('equal')}/{ag.get('values')}" if C else "; C not run")
    L_.append((1, f"check: liver tables read 0 (refused {len(R.get('liver_blocked') or [])}); frame rows {R.get('frame_rows')}; "
                  f"P1 cohort {g(R, 'follow', 'p1_cohort')}; {tline}; {gline}{gfl}{cfl}"))
    return L_


def ror_txt(key, pr, x, oc1, oc2):
    if not isinstance(x, dict):
        return f"{pr} NA"
    if x.get("error"):
        return f"{pr} ERR"
    head = f"{pr}[{x.get('n_common')}; ev {g(x, 'events', oc1)}/{g(x, 'events', oc2)}]"
    if x.get("status") != "ok":
        return f"{head} NA({str(x.get('status'))[:10]})"
    rr = x.get("ror") or {}
    return (f"{head} {oc1} {ratio_s(g(x, 'ratio', oc1))} {oc2} {ratio_s(g(x, 'ratio', oc2))} RoR "
            f"{v3(rr.get('ratio_hr'))}[{v3(rr.get('lo'))},{v3(rr.get('hi'))}] p{pstr(rr.get('p'))}")


def verdict(x):
    rr = (x or {}).get("ror") or {}
    if rr.get("p") is None:
        return "not estimable"
    if fl(rr["p"]) < 0.05:
        return "SUPPORTED" if fl(rr.get("ratio_hr")) > 1 else "OPPOSITE direction"
    return "not supported (bound only)"


def dg_t(T):
    L_ = []
    if not isinstance(T, dict) or not T:
        return L_
    if T.get("error") or T.get("status"):
        return [(1, f"T {T.get('error') or T.get('status')}")]
    for i, (oc1, oc2) in enumerate(CONTRASTS):
        key = f"{oc1}_vs_{oc2}"
        rb = g(T, "ror", key) or {}
        om = g(T, "omnibus", key) or {}
        pr = "pwv/abi"
        oms = " ".join(f"{k} p{pstr(g(v, 'wald', 'p'))} [{v.get('n_common')}; ev {g(v, 'events', oc1)}/{g(v, 'events', oc2)}]"
                       if isinstance(v, dict) and v.get("status") == "ok" else f"{k} NA" for k, v in om.items())
        lab = "primary" if i == 0 else "secondary"
        L_.append((1, f"DT{i + 1} bed-specific treatment-initiation profile ({lab}: {oc1} vs {oc2}; same people at risk for "
                      f"both; model B): {pr} RoR -> {verdict(rb.get(pr))} | "
                      + " | ".join(ror_txt(key, p_, rb.get(p_), oc1, oc2) for p_ in [f"{SH[a]}/{SH[b]}" for a, b in PAIRS6[:PRIMARY_PAIRS]])
                      + f" | omnibus (ref abi) {oms}"))
    for oc1, oc2 in CONTRASTS:
        key = f"{oc1}_vs_{oc2}"
        rb = g(T, "ror", key) or {}
        L_.append((2, f"T RoR {oc1} vs {oc2} other pairs: " + " | ".join(
            ror_txt(key, p_, rb.get(p_), oc1, oc2) for p_ in [f"{SH[a]}/{SH[b]}" for a, b in PAIRS6[PRIMARY_PAIRS:]])))
    oc1, oc2 = CONTRASTS[0]
    ra = g(T, "ror_A", f"{oc1}_vs_{oc2}") or {}
    if ra:
        L_.append((3, f"T RoR model A (age sex) {oc1} vs {oc2}: " + " | ".join(ror_txt("", p_, v, oc1, oc2) for p_, v in ra.items())))
    for oc in ("ahtn", "statin", "comp", "htn_obj"):
        r = g(T, "outcomes", oc) or {}
        if r.get("error"):
            L_.append((1, f"T {oc} ERROR {r['error']}"))
            continue
        bk = r.get("events_by_kind") or {}
        L_.append((1 if oc == "comp" else 2,
                   f"T {oc} cohort {r.get('cohort')} followed {r.get('followed')} events {r.get('events')} (c1 {bk.get('call1')} "
                   f"v2 {bk.get('v2')} c3 {bk.get('call3')}) | B: " + " ".join(
                       f"{SH[b_]} {hr_s(g(r, 'beds_B', b_), 'o_' + b_)} {g(r, 'beds_B', b_, 'n')}/{g(r, 'beds_B', b_, 'events')}"
                       for b_ in CORE4)))
        prs_ = r.get("pairs_B") or {}

        def pl(k):
            x = prs_.get(k) or {}
            head = f"{k}[{x.get('n')}/{x.get('events')}]"
            if x.get("status") != "ok":
                return f"{head} NA({str(x.get('status'))[:8]})"
            rr = (x.get("ratio") or {}).get(k) or {}
            return f"{head} {ratio_s(rr)} p{pstr(rr.get('p'))}"
        L_.append((1 if oc != "htn_obj" else 2, f"T {oc} pairs B (HR ratio, own common sample): " + " ".join(
            pl(f"{SH[a]}/{SH[b]}") for a, b in PAIRS6)))
        if oc == "comp" and isinstance(r.get("S3_B"), dict):
            s3 = r["S3_B"]
            L_.append((2, f"T comp S3 (pwv imt abi) [{s3.get('n')}/{s3.get('events')}] Wald p{pstr(g(s3, 'wald', 'p'))} " + " ".join(
                f"{k} {ratio_s(v)}" for k, v in (s3.get("ratio") or {}).items())))
    return L_


def ctrl_s(R, s):
    cs = g(R, "controls", s) or {}
    return " ".join(f"{y} {ci(c)}" + (f" p{pstr(c.get('p'))}" if c.get("b") is not None else "") + f" n {c.get('n')}"
                    for y, c in cs.items() if isinstance(c, dict))


def dg_g(G):
    L_ = []
    if not isinstance(G, dict) or not G:
        return L_
    if G.get("error") or G.get("status"):
        return [(1, f"G {G.get('error') or G.get('status')}")]
    cv, w2 = G.get("coverage") or {}, g(G, "weights", "v2") or {}
    acc = G.get("acceptance") or {}
    parts = []
    for s in ("CAD2", "T2D2"):
        a_ = acc.get(s) or {}
        c = g(G, "controls", s, a_.get("control")) or {}
        parts.append(f"{s} {g(w2, s, 'pgs_id')} coverage {v3(g(cv, s, 'coverage'))} -> {a_.get('control')} {ci(c)} "
                     f"p{pstr(c.get('p'))} {'ACCEPTED' if a_.get('accepted') else 'NOT ACCEPTED'}")
    L_.append((1, "DG0 replacement scores (pre-stated: accepted only if CAD2 -> LDL, T2D2 -> HbA1c p < 0.05 with beta > 0): "
                  + " | ".join(parts)))
    for key, lab in (("new_i", "i"), ("new_ii", "ii")):
        m = g(G, "matrix", key) or {}
        t2, t1 = m.get("T2_score_x_bed_interaction") or {}, m.get("T1_within_score_bed_heterogeneity") or {}
        if not t2:
            L_.append((1, f"DG1{lab} matrix: {m.get('status', m.get('error', 'NA'))}"))
            continue
        differ = (t2.get("perm_p") is not None and fl(t2["perm_p"]) < 0.05 and fl(t2["p"]) < 0.05)
        verdict_ = "PROFILES DIFFER across outcome liabilities" if differ else "compatible with one shared bed profile"
        o2 = g(G, "matrix", "old_" + lab, "T2_score_x_bed_interaction") or {}
        L_.append((1, f"DG1{lab} P11 D1{lab} with CAD2/T2D2 ({len(m.get('scores') or [])} scores x 4 core beds, model {lab}, n "
                      f"{m.get('n')}): score x bed interaction chi2 {v3(t2.get('chi2'))} df {t2.get('df')} p {pstr(t2.get('p'))} "
                      f"perm p {pstr(t2.get('perm_p'))} (null mean/df {v3(t2.get('perm_mean_over_df'))}) -> {verdict_} | T1 chi2 "
                      f"{v3(t1.get('chi2'))} p {pstr(t1.get('p'))} | old scores chi2 {v3(o2.get('chi2'))} p {pstr(o2.get('p'))} "
                      f"perm p {pstr(o2.get('perm_p'))}"))
    alpha = 0.05 / len(PRIM_NEW)
    parts = []
    for s in PRIM_NEW:
        j = g(G, "joint_i", s)
        if not isinstance(j, dict) or not j.get("beta"):
            continue
        pp = g(j, "heterogeneity", "p")
        het = pp is not None and fl(pp) < alpha
        top = max(L.CORE, key=lambda b_: fl(g(j, "beta", b_, "b")))
        dif = j.get("diff") or {}
        best = min(dif, key=lambda k_: fl(dif[k_].get("p")) if dif[k_].get("p") is not None else 9) if dif else None
        parts.append(f"{s} p {pstr(pp)}" + (f" HET top {SH[top]} {v3(g(j, 'beta', top, 'b'))}; "
                                            f"{best.replace('_minus_', '-').replace('abi_min', 'abi').replace('art_w', 'ret')} "
                                            f"{ci(dif.get(best), 'd')}" if het and best else ""))
    L_.append((1, f"DG2 per-score core-bed heterogeneity, new set (model i, Wald 3 df; HET p < {alpha:.2g}): " + " | ".join(parts)))
    bpl = []
    for s in PRIM_NEW:
        for b_ in L.CORE:
            r = g(G, "bp_part", s, b_) or {}
            if fl(g(r, "i", "p")) < 0.05 / (len(PRIM_NEW) * 4) and r.get("diff"):
                bpl.append(f"{s} {SH[b_]} i {v3(g(r, 'i', 'b'))} BP part {ci(r['diff'])} ({v3(r.get('share'))})")
    L_.append((1, "DG3 through visit BP, new set (model i - model ii on the model-ii sample; cells with model-i p < 0.05/36; "
                  "share in brackets): " + (" | ".join(bpl) if bpl else "none")))
    ci_ = G.get("cad_imt") or {}
    sc = G.get("score_corr_unrelated") or {}
    L_.append((1, "DG4 CAD -> IMT old vs new (beta per SD, unrelated): model i " + " ".join(
        f"{s} {ci(g(ci_, s, 'i'))}" for s in ("CAD", "CAD2", "CADm") if s in ci_) + " | model ii " + " ".join(
        f"{s} {ci(g(ci_, s, 'ii'))}" for s in ("CAD", "CAD2", "CADm") if s in ci_) + " | same-sample new-old i " + " ".join(
        f"{k.split('_')[0]} {ci(v)}" for k, v in ci_.items() if k.endswith("_minus_CAD_i")) + " | BP part " + " ".join(
        f"{s} {ci(g(ci_, s, 'bp_part', 'diff'))}" for s in ("CAD", "CAD2") if s in ci_) + " | r " + " ".join(
        f"{k} {v3(v)}" for k, v in sc.items())))
    L_.append((1, "DG5 positive controls (unrelated, model i; beta per SD of score in SD of the trait): " + " | ".join(
        f"{s}: {ctrl_s(G, s)}" for s in ("CAD2", "T2D2", "CADm", "CAD", "T2D") if s in (G.get("controls") or {}))))
    for s in NEW + ["CAD", "T2D"]:
        bi = g(G, "beds_i", s) or {}
        if bi:
            L_.append((1, f"A i {s}: " + " ".join(f"{SH[b_]} {ci(bi.get(b_))}" for b_ in L.CORE if b_ in bi)
                       + f" | {L.nrange([bi.get(b_) for b_ in L.CORE])}"))
    for s in NEW:
        j = g(G, "joint_i", s)
        if isinstance(j, dict) and j.get("diff"):
            L_.append((1, f"J i {s} n {j.get('n')} p {pstr(g(j, 'heterogeneity', 'p'))}: {L.jline(j)}"))
    for s in NEW + ["CAD", "T2D"]:
        c = cv.get(s) or {}
        wi = g(G, "weights", "v2", s) or g(G, "weights", "v1", s) or {}
        L_.append((1 if s in NEW else 2, f"cov {s} {wi.get('pgs_id')} | rows in {c.get('rows_in')} | matched {c.get('matched')} | "
                                          f"{v3(c.get('coverage'))} | direct {c.get('direct')} swap {c.get('swap')} flip "
                                          f"{c.get('flip')} pal {c.get('pal_kept')}/{c.get('pal_dropped')} | sha "
                                          f"{'ok' if wi.get('sha256_ok') else 'DIFF'}"))
    for s in NEW:
        bi = g(G, "beds_ii", s) or {}
        if bi:
            L_.append((2, f"A ii {s}: " + " ".join(f"{SH[b_]} {ci(bi.get(b_))}" for b_ in L.CORE if b_ in bi)
                       + f" | {L.nrange([bi.get(b_) for b_ in L.CORE])}"))
        j = g(G, "joint_ii", s)
        if isinstance(j, dict) and j.get("diff"):
            L_.append((2, f"J ii {s} n {j.get('n')} p {pstr(g(j, 'heterogeneity', 'p'))}: {L.jline(j)}"))
        r = g(G, "bp_part", s) or {}
        L_.append((2, f"BP part {s} (i - ii, model-ii sample): " + " ".join(f"{SH[b_]} {ci(g(r, b_, 'diff'))}" for b_ in L.CORE)))
    for s in NEW:
        bi = g(G, "beds_i", s) or {}
        L_.append((3, f"A i2 {s}: " + " ".join(f"{b_} {ci(bi.get(b_))}" for b_ in L.BEDS[4:] if b_ in bi)))
        ws = g(G, "within_sib", s) or {}
        L_.append((3, f"A sib {s}: " + " ".join(f"{SH[b_]} {ci(ws.get(b_))}" for b_ in L.CORE if b_ in ws)
                   + f" | pairs {g(ws, 'pwv', 'n_pairs')}"))
        o = g(G, "other", s) or {}
        L_.append((3, f"other {s}: " + " ".join(f"{y} {ci(o.get(y))}" for y in L.EXTRA if y in o)))
    for s in [x for x in G.get("scores") or [] if x not in NEW + ["CAD", "T2D"]]:
        bi = g(G, "beds_i", s) or {}
        L_.append((3, f"A i {s} (unchanged score): " + " ".join(f"{SH[b_]} {ci(bi.get(b_))}" for b_ in L.CORE if b_ in bi)))
    lb = G.get("labs") or {}
    L_.append((2, f"labs: NMR {L.kvs(lb.get('nmr')) if isinstance(lb.get('nmr'), dict) else lb.get('nmr')} | blood-test "
                  f"non-HDL n {g(lb, 'blood_tests', 'nhdl_bt')} TC unit {g(lb, 'blood_tests', 'tc_unit')}"))
    return L_


def dg_c(C):
    L_ = []
    if not isinstance(C, dict) or not C:
        return L_
    if C.get("error"):
        return [(1, f"C ERROR {C['error']}")]
    L_.append((1, f"R = Fig. 2g reference centiles (P10 reference subset n {C.get('reference_n_stored') or C.get('reference_n_recomputed')}; "
                  f"source {C.get('source')}): per bed and sex, ref n, ref SD; per 5-year age band n q05/q25/q50/q75/q95 of the "
                  "bed (raw units, atlas v1.5 corrected; ret = artery width, abi = lower-leg ABI), None = cell < 20"))
    for bed in CBEDS:
        for sx in ("F", "M"):
            r = g(C, "centiles", bed, sx) or {}
            bands = r.get("bands") or {}
            txt = " ".join(f"{band} {g(bands, band, 'n')}" + (
                ":" + "/".join(v3(g(bands, band, q)) for q in QK) if g(bands, band, "q50") is not None else "")
                for band in BANDS)
            L_.append((1, f"R {SH[bed]} {sx} n {r.get('n_ref')} sd {v3(r.get('sd_ref'))}"
                          + (f" {r.get('status')}" if r.get("status") else "") + f": {txt}"))
    return L_


def digest(R):
    L_ = dg_header(R)
    if R.get("fatal"):
        L_.append((1, f"FATAL {R['fatal']}"))
    L_ += dg_t(R.get("T"))
    L_ += dg_c(R.get("C"))
    L_ += dg_g(R.get("G"))
    L_.append((3, "timings s: " + " ".join(f"{k} {v3(v)}" for k, v in (R.get("timings") or {}).items())))
    return "\n".join(L.shrink(L.wrap(L_))) + "\n"


# ============================================================================ main

MODULES = ("T", "G", "C")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="HPP dataset root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", default=os.path.join(HERE, "f1_atlas_v1_5.py"))
    ap.add_argument("--prs", default=os.path.join(HERE, "f1_prs_v1_1.py"))
    ap.add_argument("--p10", default=os.path.join(HERE, "f1_atlas_outcomes_v2.py"))
    ap.add_argument("--outcomes", default=os.path.join(HERE, "f1_outcomes_v1.py"))
    ap.add_argument("--p11", default=os.path.join(HERE, "f1_outcome_liability_v1.py"))
    ap.add_argument("--weights-v1", default=os.path.join(HERE, "pgs_weights_outcome_v1"))
    ap.add_argument("--weights-v2", default=os.path.join(HERE, "pgs_weights_outcome_v2"))
    ap.add_argument("--p10-results", default=os.path.join(HERE, "f1_atlas_outcomes_out_v2", "results.json"))
    ap.add_argument("--p10-results-sha", default=EXPECT["p10res"], help="expected SHA-256 (16) of the P10 results.json")
    ap.add_argument("--seed", type=int, default=20261006, help="permutation seed (P11's; keep 20261006)")
    ap.add_argument("--p1-seed", type=int, default=20261004, help="P1 seed (usual-BP folds; keep 20261004)")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--block", type=int, default=20000, help="max variant rows per genotype read (PRS v1.1)")
    ap.add_argument("--modules", default=",".join(MODULES))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    if not os.path.isdir(a.root):
        raise SystemExit(f"dataset root not found: {a.root}")
    mods_run = [s.strip().upper() for s in a.modules.split(",") if s.strip()]
    for p_ in (a.atlas, a.prs, a.p11):
        if not os.path.exists(p_):
            raise SystemExit(f"module not found: {p_}")
    if "G" in mods_run:
        for wd in (a.weights_v1, a.weights_v2):
            if not os.path.isdir(wd):
                raise SystemExit(f"module G needs the weights folder {wd}")
    st = load_modules(a)
    t0 = time.time()
    shas = {"atlas": sha_file(a.atlas), "prs": sha_file(a.prs), "p11": sha_file(a.p11),
            "p10": sha_file(a.p10) if os.path.exists(a.p10) else "absent",
            "p1": sha_file(a.outcomes) if os.path.exists(a.outcomes) else "absent"}
    for k, wd in (("w1", a.weights_v1), ("w2", a.weights_v2)):
        mf = sorted(f for f in os.listdir(wd) if f.endswith("_manifest.tsv")) if os.path.isdir(wd) else []
        shas[k] = sha_file(os.path.join(wd, mf[0])) if len(mf) == 1 else "absent"
    mods = {}
    for k, h in shas.items():
        mods[f"{k}_sha16"] = hfmt(h) if h != "absent" else "absent"
        mods[f"{k}_ok"] = h == EXPECT[k]
    log(f"{VERSION}; atlas {vq(A.VERSION)} {mods['atlas_sha16']}; prs {vq(P.VERSION)} {mods['prs_sha16']}; P11 "
        f"{mods['p11_sha16']}; P10/P1 {st}; numpy {np.__version__}; pandas {pd.__version__}; python {sys.version.split()[0]}")
    if not all(mods[f"{k}_ok"] for k in ("atlas", "prs", "p11", "p10", "p1")):
        log("WARNING a module is not the expected version (digest header shows DIFF)")
    res = {"version": VERSION, "atlas_version": vq(A.VERSION), "modules": mods, "p10_status": st, "seed": f"{a.seed:,}",
           "p1_seed": f"{a.p1_seed:,}", "perms": a.perms, "modules_run": mods_run}
    pids = set()
    try:
        fr = A.build_frame(a.root, None)[0].reset_index(drop=True)
        res["frame_rows"] = rc(len(fr))
        pids |= set(fr["participant_id"].astype(str))
        log("frame built")
        S = None
        if ("T" in mods_run or "C" in mods_run) and st == "ok":
            t1 = time.time()
            S, ids = build_followup(a.root, fr, a.p1_seed)
            pids |= ids
            res["follow"] = S["info"]
            res.setdefault("timings", {})["follow_build"] = time.time() - t1
            log(f"follow-up data built in {time.time() - t1:.1f}s")
        if "T" in mods_run:
            if S is None:
                res["T"] = {"status": f"P10/P1 import {st}: skipped"}
            else:
                guard(res, "T", lambda: section_t(S))
                log("T done")
        if "C" in mods_run:
            guard(res, "C", lambda: section_c(fr, S, a.p10_results, a.p10_results_sha))
            log("C done")
        if "G" in mods_run:
            t1 = time.time()
            try:
                RG, ids = section_g(a.root, fr, a.weights_v1, a.weights_v2, a.block, a.perms, a.seed)
                pids |= ids
                res["G"] = RG
            except Exception as ex:
                res["G"] = {"error": err(ex)}
                log(f"G FAILED: {res['G']['error']}")
            res.setdefault("timings", {})["G"] = time.time() - t1
            log(f"G done in {time.time() - t1:.1f}s")
    except Exception as ex:
        res["fatal"] = err(ex)
        log(f"FATAL {res['fatal']}")
    res["liver_blocked"] = sorted(set(re.sub(r"\d{5,}", "#", b) for b in L.BLOCKED))
    res["runtime_s"] = time.time() - t0
    text = json.dumps(clean(res), ensure_ascii=False, sort_keys=False, indent=1)
    privacy_check(text, pids)
    try:
        dg = digest(json.loads(text))
    except Exception as ex:  # a digest failure must not lose results.json
        dg = f"F1 FOLLOWUP2 v1 DIGEST {VERSION}\nDIGEST FAILED: {err(ex)}; see results.json\n"
        log(f"digest FAILED: {err(ex)}")
    privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines (max {max(len(x) for x in dg.splitlines())} chars); total {time.time() - t0:.1f}s")
    logtext = "\n".join(LONG.sub("#", vq(x)) for x in A.LOG) + "\n"
    privacy_check(logtext, pids)
    os.makedirs(a.out)
    sums = []
    for fn_, tx in (("results.json", text), ("digest.txt", dg), ("run_log.txt", logtext)):
        with open(os.path.join(a.out, fn_), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn_}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s[:16] + "  " + s.split()[-1] for s in sums))


if __name__ == "__main__":
    main()
