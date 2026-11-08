#!/usr/bin/env python3
"""F1 Stage 0, task j: NHANES acquisition physics of blood pressure and ABI, with 2019 linked mortality (public data).

GATED: refuses to run on real data without --real. Use --simulate for a planted-answer wiring test. Unlike drivers a-i,
this driver reads the raw public files under external/nhanes/ directly (read-only; no clean table exists) and writes
only results/stage0/analysis/j_nhanes_acq/ (simulate mode: its subfolder simulate_validation/).

Label: PUBLIC-DATA SUPPORTING EVIDENCE for F1's measurement-physics question. These are analogues, not replications:
auscultatory BP and Doppler ABI, not oscillometric leg PWV. No HPP data are used.

Pre-specified analysis
----------------------
Data A (continuous NHANES 1999-2016, 9 cycles): MEC-examined records (RIDSTATR = 2) of DEMO, BPX (auscultatory
mercury BP; 3 readings after 5 min seated rest, a 4th if one was interrupted), BPQ, SMQ, DIQ, MCQ, BMX, total/HDL
cholesterol, HbA1c, serum creatinine; LEXABPI/LEXAB_B/LEXAB_C (Doppler ABI, 1999-2004, age >= 40); 2019 public-use
linked mortality files. 18-year MEC weight = 2/9 x WTMEC4YR (1999-2002) or 1/9 x WTMEC2YR (2003-2016); strata
SDMVSTRA (unique across cycles, asserted), PSUs SDMVPSU nested in strata; Taylor linearisation; lonely PSU 'adjust';
domains by zero weights, never by dropping records. Intervals: t(design df) for model coefficients.
Domains: BP = age >= 20, not pregnant (RIDEXPRG = 1 excluded), reading 1 and >= 2 of readings 2-4 present;
MORT = BP + linkage-eligible + complete M2. Time = PERMTH_EXM / 12 (minimum 0.5 month). Outcomes: all-cause death,
CVD death (UCOD_LEADING 1 or 5; cause-specific). Efron ties.
Covariates: M1 = age (per 10 y), sex, race/ethnicity (RIDRETH1; ref non-Hispanic white); M2 = M1 + smoking (former,
current), diabetes (self-report, insulin/pills, HbA1c >= 6.5 %), BMI (per 5), total and HDL cholesterol (per 10
mg/dL), BP medication (BPQ050A), eGFR CKD-EPI 2021 (per 10; creatinine 1999-2000 1.013x + 0.147, 2005-2006
0.978x - 0.016), prevalent CVD (MCQ160B-F). ABI models add seated SBP (later mean, per 10 mmHg) as in task f.
Q1 acute vs usual BP at the visit: first reading (SBP1), mean of later readings (2-4), within-visit acute excess =
SBP1 - later mean; also the all-reading mean. Survey-weighted Cox per SD (weighted SD in MORT) and per 10 mmHg, each
alone and mutually adjusted (later + excess; first + later), M1 and M2; DBP analogues (M2). Discrimination:
survey-weighted Harrell C (pair weight w_i w_j; lp ties 1/2) of M2 + first vs M2 + later mean etc.; C and C
differences with Rao-Wu PSU-bootstrap SEs (linear predictors fixed at the full-sample fit). Misclassification of
BP-defined hypertension (>= 140/90; secondary >= 130/80) by the first reading vs the later mean (survey-weighted
proportions, all and untreated), and Cox HRs of the discordant groups (ref: neither).
Q2 acquisition conditions: 6-month exam period (RIDEXMON 1 = Nov-Apr 'winter' vs May-Oct) and survey cycle. Design-
based linear regressions of SBP first / later / excess, DBP later, pulse, and (1999-2004, age >= 40) lower-leg ABI,
per-leg ABI, supine brachial and mean ankle SBP on winter + cycle indicators + M1; Wald test of cycle (design-based).
Weighted means by cycle x period. Whether conditions change mortality HRs and C: M2 vs M2 + winter with the baseline
hazard stratified by cycle, and 'acquisition-corrected' exposures (period and cycle components of the regression
removed, mean kept); low ABI (< 0.9) reclassification by raw vs corrected lower-leg ABI and HRs of both classes.
Q3 reliability and dilution: one-way ICC (weighted; Rao-Wu CIs) of readings 2 vs 3 and 1 vs 2; regression-dilution
ratio (RDR) lambda1 = coefficient of reading 3 on reading 2 + M2 covariates (design-based); Spearman-Brown for means
(lambda_k = k lambda1 / (1 + (k - 1) lambda1)). Cox per 10 mmHg (M2) of reading 1, 2, 3, mean(2, 3), mean(1-3) in the
domain with readings 1-3: observed log-HR ratios vs those predicted by lambda; corrected log-HR = observed / lambda
(delta-method SE). ABI: left vs right ICC (agreement, includes true leg differences); within-visit repeat ABI (ages
40-59, two brachial and ankle readings): lower-leg ABI from reading set 1 vs set 2, ICC and lambda1; Cox per 0.1 lower
ABI (40-59, highest leg <= 1.4) for single-set and NCHS mean-based ABI, raw and corrected.
Q4 within-day session (PHDSESN, PH/FASTQX files): NOT on disk for 1999-2016. When present, the driver uses it
automatically: session (0 morning, 1 afternoon, 2 evening) effects on BP/pulse/ABI (+ winter + cycle + M1), HRs with
and without session adjustment, session-corrected SBP and C. Otherwise it is skipped and the hub request is listed.
Data B (NHANES III 1988-1994, on disk; public analogue with richer acquisition data): adult.dat (household BP by
interviewers, 3 readings; MEC session MXPSESSR 1 morning / 2 afternoon / 3 evening, exam month MXPTIMO, day of week
MXPTIDW, Census region, phase, age in months at screener and exam), exam.dat (MEC physician BP, 3 readings; pulse;
BMI), lab.dat (lipids, HbA1c, creatinine 0.960x - 0.184), NHANES III 2019 linked mortality. Weight WTPFEX6, pseudo-
strata SDPSTRA6, pseudo-PSUs SDPPSU6. Domain: age >= 20, MEC examined, 3 MEC readings, household reading 1 and >= 2
household readings; mortality as above; covariates N2 = M2 analogue (prevalent CVD = MI, CHF, stroke).
B1 acquisition: regressions of MEC mean SBP, MEC first-reading excess, pulse, household mean SBP (negative control:
measured on another day) and the within-person contrast MEC - household on session (afternoon, evening vs morning),
weekend, month harmonics (cos, sin of 2 pi (month - 1) / 12), phase 2, + N1; Wald tests (session 2 df, month 2 df);
sensitivity + Census region.
B2 reliability: within-occasion ICC (MEC and household readings 2 vs 3) vs between-occasion ICC (household mean vs MEC
mean); RDRs: within-visit lambda (MEC reading 3 on 2 + N2; Spearman-Brown for 3) vs between-occasion lambda for the
MEC mean (household mean on MEC mean + N2) and for the household mean; by interval between screener and MEC (months).
B3 mortality: Cox per 10 mmHg (N2) of MEC mean, household mean, two-occasion mean, MEC first reading, MEC later mean;
corrected HRs of the MEC mean with within-visit vs between-occasion lambda; per SD mutual models (MEC later + MEC
excess; two-occasion mean + occasion deviation MEC - household); MEC mean HR without and with acquisition adjustment
(session, weekend, month harmonics, phase) and within sessions (interaction Wald test, 2 df); C and C differences.
Simulate mode: fake raw frames with the real variable names (wiring check against XPT headers and SAS layouts only),
planted acquisition effects, reliability, dilution and mortality effects; C-index fast vs brute-force check.

Review additions (2026-10-05, after the first real run; NOT pre-specified)
--------------------------------------------------------------------------
Fixes: DIQ070 is named DID070 in DIQ_D/DIQ_E (2005-2008), now used; multi-df Wald tests also as design-based F
(q, design df); interval variable described as screener-to-MEC; report wording hedged. Added: (A) ABI cuff size
(LEDARMCF/LEDANKCF; documented protocol change in 1999-2000) and arm circumference (BMXARMC): weighted cuff shares by
cycle; ABI, ankle and brachial cycle coefficients without/with arm circumference and cuff indicators. (B) NHANES III
session design: random household assignment to morning vs afternoon/evening (subsample weights WTPFSD6/WTPFMD6;
contrast among persons examined in the assigned session), session model + fasting hours (lab PHPFAST), venipuncture
clock time (lab PHPBEST) instead of / within session.

Run-2 changes (2026-10-05, after the hub placed PH/FASTQX on disk; NOT pre-specified; CHANGELOG_20261005.md)
-------------------------------------------------------------------------------------------------------------
Q4 now runs as pre-specified above (PHDSESN 0 morning / 1 afternoon / 2 evening in all nine codebooks on disk). Fix:
XPORT zero (read by pandas as 5.4e-79) is set to 0.0 on reading. This was needed for PHDSESN (code 0 = morning) and
also corrects the DBP readings recorded as 0 (BPXDI 0; 0.6-1.8 % of readings per cycle), which the 'x > 0' filter was
meant to set missing but kept as ~0. Every estimate that uses DBP changes slightly (DBP models; BP-defined hypertension
at >= 140/90 and >= 130/80, its prevalences and its Cox models, whose domains lose the records with a DBP of 0); no
estimate based only on SBP, pulse or ABI, and no NHANES III estimate, changes (checked against run 1). Q4 is added to
the main findings and the key-estimate table. The session is not documented as randomised in the 1999-2016
files on disk, so Q4 contrasts are between persons (no household BP exists in 1999-2016).
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib_survival as ls  # noqa: E402
import s0an_common as C  # noqa: E402
import s0lib as L  # noqa: E402

TASK = "j_nhanes_acq"
NH = L.EXT / "nhanes"
N3 = NH / "nhanes_iii"
MORT_DIR = NH / "linked_mortality_2019"
OUT_REAL = C.ANALYSIS / TASK
OUT_SIM = C.ANALYSIS / TASK / "simulate_validation"

CYCLES = [(1, "1999-2000", "", 1999), (2, "2001-2002", "_B", 2001), (3, "2003-2004", "_C", 2003),
          (4, "2005-2006", "_D", 2005), (5, "2007-2008", "_E", 2007), (6, "2009-2010", "_F", 2009),
          (7, "2011-2012", "_G", 2011), (8, "2013-2014", "_H", 2013), (9, "2015-2016", "_I", 2015)]
CYC_LAB = {k: c for k, c, _, _ in CYCLES}
MORT_LAYOUT = {"SEQN": (1, 6), "ELIGSTAT": (15, 15), "MORTSTAT": (16, 16), "UCOD_LEADING": (17, 19),
               "PERMTH_EXM": (46, 48)}
DEMO_V = ["SDDSRVYR", "RIDSTATR", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDEXMON", "RIDEXPRG", "WTMEC2YR",
          "WTMEC4YR", "SDMVPSU", "SDMVSTRA"]
BPX_V = ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4", "BPXPLS"]
FILE_V = {"BPX": BPX_V, "BPQ": ["BPQ020", "BPQ040A", "BPQ050A"], "SMQ": ["SMQ020", "SMQ040"],
          "DIQ": ["DIQ010", "DIQ050", "DIQ070"], "MCQ": ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"],
          "BMX": ["BMXBMI", "BMXARMC"], "TC": ["LBXTC"], "GHB": ["LBXGH"],
          "LEX": ["LEXBRP1", "LEXBRP2", "LEXBRPM", "LEXRPTS1", "LEXRPTS2", "LEXRPTSM", "LEXLPTS1", "LEXLPTS2",
                  "LEXLPTSM", "LEXRABPI", "LEXLABPI", "LEDARMCF", "LEDANKCF"]}
ALT_V = {"HDL": ("HDL_h", ["LBDHDL", "LBXHDD", "LBDHDD"]), "BIO": ("SCR_h", ["LBXSCR", "LBDSCR"])}
# Same question, renamed in some releases (codebooks on disk): DIQ070 "take diabetic pills" is DID070 in DIQ_D/DIQ_E
# (2005-2008; same codes 1 yes / 2 no). Review 2026-10-05: the first version treated it as absent in 2005-2008.
RENAME_V = {"DIQ": {"DID070": "DIQ070"}}
OPTIONAL_V = {"WTMEC4YR"}                 # WTMEC4YR only 1999-2002
# ABI cuff codes (LEDARMCF/LEDANKCF; 2001-2004 recoded by NCHS to the 1999-2000 scheme): 1 9 cm, 2 10 cm, 3 12 cm,
# 4 15 cm, 5 18 cm, 8 could not obtain. LEXABPI.htm: cuff protocol changed during 1999-2000 (10/12 cm -> 9-18 cm by arm
# circumference); 1999-2000 ankles were mostly measured with the 10 cm cuff, 2001-2004 ankles mostly with 15 cm.
CUFF_CM = {1: 9, 2: 10, 3: 12, 4: 15, 5: 18}
SESSION_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{y}/DataFiles/{stem}.{ext}"
N3_ADULT = ["SEQN", "DMPSTAT", "HSSEX", "HSAGEIR", "DMARETHN", "SDPPHASE", "SDPPSU6", "SDPSTRA6", "WTPFEX6",
            "HSAITMOR", "MXPSESSR", "MXPTIMO", "MXPTIDW", "MXPAXTMR", "DMPCREGN", "HAZA8AK1", "HAZA8BK1", "HAZA8CK1",
            "HAZA8DK1", "HAC1C", "HAC1D", "HAD1", "HAD3", "HAD4", "HAD6", "HAD10", "HAE2", "HAE4A", "HAE5A", "HAF10",
            "HAR1", "HAR3"]
N3_EXAM = ["SEQN", "PEP6G1", "PEP6H1", "PEP6I1", "PEP6DR", "BMPBMI"]
N3_ADULT += ["WTPFSD6", "WTPFMD6"]       # morning / afternoon-evening session subsample weights (random assignment)
N3_LAB = ["SEQN", "TCP", "HDP", "GHP", "CEP", "PHPFAST", "PHPBEST"]
N3_FILL = {"HAZA8AK1": ["888"], "HAZA8BK1": ["888"], "HAZA8CK1": ["888"], "HAZA8DK1": ["888"], "PEP6G1": ["888"],
           "PEP6H1": ["888"], "PEP6I1": ["888"], "PEP6DR": ["888"], "BMPBMI": ["8888"], "TCP": ["888"],
           "HDP": ["888"], "GHP": ["8888"], "CEP": ["8888"], "HSAITMOR": ["9999"], "MXPAXTMR": ["9999"],
           "PHPFAST": ["88888"]}
N3_CLOCK = {"PHPBEST"}                    # "HH:MM" strings (time of venipuncture), 88888 = blank but applicable

M1 = ["age10", "male", "race_mexam", "race_othhisp", "race_black", "race_other"]
M2 = M1 + ["smk_former", "smk_current", "diabetes_any", "bmi5", "tc10", "hdl10", "bp_med", "egfr10", "hx_cvd_any"]
M2A = M2 + ["sbp_later10"]
N1 = ["age10", "male", "race_black", "race_mexam", "race_other"]
N2 = N1 + ["smk_former", "smk_current", "diabetes_any", "bmi5", "tc10", "hdl10", "bp_med", "egfr10", "hx_cvd_any"]
CYC_D = [f"cyc{k}" for k in range(2, 10)]
OUTCOMES = [("death_all", "All-cause death"), ("death_cvd", "CVD death")]

INPUTS: dict[str, dict] = {}
_XPT: dict[Path, pd.DataFrame] = {}


# =================================================================================================================
# Generic helpers
# =================================================================================================================
def _register(p: Path, rows=None):
    if str(p) not in INPUTS:
        INPUTS[str(p)] = {"path": str(p.relative_to(L.HUB)), "sha256": L.sha256(p), "rows": rows}


def read_xpt(p: Path) -> pd.DataFrame:
    if p not in _XPT:
        d = pd.read_sas(p, format="xport", encoding="latin-1")
        # XPORT stores 0 as the smallest IBM float; pandas returns it as 5.4e-79 (fix 2026-10-05, run 2). Exact
        # comparisons with code 0 (PHDSESN 0 = morning) and 'x > 0' filters (BPXDI 0 = diastolic not heard) need a
        # true zero, so every numeric value with |x| < 1e-70 is set to 0.0 (no real NHANES value is that small).
        num = d.select_dtypes(include=[np.floating]).columns
        if len(num):
            v = d[num].to_numpy(copy=True)
            v[(v != 0) & (np.abs(v) < 1e-70)] = 0.0
            d[num] = v
        d["SEQN"] = d["SEQN"].astype(np.int64)
        assert d["SEQN"].is_unique, p
        _XPT[p] = d
        _register(p, len(d))
    return _XPT[p]


def xpt_columns(p: Path) -> list[str]:
    r = pd.read_sas(p, format="xport", iterator=True, encoding="latin-1")
    cols = [c if isinstance(c, str) else c.name for c in r.columns]
    r.close()
    return cols


def yn(s):
    return s.map({1: 1.0, 2: 0.0})


def ckd_epi_2021(scr, age, female):
    k = np.where(female == 1, 0.7, 0.9)
    a = np.where(female == 1, -0.241, -0.302)
    r = scr / k
    return 142 * np.minimum(r, 1) ** a * np.maximum(r, 1) ** -1.200 * 0.9938 ** age * np.where(female == 1, 1.012, 1)


def even_round(x):
    return 2 * np.round(np.asarray(x, float) / 2)


def tq(df):
    return stats.t.ppf(0.975, df) if df and df > 0 else stats.norm.ppf(0.975)


def sb(lam, k):
    """Spearman-Brown: reliability of the mean of k parallel replicates from single-replicate reliability lam."""
    return k * lam / (1 + (k - 1) * lam)


def sb_se(lam, se, k):
    """Delta-method SE of sb(lam, k)."""
    return se * k / (1 + (k - 1) * lam) ** 2


def fmt_ci(e, lo, hi, nd=3):
    return f"{e:.{nd}f} ({lo:.{nd}f}, {hi:.{nd}f})"


def wmean_dom(x, w, m):
    x = np.asarray(x, float)
    mm = m & np.isfinite(x)
    return float(np.average(x[mm], weights=w[mm]))


# =================================================================================================================
# Survey models
# =================================================================================================================
def ph_tests(f) -> dict:
    """Grambsch-Therneau PH tests (diagnostic only). Per-term statistic as in R survival < 3.0:
    test_j = sum_k (g_k - gbar) r2_kj with r2 = Schoenfeld residuals x V x d (scaled), chi2_j = test_j^2 / (V_jj d Sxx).
    lib_survival.cox_zph uses the unscaled residuals for the per-term statistic, which is not invariant to covariate
    scaling (its GLOBAL test is correct), so the per-term values are recomputed here from its outputs."""
    z = ls.cox_zph(f, "km")
    g, sc = z.attrs["g"], z.attrs["scaled"]
    wpos = f.zph_data["w_all"][f.zph_data["w_all"] > 0]
    V = f.var_model * float(np.mean(wpos))
    dn = len(g)
    xx = g - g.mean()
    test = xx @ (sc - f.coef)
    chi2 = test ** 2 / (np.diag(V) * dn * np.sum(xx ** 2))
    out = dict(zip(f.names, stats.chi2.sf(chi2, 1)))
    out["GLOBAL"] = float(z["p"].iloc[-1])
    return out


def svy_cox(d, des, dom, outcome, cols, report, strata=None):
    m = dom & d[cols + ["t", outcome]].notna().all(axis=1).values
    X = d[cols].fillna(0).values
    f = ls.coxph(d["t"].fillna(1).values, d[outcome].fillna(0).values, X, design=des.domain(m), names=cols,
                 strata=strata)
    tab = f.table("design", use_t=True)
    try:
        ph = ph_tests(f)
    except Exception:  # noqa: BLE001 - diagnostic only
        ph = {}
    rows = []
    for c in report:
        r = tab.iloc[cols.index(c)]
        rows.append({"term": c, "n_unweighted": int(m.sum()), "events_unweighted": int(d.loc[m, outcome].sum()),
                     "log_hr": r["coef"], "se_design": r["se"], "hr": r["exp_coef"], "lower": r["exp_lower"],
                     "upper": r["exp_upper"], "p": r["p"], "ph_p_term": ph.get(c, np.nan),
                     "ph_p_global": ph.get("GLOBAL", np.nan), "design_df": f.design_df})
    if f.warnings:
        rows[0]["warnings"] = "; ".join(f.warnings)
    return rows, f, m


def svy_lm(d, des, dom, y, xcols):
    m = dom & d[[y] + xcols].notna().all(axis=1).values
    X = np.column_stack([np.ones(len(d))] + [d[c].fillna(0).values for c in xcols])
    r = ls.glm_fit(d[y].fillna(0).values, X, design=des.domain(m), names=["const"] + xcols)
    return r, m


def lm_rows(r, m, terms, extra):
    tab = r.table(use_t=True).set_index("term")
    out = []
    for t in terms:
        x = tab.loc[t]
        out.append({**extra, "term": t, "n_unweighted": int(m.sum()), "coef": x["coef"], "se_design": x["se"],
                    "lower": x["lower"], "upper": x["upper"], "p": x["p"], "design_df": r.design_df})
    return out


def wald_f(w: dict, ddf) -> dict:
    """Design-based F version of a Wald chi2 (F = chi2 / q on q and design df), as in R survey's regTermTest default
    denominator (unadjusted). The chi2 p-value ignores the limited design df and is anti-conservative."""
    F = w["chi2"] / w["df"]
    return {"F": F, "design_df": ddf, "p_F": float(stats.f.sf(F, w["df"], ddf)) if ddf else np.nan}


def wald_rows(r, groups, extra):
    out = []
    for lab, cols in groups:
        idx = [r.names.index(c) for c in cols if c in r.names]
        if not idx:
            continue
        w = ls.wald_test(r.coef, r.vcov, idx=idx)
        out.append({**extra, "test": lab, "chi2": w["chi2"], "df": w["df"], "p": w["p"],
                    **wald_f(w, getattr(r, "design_df", None))})
    return out


def rc_lambda(d, des, dom, y, x, Z):
    """Regression-dilution ratio: coefficient of x when regressing the replicate y on x + Z (design-based)."""
    r, m = svy_lm(d, des, dom, y, [x] + Z)
    t = r.table(use_t=True).set_index("term").loc[x]
    return {"lambda": t["coef"], "se": t["se"], "lower": t["lower"], "upper": t["upper"], "n_unweighted": int(m.sum()),
            "design_df": r.design_df}


def corrected(log_hr, se, lam, lam_se, df):
    """log-HR / lambda with delta-method SE (lambda and log-HR uncertainty treated as independent)."""
    b = log_hr / lam
    s = np.sqrt(se ** 2 / lam ** 2 + log_hr ** 2 * lam_se ** 2 / lam ** 4)
    q = tq(df)
    return b, s, np.exp(b), np.exp(b - q * s), np.exp(b + q * s)


# =================================================================================================================
# Rao-Wu PSU bootstrap; weighted Harrell C; weighted one-way ICC
# =================================================================================================================
def raowu(des: ls.SurveyDesign, B: int, rng) -> np.ndarray:
    M = np.ones((B, des.n_psu))
    for h in range(des.n_strata):
        ps = np.flatnonzero(des.psu_stratum == h)
        nh = len(ps)
        if nh < 2:
            continue
        draws = rng.integers(0, nh, size=(B, nh - 1))
        cnt = np.zeros((B, nh))
        np.add.at(cnt, (np.repeat(np.arange(B), nh - 1), draws.ravel()), 1.0)
        M[:, ps] = cnt * nh / (nh - 1)
    return M


def rep_weights(w, keep, Mrep, psu_code):
    return np.vstack([w[keep][None, :], w[keep][None, :] * Mrep[:, psu_code[keep]]])


def cindex_reps(time, event, lp, W):
    """Survey-weighted Harrell C for each row of W (records already restricted to the domain).
    Comparable pairs: event_i = 1 and time_j > time_i; pair weight w_i w_j; concordant if lp_i > lp_j; lp ties 1/2."""
    _, r = np.unique(lp, return_inverse=True)
    K = int(r.max()) + 1
    order = np.argsort(-time, kind="mergesort")
    ts = time[order]
    evt = np.flatnonzero(event == 1)
    ev_sorted = evt[np.argsort(-time[evt], kind="mergesort")]
    te = time[ev_sorted]
    ut = np.unique(time[evt])[::-1]
    nb = np.searchsorted(-ts, -ut, side="left")           # records with time > tau
    st = np.searchsorted(-te, -ut, side="left")
    en = np.searchsorted(-te, -ut, side="right")
    out = np.empty(W.shape[0])
    for b in range(W.shape[0]):
        w = W[b]
        H = np.zeros(K)
        num = den = 0.0
        prev = 0
        for i in range(len(ut)):
            if nb[i] > prev:
                idx = order[prev:nb[i]]
                np.add.at(H, r[idx], w[idx])
                prev = nb[i]
            if prev == 0:
                continue
            ei = ev_sorted[st[i]:en[i]]
            CW = np.cumsum(H)
            ri = r[ei]
            num += float(np.sum(w[ei] * (CW[ri] - 0.5 * H[ri])))
            den += float(np.sum(w[ei])) * CW[-1]
        out[b] = num / den if den > 0 else np.nan
    return out


def cindex_brute(time, event, lp, w):
    num = den = 0.0
    for i in np.flatnonzero(event == 1):
        j = time > time[i]
        den += w[i] * w[j].sum()
        num += w[i] * np.sum(w[j] * ((lp[i] > lp[j]) + 0.5 * (lp[i] == lp[j])))
    return num / den


def icc_reps(a, b, W):
    """One-way ICC for paired readings (systematic a-b difference counts as error), Pearson r, mean difference."""
    S = W.sum(axis=1)
    ma, mb = W @ a / S, W @ b / S
    va = (W * (a[None, :] - ma[:, None]) ** 2).sum(axis=1) / S
    vb = (W * (b[None, :] - mb[:, None]) ** 2).sum(axis=1) / S
    cab = (W * (a[None, :] - ma[:, None]) * (b[None, :] - mb[:, None])).sum(axis=1) / S
    msd = (W * (a - b)[None, :] ** 2).sum(axis=1) / S / 2
    vtot = (va + vb) / 2 + (ma - mb) ** 2 / 4
    return 1 - msd / vtot, cab / np.sqrt(va * vb), ma - mb


def icc_table_row(d, des, dom, a, b, Mrep, label):
    keep = dom & d[[a, b]].notna().all(axis=1).values & (des.weights > 0)
    W = rep_weights(des.weights, keep, Mrep, des.psu_code)
    icc, r, md = icc_reps(d.loc[keep, a].values.astype(float), d.loc[keep, b].values.astype(float), W)
    q = tq(des.df)
    out = {"comparison": label, "a": a, "b": b, "n_unweighted": int(keep.sum())}
    for nm, v in (("icc_oneway", icc), ("pearson_r", r), ("mean_diff_a_minus_b", md)):
        se = float(np.std(v[1:], ddof=1))
        out.update({nm: v[0], nm + "_lower": v[0] - q * se, nm + "_upper": v[0] + q * se})
    return out


def cindex_block(d, des, dom, models, contrasts, Mrep, label, strata=None):
    allc = sorted({c for cols in models.values() for c in cols})
    rows, crow = [], []
    res = {}
    for oc, olab in OUTCOMES:
        m = dom & d[allc + ["t", oc]].notna().all(axis=1).values & (des.weights > 0)
        W = rep_weights(des.weights, m, Mrep, des.psu_code)
        tt, ee = d.loc[m, "t"].values.astype(float), d.loc[m, oc].values.astype(float)
        q = tq(des.df)
        for name, cols in models.items():
            f = ls.coxph(d["t"].fillna(1).values, d[oc].fillna(0).values, d[cols].fillna(0).values,
                         design=des.domain(m), names=cols, strata=strata)
            lp = d.loc[m, cols].fillna(0).values @ f.coef
            cc = cindex_reps(tt, ee, lp, W)
            res[(oc, name)] = cc
            se = float(np.std(cc[1:], ddof=1))
            rows.append({"block": label, "outcome": olab, "model": name, "n_unweighted": int(m.sum()),
                         "events_unweighted": int(ee.sum()), "c_index": cc[0], "se_boot": se,
                         "lower": cc[0] - q * se, "upper": cc[0] + q * se})
        for a, b in contrasts:
            dd = res[(oc, a)] - res[(oc, b)]
            se = float(np.std(dd[1:], ddof=1))
            crow.append({"block": label, "outcome": olab, "contrast": f"{a} minus {b}", "delta_c": dd[0],
                         "se_boot": se, "lower": dd[0] - q * se, "upper": dd[0] + q * se,
                         "p": 2 * stats.t.sf(abs(dd[0] / se), des.df) if se > 0 else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(crow), res


# =================================================================================================================
# Loading (real) - continuous NHANES 1999-2016
# =================================================================================================================
def cont_files(k, sfx):
    lab = {1: ("LAB13", "LAB13", "LAB10", "LAB18"), 2: ("L13_B", "L13_B", "L10_B", "L40_B"),
           3: ("L13_C", "L13_C", "L10_C", "L40_C")}
    tc, hdl, ghb, bio = lab[k] if k <= 3 else (f"TCHOL{sfx}", f"HDL{sfx}", f"GHB{sfx}", f"BIOPRO{sfx}")
    f = {"DEMO": f"DEMO{sfx}", "BPX": f"BPX{sfx}", "BPQ": f"BPQ{sfx}", "SMQ": f"SMQ{sfx}", "DIQ": f"DIQ{sfx}",
         "MCQ": f"MCQ{sfx}", "BMX": f"BMX{sfx}", "TC": tc, "HDL": hdl, "GHB": ghb, "BIO": bio}
    if k <= 3:
        f["LEX"] = {1: "LEXABPI", 2: "LEXAB_B", 3: "LEXAB_C"}[k]
    f["SESS"] = f"PH{sfx}" if k <= 3 else f"FASTQX{sfx}"
    return f


def session_requests() -> pd.DataFrame:
    rows = []
    for k, cyc, sfx, y in CYCLES:
        stem = cont_files(k, sfx)["SESS"]
        p = NH / cyc / f"{stem}.xpt"
        rows.append({"cycle": cyc, "file": f"{stem}.xpt (+ {stem}.htm)",
                     "url_xpt": SESSION_URL.format(y=y, stem=stem, ext="xpt"),
                     "url_doc": SESSION_URL.format(y=y, stem=stem, ext="htm"),
                     "local_target": f"external/nhanes/{cyc}/{stem}.xpt", "on_disk": p.exists(),
                     "variable": "PHDSESN (session: 0 morning, 1 afternoon, 2 evening; verify in the codebook)"})
    return pd.DataFrame(rows)


def mort_file(cyc):
    return MORT_DIR / f"NHANES_{cyc.replace('-', '_')}_MORT_2019_PUBLIC.dat"


def read_mort(p: Path) -> pd.DataFrame:
    m = L.read_fixed_width(p, MORT_LAYOUT, list(MORT_LAYOUT))
    for v in MORT_LAYOUT:
        m[v] = pd.to_numeric(m[v].replace(".", np.nan), errors="coerce")
    m["SEQN"] = m["SEQN"].astype(np.int64)
    assert m["SEQN"].is_unique
    _register(p, len(m))
    return m


def load_cont() -> tuple[pd.DataFrame, list[str]]:
    parts, sess_cycles = [], []
    for k, cyc, sfx, _ in CYCLES:
        f = cont_files(k, sfx)
        dem = read_xpt(NH / cyc / f"{f['DEMO']}.xpt")
        df = dem.loc[dem["RIDSTATR"] == 2, ["SEQN"] + [v for v in DEMO_V if v in dem.columns]].copy()
        for key, stem in f.items():
            if key in ("DEMO", "SESS"):
                continue
            x = read_xpt(NH / cyc / f"{stem}.xpt")
            if key in ALT_V:
                new, alts = ALT_V[key]
                src = [a for a in alts if a in x.columns]
                assert len(src) == 1, (stem, src)
                x = x[["SEQN", src[0]]].rename(columns={src[0]: new})
            else:
                ren = {a: b for a, b in RENAME_V.get(key, {}).items() if a in x.columns and b not in x.columns}
                x = x.rename(columns=ren)
                keep = [v for v in FILE_V[key] if v in x.columns]
                miss = [v for v in FILE_V[key] if v not in x.columns and v not in OPTIONAL_V]
                assert not miss, (stem, miss)
                x = x[["SEQN"] + keep]
            df = df.merge(x, on="SEQN", how="left")
        sp = NH / cyc / f"{f['SESS']}.xpt"
        if sp.exists():
            s = read_xpt(sp)
            if "PHDSESN" in s.columns:
                bad = set(s["PHDSESN"].dropna().unique()) - {0.0, 1.0, 2.0}
                if bad:   # coding assumed 0 morning / 1 afternoon / 2 evening; stop rather than mis-assign sessions
                    raise RuntimeError(f"{sp.name}: PHDSESN has codes {sorted(bad)}; check the codebook and the mapping")
                df = df.merge(s[["SEQN", "PHDSESN"]], on="SEQN", how="left")
                sess_cycles.append(cyc)
        df = df.merge(read_mort(mort_file(cyc)), on="SEQN", how="left")
        parts.append(df)
    raw = pd.concat(parts, ignore_index=True, sort=False)
    for v in ["PHDSESN", "WTMEC4YR"] + FILE_V["LEX"]:
        if v not in raw.columns:
            raw[v] = np.nan
    return raw, sess_cycles


def clock_hours(s: pd.Series) -> pd.Series:
    """'HH:MM' -> decimal hours; anything else (88888 fill, blank) -> NaN."""
    p = s.astype("object").str.extract(r"^(\d{1,2}):(\d{2})$")
    h = pd.to_numeric(p[0], errors="coerce") + pd.to_numeric(p[1], errors="coerce") / 60
    return h.where(h.between(0, 24))


def load_nh3() -> pd.DataFrame:
    def rd(dat, sas, wanted):
        pos, _ = L.read_sas_layout(sas)
        miss = [v for v in wanted if v not in pos]
        assert not miss, (sas, miss)
        r = L.read_fixed_width(dat, pos, wanted)
        _register(dat)
        _register(sas)
        out = pd.DataFrame({v: (clock_hours(r[v]) if v in N3_CLOCK else L.fill_to_nan(r[v], N3_FILL[v])
                                if v in N3_FILL else L.to_num(r[v])) for v in wanted})
        out["SEQN"] = out["SEQN"].astype(np.int64)
        assert out["SEQN"].is_unique
        return out
    a = rd(N3 / "1a" / "adult.dat", N3 / "1a" / "adult.sas", N3_ADULT)
    x = rd(N3 / "1a" / "exam.dat", N3 / "1a" / "exam.sas", N3_EXAM)
    lb = rd(N3 / "1a" / "lab.dat", N3 / "1a" / "lab.sas", N3_LAB)
    mo = read_mort(MORT_DIR / "NHANES_III_MORT_2019_PUBLIC.dat")
    return a.merge(x, on="SEQN", how="left").merge(lb, on="SEQN", how="left").merge(mo, on="SEQN", how="left")


# =================================================================================================================
# Simulation (planted answers; same raw variable names)
# =================================================================================================================
SIM = {"sd_u": 15.0, "sd_o": 6.0, "sd_e": 4.0, "beta_u10": 0.20, "winter_sbp": 2.0, "excess_mean": 3.0,
       "sess_eff": {0: 0.0, 1: -1.0, 2: -2.0}, "cyc_off": [1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -1.0, 0.0],
       "abi_winter_factor": 0.98, "beta_abi01": 0.15,
       "n3_sess_eff": {1: 0.0, 2: -1.5, 3: -3.0}, "n3_month_cos": 2.5}


def simulate_cont(rng) -> tuple[pd.DataFrame, dict]:
    frames, truth_parts = [], []
    seq = 0
    for k, cyc, sfx, yr in CYCLES:
        n = 3000
        nstr = 13 if k == 1 else 15
        s0 = 1 if k == 1 else 14 + 15 * (k - 2)
        st = s0 + rng.integers(0, nstr, n)
        psu = rng.integers(1, 3, n).astype(float)
        if k == 1:
            psu[(st == 5) & (rng.random(n) < 0.3)] = 3
        w2 = np.exp(rng.normal(9.0, 0.7, n))
        age = np.clip(rng.normal(48, 17, n), 20, 85).round()
        male = rng.binomial(1, 0.48, n)
        race = rng.choice([1, 2, 3, 4, 5], n, p=[0.17, 0.07, 0.5, 0.2, 0.06])
        winter = rng.binomial(1, 0.5, n)
        sess = rng.choice([0, 1, 2], n, p=[0.45, 0.40, 0.15])
        U = 118 + 0.5 * (age - 48) + 3 * male + rng.normal(0, SIM["sd_u"], n)
        O = rng.normal(0, SIM["sd_o"], n)
        acq = SIM["winter_sbp"] * winter + SIM["cyc_off"][k - 1] + np.vectorize(SIM["sess_eff"].get)(sess)
        A = rng.normal(SIM["excess_mean"], 3.0, n)
        T_ = U + O + acq
        R = np.column_stack([even_round(T_ + rng.normal(0, SIM["sd_e"], n) + (A if j == 0 else 0)) for j in range(4)])
        R[rng.random(n) > 0.2, 3] = np.nan
        R[rng.random(n) < 0.02, 0] = np.nan
        R[rng.random(n) < 0.03, 2] = np.nan
        Dd = even_round(75 + 0.4 * (R - 120) + rng.normal(0, 6, R.shape))
        Dd[np.isnan(R)] = np.nan
        Dd[rng.random(R.shape) < 0.01] = 0
        preg = np.where((male == 0) & (age < 45), np.where(rng.random(n) < 0.03, 1.0, 2.0), np.nan)
        smk020 = np.where(rng.random(n) < 0.45, 1.0, 2.0)
        smk040 = np.where(smk020 == 1, rng.choice([1.0, 2.0, 3.0], n, p=[0.4, 0.1, 0.5]), np.nan)
        cur = (smk020 == 1) & np.isin(smk040, [1, 2])
        dm = rng.random(n) < 0.12
        ghb = np.round(np.where(dm, rng.normal(7.4, 1.2, n), rng.normal(5.4, 0.4, n)), 1)
        diq010 = np.where(dm & (rng.random(n) < 0.7), 1.0, np.where(rng.random(n) < 0.03, 3.0, 2.0))
        # ABI (cycles 1-3, age >= 40)
        abi_tr = 1.12 - 0.003 * (age - 40) + rng.normal(0, 0.07, n)
        low = rng.random(n) < 0.06
        abi_tr = np.where(low, abi_tr - rng.uniform(0.25, 0.5, n), abi_tr)
        abi_l_tr = np.clip(abi_tr + rng.normal(0, 0.05, n), 0.3, 1.6)
        abi_r_tr = np.clip(abi_tr + rng.normal(0, 0.05, n), 0.3, 1.6)
        abi_min_tr = np.minimum(abi_l_tr, abi_r_tr)
        has_abi = (k <= 3) & (age >= 40) & (rng.random(n) < 0.88)
        two = age < 60
        brach = T_ - 3
        wf = np.where(winter == 1, SIM["abi_winter_factor"], 1.0)
        b1 = np.round(brach + rng.normal(0, 4, n))
        b2 = np.where(two, np.round(brach + rng.normal(0, 4, n)), np.nan)
        rr1 = np.round(brach * abi_r_tr * wf + rng.normal(0, 5, n))
        rr2 = np.where(two, np.round(brach * abi_r_tr * wf + rng.normal(0, 5, n)), np.nan)
        ll1 = np.round(brach * abi_l_tr * wf + rng.normal(0, 5, n))
        ll2 = np.where(two, np.round(brach * abi_l_tr * wf + rng.normal(0, 5, n)), np.nan)
        bm = np.nanmean(np.column_stack([b1, b2]), axis=1)
        rm = np.nanmean(np.column_stack([rr1, rr2]), axis=1)
        lm = np.nanmean(np.column_stack([ll1, ll2]), axis=1)
        armc = np.round(np.clip(rng.normal(32, 5, n), 18, 55), 1)
        arm_cuff = np.select([armc < 22, armc < 30, armc < 38], [1.0, 3.0, 4.0], 5.0)
        ank_cuff = np.where((k == 1) & (rng.random(n) < 0.7), 2.0, np.minimum(arm_cuff, 4.0))
        lex = {"LEXBRP1": b1, "LEXBRP2": b2, "LEXBRPM": bm, "LEXRPTS1": rr1, "LEXRPTS2": rr2, "LEXRPTSM": rm,
               "LEXLPTS1": ll1, "LEXLPTS2": ll2, "LEXLPTSM": lm, "LEXRABPI": np.round(rm / bm, 2),
               "LEXLABPI": np.round(lm / bm, 2), "LEDARMCF": arm_cuff, "LEDANKCF": ank_cuff}
        lex = {c: np.where(has_abi, v, np.nan) for c, v in lex.items()}
        lp = (0.085 * (age - 50) + 0.35 * male + 0.45 * cur + 0.4 * dm + SIM["beta_u10"] * (U - 125) / 10
              + np.where(age >= 40, SIM["beta_abi01"] * np.maximum(0, 1.1 - abi_min_tr) / 0.1, 0))
        Tm = -np.log(rng.random(n)) / (0.004 * np.exp(lp))
        cens = 2020.0 - (yr + 0.5 + rng.uniform(0, 2, n))
        death = (Tm <= cens).astype(float)
        months = np.floor(np.minimum(Tm, cens) * 12)
        elig = np.where(rng.random(n) < 0.995, 1.0, 2.0)
        ucod = np.where(death == 1, np.where(rng.random(n) < 0.3, rng.choice([1, 5], n), rng.choice([2, 3, 4, 10], n)),
                        np.nan)
        df = pd.DataFrame({
            "SEQN": seq + np.arange(1, n + 1), "SDDSRVYR": float(k), "RIDSTATR": 2.0, "RIAGENDR": np.where(male == 1, 1.0, 2.0),
            "RIDAGEYR": age, "RIDRETH1": race.astype(float),
            "RIDEXMON": np.where(rng.random(n) < 0.01, np.nan, np.where(winter == 1, 1.0, 2.0)), "RIDEXPRG": preg,
            "WTMEC2YR": w2, "WTMEC4YR": w2 / 2 if k <= 2 else np.nan, "SDMVPSU": psu, "SDMVSTRA": st.astype(float),
            "BPXSY1": R[:, 0], "BPXSY2": R[:, 1], "BPXSY3": R[:, 2], "BPXSY4": R[:, 3],
            "BPXDI1": Dd[:, 0], "BPXDI2": Dd[:, 1], "BPXDI3": Dd[:, 2], "BPXDI4": Dd[:, 3],
            "BPXPLS": np.round(72 + rng.normal(0, 11, n) + 1.5 * winter),
            "BPQ020": np.where(rng.random(n) < 0.3, 1.0, 2.0), "BPQ040A": np.nan, "BPQ050A": np.nan,
            "SMQ020": smk020, "SMQ040": smk040, "DIQ010": diq010,
            "DIQ050": np.where(diq010 == 1, np.where(rng.random(n) < 0.3, 1.0, 2.0), np.nan),
            "DIQ070": np.nan if k in (4, 5) else np.where(diq010 == 1, np.where(rng.random(n) < 0.6, 1.0, 2.0), np.nan),
            **{f"MCQ160{c}": np.where(rng.random(n) < 0.04, 1.0, 2.0) for c in "BCDEF"},
            "BMXBMI": np.round(rng.normal(28.7, 6.2, n), 1), "BMXARMC": armc, "LBXTC": np.round(rng.normal(198, 41, n)),
            "HDL_h": np.round(rng.normal(52, 15, n)), "LBXGH": ghb, "SCR_h": np.round(np.exp(rng.normal(-0.15, 0.25, n)), 2),
            **lex, "ELIGSTAT": elig, "MORTSTAT": np.where(elig == 1, death, np.nan),
            "UCOD_LEADING": np.where(elig == 1, ucod, np.nan), "PERMTH_EXM": np.where(elig == 1, months, np.nan),
            "PHDSESN": sess.astype(float)})
        df.loc[df["BPQ020"] == 1, "BPQ040A"] = np.where(rng.random(int((df["BPQ020"] == 1).sum())) < 0.8, 1.0, 2.0)
        mm = df["BPQ040A"] == 1
        df.loc[mm, "BPQ050A"] = np.where(rng.random(int(mm.sum())) < 0.8, 1.0, 2.0)
        for c in ("LBXTC", "HDL_h", "SCR_h", "BMXBMI"):
            df.loc[rng.choice(n, 120, replace=False), c] = np.nan
        frames.append(df)
        truth_parts.append(pd.DataFrame({"acq": acq, "w": w2 / 9, "age": age, "preg": preg}))
        seq += n
    raw = pd.concat(frames, ignore_index=True)
    tp = pd.concat(truth_parts, ignore_index=True)
    ad = (tp["age"] >= 20) & (tp["preg"] != 1)
    var_acq = float(np.cov(tp.loc[ad, "acq"], aweights=tp.loc[ad, "w"]))
    V = SIM["sd_u"] ** 2 + SIM["sd_o"] ** 2 + var_acq
    lam1 = V / (V + SIM["sd_e"] ** 2 + 1 / 3)
    truth = {"lambda1_within": lam1, "beta_visit_true10": SIM["beta_u10"] * SIM["sd_u"] ** 2 / V,
             "winter_sbp": SIM["winter_sbp"], "excess_mean": SIM["excess_mean"],
             "sess_eve": SIM["sess_eff"][2], "abi_winter": None}
    return raw, truth


def simulate_nh3(rng) -> tuple[pd.DataFrame, dict]:
    n = 12000
    st = rng.integers(1, 50, n)
    psu = rng.integers(1, 3, n)
    w = np.exp(rng.normal(8.0, 0.9, n))
    age = np.clip(rng.normal(47, 18, n), 20, 90).round()
    male = rng.binomial(1, 0.47, n)
    race = rng.choice([1, 2, 3, 4], n, p=[0.45, 0.27, 0.25, 0.03])
    phase = 1 + (rng.random(n) < 0.5)
    # random household assignment: morning ('standard') vs afternoon/evening ('modified'); 4 % examined off-assignment
    arm = (rng.random(n) < 0.5).astype(int)
    p_arm = {0: [0.96, 0.025, 0.015], 1: [0.04, 0.60, 0.36]}
    sess = np.where(arm == 1, rng.choice([1, 2, 3], n, p=p_arm[1]), rng.choice([1, 2, 3], n, p=p_arm[0]))
    w_sd = np.where((arm == 0) & (sess == 1), w * 2, 0.0)
    w_md = np.where((arm == 1) & (sess >= 2), w * 2, 0.0)
    fast = np.round(np.clip(np.where(sess == 1, rng.normal(13, 2, n), rng.normal(7, 2.5, n)), 0, 36), 2)
    vclock = np.round(np.vectorize({1: 9.3, 2: 14.2, 3: 18.2}.get)(sess) + rng.normal(0, 0.8, n), 2)
    month = rng.integers(1, 13, n)
    dow = rng.integers(1, 8, n)
    region = rng.integers(1, 5, n)
    hsm = age * 12 + rng.integers(0, 12, n)
    interval = rng.choice([0, 1, 2, 3], n, p=[0.35, 0.40, 0.15, 0.10])
    U = 120 + 0.55 * (age - 47) + 3 * male + rng.normal(0, SIM["sd_u"], n)
    acq = np.vectorize(SIM["n3_sess_eff"].get)(sess) + SIM["n3_month_cos"] * np.cos(2 * np.pi * (month - 1) / 12)
    Om, Oh = rng.normal(0, SIM["sd_o"], n), rng.normal(0, SIM["sd_o"], n)
    Am, Ah = rng.normal(3.0, 3.0, n), rng.normal(2.0, 3.0, n)
    mec = np.column_stack([even_round(U + Om + acq + rng.normal(0, SIM["sd_e"], n) + (Am if j == 0 else 0))
                           for j in range(3)])
    hh = np.column_stack([even_round(U + Oh + rng.normal(0, SIM["sd_e"], n) + (Ah if j == 0 else 0))
                          for j in range(3)])
    hh[rng.random(n) < 0.03, 2] = np.nan
    mec[rng.random(n) < 0.02, 1] = np.nan
    har1 = np.where(rng.random(n) < 0.5, 1.0, 2.0)
    har3 = np.where(har1 == 1, np.where(rng.random(n) < 0.55, 1.0, 2.0), np.nan)
    cur = (har1 == 1) & (har3 == 1)
    dm = rng.random(n) < 0.1
    had1 = np.where(dm & (rng.random(n) < 0.7), 1.0, 2.0)
    lp = 0.085 * (age - 50) + 0.35 * male + 0.45 * cur + 0.4 * dm + SIM["beta_u10"] * (U - 125) / 10
    Tm = -np.log(rng.random(n)) / (0.004 * np.exp(lp))
    cens = 2019.99 - (1988.8 + 3 * (phase - 1) + rng.uniform(0, 3, n))
    death = (Tm <= cens).astype(float)
    hae2 = np.where(rng.random(n) < 0.25, 1.0, 2.0)
    hae4a = np.where(hae2 == 1, np.where(rng.random(n) < 0.8, 1.0, 2.0), np.nan)
    raw = pd.DataFrame({
        "SEQN": np.arange(1, n + 1), "DMPSTAT": 2.0, "HSSEX": np.where(male == 1, 1.0, 2.0), "HSAGEIR": age,
        "DMARETHN": race.astype(float), "SDPPHASE": phase.astype(float), "SDPPSU6": psu.astype(float),
        "SDPSTRA6": st.astype(float), "WTPFEX6": w, "HSAITMOR": hsm.astype(float), "MXPSESSR": sess.astype(float),
        "MXPTIMO": month.astype(float), "MXPTIDW": dow.astype(float), "MXPAXTMR": (hsm + interval).astype(float),
        "DMPCREGN": region.astype(float), "HAZA8AK1": hh[:, 0], "HAZA8BK1": hh[:, 1], "HAZA8CK1": hh[:, 2],
        "HAZA8DK1": np.nan, "HAC1C": np.where(rng.random(n) < 0.03, 1.0, 2.0),
        "HAC1D": np.where(rng.random(n) < 0.03, 1.0, 2.0), "HAD1": had1, "HAD3": np.nan, "HAD4": np.nan,
        "HAD6": np.where(had1 == 1, np.where(rng.random(n) < 0.3, 1.0, 2.0), np.nan),
        "HAD10": np.where(had1 == 1, np.where(rng.random(n) < 0.6, 1.0, 2.0), np.nan), "HAE2": hae2, "HAE4A": hae4a,
        "HAE5A": np.where(hae4a == 1, np.where(rng.random(n) < 0.8, 1.0, 2.0), np.nan),
        "HAF10": np.where(rng.random(n) < 0.04, 1.0, 2.0), "HAR1": har1, "HAR3": har3,
        "PEP6G1": mec[:, 0], "PEP6H1": mec[:, 1], "PEP6I1": mec[:, 2],
        "PEP6DR": np.round(72 + rng.normal(0, 11, n) + 2.0 * (sess == 3)), "BMPBMI": np.round(rng.normal(27, 5.5, n), 1),
        "TCP": np.round(rng.normal(205, 43, n)), "HDP": np.round(rng.normal(50, 15, n)),
        "GHP": np.round(np.where(dm, rng.normal(7.5, 1.3, n), rng.normal(5.3, 0.4, n)), 1),
        "CEP": np.round(np.exp(rng.normal(0.1, 0.2, n)), 1), "ELIGSTAT": 1.0, "MORTSTAT": death,
        "UCOD_LEADING": np.where(death == 1, np.where(rng.random(n) < 0.35, rng.choice([1, 5], n), 2), np.nan),
        "PERMTH_EXM": np.floor(np.minimum(Tm, cens) * 12), "WTPFSD6": w_sd, "WTPFMD6": w_md, "PHPFAST": fast,
        "PHPBEST": vclock})
    for c in ("TCP", "HDP", "CEP", "BMPBMI"):
        raw.loc[rng.choice(n, 150, replace=False), c] = np.nan
    mec_mean = np.nanmean(mec, axis=1)
    ok = ~np.isnan(mec).any(axis=1)
    err_var = float(np.cov((mec_mean - U)[ok], aweights=w[ok]))
    pa = np.array(p_arm[1][1:]) / sum(p_arm[1][1:])
    truth = {"lambda_between_mec": SIM["sd_u"] ** 2 / (SIM["sd_u"] ** 2 + err_var), "beta_u10": SIM["beta_u10"],
             "sess_eve": SIM["n3_sess_eff"][3], "month_cos": SIM["n3_month_cos"],
             "arm_pm": float(pa @ np.array([SIM["n3_sess_eff"][2], SIM["n3_sess_eff"][3]]))}
    return raw, truth


def wiring_check(raw_c: pd.DataFrame, raw_3: pd.DataFrame) -> pd.DataFrame:
    """Every raw variable used exists in the real file headers / SAS layouts (schema only) and in the simulated frames."""
    rows = []
    for k, cyc, sfx, _ in CYCLES:
        f = cont_files(k, sfx)
        for key, stem in f.items():
            p = NH / cyc / f"{stem}.xpt"
            if key == "SESS":
                rows.append({"source": f"{cyc}/{stem}.xpt", "variable": "PHDSESN", "required": False,
                             "in_real_file": p.exists() and "PHDSESN" in xpt_columns(p),
                             "in_simulated": "PHDSESN" in raw_c.columns})
                continue
            cols = set(xpt_columns(p))
            if key in ALT_V:
                new, alts = ALT_V[key]
                rows.append({"source": f"{cyc}/{stem}.xpt", "variable": f"{new} <- {'/'.join(alts)}", "required": True,
                             "in_real_file": any(a in cols for a in alts), "in_simulated": new in raw_c.columns})
                continue
            alias = {b: a for a, b in RENAME_V.get(key, {}).items()}
            for v in (DEMO_V if key == "DEMO" else FILE_V[key]):
                rows.append({"source": f"{cyc}/{stem}.xpt", "variable": v + (f" (or {alias[v]})" if v in alias else ""),
                             "required": v not in OPTIONAL_V, "in_real_file": v in cols or alias.get(v) in cols,
                             "in_simulated": v in raw_c.columns})
        mp = mort_file(cyc)
        rows.append({"source": str(mp.relative_to(NH)), "variable": ",".join(MORT_LAYOUT), "required": True,
                     "in_real_file": mp.exists(), "in_simulated": all(v in raw_c.columns for v in MORT_LAYOUT)})
    for dat, sas, wanted in (("adult.dat", "adult.sas", N3_ADULT), ("exam.dat", "exam.sas", N3_EXAM),
                             ("lab.dat", "lab.sas", N3_LAB)):
        pos, _ = L.read_sas_layout(N3 / "1a" / sas)
        for v in wanted:
            rows.append({"source": f"nhanes_iii/1a/{sas}", "variable": v, "required": True, "in_real_file": v in pos,
                         "in_simulated": v in raw_3.columns})
    df = pd.DataFrame(rows)
    df["ok"] = (~df["required"] | df["in_real_file"]) & df["in_simulated"]
    bad = df[~df["ok"]]
    if len(bad):
        raise RuntimeError(f"wiring check failed: {bad[['source', 'variable']].values.tolist()[:10]}")
    return df


# =================================================================================================================
# Preparation
# =================================================================================================================
def prepare_cont(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = raw.copy()
    d["k"] = d["SDDSRVYR"].astype(int)
    d["cycle"] = d["k"].map(CYC_LAB)
    w = np.where(d["k"] <= 2, d["WTMEC4YR"] * 2 / 9, d["WTMEC2YR"] / 9)
    d["w18"] = np.nan_to_num(w.astype(float), nan=0.0)
    d["age_y"] = d["RIDAGEYR"].astype(float)
    d["age10"] = d["age_y"] / 10
    d["male"] = (d["RIAGENDR"] == 1).astype(float)
    female = 1 - d["male"]
    for c, code in (("race_mexam", 1), ("race_othhisp", 2), ("race_black", 4), ("race_other", 5)):
        d[c] = (d["RIDRETH1"] == code).astype(float).where(d["RIDRETH1"].notna())
    d["winter"] = np.where(d["RIDEXMON"].isin([1, 2]), (d["RIDEXMON"] == 1) * 1.0, np.nan)
    for k in range(2, 10):
        d[f"cyc{k}"] = (d["k"] == k).astype(float)
    S = d[["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4"]].astype(float)
    D = d[["BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"]].astype(float).where(lambda x: x > 0)
    d["n_later"] = S.iloc[:, 1:].notna().sum(axis=1)
    d["sbp_first"] = S.iloc[:, 0]
    d["sbp_later"] = S.iloc[:, 1:].mean(axis=1).where(d["n_later"] >= 2)
    d["sbp_excess"] = d["sbp_first"] - d["sbp_later"]
    d["sbp_all"] = S.mean(axis=1).where(d["sbp_first"].notna() & (d["n_later"] >= 2))
    d["sbp_r1"], d["sbp_r2"], d["sbp_r3"] = S.iloc[:, 0], S.iloc[:, 1], S.iloc[:, 2]
    d["sbp_r23"] = (d["sbp_r2"] + d["sbp_r3"]) / 2
    d["sbp_r123"] = (d["sbp_r1"] + d["sbp_r2"] + d["sbp_r3"]) / 3
    nld = D.iloc[:, 1:].notna().sum(axis=1)
    d["dbp_first"] = D.iloc[:, 0]
    d["dbp_later"] = D.iloc[:, 1:].mean(axis=1).where(nld >= 2)
    d["dbp_excess"] = d["dbp_first"] - d["dbp_later"]
    d["pulse"] = d["BPXPLS"]
    for c in ("sbp_first", "sbp_later", "sbp_excess", "sbp_all", "sbp_r1", "sbp_r2", "sbp_r3", "sbp_r23", "sbp_r123",
              "dbp_first", "dbp_later", "dbp_excess"):
        d[c + "10"] = d[c] / 10
    smk = np.select([d["SMQ020"] == 2, (d["SMQ020"] == 1) & (d["SMQ040"] == 3),
                     (d["SMQ020"] == 1) & d["SMQ040"].isin([1, 2])], ["never", "former", "current"], default="")
    d["smk_former"] = np.where(smk == "", np.nan, (smk == "former") * 1.0)
    d["smk_current"] = np.where(smk == "", np.nan, (smk == "current") * 1.0)
    dm_yes = (d["DIQ010"] == 1) | (d["DIQ050"] == 1) | (d["DIQ070"] == 1) | (d["LBXGH"] >= 6.5)
    dm_no = d["DIQ010"].isin([2, 3]) & (d["LBXGH"] < 6.5)
    d["diabetes_any"] = np.where(dm_yes, 1.0, np.where(dm_no, 0.0, np.nan))
    d["bmi5"] = d["BMXBMI"] / 5
    d["tc10"], d["hdl10"] = d["LBXTC"] / 10, d["HDL_h"] / 10
    d["bp_med"] = np.where(d["BPQ050A"] == 1, 1.0, np.where((d["BPQ020"] == 2) | (d["BPQ040A"] == 2) |
                                                            (d["BPQ050A"] == 2), 0.0, np.nan))
    scr = np.where(d["k"] == 1, 1.013 * d["SCR_h"] + 0.147, np.where(d["k"] == 4, 0.978 * d["SCR_h"] - 0.016,
                                                                      d["SCR_h"]))
    d["egfr10"] = ckd_epi_2021(scr, d["age_y"].values, female.values) / 10
    d["hx_cvd_any"] = pd.concat([yn(d[f"MCQ160{c}"]) for c in "BCDEF"], axis=1).max(axis=1)
    elig = d["ELIGSTAT"] == 1
    d["mort_eligible"] = elig.astype(int)
    d["death_all"] = np.where(elig, d["MORTSTAT"], np.nan)
    d["death_cvd"] = np.where(elig, ((d["MORTSTAT"] == 1) & d["UCOD_LEADING"].isin([1, 5])) * 1.0, np.nan)
    d["t"] = np.where(elig, np.maximum(d["PERMTH_EXM"] / 12, 0.5 / 12), np.nan)
    # ABI (1999-2004)
    d["abi_r"], d["abi_l"] = d["LEXRABPI"], d["LEXLABPI"]
    d["abi_min"] = d[["abi_r", "abi_l"]].min(axis=1)
    d["abi_max"] = d[["abi_r", "abi_l"]].max(axis=1)
    d["abi_lt09"] = (d["abi_min"] < 0.9).astype(float).where(d["abi_min"].notna())
    d["abi_per01_lower"] = np.where(d["abi_max"] <= 1.4, -d["abi_min"] / 0.1, np.nan)
    d["brach_supine"] = d["LEXBRPM"]
    d["ankle_mean"] = d[["LEXRPTSM", "LEXLPTSM"]].mean(axis=1)
    for j in (1, 2):
        b = d[f"LEXBRP{j}"]
        r_, l_ = d[f"LEXRPTS{j}"] / b, d[f"LEXLPTS{j}"] / b
        d[f"abi_low_set{j}"] = np.fmin(r_, l_).where(r_.notna() & l_.notna())
        d[f"abi_high_set{j}"] = np.fmax(r_, l_).where(r_.notna() & l_.notna())
    d["abi_low_set1_per01_lower"] = np.where(d["abi_high_set1"] <= 1.4, -d["abi_low_set1"] / 0.1, np.nan)
    # ABI cuff size (cm; 8 'could not obtain' -> NaN) and arm circumference (cuff choice depends on it)
    for c, v in (("cuff_arm_cm", "LEDARMCF"), ("cuff_ank_cm", "LEDANKCF")):
        d[c] = d[v].map(CUFF_CM)
    ok_c = d["cuff_arm_cm"].notna() & d["cuff_ank_cm"].notna()
    for c, cond in (("cuffA_10", d["cuff_ank_cm"] == 10), ("cuffA_12", d["cuff_ank_cm"] == 12),
                    ("cuffA_oth", d["cuff_ank_cm"].isin([9, 18])), ("cuffB_12", d["cuff_arm_cm"] == 12),
                    ("cuffB_oth", d["cuff_arm_cm"].isin([9, 10, 18]))):
        d[c] = cond.astype(float).where(ok_c)                   # reference: 15 cm at both sites
    d["armc5"] = d["BMXARMC"] / 5
    # session (when present)
    ses = d["PHDSESN"]
    okses = ses.isin([0, 1, 2])
    d["sess_aft"] = np.where(okses, (ses == 1) * 1.0, np.nan)
    d["sess_eve"] = np.where(okses, (ses == 2) * 1.0, np.nan)
    # domains
    adult = (d["age_y"] >= 20) & (d["RIDEXPRG"] != 1) & (d["w18"] > 0)
    bp = adult & d["sbp_first"].notna() & d["sbp_later"].notna()
    mort = bp & elig & d[M2 + ["t"]].notna().all(axis=1)
    abi = (d["k"] <= 3) & (d["age_y"] >= 40) & (d["w18"] > 0) & d["abi_min"].notna()
    R = {"adult": adult.values, "bp": bp.values, "mort": mort.values, "abi": abi.values,
         "abi_mort": (abi & elig & d[M2A + ["t"]].notna().all(axis=1)).values}
    assert d.groupby("SDMVSTRA")["k"].nunique().max() == 1, "SDMVSTRA not unique across cycles"
    return d, R


def prepare_nh3(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = raw.copy()
    d["age_y"] = d["HSAGEIR"].astype(float)
    d["age10"] = d["age_y"] / 10
    d["male"] = (d["HSSEX"] == 1).astype(float)
    female = 1 - d["male"]
    for c, code in (("race_black", 2), ("race_mexam", 3), ("race_other", 4)):
        d[c] = (d["DMARETHN"] == code).astype(float).where(d["DMARETHN"].notna())
    d["w"] = np.where(d["DMPSTAT"] == 2, d["WTPFEX6"].fillna(0), 0.0)
    ses = d["MXPSESSR"]
    ok = ses.isin([1, 2, 3])
    d["sess_aft"] = np.where(ok, (ses == 2) * 1.0, np.nan)
    d["sess_eve"] = np.where(ok, (ses == 3) * 1.0, np.nan)
    d["session"] = ses.map({1: "morning", 2: "afternoon", 3: "evening"})
    mo = d["MXPTIMO"].where(d["MXPTIMO"].between(1, 12))
    d["m_cos"] = np.cos(2 * np.pi * (mo - 1) / 12)
    d["m_sin"] = np.sin(2 * np.pi * (mo - 1) / 12)
    d["month"] = mo
    d["weekend"] = np.where(d["MXPTIDW"].between(1, 7), d["MXPTIDW"].isin([1, 7]) * 1.0, np.nan)
    d["phase2"] = (d["SDPPHASE"] == 2).astype(float)
    # Random household assignment to the morning ('standard') or afternoon/evening ('modified') session (adult.htm
    # notes on WTPFSD6/WTPFMD6): positive WTPFSD6 = assigned to and examined in the morning, positive WTPFMD6 = assigned
    # to and examined in the afternoon/evening; persons examined off-assignment have zero in both (excluded).
    sd6, md6 = d["WTPFSD6"].fillna(0), d["WTPFMD6"].fillna(0)
    d["arm_pm"] = np.where(md6 > 0, 1.0, np.where(sd6 > 0, 0.0, np.nan))
    d["w_arm"] = np.where(d["DMPSTAT"] == 2, np.where(md6 > 0, md6, sd6), 0.0)
    d["fast_h"] = d["PHPFAST"].clip(upper=24)                    # hours since last food/drink (lab file)
    d["clock_h"] = d["PHPBEST"]                                 # time of venipuncture, decimal hours
    for c, code in (("reg_mw", 2), ("reg_s", 3), ("reg_w", 4)):
        d[c] = (d["DMPCREGN"] == code).astype(float).where(d["DMPCREGN"].notna())
    okm = (d["HSAITMOR"] < 1080) & (d["MXPAXTMR"] < 1080)
    d["interval_m"] = (d["MXPAXTMR"] - d["HSAITMOR"]).where(okm)
    Mx = d[["PEP6G1", "PEP6H1", "PEP6I1"]].astype(float)
    d["mec_r1"], d["mec_r2"], d["mec_r3"] = Mx.iloc[:, 0], Mx.iloc[:, 1], Mx.iloc[:, 2]
    d["mec_mean"] = Mx.mean(axis=1, skipna=False)
    d["mec_first"] = d["mec_r1"]
    d["mec_later"] = (d["mec_r2"] + d["mec_r3"]) / 2
    d["mec_excess"] = d["mec_first"] - d["mec_later"]
    Hh = d[["HAZA8AK1", "HAZA8BK1", "HAZA8CK1", "HAZA8DK1"]].astype(float)
    nh = Hh.notna().sum(axis=1)
    d["hh_r1"], d["hh_r2"], d["hh_r3"] = Hh.iloc[:, 0], Hh.iloc[:, 1], Hh.iloc[:, 2]
    d["hh_mean"] = Hh.mean(axis=1).where((nh >= 2) & d["hh_r1"].notna())
    d["two_occ"] = (d["mec_mean"] + d["hh_mean"]) / 2
    d["occ_dev"] = d["mec_mean"] - d["hh_mean"]
    d["pulse"] = d["PEP6DR"]
    for c in ("mec_mean", "mec_first", "mec_later", "mec_excess", "hh_mean", "two_occ", "occ_dev"):
        d[c + "10"] = d[c] / 10
    smk = np.select([d["HAR1"] == 2, (d["HAR1"] == 1) & (d["HAR3"] == 2), (d["HAR1"] == 1) & (d["HAR3"] == 1)],
                    ["never", "former", "current"], default="")
    d["smk_former"] = np.where(smk == "", np.nan, (smk == "former") * 1.0)
    d["smk_current"] = np.where(smk == "", np.nan, (smk == "current") * 1.0)
    preg_only = (d["HAD3"] == 1) & (d["HAD4"] == 2)
    dself = np.where(d["HAD1"] == 1, np.where(preg_only, 0.0, 1.0), np.where(d["HAD1"] == 2, 0.0, np.nan))
    d["diabetes_any"] = np.where((dself == 1) | (d["HAD6"] == 1) | (d["HAD10"] == 1) | (d["GHP"] >= 6.5), 1.0,
                                 np.where((dself == 0) & (d["GHP"] < 6.5), 0.0, np.nan))
    d["bmi5"] = d["BMPBMI"] / 5
    d["tc10"], d["hdl10"] = d["TCP"] / 10, d["HDP"] / 10
    d["bp_med"] = np.where(d["HAE2"] == 2, 0.0, np.where(d["HAE4A"] == 2, 0.0, yn(d["HAE5A"])))
    d["egfr10"] = ckd_epi_2021(0.960 * d["CEP"] - 0.184, d["age_y"].values, female.values) / 10
    d["hx_cvd_any"] = pd.concat([yn(d["HAF10"]), yn(d["HAC1C"]), yn(d["HAC1D"])], axis=1).max(axis=1)
    elig = d["ELIGSTAT"] == 1
    d["death_all"] = np.where(elig, d["MORTSTAT"], np.nan)
    d["death_cvd"] = np.where(elig, ((d["MORTSTAT"] == 1) & d["UCOD_LEADING"].isin([1, 5])) * 1.0, np.nan)
    d["t"] = np.where(elig, np.maximum(d["PERMTH_EXM"] / 12, 0.5 / 12), np.nan)
    base = (d["age_y"] >= 20) & (d["DMPSTAT"] == 2) & (d["w"] > 0) & d["mec_mean"].notna() & d["hh_mean"].notna()
    R = {"base": base.values, "mort": (base & elig & d[N2 + ["t"]].notna().all(axis=1)).values}
    return d, R


# =================================================================================================================
# Module A: continuous NHANES
# =================================================================================================================
def design_cont(d):
    return ls.SurveyDesign(d["w18"].values, d["SDMVSTRA"].values, d["SDMVPSU"].values, nest=True, lonely_psu="adjust")


def sample_table(d, R):
    rows = []
    for k, cyc, _, _ in CYCLES:
        m = (d["k"] == k).values
        rows.append({"cycle": cyc, "mec_examined": int(m.sum()), "adults_20plus": int((m & R["adult"]).sum()),
                     "bp_domain": int((m & R["bp"]).sum()), "mortality_domain": int((m & R["mort"]).sum()),
                     "deaths": int(d.loc[m & R["mort"], "death_all"].sum()),
                     "cvd_deaths": int(d.loc[m & R["mort"], "death_cvd"].sum()),
                     "median_followup_y": float(np.median(d.loc[m & R["mort"], "t"])) if (m & R["mort"]).any() else np.nan,
                     "abi_domain": int((m & R["abi"]).sum()), "abi_mortality_domain": int((m & R["abi_mort"]).sum()),
                     "session_available": int((m & R["bp"] & d["sess_aft"].notna().values).sum()),
                     "bp_domain_with_4th_reading": int((m & R["bp"] & d["BPXSY4"].notna().values).sum())})
    return pd.DataFrame(rows)


def q1(d, des, R, Mrep, rep, K):
    dom = R["mort"]
    w = des.weights
    sc = []
    for c in ("sbp_first", "sbp_later", "sbp_excess", "sbp_all", "dbp_first", "dbp_later", "dbp_excess"):
        mm = dom & d[c].notna().values
        sd = ls.weighted_sd(d.loc[mm, c].values, w[mm])
        d[c + "_sd"] = d[c] / sd
        sc.append({"variable": c, "weighted_mean": wmean_dom(d[c], w, mm), "weighted_sd": sd,
                   "n_unweighted": int(mm.sum())})
    sc = pd.DataFrame(sc)
    K["q1_scaling"] = sc
    em = ls.svy_mean(d["sbp_excess"].values, des, domain=R["bp"])
    pc = ls.svy_mean((d["sbp_excess"] > 0).astype(float).where(d["sbp_excess"].notna()).values, des,
                     domain=R["bp"], proportion=True)
    K["q1_excess_mean"] = em
    K["q1_excess_pos"] = pc
    rep.h("Q1. Acute (first reading) vs usual (later readings) blood pressure at the visit")
    rep.p(f"Within-visit acute excess (SBP reading 1 minus mean of later readings), survey-weighted mean in the BP "
          f"domain: {fmt_ci(em['estimate'], em['lower'], em['upper'], 2)} mmHg (n = {em['n']}); share with a positive "
          f"excess {fmt_ci(pc['estimate'], pc['lower'], pc['upper'], 3)}.")
    rep.table("q1_scaling", sc, "Exposure scaling in the mortality domain (survey-weighted mean and SD)", max_rows=None)
    rows = []
    sets = [("first reading alone", ["sbp_first_sd"]), ("later mean alone", ["sbp_later_sd"]),
            ("all-reading mean alone", ["sbp_all_sd"]), ("later mean + acute excess (mutual)", ["sbp_later_sd", "sbp_excess_sd"]),
            ("first + later mean (mutual)", ["sbp_first_sd", "sbp_later_sd"])]
    for oc, olab in OUTCOMES:
        for alab, covs in (("M1", M1), ("M2", M2)):
            for mlab, ex in sets:
                r, _, _ = svy_cox(d, des, dom, oc, ex + covs, ex)
                rows += [{"outcome": olab, "adjustment": alab, "model": mlab, "scale": "per SD", **x} for x in r]
        for mlab, ex in (("first reading alone", ["sbp_first10"]), ("later mean alone", ["sbp_later10"]),
                         ("later mean + acute excess (mutual)", ["sbp_later10", "sbp_excess10"])):
            r, _, _ = svy_cox(d, des, dom, oc, ex + M2, ex)
            rows += [{"outcome": olab, "adjustment": "M2", "model": mlab, "scale": "per 10 mmHg", **x} for x in r]
        domd = dom & d[["dbp_first", "dbp_later"]].notna().all(axis=1).values
        for mlab, ex in (("DBP first reading alone", ["dbp_first_sd"]), ("DBP later mean alone", ["dbp_later_sd"]),
                         ("DBP later mean + acute excess (mutual)", ["dbp_later_sd", "dbp_excess_sd"])):
            r, _, _ = svy_cox(d, des, domd, oc, ex + M2, ex)
            rows += [{"outcome": olab, "adjustment": "M2", "model": mlab, "scale": "per SD", **x} for x in r]
    cox = pd.DataFrame(rows)
    K["q1_cox"] = cox
    rep.table("q1_svy_cox", cox, "Survey-weighted Cox: first reading, later mean and acute excess (design-based SE, "
              "t(design df) intervals)", max_rows=None)
    # discrimination
    models = {"M2 (no BP)": M2, "M2 + first reading": M2 + ["sbp_first_sd"], "M2 + later mean": M2 + ["sbp_later_sd"],
              "M2 + all-reading mean": M2 + ["sbp_all_sd"],
              "M2 + later mean + excess": M2 + ["sbp_later_sd", "sbp_excess_sd"],
              "M1 + first reading": M1 + ["sbp_first_sd"], "M1 + later mean": M1 + ["sbp_later_sd"]}
    contr = [("M2 + later mean", "M2 + first reading"), ("M2 + later mean + excess", "M2 + later mean"),
             ("M2 + later mean", "M2 (no BP)"), ("M2 + all-reading mean", "M2 + later mean"),
             ("M1 + later mean", "M1 + first reading")]
    cdf, cdd, _ = cindex_block(d, des, dom, models, contr, Mrep, "Q1 SBP")
    K["q1_c"], K["q1_dc"] = cdf, cdd
    rep.table("q1_cindex", cdf, "Survey-weighted Harrell C (Rao-Wu PSU bootstrap SE; lp fixed at full-sample fit)",
              max_rows=None)
    rep.table("q1_cindex_diff", cdd, "C-index differences", max_rows=None)
    # misclassification
    domh = R["bp"] & d[["dbp_first", "dbp_later"]].notna().all(axis=1).values
    mrows, crow = [], []
    for thr, (s_t, d_t) in (("140/90", (140, 90)), ("130/80", (130, 80))):
        hf = ((d["sbp_first"] >= s_t) | (d["dbp_first"] >= d_t)).values
        hl = ((d["sbp_later"] >= s_t) | (d["dbp_later"] >= d_t)).values
        cats = {"both": hf & hl, "first reading only": hf & ~hl, "later mean only": ~hf & hl, "neither": ~hf & ~hl}
        for slab, sub in (("all adults", domh), ("untreated (bp_med = 0)", domh & (d["bp_med"] == 0).values)):
            for cl, ind in cats.items():
                e = ls.svy_mean(ind.astype(float), des, domain=sub, proportion=True)
                mrows.append({"threshold": thr, "subgroup": slab, "quantity": f"P({cl})", "estimate": e["estimate"],
                              "lower": e["lower"], "upper": e["upper"], "n_unweighted_cell": int((sub & ind).sum()),
                              "n_unweighted_domain": int(sub.sum())})
            for qlab, num, den in (("P(not confirmed by later mean | hypertensive by first reading)", ~hl, hf),
                                   ("P(hypertensive by first reading | not by later mean)", hf, ~hl)):
                e = ls.svy_mean(num.astype(float), des, domain=sub & den, proportion=True)
                mrows.append({"threshold": thr, "subgroup": slab, "quantity": qlab, "estimate": e["estimate"],
                              "lower": e["lower"], "upper": e["upper"], "n_unweighted_cell": int((sub & den & num).sum()),
                              "n_unweighted_domain": int((sub & den).sum())})
            e = ls.svy_mean(hf.astype(float) - hl.astype(float), des, domain=sub)
            mrows.append({"threshold": thr, "subgroup": slab, "quantity": "prevalence by first reading minus by later mean",
                          "estimate": e["estimate"], "lower": e["lower"], "upper": e["upper"],
                          "n_unweighted_cell": np.nan, "n_unweighted_domain": int(sub.sum())})
        d[f"hc_both_{thr}"] = cats["both"].astype(float)
        d[f"hc_first_{thr}"] = cats["first reading only"].astype(float)
        d[f"hc_later_{thr}"] = cats["later mean only"].astype(float)
        cols = [f"hc_first_{thr}", f"hc_later_{thr}", f"hc_both_{thr}"]
        for oc, olab in OUTCOMES:
            r, _, _ = svy_cox(d, des, dom & domh, oc, cols + M2, cols)
            crow += [{"threshold": thr, "outcome": olab, "adjustment": "M2", **x} for x in r]
    mis = pd.DataFrame(mrows)
    K["q1_mis"] = mis
    rep.table("q1_htn_misclassification", mis, "BP-defined hypertension by first reading vs later mean (survey-"
              "weighted proportions; logit CIs)", max_rows=None)
    disc = pd.DataFrame(crow)
    K["q1_disc"] = disc
    rep.table("q1_discordant_cox", disc, "Cox HRs of discordant classification (ref: neither)", max_rows=None)


def q2(d, des, R, Mrep, rep, K):
    rep.h("Q2. Acquisition conditions: 6-month exam period and survey cycle")
    rep.p("RIDEXMON: 1 = November-April ('winter'), 2 = May-October. By NHANES design the mobile examination centres "
          "operate in southern locations in winter and northern locations in summer (operating practice as described by "
          "NCHS; not stated in the documentation files on disk), so the period contrast mixes "
          "season with geography; survey cycle mixes acquisition/protocol with secular trends in BP and treatment.")
    domb = R["bp"] & ~np.isnan(d["winter"].values)
    rows, tests = [], []
    X = ["winter"] + CYC_D + M1
    for y in ("sbp_first", "sbp_later", "sbp_excess", "dbp_later", "pulse"):
        r, m = svy_lm(d, des, domb, y, X)
        rows += lm_rows(r, m, ["winter"] + CYC_D, {"domain": "BP (age >= 20, 1999-2016)", "outcome": y})
        tests += wald_rows(r, [("cycle (8 df)", CYC_D), ("winter", ["winter"])], {"domain": "BP", "outcome": y})
    doma = R["abi"] & ~np.isnan(d["winter"].values)
    Xa = ["winter", "cyc2", "cyc3"] + M1
    for y in ("abi_min", "abi_r", "abi_l", "brach_supine", "ankle_mean"):
        r, m = svy_lm(d, des, doma, y, Xa)
        rows += lm_rows(r, m, ["winter", "cyc2", "cyc3"], {"domain": "ABI (age >= 40, 1999-2004)", "outcome": y})
        tests += wald_rows(r, [("cycle (2 df)", ["cyc2", "cyc3"])], {"domain": "ABI", "outcome": y})
    acq = pd.DataFrame(rows)
    K["q2_acq"] = acq
    rep.table("q2_acquisition_effects", acq, "Design-based linear regressions on period and cycle (+ age, sex, "
              "race/ethnicity); coefficients in measurement units", max_rows=None)
    K["q2_tests"] = pd.DataFrame(tests)
    rep.table("q2_acquisition_tests", K["q2_tests"], "Wald tests (design-based; chi2 p and design-based F p)",
              max_rows=None)
    q2_cuff(d, des, doma, rep, K)
    cells = []
    for k, cyc, _, _ in CYCLES:
        for per, lab in ((1.0, "Nov-Apr"), (0.0, "May-Oct")):
            m = domb & (d["k"] == k).values & (d["winter"] == per).values
            for y in ("sbp_later", "sbp_excess"):
                e = ls.svy_mean(d[y].values, des, domain=m)
                cells.append({"cycle": cyc, "period": lab, "outcome": y, "n_unweighted": int(m.sum()),
                              "weighted_mean": e["estimate"], "lower": e["lower"], "upper": e["upper"]})
    K["q2_cells"] = pd.DataFrame(cells)
    rep.table("q2_cycle_period_means", K["q2_cells"], "Weighted means by cycle x period (unadjusted)", max_rows=None)
    # corrected exposures (remove period and cycle components, keep the mean)
    for y, dom_, acqc in (("sbp_later", domb, ["winter"] + CYC_D), ("sbp_first", domb, ["winter"] + CYC_D),
                          ("abi_min", doma, ["winter", "cyc2", "cyc3"])):
        r, m = svy_lm(d, des, dom_, y, acqc + M1)
        b = r.coef[1:1 + len(acqc)]
        A = d[acqc].values @ b
        Abar = float(np.average(A[m], weights=des.weights[m]))
        d[y + "_c"] = d[y] - (A - Abar)
    d["sbp_later_c10"] = d["sbp_later_c"] / 10
    d["sbp_first_c10"] = d["sbp_first_c"] / 10
    abi_max_c = d["abi_max"] + (d["abi_min_c"] - d["abi_min"])
    d["abi_c_per01_lower"] = np.where(abi_max_c <= 1.4, -d["abi_min_c"] / 0.1, np.nan)
    d["abi_c_lt09"] = (d["abi_min_c"] < 0.9).astype(float).where(d["abi_min_c"].notna())
    # HRs with and without acquisition adjustment
    hr = []
    domm = R["mort"] & ~np.isnan(d["winter"].values)
    strata = d["k"].values
    for oc, olab in OUTCOMES:
        for ex, ex_c in (("sbp_later10", "sbp_later_c10"), ("sbp_first10", "sbp_first_c10")):
            for alab, cols, rep_, stt in (("M2 raw", [ex] + M2, ex, None),
                                          ("M2 + winter, baseline stratified by cycle", [ex, "winter"] + M2, ex, strata),
                                          ("M2, acquisition-corrected exposure", [ex_c] + M2, ex_c, None)):
                r, _, _ = svy_cox(d, des, domm, oc, cols, [rep_], strata=stt)
                hr += [{"outcome": olab, "exposure": ex, "model": alab, **x} for x in r]
        doma_m = R["abi_mort"] & ~np.isnan(d["winter"].values)
        for ex, ex_c in (("abi_per01_lower", "abi_c_per01_lower"), ("abi_lt09", "abi_c_lt09")):
            for alab, cols, rep_, stt in (("M2A raw", [ex] + M2A, ex, None),
                                          ("M2A + winter, baseline stratified by cycle", [ex, "winter"] + M2A, ex, strata),
                                          ("M2A, acquisition-corrected exposure", [ex_c] + M2A, ex_c, None)):
                r, _, _ = svy_cox(d, des, doma_m, oc, cols, [rep_], strata=stt)
                hr += [{"outcome": olab, "exposure": ex, "model": alab, **x} for x in r]
    hr = pd.DataFrame(hr)
    K["q2_hr"] = hr
    rep.table("q2_hr_acquisition_adjusted", hr, "Mortality HRs without and with acquisition adjustment (per 10 mmHg "
              "SBP; per 0.1 lower ABI; ABI < 0.9)", max_rows=None)
    # reclassification of low ABI
    m = R["abi_mort"] & d["abi_min_c"].notna().values
    rc = []
    for a_, b_ in ((1, 1), (1, 0), (0, 1), (0, 0)):
        ind = (d["abi_lt09"] == a_).values & (d["abi_c_lt09"] == b_).values
        e = ls.svy_mean(ind.astype(float), des, domain=m, proportion=True)
        rc.append({"raw_abi_lt_0.9": a_, "corrected_abi_lt_0.9": b_, "n_unweighted": int((m & ind).sum()),
                   "weighted_share": e["estimate"], "deaths": int(d.loc[m & ind, "death_all"].sum())})
    rc = pd.DataFrame(rc)
    K["q2_reclass"] = rc
    rep.table("q2_low_abi_reclassification", rc, "Low ABI (< 0.9) by raw vs acquisition-corrected lower-leg ABI "
              "(ABI mortality domain)", max_rows=None)
    # discrimination
    cdf, cdd, _ = cindex_block(d, des, domm, {"M2 + later mean (raw)": M2 + ["sbp_later10"],
                                              "M2 + later mean (corrected)": M2 + ["sbp_later_c10"]},
                               [("M2 + later mean (corrected)", "M2 + later mean (raw)")], Mrep, "Q2 SBP")
    da = R["abi_mort"] & ~np.isnan(d["winter"].values) & (d["abi_max"] <= 1.4).values & \
        d["abi_c_per01_lower"].notna().values
    cdf2, cdd2, _ = cindex_block(d, des, da, {"M2A (no ABI)": M2A, "M2A + ABI per 0.1 lower (raw)": M2A + ["abi_per01_lower"],
                                              "M2A + ABI per 0.1 lower (corrected)": M2A + ["abi_c_per01_lower"],
                                              "M2A + ABI < 0.9 (raw)": M2A + ["abi_lt09"],
                                              "M2A + ABI < 0.9 (corrected)": M2A + ["abi_c_lt09"]},
                                 [("M2A + ABI per 0.1 lower (corrected)", "M2A + ABI per 0.1 lower (raw)"),
                                  ("M2A + ABI < 0.9 (corrected)", "M2A + ABI < 0.9 (raw)"),
                                  ("M2A + ABI per 0.1 lower (raw)", "M2A (no ABI)")], Mrep, "Q2 ABI")
    c_all = pd.concat([cdf, cdf2], ignore_index=True)
    d_all = pd.concat([cdd, cdd2], ignore_index=True)
    K["q2_c"], K["q2_dc"] = c_all, d_all
    rep.table("q2_cindex", c_all, "C-index with raw vs acquisition-corrected exposures", max_rows=None)
    rep.table("q2_cindex_diff", d_all, "C-index differences", max_rows=None)


def q2_cuff(d, des, doma, rep, K):
    """Added in review (2026-10-05): the ABI cycle differences against the documented cuff-protocol change."""
    rep.p("**ABI cuff protocol (review addition).** LEXABPI.htm documents a protocol change during 1999-2000: brachial "
          "and tibial pressures were first taken with 10 and 12 cm cuffs, later with 9/12/15/18 cm cuffs chosen by arm "
          "circumference (the same cuff on arm and ankles). The cuff mix therefore differs by cycle. Below: weighted "
          "cuff-size shares by cycle, and the cycle coefficients for ABI, mean ankle SBP and supine brachial SBP "
          "before and after adding arm circumference and cuff-size indicators (same records in all three "
          "specifications; reference cuff 15 cm). Cuff size is chosen from arm circumference, so this is a "
          "descriptive decomposition, not a causal estimate of the cuff effect.")
    sh = []
    for k in (1, 2, 3):
        for site, c in (("brachial", "cuff_arm_cm"), ("ankle", "cuff_ank_cm")):
            m = doma & (d["k"] == k).values & d[c].notna().values
            for cm in sorted(CUFF_CM.values()):
                ind = (d[c] == cm).astype(float).where(d[c].notna()).values
                if not np.nansum(ind[m]):
                    continue
                e = ls.svy_mean(ind, des, domain=m, proportion=True)
                sh.append({"cycle": CYC_LAB[k], "site": site, "cuff_cm": cm, "n_unweighted": int(np.nansum(ind[m])),
                           "weighted_share": e["estimate"], "lower": e["lower"], "upper": e["upper"]})
    K["q2_cuff_share"] = pd.DataFrame(sh)
    rep.table("q2_abi_cuff_shares", K["q2_cuff_share"], "Cuff size by cycle (ABI domain, survey-weighted shares)",
              max_rows=None)
    cuffc = ["cuffA_10", "cuffA_12", "cuffA_oth", "cuffB_12", "cuffB_oth"]
    domc = doma & d[cuffc + ["armc5"]].notna().all(axis=1).values
    rows = []
    for y in ("abi_min", "ankle_mean", "brach_supine"):
        for spec, extra in (("M1", []), ("M1 + arm circumference", ["armc5"]),
                            ("M1 + arm circumference + cuff sizes", ["armc5"] + cuffc)):
            r, m = svy_lm(d, des, domc, y, ["winter", "cyc2", "cyc3"] + M1 + extra)
            rows += lm_rows(r, m, ["cyc2", "cyc3"] + [c for c in cuffc if spec.endswith("sizes")],
                            {"outcome": y, "spec": spec})
    K["q2_cuff_lm"] = pd.DataFrame(rows)
    rep.table("q2_abi_cuff_models", K["q2_cuff_lm"], "ABI-exam cycle differences (vs 1999-2000) without and with "
              "arm circumference and cuff size (design-based)", max_rows=None)


def q3(d, des, R, Mrep, rep, K):
    rep.h("Q3. Measurement reliability and regression dilution")
    dom3 = R["mort"] & d[["sbp_r1", "sbp_r2", "sbp_r3"]].notna().all(axis=1).values
    icc = [icc_table_row(d, des, R["bp"], "sbp_r2", "sbp_r3", Mrep, "SBP reading 2 vs 3 (BP domain)"),
           icc_table_row(d, des, R["bp"], "sbp_r1", "sbp_r2", Mrep, "SBP reading 1 vs 2 (BP domain)"),
           icc_table_row(d, des, R["bp"], "sbp_r1", "sbp_r3", Mrep, "SBP reading 1 vs 3 (BP domain)")]
    domlr = R["abi"] & d[["abi_r", "abi_l"]].notna().all(axis=1).values
    icc.append(icc_table_row(d, des, domlr, "abi_r", "abi_l", Mrep, "ABI right vs left leg (agreement; age >= 40)"))
    dom2 = R["abi"] & (d["age_y"] < 60).values & d[["abi_low_set1", "abi_low_set2"]].notna().all(axis=1).values
    icc.append(icc_table_row(d, des, dom2, "abi_low_set1", "abi_low_set2", Mrep,
                             "Lower-leg ABI reading set 1 vs set 2 (age 40-59)"))
    icc = pd.DataFrame(icc)
    K["q3_icc"] = icc
    rep.table("q3_icc", icc, "Within-visit reliability (one-way ICC, Pearson r, mean difference; Rao-Wu CIs)",
              max_rows=None)
    lam = []
    for lab, dom_, y, x, Z in (("SBP single reading | none", dom3, "sbp_r3", "sbp_r2", []),
                               ("SBP single reading | M1", dom3, "sbp_r3", "sbp_r2", M1),
                               ("SBP single reading | M2", dom3, "sbp_r3", "sbp_r2", M2),
                               ("ABI lower leg single set | M2A (age 40-59)",
                                dom2 & R["abi_mort"], "abi_low_set2", "abi_low_set1", M2A)):
        lam.append({"quantity": lab, **rc_lambda(d, des, dom_, y, x, Z)})
    lam = pd.DataFrame(lam)
    l1 = lam.iloc[2]
    K["q3_lambda"] = lam
    rep.table("q3_rdr", lam, "Regression-dilution ratios (replicate regressed on reading + covariates; design-based)",
              max_rows=None)
    rows = []
    kk = {"sbp_r1": 1, "sbp_r2": 1, "sbp_r3": 1, "sbp_r23": 2, "sbp_r123": 3}
    for oc, olab in OUTCOMES:
        base = None
        for ex in ("sbp_r1", "sbp_r2", "sbp_r3", "sbp_r23", "sbp_r123"):
            r, f, _ = svy_cox(d, des, dom3, oc, [ex + "10"] + M2, [ex + "10"])
            x = r[0]
            lk, lse = sb(l1["lambda"], kk[ex]), sb_se(l1["lambda"], l1["se"], kk[ex])
            b, s, h, lo, hi = corrected(x["log_hr"], x["se_design"], lk, lse, x["design_df"])
            if ex == "sbp_r2":
                base = x["log_hr"]
            rows.append({"outcome": olab, "exposure": ex, "lambda_used": lk, **{k: x[k] for k in (
                "n_unweighted", "events_unweighted", "log_hr", "se_design", "hr", "lower", "upper")},
                "corrected_log_hr": b, "corrected_se": s, "corrected_hr": h, "corrected_lower": lo, "corrected_upper": hi})
        for rr in rows:
            if rr["outcome"] == olab:
                rr["observed_ratio_to_reading2"] = rr["log_hr"] / base
                rr["predicted_ratio_from_lambda"] = rr["lambda_used"] / l1["lambda"]
    dil = pd.DataFrame(rows)
    K["q3_dil"] = dil
    rep.table("q3_dilution_sbp", dil, "SBP per 10 mmHg (M2) by number of readings averaged (domain with readings "
              "1-3): observed and within-visit-corrected HRs. Reading 1 carries the acute excess; its 'prediction' "
              "uses the single-reading lambda", max_rows=None)
    # ABI dilution (age 40-59)
    labi = lam.iloc[3]
    da = dom2 & R["abi_mort"] & (d["abi_max"] <= 1.4).values & (d["abi_high_set1"] <= 1.4).values
    arows = []
    for oc, olab in OUTCOMES:
        for ex, kx, lab in (("abi_low_set1_per01_lower", 1, "single reading set"),
                            ("abi_per01_lower", 2, "NCHS mean of two sets")):
            r, _, _ = svy_cox(d, des, da, oc, [ex] + M2A, [ex])
            x = r[0]
            lk, lse = sb(labi["lambda"], kx), sb_se(labi["lambda"], labi["se"], kx)
            b, s, h, lo, hi = corrected(x["log_hr"], x["se_design"], lk, lse, x["design_df"])
            arows.append({"outcome": olab, "exposure": f"ABI per 0.1 lower, {lab}", "lambda_used": lk,
                          **{k: x[k] for k in ("n_unweighted", "events_unweighted", "log_hr", "se_design", "hr", "lower",
                                               "upper")}, "corrected_hr": h, "corrected_lower": lo, "corrected_upper": hi})
    adil = pd.DataFrame(arows)
    K["q3_abi_dil"] = adil
    rep.table("q3_dilution_abi", adil, "ABI per 0.1 lower (M2A; age 40-59; both reading sets): observed and "
              "within-visit-corrected HRs", max_rows=None)


def q4(d, des, R, Mrep, rep, K, sess_cycles):
    rep.h("Q4. Within-day session (PHDSESN)")
    req = session_requests()
    K["q4_requests"] = req
    doms = R["bp"] & ~np.isnan(d["sess_aft"].values) & ~np.isnan(d["winter"].values)
    if doms.sum() < 500:
        K["q4_status"] = "skipped"
        rep.p("**Skipped:** the session variable PHDSESN is not on disk for any 1999-2016 cycle (it is in the PH / "
              "FASTQX fasting-questionnaire files, which are not in external/nhanes/). The driver uses these files "
              "automatically when they are placed at the local targets below (no code change). Hub request: download "
              "with scripts/fetch_one.sh (logged in external/DOWNLOAD_MANIFEST.tsv); all are open CDC files, no "
              "sign-up. The within-day analysis is done here with NHANES III instead (module B, MXPSESSR).")
        rep.table("q4_hub_request_session_files", req, "Files needed for the 1999-2016 within-day analysis",
                  max_rows=None)
        return
    K["q4_status"] = f"run (cycles with PHDSESN: {', '.join(sess_cycles) if sess_cycles else 'simulated'})"
    rep.p(f"Session available in {int(doms.sum())} BP-domain records ({K['q4_status']}).")
    rep.p("Run 2 (2026-10-05): PHDSESN read from the PH/FASTQX files placed on disk by the hub (codes 0 morning, 1 "
          "afternoon, 2 evening in all nine codebooks). The files on disk do not document how SPs were assigned to "
          "sessions, so these are between-person contrasts; the session also bundles fasting instructions, exam order "
          "and examiner shift (see NHANES III module B for the within-person and assignment-based analogues).")
    desc = []                             # run 2: descriptive session distribution (BP domain)
    w = des.weights
    for s, col in (("morning", None), ("afternoon", "sess_aft"), ("evening", "sess_eve")):
        if col is None:
            ms = doms & (d["sess_aft"].values == 0) & (d["sess_eve"].values == 0)
        else:
            ms = doms & (d[col].values == 1)
        desc.append({"session": s, "n_unweighted": int(ms.sum()), "weighted_share": float(w[ms].sum() / w[doms].sum()),
                     "sbp_later_wmean": wmean_dom(d["sbp_later"], w, ms),
                     "sbp_excess_wmean": wmean_dom(d["sbp_excess"], w, ms), "age_wmean": wmean_dom(d["age_y"], w, ms)})
    K["q4_desc"] = pd.DataFrame(desc)
    rep.table("q4_session_sample", K["q4_desc"], "BP domain by session (weighted shares and means; added in run 2)",
              max_rows=None)
    rows, tests = [], []
    X = ["sess_aft", "sess_eve", "winter"] + CYC_D + M1
    for y in ("sbp_first", "sbp_later", "sbp_excess", "dbp_later", "pulse"):
        r, m = svy_lm(d, des, doms, y, X)
        rows += lm_rows(r, m, ["sess_aft", "sess_eve"], {"outcome": y})
        tests += wald_rows(r, [("session (2 df)", ["sess_aft", "sess_eve"])], {"outcome": y})
    doma = R["abi"] & ~np.isnan(d["sess_aft"].values) & ~np.isnan(d["winter"].values)
    if doma.sum() > 200:
        for y in ("abi_min", "brach_supine", "ankle_mean"):
            r, m = svy_lm(d, des, doma, y, ["sess_aft", "sess_eve", "winter", "cyc2", "cyc3"] + M1)
            rows += lm_rows(r, m, ["sess_aft", "sess_eve"], {"outcome": y})
    se = pd.DataFrame(rows)
    K["q4_sess"] = se
    K["q4_tests"] = pd.DataFrame(tests)
    K["q4_n"] = int(doms.sum())
    rep.table("q4_session_effects", se, "Session effects (ref morning; + winter, cycle, M1)", max_rows=None)
    rep.table("q4_session_tests", pd.DataFrame(tests), "Wald tests", max_rows=None)
    r, m = svy_lm(d, des, doms, "sbp_later", ["sess_aft", "sess_eve"] + M1)
    A = d[["sess_aft", "sess_eve"]].values @ r.coef[1:3]
    d["sbp_later_s10"] = (d["sbp_later"] - (A - np.average(A[m], weights=des.weights[m]))) / 10
    domm = R["mort"] & ~np.isnan(d["sess_aft"].values)
    hr = []
    for oc, olab in OUTCOMES:
        for alab, cols, rr in (("M2 raw", ["sbp_later10"] + M2, "sbp_later10"),
                               ("M2 + session", ["sbp_later10", "sess_aft", "sess_eve"] + M2, "sbp_later10"),
                               ("M2, session-corrected SBP", ["sbp_later_s10"] + M2, "sbp_later_s10")):
            r_, _, _ = svy_cox(d, des, domm, oc, cols, [rr])
            hr += [{"outcome": olab, "model": alab, **x} for x in r_]
    hr = pd.DataFrame(hr)
    K["q4_hr"] = hr
    rep.table("q4_session_hr", hr, "SBP later mean per 10 mmHg with and without session adjustment", max_rows=None)
    cdf, cdd, _ = cindex_block(d, des, domm, {"M2 + later mean (raw)": M2 + ["sbp_later10"],
                                              "M2 + later mean (session-corrected)": M2 + ["sbp_later_s10"]},
                               [("M2 + later mean (session-corrected)", "M2 + later mean (raw)")], Mrep, "Q4 session")
    K["q4_dc"] = cdd
    rep.table("q4_cindex", cdf, "C-index", max_rows=None)
    rep.table("q4_cindex_diff", cdd, "C-index difference", max_rows=None)


# =================================================================================================================
# Module B: NHANES III
# =================================================================================================================
def nh3(d, des, R, Mrep, rep, K):
    rep.h("Module B. NHANES III (1988-1994): within-day session, calendar, household vs MEC occasion")
    base, dom = R["base"], R["mort"]
    w = des.weights
    desc = []
    for lab, m in (("all in domain", base),) + tuple((s, base & (d["session"] == s).values)
                                                       for s in ("morning", "afternoon", "evening")):
        desc.append({"group": lab, "n_unweighted": int(m.sum()), "weighted_age": wmean_dom(d["age_y"], w, m),
                     "mec_mean_sbp": wmean_dom(d["mec_mean"], w, m), "hh_mean_sbp": wmean_dom(d["hh_mean"], w, m),
                     "mec_minus_hh": wmean_dom(d["occ_dev"], w, m), "mec_first_excess": wmean_dom(d["mec_excess"], w, m),
                     "interval_months_median": float(np.nanmedian(d.loc[m, "interval_m"])) if m.any() else np.nan,
                     "deaths_in_mort_domain": int(d.loc[m & dom, "death_all"].sum())})
    rep.table("b_sample", pd.DataFrame(desc), "NHANES III analysis domain by MEC session (weighted means)",
              max_rows=None)
    # B1 acquisition
    ok = base & d[["sess_aft", "m_cos", "weekend"]].notna().all(axis=1).values
    X = ["sess_aft", "sess_eve", "weekend", "m_cos", "m_sin", "phase2"] + N1
    rows, tests = [], []
    for spec, xs in (("main", X), ("+ Census region", X + ["reg_mw", "reg_s", "reg_w"])):
        for y in ("mec_mean", "mec_excess", "pulse", "hh_mean", "occ_dev"):
            r, m = svy_lm(d, des, ok, y, xs)
            rows += lm_rows(r, m, ["sess_aft", "sess_eve", "weekend", "m_cos", "m_sin", "phase2"],
                            {"spec": spec, "outcome": y})
            tests += wald_rows(r, [("session (2 df)", ["sess_aft", "sess_eve"]), ("month harmonics (2 df)",
                                                                                 ["m_cos", "m_sin"])],
                               {"spec": spec, "outcome": y})
    acq = pd.DataFrame(rows)
    K["b_acq"] = acq
    rep.p("Outcomes: `mec_mean` (MEC physician SBP, mean of 3), `mec_excess` (MEC reading 1 minus mean of 2-3), `pulse` "
          "(MEC radial), `hh_mean` (household SBP by interviewers, another day: negative control for who attends which "
          "session), `occ_dev` (MEC minus household mean: within-person contrast that removes stable person-level "
          "differences). Month harmonics: cos/sin of 2 pi (month - 1) / 12 (cos peaks in January).")
    rep.table("b_acquisition_effects", acq, "Design-based regressions on session (ref morning), weekend, month "
              "harmonics and phase (+ age, sex, race/ethnicity)", max_rows=None)
    K["b_tests"] = pd.DataFrame(tests)
    rep.table("b_acquisition_tests", K["b_tests"], "Wald tests", max_rows=None)
    mrows = []
    for mo in range(1, 13):
        m = base & (d["month"] == mo).values
        for y in ("mec_mean", "occ_dev"):
            e = ls.svy_mean(d[y].values, des, domain=m)
            mrows.append({"month": mo, "outcome": y, "n_unweighted": int(m.sum()), "weighted_mean": e["estimate"],
                          "lower": e["lower"], "upper": e["upper"]})
    rep.table("b_month_means", pd.DataFrame(mrows), "Weighted means by MEC exam month (unadjusted)", max_rows=None)
    nh3_session_design(d, des, ok, X, rep, K)
    # B2 reliability
    icc = [icc_table_row(d, des, base, "mec_r2", "mec_r3", Mrep, "MEC reading 2 vs 3 (within occasion)"),
           icc_table_row(d, des, base, "mec_r1", "mec_r2", Mrep, "MEC reading 1 vs 2 (within occasion)"),
           icc_table_row(d, des, base, "hh_r2", "hh_r3", Mrep, "Household reading 2 vs 3 (within occasion)"),
           icc_table_row(d, des, base, "hh_mean", "mec_mean", Mrep, "Household mean vs MEC mean (between occasions)")]
    for lab, cond in (("interval 0-1 month", d["interval_m"] <= 1), ("interval >= 2 months", d["interval_m"] >= 2)):
        icc.append(icc_table_row(d, des, base & cond.values, "hh_mean", "mec_mean", Mrep,
                                 f"Household mean vs MEC mean, {lab}"))
    icc = pd.DataFrame(icc)
    K["b_icc"] = icc
    rep.table("b_icc", icc, "Within- vs between-occasion reliability (one-way ICC; Rao-Wu CIs)", max_rows=None)
    lam = []
    for lab, dom_, y, x in (("within-visit, MEC single reading (r3 on r2)", dom, "mec_r3", "mec_r2"),
                            ("between-occasion, MEC mean (household mean on MEC mean)", dom, "hh_mean", "mec_mean"),
                            ("between-occasion, household mean (MEC mean on household mean)", dom, "mec_mean", "hh_mean"),
                            ("between-occasion, MEC mean, interval 0-1 month", dom & (d["interval_m"] <= 1).values,
                             "hh_mean", "mec_mean"),
                            ("between-occasion, MEC mean, interval >= 2 months", dom & (d["interval_m"] >= 2).values,
                             "hh_mean", "mec_mean")):
        lam.append({"quantity": lab, **rc_lambda(d, des, dom_, y, x, N2)})
    lam = pd.DataFrame(lam).set_index("quantity", drop=False)
    K["b_lambda"] = lam
    rep.table("b_rdr", lam.reset_index(drop=True), "Regression-dilution ratios given N2 (design-based)", max_rows=None)
    lw = lam.iloc[0]
    lb = lam.iloc[1]
    lh = lam.iloc[2]
    rho = np.sqrt(lb["lambda"] * lh["lambda"])
    rho_se = 0.5 * rho * np.sqrt((lb["se"] / lb["lambda"]) ** 2 + (lh["se"] / lh["lambda"]) ** 2)
    lam2, lam2_se = sb(rho, 2), sb_se(rho, rho_se, 2)
    # B3 mortality
    rows = []
    for oc, olab in OUTCOMES:
        for ex in ("mec_mean", "hh_mean", "two_occ", "mec_first", "mec_later"):
            r, _, _ = svy_cox(d, des, dom, oc, [ex + "10"] + N2, [ex + "10"])
            x = r[0]
            row = {"outcome": olab, "exposure": ex + " (per 10 mmHg)", **x}
            if ex == "mec_mean":
                for nm, lk, lse in (("within-visit (Spearman-Brown, 3 readings)", sb(lw["lambda"], 3),
                                     sb_se(lw["lambda"], lw["se"], 3)),
                                    ("between-occasion", lb["lambda"], lb["se"])):
                    b, s, h, lo, hi = corrected(x["log_hr"], x["se_design"], lk, lse, x["design_df"])
                    rows.append({**row, "correction": nm, "lambda_used": lk, "corrected_hr": h, "corrected_lower": lo,
                                 "corrected_upper": hi, "corrected_log_hr": b})
            elif ex == "two_occ":
                b, s, h, lo, hi = corrected(x["log_hr"], x["se_design"], lam2, lam2_se, x["design_df"])
                rows.append({**row, "correction": "between-occasion (Spearman-Brown, 2 occasions)", "lambda_used": lam2,
                             "corrected_hr": h, "corrected_lower": lo, "corrected_upper": hi, "corrected_log_hr": b})
            elif ex == "hh_mean":
                b, s, h, lo, hi = corrected(x["log_hr"], x["se_design"], lh["lambda"], lh["se"], x["design_df"])
                rows.append({**row, "correction": "between-occasion", "lambda_used": lh["lambda"], "corrected_hr": h,
                             "corrected_lower": lo, "corrected_upper": hi, "corrected_log_hr": b})
            else:
                rows.append({**row, "correction": "none"})
    hr = pd.DataFrame(rows)
    K["b_hr"] = hr
    rep.table("b_hr_by_occasion", hr, "SBP per 10 mmHg (N2) by occasion and number of occasions; dilution-corrected",
              max_rows=None)
    # per SD mutual models
    sd = {}
    for c in ("mec_later", "mec_excess", "two_occ", "occ_dev", "mec_mean"):
        sd[c] = ls.weighted_sd(d.loc[dom, c].values, w[dom])
        d[c + "_sd"] = d[c] / sd[c]
    mrows = []
    for oc, olab in OUTCOMES:
        for mlab, ex in (("MEC later mean + MEC acute excess", ["mec_later_sd", "mec_excess_sd"]),
                         ("two-occasion mean + occasion deviation (MEC - household)", ["two_occ_sd", "occ_dev_sd"])):
            r, _, _ = svy_cox(d, des, dom, oc, ex + N2, ex)
            mrows += [{"outcome": olab, "model": mlab, "sd_of_term": sd[x["term"][:-3]], **x} for x in r]
    mut = pd.DataFrame(mrows)
    K["b_mutual"] = mut
    rep.table("b_mutual_models", mut, "Per SD, mutually adjusted (N2)", max_rows=None)
    # acquisition adjustment and within-session HRs
    doma = dom & d[["sess_aft", "m_cos", "weekend"]].notna().all(axis=1).values
    arows, trows = [], []
    for oc, olab in OUTCOMES:
        for alab, cols in (("N2", ["mec_mean10"] + N2),
                           ("N2 + session, weekend, month harmonics, phase",
                            ["mec_mean10", "sess_aft", "sess_eve", "weekend", "m_cos", "m_sin", "phase2"] + N2)):
            r, _, _ = svy_cox(d, des, doma, oc, cols, ["mec_mean10"])
            arows += [{"outcome": olab, "model": alab, "subset": "all sessions", **x} for x in r]
        for s in ("morning", "afternoon", "evening"):
            r, _, _ = svy_cox(d, des, doma & (d["session"] == s).values, oc, ["mec_mean10"] + N2, ["mec_mean10"])
            arows += [{"outcome": olab, "model": "N2", "subset": f"{s} session", **x} for x in r]
        dd = d.copy()
        dd["x_aft"] = dd["mec_mean10"] * dd["sess_aft"]
        dd["x_eve"] = dd["mec_mean10"] * dd["sess_eve"]
        cols = ["mec_mean10", "x_aft", "x_eve", "sess_aft", "sess_eve"] + N2
        _, f, _ = svy_cox(dd, des, doma, oc, cols, ["mec_mean10"])
        wt = ls.wald_test(f.coef, f.var_design, idx=[1, 2])
        trows.append({"outcome": olab, "test": "SBP x session (2 df, design-based)", **wt, **wald_f(wt, f.design_df)})
    ah = pd.DataFrame(arows)
    K["b_acq_hr"] = ah
    rep.table("b_hr_acquisition", ah, "MEC SBP per 10 mmHg without/with acquisition adjustment and within sessions",
              max_rows=None)
    K["b_inter"] = pd.DataFrame(trows)
    rep.table("b_session_interaction", K["b_inter"], "Interaction tests", max_rows=None)
    # discrimination
    models = {"N2 (no BP)": N2, "N2 + MEC mean": N2 + ["mec_mean10"], "N2 + household mean": N2 + ["hh_mean10"],
              "N2 + two-occasion mean": N2 + ["two_occ10"], "N2 + MEC first reading": N2 + ["mec_first10"]}
    contr = [("N2 + two-occasion mean", "N2 + MEC mean"), ("N2 + MEC mean", "N2 + MEC first reading"),
             ("N2 + MEC mean", "N2 + household mean"), ("N2 + MEC mean", "N2 (no BP)")]
    cdf, cdd, _ = cindex_block(d, des, dom, models, contr, Mrep, "B NHANES III")
    K["b_c"], K["b_dc"] = cdf, cdd
    rep.table("b_cindex", cdf, "C-index (NHANES III)", max_rows=None)
    rep.table("b_cindex_diff", cdd, "C-index differences (NHANES III)", max_rows=None)


def nh3_session_design(d, des, ok, X, rep, K):
    """Added in review (2026-10-05): randomised session assignment, fasting time and venipuncture clock (NHANES III)."""
    rep.p("**B1b. Session: randomised assignment, fasting and clock time (review addition).** NHANES III randomly "
          "assigned sampled households to a morning ('standard') or an afternoon/evening ('modified') MEC session "
          "(adult.htm, notes on WTPFSD6/WTPFMD6). The session subsample weights (positive only for persons examined in "
          "the assigned session, non-response-adjusted) give an assignment-based contrast of afternoon/evening vs "
          "morning that does not rely on who chooses which session (persons examined off-assignment, about 4-6 %, are "
          "excluded, so it is not a strict intention-to-treat contrast). Sessions also differ in instructed fasting "
          "(lab file PHPFAST: hours since last food or drink) and clock time (PHPBEST: time of venipuncture, a proxy "
          "for the time of the physician BP measurement, which is not recorded); afternoon and evening sessions have "
          "similar fasting times, so the evening minus afternoon difference is not a fasting effect.")
    rows, desc = [], []
    for s in ("morning", "afternoon", "evening"):
        m = ok & (d["session"] == s).values
        desc.append({"session": s, "n_unweighted": int(m.sum()),
                     "fast_h_median": float(np.nanmedian(d.loc[m, "fast_h"])),
                     "fast_h_q05": float(np.nanpercentile(d.loc[m, "fast_h"], 5)),
                     "fast_h_q95": float(np.nanpercentile(d.loc[m, "fast_h"], 95)),
                     "clock_h_median": float(np.nanmedian(d.loc[m, "clock_h"])),
                     "clock_h_q05": float(np.nanpercentile(d.loc[m, "clock_h"], 5)),
                     "clock_h_q95": float(np.nanpercentile(d.loc[m, "clock_h"], 95)),
                     "share_examined_in_assigned_session": float(np.mean(d.loc[m, "arm_pm"].notna()))})
    K["b_sess_desc"] = pd.DataFrame(desc)
    rep.table("b_session_fasting_clock", K["b_sess_desc"], "Fasting hours and venipuncture clock time by session "
              "(unweighted quantiles 5-95 %)", max_rows=None)
    # (1) assignment-based contrast with the session subsample weights
    desA = ls.SurveyDesign(d["w_arm"].values, d["SDPSTRA6"].values, d["SDPPSU6"].values, nest=True,
                           lonely_psu="adjust")
    domA = ok & ~np.isnan(d["arm_pm"].values) & (d["w_arm"].values > 0)
    for y in ("mec_mean", "occ_dev", "hh_mean", "pulse", "mec_excess"):
        for spec, xs in (("assigned afternoon/evening vs morning (subsample weights), + phase + N1",
                          ["arm_pm", "phase2"] + N1),
                         ("assigned afternoon/evening vs morning (subsample weights), unadjusted", ["arm_pm"])):
            r, m = svy_lm(d, desA, domA, y, xs)
            rows += lm_rows(r, m, ["arm_pm"], {"spec": spec, "outcome": y})
    # (2) main session model + fasting hours; (3) clock time instead of session; (4) clock within session
    okf = ok & d[["fast_h", "clock_h"]].notna().all(axis=1).values
    Xn = [c for c in X if c not in ("sess_aft", "sess_eve")]
    for y in ("mec_mean", "occ_dev", "hh_mean", "pulse"):
        for spec, xs, terms in (("session (main), fasting/clock-complete records", X, ["sess_aft", "sess_eve"]),
                                ("session + fasting hours", X + ["fast_h"], ["sess_aft", "sess_eve", "fast_h"]),
                                ("venipuncture clock hour instead of session", Xn + ["clock_h"], ["clock_h"]),
                                ("clock hour + fasting hours", Xn + ["clock_h", "fast_h"], ["clock_h", "fast_h"]),
                                ("clock hour within session", X + ["clock_h"], ["sess_aft", "sess_eve", "clock_h"])):
            r, m = svy_lm(d, des, okf, y, xs)
            rows += lm_rows(r, m, terms, {"spec": spec, "outcome": y})
    K["b_sess_design"] = pd.DataFrame(rows)
    rep.table("b_session_design", K["b_sess_design"], "Session effects: randomised-assignment contrast, fasting "
              "adjustment and clock time (design-based; mmHg, /min or mmHg per hour; + weekend, month harmonics, "
              "phase, N1 unless stated)", max_rows=None)


# =================================================================================================================
# Summary
# =================================================================================================================
def pick(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    if not m.any():
        return None
    return df[m].iloc[0]


def hr_txt(r, pre=""):
    if r is None:
        return "n/a"
    return f"{pre}{r['hr']:.3f} ({r['lower']:.3f}, {r['upper']:.3f})"


def summary(K, rep):
    S = []

    def add(section, item, value, lo=np.nan, hi=np.nan, note=""):
        S.append({"section": section, "item": item, "estimate": value, "lower": lo, "upper": hi, "note": note})
    em = K["q1_excess_mean"]
    add("Q1", "SBP acute excess at the visit (reading 1 - later mean), mmHg", em["estimate"], em["lower"], em["upper"])
    pc = K["q1_excess_pos"]
    add("Q1", "Share with a positive acute excess (proportion)", pc["estimate"], pc["lower"], pc["upper"])
    cox = K["q1_cox"]
    for oc in ("All-cause death", "CVD death"):
        for mlab, term in (("first reading alone", "sbp_first_sd"), ("later mean alone", "sbp_later_sd"),
                           ("later mean + acute excess (mutual)", "sbp_later_sd"),
                           ("later mean + acute excess (mutual)", "sbp_excess_sd")):
            r = pick(cox, outcome=oc, adjustment="M2", model=mlab, term=term, scale="per SD")
            if r is not None:
                add("Q1", f"{oc}: HR per SD, {mlab}, term {term} (M2)", r["hr"], r["lower"], r["upper"])
    for _, r in K["q1_dc"].iterrows():
        add("Q1", f"{r['outcome']}: delta C, {r['contrast']}", r["delta_c"], r["lower"], r["upper"])
    mis = K["q1_mis"]
    for thr in ("140/90", "130/80"):
        r = pick(mis, threshold=thr, subgroup="all adults",
                 quantity="P(not confirmed by later mean | hypertensive by first reading)")
        if r is not None:
            add("Q1", f"Hypertensive by first reading ({thr}) not confirmed by later mean", r["estimate"], r["lower"],
                r["upper"])
        r = pick(mis, threshold=thr, subgroup="all adults", quantity="prevalence by first reading minus by later mean")
        if r is not None:
            add("Q1", f"Prevalence excess from first reading ({thr})", r["estimate"], r["lower"], r["upper"])
    disc = K["q1_disc"]
    for oc in ("All-cause death", "CVD death"):
        r = pick(disc, threshold="140/90", outcome=oc, term="hc_first_140/90")
        if r is not None:
            add("Q1", f"{oc}: HR first-reading-only hypertension vs neither (M2)", r["hr"], r["lower"], r["upper"])
    acq = K["q2_acq"]
    for y in ("sbp_later", "sbp_excess", "pulse", "abi_min", "brach_supine", "ankle_mean"):
        r = pick(acq, outcome=y, term="winter")
        if r is not None:
            add("Q2", f"Nov-Apr vs May-Oct effect on {y} (units)", r["coef"], r["lower"], r["upper"])
    hr = K["q2_hr"]
    for ex in ("sbp_later10", "abi_per01_lower", "abi_lt09"):
        for _, r in hr[(hr["exposure"] == ex) & (hr["outcome"] == "All-cause death")].iterrows():
            add("Q2", f"All-cause death HR, {ex}, {r['model']}", r["hr"], r["lower"], r["upper"])
    for _, r in K["q2_dc"].iterrows():
        add("Q2", f"{r['outcome']}: delta C, {r['contrast']}", r["delta_c"], r["lower"], r["upper"])
    rc = K["q2_reclass"]
    nre = int(rc.loc[rc["raw_abi_lt_0.9"] != rc["corrected_abi_lt_0.9"], "n_unweighted"].sum())
    add("Q2", "Low ABI reclassified by acquisition correction (n, unweighted)", nre)
    for _, r in K["q3_icc"].iterrows():
        add("Q3", f"ICC {r['comparison']}", r["icc_oneway"], r["icc_oneway_lower"], r["icc_oneway_upper"])
    for _, r in K["q3_lambda"].iterrows():
        add("Q3", f"RDR {r['quantity']}", r["lambda"], r["lower"], r["upper"])
    for _, r in K["q3_dil"][K["q3_dil"]["outcome"] == "All-cause death"].iterrows():
        add("Q3", f"All-cause death HR per 10 mmHg, {r['exposure']} observed", r["hr"], r["lower"], r["upper"],
            f"observed/predicted log-HR ratio vs reading 2: {r['observed_ratio_to_reading2']:.3f}/"
            f"{r['predicted_ratio_from_lambda']:.3f}")
        add("Q3", f"All-cause death HR per 10 mmHg, {r['exposure']} within-visit corrected", r["corrected_hr"],
            r["corrected_lower"], r["corrected_upper"])
    for _, r in K["q3_abi_dil"].iterrows():
        add("Q3", f"{r['outcome']}: HR {r['exposure']} observed", r["hr"], r["lower"], r["upper"])
        add("Q3", f"{r['outcome']}: HR {r['exposure']} corrected", r["corrected_hr"], r["corrected_lower"],
            r["corrected_upper"])
    for _, r in K["q2_cuff_lm"][K["q2_cuff_lm"]["term"].isin(["cyc2", "cyc3"])].iterrows():
        add("Q2", f"ABI exam {r['term']} vs 1999-2000 on {r['outcome']}, {r['spec']}", r["coef"], r["lower"], r["upper"])
    t = pick(K["q2_tests"], outcome="sbp_excess", test="cycle (8 df)")
    add("Q2", "Cycle test on SBP acute excess: design-based F (8, design df)", t["F"], note=f"p_F {t['p_F']:.2e}")
    add("Q4", "Session analysis (1999-2016)", np.nan, note=K["q4_status"])
    if "q4_sess" in K:                    # run 2 (2026-10-05): Q4 estimates in the key-estimate table
        for _, r in K["q4_sess"].iterrows():
            add("Q4", f"{r['term']} vs morning on {r['outcome']} (units)", r["coef"], r["lower"], r["upper"])
        for _, r in K["q4_tests"].iterrows():
            add("Q4", f"Session test on {r['outcome']}: design-based F (2, design df)", r["F"], note=f"p_F {r['p_F']:.2e}")
        for _, r in K["q4_hr"].iterrows():
            add("Q4", f"{r['outcome']}: HR SBP later mean per 10 mmHg, {r['model']}", r["hr"], r["lower"], r["upper"])
        for _, r in K["q4_dc"].iterrows():
            add("Q4", f"{r['outcome']}: delta C, {r['contrast']}", r["delta_c"], r["lower"], r["upper"])
    if "b_acq" in K:
        ba = K["b_acq"]
        for y in ("mec_mean", "occ_dev", "hh_mean", "mec_excess", "pulse"):
            for t in ("sess_aft", "sess_eve", "m_cos", "weekend"):
                r = pick(ba, spec="main", outcome=y, term=t)
                if r is not None:
                    add("B1", f"NHANES III {t} effect on {y}", r["coef"], r["lower"], r["upper"])
        for _, r in K["b_sess_design"].iterrows():
            add("B1b", f"NHANES III {r['term']} effect on {r['outcome']}, {r['spec']}", r["coef"], r["lower"],
                r["upper"])
        for _, r in K["b_icc"].iterrows():
            add("B2", f"ICC {r['comparison']}", r["icc_oneway"], r["icc_oneway_lower"], r["icc_oneway_upper"])
        for _, r in K["b_lambda"].iterrows():
            add("B2", f"RDR {r['quantity']}", r["lambda"], r["lower"], r["upper"])
        for _, r in K["b_hr"][K["b_hr"]["outcome"] == "All-cause death"].iterrows():
            add("B3", f"All-cause death HR {r['exposure']} observed", r["hr"], r["lower"], r["upper"], r["correction"])
            if r["correction"] != "none":
                add("B3", f"All-cause death HR {r['exposure']} corrected ({r['correction']})", r["corrected_hr"],
                    r["corrected_lower"], r["corrected_upper"])
        for _, r in K["b_mutual"].iterrows():
            add("B3", f"{r['outcome']}: HR per SD {r['term']} ({r['model']})", r["hr"], r["lower"], r["upper"])
        for _, r in K["b_acq_hr"][K["b_acq_hr"]["outcome"] == "All-cause death"].iterrows():
            add("B3", f"All-cause death HR MEC SBP per 10, {r['model']}, {r['subset']}", r["hr"], r["lower"], r["upper"])
        for _, r in K["b_dc"].iterrows():
            add("B3", f"{r['outcome']}: delta C, {r['contrast']}", r["delta_c"], r["lower"], r["upper"])
    S = pd.DataFrame(S)
    rep.h("Key estimates (all from the tables above)")
    rep.table("key_estimates", S, None, max_rows=None)
    return S


def findings(K) -> list[str]:
    """Main numbers in neutral wording (all values taken from the result tables of this run)."""
    def h(r):
        return "n/a" if r is None else f"{r['hr']:.3f} ({r['lower']:.3f}, {r['upper']:.3f})"

    def c(r, col="coef", nd=2):
        return "n/a" if r is None else f"{r[col]:+.{nd}f} ({r['lower']:+.{nd}f}, {r['upper']:+.{nd}f})"

    def dc(df, oc, contrast):
        r = pick(df, outcome=oc, contrast=contrast)
        return "n/a" if r is None else f"{r['delta_c']:+.4f} ({r['lower']:+.4f}, {r['upper']:+.4f})"

    def pf(t):
        return f"F {t['F']:.1f} on {int(t['df'])} and {int(t['design_df'])} df, p {t['p_F']:.1e}"
    F = []
    em, pc = K["q1_excess_mean"], K["q1_excess_pos"]
    cox, sc = K["q1_cox"], K["q1_scaling"].set_index("variable")["weighted_sd"]
    a = {oc: [pick(cox, outcome=oc, adjustment="M2", model=m_, term=t_, scale="per SD") for m_, t_ in (
        ("first reading alone", "sbp_first_sd"), ("later mean alone", "sbp_later_sd"),
        ("later mean + acute excess (mutual)", "sbp_later_sd"), ("later mean + acute excess (mutual)", "sbp_excess_sd"))]
        for oc in ("All-cause death", "CVD death")}
    F.append(f"**Q1 (1999-2016, auscultatory BP).** Within-visit acute excess (reading 1 minus later mean): "
             f"{em['estimate']:.2f} ({em['lower']:.2f}, {em['upper']:.2f}) mmHg; positive in "
             f"{100 * pc['estimate']:.1f} %. All-cause death, M2, HR per SD: first reading {h(a['All-cause death'][0])}, "
             f"later mean {h(a['All-cause death'][1])}; mutually adjusted later mean {h(a['All-cause death'][2])} and "
             f"acute excess {h(a['All-cause death'][3])} (1 SD of the excess = {sc['sbp_excess']:.1f} mmHg, of the "
             f"later mean = {sc['sbp_later']:.1f} mmHg). CVD death: first {h(a['CVD death'][0])}, later "
             f"{h(a['CVD death'][1])}; mutual later {h(a['CVD death'][2])}, excess {h(a['CVD death'][3])}.")
    F.append(f"Discrimination (Harrell C, M2): later mean minus first reading, all-cause "
             f"{dc(K['q1_dc'], 'All-cause death', 'M2 + later mean minus M2 + first reading')}, CVD "
             f"{dc(K['q1_dc'], 'CVD death', 'M2 + later mean minus M2 + first reading')}; adding the acute excess "
             f"{dc(K['q1_dc'], 'All-cause death', 'M2 + later mean + excess minus M2 + later mean')} (all-cause); adding "
             f"SBP to M2 {dc(K['q1_dc'], 'All-cause death', 'M2 + later mean minus M2 (no BP)')} (all-cause).")
    mis, disc = K["q1_mis"], K["q1_disc"]
    r1 = pick(mis, threshold="140/90", subgroup="all adults",
              quantity="P(not confirmed by later mean | hypertensive by first reading)")
    r2 = pick(mis, threshold="140/90", subgroup="all adults", quantity="prevalence by first reading minus by later mean")
    r3 = pick(mis, threshold="140/90", subgroup="untreated (bp_med = 0)",
              quantity="P(not confirmed by later mean | hypertensive by first reading)")
    F.append(f"BP-defined hypertension (>= 140/90): of adults hypertensive by the first reading, "
             f"{100 * r1['estimate']:.1f} % ({100 * r1['lower']:.1f}, {100 * r1['upper']:.1f}) were not by the later mean "
             f"(untreated {100 * r3['estimate']:.1f} %); prevalence by the first reading is higher than by the later mean "
             f"by {100 * r2['estimate']:.1f} ({100 * r2['lower']:.1f}, {100 * r2['upper']:.1f}) percentage points. HR (M2) vs "
             f"neither, all-cause: first-reading-only {h(pick(disc, threshold='140/90', outcome='All-cause death', term='hc_first_140/90'))}, "
             f"both {h(pick(disc, threshold='140/90', outcome='All-cause death', term='hc_both_140/90'))}; CVD: "
             f"first-reading-only {h(pick(disc, threshold='140/90', outcome='CVD death', term='hc_first_140/90'))}, both "
             f"{h(pick(disc, threshold='140/90', outcome='CVD death', term='hc_both_140/90'))}.")
    acq = K["q2_acq"]
    t_ex = pick(K["q2_tests"], outcome="sbp_excess", test="cycle (8 df)")
    cp = K["q2_cells"]
    ce = cp[cp["outcome"] == "sbp_excess"]
    late = [pick(acq, outcome="sbp_excess", term=f"cyc{k}") for k in (7, 8, 9)]
    F.append(f"**Q2 (acquisition conditions).** Nov-Apr vs May-Oct (adjusted for cycle, age, sex, race): later-mean SBP "
             f"{c(pick(acq, outcome='sbp_later', term='winter'))} mmHg, pulse {c(pick(acq, outcome='pulse', term='winter'))} "
             f"/min, acute excess {c(pick(acq, outcome='sbp_excess', term='winter'))} mmHg, lower-leg ABI "
             f"{c(pick(acq, outcome='abi_min', term='winter'), nd=3)}. The within-person acute excess differed by survey "
             f"cycle (Wald chi2 {t_ex['chi2']:.1f}, 8 df; design-based {pf(t_ex)}); cycle x period cell means ranged "
             f"{ce['weighted_mean'].min():.2f} to {ce['weighted_mean'].max():.2f} mmHg, and the 2011-2012, 2013-2014 and "
             f"2015-2016 cycles were lower than 1999-2000 by {-late[0]['coef']:.2f}, {-late[1]['coef']:.2f} and "
             f"{-late[2]['coef']:.2f} mmHg (the BPX documentation on disk reports no protocol change in these cycles; the "
             f"cause cannot be identified from public files).")
    cl = K["q2_cuff_lm"]

    def cyc3(y, spec):
        return pick(cl, outcome=y, spec=spec, term="cyc3")
    s0, s2 = "M1", "M1 + arm circumference + cuff sizes"
    sh = K["q2_cuff_share"]

    def share(cyc, site, cm):
        r = pick(sh, cycle=cyc, site=site, cuff_cm=cm)
        return 0.0 if r is None else 100 * r["weighted_share"]
    F.append(f"ABI exam 2003-2004 vs 1999-2000 (M1): lower-leg ABI {c(pick(acq, outcome='abi_min', term='cyc3'), nd=3)}, "
             f"mean ankle SBP {c(pick(acq, outcome='ankle_mean', term='cyc3'), nd=1)} mmHg, supine brachial SBP "
             f"{c(pick(acq, outcome='brach_supine', term='cyc3'), nd=1)} mmHg. These cycles differ in a documented "
             f"acquisition factor: ankle pressures were taken with a 10 cm cuff in {share('1999-2000', 'ankle', 10):.0f} % "
             f"of 1999-2000 exams and with a 15 cm cuff in {share('2003-2004', 'ankle', 15):.0f} % of 2003-2004 exams. In "
             f"records with cuff data, the 2003-2004 difference in ABI was {c(cyc3('abi_min', s0), nd=3)} before and "
             f"{c(cyc3('abi_min', s2), nd=3)} after adding arm circumference and cuff sizes (ankle SBP "
             f"{c(cyc3('ankle_mean', s0), nd=1)} -> {c(cyc3('ankle_mean', s2), nd=1)} mmHg; brachial "
             f"{c(cyc3('brach_supine', s0), nd=1)} -> {c(cyc3('brach_supine', s2), nd=1)} mmHg). Cuff size is "
             f"collinear with cycle and arm size and the individual cuff coefficients are imprecise, so this is "
             f"consistent with, not proof of, a cuff-protocol origin of the cycle difference.")
    hr = K["q2_hr"]

    def chain(ex, oc):
        rows = hr[(hr["exposure"] == ex) & (hr["outcome"] == oc)]
        return " -> ".join(f"{r['hr']:.3f}" for _, r in rows.iterrows())
    rc = K["q2_reclass"]
    nre = int(rc.loc[rc["raw_abi_lt_0.9"] != rc["corrected_abi_lt_0.9"], "n_unweighted"].sum())
    F.append(f"Mortality HRs raw -> + period, cycle-stratified -> acquisition-corrected exposure (all-cause): SBP later "
             f"mean per 10 mmHg {chain('sbp_later10', 'All-cause death')}; ABI per 0.1 lower "
             f"{chain('abi_per01_lower', 'All-cause death')}; ABI < 0.9 {chain('abi_lt09', 'All-cause death')}. Low ABI "
             f"reclassified by the correction: {nre} of {int(rc['n_unweighted'].sum())}. C-index change with corrected "
             f"SBP {dc(K['q2_dc'], 'All-cause death', 'M2 + later mean (corrected) minus M2 + later mean (raw)')}.")
    icc, lam, dil = K["q3_icc"], K["q3_lambda"], K["q3_dil"]
    i23 = pick(icc, a="sbp_r2", b="sbp_r3")
    ilr = pick(icc, a="abi_r", b="abi_l")
    iab = pick(icc, a="abi_low_set1", b="abi_low_set2")
    l1, la = lam.iloc[2], lam.iloc[3]
    r2_ = pick(dil, outcome="All-cause death", exposure="sbp_r2")
    r23 = pick(dil, outcome="All-cause death", exposure="sbp_r23")
    r123 = pick(dil, outcome="All-cause death", exposure="sbp_r123")
    F.append(f"**Q3 (reliability and dilution).** Within-visit SBP ICC (readings 2 vs 3) {i23['icc_oneway']:.3f} "
             f"({i23['icc_oneway_lower']:.3f}, {i23['icc_oneway_upper']:.3f}); RDR of one reading given M2 "
             f"{l1['lambda']:.3f} ({l1['lower']:.3f}, {l1['upper']:.3f}). ABI right vs left ICC {ilr['icc_oneway']:.3f}; "
             f"lower-leg ABI set 1 vs set 2 (age 40-59) ICC {iab['icc_oneway']:.3f}, RDR {la['lambda']:.3f}. All-cause HR "
             f"per 10 mmHg: reading 2 {r2_['hr']:.3f}, mean of readings 2-3 {r23['hr']:.3f}; within-visit corrected "
             f"{r23['corrected_hr']:.3f} ({r23['corrected_lower']:.3f}, {r23['corrected_upper']:.3f}). Averaging readings "
             f"did not raise the observed log-HR as the within-visit RDR predicts (ratio to reading 2: mean of 2-3 "
             f"{r23['observed_ratio_to_reading2']:.3f} observed vs {r23['predicted_ratio_from_lambda']:.3f} predicted; mean "
             f"of 1-3 {r123['observed_ratio_to_reading2']:.3f} vs {r123['predicted_ratio_from_lambda']:.3f}; not formally "
             f"tested).")
    if "q4_sess" not in K:
        F.append(f"**Q4 (within-day session, 1999-2016):** {K['q4_status']}.")
    else:                                 # run 2 (2026-10-05)
        qs, qt, qh = K["q4_sess"], K["q4_tests"], K["q4_hr"]

        def qc(y, t, nd=2):
            return c(pick(qs, outcome=y, term=t), nd=nd)

        def qf(y):
            t = pick(qt, outcome=y, test="session (2 df)")
            return "n/a" if t is None else pf(t)

        def qchain(oc):
            rows = qh[qh["outcome"] == oc]
            return " -> ".join(f"{r['hr']:.3f}" for _, r in rows.iterrows())
        abi_txt = ""
        if pick(qs, outcome="abi_min", term="sess_eve") is not None:
            abi_txt = (f" Lower-leg ABI (1999-2004, age >= 40): afternoon {qc('abi_min', 'sess_aft', 3)}, evening "
                       f"{qc('abi_min', 'sess_eve', 3)}.")
        F.append(f"**Q4 (within-day session, 1999-2016; between persons, session not documented as randomised).** "
                 f"Versus the morning session (+ 6-month period, cycle, age, sex, race; n = {K['q4_n']}): later-mean "
                 f"SBP afternoon {qc('sbp_later', 'sess_aft')}, evening {qc('sbp_later', 'sess_eve')} mmHg "
                 f"(session {qf('sbp_later')}); first reading afternoon {qc('sbp_first', 'sess_aft')}, evening "
                 f"{qc('sbp_first', 'sess_eve')}; acute excess afternoon {qc('sbp_excess', 'sess_aft')}, evening "
                 f"{qc('sbp_excess', 'sess_eve')} mmHg; later-mean DBP afternoon {qc('dbp_later', 'sess_aft')}, evening "
                 f"{qc('dbp_later', 'sess_eve')} mmHg; pulse afternoon {qc('pulse', 'sess_aft')}, evening "
                 f"{qc('pulse', 'sess_eve')} /min.{abi_txt} SBP later mean per 10 mmHg, M2 raw -> + session -> "
                 f"session-corrected SBP: all-cause {qchain('All-cause death')}, CVD {qchain('CVD death')}; C-index "
                 f"change with session-corrected SBP: all-cause "
                 f"{dc(K['q4_dc'], 'All-cause death', 'M2 + later mean (session-corrected) minus M2 + later mean (raw)')}, "
                 f"CVD {dc(K['q4_dc'], 'CVD death', 'M2 + later mean (session-corrected) minus M2 + later mean (raw)')}.")
    if "b_acq" in K:
        ba, bt, bs = K["b_acq"], K["b_tests"], K["b_sess_design"]
        bl, bh, bm, bah = K["b_lambda"], K["b_hr"], K["b_mutual"], K["b_acq_hr"]
        tdev = pick(bt, spec="main", outcome="occ_dev", test="session (2 df)")
        F.append(f"**B1 (NHANES III, session and calendar).** MEC SBP vs morning session: afternoon "
                 f"{c(pick(ba, spec='main', outcome='mec_mean', term='sess_aft'))}, evening "
                 f"{c(pick(ba, spec='main', outcome='mec_mean', term='sess_eve'))} mmHg; household SBP measured on another "
                 f"day (negative control): afternoon {c(pick(ba, spec='main', outcome='hh_mean', term='sess_aft'))}, evening "
                 f"{c(pick(ba, spec='main', outcome='hh_mean', term='sess_eve'))}; within-person MEC minus household: "
                 f"afternoon {c(pick(ba, spec='main', outcome='occ_dev', term='sess_aft'))}, evening "
                 f"{c(pick(ba, spec='main', outcome='occ_dev', term='sess_eve'))} (session Wald chi2 {tdev['chi2']:.1f}, "
                 f"2 df; design-based {pf(tdev)}). Pulse: afternoon "
                 f"{c(pick(ba, spec='main', outcome='pulse', term='sess_aft'))}, evening "
                 f"{c(pick(ba, spec='main', outcome='pulse', term='sess_eve'))} /min. Month cos harmonic (January peak): MEC "
                 f"SBP {c(pick(ba, spec='main', outcome='mec_mean', term='m_cos'))}, household SBP "
                 f"{c(pick(ba, spec='main', outcome='hh_mean', term='m_cos'))}, MEC minus household "
                 f"{c(pick(ba, spec='main', outcome='occ_dev', term='m_cos'))}.")
        sa = "assigned afternoon/evening vs morning (subsample weights), + phase + N1"
        F.append(f"**B1b (session design checks).** Assignment-based contrast (households randomly assigned to the "
                 f"afternoon/evening vs the morning session; persons examined in the assigned session): MEC minus household "
                 f"{c(pick(bs, spec=sa, outcome='occ_dev'))} mmHg, MEC SBP {c(pick(bs, spec=sa, outcome='mec_mean'))}, "
                 f"household SBP (balance check) {c(pick(bs, spec=sa, outcome='hh_mean'))}. Adding fasting hours to the "
                 f"session model: evening MEC minus household "
                 f"{c(pick(bs, spec='session (main), fasting/clock-complete records', outcome='occ_dev', term='sess_eve'))} "
                 f"-> {c(pick(bs, spec='session + fasting hours', outcome='occ_dev', term='sess_eve'))} mmHg (fasting "
                 f"hours {c(pick(bs, spec='session + fasting hours', outcome='occ_dev', term='fast_h'), nd=3)} per hour). "
                 f"Venipuncture clock time instead of session: MEC minus household "
                 f"{c(pick(bs, spec='venipuncture clock hour instead of session', outcome='occ_dev', term='clock_h'), nd=3)} "
                 f"mmHg per hour of the day (household SBP "
                 f"{c(pick(bs, spec='venipuncture clock hour instead of session', outcome='hh_mean', term='clock_h'), nd=3)}); "
                 f"within session the clock slope is imprecise "
                 f"({c(pick(bs, spec='clock hour within session', outcome='occ_dev', term='clock_h'), nd=3)} per hour), "
                 f"because clock time and session are nearly collinear. Acute excess (reading 1 minus 2-3) by assigned "
                 f"session {c(pick(bs, spec=sa, outcome='mec_excess'))} mmHg.")
        bi = K["b_icc"]
        iw = pick(bi, a="mec_r2", b="mec_r3")
        ih = pick(bi, a="hh_r2", b="hh_r3")
        ib = pick(bi, comparison="Household mean vs MEC mean (between occasions)")
        lw3 = sb(bl.iloc[0]["lambda"], 3)
        F.append(f"**B2 (reliability).** ICC within occasion: MEC readings 2 vs 3 {iw['icc_oneway']:.3f}, household "
                 f"{ih['icc_oneway']:.3f}; between occasions (household mean vs MEC mean; household BP taken at the end of "
                 f"the household interview, MEC exam a median of about 1 month after the screener interview) "
                 f"{ib['icc_oneway']:.3f} ({ib['icc_oneway_lower']:.3f}, {ib['icc_oneway_upper']:.3f}). RDR for the "
                 f"3-reading MEC mean: {lw3:.3f} if only within-visit reading error is counted (one reading "
                 f"{bl.iloc[0]['lambda']:.3f}, Spearman-Brown) vs {bl.iloc[1]['lambda']:.3f} "
                 f"({bl.iloc[1]['lower']:.3f}, {bl.iloc[1]['upper']:.3f}) if household-vs-MEC disagreement is counted.")
        for oc in ("All-cause death", "CVD death"):
            rw = pick(bh, outcome=oc, exposure="mec_mean (per 10 mmHg)", correction="within-visit (Spearman-Brown, 3 readings)")
            rb = pick(bh, outcome=oc, exposure="mec_mean (per 10 mmHg)", correction="between-occasion")
            r2o = pick(bh, outcome=oc, exposure="two_occ (per 10 mmHg)")
            u = pick(bm, outcome=oc, term="two_occ_sd")
            v = pick(bm, outcome=oc, term="occ_dev_sd")
            F.append(f"B3 {oc}, HR per 10 mmHg (N2): MEC mean {h(rw)}; within-visit corrected {rw['corrected_hr']:.3f} "
                     f"({rw['corrected_lower']:.3f}, {rw['corrected_upper']:.3f}); between-occasion corrected "
                     f"{rb['corrected_hr']:.3f} ({rb['corrected_lower']:.3f}, {rb['corrected_upper']:.3f}); two-occasion mean "
                     f"{h(r2o)}. Per SD, mutually adjusted: two-occasion mean {h(u)}, occasion deviation (MEC minus "
                     f"household; 1 SD = {v['sd_of_term']:.1f} mmHg) {h(v)}.")
        a0 = pick(bah, outcome="All-cause death", model="N2", subset="all sessions")
        a1 = pick(bah, outcome="All-cause death", model="N2 + session, weekend, month harmonics, phase")
        ti = pick(K["b_inter"], outcome="All-cause death")
        F.append(f"B3 acquisition adjustment (all-cause, MEC SBP per 10 mmHg): {a0['hr']:.3f} -> {a1['hr']:.3f}; "
                 f"SBP x session interaction p {ti['p']:.2f} (design-based F p {ti['p_F']:.2f}). C-index, two-occasion "
                 f"minus MEC mean: all-cause "
                 f"{dc(K['b_dc'], 'All-cause death', 'N2 + two-occasion mean minus N2 + MEC mean')}, CVD "
                 f"{dc(K['b_dc'], 'CVD death', 'N2 + two-occasion mean minus N2 + MEC mean')}.")
    return F


def interpretation(K) -> list[str]:
    """Reading rules for the numbers above (review 2026-10-05). Wording limits, not new results."""
    ups = []
    for oc in ("All-cause death", "CVD death"):
        r = pick(K["q1_cox"], outcome=oc, adjustment="M2", model="later mean + acute excess (mutual)",
                 term="sbp_excess_sd", scale="per SD")
        ups.append(f"Q1 excess {oc.split()[0].lower()} {r['upper']:.3f}")
        if "b_mutual" in K:
            r = pick(K["b_mutual"], outcome=oc, term="occ_dev_sd")
            ups.append(f"B3 occasion deviation {oc.split()[0].lower()} {r['upper']:.3f}")
    out = [
        "Null mutually adjusted terms (Q1 acute excess; B3 occasion deviation) mean no detectable association with "
        f"death beyond the usual level (upper 95 % limits of the HR per SD: {'; '.join(ups)}); they do not show that "
        "the component is pure noise, and the excess term also carries reading-level error of the later mean with the "
        "opposite sign.",
        "The NHANES III session contrast is a bundle: clock time, instructed fasting (morning about 13 h, afternoon and "
        "evening about 7 h), MEC exam order and examiner shift (12 physicians took MEC BP; examiner IDs are not public). "
        "The assignment-based contrast (B1b) addresses self-selection into morning vs afternoon/evening, and the "
        "fasting-hours adjustment addresses fasting, but neither removes exam order or examiner shift; the time of day "
        "of the household BP is not recorded. Safe wording: 'BP measured later in the exam day (afternoon/evening "
        "session) was higher within the same person'; a 'clock' or circadian mechanism is not established.",
        "Between-occasion RDR treats every household-vs-MEC difference (interviewer vs physician, home vs MEC, true BP "
        "change over about a month) as error and assumes classical errors with slope 1; the corrected HRs refer to BP "
        "shared across the two settings. Within-visit RDR counts only reading-to-reading error. Neither is the "
        "long-term usual BP of the classic regression-dilution literature, so 'how much dilution matters' depends on "
        "which variation is treated as error.",
        "The winter rise appears at home and in the MEC alike (NHANES III), which is consistent with a seasonal change in "
        "BP rather than an MEC-specific artefact; month and 6-month period also track MEC location (stated as NHANES "
        "operating practice, not in the documentation files on disk).",
        "The ABI cycle differences coincide with a documented cuff-protocol change; cuff size is chosen from arm "
        "circumference, so the cuff-adjusted rows are a descriptive decomposition, not a cuff-effect estimate.",
        "Adjusting for period/cycle barely moved the SBP and ABI mortality HRs: this is a null for a highly "
        "standardised survey with auscultatory/Doppler methods; it does not transfer to HPP oscillometric leg PWV without "
        "HPP data.",
        "Global proportional-hazards tests reject in most Cox models (mainly age and long follow-up); HRs are averages "
        "over follow-up. Per-term PH p-values in this report are computed correctly here (lib_survival.cox_zph per-term "
        "values are not scale invariant; see the driver docstring of ph_tests).",
    ]
    return out


LIMITATIONS = [
    "Public-data supporting evidence: NHANES BP is auscultatory (mercury sphygmomanometer, physician examiners in the "
    "MEC; lay interviewers at home in NHANES III) and ABI is Doppler-based; F1's main measure is oscillometric leg "
    "PWV. These analyses are analogues of the F1 acquisition-physics claims, not replications.",
    "Acquisition conditions are confounded by design: the 6-month period (RIDEXMON) with geography (MECs operate in "
    "southern locations in winter: NHANES operating practice, not stated in the files on disk), survey cycle with "
    "secular trends in BP and treatment and, for ABI, with the documented cuff-protocol change, and NHANES III session "
    "with fasting instructions, exam order and examiner shift (morning vs afternoon/evening assignment was random by "
    "household; afternoon vs evening was chosen). The household-measured BP (another day) is the negative control and "
    "MEC - household the within-person contrast; 'corrected' exposures are sensitivity analyses.",
    "Within-visit reliability reflects reading-to-reading error only, so dilution corrections based on it are lower "
    "bounds for usual-BP dilution; the NHANES III household vs MEC comparison gives a between-occasion analogue "
    "(different observers and settings). The interval variable is screener interview to MEC exam (age in months at "
    "each; median about one month); the household BP was taken at the household adult interview, which falls between "
    "the two, so the true household-to-MEC interval is not known exactly.",
    "Regression calibration assumes classical non-differential error, approximately linear effects and parallel "
    "replicates (Spearman-Brown); the delta-method CI treats lambda and log-HR as independent. The mutually adjusted "
    "'acute excess' coefficient is not a clean causal contrast: reading-level error in the later mean enters the "
    "excess with the opposite sign.",
    "Outcomes are deaths only (2019 public-use linked mortality files; NCHS perturbs follow-up time and cause for some "
    "decedents); no incident non-fatal events. Follow-up is short for 2011-2016 records. CVD death = underlying cause "
    "heart disease or cerebrovascular disease (UCOD_LEADING 001, 005); other deaths are censored (cause-specific).",
    "Design: Taylor linearisation with t(design df) intervals for model coefficients; multi-df Wald tests are given "
    "as chi2 and as design-based F (q, design df), the latter preferred; C-index and ICC CIs from a Rao-Wu PSU "
    "bootstrap with linear predictors fixed at the full-sample fit (model-fitting uncertainty not propagated).",
    "Records without a first SBP reading (first attempt interrupted) are outside the BP domain; a 4th reading "
    "replaces an interrupted 2nd or 3rd reading far more often in 1999-2006 than later (sample_by_cycle.csv, "
    "column bp_domain_with_4th_reading), so the later mean is not built identically across cycles.",
    "Age is top-coded (85 in 1999-2006, 80 from 2007, 90 in NHANES III). Many estimates are reported; treat them as "
    "descriptive supporting evidence, not confirmatory tests.",
    "Q4 (run 2): the 1999-2016 session (PHDSESN) is recorded by the phlebotomist; how SPs were assigned to sessions is "
    "not documented in the files on disk, and there is no second-occasion BP, so the session contrasts are between "
    "persons and bundle clock time with fasting instructions, exam order, examiner shift and who attends when.",
]


# =================================================================================================================
# Main
# =================================================================================================================
class JReport(C.Report):
    """C.Report with a header line that matches this driver (raw public files, not a clean table)."""

    def write(self, outdir: Path, driver_file: str):
        super().write(outdir, driver_file)
        p = outdir / "report.md"
        txt = p.read_text(encoding="utf-8")
        line = ("- Real public NHANES files read directly (raw XPT / fixed-width; no clean table). "
                "The analyses marked 'review addition' were added after the first "
                "real run (2026-10-05). Run 2 (2026-10-05): Q4 runs on the PH/FASTQX session files and XPORT zeros "
                "are read as 0 (`CHANGELOG_20261005.md`)." if self.mode != "simulate" else
                "- **SIMULATED DATA** (fake raw frames with the real variable names; planted effects). These numbers "
                "say nothing about the real data.")
        for old in ("- Real clean public table.",
                    "- **SIMULATED DATA** (fake table with the same columns as the clean public table; planted effects). "
                    "These numbers say nothing about the real data."):
            txt = txt.replace(old, line, 1)
        p.write_text(txt, encoding="utf-8")


def parse_args():
    def extra(ap):
        ap.add_argument("--n-rep", type=int, default=200,
                        help="Rao-Wu PSU bootstrap replicates for C-index and ICC intervals")
        ap.add_argument("--skip-nh3", action="store_true", help="skip module B (NHANES III)")
    return C.parse_args(__doc__.splitlines()[0], extra)


def gate(args):
    if args.simulate:
        OUT_SIM.mkdir(parents=True, exist_ok=True)
        return "simulate", OUT_SIM
    if args.real:
        OUT_REAL.mkdir(parents=True, exist_ok=True)
        return "real", OUT_REAL
    print(f"[{TASK}] REFUSED: this driver fits models on the real public data. Run with --simulate for a wiring test, "
          "or with --real.", file=sys.stderr)
    raise SystemExit(2)


def main():
    import warnings
    warnings.simplefilter("ignore", pd.errors.PerformanceWarning)   # column-wise frame building; harmless
    args = parse_args()
    mode, outdir = gate(args)
    rng = np.random.default_rng(args.seed)
    t0 = _dt.datetime.now()
    rep = JReport(TASK, "F1 Stage 0 task j - NHANES acquisition physics of BP and ABI with mortality "
                  "(public-data supporting evidence)", mode, args)
    rep.p("- **Label:** public-data supporting evidence for F1 (measurement physics). No HPP data. "
          "No liver phenotypes.")
    at = len(rep.parts)
    if mode == "simulate":
        rep.p(f"- Simulate outputs are written to `results/stage0/analysis/{TASK}/simulate_validation/` (this task may "
              "write only its own folder).")
        raw_c, truth_c = simulate_cont(rng)
        raw_3, truth_3 = simulate_nh3(rng)
        sess_cycles = []
        rep.h("Wiring check against the real file headers / SAS layouts (schema only)")
        wc = wiring_check(raw_c, raw_3)
        rep.table("wiring_check", wc, None, max_rows=12)
    else:
        raw_c, sess_cycles = load_cont()
        raw_3 = None if args.skip_nh3 else load_nh3()
    rep.h("Pre-specified analysis")
    rep.p("```text\nPre-specified analysis" + __doc__.split("Pre-specified analysis", 1)[1].rstrip() + "\n```")
    K: dict = {}
    d, R = prepare_cont(raw_c)
    des = design_cont(d)
    Mrep = raowu(des, args.n_rep, rng)
    rep.h("Module A: continuous NHANES 1999-2016 - data and design")
    rep.table("design_summary", pd.DataFrame([{"module": "A continuous 1999-2016", "records": len(d),
                                               "strata": des.n_strata, "psus": des.n_psu, "design_df": des.df,
                                               "n_rep_bootstrap": args.n_rep}]), "Design", max_rows=None)
    rep.table("sample_by_cycle", sample_table(d, R), "Unweighted counts by cycle", max_rows=None)
    q1(d, des, R, Mrep, rep, K)
    print(f"[{TASK}] Q1 done {(_dt.datetime.now() - t0).seconds}s", flush=True)
    q2(d, des, R, Mrep, rep, K)
    print(f"[{TASK}] Q2 done {(_dt.datetime.now() - t0).seconds}s", flush=True)
    q3(d, des, R, Mrep, rep, K)
    print(f"[{TASK}] Q3 done {(_dt.datetime.now() - t0).seconds}s", flush=True)
    q4(d, des, R, Mrep, rep, K, sess_cycles)
    print(f"[{TASK}] Q4 done {(_dt.datetime.now() - t0).seconds}s", flush=True)
    d3 = None
    if raw_3 is not None:
        d3, R3 = prepare_nh3(raw_3)
        des3 = ls.SurveyDesign(d3["w"].values, d3["SDPSTRA6"].values, d3["SDPPSU6"].values, nest=True,
                               lonely_psu="adjust")
        M3 = raowu(des3, args.n_rep, rng)
        rep.table("design_summary_nh3", pd.DataFrame([{"module": "B NHANES III", "records": len(d3),
                                                       "strata": des3.n_strata, "psus": des3.n_psu,
                                                       "design_df": des3.df, "domain_n": int(R3["base"].sum()),
                                                       "mortality_domain_n": int(R3["mort"].sum()),
                                                       "deaths": int(d3.loc[R3["mort"], "death_all"].sum())}]),
                  "Design (NHANES III)", max_rows=None)
        nh3(d3, des3, R3, M3, rep, K)
        print(f"[{TASK}] B done {(_dt.datetime.now() - t0).seconds}s", flush=True)
    S = summary(K, rep)
    try:
        F = findings(K)
    except Exception as e:  # noqa: BLE001 - the findings text must never block the outputs
        F = [f"(findings text failed: {e!r}; see the tables)"]
    try:
        IR = interpretation(K)
    except Exception as e:  # noqa: BLE001
        IR = [f"(reading-rules text failed: {e!r})"]
    rep.parts[at:at] = ["", "## Main findings (numbers from the tables below; neutral wording)", ""] + \
        [f"- {x}" for x in F] + ["", "## Reading rules for these numbers (review 2026-10-05)", ""] + \
        [f"- {x}" for x in IR] + ["", "See the Limitations section before using any number."]
    rep.h("Limitations")
    rep.bullets(LIMITATIONS)
    planted = None
    if mode == "simulate":
        lam = K["q3_lambda"]
        l1 = lam.iloc[2]
        dil = K["q3_dil"]
        r23 = pick(dil, outcome="All-cause death", exposure="sbp_r23")
        acq = K["q2_acq"]
        wsb = pick(acq, outcome="sbp_later", term="winter")
        wab = pick(acq, outcome="abi_min", term="winter")
        abi_m = wmean_dom(d["abi_min"], des.weights, R["abi"])
        em = K["q1_excess_mean"]
        checks = [("A: within-visit RDR, single SBP reading | M2", truth_c["lambda1_within"], l1["lambda"], l1["se"]),
                  ("A: winter effect on later-mean SBP (mmHg)", truth_c["winter_sbp"], wsb["coef"], wsb["se_design"]),
                  ("A: mean first-reading excess (mmHg)", truth_c["excess_mean"], em["estimate"], em["se"]),
                  ("A: within-visit-corrected log-HR per 10 mmHg, mean of readings 2-3 (target: visit-true BP)",
                   truth_c["beta_visit_true10"], r23["corrected_log_hr"], r23["corrected_se"]),
                  ("A: winter effect on lower-leg ABI (approx. -0.02 x mean ABI)",
                   (SIM["abi_winter_factor"] - 1) * abi_m, wab["coef"], wab["se_design"])]
        if "q4_sess" in K:
            r = pick(K["q4_sess"], outcome="sbp_later", term="sess_eve")
            checks.append(("A: evening vs morning session SBP (mmHg)", truth_c["sess_eve"], r["coef"], r["se_design"]))
        if d3 is not None:
            lb = K["b_lambda"].iloc[1]
            hb = pick(K["b_hr"], outcome="All-cause death", exposure="mec_mean (per 10 mmHg)", correction="between-occasion")
            ba = K["b_acq"]
            se_ = pick(ba, spec="main", outcome="mec_mean", term="sess_eve")
            dv = pick(ba, spec="main", outcome="occ_dev", term="sess_eve")
            mc = pick(ba, spec="main", outcome="mec_mean", term="m_cos")
            hh = pick(ba, spec="main", outcome="hh_mean", term="sess_eve")
            hc = corrected(hb["log_hr"], hb["se_design"], hb["lambda_used"], lb["se"], hb["design_df"])
            checks += [("B: between-occasion RDR of the MEC mean | N2", truth_3["lambda_between_mec"], lb["lambda"], lb["se"]),
                       ("B: between-occasion-corrected log-HR per 10 mmHg, MEC mean (target: usual BP)",
                        truth_3["beta_u10"], hc[0], hc[1]),
                       ("B: evening vs morning, MEC mean SBP (mmHg)", truth_3["sess_eve"], se_["coef"], se_["se_design"]),
                       ("B: evening vs morning, MEC - household (mmHg)", truth_3["sess_eve"], dv["coef"], dv["se_design"]),
                       ("B: evening vs morning, household SBP (negative control, planted 0)", 0.0, hh["coef"], hh["se_design"]),
                       ("B: month cos harmonic, MEC mean SBP (mmHg)", truth_3["month_cos"], mc["coef"], mc["se_design"])]
            bs = K["b_sess_design"]
            sa = "assigned afternoon/evening vs morning (subsample weights), + phase + N1"
            ra = pick(bs, spec=sa, outcome="occ_dev")
            rh = pick(bs, spec=sa, outcome="hh_mean")
            rf = pick(bs, spec="session + fasting hours", outcome="occ_dev", term="sess_eve")
            rfh = pick(bs, spec="session + fasting hours", outcome="occ_dev", term="fast_h")
            checks += [("B1b: assigned afternoon/evening vs morning, MEC - household (subsample weights)",
                        truth_3["arm_pm"], ra["coef"], ra["se_design"]),
                       ("B1b: assigned afternoon/evening vs morning, household SBP (planted 0)", 0.0, rh["coef"],
                        rh["se_design"]),
                       ("B1b: evening vs morning, MEC - household, + fasting hours (fasting planted 0)",
                        truth_3["sess_eve"], rf["coef"], rf["se_design"]),
                       ("B1b: fasting hours coefficient, MEC - household (planted 0)", 0.0, rfh["coef"],
                        rfh["se_design"])]
        # C-index fast vs brute force on a subsample
        m = R["mort"].copy()
        idx = np.flatnonzero(m)[:1500]
        tt = d["t"].values[idx]
        ee = d["death_all"].values[idx]
        lp = d["sbp_later"].values[idx] + rng.normal(0, 5, len(idx)).round()
        ww = des.weights[idx]
        fast = cindex_reps(tt, ee, lp, ww[None, :])[0]
        brute = cindex_brute(tt, ee, lp, ww)
        checks.append(("C-index fast vs brute force (1,500 records, lp ties present)", brute, fast, 1e-9))
        planted = C.planted_rows(checks)
    rep.inputs = list(INPUTS.values())
    C.finish(rep, outdir, __file__, planted)


if __name__ == "__main__":
    main()
