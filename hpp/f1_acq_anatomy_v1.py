#!/usr/bin/env python3
"""F1 acquisition anatomy v1 (package P2): the within-day and calendar acquisition components of leg PWV, studied
WITHOUT changing the atlas (HPP sandbox; writes aggregates only).

The atlas (--atlas, default f1_atlas_v1_5.py beside this file, else f1_atlas_v1.py beside it) is imported and
A.build_frame(root, None) gives the v1.5 frame.  From its raw device beds this module rebuilds, with the atlas's own
functions, the quarter-adjusted beds (v1.3, A.calendar_adjust), the day-adjusted beds (v1.4, A.day_adjust) and the v1.5
beds (A.session_adjust; checked equal to the frame).  Every new correction is computed here and only evaluated.

Definitions.  r = residual of a variable on 1, age, (age-50)^2, sex and calendar-quarter fixed effects (quarters with
>= 20 rows); PWV, ABI, IMT and retinal artery width enter quarter-adjusted (atlas v1.3).  Same-day pairs = unordered
pairs of different, unrelated people (atlas relation components) measured on the same day.  Kernel: mean residual
product of same-day pairs by |dt| bin = b0 + s k(dt/tau), b0 >= 0 (day), s >= 0 (within-day), WLS over bins with
>= 100 pairs (weights = pairs), shapes exp k = e^-x, Gaussian k = e^(-x^2/2), Matern-3/2; tau on a 40-point grid
5-1440 min; z = s / jackknife SE (20 groups of days, tau fixed); tau CI from the jackknife of log tau.  Shares are /var(r).
Clock hour = local (Asia/Jerusalem) time of the vascular acquisition; hour spline = restricted cubic, 4 knots at
hour quantiles 5/35/65/95 %.  Kriging = the atlas leave-one-out day + kernel BLUP (A.krige_day generalised to shapes).

Blocks
  K  nested kernels for quarter-adjusted PWV on one complete sample: K0 age/sex (+quarter); K1 + clock-hour spline;
     K2 + BMI, sitting SBP/DBP, sitting HR, antihypertensives; K3 both.  Per model and shape: tau [CI], day and kernel
     shares, z, weighted SSE, leave-one-out kriging MSE / var (lower = better kernel), bins; hour curve f(h) - f(10).
  V  the same kernel (K0, K1) for device brachial and ankle SBP, ABI, sitting SBP/DBP/HR (separate cuff), ECG HR, BMI,
     height, IMT, retinal artery width, on the vascular clock and on each variable's own timestamp.
  X  composition-free day-process correlation of two variables: same-day stranger cross-covariance r_a(i) r_b(j) over
     sqrt(auto-cov a * auto-cov b), all same-day pairs and |dt| <= 60 min, jackknife SE; day-BLUP correlations.
  W  within person: d(variable) ~ d(clock hour) + baseline age + sex + interval (retest pairs); between-person hour
     slopes (K0 and K2 + linear hour, day-clustered SE) for comparison.
  P  placebo kernels: clock times shuffled within day; day labels shuffled within quarter (rows keep their clock time).
  J  one joint model: calendar process = covariance of different-day stranger pairs 1-7 days apart (jackknife SE; the
     full lag profile and a c0 + sc exp(-lag/L) fit, L >= 7 d, are reported too), clinic-day and within-day kernel from
     same-day pair products net of it, all as shares of one residual variance; Kish effective N.
  CP change points (penalised optimal partitioning, min segment 15 clinic days) on daily medians of age/sex residuals,
     day-order permutation null, step vs smooth-spline fit; quarterly medians; vascular_health device/version fields.
  E  PWV retest ICC, incident-hypertension OR (atlas S6 definition) and SBP->PWV slopes for raw, v1.3, v1.4, v1.5 and
     alternative corrections: g15 (v1.5 with a Gaussian free-tau kernel), h16 (explicit hour curve from K3 only),
     h16e / h16g (explicit hour + exponential / Gaussian residual kernel); 2-fold cross-fitting over clinic days;
     placebo corrections (real parameters, day labels shuffled within quarter); paired participant bootstrap of
     delta ICC and delta log OR.
  COVID  K0 kernels and the E metrics without measurements in 2020-04-01..2021-06-30.
Privacy (as the atlas): aggregates only; counts rounded to 10, cells < 20 suppressed, pair counts >= 1e6 as text;
quantiles 5-95 % only; no participant ids or per-day values; every output is scanned for 7+ digit runs and ids.
Run (sandbox pheno env, from <WORK_DIR>):
  OMP_NUM_THREADS=4 python f1_acq_anatomy_v1.py \
      --root <HPP_DATA_ROOT> --atlas f1_atlas_v1_5.py --out f1_acq_anatomy_out_v1
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):  # many small problems: avoid oversubscription
    os.environ.setdefault(_v, "4")
import argparse
import copy
import hashlib
import importlib.util
import json
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

VERSION = "f1_acq_anatomy_v1.1 2026-10-04"  # v1.1 = code review fixes (see VALIDATION.md)
A = None  # atlas module (set_atlas)
LONG = re.compile(r"\d{7,}")
ATLAS_API = ["build_frame", "rd", "rc", "rcp", "cnt", "qtiles", "pct", "zs", "fnum", "log", "LOG", "MINCELL",
             "MAIN_STAGES", "LATER_STAGES", "TZ", "VERSION", "CAL_BEDS", "calendar_quarter", "calendar_adjust",
             "day_adjust", "session_adjust", "kernel_fit", "krige_day", "relation_components", "to_local", "icc1",
             "wide", "logit", "auc", "fit", "coef", "acute_deviation", "calibration_data", "rc_acute",
             "one_row_per_participant", "s6_incident", "DIAG", "SESSION_EDGES", "SESSION_TAUS", "SESSION_JK",
             "SESSION_MIN_BIN", "SESSION_MIN_Z", "SESSION_MAX_NB"]
TAUS = tuple(float(x) for x in np.round(np.geomspace(5.0, 1440.0, 40), 2))
EDGES = (0, 5, 10, 15, 20, 30, 45, 60, 75, 90, 105, 120, 150, 180, 210, 240, 300, 360, 480)
SHAPES = ("exp", "gau", "m32")
MIN_BIN = 100
JK = 20
MIN_Z = 3.0
NB_MAX = 60
KNOT_Q = (0.05, 0.35, 0.65, 0.95)
K2COV = ["bmi", "sbp_sit", "dbp_sit", "hr_sit", "antihtn"]
MODELS = ("K0", "K1", "K2", "K3")
HOURS = (8.0, 9.0, 11.0, 12.0, 13.0)  # hour-curve contrasts vs 10:00
LAG_EDGES = (1, 8, 31, 92, 183, 366, 731)  # day-lag bins (days), last open
LAG_L = (7, 14, 30, 60, 91, 182, 365, 730)  # >= the first lag bin (1-7 d): no extrapolation below it
CP_PEN = 3.0
CP_MINSEG = 15
CP_MIN_N = 3
COVID = ("2020-04-01", "2021-06-30")
# label, variable, own clock
PANEL = [("pwv", "pwv_q", "vas"), ("dev_bsbp", "dev_bsbp", "vas"), ("dev_asbp", "dev_asbp", "vas"),
         ("abi", "abi_q", "vas"), ("sbp_sit", "sbp_sit", "bp"), ("dbp_sit", "dbp_sit", "bp"), ("hr_sit", "hr_sit", "bp"),
         ("ecg_hr", "ecg_hr", "ecg"), ("bmi", "bmi", "anthro"), ("height", "height", "anthro"),
         ("imt", "imt_q", "carotid"), ("art_w", "art_w_q", "fundus")]
OWN = {"ecg": "ecg/ecg.parquet", "anthro": "anthropometrics/anthropometrics.parquet",
       "carotid": "carotid_ultrasound/carotid_ultrasound.parquet", "fundus": "fundus/fundus.parquet"}
CP_VARS = [("pwv_raw", "pwv_raw"), ("dev_bsbp", "dev_bsbp"), ("dev_asbp", "dev_asbp"), ("abi_raw", "abi_raw"),
           ("sbp_sit", "sbp_sit"), ("hr_sit", "hr_sit"), ("height", "height")]
SECTIONS = ("K", "V", "X", "W", "P", "J", "CP", "E", "COVID")
DEVICE_RX = re.compile(r"device|software|version|firmware|serial|operator|station|machine|model|site|cohort|"
                       r"technician|room|location|instrument", re.I)
STAFF_RX = re.compile(r"operator|technician|nurse|staff|user|examiner|room|station|location", re.I)  # labels masked


# ============================================================================ atlas, logging, privacy

def set_atlas(path):
    global A
    spec = importlib.util.spec_from_file_location("f1_atlas", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["f1_atlas"] = m
    spec.loader.exec_module(m)
    miss = [x for x in ATLAS_API if not hasattr(m, x)]
    if miss:
        raise SystemExit(f"atlas module lacks {miss}; use atlas v1.5")
    A = m
    return m


def log(msg):
    A.log(msg)


def privacy_check(text, pids):
    m = LONG.search(text)
    if m:
        raise SystemExit(f"PRIVACY ABORT: digit run '{m.group(0)[:3]}...' in output")
    for p in pids:  # ids of 7+ digits are caught above
        if not (p.isdigit() and len(p) >= 7) and len(p) >= 4 and p in text:
            raise SystemExit("PRIVACY ABORT: participant id in output")


def err(ex):
    tb = traceback.extract_tb(ex.__traceback__)
    ln = tb[-1].lineno if tb else -1
    return f"{type(ex).__name__}: {str(ex)[:160]} (line {ln})"


def rc(n):
    return A.rc(n)


def fl(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


# ============================================================================ frame preparation

def restore_raw(fr):
    f0 = fr.copy()
    for c in A.CAL_BEDS:
        if c + "_raw" in f0.columns:
            f0[c] = f0[c + "_raw"]
            f0 = f0.drop(columns=[c + "_raw"])
    return f0


def clock_of(ts):
    """tz-aware or naive (UTC) timestamps -> (UTC day string, absolute minutes, local clock minutes)."""
    ts = pd.to_datetime(ts, errors="coerce", utc=True)
    if ts.notna().sum() == 0:
        n = len(ts)
        return pd.Series([None] * n, dtype=object), np.full(n, np.nan), np.full(n, np.nan)
    ts = ts.dt.as_unit("ns")
    day = ts.dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    tm = ((ts - ts.min()).dt.total_seconds() / 60.0).to_numpy(float)
    lt = A.to_local(ts)
    clk = (lt.dt.hour * 60.0 + lt.dt.minute + lt.dt.second / 60.0 + lt.dt.microsecond / 6e7).to_numpy(float)
    return day.where(ts.notna(), None), tm, clk


def own_ts(root, fr, key):
    """Earliest collection_timestamp per (participant, main stage) of an own-clock table, aligned to fr rows."""
    if key == "bp":
        return fr["bp_ts"] if "bp_ts" in fr.columns else None
    rel = OWN.get(key)
    p = os.path.join(root, rel) if (root and rel) else None
    if not p or not os.path.exists(p):
        return None
    if "collection_timestamp" not in pq.read_schema(p).names:
        return None
    d = A.rd(root, rel, columns=["collection_timestamp"])
    if d is None or "participant_id" not in d.columns or "research_stage" not in d.columns:
        return None
    d = d[d["research_stage"].isin(A.MAIN_STAGES)]
    d = d.assign(_ts=pd.to_datetime(d["collection_timestamp"], errors="coerce", utc=True))
    g = d.groupby(["participant_id", "research_stage"])["_ts"].min().reset_index()
    m = fr[["participant_id", "research_stage"]].astype(str).merge(g, how="left", on=["participant_id", "research_stage"])
    return m["_ts"]


def prep(fr, medflags, htn, root):
    """Build the analysis context: arrays aligned to the v1.5 frame rows plus the v1.3/v1.4/v1.5 PWV versions."""
    t0 = time.time()
    fr = fr.reset_index(drop=True)
    diag0 = copy.deepcopy(A.DIAG)
    fq = A.calendar_adjust(restore_raw(fr))
    fd = A.day_adjust(fq)
    f15 = A.session_adjust(fq, root)
    S = {"fr": fr, "fq": fq, "medflags": medflags, "htn": htn, "root": root, "diag": diag0, "n": len(fr), "v": {},
         "info": {}}
    a, b = f15["pwv"].to_numpy(float), fr["pwv"].to_numpy(float)
    same_na = bool((np.isnan(a) == np.isnan(b)).all())
    S["info"]["v15_reproduced_maxdiff"] = float(np.nanmax(np.abs(a - b))) if np.isfinite(a).any() else None
    S["info"]["v15_reproduced_same_missing"] = same_na
    pid = fr["participant_id"].astype(str).to_numpy()
    S["pid"] = pid
    S["pcode"] = pd.factorize(pid)[0]
    comp = A.relation_components(root)
    S["info"]["relations_excluded"] = comp is not None
    S["cid"] = pd.factorize(np.array([comp.get(p, p) for p in pid] if comp else pid, dtype=object))[0]
    S["stage"] = fr["research_stage"].astype(str).to_numpy()
    q, t = A.calendar_quarter(fr)
    S["quarter"] = q.where(~q.astype(str).str.contains("NA"), None).to_numpy(dtype=object)
    S["qcode"] = pd.factorize(S["quarter"])[0]  # -1 = missing
    tn = pd.to_datetime(t, errors="coerce")
    dayv = tn.dt.strftime("%Y-%m-%d")
    S["day"] = dayv.where(tn.notna(), None).to_numpy(dtype=object)
    S["dcode"] = pd.factorize(S["day"])[0]
    S["dix"] = ((tn.dt.normalize() - pd.Timestamp("2018-01-01")).dt.days).to_numpy(float)
    S["date"] = tn.dt.normalize()
    S["clk"] = {}
    if "vas_ts" in fr.columns:
        _, tm, ck = clock_of(fr["vas_ts"])
    else:
        tm, ck = np.full(len(fr), np.nan), np.full(len(fr), np.nan)
    S["clk"]["vas"] = {"dcode": S["dcode"], "tm": tm, "clock": ck}
    for key in ("bp", "ecg", "anthro", "carotid", "fundus"):
        try:
            ts = own_ts(root, fr, key)
            if ts is None:
                continue
            dy, tmk, ckk = clock_of(ts)
            S["clk"][key] = {"dcode": pd.factorize(dy.to_numpy(dtype=object))[0], "tm": tmk, "clock": ckk}
        except Exception as ex:
            S["info"][f"clock_{key}_error"] = err(ex)
    S["hour"] = S["clk"]["vas"]["clock"] / 60.0
    V = S["v"]

    def col(f, c):
        return pd.to_numeric(f[c], errors="coerce").to_numpy(float) if c in f.columns else np.full(len(f), np.nan)
    V["pwv_q"], V["pwv_raw"], V["pwv_d14"], V["pwv_v15"] = col(fq, "pwv"), col(fq, "pwv_raw"), col(fd, "pwv"), col(fr, "pwv")
    V["abi_q"], V["abi_raw"] = col(fq, "abi_min"), col(fq, "abi_min_raw")
    V["imt_q"], V["art_w_q"] = col(fq, "imt"), col(fq, "art_w")
    for c in ("dev_bsbp", "dev_asbp", "sbp_sit", "dbp_sit", "hr_sit", "ecg_hr", "bmi", "height", "antihtn", "age",
              "sex_male", "acq_hour"):
        V[c] = col(fr, c)
    V["sex"] = V["sex_male"]
    # v1.1: K2 composition covariates actually available (>= 50 % of PWV rows); a missing table/column drops that
    # covariate (reported) instead of emptying the K sample and every block built on it
    okp = np.isfinite(V["pwv_q"])
    S["k2cov"] = [c for c in K2COV if okp.any() and np.isfinite(V[c][okp]).mean() >= 0.5]
    S["info"]["k2_dropped"] = [c for c in K2COV if c not in S["k2cov"]]
    S["info"]["n_rows_pwv_timed"] = rc((np.isfinite(V["pwv_q"]) & np.isfinite(tm)).sum())
    S["info"]["n_days_pwv"] = rc(len(np.unique(S["dcode"][np.isfinite(V["pwv_q"]) & (S["dcode"] >= 0)])))
    ok = np.isfinite(V["pwv_q"]) & (S["dcode"] >= 0)
    nd = pd.Series(S["dcode"][ok]).value_counts()
    S["info"]["pwv_per_day"] = A.qtiles(nd) if len(nd) >= A.MINCELL else None
    S["info"]["hour_pwv"] = A.qtiles(pd.Series(S["hour"][np.isfinite(V["pwv_q"])]))
    log(f"prep: {A.cnt(len(fr))} rows; v1.5 reproduced max diff {S['info']['v15_reproduced_maxdiff']}; "
        f"{time.time() - t0:.1f}s")
    return S


# ============================================================================ design, OLS, splines

def rcs(x, knots):
    """Restricted cubic spline basis (Harrell): x and K-2 non-linear terms."""
    k = np.asarray(knots, float)
    x = np.asarray(x, float)
    cols = [x]
    den = (k[-1] - k[0]) ** 2
    for j in range(len(k) - 2):
        t = (np.clip(x - k[j], 0, None) ** 3
             - np.clip(x - k[-2], 0, None) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
             + np.clip(x - k[-1], 0, None) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])) / den
        cols.append(t)
    return np.column_stack(cols)


def knots_of(h):
    h = np.asarray(h, float)
    h = h[np.isfinite(h)]
    k = np.quantile(h, KNOT_Q)
    if len(np.unique(np.round(k, 6))) < len(KNOT_Q):
        k = np.linspace(h.min(), h.max(), len(KNOT_Q))
    return k


def xmat(S, rows, model="K0", quarter=True, knots=None, hour=None, linear_hour=False, exclude=()):
    """Design for residual models. Columns: 1, age, (age-50)^2, sex [, quarter dummies] [, hour] [, K2 covariates]."""
    V = S["v"]
    a = V["age"][rows]
    cols = [np.ones(len(rows)), a, (a - 50.0) ** 2, V["sex"][rows]]
    names = ["const", "age", "age2", "sex"]
    if quarter:
        qc = S["qcode"][rows]
        cnt = pd.Series(qc[qc >= 0]).value_counts()
        keep = sorted(cnt[cnt >= A.MINCELL].index)
        for q in keep[1:]:
            cols.append((qc == q).astype(float))
            names.append(f"q{q}")
    hcols = []
    if model in ("K1", "K3") or linear_hour:
        h = (S["hour"] if hour is None else hour)[rows]
        if linear_hour:
            B = h[:, None]
        else:
            B = rcs(h, knots)
        for j in range(B.shape[1]):
            hcols.append(len(cols))
            cols.append(B[:, j])
            names.append(f"h{j}")
    if model in ("K2", "K3"):
        for c in [c for c in S.get("k2cov", K2COV) if c not in exclude]:
            cols.append(V[c][rows])
            names.append(c)
    return np.column_stack(cols), names, hcols


def ols_cl(y, X, cl=None):
    """OLS with HC1 or cluster-robust (CR1) covariance. Returns b, cov, residuals."""
    n, k = X.shape
    XtX = np.linalg.pinv(X.T @ X)
    b = XtX @ (X.T @ y)
    e = y - X @ b
    if cl is None:
        meat = (X * (e ** 2)[:, None]).T @ X * (n / max(n - k, 1))
    else:
        U = pd.DataFrame(X * e[:, None]).groupby(np.asarray(cl)).sum().to_numpy()
        G = len(U)
        meat = U.T @ U * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return b, XtX @ meat @ XtX, e


def rows_ok(S, need, extra=None):
    ok = np.ones(S["n"], bool)
    for c in need:
        ok &= np.isfinite(S["v"][c])
    if extra is not None:
        ok &= extra
    return np.flatnonzero(ok)


# ============================================================================ pairs, kernels, kriging

def kern(d, shape, tau):
    x = np.asarray(d, float) / tau
    with np.errstate(invalid="ignore", over="ignore"):
        if shape == "exp":
            return np.exp(-x)
        if shape == "gau":
            return np.exp(-0.5 * x * x)
        if shape == "m32":
            a = np.sqrt(3.0) * x
            return (1.0 + a) * np.exp(-a)
    raise ValueError(shape)


def day_groups(dcode, rows, min_size=2):
    rows = np.asarray(rows)
    rows = rows[dcode[rows] >= 0]
    if not len(rows):
        return []
    o = rows[np.argsort(dcode[rows], kind="stable")]
    gs = np.split(o, np.flatnonzero(np.diff(dcode[o])) + 1)
    return [g for g in gs if len(g) >= min_size]


def pairs_of(S, rows, dcode, t=None):
    """Unordered same-day pairs among rows (excluding same person / relation component); with t, only timed rows."""
    rows = np.asarray(rows)
    if t is not None:
        rows = rows[np.isfinite(t[rows])]
    I, J = [np.zeros(0, int)], [np.zeros(0, int)]
    cid, pc = S["cid"], S["pcode"]
    for g in day_groups(dcode, rows):
        iu, ju = np.triu_indices(len(g), 1)
        a, b = g[iu], g[ju]
        k = (cid[a] != cid[b]) & (pc[a] != pc[b])
        I.append(a[k])
        J.append(b[k])
    return np.concatenate(I), np.concatenate(J)


def _wls(nb, sp, ke, min_bin=MIN_BIN):
    """Atlas v1.5 WLS of the bin mean products on [1, mean kernel] with b0, s >= 0. Returns (sse, b0, s) or None."""
    use = nb >= min_bin
    if use.sum() < 3:
        return None
    w = nb[use]
    cv = sp[use] / w
    if ke is None:
        b0 = max(np.average(cv, weights=w), 0.0)
        return float((w * (cv - b0) ** 2).sum()), float(b0), 0.0
    E = ke[use] / w
    Xk = np.column_stack([np.ones(len(w)), E])
    bt = np.linalg.lstsq((Xk.T * w) @ Xk, (Xk.T * w) @ cv, rcond=None)[0]
    if bt.min() < 0:
        cands = [np.array([max(np.average(cv, weights=w), 0.0), 0.0]),
                 np.array([0.0, max((w * E * cv).sum() / (w * E * E).sum(), 0.0)])]
        bt = min(cands, key=lambda q: (w * (cv - Xk @ q) ** 2).sum())
    return float((w * (cv - Xk @ bt) ** 2).sum()), float(bt[0]), float(bt[1])


def _vertex(lx, y, k):
    """log tau of the parabola through the profile minimum k and its two grid neighbours (clipped to them); the grid
    value at an edge or when the neighbours are not finite.  v1.1: the jackknife of a grid argmin is degenerate (all
    replicates on one grid point gave a zero-width CI), so the tau CI uses these interpolated minima."""
    if k <= 0 or k >= len(lx) - 1 or not np.all(np.isfinite(y[k - 1:k + 2])):
        return float(lx[k])
    x0, x1, x2 = lx[k - 1:k + 2]
    y0, y1, y2 = y[k - 1:k + 2]
    den = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
    if not a > 0:
        return float(x1)
    return float(np.clip(-b / (2 * a), x0, x2))


def kfit(dt, pr, grp, V, shapes=SHAPES, taus=TAUS, edges=EDGES, njk=JK, jk_tau=True, min_bin=MIN_BIN):
    """Kernel fits of cov(dt) = b0 + s k(dt/tau) for each shape; day-only fit; jackknife z (tau fixed) and tau CI
    (jackknife of the interpolated profile minimum in log tau, among fits with s > 0)."""
    edges = np.asarray(edges, float)
    nbin = len(edges)
    b = np.searchsorted(edges, dt, side="right") - 1
    gi = grp.astype(int) * nbin + b
    G = njk
    NB = np.bincount(gi, minlength=G * nbin).reshape(G, nbin).astype(float)
    SP = np.bincount(gi, weights=pr, minlength=G * nbin).reshape(G, nbin)
    nb, sp = NB.sum(0), SP.sum(0)
    out = {"V": float(V), "n_pairs": A.rcp(len(dt)), "fits": {}}
    d0 = _wls(nb, sp, None, min_bin)
    if d0 is None:
        out["status"] = "fewer than 3 bins with enough pairs"
        return out
    out["day_only"] = {"day": d0[1] / V, "sse": d0[0] / V ** 2}
    dj = [(_wls(nb - NB[g], sp - SP[g], None, min_bin) or (0, np.nan, 0))[1] for g in range(G)]
    dj = np.array(dj, float)
    if np.isfinite(dj).sum() > 2:
        out["day_only"]["day_se"] = float(np.sqrt((G - 1) / G * np.nansum((dj - np.nanmean(dj)) ** 2)) / V)
    for shape in shapes:
        KS = np.stack([np.bincount(gi, weights=kern(dt, shape, tau), minlength=G * nbin).reshape(G, nbin)
                       for tau in taus])  # tau x G x bin
        best, prof, profp = None, [], []
        for it, tau in enumerate(taus):
            r = _wls(nb, sp, KS[it].sum(0), min_bin)
            prof.append(r[0] if r else np.nan)
            profp.append(r[0] if (r is not None and r[2] > 0) else np.inf)
            if r is not None and (best is None or r[0] < best[0] - 1e-15):
                best = (r[0], it, r[1], r[2])
        if best is None:
            continue
        sse, it, b0, s = best
        tau = taus[it]
        fit = {"tau": tau, "day": b0 / V, "share": s / V, "b0": b0, "s": s, "sse": sse / V ** 2,
               "edge": bool(it in (0, len(taus) - 1)), "z": 0.0, "lag0": (b0 + s) / V,
               "decay": float(s * (kern(10.0, shape, tau) - kern(300.0, shape, tau)) / V)}
        if s > 0:
            sj = np.array([(_wls(nb - NB[g], sp - SP[g], KS[it].sum(0) - KS[it][g], min_bin) or (0, 0, np.nan))[2]
                           for g in range(G)], float)
            ok = np.isfinite(sj)
            if ok.sum() > 2:
                se = np.sqrt((ok.sum() - 1) / ok.sum() * ((sj[ok] - sj[ok].mean()) ** 2).sum())
                fit["z"] = float(s / se) if se > 0 else 0.0
                fit["share_se"] = float(se / V)
        if jk_tau and s > 0:
            lx = np.log(np.asarray(taus, float))
            pp = np.asarray(profp, float)
            c0 = _vertex(lx, pp, int(np.argmin(pp)))
            fit["tau_v"] = float(np.exp(c0))
            lt = []
            for g in range(G):
                yj = np.full(len(taus), np.inf)
                for k2 in range(len(taus)):
                    r = _wls(nb - NB[g], sp - SP[g], KS[k2].sum(0) - KS[k2][g], min_bin)
                    if r is not None and r[2] > 0:
                        yj[k2] = r[0]
                if np.isfinite(yj).any():
                    lt.append(_vertex(lx, yj, int(np.argmin(yj))))
            if len(lt) > 2:
                lt = np.array(lt)
                se = np.sqrt((len(lt) - 1) / len(lt) * ((lt - lt.mean()) ** 2).sum())
                tq = float(stats.t.ppf(0.975, len(lt) - 1))  # jackknife with G groups: t(G - 1) quantile
                fit["tau_lse"] = float(se)
                fit["tau_lo"] = float(np.exp(c0 - tq * se))
                fit["tau_hi"] = float(np.exp(c0 + tq * se))
        fit["profile_sse"] = {f"{taus[k]:g}": (p / V ** 2 if np.isfinite(p) else None) for k, p in enumerate(prof)
                              if k % 4 == 0 or k == it}
        out["fits"][shape] = fit
    if out["fits"]:
        out["best_shape"] = min(out["fits"], key=lambda k: out["fits"][k]["sse"])
    return out


def bins_r(dt, pr, V, edges=None):
    edges = np.asarray(A.SESSION_EDGES if edges is None else edges, float)
    labs = [f"{int(a)}-{int(b)}" for a, b in zip(edges[:-1], edges[1:])] + [f"{int(edges[-1])}+"]
    b = np.searchsorted(edges, dt, side="right") - 1
    nb = np.bincount(b, minlength=len(edges))
    cv = np.bincount(b, weights=pr, minlength=len(edges)) / np.maximum(nb, 1)
    return {lab: {"n": A.rcp(nb[k]), "r": float(cv[k] / V) if nb[k] >= A.MINCELL else None} for k, lab in enumerate(labs)}


def krige(r, t, s, tau, V, b0, shape="exp", m=NB_MAX):
    """A.krige_day generalised to kernel shapes: leave-one-out BLUP of the day + within-day artefact."""
    n = len(r)

    def kk(d):
        return np.nan_to_num(kern(d, shape, tau), nan=0.0, posinf=0.0)
    if n - 1 <= m:
        K = b0 + s * kk(np.abs(t[:, None] - t[None, :]))
        np.fill_diagonal(K, V)
        Q = np.linalg.inv(K)
        return r - (Q @ r) / np.diag(Q)
    out = np.empty(n)
    for i in range(n):
        d = np.nan_to_num(np.abs(t - t[i]), nan=np.inf)
        d[i] = -1.0
        nb = np.argsort(d, kind="stable")[1:m + 1]
        K = b0 + s * kk(np.abs(t[nb][:, None] - t[nb][None, :]))
        np.fill_diagonal(K, V)
        out[i] = (b0 + s * kk(d[nb])) @ np.linalg.solve(K, r[nb])
    return out


def krige_all(r_full, groups, t, s, tau, V, b0, shape):
    """Kriged artefact for every row in the day groups (r_full indexed by frame row)."""
    ah = np.zeros(len(r_full))
    if b0 <= 0 and s <= 0:
        return ah
    f = min(1.0, 0.95 * V / (b0 + s))
    for g in groups:
        ah[g] = krige(r_full[g], t[g], s * f, tau, V, b0 * f, shape)
    return ah


def loo_err(r_full, groups, t, V, fitd, shape):
    """Leave-one-out kriging squared errors / V for rows in day groups (>= 2 rows), aligned to concat(groups)."""
    rows = np.concatenate(groups) if groups else np.zeros(0, int)
    if not len(rows):
        return None
    ah = krige_all(r_full, groups, t, fitd.get("s", 0.0), fitd.get("tau", 60.0), V, fitd["b0"], shape)
    return (r_full[rows] - ah[rows]) ** 2 / V


def loo_mse(r_full, groups, t, V, fitd, shape):
    e = loo_err(r_full, groups, t, V, fitd, shape)
    return None if e is None else float(np.mean(e))


def paired_diff(ea, eb, dc):
    """Mean of per-row differences ea - eb with a day-clustered SE (days = independent units)."""
    if ea is None or eb is None or len(ea) != len(eb) or len(ea) < 50:
        return None
    d = ea - eb
    m = float(d.mean())
    sd = pd.Series(d).groupby(dc).agg(["sum", "count"])
    u = sd["sum"].to_numpy() - sd["count"].to_numpy() * m
    D = len(u)
    se = float(np.sqrt(D / max(D - 1, 1) * (u ** 2).sum()) / len(d))
    return {"d": m, "se": se, "z": m / se if se > 0 else None}


def kernel_block(S, rows, r, clock="vas", shapes=SHAPES, loo=False, jk_tau=True, dcode=None, t=None, bins=True):
    """Fit kernels of residual r (aligned to rows) on same-day stranger pairs of a clock."""
    ck = S["clk"][clock]
    dcode = ck["dcode"] if dcode is None else dcode
    t = ck["tm"] if t is None else t
    full = np.full(S["n"], np.nan)
    full[rows] = r
    I, J = pairs_of(S, rows, dcode, t)
    if len(I) < 3 * MIN_BIN:
        return {"status": "too few pairs", "n_pairs": A.rcp(len(I))}
    V = float(np.var(r))
    dt = np.abs(t[I] - t[J])
    pr = full[I] * full[J]
    out = kfit(dt, pr, dcode[I] % JK, V, shapes=shapes, jk_tau=jk_tau)
    out["n_rows"] = rc(len(rows))
    out["n_days"] = rc(len(np.unique(dcode[rows][dcode[rows] >= 0])))
    if bins:
        out["bins"] = bins_r(dt, pr, V)
    if loo and out.get("fits"):
        groups = day_groups(dcode, rows)
        dc = dcode[np.concatenate(groups)] if groups else None
        E = {"day_only": loo_err(full, groups, t, V, {"b0": out["day_only"]["day"] * V, "s": 0.0, "tau": 60.0}, "exp")}
        for sh, f in out["fits"].items():
            E[sh] = loo_err(full, groups, t, V, f, sh)
        lo = {k: (float(np.mean(e)) if e is not None else None) for k, e in E.items()}
        if "gau" in E and "exp" in E:
            lo["gau_minus_exp"] = paired_diff(E["gau"], E["exp"], dc)
        if "exp" in E:
            lo["exp_minus_day"] = paired_diff(E["exp"], E["day_only"], dc)
        out["loo"] = lo
    return out


# ============================================================================ K nested kernels

def k_sample(S):
    if "K" in S:
        return S["K"]
    V = S["v"]
    rows = rows_ok(S, ["pwv_q", "age", "sex"] + S.get("k2cov", K2COV), (S["dcode"] >= 0)
                   & np.isfinite(S["clk"]["vas"]["tm"]) & np.isfinite(S["hour"]))
    if len(rows) < 200:
        raise ValueError(f"K sample too small ({A.cnt(len(rows))} rows with PWV, timestamp and covariates)")
    knots = knots_of(S["hour"][rows])
    out = {"rows": rows, "knots": knots, "resid": {}, "coef": {}}
    for m in MODELS:
        X, names, hc = xmat(S, rows, m, knots=knots)
        b, C, e = ols_cl(V["pwv_q"][rows], X, S["dcode"][rows])
        out["resid"][m] = e
        out["coef"][m] = (b, C, names, hc)
    S["K"] = out
    return out


def hour_curve(b, C, hc, knots, hours=HOURS, ref=10.0):
    B = rcs(np.array(list(hours) + [ref]), knots)
    D = B[:-1] - B[-1]
    est = D @ b[hc]
    se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", D, C[np.ix_(hc, hc)], D), 0, None))
    return {f"{h:g}": {"b": float(e), "se": float(s)} for h, e, s in zip(hours, est, se)}


def block_K(S, a):
    K = k_sample(S)
    rows = K["rows"]
    out = {"n_rows": rc(len(rows)), "knots_h": [float(x) for x in K["knots"]], "models": {}}
    ky = S["v"]["pwv_q"][rows]
    for m in MODELS:
        r = K["resid"][m]
        res = kernel_block(S, rows, r, loo=True)
        res["r2_model"] = float(1 - np.var(r) / np.var(ky))
        b, C, names, hc = K["coef"][m]
        if hc:
            res["hour_curve"] = hour_curve(b, C, hc, K["knots"])
            hh = rcs(S["hour"][rows], K["knots"]) @ b[hc]
            res["hour_var_share"] = float(np.var(hh) / np.var(ky))
        out["models"][m] = res
        f = res.get("fits", {})
        log(f"K {m}: " + "; ".join(f"{sh} tau {v['tau']:g} s {v['share']:.3f} z {v['z']:.1f}" for sh, v in f.items()))
    # atlas reproduction: the v1.5 exponential fit (atlas grid and bins) on K0 residuals of the same sample
    try:
        r = K["resid"]["K0"]
        full = np.full(S["n"], np.nan)
        full[rows] = r
        t = S["clk"]["vas"]["tm"]
        I, J = pairs_of(S, rows, S["dcode"], t)
        nb, cov, best = A.kernel_fit(np.abs(t[I] - t[J]), full[I] * full[J], float(np.var(r)), S["dcode"][I] % A.SESSION_JK)
        if best is not None:
            V0 = float(np.var(r))
            out["atlas_grid_K0"] = {"tau": best[0], "day": best[1] / V0, "share": best[2] / V0, "z": best[3]}
    except Exception as ex:
        out["atlas_grid_K0"] = {"error": err(ex)}
    d = S["diag"].get("session_kernel", {}).get("pwv", {})
    out["atlas_v15_diag"] = {k: d.get(k) for k in ("tau", "day_share", "share", "z", "applied", "n_pairs", "n_days")}
    return out


# ============================================================================ V variable panel

def resid_var(S, var, clock="vas", model="K0", min_rows=200):
    ck = S["clk"].get(clock)
    if ck is None:
        return None, None
    hour = ck["clock"] / 60.0
    extra = (ck["dcode"] >= 0)
    if model == "K1":
        extra &= np.isfinite(hour)
    rows = rows_ok(S, [var, "age", "sex"], extra)
    if len(rows) < min_rows:
        return None, None
    knots = knots_of(hour[rows]) if model == "K1" else None
    X, _, _ = xmat(S, rows, model, knots=knots, hour=hour)
    b, _, e = ols_cl(S["v"][var][rows], X)
    return rows, e


def block_V(S, a):
    out = {}
    for lab, var, own in PANEL:
        if not np.isfinite(S["v"][var]).any():
            out[lab] = {"status": "missing"}
            continue
        res = {}
        for tag, clock, model in (("vas_K0", "vas", "K0"), ("vas_K1", "vas", "K1"), ("own_K0", own, "K0")):
            if tag == "own_K0" and own == "vas":
                continue
            try:
                rows, r = resid_var(S, var, clock, model)
                if rows is None:
                    res[tag] = {"status": "too few rows or no clock"}
                    continue
                kb = kernel_block(S, rows, r, clock=clock, shapes=("exp", "gau"), jk_tau=False, bins=(tag == "vas_K0"))
                ck = S["clk"][clock]
                if tag == "vas_K0":
                    bl, _, share = day_blup(r, ck["dcode"][rows])
                    kb["anova_day_share"] = share
                res[tag] = kb
            except Exception as ex:
                res[tag] = {"error": err(ex)}
        out[lab] = res
        f = res.get("vas_K0", {}).get("fits", {})
        log(f"V {lab}: " + "; ".join(f"{sh} s {v['share']:.3f} z {v['z']:.1f} tau {v['tau']:g}" for sh, v in f.items()))
    return out


# ============================================================================ X process correlations, day BLUPs

def day_blup(r, dcode):
    g = pd.DataFrame({"r": r, "d": dcode})
    g = g[g["d"] >= 0]
    st = g.groupby("d")["r"].agg(["sum", "count"])
    m = st[st["count"] >= 2]
    if len(m) < 20:
        return None, st["count"], None
    nd = m["count"].to_numpy(float)
    N, D = nd.sum(), len(nd)
    gm = g.loc[g["d"].isin(m.index), "r"]
    ss_tot = ((gm - gm.mean()) ** 2).sum()
    ss_b = ((m["sum"].to_numpy() / nd - gm.mean()) ** 2 * nd).sum()
    msw = (ss_tot - ss_b) / (N - D)
    msb = ss_b / (D - 1)
    n0 = (N - (nd ** 2).sum() / N) / (D - 1)
    vday = max(0.0, (msb - msw) / n0)
    share = float(vday / (vday + msw)) if (vday + msw) > 0 else None
    if vday <= 0:
        return st["sum"] / st["count"], st["count"], share  # unshrunk day means
    return st["sum"] / (st["count"] + msw / vday), st["count"], share


def block_X(S, a):
    labs = [p for p in PANEL if np.isfinite(S["v"][p[1]]).any()]
    R = {}
    for lab, var, _ in labs:
        rows, r = resid_var(S, var, "vas", "K0")
        if rows is None:
            continue
        full = np.full(S["n"], np.nan)
        full[rows] = r
        R[lab] = full
    names = list(R)
    t = S["clk"]["vas"]["tm"]
    allrows = np.flatnonzero((S["dcode"] >= 0) & np.isfinite(np.column_stack([R[k] for k in names])).any(axis=1))
    I, J = pairs_of(S, allrows, S["dcode"], None)
    dt = np.abs(t[I] - t[J])
    close = np.isfinite(dt) & (dt <= 60)
    grp = S["dcode"][I] % JK
    out = {"n_pairs": A.rcp(len(I)), "auto": {}, "rho": {}, "rho60": {}, "blup_r": {}}

    def stats_(x, y, mask):
        """pooled mean product over ordered pairs and its day-group jackknife replicates"""
        p1 = x[I] * y[J]
        p2 = x[J] * y[I]
        ok1, ok2 = np.isfinite(p1) & mask, np.isfinite(p2) & mask
        s = np.bincount(grp[ok1], weights=p1[ok1], minlength=JK) + np.bincount(grp[ok2], weights=p2[ok2], minlength=JK)
        n = np.bincount(grp[ok1], minlength=JK) + np.bincount(grp[ok2], minlength=JK)
        if n.sum() < 200:
            return None, None
        full_ = s.sum() / n.sum()
        jk = (s.sum() - s) / np.maximum(n.sum() - n, 1)
        return full_, jk
    var_ = {k: float(np.nanvar(R[k])) for k in names}

    def jse(jk):
        jk = jk[np.isfinite(jk)]
        return float(np.sqrt((len(jk) - 1) / len(jk) * ((jk - jk.mean()) ** 2).sum())) if len(jk) > 2 else None
    for mask, key in ((np.ones(len(I), bool), "rho"), (close, "rho60")):
        auto = {}
        for k in names:
            m, jk = stats_(R[k], R[k], mask)
            se = jse(jk) if m is not None else None
            auto[k] = (m, jk, (m / se) if (m is not None and se) else None)
            if key == "rho" and m is not None:
                out["auto"][k] = {"r": m / var_[k], "z": auto[k][2]}
        for i, k1 in enumerate(names):
            for k2 in names[i + 1:]:
                c, cj = stats_(R[k1], R[k2], mask)
                if c is None:
                    out[key][f"{k1}|{k2}"] = None
                    continue
                sc = np.sqrt(var_[k1] * var_[k2])
                rec = {"xr": float(c / sc), "xr_se": (jse(cj / sc) if cj is not None else None)}
                a1, a1j, z1 = auto[k1]
                a2, a2j, z2 = auto[k2]
                if a1 is not None and a2 is not None and a1 > 0 and a2 > 0 and (z1 or 0) >= 2 and (z2 or 0) >= 2:
                    rec["rho"] = float(c / np.sqrt(a1 * a2))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        rec["rho_se"] = jse(cj / np.sqrt(a1j * a2j))
                out[key][f"{k1}|{k2}"] = rec
    # day BLUPs (K0 residuals, vascular day), correlation over days with >= 5 rows of both
    bl = {}
    for k in names:
        rows = np.flatnonzero(np.isfinite(R[k]) & (S["dcode"] >= 0))
        b, n, share = day_blup(R[k][rows], S["dcode"][rows])
        if b is not None:
            bl[k] = (b, n)
    for i, k1 in enumerate(names):
        for k2 in names[i + 1:]:
            if k1 not in bl or k2 not in bl:
                continue
            b1, n1 = bl[k1]
            b2, n2 = bl[k2]
            dd = b1.index.intersection(b2.index)
            dd = [d for d in dd if n1[d] >= 5 and n2[d] >= 5]
            if len(dd) >= 30:
                out["blup_r"][f"{k1}|{k2}"] = {"r": float(np.corrcoef(b1[dd], b2[dd])[0, 1]), "n_days": rc(len(dd))}
    return out


# ============================================================================ W within person

def block_W(S, a):
    V, st = S["v"], S["stage"]
    s0, s1 = A.MAIN_STAGES
    out = {}
    K = k_sample(S)
    items = [("pwv_q", "pwv_q", "vas"), ("pwv_raw", "pwv_raw", "vas"), ("pwv_v15", "pwv_v15", "vas"),
             ("dev_bsbp", "dev_bsbp", "vas"), ("dev_asbp", "dev_asbp", "vas"), ("sbp_sit", "sbp_sit", "bp"),
             ("hr_sit", "hr_sit", "bp"), ("height", "height", "vas"), ("bmi", "bmi", "vas")]
    vd = pd.to_datetime(S["fr"]["visit_date"], errors="coerce")
    for lab, var, clock in items:
        try:
            ck = S["clk"].get(clock) or S["clk"]["vas"]
            h = ck["clock"] / 60.0
            df = pd.DataFrame({"p": S["pid"], "st": st, "y": V[var], "h": h, "age": V["age"], "sex": V["sex"], "vd": vd})
            df = df[np.isfinite(df["y"]) & np.isfinite(df["h"])]
            w0 = df[df["st"] == s0].drop_duplicates("p").set_index("p")
            w1 = df[df["st"] == s1].drop_duplicates("p").set_index("p")
            ix = w0.index.intersection(w1.index)
            d = pd.DataFrame({"dy": w1.loc[ix, "y"] - w0.loc[ix, "y"], "dh": w1.loc[ix, "h"] - w0.loc[ix, "h"],
                              "age0": w0.loc[ix, "age"], "sex": w0.loc[ix, "sex"],
                              "iv": (w1.loc[ix, "vd"] - w0.loc[ix, "vd"]).dt.days / 365.25}).dropna()
            res = {"n_pairs": rc(len(d))}
            if len(d) >= 100:
                X = np.column_stack([np.ones(len(d)), d["dh"], d["age0"], d["sex"], d["iv"]])
                b, C, _ = ols_cl(d["dy"].to_numpy(float), X)
                res["within"] = {"b": float(b[1]), "se": float(np.sqrt(C[1, 1]))}
                res["abs_dh"] = A.qtiles(d["dh"].abs())
                res["sd_dy"] = float(d["dy"].std())
            # between-person linear hour slopes (all rows; K0 and, for the K sample variables, K2)
            for tag, model in (("between_K0", "K0"), ("between_K2", "K2")):
                need = [var, "age", "sex"] + (S.get("k2cov", K2COV) if model == "K2" else [])
                rows = rows_ok(S, need, (ck["dcode"] >= 0) & np.isfinite(h))
                if len(rows) < 200:
                    continue
                X, names, hc = xmat(S, rows, model, hour=h, linear_hour=True, exclude=(var,))
                b, C, _ = ols_cl(V[var][rows], X, ck["dcode"][rows])
                res[tag] = {"b": float(b[hc[0]]), "se": float(np.sqrt(C[hc[0], hc[0]])), "n": rc(len(rows))}
            out[lab] = res
        except Exception as ex:
            out[lab] = {"error": err(ex)}
    out["note"] = "per clock hour, native units; within = visit-2 minus visit-1 change"
    return out


# ============================================================================ P placebo kernels

def shuffle_within(groups_key, rows, rng):
    """Permutation of rows within groups (returns new order aligned to rows)."""
    rows = np.asarray(rows)
    g = groups_key[rows]
    out = rows.copy()
    o = np.argsort(g, kind="stable")
    for blk in np.split(o, np.flatnonzero(np.diff(g[o])) + 1):
        out[blk] = rows[blk][rng.permutation(len(blk))]
    return out


def block_P(S, a, rng):
    K = k_sample(S)
    rows = K["rows"]
    clock = S["clk"]["vas"]["clock"]
    out = {"reps": a.placebo}
    for m in ("K0", "K1"):
        r = K["resid"][m]
        real = kernel_block(S, rows, r, shapes=("exp", "gau"), jk_tau=False, bins=False)
        res = {"real": {sh: {k: f.get(k) for k in ("share", "day", "z", "tau", "decay")} for sh, f in real.get("fits", {}).items()}}
        for kind in ("time_within_day", "day_within_quarter"):
            acc = {"exp": [], "gau": []}
            for _ in range(a.placebo):
                if kind == "time_within_day":
                    t = clock.copy()
                    t[rows] = clock[shuffle_within(S["dcode"], rows, rng)]
                    kb = kernel_block(S, rows, r, t=t, shapes=("exp", "gau"), jk_tau=False, bins=False)
                else:
                    dc = S["dcode"].copy()
                    dc[rows] = S["dcode"][shuffle_within(S["qcode"], rows, rng)]
                    kb = kernel_block(S, rows, r, dcode=dc, t=clock, shapes=("exp", "gau"), jk_tau=False, bins=False)
                for sh in ("exp", "gau"):
                    f = kb.get("fits", {}).get(sh)
                    if f:
                        acc[sh].append((f["share"], f["z"], f["day"], f["decay"]))
            res[kind] = {}
            for sh in ("exp", "gau"):
                x = np.array(acc[sh], float)
                rd_ = g(res, "real", sh, "decay")
                if len(x):
                    res[kind][sh] = {"p_perm": float((1 + (x[:, 3] >= rd_).sum()) / (1 + len(x))) if rd_ is not None else None,
                                     "share_mean": float(x[:, 0].mean()), "share_q95": float(np.quantile(x[:, 0], 0.95)),
                                     "z_mean": float(x[:, 1].mean()), "frac_z3": float((x[:, 1] >= MIN_Z).mean()),
                                     "day_mean": float(x[:, 2].mean()), "decay_mean": float(x[:, 3].mean()),
                                     "decay_q95": float(np.quantile(x[:, 3], 0.95))}
        out[m] = res
    return out


# ============================================================================ J joint calendar / day / within-day

def lag_cov(S, rows, r):
    """Different-day pair covariance by day lag (excluding same person / relation component across days)."""
    dix, cid = S["dix"][rows], S["cid"][rows]
    df = pd.DataFrame({"r": r, "dx": dix})
    st = df.groupby("dx")["r"].agg(["sum", "count"])
    X_, S_, N_ = st.index.to_numpy(float), st["sum"].to_numpy(), st["count"].to_numpy(float)
    iu, ju = np.triu_indices(len(st), 1)
    lag = np.abs(X_[ju] - X_[iu])
    e = np.asarray(LAG_EDGES, float)
    nb = len(e)
    b = np.searchsorted(e, lag, side="right") - 1
    ok = b >= 0
    sums = np.bincount(b[ok], weights=(S_[iu] * S_[ju])[ok], minlength=nb)
    cnts = np.bincount(b[ok], weights=(N_[iu] * N_[ju])[ok], minlength=nb)
    dps = np.bincount(b[ok], minlength=nb).astype(float)
    # day-group jackknife of the first lag bin (1-7 days): drop day pairs touching group g
    gday = np.arange(len(st)) % JK
    f1 = ok & (b == 0)
    prod1, cnt1 = (S_[iu] * S_[ju])[f1], (N_[iu] * N_[ju])[f1]
    g1, g2 = gday[iu][f1], gday[ju][f1]
    jk1 = [(prod1[(g1 != g) & (g2 != g)].sum(), cnt1[(g1 != g) & (g2 != g)].sum()) for g in range(JK)]
    dpos = {x: k for k, x in enumerate(X_)}
    ex1 = np.zeros(JK)
    exn = np.zeros(JK)
    # exclusions: pairs of the same person / relation component on different days
    o = np.argsort(cid, kind="stable")
    for g in np.split(o, np.flatnonzero(np.diff(cid[o])) + 1):
        if len(g) < 2:
            continue
        i2, j2 = np.triu_indices(len(g), 1)
        lg = np.abs(dix[g[i2]] - dix[g[j2]])
        kk = lg >= 1
        if not kk.any():
            continue
        bb = np.searchsorted(e, lg[kk], side="right") - 1
        pp = (r[g[i2]] * r[g[j2]])[kk]
        np.subtract.at(sums, bb, pp)
        np.subtract.at(cnts, bb, 1.0)
        for q_, (ia, ja) in enumerate(zip(g[i2][kk], g[j2][kk])):
            if bb[q_] == 0:
                ga, gb = gday[dpos[dix[ia]]], gday[dpos[dix[ja]]]
                for gg in range(JK):
                    if ga != gg and gb != gg:
                        ex1[gg] += pp[q_]
                        exn[gg] += 1.0
    cov = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    jv = np.array([(a_ - ex1[k]) / max(c_ - exn[k], 1.0) for k, (a_, c_) in enumerate(jk1)])
    cov1_se = float(np.sqrt((JK - 1) / JK * ((jv - jv.mean()) ** 2).sum())) if len(jv) > 2 else None
    Ek = {}
    for L in LAG_L:
        w = np.bincount(b[ok], weights=((N_[iu] * N_[ju]) * np.exp(-lag / L))[ok], minlength=nb)
        Ek[L] = w / np.maximum(np.bincount(b[ok], weights=(N_[iu] * N_[ju])[ok], minlength=nb), 1)
    return cov, dps, Ek, cnts, cov1_se


def calendar_fit(cov, dps, Ek):
    use = np.isfinite(cov) & (dps > 0)
    if use.sum() < 3:
        return None
    best = None
    w = dps[use]
    for L, E in Ek.items():
        X = np.column_stack([np.ones(use.sum()), E[use]])
        bt = np.linalg.lstsq((X.T * w) @ X, (X.T * w) @ cov[use], rcond=None)[0]
        if bt[1] < 0:
            bt = np.array([np.average(cov[use], weights=w), 0.0])
        sse = float((w * (cov[use] - X @ bt) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, L, float(bt[0]), float(bt[1]))
    return best


def joint(S, rows, r, kernel_shapes=("exp", "gau")):
    """One decomposition of var(r): calendar process (different-day pairs), then clinic day + within-day kernel fitted
    to same-day pair products net of the calendar covariance at lag 0 (b0, s >= 0)."""
    V = float(np.var(r))
    out = {"n_rows": rc(len(rows)), "V": V}
    cov, dps, Ek, cnts, cov1_se = lag_cov(S, rows, r)
    labs = [f"{a}-{b - 1}" for a, b in zip(LAG_EDGES[:-1], LAG_EDGES[1:])] + [f"{LAG_EDGES[-1]}+"]
    out["lag_r"] = {lab: (float(cov[k] / V) if np.isfinite(cov[k]) and cnts[k] >= A.MINCELL else None)
                    for k, lab in enumerate(labs)}
    cf = calendar_fit(cov, dps, Ek)
    # calendar component = covariance of different-day stranger pairs 1-7 days apart (nonparametric, jackknife SE);
    # the parametric c0 + sc exp(-lag/L) value at a 1-day lag is reported as a model-based alternative
    cal0 = max(float(cov[0]), 0.0) if np.isfinite(cov[0]) else 0.0
    out["calendar_fit_1d"] = ((cf[2] + cf[3] * np.exp(-1.0 / cf[1])) / V) if cf else None
    out["calendar_se"] = (cov1_se / V) if cov1_se is not None else None
    full = np.full(S["n"], np.nan)
    full[rows] = r
    t = S["clk"]["vas"]["tm"]
    I, J = pairs_of(S, rows, S["dcode"], t)
    if len(I) < 3 * MIN_BIN:
        out["status"] = "too few pairs"
        return out
    kb = kfit(np.abs(t[I] - t[J]), full[I] * full[J] - cal0, S["dcode"][I] % JK, V, shapes=kernel_shapes, jk_tau=False)
    if not kb.get("fits"):
        out["status"] = kb.get("status")
        return out
    sh = kb["best_shape"]
    f = kb["fits"][sh]
    out.update({"shape": sh, "tau": f["tau"], "kernel": f["share"], "z": f["z"], "decay": f["decay"],
                "calendar": cal0 / V, "calendar_L_days": cf[1] if cf else None,
                "day": f["day"], "remainder": 1.0 - cal0 / V - f["day"] - f["share"]})
    # Kish design effect for day clusters
    I, J = pairs_of(S, rows, S["dcode"], None)
    rho = float(np.mean(full[I] * full[J]) / V) if len(I) else np.nan
    nd = pd.Series(S["dcode"][rows]).value_counts().to_numpy(float)
    mw = float((nd ** 2).sum() / nd.sum())
    deff = 1.0 + (mw - 1.0) * rho
    out.update({"rho_sameday": rho, "deff": deff, "neff": rc(len(rows) / deff) if deff > 0 else None})
    return out


def block_J(S, a):
    V = S["v"]
    rows = rows_ok(S, ["pwv_raw", "pwv_q", "age", "sex"], (S["dcode"] >= 0) & np.isfinite(S["clk"]["vas"]["tm"])
                   & np.isfinite(S["hour"]) & np.isfinite(S["dix"]))
    out = {}
    knots = knots_of(S["hour"][rows])
    versions = [("raw_K0", "pwv_raw", "K0"), ("raw_K1", "pwv_raw", "K1"), ("q13_K0", "pwv_q", "K0"),
                ("v15_K0", "pwv_v15", "K0")]
    if "h16g" in S.get("Y", {}):  # alternative correction, when block E has run
        S["v"]["_h16g"] = S["Y"]["h16g"]
        versions.append(("h16g_K0", "_h16g", "K0"))
    for lab, var, m in versions:
        try:
            rr = rows[np.isfinite(V[var][rows])]
            X, _, _ = xmat(S, rr, m, quarter=False, knots=knots)
            _, _, e = ols_cl(V[var][rr], X)
            out[lab] = joint(S, rr, e)
            if lab == "raw_K0":
                Xq, _, _ = xmat(S, rr, "K0", quarter=True)
                _, _, eq = ols_cl(V[var][rr], Xq)
                out[lab]["quarter_fe_share"] = float(1 - np.var(eq) / np.var(e))
        except Exception as ex:
            out[lab] = {"error": err(ex)}
    return out


# ============================================================================ CP change points, quarterly profiles

def cp_dp(y, w, pen, minseg):
    """Optimal partitioning (weighted Gaussian change in mean), penalty per change point, minimum segment length."""
    D = len(y)
    W = np.r_[0.0, np.cumsum(w)]
    WY = np.r_[0.0, np.cumsum(w * y)]
    WY2 = np.r_[0.0, np.cumsum(w * y * y)]
    F = np.full(D + 1, np.inf)
    F[0] = -pen
    last = np.zeros(D + 1, int)
    for t in range(minseg, D + 1):
        s = np.arange(0, t - minseg + 1)
        s = s[np.isfinite(F[s])]
        if not len(s):
            continue
        ww = W[t] - W[s]
        c = F[s] + (WY2[t] - WY2[s]) - (WY[t] - WY[s]) ** 2 / ww + pen
        k = int(np.argmin(c))
        F[t], last[t] = c[k], s[k]
    cps, t = [], D
    while t > 0:
        s = last[t]
        if s > 0:
            cps.append(s)
        t = s
    return sorted(cps)


def daily(S, var):
    V = S["v"]
    rows = rows_ok(S, [var, "age", "sex"], (S["dcode"] >= 0) & np.isfinite(S["dix"]))
    if len(rows) < 500:
        return None
    X, _, _ = xmat(S, rows, "K0", quarter=False)
    _, _, e = ols_cl(V[var][rows], X)
    df = pd.DataFrame({"r": e, "dx": S["dix"][rows], "d": S["dcode"][rows]})
    g = df.groupby("dx")["r"]
    st = pd.DataFrame({"med": g.median(), "n": g.size(), "var": g.var()}).sort_index()
    st = st[st["n"] >= CP_MIN_N]
    vw = float(np.nansum(st["var"] * (st["n"] - 1)) / max((st["n"] - 1).sum(), 1))
    m, n = st["med"].to_numpy(), st["n"].to_numpy(float)
    d2 = np.diff(m) ** 2
    hn = 1.0 / n[:-1] + 1.0 / n[1:]
    v0 = max(0.0, (np.median(d2) / 0.4549 - 1.571 * vw * np.median(hn)) / 2.0)
    w = 1.0 / (v0 + 1.571 * vw / n)
    return {"dx": st.index.to_numpy(float), "m": m, "n": n, "w": w, "v0": v0, "vw": vw, "mean": float(np.nanmean(V[var][rows]))}


def cp_analyse(D_, rng, nnull):
    m, w, n, dx = D_["m"], D_["w"], D_["n"], D_["dx"]
    Dn = len(m)
    pen = CP_PEN * np.log(Dn)
    cps = cp_dp(m, w, pen, CP_MINSEG)
    bnd = [0] + cps + [Dn]
    seg = []
    for a_, b_ in zip(bnd[:-1], bnd[1:]):
        W = w[a_:b_].sum()
        seg.append((float((w[a_:b_] * m[a_:b_]).sum() / W), float(W), int(n[a_:b_].sum()), a_, b_))
    fitted = np.concatenate([np.full(b_ - a_, lv) for lv, _, _, a_, b_ in seg])
    mu = np.average(m, weights=w)
    sst = float((w * (m - mu) ** 2).sum())
    r2_cp = 1 - float((w * (m - fitted) ** 2).sum()) / sst
    q = max(3, 2 * len(cps) + 1)
    kn = np.quantile(dx, np.linspace(0.05, 0.95, q)) if q >= 3 else None
    Xs = np.column_stack([np.ones(Dn), rcs((dx - dx.min()) / 365.25, (kn - dx.min()) / 365.25)])
    bs = np.linalg.lstsq(Xs * np.sqrt(w)[:, None], m * np.sqrt(w), rcond=None)[0]
    r2_sp = 1 - float((w * (m - Xs @ bs) ** 2).sum()) / sst
    base = pd.Timestamp("2018-01-01")
    out = {"n_days": rc(Dn), "n_cp": len(cps), "pen": pen, "r2_cp": r2_cp, "r2_spline_same_df": r2_sp,
           "spline_df": q, "v_day_local": D_["v0"], "v_within": D_["vw"], "cps": [], "segments": []}
    for k, c in enumerate(cps):
        dt = base + pd.Timedelta(days=float(dx[c]))
        wk = (dt - pd.Timedelta(days=dt.weekday())).strftime("%Y-%m-%d")
        jump = seg[k + 1][0] - seg[k][0]
        z = jump / np.sqrt(1.0 / seg[k + 1][1] + 1.0 / seg[k][1])
        out["cps"].append({"week": wk, "jump": jump, "z": float(z)})
    for lv, W, nn, a_, b_ in seg:
        out["segments"].append({"level": lv + D_["mean"], "n": rc(nn), "days": rc(b_ - a_)})
    nulls = []
    for _ in range(nnull):
        p = rng.permutation(Dn)
        nulls.append(len(cp_dp(m[p], w[p], pen, CP_MINSEG)))
    if nulls:
        out["null_mean_cp"] = float(np.mean(nulls))
        out["null_frac_ge1"] = float(np.mean(np.array(nulls) >= 1))
    return out


def block_CP(S, a, rng):
    out = {"vars": {}, "quarterly": {}, "device_fields": None}
    for lab, var in CP_VARS:
        try:
            D_ = daily(S, var)
            if D_ is None:
                out["vars"][lab] = {"status": "too few rows"}
                continue
            out["vars"][lab] = cp_analyse(D_, rng, a.cp_null)
        except Exception as ex:
            out["vars"][lab] = {"error": err(ex)}
    q = pd.Series(S["quarter"])
    for lab, var in CP_VARS:
        y = pd.Series(S["v"][var])
        g = pd.DataFrame({"q": q, "y": y}).dropna()
        st = g.groupby("q")["y"].agg(["median", "size"])
        out["quarterly"][lab] = {k: {"median": float(v["median"]), "n": rc(v["size"])} for k, v in st.iterrows()
                                 if v["size"] >= A.MINCELL}
    try:
        p = os.path.join(S["root"], "vascular_health/vascular_health.parquet")
        names = pq.read_schema(p).names
        meas = {"participant_id", "research_stage", "array_index", "collection_timestamp", "collection_date", "timezone"}
        other = [c for c in names if c not in meas and not re.search(r"pressure|abi|pwv", c, re.I)]
        dev = {}
        for c in [c for c in names if DEVICE_RX.search(c)]:
            col = pq.read_table(p, columns=[c]).column(c).to_pandas()
            vc = col.astype(str).value_counts()
            if STAFF_RX.search(c):  # v1.1: staff/room labels may be names: level counts only, labels masked
                lv = {f"L{i + 1}": rc(v) for i, (k, v) in enumerate(vc.head(6).items()) if rc(v) is not None}
            else:
                lv = {re.sub(r"\d{5,}", "#", str(k))[:24]: rc(v) for k, v in vc.head(6).items() if rc(v) is not None}
            dev[c] = {"n_levels": int(min(len(vc), 999)), "top": lv}
        out["device_fields"] = {"n_columns": len(names), "other_columns": other[:20], "device_like": dev}
    except Exception as ex:
        out["device_fields"] = {"error": err(ex)}
    return out


# ============================================================================ E re-evaluation of corrections

def htn_frame(S):
    """Replica of atlas S6 at-risk frame (baseline untreated normotensive with a visit-2 row); _row = frame row."""
    fr, htn, medflags = S["fr"], S["htn"], S["medflags"]
    s0, s1 = A.MAIN_STAGES
    base = fr[fr.research_stage == s0].copy()
    base["_row"] = base.index
    base = base[(base["antihtn"] == 0) & (base["sbp_sit"] < 140) & (base["dbp_sit"] < 90)]
    if "htn_curated" in base.columns:
        base = base[base["htn_curated"].fillna(0) == 0]
    if "htn_self" in base.columns:
        base = base[base["htn_self"].fillna(0) == 0]
    v2 = fr[fr.research_stage == s1][["participant_id", "sbp_sit", "dbp_sit", "antihtn", "visit_date"]].rename(
        columns={"sbp_sit": "sbp2", "dbp_sit": "dbp2", "antihtn": "antihtn2", "visit_date": "date2"})
    d = base.merge(v2, on="participant_id", how="inner")
    if medflags is not None:
        lm = medflags[medflags["research_stage"].isin(A.LATER_STAGES)].groupby("participant_id")["antihtn"].max()
        d["antihtn_later"] = d["participant_id"].map(lm).fillna(0)
    else:
        d["antihtn_later"] = 0.0
    if htn is not None:
        lh = htn[htn["research_stage"].isin(A.LATER_STAGES)].groupby("participant_id")[["htn_curated", "htn_self"]].max()
        d["htn_later"] = d["participant_id"].map(lh.max(axis=1)).fillna(0)
    else:
        d["htn_later"] = 0.0
    d["inc_measured"] = ((d["sbp2"] >= 140) | (d["dbp2"] >= 90)).astype(float)
    d.loc[d["sbp2"].isna() & d["dbp2"].isna(), "inc_measured"] = np.nan
    d["inc_any"] = ((d["inc_measured"] == 1) | (d["antihtn2"] == 1) | (d["antihtn_later"] == 1) |
                    (d["htn_later"] == 1)).astype(float)
    return d


def or_fit(d, y, idx=None):
    """Atlas S6 per-bed logistic model for PWV values y (frame-aligned): OR per SD, CI, AUC; None below 20 events."""
    covs = ["age", "sex_male", "bmi", "sbp_sit", "dbp_sit"]
    if d is None:
        return None
    e = d.assign(_y=y[d["_row"].to_numpy()])
    if idx is not None:
        e = e.iloc[idx]
    e = e.dropna(subset=["_y", "inc_any"] + covs)
    if len(e) < 200 or e["inc_any"].sum() < A.MINCELL:
        return None
    X0 = np.column_stack([np.ones(len(e))] + [e[c].astype(float) for c in covs])
    X1 = np.column_stack([X0, A.zs(e["_y"]).to_numpy()])
    yy = e["inc_any"].to_numpy()
    b, se, p1 = A.logit(yy, X1)
    if b is None:
        return None
    return {"n": rc(len(e)), "events": rc(yy.sum()), "log_or": float(b[-1]), "se": float(se[-1]),
            "or": float(np.exp(b[-1])), "or_lo": float(np.exp(b[-1] - 1.96 * se[-1])),
            "or_hi": float(np.exp(b[-1] + 1.96 * se[-1])), "auc": float(A.auc(yy, p1))}


def eval_setup(S):
    if "EV" in S:
        return S["EV"]
    fr = S["fr"]
    s0, s1 = A.MAIN_STAGES
    ok = np.isfinite(S["v"]["pwv_q"])
    df = pd.DataFrame({"p": S["pid"], "st": S["stage"], "row": np.arange(S["n"])})[ok]
    r0 = df[df.st == s0].drop_duplicates("p").set_index("p")["row"]
    r1 = df[df.st == s1].drop_duplicates("p").set_index("p")["row"]
    ix = r0.index.intersection(r1.index)
    EV = {"i0": r0.loc[ix].to_numpy(), "i1": r1.loc[ix].to_numpy(), "htn_d": None, "slope_d": None, "cal": None,
          "art_d": None, "bp_obs": ["sbp_sit", "dbp_sit"], "errors": {}}
    # v1.1: each evaluation frame is built independently, so a missing table/column (e.g. medications) removes only
    # the metric that needs it (error note) and not the ICC and slopes
    try:
        hd = htn_frame(S)
        EV["htn_d"] = hd[["_row", "inc_any", "age", "sex_male", "bmi", "sbp_sit", "dbp_sit"]].copy()
    except Exception as ex:
        EV["errors"]["htn"] = err(ex)
    try:
        fr2, _ = A.acute_deviation(fr)
        need = ["pwv", "dev_bsbp", "sbp_sit", "dbp_sit", "acute_dev", "age", "sex_male"]
        d = A.one_row_per_participant(fr2, need)
        for c in ("sbp_sit", "acute_dev"):
            d[c + "10"] = d[c] / 10
        EV["bp_obs"] = ["sbp_sit", "dbp_sit"] + (["sbp_lie"] if d["sbp_lie"].notna().mean() > 0.5 else [])
        keep = list(dict.fromkeys(["participant_id", "pwv", "sbp_sit10", "acute_dev10", "age", "sex_male", "bmi",
                                   "dev_bsbp"] + EV["bp_obs"]))
        EV["slope_d"] = d[keep].copy()
        try:
            EV["cal"] = A.calibration_data(fr2)
        except Exception as ex:
            EV["errors"]["calibration"] = err(ex)
    except Exception as ex:
        EV["errors"]["slopes"] = err(ex)
    try:
        EV["art_d"] = A.one_row_per_participant(fr, ["pwv", "age", "sex_male"])[["pwv", "acq_hour", "age", "sex_male",
                                                                                "bmi"]].copy()
    except Exception as ex:
        EV["errors"]["acq_hour"] = err(ex)
    S["EV"] = EV
    return EV


def metrics(S, y, full=True, mask=None):
    """ICC (atlas icc1, retest pairs), incident-hypertension OR (S6), SBP->PWV slopes (S2), acq-hour artefact (S8)."""
    EV = eval_setup(S)
    yy = y.copy()
    if mask is not None:
        yy[~mask] = np.nan
    out = {}
    a0, a1 = yy[EV["i0"]], yy[EV["i1"]]
    okp = np.isfinite(a0) & np.isfinite(a1)
    out["icc"] = float(A.icc1(a0, a1)) if okp.sum() >= A.MINCELL else None
    out["icc_n"] = rc(okp.sum())
    out["htn"] = or_fit(EV["htn_d"], yy)
    if EV["slope_d"] is None:
        return out
    d = EV["slope_d"].copy()
    d["pwv"] = yy[d.index.to_numpy()]
    d = d[np.isfinite(d["pwv"])]
    f = A.fit(d, "pwv", ["sbp_sit10"], ["age", "sex_male", "bmi"], std=False)
    out["slope_sit"] = A.coef(f, "sbp_sit10")
    f = A.fit(d, "pwv", ["sbp_sit10", "acute_dev10"], ["age", "sex_male", "bmi"], std=False)
    out["slope_acute"] = A.coef(f, "acute_dev10")
    if full:
        try:
            rcr = A.rc_acute(d, EV["cal"], EV["bp_obs"], 0, np.random.default_rng(0))
            out["rc_device_given_usual"] = (rcr or {}).get("per_10mmHg", {}).get("device_sbp_given_usual", {}).get("b")
            out["rc_usual_sbp"] = (rcr or {}).get("per_10mmHg", {}).get("usual_sbp_rc", {}).get("b")
        except Exception as ex:
            out["rc_error"] = err(ex)
        if EV["art_d"] is not None:
            ad = EV["art_d"].copy()
            ad["pwv"] = yy[ad.index.to_numpy()]
            ad = ad[np.isfinite(ad["pwv"])]
            out["acq_hour"] = A.coef(A.fit(ad, "pwv", ["acq_hour"], ["age", "sex_male", "bmi"]), "acq_hour")
    return out


def fit_hour(S, rows_train, knots=None):
    """K3 hour curve (spline coefficients) fitted on the training rows of the complete K sample."""
    K = k_sample(S)
    tr = np.intersect1d(K["rows"], rows_train)
    knots = K["knots"] if knots is None else knots
    X, names, hc = xmat(S, tr, "K3", knots=knots)
    b, _, _ = ols_cl(S["v"]["pwv_q"][tr], X)
    gam = b[hc]
    ref = rcs(S["hour"][tr], knots) @ gam
    return {"knots": knots, "gam": gam, "center": float(np.mean(ref))}


def apply_hour(S, y, H):
    out = y.copy()
    h = S["hour"]
    ok = np.isfinite(h) & np.isfinite(y)
    out[ok] = y[ok] - (rcs(h[ok], H["knots"]) @ H["gam"] - H["center"])
    return out


def kcorrect(S, y, shape, taus, edges, fit_rows=None, apply_rows=None, params=None, dcode=None, t=None,
             zmin=MIN_Z):
    """Day + within-day kernel correction of y (frame-aligned) as atlas v1.5, with a chosen kernel shape and grid.
    Residual model 1, age, (age-50)^2, sex fitted on fit_rows; kernel fitted on fit_rows pairs (or params given);
    LOO kriging applied within the days of apply_rows."""
    V_ = S["v"]
    dcode = S["dcode"] if dcode is None else dcode
    t = S["clk"]["vas"]["tm"] if t is None else t
    base = rows_ok(S, ["age", "sex"], np.isfinite(y) & (dcode >= 0))
    fit_rows = base if fit_rows is None else np.intersect1d(base, fit_rows)
    apply_rows = base if apply_rows is None else np.intersect1d(base, apply_rows)
    Xf, _, _ = xmat(S, fit_rows, "K0", quarter=False)
    beta = np.linalg.lstsq(Xf, y[fit_rows], rcond=None)[0]
    rf = y[fit_rows] - Xf @ beta
    Vf = float(np.var(rf))
    info = {}
    if params is None:
        full = np.full(S["n"], np.nan)
        full[fit_rows] = rf
        I, J = pairs_of(S, fit_rows, dcode, t)
        kf = kfit(np.abs(t[I] - t[J]), full[I] * full[J], dcode[I] % JK, Vf, shapes=(shape,), taus=taus, edges=edges,
                  jk_tau=False, min_bin=MIN_BIN if edges is EDGES else A.SESSION_MIN_BIN)
        f = kf.get("fits", {}).get(shape)
        if f is None:
            return y.copy(), {"status": kf.get("status", "no fit")}
        applied = bool(f["s"] > 0 and f["z"] >= zmin)
        params = {"tau": f["tau"], "b0": f["b0"], "s": f["s"] if applied else 0.0, "V": Vf, "applied": applied,
                  "z": f["z"], "share": f["share"], "day": f["day"]}
        if not applied:
            params["b0"] = kf["day_only"]["day"] * Vf
    info.update(params)
    Xa, _, _ = xmat(S, apply_rows, "K0", quarter=False)
    full = np.full(S["n"], np.nan)
    full[apply_rows] = y[apply_rows] - Xa @ beta
    groups = day_groups(dcode, apply_rows)
    ah = krige_all(full, groups, t, params["s"], params["tau"], params["V"], params["b0"], shape)
    out = y.copy()
    out[apply_rows] = y[apply_rows] - ah[apply_rows]
    return out, info


def variants(S):
    """PWV versions: raw, v1.3, v1.4, v1.5 (atlas) and the alternative corrections (cached in S)."""
    if "Y" in S:
        return S["Y"], S["Yinfo"], S["H"]
    V = S["v"]
    Y = {"raw": V["pwv_raw"], "q13": V["pwv_q"], "d14": V["pwv_d14"], "v15": V["pwv_v15"]}
    info, H = {}, None
    try:
        Y["g15"], info["g15"] = kcorrect(S, V["pwv_q"], "gau", TAUS, EDGES)
    except Exception as ex:  # v1.1: one failing variant no longer removes the others
        info["g15"] = {"error": err(ex)}
    try:
        H = fit_hour(S, np.arange(S["n"]))
        Y["h16"] = apply_hour(S, V["pwv_q"], H)
        info["h16"] = {"gam": [float(x) for x in H["gam"]], "knots": [float(x) for x in H["knots"]]}
        Y["h16e"], info["h16e"] = kcorrect(S, Y["h16"], "exp", TAUS, EDGES)
        Y["h16g"], info["h16g"] = kcorrect(S, Y["h16"], "gau", TAUS, EDGES)
    except Exception as ex:
        info["h16"] = {"error": err(ex)}
    S["Y"], S["Yinfo"], S["H"] = Y, info, H
    return Y, info, H


def crossfit(S, rng, reps):
    """2-fold cross-fitting over clinic days: hour curve, residual model and kernel from one half, applied to the
    other. Variants: v15x (atlas exponential grid/bins), h16gx (hour + Gaussian)."""
    V = S["v"]
    days = np.unique(S["dcode"][S["dcode"] >= 0])
    res = {"v15x": [], "h16gx": []}
    par = {"v15x": [], "h16gx": []}
    loo = {"exp": [], "gau": [], "day": []}
    lerr = {"exp": [], "gau": [], "dc": []}
    for _ in range(reps):
        fold = np.zeros(S["dcode"].max() + 1, int)
        fold[rng.permutation(days)[: len(days) // 2]] = 1
        rowfold = np.where(S["dcode"] >= 0, fold[np.maximum(S["dcode"], 0)], -1)
        yv, yh = V["pwv_q"].copy(), V["pwv_q"].copy()
        for k in (0, 1):
            tr, te = np.flatnonzero(rowfold == k), np.flatnonzero(rowfold == 1 - k)
            o, inf = kcorrect(S, V["pwv_q"], "exp", A.SESSION_TAUS, A.SESSION_EDGES, fit_rows=tr, apply_rows=te,
                              zmin=A.SESSION_MIN_Z)
            yv[te] = o[te]
            par["v15x"].append((inf.get("tau"), inf.get("share"), inf.get("z")))
            try:
                K = k_sample(S)
            except Exception:  # no K sample: v15x only
                yh = None
                continue
            H = fit_hour(S, tr)
            yh0 = apply_hour(S, V["pwv_q"], H)
            o, inf = kcorrect(S, yh0, "gau", TAUS, EDGES, fit_rows=tr, apply_rows=te)
            yh[te] = o[te]
            par["h16gx"].append((inf.get("tau"), inf.get("share"), inf.get("z")))
            # cross-fitted LOO MSE of kernel shapes on the test half (K3 residuals; parameters from the train half)
            ktr, kte = np.intersect1d(K["rows"], tr), np.intersect1d(K["rows"], te)
            if len(ktr) > 200 and len(kte) > 200:
                X, _, _ = xmat(S, K["rows"], "K3", knots=K["knots"])
                pos = {r_: i for i, r_ in enumerate(K["rows"])}
                itr = np.array([pos[r_] for r_ in ktr])
                ite = np.array([pos[r_] for r_ in kte])
                beta = np.linalg.lstsq(X[itr], V["pwv_q"][ktr], rcond=None)[0]
                rtr = V["pwv_q"][ktr] - X[itr] @ beta
                rte = V["pwv_q"][kte] - X[ite] @ beta
                kb = kernel_block(S, ktr, rtr, shapes=("exp", "gau"), jk_tau=False, bins=False)
                full = np.full(S["n"], np.nan)
                full[kte] = rte
                Vte = float(np.var(rte))
                groups = day_groups(S["dcode"], kte)
                t = S["clk"]["vas"]["tm"]
                if kb.get("fits"):
                    loo["day"].append(loo_mse(full, groups, t, Vte, {"b0": kb["day_only"]["day"] * float(np.var(rtr)),
                                                                     "s": 0.0, "tau": 60.0}, "exp"))
                    ee = {}
                    for sh in ("exp", "gau"):
                        f = kb["fits"].get(sh)
                        if f:
                            ee[sh] = loo_err(full, groups, t, Vte, f, sh)
                            loo[sh].append(float(np.mean(ee[sh])))
                    if "exp" in ee and "gau" in ee and groups:
                        lerr["exp"].append(ee["exp"])
                        lerr["gau"].append(ee["gau"])
                        lerr["dc"].append(S["dcode"][np.concatenate(groups)])  # same day across reps = one cluster
        res["v15x"].append(metrics(S, yv, full=False))
        if yh is not None:
            res["h16gx"].append(metrics(S, yh, full=False))
    out = {"reps": reps}
    for k in res:
        icc = [m["icc"] for m in res[k] if m.get("icc") is not None]
        lor = [m["htn"]["log_or"] for m in res[k] if m.get("htn")]
        ss = [m["slope_sit"]["b"] for m in res[k] if m.get("slope_sit")]
        p = np.array([x for x in par[k] if x[0] is not None], float)
        out[k] = {"icc_mean": float(np.mean(icc)) if icc else None, "icc_min": float(np.min(icc)) if icc else None,
                  "icc_max": float(np.max(icc)) if icc else None,
                  "or_mean": float(np.exp(np.mean(lor))) if lor else None,
                  "slope_sit_mean": float(np.mean(ss)) if ss else None,
                  "tau_range": [float(np.nanmin(p[:, 0])), float(np.nanmax(p[:, 0]))] if len(p) else None,
                  "share_range": [float(np.nanmin(p[:, 1])), float(np.nanmax(p[:, 1]))] if len(p) else None,
                  "z_min": float(np.nanmin(p[:, 2])) if len(p) else None}
    out["loo_test_half"] = {k: (float(np.mean(v)) if v else None) for k, v in loo.items()}
    if lerr["exp"]:
        out["loo_test_half"]["gau_minus_exp"] = paired_diff(np.concatenate(lerr["gau"]), np.concatenate(lerr["exp"]),
                                                            np.concatenate(lerr["dc"]))
    return out


def placebo_correct(S, rng, reps, Y, info):
    """Corrections applied with the real-data parameters to (a) day labels shuffled within quarter (rows keep their
    clock time): every gain must vanish; (b) v1.1: clock times shuffled within day (day labels kept): the day part of
    the gain stays, the part carried by acquisition timing must vanish (real gain - time-shuffled gain = timing)."""
    d15 = S["diag"].get("session_kernel", {}).get("pwv", {})
    out = {"reps": reps}
    specs = []
    if d15.get("tau") is not None and d15.get("V"):
        Vd = float(d15["V"])
        specs.append(("v15p", "v15", Y["q13"], "exp", {"tau": float(d15["tau"]), "b0": float(d15["day_share"]) * Vd,
                                                       "s": float(d15["share"]) * Vd if d15.get("applied") else 0.0,
                                                       "V": Vd}))
    if "h16g" in info and info["h16g"].get("tau") is not None:
        p = info["h16g"]
        specs.append(("h16gp", "h16g", Y["h16"], "gau", {"tau": p["tau"], "b0": p["b0"], "s": p["s"], "V": p["V"]}))
    rows = np.flatnonzero(S["dcode"] >= 0)
    clock = S["clk"]["vas"]["clock"]
    rows_t = np.flatnonzero((S["dcode"] >= 0) & np.isfinite(clock))

    def summ(x):
        return {"mean": float(np.nanmean(x)), "q05": float(np.nanquantile(x, 0.05)),
                "q95": float(np.nanquantile(x, 0.95))} if np.isfinite(x).any() else None
    for lab, real, y0, shape, par in specs:
        m0 = metrics(S, y0, full=False)
        mr = metrics(S, Y[real], full=False) if real in Y else {}
        acc = []
        for _ in range(reps):
            dc = S["dcode"].copy()
            dc[rows] = S["dcode"][shuffle_within(S["qcode"], rows, rng)]
            yp, _ = kcorrect(S, y0, shape, TAUS, EDGES, params=dict(par), dcode=dc, t=clock)
            m = metrics(S, yp, full=False)
            tt = clock.copy()
            tt[rows_t] = clock[shuffle_within(S["dcode"], rows_t, rng)]
            yt, _ = kcorrect(S, y0, shape, TAUS, EDGES, params=dict(par), t=tt)
            mt = metrics(S, yt, full=False)
            acc.append(tuple((mm["icc"] - m0["icc"] if mm.get("icc") is not None and m0.get("icc") is not None else
                              np.nan, mm["htn"]["log_or"] - m0["htn"]["log_or"] if mm.get("htn") and m0.get("htn")
                              else np.nan) for mm in (m, mt)))
        x = np.array(acc, float).reshape(len(acc), 4)
        ds, ts_ = summ(x[:, 0]) or {}, summ(x[:, 2]) or {}
        out[lab] = {"d_icc_mean": ds.get("mean"), "d_icc_q05": ds.get("q05"), "d_icc_q95": ds.get("q95"),
                    "d_logor_mean": float(np.nanmean(x[:, 1])) if np.isfinite(x[:, 1]).any() else None,
                    "t_icc_mean": ts_.get("mean"), "t_icc_q05": ts_.get("q05"), "t_icc_q95": ts_.get("q95"),
                    "t_logor_mean": float(np.nanmean(x[:, 3])) if np.isfinite(x[:, 3]).any() else None,
                    "real_d_icc": (mr["icc"] - m0["icc"]) if mr.get("icc") is not None and m0.get("icc") is not None
                    else None, "base_icc": m0.get("icc")}
    return out


def boot_deltas(S, Y, rng, B):
    EV = eval_setup(S)
    pairs = [("v15", "q13"), ("h16g", "v15"), ("g15", "v15"), ("h16g", "q13")]
    i0, i1 = EV["i0"], EV["i1"]
    d = EV["htn_d"]
    out = {}
    for a_, b_ in pairs:
        if a_ not in Y or b_ not in Y:
            continue
        ya, yb = Y[a_], Y[b_]
        ok = np.isfinite(ya[i0]) & np.isfinite(ya[i1]) & np.isfinite(yb[i0]) & np.isfinite(yb[i1])
        j0, j1 = i0[ok], i1[ok]
        di, dl = [], []
        for _ in range(B):
            k = rng.integers(0, len(j0), len(j0))
            di.append(A.icc1(ya[j0[k]], ya[j1[k]]) - A.icc1(yb[j0[k]], yb[j1[k]]))
            if d is None:
                continue
            kk = rng.integers(0, len(d), len(d))
            fa, fb = or_fit(d, ya, kk), or_fit(d, yb, kk)
            if fa and fb:
                dl.append(fa["log_or"] - fb["log_or"])
        m_a, m_b = or_fit(d, ya), or_fit(d, yb)
        out[f"{a_}-{b_}"] = {"d_icc": float(A.icc1(ya[j0], ya[j1]) - A.icc1(yb[j0], yb[j1])), "d_icc_ci": A.pct(di),
                             "d_logor": (m_a["log_or"] - m_b["log_or"]) if m_a and m_b else None,
                             "d_logor_ci": A.pct(dl)}
    return out


def block_E(S, a, rng):
    Y, info, H = variants(S)
    out = {"variants": {}, "info": info}
    for k, y in Y.items():
        out["variants"][k] = metrics(S, y)
        out["variants"][k]["change_vs_q13_sd"] = float(np.nanstd(y - S["v"]["pwv_q"]) / np.nanstd(S["v"]["pwv_q"]))
    EVd = eval_setup(S)
    out["eval_errors"] = EVd.get("errors") or None
    # replica check against atlas S6 for the v1.5 frame
    try:
        s6 = A.s6_incident(S["fr"], S["htn"], S["medflags"]).get("per_bed", {}).get("pwv")
        mine = out["variants"]["v15"].get("htn")
        out["s6_replica_diff"] = abs(s6["or"] - mine["or"]) if s6 and mine else None
    except Exception as ex:
        out["s6_replica_diff"] = None
        out["s6_replica_error"] = err(ex)
    for name, fn in (("crossfit", lambda: crossfit(S, rng, a.cf)),
                     ("placebo", lambda: placebo_correct(S, rng, a.placebo, Y, info)),
                     ("boot", lambda: boot_deltas(S, Y, rng, a.boot))):
        t1 = time.time()
        try:
            out[name] = fn()
        except Exception as ex:
            out[name] = {"error": err(ex)}
        log(f"E {name} {time.time() - t1:.1f}s")
    return out


# ============================================================================ COVID window

def block_COVID(S, a):
    lo, hi = pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1])
    dt = S["date"]
    inwin = ((dt >= lo) & (dt <= hi)).to_numpy()
    keep = ~inwin
    out = {"window": f"{COVID[0]}..{COVID[1]}", "n_rows_excluded_pwv": rc((inwin & np.isfinite(S["v"]["pwv_q"])).sum())}
    K = k_sample(S)
    rows = K["rows"]
    sel = keep[rows]
    for m in ("K0", "K1"):
        r = K["resid"][m]
        kb = kernel_block(S, rows[sel], r[sel] - r[sel].mean(), shapes=("exp", "gau"), jk_tau=False, bins=False)
        out[m] = {sh: {k: f.get(k) for k in ("tau", "share", "day", "z", "decay", "lag0")} for sh, f in kb.get("fits", {}).items()}
    Y, _, _ = variants(S)
    out["metrics"] = {k: metrics(S, Y[k], full=False, mask=keep) for k in ("q13", "v15", "h16g") if k in Y}
    return out


# ============================================================================ digest

def v3(x):
    if x is None:
        return "NA"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "NA"
    s = f"{x:.3g}"
    return s.replace("e-0", "e-").replace("e+0", "e")


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def nz(x):
    return "NA" if x is None else x


def bse(c):
    if not isinstance(c, dict) or c.get("b") is None:
        return "NA"
    if c.get("se") is not None:
        return f"{v3(c['b'])}({v3(c['se'])})"
    if c.get("lo") is not None:
        return f"{v3(c['b'])}[{v3(c['lo'])},{v3(c['hi'])}]"
    return v3(c["b"])


def kf_s(f, short=False):
    if not isinstance(f, dict):
        return "NA"
    t = f"t{v3(f.get('tau'))}"
    if f.get("tau_lo") is not None and (fl(f.get("z")) or 0) >= 2:
        t += f"[{v3(f['tau_lo'])},{v3(f['tau_hi'])}]"
    if f.get("edge"):
        t += "!"
    s = f"{t} s{v3(f.get('share'))} z{v3(f.get('z'))} d{v3(f.get('day'))} dc{v3(f.get('decay'))}"
    if not short:
        s += f" l0={v3(f.get('lag0'))}"
    return s


def digest(R):
    """Screen-transcription digest: <= 120 lines, <= 220 characters, no leading spaces, 3 significant digits."""
    L = [f"F1_ACQ_ANATOMY {R.get('version')} | atlas {R.get('atlas_version')} | seed {R.get('seed')}"]
    I = R.get("info", {})
    L.append(f"flow: pwv timed rows {I.get('n_rows_pwv_timed')} days {I.get('n_days_pwv')} per-day q05/50/95 "
             f"{v3(g(I, 'pwv_per_day', 'q05'))}/{v3(g(I, 'pwv_per_day', 'q50'))}/{v3(g(I, 'pwv_per_day', 'q95'))}; "
             f"hour q05/50/95 {v3(g(I, 'hour_pwv', 'q05'))}/{v3(g(I, 'hour_pwv', 'q50'))}/{v3(g(I, 'hour_pwv', 'q95'))}; "
             f"relations excluded {I.get('relations_excluded')}")
    L.append(f"check: v1.5 rebuilt max|diff| {v3(I.get('v15_reproduced_maxdiff'))} same-missing "
             f"{I.get('v15_reproduced_same_missing')}; s6 replica |dOR| {v3(g(R, 'E', 's6_replica_diff'))}; "
             f"clocks {','.join(I.get('clocks', []))}"
             + (f"; K2 covariates dropped (missing) {','.join(I['k2_dropped'])}" if I.get("k2_dropped") else ""))
    for k, v in sorted(R.get("errors", {}).items()):
        L.append(f"ERROR {k}: {v}")
    K = R.get("K", {})
    if K and not K.get("error"):
        L.append("## K nested kernels, v1.3 PWV, one complete sample: t=tau min [jk CI] !=edge, s=kernel share, d=day, "
                 "dc=s[k(10)-k(300)], l0=d+s; loo=LOO kriging MSE/var; g-e=gau-exp(se by day)")
        L.append(f"K n {K.get('n_rows')} knots h {'/'.join(v3(x) for x in K.get('knots_h', []))} | atlas v1.5 run: "
                 + " ".join(f"{k} {v3(v)}" for k, v in (K.get("atlas_v15_diag") or {}).items() if k != "n_days")
                 + " | atlas grid on K0: " + " ".join(f"{k} {v3(v)}" for k, v in (K.get("atlas_grid_K0") or {}).items()))
        for m in MODELS:
            M = g(K, "models", m) or {}
            if M.get("error") or M.get("status"):
                L.append(f"K {m}: {M.get('error') or M.get('status')}")
                continue
            lo, f = M.get("loo", {}), M.get("fits", {})
            L.append(f"K {m} pairs {M.get('n_pairs')} r2 {v3(M.get('r2_model'))} | day-only d{v3(g(M, 'day_only', 'day'))} | "
                     f"sse x1e3 exp {v3((fl(g(f, 'exp', 'sse')) or 0) * 1e3)} gau {v3((fl(g(f, 'gau', 'sse')) or 0) * 1e3)} "
                     f"m32 {v3((fl(g(f, 'm32', 'sse')) or 0) * 1e3)} best {M.get('best_shape')} | loo day "
                     f"{v3(lo.get('day_only'))} exp {v3(lo.get('exp'))} gau {v3(lo.get('gau'))} m32 {v3(lo.get('m32'))} "
                     f"g-e {v3(g(lo, 'gau_minus_exp', 'd'))}({v3(g(lo, 'gau_minus_exp', 'se'))}) e-day "
                     f"{v3(g(lo, 'exp_minus_day', 'd'))}({v3(g(lo, 'exp_minus_day', 'se'))})")
            for sh in ("exp", "gau"):
                L.append(f"K {m} {sh} {kf_s(f.get(sh))}")
            if M.get("hour_curve"):
                hc = M["hour_curve"]
                L.append(f"K {m} hour f(h)-f(10) m/s: " + " ".join(f"{h:g}h {bse(hc.get(f'{h:g}'))}" for h in HOURS)
                         + f"; var share {v3(M.get('hour_var_share'))}")
        labs = [f"{int(a)}-{int(b)}" for a, b in zip(A.SESSION_EDGES[:-1], A.SESSION_EDGES[1:])] + [f"{A.SESSION_EDGES[-1]}+"]
        for m in ("K0", "K3"):
            bins = g(K, "models", m, "bins") or {}
            if m == "K0":  # pair counts once (same pairs in K3)
                L.append(f"K {m} bins r/n: " + " ".join(f"{k}:{v3(g(bins, k, 'r'))}/{nz(g(bins, k, 'n'))}" for k in labs
                                                         if k in bins))
            else:
                L.append(f"K {m} bins r: " + " ".join(f"{k}:{v3(g(bins, k, 'r'))}" for k in labs if k in bins))
    Vb = R.get("V", {})
    if Vb and not Vb.get("error"):
        L.append("## V kernel by variable: K0 vas clock exp | gau | K1 (+hour) gau | own clock K0 gau | ANOVA day share")
        for lab, _, own in PANEL:
            x = Vb.get(lab, {})
            if x.get("status"):
                L.append(f"V {lab}: {x['status']}")
                continue
            e0, g0 = g(x, "vas_K0", "fits", "exp"), g(x, "vas_K0", "fits", "gau")
            g1, go = g(x, "vas_K1", "fits", "gau"), g(x, "own_K0", "fits", "gau")
            ow = "same" if own == "vas" else (kf_s(go, True) if go else str(g(x, "own_K0", "status") or g(x, "own_K0", "error")
                                                                             or "NA")[:40])
            L.append(f"V {lab} n{g(x, 'vas_K0', 'n_rows')}: e {kf_s(e0, True)} | g {kf_s(g0, True)} | K1 {kf_s(g1, True)} | "
                     f"own {ow} | anova {v3(g(x, 'vas_K0', 'anova_day_share'))}")
    X = R.get("X", {})
    if X and not X.get("error"):
        L.append(f"## X same-day stranger cross-corr xr(se) | process corr rho (both auto z>=2) | day-BLUP r (composition-"
                 f"sensitive); pairs {X.get('n_pairs')}")
        L.append("X auto r(z): " + " ".join(f"{k} {v3(g(X, 'auto', k, 'r'))}({v3(g(X, 'auto', k, 'z'))})" for k, _, _ in PANEL
                                             if g(X, "auto", k)))
        heads = {"pwv": [p[0] for p in PANEL if p[0] != "pwv"],
                 "dev_bsbp": ["dev_asbp", "abi", "sbp_sit", "dbp_sit", "hr_sit", "ecg_hr", "bmi", "height"],
                 "dev_asbp": ["abi", "sbp_sit", "hr_sit", "bmi", "height"], "abi": ["sbp_sit", "hr_sit", "height"],
                 "sbp_sit": ["dbp_sit", "hr_sit", "bmi"]}
        names = [p[0] for p in PANEL]
        for k1, others in heads.items():
            parts = []
            for k2 in others:
                key = f"{k1}|{k2}" if names.index(k1) < names.index(k2) else f"{k2}|{k1}"
                c, bl = g(X, "rho", key), g(X, "blup_r", key)
                if c is None and bl is None:
                    continue
                parts.append(f"{k2} {v3(g(c, 'xr'))}({v3(g(c, 'xr_se'))})|{v3(g(c, 'rho'))}|{v3(g(bl, 'r'))}")
            if parts:
                L.append(f"X {k1}: " + " ".join(parts))
    W = R.get("W", {})
    if W and not W.get("error"):
        L.append("## W clock-hour slope (native units per hour): within-person d~dh (se) n |dh| q50 | between K0 | between K2")
        for lab in ("pwv_q", "pwv_raw", "pwv_v15", "dev_bsbp", "dev_asbp", "sbp_sit", "hr_sit", "height", "bmi"):
            x = W.get(lab, {})
            if x.get("error"):
                L.append(f"W {lab}: {x['error']}")
                continue
            L.append(f"W {lab}: within {bse(x.get('within'))} n {x.get('n_pairs')} |dh| {v3(g(x, 'abs_dh', 'q50'))} | "
                     f"K0 {bse(x.get('between_K0'))} | K2 {bse(x.get('between_K2'))}")
    P = R.get("P", {})
    if P and not P.get("error"):
        L.append(f"## P placebo kernels ({P.get('reps')} reps): p = permutation p of the real decay, s mean/q95, z mean, "
                 "f = frac z>=3, d = day share mean, dc = decay mean/q95")
        for m in ("K0", "K1"):
            x = P.get(m, {})
            re_ = x.get("real", {})
            L.append(f"P {m} real: exp s{v3(g(re_, 'exp', 'share'))} z{v3(g(re_, 'exp', 'z'))} d{v3(g(re_, 'exp', 'day'))} "
                     f"dc{v3(g(re_, 'exp', 'decay'))} | gau s{v3(g(re_, 'gau', 'share'))} z{v3(g(re_, 'gau', 'z'))} "
                     f"d{v3(g(re_, 'gau', 'day'))} dc{v3(g(re_, 'gau', 'decay'))}")
            for kind, short in (("time_within_day", "time-in-day"), ("day_within_quarter", "day-in-qtr")):
                y = x.get(kind, {})
                L.append(f"P {m} {short}: " + " | ".join(
                    f"{sh} p{v3(g(y, sh, 'p_perm'))} s{v3(g(y, sh, 'share_mean'))}/{v3(g(y, sh, 'share_q95'))} z{v3(g(y, sh, 'z_mean'))} "
                    f"f{v3(g(y, sh, 'frac_z3'))} d{v3(g(y, sh, 'day_mean'))} dc{v3(g(y, sh, 'decay_mean'))}/"
                    f"{v3(g(y, sh, 'decay_q95'))}" for sh in ("exp", "gau")))
    J = R.get("J", {})
    if J and not J.get("error"):
        L.append("## J joint shares of one residual variance: cal = 1-7 d different-day r (se; exp-fit at 1 d, L; quarter-FE "
                 "share) | day | within-day kernel | remainder | Kish")
        for lab in ("raw_K0", "raw_K1", "q13_K0", "v15_K0", "h16g_K0"):
            x = J.get(lab)
            if not isinstance(x, dict):
                continue
            if x.get("error") or x.get("status"):
                L.append(f"J {lab}: {x.get('error') or x.get('status')}")
                continue
            qfe = f"; qFE {v3(x.get('quarter_fe_share'))}" if x.get("quarter_fe_share") is not None else ""
            L.append(f"J {lab} n{x.get('n_rows')}: cal {v3(x.get('calendar'))}({v3(x.get('calendar_se'))}; fit "
                     f"{v3(x.get('calendar_fit_1d'))} L{v3(x.get('calendar_L_days'))}d{qfe}) day {v3(x.get('day'))} within {v3(x.get('kernel'))} "
                     f"({x.get('shape')} t{v3(x.get('tau'))} z{v3(x.get('z'))} dc{v3(x.get('decay'))}) rem {v3(x.get('remainder'))} | rho "
                     f"{v3(x.get('rho_sameday'))} deff {v3(x.get('deff'))} neff {x.get('neff')}")
            if lab in ("raw_K0", "q13_K0"):
                lr = x.get("lag_r") or {}
                L.append(f"J {lab} lag r: " + " ".join(f"{k}:{v3(lr.get(k))}" for k in
                                                        [f"{a}-{b - 1}" for a, b in zip(LAG_EDGES[:-1], LAG_EDGES[1:])]
                                                        + [f"{LAG_EDGES[-1]}+"]))
    C = R.get("CP", {})
    if C and not C.get("error"):
        L.append(f"## CP change points on daily medians (pen {CP_PEN}*log D, min {CP_MINSEG} days): n, r2 step vs spline "
                 "(same df), null mean/frac>=1 | week:jump(z)")
        for lab, _ in CP_VARS:
            x = g(C, "vars", lab) or {}
            if x.get("error") or x.get("status"):
                L.append(f"CP {lab}: {x.get('error') or x.get('status')}")
                continue
            L.append(f"CP {lab} days {x.get('n_days')}: n {x.get('n_cp')} r2 {v3(x.get('r2_cp'))} vs "
                     f"{v3(x.get('r2_spline_same_df'))}(df{x.get('spline_df')}) null {v3(x.get('null_mean_cp'))}/"
                     f"{v3(x.get('null_frac_ge1'))} | " + " ".join(f"{c['week'][2:].replace('-', '')}:{v3(c['jump'])}"
                                                                  f"({v3(c['z'])})" for c in (x.get("cps") or [])[:8]))
        qall = sorted({k for lab in ("pwv_raw", "dev_bsbp", "dev_asbp", "abi_raw", "sbp_sit")
                       for k in (g(C, "quarterly", lab) or {})})
        if qall:  # v1.1: quarter labels once, then values only (NA = cell < 20) to save digest lines
            L.append(f"Q quarters: {qall[0][2:]}..{qall[-1][2:]} ({len(qall)}): " + " ".join(k[2:] for k in qall))
        for lab in ("pwv_raw", "dev_bsbp", "dev_asbp", "abi_raw", "sbp_sit"):
            q = g(C, "quarterly", lab) or {}
            if q:
                L.append(f"Q {lab}: " + " ".join(v3(g(q, k, "median")) for k in qall))
        dfi = C.get("device_fields") or {}
        if dfi.get("error"):
            L.append(f"vascular_health fields: {dfi['error']}")
        else:
            L.append(f"vascular_health fields: n {dfi.get('n_columns')} other {','.join(dfi.get('other_columns') or [])} | "
                     "device-like " + "; ".join(f"{k} lv{v.get('n_levels')} {v.get('top')}"
                                                for k, v in (dfi.get("device_like") or {}).items()))
    E = R.get("E", {})
    if E and not E.get("error"):
        L.append("## E PWV re-evaluation: retest ICC(n) | incident HTN OR/SD [CI] ev auc | per 10 mmHg sit(se) acute(se) | "
                 "RC device|usual | acq_hour std beta | SD change vs v1.3")
        for k in ("raw", "q13", "d14", "v15", "g15", "h16", "h16e", "h16g"):
            m = g(E, "variants", k)
            if not m:
                continue
            h = m.get("htn") or {}
            L.append(f"E {k}: icc {v3(m.get('icc'))}({m.get('icc_n')}) | OR {v3(h.get('or'))}[{v3(h.get('or_lo'))},"
                     f"{v3(h.get('or_hi'))}] ev{nz(h.get('events'))} auc{v3(h.get('auc'))} | sit {bse(m.get('slope_sit'))} ac "
                     f"{bse(m.get('slope_acute'))} | rc {v3(m.get('rc_device_given_usual'))}|{v3(m.get('rc_usual_sbp'))} | "
                     f"hr {bse(m.get('acq_hour'))} | chg {v3(m.get('change_vs_q13_sd'))}")
        if E.get("eval_errors") or E.get("s6_replica_error"):
            L.append("E eval errors: " + "; ".join(f"{k} {str(v)[:70]}" for k, v in
                                                    sorted((E.get("eval_errors") or {}).items())) +
                     (f"; s6 replica {str(E['s6_replica_error'])[:60]}" if E.get("s6_replica_error") else ""))
        inf = E.get("info") or {}
        L.append("E params: " + " ".join((f"{k} ERROR {inf[k]['error'][:60]};" if inf[k].get("error") else
                                          f"{k} t{v3(inf[k].get('tau'))} s{v3(inf[k].get('share'))} d{v3(inf[k].get('day'))} "
                                          f"z{v3(inf[k].get('z'))} applied {inf[k].get('applied')};")
                                         for k in ("g15", "h16", "h16e", "h16g") if isinstance(inf.get(k), dict)
                                         and (k != "h16" or inf[k].get("error"))))
        cfx = E.get("crossfit") or {}
        if cfx.get("error"):
            L.append(f"E crossfit: {cfx['error']}")
        else:
            for k in ("v15x", "h16gx"):
                x = cfx.get(k) or {}
                L.append(f"E crossfit {k} ({cfx.get('reps')}x2): icc {v3(x.get('icc_mean'))} [{v3(x.get('icc_min'))},"
                         f"{v3(x.get('icc_max'))}] OR {v3(x.get('or_mean'))} sit {v3(x.get('slope_sit_mean'))} | tau "
                         f"{'-'.join(v3(t) for t in (x.get('tau_range') or []))} s "
                         f"{'-'.join(v3(t) for t in (x.get('share_range') or []))} zmin {v3(x.get('z_min'))}")
            lo = cfx.get("loo_test_half") or {}
            L.append(f"E crossfit LOO MSE/var on test half (K3 residuals): day {v3(lo.get('day'))} exp {v3(lo.get('exp'))} "
                     f"gau {v3(lo.get('gau'))} gau-exp {v3(g(lo, 'gau_minus_exp', 'd'))}({v3(g(lo, 'gau_minus_exp', 'se'))})")
        pl = E.get("placebo") or {}
        L.append("E placebo (real params; dICC vs uncorrected, mean [q05,q95]): day-in-qtr | time-in-day | real: " +
                 (pl.get("error") or " | ".join(
                     f"{k} {v3(v.get('d_icc_mean'))}[{v3(v.get('d_icc_q05'))},{v3(v.get('d_icc_q95'))}] dlogOR "
                     f"{v3(v.get('d_logor_mean'))} | {v3(v.get('t_icc_mean'))}[{v3(v.get('t_icc_q05'))},"
                     f"{v3(v.get('t_icc_q95'))}] dlogOR {v3(v.get('t_logor_mean'))} | {v3(v.get('real_d_icc'))} base "
                     f"{v3(v.get('base_icc'))}" for k, v in pl.items() if isinstance(v, dict))))
        bt = E.get("boot") or {}
        L.append("E bootstrap: " + (bt.get("error") or " | ".join(
            f"{k} dICC {v3(v.get('d_icc'))}{ci(v.get('d_icc_ci'))} dlogOR {v3(v.get('d_logor'))}{ci(v.get('d_logor_ci'))}"
            for k, v in bt.items() if isinstance(v, dict))))
    CV = R.get("COVID", {})
    if CV and not CV.get("error"):
        L.append(f"## COVID window {CV.get('window')} excluded (pwv rows {CV.get('n_rows_excluded_pwv')})")
        for m in ("K0", "K1"):
            x = CV.get(m) or {}
            L.append(f"COVID {m}: exp {kf_s(x.get('exp'), True)} | gau {kf_s(x.get('gau'), True)}")
        for k in ("q13", "v15", "h16g"):
            m = g(CV, "metrics", k)
            if not m:
                continue
            h = m.get("htn") or {}
            L.append(f"COVID E {k}: icc {v3(m.get('icc'))}({m.get('icc_n')}) OR {v3(h.get('or'))}[{v3(h.get('or_lo'))},"
                     f"{v3(h.get('or_hi'))}] ev{nz(h.get('events'))} sit {bse(m.get('slope_sit'))}")
    L.append(f"runtime {v3(R.get('runtime_s'))}s; blocks " + " ".join(f"{k} {v3(v)}" for k, v in (R.get("timings") or {}).items()))
    def wrap(ln):
        ln = re.sub(r"(?<![\w.])(-?)0\.(\d)", r"\1.\2", ln.strip())  # 0.123 -> .123 (3 significant digits kept)
        w = []
        while len(ln) > 220:
            w.append(ln[:220])
            ln = "+" + ln[220:]
        return w + [ln]
    W_ = [wrap(ln) for ln in L]
    # v1.1: above 120 lines, drop optional lines (all also in results.json) in this order before truncating, so the
    # E and COVID blocks at the end are never cut
    drop = []
    for pre in ("Q ", "J raw_K0 lag r", "J q13_K0 lag r", "K K3 bins", "K K0 bins", "X sbp_sit:", "X abi:", "X dev_asbp:"):
        if sum(map(len, W_)) + (1 if drop else 0) <= 120:
            break
        keep = [w for w in W_ if not w[0].startswith(pre)]
        if len(keep) < len(W_):
            drop.append(pre.strip().rstrip(":"))
            W_ = keep
    out = [x for w in W_ for x in w]
    if drop:
        out.append(f"OMITTED (see results.json): {', '.join(drop)}")
    if len(out) > 120:
        out = out[:119] + [f"TRUNCATED {len(out) - 119} lines (see results.json)"]
    return "\n".join(out) + "\n"


def ci(c):
    return "[NA]" if not c or c[0] is None else f"[{v3(c[0])},{v3(c[1])}]"


# ============================================================================ main

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


def analyse(S, a):
    """Run the blocks on a prepared context; returns the results dict."""
    secs = [s.strip() for s in a.sections.split(",") if s.strip()]
    rng = np.random.default_rng(a.seed)
    R = {"version": VERSION, "atlas_version": A.VERSION, "seed": f"{a.seed:,}", "sections": secs,
         "settings": {"taus": [TAUS[0], TAUS[-1], len(TAUS)], "edges": list(EDGES), "min_bin": MIN_BIN, "jk": JK,
                      "min_z": MIN_Z, "knot_q": list(KNOT_Q), "k2": K2COV, "covid": list(COVID), "cp_pen": CP_PEN,
                      "cp_minseg": CP_MINSEG, "placebo": a.placebo, "cf": a.cf, "boot": a.boot, "cp_null": a.cp_null},
         "info": dict(S["info"], clocks=sorted(S["clk"])), "errors": {}, "timings": {}}
    fns = {"K": lambda: block_K(S, a), "V": lambda: block_V(S, a), "X": lambda: block_X(S, a),
           "W": lambda: block_W(S, a), "P": lambda: block_P(S, a, rng), "J": lambda: block_J(S, a),
           "CP": lambda: block_CP(S, a, rng), "E": lambda: block_E(S, a, rng), "COVID": lambda: block_COVID(S, a)}
    order = [s for s in ("K", "V", "X", "W", "P", "E", "J", "CP", "COVID") if s in secs]
    for key in order:
        t1 = time.time()
        try:
            R[key] = fns[key]()
        except Exception as ex:
            R[key] = {"error": err(ex)}
            R["errors"][key] = R[key]["error"]
            log(f"{key} FAILED: {R[key]['error']}")
        R["timings"][key] = time.time() - t1
        log(f"block {key} done in {R['timings'][key]:.1f}s")
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--atlas", default=None, help="atlas v1.5 script (default f1_atlas_v1_5.py, else f1_atlas_v1.py, "
                                                  "beside this file)")
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--placebo", type=int, default=50, help="placebo replicates (kernels and corrections); "
                                                            "minimum permutation p = 1/(reps + 1)")
    ap.add_argument("--cf", type=int, default=5, help="cross-fitting repetitions (each 2-fold over clinic days)")
    ap.add_argument("--boot", type=int, default=200, help="paired participant bootstrap for delta ICC / log OR")
    ap.add_argument("--cp-null", type=int, default=20, help="day-order permutations for the change-point null")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    here = os.path.dirname(os.path.abspath(__file__))
    path = a.atlas or next((p for p in (os.path.join(here, "f1_atlas_v1_5.py"), os.path.join(here, "f1_atlas_v1.py"))
                            if os.path.exists(p)), None)
    if path is None:
        raise SystemExit("atlas not found: pass --atlas f1_atlas_v1_5.py")
    set_atlas(path)
    t0 = time.time()
    import pyarrow
    import scipy
    log(f"{VERSION}; atlas {A.VERSION} ({os.path.basename(path)}); numpy {np.__version__}; pandas {pd.__version__}; "
        f"scipy {scipy.__version__}; pyarrow {pyarrow.__version__}; python {sys.version.split()[0]}; "
        f"threads {os.environ.get('OMP_NUM_THREADS')}")
    if not str(A.VERSION).startswith("f1_atlas_v1.5"):
        log(f"WARNING: atlas version {A.VERSION} is not v1.5")
    fr, medflags, htn = A.build_frame(a.root, None)
    pids = set(fr["participant_id"].astype(str))
    S = prep(fr, medflags, htn, a.root)
    R = analyse(S, a)
    R["runtime_s"] = time.time() - t0
    text = json.dumps(clean(R), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    privacy_check(text, pids)
    dg = digest(json.loads(text))
    privacy_check(dg, pids)
    log(f"digest {dg.count(chr(10))} lines (max {max(len(x) for x in dg.splitlines())} chars); total {time.time() - t0:.1f}s")
    logtext = "\n".join(A.LOG) + "\n"
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
