#!/usr/bin/env python3
"""F1 multi-bed vascular atlas — sandbox analysis programme v1 (writes aggregates only).

Paper F1 asks whether femoral-ankle PWV, carotid IMT, retinal microvasculature and ABI share upstream drivers,
and how much of PWV's blood-pressure association is acquisition-time physics.  Sections:

  S0 facts       cohort flow, overlaps, descriptives, measurement validity (left/right agreement, between-visit
                 ICC per bed, BP replicate reliability, device vs visit BP, acquisition time of day)
  S1 atlas       standardized betas of ~30 exposures x beds (4 covariate sets); common-sample bed contrasts with
                 analytic robust (SUR-equivalent) joint covariance, Wald heterogeneity test, paired bootstrap,
                 TOST equivalence, reliability-disattenuated contrasts, partial correlations, joint models
  S2 pressure    acquisition-pressure decomposition of faPWV: acute deviation (device brachial SBP beyond visit
                 BP), within-person change, lagged slope, per-exposure three-way shares with naive and
                 regression-calibrated chronic BP, CAVI0-style pressure-corrected stiffness, sensitivity sets
  S3 hypoxia     SpO2-based hypoxic burden (Azarbarzin-style, from the per-night SpO2 series and respiratory
                 events), sleep hypoxic load below 90 %, their links with AHI/ODI/T90 and with each bed
  S4 ecg         ECG LV voltage (Sokolow-Lyon, Cornell, Cornell product) from the device amplitude table
  S5 starters    antihypertensive starters between visits as a positive control on pressure-decomposed PWV;
                 other drug classes reported only above a minimum number of starters
  S6 incident    incident hypertension and SBP change by baseline bed (each bed and jointly)
  S7 vascage     bed-specific age slopes, age-equivalent effects of exposures, sex x age reference quantiles
  S8 extended    nonlinearity (quintiles), sex strata, negative-control exposure (birth month), acquisition
                 artefacts (height, heart rate, time of day, season), ABI < 0.9 / > 1.4 exclusion

Run (sandbox `pheno` env; numpy / pandas / scipy only):
  python f1_atlas_v1.py --root <HPP_DATA_ROOT> --out f1_atlas_out_v1

Privacy: aggregates only; counts rounded to 10, cells < 20 suppressed; quantiles 5-95 % only; refuses to run if
the output folder exists; scans every output for 7+ digit runs and participant IDs before writing.
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

VERSION = "f1_atlas_v1.5 2026-10-03"
DIAG = {}  # v1.1: acquisition diagnostics captured while building the frame (aggregates only)
MINCELL = 20
MAIN_STAGES = ("00_00_visit", "02_00_visit")
LATER_STAGES = ("01_00_call", "02_00_visit", "03_00_call")
QS = (0.05, 0.25, 0.5, 0.75, 0.95)
TOST_DELTA = 0.05          # equivalence margin for bed contrasts, SD units
MIN_STARTERS = 100         # report a drug-class starter model only above this many starters
LOG = []
TZ = "Asia/Jerusalem"


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    LOG.append(line)
    print(line, flush=True)


# ============================================================================ privacy helpers

def rc(n):
    """Rounded count, or None when below the minimum cell size."""
    n = int(n)
    return None if n < MINCELL else int(round(n / 10.0) * 10)


def cnt(n):
    """Count for log lines: rounded to 10 (<20 shown as '<20'), thousands separated (no long digit runs)."""
    r = rc(n)
    return "<20" if r is None else f"{r:,}"


def fnum(x):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    x = float(x)
    if not np.isfinite(x):
        return None
    if x == 0:
        return "0"
    a = abs(x)
    if 1e-3 <= a < 1e5:
        return f"{x:.4g}"
    return f"{x:.3e}"


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating, int, np.integer, bool, np.bool_)):
        return fnum(obj)
    return obj


def privacy_check(text, pids):
    m = re.search(r"\d{7,}", text)
    if m:
        raise SystemExit(f"PRIVACY ABORT: digit run '{m.group(0)[:3]}...' in output")
    for p in pids:
        if p in text:
            raise SystemExit("PRIVACY ABORT: participant id in output")


def qtiles(s):
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < MINCELL:
        return {"n": rc(len(s))}
    return {"n": rc(len(s)), **{f"q{int(q * 100):02d}": float(s.quantile(q)) for q in QS}}


def pct(lo_hi_arr):
    a = np.asarray(lo_hi_arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return [None, None]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


# ============================================================================ loading

def rd(root, rel, columns=None):
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        log(f"missing table {rel}")
        return None
    try:
        d = pd.read_parquet(p, columns=columns)
    except Exception as ex:  # a requested column may be absent
        log(f"read {rel} with columns failed ({type(ex).__name__}); reading all columns")
        d = pd.read_parquet(p)
    d = d.reset_index()
    d["participant_id"] = d["participant_id"].astype(str)
    if "research_stage" in d.columns:
        d["research_stage"] = d["research_stage"].astype(str)
    return d


def num(d, cols):
    for c in cols:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
            d[c] = d[c].replace([np.inf, -np.inf], np.nan)
    return d


def rowmean(d, cols):
    cols = [c for c in cols if c in d.columns]
    if not cols:
        return pd.Series(np.nan, index=d.index)
    return d[cols].astype(float).mean(axis=1, skipna=True)


def per_visit(d, cols, how="mean"):
    cols = [c for c in cols if c in d.columns]
    g = d.groupby(["participant_id", "research_stage"])[cols]
    out = g.median() if how == "median" else g.mean()
    return out.reset_index()


def first_val(d, col, name):
    if col not in d.columns:
        return None
    return d.groupby(["participant_id", "research_stage"])[col].min().reset_index().rename(columns={col: name})


def to_local(ts):
    ts = pd.to_datetime(ts, errors="coerce", utc=True)
    try:
        return ts.dt.tz_convert(TZ)
    except Exception:
        return ts


ATC_CLASSES = {
    "antihtn": ("C02", "C03", "C07", "C08", "C09"),
    "statin": ("C10AA",),
    "lipid_any": ("C10",),
    "glp1": ("A10BJ",),
    "sglt2": ("A10BK",),
    "metformin": ("A10BA02",),
    "insulin": ("A10A",),
    "diabetes_any": ("A10",),
}


def atc_flags(med):
    """Per (participant, stage) medication-class flags from the ATC list columns (vectorised)."""
    parts = []
    for c in ("atc3", "atc4", "atc5"):
        if c not in med.columns:
            continue
        e = med[["participant_id", "research_stage", c]].explode(c)
        e = e.rename(columns={c: "code"}).dropna(subset=["code"])
        parts.append(e)
    keys = med[["participant_id", "research_stage"]].drop_duplicates()
    if not parts:
        out = keys.copy()
        for k in ATC_CLASSES:
            out[k] = 0.0
        return out
    e = pd.concat(parts, ignore_index=True)
    e["code"] = e["code"].astype(str)
    for k, ps in ATC_CLASSES.items():
        e[k] = e["code"].str.startswith(ps).astype(float)
    f = e.groupby(["participant_id", "research_stage"])[list(ATC_CLASSES)].max().reset_index()
    out = keys.merge(f, how="left")
    for k in ATC_CLASSES:
        out[k] = out[k].fillna(0.0)
    return out


# ============================================================================ hypoxic burden (S3 inputs)

HB_RESP = ("A/H obstructive", "A/H central", "A/H unclassified")
_EVT = {}


def load_resp_events(root, files):
    """Respiratory events per participant: absolute event end (ns, UTC) and duration (s)."""
    out = {}
    n_all = 0
    for rel in files:
        p = os.path.join(root, "sleep", str(rel).lstrip("./"))
        if not os.path.exists(p):
            continue
        try:
            e = pd.read_parquet(p, columns=["event", "event_end", "start", "end"])
        except Exception as ex:
            log(f"events {os.path.basename(p)} unreadable: {type(ex).__name__}")
            continue
        n_all += len(e)
        e = e[e["event"].astype(str).isin(HB_RESP)].reset_index()
        if not len(e):
            continue
        end = pd.to_datetime(e["event_end"], errors="coerce", utc=True).dt.as_unit("ns")  # HPP files are [us]
        dur = (pd.to_numeric(e["end"], errors="coerce") - pd.to_numeric(e["start"], errors="coerce")).clip(5, 120)
        ok = end.notna().values
        e = pd.DataFrame({"pid": e["participant_id"].astype(str).values[ok],
                          "end_ns": end[ok].astype("int64").values, "dur": dur.values[ok]})
        for pid, g in e.groupby("pid"):
            out.setdefault(pid, []).append(g[["end_ns", "dur"]].to_numpy(dtype=float))
    out = {k: np.vstack(v) for k, v in out.items()}
    for k in out:
        out[k] = out[k][np.argsort(out[k][:, 0])]
    log(f"respiratory events: {cnt(sum(len(v) for v in out.values()))} of {cnt(n_all)} event rows, "
        f"{cnt(len(out))} participants")
    return out


def _read_spo2(night_dir):
    fs = sorted(glob.glob(os.path.join(night_dir, "source=spo2", "*.parquet")))
    if not fs:
        return None, None
    d = pd.concat([pd.read_parquet(f) for f in fs])
    ts = pd.to_datetime(d.index.get_level_values("collection_timestamp"), utc=True).as_unit("ns")  # [us] in HPP
    v = pd.to_numeric(d["values"], errors="coerce").to_numpy(dtype=float)
    o = np.argsort(ts.asi8)
    return ts.asi8[o], v[o]


def hb_night(args):
    """Hypoxic burden for one night.

    SpO2 is put on a regular grid (1 s for a 1 Hz series); values outside 50-100 % are invalid.  Respiratory
    events whose end lies inside the recording are taken from the events table.  The participant-night
    ensemble average of SpO2 around event ends (-120..+120 s) defines the search window (last peak before the
    nadir to first peak after it).  Per event, the area between the pre-event baseline (max SpO2 in the 100 s
    before event end) and the SpO2 curve inside the window is summed once per second (overlaps counted once).
    HB = total area (%min) / sleep time (h).  Also returns the sleep hypoxic load below 90 % (%min/h), mean SpO2,
    time below 90 % (% of valid time) and the valid hours.
    """
    key, night_dir, pid, tst_h = args
    res = {"key": key, "hb": np.nan, "hload90": np.nan, "spo2_mean_ts": np.nan, "t90_pct_ts": np.nan,
           "valid_h": np.nan, "n_resp": 0}
    try:
        ts, v = _read_spo2(night_dir)
    except Exception:
        return res
    if ts is None or len(ts) < 10:
        return res
    dt = float(np.median(np.diff(ts))) / 1e9
    if not np.isfinite(dt) or dt <= 0:
        return res
    t = (ts - ts[0]) / 1e9
    if dt < 0.5:  # HPP SpO2 series: 1 sample/s stored with 1 ms timestamp steps (sandbox 2026-10-03)
        t, dt = t / dt, 1.0
    step = 1.0 if dt < 1.5 else dt
    gi = np.round(t / step).astype(np.int64)
    n = int(gi.max()) + 1
    if n > 200000:
        return res
    s = np.full(n, np.nan)
    v = np.where((v >= 50) & (v <= 100), v, np.nan)
    s[gi] = v
    valid = np.isfinite(s)
    valid_h = valid.sum() * step / 3600.0
    res["valid_h"] = valid_h
    if valid.sum() < 10:
        return res
    res["spo2_mean_ts"] = float(np.nanmean(s))
    res["t90_pct_ts"] = float(100.0 * np.mean(s[valid] < 90))
    hours = tst_h if (tst_h is not None and np.isfinite(tst_h) and tst_h > 0) else valid_h
    res["hload90"] = float(np.nansum(np.clip(90.0 - s, 0, None)) * step / 60.0 / hours) if hours > 0 else np.nan
    ev = _EVT.get(pid)
    if ev is None or step > 2 or valid_h < 2:
        return res
    t0 = ts[0]
    ends = (ev[:, 0] - t0) / 1e9
    keep = (ends > 120) & (ends < (n - 1) * step - 120)
    ends = ends[keep]
    res["n_resp"] = int(len(ends))
    if len(ends) == 0:
        res["hb"] = 0.0 if valid_h >= 4 else np.nan
        return res
    W = int(round(120 / step))
    idx = np.round(ends / step).astype(np.int64)
    win = np.arange(-W, W + 1)
    off = win * step
    w0, w1 = int(np.searchsorted(off, -30)), int(np.searchsorted(off, 60))   # default window (< 3 events)
    if len(ends) >= 3:
        avg = np.nanmean(s[idx[:, None] + win[None, :]], axis=0)
        sel = (off >= -30) & (off <= 90) & np.isfinite(avg)
        if sel.any():
            nad = int(np.argmin(np.where(sel, avg, np.inf)))
            pre = np.arange(int(np.searchsorted(off, -90)), nad + 1)
            post = np.arange(nad, int(np.searchsorted(off, 90, side="right")))
            if np.isfinite(avg[pre]).any() and np.isfinite(avg[post]).any():
                # window = from the end of the pre-event plateau to the return to the post-event plateau:
                # the last / first point within 5 % of the drop from the respective peak (flat, noisy
                # plateaus would otherwise move a plain argmax far from the desaturation)
                vn = avg[nad]
                p0 = np.nanmax(avg[pre])
                p1 = np.nanmax(avg[post])
                a0 = np.flatnonzero(avg[pre] >= p0 - 0.05 * (p0 - vn))
                a1 = np.flatnonzero(avg[post] >= p1 - 0.05 * (p1 - vn))
                w0 = int(pre[a0[-1]]) if len(a0) else w0
                w1 = int(post[a1[0]]) if len(a1) else w1
    res["hb_w0"], res["hb_w1"] = float(off[w0]), float(off[w1])
    if w1 <= w0:
        return res
    used = np.zeros(n, dtype=bool)
    area = 0.0
    B = int(round(100 / step))
    for i in idx:
        base = np.nanmax(s[max(i - B, 0): i + 1]) if np.isfinite(s[max(i - B, 0): i + 1]).any() else np.nan
        if not np.isfinite(base):
            continue
        a, b = i + win[w0], i + win[w1] + 1
        seg = slice(a, b)
        d = np.clip(base - s[seg], 0, None)
        d = np.where(np.isfinite(d) & ~used[seg], d, 0.0)
        area += d.sum() * step / 60.0
        used[seg] = True
    res["hb"] = float(area / hours) if hours > 0 else np.nan
    return res


def hypoxic_burden(root, workers, max_nights=None):
    s = rd(root, "sleep/sleep.parquet",
           columns=["channels_time_series", "events_time_series", "total_sleep_time", "night",
                    "quality_score_spo2"])
    if s is None:
        return None, {}
    s = s[s["research_stage"].isin(MAIN_STAGES)].copy()
    tst = pd.to_numeric(s["total_sleep_time"], errors="coerce").astype(float)
    unit = "s" if tst.median() > 2000 else "min"
    s["tst_h"] = tst / (3600.0 if unit == "s" else 60.0)
    global _EVT
    _EVT = load_resp_events(root, sorted(set(s["events_time_series"].dropna().astype(str))))
    jobs = []
    for i, r in enumerate(s.itertuples(index=False)):
        if not isinstance(r.channels_time_series, str):
            continue
        nd = os.path.join(root, "sleep", r.channels_time_series.lstrip("./"))
        jobs.append((i, nd, r.participant_id, r.tst_h))
    if max_nights:
        jobs = jobs[:max_nights]
    t0 = time.time()
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            out = pool.map(hb_night, jobs, chunksize=20)
    else:
        out = [hb_night(j) for j in jobs]
    r = pd.DataFrame(out).set_index("key")
    s = s.reset_index(drop=True).join(r, how="left")
    log(f"hypoxic burden: {cnt(len(jobs))} nights in {time.time() - t0:.0f}s; HB computed for "
        f"{cnt(int(s['hb'].notna().sum()))}")
    ok = s["valid_h"] >= 4
    meta = {"nights": rc(len(s)), "nights_with_series": rc(len(jobs)), "nights_valid_4h": rc(int(ok.sum())),
            "nights_hb": rc(int((ok & s["hb"].notna()).sum())), "tst_unit_assumed": unit,
            "valid_hours": qtiles(s["valid_h"]), "resp_events_per_night": qtiles(s.loc[ok, "n_resp"]),
            "window_start_s": qtiles(s.loc[ok, "hb_w0"]) if "hb_w0" in s else None,
            "window_end_s": qtiles(s.loc[ok, "hb_w1"]) if "hb_w1" in s else None}
    s = s[ok]
    v = s.groupby(["participant_id", "research_stage"])[["hb", "hload90", "spo2_mean_ts", "t90_pct_ts"]].median()
    v["n_nights_hb"] = s.groupby(["participant_id", "research_stage"])["hb"].count()
    return v.reset_index(), meta


# ============================================================================ calendar adjustment (v1.2)

CAL_BEDS = ["pwv", "pwv_r", "pwv_l", "imt", "art_w", "abi_min"]
CAL_ADJUST = True
DAY_ADJUST = True  # v1.4: remove measurement-day (session) effects after the quarter adjustment
SESSION_ADJUST = True  # v1.5: refine the day adjustment by a sub-day time kernel (joint day + session BLUP)


def calendar_quarter(fr):
    """Calendar quarter of the vascular acquisition (device timestamp, else visit date)."""
    t = pd.to_datetime(fr["visit_date"], errors="coerce")
    if "vas_ts" in fr.columns:
        t = pd.to_datetime(fr["vas_ts"], errors="coerce", utc=True).dt.tz_localize(None).fillna(t)
    return t.dt.year.astype("Int64").astype(str) + "Q" + t.dt.quarter.astype("Int64").astype(str), t


def calendar_adjust(fr):
    """Remove calendar-quarter effects (estimated with age, age^2 and sex; both visits pooled) from each device bed.
    Quarters with < MINCELL measurements are left unadjusted. The weighted mean effect is zero, so the overall mean is
    unchanged. The raw values are kept as <bed>_raw."""
    q, _ = calendar_quarter(fr)
    fr = fr.copy()
    fr["_calq"] = q.where(~q.str.contains("NA"), None)
    eff_out = {}
    for c in [b for b in CAL_BEDS if b in fr.columns]:
        fr[c + "_raw"] = fr[c]
        e = fr.dropna(subset=[c, "age", "sex_male", "_calq"])
        n_q = e["_calq"].value_counts()
        keep = n_q[n_q >= MINCELL].index
        e = e[e["_calq"].isin(keep)]
        if len(keep) < 2 or len(e) < 200:
            continue
        qs = sorted(keep)
        D = pd.get_dummies(pd.Categorical(e["_calq"], categories=qs), dtype=float).values[:, 1:]
        a = e["age"].astype(float).values
        X = np.column_stack([np.ones(len(e)), a, (a - 50) ** 2, e["sex_male"].astype(float).values, D])
        b = beta_only(e[c].astype(float).values, X)
        eff = np.r_[0.0, b[4:]]
        w = np.array([n_q[k] for k in qs], dtype=float)
        eff = eff - (eff * w).sum() / w.sum()
        emap = dict(zip(qs, eff))
        adj = fr["_calq"].map(emap).astype(float).fillna(0.0)
        fr[c] = fr[c] - adj.where(fr[c].notna(), 0.0)
        eff_out[c] = {k: {"effect": float(emap[k]), "n": rc(n_q[k])} for k in qs}
    DIAG["calendar_effects"] = eff_out
    return fr.drop(columns=["_calq"])


def day_adjust(fr):
    """v1.4: remove measurement-day (session) effects from each device bed, after the quarter adjustment.
    Residuals of the bed on age, (age-50)^2 and sex; day variance share by one-way ANOVA moments over days with >= 2
    measurements; each value is corrected by the leave-one-out shrunken mean residual of the other measurements of the
    same day, sum_{j != i} r_j / (n_d - 1 + k), k = var_within / var_day (BLUP of the day effect without the person's own
    value). Days with one measurement and beds without a positive day variance are left unchanged."""
    _, t = calendar_quarter(fr)
    fr = fr.copy()
    day = t.dt.strftime("%Y-%m-%d")
    out = {}
    for c in [b for b in CAL_BEDS if b in fr.columns]:
        ok = fr[c].notna() & fr["age"].notna() & fr["sex_male"].notna() & day.notna()
        if ok.sum() < 200:
            continue
        a = fr.loc[ok, "age"].astype(float).values
        X = np.column_stack([np.ones(ok.sum()), a, (a - 50) ** 2, fr.loc[ok, "sex_male"].astype(float).values])
        y = fr.loc[ok, c].astype(float).values
        r = y - X @ beta_only(y, X)
        g = pd.DataFrame({"r": r, "d": day[ok].values})
        st = g.groupby("d")["r"].agg(["sum", "count"])
        m = st[st["count"] >= 2]
        if len(m) < 20:
            continue
        nd = m["count"].values.astype(float)
        N, D = nd.sum(), len(nd)
        gm = g.loc[g["d"].isin(m.index), "r"]
        ss_tot = ((gm - gm.mean()) ** 2).sum()
        ss_b = ((m["sum"].values / nd - gm.mean()) ** 2 * nd).sum()
        msw = (ss_tot - ss_b) / (N - D)
        msb = ss_b / (D - 1)
        n0 = (N - (nd ** 2).sum() / N) / (D - 1)
        vday = max(0.0, (msb - msw) / n0)
        if vday <= 0:
            out[c] = {"day_share": 0.0, "n_days": rc(D)}
            continue
        k = msw / vday
        S = g["d"].map(st["sum"]).values
        n = g["d"].map(st["count"]).values.astype(float)
        u = np.where(n >= 2, (S - g["r"].values) / (n - 1 + k), 0.0)
        fr.loc[ok, c] = y - u
        out[c] = {"day_share": float(vday / (vday + msw)), "n_days": rc(D), "k": float(k),
                  "median_n_per_day": float(np.median(nd))}
    DIAG["day_effects"] = out
    return fr


SESSION_EDGES = (0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 360)  # v1.5: |dt| bins (min), last bin open
SESSION_TAUS = (5, 10, 15, 20, 30, 45, 60, 90, 120, 180)  # kernel range grid (min)
SESSION_MIN_BIN = 100  # pairs per bin used in the kernel fit
SESSION_MIN_Z = 3.0    # kernel amplitude / jackknife SE needed to apply it
SESSION_JK = 20        # jackknife groups of days
SESSION_MAX_NB = 60    # kriging neighbours per measurement (nearest in time)


def rcp(n):
    """v1.5: rounded pair count; counts of a million or more as a string (no 7-digit run in the outputs)."""
    return rc(n) if n < 10 ** 6 else ">=1e6"


def relation_components(root):
    """v1.5: participant -> relation component (any participant_relation type, transitive); None if unreadable."""
    try:
        rel = rd(root, "participant_relation/participant_relation.parquet") if root else None
        if rel is None:
            return None
        rel = rel[rel["relation_type"].notna() & rel["relative_participant_id"].notna()]
        par = {}

        def find(x):
            par.setdefault(x, x)
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        for x, y in zip(rel["participant_id"].astype(str), rel["relative_participant_id"].astype("int64").astype(str)):
            rx, ry = find(x), find(y)
            if rx != ry:
                par[rx] = ry
        return {k: find(k) for k in list(par)}
    except Exception as ex:
        DIAG["session_relation_error"] = f"{type(ex).__name__}: {str(ex)[:150]}"
        return None


def kernel_fit(dt, pr, V, grp):
    """v1.5: WLS fit of cov(dt) = b0 + s exp(-dt/tau) (b0 = day component, s = time kernel, both >= 0) to the mean
    residual products of same-day pairs by |dt| bin (bins with >= SESSION_MIN_BIN pairs, weights = pairs), tau on a
    grid; z = s / jackknife SE over SESSION_JK groups of days (grp). Returns bin n, bin cov, (tau, b0, s, z) or None."""
    nbin = len(SESSION_EDGES)
    b = np.searchsorted(np.asarray(SESSION_EDGES, float), dt, side="right") - 1

    def fit(m, taus):
        nb = np.bincount(b[m], minlength=nbin).astype(float)
        cov = np.bincount(b[m], weights=pr[m], minlength=nbin) / np.maximum(nb, 1)
        use = nb >= SESSION_MIN_BIN
        if use.sum() < 3:
            return nb, cov, None
        w, cv, best = nb[use], cov[use], None
        for tau in taus:
            E = (np.bincount(b[m], weights=np.exp(-dt[m] / tau), minlength=nbin) / np.maximum(nb, 1))[use]
            Xk = np.column_stack([np.ones(len(w)), E])
            bt = np.linalg.lstsq((Xk.T * w) @ Xk, (Xk.T * w) @ cv, rcond=None)[0]
            if bt.min() < 0:  # two-parameter NNLS: the better of the one-parameter fits
                cands = [np.array([max(np.average(cv, weights=w), 0.0), 0.0]),
                         np.array([0.0, max((w * E * cv).sum() / (w * E * E).sum(), 0.0)])]
                bt = min(cands, key=lambda q: (w * (cv - Xk @ q) ** 2).sum())
            sse = float((w * (cv - Xk @ bt) ** 2).sum())
            if best is None or sse < best[0] - 1e-15:
                best = (sse, tau, float(bt[0]), float(bt[1]))
        return nb, cov, best
    nb, cov, best = fit(np.ones(len(dt), bool), SESSION_TAUS)
    if best is None:
        return nb, cov, None
    _, tau, b0, s = best
    z = 0.0
    if s > 0:
        sj = np.array([q[3] for q in (fit(grp != g, (tau,))[2] for g in range(SESSION_JK)) if q is not None])
        se = np.sqrt((len(sj) - 1) / len(sj) * ((sj - sj.mean()) ** 2).sum()) if len(sj) > 2 else np.inf
        z = float(s / se) if se > 0 else 0.0
    return nb, cov, (tau, b0, s, z)


def krige_day(r, t, s, tau, V, b0=0.0, m=SESSION_MAX_NB):
    """v1.5: leave-one-out kriging of the measurement artefact within one day: a_i = c_{i,-i} K_{-i,-i}^{-1} r_{-i},
    K = V on the diagonal, b0 + s exp(-|dt|/tau) off it (b0 only when a time is missing). Up to m neighbours: closed
    form r_i - (K^-1 r)_i / (K^-1)_ii; larger days: the m measurements nearest in time to i."""
    n = len(r)
    if n - 1 <= m:
        K = b0 + s * np.nan_to_num(np.exp(-np.abs(t[:, None] - t[None, :]) / tau), nan=0.0)
        np.fill_diagonal(K, V)
        Q = np.linalg.inv(K)
        return r - (Q @ r) / np.diag(Q)
    out = np.empty(n)
    for i in range(n):
        d = np.nan_to_num(np.abs(t - t[i]), nan=np.inf)
        d[i] = -1.0
        nb = np.argsort(d, kind="stable")[1:m + 1]
        K = b0 + s * np.nan_to_num(np.exp(-np.abs(t[nb][:, None] - t[nb][None, :]) / tau), nan=0.0)
        np.fill_diagonal(K, V)
        out[i] = (b0 + s * np.exp(-d[nb] / tau)) @ np.linalg.solve(K, r[nb])
    return out


def session_adjust(fr, root=None, apply=True):
    """v1.5: day_adjust (v1.4), refined by a joint day + time-kernel model where a sub-day kernel is supported.
    r = residual of the quarter-adjusted bed on age, (age-50)^2 and sex. Same-day pairs of different, unrelated people
    (relation components when participant_relation is readable, else all pairs) with vascular timestamps give the
    residual covariance by |dt| bin; kernel_fit gives cov(dt) = b0 + s exp(-dt/tau). Where s > 0 and
    z >= SESSION_MIN_Z, each value is corrected by krige_day (leave-one-out BLUP of its day + session artefact from the
    other measurements of the day, b0 only for rows without a timestamp; with s = 0 this is the day_adjust formula).
    Other beds, and all beds when apply=False, keep the day_adjust values; the kernel is reported in both cases."""
    out = {}
    fd = day_adjust(fr)
    DIAG["session_kernel"] = out
    if "vas_ts" not in fr.columns:
        return fd
    _, t = calendar_quarter(fr)
    day = t.dt.strftime("%Y-%m-%d")
    ts = pd.to_datetime(fr["vas_ts"], errors="coerce", utc=True)
    tm = ((ts - ts.min()).dt.total_seconds() / 60.0).values
    comp = relation_components(root)
    DIAG["session_relations_excluded"] = comp is not None
    pid = fr["participant_id"].astype(str).values
    cid = pd.factorize(np.array([comp.get(p, p) for p in pid] if comp else pid, dtype=object))[0]
    pcode = pd.factorize(pid)[0]
    labs = [f"{a}-{b}" for a, b in zip(SESSION_EDGES[:-1], SESSION_EDGES[1:])] + [f"{SESSION_EDGES[-1]}+"]
    for c in [b for b in CAL_BEDS if b in fr.columns]:
        ok = (fr[c].notna() & fr["age"].notna() & fr["sex_male"].notna() & day.notna()).values
        if ok.sum() < 200:
            continue
        a = fr.loc[ok, "age"].astype(float).values
        X = np.column_stack([np.ones(ok.sum()), a, (a - 50) ** 2, fr.loc[ok, "sex_male"].astype(float).values])
        y = fr.loc[ok, c].astype(float).values
        r = y - X @ beta_only(y, X)
        V = float(r.var())
        tt, cc, pc = tm[ok], cid[ok], pcode[ok]
        dk = pd.factorize(day[ok])[0]
        o = np.lexsort((tt, dk))
        days = [g for g in np.split(o, np.flatnonzero(np.diff(dk[o])) + 1) if len(g) >= 2]
        I, J = [np.zeros(0, int)], [np.zeros(0, int)]
        for g in days:
            g = g[np.isfinite(tt[g])]
            iu, ju = np.triu_indices(len(g), 1)
            k = (cc[g][iu] != cc[g][ju]) & (pc[g][iu] != pc[g][ju])
            I.append(g[iu[k]])
            J.append(g[ju[k]])
        I, J = np.concatenate(I), np.concatenate(J)
        dt = np.abs(tt[I] - tt[J])
        nb, cov, best = kernel_fit(dt, r[I] * r[J], V, dk[I] % SESSION_JK)
        e = {"V": V, "n_pairs": rcp(len(dt)), "n_days": rc(len(days)), "applied": False,
             "bins": {lab: {"n": rcp(nb[k]), "r": float(cov[k] / V) if nb[k] >= MINCELL else None}
                      for k, lab in enumerate(labs)}}
        out[c] = e
        y4 = fd.loc[ok, c].astype(float).values
        y2 = y4
        if best is not None:
            tau, b0, s, z = best
            e.update({"tau": tau, "day_share": b0 / V, "share": s / V, "z": z})
            e["applied"] = bool(apply and s > 0 and z >= SESSION_MIN_Z)
            if e["applied"]:
                f = min(1.0, 0.95 * V / (b0 + s))
                ah = np.zeros(len(r))
                for g in days:
                    ah[g] = krige_day(r[g], tt[g], s * f, tau, V, b0 * f)
                y2 = y - ah
                fd.loc[ok, c] = y2
                e["change_vs_day_sd"] = float(np.std(y2 - y4) / np.sqrt(V))
        r2 = y2 - X @ beta_only(y2, X)  # final values (v1.5, or v1.4 where not applied): bin r after
        bi = np.searchsorted(np.asarray(SESSION_EDGES, float), dt, side="right") - 1
        nb2 = np.bincount(bi, minlength=len(labs))
        cov2 = np.bincount(bi, weights=r2[I] * r2[J], minlength=len(labs)) / np.maximum(nb2, 1)
        for k, lab in enumerate(labs):
            if nb2[k] >= MINCELL:
                e["bins"][lab]["r_after"] = float(cov2[k] / r2.var())
    return fd


# ============================================================================ frame

def build_frame(root, hb=None):
    t0 = time.time()
    pop = rd(root, "population/population.parquet")
    ev = rd(root, "events/events.parquet")
    pop["sex_male"] = pd.to_numeric(pop["sex"].astype(str), errors="coerce")  # coding 9: 0 female, 1 male
    pop["birth_month"] = pd.to_numeric(pop["month_of_birth"].astype(str), errors="coerce") \
        if "month_of_birth" in pop.columns else np.nan
    pop = pop[["participant_id", "sex_male", "year_of_birth", "birth_month"]].drop_duplicates("participant_id")
    ev = ev[ev["research_stage"].isin(MAIN_STAGES)]
    ev = num(ev, ["age_at_research_stage"])
    ev["visit_date"] = pd.to_datetime(ev["research_stage_date"], errors="coerce")
    ev = ev.groupby(["participant_id", "research_stage"]).agg(
        age=("age_at_research_stage", "first"), visit_date=("visit_date", "first")).reset_index()

    tabs = {}
    v = rd(root, "vascular_health/vascular_health.parquet")
    vcols = ["r_ankle_pressure", "l_ankle_pressure", "r_brachial_pressure", "l_brachial_pressure",
             "r_abi", "l_abi", "from_r_thigh_to_r_ankle_pwv", "from_l_thigh_to_l_ankle_pwv"]
    v = num(v, vcols)
    v["pwv_r"] = v["from_r_thigh_to_r_ankle_pwv"]
    v["pwv_l"] = v["from_l_thigh_to_l_ankle_pwv"]
    v["pwv"] = rowmean(v, ["pwv_r", "pwv_l"])
    v["abi_r"], v["abi_l"] = v["r_abi"], v["l_abi"]
    v["abi_min"] = v[["r_abi", "l_abi"]].min(axis=1, skipna=True)
    v["abi_max"] = v[["r_abi", "l_abi"]].max(axis=1, skipna=True)
    v["dev_bsbp"] = rowmean(v, ["r_brachial_pressure", "l_brachial_pressure"])
    v["dev_bsbp_r"], v["dev_bsbp_l"] = v["r_brachial_pressure"], v["l_brachial_pressure"]
    v["dev_asbp"] = rowmean(v, ["r_ankle_pressure", "l_ankle_pressure"])
    try:  # v1.1: arrays per visit and PWV by array order (between-visit PWV shift diagnosis)
        k = v.groupby(["participant_id", "research_stage"]).size()
        DIAG["vascular_arrays_per_visit"] = {st: qtiles(k[k.index.get_level_values(1) == st]) for st in MAIN_STAGES}
        if "collection_timestamp" in v.columns:
            vo = v[["participant_id", "research_stage", "pwv"]].copy()
            vo["_t"] = pd.to_datetime(v["collection_timestamp"], errors="coerce", utc=True)
            vo["_ord"] = vo.groupby(["participant_id", "research_stage"])["_t"].rank(method="first")
            DIAG["pwv_by_array_order"] = {st: {str(o): qtiles(vo.loc[(vo["research_stage"] == st) & (vo["_ord"] == o), "pwv"])
                                               for o in (1, 2, 3)} for st in MAIN_STAGES}
    except Exception as ex:
        DIAG["arrays_error"] = f"{type(ex).__name__}: {str(ex)[:150]}"
    vv = per_visit(v, ["pwv", "pwv_r", "pwv_l", "abi_r", "abi_l", "abi_min", "abi_max", "dev_bsbp", "dev_bsbp_r",
                       "dev_bsbp_l", "dev_asbp"])
    ts = first_val(v, "collection_timestamp", "vas_ts")
    if ts is not None:
        vv = vv.merge(ts, how="left")
    tabs["vascular"] = vv

    b = rd(root, "blood_pressure/blood_pressure.parquet")
    bmap = {"sitting_blood_pressure_systolic": "sbp_sit", "sitting_blood_pressure_diastolic": "dbp_sit",
            "sitting_blood_pressure_pulse_rate": "hr_sit", "lying_blood_pressure_systolic": "sbp_lie",
            "lying_blood_pressure_diastolic": "dbp_lie", "lying_blood_pressure_pulse_rate": "hr_lie",
            "standing_one_min_blood_pressure_systolic": "sbp_st1",
            "standing_three_min_blood_pressure_systolic": "sbp_st3",
            "sitting_second_blood_pressure_systolic": "sbp_sit2",
            "sitting_second_blood_pressure_diastolic": "dbp_sit2"}
    b = num(b, list(bmap)).rename(columns=bmap)
    bb = per_visit(b, list(bmap.values()))
    ts = first_val(b, "collection_timestamp", "bp_ts")
    if ts is not None:
        bb = bb.merge(ts, how="left")
    tabs["bp"] = bb

    a = rd(root, "anthropometrics/anthropometrics.parquet")
    amap = {"bmi": "bmi", "waist_circumference": "waist", "hip_circumference": "hip",
            "waist_to_hip_ratio": "whr", "height": "height", "weight": "weight"}
    a = num(a, list(amap)).rename(columns=amap)
    tabs["anthro"] = per_visit(a, list(amap.values()))

    bc = rd(root, "body_composition/body_composition.parquet")
    bcmap = {"total_scan_vat_mass": "vat", "total_scan_sat_mass": "sat",
             "body_comp_total_fat_mass": "fat_total", "body_comp_total_lean_mass": "lean_total"}
    legc = ["body_comp_leg_left_fat_mass", "body_comp_leg_right_fat_mass", "body_comp_leg_left_lean_mass",
            "body_comp_leg_right_lean_mass"]
    bc = num(bc, list(bcmap) + legc).rename(columns=bcmap)
    bc["leg_fat"] = bc[legc[:2]].sum(axis=1, min_count=2) if legc[0] in bc.columns else np.nan
    bc["leg_lean"] = bc[legc[2:]].sum(axis=1, min_count=2) if legc[2] in bc.columns else np.nan
    tabs["dxa"] = per_visit(bc, list(bcmap.values()) + ["leg_fat", "leg_lean"])

    c = rd(root, "carotid_ultrasound/carotid_ultrasound.parquet")
    c = num(c, ["imt_left", "imt_right"])
    c["imt"] = rowmean(c, ["imt_left", "imt_right"])
    c["imt_max"] = c[["imt_left", "imt_right"]].max(axis=1, skipna=True)
    c["imt_l"], c["imt_r"] = c["imt_left"], c["imt_right"]
    tabs["carotid"] = per_visit(c, ["imt", "imt_max", "imt_l", "imt_r"])

    f = rd(root, "fundus/microvasculature.parquet")
    fmap = {"artery_average_width": "art_w", "vein_average_width": "vein_w", "fractal_dimension": "fd",
            "vessel_density": "vdens", "artery_distance_tortuosity": "art_tort",
            "vein_distance_tortuosity": "vein_tort"}
    for fn, short in fmap.items():
        f = num(f, [f"{fn}_left", f"{fn}_right"])
        f[short] = rowmean(f, [f"{fn}_left", f"{fn}_right"])
    for side, col in (("l", "artery_average_width_left"), ("r", "artery_average_width_right")):
        f["art_w_" + side] = f[col] if col in f.columns else np.nan
    f["avr"] = f["art_w"] / f["vein_w"]
    tabs["retina"] = per_visit(f, list(fmap.values()) + ["avr", "art_w_l", "art_w_r"])

    lv = rd(root, "liver_ultrasound/liver_ultrasound.parquet", columns=["attenuation_coefficient_qbox"])
    if lv is not None:  # F1 uses liver attenuation only, as one exposure row
        lv = num(lv, ["attenuation_coefficient_qbox"]).rename(columns={"attenuation_coefficient_qbox": "att"})
        tabs["liver"] = per_visit(lv, ["att"], "median")

    s = rd(root, "sleep/sleep.parquet", columns=["ahi", "odi", "saturation_below_90", "saturation_mean",
                                                  "total_sleep_time", "desaturations_mean_nadir"])
    smap = {"ahi": "ahi", "odi": "odi", "saturation_below_90": "t90", "saturation_mean": "spo2_mean",
            "total_sleep_time": "tst", "desaturations_mean_nadir": "desat_nadir"}
    s = num(s, list(smap)).rename(columns=smap)
    s = s[s["research_stage"].isin(MAIN_STAGES)]
    ss = per_visit(s, list(smap.values()), "median")
    tabs["sleep"] = ss.merge(s.groupby(["participant_id", "research_stage"]).size().rename("n_nights")
                             .reset_index(), how="left")
    if hb is not None and len(hb):
        tabs["hb"] = hb

    ig = rd(root, "cgm/iglu.parquet", columns=["cgm_mean", "cgm_cv", "cgm_above_140", "cgm_gmi"])
    if ig is not None:
        ig = num(ig, ["cgm_mean", "cgm_cv", "cgm_above_140", "cgm_gmi"]).rename(columns={"cgm_above_140": "cgm_above140"})
        tabs["cgm"] = per_visit(ig, ["cgm_mean", "cgm_cv", "cgm_above140", "cgm_gmi"])

    e = rd(root, "ecg/ecg.parquet", columns=["r_mv_V5", "r_mv_V6", "s_mv_V1", "r_mv_aVL", "s_mv_V3", "qrs_ms",
                                              "hr_bpm"])
    if e is not None:
        e = num(e, ["r_mv_V5", "r_mv_V6", "s_mv_V1", "r_mv_aVL", "s_mv_V3", "qrs_ms", "hr_bpm"])
        e["sokolow"] = e["s_mv_V1"].abs() + e[["r_mv_V5", "r_mv_V6"]].max(axis=1, skipna=False)
        e["cornell"] = e["r_mv_aVL"].abs() + e["s_mv_V3"].abs()
        e["cornell_prod"] = e["cornell"] * e["qrs_ms"]
        e["ecg_hr"], e["qrs"] = e["hr_bpm"], e["qrs_ms"]
        tabs["ecg"] = per_visit(e, ["sokolow", "cornell", "cornell_prod", "ecg_hr", "qrs"])

    lf = rd(root, "lifestyle_and_environment/lifestyle_and_environment.parquet", columns=["smoking_current_status"])
    if lf is not None:
        st = pd.to_numeric(lf["smoking_current_status"].astype(object), errors="coerce")
        lf["smk_current"] = np.where(st.isin([1, 2]), 1.0, np.where(st == 0, 0.0, np.nan))
        tabs["smoke"] = lf.groupby(["participant_id", "research_stage"])["smk_current"].max().reset_index()

    med = rd(root, "medications/medications.parquet", columns=["atc3", "atc4", "atc5"])
    medflags = None
    if med is not None:
        medflags = atc_flags(med)
        tabs["meds"] = medflags[medflags["research_stage"].isin(MAIN_STAGES)]

    h = rd(root, "curated_phenotypes/hypertension.parquet",
           columns=["hypertension__curated_phenotype", "hypertension__self_reported_hypertension"])
    htn = None
    if h is not None:
        cur = h["hypertension__curated_phenotype"].astype(str).str.lower()
        h["htn_curated"] = np.where(cur.isin(["nan", "<na>", "none"]), np.nan,
                                    (cur.str.contains("hypertens") & ~cur.str.startswith("non")).astype(float))
        sr = h["hypertension__self_reported_hypertension"].astype(object)
        h["htn_self"] = pd.to_numeric(sr.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}), errors="coerce")
        htn = h.groupby(["participant_id", "research_stage"])[["htn_curated", "htn_self"]].max().reset_index()
        tabs["htn"] = htn[htn["research_stage"].isin(MAIN_STAGES)]

    fr = ev.copy()
    for k, t in tabs.items():
        t = t[t["research_stage"].isin(MAIN_STAGES)]
        fr = fr.merge(t, on=["participant_id", "research_stage"], how="left")
    fr = fr.merge(pop, on="participant_id", how="left")
    miss_age = fr["age"].isna() & fr["visit_date"].notna() & fr["year_of_birth"].notna()
    fr.loc[miss_age, "age"] = fr.loc[miss_age, "visit_date"].dt.year - pd.to_numeric(
        fr.loc[miss_age, "year_of_birth"], errors="coerce")
    for k in ATC_CLASSES:
        if k in fr.columns:
            fr[k] = fr[k].fillna(0.0)

    # labs: nearest draw to the visit date within 180 days
    bt = rd(root, "blood_tests/blood_tests.parquet",
            columns=["collection_date", "bt__triglycerides_float_value", "bt__ldl_cholesterol_float_value",
                     "bt__hdl_cholesterol_float_value", "bt__hba1c_float_value", "bt__glucose_float_value",
                     "bt__urate_float_value", "bt__creatinine_float_value", "urine__acr_float_value"])
    labmap = {"bt__triglycerides_float_value": "tg", "bt__ldl_cholesterol_float_value": "ldl",
              "bt__hdl_cholesterol_float_value": "hdl", "bt__hba1c_float_value": "hba1c",
              "bt__glucose_float_value": "glucose", "bt__urate_float_value": "urate",
              "bt__creatinine_float_value": "creat", "urine__acr_float_value": "acr"}
    bt = num(bt, list(labmap)).rename(columns=labmap)
    bt["lab_date"] = pd.to_datetime(bt["collection_date"], errors="coerce")
    labs = [c for c in labmap.values() if c in bt.columns]
    bt = bt.dropna(subset=["lab_date"])[["participant_id", "lab_date"] + labs]
    bt = bt.dropna(subset=labs, how="all")
    left = fr[["participant_id", "research_stage", "visit_date"]].dropna(subset=["visit_date"]).sort_values("visit_date")
    m = pd.merge_asof(left, bt.sort_values("lab_date"), left_on="visit_date", right_on="lab_date",
                      by="participant_id", direction="nearest", tolerance=pd.Timedelta(days=180))
    m["lab_gap_days"] = (m["lab_date"] - m["visit_date"]).dt.days
    fr = fr.merge(m.drop(columns=["visit_date", "lab_date"]), on=["participant_id", "research_stage"], how="left")

    cr = fr["creat"].copy()
    if cr.median(skipna=True) > 20:
        cr = cr / 88.4
    female = fr["sex_male"] == 0
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = cr / kappa
    fr["egfr"] = (142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.2 *
                  0.9938 ** fr["age"] * np.where(female, 1.012, 1.0))

    # derived variables
    for col in ("tg", "acr", "vat", "sat", "hb", "hload90"):
        if col in fr.columns:
            x = fr[col].where(fr[col] > 0)
            fr["log_" + col] = np.log(x)
    if "hb" in fr.columns:
        fr["log1p_hb"] = np.log1p(fr["hb"].clip(lower=0))
    for col in ("ahi", "odi", "t90"):
        if col in fr.columns:
            fr["log1p_" + col] = np.log1p(fr[col].clip(lower=0))
    fr["pp_sit"] = fr["sbp_sit"] - fr["dbp_sit"]
    fr["map_sit"] = fr["dbp_sit"] + (fr["sbp_sit"] - fr["dbp_sit"]) / 3.0
    if "vas_ts" in fr.columns:
        lt = to_local(fr["vas_ts"])
        fr["acq_hour"] = lt.dt.hour + lt.dt.minute / 60.0
        mon = lt.dt.month
    else:
        mon = fr["visit_date"].dt.month
    fr["season_sin"] = np.sin(2 * np.pi * mon / 12.0)
    fr["season_cos"] = np.cos(2 * np.pi * mon / 12.0)
    fr["bm_sin"] = np.sin(2 * np.pi * fr["birth_month"] / 12.0)
    fr["bm_cos"] = np.cos(2 * np.pi * fr["birth_month"] / 12.0)
    if CAL_ADJUST:  # v1.2: remove calendar-quarter measurement drift from the device beds (raw kept as <bed>_raw)
        fr = calendar_adjust(fr)
        if DAY_ADJUST:  # v1.4: and measurement-day (session) effects; v1.5: refined by a joint day + time kernel
            fr = session_adjust(fr, root, apply=SESSION_ADJUST)
    # CAVI0 (Spronck 2015): 2 rho PWV^2 / Pd - ln(Pd / Pref); Pd = sitting DBP, Pref = 100 mmHg
    pd_pa = fr["dbp_sit"] * 133.322
    fr["cavi0"] = (2 * 1050.0 * fr["pwv"] ** 2 / pd_pa) - np.log(fr["dbp_sit"] / 100.0)
    fr.loc[~np.isfinite(fr["cavi0"]), "cavi0"] = np.nan
    if "pwv_raw" in fr.columns:
        fr["cavi0_raw"] = (2 * 1050.0 * fr["pwv_raw"] ** 2 / pd_pa) - np.log(fr["dbp_sit"] / 100.0)
        fr.loc[~np.isfinite(fr["cavi0_raw"]), "cavi0_raw"] = np.nan
    # adverse-oriented copies (higher = worse) for bed contrasts
    for bed, sign in ORIENT.items():
        if bed in fr.columns:
            fr["o_" + bed] = sign * fr[bed]
    log(f"frame built: {cnt(len(fr))} visit rows in {time.time() - t0:.1f}s")
    return fr, medflags, htn


# ============================================================================ regression helpers

def ols(y, X):
    n, k = X.shape
    xtx = np.linalg.pinv(X.T @ X)
    b = xtx @ (X.T @ y)
    e = y - X @ b
    meat = (X * (e ** 2)[:, None]).T @ X
    V = xtx @ meat @ xtx * (n / max(n - k, 1))
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    r2 = 1 - (e @ e) / max(((y - y.mean()) ** 2).sum(), 1e-12)
    return b, se, r2


def beta_only(y, X):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def zs(s):
    s = s.astype(float)
    sd = s.std()
    return (s - s.mean()) / sd if sd and np.isfinite(sd) and sd > 0 else s * np.nan


def p_norm(b, s):
    if b is None or s is None or not (np.isfinite(b) and np.isfinite(s)) or s <= 0:
        return None
    return float(2 * stats.norm.sf(abs(b / s)))


def design(d, xs, covs, std_cols=()):
    parts, names = [np.ones(len(d))], ["_const"]
    for c in list(xs) + list(covs):
        x = d[c].astype(float)
        if c in std_cols:
            x = zs(x)
        parts.append(x.values)
        names.append(c)
    return np.column_stack(parts), names


def fit(d, y, xs, covs, std=True, min_n=100):
    cols = list(dict.fromkeys([y] + list(xs) + list(covs)))
    dd = d[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(dd) < min_n:
        return None
    yv = zs(dd[y]) if std else dd[y].astype(float)
    X, names = design(dd, xs, covs, set(xs) if std else set())
    ok = np.isfinite(X).all(axis=1) & np.isfinite(yv.values)
    X, yy = X[ok], yv.values[ok]
    if len(yy) < min_n:
        return None
    keep = np.r_[True, X[:, 1:].std(axis=0) > 0]
    X = X[:, keep]
    names = [nm for nm, k in zip(names, keep) if k]
    b, se, r2 = ols(yy, X)
    out = {"n": rc(len(yy)), "r2": r2, "coef": {}}
    for nm, bi, si in zip(names, b, se):
        if nm in xs:
            out["coef"][nm] = {"b": bi, "se": si, "lo": bi - 1.96 * si, "hi": bi + 1.96 * si, "p": p_norm(bi, si)}
    return out


def coef(f, x):
    return None if not f else f["coef"].get(x)


def logit(y, X, iters=60):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X
        g = X.T @ (y - p)
        try:
            step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), g)
        except np.linalg.LinAlgError:
            return None, None, None
        b = b + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = np.clip(X @ b, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    H = (X * (p * (1 - p))[:, None]).T @ X
    V = np.linalg.pinv(H)
    return b, np.sqrt(np.clip(np.diag(V), 0, None)), p


def auc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    if n1 < 5 or n0 < 5:
        return np.nan
    r = stats.rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def abi_clin(d):
    """ABI for the clinical thresholds (< 0.9, > 1.4): the unadjusted value when calendar adjustment is on."""
    return d["abi_min_raw"] if "abi_min_raw" in d.columns else d["abi_min"]


def one_row_per_participant(fr, need):
    d = fr.replace([np.inf, -np.inf], np.nan).dropna(subset=need).copy()
    d["_ord"] = d["research_stage"].map({MAIN_STAGES[0]: 0, MAIN_STAGES[1]: 1})
    return d.sort_values("_ord").drop_duplicates("participant_id", keep="first")


def icc1(a, b):
    """One-way random ICC(1) for two measurements per subject."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 20:
        return np.nan
    m = (a + b) / 2
    gm = m.mean()
    msb = 2 * ((m - gm) ** 2).sum() / (n - 1)
    msw = (((a - m) ** 2) + ((b - m) ** 2)).sum() / n
    return (msb - msw) / (msb + msw)


def icc_c(a, b):
    """Two-way consistency ICC(3,1) for two measurements per subject (ignores a common shift between occasions)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 20:
        return np.nan
    m = (a + b) / 2
    gm = m.mean()
    msr = 2 * ((m - gm) ** 2).sum() / (n - 1)
    ea = a - m - (a.mean() - gm)
    eb = b - m - (b.mean() - gm)
    mse = ((ea ** 2).sum() + (eb ** 2).sum()) / (n - 1)
    return (msr - mse) / (msr + mse)


def wide(fr, cols):
    w = fr.pivot_table(index="participant_id", columns="research_stage", values=cols, aggfunc="first")
    return w


# ============================================================================ variable sets

ORIENT = {"pwv": 1, "imt": 1, "art_w": -1, "abi_min": -1, "vein_w": 1, "fd": -1, "art_tort": 1, "avr": -1,
          "vdens": -1, "sokolow": 1, "cornell": 1, "cornell_prod": 1, "cavi0": 1}
BEDS = [("pwv", "faPWV"), ("imt", "cIMT"), ("art_w", "retina_artery_width"), ("abi_min", "ABI_min"),
        ("vein_w", "retina_vein_width"), ("avr", "retina_AV_width_ratio"), ("fd", "retina_fractal_dim"),
        ("art_tort", "retina_artery_tortuosity"), ("vdens", "retina_vessel_density"),
        ("sokolow", "ECG_Sokolow_Lyon"), ("cornell", "ECG_Cornell"), ("cavi0", "faPWV_CAVI0")]
CORE = ["pwv", "imt", "art_w", "abi_min"]
EXPOSURES = ["age", "bmi", "waist", "whr", "log_vat", "log_sat", "leg_fat", "leg_lean", "fat_total", "att",
             "log1p_ahi", "log1p_odi", "log1p_t90", "log1p_hb", "log_hload90", "spo2_mean", "log_tg", "ldl", "hdl",
             "hba1c", "glucose", "cgm_mean", "cgm_cv", "sbp_sit", "dbp_sit", "pp_sit", "hr_sit", "urate", "egfr",
             "log_acr", "smk_current", "height"]
COVSETS = {"M1_age_sex": ["age", "sex_male"],
           "M2_plus_bmi": ["age", "sex_male", "bmi"],
           "M3_plus_bp_meds": ["age", "sex_male", "bmi", "sbp_sit", "antihtn"],
           "M4_plus_smoking_lipid_glyc_meds": ["age", "sex_male", "bmi", "sbp_sit", "antihtn", "smk_current",
                                               "statin", "diabetes_any"]}
SHARE_EXPOSURES = ["waist", "whr", "bmi", "log_vat", "leg_fat", "att", "log1p_ahi", "log1p_odi", "log1p_t90",
                   "log1p_hb", "log_tg", "ldl", "hba1c", "cgm_mean", "egfr", "smk_current", "age"]


def avail(fr, cols):
    return [c for c in cols if c in fr.columns and fr[c].notna().sum() >= 100]


# ============================================================================ S0 facts and measurement validity

def s0_facts(fr):
    out = {}
    mods = {"pwv": "pwv", "abi": "abi_min", "imt": "imt", "retina": "art_w", "liver_att": "att", "sleep": "ahi",
            "hb": "hb", "dxa": "vat", "bp": "sbp_sit", "cgm": "cgm_mean", "ecg": "sokolow", "acr": "acr",
            "tg_lab": "tg", "device_bsbp": "dev_bsbp"}
    mods = {k: c for k, c in mods.items() if c in fr.columns}
    out["n_by_stage"] = {k: {st: rc(fr.loc[fr.research_stage == st, c].notna().sum()) for st in MAIN_STAGES}
                         for k, c in mods.items()}
    have = {k: fr[c].notna() for k, c in mods.items()}
    four = have["pwv"] & have["imt"] & have["retina"] & have["abi"]
    ov = {"four_beds": {st: rc((four & (fr.research_stage == st)).sum()) for st in MAIN_STAGES},
          "four_beds_any_visit_participants": rc(fr.loc[four, "participant_id"].nunique())}
    for extra in ("dxa", "sleep", "hb", "cgm", "ecg", "liver_att"):
        if extra in have:
            ov[f"four_beds_plus_{extra}"] = rc(fr.loc[four & have[extra], "participant_id"].nunique())
    pairs = {}
    for i, a in enumerate(["pwv", "imt", "retina", "abi"]):
        for b in ["pwv", "imt", "retina", "abi"][i + 1:]:
            pairs[f"{a}_and_{b}"] = rc(fr.loc[have[a] & have[b], "participant_id"].nunique())
    ov["pairs_participants"] = pairs
    out["overlap"] = ov
    if "vas_ts" in fr.columns and "bp_ts" in fr.columns:
        g = (pd.to_datetime(fr["vas_ts"], utc=True, errors="coerce") - pd.to_datetime(fr["bp_ts"], utc=True, errors="coerce"))
        gm = (g.dt.total_seconds() / 60.0).dropna()
        out["pwv_minus_bp_minutes"] = qtiles(gm)
        if len(gm):
            out["pwv_bp_gap_fraction"] = {"within_60min": float((gm.abs() <= 60).mean()),
                                          "within_6h": float((gm.abs() <= 360).mean()),
                                          "same_day": float((gm.abs() <= 1440).mean())}
    if "acq_hour" in fr.columns:
        out["acquisition_hour"] = qtiles(fr.loc[fr["pwv"].notna(), "acq_hour"])
    w = wide(fr, ["visit_date"])
    if ("visit_date", MAIN_STAGES[1]) in w.columns:
        iv = (w[("visit_date", MAIN_STAGES[1])] - w[("visit_date", MAIN_STAGES[0])]).dt.days.dropna() / 365.25
        out["visit_interval_years"] = qtiles(iv)
    out["descriptives"] = {c: qtiles(fr[c]) for c in
                           ["age", "pwv", "pwv_r", "pwv_l", "imt", "art_w", "vein_w", "avr", "fd", "art_tort",
                            "abi_min", "abi_r", "abi_l", "dev_bsbp", "dev_asbp", "sbp_sit", "sbp_sit2", "dbp_sit",
                            "sbp_lie", "sbp_st1", "sbp_st3", "hr_sit", "bmi", "waist", "height", "vat", "leg_fat",
                            "leg_lean", "att", "ahi", "odi", "t90", "hb", "hload90", "spo2_mean", "spo2_mean_ts",
                            "t90_pct_ts", "tg", "ldl", "hdl", "hba1c", "glucose", "cgm_mean", "cgm_cv", "egfr", "acr",
                            "sokolow", "cornell", "cornell_prod", "qrs", "cavi0", "lab_gap_days"]
                           if c in fr.columns}
    out["descriptives_by_sex"] = {
        str(int(sx)): {c: qtiles(fr.loc[fr["sex_male"] == sx, c]) for c in CORE + ["sokolow", "sbp_sit", "age"]
                       if c in fr.columns} for sx in (0, 1)}
    out["fractions"] = {k: float(fr[k].mean()) for k in ["antihtn", "statin", "diabetes_any", "glp1", "smk_current",
                                                          "htn_curated"] if k in fr.columns and fr[k].notna().any()}
    out["abi_categories"] = {"lt_0.9": rc((abi_clin(fr) < 0.9).sum()), "gt_1.4": rc((fr["abi_max"] > 1.4).sum()),
                             "measured": rc(fr["abi_min"].notna().sum())}
    # measurement validity
    val = {}
    for a_, b_, name in (("pwv_l", "pwv_r", "pwv_left_right"), ("abi_l", "abi_r", "abi_left_right"),
                         ("imt_l", "imt_r", "imt_left_right"), ("art_w_l", "art_w_r", "retina_art_w_left_right"),
                         ("dev_bsbp_l", "dev_bsbp_r", "device_brachial_sbp_left_right"),
                         ("sbp_sit", "sbp_sit2", "sitting_sbp_first_second"), ("dev_bsbp", "sbp_sit", "device_vs_sitting_sbp"),
                         ("dev_bsbp", "sbp_lie", "device_vs_lying_sbp")):
        if a_ in fr.columns and b_ in fr.columns:
            d = fr[[a_, b_]].dropna()
            if len(d) >= MINCELL:
                diff = d[a_] - d[b_]
                val[name] = {"n": rc(len(d)), "pearson": float(d.corr().iloc[0, 1]), "icc1": icc1(d[a_], d[b_]),
                             "mean_diff": float(diff.mean()), "sd_diff": float(diff.std())}
    out["agreement_within_visit"] = val
    trt = {}
    bedcols = [c for c in ["pwv", "imt", "art_w", "abi_min", "vein_w", "fd", "sokolow", "cornell", "sbp_sit", "dev_bsbp",
                           "hb", "log1p_ahi", "log1p_odi", "bmi", "waist", "log_vat", "cavi0"] if c in fr.columns]
    w = wide(fr, bedcols + ["antihtn"])
    for c in bedcols:
        if (c, MAIN_STAGES[0]) in w.columns and (c, MAIN_STAGES[1]) in w.columns:
            p = w[[(c, MAIN_STAGES[0]), (c, MAIN_STAGES[1])]].dropna()
            if len(p) >= MINCELL:
                trt[c] = {"n": rc(len(p)), "icc1": icc1(p.iloc[:, 0], p.iloc[:, 1]),
                          "pearson": float(p.corr().iloc[0, 1]), "mean_change": float((p.iloc[:, 1] - p.iloc[:, 0]).mean())}
    out["between_visit_reliability"] = trt
    return out


# ============================================================================ S0b calendar drift (v1.1)

DRIFT_COLS = ["pwv", "pwv_r", "pwv_l", "dev_bsbp", "dev_asbp", "abi_min", "imt", "art_w", "cavi0", "hr_sit", "sbp_sit",
              "height", "bmi", "age"]


def s0b_drift(fr):
    """Between-visit shift diagnosis: per-stage and per-calendar-quarter distributions, adjusted calendar trend within
    each stage, stage effect at the same calendar quarter, and retest ICC after removing calendar-quarter means."""
    out = {"by_stage": {}, "by_quarter": {}, "calendar_trend_per_year": {}, "stage2_vs_stage0_same_quarter": {},
           "retest_alternatives": {}, "acquisition": DIAG}
    vd = pd.to_datetime(fr["visit_date"], errors="coerce")
    t = vd
    if "vas_ts" in fr.columns:
        tv = pd.to_datetime(fr["vas_ts"], errors="coerce", utc=True).dt.tz_localize(None)
        t = tv.fillna(vd)
    d = fr.assign(_t=t)
    d = d[d["_t"].notna()].copy()
    for c in CAL_BEDS + ["cavi0"]:  # v1.2: diagnose drift on raw values; keep the adjusted value as <bed>_adj
        if c + "_raw" in d.columns:
            d[c + "_adj"] = d[c]
            d[c] = d[c + "_raw"]
    d["_q"] = d["_t"].dt.year.astype(str) + "Q" + d["_t"].dt.quarter.astype(str)
    d["_yr"] = (d["_t"] - pd.Timestamp("2018-01-01")).dt.days / 365.25
    cols = [c for c in DRIFT_COLS if c in d.columns]
    for c in cols:
        out["by_stage"][c] = {st: qtiles(d.loc[d["research_stage"] == st, c]) for st in MAIN_STAGES}
        bq = {}
        for (q, st), s in d.groupby(["_q", "research_stage"])[c]:
            s = s.dropna()
            if len(s) >= MINCELL:
                bq[f"{q}|{st[:2]}"] = {"n": rc(len(s)), "median": float(s.median())}
        out["by_quarter"][c] = bq
    covs = [c for c in ["age", "sex_male", "sbp_sit", "hr_sit", "height"] if c in d.columns]
    for c in [x for x in ["pwv", "dev_bsbp", "abi_min", "imt", "art_w"] if x in d.columns]:
        cv = [x for x in covs if x != c]
        r = {}
        for st in MAIN_STAGES:
            f = fit(d[d["research_stage"] == st], c, ["_yr"], cv, std=False)
            r[st] = (coef(f, "_yr") | {"n": f["n"]}) if f else None
        out["calendar_trend_per_year"][c] = r
        e = d.dropna(subset=[c] + cv).copy()
        st_per_q = e.groupby("_q")["research_stage"].nunique()
        n_per = e.groupby(["_q", "research_stage"]).size().unstack(fill_value=0)
        both = [q for q in st_per_q.index if st_per_q[q] == 2 and (n_per.loc[q] >= MINCELL).all()]
        e = e[e["_q"].isin(both)]
        if len(both) >= 1 and len(e) >= 200:
            e["stage2"] = (e["research_stage"] == MAIN_STAGES[1]).astype(float)
            parts = [np.ones(len(e)), e["stage2"].values] + [e[x].astype(float).values for x in cv]
            if len(both) > 1:
                parts.append(pd.get_dummies(e["_q"], drop_first=True, dtype=float).values)
            b, se, _ = ols(e[c].astype(float).values, np.column_stack(parts))
            out["stage2_vs_stage0_same_quarter"][c] = {"n": rc(len(e)), "n_quarters": len(both), "b": b[1], "se": se[1],
                                                        "lo": b[1] - 1.96 * se[1], "hi": b[1] + 1.96 * se[1],
                                                        "p": p_norm(b[1], se[1])}
    for c in [x for x in ["pwv", "abi_min", "imt", "art_w", "cavi0", "dev_bsbp", "sbp_sit"] if x in d.columns]:
        qm = d.groupby("_q")[c].transform(lambda s: s.mean() if s.notna().sum() >= MINCELL else np.nan)
        d[c + "_dq"] = d[c] - qm + d[c].mean()
        w = wide(d, [c, c + "_dq"])
        if (c, MAIN_STAGES[0]) not in w.columns or (c, MAIN_STAGES[1]) not in w.columns:
            continue
        p = w[[(c, MAIN_STAGES[0]), (c, MAIN_STAGES[1])]].dropna()
        p2 = w[[(c + "_dq", MAIN_STAGES[0]), (c + "_dq", MAIN_STAGES[1])]].dropna()
        if len(p) < MINCELL:
            continue
        r = {"n": rc(len(p)), "icc1": icc1(p.iloc[:, 0], p.iloc[:, 1]), "icc_c": icc_c(p.iloc[:, 0], p.iloc[:, 1]),
             "pearson": float(p.corr().iloc[0, 1]), "mean_change": float((p.iloc[:, 1] - p.iloc[:, 0]).mean()),
             "sd_change": float((p.iloc[:, 1] - p.iloc[:, 0]).std())}
        if len(p2) >= MINCELL:
            r.update({"n_dq": rc(len(p2)), "icc1_dq": icc1(p2.iloc[:, 0], p2.iloc[:, 1]),
                      "icc_c_dq": icc_c(p2.iloc[:, 0], p2.iloc[:, 1]), "pearson_dq": float(p2.corr().iloc[0, 1]),
                      "mean_change_dq": float((p2.iloc[:, 1] - p2.iloc[:, 0]).mean())})
        if c + "_adj" in d.columns:  # v1.2: the calendar-adjusted values actually used downstream
            wa = wide(d, [c + "_adj"])
            if (c + "_adj", MAIN_STAGES[0]) in wa.columns and (c + "_adj", MAIN_STAGES[1]) in wa.columns:
                pa = wa[[(c + "_adj", MAIN_STAGES[0]), (c + "_adj", MAIN_STAGES[1])]].dropna()
                if len(pa) >= MINCELL:
                    r.update({"icc1_adj": icc1(pa.iloc[:, 0], pa.iloc[:, 1]), "pearson_adj": float(pa.corr().iloc[0, 1]),
                              "mean_change_adj": float((pa.iloc[:, 1] - pa.iloc[:, 0]).mean())})
        out["retest_alternatives"][c] = r
    return out


# ============================================================================ S1 atlas

def multi_fit(Y, X, j=1):
    """OLS of several outcomes on one design; returns row j of B and its robust joint covariance (outcomes x outcomes)."""
    n, k = X.shape
    G = np.linalg.pinv(X.T @ X)
    B = G @ (X.T @ Y)
    E = Y - X @ B
    a = (G @ X.T)[j]
    AE = a[:, None] * E
    V = AE.T @ AE * (n / max(n - k, 1))
    return B[j], V


def contrasts(bvec, V, beds, pairs):
    out = {}
    ix = {b: i for i, b in enumerate(beds)}
    for a_, c_ in pairs:
        cvec = np.zeros(len(beds))
        cvec[ix[a_]], cvec[ix[c_]] = 1, -1
        d = float(cvec @ bvec)
        s = float(np.sqrt(max(cvec @ V @ cvec, 0)))
        out[f"{a_}_minus_{c_}"] = {"d": d, "se": s, "lo": d - 1.96 * s, "hi": d + 1.96 * s, "p": p_norm(d, s),
                                   "tost_equivalent": bool(abs(d) + 1.645 * s < TOST_DELTA)}
    C = np.zeros((len(beds) - 1, len(beds)))
    for i in range(1, len(beds)):
        C[i - 1, 0], C[i - 1, i] = 1, -1
    cb = C @ bvec
    try:
        W = float(cb @ np.linalg.solve(C @ V @ C.T, cb))
        p = float(stats.chi2.sf(W, len(beds) - 1))
    except np.linalg.LinAlgError:
        W, p = None, None
    return out, {"wald": W, "df": len(beds) - 1, "p": p}


def bh(pvals):
    p = np.array([np.nan if v is None else v for v in pvals], dtype=float)
    ok = np.isfinite(p)
    q = np.full(len(p), np.nan)
    if ok.sum():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        qq = np.empty_like(r)
        qq[o] = np.clip(r, 0, 1)
        q[ok] = qq
    return q


def s1_atlas(fr, boot, rng):
    out = {"per_bed": {}, "common": {}, "joint": {}}
    exps = avail(fr, EXPOSURES)
    for bed, label in BEDS:
        if bed not in fr.columns or fr[bed].notna().sum() < 100:
            continue
        d = one_row_per_participant(fr, [bed, "age", "sex_male"])
        res = {"n_participants": rc(len(d)), "orientation": ORIENT.get(bed, 1)}
        for ex in exps:
            if ex == bed:
                continue
            r = {}
            for mname, covs in COVSETS.items():
                covs = [c for c in covs if c != ex and c in d.columns]
                f = fit(d, bed, [ex], covs)
                if f:
                    r[mname] = {"n": f["n"], **f["coef"].get(ex, {})}
            res[ex] = r
        out["per_bed"][label] = res

    # reliability for disattenuation (between-visit ICC per core bed; bootstrap over pairs)
    w = wide(fr, CORE)
    rel_pairs = {}
    for c in CORE:
        if (c, MAIN_STAGES[0]) in w.columns and (c, MAIN_STAGES[1]) in w.columns:
            p = w[[(c, MAIN_STAGES[0]), (c, MAIN_STAGES[1])]].dropna().to_numpy()
            if len(p) >= 50:
                rel_pairs[c] = p
    rel = {c: icc1(p[:, 0], p[:, 1]) for c, p in rel_pairs.items()}
    rel_boot = {c: np.array([icc1(*p[rng.integers(0, len(p), len(p))].T) for _ in range(boot)])
                for c, p in rel_pairs.items()}
    out["common"]["reliability_icc_between_visits"] = {c: {"icc": rel[c], "ci": pct(rel_boot[c]),
                                                            "n_pairs": rc(len(rel_pairs[c]))} for c in rel}

    need = CORE + ["age", "sex_male"]
    d = one_row_per_participant(fr, need)
    out["common"]["n_participants"] = rc(len(d))
    if len(d) < 200:
        return out
    oc = ["o_" + b for b in CORE]
    R, resid = {}, {}
    X = np.column_stack([np.ones(len(d)), d["age"].astype(float), d["sex_male"].astype(float)])
    for b_ in oc:
        yb = zs(d[b_]).values
        resid[b_] = yb - X @ beta_only(yb, X)
    for i in oc:
        R[i] = {j: float(np.corrcoef(resid[i], resid[j])[0, 1]) for j in oc}
    out["common"]["partial_corr_age_sex_oriented"] = R
    pairs = [("pwv", "imt"), ("pwv", "art_w"), ("pwv", "abi_min"), ("imt", "art_w"), ("imt", "abi_min"),
             ("art_w", "abi_min")]
    for mname, covs0 in (("M1_age_sex", COVSETS["M1_age_sex"]), ("M3_plus_bp_meds", COVSETS["M3_plus_bp_meds"])):
        mres = {}
        for ex in exps:
            covs = [c for c in covs0 if c != ex and c in d.columns]
            dd = d[CORE + oc + [ex] + covs].replace([np.inf, -np.inf], np.nan).dropna()
            dd = dd.loc[:, ~dd.columns.duplicated()]
            if len(dd) < 200 or dd[ex].std() == 0:
                continue
            Xd, _ = design(dd, [ex], covs, std_cols={ex})
            Y = np.column_stack([zs(dd[b_]).values for b_ in oc])
            bvec, V = multi_fit(Y, Xd)
            con, het = contrasts(bvec, V, CORE, pairs)
            Yraw = np.column_stack([zs(dd[b_]).values for b_ in CORE])
            braw, Vraw = multi_fit(Yraw, Xd)
            con_raw, _ = contrasts(braw, Vraw, CORE, [("pwv", "imt"), ("pwv", "art_w")])
            n = len(dd)
            bs = np.empty((boot, len(CORE)))
            for k in range(boot):
                idx = rng.integers(0, n, n)
                bs[k] = np.linalg.lstsq(Xd[idx], Y[idx], rcond=None)[0][1]
            dis = {}
            for a_, c_ in pairs:
                if a_ not in rel_boot or c_ not in rel_boot or rel[a_] <= 0.05 or rel[c_] <= 0.05:
                    continue
                i, j = CORE.index(a_), CORE.index(c_)
                ra = np.where(rel_boot[a_] > 0.05, rel_boot[a_], np.nan)
                rc_ = np.where(rel_boot[c_] > 0.05, rel_boot[c_], np.nan)
                dis[f"{a_}_minus_{c_}"] = {"d": float(bvec[i] / np.sqrt(rel[a_]) - bvec[j] / np.sqrt(rel[c_])),
                                           "ci": pct(bs[:, i] / np.sqrt(ra) - bs[:, j] / np.sqrt(rc_))}
            mres[ex] = {"n": rc(n),
                        "beta_oriented": {b_: {"b": bvec[i], "se": float(np.sqrt(V[i, i])), "boot_ci": pct(bs[:, i])}
                                          for i, b_ in enumerate(CORE)},
                        "diff_oriented": {k: {**v, "boot_ci": pct(bs[:, CORE.index(k.split('_minus_')[0])] -
                                                                  bs[:, CORE.index(k.split('_minus_')[1])])}
                                          for k, v in con.items()},
                        "heterogeneity": het, "diff_raw_sign": con_raw, "diff_disattenuated": dis}
        qs = bh([v["heterogeneity"]["p"] for v in mres.values()])
        for (k, v), q in zip(mres.items(), qs):
            v["heterogeneity"]["q_bh"] = q
        out["common"][mname] = mres
    # decision-rule tally (pre-specified: >= 2 nodes with PWV-IMT or PWV-retina CI excluding 0)
    tally = {}
    for mname in ("M1_age_sex", "M3_plus_bp_meds"):
        m = out["common"].get(mname, {})
        hits_o = [ex for ex, v in m.items() if ex != "age" and any(
            (v["diff_oriented"][k]["lo"] > 0 or v["diff_oriented"][k]["hi"] < 0)
            for k in ("pwv_minus_imt", "pwv_minus_art_w"))]
        hits_r = [ex for ex, v in m.items() if ex != "age" and any(
            (v["diff_raw_sign"][k]["lo"] > 0 or v["diff_raw_sign"][k]["hi"] < 0)
            for k in ("pwv_minus_imt", "pwv_minus_art_w"))]
        tally[mname] = {"n_nodes_oriented": len(hits_o), "nodes_oriented": hits_o,
                        "n_nodes_raw_sign": len(hits_r), "nodes_raw_sign": hits_r}
    out["decision_rule_tally"] = tally
    joint_x = ["bmi", "log_vat", "leg_fat", "leg_lean", "att", "log1p_t90", "log1p_hb", "log_tg", "ldl", "hdl", "hba1c",
               "sbp_sit", "hr_sit", "smk_current"]
    for bed, label in BEDS:
        if bed not in fr.columns or fr[bed].notna().sum() < 100:
            continue
        d1 = one_row_per_participant(fr, [bed, "age", "sex_male"])
        xs = [x for x in avail(d1, joint_x)]
        jc = [c for c in ["age", "sex_male", "antihtn", "statin", "diabetes_any"] if c in d1.columns]
        f = fit(d1, bed, xs, jc)
        if f:
            out["joint"][label] = f
        xs2 = [x for x in xs if x not in ("log1p_hb",)]
        f2 = fit(d1, bed, xs2, jc)
        if f2:
            out["joint"][label + "_without_hb"] = f2
    return out


# ============================================================================ S2 pressure decomposition

def acute_deviation(fr):
    d = fr.copy()
    base_cols = ["sbp_sit", "dbp_sit"]
    if d["sbp_lie"].notna().mean() > 0.5:
        base_cols.append("sbp_lie")
    m = d[["dev_bsbp", "age", "sex_male"] + base_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    X = np.column_stack([np.ones(m.sum())] + [d.loc[m, c].astype(float).values for c in base_cols + ["age", "sex_male"]])
    b, _, r2 = ols(d.loc[m, "dev_bsbp"].astype(float).values, X)
    d["acute_dev"] = np.nan
    d.loc[m, "acute_dev"] = d.loc[m, "dev_bsbp"].astype(float).values - X @ b
    info = {"r2_device_on_visit_bp": r2, "predictors": base_cols + ["age", "sex_male"],
            "acute_dev_sd": float(d["acute_dev"].std())}
    return d, info


def calibration_data(fr):
    """Visit-1 rows of participants untreated at both visits with sitting BP at both (replicate = visit 2)."""
    w = wide(fr, ["sbp_sit", "dbp_sit", "antihtn"])
    s0, s1 = MAIN_STAGES
    need = [("sbp_sit", s0), ("sbp_sit", s1), ("dbp_sit", s0), ("dbp_sit", s1)]
    if not all(k in w.columns for k in need):
        return None
    keep = (w[("antihtn", s0)] == 0) & (w[("antihtn", s1)] == 0)
    rep = pd.DataFrame({"participant_id": w.index, "sbp_rep": w[("sbp_sit", s1)].values,
                        "dbp_rep": w[("dbp_sit", s1)].values})[keep.values].dropna()
    v1 = fr[fr.research_stage == s0]
    return v1.merge(rep, on="participant_id", how="inner")


def rc_predict(cal, target, preds, apply_to):
    """Regression calibration: E[usual BP | observed BP measures, covariates] fitted where a replicate exists.

    `target` is the other-visit sitting BP of untreated participants (independent measurement error); every
    covariate of the outcome model must be among `preds`.
    """
    c = cal[[target] + preds].replace([np.inf, -np.inf], np.nan).dropna()
    if len(c) < 100:
        return None, None
    X = np.column_stack([np.ones(len(c))] + [c[p].astype(float).values for p in preds])
    b = beta_only(c[target].astype(float).values, X)
    Xa = np.column_stack([np.ones(len(apply_to))] + [apply_to[p].astype(float).values for p in preds])
    return Xa @ b, {"n": rc(len(c)), "r2": float(1 - np.var(c[target].values - X @ b) / np.var(c[target].values)),
                    "coef_observed_sbp": float(b[1 + preds.index("sbp_sit")]) if "sbp_sit" in preds else None,
                    "coef_device_sbp": float(b[1 + preds.index("dev_bsbp")]) if "dev_bsbp" in preds else None}


def three_way(y, X0, Xc, Xa):
    return beta_only(y, X0)[1], beta_only(y, Xc)[1], beta_only(y, Xa)[1]


def rc_acute(d, cal, bp_obs, boot, rng):
    """Acquisition coupling with regression-calibrated chronic BP: PWV ~ E[usual SBP, DBP | all BP measures]
    + device brachial SBP + age + sex + BMI.  The device coefficient is the pressure-at-acquisition slope holding
    usual BP fixed (an observed-BP adjustment leaves usual-BP error in the 'acute deviation')."""
    if cal is None:
        return None
    covs = ["age", "sex_male", "bmi"]
    preds = list(dict.fromkeys(bp_obs + ["dev_bsbp"] + covs))
    e = d.dropna(subset=preds + ["pwv"]).copy()
    if len(e) < 300:
        return None

    def est(cal_, e_):
        p1, info = rc_predict(cal_, "sbp_rep", preds, e_)
        p2, _ = rc_predict(cal_, "dbp_rep", preds, e_)
        if p1 is None or p2 is None:
            return None, None
        X = np.column_stack([np.ones(len(e_)), p1 / 10, p2 / 10, e_["dev_bsbp"].values / 10] +
                            [e_[c].astype(float).values for c in covs])
        return beta_only(e_["pwv"].astype(float).values, X), info

    b, info = est(cal, e)
    if b is None:
        return None
    cal_e = cal.dropna(subset=["sbp_rep", "dbp_rep"] + preds)
    bs = []
    for _ in range(boot):
        bb, _ = est(cal_e.iloc[rng.integers(0, len(cal_e), len(cal_e))], e.iloc[rng.integers(0, len(e), len(e))])
        if bb is not None:
            bs.append(bb[1:4])
    bs = np.array(bs)
    return {"n": rc(len(e)), "calibration": info,
            "per_10mmHg": {"usual_sbp_rc": {"b": b[1], "ci": pct(bs[:, 0]) if len(bs) else None},
                           "usual_dbp_rc": {"b": b[2], "ci": pct(bs[:, 1]) if len(bs) else None},
                           "device_sbp_given_usual": {"b": b[3], "ci": pct(bs[:, 2]) if len(bs) else None}}}


def s2_pressure(fr, boot, rng):
    out = {}
    fr2, info = acute_deviation(fr)
    out["acute_dev_model"] = info
    need = ["pwv", "dev_bsbp", "sbp_sit", "dbp_sit", "acute_dev", "age", "sex_male"]
    d = one_row_per_participant(fr2, need)
    out["n_participants"] = rc(len(d))
    if len(d) < 200:
        return out
    out["corr"] = {"dev_bsbp_vs_sbp_sit": float(d[["dev_bsbp", "sbp_sit"]].corr().iloc[0, 1]),
                   "dev_asbp_vs_dev_bsbp": float(d[["dev_asbp", "dev_bsbp"]].dropna().corr().iloc[0, 1])
                   if d["dev_asbp"].notna().sum() > 100 else None}
    out["device_minus_sitting_sbp"] = qtiles(d["dev_bsbp"] - d["sbp_sit"])
    for c in ("sbp_sit", "dev_bsbp", "acute_dev", "dev_asbp", "map_sit"):
        d[c + "10"] = d[c] / 10
    C2 = ["age", "sex_male", "bmi"]
    cs = {"a_sitting_only": fit(d, "pwv", ["sbp_sit10"], C2, std=False),
          "b_device_only": fit(d, "pwv", ["dev_bsbp10"], C2, std=False),
          "c_sitting_plus_acute": fit(d, "pwv", ["sbp_sit10", "acute_dev10"], C2, std=False),
          "d_sitting_plus_ankle_device": fit(d, "pwv", ["sbp_sit10", "dev_asbp10"], C2, std=False),
          "e_map_plus_acute": fit(d, "pwv", ["map_sit10", "acute_dev10"], C2, std=False),
          "f_sitting_plus_acute_plus_hr": fit(d, "pwv", ["sbp_sit10", "acute_dev10"], C2 + ["hr_sit"], std=False)}
    for bed in ("imt", "art_w", "abi_min", "sokolow", "cavi0"):
        if bed in d.columns:
            cs[f"{bed}_sitting_plus_acute"] = fit(d, bed, ["sbp_sit10", "acute_dev10"], C2, std=False)
    out["cross_sectional_per_10mmHg"] = cs
    # by sex and age group
    strat = {}
    for sx in (0, 1):
        strat[f"sex_{sx}"] = fit(d[d.sex_male == sx], "pwv", ["sbp_sit10", "acute_dev10"], ["age", "bmi"], std=False)
    for lo, hi in ((0, 50), (50, 60), (60, 200)):
        strat[f"age_{lo}_{hi}"] = fit(d[(d.age >= lo) & (d.age < hi)], "pwv", ["sbp_sit10", "acute_dev10"], C2, std=False)
    out["acute_by_stratum"] = strat
    # within-person change
    w = wide(fr2.dropna(subset=["pwv", "dev_bsbp", "sbp_sit"]),
             ["pwv", "dev_bsbp", "sbp_sit", "dbp_sit", "map_sit", "bmi", "age", "visit_date", "antihtn", "hr_sit",
              "acute_dev"])
    s0, s1 = MAIN_STAGES
    if all((v, st) in w.columns for v in ("pwv", "dev_bsbp", "sbp_sit") for st in MAIN_STAGES):
        ch = pd.DataFrame({
            "d_pwv": w[("pwv", s1)] - w[("pwv", s0)], "pwv0": w[("pwv", s0)],
            "d_dev10": (w[("dev_bsbp", s1)] - w[("dev_bsbp", s0)]) / 10,
            "d_sit10": (w[("sbp_sit", s1)] - w[("sbp_sit", s0)]) / 10,
            "d_map10": (w[("map_sit", s1)] - w[("map_sit", s0)]) / 10,
            "d_acute10": (w[("acute_dev", s1)] - w[("acute_dev", s0)]) / 10,
            "d_hr": w[("hr_sit", s1)] - w[("hr_sit", s0)],
            "d_bmi": w[("bmi", s1)] - w[("bmi", s0)], "age0": w[("age", s0)],
            "interval": (pd.to_datetime(w[("visit_date", s1)]) - pd.to_datetime(w[("visit_date", s0)])).dt.days / 365.25,
        }).reset_index()
        ch = ch.merge(fr[["participant_id", "sex_male"]].drop_duplicates("participant_id"), how="left")
        cv = ["age0", "sex_male", "d_bmi", "interval"]
        out["within_person"] = {
            "n_pairs": rc(ch.dropna(subset=["d_pwv", "d_dev10", "d_sit10"]).shape[0]),
            "a_d_sitting": fit(ch, "d_pwv", ["d_sit10"], cv, std=False),
            "b_d_device": fit(ch, "d_pwv", ["d_dev10"], cv, std=False),
            "c_d_sitting_plus_d_acute": fit(ch, "d_pwv", ["d_sit10", "d_acute10"], cv, std=False),
            "d_d_map": fit(ch, "d_pwv", ["d_map10"], cv, std=False),
            "e_d_sitting_plus_d_acute_plus_d_hr": fit(ch, "d_pwv", ["d_sit10", "d_acute10", "d_hr"], cv, std=False),
            "d_pwv": qtiles(ch["d_pwv"]),
            "benchmark_note": "McNally 2024 meta-analysis: -0.65 m/s per -10 mmHg MAP (central/cf PWV)"}
        lag = pd.DataFrame({"pwv1": w[("pwv", s1)], "sit0_10": w[("sbp_sit", s0)] / 10,
                            "sit1_10": w[("sbp_sit", s1)] / 10, "age1": w[("age", s1)],
                            "bmi1": w[("bmi", s1)]}).reset_index().merge(
            fr[["participant_id", "sex_male"]].drop_duplicates("participant_id"), how="left").dropna()
        if len(lag) >= 200:
            Xs = np.column_stack([np.ones(len(lag)), lag["age1"], lag["sex_male"].astype(float), lag["bmi1"]])
            X0, X1 = np.column_stack([Xs, lag["sit0_10"]]), np.column_stack([Xs, lag["sit1_10"]])
            y = lag["pwv1"].values.astype(float)
            b0, b1 = beta_only(y, X0)[-1], beta_only(y, X1)[-1]
            rs = []
            for _ in range(boot):
                idx = rng.integers(0, len(lag), len(lag))
                rs.append(beta_only(y[idx], X0[idx])[-1] / beta_only(y[idx], X1[idx])[-1])
            out["lagged"] = {"n": rc(len(lag)), "slope_prior_visit_sbp": b0, "slope_same_visit_sbp": b1,
                             "ratio": b0 / b1 if b1 else None, "ratio_ci": pct(rs)}
    # three-way shares, naive and regression-calibrated chronic BP
    cal = calibration_data(fr2)
    bp_obs = ["sbp_sit", "dbp_sit"] + (["sbp_lie"] if d["sbp_lie"].notna().mean() > 0.5 else [])
    out["rc_acute"] = rc_acute(d, cal, bp_obs, boot, rng)
    shares, rcinfo = {}, {}
    for ex in avail(d, SHARE_EXPOSURES):
        e = d.dropna(subset=[ex]).copy()
        if len(e) < 300 or e[ex].std() == 0:
            continue
        base = ["age", "sex_male"] if ex != "age" else ["sex_male"]
        X0, _ = design(e, [ex], base, std_cols={ex})
        Xc, _ = design(e, [ex], base + ["sbp_sit", "dbp_sit"], std_cols={ex})
        Xa, _ = design(e, [ex], base + ["sbp_sit", "dbp_sit", "acute_dev"], std_cols={ex})
        y = zs(e["pwv"]).values
        t, c, a = three_way(y, X0, Xc, Xa)
        rec = {"n": rc(len(e)), "beta_total": t, "beta_given_visit_bp": c, "beta_given_visit_bp_and_acute": a,
               "share_visit_bp": (t - c) / t if t else None, "share_acute": (c - a) / t if t else None}
        bt = np.empty((boot, 3))
        for k in range(boot):
            idx = rng.integers(0, len(e), len(e))
            bt[k] = three_way(y[idx], X0[idx], Xc[idx], Xa[idx])
        rec["share_visit_bp_ci"] = pct((bt[:, 0] - bt[:, 1]) / bt[:, 0])
        rec["share_acute_ci"] = pct((bt[:, 1] - bt[:, 2]) / bt[:, 0])
        rec["beta_total_ci"] = pct(bt[:, 0])
        rec["beta_resid_ci"] = pct(bt[:, 2])
        # regression calibration of chronic BP (replicate = other-visit sitting BP in the untreated); the
        # calibration uses every BP measure of the visit (incl. the device SBP), the exposure and covariates
        if cal is not None and ex in cal.columns:
            preds = list(dict.fromkeys(bp_obs + ["dev_bsbp", ex] + base))
            ps, info_s = rc_predict(cal, "sbp_rep", preds, e)
            pdb, _ = rc_predict(cal, "dbp_rep", preds, e)
            if ps is not None and pdb is not None:
                e["sbp_rc"], e["dbp_rc"] = ps, pdb
                Xrc, _ = design(e, [ex], base + ["sbp_rc", "dbp_rc"], std_cols={ex})
                Xra, _ = design(e, [ex], base + ["sbp_rc", "dbp_rc", "dev_bsbp"], std_cols={ex})
                crc, arc = beta_only(y, Xrc)[1], beta_only(y, Xra)[1]
                rec["rc"] = {"calibration": info_s, "beta_given_rc_bp": crc, "beta_given_rc_bp_and_device": arc,
                             "share_rc_bp": (t - crc) / t if t else None,
                             "share_acute_after_rc": (crc - arc) / t if t else None}
                cal_e = cal.dropna(subset=["sbp_rep", "dbp_rep"] + preds)
                sdx = float(e[ex].std())
                bb = []
                for k in range(boot):
                    ci = cal_e.iloc[rng.integers(0, len(cal_e), len(cal_e))]
                    idx = rng.integers(0, len(e), len(e))
                    eb = e.iloc[idx].copy()
                    p1, _ = rc_predict(ci, "sbp_rep", preds, eb)
                    p2, _ = rc_predict(ci, "dbp_rep", preds, eb)
                    if p1 is None or p2 is None:
                        continue
                    eb["sbp_rc"], eb["dbp_rc"] = p1, p2
                    yb = y[idx]
                    Xtb, _ = design(eb, [ex], base)
                    Xrcb, _ = design(eb, [ex], base + ["sbp_rc", "dbp_rc"])
                    Xrab, _ = design(eb, [ex], base + ["sbp_rc", "dbp_rc", "dev_bsbp"])
                    bb.append((beta_only(yb, Xtb)[1] * sdx, beta_only(yb, Xrcb)[1] * sdx, beta_only(yb, Xrab)[1] * sdx))
                bb = np.array(bb)
                if len(bb) >= 20:
                    rec["rc"]["share_rc_bp_ci"] = pct((bb[:, 0] - bb[:, 1]) / bb[:, 0])
                    rec["rc"]["share_acute_after_rc_ci"] = pct((bb[:, 1] - bb[:, 2]) / bb[:, 0])
                    rec["rc"]["beta_resid_rc_ci"] = pct(bb[:, 2])
        shares[ex] = rec
    out["exposure_shares_pwv"] = shares
    out["regression_calibration_n"] = rc(len(cal)) if cal is not None else None
    # sensitivity sets for the shares
    sens = {}
    for ex in avail(d, ["waist", "log_vat", "att", "log1p_t90", "log1p_hb", "log_tg", "hba1c"]):
        for tag, sub, extra in (("excl_abi_lt_0.9_gt_1.4", d[(abi_clin(d) >= 0.9) & (d["abi_max"] <= 1.4)], []),
                                ("adj_leg_fat_lean", d, ["leg_fat", "leg_lean"]),
                                ("adj_height_hr", d, ["height", "hr_sit"]),
                                ("adj_acq_hour_season", d, ["acq_hour", "season_sin", "season_cos"]),
                                ("untreated_only", d[d["antihtn"] == 0], [])):
            extra = [x for x in extra if x in sub.columns]
            e = sub.dropna(subset=[ex] + extra)
            if len(e) < 300:
                continue
            ft = fit(e, "pwv", [ex], ["age", "sex_male"] + extra)
            fc = fit(e, "pwv", [ex], ["age", "sex_male", "sbp_sit", "dbp_sit"] + extra)
            fa = fit(e, "pwv", [ex], ["age", "sex_male", "sbp_sit", "dbp_sit", "acute_dev"] + extra)
            if ft and fc and fa:
                t, c, a_ = ft["coef"][ex]["b"], fc["coef"][ex]["b"], fa["coef"][ex]["b"]
                sens.setdefault(ex, {})[tag] = {"n": ft["n"], "beta_total": t, "beta_given_visit_bp": c,
                                                "beta_given_visit_bp_and_acute": a_,
                                                "share_visit_bp": (t - c) / t if t else None,
                                                "share_acute": (c - a_) / t if t else None,
                                                "resid_lo": fa["coef"][ex]["lo"], "resid_hi": fa["coef"][ex]["hi"]}
    out["exposure_shares_sensitivity"] = sens
    # CAVI0-style pressure-corrected stiffness: exposure betas on CAVI0 vs PWV
    cav = {}
    if "cavi0" in d.columns:
        cav["bp_dependence"] = {"pwv": fit(d, "pwv", ["sbp_sit10", "acute_dev10"], C2, std=False),
                                "cavi0": fit(d, "cavi0", ["sbp_sit10", "acute_dev10"], C2, std=False)}
        for ex in avail(d, ["waist", "bmi", "log_vat", "att", "log1p_hb", "log_tg", "hba1c", "age"]):
            covs = ["age", "sex_male"] if ex != "age" else ["sex_male"]
            fp, fcv = fit(d, "pwv", [ex], covs), fit(d, "cavi0", [ex], covs)
            cav[ex] = {"pwv": coef(fp, ex), "cavi0": coef(fcv, ex)}
    out["cavi0_sensitivity"] = cav
    # other beds: share via visit BP
    comp = {}
    for bed in ("imt", "abi_min", "art_w", "sokolow"):
        if bed not in fr.columns:
            continue
        comp[bed] = {}
        for ex in avail(fr, ["waist", "bmi", "log_vat", "att", "log1p_ahi", "log1p_hb", "log_tg", "hba1c"]):
            e = one_row_per_participant(fr, [bed, ex, "sbp_sit", "dbp_sit", "age", "sex_male"])
            if len(e) < 300:
                continue
            ft = fit(e, bed, [ex], ["age", "sex_male"])
            fc = fit(e, bed, [ex], ["age", "sex_male", "sbp_sit", "dbp_sit"])
            if ft and fc:
                bt_, bc_ = ft["coef"][ex]["b"], fc["coef"][ex]["b"]
                comp[bed][ex] = {"n": ft["n"], "beta_total": bt_, "beta_given_visit_bp": bc_,
                                 "share_visit_bp": (bt_ - bc_) / bt_ if bt_ else None}
    out["exposure_shares_other_beds"] = comp
    rel = {}
    r2 = fr.dropna(subset=["sbp_sit", "sbp_sit2"])
    if len(r2) >= 50:
        rel["sit_vs_sit2"] = {"n": rc(len(r2)), "corr": float(r2[["sbp_sit", "sbp_sit2"]].corr().iloc[0, 1]),
                              "sd_diff": float((r2["sbp_sit"] - r2["sbp_sit2"]).std())}
    if cal is not None and len(cal) >= 50:
        rel["between_visits_untreated"] = {"n": rc(len(cal)), "corr": float(cal[["sbp_sit", "sbp_rep"]].corr().iloc[0, 1])}
    out["bp_reliability"] = rel
    return out


# ============================================================================ S3 hypoxia

def s3_hypoxia(fr, hb_meta):
    out = {"computation": hb_meta}
    if "hb" not in fr.columns or fr["hb"].notna().sum() < 100:
        out["note"] = "hypoxic burden not available"
        return out
    d = one_row_per_participant(fr, ["hb", "age", "sex_male"])
    cols = [c for c in ["log1p_hb", "log_hload90", "log1p_ahi", "log1p_odi", "log1p_t90", "spo2_mean", "spo2_mean_ts",
                        "t90_pct_ts", "desat_nadir", "bmi", "log_vat"] if c in d.columns]
    out["n_participants"] = rc(len(d))
    out["spearman"] = clean(d[cols].corr(method="spearman").round(4).to_dict())
    out["hb_by_ahi_category"] = {}
    if "ahi" in d.columns:
        for lo, hi in ((0, 5), (5, 15), (15, 30), (30, 1e9)):
            sub = d[(d["ahi"] >= lo) & (d["ahi"] < hi)]
            out["hb_by_ahi_category"][f"{lo}_{hi if hi < 1e9 else 'plus'}"] = qtiles(sub["hb"])
    beds = {}
    for bed in ["pwv", "imt", "art_w", "abi_min", "sokolow", "sbp_sit"]:
        if bed not in fr.columns:
            continue
        e = one_row_per_participant(fr, [bed, "age", "sex_male", "bmi"])
        r = {}
        for ex in avail(e, ["log1p_hb", "log_hload90", "log1p_ahi", "log1p_odi", "log1p_t90"]):
            r[ex] = {"M2": coef(fit(e, bed, [ex], ["age", "sex_male", "bmi"]), ex),
                     "M3_vat_bp": coef(fit(e, bed, [ex], avail(e, ["age", "sex_male", "bmi", "log_vat", "sbp_sit", "antihtn"])
                                           if bed != "sbp_sit" else avail(e, ["age", "sex_male", "bmi", "log_vat"])), ex)}
        # SpO2-based vs PAT-based: mutual adjustment
        if all(c in e.columns for c in ("log1p_hb", "log1p_ahi")):
            f = fit(e, bed, ["log1p_hb", "log1p_ahi"], ["age", "sex_male", "bmi"])
            r["hb_and_ahi_mutual"] = f
        beds[bed] = r
    out["beds"] = beds
    return out


# ============================================================================ S4 ECG

def s4_ecg(fr):
    out = {}
    if "sokolow" not in fr.columns:
        return {"note": "no ECG table"}
    d = one_row_per_participant(fr, ["sokolow", "age", "sex_male"])
    out["n_participants"] = rc(len(d))
    out["lvh_prevalence"] = {
        "sokolow_ge_3.5mV": float((d["sokolow"] >= 3.5).mean()),
        "cornell_sex_specific": float(((d["cornell"] > np.where(d["sex_male"] == 1, 2.8, 2.0))).mean())
        if "cornell" in d.columns else None,
        "cornell_product_ge_244": float((d["cornell_prod"] >= 244).mean()) if "cornell_prod" in d.columns else None}
    r = {}
    for y in ("sokolow", "cornell", "cornell_prod"):
        if y not in d.columns:
            continue
        r[y] = {}
        for ex in avail(d, ["sbp_sit", "dev_bsbp", "pwv", "imt", "art_w", "abi_min", "bmi", "log_vat", "log1p_hb", "age"]):
            covs = ["age", "sex_male"] if ex != "age" else ["sex_male"]
            r[y][ex] = {"M1": coef(fit(d, y, [ex], covs), ex),
                        "M2_bmi": coef(fit(d, y, [ex], covs + ["bmi"]), ex) if ex != "bmi" else None}
    out["associations"] = r
    e = d[d["qrs"] < 120] if "qrs" in d.columns else d
    out["excluding_qrs_ge_120"] = {"n": rc(len(e)),
                                   "sokolow_on_sbp": coef(fit(e, "sokolow", ["sbp_sit"], ["age", "sex_male", "bmi"]), "sbp_sit")}
    return out


# ============================================================================ S5 drug starters

def s5_starters(fr, medflags):
    out = {}
    if medflags is None:
        return {"error": "no medications table"}
    s0, s1 = MAIN_STAGES
    w = medflags.pivot_table(index="participant_id", columns="research_stage", values=list(ATC_CLASSES), aggfunc="max")
    stages = set(medflags["research_stage"])
    later = [st for st in LATER_STAGES if st in stages]
    out["stages_with_med_records"] = {st: rc((medflags["research_stage"] == st).sum()) for st in sorted(stages)
                                      if st in (s0,) + LATER_STAGES}
    fr2, _ = acute_deviation(fr)
    cols = ["pwv", "dev_bsbp", "sbp_sit", "acute_dev", "imt", "art_w", "abi_min", "bmi", "age", "visit_date", "sokolow"]
    pw = wide(fr2, [c for c in cols if c in fr2.columns])
    have_v2_med = w.index[w[(list(ATC_CLASSES)[0], s1)].notna()] if (list(ATC_CLASSES)[0], s1) in w.columns else pd.Index([])
    for drug in ATC_CLASSES:
        if (drug, s0) not in w.columns:
            continue
        base = w[(drug, s0)].fillna(0) == 1
        lat = pd.concat([w[(drug, st)] for st in later if (drug, st) in w.columns], axis=1)
        later_any = lat.max(axis=1).fillna(0) == 1
        at_v2 = w[(drug, s1)].fillna(0) == 1 if (drug, s1) in w.columns else later_any
        has_base = w[(drug, s0)].notna()
        starters = w.index[has_base & ~base & later_any & at_v2]
        never = w.index[has_base & ~base & ~later_any].intersection(have_v2_med)
        rec = {"baseline_users": rc(base.sum()), "starters_present_at_v2": rc(len(starters)), "never_users": rc(len(never))}
        grp = pd.Series(np.nan, index=pw.index)
        grp[grp.index.isin(starters)] = 1.0
        grp[grp.index.isin(never)] = 0.0
        ch = pd.DataFrame({"start": grp})
        for c in ("pwv", "dev_bsbp", "sbp_sit", "acute_dev", "imt", "art_w", "abi_min", "bmi", "sokolow"):
            if (c, s0) in pw.columns and (c, s1) in pw.columns:
                ch["d_" + c] = pw[(c, s1)] - pw[(c, s0)]
                ch[c + "0"] = pw[(c, s0)]
        ch["age0"] = pw[("age", s0)] if ("age", s0) in pw.columns else np.nan
        ch["interval"] = (pd.to_datetime(pw[("visit_date", s1)]) - pd.to_datetime(pw[("visit_date", s0)])).dt.days / 365.25 \
            if ("visit_date", s1) in pw.columns else np.nan
        for c in ("d_dev_bsbp", "d_sbp_sit", "d_acute_dev"):
            if c in ch.columns:
                ch[c + "10"] = ch[c] / 10
        ch = ch.reset_index().merge(fr[["participant_id", "sex_male"]].drop_duplicates("participant_id"), how="left")
        n_st = int(((ch["start"] == 1) & ch.get("d_pwv", pd.Series(np.nan, index=ch.index)).notna()).sum())
        rec["starters_with_pwv_both"] = rc(n_st)
        if drug == "antihtn" or n_st >= MIN_STARTERS:
            cv = ["pwv0", "age0", "sex_male", "d_bmi", "interval"]
            m = {"a_ancova": fit(ch, "d_pwv", ["start"], cv, std=False, min_n=50),
                 "b_plus_d_device_and_d_sitting": fit(ch, "d_pwv", ["start"], cv + ["d_dev_bsbp10", "d_sbp_sit10"],
                                                      std=False, min_n=50),
                 "c_plus_d_sitting_only": fit(ch, "d_pwv", ["start"], cv + ["d_sbp_sit10"], std=False, min_n=50),
                 "d_plus_d_acute_only": fit(ch, "d_pwv", ["start"], cv + ["d_acute_dev10"], std=False, min_n=50),
                 "bp_change_sitting": fit(ch, "d_sbp_sit", ["start"], ["sbp_sit0", "age0", "sex_male", "interval"],
                                          std=False, min_n=50),
                 "bp_change_device": fit(ch, "d_dev_bsbp", ["start"], ["dev_bsbp0", "age0", "sex_male", "interval"],
                                         std=False, min_n=50)}
            for bed in ("imt", "art_w", "abi_min", "sokolow"):
                if "d_" + bed in ch.columns:
                    m[f"other_bed_{bed}"] = fit(ch, "d_" + bed, ["start"], [bed + "0", "age0", "sex_male", "d_bmi", "interval"],
                                                std=False, min_n=50)
            rec["models"] = {k: coef(v, "start") | {"n": v["n"]} if v and coef(v, "start") else None for k, v in m.items()}
        out[drug] = rec
    return out


# ============================================================================ S6 incident hypertension

def s6_incident(fr, htn, medflags):
    out = {}
    s0, s1 = MAIN_STAGES
    base = fr[fr.research_stage == s0].copy()
    flow = {"baseline": rc(len(base))}
    base = base[(base["antihtn"] == 0) & (base["sbp_sit"] < 140) & (base["dbp_sit"] < 90)]
    flow["untreated_measured_normotensive"] = rc(len(base))
    if "htn_curated" in base.columns:
        base = base[base["htn_curated"].fillna(0) == 0]
        flow["not_curated_hypertensive"] = rc(len(base))
    if "htn_self" in base.columns:
        base = base[base["htn_self"].fillna(0) == 0]
        flow["not_self_reported"] = rc(len(base))
    out["flow"] = flow
    v2 = fr[fr.research_stage == s1][["participant_id", "sbp_sit", "dbp_sit", "antihtn", "visit_date"]].rename(
        columns={"sbp_sit": "sbp2", "dbp_sit": "dbp2", "antihtn": "antihtn2", "visit_date": "date2"})
    d = base.merge(v2, on="participant_id", how="inner")
    if medflags is not None:
        lm = medflags[medflags["research_stage"].isin(LATER_STAGES)].groupby("participant_id")["antihtn"].max()
        d["antihtn_later"] = d["participant_id"].map(lm).fillna(0)
    else:
        d["antihtn_later"] = 0.0
    if htn is not None:
        lh = htn[htn["research_stage"].isin(LATER_STAGES)].groupby("participant_id")[["htn_curated", "htn_self"]].max()
        d["htn_later"] = d["participant_id"].map(lh.max(axis=1)).fillna(0)
    else:
        d["htn_later"] = 0.0
    d["inc_measured"] = ((d["sbp2"] >= 140) | (d["dbp2"] >= 90)).astype(float)
    d.loc[d["sbp2"].isna() & d["dbp2"].isna(), "inc_measured"] = np.nan
    d["inc_any"] = ((d["inc_measured"] == 1) | (d["antihtn2"] == 1) | (d["antihtn_later"] == 1) |
                    (d["htn_later"] == 1)).astype(float)
    d["interval"] = (pd.to_datetime(d["date2"]) - pd.to_datetime(d["visit_date"])).dt.days / 365.25
    out["n_at_risk_with_v2"] = rc(len(d))
    out["incidence"] = {"measured": rc(d["inc_measured"].sum()), "any_definition": rc(d["inc_any"].sum())}
    out["interval_years"] = qtiles(d["interval"])
    beds = [b for b in CORE + ["sokolow"] if b in d.columns]
    res = {}
    covs = ["age", "sex_male", "bmi", "sbp_sit", "dbp_sit"]
    for bed in beds:
        e = d.dropna(subset=["o_" + bed, "inc_any"] + covs)
        if len(e) < 200 or e["inc_any"].sum() < MINCELL:
            continue
        X0 = np.column_stack([np.ones(len(e))] + [e[c].astype(float) for c in covs])
        X1 = np.column_stack([X0, zs(e["o_" + bed]).values])
        y = e["inc_any"].values
        b, se, p1 = logit(y, X1)
        _, _, p0 = logit(y, X0)
        if b is None:
            continue
        r = {"n": rc(len(e)), "events": rc(y.sum()), "log_or_per_sd": b[-1], "se": se[-1],
             "or": float(np.exp(b[-1])), "or_lo": float(np.exp(b[-1] - 1.96 * se[-1])),
             "or_hi": float(np.exp(b[-1] + 1.96 * se[-1])), "auc_base": auc(y, p0), "auc_with_bed": auc(y, p1)}
        cont = e.dropna(subset=["sbp2"]).copy()
        cont = cont[cont["antihtn2"] == 0]
        cont["o_bed_z"] = zs(cont["o_" + bed])
        f = fit(cont, "sbp2", ["o_bed_z"], covs + ["interval"], std=False)
        r["sbp_v2_per_sd_untreated"] = coef(f, "o_bed_z") | {"n": f["n"]} if f else None
        res[bed] = r
    out["per_bed"] = res
    # v1.1: joint models always report their sample and status; three-bed fallbacks when four beds are too few
    joint = {}
    for label, bset in [("four_beds", CORE)] + [(f"without_{x}", [b for b in CORE if b != x]) for x in CORE]:
        e = d.dropna(subset=["o_" + b for b in bset] + ["inc_any"] + covs)
        info = {"n": rc(len(e)), "events": rc(e["inc_any"].sum()) if len(e) else None}
        if len(e) >= 300 and e["inc_any"].sum() >= MINCELL:
            X = np.column_stack([np.ones(len(e))] + [e[c].astype(float) for c in covs] +
                                [zs(e["o_" + b]).values for b in bset])
            b, se, _ = logit(e["inc_any"].values, X)
            if b is None:
                info["status"] = "logit failed"
            else:
                k = len(bset)
                info["status"] = "ok"
                info.update({bd: {"or": float(np.exp(b[-k + i])), "or_lo": float(np.exp(b[-k + i] - 1.96 * se[-k + i])),
                                  "or_hi": float(np.exp(b[-k + i] + 1.96 * se[-k + i]))} for i, bd in enumerate(bset)})
        else:
            info["status"] = "too few (needs n>=300 and >=20 events)"
        joint[label] = info
    out["joint"] = joint
    return out


# ============================================================================ S7 vascular-age equivalents

def s7_vascage(fr, boot, rng):
    out = {"age_slopes": {}, "age_equivalents": {}, "reference_quantiles": {}}
    ldl_unit = 38.67 if fr["ldl"].median(skipna=True) > 20 else 1.0
    contrasts_ = {"waist": 10.0, "bmi": 5.0, "sbp_sit": 10.0, "hba1c": 0.5, "ldl": ldl_unit, "smk_current": 1.0,
                  "cgm_mean": 10.0}
    sd_contrasts = ["log_vat", "log_tg", "log1p_hb", "log1p_t90", "leg_fat", "att"]
    for bed in [b for b in CORE + ["sokolow"] if b in fr.columns]:
        ob = "o_" + bed
        d = one_row_per_participant(fr, [ob, "age", "sex_male"])
        if len(d) < 200:
            continue
        f = fit(d, ob, ["age"], ["sex_male"], std=False)
        out["age_slopes"][bed] = {"per_year_oriented_units": coef(f, "age"), "n": f["n"] if f else None}
        d["age2c"] = (d["age"] - 50) ** 2
        f2 = fit(d, ob, ["age", "age2c"], ["sex_male"], std=False)
        out["age_slopes"][bed]["quadratic"] = coef(f2, "age2c") if f2 else None
        eq = {}
        for ex in avail(d, list(contrasts_) + sd_contrasts):
            e = d.dropna(subset=[ex])
            if len(e) < 200:
                continue
            delta = contrasts_.get(ex, float(e[ex].std()))
            X = np.column_stack([np.ones(len(e)), e["age"].astype(float), e["sex_male"].astype(float), e[ex].astype(float)])
            y = e[ob].astype(float).values
            b = beta_only(y, X)
            est = b[3] * delta / b[1] if b[1] else np.nan
            bs = []
            for _ in range(boot):
                idx = rng.integers(0, len(e), len(e))
                bb = beta_only(y[idx], X[idx])
                bs.append(bb[3] * delta / bb[1] if bb[1] else np.nan)
            eq[ex] = {"contrast": delta if ex in contrasts_ else "1 SD", "years": est, "ci": pct(bs), "n": rc(len(e))}
        out["age_equivalents"][bed] = eq
        ref = {}
        for sx in (0, 1):
            for lo, hi in ((20, 40), (40, 50), (50, 60), (60, 70), (70, 100)):
                sub = d[(d["sex_male"] == sx) & (d["age"] >= lo) & (d["age"] < hi)][bed]
                ref[f"sex{sx}_age{lo}_{hi}"] = qtiles(sub)
        out["reference_quantiles"][bed] = ref
    return out


# ============================================================================ S8 extended data

def s8_extended(fr, rng):
    out = {"nonlinearity": {}, "sex_strata": {}, "negative_control_birth_month": {}, "artefacts": {},
           "abi_exclusion": {}}
    beds = [b for b in CORE + ["sokolow"] if b in fr.columns]
    for bed in beds:
        d = one_row_per_participant(fr, [bed, "age", "sex_male"])
        nl = {}
        for ex in avail(d, ["bmi", "waist", "log_vat", "leg_fat", "sbp_sit", "log_tg", "hba1c", "log1p_hb", "age", "height"]):
            e = d.dropna(subset=[ex]).copy()
            if len(e) < 500:
                continue
            q = e[ex].quantile([0.2, 0.4, 0.6, 0.8]).values
            if len(np.unique(q)) < 4:
                continue
            e["qi"] = np.searchsorted(q, e[ex].values, side="right")
            covs = ["age", "sex_male"] if ex != "age" else ["sex_male"]
            y = zs(e[bed]).values
            D = np.column_stack([(e["qi"] == k).astype(float) for k in range(1, 5)])
            C = np.column_stack([e[c].astype(float) for c in covs])
            Xq = np.column_stack([np.ones(len(e)), D, C])
            Xl = np.column_stack([np.ones(len(e)), e[ex].astype(float), C])
            Xf = np.column_stack([np.ones(len(e)), e[ex].astype(float), D, C])  # v1.1: nested (linear + quintiles)
            bq, seq, _ = ols(y, Xq)
            rss_q = ((y - Xf @ beta_only(y, Xf)) ** 2).sum()
            rss_l = ((y - Xl @ beta_only(y, Xl)) ** 2).sum()
            df1, df2 = 4, len(e) - Xf.shape[1]
            F = ((rss_l - rss_q) / df1) / (rss_q / df2)
            counts = [rc((e["qi"] == k).sum()) for k in range(5)]
            nl[ex] = {"quintile_cuts": q.tolist(), "n_per_quintile": counts,
                      "beta_vs_q1_sd": [{"b": bq[k], "se": seq[k]} for k in range(1, 5)],
                      "f_nonlinear": float(F), "p_nonlinear": float(stats.f.sf(F, df1, df2))}
        out["nonlinearity"][bed] = nl
        ss = {}
        for ex in avail(d, ["bmi", "waist", "log_vat", "sbp_sit", "log_tg", "hba1c", "ldl", "log1p_hb", "smk_current"]):
            e = d.dropna(subset=[ex]).copy()
            r = {f"sex_{sx}": coef(fit(e[e.sex_male == sx], bed, [ex], ["age"]), ex) for sx in (0, 1)}
            e["ex_z"] = zs(e[ex])
            e["inter"] = e["ex_z"] * e["sex_male"]
            fi = fit(e, bed, ["inter"], ["ex_z", "age", "sex_male"], std=False)
            r["interaction"] = coef(fi, "inter")
            ss[ex] = r
        out["sex_strata"][bed] = ss
        e = d.dropna(subset=["bm_sin", "bm_cos"])
        if len(e) >= 200:
            fo = fit(e, bed, ["bm_sin", "bm_cos"], ["age", "sex_male"], std=True)
            if fo:
                bvec = np.array([fo["coef"]["bm_sin"]["b"], fo["coef"]["bm_cos"]["b"]])
                sev = np.array([fo["coef"]["bm_sin"]["se"], fo["coef"]["bm_cos"]["se"]])
                W = float(((bvec / sev) ** 2).sum())
                out["negative_control_birth_month"][bed] = {"n": fo["n"], "wald_approx": W, "p": float(stats.chi2.sf(W, 2))}
        art = {}
        for ex in avail(d, ["height", "hr_sit", "acq_hour", "season_sin", "season_cos", "leg_lean"]):
            art[ex] = coef(fit(d, bed, [ex], ["age", "sex_male", "bmi"]), ex)
        out["artefacts"][bed] = art
    d = one_row_per_participant(fr, ["pwv", "age", "sex_male", "abi_min", "abi_max"])
    keep = d[(abi_clin(d) >= 0.9) & (d["abi_max"] <= 1.4)]
    out["abi_exclusion"] = {"n_all": rc(len(d)), "n_kept": rc(len(keep)),
                            **{ex: {"all": coef(fit(d, "pwv", [ex], ["age", "sex_male"]), ex),
                                    "kept": coef(fit(keep, "pwv", [ex], ["age", "sex_male"]), ex)}
                               for ex in avail(d, ["bmi", "waist", "log_vat", "sbp_sit", "log_tg", "hba1c", "leg_fat"])}}
    return out



# ============================================================================ digest (for screen transcription)

def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def v3(x):
    if x is None:
        return "NA"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "NA"
    return f"{x:.3g}"


def ci3(c):
    return "[NA]" if not c or c[0] is None else f"[{v3(c[0])},{v3(c[1])}]"


def cf(c):
    """coefficient dict -> 'b [lo,hi]'."""
    if not isinstance(c, dict):
        return "NA"
    b = c.get("b", c.get("d"))
    if b is None:
        return "NA"
    if "lo" in c:
        return f"{v3(b)} [{v3(c['lo'])},{v3(c['hi'])}]"
    return f"{v3(b)} {ci3(c.get('ci') or c.get('boot_ci'))}"


def digest(res):
    L = [f"F1 ATLAS DIGEST  {res.get('version')}  boot {res.get('boot')}"]
    s0 = res.get("S0_facts", {})
    L.append("## S0 facts")
    L.append("four beds same visit v1/v2: " + " / ".join(str(g(s0, "overlap", "four_beds", st)) for st in MAIN_STAGES)
             + f"; participants any visit {g(s0, 'overlap', 'four_beds_any_visit_participants')}")
    for k in ("four_beds_plus_dxa", "four_beds_plus_sleep", "four_beds_plus_hb", "four_beds_plus_cgm", "four_beds_plus_ecg"):
        L.append(f"{k}: {g(s0, 'overlap', k)}")
    L.append("pairs: " + "; ".join(f"{k} {v}" for k, v in (g(s0, "overlap", "pairs_participants") or {}).items()))
    L.append(f"pwv-bp gap min q05/q50/q95: {v3(g(s0, 'pwv_minus_bp_minutes', 'q05'))}/"
             f"{v3(g(s0, 'pwv_minus_bp_minutes', 'q50'))}/{v3(g(s0, 'pwv_minus_bp_minutes', 'q95'))}; "
             f"fractions {g(s0, 'pwv_bp_gap_fraction')}")
    L.append(f"visit interval y q05/q50/q95: {v3(g(s0, 'visit_interval_years', 'q05'))}/"
             f"{v3(g(s0, 'visit_interval_years', 'q50'))}/{v3(g(s0, 'visit_interval_years', 'q95'))}")
    for k, v in (g(s0, "between_visit_reliability") or {}).items():
        L.append(f"retest {k}: n {v.get('n')} icc {v3(v.get('icc1'))} r {v3(v.get('pearson'))} mean change {v3(v.get('mean_change'))}")
    for k, v in (g(s0, "agreement_within_visit") or {}).items():
        L.append(f"agree {k}: n {v.get('n')} r {v3(v.get('pearson'))} icc {v3(v.get('icc1'))} "
                 f"diff {v3(v.get('mean_diff'))} sd {v3(v.get('sd_diff'))}")
    for c in ("pwv", "imt", "art_w", "abi_min", "dev_bsbp", "sbp_sit", "hb", "sokolow", "age"):
        q = g(s0, "descriptives", c) or {}
        L.append(f"desc {c}: n {q.get('n')} q05 {v3(q.get('q05'))} q50 {v3(q.get('q50'))} q95 {v3(q.get('q95'))}")
    L.append(f"abi categories: {g(s0, 'abi_categories')}; fractions: "
             + ", ".join(f"{k} {v3(v)}" for k, v in (g(s0, "fractions") or {}).items()))
    sd_ = res.get("S0b_drift", {})
    L.append("## S0b drift (v1.1)")
    acq = sd_.get("acquisition") or {}
    L.append("arrays per visit q05/q50/q95: " + "; ".join(
        f"{st[:2]} n {v.get('n')} {v3(v.get('q05'))}/{v3(v.get('q50'))}/{v3(v.get('q95'))}"
        for st, v in (acq.get("vascular_arrays_per_visit") or {}).items()))
    L.append("pwv by array order q50 (n): " + "; ".join(
        f"{st[:2]} " + " ".join(f"#{o} {v3(v.get('q50'))}({v.get('n')})" for o, v in od.items())
        for st, od in (acq.get("pwv_by_array_order") or {}).items()))
    if acq.get("arrays_error"):
        L.append(f"arrays error: {acq.get('arrays_error')}")
    L.append("stage q25/q50/q75 (00 | 02): " + "; ".join(
        f"{c} " + " | ".join(f"{v3(g(v, st, 'q25'))}/{v3(g(v, st, 'q50'))}/{v3(g(v, st, 'q75'))}" for st in MAIN_STAGES)
        for c, v in (sd_.get("by_stage") or {}).items()))
    for c in ("pwv", "pwv_r", "pwv_l", "dev_bsbp", "abi_min", "imt", "hr_sit"):
        bq = g(sd_, "by_quarter", c) or {}
        L.append(f"quarter {c}: " + " ".join(f"{k}:{v3(v.get('median'))}({v.get('n')})" for k, v in bq.items()))
    for c, v in (sd_.get("calendar_trend_per_year") or {}).items():
        L.append(f"trend/yr {c}: " + "; ".join(f"{st[:2]} {cf(e)} n {e.get('n') if e else None}" for st, e in v.items()))
    L.append("stage02-stage00 same quarter: " + "; ".join(
        f"{c} {cf(v)} n {v.get('n')} q {v.get('n_quarters')}" for c, v in (sd_.get("stage2_vs_stage0_same_quarter") or {}).items()))
    for c, v in (sd_.get("retest_alternatives") or {}).items():
        L.append(f"retest2 {c}: n {v.get('n')} icc1 {v3(v.get('icc1'))} iccC {v3(v.get('icc_c'))} r {v3(v.get('pearson'))} "
                 f"chg {v3(v.get('mean_change'))} sd {v3(v.get('sd_change'))} | demeaned n {v.get('n_dq')} icc1 "
                 f"{v3(v.get('icc1_dq'))} iccC {v3(v.get('icc_c_dq'))} r {v3(v.get('pearson_dq'))} chg {v3(v.get('mean_change_dq'))}"
                 + (f" | adjusted icc1 {v3(v.get('icc1_adj'))} r {v3(v.get('pearson_adj'))} chg {v3(v.get('mean_change_adj'))}"
                    if "icc1_adj" in v else ""))
    L.append(f"calendar adjustment: {'on' if res.get('calendar_adjust') else 'off'} (beds {','.join(CAL_BEDS)}; "
             "effects = quarter fixed effects given age, age^2, sex; weighted mean 0)")
    for c, v in (acq.get("calendar_effects") or {}).items():
        L.append(f"caleff {c}: " + " ".join(f"{k}:{v3(e.get('effect'))}({e.get('n')})" for k, e in v.items()))
    if acq.get("day_effects") is not None or res.get("day_adjust") is not None:
        L.append(f"day adjustment: {'on' if res.get('day_adjust') else 'off'} | " + "; ".join(
            f"{c} share {v3(v.get('day_share'))} k {v3(v.get('k'))} days {v.get('n_days')} med_n {v3(v.get('median_n_per_day'))}"
            for c, v in (acq.get("day_effects") or {}).items()))
    if acq.get("session_kernel") is not None or res.get("session_adjust") is not None:
        L.append(f"session adjustment: {'on' if res.get('session_adjust') else 'off'} (v1.5: same-day unrelated pair cov = "
                 f"day + s*exp(-dt/tau), relatives excluded {acq.get('session_relations_excluded')}; applied where "
                 f"z >= {SESSION_MIN_Z:g} as LOO day+session BLUP, else v1.4; share/day = s/var, b0/var; bins |dt| min: "
                 "r before>after/n pairs)")
        labs = [f"{a}-{b}" for a, b in zip(SESSION_EDGES[:-1], SESSION_EDGES[1:])] + [f"{SESSION_EDGES[-1]}+"]

        def r2(x):
            return re.sub(r"^(-?)0\.", r"\1.", f"{float(x):.2f}") if x is not None else "NA"
        for c, v in (acq.get("session_kernel") or {}).items():
            bins = v.get("bins") or {}
            L.append(f"session kernel {c}: share {v3(v.get('share'))} tau {v.get('tau')} z {v3(v.get('z'))} day "
                     f"{v3(v.get('day_share'))} applied {v.get('applied')} chg {v3(v.get('change_vs_day_sd'))} pairs "
                     f"{v.get('n_pairs')} |" + "".join(
                         f" {k}:{r2(bins[k].get('r'))}>{r2(bins[k].get('r_after'))}/{bins[k].get('n')}"
                         for k in labs if k in bins))
    s1 = res.get("S1_atlas", {})
    L.append("## S1 atlas (common sample n %s)" % g(s1, "common", "n_participants"))
    L.append("reliability icc: " + "; ".join(f"{k} {v3(v.get('icc'))} {ci3(v.get('ci'))} n {v.get('n_pairs')}"
                                             for k, v in (g(s1, "common", "reliability_icc_between_visits") or {}).items()))
    for m, t in (g(s1, "decision_rule_tally") or {}).items():
        L.append(f"DECISION {m}: oriented {t.get('n_nodes_oriented')} {','.join(t.get('nodes_oriented', []))} | "
                 f"raw-sign {t.get('n_nodes_raw_sign')}")
    for m in ("M1_age_sex", "M3_plus_bp_meds"):
        L.append(f"# {m}: exposure | n | oriented beta pwv imt art_w abi | pwv-imt [CI] | pwv-art_w [CI] | Wald p | q")
        for ex, v in (g(s1, "common", m) or {}).items():
            b = v.get("beta_oriented", {})
            dd = v.get("diff_oriented", {})
            L.append(f"{ex} | {v.get('n')} | " + " ".join(v3(g(b, bd, "b")) for bd in CORE) +
                     f" | {cf(dd.get('pwv_minus_imt'))} | {cf(dd.get('pwv_minus_art_w'))} | "
                     f"{v3(g(v, 'heterogeneity', 'p'))} | {v3(g(v, 'heterogeneity', 'q_bh'))}")
    s2 = res.get("S2_pressure", {})
    L.append("## S2 pressure (n %s)" % s2.get("n_participants"))
    L.append(f"acute model r2 {v3(g(s2, 'acute_dev_model', 'r2_device_on_visit_bp'))} sd {v3(g(s2, 'acute_dev_model', 'acute_dev_sd'))}; "
             f"device-sitting q50 {v3(g(s2, 'device_minus_sitting_sbp', 'q50'))}")
    cs = g(s2, "cross_sectional_per_10mmHg") or {}
    L.append(f"per 10 mmHg: sitting only {cf(g(cs, 'a_sitting_only', 'coef', 'sbp_sit10'))}; device only "
             f"{cf(g(cs, 'b_device_only', 'coef', 'dev_bsbp10'))}; sitting+acute: sit {cf(g(cs, 'c_sitting_plus_acute', 'coef', 'sbp_sit10'))} "
             f"acute {cf(g(cs, 'c_sitting_plus_acute', 'coef', 'acute_dev10'))}")
    ra = g(s2, "rc_acute", "per_10mmHg") or {}
    L.append(f"RC: usual sbp {cf(ra.get('usual_sbp_rc'))}; device given usual {cf(ra.get('device_sbp_given_usual'))}; "
             f"calibration {g(s2, 'rc_acute', 'calibration')}")
    for bed in ("imt", "art_w", "abi_min", "sokolow", "cavi0"):
        L.append(f"{bed} sitting+acute: acute {cf(g(cs, bed + '_sitting_plus_acute', 'coef', 'acute_dev10'))}")
    for k, v in (g(s2, "acute_by_stratum") or {}).items():
        L.append(f"acute {k}: {cf(g(v, 'coef', 'acute_dev10'))}")
    wp = g(s2, "within_person") or {}
    L.append(f"within-person n {wp.get('n_pairs')}: d_sit {cf(g(wp, 'a_d_sitting', 'coef', 'd_sit10'))}; d_device "
             f"{cf(g(wp, 'b_d_device', 'coef', 'd_dev10'))}; d_sit+d_acute: {cf(g(wp, 'c_d_sitting_plus_d_acute', 'coef', 'd_sit10'))} / "
             f"{cf(g(wp, 'c_d_sitting_plus_d_acute', 'coef', 'd_acute10'))}; d_map {cf(g(wp, 'd_d_map', 'coef', 'd_map10'))}")
    L.append(f"lagged ratio {v3(g(s2, 'lagged', 'ratio'))} {ci3(g(s2, 'lagged', 'ratio_ci'))} n {g(s2, 'lagged', 'n')}")
    L.append("# shares: exposure | n | beta total | share visit BP [CI] | share acute [CI] | RC share [CI] | RC acute share [CI]")
    for ex, v in (g(s2, "exposure_shares_pwv") or {}).items():
        r_ = v.get("rc", {})
        L.append(f"{ex} | {v.get('n')} | {v3(v.get('beta_total'))} | {v3(v.get('share_visit_bp'))} {ci3(v.get('share_visit_bp_ci'))} | "
                 f"{v3(v.get('share_acute'))} {ci3(v.get('share_acute_ci'))} | {v3(r_.get('share_rc_bp'))} {ci3(r_.get('share_rc_bp_ci'))} | "
                 f"{v3(r_.get('share_acute_after_rc'))} {ci3(r_.get('share_acute_after_rc_ci'))}")
    cav = g(s2, "cavi0_sensitivity", "bp_dependence") or {}
    L.append(f"CAVI0 vs PWV sitting/acute: pwv {cf(g(cav, 'pwv', 'coef', 'sbp_sit10'))} / {cf(g(cav, 'pwv', 'coef', 'acute_dev10'))}; "
             f"cavi0 {cf(g(cav, 'cavi0', 'coef', 'sbp_sit10'))} / {cf(g(cav, 'cavi0', 'coef', 'acute_dev10'))}")
    L.append(f"bp reliability: {g(s2, 'bp_reliability')}")
    s3 = res.get("S3_hypoxia", {})
    L.append("## S3 hypoxia")
    L.append(f"computation: {s3.get('computation')}")
    L.append(f"spearman hb: " + ", ".join(f"{k} {v3(v)}" for k, v in (g(s3, "spearman", "log1p_hb") or {}).items()))
    for bed, r_ in (g(s3, "beds") or {}).items():
        L.append(f"{bed}: " + "; ".join(f"{ex} M2 {cf(v.get('M2'))} M3 {cf(v.get('M3_vat_bp'))}"
                                         for ex, v in r_.items() if isinstance(v, dict) and "M2" in v))
    s4 = res.get("S4_ecg", {})
    L.append("## S4 ecg n %s; LVH %s" % (s4.get("n_participants"), s4.get("lvh_prevalence")))
    for y in ("sokolow", "cornell"):
        L.append(f"{y}: " + "; ".join(f"{ex} {cf(v.get('M1'))}" for ex, v in (g(s4, "associations", y) or {}).items()))
    s5 = res.get("S5_starters", {})
    L.append("## S5 starters")
    for drug, v in s5.items():
        if not isinstance(v, dict) or "baseline_users" not in v:
            continue
        L.append(f"{drug}: base users {v.get('baseline_users')} starters {v.get('starters_present_at_v2')} with pwv both "
                 f"{v.get('starters_with_pwv_both')} never {v.get('never_users')}")
        for k, m in (v.get("models") or {}).items():
            L.append(f"  {k}: {cf(m)} n {m.get('n') if m else None}")
    s6 = res.get("S6_incident_htn", {})
    L.append("## S6 incident hypertension: flow %s; at risk %s; incidence %s" % (s6.get("flow"), s6.get("n_at_risk_with_v2"),
                                                                             s6.get("incidence")))
    for bed, v in (s6.get("per_bed") or {}).items():
        L.append(f"{bed}: OR {v3(v.get('or'))} [{v3(v.get('or_lo'))},{v3(v.get('or_hi'))}] auc {v3(v.get('auc_base'))}->"
                 f"{v3(v.get('auc_with_bed'))}; sbp v2 per SD {cf(v.get('sbp_v2_per_sd_untreated'))}")
    for lab, v in (s6.get("joint") or {}).items():
        L.append(f"joint {lab}: n {v.get('n')} ev {v.get('events')} {v.get('status')} " + " ".join(
            f"{bd} {v3(e.get('or'))} [{v3(e.get('or_lo'))},{v3(e.get('or_hi'))}]" for bd, e in v.items() if isinstance(e, dict)))
    s7 = res.get("S7_vascular_age", {})
    L.append("## S7 vascular age")
    for bed, v in (g(s7, "age_slopes") or {}).items():
        L.append(f"{bed} age slope {cf(v.get('per_year_oriented_units'))}; equivalents: " + "; ".join(
            f"{ex} {v3(e.get('years'))}y {ci3(e.get('ci'))}" for ex, e in (g(s7, "age_equivalents", bed) or {}).items()))
    s8 = res.get("S8_extended", {})
    L.append("## S8 extended")
    for bed, v in (g(s8, "nonlinearity") or {}).items():
        L.append(f"nonlinear p {bed}: " + ", ".join(f"{ex} {v3(e.get('p_nonlinear'))}" for ex, e in v.items()))
    L.append("negative control birth month p: " + ", ".join(f"{bed} {v3(v.get('p'))}" for bed, v in
                                                             (g(s8, "negative_control_birth_month") or {}).items()))
    for bed, v in (g(s8, "sex_strata") or {}).items():
        L.append(f"sex interaction {bed}: " + ", ".join(f"{ex} {cf(e.get('interaction'))}" for ex, e in v.items()))
    for bed, v in (g(s8, "artefacts") or {}).items():
        L.append(f"artefacts {bed}: " + ", ".join(f"{ex} {cf(e)}" for ex, e in v.items()))
    ab = g(s8, "abi_exclusion") or {}
    L.append(f"abi exclusion n {ab.get('n_all')}->{ab.get('n_kept')}: " + ", ".join(
        f"{ex} {cf(v.get('all'))}->{cf(v.get('kept'))}" for ex, v in ab.items() if isinstance(v, dict)))
    for k, v in res.items():
        if isinstance(v, dict) and "error" in v:
            L.append(f"ERROR {k}: {v['error']}")
    return "\n".join(L) + "\n"


# ============================================================================ main

SECTIONS = ("S0", "S0b", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--skip-hb", action="store_true", help="skip the per-night hypoxic-burden computation")
    ap.add_argument("--max-nights", type=int, default=None, help="debug: limit nights for hypoxic burden")
    ap.add_argument("--no-calendar-adjust", action="store_true", help="v1.2: keep raw device beds (sensitivity)")
    ap.add_argument("--no-day-adjust", action="store_true", help="v1.4: quarter adjustment only (sensitivity)")
    ap.add_argument("--no-session-adjust", action="store_true", help="v1.5: report the time kernel, do not remove it")
    a = ap.parse_args()
    global CAL_ADJUST, DAY_ADJUST, SESSION_ADJUST
    CAL_ADJUST = not a.no_calendar_adjust
    DAY_ADJUST = not a.no_day_adjust
    SESSION_ADJUST = not a.no_session_adjust
    if os.path.exists(a.out):
        raise SystemExit(f"output directory exists: {a.out} (refusing to overwrite)")
    sections = [s.strip() for s in a.sections.split(",") if s.strip()]
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    log(f"{VERSION}; numpy {np.__version__}; pandas {pd.__version__}; python {sys.version.split()[0]}; boot {a.boot}")
    hb, hb_meta = None, {"skipped": True}
    if not a.skip_hb:
        try:
            hb, hb_meta = hypoxic_burden(a.root, a.workers, a.max_nights)
        except Exception as ex:
            hb_meta = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
            log(f"hypoxic burden FAILED: {hb_meta['error']}")
    fr, medflags, htn = build_frame(a.root, hb)
    pids = set(fr["participant_id"].astype(str))
    res = {"version": VERSION, "boot": a.boot, "sections": sections, "calendar_adjust": CAL_ADJUST,
           "day_adjust": CAL_ADJUST and DAY_ADJUST, "session_adjust": CAL_ADJUST and DAY_ADJUST and SESSION_ADJUST}
    fns = {"S0": ("S0_facts", lambda: s0_facts(fr)),
           "S0b": ("S0b_drift", lambda: s0b_drift(fr)),
           "S1": ("S1_atlas", lambda: s1_atlas(fr, a.boot, rng)),
           "S2": ("S2_pressure", lambda: s2_pressure(fr, a.boot, rng)),
           "S3": ("S3_hypoxia", lambda: s3_hypoxia(fr, hb_meta)),
           "S4": ("S4_ecg", lambda: s4_ecg(fr)),
           "S5": ("S5_starters", lambda: s5_starters(fr, medflags)),
           "S6": ("S6_incident_htn", lambda: s6_incident(fr, htn, medflags)),
           "S7": ("S7_vascular_age", lambda: s7_vascage(fr, a.boot, rng)),
           "S8": ("S8_extended", lambda: s8_extended(fr, rng))}
    for key in sections:
        name, fn = fns[key]
        t1 = time.time()
        try:
            res[name] = fn()
            log(f"{name} done in {time.time() - t1:.1f}s")
        except Exception as ex:  # keep other sections
            import traceback
            tb = traceback.extract_tb(ex.__traceback__)[-1]
            res[name] = {"error": f"{type(ex).__name__}: {str(ex)[:200]} (line {tb.lineno})"}
            log(f"{name} FAILED: {type(ex).__name__}: {str(ex)[:200]} (line {tb.lineno})")
    res["runtime_s"] = time.time() - t0
    text = json.dumps(clean(res), ensure_ascii=False, indent=1, sort_keys=True)
    privacy_check(text, pids)
    logtext = "\n".join(LOG) + "\n"
    privacy_check(logtext, pids)
    dg = digest(json.loads(text))
    privacy_check(dg, pids)
    os.makedirs(a.out)
    files = {"results.json": text, "digest.txt": dg, "run_log.txt": logtext}
    sums = []
    for fn_, tx in files.items():
        with open(os.path.join(a.out, fn_), "w", encoding="utf-8") as fh:
            fh.write(tx)
        sums.append(f"{hashlib.sha256(tx.encode('utf-8')).hexdigest()}  {fn_}")
    with open(os.path.join(a.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print("\n".join(s[:16] + "  " + s.split()[-1] for s in sums))
    log(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
