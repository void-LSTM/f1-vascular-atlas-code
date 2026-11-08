"""Counts and design-information audit for the existing public IMPACT release."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_SHA = "070d98e5ff905abef4565e108eb8058a87f2c457cbb3b435b760175c7f2ea137"
HORIZON_DAYS = 5 * 365.25
BINARY = ["female", "hemodialysis", "diabetes", "hx_old_mi", "hx_stroke"]
CLINICAL = ["age_y", *BINARY, "egfr_jsn"]
EXTRA = ["htn", "smoking"]
MARKERS = ["bapwv_max_cms", "abi_adverse"]
FLAGS = ["mi", "stroke", "death"]
TIMES = ["mi_fu_days", "stroke_fu_days", "death_fu_days"]
USED = ["row_id", "age_y", *BINARY, *EXTRA, "creatinine_mgdl", "abi_min", "bapwv_max_cms", *FLAGS, *TIMES]


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_impact(path):
    if fingerprint(path) != SOURCE_SHA:
        raise ValueError("IMPACT source hash mismatch")
    return pd.read_parquet(path, columns=USED)


def endpoint(data):
    times = data[TIMES].to_numpy(float)
    flags = data[FLAGS].to_numpy(float)
    valid = np.isfinite(times).all(axis=1) & (times > 0).all(axis=1) & np.isin(flags, [0, 1]).all(axis=1)
    end = np.where(flags[:, :2] == 0, times[:, :2], np.inf).min(axis=1)
    first = np.where(flags[:, :2] == 1, times[:, :2], np.inf).min(axis=1)
    death_time = times[:, 2]
    terminal = (flags[:, 2] == 1) & (death_time <= end) & (death_time <= HORIZON_DAYS) & (death_time <= first)
    nonfatal = valid & (first <= end) & (first <= HORIZON_DAYS) & (first <= death_time)
    nonfatal &= ~((flags[:, 2] == 1) & (first == death_time))
    t = np.minimum.reduce([end, first, death_time, np.full(len(data), HORIZON_DAYS)])
    which = np.where(nonfatal, 1, np.where(valid & terminal, 2, 0))
    return pd.DataFrame(dict(time_days=t, status=which, valid=valid,
                             first_before_competing=nonfatal,
                             mi_first=nonfatal & (flags[:, 0] == 1) & (times[:, 0] == first),
                             stroke_first=nonfatal & (flags[:, 1] == 1) & (times[:, 1] == first),
                             same_day_death=np.isfinite(first) & (first == death_time) & (flags[:, 2] == 1),
                             apparent_event_beyond_complete_followup=np.isfinite(first) & (first > end),
                             event_after_death=np.isfinite(first) & (first > death_time) & (flags[:, 2] == 1)),
                        index=data.index)


def prepare(data):
    if data.row_id.isna().any() or not data.row_id.is_unique:
        raise ValueError("released row key invalid")
    d = data.copy()
    for c in BINARY + EXTRA:
        d[c] = d[c].where(d[c].isin([0, 1]))
    d["egfr_jsn"] = (194 * d.creatinine_mgdl.where(d.creatinine_mgdl > 0)**-1.094 *
                     d.age_y.where(d.age_y > 0)**-.287 * np.where(d.female == 1, .739, 1))
    d["abi_adverse"] = -d.abi_min
    info = endpoint(d)
    base = d.age_y.ge(20) & d.abi_min.ge(.9) & np.isfinite(d[MARKERS]).all(axis=1) & info.valid
    keep = base & np.isfinite(d[CLINICAL]).all(axis=1)
    return d.join(info), base, keep


def precision_proxy(d, clinical=CLINICAL):
    x = d[clinical + MARKERS].to_numpy(float)
    x = x - x.mean(axis=0)
    sd = x.std(axis=0)
    if not len(d) or d.status.eq(1).sum() == 0 or np.any(sd <= 0):
        return None
    x = x / sd
    events = int(d.status.eq(1).sum())
    widths = {}
    for j, marker in enumerate(clinical + MARKERS):
        if marker not in MARKERS:
            continue
        others = np.delete(x, j, axis=1)
        residual = x[:,j] - others @ np.linalg.lstsq(others,x[:,j],rcond=None)[0]
        widths[marker] = float(1.96 / np.sqrt(events * np.mean(residual**2)))
    return widths


def audit(raw):
    d, base, keep = prepare(raw)
    subset = d[keep]
    def counts(m):
        s = d[m]
        return dict(n=len(s), first_mi_or_stroke=int(s.status.eq(1).sum()),
                    mi_first_including_both=int(s.mi_first.sum()), stroke_first_including_both=int(s.stroke_first.sum()),
                    both_on_first_day=int((s.mi_first & s.stroke_first).sum()),
                    competing_death=int(s.status.eq(2).sum()),
                    administrative_before_horizon=int((s.status.eq(0) & s.time_days.lt(HORIZON_DAYS)).sum()),
                    event_same_day_as_death_any_followup=int(s.same_day_death.sum()),
                    event_same_day_as_death_in_risk_window=int((s.same_day_death & s.status.eq(2)).sum()),
                    event_after_death=int(s.event_after_death.sum()),
                    apparent_event_beyond_complete_followup=int(s.apparent_event_beyond_complete_followup.sum()))
    return dict(raw_rows=len(d), horizon_days=HORIZON_DAYS, source_eligible=counts(base),
                complete_case=counts(keep), excluded_for_missing_adjustment=int((base & ~keep).sum()),
                expected_log_hr_halfwidth=precision_proxy(subset),
                source_sha256=SOURCE_SHA)


if __name__ == "__main__":
    print(json.dumps(audit(read_impact(Path(sys.argv[1]))), indent=2))
