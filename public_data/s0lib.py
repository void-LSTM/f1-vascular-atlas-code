"""Shared helpers for the F1 Stage 0 parsers (public data; parsing, cleaning, description only).

No association, regression, survival or MR model is fitted anywhere in Stage 0. The helpers below only read
source files, write cleaned parquet tables and write Markdown data dictionaries with descriptive statistics.

Runs with numpy 1.26.4, pandas 2.1.0 and pyarrow 15. Legacy .xls files are read by a second Python with xlrd
(READERS_PYTHON) in a subprocess, which returns the raw cell values as JSON; all cleaning then happens in the main
environment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HUB = Path(os.environ.get("F1_WORKDIR", Path(__file__).resolve().parents[1]))
EXT = HUB / "external"
PAPER = HUB
OUT = PAPER / "results" / "stage0"
CLEAN = OUT / "clean"
DICT = OUT / "dictionaries"
READERS_PY = Path(os.environ.get("READERS_PYTHON", HUB / ".venv-readers" / "bin" / "python"))

CLEAN.mkdir(parents=True, exist_ok=True)
DICT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_XLS_SNIPPET = r"""
import json, sys, xlrd
wb = xlrd.open_workbook(sys.argv[1])
out = {}
for sh in wb.sheets():
    rows = []
    for i in range(sh.nrows):
        row = []
        for j in range(sh.ncols):
            c = sh.cell(i, j)
            # ctype: 0 empty, 1 text, 2 number, 3 date, 4 bool, 5 error, 6 blank
            if c.ctype in (0, 6):
                row.append(None)
            elif c.ctype == 5:
                row.append({"error": int(c.value)})
            else:
                row.append(c.value)
        rows.append(row)
    out[sh.name] = rows
json.dump(out, sys.stdout)
"""


def read_xls(path: Path) -> dict:
    """Read a legacy .xls workbook via xlrd in .venv-readers; returns {sheet: [[cell, ...], ...]}."""
    res = subprocess.run([str(READERS_PY), "-c", _XLS_SNIPPET, str(path)], capture_output=True, text=True,
                         check=True)
    return json.loads(res.stdout)


def xls_sheet_to_frame(rows: list, header_row: int = 0) -> pd.DataFrame:
    header = rows[header_row]
    data = rows[header_row + 1:]
    cols = [str(h) if h is not None else f"Unnamed: {i}" for i, h in enumerate(header)]
    return pd.DataFrame(data, columns=cols)


def read_sas_layout(sas_path: Path) -> tuple[dict, dict]:
    """Parse a NCHS SAS read-in program: returns ({var: (start, end)}, {var: label}) (1-based inclusive)."""
    txt = Path(sas_path).read_text(encoding="latin-1")
    m = re.search(r"\bINPUT\b(.*?);", txt, flags=re.S)
    if not m:
        raise ValueError(f"No INPUT block in {sas_path}")
    pos = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        mm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+\$?\s*(\d+)(?:\s*-\s*(\d+))?$", line)
        if not mm:
            raise ValueError(f"Unparsed INPUT line in {sas_path}: {line!r}")
        s = int(mm.group(2))
        e = int(mm.group(3)) if mm.group(3) else s
        pos[mm.group(1)] = (s, e)
    labels = {}
    lm = re.search(r"\bLABEL\b(.*?);", txt, flags=re.S)
    if lm:
        for mm in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\"([^\"]*)\"", lm.group(1)):
            labels[mm.group(1)] = mm.group(2).strip()
    return pos, labels


def read_fixed_width(dat_path: Path, layout: dict, variables: list[str]) -> pd.DataFrame:
    """Read selected variables from a fixed-width file as stripped strings ('' -> None)."""
    cols = {v: [] for v in variables}
    spans = [(v, layout[v][0] - 1, layout[v][1]) for v in variables]
    with open(dat_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.rstrip("\r\n")
            for v, s, e in spans:
                val = line[s:e].strip() if len(line) >= s else ""
                cols[v].append(val if val != "" else None)
    return pd.DataFrame(cols)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def fill_to_nan(s: pd.Series, codes: list) -> pd.Series:
    """Set NCHS fill codes (e.g. 8888 'blank but applicable') to NaN on a raw string series, then numeric."""
    raw = s.astype("object")
    mask = raw.isin([str(c) for c in codes])
    out = to_num(raw.where(~mask))
    return out


# ----------------------------------------------------------------------------------------------------------------
# Cleaning helpers
# ----------------------------------------------------------------------------------------------------------------
def raw_to_text(v) -> object:
    """Represent an original cell value as text (keeps the original spelling of numbers)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def make_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns with mixed types are converted to text so that pyarrow can store them."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            types = set(type(x) for x in df[c].dropna())
            if len(types) > 1 or (types and not types <= {str}):
                df[c] = df[c].map(raw_to_text)
    return df


def write_parquet(df: pd.DataFrame, name: str) -> Path:
    path = CLEAN / f"{name}.parquet"
    make_parquet_safe(df).to_parquet(path, index=False, engine="pyarrow")
    return path


def mean_available(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].mean(axis=1, skipna=True)


# ----------------------------------------------------------------------------------------------------------------
# Description
# ----------------------------------------------------------------------------------------------------------------
def _fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if x.is_integer() and abs(x) < 1e12:
        return str(int(x))
    ax = abs(x)
    if ax >= 100:
        s = f"{x:.1f}"
    elif ax >= 1:
        s = f"{x:.2f}"
    elif ax >= 0.01:
        s = f"{x:.3f}"
    else:
        return f"{x:.3g}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def describe_series(s: pd.Series, max_levels: int = 8) -> dict:
    n = len(s)
    nn = int(s.notna().sum())
    d = {"n_nonmissing": nn, "n_missing": n - nn, "pct_missing": (100.0 * (n - nn) / n) if n else float("nan")}
    sv = s.dropna()
    is_bool = pd.api.types.is_bool_dtype(s)
    if pd.api.types.is_numeric_dtype(s) and not is_bool and nn > 0:
        uniq = sv.nunique()
        if uniq <= 6:
            vc = sv.value_counts().sort_index()
            d["summary"] = "; ".join(f"{_fmt(k)}: {v}" for k, v in vc.items())
        else:
            q = sv.quantile([0.05, 0.5, 0.95]).values
            d["summary"] = (f"{_fmt(sv.min())} to {_fmt(sv.max())}; median {_fmt(q[1])} "
                            f"[P5 {_fmt(q[0])}, P95 {_fmt(q[2])}]; mean {_fmt(sv.mean())}")
    elif nn > 0:
        vc = sv.astype(str).value_counts()
        parts = [f"{k[:40]}: {v}" for k, v in vc.head(max_levels).items()]
        if len(vc) > max_levels:
            parts.append(f"... ({len(vc)} distinct)")
        d["summary"] = "; ".join(parts)
    else:
        d["summary"] = "all missing"
    return d


def md_escape(x: str) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def write_dictionary(df: pd.DataFrame, name: str, title: str, sources: list[Path], parser: str,
                     varmeta: dict, intro: list[str] | None = None, notes: list[str] | None = None,
                     extra_sections: list[tuple[str, str]] | None = None) -> Path:
    """Write results/stage0/dictionaries/<name>.md.

    varmeta: {column: {"kind": original|harmonised|derived|qc, "meaning": str, "unit": str, "flag": str}}
    Columns without metadata are listed with kind 'original' and meaning 'see source'.
    """
    lines = [f"# {title}", ""]
    lines.append(f"- Clean table: `results/stage0/clean/{name}.parquet` ({len(df)} rows x {df.shape[1]} columns)")
    lines.append(f"- Parser: `scripts/stage0/{parser}`")
    for p in sources:
        p = Path(p)
        rel = p.relative_to(HUB) if str(p).startswith(str(HUB)) else p
        lines.append(f"- Source: `{rel}` (SHA-256 `{sha256(p)}`)")
    lines.append("- Stage 0 rule: parsing, cleaning and description only; no association/regression/survival/MR "
                 "model was fitted.")
    lines.append("")
    if intro:
        lines += intro + [""]
    if notes:
        lines.append("## Decoding notes and flagged uncertainty")
        lines.append("")
        lines += [f"- {n}" for n in notes]
        lines.append("")
    if extra_sections:
        for head, body in extra_sections:
            lines.append(f"## {head}")
            lines.append("")
            lines.append(body.rstrip())
            lines.append("")
    lines.append("## Variables")
    lines.append("")
    lines.append("Kind: `original` = column as in the source file (name kept); `harmonised` = cleaned analysis "
                 "variable with explicit unit; `derived` = computed here; `qc` = quality-control flag. "
                 "Range shows min to max; median [P5, P95]; mean, or the level counts for categorical/binary "
                 "variables.")
    lines.append("")
    lines.append("| # | Variable | Kind | Meaning | Unit | Non-missing | Missing % | Range / levels | Flag |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(df.columns, 1):
        meta = varmeta.get(c, {})
        d = describe_series(df[c])
        lines.append("| " + " | ".join([
            str(i), f"`{md_escape(c)}`", meta.get("kind", "original"), md_escape(meta.get("meaning", "see source")),
            md_escape(meta.get("unit", "")), str(d["n_nonmissing"]), f"{d['pct_missing']:.1f}",
            md_escape(d["summary"]), md_escape(meta.get("flag", "")),
        ]) + " |")
    lines.append("")
    path = DICT / f"{name}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def md_table(df: pd.DataFrame, floatfmt: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(md_escape(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                vals.append(_fmt(float(v), floatfmt))
            else:
                vals.append(md_escape(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def followup_summary(time: pd.Series, event: pd.Series, unit: str) -> dict:
    t = time.dropna()
    e = event[time.notna()]
    q = t.quantile([0.25, 0.5, 0.75]).values
    return {
        "n": int(len(t)), "events": int(e.sum()), "pct_events": 100.0 * e.mean() if len(e) else float("nan"),
        "fu_median": q[1], "fu_q1": q[0], "fu_q3": q[2], "fu_min": t.min(), "fu_max": t.max(),
        "person_time": t.sum(), "unit": unit,
        "n_time_le0": int((t <= 0).sum()),
    }


def events_table(df: pd.DataFrame, pairs: list[tuple[str, str, str]], unit: str) -> pd.DataFrame:
    """pairs: (label, event_col, time_col). Descriptive only: counts, % and follow-up distribution."""
    rows = []
    for lab, ev, tm in pairs:
        s = followup_summary(df[tm], df[ev], unit)
        rows.append({"outcome": lab, "event column": ev, "time column": tm, "N with time": s["n"],
                     "events": s["events"], "% events": s["pct_events"],
                     f"median FU ({unit})": s["fu_median"], f"IQR ({unit})": f"{_fmt(s['fu_q1'])}-{_fmt(s['fu_q3'])}",
                     f"max FU ({unit})": s["fu_max"], f"person-{unit}": s["person_time"],
                     "time<=0": s["n_time_le0"]})
    return pd.DataFrame(rows)


def weighted_mean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & w.notna() & (w > 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.average(x[m].astype(float), weights=w[m].astype(float)))
