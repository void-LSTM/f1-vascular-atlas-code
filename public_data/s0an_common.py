"""Shared infrastructure for the gated F1 Stage 0 analysis drivers (`analysis_*.py`).

Gate
----
Every driver refuses to touch real data unless `--real` is passed. `--simulate` builds a fake table with the same
columns as the clean public table and runs the whole pipeline end to end (wiring test); in that mode the real
parquet is never opened: only its schema (column names and types) is read, to check the wiring.

Outputs
-------
real mode     : results/stage0/analysis/<task>/            (report.md, *.csv, provenance.json)
simulate mode : results/stage0/validation/drivers/<task>/  (same files + planted-vs-estimated check)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER = Path(os.environ.get("F1_WORKDIR", HERE.parent))
CLEAN = PAPER / "results" / "stage0" / "clean"
ANALYSIS = PAPER / "results" / "stage0" / "analysis"
SIMOUT = PAPER / "results" / "stage0" / "validation" / "drivers"

_REAL_TOKEN = object()        # only `gate` hands this out, and only with --real


class GateError(SystemExit):
    pass


def parse_args(description: str, extra=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=description)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--real", action="store_true",
                   help="run on the real clean table")
    g.add_argument("--simulate", action="store_true",
                   help="build a fake table with the same columns and run end to end (wiring test)")
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap replicates where applicable")
    if extra:
        extra(ap)
    return ap.parse_args()


def gate(args, task: str):
    """Return (mode, outdir, token). Exits with status 2 unless --real or --simulate was given."""
    if args.simulate:
        out = SIMOUT / task
        out.mkdir(parents=True, exist_ok=True)
        return "simulate", out, None
    if args.real:
        out = ANALYSIS / task
        out.mkdir(parents=True, exist_ok=True)
        return "real", out, _REAL_TOKEN
    msg = (f"[{task}] REFUSED: this driver fits models on the real public data. Run with --simulate "
           "for a wiring test, or with --real.")
    print(msg, file=sys.stderr)
    raise GateError(2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def real_schema(table: str) -> dict:
    """Column name -> pyarrow type string of the clean parquet (schema only; no data are read)."""
    import pyarrow.parquet as pq
    sch = pq.read_schema(CLEAN / f"{table}.parquet")
    return {f.name: str(f.type) for f in sch}


def _kind(t: str) -> str:
    t = t.lower()
    if t.startswith(("int", "uint", "double", "float", "bool")) or t in ("int64", "float64"):
        return "numeric"
    if "timestamp" in t or "datetime" in t:
        return "datetime"
    return "text"


def check_columns(sim: pd.DataFrame, table: str, used: list[str]) -> pd.DataFrame:
    """Wiring check: every used column exists in the real schema and in the simulated table, same kind."""
    sch = real_schema(table)
    rows = []
    for c in used:
        rk = _kind(sch[c]) if c in sch else "MISSING"
        sk = _kind(str(sim[c].dtype)) if c in sim.columns else "MISSING"
        rows.append({"column": c, "real_type": sch.get(c, "MISSING"), "sim_dtype": str(sim[c].dtype)
                     if c in sim.columns else "MISSING", "ok": bool(rk != "MISSING" and rk == sk)})
    df = pd.DataFrame(rows)
    if not df["ok"].all():
        bad = df.loc[~df["ok"], "column"].tolist()
        raise RuntimeError(f"wiring check failed for {table}: {bad}")
    return df


def load_real(table: str, token, columns: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Load the real clean table. Only callable with the token returned by gate(--real)."""
    if token is not _REAL_TOKEN:
        raise PermissionError("real data may only be loaded after gate(--real)")
    path = CLEAN / f"{table}.parquet"
    df = pd.read_parquet(path, columns=columns)
    return df, {"table": table, "path": str(path), "sha256": sha256(path), "rows": len(df)}


def fmt(v, nd=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        av = abs(v)
        if av != 0 and (av < 1e-3 or av >= 1e5):
            return f"{v:.3e}"
        return f"{v:.{nd}g}"
    return str(v)


def md_table(df: pd.DataFrame, nd: int = 4, max_rows: int | None = None) -> str:
    if df is None or len(df) == 0:
        return "_(no rows)_"
    d = df if max_rows is None else df.head(max_rows)
    cols = list(d.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(fmt(r[c], nd).replace("|", "/") for c in cols) + " |")
    if max_rows is not None and len(df) > max_rows:
        out.append(f"\n_({len(df) - max_rows} more rows in the CSV)_")
    return "\n".join(out)


class Report:
    """Collects markdown sections and tables; writes report.md, one CSV per table and provenance.json."""

    def __init__(self, task: str, title: str, mode: str, args):
        self.task, self.title, self.mode, self.args = task, title, mode, args
        self.parts: list[str] = []
        self.tables: dict[str, pd.DataFrame] = {}
        self.inputs: list[dict] = []
        self.t0 = _dt.datetime.now()

    def h(self, text: str, level: int = 2):
        self.parts += ["", "#" * level + " " + text, ""]

    def p(self, text: str):
        self.parts.append(text)

    def bullets(self, items):
        self.parts += [f"- {x}" for x in items]

    def table(self, name: str, df: pd.DataFrame, caption: str | None = None, nd: int = 4,
              max_rows: int | None = 60):
        self.tables[name] = df.copy()
        if caption:
            self.parts += ["", f"**{caption}** (`{name}.csv`)", ""]
        self.parts.append(md_table(df, nd, max_rows))
        self.parts.append("")

    def write(self, outdir: Path, driver_file: str):
        for name, df in self.tables.items():
            df.to_csv(outdir / f"{name}.csv", index=False)
        head = [f"# {self.title}", "",
                f"- Task `{self.task}`; mode **{self.mode.upper()}**; generated {self.t0:%Y-%m-%d %H:%M}; "
                f"driver `scripts/stage0/{Path(driver_file).name}`.",
                ("- **SIMULATED DATA** (fake table with the same columns as the clean public table; planted effects). "
                 "These numbers say nothing about the real data." if self.mode == "simulate" else
                 "- Real clean public table.")]
        (outdir / "report.md").write_text("\n".join(head + self.parts) + "\n", encoding="utf-8")
        prov = {"task": self.task, "mode": self.mode, "argv": sys.argv, "started": self.t0.isoformat(),
                "finished": _dt.datetime.now().isoformat(), "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "driver_sha256": sha256(Path(driver_file)),
                "lib_sha256": {n: sha256(HERE / n) for n in ("lib_survival.py", "lib_mr.py", "s0an_common.py")
                               if (HERE / n).exists()},
                "inputs": self.inputs, "seed": getattr(self.args, "seed", None)}
        (outdir / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")


def planted_rows(checks: list[tuple]) -> pd.DataFrame:
    """checks: (label, planted, estimate, se). Flags |est - planted| <= 3 SE (wiring check, not validation)."""
    rows = []
    for lab, tru, est, se in checks:
        ok = bool(np.isfinite(est) and np.isfinite(se) and abs(est - tru) <= 3 * se)
        rows.append({"quantity": lab, "planted": tru, "estimate": est, "se": se,
                     "z": (est - tru) / se if se and np.isfinite(se) and se > 0 else np.nan,
                     "within_3se": ok})
    return pd.DataFrame(rows)


def finish(rep: Report, outdir: Path, driver_file: str, planted: pd.DataFrame | None = None):
    if planted is not None:
        rep.h("Planted-value check (simulate mode only)")
        rep.p("Wiring check: each estimate should be within about 3 SE of the value planted in the simulated table "
              "(approximate where the fitted model is not the data-generating model, e.g. non-collapsible Cox "
              "models with an omitted covariate).")
        rep.table("planted_check", planted, "Planted vs estimated")
    rep.write(outdir, driver_file)
    status = "" if planted is None else f"; planted within 3 SE: {int(planted['within_3se'].sum())}/{len(planted)}"
    print(f"[{rep.task}] {rep.mode}: wrote {outdir}{status}")
