#!/usr/bin/env python3
"""derive_m_crossbed_profile_nhanes.py - F1 Stage 0 derived contrast (exploratory; public NHANES only).

Does the heart-vs-cerebrovascular death profile differ between beds? Compares the Lunn-McNeil log ratio
log(HR heart / HR cerebrovascular) of a binary ABI marker (NHANES 1999-2004, task m_outcome_contrasts, M2) with that
of binary retinal signs (NHANES 2005-2008, task m_nhanes_retina_mortality, M2). The two samples are different NHANES
cycles (independent people), the death causes are the same UCOD_LEADING groups (heart 1, cerebrovascular 5) and the
same 2019 linked mortality, so the difference of log ratios has variance se1^2 + se2^2. Reference distribution:
t with the smaller design df (conservative). Reads only the two CSVs; writes one CSV and one md.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

import pandas as pd
from scipy import stats

PAPER = Path(os.environ.get("F1_WORKDIR", Path(__file__).resolve().parents[1]))
A = PAPER / "results/stage0/analysis"
OUT = A / "m_crossbed_profile_nhanes"
F_ABI = A / "m_outcome_contrasts/contrasts_primary.csv"
F_RET = A / "m_nhanes_retina_mortality/heart_vs_cereb_contrast.csv"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    abi = pd.read_csv(F_ABI)
    ret = pd.read_csv(F_RET)
    a = abi.loc[abi["id"] == "C1-NH9904-abi_lt09-M2"].iloc[0]
    a_line = int(abi.index[abi["id"] == "C1-NH9904-abi_lt09-M2"][0]) + 2
    rows = []
    for term, label in [("avn", "AV nicking (definite)"), ("ret_any", "Any retinopathy")]:
        m = (ret["model"] == "M2") & (ret["term"] == term) & (ret["exposure"] == label)
        r = ret.loc[m].iloc[0]
        r_line = int(ret.index[m][0]) + 2
        d = float(a["log_ratio"]) - float(r["log_ratio"])
        se = math.sqrt(float(a["se"]) ** 2 + float(r["se_design"]) ** 2)
        df = int(min(float(a["design_df"]), float(r["design_df"])))
        t = d / se
        p = 2 * stats.t.sf(abs(t), df)
        q = stats.t.ppf(0.975, df)
        rows.append(dict(
            bed_A="ABI < 0.9 (vs 1.00-1.40), NHANES 1999-2004", bed_B=f"{label}, NHANES 2005-2008",
            ratio_A=math.exp(float(a["log_ratio"])), ratio_B=math.exp(float(r["log_ratio"])),
            ratio_of_ratios=math.exp(d), lower=math.exp(d - q * se), upper=math.exp(d + q * se), p=p, df=df,
            n_A=int(a["n"]), heart_cereb_events_A=f"{int(a['events_A'])}/{int(a['events_B'])}",
            n_B=int(r["n_unweighted"]), heart_cereb_events_exposed_B=f"{int(r['heart_events_exposed'])}/{int(r['cereb_events_exposed'])}",
            source_A=f"{F_ABI.relative_to(PAPER)}:{a_line}", source_B=f"{F_RET.relative_to(PAPER)}:{r_line}"))
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT / "crossbed_profile.csv", index=False)
    lines = [
        "# Derived: heart-vs-cerebrovascular death profile, ABI vs retinal signs (NHANES; exploratory)",
        "",
        "Ratio of ratios = [HR heart / HR cerebrovascular for ABI < 0.9] / [same for the retinal sign]; independent",
        "NHANES cycles, so var = se_A^2 + se_B^2; t(df = smaller design df). Not same-person; different exam cycles.",
        "",
        "| Bed A | Bed B | ratio A | ratio B | ratio of ratios [95% CI] | p | df | source A | source B |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['bed_A']} | {r['bed_B']} | {r['ratio_A']:.3g} | {r['ratio_B']:.3g} | "
                     f"{r['ratio_of_ratios']:.3g} [{r['lower']:.3g}, {r['upper']:.3g}] | {r['p']:.2g} | {r['df']} | "
                     f"`{r['source_A']}` | `{r['source_B']}` |")
    lines += ["", f"Inputs: `{F_ABI.relative_to(PAPER)}` sha256 {sha(F_ABI)[:16]}; "
                  f"`{F_RET.relative_to(PAPER)}` sha256 {sha(F_RET)[:16]}.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
