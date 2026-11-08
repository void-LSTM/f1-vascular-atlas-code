#!/usr/bin/env python3
"""P15: aggregate-only diagnostic for P14 prospective denominators. No new fits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import importlib.util

VERSION = 'f1_longitudinal_feasibility_v1.0 2026-10-06'


def funnel(p, fr, tech, proxies):
    """Keep P14 joins, dates and complete-case specification exactly; diagnose losses."""
    if fr.duplicated(['participant_id', 'research_stage']).any():
        raise ValueError('nonunique participant stage frame')
    baseline = fr[fr.research_stage == p.STAGES[0]].copy()
    followup = fr[fr.research_stage == p.STAGES[1]].copy()
    joined = baseline.merge(followup, on='participant_id', suffixes=('_0', '_1'), validate='one_to_one')
    dt = (p.O.day(joined.visit_date_1)-p.O.day(joined.visit_date_0)).dt.total_seconds()/86400/365.25
    dt.index = joined.index
    joined['interval_years'] = dt
    joined['nk'] = joined.nk_0
    finite = joined.replace([np.inf, -np.inf], np.nan)
    c = p.count
    out = {'visits': [c(len(baseline)), c(len(followup))],
           'baseline_NK': c(baseline.nk.notna().sum()),
           'baseline_NK_followup_visit': c(joined.nk_0.notna().sum()),
           'both_visits': c(len(joined)),
           'interval': {'missing': c(dt.isna().sum()), 'nonpositive': c((dt<=0).sum()),
                        'above5': c((dt>5).sum()), 'valid': c(((dt>0)&(dt<=5)).sum())},
           'beds': {}}
    for bed in p.REPEAT_BEDS:
        y0, y1 = 'o_'+bed+'_0', 'o_'+bed+'_1'
        mask = pd.Series(True, index=joined.index)
        row = {}
        blocks = [('two_beds', [y0, y1]), ('baseline_NK', ['nk']),
                  ('interval', ['interval_years']), ('baseline_cov', [x+'_0' for x in p.COVS]),
                  ('RNA_technical', [x+'_0' for x in tech]), ('composition', [x+'_0' for x in proxies])]
        for name, cols in blocks:
            ok = finite[cols].notna().all(axis=1)
            if name == 'interval': ok &= (dt>0)&(dt<=5)
            mask &= ok
            row[name] = c(mask.sum())
        cov = [x+'_0' for x in p.COVS+tech+proxies]+[y0, 'interval_years']
        need = list(dict.fromkeys(['nk', y1]+cov))
        direct = finite.loc[(dt>0)&(dt<=5)].dropna(subset=need)
        if len(direct) != int(mask.sum()): raise ValueError('funnel differs from P14 complete cases')
        row['complete'] = c(len(direct))
        row['change_cov_complete'] = c(direct[['sbp_sit_1', 'bmi_1']].notna().all(axis=1).sum())
        row['future_RNA_available'] = c(direct.nk_1.notna().sum())
        # Independently report baseline-NK overlap before requiring baseline bed/dates/covariates.
        row['baseline_NK_later_bed'] = c(finite[['nk', y1]].notna().all(axis=1).sum())
        out['beds'][bed] = row
    return out


def digest(result):
    f = result['funnel']
    lines = [f'F1 P15 | {VERSION}', 'Feasibility only; P14 definitions unchanged; no new hypothesis fits.',
             'Counts rounded10; cells below20 suppressed. Baseline NK does not require future RNA.',
             f"Visits={f['visits']} both_visits={f['both_visits']}",
             f"BaselineNK={f['baseline_NK']} with_followup_visit={f['baseline_NK_followup_visit']}",
             'Interval '+json.dumps(f['interval'], separators=(',', ':'))]
    for bed, row in f['beds'].items():
        lines += [f'{bed} funnel '+json.dumps({k: row[k] for k in
                  ['two_beds','baseline_NK','interval','baseline_cov','RNA_technical','composition']}, separators=(',', ':')),
                  f'{bed} detail '+json.dumps({k: row[k] for k in
                  ['complete','change_cov_complete','future_RNA_available','baseline_NK_later_bed']}, separators=(',', ':'))]
    lines += ['RNA strict inventory '+json.dumps(result['inventory']['rna'], separators=(',', ':')),
              'P14 prior_n '+json.dumps(result['prior_n'], separators=(',', ':')),
              'F1_P15_DONE errors=[]']
    if len(lines)>120 or any(len(s)>220 for s in lines): raise ValueError('digest size contract')
    return '\n'.join(lines)+'\n'


def main():
    ap = argparse.ArgumentParser()
    for arg in ['root','out','p14','atlas','omics','nkhr','reference','prior']: ap.add_argument('--'+arg, required=True)
    args = ap.parse_args()
    if Path(args.out).exists(): raise SystemExit('Refusing existing output')
    start = time.time()
    spec = importlib.util.spec_from_file_location('p14_p15', args.p14)
    p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
    p.setup(args.atlas, args.omics, args.nkhr)
    fr = p.A.build_frame(args.root, None)[0].reset_index(drop=True)
    fr['participant_id'] = fr.participant_id.astype(str)
    rna, tech, proxies, _ = p.load_rna(args.root, args.reference)
    ids = set(fr.participant_id)|set(rna.participant_id)
    joined = fr.merge(rna[['participant_id','research_stage','nk']+tech+proxies],
                      on=['participant_id','research_stage'], how='left', validate='many_to_one')
    prior = json.loads(Path(args.prior).read_text())
    result = {'version': VERSION, 'funnel': funnel(p, joined, tech, proxies),
              'inventory': p.inventory(args.root, fr, rna),
              'prior_n': {b: prior['prospective'][b]['n'] for b in p.REPEAT_BEDS},
              'runtime_s': time.time()-start}
    dg = digest(result); text = json.dumps(p.clean(result), sort_keys=True, allow_nan=False)
    p.O.privacy_check(text, ids); p.O.privacy_check(dg, ids)
    out = Path(args.out); out.mkdir()
    (out/'results.json').write_text(text+'\n'); (out/'digest.txt').write_text(dg)
    (out/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256((out/n).read_bytes()).hexdigest()}  {n}\n'
                                       for n in ['results.json','digest.txt']))
    print('F1_P15_DONE errors=[] runtime=%.1fs'%(time.time()-start), flush=True)
    print('digest lines=%d sq12=%s'%(dg.count('\n'), hashlib.sha256(re.sub(' +',' ',dg).encode()).hexdigest()[:12]))

if __name__ == '__main__': main()
