#!/usr/bin/env python3
"""P21: fixed prospective continuous-task counts, no fitted associations/prediction."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TARGETS = ['sbp_sit', 'dbp_sit', 'ldl', 'glucose', 'hba1c', 'egfr', 'pwv', 'imt', 'abi_min']
BEDS = ['pwv', 'imt', 'art_w', 'abi_min']
BLOCKS = {b: [b] for b in BEDS}
BLOCKS.update(three=['pwv', 'imt', 'abi_min'], four=BEDS)
SOURCES = {'bp': ('blood_pressure/blood_pressure.parquet', ['sitting_blood_pressure_systolic', 'sitting_blood_pressure_diastolic']),
           'bmi': ('anthropometrics/anthropometrics.parquet', ['bmi']),
           'vascular': ('vascular_health/vascular_health.parquet', ['from_r_thigh_to_r_ankle_pwv', 'from_l_thigh_to_l_ankle_pwv', 'r_abi', 'l_abi']),
           'carotid': ('carotid_ultrasound/carotid_ultrasound.parquet', ['imt_left', 'imt_right']),
           'retina': ('fundus/microvasculature.parquet', ['artery_average_width_left', 'artery_average_width_right'])}
MODALITY = dict(sbp_sit='bp', dbp_sit='bp', pwv='vascular', abi_min='vascular', imt='carotid', art_w='retina')
LABS = dict(bt__ldl_cholesterol_float_value='ldl', bt__glucose_float_value='glucose',
            bt__hba1c_float_value='hba1c', bt__creatinine_float_value='creat')
KEYS = ['participant_id', 'research_stage']
A = None

def count(n):
    return '<20' if n < 20 else int(round(float(n) / 10) * 10)

def day(s):
    return pd.to_datetime(s, errors='coerce', utc=True).dt.tz_convert('Asia/Jerusalem').dt.tz_localize(None).dt.normalize().dt.as_unit('ns')

def lab_table(raw):
    """Exclude a whole ambiguous date; never choose the most favourable duplicate."""
    d = raw.rename(columns=LABS).copy()
    d['lab_date'] = day(d.collection_date)
    for c in LABS.values():
        d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan) if c in d else np.nan
    d = d.dropna(subset=['lab_date']).dropna(subset=list(LABS.values()), how='all')
    grp = d.groupby(['participant_id', 'lab_date'], sort=False)
    ambiguous = grp[list(LABS.values())].nunique(dropna=True).gt(1).any(axis=1)
    result = grp[list(LABS.values())].first().loc[~ambiguous].reset_index()
    return result, count(int(ambiguous.sum()))

def match_labs(visits, labs):
    left = visits[KEYS + ['visit_date']].dropna(subset=['visit_date']).copy()
    left['visit_date'] = day(left.visit_date)
    out = []
    for stage, direction in zip(A.MAIN_STAGES, ['backward', 'nearest']):
        l = left[left.research_stage == stage].sort_values('visit_date')
        if l.empty or labs.empty:
            continue
        m = pd.merge_asof(l, labs.sort_values('lab_date'), left_on='visit_date', right_on='lab_date',
                          by='participant_id', direction=direction, tolerance=pd.Timedelta(days=180))
        # Equidistant distinct dates at follow-up are ambiguous rather than arbitrary nearest matches.
        if direction == 'nearest':
            back = pd.merge_asof(l, labs.sort_values('lab_date'), left_on='visit_date', right_on='lab_date', by='participant_id', direction='backward', tolerance=pd.Timedelta(days=180))
            forward = pd.merge_asof(l, labs.sort_values('lab_date'), left_on='visit_date', right_on='lab_date', by='participant_id', direction='forward', tolerance=pd.Timedelta(days=180))
            tied = ((back.visit_date-back.lab_date).to_numpy() == (forward.lab_date-forward.visit_date).to_numpy()) & (back.lab_date.to_numpy() != forward.lab_date.to_numpy())
            m.loc[tied, list(LABS.values()) + ['lab_date']] = np.nan
        out.append(m.drop(columns='visit_date'))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=KEYS + list(LABS.values()) + ['lab_date'])

def acquisition(raw, root, name):
    path, values = SOURCES[name]
    d = raw.copy()
    usable = pd.concat([pd.to_numeric(d[c], errors='coerce').replace([np.inf,-np.inf],np.nan) for c in values if c in d], axis=1).notna().any(axis=1)
    d = d[usable].copy()
    if name == 'retina' and not any(c in d for c in ['collection_timestamp', 'collection_date']):
        # Processed calibre must match photograph metadata on the full source array key.
        photo = A.rd(root, 'fundus/fundus.parquet')
        keys = KEYS + ['array_index']
        if photo is not None and all(k in d and k in photo for k in keys) and not d.duplicated(keys).any() and not photo.duplicated(keys).any():
            cols = [c for c in ['collection_timestamp','collection_date'] if c in photo]
            d = d.merge(photo[keys + cols], on=keys, how='left', validate='one_to_one')
    ts = day(d.collection_timestamp) if 'collection_timestamp' in d else pd.Series(pd.NaT, index=d.index, dtype='datetime64[ns]')
    if 'collection_date' in d:
        ts = ts.fillna(day(d.collection_date))
    d['_ts'] = ts
    g = d.groupby(KEYS)
    t = g['_ts'].agg(['min', 'max', 'count', 'size']).reset_index()
    # Every row contributing a value must have known timing, not just the first available row.
    t.loc[t['count'] != t['size'], ['min', 'max']] = pd.NaT
    qc = {'value_rows': count(len(d)), 'missing_timing_rows': count(ts.isna().sum()),
          'visits_with_all_timing': count(t['min'].notna().sum())}
    return t[KEYS + ['min', 'max']].rename(columns={'min':name+'_lo', 'max':name+'_hi'}), qc

def prepare(fr, dates):
    if fr.duplicated(KEYS).any():
        raise ValueError('duplicate participant stage')
    f = fr.replace([np.inf,-np.inf],np.nan).copy()
    for d in dates:
        f = f.merge(d, on=KEYS, how='left', validate='one_to_one')
    s0,s1 = A.MAIN_STAGES
    p = f[f.research_stage == s0].merge(f[f.research_stage == s1], on='participant_id', suffixes=('_0','_1'), validate='one_to_one')
    p['start'] = day(p.visit_date_0) + pd.Timedelta(days=180)
    p['lower'] = day(p.visit_date_0) - pd.Timedelta(days=180)
    return p

def baseline_time(p, name):
    return p[name+'_lo_0'].ge(p.lower) & p[name+'_hi_0'].le(p.start)

def audit(p, relations):
    results = {}
    core = p[['age_0','sex_male_0','bmi_0','sbp_sit_0','dbp_sit_0']].notna().all(axis=1) & baseline_time(p,'bp') & baseline_time(p,'bmi')
    for target in TARGETS:
        values = p[[target+'_0',target+'_1']].notna().all(axis=1)
        if target in MODALITY:
            modality = MODALITY[target]
            lo,hi = p[modality+'_lo_1'],p[modality+'_hi_1']
            base = baseline_time(p,modality)
        else:
            lo = hi = p.lab_date_1
            base = p.lab_date_0.ge(p.lower) & p.lab_date_0.le(day(p.visit_date_0))
        chronology = lo.gt(p.start + pd.Timedelta(days=.5*365.25)) & hi.le(p.start + pd.Timedelta(days=5*365.25))
        eligible = values & base & chronology & core
        row = {'paired_values':count(values.sum()), 'strict_clinical_target':count(eligible.sum()), 'blocks':{}}
        for block,beds in BLOCKS.items():
            use = eligible.copy()
            for bed in beds:
                use &= p[bed+'_0'].notna() & baseline_time(p,MODALITY[bed])
            d = p[use]
            info = {'n':count(len(d)), 'status':'NA_n_lt_200'}
            if relations is None:
                info['status'] = 'NA_relationship_metadata'
            else:
                groups = np.array([relations.get(str(x),str(x)) for x in d.participant_id])
                ng = len(np.unique(groups)); info['groups'] = count(ng)
                if len(d)>=200 and ng>=100:
                    rng = np.random.default_rng(20261014)
                    labels = np.unique(groups); remap = dict(zip(labels,rng.permutation(len(labels))))
                    g = np.array([remap[x] for x in groups]); folds = list(GroupKFold(5).split(d,groups=g))
                    nt = [len(train) for train,_ in folds]; gt = [len(np.unique(g[train])) for train,_ in folds]
                    varies = all(np.nanstd(d[target+'_1'].to_numpy(float)[train])>1e-12 for train,_ in folds)
                    info.update(min_training_n=count(min(nt)),min_training_groups=count(min(gt)),status='FEASIBLE_COUNTS_ONLY' if min(nt)>=100 and min(gt)>=50 and varies else 'NA_training_gate')
                elif ng<100:
                    info['status'] = 'NA_groups_lt_100'
            row['blocks'][block] = info
        results[target] = row
    return results

def digest(r):
    lines=['F1_P21 clinical task feasibility: counts only; no prediction fit', 'Paired visits n='+str(r['paired_visits']), 'Ambiguous lab person-dates='+str(r['lab_conflicting_dates'])]
    for name,info in r['timing'].items():
        lines.append('TIME '+name+' '+' '.join(k+'='+str(v) for k,v in info.items()))
    for target,row in r['tasks'].items():
        lines.append('TASK '+target+' values='+str(row['paired_values'])+' strict='+str(row['strict_clinical_target']))
        for name,info in row['blocks'].items():
            lines.append('BLOCK '+target+'/'+name+' '+' '.join(k+'='+str(v) for k,v in info.items()))
    lines.append('F1_P21_DONE errors=[]')
    if len(lines)>120 or any(len(x)>220 for x in lines): raise ValueError('digest size')
    return '\n'.join(lines)+'\n'

def main():
    global A
    ap=argparse.ArgumentParser()
    for name in ['root','out','atlas']:ap.add_argument('--'+name,required=True)
    args=ap.parse_args(); output=Path(args.out)
    if output.exists():raise SystemExit('existing output')
    path=Path(args.atlas)
    if hashlib.sha256(path.read_bytes()).hexdigest()!='62344144b83db3f2f7e6492bd05fd7616d48a442bba88134aa1e0360110fa253':
        raise SystemExit('immutable atlas dependency mismatch')
    spec=importlib.util.spec_from_file_location('atlas_p21',path);A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
    # Explicitly disable both full-cohort acquisition transforms before frame construction.
    A.CAL_ADJUST=False; A.DAY_ADJUST=False; A.SESSION_ADJUST=False
    start=time.time(); fr=A.build_frame(args.root,None)[0]
    ids=set(fr.participant_id.astype(str))
    raw=A.rd(args.root,'blood_tests/blood_tests.parquet'); labs,ambiguous=lab_table(raw)
    selected=match_labs(fr,labs)
    fr=fr.drop(columns=list(LABS.values())+['egfr'],errors='ignore').merge(selected,on=KEYS,how='left',validate='one_to_one')
    # Atlas creatinine conversion is retained for availability only; no disease threshold is inferred.
    cr=fr.creat.copy(); cr=cr/88.4 if cr.median(skipna=True)>20 else cr
    female=fr.sex_male.eq(0); ratio=cr/np.where(female,.7,.9)
    fr['egfr']=142*np.minimum(ratio,1)**np.where(female,-.241,-.302)*np.maximum(ratio,1)**-1.2*.9938**fr.age*np.where(female,1.012,1)
    dates=[]; timing={}
    for name,(source,_) in SOURCES.items():
        d,qc=acquisition(A.rd(args.root,source),args.root,name);dates.append(d);timing[name]=qc
    p=prepare(fr,dates); rel=A.relation_components(args.root)
    result={'paired_visits':count(len(p)),'lab_conflicting_dates':ambiguous,'timing':timing,'tasks':audit(p,rel),'inference_performed':False}
    dg=digest(result); js=json.dumps(result,sort_keys=True,allow_nan=False)+'\n'
    for text in [dg,js]:
        A.privacy_check(text,ids)
        if re.search(r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b',text):raise SystemExit('fine date privacy abort')
    output.mkdir();(output/'digest.txt').write_text(dg);(output/'results.json').write_text(js)
    (output/'SHA256SUMS').write_text(''.join(hashlib.sha256((output/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['digest.txt','results.json']))
    print('F1_P21_DONE errors=[] runtime=%.1fs'%(time.time()-start),flush=True)
    print('digest lines=%d sq12=%s'%(len(dg.splitlines()),hashlib.sha256(re.sub(' +',' ',dg).encode()).hexdigest()[:12]))

if __name__=='__main__': main()
