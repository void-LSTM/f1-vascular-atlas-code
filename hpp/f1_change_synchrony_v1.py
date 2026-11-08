#!/usr/bin/env python3
"""P16: fixed aggregate-only observed vascular change synchrony; no RNA."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
from scipy import stats

VERSION='f1_change_synchrony_v1.0 2026-10-07'
BEDS=['pwv','imt','abi_min']
PAIRS=[(0,1),(0,2),(1,2)]
LABELS=['pwv-imt','pwv-abi','imt-abi']
MIN_N=200
A=O=None

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s)
    sys.modules[name]=m; s.loader.exec_module(m); return m

def count(n): return '<20' if n<20 else int(round(float(n)/10)*10)

def prepare(fr):
    fr=fr.reset_index(drop=True).copy(); fr['participant_id']=fr.participant_id.astype(str)
    if fr.duplicated(['participant_id','research_stage']).any(): raise ValueError('nonunique participant/stage')
    s0,s1=A.MAIN_STAGES
    d=fr.loc[fr.research_stage==s0].merge(fr.loc[fr.research_stage==s1],on='participant_id',
                                       suffixes=('_0','_1'),validate='one_to_one')
    t0=O.day(d.visit_date_0); t1=O.day(d.visit_date_1)
    d['years']=(t1-t0).dt.total_seconds()/86400/365.25
    d['season_match']=(t0.dt.quarter==t1.dt.quarter).to_numpy()
    d=d.replace([np.inf,-np.inf],np.nan)
    inv={'both_visits':count(len(d)),'repeat':{},'pair_repeat':{},
         'interval_missing':count(d.years.isna().sum()),'interval_short':count((d.years<=.5).sum()),
         'interval_long':count((d.years>5).sum())}
    for b in BEDS+['art_w']:
        inv['repeat'][b]=count(d[[b+'_0',b+'_1']].notna().all(axis=1).sum()) if b+'_0' in d else '<20'
    for i,j in PAIRS:
        cols=[b+'_'+s for b in [BEDS[i],BEDS[j]] for s in ['0','1']]
        inv['pair_repeat'][LABELS[PAIRS.index((i,j))]]=count(d[cols].notna().all(axis=1).sum())
    need=[b+'_'+s for b in BEDS for s in ['0','1']]
    common=d.dropna(subset=need)
    inv['three_repeat']=count(len(common))
    common=common.loc[(common.years>.5)&(common.years<=5)]
    inv['three_valid_interval']=count(len(common))
    common=common.dropna(subset=['age_0','sex_male_0']).reset_index(drop=True)
    inv['primary_common']=count(len(common))
    return common,inv

def arrays(d,raw=False,conditional=True,pressure=False):
    y=np.column_stack([A.ORIENT[b]*(d[b+('_raw' if raw else '')+'_1']-d[b+('_raw' if raw else '')+'_0'])/d.years for b in BEDS])
    cov=['age_0','sex_male_0','years']+[b+('_raw' if raw else '')+'_0' for b in BEDS] if conditional else []
    X=np.column_stack([np.ones(len(d))]+[d[c].to_numpy(float) for c in cov])
    if pressure:
        X=np.column_stack([X]+[((d[c+'_1']-d[c+'_0'])/d.years).to_numpy(float) for c in ['sbp_sit','dbp_sit','bmi']])
    X=X[:,np.r_[True,X[:,1:].std(axis=0)>1e-12]]
    if not np.isfinite(X).all() or not np.isfinite(y).all() or np.linalg.matrix_rank(X)!=X.shape[1]:
        raise np.linalg.LinAlgError('invalid/rank-deficient change design')
    y=y-X@np.linalg.lstsq(X,y,rcond=None)[0]
    if np.any(y.std(axis=0)<1e-12): raise np.linalg.LinAlgError('degenerate residual change')
    R=np.corrcoef(y,rowvar=False)
    return np.array([R[i,j] for i,j in PAIRS])

def estimate(d,boot,rng,raw=False,conditional=True,pressure=False):
    if len(d)<MIN_N: return {'n':count(len(d)),'status':'NA_n_lt_200'}
    r=arrays(d,raw,conditional,pressure); rr=[]; failed=0
    for _ in range(boot):
        b=d.iloc[rng.integers(0,len(d),len(d))]
        try: rr.append(arrays(b,raw,conditional,pressure))
        except np.linalg.LinAlgError: failed+=1
    if len(rr)<.95*boot: return {'n':count(len(d)),'status':'NA_boot_fraction','B_used':len(rr),'B_failed':failed}
    rr=np.asarray(rr); z=np.arctanh(np.clip(r,-.999999,.999999))
    zs=np.arctanh(np.clip(rr,-.999999,.999999)).std(axis=0,ddof=1)
    if np.any(zs<=0): raise ValueError('degenerate bootstrapSE')
    pv=2*stats.norm.sf(abs(z/zs)); bound=np.arctanh(.2)
    tost=np.maximum(stats.norm.sf((z+bound)/zs),stats.norm.cdf((z-bound)/zs))
    rows={}
    for j,label in enumerate(LABELS):
        rows[label]={'r':r[j],'ci95':np.percentile(rr[:,j],[2.5,97.5]),
                     'ci90':np.percentile(rr[:,j],[5,95]),'fisher_boot_se':zs[j],
                     'p_zero':pv[j],'q_zero3':A.bh(pv)[j],'p_tost_bound02':tost[j]}
    contrasts={}
    for i,j in PAIRS:
        contrasts[LABELS[i]+'_minus_'+LABELS[j]]={'difference':r[i]-r[j],
            'ci95':np.percentile(rr[:,i]-rr[:,j],[2.5,97.5])}
    return {'n':count(len(d)),'status':'estimated','B_used':len(rr),'B_failed':failed,'pairs':rows,
            'all_three_tost_pass':bool(np.all(tost<.05)),'contrasts':contrasts}

def analyse(fr,boot,seed):
    d,inv=prepare(fr); rng=np.random.default_rng(seed)
    result={'inventory':inv,'models':{},'interval_q05_25_50_75_95':A.qtiles(d.years)}
    result['models']['primary']=estimate(d,boot,rng)
    result['models']['unadjusted']=estimate(d,boot,rng,conditional=False)
    rawcols=[b+'_raw_'+s for b in BEDS for s in ['0','1']]
    if all(c in d for c in rawcols) and d[rawcols].notna().all().all():
        result['models']['raw_same_people']=estimate(d,boot,rng,raw=True)
    else: result['models']['raw_same_people']={'n':count(len(d)),'status':'NA_raw_common_incomplete'}
    pressurecols=[c+'_'+s for c in ['sbp_sit','dbp_sit','bmi'] for s in ['0','1']]
    p=d.dropna(subset=pressurecols)
    result['models']['pressure_adiposity']=estimate(p,boot,rng,pressure=True)
    result['models']['season_matched']=estimate(d.loc[d.season_match],boot,rng)
    return result

def clean(obj):
    if isinstance(obj,dict): return {str(k):clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple,np.ndarray)): return [clean(v) for v in obj]
    if isinstance(obj,(float,np.floating)): return A.fnum(obj)
    if isinstance(obj,np.integer): return int(obj)
    return obj

def fmt(x): return 'NA' if x is None else str(A.fnum(float(x)))

def digest(r):
    lines=['F1 P16 | '+VERSION,'Observed conditional change coupling, not biological independence or causality.',
           'Counts rounded10/cells below20; strict stages; interval>0.5 and<=5y; primary gate200.',
           'Primary residualisation: age/sex/interval/all3baseline beds; bootstrap conditional on atlas transform.',
           'Inventory '+json.dumps(r['inventory'],separators=(',',':')),
           'Common interval quantiles '+str(r['interval_q05_25_50_75_95'])]
    # Keep dense inventory split into a separate bounded line when needed.
    if len(lines[4])>220:
        inv=r['inventory']; lines[4]='Inventory '+json.dumps({k:v for k,v in inv.items() if k not in ['repeat','pair_repeat']},separators=(',',':'))
        lines.insert(5,'Repeat '+json.dumps(inv['repeat'],separators=(',',':'))+' paired '+json.dumps(inv['pair_repeat'],separators=(',',':')))
    for name,m in r['models'].items():
        lines.append(f"{name} n={m['n']} {m['status']} B={m.get('B_used','NA')}/{m.get('B_failed','NA')} allTOST={m.get('all_three_tost_pass','NA')}")
        for pair,v in m.get('pairs',{}).items():
            lo,hi=v['ci95']; l90,h90=v['ci90']
            lines.append(f" {pair} r={fmt(v['r'])} [{fmt(lo)},{fmt(hi)}] p={fmt(v['p_zero'])} q={fmt(v['q_zero3'])} TOST={fmt(v['p_tost_bound02'])} CI90=[{fmt(l90)},{fmt(h90)}]")
        for pair,v in m.get('contrasts',{}).items():
            lines.append(f" {pair} d={fmt(v['difference'])} [{fmt(v['ci95'][0])},{fmt(v['ci95'][1])}]")
    lines.append('F1_P16_DONE errors=[]')
    if len(lines)>120 or any(len(x)>220 for x in lines): raise ValueError('digest size contract')
    return '\n'.join(lines)+'\n'

def main():
    global A,O
    ap=argparse.ArgumentParser()
    for c in ['root','out','atlas','omics']: ap.add_argument('--'+c,required=True)
    ap.add_argument('--boot',type=int,default=1000); ap.add_argument('--seed',type=int,default=20261007)
    args=ap.parse_args()
    if args.boot<20 or Path(args.out).exists(): raise SystemExit('invalid boot or existing output')
    A=load(args.atlas,'atlas_p16'); O=load(args.omics,'omics_p16'); O.A=A
    start=time.time(); fr=A.build_frame(args.root,None)[0]
    ids=set(fr.participant_id.astype(str)); r=analyse(fr,args.boot,args.seed)
    r.update(version=VERSION,bootstrap=args.boot,seed_label=f'{args.seed//10000}-{args.seed%10000:04d}',runtime_s=time.time()-start)
    r=clean(r); dg=digest(r); js=json.dumps(r,sort_keys=True,allow_nan=False)
    O.privacy_check(js,ids); O.privacy_check(dg,ids)
    out=Path(args.out); out.mkdir()
    (out/'results.json').write_text(js+'\n'); (out/'digest.txt').write_text(dg)
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['results.json','digest.txt']))
    print('F1_P16_DONE errors=[] runtime=%.1fs'%(time.time()-start),flush=True)
    print('digest lines=%d sq12=%s'%(dg.count('\n'),hashlib.sha256(re.sub(' +',' ',dg).encode()).hexdigest()[:12]))

if __name__=='__main__': main()
