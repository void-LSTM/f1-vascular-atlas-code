#!/usr/bin/env python3
"""P17: relationship-cluster P16 sensitivity, pairwise cohorts and omics feasibility."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
from scipy import stats
import f1_change_synchrony_v1 as P

MIN_GROUPS=100
VERSION='f1_longitudinal_extension_v1.0 2026-10-07'

def paired_frame(fr):
    s0,s1=P.A.MAIN_STAGES
    if fr.duplicated(['participant_id','research_stage']).any():
        raise ValueError('duplicate visit keys')
    d=fr.loc[fr.research_stage==s0].merge(fr.loc[fr.research_stage==s1],on='participant_id',suffixes=('_0','_1'),validate='one_to_one')
    d['years']=(P.O.day(d.visit_date_1)-P.O.day(d.visit_date_0)).dt.total_seconds()/86400/365.25
    return d.replace([np.inf,-np.inf],np.nan)

def group_info(d,relations):
    if relations is None: return None,{'n':P.count(len(d)),'status':'NA_relationship_metadata'}
    g=np.array([relations.get(str(p),str(p)) for p in d.participant_id])
    _,inv,counts=np.unique(g,return_inverse=True,return_counts=True)
    info={'n':P.count(len(d)),'groups':P.count(len(counts)),
          'multi_groups':P.count((counts>1).sum()),'people_in_multi':P.count(counts[counts>1].sum())}
    return [np.flatnonzero(inv==k) for k in range(len(counts))],info

def group_draw(blocks,rng):
    return np.concatenate([blocks[k] for k in rng.integers(0,len(blocks),len(blocks))])

def pair_arrays(d,beds):
    y=np.column_stack([P.A.ORIENT[b]*(d[b+'_1']-d[b+'_0'])/d.years for b in beds])
    cov=['age_0','sex_male_0','years']+[b+'_0' for b in beds]
    X=np.column_stack([np.ones(len(d))]+[d[c].to_numpy(float) for c in cov])
    X=X[:,np.r_[True,X[:,1:].std(axis=0)>1e-12]]
    if not np.isfinite(X).all() or not np.isfinite(y).all() or np.linalg.matrix_rank(X)!=X.shape[1]:
        raise np.linalg.LinAlgError('invalid change design')
    y-=X@np.linalg.lstsq(X,y,rcond=None)[0]
    if np.any(y.std(axis=0)<1e-12): raise np.linalg.LinAlgError('degenerate changes')
    return np.array([np.corrcoef(y,rowvar=False)[0,1]])

def estimate(d,relations,boot,rng,fn,labels):
    blocks,info=group_info(d,relations)
    if blocks is None: return info
    if len(d)<200: return {**info,'status':'NA_n_lt_200'}
    if len(blocks)<MIN_GROUPS: return {**info,'status':'NA_groups_lt_100'}
    try: r=fn(d)
    except np.linalg.LinAlgError: return {**info,'status':'NA_design'}
    draws=[]; failed=0
    for _ in range(boot):
        try: draws.append(fn(d.iloc[group_draw(blocks,rng)]))
        except np.linalg.LinAlgError: failed+=1
    if len(draws)<.95*boot: return {**info,'status':'NA_boot_fraction','B_used':len(draws),'B_failed':failed}
    rr=np.asarray(draws); z=np.arctanh(np.clip(r,-.999999,.999999))
    se=np.arctanh(np.clip(rr,-.999999,.999999)).std(axis=0,ddof=1)
    if np.any(se<=0): return {**info,'status':'NA_boot_SE'}
    p=2*stats.norm.sf(abs(z/se)); bound=np.arctanh(.2)
    tost=np.maximum(stats.norm.sf((z+bound)/se),stats.norm.cdf((z-bound)/se))
    rows={}
    for j,label in enumerate(labels):
        rows[label]={'r':r[j],'ci95':np.percentile(rr[:,j],[2.5,97.5]),'ci90':np.percentile(rr[:,j],[5,95]),
                     'p_zero':p[j],'p_tost':tost[j],'q_zero':P.A.bh(p)[j] if len(labels)==3 else None}
    return {**info,'status':'estimated','B_used':len(draws),'B_failed':failed,'pairs':rows,'allTOST':bool(np.all(tost<.05))}

def omics_inventory(root,fr):
    d=paired_frame(fr); valid=d[(d.years>.5)&(d.years<=5)]
    out={}; stages=P.A.MAIN_STAGES
    for name,path in [('NMR','nightingale_metabolomics/nightingale_metabolomics.parquet'),('lipid','serum_lipidomics/serum_lipidomics.parquet')]:
        t=P.A.rd(root,path)
        if t is None: out[name]={'status':'table_unavailable'}; continue
        t=t.copy(); t['participant_id']=t.participant_id.astype(str)
        meta={'participant_id','research_stage','cohort','array_index','index','collection_date','collection_timestamp','timezone'}
        cols=[c for c in t if c not in meta and not str(c).startswith('synthetic_pad') and pd.api.types.is_numeric_dtype(t[c]) and not pd.api.types.is_bool_dtype(t[c])]
        t=t.loc[t[cols].notna().any(axis=1)] if cols else t.iloc[:0]
        ids=[set(t.loc[t.research_stage==s,'participant_id']) for s in stages]
        row={'base':P.count(len(ids[0])),'later':P.count(len(ids[1])),'repeat':P.count(len(ids[0]&ids[1])),'beds':{}}
        for b in P.BEDS:
            x=valid.loc[valid.participant_id.astype(str).isin(ids[0])]
            later=x.loc[x[b+'_1'].notna()]
            both=later.dropna(subset=[b+'_0'])
            basic=both.dropna(subset=['age_0','sex_male_0'])
            clinical=basic.dropna(subset=['bmi_0','sbp_sit_0','antihtn_0'])
            row['beds'][b]={'later_bed':P.count(len(later)),'paired_bed':P.count(len(both)),
                            'basic':P.count(len(basic)),'clinical':P.count(len(clinical)),
                            'repeat_omics_basic':P.count(basic.participant_id.astype(str).isin(ids[1]).sum())}
        out[name]=row
    return out

def analyse(fr,relations,boot,seed):
    common,inv=P.prepare(fr); rng=np.random.default_rng(seed)
    r={'inventory':inv,'models':{}}
    sets=[('common_primary',common,False,True,False),('common_unadjusted',common,False,False,False),
          ('common_raw',common,True,True,False),
          ('common_pressure',common.dropna(subset=[c+'_'+s for c in ['sbp_sit','dbp_sit','bmi'] for s in ['0','1']]),False,True,True),
          ('common_season',common.loc[common.season_match],False,True,False)]
    for name,d,raw,conditional,pressure in sets:
        if raw and not d[[b+'_raw_'+s for b in P.BEDS for s in ['0','1']]].notna().all().all():
            r['models'][name]={'n':P.count(len(d)),'status':'NA_raw_common_incomplete'}; continue
        r['models'][name]=estimate(d,relations,boot,rng,lambda x:P.arrays(x,raw,conditional,pressure),P.LABELS)
    full=paired_frame(fr); full=full.loc[(full.years>.5)&(full.years<=5)]
    for (i,j),label in zip(P.PAIRS,P.LABELS):
        beds=[P.BEDS[i],P.BEDS[j]]
        need=[b+'_'+s for b in beds for s in ['0','1']]+['age_0','sex_male_0']
        d=full.dropna(subset=need)
        r['models']['pair_'+label]=estimate(d,relations,boot,rng,lambda x:pair_arrays(x,beds),[label])
    # Across the three different-sample pair analyses, adjust zero tests together; never compare coefficients across samples.
    vals=[r['models']['pair_'+k].get('pairs',{}).get(k,{}).get('p_zero',np.nan) for k in P.LABELS]
    good=np.isfinite(vals); q=np.full(3,np.nan)
    if good.any(): q[good]=P.A.bh(np.array(vals)[good])
    for k,v in zip(P.LABELS,q):
        if np.isfinite(v): r['models']['pair_'+k]['pairs'][k]['q_zero']=float(v)
    return r

def digest(r):
    lines=['F1 P17 | '+VERSION,'Relationship-cluster P16 sensitivity; fixed pairwise cohorts; omics feasibility only.',
           'Strict stages; interval>0.5<=5y; n>=200 andgroups>=100; no biological independence claim.']
    for name,m in r['models'].items():
        lines.append(f"{name} n={m['n']} {m['status']} g={m.get('groups','NA')} multi={m.get('multi_groups','NA')} peopleMulti={m.get('people_in_multi','NA')} B={m.get('B_used','NA')}/{m.get('B_failed','NA')} allTOST={m.get('allTOST','NA')}")
        for label,v in m.get('pairs',{}).items():
            f=P.fmt
            lines.append(f" {label} r={f(v['r'])} CI95=[{f(v['ci95'][0])},{f(v['ci95'][1])}] q={f(v['q_zero'])} TOST={f(v['p_tost'])} CI90=[{f(v['ci90'][0])},{f(v['ci90'][1])}]")
    for layer,m in r['omics'].items():
        lines.append(layer+' '+json.dumps({k:v for k,v in m.items() if k!='beds'},separators=(',',':')))
        for b,v in m.get('beds',{}).items(): lines.append(layer+' '+b+' '+json.dumps(v,separators=(',',':')))
    lines.append('F1_P17_DONE errors=[]')
    if len(lines)>120 or any(len(l)>220 for l in lines): raise ValueError('digest size')
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser()
    for k in ['root','out','atlas','omics']: ap.add_argument('--'+k,required=True)
    ap.add_argument('--boot',type=int,default=1000); ap.add_argument('--seed',type=int,default=20261008)
    a=ap.parse_args(); target=Path(a.out)
    if target.exists() or a.boot<20: raise SystemExit('existing output or invalid boot')
    P.A=P.load(a.atlas,'atlas_p17'); P.O=P.load(a.omics,'omics_p17'); P.O.A=P.A
    start=time.time(); fr=P.A.build_frame(a.root,None)[0].reset_index(drop=True)
    ids=set(fr.participant_id.astype(str)); rel=P.A.relation_components(a.root)
    r=analyse(fr,rel,a.boot,a.seed); r['omics']=omics_inventory(a.root,fr)
    r=P.clean(r); dg=digest(r); js=json.dumps(r,sort_keys=True,allow_nan=False)
    P.O.privacy_check(dg,ids); P.O.privacy_check(js,ids)
    target.mkdir(); (target/'digest.txt').write_text(dg); (target/'results.json').write_text(js+'\n')
    (target/'SHA256SUMS').write_text(''.join(hashlib.sha256((target/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['digest.txt','results.json']))
    print('F1_P17_DONE errors=[] runtime=%.1fs'%(time.time()-start),flush=True)
    print('digest lines=%d sq12=%s'%(len(dg.splitlines()),hashlib.sha256(re.sub(' +',' ',dg).encode()).hexdigest()[:12]))

if __name__=='__main__': main()
