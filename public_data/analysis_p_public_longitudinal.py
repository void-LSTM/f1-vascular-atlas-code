#!/usr/bin/env python3
"""L15 fixed public two-modality changes; see dated protocol."""
import os
for e in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[e]='1'
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
PAPER=Path(os.environ.get("F1_WORKDIR", Path(__file__).resolve().parents[1]))
PROTOCOL=PAPER/'docs/L15_PUBLIC_LONGITUDINAL_PROTOCOL_20261007.md'

def corr(d,cols,rank=False):
    X=np.column_stack([np.ones(len(d))]+[d[c].to_numpy(float) for c in cols])
    Y=d[['dy1','dy2']].to_numpy(float).copy()
    if rank:Y=np.column_stack([stats.norm.ppf((stats.rankdata(Y[:,j])-.5)/len(Y)) for j in range(2)])
    if not np.isfinite(X).all() or not np.isfinite(Y).all():raise np.linalg.LinAlgError('nonfinite')
    X=X[:,np.r_[True,X[:,1:].std(axis=0)>1e-12]]
    if np.linalg.matrix_rank(X)!=X.shape[1]:raise np.linalg.LinAlgError('design')
    Y-=X@np.linalg.lstsq(X,Y,rcond=None)[0]
    if min(Y.std(axis=0))<1e-12:raise np.linalg.LinAlgError('degenerate')
    return float(np.corrcoef(Y,rowvar=False)[0,1])

def fit(d,cols,B,rng,rank=False,site=False,infer=False):
    z=d.dropna(subset=cols+['dy1','dy2']).reset_index(drop=True)
    if len(z)<100:return {'n':len(z),'status':'NA_n_lt_100'}
    try:r=corr(z,cols,rank)
    except np.linalg.LinAlgError:return {'n':len(z),'status':'NA_design'}
    blocks=[np.flatnonzero(z.site.to_numpy()==s) for s in sorted(z.site.unique())] if site else [np.arange(len(z))]
    boot=[];failed=0
    for _ in range(B):
        ix=np.concatenate([rng.choice(a,len(a),replace=True) for a in blocks])
        try:boot.append(corr(z.iloc[ix],cols,rank))
        except np.linalg.LinAlgError:failed+=1
    out={'n':len(z),'status':'estimated' if len(boot)>=.95*B else 'NA_boot_fraction','B_used':len(boot),'B_failed':failed}
    if out['status']!='estimated':return out
    out.update(r=r,ci95=np.percentile(boot,[2.5,97.5]).tolist())
    if infer and len(z)>=200:
        se=np.std(np.arctanh(np.clip(boot,-.999999,.999999)),ddof=1);zz=np.arctanh(np.clip(r,-.999999,.999999));bound=np.arctanh(.2)
        out.update(ci90=np.percentile(boot,[5,95]).tolist(),p_zero=float(2*stats.norm.sf(abs(zz/se))),p_tost=float(max(stats.norm.sf((zz+bound)/se),stats.norm.cdf((zz-bound)/se))))
    return out

def analyse(data,B,seed):
    rng=np.random.default_rng(seed);out={}
    for cohort,d in data.items():
        basic=['age','male','y1_0','y2_0'];fixed=[c for c in d if c.startswith('site_') or c.startswith('arm_')]
        if cohort=='xin':basic+=fixed
        extra=['bmi','sbp','dbp','d_sbp','d_dbp']+(['d_bmi','dm'] if cohort=='pci' else [])
        out[cohort]={}
        for name,cols,rank in [('primary',basic,False),('unadjusted',[],False),('pressure',basic+extra,False),('rank',basic,True)]:
            out[cohort][name]=fit(d,cols,B,rng,rank,cohort=='xin',cohort=='pci' and name=='primary')
    return out

def prepare():
    p=PAPER/'results/stage0/clean/pci_cavi_2016.parquet';pci=pd.read_parquet(p)
    x=pd.read_parquet(PAPER/'results/stage0/clean/xinruikang_long.parquet');q=pd.read_parquet(PAPER/'results/stage0/clean/xinruikang_participants.parquet')
    if x.duplicated(['participant_id','week']).any():raise ValueError('duplicate visit keys')
    a=x[x.week==0].merge(x[x.week==12],on='participant_id',suffixes=('_0','_1'),validate='one_to_one').merge(q[['participant_id','age_y','sex_male','bmi_kgm2_calc']],on='participant_id',validate='one_to_one')
    years=(a.visit_date_1-a.visit_date_0).dt.total_seconds()/86400/365.25;a=a[(years>=.1)&(years<=.5)].copy()
    a['y1_0']=a.pwv_mean_ms_0;a['y2_0']=a[['abi_l_0','abi_r_0']].min(axis=1,skipna=False)
    a['dy1']=a.pwv_mean_ms_1-a.y1_0;a['dy2']=a.y2_0-a[['abi_l_1','abi_r_1']].min(axis=1,skipna=False)
    a['age']=a.age_y;a['male']=a.sex_male;a['bmi']=a.bmi_kgm2_calc
    a['sbp']=a.sbp_mmhg_0;a['dbp']=a.dbp_mmhg_0;a['d_sbp']=a.sbp_mmhg_1-a.sbp;a['d_dbp']=a.dbp_mmhg_1-a.dbp
    a['d_bmi']=np.nan # weight not repeated in the released examination sheets; preserve unavailable sensitivity
    a['site']=a.site_0
    dummy=pd.get_dummies(a[['site_0','arm_code_0']],prefix=['site','arm'],drop_first=True,dtype=float)
    a=pd.concat([a,dummy],axis=1)
    cols=['age','male','bmi','sbp','dbp','d_bmi','d_sbp','d_dbp','y1_0','y2_0','dy1','dy2','site']+list(dummy)
    # Exclude original site_0 text column from the numerical design by selecting only fixed dummy names.
    return {'pci':pci,'xin':a[cols]},[p,PAPER/'results/stage0/clean/xinruikang_long.parquet',PAPER/'results/stage0/clean/xinruikang_participants.parquet']

def simulate(seed):
    rng=np.random.default_rng(seed);n=800
    d=pd.DataFrame({c:rng.normal(size=n) for c in ['age','bmi','sbp','dbp','d_bmi','d_sbp','d_dbp','y1_0','y2_0']})
    d['male']=rng.binomial(1,.5,n);d['dm']=rng.binomial(1,.5,n)
    e=rng.normal(size=n);d['dy1']=d.age+.7*d.y1_0+e;d['dy2']=d.age+.5*d.y2_0+.75*e+rng.normal(size=n)*.3
    return {'pci':d}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--real-authorised',action='store_true');ap.add_argument('--simulate',action='store_true');ap.add_argument('--out',required=True);ap.add_argument('--boot',type=int,default=2000);ap.add_argument('--seed',type=int,default=20261009);a=ap.parse_args()
    if a.real_authorised==a.simulate:raise SystemExit('choose --simulate or --real-authorised')
    out=Path(a.out)
    if out.exists():raise SystemExit('refusing existing output')
    data,inputs=(simulate(a.seed),[]) if a.simulate else prepare()
    r=analyse(data,a.boot,a.seed);out.mkdir(parents=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    prov={'mode':'real_authorised' if a.real_authorised else 'synthetic','formal_registration':False,'protocol_sha256':sha(PROTOCOL) if PROTOCOL.exists() else None,'script_sha256':sha(Path(__file__)),'boot':a.boot,'seed':a.seed,'inputs':[{'path':str(p),'sha256':sha(p)} for p in inputs]}
    (out/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');(out/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    (out/'report.md').write_text('# L15 public two-modality longitudinal supplement\n\nSee fixed protocol and source/QC caveats. PCI: released rows, CAVI rather than HPPfaPWV; unresolved206-vs205 rows. Xinruikang: descriptive only; pressure sensitivity conditions on baseline BMI and BP change; repeated BMI absent.\n\n'+json.dumps(r,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(out.iterdir()) if p.is_file()))
    print(json.dumps(r,indent=2));print('L15_DONE')
if __name__=='__main__':main()
