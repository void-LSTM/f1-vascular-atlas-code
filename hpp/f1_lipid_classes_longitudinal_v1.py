#!/usr/bin/env python3
"""P19 fixed lipid-class baseline summaries; inherits reviewed P18v1.1 estimator."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import f1_baseline_molecular_longitudinal_v1 as R
P=R.P
CLASSES=['TAG','PC','SM']
VERSION='f1_lipid_classes_longitudinal_v1.0 2026-10-07'

def class_score(d,prefix):
    cols=[c for c in d if c.startswith(prefix) and pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])]
    if not cols:return np.full(len(d),np.nan),0
    X=d[cols].to_numpy(float,na_value=np.nan);X[~np.isfinite(X)]=np.nan
    keep=np.isfinite(X).mean(axis=0)>=.5;X=X[:,keep]
    if not X.shape[1]:return np.full(len(d),np.nan),0
    mean=np.nanmean(X,axis=0);sd=np.nanstd(X,axis=0);keep=sd>1e-12
    X=X[:,keep];mean=mean[keep];sd=sd[keep]
    if len(sd)<3:return np.full(len(d),np.nan),int(len(sd))
    Z=(np.clip(X,mean-5*sd,mean+5*sd)-mean)/sd
    good=np.isfinite(Z).sum(axis=1)>=np.ceil(.8*Z.shape[1]);out=np.full(len(d),np.nan)
    out[good]=np.nanmean(Z[good],axis=1)
    return out,int(Z.shape[1])

def samples(root):
    d=P.A.rd(root,'serum_lipidomics/serum_lipidomics.parquet')
    if d is None:return None,{}
    d=d.loc[d.research_stage==P.A.MAIN_STAGES[0]].copy().reset_index(drop=True)
    date=P.O.day(d.collection_date) if 'collection_date' in d else pd.Series(pd.NaT,index=d.index)
    if 'collection_timestamp' in d:date=date.fillna(P.O.day(d.collection_timestamp))
    s=pd.DataFrame({'participant_id':d.participant_id.astype(str),'sample_date':date})
    s['hour']=P.O.draw_hour(d.collection_timestamp) if 'collection_timestamp' in d else np.nan
    info={}
    for c in CLASSES:s[c],info[c]=class_score(d,c.lower()+'_')
    return s,info

def analyse(root,fr,relations,boot,seed):
    s,features=samples(root);d,flow=R.join_samples(fr,s)
    r={'inventory':{'flow':flow,'retained_features':features},'models':{},'common':{}}
    rng=np.random.default_rng(seed)
    for marker in CLASSES:
        for b in P.BEDS:
            key=marker+'_'+b;r['models'][key]={}
            if d is None:r['models'][key]['C']={'n':'<20','status':'NA_table'};continue
            dc=d.dropna(subset=[marker,b+'_1']+R.covs(b,'C'))
            for m in ['A','C','L','D30']:
                dm=dc.dropna(subset=R.covs(b,'L')) if m=='L' else dc.loc[dc.abs_gap<=30] if m=='D30' else dc
                r['models'][key][m]=R.estimate(dm,marker,[b],'C' if m=='D30' else m,relations,boot,rng)
        if d is None:r['common'][marker]={'n':'<20','status':'NA_table'};continue
        need=list(dict.fromkeys([marker]+[b+'_1' for b in P.BEDS]+sum([R.covs(b,'C') for b in P.BEDS],[])))
        r['common'][marker]=R.estimate(d.dropna(subset=need),marker,P.BEDS,'C',relations,boot,rng)
    for m in ['A','C','L','D30']:R.adjust([r['models'][c+'_'+b].get(m,{}).get('beds',{}).get(b) for c in CLASSES for b in P.BEDS],9)
    R.adjust([r['common'][c].get('joint') for c in CLASSES],3)
    R.adjust([r['common'][c].get('contrasts',{}).get(k) for c in CLASSES for k in P.LABELS],9)
    return r

def digest(r):
    # Reuse exact row formatting; replace module-specific inventory/header only.
    x=dict(r);x['inventory']={};lines=R.digest(x).splitlines()
    lines[:4]=['F1 P19 | '+VERSION,'Baseline TAG/PC/SM classmean -> laterbed conditional on baseline; fixed9primaryBH.',
               'Prefixonly; >=50%feature prevalence; >=3members; >=80%rowcoverage; no association selection.',
               'A/C/L/D30 asP18v1.1; strictdates/stages; n/groups>=100; familybootstrap1000; fixedscales.']
    flow=r['inventory']['flow'];feat=r['inventory']['retained_features']
    lines.insert(4,'Inventory features='+json.dumps(feat,separators=(',',':'))+' stage='+str(flow.get('stage_overlap','NA'))+' dated='+str(flow.get('dated_unique','NA'))+' ties='+str(flow.get('ambiguous_ties','NA')))
    lines[-1]='F1_P19_DONE errors=[]'
    if len(lines)>120 or any(len(x)>220 for x in lines):raise ValueError('digest size')
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser()
    for k in ['root','out','atlas','omics']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--boot',type=int,default=1000);ap.add_argument('--seed',type=int,default=20261011)
    a=ap.parse_args();out=Path(a.out)
    if out.exists() or a.boot<20:raise SystemExit('existing output or invalid bootstrap')
    P.A=P.load(a.atlas,'atlas_p19');P.O=P.load(a.omics,'omics_p19');P.O.A=P.A
    start=time.time();fr=P.A.build_frame(a.root,None)[0].reset_index(drop=True)
    r=P.clean(analyse(a.root,fr,P.A.relation_components(a.root),a.boot,a.seed));dg=digest(r);js=json.dumps(r,sort_keys=True,allow_nan=False)+'\n'
    ids=set(fr.participant_id.astype(str));P.O.privacy_check(dg,ids);P.O.privacy_check(js,ids)
    out.mkdir();(out/'digest.txt').write_text(dg);(out/'results.json').write_text(js)
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['digest.txt','results.json']))
    print('F1_P19_DONE errors=[] runtime=%.1fs'%(time.time()-start),flush=True)

if __name__=='__main__':main()
