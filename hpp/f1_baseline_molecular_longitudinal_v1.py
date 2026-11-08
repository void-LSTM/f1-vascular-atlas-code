#!/usr/bin/env python3
"""P18 fixed baseline molecular panel; aggregate-only prospective conditional associations."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import f1_longitudinal_extension_v1 as Q
P=Q.P
MARKERS=['apob','glyca','cer16_24','cer18_24','cer241_24']
VERSION='f1_baseline_molecular_longitudinal_v1.1 2026-10-07'

def samples(root,layer):
    folder='nightingale_metabolomics' if layer=='NMR' else 'serum_lipidomics'
    d=P.A.rd(root,folder+'/'+folder+'.parquet')
    if d is None: return None
    d=d.loc[d.research_stage==P.A.MAIN_STAGES[0]].copy().reset_index(drop=True)
    d['participant_id']=d.participant_id.astype(str)
    date=P.O.day(d.collection_date) if 'collection_date' in d else pd.Series(pd.NaT,index=d.index)
    if 'collection_timestamp' in d: date=date.fillna(P.O.day(d.collection_timestamp))
    d['sample_date']=date
    d['hour']=P.O.draw_hour(d.collection_timestamp) if 'collection_timestamp' in d else np.nan
    names=MARKERS[:2] if layer=='NMR' else MARKERS[2:]
    for marker in names:
        if layer=='NMR':
            x=pd.to_numeric(d[marker],errors='coerce') if marker in d else pd.Series(np.nan,index=d.index)
            d[marker]=np.log(x.where(x>0))
        else:
            src={'cer16_24':'cer_d18:1/c16:0','cer18_24':'cer_d18:1/c18:0','cer241_24':'cer_d18:1/c24:1'}[marker]
            d[marker]=pd.to_numeric(d[src],errors='coerce')-pd.to_numeric(d['cer_d18:1/c24:0'],errors='coerce') if src in d and 'cer_d18:1/c24:0' in d else np.nan
    return d[['participant_id','sample_date','hour']+names].replace([np.inf,-np.inf],np.nan)

def join_samples(fr,s):
    if s is None: return None,{'status':'NA_table'}
    d=Q.paired_frame(fr);d=d.loc[(d.years>.5)&(d.years<=5)]
    d=d.merge(s,on='participant_id',how='inner',validate='one_to_many')
    flow={'stage_overlap':P.count(d.participant_id.nunique())}
    d['gap']=(P.O.day(d.sample_date)-P.O.day(d.visit_date_0)).dt.total_seconds()/86400
    d=d.loc[d.gap.notna()&(d.gap.abs()<=365)&(P.O.day(d.sample_date)<P.O.day(d.visit_date_1))].copy()
    d['abs_gap']=d.gap.abs();d['min_gap']=d.groupby('participant_id').abs_gap.transform('min')
    d=d.loc[d.abs_gap==d.min_gap]
    ties=d.participant_id.duplicated(keep=False);flow['ambiguous_ties']=P.count(d.loc[ties].participant_id.nunique())
    d=d.loc[~ties].copy();d['gap_years']=d.gap/365.25
    flow['dated_unique']=P.count(len(d));flow['gap_abs_q05_25_50_75_95']=P.O.qt(d.abs_gap.to_numpy())
    return d,flow

def covs(bed,model):
    c=['age_0','sex_male_0','years','gap_years',bed+'_0']
    if model!='A': c+=['bmi_0','sbp_sit_0','antihtn_0']
    if model=='L': c+=['ldl_0','hdl_0','log_tg_0','statin_0']
    return c

def fit(d,marker,beds,model,scales):
    x=(d[marker].to_numpy(float)-scales['xm'])/scales['xs'];betas=[]
    hr=d.hour.to_numpy(float);hc=np.where(np.isfinite(hr),hr,scales['hm'])-scales['hm']
    for b in beds:
        y=(P.A.ORIENT[b]*d[b+'_1'].to_numpy(float)-scales[b][0])/scales[b][1]
        raw=np.column_stack([d[c].to_numpy(float) for c in covs(b,model)]+[hc,hc**2,(~np.isfinite(hr)).astype(float)])
        # A rare nuisance level can be absent in a whole-family resample. Its all-zero/constant
        # column contains no information in that resample; omit it, preserving exposure and intercept.
        keep=scales['keep_'+b] & (raw.std(axis=0)>1e-12)
        X=np.column_stack([np.ones(len(d)),x,raw[:,keep]])
        if not np.isfinite(X).all() or not np.isfinite(y).all() or np.linalg.matrix_rank(X)!=X.shape[1] or len(d)-X.shape[1]<20:
            raise np.linalg.LinAlgError('rank or residual degrees of freedom')
        betas.append(np.linalg.lstsq(X,y,rcond=None)[0][1])
    return np.asarray(betas)

def estimate(d,marker,beds,model,relations,boot,rng):
    blocks,info=Q.group_info(d,relations)
    if len(d)<100: return {**info,'status':'NA_n_lt_100'}
    if blocks is None: return info
    if len(blocks)<100: return {**info,'status':'NA_groups_lt_100'}
    if np.isfinite(d.hour).sum()<20: return {**info,'status':'NA_draw_hour'}
    sc={'xm':float(d[marker].mean()),'xs':float(d[marker].std(ddof=0)),'hm':float(d.hour.median())}
    if sc['xs']<=1e-12: return {**info,'status':'NA_exposure_SD'}
    hc=d.hour.fillna(sc['hm']).to_numpy()-sc['hm']
    for b in beds:
        sc[b]=(float(P.A.ORIENT[b]*d[b+'_1'].mean()),float(d[b+'_1'].std(ddof=0)))
        if sc[b][1]<=1e-12: return {**info,'status':'NA_outcome_SD'}
        raw=np.column_stack([d[c].to_numpy(float) for c in covs(b,model)]+[hc,hc**2,d.hour.isna().to_numpy(float)])
        sc['keep_'+b]=raw.std(axis=0)>1e-12
    try: point=fit(d,marker,beds,model,sc)
    except np.linalg.LinAlgError:return {**info,'status':'NA_design'}
    draws=[];failed=0
    for _ in range(boot):
        try: draws.append(fit(d.iloc[Q.group_draw(blocks,rng)],marker,beds,model,sc))
        except np.linalg.LinAlgError:failed+=1
    if len(draws)<.95*boot:return {**info,'status':'NA_boot_fraction','B_used':len(draws),'B_failed':failed}
    bb=np.asarray(draws);se=bb.std(axis=0,ddof=1)
    if np.any(se<=1e-12):return {**info,'status':'NA_boot_SE'}
    out={**info,'status':'estimated','B_used':len(bb),'B_failed':failed,'beds':{}}
    for j,b in enumerate(beds):out['beds'][b]={'beta':point[j],'ci95':np.percentile(bb[:,j],[2.5,97.5]),'p':2*stats.norm.sf(abs(point[j]/se[j]))}
    if len(beds)==3:
        H=np.array([[1,-1,0],[1,0,-1],[0,1,-1]])
        delta=H@point;bd=bb@H.T;sd=bd.std(axis=0,ddof=1)
        out['contrasts']={k:{'beta':delta[j],'ci95':np.percentile(bd[:,j],[2.5,97.5]),'p':float(2*stats.norm.sf(abs(delta[j]/sd[j]))) if sd[j]>1e-12 else None} for j,k in enumerate(P.LABELS)}
        hh=H[:2];V=np.cov(bb@hh.T,rowvar=False);rank=np.linalg.matrix_rank(V)
        w=hh@point;stat=float(w@np.linalg.pinv(V)@w)
        out['joint']={'Q':stat,'df':int(rank),'p':float(stats.chi2.sf(stat,rank)) if rank==2 else None}
    return out

def adjust(records,expected):
    ps=[r['p'] if r and r.get('p') is not None else 1. for r in records]
    ps+= [1.]*(expected-len(ps));qs=P.A.bh(ps)
    for r,q in zip(records,qs):
        if r is not None:r['q']=float(q)

def analyse(root,fr,relations,boot,seed):
    rng=np.random.default_rng(seed);r={'inventory':{},'models':{},'common':{}}
    for layer in ['NMR','lipid']:
        d,flow=join_samples(fr,samples(root,layer));r['inventory'][layer]=flow
        for marker in MARKERS[:2] if layer=='NMR' else MARKERS[2:]:
            for b in P.BEDS:
                key=marker+'_'+b;r['models'][key]={}
                if d is None:r['models'][key]={'C':{'n':'<20','status':'NA_table'}};continue
                need=[marker,b+'_1']+covs(b,'C');dc=d.dropna(subset=need)
                for m in ['A','C','L','D30']:
                    dm=dc.dropna(subset=covs(b,'L')) if m=='L' else dc.loc[dc.abs_gap<=30] if m=='D30' else dc
                    r['models'][key][m]=estimate(dm,marker,[b],'C' if m=='D30' else m,relations,boot,rng)
            if d is None:r['common'][marker]={'n':'<20','status':'NA_table'};continue
            need=list(dict.fromkeys([marker]+[b+'_1' for b in P.BEDS]+sum([covs(b,'C') for b in P.BEDS],[])))
            r['common'][marker]=estimate(d.dropna(subset=need),marker,P.BEDS,'C',relations,boot,rng)
    for m in ['A','C','L','D30']:
        rec=[r['models'][marker+'_'+b].get(m,{}).get('beds',{}).get(b) for marker in MARKERS for b in P.BEDS]
        adjust(rec,15)
    adjust([r['common'][marker].get('joint') for marker in MARKERS],5)
    adjust([r['common'][marker].get('contrasts',{}).get(k) for marker in MARKERS for k in P.LABELS],15)
    return r

def digest(r):
    lines=['F1 P18 | '+VERSION,'Baseline molecular -> later bed conditional on baseline; no repeat-omics/RNA requirement.',
           'Same stage, dated gap<=365d, beforefollowup; interval>0.5<=5y; n/groups>=100; fixed15primaryBH.',
           'A basic; C primary clinical; L pluslipidpanel; D30 clinical gap<=30d. Familybootstrap1000; fixedscales.']
    f=P.fmt
    for layer,row in r['inventory'].items():lines.append(layer+' '+json.dumps(row,separators=(',',':')))
    for key,models in r['models'].items():
        for m,v in models.items():
            text=f"{key} {m} n={v['n']} g={v.get('groups','NA')} {v['status']}"
            if v['status']=='estimated':
                z=next(iter(v['beds'].values()));text+=f" b={f(z['beta'])} CI=[{f(z['ci95'][0])},{f(z['ci95'][1])}] q={f(z['q'])} B={v['B_used']}/{v['B_failed']}"
            lines.append(text)
    for marker,v in r['common'].items():
        lines.append(f"common {marker} n={v['n']} g={v.get('groups','NA')} {v['status']} jointq={f(v.get('joint',{}).get('q'))} B={v.get('B_used','NA')}/{v.get('B_failed','NA')}")
        for k,z in v.get('contrasts',{}).items():lines.append(f" {k} d={f(z['beta'])} CI=[{f(z['ci95'][0])},{f(z['ci95'][1])}] q={f(z['q'])}")
    lines.append('F1_P18_DONE errors=[]')
    if len(lines)>120 or any(len(x)>220 for x in lines):raise ValueError('digest size')
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser()
    for k in ['root','out','atlas','omics']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--boot',type=int,default=1000);ap.add_argument('--seed',type=int,default=20261010)
    a=ap.parse_args();out=Path(a.out)
    if out.exists() or a.boot<20:raise SystemExit('existing output or invalid bootstrap')
    P.A=P.load(a.atlas,'atlas_p18');P.O=P.load(a.omics,'omics_p18');P.O.A=P.A
    start=time.time();fr=P.A.build_frame(a.root,None)[0].reset_index(drop=True)
    r=analyse(a.root,fr,P.A.relation_components(a.root),a.boot,a.seed);r=P.clean(r)
    dg=digest(r);js=json.dumps(r,sort_keys=True,allow_nan=False)+'\n';ids=set(fr.participant_id.astype(str))
    P.O.privacy_check(dg,ids);P.O.privacy_check(js,ids)
    out.mkdir();(out/'digest.txt').write_text(dg);(out/'results.json').write_text(js)
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['digest.txt','results.json']))
    print('F1_P18_DONE errors=[] runtime=%.1fs'%(time.time()-start),flush=True)

if __name__=='__main__':main()
