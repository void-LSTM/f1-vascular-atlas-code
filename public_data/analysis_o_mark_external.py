#!/usr/bin/env python3
"""Fixed L14 MARK clinical CAVI/IMT profiles; see dated protocol, no selection."""
import os
for _e in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[_e]='1'
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import s0an_common as C
import analysis_m_twobed_replication as M

TASK='o_mark_external'
TERMS=['age','male','dbp','pp','bmi','hr','glucose','hdl','smoker','aht']
BINARY={'male','smoker','aht'}
PROTOCOL=C.PAPER/'docs/L14_MARK_EXTERNAL_PROTOCOL_20261007.md'

def model_specs(d):
    out=[('MA',TERMS,False,False),('RANK',TERMS,True,False),
         ('SBP',[c for c in TERMS if c not in ['dbp','pp']]+['sbp'],False,False),
         ('UNTREATED',[c for c in TERMS if c!='aht'],False,True)]
    for c in TERMS+['sbp']:
        out.append(('MIN:'+c,[c]+[a for a in ['age','male'] if a!=c],False,False))
    return out

def fit(d, cols, rank=False):
    X=np.ones((len(d),len(cols)+1))
    for i,c in enumerate(cols,1):
        a=d[c].to_numpy(float)
        X[:,i]=a if c in BINARY else (a-a.mean())/a.std(ddof=1)
    Y=d[['cavi','imt']].to_numpy(float)
    if rank:
        Y=np.column_stack([stats.norm.ppf((stats.rankdata(Y[:,i])-.5)/len(Y)) for i in range(2)])
    Y=(Y-Y.mean(axis=0))/Y.std(axis=0,ddof=1)
    if not np.isfinite(X).all() or not np.isfinite(Y).all() or np.linalg.matrix_rank(X)!=X.shape[1]:
        raise np.linalg.LinAlgError('nonfinite or rank-deficient design')
    return np.linalg.lstsq(X,Y,rcond=None)[0][1:]

def bh(p):
    p=np.asarray(p,float); order=np.argsort(p); q=np.empty(len(p))
    q[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return q

def simulated(seed):
    rng=np.random.default_rng(seed); n=800
    d=pd.DataFrame({c:rng.normal(size=n) for c in TERMS+['sbp']})
    for c in BINARY: d[c]=rng.binomial(1,.4,n)
    d['cavi']=.7*d.age+.6*d.dbp-.5*d.bmi+rng.normal(size=n)
    d['imt']=.7*d.age-.5*d.dbp+.5*d.bmi+rng.normal(size=n)
    return d

def analyse(d,nboot,seed):
    d=d.dropna(subset=TERMS+['sbp','cavi','imt']).reset_index(drop=True)
    if len(d)<100: raise RuntimeError('primary n<100')
    rng=np.random.default_rng(seed); effects=[]; tests=[]; runs=[]; stored={}
    for name,cols,rank,untreated in model_specs(d):
        z=d.loc[d.aht==0].reset_index(drop=True) if untreated else d
        if len(z)<100:
            runs.append({'model':name,'n':len(z),'status':'NA_n_lt_100'}); continue
        full=fit(z,cols,rank); boot=[]; failed=0
        # Paired resampling, same participants/outcomes; models have separate deterministic streams.
        for _ in range(nboot):
            b=z.iloc[rng.integers(0,len(z),len(z))]
            try: boot.append(fit(b,cols,rank))
            except np.linalg.LinAlgError: failed+=1
        if len(boot)<.95*nboot: raise RuntimeError('bootstrap usable fraction<.95: '+name)
        boot=np.asarray(boot); stored[name.replace(':','_')]=boot
        runs.append({'model':name,'n':len(z),'B_used':len(boot),'B_failed':failed,'status':'OK',
                     'design_condition':float(np.linalg.cond(np.column_stack([np.ones(len(z))]+[
                         z[c].to_numpy() if c in BINARY else (z[c]-z[c].mean()).to_numpy()/z[c].std() for c in cols])))})
        focus=[name.split(':')[1]] if name.startswith('MIN:') else cols
        for c in focus:
            j=cols.index(c)
            for contrast,v,bv in [('cavi',full[j,0],boot[:,j,0]),('imt',full[j,1],boot[:,j,1]),
                                 ('delta',full[j,0]-full[j,1],boot[:,j,0]-boot[:,j,1])]:
                effects.append({'model':name,'driver':c,'contrast':contrast,'n':len(z),**M.summ(v,bv)})
        if not name.startswith('MIN:'):
            a,b=full.T; BA,BB=boot[:,:,0],boot[:,:,1]
            q,df,p=M.wald(a-b,BA-BB)
            tests.append({'model':name,'test':'equality','n':len(z),'statistic':q,'df':df,'p':p,
                          'cov_condition':float(np.linalg.cond(np.cov(BA-BB,rowvar=False)))})
            gain,q,df,p=M.gain_only(a,b,BA,BB)
            tests.append({'model':name,'test':'gain_only','n':len(z),'statistic':q,'df':df,'p':p,'gain':gain})
            if name=='MA':
                jj=[i for i,c in enumerate(cols) if c not in ['age','male']]
                q,df,p=M.wald((a-b)[jj],(BA-BB)[:,jj])
                tests.append({'model':name,'test':'equality_without_age_sex','n':len(z),
                              'statistic':q,'df':df,'p':p})
    ef=pd.DataFrame(effects); mask=(ef.model=='MA')&(ef.contrast=='delta')
    ef.loc[mask,'q_primary10']=bh(ef.loc[mask,'p'])
    return ef,pd.DataFrame(tests),pd.DataFrame(runs),stored,d

def concordance(ef):
    atl,_,_=M.parse_hpp(); rows=[]
    mapping={'age':'age','dbp':'dbp_sit','pp':'pp_sit','bmi':'bmi','glucose':'glucose',
             'hdl':'hdl','smoker':'smk_current','sbp':'sbp_sit','hr':'hr_sit'}
    for c,h in mapping.items():
        rr=atl.get(('M1',h)); e=ef[(ef.model=='MIN:'+c)&(ef.contrast=='delta')]
        if rr is None or len(e)!=1: continue
        v=float(e.est.iloc[0]); hv=rr['pwv_imt']
        rows.append({'driver':c,'mark_delta':v,'hpp_delta':hv,'same_sign':bool(v*hv>0),
                     'hpp_source':str(M.HPP_FILES['ATL'].relative_to(C.PAPER))+':'+str(rr['line'])})
    return pd.DataFrame(rows)

def main():
    args=C.parse_args(__doc__); mode,out,token=C.gate(args,TASK)
    if (out/'provenance.json').exists(): raise SystemExit('Refusing existing completed output')
    if mode=='real': d,source=C.load_real('mark_2015',token)
    else: d=simulated(args.seed); source={'mode':'synthetic_known_nonproportional'}
    ef,tests,runs,boot,common=analyse(d,args.n_boot,args.seed)
    ef.to_csv(out/'effects.csv',index=False); tests.to_csv(out/'tests.csv',index=False)
    runs.to_csv(out/'analysis_sets.csv',index=False); np.savez_compressed(out/'bootstrap_coefficients.npz',**boot)
    if mode=='real': concordance(ef).to_csv(out/'hpp_direction_comparison.csv',index=False)
    prov={'mode':mode,'source':source,'protocol_sha256':C.sha256(PROTOCOL) if PROTOCOL.exists() else None,
          'driver_sha256':C.sha256(Path(__file__)),'reused_utility_sha256':C.sha256(Path(M.__file__)),
          'seed':args.seed,'n_boot':args.n_boot,'primary_n':len(common),
          'primary_terms':TERMS,'formal_registration':False}
    (out/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    text=['# L14 MARK continuous CAVI/IMT external clinical profiles','',
          '- Mode: '+mode+'; fixed protocol: docs/L14_MARK_EXTERNAL_PROTOCOL_20261007.md.',
          '- Partial two-modality clinical cross-sectional validation; CAVI differs from HPPfaPWV.',
          '- No ABI/retina/events/NK-specific data. All primary/null/sensitivity results retained.',
          '- Gain-only tests use an exploratory asymptotic Wald reference; no calibrated exact P claim.','',
          '## Analysis sets','',runs.to_csv(index=False),'## Joint tests','',tests.to_csv(index=False),
          '## Primary paired contrasts (CAVI minus IMT)','',
          ef.loc[(ef.model=='MA')&(ef.contrast=='delta')].to_csv(index=False)]
    (out/'report.md').write_text('\n'.join(text)+'\n')
    (out/'SHA256SUMS').write_text(''.join(C.sha256(p)+'  '+p.name+'\n' for p in sorted(out.iterdir()) if p.is_file()))
    print(tests.to_string(index=False)); print('DONE',mode,'primary_n',len(common),'out',out)

if __name__=='__main__': main()
