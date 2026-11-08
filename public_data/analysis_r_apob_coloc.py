#!/usr/bin/env python3
"""L17 fixed APOB cis-lead ratio and dense regional coloc; public data only."""
import argparse, hashlib, json, platform, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
import analysis_i_mr as I

IDS=['asi','cimt','crae','crve','pad_mvp']
REGION={'37':(20724301,21766945),'38':(20501429,21544073)}
EXPOSURE=dict(id='apob',path='25a_apoB_noUKB_Karjalainen2024/GCST90301946.h.tsv.gz',sep='\t',build='38',
 cols=dict(I._SSF,snp='rsid'),n_default=136016,kind='cont')
SPECS=[EXPOSURE]+[next(s.copy() for s in I.SPECS if s['id']==k) for k in IDS]

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def qualify(d):
 d=d.copy();d['pair']=[''.join(sorted((str(a),str(b)))) for a,b in zip(d.ea,d.oa)]
 ok=d.ea.isin(list('ACGT')) & d.oa.isin(list('ACGT')) & d.ea.ne(d.oa)
 ok &= np.isfinite(d.beta)&np.isfinite(d.se)&d.se.gt(0)&np.isfinite(d.p)&d.p.between(0,1)&d.eaf.gt(0)&d.eaf.lt(1)
 return d.loc[ok].reset_index(drop=True)

def scan(task):
 import pyarrow as pa, pyarrow.csv as cv
 spec,root=task;path=Path(root)/spec['path'];cols=list(dict.fromkeys(spec['cols'].values()))
 reader=cv.open_csv(pa.input_stream(str(path),compression='gzip'),read_options=cv.ReadOptions(block_size=16<<20),
  parse_options=cv.ParseOptions(delimiter=spec['sep']),
  convert_options=cv.ConvertOptions(include_columns=cols,column_types={c:pa.string() for c in cols},strings_can_be_null=True,null_values=I.NA_VALUES))
 parts=[];a,b=REGION[spec['build']];n=0
 for batch in reader:
  raw=batch.to_pandas();n+=len(raw)
  ch=raw[spec['cols']['chr']].str.replace(r'^chr','',regex=True)
  pos=pd.to_numeric(raw[spec['cols']['pos']],errors='coerce')
  keep=(ch=='2')&pos.between(a,b)
  if keep.any():parts.append(I.standardize(spec,raw.loc[keep]))
 if not parts:raise ValueError('no regional variants '+spec['id'])
 d=qualify(pd.concat(parts,ignore_index=True));print('L17_EXTRACT',spec['id'],n,len(d),flush=True)
 return spec['id'],d,dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path),rows_scanned=n,qualified_region=len(d))

def abf(beta,se,prior):
 v=np.square(se);w=prior**2
 return .5*(np.log(v/(v+w))+np.square(beta/se)*w/(v+w))

def combine(l1,l2,p12=1e-5):
 a=logsumexp(l1);b=logsumexp(l2);same=logsumexp(l1+l2)
 delta=min(float(same-a-b),0.0)
 distinct=(a+b+np.log(-np.expm1(delta))) if delta<0 else -np.inf
 ll=np.array([0,np.log(1e-4)+a,np.log(1e-4)+b,2*np.log(1e-4)+distinct,np.log(p12)+same])
 return np.exp(ll-logsumexp(ll))

def trait_sd(d):
 # Official coloc sdY.est no-intercept regression of N*2f(1-f) on inverse varbeta.
 xx=1/np.square(d.se.to_numpy());yy=d.n.to_numpy()*2*d.eaf.to_numpy()*(1-d.eaf.to_numpy())
 return float(np.sqrt(np.dot(xx,yy)/np.dot(xx,xx)))

def orientation(x,y):
 ea,oa=x.ea,x.oa;ya,yb=y.ea,y.oa
 if {ea,oa} in ({'A','T'},{'C','G'}):raise ValueError('palindrome')
 comp=str.maketrans('ACGT','TGCA')
 for a,b in ((ya,yb),(ya.translate(comp),yb.translate(comp))):
  if (ea,oa)==(a,b):return 1
  if (ea,oa)==(b,a):return -1
 raise ValueError('allele mismatch')

def identities(regions):
 # Native build37 position lookup to rsID from independent regional files; ambiguous mappings dropped.
 ref=pd.concat([regions[k] for k in ['cimt','crae','crve']],ignore_index=True)
 ref=ref.loc[ref.snp.notna(),['pos','pair','snp']].drop_duplicates()
 bad=ref.groupby(['pos','pair']).snp.nunique();bad=set(bad[bad>1].index)
 lookup={(int(r.pos),r.pair):r.snp for r in ref.itertuples() if (r.pos,r.pair) not in bad}
 out={}
 for key,d in regions.items():
  d=d.copy()
  if key=='asi':d['snp']=[lookup.get((int(p),q),pd.NA) for p,q in zip(d.pos,d.pair)]
  d=d.loc[d.snp.notna()].copy();d['key']=d.snp.astype(str)+':'+d.pair
  # Drop every duplicate mapping, rather than favouring association strength.
  d=d.loc[~d.snp.duplicated(keep=False)].reset_index(drop=True);out[key]=d
 return out,len(bad)

def analyse(regions):
 regions,ambiguous=identities(regions);x=regions['apob'];xp=x.loc[~x.pair.isin(['AT','CG'])].sort_values(['p','snp'])
 if xp.empty:raise ValueError('no nonpalindromic cis candidate')
 lead=xp.iloc[0];f=float((lead.beta/lead.se)**2);strong=bool(lead.p<5e-8 and f>=10)
 records=[]
 for k in IDS:
  y=regions[k];spec=next(s for s in SPECS if s['id']==k)
  m=x.merge(y,on='key',suffixes=('_x','_y'));coverage=len(m)/min(len(x),len(y)) if min(len(x),len(y)) else 0
  rec=dict(outcome=k,exposure_qualified=len(x),outcome_qualified=len(y),shared=len(m),coverage=coverage,
   min_p_exposure=float(x.p.min()),min_p_outcome=float(y.p.min()),wald_status='NA',coloc_status='NA',wald_p=1.0)
  hit=y.loc[y.key.eq(lead.key)]
  if strong and len(hit)==1:
   yr=hit.iloc[0];sign=orientation(lead,yr)*spec.get('adverse',1)
   beta=float(sign*yr.beta/lead.beta);se=float(yr.se/abs(lead.beta))
   rec.update(wald_status='OK',wald_beta=beta,wald_se=se,wald_ci=[beta-1.95996398454*se,beta+1.95996398454*se],
    wald_p=float(2*norm.sf(abs(beta/se))),outcome_units=spec['units'])
  elif not strong:rec['wald_status']='NA_WEAK_EXPOSURE'
  else:rec['wald_status']='NA_LEAD_MISSING'
  if len(m)>=1000 and coverage>=.8:
   xx=pd.DataFrame({'beta':m.beta_x,'se':m.se_x,'eaf':m.eaf_x,'n':m.n_x})
   yy=pd.DataFrame({'beta':m.beta_y,'se':m.se_y,'eaf':m.eaf_y,'n':m.n_y})
   sd=trait_sd(yy) if spec['kind']=='cont' else None;pr=.15*sd if sd is not None else .2
   l1=abf(xx.beta.to_numpy(),xx.se.to_numpy(),.15);l2=abf(yy.beta.to_numpy(),yy.se.to_numpy(),pr)
   pp={str(p):combine(l1,l2,p).tolist() for p in [1e-6,1e-5,1e-4]};v=pp['1e-05'];cond=v[4]/(v[3]+v[4])
   supported=bool(rec['min_p_exposure']<5e-8 and rec['min_p_outcome']<5e-8 and min(z[4] for z in pp.values())>=.8 and cond>=.9)
   rec.update(coloc_status='OK',outcome_estimated_sd=sd,posterior_by_p12=pp,conditional_shared=cond,shared_signal_support=supported)
  else:rec['coloc_status']='NA_DENSE_COVERAGE'
  records.append(rec)
 ps=np.array([z['wald_p'] for z in records]);order=np.argsort(ps);q=np.empty(5);q[order]=np.minimum(1,np.minimum.accumulate((ps[order]*5/np.arange(1,6))[::-1])[::-1])
 for z,qq in zip(records,q):z['wald_q']=float(qq);z['signed_genetic_support']=bool(z['wald_status']=='OK' and qq<.05)
 return dict(lead=dict(rsid=str(lead.snp),alleles=[lead.ea,lead.oa],p=float(lead.p),beta=float(lead.beta),se=float(lead.se),F=f,strong=strong),ambiguous_build37_mappings=ambiguous,results=records)

def selftest(out):
 out.mkdir(parents=True,exist_ok=False)
 l1=np.array([2.,-1.,.5]);l2=np.array([-1.,3.,.2]);actual=combine(l1,l2)
 bf1,bf2=np.exp(l1),np.exp(l2)
 direct=np.array([1,1e-4*bf1.sum(),1e-4*bf2.sum(),1e-8*sum(bf1[i]*bf2[j] for i in range(3) for j in range(3) if i!=j),1e-5*(bf1*bf2).sum()]);direct/=direct.sum()
 assert np.allclose(actual,direct,rtol=1e-12)
 s=np.ones(1100)*.01;b=np.zeros(1100);b[50]=.3;different=np.roll(b,150)
 assert combine(abf(b,s,.15),abf(b,s,.15))[4]>.99
 assert combine(abf(b,s,.15),abf(different,s,.15))[3]>.99
 x=pd.Series({'ea':'A','oa':'C'});y=pd.Series({'ea':'C','oa':'A'})
 assert orientation(x,y)==-1 and orientation(x,pd.Series({'ea':'T','oa':'G'}))==1
 d=pd.DataFrame(dict(snp=['rs'+str(i+1) for i in range(1100)],pos=np.arange(1100)+21224301,pos38=np.arange(1100)+21001429,ea='A',oa='C',beta=b,se=s,p=2*norm.sf(abs(b/s)),eaf=.3,n=136016))
 regions={k:qualify(d.copy()) for k in ['apob']+IDS}
 for k in IDS:regions[k]['beta']=b*.5;regions[k]['p']=2*norm.sf(abs(b*.5/s))
 a=analyse(regions);assert all(r['shared']==1100 and abs(abs(r['wald_beta'])-.5)<1e-12 for r in a['results'])
 # Mock file exercises streaming, metadata standardisation and identity mapping end to end.
 raw=out/'mock';raw.mkdir();rawdata=pd.DataFrame(dict(chromosome=2,base_pair_location=d.pos38,effect_allele='A',other_allele='C',beta=b,standard_error=s,effect_allele_frequency=.3,p_value=d.p,rsid=d.snp))
 rawdata.to_csv(raw/'mock.tsv.gz',sep='\t',index=False,compression='gzip');spec=EXPOSURE.copy();spec['path']='mock.tsv.gz'
 key,extracted,_=scan((spec,raw));assert key=='apob' and len(extracted)==1100
 (out/'known_answer.txt').write_text('PASS ABF/direct H0-H4 enumeration; shared vs distinct; reversed/strand orientation; fixed-five ratio recovery; native-region mock-file streaming.\n')
 print('L17_SELFTEST_PASS',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--real',action='store_true');p.add_argument('--gwas-root',type=Path);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
 if a.self_test:selftest(a.outdir);return
 if not a.real or a.gwas_root is None:p.error('--real and --gwas-root are required')
 a.outdir.mkdir(parents=True,exist_ok=False);start=time.time()
 with ProcessPoolExecutor(max_workers=2) as pool:ex=list(pool.map(scan,[(s,a.gwas_root) for s in SPECS]))
 regions={k:d for k,d,_ in ex}
 for k,d,_ in ex:d.to_parquet(a.outdir/(k+'_region.parquet'),index=False)
 result=analyse(regions);result.update(protocol='L17_APOB_CIS_COLOCALISATION_PROTOCOL_20261007',region=REGION,
  input_files={k:r for k,_,r in ex},code_files={str(q.name):sha(q) for q in [Path(__file__),Path(I.__file__),Path(I.M.__file__),Path(I.C.__file__)]},
  versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},runtime_seconds=time.time()-start)
 (a.outdir/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 pd.json_normalize(result['results']).to_csv(a.outdir/'summary.csv',index=False)
 (a.outdir/'SHA256SUMS').write_text(''.join(sha(q)+'  '+q.name+'\n' for q in sorted(a.outdir.iterdir()) if q.is_file()))
 print('L17_COMPLETE',json.dumps(result['lead']),flush=True)
 for r in result['results']:print('L17_RESULT',json.dumps(r),flush=True)
if __name__=='__main__':main()
