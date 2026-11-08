#!/usr/bin/env python3
"""P24: delete relationship groups; re-fit immutable acquisition correction."""
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
import f1_longitudinal_extension_v1 as Q

VERSION='f1_correction_jackknife_v1.0 2026-10-07'
DEPS={
 'atlas':'62344144b83db3f2f7e6492bd05fd7616d48a442bba88134aa1e0360110fa253',
 'omics':'076251b895fc6314af6c45a35c23920b1715c26d4a718cc6b68c90fb828a6ae4',
 'p16':'b0d6cada571934ab7ef3171a67cc1e53dd3e0275a83d43f0b628b7f7a7cb3730',
 'p17':'2782cb44eff943da7559be791cd77a863942bc5252ece2f6b8306ec735de12f0'}


def restore_raw(fr):
    d=fr.copy()
    for c in P.A.CAL_BEDS:
        if c not in d: continue
        if c+'_raw' not in d: raise ValueError('missing immutable raw column')
        d[c]=d[c+'_raw']
    return d


def refit(raw,root):
    P.A.DIAG.clear()
    d=P.A.calendar_adjust(raw)
    return P.A.session_adjust(d,root,apply=P.A.SESSION_ADJUST)


def partition(fr,relations,k,seed):
    people=fr[['participant_id']].drop_duplicates().reset_index(drop=True)
    blocks,info=Q.group_info(people,relations)
    if blocks is None or len(blocks)<max(100,k): return None,info
    rng=np.random.default_rng(seed)
    parts=np.array_split(rng.permutation(len(blocks)),k)
    pid_to_part={str(people.participant_id.iloc[i]):j for j,a in enumerate(parts) for b in a for i in blocks[b]}
    tags=np.array([pid_to_part[str(p)] for p in fr.participant_id])
    sizes=np.array([sum(len(blocks[b]) for b in a) for a in parts])
    size_quantiles=P.A.qtiles(pd.Series(sizes))
    info.update(delete_groups=k,people_per_delete_group_q05_25_50_75_95={q:P.count(v) for q,v in size_quantiles.items()})
    if sizes.min()==0 or sizes.max()/sizes.min()>1.5:
        info['status']='NA_delete_group_imbalance'; return None,info
    return tags,info


def inference(point,replicates):
    rr=np.asarray(replicates);k=len(rr)
    z=np.arctanh(np.clip(point,-.999999,.999999));zz=np.arctanh(np.clip(rr,-.999999,.999999))
    # Centre at the full estimate, as in the standard DAGJK replication formula.
    se=np.sqrt((k-1)/k*np.sum((zz-z)**2,axis=0))
    if np.any(se<=0): return {'status':'NA_jackknife_SE'}
    p=2*stats.t.sf(abs(z/se),k-1);bound=np.arctanh(.2)
    tost=np.maximum(stats.t.sf((z+bound)/se,k-1),stats.t.cdf((z-bound)/se,k-1))
    rows={}
    for j,label in enumerate(P.LABELS):
        rows[label]={'r':point[j],'fisher_se':se[j],
          'ci95':np.tanh(z[j]+np.array([-1,1])*stats.t.ppf(.975,k-1)*se[j]),
          'ci90':np.tanh(z[j]+np.array([-1,1])*stats.t.ppf(.95,k-1)*se[j]),
          'q_zero3':P.A.bh(p)[j],'p_tost':tost[j]}
    return {'status':'estimated_approximate_jackknife','pairs':rows,'allTOST':bool(np.all(tost<.05))}


def analyse(fr,root,relations,k=50,seed=20261007):
    d,_=P.prepare(fr);blocks,commoninfo=Q.group_info(d,relations)
    r={'common':commoninfo,'delete_groups':k,'models':{}}
    if blocks is None: r['status']='NA_relationship_metadata';return r
    if len(d)<200 or len(blocks)<100: r['status']='NA_common_n200_groups100';return r
    raw=restore_raw(fr);rebuilt=refit(raw,root)
    if any(not np.allclose(fr[c],rebuilt[c],atol=1e-8,rtol=1e-8,equal_nan=True) for c in P.A.CAL_BEDS if c in fr):
        r['status']='NA_correction_reconstruction';return r
    re_d,_=P.prepare(rebuilt)
    if list(re_d.participant_id.astype(str))!=list(d.participant_id.astype(str)):
        r['status']='NA_reconstruction_roster';return r
    r['reconstruction']='PASS_all_device_beds_1e-8'
    original_kernel={c:bool(v.get('applied',False)) for c,v in P.A.DIAG.get('session_kernel',{}).items()}
    tags,info=partition(fr,relations,k,seed);r['calibration_partition']=info
    if tags is None: r['status']=info.get('status','NA_calibration_groups');return r
    point=P.arrays(d);fixed=[];full=[];failures={};switches={c:0 for c in P.A.CAL_BEDS}
    r['point_identity']='PASS' if np.allclose(P.arrays(re_d),point,atol=1e-10,rtol=1e-10) else 'FAIL'
    if r['point_identity']!='PASS':r['status']='NA_point_reconstruction';return r
    for j in range(k):
        keep=tags!=j
        try:
            f=fr.loc[keep].reset_index(drop=True);b=refit(raw.loc[keep].reset_index(drop=True),root)
            fd,_=P.prepare(f);bd,_=P.prepare(b);gb,_=Q.group_info(fd,relations)
            if len(fd)<200 or len(gb)<100:raise ValueError('retained sample gate')
            if list(fd.participant_id.astype(str))!=list(bd.participant_id.astype(str)):
                raise ValueError('retained roster mismatch')
            x=P.arrays(fd);y=P.arrays(bd)
            fixed.append(x);full.append(y)
            kernel=P.A.DIAG.get('session_kernel',{})
            for c in switches:
                switches[c]+=int(bool(kernel.get(c,{}).get('applied',False))!=original_kernel.get(c,False))
        except (ValueError,np.linalg.LinAlgError) as e:
            key=type(e).__name__;failures[key]=failures.get(key,0)+1
        print('P24 delete group %d/%d'%(j+1,k),flush=True)
    r.update(replicates_used=len(full),replicates_failed=sum(failures.values()),failure_types=failures,kernel_gate_switches=switches)
    if len(full)!=k:r['status']='NA_incomplete_delete_groups';return r
    r['models']['fixed_transform_same_deletions']=inference(point,fixed)
    r['models']['refitted_transform_same_deletions']=inference(point,full)
    r['refit_minus_fixed_replicate_q05_25_50_75_95']={label:P.A.qtiles(pd.Series(np.array(full)[:,j]-np.array(fixed)[:,j])) for j,label in enumerate(P.LABELS)}
    r['status']='COMPLETE'
    return r


def digest(r):
    lines=['F1 P24 | '+VERSION,'Immutable atlas v1.5 correction refit; same three observed conditional change pairs.',
      str(r['delete_groups'])+' balanced delete groups of whole relationship components; no duplicate measurements; gates n200/g100.',
      'Approximate DAGJK Fisher-z/t intervals, not a validated correction-error or clinic-day-independent model.',
      'Post-result sensitivity; original P16/P17 primary/multiplicity unchanged; fixed |r|<.20 intersection-union.',
      'Common '+json.dumps(r['common'],separators=(',',':'))]
    for key in ['status','reconstruction','point_identity','delete_groups','replicates_used','replicates_failed','failure_types','kernel_gate_switches']:
        if key in r:lines.append(key+' '+str(r[key]))
    for key,v in r.get('calibration_partition',{}).items():lines.append('Partition '+key+' '+str(v))
    for name,m in r['models'].items():
        lines.append(name+' '+m['status']+' allTOST='+str(m.get('allTOST','NA')))
        for label,v in m.get('pairs',{}).items():
            lines.append(' '+label+' r='+str(v['r'])+' SEz='+str(v['fisher_se'])+' CI95='+str(v['ci95'])+' q='+str(v['q_zero3'])+' TOST='+str(v['p_tost'])+' CI90='+str(v['ci90']))
    for key,v in r.get('refit_minus_fixed_replicate_q05_25_50_75_95',{}).items():lines.append('RefitMinusFixed '+key+' '+str(v))
    lines.append('F1_P24_DONE errors=[]')
    if len(lines)>120 or any(len(x)>220 for x in lines):raise ValueError('digest size contract')
    return '\n'.join(lines)+'\n'


def main():
    ap=argparse.ArgumentParser()
    for c in ['root','out','atlas','omics','p16','p17']:ap.add_argument('--'+c,required=True)
    ap.add_argument('--groups',type=int,default=50);ap.add_argument('--seed',type=int,default=20261007)
    a=ap.parse_args()
    if Path(a.out).exists() or a.groups<20:raise SystemExit('existing output or invalid delete groups')
    for c,h in DEPS.items():
        if hashlib.sha256(Path(getattr(a,c)).read_bytes()).hexdigest()!=h:raise SystemExit('dependency mismatch '+c)
    if Path(a.p16).resolve()!=Path(P.__file__).resolve() or Path(a.p17).resolve()!=Path(Q.__file__).resolve():
        raise SystemExit('import/dependency path mismatch')
    P.A=P.load(a.atlas,'atlas_p24');P.O=P.load(a.omics,'omics_p24');P.O.A=P.A
    start=time.time();fr=P.A.build_frame(a.root,None)[0]
    r=analyse(fr,a.root,P.A.relation_components(a.root),a.groups,a.seed)
    r.update(version=VERSION,runtime_s=time.time()-start,seed_label='2026-10-07')
    r=P.clean(r);js=json.dumps(r,sort_keys=True,allow_nan=False);dg=digest(r)
    ids=set(fr.participant_id.astype(str));P.O.privacy_check(js,ids);P.O.privacy_check(dg,ids)
    out=Path(a.out);out.mkdir();(out/'results.json').write_text(js+'\n');(out/'digest.txt').write_text(dg)
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['results.json','digest.txt']))
    print('F1_P24_DONE errors=[] runtime=%.1fs'%(time.time()-start),flush=True)
    print('digest lines=%d sq12=%s'%(dg.count('\n'),hashlib.sha256(re.sub(' +',' ',dg).encode()).hexdigest()[:12]))


if __name__=='__main__':main()
