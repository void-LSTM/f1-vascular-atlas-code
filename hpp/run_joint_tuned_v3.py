"""Bounded nested-family tuning of full-input residual multitask networks.

Every preprocessing/selection step uses only the current outer training set.
Participant arrays and process-pool IPC stay inside HPP; only approved aggregates export.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse, hashlib, importlib.util, json, multiprocessing, re, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold

SEEDS=[20261012,20261023,20261104]
NEURAL_GRID=[(h,l2) for h in [4,8,16] for l2 in [.001,.01,.1]]
LINEAR_GRID=[(0,l2) for l2 in [.001,.01,.1]]
MODELS=['fixed_linear','fixed_joint','tuned_linear','separate_residual','joint_residual']
MAXITER=3000
EXPECTED={'core':'47fd8d640cbfa0b0a584c1b5261e56663a8f6ebe7eae405a701b7cb88064df05',
          'adapter':'0a1a4be5e4b8d592cd9bf3db83b14f56d8720eb6ee26830c0e3baf1b1bc2e1b7',
          'evaluator':'cda5d3aa8201b920392cee6e737e3adc02ae5d5d9768ba70224e217a1cc48df3',
          'policy':'94151469629e32b7f5b64505d8b9b3e44533e67cc4ed6f04f0ee32c4f517c297'}

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def verified(path,kind):
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=EXPECTED[kind]:raise ValueError('frozen_hash_'+kind)
    return load(path,'tuning_'+kind)
def worker_init(core_path):
    global CORE
    CORE=verified(core_path,'core')

def grouped_folds(groups,seed,n=3):
    groups=np.asarray(groups);unique=np.unique(groups);rng=np.random.default_rng(seed)
    lookup=dict(zip(unique,rng.permutation(len(unique))))
    g=np.asarray([lookup[x] for x in groups]);fold=np.full(len(g),-1,int)
    for f,(_,test) in enumerate(GroupKFold(n).split(g,groups=g)):fold[test]=f
    return fold

def prepare(data,active,tr,te,evaluator):
    """Fresh split-specific transforms, including within every inner split."""
    mask=data['mask'][:,active];Y=data['Y'][:,active];kinds=[data['kinds'][t] for t in active]
    X,Xtest=evaluator.preprocess(data['X'][tr],data['X'][te])
    full=[evaluator.all_information(data,t) for t in active]
    fp=[evaluator.preprocess(z[tr],z[te],mask[tr,j]) for j,z in enumerate(full)]
    zp=[evaluator.preprocess(data['Z'][t][tr],data['Z'][t][te],mask[tr,j]) for j,t in enumerate(active)]
    yy,yv=Y[tr].copy(),Y[te].copy();center=np.zeros(len(active));scale=np.ones(len(active));den=np.ones(len(active))
    for j,k in enumerate(kinds):
        m=mask[tr,j]
        if m.sum()<100 or len(np.unique(data['groups'][tr][m]))<25:raise ValueError('inner_information_gate')
        if k=='continuous':
            center[j],scale[j]=yy[m,j].mean(),yy[m,j].std()
            if scale[j]<1e-10:raise ValueError('constant_training_target')
            yy[m,j]=(yy[m,j]-center[j])/scale[j]
            observed=mask[te,j];yv[observed,j]=(yv[observed,j]-center[j])/scale[j]
        else:
            pi=yy[m,j].mean()
            if min(yy[m,j].sum(),m.sum()-yy[m,j].sum())<20:raise ValueError('inner_binary_class_gate')
            den[j]=pi*(1-pi)
    return dict(X=X,Xtest=Xtest,F=[a for a,b in fp],Ftest=[b for a,b in fp],
                Z=[a for a,b in zp],Ztest=[b for a,b in zp],Y=yy,Ytest=yv,
                mask=mask[tr],test_mask=mask[te],kinds=kinds,
                tasks=[data['task_names'][t] for t in active],center=center,scale=scale,den=den)

def fit_model(core,p,h,l2,seed,head=None,private=False):
    ix=list(range(len(p['tasks']))) if head is None else [head]
    Z=p['Z'] if private else p['F'];Zt=p['Ztest'] if private else p['Ftest']
    X=p['X'] if h else np.empty((len(p['Y']),0))
    Xt=p['Xtest'] if h else np.empty((len(p['Ytest']),0))
    model=core.fit_joint(X,[Z[j] for j in ix],p['Y'][:,ix],p['mask'][:,ix],
                         [p['kinds'][j] for j in ix],seed=seed,hidden=h,l2=l2,
                         task_names=[p['tasks'][j] for j in ix],maxiter=MAXITER)
    receipt={k:model[k] for k in ['converged','optimizer_status','iterations','function_evaluations','objective','max_abs_gradient']}
    pred=core.predict_joint(model,Xt,[Zt[j] for j in ix]) if receipt['converged'] else None
    return pred,receipt

def validation_error(p,pred,ix):
    if pred is None:return np.full(len(ix),np.nan)
    loss=[]
    for k,j in enumerate(ix):
        m=p['test_mask'][:,j]
        loss.append(float(np.sum((pred[m,k]-p['Ytest'][m,j])**2/p['den'][j])))
    return np.asarray(loss)

def candidate_job(args):
    p,config,seed=args;h,l2=config;T=len(p['tasks']);ix=list(range(T));receipts=[]
    joint,q=fit_model(CORE,p,h,l2,seed);receipts.append(q)
    result={'joint':validation_error(p,joint,ix),'separate':np.full(T,np.nan),'receipts':receipts}
    if h:
        for j in range(T):
            pred,q=fit_model(CORE,p,h,l2,seed,head=j);receipts.append(q)
            result['separate'][j]=validation_error(p,pred,[j])[0]
    return result

def choose(scores,grid):
    valid=np.isfinite(scores)
    if not valid.any():raise ValueError('no_converged_inner_candidate')
    best=min((i for i in range(len(grid)) if valid[i]),key=lambda i:(scores[i],grid[i][0],-grid[i][1]))
    return grid[best],float(scores[best])

def select(data,active,tr,seed,evaluator,pool,neural_grid=NEURAL_GRID,linear_grid=LINEAR_GRID,inner_n=3):
    """No outer-test data are included in preparation or candidate scoring."""
    idx=np.flatnonzero(tr);fold=grouped_folds(data['groups'][tr],seed,inner_n);T=len(active)
    grid=list(linear_grid)+list(neural_grid);sums=np.zeros((len(grid),T));sep=np.zeros((len(neural_grid),T))
    total=np.zeros(T,int);fit_count=fail_count=0
    for f in range(inner_n):
        it,iv=np.zeros(len(tr),bool),np.zeros(len(tr),bool)
        it[idx[fold!=f]]=True;iv[idx[fold==f]]=True
        if np.any((it|iv)&~tr) or np.any(it&iv):raise AssertionError('outer_test_in_inner_split')
        if set(data['groups'][it])&set(data['groups'][iv]):raise AssertionError('inner_family_split')
        p=prepare(data,active,it,iv,evaluator);total+=p['test_mask'].sum(axis=0)
        jobs=[(p,c,seed+f) for c in grid]
        results=list(pool.map(candidate_job,jobs)) if pool is not None else [candidate_job(z) for z in jobs]
        for i,res in enumerate(results):
            sums[i]+=res['joint'];fit_count+=len(res['receipts']);fail_count+=sum(not q['converged'] for q in res['receipts'])
            if i>=len(linear_grid):sep[i-len(linear_grid)]+=res['separate']
    per_head=sums/total[None,:];sep/=total[None,:]
    lin,ls=choose(per_head[:len(linear_grid)].mean(axis=1),linear_grid)
    joint,js=choose(per_head[len(linear_grid):].mean(axis=1),neural_grid)
    independent=[choose(sep[:,j],neural_grid) for j in range(T)]
    return {'linear':lin,'joint':joint,'separate':[a for a,b in independent],
            'scores':{'linear':ls,'joint':js,'separate':[b for a,b in independent]},
            'inner_fit_count':fit_count,'inner_nonconverged':fail_count,
            'candidate_scores':{'linear':per_head[:len(linear_grid)].mean(axis=1).tolist(),
                                'joint':per_head[len(linear_grid):].mean(axis=1).tolist(),
                                'separate':sep.tolist()}}

def evaluate(data,core,evaluator,active,folds,pool,neural_grid=NEURAL_GRID,linear_grid=LINEAR_GRID,outer_folds=range(5)):
    T=len(active);N=len(data['X']);R=len(folds)
    prediction={m:np.full((R,N,T),np.nan) for m in MODELS};normal=np.full((R,N,T),np.nan)
    receipts=[];selections=[];inner_fits=inner_failed=0
    for r,fold in enumerate(folds):
        for f in outer_folds:
            tr,te=fold!=f,fold==f;seed=SEEDS[r]+f
            chosen=select(data,active,tr,seed+700,evaluator,pool,neural_grid,linear_grid)
            selections.append({'repeat':r+1,'fold':f+1,**chosen})
            inner_fits+=chosen['inner_fit_count'];inner_failed+=chosen['inner_nonconverged']
            p=prepare(data,active,tr,te,evaluator)
            normal[r,te]=np.array([p['scale'][j]**2 if k=='continuous' else p['den'][j] for j,k in enumerate(p['kinds'])])
            for name in MODELS:
                if name=='separate_residual':
                    out=np.full((te.sum(),T),np.nan)
                    for j,(h,l2) in enumerate(chosen['separate']):
                        pred,q=fit_model(core,p,h,l2,seed,head=j);receipts.append({'repeat':r+1,'fold':f+1,'model':name,'head':p['tasks'][j],'hidden':h,'l2':l2,**q})
                        if pred is None:raise RuntimeError('selected_outer_fit_nonconverged_'+name)
                        out[:,j]=pred[:,0]
                else:
                    h,l2,private=(0,.01,False) if name=='fixed_linear' else ((4,.01,True) if name=='fixed_joint' else (*chosen['linear'],False) if name=='tuned_linear' else (*chosen['joint'],False))
                    out,q=fit_model(core,p,h,l2,seed,private=private);receipts.append({'repeat':r+1,'fold':f+1,'model':name,'hidden':h,'l2':l2,**q})
                    if out is None:raise RuntimeError('selected_outer_fit_nonconverged_'+name)
                for j,k in enumerate(p['kinds']):
                    if k=='continuous':out[:,j]=out[:,j]*p['scale'][j]+p['center'][j]
                prediction[name][r,te]=out
            print('F1_TUNED outer_complete repeat=%d fold=%d inner_fits=%d rejected_fits=%d'%(r+1,f+1,chosen['inner_fit_count'],chosen['inner_nonconverged']),flush=True)
    return prediction,normal,receipts,selections,{'inner_fits':inner_fits,'inner_nonconverged':inner_failed,'outer_fits':len(receipts),'all_outer_converged':all(q['converged'] for q in receipts)}

def contrasts(data,active,pred,norm,metrics,evaluator):
    aliases={'clinical_own':'fixed_linear','joint_no_crossvascular':'fixed_joint',
             'independent_linear':'tuned_linear','independent_network':'separate_residual','joint_network':'joint_residual'}
    raw=evaluator.compare_bootstrap(data,active,{k:pred[v] for k,v in aliases.items()},norm,{k:metrics[v] for k,v in aliases.items()})
    return {aliases[k.split('_minus_')[0]]+'_minus_'+aliases[k.split('_minus_')[1]]:v for k,v in raw.items()}

def export(out,result,data,evaluator,A):
    js=json.dumps(A.clean(result),sort_keys=True,allow_nan=False)+'\n';dg=evaluator.digest(result)
    exports={'results.json':js,'digest.txt':dg}
    lines=['F1_TUNED candidate selection; model counts, not participant cells']
    for z in result['selections']:
        for m in ['linear','joint']:
            h,l2=z[m];lines.append('repeat=%d fold=%d model=%s hidden=%d l2=%g inner_error=%.5g'%(z['repeat'],z['fold'],m,h,l2,z['scores'][m]))
        for j,(h,l2) in enumerate(z['separate']):lines.append('repeat=%d fold=%d model=separate head=%s hidden=%d l2=%g inner_error=%.5g'%(z['repeat'],z['fold'],result['active_heads'][j],h,l2,z['scores']['separate'][j]))
    if len(lines)<=120:exports['selection.txt']='\n'.join(lines)+'\n'
    else:
        for i,at in enumerate(range(0,len(lines),100),1):exports['selection_p%d.txt'%i]='\n'.join(lines[at:at+100])+'\n'
    rows=['F1_TUNED repeat macro error; no best-repeat selection']
    for r,s in enumerate(result['repeat_metrics'],1):
        for m,v in s.items():rows.append('repeat=%d model=%s error=%s'%(r,m,A.fnum(v['macro_normalized_error'])))
    exports['repeat_metrics.txt']='\n'.join(rows)+'\n'
    for task in result['active_heads']:
        rows=['F1_TUNED calibration '+task]
        for m in ['tuned_linear','joint_residual']:
            for row in result['calibration_bins'][m][task]:rows.append(m+' '+json.dumps(A.clean(row),sort_keys=True,separators=(',',':')))
        exports['calibration_'+task+'.txt']='\n'.join(rows)+'\n'
    rows=['F1_TUNED aggregate coverage']
    def walk(d,prefix):
        if isinstance(d,dict):
            for k,v in d.items():walk(v,prefix+'.'+k)
        else:
            text=prefix+' '+json.dumps(A.clean(d),separators=(',',':'))
            if len(text)<220:rows.append(text)
    walk(result['data_qc'],'qc');walk(result['fit_summary'],'fits')
    for i,at in enumerate(range(0,len(rows),100),1):exports['coverage_p%d.txt'%i]='\n'.join(rows[at:at+100])+'\n'
    for n,s in exports.items():
        A.privacy_check(s,set(map(str,data['ids'])))
        if re.search(r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b',s):raise ValueError('fine_date')
        if n.endswith('.txt') and (len(s.splitlines())>120 or any(len(x)>220 for x in s.splitlines())):raise ValueError('export_bounds_'+n)
    out.mkdir()
    for n,s in exports.items():(out/n).write_text(s)
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in exports))
    return hashlib.sha256(dg.encode()).hexdigest()[:12]

def main():
    ap=argparse.ArgumentParser()
    for k in ['root','out','core','adapter','evaluator','policy','atlas','p21','p13','outcomes','p10']:ap.add_argument('--'+k,required=True)
    ap.add_argument('--counts-only',action='store_true')
    ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();out=Path(a.out)
    if out.exists():raise SystemExit('output_exists')
    core=verified(a.core,'core');d=verified(a.adapter,'adapter');e=verified(a.evaluator,'evaluator');start=time.time()
    policy=verified(a.policy,'policy');policy.apply(d)
    data=policy.build(d,a.root,a.atlas,a.p21,a.p13,a.outcomes,a.p10)
    folds=[e.fold_labels(data['groups'],s) for s in SEEDS];active,audit=e.eligibility(data,folds)
    if a.counts_only:
        A=e.load(a.atlas,'source_counts_export');result={'analysis_id':'recorded_stage_clinical_source_sensitivity_v3','status':'COUNTS_ONLY_COMPLETE','eligibility':audit,'htn_source_qc':data['qc']['htn'],'models_fitted':0,'original_five_head_primary_available':False,'runtime_seconds':time.time()-start}
        js=json.dumps(A.clean(result),sort_keys=True,allow_nan=False)+'\n';lines=['F1_STAGE_HTN source-availability sensitivity; no fitted model or onset-date claim']
        def walk(z,prefix):
            if isinstance(z,dict):
                for k,v in z.items():walk(v,prefix+'/'+k)
            else:lines.append(prefix+' '+json.dumps(A.clean(z),separators=(',',':')))
        walk(audit,'eligibility');walk(result['htn_source_qc'],'htn');lines+=['F1_STAGE_HTN_DONE'];dg='\n'.join(lines)+'\n'
        if len(lines)>120 or any(len(x)>220 for x in lines):raise ValueError('counts_digest_bounds')
        for s in [js,dg]:A.privacy_check(s,set(map(str,data['ids'])))
        out.mkdir();(out/'results.json').write_text(js);(out/'digest.txt').write_text(dg);(out/'SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['results.json','digest.txt']))
        print('F1_STAGE_HTN_DONE seconds=%.1f sha12=%s'%(result['runtime_seconds'],hashlib.sha256(dg.encode()).hexdigest()[:12]),flush=True);return
    if len(active)<2:raise ValueError('insufficient_active_heads')
    ctx=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=ctx,initializer=worker_init,initargs=(a.core,)) as pool:
        pred,norm,receipts,selections,fit_summary=evaluate(data,core,e,active,folds,pool)
    metrics=e.score(data,active,pred,norm)
    names=[data['task_names'][t] for t in active]
    result={'analysis_id':'atlas_nested_residual_tuning_v3','status':'COMPLETE' if len(active)==7 else 'PARTIAL_HEAD_COVERAGE',
            'active_heads':names,'eligibility':audit,'metrics':metrics,'data_qc':data['qc'],
            'n_union':e.count(len(data['X'])),'groups_union':e.count(len(np.unique(data['groups']))),
            'units':dict(zip(data['task_names'],data['units'])),'selections':selections,'fit_diagnostics':receipts,'fit_summary':fit_summary,
            'five_head_primary_available':False,
            'six_phenotype_secondary_available':all(t in names for t in data['task_names'] if data['kinds'][data['task_names'].index(t)]=='continuous'),
            'scope':'Exploratory revision motivated by inspected v2 results; declared stage-availability clinical sensitivity. Inner3fold selection,outer3x5 evaluation. Original strict disease primary NA preserved.',
            'grid':{'neural':NEURAL_GRID,'linear':LINEAR_GRID},'comparisons':contrasts(data,active,pred,norm,metrics,e),
            'calibration_bins':e.calibration_bins(data,active,pred),
            'repeat_metrics':[e.score(data,active,{k:v[r:r+1] for k,v in pred.items()},norm[r:r+1]) for r in range(len(folds))],
            'uncertainty':'Paired family bootstrap of frozen outer OOF predictions; conditional on fitted/tuned models and folds, not full-refit or independent design validation.',
            'runtime_seconds':time.time()-start}
    A=e.load(a.atlas,'tuning_export_atlas');sha=export(out,result,data,e,A)
    print('F1_TUNED_DONE seconds=%.1f status=%s sha12=%s'%(result['runtime_seconds'],result['status'],sha),flush=True)

if __name__=='__main__':main()
