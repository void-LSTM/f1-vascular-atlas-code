#!/usr/bin/env python3
"""F1 joint-network grouped validation. All participant arrays stay in HPP memory.

The primary estimand averages losses over repeats, not ensemble predictions.
Returned intervals condition on fitted OOF models; no full-refit CI claim.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold

TASKS = ['pwv', 'imt', 'abi', 'retina_level', 'htn_objective', 'sbp_followup', 'dbp_followup']
MODELS = ['clinical_own', 'independent_linear', 'independent_network',
          'joint_network', 'joint_no_crossvascular']
SEEDS = [20261012, 20261023, 20261104]
BOOT = 1000

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def count(n):
    return '<20' if n < 20 else int(round(float(n) / 10) * 10)

def preprocess(train, test, fit_mask=None):
    """Transform all rows using only the declared eligible training subset.

    Shared X uses the training union. Private/head-specific designs use only
    that head's observed training cases, so other heads' availability or future
    query times cannot alter this head's imputation, column set or ridge scale.
    """
    train, test = np.asarray(train, float), np.asarray(test, float)
    fit = train if fit_mask is None else train[np.asarray(fit_mask, bool)]
    if not len(fit):
        raise ValueError('preprocess_empty_training_subset')
    keep = np.isfinite(fit).any(axis=0)
    train, test = train[:, keep], test[:, keep]
    fit = fit[:, keep]
    if train.shape[1] == 0:
        return train, test
    median = np.nanmedian(np.where(np.isfinite(fit), fit, np.nan), axis=0)
    missing_train, missing_test = ~np.isfinite(train), ~np.isfinite(test)
    a = np.where(missing_train, median, train)
    b = np.where(missing_test, median, test)
    # Missing indicators are established in training, never by test availability.
    missing_fit = ~np.isfinite(fit)
    indicators = missing_fit.any(axis=0)
    fitted = np.where(missing_fit, median, fit)
    fitted = np.column_stack([fitted, missing_fit[:, indicators].astype(float)])
    a = np.column_stack([a, missing_train[:, indicators].astype(float)])
    b = np.column_stack([b, missing_test[:, indicators].astype(float)])
    mean, sd = fitted.mean(axis=0), fitted.std(axis=0)
    variable = sd > 1e-10
    return (a[:, variable] - mean[variable]) / sd[variable], (b[:, variable] - mean[variable]) / sd[variable]

def fold_labels(groups, seed):
    labels = np.unique(groups)
    rng = np.random.default_rng(seed)
    lookup = dict(zip(labels, rng.permutation(len(labels))))
    shuffled = np.asarray([lookup[g] for g in groups])
    fold = np.full(len(groups), -1, int)
    for k, (_, test) in enumerate(GroupKFold(5).split(shuffled, groups=shuffled)):
        fold[test] = k
    if any(len(np.unique(fold[groups == g])) != 1 for g in labels):
        raise AssertionError('family_split')
    return fold

def eligibility(data, folds):
    mask, y, groups = data['mask'], data['Y'], data['groups']
    active, audit = [], {}
    for t, name in enumerate(data['task_names']):
        m = mask[:, t]
        info = {'n': count(m.sum()), 'groups': count(len(np.unique(groups[m])))}
        ok = m.sum() >= 250 and len(np.unique(groups[m])) >= 100
        binary = data['kinds'][t] == 'binary'
        if binary:
            e = int(y[m, t].sum())
            info['events'] = count(e)
            ok &= min(e, int(m.sum()) - e) >= 80
        minimum_n, minimum_groups, minimum_events = [], [], []
        for fold in folds:
            for f in range(5):
                tr = m & (fold != f)
                minimum_n.append(int(tr.sum()))
                minimum_groups.append(len(np.unique(groups[tr])))
                ok &= tr.sum() >= 200 and len(np.unique(groups[tr])) >= 50
                ok &= np.std(y[tr, t]) > 1e-10 if tr.any() else False
                if binary:
                    ne = min(y[tr, t].sum(), tr.sum() - y[tr, t].sum())
                    minimum_events.append(int(ne))
                    ok &= ne >= 50
        info.update(min_training_n=count(min(minimum_n)), min_training_groups=count(min(minimum_groups)),
                    status='eligible' if ok else 'NA_fixed_information_gate')
        if binary:
            info['min_training_class_n'] = count(min(minimum_events))
        audit[name] = info
        if ok:
            active.append(t)
    return active, audit

def all_information(data, t):
    names = list(data['feature_names'])
    values = [data['X']]
    own_name = {'pwv': 'pwv', 'imt': 'imt', 'abi': 'adverse_abi'}
    for j, name in enumerate(data['skip_names'][t]):
        canonical = own_name.get(data['task_names'][t], name) if name == 'own_baseline' else name
        if canonical in names:
            # Names alone must not conceal differing content or orientation.
            observed = data['mask'][:, t]
            source = data['X'][observed, names.index(canonical)]
            other = data['Z'][t][observed, j]
            if not np.allclose(source, other, rtol=1e-10, atol=1e-12, equal_nan=True):
                raise ValueError('duplicate_predictor_values_disagree_' + canonical)
        else:
            names.append(canonical)
            values.append(data['Z'][t][:, j:j + 1])
    return np.column_stack(values)

def convergence(model):
    """Strict fit receipt, adapted to documented core result fields."""
    keys = ['converged','optimizer_status','iterations','function_evaluations',
            'initial_objective','objective','max_abs_gradient']
    receipt = {k:model[k] for k in keys if k in model}
    success = model.get('converged', False)
    return bool(success), receipt

def cross_validate(data, core, active, folds):
    n, T, R = len(data['X']), len(active), len(folds)
    prediction = {name: np.full((R, n, T), np.nan) for name in MODELS}
    normalizer = np.full((R, n, T), np.nan)
    receipts = []
    kinds = [data['kinds'][t] for t in active]
    task_names = [data['task_names'][t] for t in active]
    mask = data['mask'][:, active]
    y = data['Y'][:, active]
    full = [all_information(data, t) for t in active]
    no_vasc = [i for i, name in enumerate(data['feature_names'])
               if not any(k in name.lower() for k in ['pwv', 'imt', 'abi', 'art_w', 'retina'])]
    for r, fold in enumerate(folds):
        for f in range(5):
            train, test = fold != f, fold == f
            X, Xtest = preprocess(data['X'][train], data['X'][test])
            Xno, Xnotest = preprocess(data['X'][train][:, no_vasc], data['X'][test][:, no_vasc])
            zpair = [preprocess(data['Z'][t][train], data['Z'][t][test], mask[train, j])
                     for j, t in enumerate(active)]
            fpair = [preprocess(z[train], z[test], mask[train, j]) for j, z in enumerate(full)]
            Z, Ztest = [a for a, b in zpair], [b for a, b in zpair]
            F, Ftest = [a for a, b in fpair], [b for a, b in fpair]
            yy = y[train].copy()
            center, scale = np.zeros(T), np.ones(T)
            for j, kind in enumerate(kinds):
                observed = mask[train, j]
                if kind == 'continuous':
                    center[j], scale[j] = yy[observed, j].mean(), yy[observed, j].std()
                    yy[observed, j] = (yy[observed, j] - center[j]) / scale[j]
                    normalizer[r, test, j] = scale[j] ** 2
                else:
                    prevalence = yy[observed, j].mean()
                    normalizer[r, test, j] = prevalence * (1 - prevalence)
            # NaN labels remain masked and must never affect the objective.
            fit_seed = SEEDS[r] + f
            for name in MODELS:
                if name == 'independent_network':
                    out = np.full((test.sum(), T), np.nan)
                    for j in range(T):
                        model = core.fit_joint(X, [Z[j]], yy[:, j:j+1], mask[train, j:j+1], [kinds[j]],
                                               seed=fit_seed, hidden=4, l2=.01, task_names=[task_names[j]])
                        success, receipt = convergence(model)
                        receipts.append({'repeat':r + 1,'fold':f + 1,'model':name,'head':task_names[j],**receipt})
                        if not success:
                            raise RuntimeError('nonconverged_' + name + '_' + task_names[j])
                        out[:, j] = core.predict_joint(model, Xtest, [Ztest[j]])[:, 0]
                else:
                    xx, xt, zz, zt, hidden = X, Xtest, Z, Ztest, 4
                    if name in ['clinical_own', 'independent_linear']:
                        xx, xt, hidden = np.empty((train.sum(), 0)), np.empty((test.sum(), 0)), 0
                        if name == 'independent_linear':
                            zz, zt = F, Ftest
                    elif name == 'joint_no_crossvascular':
                        xx, xt = Xno, Xnotest
                    model = core.fit_joint(xx, zz, yy, mask[train], kinds, seed=fit_seed,
                                           hidden=hidden, l2=.01, task_names=task_names)
                    success, receipt = convergence(model)
                    receipts.append({'repeat':r + 1,'fold':f + 1,'model':name,**receipt})
                    if not success:
                        raise RuntimeError('nonconverged_' + name)
                    out = core.predict_joint(model, xt, zt)
                for j, kind in enumerate(kinds):
                    if kind == 'continuous':
                        out[:, j] = out[:, j] * scale[j] + center[j]
                prediction[name][r, test] = out
            print('F1_NETWORK fold_complete repeat=%d fold=%d' % (r + 1, f + 1), flush=True)
    return prediction, normalizer, receipts

def weighted_auc(y, p, w):
    order = np.argsort(p, kind='mergesort')
    yy, pp, ww = y[order], p[order], w[order]
    _, first = np.unique(pp, return_index=True)
    positives = np.add.reduceat(ww * yy, first)
    negatives = np.add.reduceat(ww * (1 - yy), first)
    denominator = positives.sum() * negatives.sum()
    return float(np.sum(positives * (np.cumsum(negatives) - .5 * negatives)) / denominator) if denominator > 0 else np.nan

def binary_calibration(y, p, w):
    """Unpenalized logistic recalibration; undefined fits remain unavailable."""
    lp = logit(np.clip(p, 1e-8, 1 - 1e-8))
    positive, negative = (y == 1) & (w > 0), (y == 0) & (w > 0)
    if not positive.any() or not negative.any():
        return np.nan, np.nan
    center = np.average(lp, weights=w)
    scale = np.sqrt(np.average((lp-center)**2, weights=w))
    if scale < 1e-10:
        return np.nan, np.nan
    # Complete or quasi-complete separation has no finite slope MLE.
    if (lp[negative].max() <= lp[positive].min() or
            lp[positive].max() <= lp[negative].min()):
        return np.nan, np.nan
    a = np.column_stack([np.ones(len(lp)), (lp-center)/scale])
    normalized_w = w / w.sum()
    def objective(beta):
        z = a @ beta
        loss = np.dot(normalized_w, np.logaddexp(0., z)-y*z)
        gradient = a.T @ (normalized_w*(expit(z)-y))
        return float(loss), gradient
    initial = np.array([logit(np.average(y, weights=w)), 0.])
    fitted = minimize(objective, initial, jac=True, method='BFGS',
                      options={'maxiter':200, 'gtol':1e-8})
    loss, gradient = objective(fitted.x)
    if (not np.isfinite(loss) or not np.isfinite(fitted.x).all() or
            np.max(np.abs(gradient)) > 1e-7):
        return np.nan, np.nan
    slope = fitted.x[1]/scale
    return float(fitted.x[0]-slope*center), float(slope)

def one_metrics(y, predictions, w, kind):
    mean = np.average(y, weights=w)
    mse = float(np.mean([np.average((y - p) ** 2, weights=w) for p in predictions]))
    result = {}
    if kind == 'continuous':
        variance = np.average((y - mean) ** 2, weights=w)
        result.update(mse=mse, rmse=float(np.sqrt(mse)),
                      mae=float(np.mean([np.average(abs(y-p), weights=w) for p in predictions])),
                      r2=float(1-mse/variance), mean_error=float(np.mean([np.average(p-y,weights=w) for p in predictions])))
        slopes, intercepts = [], []
        for p in predictions:
            mp = np.average(p, weights=w)
            vp = np.average((p-mp)**2, weights=w)
            slope = np.average((p-mp)*(y-mean), weights=w)/vp if vp>1e-12 else np.nan
            slopes.append(slope); intercepts.append(mean-slope*mp)
    else:
        pclip = np.clip(predictions, 1e-8, 1-1e-8)
        result.update(brier=mse, auc=float(np.mean([weighted_auc(y,p,w) for p in predictions])),
                      logloss=float(np.mean([np.average(-y*np.log(p)-(1-y)*np.log1p(-p),weights=w) for p in pclip])),
                      expected_over_observed=float(np.mean([np.average(p,weights=w)/mean for p in predictions])))
        estimates = [binary_calibration(y,p,w) for p in predictions]
        intercepts, slopes = zip(*estimates)
    result.update(calibration_intercept=float(np.mean(intercepts)),calibration_slope=float(np.mean(slopes)))
    return result

def score(data, active, prediction, normalizer, weights=None, detail=True):
    weights = np.ones(len(data['X'])) if weights is None else weights
    result = {}
    for model, pp in prediction.items():
        row = {'heads':{},'macro_normalized_error':0.}
        for j,t in enumerate(active):
            use = data['mask'][:,t] & (weights>0)
            y, p, w = data['Y'][use,t], pp[:,use,j], weights[use]
            scaled = np.mean((p-y[None,:])**2 / normalizer[:,use,j],axis=0)
            value = float(np.average(scaled,weights=w))
            metrics = one_metrics(y,p,w,data['kinds'][t]) if detail else {}
            metrics['native_squared_error'] = float(np.average(np.mean((p-y[None,:])**2,axis=0),weights=w))
            metrics['normalized_error'] = value
            row['heads'][data['task_names'][t]] = metrics
        row['macro_normalized_error'] = float(np.mean([z['normalized_error'] for z in row['heads'].values()]))
        result[model] = row
    return result

def interval(point, values):
    v=np.asarray(values,float); v=v[np.isfinite(v)]
    if len(v)<.8*BOOT:
        return {'estimate':point,'status':'NA_insufficient_bootstrap','valid':count(len(v))}
    limits=np.quantile(v,[.025,.975])
    return {'estimate':point,'ci95':limits.tolist(),'valid':count(len(v)),
            'bootstrap_median':float(np.median(v)), 'point_outside_interval':bool(point<limits[0] or point>limits[1]),
            'status':'conditional_frozen_OOF'}

def compare_bootstrap(data, active, prediction, normalizer, point):
    rng=np.random.default_rng(20261031)
    labels,inverse=np.unique(data['groups'],return_inverse=True)
    comparisons=[('independent_linear','joint_network'),('independent_network','joint_network'),
                 ('clinical_own','joint_network'),('joint_no_crossvascular','joint_network')]
    draws={a+'_minus_'+b:{'macro':[],'heads':{data['task_names'][t]:[] for t in active},
                            'native':{data['task_names'][t]:[] for t in active}} for a,b in comparisons}
    for _ in range(BOOT):
        group_w=np.bincount(rng.integers(0,len(labels),len(labels)),minlength=len(labels))
        s=score(data,active,prediction,normalizer,group_w[inverse],detail=False)
        for a,b in comparisons:
            d=draws[a+'_minus_'+b]
            d['macro'].append(s[a]['macro_normalized_error']-s[b]['macro_normalized_error'])
            for name in d['heads']:
                d['heads'][name].append(s[a]['heads'][name]['normalized_error']-s[b]['heads'][name]['normalized_error'])
                d['native'][name].append(s[a]['heads'][name]['native_squared_error']-s[b]['heads'][name]['native_squared_error'])
    result={}
    for a,b in comparisons:
        key=a+'_minus_'+b; d=draws[key]
        result[key]={'macro':interval(point[a]['macro_normalized_error']-point[b]['macro_normalized_error'],d['macro']),
                     'heads':{name:interval(point[a]['heads'][name]['normalized_error']-point[b]['heads'][name]['normalized_error'],v) for name,v in d['heads'].items()}}
        for name,z in result[key]['heads'].items():
            z['native_error_reduction']=interval(point[a]['heads'][name]['native_squared_error']-point[b]['heads'][name]['native_squared_error'],d['native'][name])
    return result

def calibration_bins(data,active,prediction):
    result={}
    for model,pp in prediction.items():
        result[model]={}
        for j,t in enumerate(active):
            use=data['mask'][:,t]
            y=data['Y'][use,t]
            rows=[]
            for r in range(len(pp)):
                p=pp[r,use,j]
                edges=np.quantile(p,[.25,.5,.75])
                bins=np.searchsorted(edges,p,side='right')
                for k in range(4):
                    m=bins==k
                    row={'repeat':r+1,'bin':k+1,'n':count(m.sum())}
                    safe=m.sum()>=20
                    if data['kinds'][t]=='binary':
                        safe &= min(int(y[m].sum()),int(m.sum()-y[m].sum()))>=20
                    if safe:
                        row.update(predicted_mean=float(p[m].mean()),observed_mean=float(y[m].mean()))
                    else:
                        row['status']='suppressed_small_cell'
                    rows.append(row)
            result[model][data['task_names'][t]]=rows
    return result

def digest(result):
    def fmt(x):
        if not np.isfinite(x):
            return 'NA'
        if x==0:
            return '0'
        return ('%.4g'%x) if 1e-3<=abs(x)<1e5 else ('%.3e'%x)
    lines=['F1_JOINT_NETWORK genuine shared-trunk masked multitask analysis',
           'Research forecast; retinal LEVEL and clinical SBP/DBP phenotypes; disease HTN gate retained.',
           'CI conditional on frozen family-grouped OOF predictions; repeat-average loss.',
           'STATUS '+result['status']]
    for name,row in result['eligibility'].items():
        lines.append('ELIG '+name+' '+json.dumps(row,sort_keys=True,separators=(',',':')))
    if 'metrics' in result:
        for model,row in result['metrics'].items():
            lines.append('MACRO '+model+' error='+fmt(row['macro_normalized_error']))
            for name,m in row['heads'].items():
                keys=['normalized_error','mse','mae','r2','brier','auc']
                lines.append('METRIC '+model+'/'+name+' '+' '.join(k+'='+fmt(m[k]) for k in keys if k in m))
                lines.append('CAL '+model+'/'+name+' intercept='+fmt(m['calibration_intercept'])+' slope='+fmt(m['calibration_slope']))
        for name,c in result['comparisons'].items():
            for target,z in [('macro',c['macro'])]+list(c['heads'].items()):
                ci=z.get('ci95',[np.nan,np.nan])
                line='DELTA '+name+'/'+target+' d='+fmt(z['estimate'])+' CI=['+fmt(ci[0])+','+fmt(ci[1])+']'
                if 'native_error_reduction' in z:
                    v=z['native_error_reduction']; ni=v.get('ci95',[np.nan,np.nan])
                    line+=' native='+fmt(v['estimate'])+'['+fmt(ni[0])+','+fmt(ni[1])+']'
                lines.append(line)
    lines.append('F1_NETWORK_DONE')
    if len(lines)>120 or any(len(line)>220 for line in lines):
        raise ValueError('digest_size')
    return '\n'.join(lines)+'\n'

def readbacks(result, A):
    """Bounded aggregate pages; full model results remain inside the sandbox."""
    outputs={}
    lines=['F1_NETWORK aggregate coverage and inputs']
    def walk(value, prefix):
        if isinstance(value,dict):
            for key,item in value.items():walk(item,prefix+'.'+str(key))
        else:
            text=json.dumps(A.clean(value),sort_keys=True,separators=(',',':'))
            if len(prefix)+len(text)<210:
                lines.append(prefix+' '+text)
    walk(result['data_qc'],'qc')
    for page,at in enumerate(range(0,len(lines),100),1):
        outputs['coverage_p%d.txt'%page]='\n'.join(lines[at:at+100])+'\n'
    if 'calibration_bins' in result:
        for target in result['active_heads']:
            rows=['F1_NETWORK calibration '+target+'; individual repeats, approved aggregate bins']
            for model in ['independent_linear','joint_network']:
                for row in result['calibration_bins'][model][target]:
                    rows.append(model+' '+json.dumps(A.clean(row),sort_keys=True,separators=(',',':')))
            outputs['calibration_'+target+'.txt']='\n'.join(rows)+'\n'
        rows=['F1_NETWORK repeat-level macro error; no best-repeat selection']
        for r,summary in enumerate(result['repeat_metrics'],1):
            for model,record in summary.items():
                rows.append('repeat=%d model=%s error=%s'%(r,model,A.fnum(record['macro_normalized_error'])))
        outputs['repeat_metrics.txt']='\n'.join(rows)+'\n'
    for value in outputs.values():
        if len(value.splitlines())>120 or any(len(line)>220 for line in value.splitlines()):
            raise ValueError('readback_size')
    return outputs

def main():
    ap=argparse.ArgumentParser()
    for name in ['root','out','atlas','p21','p13','outcomes','p10']:
        ap.add_argument('--'+name,required=True)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise SystemExit('output_exists')
    here=Path(__file__).resolve().parent
    core=load(here/'atlas_joint_network_v1.py','jointcore')
    adapter=load(here/'network_data_clinical_v2.py','jointdata')
    start=time.time()
    data=adapter.build_data(args.root,args.atlas,args.p21,args.p13,args.outcomes,args.p10)
    data['groups']=np.asarray(data['groups']); data['mask']=np.asarray(data['mask'],bool)
    folds=[fold_labels(data['groups'],seed) for seed in SEEDS]
    active,audit=eligibility(data,folds)
    result={'analysis_id':'atlas_joint_vascular_clinical_phenotypes_v2','status':'NA_fixed_information_gate',
            'eligibility':audit,'data_qc':data['qc'],'units':dict(zip(data['task_names'],data['units'])),
            'n_union':count(len(data['X'])),'groups_union':count(len(np.unique(data['groups']))),
            'active_heads':[data['task_names'][t] for t in active],
            'uncertainty':'Paired family bootstrap of fixed out-of-fold predictions; conditional on fitted models and fixed folds.',
            'input_features':data['feature_names'],'specific_features':data['skip_names']}
    result['five_head_primary_available'] = all(name in result['active_heads'] for name in TASKS[:5])
    result['all_seven_heads_available'] = len(active)==len(data['task_names'])
    result['six_phenotype_secondary_available'] = all(name in result['active_heads'] for name in TASKS if name != 'htn_objective')
    result['scope'] = 'Exploratory extension to clinical BP phenotypes after disease-head information gate failure; preserves original five-head primary NA.'
    # Preserve a useful partial joint fit, but never replace the fixed five-head primary.
    if len(active)>=2:
        prediction,normalizer,receipts=cross_validate(data,core,active,folds)
        result['metrics']=score(data,active,prediction,normalizer)
        result['comparisons']=compare_bootstrap(data,active,prediction,normalizer,result['metrics'])
        result['calibration_bins']=calibration_bins(data,active,prediction)
        result['repeat_metrics']=[score(data,active,{k:v[r:r+1] for k,v in prediction.items()},normalizer[r:r+1]) for r in range(len(folds))]
        result['fit_diagnostics']=receipts
        result['status']='COMPLETE' if result['all_seven_heads_available'] else 'PARTIAL_HEAD_COVERAGE'
    result['runtime_seconds']=time.time()-start
    A=load(args.atlas,'export_atlas')
    # Real identifiers exist only for the existing privacy checker and are never serialized.
    js=json.dumps(A.clean(result),sort_keys=True,allow_nan=False)+'\n'
    dg=digest(result)
    exports={'results.json':js,'digest.txt':dg,**readbacks(result,A)}
    for value in exports.values():
        A.privacy_check(value,set(map(str,data['ids'])))
        if re.search(r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b',value):raise ValueError('fine_date')
    out.mkdir()
    for name,value in exports.items():
        (out/name).write_text(value)
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in exports))
    print('F1_NETWORK_DONE status=%s seconds=%.1f sq12=%s'%(result['status'],result['runtime_seconds'],hashlib.sha256(re.sub(' +',' ',dg).encode()).hexdigest()[:12]),flush=True)

if __name__=='__main__':main()
