#!/usr/bin/env python3
"""P23: baseline-observed repeat-selection sensitivity; aggregate outputs only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.exceptions import ConvergenceWarning
import f1_change_synchrony_v1 as P
import f1_longitudinal_extension_v1 as Q

VERSION = 'f1_selection_weighting_v1.0 2026-10-07'
PRED = ['age', 'sex_male', 'bmi', 'sbp_sit', 'dbp_sit', 'antihtn',
        'pwv_raw', 'imt_raw', 'abi_min_raw']
MODES = ['supported_unweighted', 'ipw_untrimmed', 'ipw_cap95', 'ipw_cap10']
DEPS = {
    'atlas': '62344144b83db3f2f7e6492bd05fd7616d48a442bba88134aa1e0360110fa253',
    'omics': '076251b895fc6314af6c45a35c23920b1715c26d4a718cc6b68c90fb828a6ae4',
    'p16': 'b0d6cada571934ab7ef3171a67cc1e53dd3e0275a83d43f0b628b7f7a7cb3730',
    'p17': '2782cb44eff943da7559be791cd77a863942bc5252ece2f6b8306ec735de12f0'}


def target_frame(fr, relations):
    common, inv = P.prepare(fr)
    base = fr.loc[fr.research_stage == P.A.MAIN_STAGES[0]].copy()
    base['participant_id'] = base.participant_id.astype(str)
    raw = [b + '_raw' for b in P.BEDS]
    if any(c not in base for c in PRED):
        return None, {'status': 'NA_baseline_schema', 'original_common': inv['primary_common']}
    base = base.replace([np.inf, -np.inf], np.nan).dropna(subset=raw + ['age', 'sex_male'])
    t = P.O.day(base.visit_date)
    base['quarter'] = (t.dt.year.astype('Int64').astype(str) + 'Q' + t.dt.quarter.astype('Int64').astype(str)).to_numpy()
    base = base.loc[t.notna().to_numpy()].copy()
    base['selected'] = base.participant_id.isin(set(common.participant_id.astype(str))).astype(int)
    attendance = set(fr.loc[fr.research_stage == P.A.MAIN_STAGES[1], 'participant_id'].astype(str))
    base['attended'] = base.participant_id.isin(attendance).astype(int)
    quarters = []
    support = []
    for quarter, b in base.groupby('quarter', sort=True):
        yes = int(b.selected.sum()); no = len(b) - yes
        keep = yes >= 20 and no >= 20
        quarters.append({'quarter': quarter, 'baseline': P.count(len(b)),
                         'attended': P.count(b.attended.sum()), 'selected': P.count(yes),
                         'not_selected': P.count(no), 'supported': keep})
        if keep: support.append(quarter)
    audit = {'status': 'ready', 'baseline_eligible': P.count(len(base)),
             'original_common': inv['primary_common'], 'quarter_counts': quarters,
             'supported_quarters': len(support), 'selection_definition': 'complete three-bed repeat with original P16 interval/covariate gates'}
    base = base.loc[base.quarter.isin(support)].reset_index(drop=True)
    # A left merge preserves the baseline target; follow-up is used only as an outcome.
    d = base.merge(common, on='participant_id', how='left', suffixes=('', '_paired'), validate='one_to_one')
    if relations is None: audit['status'] = 'NA_relationship_metadata'; return None, audit
    d['group'] = [relations.get(str(p), str(p)) for p in d.participant_id]
    audit.update(supported_baseline=P.count(len(d)), supported_selected=P.count(d.selected.sum()),
                 supported_groups=P.count(d.group.nunique()))
    return d, audit


def probabilities(d):
    y = d.selected.to_numpy(int); groups = d.group.to_numpy(str)
    if len(np.unique(groups)) < 100 or y.sum() < 200 or (1-y).sum() < 20:
        raise ValueError('baseline/selected/group gate')
    x = d[PRED].to_numpy(float)
    quarter = pd.get_dummies(pd.Categorical(d.quarter, categories=sorted(d.quarter.unique())), dtype=float).to_numpy()[:, 1:]
    pred = np.full(len(d), np.nan)
    for train, test in GroupKFold(5).split(x, y, groups):
        if min(y[train].sum(), (1-y[train]).sum()) < 20:
            raise ValueError('training class gate')
        a = x[train].copy(); b = x[test].copy()
        med = np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0 for v in a.T])
        ma = ~np.isfinite(a); mb = ~np.isfinite(b)
        a = np.where(ma, med, a); b = np.where(mb, med, b)
        mean = a.mean(axis=0); sd = a.std(axis=0); sd[sd < 1e-12] = 1
        a = np.column_stack([(a-mean)/sd, ma.astype(float), quarter[train]])
        b = np.column_stack([(b-mean)/sd, mb.astype(float), quarter[test]])
        fit = LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=1000)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            fit.fit(a, y[train])
        pred[test] = fit.predict_proba(b)[:, 1]
    if not np.isfinite(pred).all(): raise ValueError('invalid probabilities')
    return pred


def weights(d, prob):
    s = d.selected.to_numpy(bool)
    w = s.mean() / np.maximum(prob[s], 1e-6)  # numerical floor only; not a positivity repair
    return [np.ones(s.sum()), w, np.minimum(w, np.percentile(w, 95)), np.minimum(w, 10)]


def ess(w): return float(w.sum()**2 / np.dot(w, w))


def weighted_change(d, w):
    y = np.column_stack([P.A.ORIENT[b]*(d[b+'_1']-d[b+'_0'])/d.years for b in P.BEDS])
    cov = ['age_0', 'sex_male_0', 'years'] + [b+'_0' for b in P.BEDS]
    x = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cov])
    x = x[:, np.r_[True, x[:, 1:].std(axis=0) > 1e-12]]
    w = w/w.mean(); sw = np.sqrt(w)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.linalg.matrix_rank(x*sw[:, None]) != x.shape[1]:
        raise ValueError('weighted design')
    e = y - x @ np.linalg.lstsq(x*sw[:, None], y*sw[:, None], rcond=None)[0]
    e -= np.average(e, weights=w, axis=0)
    v = (e*w[:, None]).T @ e / w.sum()
    if np.any(np.diag(v) < 1e-12): raise ValueError('degenerate variance')
    cor = v / np.sqrt(np.outer(np.diag(v), np.diag(v)))
    return np.array([cor[i, j] for i, j in P.PAIRS])


def balance(d, w):
    out = {}; s = d.selected.to_numpy(bool)
    for c in PRED:
        x = d[c].to_numpy(float); ok = np.isfinite(x); chosen = ok & s
        if ok.sum() < 20 or chosen.sum() < 20 or np.std(x[ok]) < 1e-12: continue
        out[c] = [(np.average(x[chosen], weights=a[ok[s]])-x[ok].mean())/x[ok].std() for a in w]
    return out


def summary(point, draws):
    rr = np.asarray(draws); z = np.arctanh(np.clip(point, -.999999, .999999))
    se = np.arctanh(np.clip(rr, -.999999, .999999)).std(axis=0, ddof=1)
    if np.any(se <= 0): return {'status': 'NA_boot_SE'}
    p = 2*stats.norm.sf(abs(z/se)); bound = np.arctanh(.2)
    tost = np.maximum(stats.norm.sf((z+bound)/se), stats.norm.cdf((z-bound)/se))
    rows = {k: {'r': point[j], 'ci95': np.percentile(rr[:, j], [2.5, 97.5]),
                'ci90': np.percentile(rr[:, j], [5, 95]), 'q_zero3': P.A.bh(p)[j],
                'p_tost': tost[j]} for j, k in enumerate(P.LABELS)}
    return {'status': 'estimated', 'pairs': rows, 'allTOST': bool(np.all(tost < .05))}


def analyse(fr, relations, boot=500, seed=20261007):
    d, audit = target_frame(fr, relations); r = {'audit': audit, 'models': {}, 'bootstrap_attempts': boot}
    if d is None: return r
    s = d.selected.to_numpy(bool); selected = d.loc[s].reset_index(drop=True)
    r['supported_selected_groups'] = P.count(selected.group.nunique())
    if len(selected) < 200 or selected.group.nunique() < 100:
        r['status'] = 'NA_selected_n200_groups100'; return r
    try: prob = probabilities(d); ww = weights(d, prob)
    except (ValueError, ConvergenceWarning): r['status'] = 'NA_selection_fit'; return r
    r['probability_q05_25_50_75_95'] = P.A.qtiles(pd.Series(prob))
    r['selected_probability_q05_25_50_75_95'] = P.A.qtiles(pd.Series(prob[s]))
    r['baseline_probability_lt02'] = P.count((prob < .02).sum())
    r['selected_probability_lt02'] = P.count((prob[s] < .02).sum())
    r['balance_standardised_mean_difference_by_mode'] = balance(d, ww)
    active = []
    for name, w in zip(MODES, ww):
        m = {'n': P.count(len(selected)), 'ESS': P.count(ess(w)), 'weight_q05_25_50_75_95': P.A.qtiles(pd.Series(w))}
        r['models'][name] = m
        if ess(w) < 200: m['status'] = 'NA_ESS_lt_200'; active.append(False)
        else:
            try: m['_point'] = weighted_change(selected, w); active.append(True)
            except ValueError: m['status'] = 'NA_outcome_design'; active.append(False)
    blocks, _ = Q.group_info(d, relations); rng = np.random.default_rng(seed)
    draws = [[] for _ in MODES]; failed = [0 for _ in MODES]
    for j in range(boot):
        b = d.iloc[Q.group_draw(blocks, rng)].reset_index(drop=True)
        try: p = probabilities(b); wb = weights(b, p); bs = b.loc[b.selected.astype(bool)]
        except (ValueError, ConvergenceWarning):
            for k, a in enumerate(active): failed[k] += int(a)
            continue
        for k, a in enumerate(active):
            if not a: continue
            try: draws[k].append(weighted_change(bs, wb[k]))
            except (ValueError, np.linalg.LinAlgError): failed[k] += 1
        if (j+1) % 100 == 0: print('P23 bootstrap %d/%d' % (j+1, boot), flush=True)
    for k, name in enumerate(MODES):
        m = r['models'][name]
        if not active[k]: continue
        m.update(B_used=len(draws[k]), B_failed=failed[k]); point = m.pop('_point')
        if len(draws[k]) < .95*boot: m['status'] = 'NA_boot_fraction'; continue
        m.update(summary(point, draws[k]))
    r['status'] = 'COMPLETE'
    return r


def digest(r):
    lines = ['F1 P23 | '+VERSION, 'Observed-baseline selection sensitivity; no unmeasured-selection or biological-independence claim.',
             'Target: eligible baseline three raw measures in quarters with >=20 selected AND >=20 nonselected.',
             'P16 residualisation; family5fold selection fit refitted in each family bootstrap; gates n200/g100/ESS200.',
             'Primary untrimmed IPW; cap95/cap10 sensitivities retained; atlas transform fixed here.']
    for k,v in r['audit'].items():
        if k not in ['quarter_counts', 'selection_definition']: lines.append('Audit '+k+' '+str(v))
    for v in r['audit'].get('quarter_counts', []): lines.append('Quarter '+json.dumps(v, separators=(',', ':')))
    for key in ['status', 'supported_selected_groups', 'probability_q05_25_50_75_95', 'selected_probability_q05_25_50_75_95', 'baseline_probability_lt02', 'selected_probability_lt02']:
        if key in r: lines.append(key+' '+str(r[key]))
    for name, m in r['models'].items():
        lines.append(name+' '+json.dumps({k:v for k,v in m.items() if k not in ['pairs','weight_q05_25_50_75_95']}, separators=(',', ':')))
        lines.append(name+' weights '+str(m['weight_q05_25_50_75_95']))
        for label, v in m.get('pairs', {}).items():
            lines.append(' '+label+' r='+str(v['r'])+' CI95='+str(v['ci95'])+' q='+str(v['q_zero3'])+' TOST='+str(v['p_tost'])+' CI90='+str(v['ci90']))
    for c, v in r.get('balance_standardised_mean_difference_by_mode', {}).items(): lines.append('Balance '+c+' '+str(v))
    lines.append('F1_P23_DONE errors=[]')
    if len(lines)>120 or any(len(v)>220 for v in lines): raise ValueError('digest size contract')
    return '\n'.join(lines)+'\n'


def main():
    ap = argparse.ArgumentParser()
    for c in ['root', 'out', 'atlas', 'omics', 'p16', 'p17']: ap.add_argument('--'+c, required=True)
    ap.add_argument('--boot', type=int, default=500); ap.add_argument('--seed', type=int, default=20261007)
    a = ap.parse_args()
    if Path(a.out).exists() or a.boot < 20: raise SystemExit('existing output or invalid bootstrap')
    for c, h in DEPS.items():
        if hashlib.sha256(Path(getattr(a, c)).read_bytes()).hexdigest()!=h: raise SystemExit('dependency mismatch '+c)
    if Path(a.p16).resolve()!=Path(P.__file__).resolve() or Path(a.p17).resolve()!=Path(Q.__file__).resolve():
        raise SystemExit('import/dependency path mismatch')
    P.A=P.load(a.atlas,'atlas_p23'); P.O=P.load(a.omics,'omics_p23'); P.O.A=P.A
    start=time.time(); fr=P.A.build_frame(a.root,None)[0]
    r=analyse(fr,P.A.relation_components(a.root),a.boot,a.seed)
    r.update(version=VERSION,runtime_s=time.time()-start,seed_label='2026-10-07')
    r=P.clean(r); js=json.dumps(r,sort_keys=True,allow_nan=False); dg=digest(r)
    ids=set(fr.participant_id.astype(str)); P.O.privacy_check(js,ids); P.O.privacy_check(dg,ids)
    out=Path(a.out); out.mkdir(); (out/'results.json').write_text(js+'\n'); (out/'digest.txt').write_text(dg)
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in ['results.json','digest.txt']))
    print('F1_P23_DONE errors=[] runtime=%.1fs'%(time.time()-start),flush=True)
    print('digest lines=%d sq12=%s'%(dg.count('\n'),hashlib.sha256(re.sub(' +',' ',dg).encode()).hexdigest()[:12]))


if __name__=='__main__': main()
