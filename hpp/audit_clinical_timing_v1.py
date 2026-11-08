"""Counts-only audit of frozen nominal-visit HTN versus actual BP-assessment horizon.

No model fit, participant export or modification of the original labels/package.
The alternative is a measurement-horizon diagnostic, not a selected outcome.
"""
import argparse, hashlib, importlib.util, inspect, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED_ADAPTER='0a1a4be5e4b8d592cd9bf3db83b14f56d8720eb6ee26830c0e3baf1b1bc2e1b7'
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def frozen_funnel(adapter,*args):
    """Add counts to the verified original function; never alter its masks/labels."""
    fn=adapter.htn_labels;src=inspect.getsource(fn)
    assert src.count('    return y, qc')==1 and src.count('    for r in sources:')==1
    src=src.replace('def htn_labels(', 'def audit_frozen_htn_labels(',1)
    src=src.replace('    for r in sources:',"    uncertain_by_component = {}\n    for r in sources:",1)
    src=src.replace("        ambiguous |= set(r.loc[uncertain, 'participant_id'])",
                    "        uncertain_by_component[comp] = set(r.loc[uncertain, 'participant_id'])\n        ambiguous |= set(r.loc[uncertain, 'participant_id'])",1)
    addition="""    before_ambiguity = valid_horizon & ids.isin(eligible_ids) & (bp_valid | (objective == 1))
    qc['funnel'] = {
        'labels_before_ambiguity': count(before_ambiguity.sum()),
        'events_before_ambiguity': count((before_ambiguity & (objective == 1)).sum()),
        'labels_excluded_by_ambiguity': count((before_ambiguity & ids.isin(ambiguous)).sum()),
        'events_excluded_by_ambiguity': count((before_ambiguity & ids.isin(ambiguous) & (objective == 1)).sum()),
        'events_excluded_not_known_free': count((valid_horizon & ~ids.isin(eligible_ids) & (objective == 1)).sum()),
        'undated_source_events_excluded': {c: count((before_ambiguity & ids.isin(s) & (objective == 1)).sum()) for c,s in uncertain_by_component.items()}}
"""
    src=src.replace('    return y, qc',addition+'    return y, qc',1)
    namespace=dict(fn.__globals__);exec(compile(src,'<verified_frozen_htn_counts>','exec'),namespace)
    y,q=namespace['audit_frozen_htn_labels'](*args)
    return y,q['funnel']

def comparison(adapter,p,fr,stage_days,badkeys,root,A,O,P):
    original=adapter.htn_labels(p,fr,stage_days,badkeys,root,A,O,P)
    same,funnel=frozen_funnel(adapter,p,fr,stage_days,badkeys,root,A,O,P)
    if not np.array_equal(same,original[0],equal_nan=True):raise ValueError('instrumentation_changed_original_labels')
    nominal=P.day(p.visit_date_1)
    assessment=pd.concat([nominal,p.bp_hi_1],axis=1).max(axis=1)
    later=p.copy();later['visit_date_1']=assessment
    revised=adapter.htn_labels(later,fr,stage_days,badkeys,root,A,O,P)
    y0,y1=original[0],revised[0];count=adapter.count
    timed=(p.bp_lo_1>p.start+pd.Timedelta(days=.5*365.25))&(p.bp_hi_1<=p.start+pd.Timedelta(days=5*365.25))
    high=(p.sbp_sit_1>=140)|(p.dbp_sit_1>=90)
    after=p.bp_hi_1.gt(nominal)
    changes={
        'baseline_core_timed':count(len(p)),
        'actual_future_BP_timed':count(timed.sum()),
        'actual_future_BP_after_nominal_visit2':count((timed&after).sum()),
        'high_actual_future_BP_after_nominal_visit2':count((timed&after&high).sum()),
        'labels_gained':count((np.isfinite(y1)&~np.isfinite(y0)).sum()),
        'labels_lost':count((np.isfinite(y0)&~np.isfinite(y1)).sum()),
        'positive_labels_gained':count(((y1==1)&(y0!=1)).sum()),
        'positive_labels_lost':count(((y0==1)&(y1!=1)).sum()),
        'label_disagreements_on_common_observed':count((np.isfinite(y0)&np.isfinite(y1)&(y0!=y1)).sum())}
    qc={'original_nominal_visit2':original[1],'original_funnel':funnel,
        'diagnostic_actual_BP_assessment':revised[1],'paired_changes':changes,
        'alternative_rule':'horizon=max(actual_recorded_visit2,latest_contributing_visit2_BP_acquisition); original_landmark_and_known_free_rules_retained',
        'models_fitted':0,'original_labels_used_by_build_data':True,
        'disease_performance_available':False,'participant_arrays_exportable':False}
    return original[0],original[1],qc

def main():
    ap=argparse.ArgumentParser()
    for k in ['root','out','adapter','atlas','p21','p13','outcomes','p10']:ap.add_argument('--'+k,required=True)
    a=ap.parse_args();out=Path(a.out)
    if out.exists():raise SystemExit('output_exists')
    path=Path(a.adapter)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_ADAPTER:raise ValueError('adapter_hash')
    d=load(path,'frozen_clinical_adapter');original=d.htn_labels;captured={}
    class Frozen:
        htn_labels=staticmethod(original)
        count=staticmethod(d.count)
    def inspect(*args):
        y,q,extra=comparison(Frozen,*args);captured.update(extra);return y,q
    d.htn_labels=inspect;start=time.time()
    data=d.build_data(a.root,a.atlas,a.p21,a.p13,a.outcomes,a.p10)
    result={'analysis_id':'clinical_outcome_timing_audit_v1','status':'COUNTS_ONLY_COMPLETE',
            'scope':'Audit only; original disease gate and primary NA not changed.',
            'audit':captured,'runtime_seconds':time.time()-start,
            'source_adapter_sha256':EXPECTED_ADAPTER}
    A=d.load(a.atlas,'audit_export_atlas')
    js=json.dumps(A.clean(result),sort_keys=True,allow_nan=False)+'\n'
    lines=['F1_CLINICAL_OUTCOME_TIMING counts-only diagnostic; no prediction models',
           'Original labels/gates/results preserved; alternative is actual BP-assessment horizon.']
    for section in ['original_nominal_visit2','original_funnel','diagnostic_actual_BP_assessment','paired_changes']:
        for k,v in captured[section].items():
            line=section+'/'+k+' '+json.dumps(A.clean(v),separators=(',',':'))
            if len(line)<220:lines.append(line)
    lines+=['F1_CLINICAL_TIMING_DONE'];dg='\n'.join(lines)+'\n'
    if len(lines)>120 or any(len(x)>220 for x in lines):raise ValueError('digest_size')
    for s in [js,dg]:
        A.privacy_check(s,set(map(str,data['ids'])))
        if re.search(r'\b(?:19|20)\d{2}[-/]\d{1,2}[-/]\d{1,2}\b',s):raise ValueError('fine_date')
    out.mkdir();exports={'results.json':js,'digest.txt':dg}
    for n,s in exports.items():(out/n).write_text(s)
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256((out/n).read_bytes()).hexdigest()+'  '+n+'\n' for n in exports))
    print('F1_CLINICAL_TIMING_DONE seconds=%.1f sha12=%s'%(result['runtime_seconds'],hashlib.sha256(dg.encode()).hexdigest()[:12]),flush=True)

if __name__=='__main__':main()
