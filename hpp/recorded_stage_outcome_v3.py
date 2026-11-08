"""Declared source-availability sensitivity; never claim disease-onset dates.

Anchor only undated curated/self-report HTN records to the actual recorded event
Date/Timestamp of their identical participant/stage key. Baseline positives remain
prevalent; follow-up classification does not itself create an objective event.
Original date-qualified BP/treatment/ICD definition, horizon and gates remain.
"""
import numpy as np
import pandas as pd

def apply(adapter):
    original_sources=adapter._clinical_source_rows;original_labels=adapter.htn_labels
    def labels(p,fr,stage_days,badkeys,root,A,O,P):
        y0,q0=original_labels(p,fr,stage_days,badkeys,root,A,O,P)
        sources=original_sources(root,A,O,P,badkeys);updated=[];report=None;count=adapter.count
        stats={}
        for r in sources:
            if len(r) and r['_component'].iloc[0]=='report':
                r=r.copy();keys=pd.MultiIndex.from_frame(r[adapter.KEYS])
                linked=pd.Series(stage_days.reindex(keys).to_numpy(),index=r.index)
                missing=r['_when'].isna();usable=missing&linked.notna()
                r['_when']=r['_when'].where(~missing,linked)
                cur=adapter.getnum(r,'dx_any').eq(1);self=adapter.getnum(r,'self').eq(1)
                for stage,tag in zip(A.MAIN_STAGES,['baseline','followup']):
                    at=r.research_stage.eq(stage)
                    for component,mask in [('curated',cur),('self',self)]:
                        stats[tag+'_undated_positive_'+component]=count(r.loc[at&missing&mask,'participant_id'].nunique())
                stats['report_records_with_recorded_stage_date']=count(usable.sum())
                stats['report_records_still_untimed']=count(r['_when'].isna().sum())
                report=r
            updated.append(r)
        adapter._clinical_source_rows=lambda *args:updated
        try:y1,q1=original_labels(p,fr,stage_days,badkeys,root,A,O,P)
        finally:adapter._clinical_source_rows=original_sources
        ids=p.participant_id.astype(str);starts=pd.Series(p.start.to_numpy(),index=ids)
        horizons=pd.Series(P.day(p.visit_date_1).to_numpy(),index=ids)
        conflict=set();clear_baseline=set()
        if report is not None:
            r=report;rid=r.participant_id.astype(str);st=rid.map(starts);en=rid.map(horizons)
            strict=adapter.getnum(r,'dx_strict').eq(1)|adapter.getnum(r,'self').eq(1)
            post=r['_when'].gt(st)&r['_when'].le(en)
            conflict=set(rid.loc[strict&post])
            negative=adapter.getnum(r,'dx_any').eq(0)&adapter.getnum(r,'self').eq(0)
            clear_baseline=set(rid.loc[negative&r.research_stage.eq(A.MAIN_STAGES[0])&r['_when'].le(st)])
        contradictory=(y1==0)&ids.isin(conflict).to_numpy()
        y1[contradictory]=np.nan
        gain=(y1==1)&(y0!=1);loss=(y0==1)&(y1!=1)
        qc={'original_strict':q0,'recorded_stage_sensitivity':q1,'source_stage':stats,
            'labels_after_report_conflict_mask':count(np.isfinite(y1).sum()),
            'events_after_report_conflict_mask':count(np.nansum(y1)),
            'reported_positive_without_objective_evidence_masked':count(contradictory.sum()),
            'positive_labels_gained':count(gain.sum()),'positive_labels_lost':count(loss.sum()),
            'gained_events_with_explicit_negative_baseline_curated_and_self_report':count((gain&ids.isin(clear_baseline)).sum()),
            'record_time':'actual_same_person_stage_event_Date_or_Timestamp_as_record_availability_proxy; not_own_collection_or_disease_onset',
            'event_definition':'same_dated_BP140or90_or_treatment_or_ICD11; followup_report_alone_not_an_objective_event',
            'original_primary':'original_strict_five_head_primary_NA_preserved',
            'date_imputation':'no_calendar_median_or_invented_date; explicitly_declared_recorded_stage_proxy',
            'clinical_claim':'recorded_objective_HTN_assessment_sensitivity_not_adjudicated_incidence_or_hard_CV_event'}
        return y1,qc
    adapter.htn_labels=labels
    adapter.TASK_NAMES=list(adapter.TASK_NAMES);adapter.TASK_NAMES[4]='htn_recorded_stage'
    adapter.UNITS=list(adapter.UNITS);adapter.UNITS[4]='binary_stage_anchored_recorded_objective_HTN'

def build(adapter,*args):
    data=adapter.build_data(*args)
    # Correct the fixed task-label list used by the original build_data QC only.
    data['qc']['task_counts']={name:adapter.count(data['mask'][:,j].sum()) for j,name in enumerate(data['task_names'])}
    data['qc']['task_groups']={name:adapter.count(len(np.unique(data['groups'][data['mask'][:,j]]))) for j,name in enumerate(data['task_names'])}
    data['qc']['adapter']='frozen_v2_with_declared_recorded_stage_outcome_sensitivity_v3'
    return data
