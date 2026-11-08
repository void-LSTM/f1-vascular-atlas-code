#!/usr/bin/env python3
"""Strict-timeline, masked five-output HPP adapter; memory only, never exports rows.

Calling code owns model fitting, fold-only preprocessing and privacy-checked output.
All values returned in X/Z/Y remain inside HPP.  qc contains aggregate counts only.
"""
from pathlib import Path
import hashlib
import importlib.util
import numpy as np
import pandas as pd

EXPECTED = {
    'atlas': '62344144b83db3f2f7e6492bd05fd7616d48a442bba88134aa1e0360110fa253',
    'p21': 'c42a1ba7c6d8b62bd573a8f2d2c46a4a3487d14bd3b79b36f4b40b1ca3e17f64',
    'p13': '0a753b8209281294089b16a0cc2dcc3e8fa3ada61631b8f2e51d5d14632f47ea',
    'outcomes': '9bee14ddca223c57669df4d2b816613f33942ed3c4ab439eda08f8c1d8b7775e',
    'p10': 'e8cc37eaa90adef602d8b7b6f29bb2bfb8c4be14b2f8302ff2cab7be3b46743e',
}
CORE = ['age', 'sex_male', 'bmi', 'sbp_sit', 'dbp_sit']
DEEP_SOURCES = {
    'pulse_optional': ('blood_pressure/blood_pressure.parquet', ['sitting_blood_pressure_pulse_rate']),
    'waist_optional': ('anthropometrics/anthropometrics.parquet', ['waist_circumference']),
    'dxa_optional': ('body_composition/body_composition.parquet', ['total_scan_vat_mass', 'total_scan_sat_mass']),
    'liver_optional': ('liver_ultrasound/liver_ultrasound.parquet', ['attenuation_coefficient_qbox']),
    'sleep_optional': ('sleep/sleep.parquet', ['odi', 'saturation_mean']),
    'cgm_optional': ('cgm/iglu.parquet', ['cgm_mean']),
}
DEEP_MODALITY = {'hr_sit': 'pulse_optional', 'waist': 'waist_optional', 'vat': 'dxa_optional',
                 'sat': 'dxa_optional', 'att': 'liver_optional', 'odi': 'sleep_optional',
                 'spo2_mean': 'sleep_optional', 'cgm_mean': 'cgm_optional'}
FEATURE_NAMES = CORE + ['pwv', 'imt', 'adverse_abi', 'adverse_retina', 'ldl', 'glucose', 'hba1c', 'egfr'] + list(DEEP_MODALITY)
TASK_NAMES = ['pwv', 'imt', 'abi', 'retina_level', 'htn_objective']
KINDS = ['continuous'] * 4 + ['binary']
UNITS = ['m/s', 'mm', 'negative_ABI_ratio', 'negative_retinal_source_width', 'binary_recorded_objective_HTN']
KEYS = ['participant_id', 'research_stage']


def count(n):
    return '<20' if int(n) < 20 else int(round(float(n) / 10) * 10)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def getnum(d, name):
    if name not in d:
        return pd.Series(np.nan, index=d.index)
    return pd.to_numeric(d[name], errors='coerce').replace([np.inf, -np.inf], np.nan)


def source_days(d, P):
    """Actual collection date/timestamp only; no event-stage or median imputation."""
    t = pd.Series(pd.NaT, index=d.index, dtype='datetime64[ns]')
    for col in ['collection_timestamp', 'collection_date']:
        if col in d:
            t = t.fillna(P.day(d[col]))
    if 'date' in d:
        # Frozen O.med_rows/cond_rows have already converted this derived field
        # to naive Jerusalem local time. Do not shift it through UTC a second time.
        local = pd.to_datetime(d['date'], errors='coerce')
        if isinstance(local.dtype, pd.DatetimeTZDtype):
            local = local.dt.tz_convert('Asia/Jerusalem').dt.tz_localize(None)
        t = t.fillna(local.dt.normalize().dt.as_unit('ns'))
    return t


def remove_ambiguous(d, badkeys):
    if d is None:
        return None
    if not len(badkeys):
        return d.copy()
    return d.loc[~pd.MultiIndex.from_frame(d[KEYS]).isin(badkeys)].copy()


def event_days(raw, P):
    e = raw.copy()
    e['_event_day'] = P.day(e.research_stage_date)
    g = e.groupby(KEYS)._event_day
    bad = g.nunique().gt(1)
    badkeys = bad[bad].index
    dated = g.min().loc[~bad].rename('_stage_day')
    return dated, badkeys


def baseline_labs(fr, A, P, root):
    """Frozen P21 backward join and duplicate exclusions; no predictive fitting."""
    raw = A.rd(root, 'blood_tests/blood_tests.parquet')
    if raw is None:
        out = fr.copy()
        for name in ['ldl', 'glucose', 'hba1c', 'egfr']:
            out[name] = np.nan
        return out, {'status': 'source_unavailable', 'ambiguous_dates': '<20'}
    labs, ambiguous = P.lab_table(raw)
    base = fr[fr.research_stage == A.MAIN_STAGES[0]]
    selected = P.match_labs(base, labs)
    result = fr.drop(columns=list(P.LABS.values()) + ['egfr'], errors='ignore').merge(
        selected, on=KEYS, how='left', validate='one_to_one')
    # Retain the frozen atlas source-unit convention (mg/dL vs umol/L).
    # This is source harmonisation, never optimized against a head, split or metric.
    creat = getnum(result, 'creat')
    source_umol = bool(creat.median(skipna=True) > 20)
    cr = creat / 88.4 if source_umol else creat
    female = result.sex_male.eq(0)
    ratio = cr / np.where(female, .7, .9)
    result['egfr'] = 142 * np.minimum(ratio, 1) ** np.where(female, -.241, -.302) * np.maximum(ratio, 1) ** -1.2 * .9938 ** result.age * np.where(female, 1.012, 1)
    result.loc[~np.isfinite(result.egfr) | (cr <= 0), 'egfr'] = np.nan
    return result, {'status': 'prior_draw_within_180_days', 'ambiguous_dates': ambiguous,
                    'creatinine_convention': 'frozen_atlas_median_above20_convert_by88.4',
                    'creatinine_source_interpreted_unit': 'umol/L' if source_umol else 'mg/dL',
                    'optional_labs': ['ldl', 'glucose', 'hba1c', 'egfr'],
                    'HDL_TG': 'not_in_fixed_input_set'}


def _clinical_source_rows(root, A, O, P, badkeys):
    """Frozen HTN content flags with actual dates and ambiguous-stage exclusion."""
    sources = []
    med = remove_ambiguous(O.med_rows(root), badkeys)
    cond = remove_ambiguous(O.cond_rows(root), badkeys)
    htn = remove_ambiguous(O.htn_rows(root), badkeys)
    if med is not None:
        med['_when'] = source_days(med, P)
        med['_positive'] = getnum(med, 'med').eq(1)
        med['_component'] = 'med'
        sources.append(med)
    if cond is not None:
        cond['_when'] = source_days(cond, P)
        cond['_positive'] = getnum(cond, 'icd').eq(1)
        cond['_component'] = 'icd'
        sources.append(cond)
    if htn is not None:
        # The frozen parser intentionally returns no dates. Recover source date columns
        # by original row position before applying the same ambiguous-stage removal.
        raw = A.rd(root, 'curated_phenotypes/hypertension.parquet')
        whole = O.htn_rows(root)
        if len(raw) != len(whole):
            raise ValueError('curated_htn_row_alignment')
        whole['_when'] = source_days(raw.reset_index(drop=True), P).to_numpy()
        htn = remove_ambiguous(whole, badkeys)
        htn['_positive'] = getnum(htn, 'dx_any').eq(1) | getnum(htn, 'self').eq(1)
        htn['_component'] = 'report'
        sources.append(htn)
    return sources


def htn_labels(p, fr, stage_days, badkeys, root, A, O, P):
    """Known-free at landmark; incident recorded objective HTN through dated visit2.

    Evaluation horizon (not event onset) must be >0.5 and <=5y after landmark.
    Events during the first0.5y after landmark are retained. Negative labels need
    timed visit2 BP. Undated positive evidence potentially before the horizon, or
    retrospective explicitly pre-landmark history, makes the head unavailable.
    """
    ids = p.participant_id.astype(str)
    starts = pd.Series(p.start.to_numpy(), index=ids)
    horizons = pd.Series(P.day(p.visit_date_1).to_numpy(), index=ids)
    pre = {c: set() for c in ['med', 'icd', 'report']}
    incident = {c: set() for c in ['med', 'icd']}
    ambiguous, early = set(), set()
    sources = _clinical_source_rows(root, A, O, P, badkeys)
    for r in sources:
        r = r[r.participant_id.astype(str).isin(ids)].copy()
        r['participant_id'] = r.participant_id.astype(str)
        st = r.participant_id.map(starts)
        en = r.participant_id.map(horizons)
        when = r['_when']
        pos = r['_positive']
        comp = r['_component'].iloc[0] if len(r) else None
        if comp is None:
            continue
        pre[comp] |= set(r.loc[pos & when.notna() & when.le(st), 'participant_id'])
        keys = pd.MultiIndex.from_frame(r[KEYS])
        stage = pd.Series(stage_days.reindex(keys).to_numpy(), index=r.index)
        # An undated row attached only to a definitely later stage is outside scope.
        uncertain = pos & when.isna() & (stage.isna() | stage.le(en))
        ambiguous |= set(r.loc[uncertain, 'participant_id'])
        post = pos & when.gt(st) & when.le(en)
        if comp == 'icd' and 'start_year' in r:
            historical = post & getnum(r, 'start_year').lt(st.dt.year)
            ambiguous |= set(r.loc[historical, 'participant_id'])
            post &= ~historical
        if comp in incident:
            incident[comp] |= set(r.loc[post, 'participant_id'])
            early |= set(r.loc[post & when.le(st + pd.Timedelta(days=.5 * 365.25)), 'participant_id'])
    # Reuse frozen baseline eligibility with flags rebuilt only from known by-landmark information.
    base = fr[fr.research_stage == A.MAIN_STAGES[0]].copy()
    base['participant_id'] = base.participant_id.astype(str)
    base['antihtn'] = base.participant_id.isin(pre['med']).astype(float)
    base['htn_curated'] = base.participant_id.isin(pre['report']).astype(float)
    base['htn_self'] = base['htn_curated']
    bd = pd.Series(P.day(p.visit_date_0).to_numpy(), index=ids)
    eligible_base, _ = O.make_cohort(base, bd, pre['icd'], None)
    eligible_ids = set(eligible_base.participant_id.astype(str))
    # Any additional measured mean BP known by landmark also removes known prevalent HTN.
    fb = fr.copy()
    fs = fb.participant_id.astype(str).map(starts)
    measured_pre = fb.bp_hi.notna() & fb.bp_hi.le(fs) & ((fb.sbp_sit >= 140) | (fb.dbp_sit >= 90))
    eligible_ids -= set(fb.loc[measured_pre, 'participant_id'].astype(str))
    horizon_years = (P.day(p.visit_date_1) - p.start).dt.total_seconds() / (365.25 * 86400)
    valid_horizon = horizon_years.gt(.5) & horizon_years.le(5)
    bp_valid = (p[['sbp_sit_1', 'dbp_sit_1']].notna().all(axis=1) & p.bp_lo_1.gt(p.start)
                & p.bp_hi_1.le(P.day(p.visit_date_1)))
    contact = pd.DataFrame({'kind': 'v2', 'med': ids.isin(incident['med']).astype(float),
                            'icd': ids.isin(incident['icd']).astype(float),
                            'sbp2': p.sbp_sit_1.where(bp_valid), 'dbp2': p.dbp_sit_1.where(bp_valid)})
    objective = O.ev_objective(contact).to_numpy(float)
    use = valid_horizon & ids.isin(eligible_ids) & ~ids.isin(ambiguous) & (bp_valid | (objective == 1))
    y = np.where(use, objective, np.nan)
    qc = {'known_free_at_landmark': count(ids.isin(eligible_ids).sum()),
          'undated_positive_or_retrospective_ambiguity': count(ids.isin(ambiguous).sum()),
          'valid_evaluation_horizon': count(valid_horizon.sum()),
          'timed_visit2_bp': count(bp_valid.sum()),
          'labelled': count(use.sum()), 'events': count(np.nansum(y)),
          'early_postlandmark_events_retained': count((use & ids.isin(early)).sum()),
          'definition': 'BP140or90_or_antiHTN_or_ICD11_HTN',
          'negative_rule': 'timed_visit2_BP_and_no_recorded_objective_event',
          'date_imputation': 'none', 'event_time_rule': 'landmark_through_personal_visit2',
          'evaluation_horizon_rule': 'greater_than_half_year_at_most_five_years',
          'not_adjudicated_hard_CV_event': True}
    return y, qc


def build_data(root, atlas_path, p21_path, p13_path, outcomes_path, p10_path):
    """Return sandbox-only arrays plus privacy-safe aggregate qc; write nothing."""
    paths = dict(atlas=atlas_path, p21=p21_path, p13=p13_path, outcomes=outcomes_path, p10=p10_path)
    for name, path in paths.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != EXPECTED[name]:
            raise ValueError('dependency_hash_' + name)
    P13 = load(p13_path, 'network_p13')
    P13.load_modules({k: Path(paths[k]) for k in ['atlas', 'outcomes', 'p10']})
    A, O = P13.A, P13.O
    P = load(p21_path, 'network_p21'); P.A = A
    A.CAL_ADJUST = A.DAY_ADJUST = A.SESSION_ADJUST = False
    fr = A.build_frame(str(root), None)[0].replace([np.inf, -np.inf], np.nan)
    fr['participant_id'] = fr.participant_id.astype(str)
    stage_days, badkeys = event_days(A.rd(root, 'events/events.parquet'), P)
    fr = remove_ambiguous(fr, badkeys)
    if fr.duplicated(KEYS).any():
        raise ValueError('duplicate_person_stage')
    fr, labqc = baseline_labs(fr, A, P, str(root))
    timing = {}
    # Extend only this in-memory dependency instance. Keep optional source timing
    # separate, so an extra pulse/waist row cannot change the core BP/BMI cohort.
    P.SOURCES = dict(P.SOURCES, **DEEP_SOURCES)
    for name, (source, _) in P.SOURCES.items():
        raw = remove_ambiguous(A.rd(root, source), badkeys)
        if raw is None or not any(c in raw.columns for c in P.SOURCES[name][1]):
            dates = pd.DataFrame(columns=KEYS + [name + '_lo', name + '_hi'])
            timing[name] = {'status': 'source_or_value_columns_unavailable'}
        else:
            dates, timing[name] = P.acquisition(raw, str(root), name)
        fr = fr.merge(dates, on=KEYS, how='left', validate='one_to_one')
        for suffix in ['_lo', '_hi']:
            fr[name + suffix] = pd.to_datetime(fr[name + suffix]).astype('datetime64[ns]')
    s0, s1 = A.MAIN_STAGES
    p = fr[fr.research_stage == s0].merge(fr[fr.research_stage == s1], on='participant_id', how='left', suffixes=('_0', '_1'), validate='one_to_one')
    p['start'] = P.day(p.visit_date_0) + pd.Timedelta(days=180)
    p['lower'] = P.day(p.visit_date_0) - pd.Timedelta(days=180)
    core = p[[c + '_0' for c in CORE]].notna().all(axis=1) & P.baseline_time(p, 'bp') & P.baseline_time(p, 'bmi')
    p = p[core].reset_index(drop=True)
    X = pd.DataFrame({c: getnum(p, c + '_0') for c in CORE})
    for bed, col, sign in [('pwv', 'pwv', 1), ('imt', 'imt', 1), ('abi_min', 'adverse_abi', -1), ('art_w', 'adverse_retina', -1)]:
        known = P.baseline_time(p, P.MODALITY[bed])
        X[col] = (sign * getnum(p, bed + '_0')).where(known)
    for col in ['ldl', 'glucose', 'hba1c', 'egfr']:
        X[col] = getnum(p, col + '_0')
    for col, modality in DEEP_MODALITY.items():
        X[col] = getnum(p, col + '_0').where(P.baseline_time(p, modality))
    Y = np.full((len(p), 5), np.nan)
    Z, skip_names = [], []
    for j, (bed, modality, sign) in enumerate([('pwv', 'vascular', 1), ('imt', 'carotid', 1), ('abi_min', 'vascular', -1), ('art_w', 'retina', -1)]):
        lo, hi = p[modality + '_lo_1'], p[modality + '_hi_1']
        future = lo.gt(p.start + pd.Timedelta(days=.5 * 365.25)) & hi.le(p.start + pd.Timedelta(days=5 * 365.25))
        own = (sign * getnum(p, bed + '_0')).where(P.baseline_time(p, modality))
        if j < 3:
            future &= own.notna() & P.baseline_time(p, modality)
        later = sign * getnum(p, bed + '_1')
        use = future & later.notna()
        Y[:, j] = later.where(use).to_numpy(float)
        # Realized target acquisition midpoint is research forecast conditioning only.
        midpoint = lo + (hi - lo) / 2
        years = (midpoint - p.start).dt.total_seconds() / (365.25 * 86400)
        z = X[CORE].copy()
        if j < 3:
            z['own_baseline'] = own
        z['target_years_from_landmark'] = years.where(use)
        Z.append(z.to_numpy(float)); skip_names.append(list(z.columns))
    Y[:, 4], htnqc = htn_labels(p, fr, stage_days, badkeys, str(root), A, O, P)
    Z.append(X[CORE].to_numpy(float)); skip_names.append(list(CORE))
    mask = np.isfinite(Y)
    any_target = mask.any(axis=1)
    relations = A.relation_components(str(root))
    if relations is None:
        raise ValueError('relationship_metadata_missing')
    ids = p.participant_id.astype(str).to_numpy()
    groups = np.array([str(relations.get(i, i)) for i in ids])
    qc = {'adapter': 'network_data_v1', 'baseline_core_timed': count(len(p)), 'union_labelled': count(any_target.sum()),
          'ambiguous_event_stage_keys': count(len(badkeys)), 'labs': labqc, 'timing': timing, 'htn': htnqc,
          'task_counts': {name: count(mask[:, j].sum()) for j, name in enumerate(TASK_NAMES)},
          'task_groups': {name: count(len(np.unique(groups[mask[:, j]]))) for j, name in enumerate(TASK_NAMES)},
          'all_three_vascular_outputs': count(mask[:, :3].all(axis=1).sum()),
          'all_five_outputs': count(mask.all(axis=1).sum()),
          'baseline_feature_available': {name: count(X[name].notna().to_numpy()[any_target].sum()) for name in FEATURE_NAMES},
          'retina_target': 'later_level_without_baseline_retina_requirement',
          'deep_inputs': {'fixed_features': list(DEEP_MODALITY), 'units': 'source_native',
                          'scaling': 'none_in_adapter_training_only_in_model',
                          'timing': 'separate_optional_modality_all_contributors_within_baseline_window'},
          'global_acquisition_transforms': 'disabled', 'source_dates': 'actual_no_imputation',
          'participant_arrays_exportable': False}
    return dict(X=X[FEATURE_NAMES].to_numpy(float)[any_target], Z=[z[any_target] for z in Z], Y=Y[any_target],
                mask=mask[any_target], groups=groups[any_target], ids=ids[any_target], feature_names=list(FEATURE_NAMES),
                skip_names=skip_names, task_names=list(TASK_NAMES), kinds=list(KINDS), qc=qc, units=list(UNITS))
