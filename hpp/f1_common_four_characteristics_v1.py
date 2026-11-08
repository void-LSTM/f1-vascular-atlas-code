#!/usr/bin/env python3
"""Descriptive aggregates for the exact X1 common-four selected-visit cohort.

No exposure assembly, regression, prediction, inference or participant-level output.
The frozen atlas builder reconstructs its existing correction in memory.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

VERSION = 'f1_common_four_characteristics_v1.0 October 13, 2026'
ATLAS_SHA = '62344144b83db3f2f7e6492bd05fd7616d48a442bba88134aa1e0360110fa253'
EXPOSOME_SHA = '69bdec161263ad8c63d3f57328eecf42879cbe5d36f34f89179c73dbd37e1227'
CORE = ['pwv', 'imt', 'art_w', 'abi_min']
ORIENT = {'pwv': 1, 'imt': 1, 'art_w': -1, 'abi_min': -1}
KEY = ['participant_id', 'research_stage']
CONTINUOUS = {
    'age': ('Age at selected visit', 'years'),
    'bmi': ('Body mass index', 'kg/m^2'),
    'sbp_sit': ('Sitting systolic blood pressure', 'mmHg'),
    'dbp_sit': ('Sitting diastolic blood pressure', 'mmHg'),
}
BINARY = {
    'sex_male': ('Recorded male sex', '1 male; 0 female'),
    'smk_current': ('Current smoking', 'atlas source codes 1/2 current; 0 not current'),
    'antihtn': ('Recorded antihypertensive use under atlas convention', 'ATC C02/C03/C07/C08/C09'),
    'statin': ('Recorded statin use under atlas convention', 'ATC C10AA'),
}
MEASURES = {
    'pwv': ('Femoral-ankle pulse-wave velocity', 'm/s'),
    'imt': ('Mean carotid intima-media thickness', 'mm'),
    'art_w': ('Retinal arterial width', 'source image units; not calibrated micrometres'),
    'abi_min': ('Minimum-side ankle-brachial index', 'dimensionless ratio'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_dependencies(atlas_path, exposome_path):
    if sha(atlas_path) != ATLAS_SHA:
        raise ValueError('atlas dependency mismatch')
    if sha(exposome_path) != EXPOSOME_SHA:
        raise ValueError('exposome source dependency mismatch')


def load_atlas(path):
    spec = importlib.util.spec_from_file_location('atlas_common_four_frozen', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def numeric(series):
    return pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan)


def denominator(A, values, n_total):
    n = int(values.notna().sum())
    rare_partition = 0 < min(n, n_total - n) < A.MINCELL
    return {'n_total_rounded10': A.rc(n_total),
            'n_observed_rounded10': None if rare_partition else A.rc(n),
            'n_missing_rounded10': None if rare_partition else A.rc(n_total - n),
            'denominator_status': 'complementary_suppression' if rare_partition else 'rounded_and_primary_suppressed',
            'small_counts': 'null means suppressed below twenty, including zero'}


def continuous(A, values, n_total):
    values = numeric(values)
    out = denominator(A, values, n_total)
    valid = values.dropna()
    if len(valid) < A.MINCELL:
        out['status'] = 'suppressed_lt20_observed'
        return out
    out.update(status='reported', mean=float(valid.mean()), sd=float(valid.std(ddof=1)),
               q25=float(valid.quantile(.25)), median=float(valid.median()), q75=float(valid.quantile(.75)))
    return out


def binary(A, values, n_total):
    values = numeric(values)
    if not values.dropna().isin([0, 1]).all():
        raise ValueError('unexpected binary coding')
    out = denominator(A, values, n_total)
    yes, no = int((values == 1).sum()), int((values == 0).sum())
    # Complementary suppression: no percentage or other category can recover a small level.
    if min(yes, no) < A.MINCELL or out['denominator_status'] == 'complementary_suppression':
        out.update(status='suppressed_small_binary_level', yes_n_rounded10=None,
                   no_n_rounded10=None, yes_percent_rounded_counts=None)
    else:
        ry, rn = A.rc(yes), A.rc(no)
        out.update(status='reported', yes_n_rounded10=ry, no_n_rounded10=rn,
                   yes_percent_rounded_counts=round(100 * ry / (ry + rn), 1))
    return out


def stage_counts(A, selected):
    counts = [int((selected.research_stage == st).sum()) for st in A.MAIN_STAGES]
    if min(counts) < A.MINCELL:
        return {'status': 'suppressed_small_stage_level',
                'counts_rounded10': {st: None for st in A.MAIN_STAGES}}
    return {'status': 'reported', 'counts_rounded10': {st: A.rc(n) for st, n in zip(A.MAIN_STAGES, counts)}}


def select_common_four(A, frame):
    if list(A.CORE) != CORE or any(A.ORIENT[b] != ORIENT[b] for b in CORE):
        raise ValueError('unexpected core or orientation')
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError('nonunique or missing frame keys')
    if not frame.research_stage.isin(A.MAIN_STAGES).all():
        raise ValueError('unexpected research stage')
    need = ['o_' + b for b in CORE] + ['age', 'sex_male']
    required = need + CORE + [b + '_raw' for b in CORE]
    if any(c not in frame for c in required):
        raise ValueError('missing required atlas field')
    # EXACT X1 selection from frozen f1_exposome_v1.py:913, on unchanged atlas fields.
    selected = A.one_row_per_participant(frame, need)
    if selected.participant_id.duplicated().any():
        raise ValueError('duplicate selected person')
    for b in CORE:
        if not np.allclose(selected['o_' + b], A.ORIENT[b] * selected[b], equal_nan=True):
            raise ValueError('orientation identity failure')
    return selected


def medication_coverage(A, selected, medflags):
    if medflags is None:
        return {'status': 'medication_source_unavailable',
                'interpretation': 'Atlas zero-filled flags do not establish absence of medication use.'}
    if any(c not in medflags for c in KEY) or medflags.duplicated(KEY).any():
        raise ValueError('invalid medication source keys')
    keys = medflags[KEY].copy()
    keys['source_present'] = 1
    joined = selected[KEY].merge(keys, on=KEY, how='left', validate='one_to_one')
    present = joined.source_present.notna()
    rare_partition = 0 < min(present.sum(), (~present).sum()) < A.MINCELL
    return {'status': 'complementary_suppression' if rare_partition else 'reported',
            'source_record_present_n_rounded10': None if rare_partition else A.rc(present.sum()),
            'source_record_absent_n_rounded10': None if rare_partition else A.rc((~present).sum()),
            'interpretation': 'A source row indicates recorded medication data, not complete medication ascertainment. Atlas flags fill unmatched visits to zero.'}


def summarise(A, frame, medflags=None, expected_n_common=5810):
    selected = select_common_four(A, frame)
    if A.rc(len(selected)) != expected_n_common:
        raise ValueError('rounded common-four count does not match accepted X1')
    rows = []
    for c, (label, unit) in CONTINUOUS.items():
        row = {'variable': c, 'label': label, 'units': unit, 'scale': 'untransformed selected-visit atlas field'}
        row.update(continuous(A, selected[c], len(selected)) if c in selected else {'status': 'field_unavailable'})
        rows.append(row)
    for c, (label, coding) in BINARY.items():
        row = {'variable': c, 'label': label, 'coding': coding}
        row.update(binary(A, selected[c], len(selected)) if c in selected else {'status': 'field_unavailable'})
        if c in ('antihtn', 'statin'):
            row['missingness_note'] = 'Atlas fills absent source matches to zero; see medication_source_coverage. This is recorded-use coding, not disease prevalence.'
        rows.append(row)
    vascular = []
    for b, (label, unit) in MEASURES.items():
        vascular.append({'variable': b, 'label': label, 'native_units': unit,
                        'adverse_orientation_multiplier': A.ORIENT[b],
                        'adverse_orientation_note': 'Unstandardized sign multiplication; larger oriented values denote the prespecified adverse direction. No residualization or z scoring.',
                        'raw_native': continuous(A, selected[b + '_raw'], len(selected)),
                        'v15_corrected_native': continuous(A, selected[b], len(selected)),
                        'v15_corrected_adverse_oriented': continuous(A, selected['o_' + b], len(selected))})
    return {'version': VERSION,
            'cohort': {'label': 'Characteristics at the selected atlas visit in the X1 common-four cohort',
                       'n_persons_rounded10': A.rc(len(selected)),
                       'selection': 'Exact X1: finite four adverse-oriented measures, age and sex; atlas selector prefers stage00 then stage02; one row per person. No BMI, exposure or other clinical complete-case restriction.',
                       'stage_counts': stage_counts(A, selected),
                       'expected_rounded_count': expected_n_common,
                       'expected_rounded_count_gate': 'PASS',
                       'selection_source': 'f1_exposome_v1.py line 913; atlas one_row_per_participant lines 1017-1020'},
            'characteristics': rows, 'vascular_measurements': vascular,
            'medication_source_coverage': medication_coverage(A, selected, medflags),
            'reporting': {'counts': 'Round to nearest ten using frozen atlas rc; all cells below twenty including zero are null. Binary and selected-stage partitions use complementary suppression. Positive rare observed/missing or medication-source cells suppress both partition counts and binary percentages; a truly empty cell remains primary-suppressed without suppressing its large complement.',
                          'denominators': 'Each field reports observed and missing counts against the selected person cohort. Missing counts below twenty are suppressed, never imputed for this helper.',
                          'percentages': 'Binary percentages use rounded yes and no counts and one decimal; they are approximate and suppressed if either level is below twenty.',
                          'continuous': 'Mean, sample SD, median and IQR on untransformed values; no min/max or tails; suppress statistics below twenty observed.',
                          'interpretation': 'Descriptive only. Common cohort membership does not imply a single enrollment-baseline visit or complete data for every exposure-specific analysis.'}}


def audit_reconstruction(A, frame, reference):
    """Compare the existing correction and reported aggregate frame, without refitting analyses."""
    if reference.get('version') != A.VERSION:
        raise ValueError('reference atlas version mismatch')
    for key, attr in [('calendar_adjust', 'CAL_ADJUST'), ('day_adjust', 'DAY_ADJUST'), ('session_adjust', 'SESSION_ADJUST')]:
        if reference.get(key) is not True or getattr(A, attr) is not True:
            raise ValueError('correction configuration mismatch')
    saved = reference.get('S0b_drift', {}).get('acquisition', {})
    if A.DIAG.get('session_relations_excluded') is not True or saved.get('session_relations_excluded') is not True:
        raise ValueError('relationship exclusion unavailable')
    for key in ('calendar_effects', 'day_effects', 'session_kernel', 'session_relations_excluded'):
        if key not in saved or key not in A.DIAG or A.clean(A.DIAG[key]) != saved[key]:
            raise ValueError('saved correction diagnostic mismatch')
    for b, expected in [('pwv', True), ('imt', False), ('art_w', False), ('abi_min', True)]:
        if A.DIAG.get('session_kernel', {}).get(b, {}).get('applied') is not expected:
            raise ValueError('unexpected core kernel application')
    facts = reference.get('S0_facts', {})
    for b in CORE:
        if A.clean(A.qtiles(frame[b])) != facts.get('descriptives', {}).get(b):
            raise ValueError('saved vascular summary mismatch')
    if A.rc(len(frame)) != 12950 or A.rc(frame.participant_id.nunique()) != 10430:
        raise ValueError('accepted calibration frame count mismatch')
    if facts.get('overlap', {}).get('four_beds_any_visit_participants') != 5810:
        raise ValueError('reference common-four count mismatch')
    return {'status': 'PASS', 'scope': 'Frozen correction configuration, relationship exclusion, full saved cleaned calendar/day/kernel diagnostic records, all-four visit summary quantiles, and disclosure-rounded calibration/common-four counts. Does not certify identical participant membership from aggregate data.'}


def digest(report):
    lines = ['F1 common-four selected-visit characteristics', 'Descriptive aggregates only; no exposure models.',
             'Persons ' + str(report['cohort']['n_persons_rounded10']),
             'Stage counts ' + json.dumps(report['cohort']['stage_counts'], sort_keys=True)]
    for row in report['characteristics']:
        lines.append(json.dumps(row, sort_keys=True))
    for row in report['vascular_measurements']:
        lines.append(json.dumps(row, sort_keys=True))
    lines.append('Medication source ' + json.dumps(report['medication_source_coverage'], sort_keys=True))
    lines.append('Privacy and denominator notes ' + json.dumps(report['reporting'], sort_keys=True))
    return '\n'.join(lines) + '\n'


def write_outputs(A, report, all_pids, out):
    out = Path(out)
    if out.exists():
        raise ValueError('output exists; refusing overwrite')
    report = A.clean(report)
    js = json.dumps(report, sort_keys=True, allow_nan=False, indent=2) + '\n'
    dg = digest(report)
    A.privacy_check(js, all_pids)
    A.privacy_check(dg, all_pids)
    out.mkdir(parents=False)
    (out/'results.json').write_text(js)
    (out/'digest.txt').write_text(dg)
    (out/'SHA256SUMS').write_text(''.join(sha(out/f) + '  ' + f + '\n' for f in ('results.json', 'digest.txt')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('root', 'out', 'atlas', 'exposome-source', 'atlas-results'):
        ap.add_argument('--' + arg, required=True)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise ValueError('output exists; refusing overwrite')
    verify_dependencies(args.atlas, args.exposome_source)
    A = load_atlas(args.atlas)
    # Suppress source build log messages; no participant-derived logs leave the builder.
    A.log = lambda message: None
    frame, medflags, _ = A.build_frame(args.root, None)
    reference = json.loads(Path(args.atlas_results).read_text())
    audit = audit_reconstruction(A, frame, reference)
    report = summarise(A, frame, medflags)
    report['reconstruction_audit'] = audit
    report['dependencies'] = {'atlas_version': A.VERSION,
                              'atlas_sha256_groups': ':'.join(ATLAS_SHA[i:i+4] for i in range(0,64,4)),
                              'exposome_source_sha256_groups': ':'.join(EXPOSOME_SHA[i:i+4] for i in range(0,64,4)),
                              'existing_atlas_results_sha256_groups': ':'.join(sha(args.atlas_results)[i:i+4] for i in range(0,64,4)),
                              'numpy_version': np.__version__, 'pandas_version': pd.__version__}
    write_outputs(A, report, set(frame.participant_id.astype(str)), args.out)
    print('F1_COMMON_FOUR_CHARACTERISTICS_DONE errors=[]', flush=True)


if __name__ == '__main__':
    try:
        main()
    except SystemExit as exc:
        if exc.code == 0:
            raise
        print('F1_COMMON_FOUR_CHARACTERISTICS_ABORT SystemExit', file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        # Do not print data-bearing exception messages or frame contents.
        print('F1_COMMON_FOUR_CHARACTERISTICS_ABORT ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
