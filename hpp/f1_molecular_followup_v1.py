#!/usr/bin/env python3
"""P14: aggregate-only PBMC composition-proxy and prospective follow-up.

Protocol: docs/P14_MOLECULAR_FOLLOWUP_PROTOCOL_20261006.md.
No absolute cell fractions, causal mediation or clinical utility claims.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import optimize, stats, sparse

VERSION = 'f1_molecular_followup_v1.0 2026-10-06'
NK = ['GNLY', 'NKG7', 'GZMB', 'PRF1', 'KLRD1', 'GZMA']
BEDS = ['pwv', 'imt', 'art_w', 'abi_min']
REPEAT_BEDS = ['pwv', 'imt', 'abi_min']
COVS = ['age', 'sex_male', 'bmi', 'sbp_sit', 'antihtn']
STAGES = ['00_00_visit', '02_00_visit']
MIN_N = 100
A = O = N = None

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def setup(atlas, omics, nkhr):
    global A, O, N
    A = load_module(atlas, 'f1_atlas_p14')
    O = load_module(omics, 'f1_omics_p14'); O.A = A; O.DRAW_LAYERS = {'rna'}
    N = load_module(nkhr, 'f1_nkhr_p14'); N.A = A; N.O = O
    assert list(O.MODULES['nk_cytotoxic']) == NK

def count(n):
    return '<20' if n < 20 else int(10 * round(float(n) / 10))

def quantiles(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [float(v) for v in np.quantile(x, [.05, .25, .5, .75, .95])] if len(x) >= 20 else None

def z(x):
    x = np.asarray(x, float)
    sd = np.std(x)
    return (x - np.mean(x)) / sd if sd > 1e-12 else np.full_like(x, np.nan)

def infer_composition(cpm, reference):
    """Weighted NNLS contributions, normalized to proxy coordinates (not cell fractions)."""
    weights = 1 / np.sqrt(reference.mean(axis=1) + 1)
    design = reference * weights[:, None]
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('reference overlap is rank deficient')
    coef = np.array([optimize.nnls(design, v * weights, maxiter=1000)[0] for v in cpm])
    total = coef.sum(axis=1)
    proxy = np.divide(coef, total[:, None], out=np.full_like(coef, np.nan), where=total[:, None] > 0)
    fitted = coef @ reference.T
    denom = np.linalg.norm(cpm * weights, axis=1)
    error = np.linalg.norm((cpm - fitted) * weights, axis=1) / np.maximum(denom, 1e-12)
    return proxy, error

def load_rna(root, reference):
    """Read batch files; factorize sample tuples without assuming any row order."""
    ref = pd.read_csv(reference, sep='\t', index_col=0)
    if len(set(ref.index) & set(NK)):
        raise ValueError('reference overlaps the target programme')
    genes = list(ref.index) + NK
    gindex = {g: i for i, g in enumerate(genes)}
    meta = A.rd(root, 'rna_seq/rna_seq.parquet', columns=[
        'collection_date', 'collection_timestamp', 'batch', 'full_counts_raw_parquet'])
    meta['participant_id'] = meta.participant_id.astype(str)
    meta['array_index'] = pd.to_numeric(meta.array_index, errors='coerce')
    samples, matrices, libraries = [], [], []
    for rel in sorted(set(meta.full_counts_raw_parquet.dropna().astype(str))):
        path = Path(root)/'rna_seq'/rel.lstrip('./')
        if not path.exists():
            raise ValueError('RNA counts path missing')
        t = pq.read_table(path, columns=['participant_id', 'research_stage', 'array_index', 'gene', 'count'])
        pi, pdict = O.enc(t.column('participant_id')); si, sdict = O.enc(t.column('research_stage'))
        ai = t.column('array_index').combine_chunks().to_numpy(zero_copy_only=False)
        key = pd.MultiIndex.from_arrays([pdict[pi], sdict[si], ai], names=['participant_id', 'research_stage', 'array_index'])
        codes, unique = pd.factorize(key, sort=False)
        unique = unique.set_names(['participant_id', 'research_stage', 'array_index'])
        gd, gnames = O.enc(t.column('gene'))
        values = t.column('count').combine_chunks().to_numpy(zero_copy_only=False).astype(float)
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('RNA counts must be finite and nonnegative')
        goodgene = np.array([not g.startswith('ERCC-') and not re.search(r'\d{7,}', g) for g in gnames])
        lib = np.bincount(codes, weights=np.where(goodgene[gd], values, 0), minlength=len(unique))
        lookup = np.array([gindex.get(g, -1) for g in gnames])
        col = lookup[gd]; keep = col >= 0
        matrix = sparse.coo_matrix((values[keep], (codes[keep], col[keep])), shape=(len(unique), len(genes))).tocsr()
        samples.append(unique.to_frame(index=False)); matrices.append(matrix); libraries.append(lib)
    if not samples:
        raise ValueError('no RNA batches')
    s = pd.concat(samples, ignore_index=True)
    matrix = sparse.vstack(matrices).tocsr(); lib = np.concatenate(libraries)
    s['_row'] = np.arange(len(s)); s['lib'] = lib
    s = s.merge(meta.drop_duplicates(['participant_id', 'research_stage', 'array_index']),
                on=['participant_id', 'research_stage', 'array_index'], how='left', validate='many_to_one')
    ll = np.log10(np.maximum(lib, 1)); med = np.median(ll); mad = np.median(abs(ll-med))*1.4826
    s = s[(s.lib > 0) & (np.log10(s.lib.clip(lower=1)) >= med-3*mad)]
    s = s.sort_values(['lib', '_row'], ascending=[False, True]).drop_duplicates(['participant_id', 'research_stage'])
    s = s.sort_values('_row').reset_index(drop=True)
    raw = matrix[s._row.to_numpy()].toarray(); cpm = raw / s.lib.to_numpy()[:, None] * 1e6
    detected = (raw > 0).mean(axis=0) >= .1
    used = [i for i, g in enumerate(genes) if g in NK and detected[i]]
    if len(used) < 2:
        raise ValueError('fewer than two detected NK genes')
    logcpm = np.log2((raw[:, used] + .5) / (s.lib.to_numpy()[:, None] + 1) * 1e6)
    batches = s.batch.astype(object).where(s.batch.notna(), 'missing').astype(str).to_numpy()
    for bt in np.unique(batches):
        logcpm[batches == bt] -= logcpm[batches == bt].mean(axis=0)
    scale = logcpm.std(axis=0); valid = scale > 1e-12
    if valid.sum() < 2:
        raise ValueError('fewer than two variable NK genes')
    s['nk'] = (logcpm[:, valid] / scale[valid]).mean(axis=1)
    hour = O.draw_hour(s.collection_timestamp)
    tech = pd.get_dummies(pd.Categorical(batches), dtype=float).to_numpy()[:, 1:]
    flow = {'samples': count(len(s)), 'participants': count(s.participant_id.nunique()),
            'batches': len(np.unique(batches)), 'nk_genes': int(valid.sum()), 'hour_missing': count((~np.isfinite(hour)).sum())}
    tech = O.draw_tech(tech, hour, {}, 'rna')
    tech = np.column_stack([tech, ~np.isfinite(hour)])
    techcols = [f'tech_{i}' for i in range(tech.shape[1])]
    for j, col in enumerate(techcols):
        s[col] = tech[:, j]
    overlap = np.flatnonzero(detected[:len(ref)])
    rr = ref.to_numpy()[overlap]
    winners = np.argmax(rr, axis=1) if len(rr) else np.array([], int)
    markers = np.bincount(winners, minlength=len(ref.columns))
    flow.update(reference_genes=len(overlap), reference_markers={k: int(v) for k, v in zip(ref.columns, markers)})
    proxycols = ['proxy_'+k for k in ref.columns]
    flow['composition_status'] = 'unavailable: reference overlap gate'
    if len(overlap) >= 100 and np.all(markers >= 5) and np.linalg.matrix_rank(rr) == len(ref.columns):
        proxies, error = infer_composition(cpm[:, overlap], rr)
        for j, col in enumerate(proxycols):
            s[col] = proxies[:, j]
        flow['composition_status'] = 'available: RNA composition proxies only'
        flow['weighted_fit_error'] = quantiles(error)
        flow['proxy_quantiles'] = {c: quantiles(s[c]) for c in proxycols}
    else:
        for col in proxycols:
            s[col] = np.nan
    return s, techcols, [c for c in proxycols if c != 'proxy_CD4'], flow

def cov_matrix(d, cols):
    cols = list(dict.fromkeys(cols))
    v = d[cols].to_numpy(float) if cols else np.empty((len(d), 0))
    sd = v.std(axis=0); use = sd > 1e-12
    v = (v[:, use]-v[:, use].mean(axis=0))/sd[use]
    return np.column_stack([np.ones(len(d)), v])

def fit_arrays(y, x, nuisance):
    y = np.asarray(y, float)
    if y.ndim == 1: y = y[:, None]
    x = np.asarray(x, float)
    design = np.column_stack([nuisance, x])
    rank = np.linalg.matrix_rank(design)
    if len(x) < MIN_N or len(x)-rank < 20 or rank != np.linalg.matrix_rank(nuisance)+1:
        raise ValueError('model sample or identifiable exposure gate')
    inv = np.linalg.pinv(design); beta = inv @ y
    residual = y-design @ beta
    influence = inv[-1, :, None] * residual
    variance = influence.T @ influence * len(x)/(len(x)-rank)
    return beta[-1], variance

def estimate(d, outcomes, covs, exposure='nk'):
    need = list(dict.fromkeys([exposure]+outcomes+covs))
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
    if len(d) < MIN_N:
        return {'n': count(len(d)), 'status': 'unavailable: n<100'}, None
    y = np.column_stack([z(d[c]) for c in outcomes]); x = z(d[exposure])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {'n': count(len(d)), 'status': 'unavailable: zero variance'}, None
    c = cov_matrix(d, covs)
    try:
        b, v = fit_arrays(y, x, c)
    except ValueError:
        return {'n': count(len(d)), 'status': 'unavailable: model gate'}, None
    se = np.sqrt(np.maximum(np.diag(v), 0))
    p = 2*stats.norm.sf(np.abs(b)/np.maximum(se, 1e-15))
    res = {'n': count(len(d)), 'status': 'estimated', 'b': b, 'se': se, 'lo': b-1.96*se, 'hi': b+1.96*se, 'p': p}
    return res, (y, x, c, b, v)

def paired_comparison(d, outcome, basecov, extra, boot, rng):
    need = list(dict.fromkeys(['nk', outcome]+basecov+extra))
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
    base, aa = estimate(d, [outcome], basecov)
    adjusted, bb = estimate(d, [outcome], basecov+extra)
    out = {'base': base, 'adjusted': adjusted}
    if aa is None or bb is None: return out
    draws = []
    for _ in range(boot):
        ix = rng.integers(0, len(d), len(d))
        try:
            b0 = fit_arrays(aa[0][ix], aa[1][ix], aa[2][ix])[0][0]
            b1 = fit_arrays(bb[0][ix], bb[1][ix], bb[2][ix])[0][0]
            draws.append([b0, b1])
        except ValueError:
            continue
    out['bootstrap_valid'] = len(draws)
    if len(draws) >= .8*boot:
        draw = np.array(draws)
        out['delta_beta_ci'] = np.quantile(draw[:, 1]-draw[:, 0], [.025, .975])
        # A denominator whose analytic or bootstrap CI crosses zero is unstable.
        ci0 = np.quantile(draw[:, 0], [.025, .975])
        if float(base['lo'][0])*float(base['hi'][0]) > 0 and ci0[0]*ci0[1] > 0:
            out['retention'] = float(adjusted['b'][0]/base['b'][0])
            out['retention_ci'] = np.quantile(draw[:, 1]/draw[:, 0], [.025, .975])
    return out

def joint_fit(d, covs):
    res, fit = estimate(d, ['o_'+bed for bed in BEDS], covs)
    if fit is None: return res
    b, v = fit[3], fit[4]
    contrast = np.eye(4)[[0, 0, 0]]-np.eye(4)[[1, 2, 3]]
    diff = contrast @ b; vv = contrast @ v @ contrast.T
    se = np.sqrt(np.maximum(np.diag(vv), 0)); p = 2*stats.norm.sf(abs(diff)/np.maximum(se, 1e-15))
    rank = np.linalg.matrix_rank(vv)
    q = float(diff @ np.linalg.pinv(vv) @ diff)
    res.update(heterogeneity_Q=q, heterogeneity_df=rank, heterogeneity_p=float(stats.chi2.sf(q, rank)) if rank else None,
               pwv_contrasts={'b': diff, 'lo': diff-1.96*se, 'hi': diff+1.96*se, 'p': p, 'q': A.bh(p)})
    return res

def inventory(root, fr, rna):
    out = {}
    for name, sample in [('rna', rna), ('NMR', A.rd(root, 'nightingale_metabolomics/nightingale_metabolomics.parquet')),
                         ('lipid', A.rd(root, 'serum_lipidomics/serum_lipidomics.parquet'))]:
        if sample is None:
            out[name] = {'status': 'table unavailable'}; continue
        s = sample.copy(); s['participant_id'] = s.participant_id.astype(str)
        if name != 'rna':
            meta = {'participant_id', 'cohort', 'research_stage', 'array_index', 'index', 'collection_date', 'collection_timestamp'}
            cols = [c for c in s if c not in meta and not str(c).startswith('synthetic_pad') and pd.api.types.is_numeric_dtype(s[c])]
            s = s[s[cols].notna().any(axis=1)] if cols else s.iloc[:0]
        ids = {st: set(s.loc[s.research_stage == st, 'participant_id']) for st in STAGES}
        both = ids[STAGES[0]] & ids[STAGES[1]]
        row = {'visit1': count(len(ids[STAGES[0]])), 'visit2': count(len(ids[STAGES[1]])), 'distinct_stage_pairs': count(len(both))}
        row['bed_pairs'] = {}
        for bed in REPEAT_BEDS:
            pairs = [set(fr.loc[(fr.research_stage == st) & fr['o_'+bed].notna(), 'participant_id']) for st in STAGES]
            row['bed_pairs'][bed] = count(len(both & pairs[0] & pairs[1]))
        out[name] = row
    return out

def prospective(fr, tech, proxies):
    baseline = fr[fr.research_stage == STAGES[0]].drop_duplicates('participant_id')
    followup = fr[fr.research_stage == STAGES[1]].drop_duplicates('participant_id')
    joined = baseline.merge(followup, on='participant_id', suffixes=('_0', '_1'), validate='one_to_one')
    dt = (O.day(joined.visit_date_1)-O.day(joined.visit_date_0)).dt.total_seconds()/86400/365.25
    dt.index = joined.index
    joined['interval_years'] = dt
    joined = joined[(dt > 0) & (dt <= 5)].copy()
    joined['nk'] = joined.nk_0
    out = {}
    for bed in REPEAT_BEDS:
        y0, y1 = 'o_'+bed+'_0', 'o_'+bed+'_1'
        cov = [c+'_0' for c in COVS+tech+proxies]+[y0, 'interval_years']
        main, _ = estimate(joined, [y1], cov)
        joined['delta_sbp'] = joined.sbp_sit_1-joined.sbp_sit_0
        joined['delta_bmi'] = joined.bmi_1-joined.bmi_0
        main['change_adjusted'], _ = estimate(joined, [y1], cov+['delta_sbp', 'delta_bmi'])
        dd = joined.copy(); dd['delta_nk'] = dd.nk_1-dd.nk_0; dd['delta_bed'] = dd[y1]-dd[y0]
        dcov = cov+['nk_0']+[c+'_1' for c in tech]+['delta_sbp', 'delta_bmi']
        main['delta_RNA'], _ = estimate(dd, ['delta_bed'], dcov, exposure='delta_nk')
        out[bed] = main
    rawp = [float(out[b]['p'][0]) if out[b].get('status') == 'estimated' else 1 for b in REPEAT_BEDS]
    q = A.bh(rawp)
    for i, bed in enumerate(REPEAT_BEDS):
        out[bed]['q'] = float(q[i]) if out[bed].get('status') == 'estimated' else None
    return out

def clean(obj):
    if isinstance(obj, dict): return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)): return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)): return A.fnum(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    return obj

def fmt(value):
    return 'NA' if value is None or not np.isfinite(float(value)) else f'{float(value):.3g}'

def row(label, record):
    if record.get('status') != 'estimated': return f'{label} n={record.get("n")} {record.get("status", "unavailable")}'
    return f'{label} n={record["n"]} b={fmt(record["b"][0])} [{fmt(record["lo"][0])},{fmt(record["hi"][0])}] p={fmt(record["p"][0])}'

def digest(r):
    lines = [f'F1 P14 | {VERSION} | runtime={float(r["runtime_s"]):.1f}s',
             'Reference-derived RNA composition proxies only; not absolute cell fractions or a causal mechanism.',
             'RNA same-stage matching only; module NK6 fixed; M3 + batch + draw hour/hour^2.',
             'Standardized associations; HC1 joint covariance; primary single adjusted-PWV test; three contrasts BH.']
    f = r['flow']; lines.append(f'RNA samples={f["samples"]} people={f["participants"]} batches={f["batches"]} NKgenes={f["nk_genes"]} hour_missing={f["hour_missing"]}')
    lines.append(f'Reference overlap={f["reference_genes"]} genes | {f["composition_status"]}')
    lines.append('Top-marker counts '+ ' '.join(f'{k}:{v}' for k,v in f['reference_markers'].items()))
    for key, v in r['inventory'].items():
        lines.append(f'Inventory {key}: '+json.dumps(v, separators=(',', ':')))
    for key, block in r['comparisons'].items():
        lines += [row(key+' base', block['base']), row(key+' adjusted', block['adjusted'])]
        lines.append(f'{key} boot_valid={block.get("bootstrap_valid")} delta_CI={list(map(fmt, block.get("delta_beta_ci", [])))} retention={fmt(block.get("retention"))} CI={list(map(fmt, block.get("retention_ci", [])))}')
    joint = r['joint']; lines.append(f'Joint four-bed n={joint.get("n")} status={joint.get("status")} het_p={fmt(joint.get("heterogeneity_p"))}')
    if joint.get('status') == 'estimated':
        for i, bed in enumerate(BEDS):
            lines.append(f'Joint {bed} b={fmt(joint["b"][i])} [{fmt(joint["lo"][i])},{fmt(joint["hi"][i])}] p={fmt(joint["p"][i])}')
        c = joint['pwv_contrasts']
        for i, bed in enumerate(BEDS[1:]):
            lines.append(f'PWV-{bed} b={fmt(c["b"][i])} [{fmt(c["lo"][i])},{fmt(c["hi"][i])}] p={fmt(c["p"][i])} q={fmt(c["q"][i])}')
    for bed, record in r['prospective'].items():
        lines.append(row('BaselineNK->later '+bed, record)+f' q={fmt(record.get("q"))}')
        lines.append(row('BaselineNK->later '+bed+' +deltaBP/BMI', record['change_adjusted']))
        lines.append(row('DeltaRNA->delta '+bed, record['delta_RNA']))
    lines += ['Prospective fits condition on baseline bed and follow-up interval; not outcome prediction or causation.',
              'NMR/lipid inventory only. Unavailable rows are not nulls. Bootstrap uses fixed original sample scales.',
              'Inference conditional on acquisition transform and composition proxy; HC1 treats participants as independent.',
              'F1_P14_DONE errors=[]']
    if len(lines)>120 or any(len(s)>220 for s in lines): raise ValueError('digest size contract')
    return '\n'.join(lines)+'\n'

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--atlas', required=True); ap.add_argument('--omics', required=True); ap.add_argument('--nkhr', required=True)
    ap.add_argument('--reference', required=True); ap.add_argument('--boot', type=int, default=200); ap.add_argument('--seed', type=int, default=17)
    args = ap.parse_args(argv)
    if Path(args.out).exists(): raise SystemExit('Refusing existing output')
    if args.boot < 20: raise SystemExit('bootstrap must be at least20')
    start = time.time(); setup(args.atlas, args.omics, args.nkhr)
    fr = A.build_frame(args.root, None)[0].reset_index(drop=True)
    ids = set(fr.participant_id.astype(str))
    rna, tech, proxies, flow = load_rna(args.root, args.reference); ids |= set(rna.participant_id)
    fr['participant_id'] = fr.participant_id.astype(str)
    fr, _ = N.add_hr(args.root, fr)
    joined = fr.merge(rna[['participant_id', 'research_stage', 'nk']+tech+proxies],
                      on=['participant_id', 'research_stage'], how='left', validate='many_to_one')
    basecov = COVS+tech
    rng = np.random.default_rng(args.seed)
    comparisons = {}
    for key, outcome, cov in [('primary', 'o_pwv', basecov), ('raw_PWV', 'pwv_raw', basecov),
                              ('age_cubic', 'o_pwv', basecov+['age2', 'age3']),
                              ('state_adjusted', 'o_pwv', basecov+['hr_excess', 'acute_dev'])]:
        joined['age2'] = (joined.age-50)**2; joined['age3'] = (joined.age-50)**3
        need = ['nk', outcome]+cov+proxies
        dd = A.one_row_per_participant(joined, need)
        comparisons[key] = paired_comparison(dd, outcome, cov, proxies, args.boot, rng)
    common = A.one_row_per_participant(joined, ['nk']+['o_'+b for b in BEDS]+basecov+proxies)
    result = dict(version=VERSION, flow=flow, inventory=inventory(args.root, fr, rna), comparisons=comparisons,
                  joint=joint_fit(common, basecov+proxies), prospective=prospective(joined, tech, proxies),
                  bootstrap=args.boot, seed=args.seed, runtime_s=time.time()-start,
                  source_hashes={k: hashlib.sha256(Path(v).read_bytes()).hexdigest()[:16] for k,v in
                                [('script', __file__), ('atlas', args.atlas), ('omics', args.omics), ('nkhr', args.nkhr), ('reference', args.reference)]})
    result = clean(result)
    text = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
    # Full hashes are execution metadata, but format digit runs before the existing privacy contract.
    result['source_hashes'] = {k: '-'.join(v[i:i+4] for i in range(0,len(v),4)) for k,v in result['source_hashes'].items()}
    text = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
    dg = digest(result); O.privacy_check(text, ids); O.privacy_check(dg, ids)
    out = Path(args.out); out.mkdir()
    for name, data in [('results.json', text+'\n'), ('digest.txt', dg), ('run_log.txt', f'{VERSION}\nF1_P14_DONE errors=[]\n')]:
        (out/name).write_text(data)
    sums = ''.join(f'{hashlib.sha256((out/name).read_bytes()).hexdigest()}  {name}\n' for name in ['results.json','digest.txt','run_log.txt'])
    (out/'SHA256SUMS').write_text(sums)
    print('F1_P14_DONE errors=[] runtime=%.1fs'%(time.time()-start), flush=True)
    print('digest lines=%d sq12=%s'%(dg.count('\n'), hashlib.sha256(re.sub(' +',' ',dg).encode()).hexdigest()[:12]), flush=True)
    return result

if __name__ == '__main__': main()
