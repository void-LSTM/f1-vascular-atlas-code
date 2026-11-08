#!/usr/bin/env python3
"""Freeze public reference proxy columns; never reads HPP data."""
import argparse
import hashlib
import json
from pathlib import Path
import importlib.util
import pandas as pd

GROUPS = {
    'NK': ['NK'], 'CD4': ['T CD4 Naive', 'T CD4 Memory'],
    'CD8': ['T CD8 Memory', 'T CD8 Naive'], 'B': ['B Naive', 'B Memory', 'Plasmablasts'],
    'mono': ['Monocytes C', 'Monocytes NC+I'], 'DC': ['mDCs', 'pDCs'],
    'otherT': ['MAIT', 'T gd Vd2', 'T gd non-Vd2'],
    'LDneut': ['Neutrophils LD'], 'LDbaso': ['Basophils LD'],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    source, out = Path(a.source), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == 'a86eaadb747a9d69e9a22df2e0d01af785244211c68111b568e603d57816bb23'
    root = Path(__file__).resolve().parent
    excluded = set()
    for f in ['f1_omics_v1_2.py', 'f1_nk_hr_v1.py']:
        spec = importlib.util.spec_from_file_location('reference_definitions', root/f)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        excluded.update(g for genes in module.MODULES.values() for g in genes)
    x = pd.read_csv(source, sep='\t', index_col=0)
    x = x.loc[~x.index.isin(excluded)]
    ref = pd.DataFrame({k: x[v].mean(axis=1) for k, v in GROUPS.items()})
    ref.index.name = 'gene'
    ref.to_csv(out/'pbmc_reference_proxy.tsv', sep='\t', float_format='%.10g')
    meta = dict(source_url='https://raw.githubusercontent.com/giannimonaco/ABIS/master/data/sigmatrixRNAseq.txt',
                source_sha256=hashlib.sha256(raw).hexdigest(), groups=GROUPS,
                excluded_programme_genes=sorted(excluded), genes=len(ref),
                interpretation='RNA composition proxy; not validated cell fractions or ABIS absolute deconvolution')
    (out/'REFERENCE.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(f'Frozen {len(ref)} genes x {len(ref.columns)} coarse reference columns.')

if __name__ == '__main__':
    main()
