#!/usr/bin/env python3
"""Read only the public MARK SAV; explicit labels and QC, no associations."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

PAPER = Path(os.environ.get("F1_WORKDIR", Path(__file__).resolve().parents[1]))
RAW = PAPER / 'external/small_open/f1_mark_2015/mark_s1.sav'
EXPECTED = 'd9bd41cb191a4b1d7cae3f3d81be95c32cc0369861d3356735f7ff43f3b3094b'
MAP = {'Age':'age','Sex':'male','Somker':'smoker','Drugs_HTA':'aht','BMI':'bmi',
       'SBP':'sbp','DBP':'dbp','HR':'hr','Glucose':'glucose','HDL_C':'hdl','CAVI':'cavi','IMT':'imt'}
BOUNDS = {'age':(18,100),'bmi':(10,70),'sbp':(60,260),'dbp':(30,160),'hr':(20,200),
          'glucose':(20,600),'hdl':(5,200),'cavi':(1,30),'imt':(.1,3)}

def main():
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == EXPECTED
    d, meta = pyreadstat.read_sav(RAW)
    assert len(d) == 500
    for c, lab in [('Sex',{0.:'Women',1.:'Men'}),('Somker',{0.:'No',1.:'Yes'}),
                   ('Drugs_HTA',{0.:'No',1.:'Yes'})]:
        assert meta.variable_value_labels[c] == lab
    x = d[list(MAP)].rename(columns=MAP).replace([np.inf,-np.inf],np.nan)
    for c in ['male','smoker','aht']:
        assert set(x[c].dropna().unique()) <= {0.,1.}
    for c, (lo,hi) in BOUNDS.items():
        assert x[c].dropna().between(lo,hi).all(), c
    x['pp'] = x.sbp-x.dbp
    assert x.pp.dropna().between(5,160).all()
    out = PAPER/'results/stage0/clean/mark_2015.parquet'
    assert not out.exists(), 'Refusing existing clean table'
    x.to_parquet(out,index=False)
    receipt = {'raw_path':str(RAW),'raw_sha256':EXPECTED,'clean_path':str(out),
               'clean_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'rows':len(x),
               'complete_primary':int(x.notna().all(axis=1).sum()),'labels_verified':True,
               'range_assertions_passed':True,'variable_mapping':MAP,'fits_run':False}
    (PAPER/'results/stage0/feasibility_mark_20261007/clean_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    manifest = PAPER/'results/stage0/clean/MANIFEST.tsv'
    with manifest.open('a') as f:
        f.write('mark_2015\t%d\t%d\t%s\n'%(len(x),len(x.columns),receipt['clean_sha256']))
    print(json.dumps(receipt))

if __name__=='__main__': main()
