#!/usr/bin/env python3
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd

SOURCE_MAP = {
    'chicken':'Poultry','broiler':'Poultry','broiler chicken':'Poultry','hen':'Poultry','poultry':'Poultry','turkey':'Poultry',
    'cattle':'Ruminant','cow':'Ruminant','beef':'Ruminant','sheep':'Ruminant','goat':'Ruminant','ruminant':'Ruminant','ruminants':'Ruminant',
    'pig':'Pig','swine':'Pig','hog':'Pig','pork':'Pig',
}

def read_table(path):
    path=Path(path)
    if not path.exists() or path.stat().st_size==0: raise FileNotFoundError(f'Missing/empty input: {path}')
    return pd.read_csv(path,sep='\t' if path.suffix.lower() in {'.tsv','.txt'} else ',',dtype=str,low_memory=False)

def normalise_species(s):
    if pd.isna(s): return s
    x=str(s).strip().lower().replace('_',' ')
    if x in {'c. coli','c coli','campylobacter coli','coli'}: return 'C. coli'
    if x in {'c. jejuni','c jejuni','campylobacter jejuni','jejuni'}: return 'C. jejuni'
    return str(s).strip()

def normalise_source(s):
    if pd.isna(s): return s
    x=re.sub(r'\s+',' ',str(s).strip().lower())
    if x in SOURCE_MAP: return SOURCE_MAP[x]
    if 'chicken' in x or 'poultry' in x or 'broiler' in x: return 'Poultry'
    if any(k in x for k in ['cattle','cow','sheep','goat','ruminant','beef']): return 'Ruminant'
    if any(k in x for k in ['pig','swine','pork','hog']): return 'Pig'
    return str(s).strip()

def filter_species(df,species_col,species_value):
    if species_col and species_col in df.columns:
        want=normalise_species(species_value); vals=df[species_col].map(normalise_species); return df.loc[vals.eq(want)].copy()
    return df.copy()

def detect_loci(dfs,prefix='CAMP',regex=''):
    common=set(dfs[0].columns)
    for d in dfs[1:]: common &= set(d.columns)
    cols=sorted(common)
    loci=[c for c in cols if re.compile(regex).search(c)] if regex else [c for c in cols if str(c).startswith(prefix)]
    if not loci: raise ValueError(f'No shared loci detected with prefix={prefix!r} regex={regex!r}')
    return loci

def choose_group_col(df,requested='auto'):
    if requested and requested!='auto':
        if requested not in df.columns: raise ValueError(f'Requested GROUP_COL {requested!r} not found')
        return requested
    for c in ['LIN17','LIN_17','LINcode17','cgMLST_CC','cgMLST clonal complex','MLST_CC','clonal_complex','ST']:
        if c in df.columns and df[c].notna().any(): return c
    if 'LINcode' in df.columns: return '__DERIVED_LIN17__'
    raise ValueError('Could not find a lineage grouping column. Set GROUP_COL in config_coli.sh')

def derive_group_series(df,group_col):
    if group_col!='__DERIVED_LIN17__':
        s=df[group_col].astype(str).replace({'nan':np.nan,'None':np.nan,'':np.nan}); return s.fillna('__ID__'+df.index.astype(str))
    def f(x):
        if pd.isna(x): return None
        parts=[p for p in re.split(r'[_\.-]',str(x).strip()) if p!='']; return '_'.join(parts[:18]) if parts else None
    return df['LINcode'].map(f).fillna('__ID__'+df.index.astype(str))

def analysis_sources(s):
    return [normalise_source(x) for x in str(s).split(',') if str(x).strip()]
