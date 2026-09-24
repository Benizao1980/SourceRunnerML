#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from coli_common import read_table,filter_species,normalise_source,choose_group_col,derive_group_series,analysis_sources,detect_loci

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--global-source',required=True); p.add_argument('--peru-source',required=True); p.add_argument('--outdir',required=True)
    p.add_argument('--id-col',default='id'); p.add_argument('--source-col',default='source'); p.add_argument('--country-col',default='country')
    p.add_argument('--species-col',default='species'); p.add_argument('--species-value',default='C. coli'); p.add_argument('--sources',default='Poultry,Ruminant')
    p.add_argument('--group-col',default='auto'); p.add_argument('--folds',type=int,default=5); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--loci-prefix',default='CAMP'); p.add_argument('--loci-regex',default='')
    a=p.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    g=filter_species(read_table(a.global_source),a.species_col,a.species_value); l=filter_species(read_table(a.peru_source),a.species_col,a.species_value)
    for name,d in [('global',g),('peru',l)]:
        if a.id_col not in d or a.source_col not in d: raise ValueError(f'{name}: missing {a.id_col}/{a.source_col}')
        d[a.id_col]=d[a.id_col].astype(str); d['source_class']=d[a.source_col].map(normalise_source)
    sources=analysis_sources(a.sources); l=l[l['source_class'].isin(sources)].copy(); peru_ids=set(l[a.id_col])
    mask=~g[a.id_col].isin(peru_ids); removed_by_id=int((~mask).sum())
    if a.country_col in g.columns:
        peru_country=g[a.country_col].astype(str).str.strip().str.lower().eq('peru'); removed_by_country=int((mask & peru_country).sum()); mask &= ~peru_country
    else: removed_by_country=0
    g=g.loc[mask & g['source_class'].isin(sources)].copy()
    if l[a.id_col].duplicated().any(): raise ValueError('Peru source table has duplicate IDs after filtering')
    loci=detect_loci([g,l],a.loci_prefix,a.loci_regex)
    l_idx=l.set_index(a.id_col,drop=False); gc=choose_group_col(l_idx,a.group_col); groups=derive_group_series(l_idx,gc); y=l_idx['source_class']
    cv=StratifiedGroupKFold(n_splits=a.folds,shuffle=True,random_state=a.seed); fold=np.full(len(l_idx),-1,int); X=np.zeros((len(l_idx),1))
    for k,(_,te) in enumerate(cv.split(X,y,groups)): fold[te]=k
    if (fold<0).any(): raise RuntimeError('Some isolates did not receive a fold')
    man=pd.DataFrame({'id':l_idx[a.id_col].values,'source_class':y.values,'fold':fold,'group':groups.values})
    bad=man.groupby('group')['fold'].nunique(); bad=bad[bad>1]
    if len(bad): raise RuntimeError(f'{len(bad)} lineage groups cross folds')
    ct=pd.crosstab(man['fold'],man['source_class'])
    if (ct==0).any().any(): raise RuntimeError(f'At least one fold lacks a source class:\n{ct}')
    g.to_csv(out/'global_nonPeru_clean.tsv',sep='\t',index=False); l.to_csv(out/'peru_known_source_clean.tsv',sep='\t',index=False); man.to_csv(out/'coli_binary_5fold_manifest.csv',index=False); (out/'shared_loci.txt').write_text('\n'.join(loci)+'\n')
    meta={'removed_global_exact_peru_ids':removed_by_id,'removed_global_country_peru':removed_by_country,'global_n':len(g),'peru_n':len(l),'sources':sources,'group_column':gc,'n_loci_shared':len(loci),'fold_counts':ct.to_dict()}
    (out/'benchmark_setup.json').write_text(json.dumps(meta,indent=2,default=str)); print(json.dumps(meta,indent=2,default=str)); print('\nFold table:\n',ct)
if __name__=='__main__': main()
