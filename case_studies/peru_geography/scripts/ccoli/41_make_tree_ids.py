#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from coli_common import read_table,filter_species,normalise_source,choose_group_col,derive_group_series

def main():
    p=argparse.ArgumentParser(); p.add_argument('--human',required=True); p.add_argument('--peru-source',required=True); p.add_argument('--global-source'); p.add_argument('--global-max',type=int,default=0); p.add_argument('--out-ids',required=True); p.add_argument('--out-meta',required=True); p.add_argument('--id-col',default='id'); p.add_argument('--source-col',default='source'); p.add_argument('--species-col',default='species'); p.add_argument('--species-value',default='C. coli'); p.add_argument('--seed',type=int,default=42)
    a=p.parse_args(); h=filter_species(read_table(a.human),a.species_col,a.species_value); l=filter_species(read_table(a.peru_source),a.species_col,a.species_value); h['tree_role']='Human_Peru'; l['tree_role']='Source_Peru'; frames=[h,l]
    if a.global_source and a.global_max>0:
        g=filter_species(read_table(a.global_source),a.species_col,a.species_value); g=g[~g[a.id_col].astype(str).isin(set(h[a.id_col].astype(str))|set(l[a.id_col].astype(str)))].copy(); g['tree_role']='Source_Global'
        g['__source']=g[a.source_col].map(normalise_source) if a.source_col in g else 'Unknown'
        try:
            gc=choose_group_col(g.set_index(a.id_col,drop=False),'auto'); g['__group']=derive_group_series(g.set_index(a.id_col,drop=False),gc).values
        except Exception: g['__group']=g[a.id_col].astype(str)
        reps=g.groupby(['__source','__group'],dropna=False,sort=False).sample(n=1,random_state=a.seed) if len(g) else g
        if len(reps)>=a.global_max: gsel=reps.sample(n=a.global_max,random_state=a.seed)
        else:
            rem=g[~g.index.isin(reps.index)]; n=min(a.global_max-len(reps),len(rem)); gsel=pd.concat([reps,rem.sample(n=n,random_state=a.seed)],ignore_index=False)
        frames.append(gsel)
    d=pd.concat(frames,ignore_index=True,sort=False); d[a.id_col]=d[a.id_col].astype(str); d=d.drop_duplicates(a.id_col,keep='first'); ids=Path(a.out_ids); ids.parent.mkdir(parents=True,exist_ok=True); ids.write_text('\n'.join(sorted(d[a.id_col]))+'\n'); d.to_csv(a.out_meta,sep='\t',index=False); print(f'{len(d)} tree taxa -> {ids}'); print(d['tree_role'].value_counts().to_string())
if __name__=='__main__': main()
