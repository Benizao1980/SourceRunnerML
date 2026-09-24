#!/usr/bin/env python3
import argparse,re
from pathlib import Path
import numpy as np,pandas as pd
from coli_common import read_table

def main():
    p=argparse.ArgumentParser(); p.add_argument('--sources',required=True); p.add_argument('--human',required=True); p.add_argument('--out',required=True); p.add_argument('--id-col',default='id'); p.add_argument('--source-col',default='source_class'); p.add_argument('--loci-prefix',default='CAMP'); p.add_argument('--loci-regex',default=''); p.add_argument('--min-comparable',type=int,default=100)
    a=p.parse_args(); s=read_table(a.sources); h=read_table(a.human); common=set(s.columns)&set(h.columns); loci=sorted([c for c in common if (re.search(a.loci_regex,c) if a.loci_regex else str(c).startswith(a.loci_prefix))])
    S=s[loci].apply(pd.to_numeric,errors='coerce').to_numpy(float); H=h[loci].apply(pd.to_numeric,errors='coerce').to_numpy(float); src=s[a.source_col].astype(str).to_numpy(); sid=s[a.id_col].astype(str).to_numpy(); rows=[]
    cats=sorted(pd.unique(src)); mats={c:np.flatnonzero(src==c) for c in cats}
    for i,x in enumerate(H):
        rec={'id':str(h.iloc[i][a.id_col])}; best=(np.inf,None,None,None)
        for c,idx in mats.items():
            A=S[idx]; valid=np.isfinite(A)&np.isfinite(x); comp=valid.sum(axis=1); diff=((A!=x)&valid).sum(axis=1); prop=np.divide(diff,comp,out=np.full(len(idx),np.inf),where=comp>=a.min_comparable); j=int(np.argmin(prop))
            rec[f'nearest_{c}_distance']=float(prop[j]) if np.isfinite(prop[j]) else np.nan; rec[f'nearest_{c}_id']=sid[idx[j]] if np.isfinite(prop[j]) else ''; rec[f'nearest_{c}_comparable_loci']=int(comp[j]) if np.isfinite(prop[j]) else 0
            if prop[j]<best[0]: best=(prop[j],c,sid[idx[j]],int(comp[j]))
        rec['nearest_source']=best[1]; rec['nearest_source_id']=best[2]; rec['nearest_source_distance']=best[0] if np.isfinite(best[0]) else np.nan; rec['nearest_source_comparable_loci']=best[3]; rows.append(rec)
        if (i+1)%25==0: print(f'{i+1}/{len(H)} humans processed',flush=True)
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,sep='\t',index=False); print(out)
if __name__=='__main__': main()
