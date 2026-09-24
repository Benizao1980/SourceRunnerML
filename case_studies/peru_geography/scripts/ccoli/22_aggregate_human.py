#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--parts',required=True); p.add_argument('--outdir',required=True); a=p.parse_args(); parts=Path(a.parts); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for pan in ['Global_only_nonPeru','Peru_local','Global_plus_Peru']:
        f=parts/f'human_predictions__{pan}.tsv'; d=pd.read_csv(f,sep='\t',dtype={'id':str}); d.to_csv(out/f.name,sep='\t',index=False)
        vc=d['filtered_prediction'].value_counts()
        for src,n in vc.items(): rows.append({'reference':pan,'filtered_prediction':src,'n':int(n),'proportion':float(n/len(d))})
    pd.DataFrame(rows).to_csv(out/'human_prediction_summary.tsv',sep='\t',index=False); print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
