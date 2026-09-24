#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,recall_score

def metrics(d):
    y=d.true_source.astype(str); p=d.predicted_source.astype(str); labels=sorted(y.unique())
    z={'n':len(d),'accuracy':accuracy_score(y,p),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,labels=labels,average='macro',zero_division=0)}
    for lab in labels: z[f'{lab}_recall']=recall_score(y,p,labels=[lab],average='macro',zero_division=0)
    return z

def main():
    p=argparse.ArgumentParser(); p.add_argument('--parts',required=True); p.add_argument('--outdir',required=True); a=p.parse_args(); parts=Path(a.parts); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    panels=['Global_only_nonPeru','Peru_local_OOF','Global_plus_Peru_OOF']; allp={}
    for pan in panels:
        fs=sorted(parts.glob(f'benchmark__{pan}__fold*.tsv'))
        if not fs: raise FileNotFoundError(f'No parts for {pan}')
        d=pd.concat([pd.read_csv(f,sep='\t',dtype={'id':str}) for f in fs],ignore_index=True)
        if pan=='Global_only_nonPeru': d=d.drop_duplicates('id')
        elif d['id'].duplicated().any(): raise ValueError(f'Duplicate OOF predictions in {pan}')
        allp[pan]=d; d.to_csv(out/f'production_predictions__{pan}.tsv',sep='\t',index=False)
    rows=[{'reference':pan,**metrics(d)} for pan,d in allp.items()]; pd.DataFrame(rows).to_csv(out/'production_benchmark_summary.tsv',sep='\t',index=False)
    fr=[]
    for pan,d in allp.items():
        for fold,b in d.groupby('fold'): fr.append({'reference':pan,'fold':int(fold),**metrics(b)})
    pd.DataFrame(fr).sort_values(['reference','fold']).to_csv(out/'production_benchmark_fold_metrics.tsv',sep='\t',index=False)
    metas=[]
    for f in sorted(parts.glob('metadata__*.json')):
        x=json.loads(f.read_text()); x['file']=f.name; metas.append(x)
    pd.DataFrame(metas).to_csv(out/'production_model_metadata.tsv',sep='\t',index=False); print(pd.DataFrame(rows).to_string(index=False)); print('\nFinal:',out)
if __name__=='__main__': main()
