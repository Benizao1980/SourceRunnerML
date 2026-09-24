#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
import pandas as pd
from coli_common import read_table
from coli_xgb_ensemble import fit_predict_ensemble

def main():
    p=argparse.ArgumentParser(); p.add_argument('--panel',required=True,choices=['Global_only_nonPeru','Peru_local_OOF','Global_plus_Peru_OOF']); p.add_argument('--fold',type=int,required=True)
    p.add_argument('--global-file',required=True); p.add_argument('--peru-file',required=True); p.add_argument('--manifest',required=True); p.add_argument('--outdir',required=True)
    p.add_argument('--id-col',default='id'); p.add_argument('--missingness',type=float,default=.2); p.add_argument('--cpus',type=int,default=8); p.add_argument('--loci-prefix',default='CAMP'); p.add_argument('--loci-regex',default='')
    a=p.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    g=read_table(a.global_file); l=read_table(a.peru_file); m=pd.read_csv(a.manifest,dtype={'id':str}); g[a.id_col]=g[a.id_col].astype(str); l[a.id_col]=l[a.id_col].astype(str)
    l=l.merge(m[['id','fold']],left_on=a.id_col,right_on='id',how='inner',validate='one_to_one',suffixes=('','_m'))
    if a.panel=='Global_only_nonPeru': train=g; test=l.copy(); tag='all'
    else:
        if a.fold<0: raise ValueError('OOF panels require fold >=0')
        train_local=l[l['fold']!=a.fold].drop(columns=['fold']); test=l[l['fold']==a.fold].copy(); tag=str(a.fold)
        train=train_local if a.panel=='Peru_local_OOF' else pd.concat([g,train_local],ignore_index=True,sort=False)
    common=set(train.columns)&set(test.columns); loci=sorted([c for c in common if (re.search(a.loci_regex,c) if a.loci_regex else str(c).startswith(a.loci_prefix))])
    res=fit_predict_ensemble(train,test,loci,missingness=a.missingness,cpus=a.cpus)
    o=pd.DataFrame({'id':test[a.id_col].astype(str).values,'fold':test['fold'].values,'true_source':test['source_class'].values,'predicted_source':res['pred'],'max_probability':res['max_prob']})
    for j,c in enumerate(res['classes']): o['prob_'+str(c)]=res['prob'][:,j]
    o['filtered_prediction']=o['predicted_source'].where(o['max_probability']>=0.60,'Uncertain')
    fn=out/f'benchmark__{a.panel}__fold{tag}.tsv'; o.to_csv(fn,sep='\t',index=False)
    meta={'panel':a.panel,'fold':a.fold,'n_train':len(train),'n_test':len(test),'n_loci':len(res['loci']),'classes':res['classes']}; (out/f'metadata__{a.panel}__fold{tag}.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta,indent=2)); print(fn)
if __name__=='__main__': main()
