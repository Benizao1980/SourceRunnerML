#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
import pandas as pd
from coli_common import read_table
from coli_xgb_ensemble import fit_predict_ensemble

def main():
    p=argparse.ArgumentParser(); p.add_argument('--panel',required=True,choices=['Global_only_nonPeru','Peru_local','Global_plus_Peru']); p.add_argument('--global-file',required=True); p.add_argument('--peru-file',required=True); p.add_argument('--human-file',required=True); p.add_argument('--outdir',required=True); p.add_argument('--id-col',default='id'); p.add_argument('--missingness',type=float,default=.2); p.add_argument('--cpus',type=int,default=8); p.add_argument('--loci-prefix',default='CAMP'); p.add_argument('--loci-regex',default='')
    a=p.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); g=read_table(a.global_file); l=read_table(a.peru_file); h=read_table(a.human_file)
    train=g if a.panel=='Global_only_nonPeru' else l if a.panel=='Peru_local' else pd.concat([g,l],ignore_index=True,sort=False)
    common=set(train.columns)&set(h.columns); loci=sorted([c for c in common if (re.search(a.loci_regex,c) if a.loci_regex else str(c).startswith(a.loci_prefix))])
    res=fit_predict_ensemble(train,h,loci,missingness=a.missingness,cpus=a.cpus)
    o=pd.DataFrame({'id':h[a.id_col].astype(str).values,'predicted_source':res['pred'],'max_probability':res['max_prob']})
    for j,c in enumerate(res['classes']): o['prob_'+str(c)]=res['prob'][:,j]
    o['filtered_prediction']=o['predicted_source'].where(o['max_probability']>=0.60,'Uncertain')
    fn=out/f'human_predictions__{a.panel}.tsv'; o.to_csv(fn,sep='\t',index=False); (out/f'metadata__{a.panel}.json').write_text(json.dumps({'panel':a.panel,'n_train':len(train),'n_human':len(h),'n_loci':len(res['loci']),'classes':res['classes']},indent=2)); print(fn)
if __name__=='__main__': main()
