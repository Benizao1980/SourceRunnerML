#!/usr/bin/env python3
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

def fit_predict_ensemble(train,test,loci,source_col='source_class',missingness=0.20,cpus=8,seeds=range(25,75)):
    Xtr=train[loci].apply(pd.to_numeric,errors='coerce'); Xte=test[loci].apply(pd.to_numeric,errors='coerce')
    keep=Xtr.columns[Xtr.isna().mean().le(float(missingness))].tolist()
    if len(keep)<10: raise ValueError(f'Only {len(keep)} loci survive missingness threshold')
    imp=SimpleImputer(strategy='most_frequent'); A=imp.fit_transform(Xtr[keep]); B=imp.transform(Xte[keep])
    le=LabelEncoder().fit(train[source_col].astype(str)); y=le.transform(train[source_col].astype(str)); classes=list(le.classes_)
    probs=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); idx=[]
        for c in range(len(classes)):
            w=np.flatnonzero(y==c)
            if len(w): idx.extend(rng.choice(w,size=len(w),replace=True).tolist())
        idx=np.asarray(idx,dtype=int); rng.shuffle(idx)
        kw=dict(n_estimators=300,max_depth=6,learning_rate=0.05,subsample=0.85,colsample_bytree=0.85,n_jobs=int(cpus),random_state=int(seed),verbosity=0,tree_method='hist')
        if len(classes)==2: model=XGBClassifier(objective='binary:logistic',eval_metric='logloss',**kw)
        else: model=XGBClassifier(objective='multi:softprob',num_class=len(classes),eval_metric='mlogloss',**kw)
        model.fit(A[idx],y[idx]); p=model.predict_proba(B)
        if p.ndim==1: p=np.column_stack([1-p,p])
        probs.append(p)
    meanp=np.mean(np.stack(probs,axis=0),axis=0); winner=meanp.argmax(axis=1)
    return {'classes':classes,'loci':keep,'prob':meanp,'pred':np.array(classes,dtype=object)[winner],'max_prob':meanp.max(axis=1)}
