"""Synthetic-only group boundaries, nested preprocessing and CV-only integration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from sourcerunner_validation import (ValidationDesignError, grouped_splits,
    prepare_grouped_validation, collision_audit, audit_training_transform, load_validation_config)
from sourcerunner_full_validation import make_pipeline
from sourcerunner_features import AllelePreprocessor
from sourcerunner_tuning import fit_with_optional_tuning


def synthetic():
    # Every group includes both sources; unique profiles are separable from group metadata.
    rows=[]
    for g in range(12):
        for c in range(2):
            for j in range(2):
                rows.append({'id':f's{g}_{c}_{j}','lineage':f'L{g:02}',
                             'source':'A' if c==0 else 'B',
                             'CAMP1':g*10+c*2+j+1,'CAMP2':c+1})
    return pd.DataFrame(rows)


class GroupedValidationTests(unittest.TestCase):
    def setUp(self):
        self.frame=synthetic(); self.X=self.frame[['CAMP1','CAMP2']]
        self.y=self.frame.source.map({'A':0,'B':1}).to_numpy()
        self.groups=self.frame.lineage.to_numpy()
        self.config={'strategy':'grouped','group_column':'lineage','id_column':'id','collision_policy':'error'}

    def test_deterministic_complete_group_and_class_support(self):
        splits=grouped_splits(self.X,self.y,self.groups,3,42)
        again=grouped_splits(self.X,self.y,self.groups,3,42)
        for (tr,te),(a,b) in zip(splits,again):
            np.testing.assert_array_equal(tr,a); np.testing.assert_array_equal(te,b)
            self.assertFalse(set(self.groups[tr]) & set(self.groups[te]))
            self.assertEqual(set(self.y[tr]),{0,1}); self.assertEqual(set(self.y[te]),{0,1})
        self.assertEqual(sorted(np.concatenate([te for _,te in splits])),list(range(len(self.y))))

    def test_ids_and_infeasible_design_fail(self):
        bad=self.frame.copy(); bad.loc[0,'id']=bad.loc[1,'id']
        with self.assertRaisesRegex(ValidationDesignError,'Duplicate'):
            prepare_grouped_validation(bad,self.X,self.y,self.config,3,42)
        bad=self.frame.copy(); bad.loc[0,'lineage']=' '
        with self.assertRaisesRegex(ValidationDesignError,'Missing'):
            prepare_grouped_validation(bad,self.X,self.y,self.config,3,42)
        with self.assertRaisesRegex(ValidationDesignError,'fewer than'):
            grouped_splits(self.X,self.y,np.where(self.y==0,'one',self.groups),3,42)
        config=dict(self.config,group_column='CAMP1')
        with self.assertRaisesRegex(ValidationDesignError,'metadata columns'):
            prepare_grouped_validation(self.frame,self.X,self.y,config,3,42)

        with self.assertRaisesRegex(ValidationDesignError,'metadata columns'):
            prepare_grouped_validation(self.frame,self.X.assign(source=self.y),self.y,
                                       self.config,3,42,source_column='source')

    def test_raw_and_transformed_collisions_detected(self):
        with self.assertRaisesRegex(ValidationDesignError,'collision'):
            collision_audit([[1,np.nan]],[[1,np.nan]])
        self.assertEqual(collision_audit([[1]],[[1]],'warn')['distinct_shared_profiles'],1)
        # Test and training differ at B, but training-only filtering drops B.
        X=pd.DataFrame({'A':[1.,1.,1.],'B':[np.nan,np.nan,99.]})
        with self.assertRaisesRegex(ValidationDesignError,'transformed'):
            audit_training_transform(X,np.array([0,1]),np.array([2]),.2,'error')

    def test_nested_fits_never_contain_outer_or_inner_heldout_groups(self):
        tr,te=grouped_splits(self.X,self.y,self.groups,3,42)[0]
        observed=[]; original=AllelePreprocessor.fit
        def spy(step,X,y=None):
            observed.append(set(self.frame.loc[X.index,'lineage']))
            return original(step,X,y)
        with patch.object(AllelePreprocessor,'fit',spy):
            model,details=fit_with_optional_tuning(
                make_pipeline('random_forest',1,42,.2),self.X.iloc[tr],self.y[tr],
                model_name='random_forest',grids={'random_forest':{'n_estimators':[3,5]}},
                metric='balanced_accuracy',inner_folds=2,seed=42,groups=self.groups[tr])
        outer_test=set(self.groups[te]); outer_train=set(self.groups[tr])
        self.assertTrue(all(not (fit & outer_test) for fit in observed))
        expected=[set(p['train_groups']) for p in details['inner_group_partitions']]
        self.assertTrue(all(fit in expected+[outer_train] for fit in observed))
        for p in details['inner_group_partitions']:
            self.assertFalse(set(p['train_groups']) & set(p['test_groups']))
        self.assertEqual(len(model.predict(self.X.iloc[te])),len(te))

    def test_changed_outer_values_cannot_affect_tuned_model(self):
        tr,te=grouped_splits(self.X,self.y,self.groups,3,42)[0]
        changed=self.X.copy(); changed.loc[self.X.index[te],'CAMP1']=999999
        fits=[]
        for frame in (self.X,changed):
            model,detail=fit_with_optional_tuning(make_pipeline('random_forest',1,42,.2),
                frame.iloc[tr],self.y[tr],model_name='random_forest',
                grids={'random_forest':{'n_estimators':[3,5]}},metric='balanced_accuracy',
                inner_folds=2,seed=42,groups=self.groups[tr])
            fits.append((model,detail))
        self.assertEqual(fits[0][1],fits[1][1])
        np.testing.assert_array_equal(fits[0][0].named_steps['alleles'].imputer_.statistics_,
                                      fits[1][0].named_steps['alleles'].imputer_.statistics_)

    def test_grouped_cli_is_cv_only_and_writes_reusable_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); self.frame.to_csv(p/'train.tsv',sep='\t',index=False)
            self.frame.iloc[:2].to_csv(p/'query.tsv',sep='\t',index=False)
            (p/'design.json').write_text(json.dumps(self.config))
            (p/'tune.json').write_text(json.dumps({'random_forest':{'n_estimators':[3,5]}}))
            cmd=[sys.executable,str(ROOT/'scripts/sourcerunner_full_validation.py'),
                 '--train_file',str(p/'train.tsv'),'--predict_file',str(p/'query.tsv'),
                 '--source_col','source','--keep_sources','A,B','--models','random_forest',
                 '--validation_config',str(p/'design.json'),'--tuning_config',str(p/'tune.json'),
                 '--inner_cv_folds','2','--cv_folds','3','--cpus','1','--output_dir',str(p/'outputs')]
            refused=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(refused.returncode,0); self.assertIn('requires --cv_only',refused.stderr)
            result=subprocess.run(cmd+['--cv_only'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            out=next((p/'outputs').iterdir())
            memberships=pd.read_csv(out/'validation_fold_membership.tsv',sep='\t')
            for _,fold in memberships.groupby('fold'):
                self.assertFalse(set(fold.loc[fold.role=='train','group']) & set(fold.loc[fold.role=='test','group']))
            oof=pd.read_csv(out/'cv_oof_predictions__random_forest.tsv',sep='\t')
            self.assertEqual(set(oof.sample_id),set(self.frame.id))
            np.testing.assert_allclose(oof.filter(regex='^prob_').sum(axis=1),1.)
            manifest=json.loads((out/'validation_manifest.json').read_text())
            self.assertEqual(manifest['seed'],42)
            self.assertIn('validation_config',manifest['input_sha256'])
            self.assertFalse((out/'final_model_fit_all_training.pkl').exists())
            self.assertFalse((out/'bootstrap_oob_metrics_all_replicates.tsv').exists())

    def test_unknown_config_does_not_silently_downgrade(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.json'; p.write_text('{"strategy":"temporal"}')
            with self.assertRaises(ValidationDesignError): load_validation_config(p)


if __name__=='__main__': unittest.main()
