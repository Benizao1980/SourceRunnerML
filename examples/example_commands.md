# Example commands

These commands assume you are running from the **repository root**.

For explanation of each step, see [`../docs/getting_started.md`](../docs/getting_started.md).

## 1. Preflight Campylobacter cgMLST inputs

```bash
python scripts/source_runner_preflight.py \
  --train TRAIN.cgmlst.tsv \
  --predict PREDICT.cgmlst.tsv \
  --source_col source \
  --id_col id \
  --loci_pattern '^CAMP' \
  --outdir preflight \
  --write_tsv
```

The explicit `--loci_pattern '^CAMP'` is important because the preflight utility's historical default locus list is the seven-locus MLST scheme rather than cgMLST.

## 2. Small debug run

Use this first to confirm that files, packages and output paths work.

```bash
python scripts/sourcerunner_full_validation.py \
  --train_file TRAIN.cgmlst.sourcerunner.train.tsv \
  --predict_file PREDICT.cgmlst.sourcerunner.predict.tsv \
  --output_dir sourcerunner_debug \
  --run_name debug \
  --source_col source \
  --keep_sources Poultry,Ruminant,Pig \
  --loci_prefix CAMP \
  --models random_forest,logreg \
  --cv_folds 3 \
  --bootstrap 5 \
  --burn_in 1 \
  --pred_bootstrap 3 \
  --max_train_rows 300 \
  --max_predict_rows 100 \
  --cpus 4
```

Do not treat the debug settings as final validation settings.

## 3. Full three-source Campylobacter run

```bash
python scripts/sourcerunner_full_validation.py \
  --train_file CAMPY_nonhuman.cgmlst.sourcerunner.train.tsv \
  --predict_file CAMPY_predict.cgmlst.sourcerunner.predict.tsv \
  --output_dir sourcerunner_full \
  --run_name campylobacter_cgmlst_three_source_full \
  --source_col source \
  --keep_sources Poultry,Ruminant,Pig \
  --loci_prefix CAMP \
  --models random_forest,xgboost,logreg,lightgbm,catboost \
  --cv_folds 5 \
  --bootstrap 100 \
  --burn_in 25 \
  --pred_bootstrap 50 \
  --min_confidence 0.60 \
  --cpus 8
```

## 4. Alternative source set

For a poultry/ruminant/wild-bird analysis, change only the candidate source set (assuming these exact labels occur in the training data):

```bash
--keep_sources Poultry,Ruminant,"Wild bird"
```

## 5. Post-process a completed run

```bash
python scripts/sourcerunner_prediction_postprocess.py \
  --predictions path/to/human_predictions_bootstrap_ensemble.tsv \
  --metadata path/to/original_metadata.csv \
  --model_comparison path/to/model_comparison_cv_summary.tsv \
  --bootstrap_metrics path/to/bootstrap_oob_metrics_all_replicates.tsv \
  --outdir path/to/enriched_outputs \
  --id_col id
```

`human_predictions_bootstrap_ensemble.tsv` is a historical filename from Campylobacter development; the prediction set does not have to contain human isolates.

## 6. Manuscript-scale Peru analysis

Do **not** reproduce the Peru geographic-context manuscript by simply running the generic command above. That study used a frozen lineage-blocked design, explicit geographic leakage removal, fixed folds, size-matched controls and additional sensitivity analyses.

Use the dedicated documentation in [`peru_geographic_context/`](peru_geographic_context/) instead.

## 7. Salmonella pilot

The Salmonella directory is a feasibility scaffold, not a completed analysis. Follow [`salmonella_pilot/README.md`](salmonella_pilot/README.md) and issue #5 before attempting a full Salmonella run, because the current full-validation wrapper still assumes a common locus prefix.
