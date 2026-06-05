# Example commands

## Debug run

```bash
python sourcerunner_full_validation.py \
  --train_file TRAIN.cgmlst.sourcerunner.train.tsv \
  --predict_file HUMAN.cgmlst.sourcerunner.predict.tsv \
  --output_dir sourcerunner_debug \
  --run_name debug \
  --source_col reduced \
  --keep_sources Poultry,Ruminant,Pig \
  --models random_forest,logreg \
  --cv_folds 3 \
  --bootstrap 5 \
  --burn_in 1 \
  --pred_bootstrap 3 \
  --max_train_rows 300 \
  --max_predict_rows 100 \
  --cpus 8
```

## Full C. coli-style run

```bash
python sourcerunner_full_validation.py \
  --train_file Coli_nonhuman.cgmlst.sourcerunner.train.tsv \
  --predict_file Coli_human.cgmlst.sourcerunner.predict.tsv \
  --output_dir sourcerunner_coli_full \
  --run_name coli_cgmlst_three_source_full \
  --source_col reduced \
  --keep_sources Poultry,Ruminant,Pig \
  --models random_forest,xgboost,logreg,lightgbm,catboost \
  --cv_folds 5 \
  --bootstrap 100 \
  --burn_in 25 \
  --pred_bootstrap 50 \
  --min_confidence 0.60 \
  --cpus 8
```

## Full C. jejuni-style run

```bash
python sourcerunner_full_validation.py \
  --train_file Jejuni_nonhuman.cgmlst.sourcerunner.train.tsv \
  --predict_file Jejuni_human.cgmlst.sourcerunner.predict.tsv \
  --output_dir sourcerunner_jejuni_full \
  --run_name jejuni_cgmlst_poultry_ruminant_wildbird_full \
  --source_col reduced \
  --keep_sources Poultry,Ruminant,"Wild bird" \
  --models random_forest,xgboost,logreg,lightgbm,catboost \
  --cv_folds 5 \
  --bootstrap 100 \
  --burn_in 25 \
  --pred_bootstrap 50 \
  --min_confidence 0.60 \
  --cpus 8
```

## Post-process completed full run

```bash
python sourcerunner_prediction_postprocess.py \
  --predictions path/to/human_predictions_bootstrap_ensemble.tsv \
  --metadata path/to/original_metadata.csv \
  --model_comparison path/to/model_comparison_cv_summary.tsv \
  --bootstrap_metrics path/to/bootstrap_oob_metrics_all_replicates.tsv \
  --outdir path/to/enriched_outputs \
  --id_col id
```
