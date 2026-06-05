# SourceRunnerML v1.0.0

This update collects the current working SourceRunnerML scripts used for cgMLST-based Campylobacter source attribution.

## Core scripts

- `SourceRunnerML.py` / `SourceRunnerML_v1_0.py` — patched SourceRunnerML runner with fixed self-test label handling, `--skip_selftest`, and optional STRUCTURE-style prediction plots.
- `utils_v1_0.py` — utility functions for locus parsing, source-label processing, classifiers, bootstrap training, metrics, and legacy plotting helpers.
- `sourcerunner_full_validation.py` — full validation wrapper used for the C. coli and C. jejuni cgMLST runs. It compares models, performs cross-validation, fits bootstrap replicate models, predicts with post-burn-in models, and exports uncertainty summaries.
- `sourcerunner_prediction_postprocess.py` — post-processing script that enriches prediction tables with metadata and generates summaries/plots by country, year, ST, clonal complex, cgST and LINcode.
- `source_runner_preflight.py` — input checker/formatter for SourceRunner-ready TSVs.

## Recommended full-validation workflow

Run the full validation wrapper on SourceRunner-ready TSVs containing `CAMP` cgMLST loci and a source column such as `reduced`.

```bash
python sourcerunner_full_validation.py \
  --train_file TRAIN.cgmlst.sourcerunner.train.tsv \
  --predict_file HUMAN.cgmlst.sourcerunner.predict.tsv \
  --output_dir sourcerunner_outputs \
  --run_name campylobacter_cgmlst_full \
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

For C. jejuni poultry/ruminant/wild-bird attribution:

```bash
--keep_sources Poultry,Ruminant,"Wild bird"
```

## Debug/smoke-test workflow

Use a small subset first to confirm the environment works:

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

## Post-processing

After a full validation run:

```bash
python sourcerunner_prediction_postprocess.py \
  --predictions path/to/human_predictions_bootstrap_ensemble.tsv \
  --metadata path/to/original_metadata.csv \
  --model_comparison path/to/model_comparison_cv_summary.tsv \
  --bootstrap_metrics path/to/bootstrap_oob_metrics_all_replicates.tsv \
  --outdir path/to/enriched_outputs \
  --id_col id
```

This writes compact enriched prediction tables, metadata summaries, and plots.

## Key outputs from `sourcerunner_full_validation.py`

- `model_comparison_cv_summary.tsv`
- `cv_classification_report__<model>.json`
- `cv_confusion_matrix__<model>.tsv`
- `bootstrap_oob_metrics_all_replicates.tsv`
- `bootstrap_oob_metrics_post_burnin_summary.tsv`
- `human_predictions_bootstrap_ensemble.tsv`
- `prediction_probability_means.tsv`
- `prediction_probability_sds.tsv`
- `source_attribution_summary_filtered.tsv`
- `source_attribution_summary_raw.tsv`
- `source_by_clonal_complex_filtered.tsv`
- `prediction_uncertainty_summary.txt/json`
- `final_model_fit_all_training.pkl`

## Notes

- The full-validation wrapper estimates prediction uncertainty using independently trained bootstrap replicate models. This is preferred over repeating `predict_proba` on a single fitted model.
- Use `--selection_metric balanced_accuracy` by default for imbalanced source-attribution training data.
- For older cluster environments, this code remains Python 3.6 compatible.
