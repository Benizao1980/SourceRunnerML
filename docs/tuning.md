# Optional hyperparameter tuning

SourceRunnerML uses its established classifier defaults unless `--tuning_config` is supplied. Tuning is an **optional development step**, not a substitute for grouped/external validation.

## Config and command

Create a JSON object keyed by the requested model names, each containing lists of values for actual scikit-learn-style estimator hyperparameters. For example [`examples/tuning.example.json`](../examples/tuning.example.json) provides random forest (`n_estimators`, `max_depth`) and logistic regression (`C`) grids.

```bash
python scripts/sourcerunner_full_validation.py \
  --train_file TRAIN.tsv --predict_file QUERY.tsv \
  --source_col source --keep_sources Poultry,Ruminant,Pig \
  --loci_prefix CAMP --models random_forest,logreg \
  --tuning_config examples/tuning.example.json --inner_cv_folds 3 \
  --selection_metric balanced_accuracy --cv_folds 5 \
  --bootstrap 25 --burn_in 5 --pred_bootstrap 20 \
  --output_dir tuned_outputs --run_name example --cpus 4
```

### Validation rules

- The JSON keys must refer to requested models, and parameter names must exist on their estimators.
- Candidate values must be lists of JSON scalars; the grid is limited to 64 configurations **per model**.
- CPU/thread counts, verbosity and random seeds cannot be tuned. Inner grid searches use `n_jobs=1` while each candidate estimator uses the specified `--cpus`.
- Within each *outer CV fold*, GridSearchCV fits and selects hyperparameters using only the outer training rows. The selected estimator then predicts that fold's held-out observations.
- Within each *bootstrap replicate*, tuning uses only sampled in-bag rows before OOB scoring. This prevents bootstrap-held-out data from selecting the hyperparameters used to score them.
- The final prediction model is tuned on the **full source-labelled training set** and applied only to the separate query table. These are not the held-out CV/OOB scores.
- For inner CV, the requested `--inner_cv_folds` is reduced if the in-partition minimum class count is smaller; at least 2 examples/class are required.

### Outputs and interpretation

Check `run_setup.json` for your requested grids, `cv_fold_retained_loci__MODEL.tsv` for per-fold tuned parameters and retained loci, `bootstrap_oob_metrics_all_replicates.tsv` for per-replicate tuning, and `final_model_tuning.json` for selected full-training parameters.

Because **model selection among candidate model families** is still based on the same outer CV used to report the best family, the reported winning CV score can exhibit model-selection optimism. Use a genuinely external or nested-across-model-families test for a definitive generalisation estimate. Similarly, ordinary StratifiedKFold does **not** protect against geographic or clonal-lineage overlap. The Peru manuscript benchmark is a separate frozen design and is not rerun by this option.

Important: numerical allele IDs are nominal identifiers, not meaningful distances. Hyperparameter tuning does not resolve the need to assess feature encoding and population structure.
