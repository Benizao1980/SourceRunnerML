# SourceRunnerML v1.0.0

SourceRunnerML is a research software framework for **cgMLST-based microbial source attribution**. It trains supervised machine-learning models on source-labelled isolates and estimates the likely source composition of new isolates while retaining uncertainty in the predictions.

The current stable workflow has been developed and used for *Campylobacter jejuni* and *Campylobacter coli*. The framework is designed to be extensible to other bacterial pathogens. A worked *Salmonella* feasibility example is now being developed; see [`examples/salmonella_pilot/`](examples/salmonella_pilot/) and [`docs/salmonella_pilot_plan.md`](docs/salmonella_pilot_plan.md).

> **Status:** active research software. Version 1.0.0 provides a stable working Campylobacter workflow; organism-specific validation is required before applying the framework to a new pathogen or source scheme.

## What SourceRunnerML does

The v1.0.0 full-validation workflow:

- accepts source-labelled cgMLST allele matrices for model training;
- compares multiple classifiers, including random forest, logistic regression, XGBoost, LightGBM and CatBoost when installed;
- uses cross-validation and balanced-accuracy-aware model selection;
- performs independently trained bootstrap replicates with out-of-bag validation;
- produces a **single multiclass source-probability vector per isolate** for the selected source set;
- summarizes prediction uncertainty across bootstrap models;
- exports confusion matrices, classification reports, source-attribution summaries and metadata-enriched outputs.

Source attribution is probabilistic. Predictions should be interpreted as evidence of genomic similarity to represented source populations, not proof of an individual transmission event.

## Manuscript-scale worked study: Peru geographic context

A detailed worked study is available at [`examples/peru_geographic_context/`](examples/peru_geographic_context/).

This documents the frozen analysis used to test how **geographic composition of source reference genomes changes known-source attribution performance in Peruvian *Campylobacter jejuni***. It includes:

- leakage auditing and removal of Peruvian isolates from the global training panel;
- a fixed lineage-blocked fivefold benchmark;
- Global vs Peru-local vs Global+Peru reference comparisons;
- the frozen 50-model XGBoost production profile;
- size-matched geographic controls and equal-n learning curves;
- human source attribution for 981 Peruvian *C. jejuni* isolates;
- nearest-source cgMLST validation;
- lineage-dependent rescue analyses;
- Prokka -> PIRATE -> alignment-QC -> IQ-TREE phylogenetic reconstruction;
- a detailed plan for the ongoing *C. coli* extension.

The worked study is deliberately separated from the general SourceRunnerML API. SourceRunnerML remains reusable software; the Peru folder records a manuscript-specific scientific design, frozen model settings and provenance needed to reproduce that analysis.

## Repository layout

- `scripts/SourceRunnerML.py` / `scripts/SourceRunnerML_v1_0.py` - core SourceRunnerML runner.
- `scripts/sourcerunner_full_validation.py` - recommended v1.0.0 validation and prediction wrapper.
- `scripts/sourcerunner_prediction_postprocess.py` - metadata enrichment and summary/plot generation.
- `scripts/source_runner_preflight.py` - input checking and formatting helpers.
- `scripts/utils_v1_0.py` - shared model, preprocessing, metrics and plotting functions.
- `docs/workflow.md` - conceptual workflow and interpretation guide.
- `examples/` - example commands and worked-study documentation, including the Peru geographic-context analysis.
- `tests/` - lightweight import/smoke tests.

## Input requirements

The full-validation wrapper expects tab-separated files containing:

1. an isolate/sample identifier;
2. a **source label column** in the training file (specified with `--source_col`);
3. cgMLST allele columns shared by the training and prediction datasets.

For the current Campylobacter workflow, cgMLST loci are identified using a common prefix (normally `CAMP`). Missing alleles are supported and loci with excessive missingness can be filtered before imputation.

The planned Salmonella extension will remove the assumption that all cgMLST columns share a simple locus prefix and will document an explicit locus-selection workflow suitable for EnteroBase Salmonella profiles.

Full genomic datasets and analysis outputs are intentionally not stored in this repository. Worked examples should use small public or redistributable subsets, or scripts that reconstruct them from public resources.

## Recommended full-validation workflow

```bash
python scripts/sourcerunner_full_validation.py \
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

For a *C. jejuni* poultry/ruminant/wild-bird analysis:

```bash
--keep_sources Poultry,Ruminant,"Wild bird"
```

For a quick smoke-test run, reduce the number of folds/bootstrap replicates and use `--max_train_rows` / `--max_predict_rows`; examples are provided in [`examples/example_commands.md`](examples/example_commands.md).

## Key outputs

`sourcerunner_full_validation.py` writes, among other files:

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
- `prediction_uncertainty_summary.txt/json`
- `final_model_fit_all_training.pkl`

Prediction uncertainty is estimated using **independently trained bootstrap replicate models**, rather than repeatedly calling `predict_proba` on one fitted model. Balanced accuracy is the default model-selection metric for imbalanced source-attribution datasets.

## Post-processing

```bash
python scripts/sourcerunner_prediction_postprocess.py \
  --predictions path/to/human_predictions_bootstrap_ensemble.tsv \
  --metadata path/to/original_metadata.csv \
  --model_comparison path/to/model_comparison_cv_summary.tsv \
  --bootstrap_metrics path/to/bootstrap_oob_metrics_all_replicates.tsv \
  --outdir path/to/enriched_outputs \
  --id_col id
```

This produces compact prediction tables and summaries by available metadata such as country, year, sequence type, clonal complex, cgST and LIN code.

## Validation and responsible use

For new organisms or source schemes, users should explicitly evaluate:

- representation and imbalance of candidate source populations;
- lineage leakage between training and validation sets;
- geographic and temporal generalizability;
- per-source classification performance and confusion;
- probability calibration and low-confidence predictions;
- whether the available reference collection can genuinely discriminate the proposed source categories.

The planned Salmonella worked example will demonstrate lineage-grouped validation, geographic holdouts where feasible, probability calibration and propagation of source-attribution uncertainty into downstream epidemiologic analyses.

## Software notes

- Python 3.6 compatibility is retained for older HPC environments in the v1.0.0 workflow.
- Some optional classifiers require their corresponding Python packages.
- Several experimental options in the core runner are retained for development but are not part of the recommended v1.0.0 workflow; use the documented full-validation wrapper for reproducible analyses.

## License

MIT License. See [`LICENSE`](LICENSE).
