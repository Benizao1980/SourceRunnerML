# Getting started with SourceRunnerML

This guide is the shortest supported route from a source-labelled cgMLST table to a validated SourceRunnerML run.

If you only want the conceptual background, see [`workflow.md`](workflow.md). For terminology, see [`glossary.md`](glossary.md).

## 1. Install

```bash
git clone https://github.com/Benizao1980/SourceRunnerML.git
cd SourceRunnerML
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test_imports.py
```

Use `requirements-minimal.txt` if you only need the base scientific Python stack and plan to run random forest / logistic regression. The full `requirements.txt` also installs XGBoost, LightGBM, CatBoost and additional plotting/explanation packages.

## 2. Prepare the two input tables

SourceRunnerML's full-validation wrapper expects tab-separated (`.tsv`) files.

### Training/reference file

Each row is a source-labelled isolate. At minimum it needs:

- a stable isolate identifier (strongly recommended);
- a source column;
- cgMLST allele columns.

Example:

```text
id	source	CAMP0001	CAMP0002	CAMP0003
A01	Poultry	12	7	103
A02	Ruminant	8	4	55
A03	Pig	31	9	18
```

### Prediction file

The prediction table needs the same cgMLST feature columns but does not need a source label.

```text
id	CAMP0001	CAMP0002	CAMP0003
H01	12	7	104
H02	8	4	55
```

Metadata columns may be retained. The model uses only the selected/shared cgMLST loci.

## 3. Run preflight QC first

The preflight utility checks delimiters, IDs, source counts, shared loci, missingness and non-numeric allele calls.

For Campylobacter cgMLST columns beginning with `CAMP`:

```bash
python scripts/source_runner_preflight.py \
  --train TRAIN.tsv \
  --predict PREDICT.tsv \
  --source_col source \
  --id_col id \
  --loci_pattern '^CAMP' \
  --outdir preflight \
  --write_tsv
```

Why specify `--loci_pattern '^CAMP'`? The preflight script's historical default locus list is the seven-locus MLST scheme. For cgMLST you should explicitly select the cgMLST columns.

Review:

```text
preflight/source_runner_preflight_report.txt
preflight/source_runner_preflight_report.json
```

Do not continue blindly after warnings. In particular, check:

- source counts;
- duplicate IDs;
- number of shared loci;
- loci/isolates with high missingness;
- unexpected non-numeric allele calls;
- whether your source labels exactly match the classes you intend to model.

## 4. Run a small debug analysis

Before a full run, test the input and environment with a deliberately small analysis:

```bash
python scripts/sourcerunner_full_validation.py \
  --train_file TRAIN.tsv \
  --predict_file PREDICT.tsv \
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

A successful debug run should produce an output directory containing `run_setup`, model-comparison, cross-validation and bootstrap files.

These settings are only for pipeline testing. Do not quote the debug-run performance as final scientific results.

## 5. Run the full validation workflow

A typical Campylobacter run is:

```bash
python scripts/sourcerunner_full_validation.py \
  --train_file TRAIN.tsv \
  --predict_file PREDICT.tsv \
  --output_dir sourcerunner_outputs \
  --run_name campylobacter_cgmlst_full \
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

The wrapper will attempt each requested classifier. If an optional package is unavailable, that model can fail while the remaining models continue; always inspect `model_comparison_cv_summary.tsv` to confirm what actually ran.

## 6. Read the outputs in this order

### A. `run_setup.txt` / `run_setup.json`

Confirm:

- the files used;
- source classes retained;
- training source counts;
- number of shared loci;
- number of loci retained after missingness filtering;
- model list and validation settings.

If this file does not describe the analysis you intended, stop here.

### B. `model_comparison_cv_summary.tsv`

This is the first model-performance summary. Balanced accuracy is the default model-selection metric because it gives each source class equal weight regardless of class size.

### C. Confusion matrices and class-specific reports

Inspect:

- `cv_confusion_matrix__<model>.tsv`
- `cv_classification_report__<model>.json`

A model can have an attractive overall score while failing badly on one source. Source-specific recall and confusion are therefore essential.

### D. Bootstrap/OOB validation

Inspect:

- `bootstrap_oob_metrics_all_replicates.tsv`
- `bootstrap_oob_metrics_post_burnin_summary.tsv`

These quantify performance across independently retrained bootstrap models using out-of-bag observations.

### E. Prediction probabilities and uncertainty

Inspect:

- `prediction_probability_means.tsv`
- `prediction_probability_sds.tsv`
- `human_predictions_bootstrap_ensemble.tsv` (historical filename; the prediction set need not be human)

Retain the full probability vector where possible. The maximum-probability source label is a convenient summary, not ground truth.

## 7. Optional post-processing

```bash
python scripts/sourcerunner_prediction_postprocess.py \
  --predictions path/to/human_predictions_bootstrap_ensemble.tsv \
  --metadata path/to/metadata.csv \
  --model_comparison path/to/model_comparison_cv_summary.tsv \
  --bootstrap_metrics path/to/bootstrap_oob_metrics_all_replicates.tsv \
  --outdir enriched_outputs \
  --id_col id
```

This can enrich prediction tables with available metadata and generate compact summaries/plots.

## 8. Before treating a result as publication-grade

The standard wrapper uses ordinary stratified cross-validation. That is useful but may be optimistic when closely related isolates occur across training and validation folds.

For publication analyses, consider whether you also need:

- lineage-grouped validation;
- geographic holdouts;
- temporal holdouts;
- size-matched or class-balanced sensitivity analyses;
- calibration assessment;
- nearest-neighbour / nearest-source sensitivity checks.

The Peru geographic-context study shows one manuscript-scale implementation of lineage-blocked and geographic validation: [`../examples/peru_geographic_context/`](../examples/peru_geographic_context/).

## 9. Common failure modes

### “No shared CAMP loci detected”

The training and prediction tables do not share columns beginning with the selected `--loci_prefix`, or your organism uses a different locus naming scheme.

For Campylobacter, check that both files contain the full `CAMP...` cgMLST columns. General locus selection is being extended beyond a common prefix; see issue #5.

### Source class disappears

`--keep_sources` uses exact labels. Check spelling, capitalisation and source-label harmonisation before fitting.

### One source has very poor recall

Do not hide it behind overall accuracy. Check sample size, lineage composition, metadata quality and whether that source is genomically separable from the others.

### Human/prediction assignments look very confident

Confidence alone is not proof of correctness. Check known-source validation, reference-set representation, lineage leakage and external sensitivity analyses.

### Full run is too slow

Use the debug settings first, then scale bootstrap replicates/models/CPUs only after the input and output structure are confirmed.

## 10. Which example should I use?

- **General Campylobacter use:** [`../examples/example_commands.md`](../examples/example_commands.md)
- **Detailed manuscript reproducibility:** [`../examples/peru_geographic_context/`](../examples/peru_geographic_context/)
- **Adapting to Salmonella:** [`../examples/salmonella_pilot/`](../examples/salmonella_pilot/)
