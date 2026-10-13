# SourceRunnerML guided tutorial: from allele tables to source attribution

**Audience:** microbial genomicists new to supervised ML, including students and public-health collaborators. Allow a short teaching session to run the toy example, then budget separately for real-data curation and scientific validation.

**Start here if you want to actually run something.** The tutorial uses simulated allele profiles, not Peru, FoodNet or EnteroBase data. No data access is needed.

## 1. The scientific problem

Given source-labelled non-human isolates, can we learn genomic features associated with sampled reservoirs and then estimate which of those reference populations an unlabelled (often human) isolate most resembles?

This is a **closed-set, supervised multiclass classifier**. Its answers depend on the reference collection. A high Poultry probability means strong model association with *represented poultry reference isolates*, not proof of poultry exposure, infection route or an individual transmission event.

## 2. Glossary in context

| Term | Meaning in this tutorial |
|---|---|
| Isolate | A sequenced microbial sample represented by one row |
| cgMLST | Core-genome multilocus sequence typing; allele calls at many conserved loci |
| Allele | An identifier for a sequence variant at a locus; **its numeric value is not a genetic distance** |
| Feature | A locus column, such as CAMP0001 |
| Label / class | Known source category for training, e.g. Poultry |
| Reference / training set | Source-labelled isolates used to fit the classifier |
| Prediction set | Isolates lacking model source labels for which probabilities are requested |
| Fold / cross-validation (CV) | A partition held out from model fitting to measure generalisation |
| Balanced accuracy | Mean sensitivity (recall) across source classes |
| OOF / OOB | Out-of-fold predictions / out-of-bag observations for bootstrap models |
| Data leakage | Test information unintentionally influencing training or preprocessing |
| Calibration | Whether predictions made with stated probability match observed frequencies |
| Domain shift | Change in lineage, geography, time or sampling between references and targets |

For fuller explanations see [glossary.md](glossary.md) and [workflow.md](workflow.md).

## 3. Install the project

From a shell with Python 3 and Git:

~~~bash
git clone https://github.com/Benizao1980/SourceRunnerML.git
cd SourceRunnerML
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-minimal.txt
python scripts/sourcerunner_full_validation.py --help
~~~

For all optional model families and the legacy import smoke-test, install the full requirements instead. The older `tests/smoke_test_imports.py` loads legacy utilities which unconditionally import XGBoost, LightGBM and CatBoost, so it is **not** an appropriate check for a minimal install. A first lesson needs only random forest and logistic regression.

## 4. Generate reproducible toy data

From the **repository root**:

~~~bash
python examples/toy_quickstart/make_toy_data.py --outdir toy_data
~~~

This writes two tab-delimited files:

- `toy_data/train.tsv` — 180 artificial source-labelled isolates (Poultry, Ruminant, Pig), 60 each.
- `toy_data/predict.tsv` — 30 artificial prediction isolates. The extra demo_truth column exists only for checking this toy lesson; the model does not use it.

There are 120 invented CAMP loci shared across both tables. The data include deliberate missing calls and artificial class signal. They **must not** be interpreted as evidence about biological source specificity or expected real-world accuracy.

Inspect the tabular input:

~~~bash
head -n 2 toy_data/train.tsv | cut -f 1-8
head -n 2 toy_data/predict.tsv | cut -f 1-8
~~~

In real data, source labels should be harmonised from a retained raw-source field and audited for missing/uncertain provenance.

## 5. Preflight before fitting

~~~bash
python scripts/source_runner_preflight.py \
  --train toy_data/train.tsv \
  --predict toy_data/predict.tsv \
  --source_col source \
  --id_col id \
  --min_source_size 20 \
  --loci_pattern '^CAMP' \
  --outdir toy_preflight \
  --write_tsv
~~~

Inspect `toy_preflight/source_runner_preflight_report.txt` and its JSON equivalent. Check source counts, shared loci, duplicate IDs, non-numeric calls, and missingness. The pattern flag is important: the preflight's historical default otherwise looks for seven MLST housekeeping loci.

## 6. Fit a small educational model

~~~bash
python scripts/sourcerunner_full_validation.py \
  --train_file toy_data/train.tsv \
  --predict_file toy_data/predict.tsv \
  --output_dir toy_results \
  --run_name toy_first_run \
  --source_col source \
  --keep_sources Poultry,Ruminant,Pig \
  --loci_prefix CAMP \
  --models random_forest,logreg \
  --cv_folds 3 \
  --bootstrap 5 \
  --burn_in 1 \
  --pred_bootstrap 3 \
  --min_confidence 0.60 \
  --cpus 2
~~~

The wrapper writes a timestamped folder under `toy_results/`; follow the printed path. The toy data are small, but actual runs can need substantial memory.

## 7. Read outputs like a scientist

Start with these files **in order**:

1. `run_setup.json` — were all intended sources, loci and files used?
2. `model_comparison_cv_summary.tsv` — which model succeeded and what is its balanced accuracy?
3. `cv_confusion_matrix__random_forest.tsv` — which source classes get confused?
4. `bootstrap_oob_metrics_post_burnin_summary.tsv` — how variable is the repeated in-bag/out-of-bag evaluation?
5. `human_predictions_bootstrap_ensemble.tsv` — inspect each probability vector and Uncertain calls. The filename is historical: any organisms/target categories can be predicted.

The total probability across the candidate classes should be approximately 1.00 for each prediction. Probability **disagreement between bootstrap models** is different from **calibration**; neither by itself proves a prediction is correct.

## 8. Critical scientific cautions

- **Not an external test.** A toy run is an operational check, not validation for real-world attribution.
- **Current wrapper uses ordinary stratified CV.** Closely related isolates can be split across folds. For outbreak-prone or clonal taxa, add lineage-blocked CV, time/geography holdouts and external validation.
- **Preprocessing caveat (current v1 wrapper):** it fits the imputer and missingness filter before splitting for CV and bootstrap/OOB. This can leak information from held-out rows into preprocessing and make validation optimistic (see [issue #13](https://github.com/Benizao1980/SourceRunnerML/issues/13)). Do **not** claim fully leakage-safe CV/OOB from this wrapper until fold-contained preprocessing is implemented and verified.
- **Allele numbering is nominal.** Numeric classifiers currently see allele IDs as numbers. Especially for logistic regression, numeric ordering should not be misread as biological distance; test alternative representations/sensitivity.
- **Class imbalance and reference bias matter.** Look at each source's recall, balanced accuracy, and sensitivity to panel composition.
- **Labels are conditional on candidate classes.** An unrepresented reservoir does not have its own output probability.
- **Be careful with 0.60.** The uncertainty cutoff is a reporting policy, not an automatically calibrated probability.

## 9. Move from toy data to real isolates

Prepare matching cgMLST tables using a consistent allele scheme; exclude leakage and biological duplicates; preserve stable isolate IDs, source, geography, year and lineage metadata. Run preflight. Then validate source labels and sampling coverage **before** tuning classifiers.

For a real manuscript example of why lineage structure and geography matter, use [the Peru study](../examples/peru_geographic_context/START_HERE.md). For extending the framework to Salmonella, use [the general pilot](../examples/salmonella_pilot/) or [the Erika collaboration strand](../examples/salmonella_erika_k01/).

## 10. Exercises

1. Change a few raw labels from Poultry to an unrecognised spelling. What does exact-match --keep_sources do?
2. Drop 20 CAMP columns from the query file. How many shared loci remain?
3. Reduce one source to only a handful of genomes. Which class-specific metric becomes unreliable?
4. Compare random-stratified validation with a **separate** lineage-blocked test. Why might performance fall?
5. Imagine an unrepresented reservoir. Why does a high model probability not exclude it?

**Take-home:** a model that runs is not necessarily a model that generalises. Source definition, sampling and validation are part of the biological analysis.
