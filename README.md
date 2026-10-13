# SourceRunnerML v1.0.0

SourceRunnerML is a research software framework for **cgMLST-based microbial source attribution**. It learns genomic patterns from source-labelled isolates and estimates how strongly new isolates resemble the represented source populations.

> **Important:** source attribution is probabilistic. A prediction means “most similar to the represented reference population(s) under this model”; it does **not** prove an individual transmission event.

The stable v1.0 workflow has been developed primarily with *Campylobacter jejuni* and *Campylobacter coli*. Adaptation to other organisms requires organism-specific input handling and validation.

## Start here

| I want to… | Read / run |
|---|---|
| **try SourceRunnerML without obtaining any genomes** | **[`docs/tutorial.md`](docs/tutorial.md) — runnable synthetic-data lesson** |
| install SourceRunnerML and run a first analysis | [`docs/getting_started.md`](docs/getting_started.md) |
| understand the modelling workflow and validation | [`docs/workflow.md`](docs/workflow.md) |
| look up terminology | [`docs/glossary.md`](docs/glossary.md) |
| see copy/paste command templates | [`examples/example_commands.md`](examples/example_commands.md) |
| understand the completed Peru geographic-context study | [`examples/peru_geographic_context/START_HERE.md`](examples/peru_geographic_context/START_HERE.md) |
| follow the planned Salmonella feasibility work | [`examples/salmonella_pilot/`](examples/salmonella_pilot/) |
| see the Salmonella / Erika Austhof K01 study strand | [`examples/salmonella_erika_k01/`](examples/salmonella_erika_k01/) |

## First run: synthetic example

If you are teaching or evaluating the software, start **without private data**. This is a simulated input/output demonstration, **not a biological benchmark**.

```bash
git clone https://github.com/Benizao1980/SourceRunnerML.git
cd SourceRunnerML
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-minimal.txt
python examples/toy_quickstart/make_toy_data.py --outdir toy_data
python scripts/source_runner_preflight.py --train toy_data/train.tsv --predict toy_data/predict.tsv --source_col source --loci_pattern '^CAMP' --min_source_size 20 --outdir toy_preflight
```

Next, use the **[guided tutorial](docs/tutorial.md)** for a short model-fitting command, source-attribution vocabulary, interpretation exercises, and validation warnings.

## Recommended workflow

For most users, the supported path is:

1. **Prepare two tab-separated files**: a source-labelled training/reference set and a prediction set.
2. **Run preflight QC** with `scripts/source_runner_preflight.py`.
3. **Run model comparison + validation + prediction** with `scripts/sourcerunner_full_validation.py`.
4. **Inspect class-specific performance and uncertainty**, not only overall accuracy.
5. **Optionally enrich/plot predictions** with `scripts/sourcerunner_prediction_postprocess.py`.
6. For publication analyses, add validation appropriate to the biology (for example lineage-blocked or geographic holdouts).

**Validation:** the general wrapper now fits locus filtering and imputation independently within each CV fold and bootstrap/OOB training partition (see [validation tests](tests/test_validation_features.py)). It still uses ordinary stratified CV, which can be optimistic when closely related lineages cross folds; grouped, geographic and temporal validation remain study-specific (see the [tutorial](docs/tutorial.md#8-critical-scientific-cautions)). The Peru manuscript study uses a separate, frozen **lineage-blocked** design documented in its worked-study directory; do not assume the generic wrapper reproduces that study automatically.

## Install

```bash
git clone https://github.com/Benizao1980/SourceRunnerML.git
cd SourceRunnerML
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test_imports.py
```

`requirements.txt` installs the optional tree/boosting classifiers used by the full model-comparison workflow. For a lighter random-forest/logistic-regression environment, use `requirements-minimal.txt` instead.

## Minimal input structure

Training/reference TSV:

```text
id    source      CAMP0001    CAMP0002    CAMP0003    ...
A01   Poultry     12          7           103         ...
A02   Ruminant    8           4           55          ...
```

Prediction TSV:

```text
id    CAMP0001    CAMP0002    CAMP0003    ...
H01   12          7           104         ...
H02   8           4           55          ...
```

The training file needs a source-label column (`--source_col`). Select cgMLST loci with a verified `--loci_file` (strict one-name-per-line list), `--loci_pattern` (regex), or `--loci_prefix` (default `CAMP` for backwards compatibility). Explicit file mode takes precedence over regex, which takes precedence over prefix. Preflight and full validation share these rules; see [getting started](docs/getting_started.md).

## Quick debug run

```bash
python scripts/sourcerunner_full_validation.py \
  --train_file TRAIN.cgmlst.sourcerunner.train.tsv \
  --predict_file PREDICT.cgmlst.sourcerunner.predict.tsv \
  --output_dir sourcerunner_debug \
  --run_name debug \
  --source_col source \
  --keep_sources Poultry,Ruminant,Pig \
  --models random_forest,logreg \
  --cv_folds 3 \
  --bootstrap 5 \
  --burn_in 1 \
  --pred_bootstrap 3 \
  --max_train_rows 300 \
  --max_predict_rows 100 \
  --cpus 4
```

Use the small debug settings only to check that the pipeline and input format work. They are not publication-grade validation settings.

## What the full-validation wrapper does

The recommended wrapper can:

- retain a defined candidate source set;
- filter loci with excessive missingness and impute remaining missing alleles;
- compare random forest, logistic regression, XGBoost, LightGBM and CatBoost when installed;
- select models using balanced accuracy (default) or another supported metric;
- write out-of-fold cross-validation predictions, confusion matrices and class-specific reports;
- retrain independent bootstrap models and evaluate out-of-bag performance;
- average prediction probabilities across independently trained replicate models;
- report uncertainty and low-confidence predictions rather than silently forcing certainty.

## Key outputs

Common outputs include:

- `run_setup.txt/json` — exact run settings and retained-locus counts;
- `model_comparison_cv_summary.tsv` — model comparison;
- `cv_confusion_matrix__<model>.tsv` — source-specific errors;
- `cv_classification_report__<model>.json` — precision/recall/F1 by source;
- `bootstrap_oob_metrics_all_replicates.tsv` — bootstrap/OOB validation;
- `bootstrap_oob_metrics_post_burnin_summary.tsv` — summary after the configured burn-in;
- `human_predictions_bootstrap_ensemble.tsv` (historical filename) — ensemble predictions for the supplied prediction set;
- `prediction_probability_means.tsv` and `prediction_probability_sds.tsv` — probability and uncertainty summaries;
- `final_model_fit_all_training.pkl` — final fitted model object.

The prediction set does not have to be human; some historical filenames retain the Campylobacter development terminology.

## Repository structure

```text
SourceRunnerML/
├── scripts/                    # executable workflow code
├── docs/                       # getting-started, concepts and terminology
├── examples/                   # command templates and worked studies
│   ├── peru_geographic_context/
│   ├── salmonella_pilot/
│   ├── salmonella_erika_k01/
│   └── toy_quickstart/
├── tests/                      # lightweight smoke tests
├── requirements.txt
└── README.md
```

For first-time use, start with `sourcerunner_full_validation.py`. `SourceRunnerML.py`, `SourceRunnerML_v1_0.py` and `utils_v1_0.py` are lower-level research/development code and are retained for compatibility and method development.

## Worked studies vs reusable software

The `examples/` directory deliberately separates **general software behaviour** from **study-specific scientific designs**.

- **Peru geographic context**: a frozen manuscript-scale analysis with leakage auditing, fixed lineage-blocked folds, size-matched controls, human attribution, nearest-source validation and phylogenetic reconstruction.
- **Salmonella pilot**: a planning/feasibility scaffold. It does not yet claim Salmonella performance results.
- **Erika / K01 Salmonella**: a separate collaborative study design, including EnteroBase, hierCC, geography and independent epidemiological comparisons; no Salmonella results claimed.

Full private genomic datasets and large derived outputs are not stored in this repository. Worked studies should contain redistributable data, aggregate results, or instructions for rebuilding inputs from their original sources.

## Optional hyperparameter tuning

An opt-in `--tuning_config examples/tuning.example.json` enables classifier-specific parameter grids using *inner cross-validation within each training fold or bootstrap in-bag sample*. Defaults are unchanged when omitted. Read [the tuning guide](docs/tuning.md) before using on a scientific dataset; it does not remove lineage/geographic validation requirements.

## Validation principles

Before interpreting source predictions, check:

- sample numbers and imbalance by source;
- whether closely related lineages occur in both training and validation folds;
- geographic and temporal representativeness;
- source-specific recall/precision and confusion, not just overall accuracy;
- calibration/uncertainty where relevant;
- whether the proposed source categories are genuinely distinguishable with the available reference collection.

See [`docs/workflow.md`](docs/workflow.md) for interpretation guidance and [`docs/glossary.md`](docs/glossary.md) for definitions.

## Status and development priorities

SourceRunnerML v1.0.0 is active research software. Current priorities are tracked as focused GitHub issues, particularly:

- monitoring performance of organism-specific locus selection (issue #5, implemented in the generic wrapper);
- completing the Peru *C. coli* extension and deciding its validation design (issue #11);
- optional broader hyperparameter tuning (issue #4).

## License

MIT License. See [`LICENSE`](LICENSE).
