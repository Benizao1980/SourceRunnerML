# SourceRunnerML workflow guide

This guide explains the recommended SourceRunnerML v1.0.0 workflow and how its outputs should be interpreted.

For a practical first run, start with [`getting_started.md`](getting_started.md). For definitions, see [`glossary.md`](glossary.md).

## 1. What SourceRunnerML is trying to estimate

SourceRunnerML uses source-labelled genomic profiles to learn patterns associated with represented source populations. It then estimates how strongly new isolates match those source populations.

The output is **source association under a specified reference set and model**, not proof of a particular transmission event.

A source-attribution analysis therefore depends on three things at once:

1. the genomes/features supplied to the model;
2. the source labels and candidate source definitions;
3. the validation design used to test generalisation.

Reference-set composition is part of the scientific model, not a neutral background detail.

## 2. Stable v1.0 input representation

The current stable workflow operates on cgMLST allele matrices.

Each row is an isolate and each selected cgMLST locus is a model feature. The training/reference table additionally contains a source-label column. The prediction table contains the isolates for which source probabilities are requested.

For the current Campylobacter full-validation wrapper, loci are selected using a common prefix, normally `CAMP`.

This is convenient for PubMLST Campylobacter exports but is not sufficiently general for every organism. Generalising locus selection beyond a common prefix is an active development item (issue #5).

## 3. Candidate source set

SourceRunnerML v1.0 uses **multiclass source attribution**.

For one isolate and a three-source model, a prediction might look like:

```text
Poultry 0.82 | Ruminant 0.14 | Pig 0.04
```

Those probabilities sum to 1 across the represented candidate sources.

This means:

- the model compares sources simultaneously;
- probabilities are conditional on the candidate source set;
- an unrepresented source cannot be inferred directly;
- a low-confidence result should not automatically be interpreted as evidence for an unknown source.

Earlier exploratory SourceRunnerML code included one-vs-rest/source-affinity approaches. Those are not the recommended v1.0 interpretation.

## 4. Reference-set audit before modelling

Before training, document and check:

- source-label harmonisation;
- sample number per source;
- lineage distribution by source;
- geography and year by source where available;
- duplicate/overlapping isolates;
- missingness by locus and isolate;
- whether the proposed source classes are sufficiently represented and biologically interpretable.

Poorly represented or poorly discriminated categories should be merged, excluded or reported at a coarser level rather than retained simply because a label exists.

For geographic comparisons, explicitly audit whether isolates from the intended test geography occur in the training/reference panel. The Peru worked study demonstrates this with a formal leakage audit.

## 5. Preprocessing

The recommended full-validation wrapper:

1. restricts the training data to `--keep_sources`;
2. finds loci shared between training and prediction tables using `--loci_prefix`;
3. removes loci whose training-set missingness exceeds `--missingness`;
4. fits a most-frequent-value imputer on the training data;
5. applies the same retained loci and imputer to the prediction data.

Preprocessing must be defined from training/reference data rather than using information from held-out prediction targets.

Run `scripts/source_runner_preflight.py` before fitting the model to inspect IDs, source counts, shared loci and missingness.

## 6. Standard model comparison

The full-validation wrapper can compare:

- random forest;
- logistic regression;
- XGBoost;
- LightGBM;
- CatBoost.

Optional classifiers require their corresponding packages. A missing optional package can cause that model to be skipped/failed while the remaining models continue, so always inspect the model-comparison output rather than assuming every requested model ran.

Balanced accuracy is the default model-selection metric because it gives equal weight to each source class. Macro-F1, weighted-F1 and ordinary accuracy are also reported/supported.

## 7. Validation: what the general wrapper does and does not do

### Standard v1.0 wrapper

`scripts/sourcerunner_full_validation.py` uses **stratified cross-validation** for model comparison. This preserves approximate source-class proportions across folds.

It also performs independent bootstrap fitting with out-of-bag (OOB) evaluation.

These are useful baseline validation procedures, but stratification alone does **not** prevent closely related lineages from appearing in both training and test folds.

### Publication-grade study designs

Depending on the biological question, stronger validation may be needed, for example:

- lineage-grouped cross-validation;
- geographic holdout validation;
- temporal holdouts;
- source-balanced or size-matched sensitivity analyses;
- probability-calibration assessment;
- nearest-source genomic sensitivity analyses.

These are not all automatic defaults of the general wrapper.

The Peru geographic-context study uses a separate frozen lineage-blocked design. Its exact folds, leakage controls and manuscript-specific settings are documented under [`../examples/peru_geographic_context/`](../examples/peru_geographic_context/).

This distinction is important: **running the general wrapper is not the same as reproducing the Peru manuscript analysis.**

## 8. Bootstrap/OOB uncertainty

After choosing the best-performing requested model, SourceRunnerML trains independent bootstrap replicate models.

For each replicate:

- the model is fitted to a stratified bootstrap sample;
- isolates not selected into the bootstrap sample form the OOB validation set;
- OOB performance is recorded;
- fitted replicate models are retained for prediction uncertainty.

Prediction probabilities are then averaged across a configured number of independently trained replicate models.

Typical uncertainty outputs include:

- mean probability for each source;
- standard deviation of probability for each source;
- replicate-model agreement/consensus;
- confidence/entropy summaries;
- a low-confidence label according to the configured threshold.

The most-likely label is a convenient summary. The full probability vector should be retained whenever downstream interpretation depends on uncertainty.

## 9. How to interpret performance

Do not rely on one overall metric.

At minimum inspect:

- balanced accuracy;
- per-source recall;
- per-source precision;
- macro-F1;
- the confusion matrix;
- OOB variability across bootstrap models.

A model may have high overall accuracy while systematically failing on a minority source.

When the true source of human infection is unknown, validation should rely primarily on held-out **known-source** isolates. Human clinical predictions are the target inference, not a gold-standard validation set.

External epidemiological exposure information can support interpretation but should be distinguished from genomic model validation.

## 10. Lineage and geographic leakage

Two common ways to overestimate performance are:

### Lineage leakage

Closely related genomes occur in both training and validation folds. The model may recognise lineage identity rather than a source signal that generalises to unseen lineages.

### Geographic leakage

Genomes from the intended test geography are present in the supposedly external/global training panel. This can make transfer to that geography appear better than it is.

The appropriate response is not simply to remove all local data. Instead, define the scientific question and construct validation accordingly.

For example, the Peru study asks both:

- how a non-Peru global panel performs on Peru known-source isolates; and
- whether adding Peru-local reference genomes improves performance.

The clean comparison therefore requires removing Peru leakage from the “non-Peru global” panel while retaining local genomes in the explicitly local/combined conditions.

## 11. Prediction uncertainty in downstream analyses

When source attribution is used upstream of epidemiologic models, avoid treating the maximum-probability source label as error-free.

Possible approaches include:

- retaining source probabilities directly;
- repeatedly sampling source assignments from each isolate's probability vector and refitting the downstream model;
- comparing hard-label and probability-aware sensitivity analyses.

The correct approach depends on the downstream statistical model and study design.

## 12. Practical interpretation checklist

Before making a biological claim, ask:

- Were the source categories defined before looking at the predictions?
- Is every source adequately represented?
- Did training and validation share close lineages?
- Did training and validation share the same geography?
- What is recall for each source?
- Which sources are confused with one another?
- How variable are predictions across bootstrap models?
- Are low-confidence predictions being retained or hidden?
- Could an important source be absent from the candidate set?
- Does an independent genomic or epidemiological sensitivity analysis support the same conclusion?

## 13. Worked examples

- [`../examples/peru_geographic_context/`](../examples/peru_geographic_context/) — manuscript-scale geographic-context analysis with leakage auditing, lineage-blocked validation, size controls, human attribution, nearest-source validation and phylogeny.
- [`../examples/salmonella_pilot/`](../examples/salmonella_pilot/) — planned feasibility scaffold for adapting SourceRunnerML to Salmonella. No Salmonella performance results are currently claimed.
