# Salmonella worked-example plan

## Purpose

This pilot will provide a small, transparent worked example showing how SourceRunnerML can be adapted from the current Campylobacter cgMLST workflow to source-labelled *Salmonella* data.

The pilot is a **feasibility and training example**, not the final K01 source-attribution analysis. It should be small enough to run quickly, use only public/redistributable data, and make every preprocessing and validation decision explicit.

## Questions the pilot should answer

1. Can SourceRunnerML ingest a Salmonella cgMLST matrix exported from EnteroBase without relying on Campylobacter-specific locus naming?
2. Can a small set of well-represented Salmonella source categories be discriminated above chance using the existing multiclass framework?
3. Do results remain sensible when validation is grouped by genomic lineage rather than using only random stratified folds?
4. Can the pipeline produce interpretable per-isolate source probabilities, calibration summaries and bootstrap uncertainty outputs?
5. What changes are required before the framework is ready for the full K01 reference collection?

## Initial scope

Target a deliberately modest public reference subset, initially aiming for roughly 300-500 isolates per source where metadata quality permits.

Candidate source categories:

- poultry;
- cattle/ruminant;
- wild bird;
- environmental source if there is adequate, interpretable coverage.

The final pilot categories should be chosen **after** inspecting metadata completeness and genomic representation. Poorly represented or ambiguously labelled sources should be excluded or merged rather than forced into the example.

## Data and metadata checks

Before model fitting:

- retain only isolates with usable cgMLST profiles and interpretable source metadata;
- record source, serotype, country/region, year, cgMLST/cgST and available EnteroBase hierarchical cluster identifiers;
- quantify missingness by locus and isolate;
- inspect imbalance by source, serotype, geography and lineage;
- document every source-label harmonization rule;
- keep a reproducible list/query of the public records used rather than committing large genomic datasets to GitHub.

## Required software change

The current Campylobacter wrapper detects loci using a common prefix such as `CAMP`. Salmonella locus naming may not follow one simple prefix. The pilot should therefore add an explicit mechanism for selecting the cgMLST feature columns, for example:

- a start/end column range;
- a file containing locus names; or
- schema-aware column selection from an EnteroBase export.

The chosen implementation should be documented and covered by a small test.

## Baseline analysis

### 1. Preflight

- validate input dimensions and identifiers;
- report source counts;
- report loci retained after missingness filtering;
- fail clearly if expected source or locus columns are absent.

### 2. Baseline models

Compare at minimum:

- random forest;
- logistic regression;
- XGBoost when available.

Use balanced accuracy as the primary baseline model-selection metric, alongside macro-F1, source-specific precision/recall and confusion matrices.

### 3. Validation

Run two levels of validation:

- ordinary stratified cross-validation as a baseline comparison;
- **lineage-grouped cross-validation** using an EnteroBase genomic cluster level selected before analysis to reduce leakage between closely related isolates.

Where sample size and metadata permit, add a geographic holdout as a small external stress test.

### 4. Probability and uncertainty

For each held-out/prediction isolate, export:

- the multiclass probability vector across candidate sources;
- most-likely source;
- maximum probability;
- bootstrap mean and SD by source;
- replicate-model agreement/consensus;
- an uncertainty summary.

Add a probability-calibration assessment for the retained source classes.

## Planned worked-example outputs

The final example should contain small text/figure outputs sufficient to understand the workflow without distributing a large genomic dataset:

- `pilot_metadata_summary.tsv`
- `model_comparison_cv_summary.tsv`
- `lineage_grouped_cv_summary.tsv`
- one confusion matrix per retained model
- `prediction_probability_means.tsv`
- `prediction_probability_sds.tsv`
- calibration summary/plot
- `pilot_results_summary.md`

## Success criteria

The pilot will be considered useful if it:

1. runs end-to-end from a documented Salmonella cgMLST input format;
2. demonstrates that at least some biologically meaningful source categories can be separated with interpretable held-out performance;
3. shows the difference between random and lineage-grouped validation;
4. produces reproducible probability and uncertainty outputs;
5. clearly identifies source categories or sampling structures that are **not** adequately supported.

A negative result for a particular source category is still informative and should lead to merging/exclusion rather than overconfident attribution.

## Relationship to the K01

The full K01 analysis will extend beyond this example by using a substantially larger curated reference collection and by formally addressing source, serotype, geography and year imbalance; lineage-aware and geographic validation; probability calibration; and propagation of source-attribution uncertainty into downstream epidemiologic models.

The worked example will therefore serve as both preliminary feasibility evidence and a training scaffold for collaborative development of the Salmonella implementation.
