# SourceRunnerML workflow guide

This guide describes the recommended SourceRunnerML v1.0.0 workflow and how its outputs should be interpreted. It complements the top-level README and is intentionally conservative about source-attribution claims.

## 1. Scope

SourceRunnerML is a decision-support framework for microbial source attribution using source-labelled genomic profiles. It learns genomic patterns associated with represented source populations and estimates how strongly new isolates match those patterns.

It does **not** prove individual transmission events, assume that all possible sources are represented, or guarantee that source categories are biologically distinguishable. Attribution quality depends on the composition, metadata quality and genomic structure of the reference collection.

## 2. Current stable input

The current v1.0.0 workflow operates on cgMLST allele matrices.

Each row represents an isolate and each cgMLST locus is a feature. Training data additionally require a source-label column. Missing alleles are supported and can be filtered/imputed during preprocessing.

The current Campylobacter wrapper identifies loci by a common prefix (normally `CAMP`). This is convenient for PubMLST Campylobacter exports but is not sufficiently general for all organisms. The Salmonella worked example will add and document an explicit locus-selection step suitable for EnteroBase Salmonella data.

## 3. Recommended v1.0.0 modelling mode

The recommended full-validation workflow is **multiclass source attribution** across a predefined candidate source set.

For each prediction isolate, the selected model produces one vector of source probabilities across the candidate classes. The probabilities sum to 1.0 for that model/source set. The output therefore represents a selection among the represented candidate sources rather than a collection of independent binary source models.

Low-confidence predictions should retain their full probability vector and uncertainty metrics. A most-likely source may be useful for summaries, but downstream analyses should not treat that label as known with certainty.

Earlier exploratory SourceRunnerML code included one-vs-rest/source-affinity concepts. Those approaches are not the recommended v1.0.0 validation path and should not be mixed with the multiclass interpretation without explicit method development and validation.

## 4. Preprocessing and reference-set checks

Before model training:

- harmonize source labels and document any merged categories;
- quantify sample numbers by source and relevant metadata such as lineage, geography and year;
- detect and filter loci with excessive missingness;
- impute missing values using the documented workflow;
- assess whether severe class imbalance requires class weighting or balanced resampling;
- decide whether proposed source categories have adequate representation to support independent inference.

Reference-set composition is part of the model, not a neutral background detail. Poorly represented or poorly discriminated sources should be merged, excluded, or reported at a coarser resolution supported by validation.

## 5. Model comparison and validation

The full-validation wrapper can compare random forest, logistic regression, XGBoost, LightGBM and CatBoost when installed. Balanced accuracy is the default selection metric for imbalanced source datasets.

The current Campylobacter workflow provides stratified cross-validation, confusion matrices, per-class classification reports and bootstrap/out-of-bag validation. Bootstrap prediction uses independently trained replicate models to quantify variation in source probabilities.

For new organisms, additional validation should be chosen to match the biology and sampling design. The planned Salmonella extension will add:

- lineage-grouped cross-validation to reduce leakage among closely related isolates;
- geographic holdouts where sample size permits;
- probability-calibration assessment;
- explicit evaluation of reference-set imbalance by source, serotype, geography and year.

When the true source of human infection is unknown, model validation should rely primarily on held-out source-labelled non-human isolates. Human epidemiologic exposure data can provide complementary external evidence but should not be described as a gold standard.

## 6. Prediction uncertainty

SourceRunnerML v1.0.0 uses independently trained bootstrap replicate models to estimate prediction uncertainty. Typical outputs include:

- mean source probabilities across replicate models;
- standard deviations of predicted probabilities;
- agreement/consensus among replicate predictions;
- entropy- or confidence-based uncertainty summaries;
- optional filtering of low-confidence most-likely labels.

Uncertainty should be carried into interpretation rather than removed by forcing every isolate into a single high-confidence source category.

## 7. Linking attribution to epidemiologic models

When source attribution is used upstream of epidemiologic analyses, uncertainty in source membership should be propagated downstream. For the planned Salmonella example, the proposed approach is to repeatedly sample one source assignment for each isolate from its predicted multiclass probability vector, aggregate integer source-specific counts, fit the downstream epidemiologic model in each replicate dataset, and combine estimates across replicates.

Hard maximum-probability assignments and probability-weighted summaries can be retained as sensitivity analyses.

## 8. Interpretation best practices

- Report candidate source definitions and reference-set composition clearly.
- Report source-specific performance, not only overall accuracy.
- Use grouped/held-out validation where related lineages could otherwise leak between training and test sets.
- Treat human-source assignments as probabilistic estimates of likely source association.
- Prefer coarser, defensible source categories over fine categories that cannot be discriminated.
- Preserve low-confidence and uncertain predictions in downstream interpretation.
- Document all preprocessing, source-label harmonization and model-selection decisions.

## 9. Salmonella worked example

A small public/redistributable Salmonella feasibility example is planned in [`../examples/salmonella_pilot/`](../examples/salmonella_pilot/). Its purpose is to demonstrate that the framework can ingest Salmonella cgMLST profiles and to establish a transparent baseline for subsequent organism-specific development. It is not intended to substitute for the full K01 reference collection or final validation analysis.

See [`salmonella_pilot_plan.md`](salmonella_pilot_plan.md) for the proposed workflow and success criteria.
