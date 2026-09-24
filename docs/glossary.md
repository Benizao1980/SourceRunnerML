# SourceRunnerML glossary

This glossary defines terms as they are used in this repository. It is intended to make the software and worked studies easier to follow for users who are new to source attribution, cgMLST or machine-learning validation.

## Allele matrix

A table in which each row is an isolate and each genomic locus is represented by an allele identifier. SourceRunnerML treats the selected allele columns as model features.

## Balanced accuracy

The mean recall across classes. In a three-source model, each source contributes equally to the final score regardless of how many isolates it contains.

This is the default model-selection metric in the full-validation wrapper because source datasets are often imbalanced.

## Bootstrap model

A model trained on a bootstrap resample of the training data. SourceRunnerML trains independent bootstrap replicate models rather than repeatedly querying one fitted model.

## Burn-in

In the current wrapper, the first configured number of bootstrap replicates excluded from the post-burn-in summary. This is a workflow parameter retained from development; always report the value used if relying on the post-burn-in summary.

## Candidate source set

The source categories that the model is allowed to distinguish, for example `Poultry`, `Ruminant` and `Pig`.

Predicted probabilities are conditional on this candidate set. A source that is absent from the training data cannot be inferred directly.

## Calibration

The agreement between predicted probabilities and observed frequencies. For example, among predictions assigned probability 0.8 to a source, a well-calibrated model would be correct about 80% of the time in an appropriate validation dataset.

Calibration is separate from discrimination/accuracy.

## cgMLST

**Core-genome multilocus sequence typing.** A genome is represented by allele calls across a defined set of core loci. SourceRunnerML v1.0 uses cgMLST allele matrices as its principal genomic feature representation.

## Class

A model category. In SourceRunnerML this normally means a source category such as poultry or ruminant.

## Clonal complex (CC)

A conventional grouping of related MLST sequence types. It can be useful for lineage-aware summaries or validation, although cgMLST provides much finer genomic resolution.

## Confusion matrix

A table comparing true source labels with predicted source labels in a known-source validation set. It shows which sources are being confused with one another.

## Cross-validation (CV)

A validation approach that repeatedly divides known-source data into training and held-out subsets. Predictions for held-out isolates are used to estimate performance.

The general v1.0 full-validation wrapper uses stratified cross-validation. Some manuscript analyses in this repository use stronger lineage-blocked designs implemented separately.

## Entropy

A summary of uncertainty in a probability vector. Predictions distributed fairly evenly across several sources have higher entropy than predictions concentrated strongly on one source.

## Feature

A variable supplied to the model. In the current SourceRunnerML workflow, features are usually cgMLST allele columns.

## Geographic holdout

A validation design in which one geographic population is excluded from model training and used for testing. It asks whether a model transfers to a different location rather than merely fitting the geography represented in training.

## Geographic leakage

When isolates from the intended test geography are inadvertently included in the training/reference panel. This can make geographic generalisation look better than it really is.

The Peru worked study explicitly audits and removes Peruvian isolates from the non-Peru global reference panel before comparison.

## Grouped cross-validation

Cross-validation in which related isolates are kept in the same fold according to a grouping variable, for example a lineage or genomic cluster. It reduces leakage of near-identical or closely related genomes across training and test folds.

## Imputation

Replacing missing feature values with estimated/substitute values so a model can be fitted. The current full-validation wrapper uses training-set most-frequent allele imputation after locus filtering.

## Lineage

A genetically related group of isolates. Depending on the study, lineage may be represented by MLST clonal complex, cgMLST cluster, LIN code or another pre-defined genomic grouping.

## Lineage leakage

When closely related isolates from the same lineage appear in both training and validation folds. A model may then recognise lineage identity rather than learn a source signal that generalises to unseen lineages.

## LIN code

A hierarchical genomic clustering label based on allelic distance thresholds. In Campylobacter analyses it can provide a fine-scale grouping variable for lineage-blocked validation.

## Locus missingness

The fraction of isolates lacking a usable allele call at a locus. The full-validation wrapper can remove loci whose missingness exceeds a configured threshold.

## Macro F1

The unweighted mean of the F1 score calculated separately for each class. Like balanced accuracy, it prevents large classes from dominating the summary, but it incorporates both precision and recall.

## MLST

**Multilocus sequence typing.** The traditional Campylobacter MLST scheme uses seven housekeeping loci. MLST sequence type and clonal complex are useful metadata but are much lower resolution than cgMLST.

## Model comparison

Running several candidate algorithms on the same validation design and comparing their held-out performance. SourceRunnerML can compare random forest, logistic regression, XGBoost, LightGBM and CatBoost when installed.

## Multiclass source attribution

A model in which all candidate sources are considered simultaneously. Each isolate receives one probability vector across the source classes.

This is the recommended SourceRunnerML v1.0 interpretation.

## Nearest-source analysis

A non-model sensitivity analysis that identifies the closest source-labelled reference genome(s) under a genomic distance such as cgMLST allelic distance. It is useful as an independent check but is not equivalent to the machine-learning model.

## Out-of-bag (OOB) validation

In a bootstrap replicate, some observations are not selected into the bootstrap training sample. These out-of-bag observations can be used as a held-out validation set for that replicate.

## Out-of-fold (OOF) prediction

A prediction made for an isolate by a model that was not trained on that isolate. Combining OOF predictions across folds produces a leakage-resistant prediction set for known-source performance estimation, subject to the fold design used.

## Precision

Among isolates predicted as a given source, the proportion whose known source is actually that source.

## Prediction set

The isolates for which source probabilities are requested after model development. The prediction set may contain human clinical isolates, but it does not have to be human.

Some historical output filenames still use the word `human` because the workflow was developed for Campylobacter human attribution.

## Probability vector

The probabilities assigned to all candidate sources for one isolate. In a multiclass model these probabilities sum to 1 across the represented classes.

Example:

```text
Poultry 0.82 | Ruminant 0.14 | Pig 0.04
```

## Recall

Among isolates truly belonging to a source, the proportion correctly predicted as that source. This is especially useful for identifying a poorly recognised minority source.

## Reference set / training set

The source-labelled isolates used to train a source-attribution model. Reference-set composition strongly affects what the model can learn and what its predictions mean.

## Sequence type (ST)

The MLST allele-profile identifier assigned to an isolate under a defined MLST scheme.

## Source attribution

Inference about which represented source population a genome most closely resembles according to the trained model.

Source attribution is not the same as demonstrating a specific transmission event or identifying the immediate exposure that infected an individual patient.

## Source label harmonisation

Mapping heterogeneous metadata labels onto a defined set of model classes, for example `chicken`, `broiler` and `poultry meat` into a documented `Poultry` category where scientifically appropriate.

Harmonisation rules should be explicit and reproducible.

## Stratified cross-validation

Cross-validation designed so each fold contains approximately the same class proportions. This helps with source imbalance but does **not** by itself prevent related lineages from appearing in both training and test folds.

## Uncertainty

Variation or ambiguity in a source prediction. SourceRunnerML summarises uncertainty using replicate-model probabilities, their standard deviations, consensus/agreement and related summaries.

## Unknown / low-confidence prediction

A prediction whose maximum source probability does not meet a chosen confidence threshold. This should be interpreted as insufficient confidence among the represented candidate sources, not automatically as evidence for an unrepresented source.

## UFBoot / SH-aLRT

Phylogenetic branch-support procedures used by IQ-TREE. They appear in the Peru worked-study phylogenetic workflow but are not part of SourceRunnerML's machine-learning attribution itself.

## Validation set

Known-source isolates held out from model fitting and used to estimate how well the method predicts data it did not train on. The scientific value of validation depends heavily on how the holdout is constructed.
