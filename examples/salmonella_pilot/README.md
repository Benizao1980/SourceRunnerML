# Salmonella pilot worked example

This directory is a placeholder for a small, reproducible demonstration of adapting SourceRunnerML to source-labelled *Salmonella* cgMLST data.

## Purpose

The pilot is intended to show that the existing SourceRunnerML workflow can:

1. ingest a curated *Salmonella* cgMLST allele matrix;
2. harmonize a small number of biologically meaningful source categories;
3. compare baseline multiclass classifiers;
4. perform bootstrap/OOB validation and produce per-isolate source probability vectors;
5. report model performance and uncertainty transparently.

It is **not** intended to represent the final K01 training/reference dataset or final source-attribution analysis.

## Proposed pilot design

### Candidate source classes

Start with three well-represented categories, prioritising data quality over breadth. Preferred initial classes are:

- Poultry
- Cattle / ruminant
- Wild bird

If one category has inadequate coverage, substitute another well-represented environmental or animal source rather than forcing a sparse class.

### Target size

For a fast worked example, aim for approximately 300–500 isolates per source class, balanced where practical. Record the full source counts before and after filtering so that the worked example remains transparent about resampling and class balance.

### Metadata to retain

At minimum:

- isolate/sample identifier
- source label (raw and harmonized)
- country / state where available
- collection year
- serotype
- cgMLST / EnteroBase cluster identifiers where available
- cgMLST allele columns

No patient-identifiable or restricted metadata should be committed to this repository.

## Planned workflow

1. **Reference-set curation**
   - document inclusion/exclusion rules;
   - harmonize raw source labels;
   - quantify source, serotype, geography and year distributions;
   - define a small pilot source set.

2. **Preflight / QC**
   - check required columns;
   - quantify missing cgMLST loci;
   - exclude isolates/loci failing predefined missingness thresholds;
   - save a compact QC summary.

3. **Baseline modelling**
   - compare random forest, logistic regression and XGBoost;
   - use balanced accuracy as the primary selection metric;
   - retain full multiclass probability vectors.

4. **Validation**
   - baseline stratified 5-fold cross-validation;
   - bootstrap/OOB validation;
   - add lineage-grouped cross-validation if cluster labels are available;
   - add a geographic holdout if the pilot contains adequate representation from more than one region;
   - inspect confusion matrices and class-specific performance.

5. **Uncertainty / calibration**
   - summarize bootstrap variability and entropy;
   - assess probability calibration where sample size permits;
   - retain low-confidence predictions rather than forcing interpretation.

6. **Outputs for the worked example**
   - dataset summary table;
   - model-comparison table;
   - confusion matrix;
   - per-source performance summary;
   - example probability-vector plot;
   - short interpretation of what worked, what did not, and what remains for the full K01 implementation.

## Planned repository structure

```text
examples/salmonella_pilot/
├── README.md
├── data/
│   └── README.md
├── config/
│   └── source_harmonization.example.tsv
├── scripts/
│   └── run_pilot.sh
└── results/
    └── README.md
```

The repository should contain only redistributable/example data or scripts that reproduce data acquisition/processing from public sources. Full external datasets and large output directories should remain outside GitHub.

## Success criteria

The pilot is successful if it demonstrates an end-to-end *Salmonella* run and provides an honest estimate of whether the selected source categories are genomically distinguishable. High headline accuracy is not required; poor discrimination is itself informative and should be documented rather than hidden.

## Next steps

- Confirm the public source dataset and permitted redistribution.
- Agree the first three source categories.
- Add lineage/group metadata suitable for leakage-resistant validation.
- Run a small smoke-test dataset before scaling the pilot.
- Replace placeholders below with real commands and outputs once the first run is complete.

### Status

**Planning / placeholder stage — no Salmonella performance results are claimed yet.**
