# Salmonella pilot worked example

This directory is a **planning and feasibility scaffold** for adapting SourceRunnerML to source-labelled *Salmonella* cgMLST data.

> **Current status:** no Salmonella performance results are claimed. The pilot should not be treated as a validated Salmonella source-attribution model.

## Why this example exists

The current stable SourceRunnerML workflow was developed primarily with Campylobacter data. The Salmonella pilot is intended to test, transparently and on a small public/redistributable dataset, what needs to change before the framework is used for a larger Salmonella analysis.

The pilot should answer whether SourceRunnerML can:

1. ingest a curated Salmonella cgMLST allele matrix;
2. harmonise a small number of biologically meaningful source categories;
3. compare baseline multiclass classifiers;
4. perform bootstrap/OOB validation and return per-isolate source probability vectors;
5. add lineage-aware and geographic validation appropriate to Salmonella population structure;
6. report limitations and uncertainty as clearly as performance.

It is **not** the final K01 training/reference dataset or final source-attribution analysis.

## Important software limitation before running

The general v1.0 full-validation wrapper currently selects genomic loci using one common `--loci_prefix`. This works for Campylobacter `CAMP...` loci but may not match a Salmonella/EnteroBase export cleanly.

General locus selection is therefore tracked as **issue #5**. The preferred long-term solution is an explicit, reproducible feature-selection mechanism (for example a locus-list file, range or documented pattern) shared by preflight and full validation.

The included [`scripts/run_pilot.sh`](scripts/run_pilot.sh) is intentionally guarded: it will stop unless a locus prefix is supplied explicitly. That prevents the placeholder example from silently running with the wrong genomic columns.

## Proposed pilot design

### Candidate source classes

Start with a small number of well-represented, interpretable classes rather than maximising the number of labels. Initial candidates are:

- Poultry
- Cattle / ruminant
- Wild bird

If a class is sparse or poorly labelled, substitute or merge it rather than forcing an unstable category.

### Target size

For a fast feasibility example, aim for roughly 300–500 isolates per source where metadata quality permits. Record source counts before and after every filtering/subsampling step.

### Metadata to retain

At minimum:

- stable isolate/sample identifier;
- raw source label;
- harmonised source label;
- geography where available;
- collection year;
- serotype;
- cgMLST / EnteroBase cluster identifiers where available;
- cgMLST allele calls.

Do not commit patient-identifiable, restricted or large raw datasets to this repository.

## Recommended run order

1. **Choose the public dataset and confirm redistribution/access terms.**
2. **Inspect source metadata before defining classes.** Use [`config/source_harmonization.example.tsv`](config/source_harmonization.example.tsv) only as a template, not as a validated universal mapping.
3. **Resolve/define Salmonella locus selection.** Prefer implementing issue #5; only use a common prefix if the selected export genuinely supports one.
4. **Run preflight QC.** Quantify source counts, duplicate IDs, shared loci and missingness.
5. **Run a very small smoke test.** Start with random forest and logistic regression before scaling.
6. **Compare baseline models.** Add XGBoost when the input and environment are stable.
7. **Add stronger validation.** Compare ordinary stratified CV with lineage-grouped CV; use a geographic holdout if adequately powered.
8. **Assess calibration and uncertainty.** Retain low-confidence probability vectors rather than forcing hard assignments.
9. **Only then add compact results to this repository.**

See [`PLAN.md`](PLAN.md) for the checklist.

## Planned repository structure

```text
examples/salmonella_pilot/
├── README.md
├── PLAN.md
├── config/
│   └── source_harmonization.example.tsv
├── data/
│   └── README.md
├── scripts/
│   └── run_pilot.sh
└── results/
    └── README.md
```

## What would count as a useful pilot?

The goal is not a high headline accuracy number. A useful pilot should:

- run end-to-end from a documented public input format;
- quantify whether the chosen source categories are genomically distinguishable;
- expose the difference between ordinary and lineage-aware validation;
- produce interpretable probability/uncertainty outputs;
- identify classes or sampling structures that are not adequately supported;
- document every source-label and preprocessing decision.

A negative result for a source category is informative. It should lead to coarser source definitions, better reference sampling or exclusion—not overconfident attribution.

## Relationship to the rest of SourceRunnerML

For general software usage, start with [`../../docs/getting_started.md`](../../docs/getting_started.md).

For a completed example of study-specific leakage control and lineage-aware validation, see the [`../peru_geographic_context/`](../peru_geographic_context/) worked study.
