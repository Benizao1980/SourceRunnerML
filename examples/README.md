# SourceRunnerML examples

This directory contains reproducible command examples and worked-study scaffolds. Full cgMLST datasets and large SourceRunnerML outputs are intentionally not stored in GitHub.

Use SourceRunner-ready TSV files with:

- an isolate/sample identifier column, usually `id` or `isolate`;
- a source-label column for training, usually `source` or `reduced`;
- cgMLST allele columns shared between training and prediction datasets.

For quick testing, use `--max_train_rows` and `--max_predict_rows` before launching full HPC jobs.

## Available examples

- [`example_commands.md`](example_commands.md) - Campylobacter-oriented debug and full-validation command templates.
- [`peru_geographic_context/`](peru_geographic_context/) - detailed manuscript-scale worked study of geographic reference-population effects in Peruvian *Campylobacter jejuni*, including frozen benchmark results, leakage control, lineage-blocked validation, size-matched controls, human attribution, nearest-source validation, PIRATE/IQ-TREE phylogeny, and the planned *C. coli* extension.
- [`salmonella_pilot/`](salmonella_pilot/) - planned small Salmonella feasibility example for adapting the framework to EnteroBase Salmonella cgMLST data.

The Peru worked study is intentionally separated from the general SourceRunnerML API: it documents the exact scientific design and frozen manuscript analysis profile. The Salmonella pilot is likewise separated from the final K01 analysis: it is a worked example to demonstrate data ingestion, baseline multiclass source attribution and validation scaffolding, not a definitive source-attribution model for US salmonellosis.
