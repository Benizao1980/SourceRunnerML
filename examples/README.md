# SourceRunnerML examples

This directory contains reproducible command examples and small worked-example scaffolds. Full cgMLST datasets and large SourceRunnerML outputs are intentionally not stored in GitHub.

Use SourceRunner-ready TSV files with:

- an isolate/sample identifier column, usually `id` or `isolate`;
- a source-label column for training, usually `source` or `reduced`;
- cgMLST allele columns shared between training and prediction datasets.

For quick testing, use `--max_train_rows` and `--max_predict_rows` before launching full HPC jobs.

## Available examples

- [`example_commands.md`](example_commands.md) - Campylobacter-oriented debug and full-validation command templates.
- [`salmonella_pilot/`](salmonella_pilot/) - planned small Salmonella feasibility example for adapting the framework to EnteroBase Salmonella cgMLST data.

The Salmonella pilot is intentionally separated from the final K01 analysis: it is a worked example to demonstrate data ingestion, baseline multiclass source attribution and validation scaffolding, not a definitive source-attribution model for US salmonellosis.
