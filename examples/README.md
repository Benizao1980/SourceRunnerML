# SourceRunnerML examples

This directory contains command examples only. Full cgMLST datasets and SourceRunner outputs are intentionally not stored in GitHub.

Use SourceRunner-ready TSV files with:

- an isolate/sample identifier column, usually `id` or `isolate`
- a source label column for training, usually `reduced`
- cgMLST allele columns with a common prefix, usually `CAMP`

For quick testing, use `--max_train_rows` and `--max_predict_rows` before launching full HPC jobs.
