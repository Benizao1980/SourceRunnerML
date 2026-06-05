# SourceRunnerML v1.0.0

This is the first stable SourceRunnerML release for cgMLST-based microbial source attribution workflows.

## Highlights

- Stable SourceRunnerML command-line runner.
- Full-validation workflow with model comparison across random forest, logistic regression, XGBoost, LightGBM and CatBoost.
- Proper bootstrap replicate model fitting for prediction uncertainty.
- Cross-validation outputs including model comparison, classification reports and confusion matrices.
- Metadata-rich post-processing of predictions by country, year, ST, clonal complex, cgST and LINcode.
- Plot export for source attribution, model comparison and uncertainty summaries.
- Python 3.6-compatible scripts for older HPC environments.
- SLURM templates for C. coli and C. jejuni cgMLST workflows.

## Recommended tag

`v1.0.0`

## Notes

The legacy filenames `SourceRunnerML_v0_5_3.py` and `utils_v0_5.py` are retained as compatibility shims. New code should use `SourceRunnerML.py`, `SourceRunnerML_v1_0.py` and `utils_v1_0.py`.
