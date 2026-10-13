# Peru aggregate benchmark results

The TSV here reproduces **aggregate, previously documented known-source *C. jejuni* confusion counts and rounded headline metrics** from the companion [study README](../README.md). It contains **no patient-level records and no individual genomes**.

~~~bash
python examples/peru_geographic_context/scripts/verify_benchmark_summary.py
~~~

Run from the SourceRunnerML repository root. This checks that the reported poultry/ruminant confusion counts sum to the documented test class sizes and that balanced accuracy calculated from those counts matches the rounded reported values. It **does not** reproduce a trained model, inspect fold assignments, verify macro-F1, or independently establish scientific validity.

Re-running the scientific comparison requires the original inputs, frozen fold manifest, preprocessing/model settings and study-specific scripts described in [START_HERE.md](../START_HERE.md) and [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

