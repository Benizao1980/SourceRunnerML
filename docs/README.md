# SourceRunnerML documentation

Use this directory as the documentation index.

## For new users

Start with [`getting_started.md`](getting_started.md). It covers installation, input tables, preflight QC, a small debug run, the full-validation command, output interpretation and common failure modes.

## For methods and interpretation

Read [`workflow.md`](workflow.md). It explains the modelling logic, reference-set checks, validation, bootstrap/OOB uncertainty and interpretation limits.

## For terminology

Use [`glossary.md`](glossary.md). It defines the terms used throughout the repository, including cgMLST, OOF, OOB, balanced accuracy, lineage leakage, geographic leakage, calibration and source-attribution probability vectors.

## Worked studies

Worked studies live under [`../examples/`](../examples/), not in `docs/`, so that study-specific scientific designs remain clearly separated from the reusable software documentation.

- [`../examples/peru_geographic_context/`](../examples/peru_geographic_context/) — frozen manuscript-scale Peru analysis and reproducibility record.
- [`../examples/salmonella_pilot/`](../examples/salmonella_pilot/) — planned Salmonella feasibility/adaptation scaffold; no Salmonella performance results are currently claimed.

## Design principle

The repository distinguishes three layers:

1. **Reusable software** — `scripts/`
2. **General documentation** — `docs/`
3. **Study-specific designs and examples** — `examples/`

This separation is intentional. A worked study may use stricter validation or frozen settings that are not the defaults of the general wrapper.
