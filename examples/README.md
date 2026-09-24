# SourceRunnerML examples

This directory contains **command templates and study-specific worked examples**. It is deliberately separate from the reusable software in `scripts/` and the general documentation in `docs/`.

If you are new to SourceRunnerML, start with [`../docs/getting_started.md`](../docs/getting_started.md) rather than opening a worked study first.

## What is here?

### `example_commands.md`

Copy/paste templates for:

- Campylobacter cgMLST preflight;
- a small debug run;
- a standard full-validation run;
- post-processing.

These use the **general v1.0 wrapper**.

### `peru_geographic_context/`

A manuscript-scale worked study of geographic reference-population effects in Peruvian *Campylobacter jejuni*.

It documents study-specific choices that are **stricter than the generic wrapper defaults**, including:

- removal of Peru leakage from the non-Peru global reference panel;
- fixed lineage-blocked validation folds;
- Global vs Peru-local vs combined reference comparisons;
- size-matched controls and learning curves;
- human attribution and nearest-source sensitivity analyses;
- PIRATE/IQ-TREE phylogenetic reconstruction;
- the plan for the *C. coli* extension.

Use this folder when reproducing or extending that study. Do not treat its frozen model settings as universal SourceRunnerML defaults.

### `salmonella_pilot/`

A planned feasibility example for adapting SourceRunnerML to Salmonella/EnteroBase cgMLST data.

It currently contains planning, source-harmonisation and runner scaffolding. **No Salmonella performance results are claimed yet.**

The major software issue to resolve before a definitive Salmonella workflow is general locus selection beyond the Campylobacter-style common prefix (issue #5).

## Data policy

Do not commit restricted clinical metadata, private genomes or large derived analysis directories to this repository.

Worked studies should contain one or more of:

- small redistributable example data;
- aggregate non-sensitive results;
- scripts/configuration;
- exact instructions for recreating inputs from their original public or controlled-access source.

## Input reminder

The standard full-validation wrapper expects SourceRunner-ready TSVs with:

- a source-label column in the training table;
- cgMLST allele columns shared between training and prediction tables;
- stable isolate IDs strongly recommended for auditability.

See [`../docs/getting_started.md`](../docs/getting_started.md) for the full input/preflight workflow.
