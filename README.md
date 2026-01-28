<p align="left">
  <img src="sourcerunner_logo_variantB_tron.svg" width="360" alt="SourceRunnerML logo">
</p>

# SourceRunnerML

*Uncertainty-aware microbial source attribution from population genomic data.*

---

## Overview

**SourceRunnerML** is a framework for microbial source attribution designed for
**structured, recombining pathogen populations** and **overlapping ecological niches**.

Rather than forcing isolates into single definitive categories, SourceRunnerML
represents **partial attribution, mixed signals, and uncertainty explicitly**.
It is intended as a **decision-support tool** for genomic epidemiology and
surveillance, not a black-box classifier.

At its core, SourceRunnerML treats source attribution as a **population-level
inference problem**, shaped by lineage structure, gene flow, and incomplete
sampling.

---

## Conceptual position

SourceRunnerML is part of a broader analytical programme:

- **PANOPTICON** explores population structure, gene content, and ecological signal
- **SourceRunnerML** uses that structure to infer likely sources of infection
- **BAMPS-ML** extends the same principles to antimicrobial resistance phenotypes

Together, these tools support a coherent workflow from  
**population structure → transmission inference → phenotype prediction**.

---

## What SourceRunnerML does

At a high level, SourceRunnerML:

- learns genomic patterns associated with known sources
- applies these patterns to new or unknown isolates
- produces **per-isolate source affinity profiles**, not just point predictions
- exposes uncertainty arising from recombination, overlap, and missing data

The workflow is:

- Input genomic feature matrices (currently cgMLST alleles)
- Preprocess data (missingness, balancing, lineage-aware filtering)
- Train machine-learning models (multiclass or one-vs-rest)
- Optionally stabilise estimates using bootstrapping
- Generate per-isolate probability profiles with confidence intervals
- Summarise attribution patterns for downstream epidemiological interpretation

---

## Attribution modes

SourceRunnerML supports two complementary inference modes:

- **Multiclass mode**  
  Forced single-source attribution (useful for benchmarking and controlled datasets)

- **Profile-based OVR mode** *(recommended)*  
  One-vs-rest models per source, producing:
  - per-source affinity scores
  - co-attribution where appropriate
  - explicit **Uncertain / Unknown** outcomes

OVR mode is designed for **environmental, zoonotic, and mixed-source datasets**.

➡️ For a detailed conceptual walkthrough and interpretation guidance, see:  
**[`docs/workflow.md`](https://github.com/Benizao1980/SourceRunnerML/blob/main/docs/workflow.md)**

---

## Input data

Current versions of SourceRunnerML operate on **cgMLST allele matrices**
exported from **PubMLST (v2 cgMLST scheme)**.

### Training file (`--train_file`)
- one row per isolate
- `sample_id`
- `source` (known source label)
- cgMLST locus columns (numeric; missing values allowed)

### Prediction file (`--predict_file`)
- same format as training file
- `source` column optional or ignored
- supports unknown and environmental isolates

Training and prediction files must share overlapping loci; missing data are
handled internally.

> The internal architecture is designed to support alternative genomic feature
> representations (e.g. gene presence/absence, unitigs) in future releases.

---

## Quick start

```bash
python SourceRunnerML_v0_5_4.py \
  --train_file TRAIN.tsv \
  --predict_file PRED.tsv \
  --predict_mode ovr_profile \
  --bootstrap 50 \
  --pred_bootstrap 25 \
  --level0 generalist_specialist
