# SourceRunnerML v0.5.4

SourceRunnerML is a flexible and extensible machine-learning pipeline for **microbial source attribution** using cgMLST data. It is designed for **One Health surveillance**, supporting host, environmental, and mixed-source isolates with explicit handling of uncertainty.

The pipeline supports **multiclass**, **ensemble**, and **one-vs-rest (OVR) profile-based** attribution, with bootstrapping, confidence intervals, and transparent “Uncertain / Mixed” outcomes.

---

## 🔎 Overview

**SourceRunnerML** is designed as a **decision-support tool** for microbial source attribution, rather than a black-box classifier. It learns genomic patterns associated with known sources and applies them to new isolates, while explicitly representing uncertainty, mixed signals, and unknown origins.

At a high level, the workflow is:
- Input genomic feature matrices (cgMLST alleles) for training and prediction
- Preprocess and balance data, handling missingness and class imbalance
- Train machine-learning models (multiclass or one-vs-rest) with optional bootstrapping and ensembles
- Generate per-isolate source profiles, including confidence intervals and “Uncertain / Mixed” outcomes
- Summarise attribution patterns for surveillance, epidemiology, and downstream interpretation

SourceRunnerML supports both:
- **forced single-source attribution** (multiclass mode), and  
- **profile-based attribution** (**OVR mode**), which is recommended for environmental and mixed-source datasets.

➡️ For a detailed conceptual walkthrough, interpretation guidance, and best-practice recommendations, see:  
**[`docs/workflow_guide.md`](https://github.com/Benizao1980/SourceRunnerML/blob/main/docs/workflow.md)**

---

## ✨ New in v0.5.4
- **OVR profile mode** (`--predict_mode ovr_profile`)
  - One binary model per source
  - Per-isolate source affinity profiles
  - Supports co-attribution and explicit Unknown calls
- **Bootstrap-aware prediction uncertainty**
  - Mean score, SD, and confidence intervals per source
- **Level 0 classification**
  - Generalist vs specialist (entropy + max score)
- **Config file support** (`--config`)
  - YAML-based configuration with CLI overrides
- Robustness and stability improvements over v0.5.x

---

## 📥 Input data requirements

**Current versions of SourceRunnerML are designed to operate on cgMLST allele matrices
exported from PubMLST (v2 cgMLST scheme).**

The pipeline expects tabular outputs derived from PubMLST allele calling, where rows
represent isolates and columns represent cgMLST loci.

### Supported input (current)

- PubMLST cgMLST v2 allele tables
- One row per isolate
- One column per cgMLST locus
- Numeric allele identifiers (missing values allowed)

This includes standard PubMLST exports used for *Campylobacter*, *Salmonella*,
*Escherichia coli*, and related enteric pathogens.

### Required files

#### 1. Training file (`--train_file`)
A `.tsv` or `.csv` file with:
- `sample_id` (unique isolate identifier)
- `source` (known source label for training)
- cgMLST locus columns

#### 2. Prediction file (`--predict_file`)
Same format as the training file, except:
- the `source` column is optional or ignored
- unknown or environmental isolates are allowed

Training and prediction files must share overlapping cgMLST loci; missing values
are handled internally.

> **Note**  
> While SourceRunnerML currently targets PubMLST cgMLST outputs, the internal
> architecture is designed to support alternative genomic feature matrices
> (e.g. gene presence/absence, unitigs) in future releases.

---

## 🚀 Quick start

```bash
python SourceRunnerML_v0_5_4.py \
  --train_file TRAIN.tsv \
  --predict_file PRED.tsv \
  --run_name demo_run \
  --predict_mode ovr_profile \
  --bootstrap 50 \
  --pred_bootstrap 25 \
  --level0 generalist_specialist

