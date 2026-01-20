# SourceRunnerML Workflow Guide

This guide explains how to think about SourceRunnerML conceptually, not just how to run it.

---

## 1. Input data
SourceRunnerML expects:
- cgMLST or gene presence–absence matrices
- One row per isolate
- One column per locus or gene
- A source label column for training

Environmental and mixed samples are fully supported.

---

## 2. Preprocessing
- Locus columns are auto-detected
- Values are coerced to numeric
- Missingness is filtered and/or imputed
- Class imbalance is addressed via resampling

---

## 3. Model training
You can choose between:
- **Multiclass mode**: forced single-source prediction
- **OVR profile mode**: one model per source

OVR mode is recommended for:
- wastewater
- surface water
- wildlife
- mixed or unknown reservoirs

---

## 4. Prediction & uncertainty
For each isolate:
- Per-source scores are generated
- Bootstrap models provide:
  - mean score
  - standard deviation
  - confidence intervals
- Thresholding yields:
  - single-source
  - multi-source
  - Uncertain outcomes

---

## 5. Level 0: generalist vs specialist
Isolates are summarised as:
- **Specialist**: one dominant source signal
- **Generalist**: diffuse or mixed signals

This helps interpret environmental and wastewater isolates.

---

## 6. Interpretation best practices
- Treat scores as **evidence**, not absolute truth
- Avoid over-interpreting weak or mixed signals
- Use hierarchy and Unknown categories explicitly
- Prefer robustness checks over single “best” models

---

## 7. Outputs
SourceRunnerML produces:
- Tabular predictions
- Uncertainty summaries
- Diagnostics and metadata reports
- Visualisation-ready files

---

## Final note
SourceRunnerML is designed to support **epidemiological reasoning**, not replace it. Use it as a decision-support tool, not a black-box oracle.
