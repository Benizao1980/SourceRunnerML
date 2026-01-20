# SourceRunnerML Workflow Guide

This guide explains how to interpret and use SourceRunnerML conceptually. It is intended
to complement the README and help users make biologically sensible decisions when
applying the pipeline.

---

## 1. What SourceRunnerML does (and does not do)

SourceRunnerML is a **decision-support framework** for microbial source attribution.
It identifies genomic patterns associated with known sources and quantifies how
strongly new isolates match those patterns.

It does **not**:
- force every isolate into a single source,
- assume all sources are known or represented,
- treat predictions as absolute truth.

Instead, it explicitly supports **uncertainty, mixed attribution, and unknown origins**.

---

## 2. Input data

SourceRunnerML currently operates on **cgMLST allele matrices exported from PubMLST
(v2 cgMLST scheme)**.

Each row represents an isolate and each column represents a cgMLST locus. Allele values
are treated as numeric features; missing alleles are allowed and handled internally.

Environmental, wastewater, and wildlife isolates are fully supported.

Although cgMLST alleles form the current input standard, most modelling steps are
feature-agnostic, enabling future expansion beyond PubMLST outputs.

---

## 3. Preprocessing

Before model training, SourceRunnerML:
- detects cgMLST locus columns automatically,
- coerces mixed-type values to numeric,
- filters loci and isolates based on missingness,
- imputes missing values (configurable),
- addresses class imbalance via resampling.

These steps aim to minimise technical bias while preserving biological signal.

---

## 4. Modelling modes

### Multiclass mode
- Produces a single “best” source per isolate
- Appropriate for well-defined host datasets
- Less suitable for environmental or mixed reservoirs

### OVR profile mode (recommended for surveillance)
- Trains one binary model per source
- Produces a per-isolate **source affinity profile**
- Supports:
  - co-attribution,
  - Uncertain / Unknown outcomes,
  - downstream hierarchy or grouping

---

## 5. Uncertainty and bootstrapping

SourceRunnerML uses bootstrapping to:
- estimate variability in model predictions,
- generate confidence intervals per source,
- reduce sensitivity to sampling noise.

Uncertainty is a feature, not a failure mode.

---

## 6. Level 0: generalist vs specialist

In OVR mode, isolates are summarised as:
- **Specialist**: one dominant source signal
- **Generalist**: diffuse or mixed signals

This distinction is particularly informative for:
- wastewater,
- surface water,
- wildlife,
- long-term environmental persistence.

---

## 7. Interpretation best practices

- Treat source scores as **relative evidence**, not absolute probabilities
- Be cautious with low-confidence or mixed predictions
- Prefer “Unknown” to overconfident attribution
- Use hierarchy and aggregation where appropriate

SourceRunnerML is designed to support epidemiological reasoning, not replace it.

---

## 8. Outputs

Typical outputs include:
- per-sample source profiles,
- confidence intervals,
- summary tables for surveillance,
- visualisation-ready files for downstream tools.

---

## Final note

SourceRunnerML is most powerful when used iteratively:
- refine training sets,
- reassess thresholds,
- validate against external data.

It is intentionally conservative in attribution to encourage responsible interpretation.
