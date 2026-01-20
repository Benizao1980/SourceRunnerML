This document outlines the planned development trajectory of SourceRunnerML.

---

## ✅ Current (v0.5.x)
- Multiclass and OVR profile-based source attribution
- Bootstrap training and prediction uncertainty
- Ensemble classifiers
- Explicit Uncertain / Mixed outcomes
- Generalist vs specialist classification
- Config-driven execution

---

## 🔜 Near-term (v0.6)
### Hierarchical attribution
- Host vs environment
- Livestock / companion / wildlife / human
- Sub-sources within livestock
- Hierarchy-aware Unknown handling

### Calibration & robustness
- Probability calibration (isotonic / sigmoid)
- Lineage-blocked cross-validation (ST / CC)
- Improved uncertainty summaries

---

## 🧪 Mid-term (v0.7–0.8)
### Phylogeny-aware modelling
- Core-genome PCs as covariates
- Optional relatedness correction
- Phylogeny-aware imputation

### Explainability
- SHAP integration (optional)
- Feature importance summaries
- Accessory vs core signal partitioning

---

## 🌐 Long-term vision
- Assembly-level inputs (k-mers / unitigs)
- Out-of-distribution detection
- Continual learning with new isolates
- Pretrained pathogen-specific models
- Workflow support (Snakemake / Nextflow)

---

## Guiding principles
- Biological realism > forced classification
- Transparent uncertainty
- Modular, testable components
- Reproducibility first
