## Overview

**SourceRunnerML** is designed as a decision-support tool for microbial source attribution, rather than a black-box classifier. It learns genomic patterns associated with known sources and applies them to new isolates, while explicitly representing uncertainty, mixed signals, and unknown origins.

At a high level, the workflow is:
- Input genomic feature matrices (cgMLST alleles or gene presence/absence) for training and prediction
- Preprocess and balance data, handling missingness and class imbalance
- Train machine-learning models (multiclass or one-vs-rest) with optional bootstrapping and ensembles
- Generate per-isolate source profiles, including confidence intervals and “Uncertain / Mixed” outcomes
- Summarise attribution patterns for surveillance, epidemiology, and downstream interpretation

SourceRunnerML supports both:
- forced single-source attribution (multiclass mode), and
- profile-based attribution (`OVR mode`), which is recommended for and mixed-source datasets.

For a detailed conceptual walkthrough, interpretation guidance, and best-practice recommendations, see:
docs/workflow_guide.md (link)
