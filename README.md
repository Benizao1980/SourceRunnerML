<p align="center">
  <img src="sourcerunner_logo_variantB_tron.svg" width="360" alt="SourceRunnerML logo">
</p>

# SourceRunnerML

*Uncertainty-aware microbial source attribution from population genomic data.*

SourceRunnerML is a framework for microbial source attribution built for the kinds of
pathogen populations we actually work with: **structured, recombining, incompletely
sampled, and ecologically overlapping**.

The central idea is simple: in most real datasets, source attribution is not a question
with a single correct answer. Forcing isolates into definitive categories can be
convenient, but it often hides the biology. SourceRunnerML is designed to make that
uncertainty explicit and interpretable, rather than smoothing it away.

---

## Where this fits

SourceRunnerML is part of a deliberately connected set of tools:

- **PANOPTICON** is used to explore population structure, gene content, and ecological signal
- **SourceRunnerML** uses those patterns to infer likely sources of infection
- **BAMPS-ML** applies the same philosophy to antimicrobial resistance phenotypes

Together, these tools support a workflow that moves from  
**population structure → transmission inference → phenotype prediction**, while keeping
lineage effects and uncertainty visible throughout.

---

## What SourceRunnerML actually does

At a practical level, SourceRunnerML learns genomic patterns associated with known sources
and applies them to new isolates. Instead of producing a single label per isolate, it
outputs **source affinity profiles** with associated uncertainty.

In broad terms, the workflow is:

- take genomic feature matrices (currently cgMLST alleles)
- apply basic filtering and balancing, accounting for missingness
- train machine-learning models using either multiclass or one-vs-rest strategies
- optionally stabilise estimates using bootstrapping
- generate per-isolate probability profiles and confidence intervals
- summarise attribution patterns for downstream epidemiological interpretation

The emphasis is on **robustness and interpretability**, not on squeezing out marginal gains
in headline accuracy.

---

## Attribution modes

SourceRunnerML supports two complementary inference modes.

### Multiclass mode
A conventional forced-choice classifier that assigns each isolate to a single source.
This is mainly useful for benchmarking, method comparison, or tightly controlled datasets.

### Profile-based one-vs-rest (OVR) mode *(recommended)*
One binary model per source, producing:
- per-source affinity scores
- support for co-attribution
- explicit **Uncertain / Unknown** outcomes where appropriate

OVR mode is intended for **environmental, zoonotic, and mixed-source datasets**, where
overlap between reservoirs is expected rather than exceptional.

➡️ A more detailed discussion of interpretation and best-practice use is provided in  
**[`docs/workflow.md`](https://github.com/Benizao1980/SourceRunnerML/blob/main/docs/workflow.md)**

---

## Input data

Current versions of SourceRunnerML operate on **cgMLST allele matrices**
exported from **PubMLST (v2 cgMLST scheme)**.

This reflects the original use cases (*Campylobacter* and other enteric bacteria), but the
internal design is intentionally general.

### Training file (`--train_file`)
- one row per isolate
- `sample_id`
- `source` (known source label)
- cgMLST locus columns (numeric; missing values allowed)

### Prediction file (`--predict_file`)
- same structure as the training file
- `source` column optional or ignored
- supports unknown and environmental isolates

Training and prediction files must share overlapping loci. Missing data are handled
internally.

> Although cgMLST is currently the primary input, the architecture is designed to support
> alternative genomic feature representations (e.g. gene presence/absence, unitigs) in
> future releases.

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
```
