# Peru Campylobacter geographic-context source attribution

**Status: active analysis / manuscript preparation (24 September 2026)**

This directory documents the Peru source-attribution case study built around [SourceRunnerML](../../README.md). The central question is not simply *which reservoir is assigned to human Campylobacter isolates?* but **how much source-attribution performance depends on the geographic composition of the reference population**.

The current analysis is strongest for *Campylobacter jejuni*. A deliberately narrower *C. coli* verification is the next analytical step.

## Main conclusion so far

For Peruvian *C. jejuni*, human isolates are overwhelmingly assigned to poultry-associated genomic populations across global-only, Peru-only and combined reference populations. At the same time, benchmark performance on known-source Peruvian isolates depends strongly on geographic representation, especially for ruminant isolates.

The best interpretation is therefore:

> **Global reference collections provide breadth, geographically matched source genomes provide locally relevant ecological-genetic information, and combining both yields the strongest overall source attribution.**

This is not a claim that local reference panels are universally superior to global panels. The benefit is lineage-dependent and is strongest where local genomes restore source-informative representation of locally circulating lineages.

---

## Study components

### 1. Primary *C. jejuni* geographic-context benchmark

Known-source Peru *C. jejuni*:

| Source | n |
|---|---:|
| Poultry | 1,076 |
| Ruminant | 149 |
| Pig | 11 |
| **Total** | **1,236** |

The formal benchmark is poultry vs ruminant only (`n = 1,225`) because the Peru pig class (`n = 11`) is too small for a stable formal performance comparison.

Original global source panel:

| Source | n |
|---|---:|
| Poultry | 11,142 |
| Ruminant | 6,126 |
| Pig | 104 |
| **Total** | **17,372** |

A leakage audit identified **127 Peru isolates in the original global panel**. All 127 were removed before the production geography benchmark. The cleaned non-Peru global panel contains:

| Source | n |
|---|---:|
| Poultry | 11,015 |
| Ruminant | 6,126 |
| Pig | 104 |
| **Total** | **17,245** |

The primary benchmark uses **1,142 shared CAMP cgMLST loci**.

### Frozen validation design

- 5-fold `StratifiedGroupKFold`
- fixed folds reused for every reference-population comparison
- grouping preferentially by `LINcode[17]`
- exact cgMLST-profile fallback where needed
- no fine genomic group is allowed to cross train/test
- every Peru known-source isolate is tested once
- identical test folds are used for Global, Peru-local and Global+Peru comparisons

Three reference-population designs are compared:

1. **Non-Peru global**
2. **Peru-local out-of-fold**
3. **Global + Peru-local out-of-fold**

### Production benchmark result

| Reference population | Balanced accuracy | Macro F1 | Poultry recall | Ruminant recall |
|---|---:|---:|---:|---:|
| Non-Peru global | 0.754 | 0.816 | 0.997 | 0.510 |
| Peru-local OOF | 0.875 | 0.905 | 0.991 | 0.758 |
| Global + Peru-local OOF | **0.893** | **0.922** | **0.993** | **0.792** |

Key effect sizes:

- Peru-local minus global balanced accuracy: **+0.121**
- Global+Peru minus global: **+0.139**
- Global+Peru minus Peru-local: **+0.018**

The global reference population performs extremely well for poultry but poorly for Peruvian ruminant isolates. Adding Peru source genomes reduces ruminant misclassification from **73/149 to 31/149**, while poultry classification remains essentially unchanged.

Confusion matrices:

| Model | Poultry→Poultry | Poultry→Ruminant | Ruminant→Poultry | Ruminant→Ruminant |
|---|---:|---:|---:|---:|
| Global | 1,073 | 3 | 73 | 76 |
| Peru-local | 1,066 | 10 | 36 | 113 |
| Global+Peru | 1,069 | 7 | 31 | 118 |

### Size-matched geography control

To test whether the Peru-local advantage was simply caused by training-set size, global and Peru source populations were repeatedly downsampled to equal numbers per source.

Across 5 folds × 10 repeats:

- global mean balanced accuracy: **0.751**
- Peru-local mean balanced accuracy: **0.913**
- mean Peru-local advantage: **+0.163**
- the direction of effect was positive in every fold

Equal-size learning curves also favored Peru-local training at every tested sample size:

| genomes/source | Global BA | Peru-local BA |
|---:|---:|---:|
| 10 | 0.616 | 0.743 |
| 25 | 0.707 | 0.842 |
| 50 | 0.714 | 0.864 |
| 75 | 0.738 | 0.883 |
| 100 | 0.727 | 0.906 |

This is the key evidence that the geographic effect is not explained by simply having more genomes.

### Lineage-dependent rescue

Ruminant recall differs strongly among folds, and local reference genomes help most when they add source-consistent representation of the relevant lineage.

Fold-level ruminant recall:

| Fold | Global | Peru-local | Global+Peru |
|---:|---:|---:|---:|
| 1 | 0.267 | 0.367 | 0.500 |
| 2 | 0.733 | 0.933 | 0.933 |
| 3 | 0.633 | 0.867 | 0.867 |
| 4 | 0.800 | 0.900 | 0.933 |
| 5 | 0.103 | 0.724 | 0.724 |

Two focal examples explain the mechanism:

- **Fold 1 / ST-179 complex**: 9 test ruminants. Same-LIN10 out-of-fold local support is sparse and source-mixed (1 ruminant + 1 poultry). Rescue is therefore partial.
- **Fold 5 / ST-48 complex**: 14 of 29 test ruminants. Same-LIN10 out-of-fold local support is source-consistent (4 ruminant + 0 poultry). Ruminant recall rises strongly after adding Peru genomes.

These are fold-level recall values; they must not be reported as ST-specific performance estimates.

---

## Human *C. jejuni* attribution

Primary human prediction set: **981 Peru human *C. jejuni***.

Using a 0.60 maximum-probability threshold for filtered predictions:

| Training population | Poultry | Ruminant | Pig | Uncertain |
|---|---:|---:|---:|---:|
| Global-only | 915 (93.27%) | 7 (0.71%) | 0 | 59 (6.01%) |
| Peru-contextual | 924 (94.19%) | 40 (4.08%) | 0 | 17 (1.73%) |
| Global+Peru | 915 (93.27%) | 29 (2.96%) | 2 (0.20%) | 35 (3.57%) |

Thus, **>93% of human isolates are poultry-associated under every training strategy**.

Independent nearest-source cgMLST comparison gives the same high-level signal:

- nearest poultry: **925/981 (94.29%)**
- nearest ruminant: **56/981**

Additional sensitivity analyses already completed include:

- repeated balanced Poultry/Ruminant training (`200` replicates; mean balanced accuracy ≈ `0.939`)
- cgMLST-clonal-complex grouped CV
- MLST-clonal-complex grouped CV
- nearest-source cgMLST distance analysis
- lineage-focused exception analyses

These analyses are supportive rather than the central manuscript result.

---

## *C. jejuni* phylogeny for Figure 1

A whole-population phylogeny was generated independently of the classifier analysis.

Initial population:

- **2,217 Peru *C. jejuni* genomes**
- Prokka annotation
- PIRATE pangenome analysis
- PIRATE core alignment: **1,405,341 bp**

Alignment QC identified 23 genomes with >50% missing/ambiguous core-alignment sequence. These were excluded before the final tree, leaving:

- **2,194 genomes in the final phylogeny**

The production workflow is included in:

```text
scripts/cjejuni_tree/
```

The workflow includes the original 2,217-genome Prokka → PIRATE → IQ-TREE launcher, the streaming alignment-QC recovery step, and the final higher-memory IQ-TREE submission.

Important operational lessons retained in the scripts:

- University of Arizona array limits require batching/offset handling for >2,000 genomes.
- PIRATE completed successfully; it did not need to be repeated after the first IQ-TREE failure.
- the first IQ-TREE run failed because five sequences were entirely missing and 23 exceeded 50% missingness
- filtering is streamed on a compute node rather than loading the ~3 GB FASTA into memory on the login node
- ModelFinder required >160 GB RAM; the successful rerun used a 240 GB IQ-TREE allocation

The final tree is intended to use pubMLST IDs as tip names, allowing direct joins to the Figure 1 metadata table.

---

## Interpretation boundaries

`Poultry-associated` means genomic population association with the sampled poultry reservoir. It must not be interpreted as proof that an individual infection came directly from retail chicken meat or from a specific exposure event.

Likewise, `geographic context` is not treated as a causal variable. It is a proxy for local ecological-genetic structure: locally circulating lineages, host associations, agricultural systems and source populations represented in the training data.

---

## What remains unfinished

The major remaining analytical block is ***C. coli***. The *C. coli* analysis is designed as an independent/generalisation test rather than a second full-scale analysis.

Current blocker: the three SourceRunner-ready full-cgMLST tables have not yet been located/confirmed:

```text
Ccoli_global_source.tsv
Ccoli_peru_source.tsv
Ccoli_peru_human.tsv
```

There is already substantial *C. coli* metadata and lineage information, including ST-828- and ST-1150-complex isolates, but the formal verification requires the full CAMP cgMLST feature matrices.

A complete prepared analysis bundle is included under:

```text
scripts/ccoli/
```

**Do not run the model stage until the inventory/preflight confirms the input counts and feature matrix.**

See [NEXT_STEPS_CCOLI.md](NEXT_STEPS_CCOLI.md).

---

## Repository structure

```text
case_studies/peru_geography/
├── README.md
├── REPRODUCIBILITY.md
├── NEXT_STEPS_CCOLI.md
├── STATUS_2026-09-24.md
├── results/
│   └── cjejuni_primary_benchmark.tsv
└── scripts/
    ├── cjejuni_tree/
    │   ├── submit_peru2217_pirate_iqtree_v5.sh
    │   ├── resubmit_cjejuni_iqtree_qc50_v2.sh
    │   └── run_cjejuni_iqtree_qc50_240G.sh
    └── ccoli/
        └── ... inventory, benchmark, human prediction, distance and tree scripts
```

Large genomic datasets, human metadata and large alignments/trees are deliberately **not** committed here. This repository documents the computational workflow and contains reusable code, not restricted study data.

---

## Relationship to SourceRunnerML

This case study uses the SourceRunnerML framework for cgMLST-based source attribution and extends it with a study-specific validation design for comparing geographically different training populations.

The production Peru work used XGBoost ensembles with:

- 300 trees
- maximum depth 6
- learning rate 0.05
- subsample 0.85
- column subsample 0.85
- 20% training-locus missingness threshold
- most-frequent imputation learned from training data only
- source-stratified bootstrap sampling
- 50 prediction models, seeds 25–74
- 0.60 confidence threshold for filtered human predictions

Formal known-source benchmark metrics use the unfiltered maximum-probability class; thresholded calls are retained as an uncertainty/coverage diagnostic.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full repeatable analysis recipe.
