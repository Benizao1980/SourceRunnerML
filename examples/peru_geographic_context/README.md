# Peru Campylobacter geographic-context source attribution

This directory documents the manuscript-scale analysis used to test how geographic composition of reference genomes affects cgMLST-based source attribution of *Campylobacter jejuni* in Peru, with a planned/ongoing extension to *C. coli*.

The analysis is linked directly to **SourceRunnerML**, but it is intentionally documented as a **study-specific reproducibility profile**, because the production manuscript analysis used a frozen XGBoost ensemble and fixed lineage-blocked folds rather than every option exposed by the general-purpose wrapper.

## Scientific question

The primary question is not simply whether human *C. jejuni* isolates can be classified by source. It is whether the **geographic composition of the source-labelled reference population changes known-source classification performance and downstream inference for human isolates**.

Three reference strategies are compared:

1. **Non-Peru global** — broad source diversity, with every Peruvian source isolate removed.
2. **Peru-local** — geographically matched Peruvian source isolates, evaluated out-of-fold for the known-source benchmark.
3. **Global + Peru-local** — global breadth plus geographically matched local source diversity.

The main conclusion to date is that global and local reference panels are **complementary**: local genomes substantially improve recognition of Peruvian ruminant-associated diversity, while the combined global-plus-local panel gives the best overall benchmark performance.

## Current study status

### *C. jejuni* — production analysis complete

The primary known-source benchmark is frozen and complete.

Peru known-source isolates:

- poultry: **1,076**
- ruminant: **149**
- pig: **11**
- total: **1,236**

Formal benchmark:

- poultry + ruminant only: **1,225 isolates**
- pig excluded from the formal benchmark because only 11 Peru pig isolates were available
- 5-fold `StratifiedGroupKFold`
- fine genomic grouping based on `LINcode[17]` where available, with exact cgMLST-profile fallback
- no fine genomic group crosses train/test folds
- identical test folds for all three reference strategies

Original global source panel:

- poultry: 11,142
- ruminant: 6,126
- pig: 104
- total: 17,372

Leakage audit:

- 127 Peruvian isolates were identified in the original global source panel
- all 127 were removed before the production benchmark
- clean non-Peru global panel: **17,245 isolates**
  - poultry: 11,015
  - ruminant: 6,126
  - pig: 104

Human Peru *C. jejuni* prediction set:

- **981 isolates**

### *C. coli* — next analytical priority

The *C. coli* analysis is intentionally narrower. Its purpose is to test whether the geographic-reference effect generalises across species and whether major *C. coli* lineages, especially ST-828 complex and ST-1150 complex, behave differently.

Before running the *C. coli* benchmark, the source-labelled cgMLST matrices must be reconstructed/verified. Do **not** assume that the placeholder input tables used by development scripts already exist.

Planned sequence:

1. inventory all source-labelled *C. coli* data;
2. identify and remove Peru leakage from the global panel;
3. count usable Peru poultry/ruminant/pig source isolates;
4. determine whether a formal lineage-blocked Peru benchmark is defensible;
5. run human prediction under global / Peru / combined panels;
6. compare ST-828 and ST-1150 complex;
7. perform nearest-source cgMLST validation;
8. build a *C. coli* PIRATE/IQ-TREE phylogeny.

If local *C. coli* source counts are too small for a defensible formal benchmark, the analysis should be reported as a **secondary lineage-resolved validation**, rather than forcing the *C. jejuni* design onto inadequate data.

## Production *C. jejuni* model profile

The frozen manuscript benchmark used an XGBoost ensemble with the following settings:

- 50 independently fitted models
- seeds: **25–74**
- `n_estimators = 300`
- `max_depth = 6`
- `learning_rate = 0.05`
- `subsample = 0.85`
- `colsample_bytree = 0.85`
- training-locus missingness threshold: **0.20**
- training-only most-frequent imputation
- stratified class bootstrap for each ensemble member
- ensemble probability = mean class probability across models
- human assignments filtered at maximum class probability **0.60** for descriptive attribution summaries

For benchmark performance, class predictions are based on the maximum-probability class so that balanced accuracy corresponds to the mean of class recalls. The 0.60 threshold is retained separately for filtered human-attribution summaries.

The software environment used for the production run was:

- Python 3.8
- NumPy 1.24.4
- pandas 2.0.3
- scikit-learn 1.3.2
- XGBoost 2.1.4

## Frozen benchmark results

| Reference strategy | Balanced accuracy | Macro F1 | Poultry recall | Ruminant recall |
|---|---:|---:|---:|---:|
| Non-Peru global | 0.754 | 0.816 | 0.997 | 0.510 |
| Peru-local OOF | 0.875 | 0.905 | 0.991 | 0.758 |
| Global + Peru-local OOF | **0.893** | **0.922** | 0.993 | **0.792** |

Confusion counts:

- global: poultry 1,073/1,076 correct; ruminant 76/149 correct
- Peru-local: poultry 1,066/1,076 correct; ruminant 113/149 correct
- combined: poultry 1,069/1,076 correct; ruminant 118/149 correct

Thus adding Peru-local source genomes reduces ruminant errors from **73 to 31** while retaining almost complete poultry recognition.

## Size-matched geographic control

To test whether the Peru-local advantage is simply due to training-set size or class balance, global and Peru-local panels were repeatedly downsampled to equal numbers of poultry and ruminant genomes.

Across 5 folds × 10 repeats:

- mean global balanced accuracy: **0.751**
- mean Peru-local balanced accuracy: **0.913**
- mean local advantage: **+0.163**
- local advantage was positive in every fold

Equal-per-source learning curves also favoured Peru-local training at every tested size:

| Genomes/source | Global BA | Peru-local BA |
|---:|---:|---:|
| 10 | 0.616 | 0.743 |
| 25 | 0.707 | 0.842 |
| 50 | 0.714 | 0.864 |
| 75 | 0.738 | 0.883 |
| 100 | 0.727 | 0.906 |

This supports a **representation/geography effect**, not merely a larger-training-set effect.

## Human *C. jejuni* attribution

Filtered predictions for 981 human isolates:

### Global-only

- poultry: 915 (93.27%)
- ruminant: 7 (0.71%)
- uncertain: 59 (6.01%)

### Peru-contextual

- poultry: 924 (94.19%)
- ruminant: 40 (4.08%)
- uncertain: 17 (1.73%)

### Global + Peru

- poultry: 915 (93.27%)
- ruminant: 29 (2.96%)
- pig: 2 (0.20%)
- uncertain: 35 (3.57%)

Across all strategies, more than 93% of human isolates are assigned to poultry-associated genomic populations.

Interpretation: this is evidence of genomic similarity to represented source populations; it is **not** proof of a direct individual transmission event or proof that a specific food item caused infection.

## Independent genomic checks

Nearest-source cgMLST analysis gives:

- 925/981 human isolates (94.29%) nearest to poultry
- 56/981 nearest to ruminant

Balanced poultry/ruminant sensitivity analysis (200 repeats, 149 genomes/source):

- mean accuracy / balanced accuracy / macro F1 ≈ **0.939**
- modal human classifications: 743 poultry, 110 ruminant, 128 uncertain

Lineage-aware grouped cross-validation remains more difficult than isolate-random validation, as expected:

- cgMLST-clonal-complex grouped CV: BA 0.674; poultry recall 0.998; ruminant recall 0.349
- MLST-clonal-complex grouped CV: BA 0.620; poultry recall 0.998; ruminant recall 0.242

These analyses are sensitivity checks rather than replacements for the fixed production geography benchmark.

## Lineage-dependent rescue

The benefit of local genomes is heterogeneous because local source genomes help only when they add source-informative representation of the relevant lineage.

Two focal examples from the frozen fold design:

- **Fold 1:** nine ST-179-complex ruminant test isolates. Out-of-fold same-LIN10 local support is source-mixed (1 ruminant + 1 poultry). Fold-level ruminant recall improves from 0.267 (global) to 0.500 (combined).
- **Fold 5:** fourteen ST-48-complex ruminant test isolates. Out-of-fold same-LIN10 support is source-consistent (4 ruminant + 0 poultry). Fold-level ruminant recall improves from 0.103 (global) to 0.724 (combined).

These are **fold-level recalls**, not lineage-specific recall estimates.

## *C. jejuni* phylogeny

The population phylogeny is complete.

Pipeline:

1. Prokka annotation of 2,217 genomes
2. PIRATE pangenome analysis
3. extraction of the PIRATE core alignment
4. per-taxon alignment QC
5. exclusion of taxa with >50% missing/ambiguous sequence
6. IQ-TREE maximum-likelihood phylogeny

Final alignment / tree facts:

- starting genomes: **2,217**
- PIRATE core alignment: **1,405,341 bp**
- taxa excluded for >50% missing/ambiguous alignment: **23**
- final tree taxa: **2,194**

The 23 excluded IDs are documented in the analysis output and should be preserved alongside the tree. IQ-TREE may collapse exact duplicate sequences internally during optimisation and reattach them to the final tree; the manuscript/tree-tip count remains 2,194.

## Reproducing the study

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for a full start-to-finish protocol, including:

- data partitioning and leakage audit;
- fold construction;
- production ensemble settings;
- global/local/combined out-of-fold benchmark;
- size-matched controls;
- human prediction;
- nearest-source validation;
- PIRATE/IQ-TREE phylogeny;
- required output checks;
- the proposed *C. coli* extension.

## Relationship to SourceRunnerML

SourceRunnerML is the general software framework. This folder records one frozen, manuscript-specific analysis profile used to evaluate **geographic context in source attribution**.

The general SourceRunnerML wrappers may evolve. For paper-level reproduction, preserve:

- the exact input manifests;
- the exact fixed fold assignments;
- the cleaned non-Peru reference panel;
- model seeds and hyperparameters;
- software versions;
- raw per-isolate probabilities;
- ensemble metadata;
- phylogeny QC/exclusion lists.

That separation allows SourceRunnerML to remain reusable software while this directory serves as a transparent provenance record for the Peru analysis.
