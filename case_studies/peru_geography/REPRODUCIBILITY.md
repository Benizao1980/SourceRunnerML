# Reproducibility guide

This document records the computational design used for the Peru Campylobacter geographic-context analysis. It is intended both as a paper audit trail and as a guide for repeating the analysis on another species or geography.

## 1. Data model

Each SourceRunner input table should contain:

- a stable isolate identifier (`id`)
- source label
- country/geography
- species
- lineage fields where available (`ST`, clonal complex, cgST, LIN code)
- CAMP cgMLST allele columns

Human prediction rows must be held out from source-labelled training.

### Source label normalization

Study-specific raw labels (e.g. chicken, cattle, sheep, pig) are collapsed into the model source classes. For the primary C. jejuni benchmark:

- chicken/poultry → `Poultry`
- cattle/sheep/ruminant labels → `Ruminant`
- pig → `Pig`

The formal performance benchmark uses Poultry and Ruminant only.

## 2. Leakage audit

Before any geographic comparison, remove geographic leakage from the global source panel.

For C. jejuni, 127 Peru rows were present in the original global reference population. All were removed.

A robust implementation should remove global rows by both:

1. `country == Peru` where reliable geography exists; and
2. exact isolate-ID overlap with the Peru source panel.

Keep an audit file reporting counts removed by each rule.

## 3. Shared feature construction

The production C. jejuni analysis used 1,142 shared CAMP cgMLST loci.

Rules:

1. identify CAMP locus columns shared by train and prediction/test tables
2. coerce allele calls to numeric values
3. malformed/multi-allelic/non-numeric calls become missing
4. remove loci exceeding the training missingness threshold (20%)
5. fit most-frequent imputation on training data only
6. apply the fitted imputer to held-out/test/human rows

Never learn imputation values from the held-out fold or human prediction set.

## 4. Frozen lineage-blocked folds

The formal Peru benchmark contains:

```text
1,076 Poultry
149 Ruminant
1,225 total
```

Use a single fixed 5-fold `StratifiedGroupKFold` assignment and reuse it for every reference-population condition.

Preferred group:

```text
LINcode[17]
```

Fallback: exact cgMLST profile when an isolate lacks the preferred lineage grouping.

Required checks:

- every isolate appears in exactly one test fold
- both source classes are represented in every fold
- no grouping unit crosses train/test
- the identical fold manifest is reused for Global, Local and Combined analyses

This prevents the geographic comparison from being confounded by different test sets.

## 5. Reference-population conditions

For each fold, evaluate the same held-out Peru known-source isolates against:

### Global-only

Training = cleaned non-Peru global source population.

### Peru-local OOF

Training = Peru known-source isolates not in the test fold.

### Global+Peru OOF

Training = cleaned non-Peru global source population + Peru training folds.

The test fold is never included in local or combined training.

## 6. Production classifier settings

The production C. jejuni run used an XGBoost bootstrap ensemble with:

```text
n_estimators        300
max_depth           6
learning_rate       0.05
subsample           0.85
colsample_bytree    0.85
missingness cutoff  0.20
bootstrap models    50
seeds               25..74
```

Each ensemble model uses source-stratified bootstrap sampling. Prediction probabilities are averaged across the 50 fitted models.

For human attribution, a maximum mean class probability `<0.60` is labelled `Uncertain` in the filtered output.

For the formal known-source benchmark, performance metrics use the maximum-probability class **without discarding low-confidence samples**. Confidence-threshold outputs are retained separately as coverage/uncertainty diagnostics.

## 7. Primary performance metrics

Report at minimum:

- accuracy
- balanced accuracy
- macro F1
- Poultry recall
- Ruminant recall
- confusion matrix
- mean/median maximum class probability
- fraction below the 0.60 confidence threshold
- ensemble consensus / probability margin / entropy where available

Primary C. jejuni production result:

```text
Reference                 BA       MacroF1   P-recall   R-recall
Non-Peru global          0.754     0.816      0.997      0.510
Peru-local OOF           0.875     0.905      0.991      0.758
Global + Peru OOF        0.893     0.922      0.993      0.792
```

## 8. Size-matched controls

A larger reference population can outperform a smaller one simply because it has more training observations. The geography claim therefore requires equal-size controls.

Repeat training after downsampling Global and Peru panels to equal numbers per source. Use multiple repeats for each fold and sample size.

The completed C. jejuni learning-curve design tested:

```text
10, 25, 50, 75, 100 genomes/source
```

Peru-local balanced accuracy was higher at every tested size.

Replicate-level p-values, if calculated, should be described as descriptive rather than treated as independent biological replicates.

## 9. Lineage-support audit

To explain heterogeneous fold behavior, quantify the source composition of local training genomes related to each held-out ruminant isolate.

A useful summary is same-LIN10 local support:

- ruminant-only
- mixed ruminant+poultry
- poultry-only
- none

Completed C. jejuni fold counts:

| Fold | R-only | Mixed | P-only | None | n ruminants |
|---:|---:|---:|---:|---:|---:|
| 1 | 6 | 16 | 3 | 5 | 30 |
| 2 | 24 | 4 | 1 | 1 | 30 |
| 3 | 21 | 6 | 2 | 1 | 30 |
| 4 | 21 | 3 | 1 | 5 | 30 |
| 5 | 18 | 8 | 0 | 3 | 29 |

Use focal lineages to illustrate mechanism, not to claim lineage-specific recall unless performance was explicitly computed within the lineage.

## 10. Human predictions

Once the benchmark design is frozen, fit each reference-population design to the relevant complete training set and predict the 981 Peru human C. jejuni isolates.

Retain:

- raw predicted class
- filtered predicted class (0.60 threshold)
- probability for every source
- max probability
- margin between first and second source
- entropy
- bootstrap consensus
- ST/CC/cgST/LIN metadata

The principal result is the stability of the poultry-associated signal across reference-population designs, not a single preferred human classifier.

## 11. Independent nearest-source validation

Compute cgMLST distance without imputation:

```text
distance = differing alleles / comparable loci
```

For every human isolate identify the nearest genome from each source and the nearest source overall. Require a sensible minimum number of comparable loci.

This analysis answers a different question from the classifier and therefore provides useful independent support.

## 12. C. jejuni phylogeny

The tested HPC workflow is under `scripts/cjejuni_tree/`.

### Initial workflow

```bash
./submit_peru2217_pirate_iqtree_v5.sh
```

Stages:

1. validate an explicit 2,217-genome manifest
2. batched Prokka arrays
3. collect one GFF per expected isolate
4. PIRATE pangenome analysis
5. copy the PIRATE core alignment
6. IQ-TREE 2 ModelFinder + UFBoot + SH-aLRT

IQ-TREE command:

```bash
iqtree2 \
  -s CORE_ALIGNMENT.fasta \
  -m MFP \
  -B 1000 \
  -alrt 1000 \
  -T 32 \
  -safe \
  -seed 42 \
  --prefix PREFIX
```

### Alignment QC recovery

The initial core alignment contained 23 taxa with >50% missing/ambiguous sequence; five were effectively 100% missing and caused IQ-TREE input QC failure.

Do not rerun PIRATE. Filter the completed alignment:

```bash
./resubmit_cjejuni_iqtree_qc50_v2.sh
```

This submits a streaming QC job and an IQ-TREE dependency. The streaming design is important: the complete alignment is too large to load into memory on the login node.

Final population:

```text
2,217 initial genomes
23 excluded at >50% missing/ambiguous
2,194 final tree tips
1,405,341 alignment positions
```

### IQ-TREE memory recovery

ModelFinder exceeded a 160 GB allocation. The final stage was resubmitted at 240 GB:

```bash
sbatch run_cjejuni_iqtree_qc50_240G.sh
```

Only the IQ-TREE stage needs rerunning after an OOM failure; the filtered alignment and PIRATE output remain valid.

## 13. C. coli workflow

The complete prepared C. coli bundle is under:

```text
scripts/ccoli/
```

It contains:

- input inventory/preflight
- leakage cleaning
- fold construction
- production geography workers/aggregation
- human prediction workers/aggregation
- nearest-source cgMLST validation
- tree-ID construction
- Prokka → PIRATE → IQ-TREE launcher

The first command after filling `config_coli.sh` is deliberately:

```bash
./00_inventory_and_preflight.sh
```

Do not submit models before reviewing the source counts and shared-locus coverage.

## 14. Data-release policy

This repository should contain:

- code
- parameter settings
- fold-design logic
- non-sensitive aggregate results
- synthetic/example input formats

Do not commit restricted clinical metadata, identifiable participant data, raw private genomes, or very large derived files. Public isolate IDs can be released when consistent with the underlying data-source permissions and manuscript data-access plan.
