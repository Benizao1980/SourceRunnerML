# Reproducibility protocol: Peru geographic-context source attribution

This document records the analytical design and practical steps required to reproduce the Peru *Campylobacter* source-attribution study.

The aim is to make the manuscript analysis reproducible without conflating it with every option in the general SourceRunnerML software. The production *C. jejuni* benchmark is frozen; the *C. coli* extension is the next analysis and should follow the same principles only where the data support them.

---

## 1. Data required

Prepare three species-specific tables for each organism:

1. **Global source-labelled reference table**
2. **Peru source-labelled reference table**
3. **Peru human prediction table**

Each table should contain, where available:

- stable isolate ID
- isolate/sample name
- country
- year
- raw source label
- reduced source class
- MLST ST
- MLST clonal complex
- cgST
- LIN code / LINcode[17]
- cgMLST loci (for the current Campylobacter scheme, columns prefixed `CAMP`)

The global and Peru source tables must contain only non-human, source-labelled isolates used for training/evaluation. Human isolates are prediction targets and must never leak into the source-labelled training set.

### Required source harmonisation

Map raw labels to the study-level classes used in the model. For the primary *C. jejuni* analysis:

- chicken / poultry -> `Poultry`
- cattle / sheep / goat / ruminant -> `Ruminant`
- pig / swine -> `Pig`
- wild bird labels -> `Wild bird` where used in sensitivity analyses

Keep the original raw label in a separate column.

---

## 2. Species separation

Do not mix *C. jejuni* and *C. coli* in one model.

For each species:

- subset the source-labelled global data;
- subset the Peru source-labelled data;
- subset the Peru human data;
- identify the cgMLST loci shared between source and prediction tables;
- run the entire pipeline independently.

The two species differ in population structure and may differ in the source classes that can be discriminated reliably.

---

## 3. Leakage audit

This is mandatory before any geography comparison.

The global panel may contain Peru isolates. A global-vs-local comparison is invalid if the supposedly global panel already contains local Peruvian source genomes.

### *C. jejuni* production audit

Original global panel:

- total: 17,372
- poultry: 11,142
- ruminant: 6,126
- pig: 104

Audit result:

- 127 Peru isolates were present in the original global panel
- all 127 were removed

Clean non-Peru global panel:

- total: 17,245
- poultry: 11,015
- ruminant: 6,126
- pig: 104

The audit should be based on both:

1. exact stable isolate ID overlap; and
2. geography metadata (`country == Peru`) where present.

Record every removed ID in a file such as:

```text
leakage_audit/peru_ids_removed_from_global.txt
```

Also record counts before and after removal.

### Important

If the same genome is represented under multiple aliases, ID matching alone may miss leakage. Where possible, supplement ID matching with cgMLST/profile or assembly-level deduplication checks.

---

## 4. cgMLST feature preparation

For the production *C. jejuni* workflow:

1. identify `CAMP` loci shared between training and target tables;
2. coerce allele calls to numeric values;
3. treat malformed, multi-allelic or missing calls as missing;
4. remove loci with >20% missingness in the **training data**;
5. fit the imputer on training data only;
6. use most-frequent allele imputation per retained locus;
7. apply the fitted imputer to validation/test/human matrices.

Do not fit missing-value handling on the test or human prediction set.

The final production *C. jejuni* attribution analyses used **1,142 shared cgMLST loci**.

Because loci retained can vary if input tables differ, always write the retained-locus list to disk.

Suggested output:

```text
manifests/retained_cgmlst_loci.txt
```

---

## 5. Frozen known-source benchmark

### Benchmark population

Peru known-source *C. jejuni*:

- poultry: 1,076
- ruminant: 149
- pig: 11

The formal benchmark is poultry vs ruminant only:

- total benchmark isolates: 1,225

Pig is excluded from the formal benchmark because 11 local pig genomes do not provide a stable class-performance estimate.

### Why out-of-fold local references are required

A Peru isolate being evaluated must not also appear in the Peru-local training set.

The correct comparison is therefore:

- **Global:** train on cleaned non-Peru global panel; test each Peru benchmark isolate in its fixed fold.
- **Peru-local OOF:** train on Peru source isolates outside the held-out fold; test the held-out Peru fold.
- **Combined OOF:** train on cleaned global + Peru source isolates outside the held-out fold; test the held-out Peru fold.

All three strategies must use the same held-out isolates.

---

## 6. Fixed fold construction

Use a fixed five-fold design and preserve the manifest in version-controlled provenance.

Production design:

- 5 folds
- `StratifiedGroupKFold`
- grouping by a fine genomic group derived from `LINcode[17]`
- exact cgMLST-profile fallback when LIN17 is unavailable
- random seed: 42

Requirements:

- no fine genomic group may occur in both train and test within a fold;
- every benchmark isolate is tested exactly once;
- both poultry and ruminant must be represented in each held-out fold;
- the fold assignments are identical for Global / Peru-local / Combined.

Write a manifest with at least:

```text
id
source_class
LINcode17
fine_cluster
broad_lineage
fold
```

The fold manifest is part of the frozen analysis and should not be regenerated casually once manuscript results are derived from it.

---

## 7. Production XGBoost ensemble

The production benchmark uses 50 independently trained XGBoost models.

Frozen parameters:

```python
n_estimators = 300
max_depth = 6
learning_rate = 0.05
subsample = 0.85
colsample_bytree = 0.85
```

Replicate seeds:

```text
25, 26, ..., 74
```

Each ensemble member:

1. receives the fold-specific training matrix;
2. applies the training-locus missingness filter;
3. fits a training-only most-frequent imputer;
4. draws a stratified class bootstrap from the training set;
5. fits XGBoost to that bootstrap sample;
6. predicts class probabilities for the held-out isolates.

For each held-out isolate, average class probabilities across the 50 models.

For benchmark metrics use:

```text
predicted_source = argmax(mean_probability)
```

Do not apply the 0.60 human-attribution uncertainty threshold when calculating the core benchmark balanced accuracy. Otherwise uncertain calls alter the definition of class recall.

For human descriptive attribution summaries, retain both:

- raw maximum-probability class; and
- filtered class, where max probability <0.60 is labelled `Uncertain`.

---

## 8. Benchmark aggregation

Concatenate out-of-fold predictions so every Peru benchmark isolate contributes exactly once per strategy.

Report at minimum:

- balanced accuracy
- macro F1
- poultry recall
- ruminant recall
- confusion matrix
- fold-level metrics

Frozen production results:

| Reference | Balanced accuracy | Macro F1 | Poultry recall | Ruminant recall |
|---|---:|---:|---:|---:|
| Non-Peru global | 0.754 | 0.816 | 0.997 | 0.510 |
| Peru-local OOF | 0.875 | 0.905 | 0.991 | 0.758 |
| Global + Peru-local OOF | 0.893 | 0.922 | 0.993 | 0.792 |

Confusion matrices:

### Global

| True | Pred poultry | Pred ruminant |
|---|---:|---:|
| Poultry | 1,073 | 3 |
| Ruminant | 73 | 76 |

### Peru-local

| True | Pred poultry | Pred ruminant |
|---|---:|---:|
| Poultry | 1,066 | 10 |
| Ruminant | 36 | 113 |

### Combined

| True | Pred poultry | Pred ruminant |
|---|---:|---:|
| Poultry | 1,069 | 7 |
| Ruminant | 31 | 118 |

The central effect is improved ruminant recognition with almost no loss of poultry recognition.

---

## 9. Fold-level lineage analysis

Ruminant recall varies strongly by fold because local reference benefit depends on whether source-informative representatives of the held-out lineage exist in the remaining Peru training set.

Ruminant recall by fold (computational fold labels 0-4):

| Fold | Global | Peru-local | Combined |
|---:|---:|---:|---:|
| 0 | 0.267 | 0.367 | 0.500 |
| 1 | 0.733 | 0.933 | 0.933 |
| 2 | 0.633 | 0.867 | 0.867 |
| 3 | 0.800 | 0.900 | 0.933 |
| 4 | 0.103 | 0.724 | 0.724 |

When presenting manuscript folds as 1-5, add one to the computational labels.

Two focal examples:

- manuscript Fold 1 / computational fold 0: nine ST-179-complex ruminants; same-LIN10 OOF support is 1 ruminant + 1 poultry; sparse/mixed local representation.
- manuscript Fold 5 / computational fold 4: fourteen ST-48-complex ruminants; same-LIN10 OOF support is 4 ruminant + 0 poultry; source-consistent local representation.

Do not describe the fold-level recall values as ST-specific recall.

---

## 10. Size-matched geographic control

The main benchmark compares panels with very different sample sizes, so a separate control tests whether the local advantage survives when panel size is held equal.

Recommended design:

- use the same five held-out folds;
- for each fold, repeatedly downsample the available training data to equal source-class sizes;
- match global and Peru-local training counts;
- run 10 random repeats per fold;
- calculate balanced accuracy for every repeat.

Frozen result:

- global mean BA: 0.751
- Peru-local mean BA: 0.913
- mean difference: +0.163
- local advantage positive in every fold

Replicate-level p-values, if shown, are descriptive; the important result is the consistency and effect magnitude.

---

## 11. Equal-n learning curves

A second size control trains each geography at fixed equal numbers of genomes per source.

Frozen values:

| n/source | Global | Peru-local |
|---:|---:|---:|
| 10 | 0.616 | 0.743 |
| 25 | 0.707 | 0.842 |
| 50 | 0.714 | 0.864 |
| 75 | 0.738 | 0.883 |
| 100 | 0.727 | 0.906 |

The geographic-context advantage is therefore visible across the tested training sizes.

---

## 12. Human prediction

Once the benchmark design is frozen, fit source models and predict the 981 human Peru *C. jejuni* isolates.

Use the same production feature handling and XGBoost ensemble profile.

Retain per-isolate:

- mean probability for each source
- maximum probability
- top source
- second source
- probability margin
- entropy if calculated
- filtered source at 0.60
- ST / CC / cgST / LIN code / clinical metadata where available

Frozen summaries:

### Global-only

- 915 poultry
- 7 ruminant
- 59 uncertain

### Peru-contextual

- 924 poultry
- 40 ruminant
- 17 uncertain

### Combined

- 915 poultry
- 29 ruminant
- 2 pig
- 35 uncertain

This robustly supports a dominant poultry-associated population signal while also showing that minority assignments depend on reference composition.

---

## 13. Nearest-source cgMLST validation

Classifier output should be checked against direct genomic proximity.

For each human isolate:

1. calculate allele mismatches to every source-labelled reference using only loci comparable for that pair;
2. divide mismatches by number of comparable loci;
3. identify the nearest source genome in each source class;
4. identify the nearest source overall;
5. retain both raw mismatch count and proportional distance.

Frozen result:

- nearest poultry: 925/981 (94.29%)
- nearest ruminant: 56/981

This provides an independent check that the main poultry signal is not purely a classifier artefact.

---

## 14. Additional sensitivity analyses

Useful secondary checks include:

### Balanced training

200 repeats, 149 genomes/source:

- mean accuracy / BA / macro F1 ≈ 0.939
- modal human predictions: 743 poultry, 110 ruminant, 128 uncertain

### Grouped cross-validation

cgMLST-CC grouping:

- accuracy 0.919
- BA 0.674
- macro F1 0.734
- poultry recall 0.998
- ruminant recall 0.349

MLST-CC grouping:

- accuracy 0.906
- BA 0.620
- macro F1 0.667

These deliberately harsher grouped tests are useful for assessing extrapolation to underrepresented lineages.

---

## 15. Core-genome phylogeny

The phylogeny provides population-genomic context for Figure 1.

### Step 15.1: annotation

Annotate all target assemblies using Prokka.

Example:

```bash
prokka \
  --outdir prokka/ISOLATE_ID \
  --prefix ISOLATE_ID \
  --genus Campylobacter \
  --species jejuni \
  assembly.fasta
```

### Step 15.2: PIRATE

The production run used PIRATE on all 2,217 annotated genomes.

Representative command profile:

```bash
PIRATE \
  -i gffs/ \
  -o pirate_out/ \
  -s '50,60,70,80,90,95,98' \
  --pan-opt '--diamond' \
  -a \
  -z 1
```

Record software/environment version and the exact command in the run log.

Final core alignment:

- 2,217 taxa before QC
- 1,405,341 nucleotide positions

### Step 15.3: per-taxon alignment QC

For each taxon, calculate the fraction of the core alignment that is not unambiguous A/C/G/T.

Exclude taxa with:

```text
missing_or_ambiguous_fraction > 0.50
```

Production outcome:

- 23 excluded
- 2,194 retained

Always write:

```text
core_alignment_sequence_qc.tsv
core_alignment_excluded_qc50.txt
core_alignment_retained_qc50.txt
```

### Step 15.4: IQ-TREE

Production IQ-TREE profile:

```bash
iqtree2 \
  -s Peru_Cjejuni_PIRATE_core_alignment_qc50.fasta \
  -m MFP \
  -B 1000 \
  -alrt 1000 \
  -T 32 \
  -safe \
  -seed 42 \
  --prefix Peru_Cjejuni_PIRATE_core_IQTree_qc50
```

For this unusually large alignment, ModelFinder required approximately 207 GB RAM according to IQ-TREE and a 160 GB job was killed by the scheduler. The successful rerun therefore requested 240 GB RAM.

Preserve:

- `.treefile`
- `.contree` if produced
- `.iqtree`
- `.log`
- model/checkpoint files where useful
- filtered alignment
- excluded/retained lists

Do not interpret IQ-TREE's temporary removal of exact duplicate sequences during optimisation as a reduction in the final study population. Duplicate tips are restored to the output tree; the final tree population is 2,194 isolates.

---

## 16. Expected project directory structure

A practical working layout is:

```text
SourceAttribution_Peru/
├── data_inputs/
├── fixed_holdout_design/
├── primary_geography_benchmark/
│   └── production_sourcerunner_ensemble/
├── figures/
│   └── Figure1/
│       └── Peru2217_PIRATE_IQTree/
├── existing_results/
├── scripts_reusable/
└── Ccoli_verification/
```

Never treat HPC paths as public data locations. They are provenance hints only; public release should use redistributable manifests or reconstruction instructions.

---

## 17. Output validation checklist

Before freezing a result, verify all of the following:

- [ ] species-specific input tables
- [ ] source labels harmonised
- [ ] all Peru isolates removed from global panel
- [ ] retained cgMLST loci recorded
- [ ] fixed fold manifest saved
- [ ] no genomic group crosses train/test within a fold
- [ ] each benchmark isolate predicted exactly once per strategy
- [ ] same test folds used by Global / Peru / Combined
- [ ] raw per-class probabilities retained
- [ ] benchmark metrics use argmax class, not uncertainty filtering
- [ ] human filtered predictions use documented confidence threshold
- [ ] size-matched control performed
- [ ] nearest-source analysis performed
- [ ] phylogeny input IDs reconcile with metadata
- [ ] alignment QC/exclusion list preserved
- [ ] software versions and seeds recorded

---

## 18. *C. coli* next-step protocol

The next analysis should begin with **inventory**, not modelling.

### Step A — locate/rebuild input matrices

We currently have evidence for substantial *C. coli* metadata, including ST-1150 and Peru poultry isolates, but the required full cgMLST source/human matrices must be located or reconstructed.

Search for the original combined Campylobacter pubMLST/cgMLST exports before redownloading data. The likely simplest route is to split an existing combined export by species.

Required outputs:

```text
Ccoli_global_source.tsv
Ccoli_peru_source.tsv
Ccoli_peru_human.tsv
```

These names are conventions, not assumptions that the files already exist.

### Step B — inventory counts

Report:

- total rows by table
- source counts
- country counts
- Peru overlaps between global and local
- number of shared cgMLST loci
- missingness distribution
- ST / CC counts
- ST-828-complex counts
- ST-1150-complex counts

### Step C — decide benchmark design

If Peru-local poultry and ruminant counts are adequate across multiple lineages, reproduce the *C. jejuni* binary P/R benchmark.

If not, do not force it. Use:

- human predictions under Global / Peru / Combined where trainable;
- lineage-resolved ST-828 vs ST-1150 analysis;
- nearest-source cgMLST;
- phylogenetic validation.

### Step D — phylogeny

Build a *C. coli* core-genome phylogeny using the same conceptual Prokka -> PIRATE -> alignment-QC -> IQ-TREE pipeline.

Particular attention should be paid to:

- ST-828 complex
- ST-1150 complex
- whether human isolates cluster with poultry-associated or other source-associated diversity
- whether source attribution differs systematically by major lineage

---

## 19. Interpretation guardrails

The output is source attribution, not transmission reconstruction.

Preferred language:

> human isolates are genetically most similar to / attributed to poultry-associated source populations represented in the reference collection.

Avoid unsupported causal language such as:

> these infections came directly from chicken meat.

Likewise, 'geographic context' should be understood as a proxy for locally circulating ecological-genetic structure. Geography itself is not a biological mechanism.

The central study result is therefore:

> global source collections provide breadth, geographically matched local genomes add locally relevant source-associated lineage representation, and combining the two gives the strongest overall attribution performance in the Peruvian *C. jejuni* benchmark.
