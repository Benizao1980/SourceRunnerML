# Status and next steps

Last updated: September 2026

## Where the project currently stands

### Manuscript story

The *C. jejuni* analysis now supports a clear central result:

> Human Peruvian *C. jejuni* isolates are overwhelmingly poultry-associated across attribution strategies, but known-source benchmark performance is strongly affected by geographic composition of the reference population. Peru-local genomes improve recognition of locally circulating ruminant-associated diversity, while the combined global-plus-Peru panel performs best overall.

This is deliberately not framed as "local beats global". Global breadth and local ecological/genetic relevance are complementary.

### Production *C. jejuni* geography benchmark — complete

Frozen benchmark population:

- Peru poultry: 1,076
- Peru ruminant: 149
- formal P/R benchmark: 1,225 isolates
- 5 fixed lineage-blocked folds
- 50-model XGBoost ensemble per production comparison
- clean non-Peru global source panel after removal of 127 Peru rows

Frozen benchmark balanced accuracy:

- Global: 0.754
- Peru-local OOF: 0.875
- Global + Peru OOF: 0.893

Main error correction:

- ruminant errors fall from 73/149 under global-only training to 31/149 under combined training
- poultry recognition remains essentially unchanged

### Training-size control — complete

Repeated size-matched benchmarking shows the Peru-local advantage is not explained simply by panel size.

- global mean BA: 0.751
- Peru-local mean BA: 0.913
- mean difference: +0.163

Equal-n/source learning curves favour Peru-local training at every tested training size from 10 to 100 genomes/source.

### Human *C. jejuni* attribution — complete

Human prediction set: 981 isolates.

More than 93% are classified as poultry-associated under each principal reference strategy.

Combined global + Peru filtered predictions:

- poultry 915
- ruminant 29
- pig 2
- uncertain 35

### Independent cgMLST validation — complete

Nearest-source comparison:

- 925/981 (94.29%) nearest poultry
- 56/981 nearest ruminant

This independently supports the dominant poultry-associated signal.

### Lineage mechanism analysis — complete

Fold-level analyses show that geographic rescue is lineage-dependent.

Local genomes are most useful when the non-held-out local reference set contains source-informative representatives of the same locally circulating genomic lineage.

Focal manuscript examples:

- ST-179 complex: sparse/source-mixed local representation; weaker rescue
- ST-48 complex: source-consistent local representation; strong rescue

### *C. jejuni* pangenome/core phylogeny — complete

- 2,217 genomes entered Prokka/PIRATE
- PIRATE core alignment: 1,405,341 bp
- 23 taxa excluded at >50% missing/ambiguous core alignment
- final IQ-TREE population: 2,194 taxa
- IQ-TREE required a high-memory rerun because ModelFinder estimated ~207 GB RAM; the successful job used 240 GB

The tree, QC list, alignment and IQ-TREE outputs should be retained as manuscript provenance.

### Manuscript status

The Results/Discussion have already been restructured around:

1. study/reference-population context;
2. robust poultry association among human *C. jejuni*;
3. improvement from geographic context in known-source classification;
4. independent genomic/sensitivity validation;
5. lineage-concentrated minority/non-poultry predictions;
6. *C. coli* as the cross-species validation/generalisation analysis.

The principal remaining biological analysis is *C. coli*.

---

# What should happen next

## Priority 1 — reconstruct the *C. coli* input matrices

Do not start by fitting models.

First locate the original species-mixed pubMLST/cgMLST exports and determine whether the required *C. coli* matrices can be reconstructed from data already held locally/Box/HPC.

We need three tables:

```text
Ccoli_global_source.tsv
Ccoli_peru_source.tsv
Ccoli_peru_human.tsv
```

These are desired output names, not assumed existing filenames.

Each should contain full *C. coli* cgMLST allele profiles plus stable metadata.

## Priority 2 — *C. coli* inventory and leakage audit

Before deciding the model design, report:

- number of human *C. coli* isolates
- number of global source-labelled *C. coli*
- number of Peru source-labelled *C. coli*
- counts by poultry / ruminant / pig / other candidate source
- exact Peru overlap in global reference data
- shared cgMLST locus count
- missingness
- ST and CC composition
- ST-828-complex counts by source/geography
- ST-1150-complex counts by source/geography

The decision point is whether local P/R source counts are sufficient for a defensible formal benchmark.

## Priority 3 — choose one of two *C. coli* analysis paths

### Path A: sufficient Peru-local source data

Reproduce the primary *C. jejuni* design:

- clean non-Peru global
- Peru-local OOF
- combined OOF
- lineage-blocked fivefold benchmark
- XGBoost ensemble
- P/R balanced accuracy and class recalls
- human predictions
- nearest-source validation
- lineage stratification

### Path B: insufficient Peru-local source data

Do not force a formal benchmark.

Instead use *C. coli* as a secondary generalisation test:

- human attribution using the best-supported source panels
- ST-828 vs ST-1150 source-probability profiles
- uncertainty comparison
- nearest-source cgMLST
- core-genome phylogeny
- qualitative/quantitative comparison with *C. jejuni*

This is scientifically preferable to reporting unstable class metrics from tiny source groups.

## Priority 4 — build the *C. coli* phylogeny in parallel

Once the target IDs and assemblies are reconciled:

1. Prokka
2. PIRATE
3. core-alignment taxon QC
4. IQ-TREE

Use the successful *C. jejuni* workflow as the template, but size resources to the *C. coli* collection.

Particular attention:

- ST-828 complex
- ST-1150 complex
- source-associated clustering
- whether human ST-1150 isolates occupy a distinct source/ecological position

## Priority 5 — finish Figure 4 and the manuscript placeholders

Figure 4 should ultimately answer a single question:

> Does the geography/source-attribution result observed in *C. jejuni* generalise to *C. coli*, or is reservoir association species/lineage dependent?

Suggested panels:

- A: human *C. coli* attribution across reference strategies
- B: benchmark/uncertainty performance, if defensible
- C: ST-828 vs ST-1150 attribution profiles
- D: predicted source × lineage × clinical category
- E: nearest-source or phylogenetic validation

If formal *C. coli* benchmarking is not defensible, redesign panel B rather than forcing symmetry with *C. jejuni*.

---

# Repository/reproducibility tasks

This worked-study folder should remain linked to SourceRunnerML rather than replacing the software documentation.

Before manuscript submission, add or archive:

- fixed *C. jejuni* fold manifest
- clean non-Peru reference manifest
- model metadata/seeds
- retained-locus manifest
- per-isolate production probabilities, where redistributable
- size-matched benchmark summary
- learning-curve summary
- phylogeny QC list
- exact IQ-TREE model/run report
- *C. coli* equivalent manifests when complete

Large genomic datasets should not be committed directly if redistribution is restricted. Prefer stable public IDs plus reconstruction scripts/manifests.

---

# Immediate next action

The next computational task is therefore:

> **Find/reconstruct the full *C. coli* cgMLST source and human matrices, run a source-count/leakage inventory, and only then choose the formal benchmark design.**
