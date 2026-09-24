# Next step: C. coli verification

The C. coli analysis should be deliberately narrower than the completed C. jejuni analysis. Its purpose is to test whether the geographic-context result generalises across Campylobacter species and whether the major C. coli lineages (especially ST-828 and ST-1150 complexes) show different reservoir structure.

## Decision rule

Do **not** force an identical formal benchmark if the Peru known-source C. coli sample sizes do not support it.

Two acceptable outcomes are:

### A. Adequate Peru source counts

If Poultry and Ruminant both have sufficient source-labelled Peru C. coli genomes:

1. repeat the frozen-lineage benchmark
2. Global-only non-Peru
3. Peru-local OOF
4. Global+Peru OOF
5. compare balanced accuracy, macro F1 and source-specific recall
6. run human predictions under all three panels
7. stratify results by ST-828 vs ST-1150 complex
8. run nearest-source cgMLST validation
9. build the Figure 4 C. coli phylogeny

Pig should be added to the formal benchmark only if the inventory shows a defensible Peru sample size.

### B. Peru source counts too small

If one or more key source classes are sparse, do not manufacture a formal benchmark. Use C. coli as a secondary validation/generalisation analysis:

1. human attribution under available global/local/combined panels
2. lineage-resolved ST-828/ST-1150 predictions
3. nearest-source cgMLST distances
4. phylogenetic placement against Peru source isolates
5. report the limited local reference sample size explicitly

A different C. coli result would be informative: it would demonstrate species/lineage dependence rather than invalidate the C. jejuni finding.

## Input discovery is the immediate task

The prepared scripts currently expect:

```text
GLOBAL_SOURCE_TSV=/.../Ccoli_global_source.tsv
PERU_SOURCE_TSV=/.../Ccoli_peru_source.tsv
HUMAN_TSV=/.../Ccoli_peru_human.tsv
```

These are placeholders. Before submitting anything, locate the full cgMLST source used for the original Campylobacter analyses.

Likely routes:

- original pubMLST full-profile export containing both C. jejuni and C. coli
- an existing combined training table used before species splitting
- SourceRunner-ready training/prediction tables elsewhere in the project/Box storage

The relevant table must contain many `CAMP...` cgMLST loci, not merely seven-locus MLST, ST, clonal complex and LIN code.

## Run order once inputs exist

From `scripts/ccoli/`:

```bash
nano config_coli.sh
./00_inventory_and_preflight.sh
```

**Stop after inventory and inspect counts.**

Then, if defensible:

```bash
./01_prepare_benchmark.sh
./10_submit_geography_benchmark.sh
./20_submit_human_predictions.sh
./30_submit_nearest_source.sh
./40_make_tree_ids.sh
./50_submit_coli_pirate_iqtree.sh
```

See `scripts/ccoli/README_RUN_ME.md` for full operational detail.

## Outputs needed for Figure 4

Preferred panels:

- **a**: human C. coli attribution under Global / Peru / Combined panels
- **b**: known-source CV performance if sample size permits
- **c**: ST-828 vs ST-1150 source-attribution profiles
- **d**: predicted source × lineage × clinical category (descriptive only)
- **e**: nearest-source cgMLST or phylogenetic validation

Do not add Gubbins, human-as-source models, extensive case/carrier tests or multiple secondary sensitivity analyses unless the primary C. coli result gives a specific reason.
