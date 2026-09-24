# C. coli geographic-context verification + Figure 4 tree bundle

This bundle is designed to mirror the successful C. jejuni analysis while keeping the C. coli verification deliberately focused.

## What it does

### Track A — geographic-context verification
1. Inventory/audit the C. coli input tables and contig coverage.
2. Remove Peru rows and exact Peru IDs from the global source panel.
3. Build a common known-source Peru benchmark.
4. Freeze lineage-blocked 5-fold cross-validation using LIN17 (preferred), then cgMLST_CC / MLST_CC as fallbacks.
5. Run the same three reference-population designs: Global-only non-Peru, Peru-local OOF, and Global + Peru-local OOF.
6. Aggregate balanced accuracy, macro-F1 and per-source recall.
7. Predict human C. coli under all three reference-population designs.
8. Run an independent nearest-source cgMLST analysis.

### Track B — Figure 4 pangenome/tree
1. Build an explicit tree ID list from human + Peru source C. coli, with optional global representatives.
2. Map IDs to contigs.
3. Prokka annotation using batched SLURM array tasks.
4. PIRATE pangenome + >=95% core alignment.
5. IQ-TREE 2 with ModelFinder, UFBoot and SH-aLRT.

## Important design choice

The default verification is **Poultry vs Ruminant**, matching the formal C. jejuni benchmark and avoiding unstable inference from tiny pig classes. After the inventory, edit `config_coli.sh` if C. coli has enough pig isolates to justify a three-source analysis.

## Run order

Edit `config_coli.sh`, especially `GLOBAL_SOURCE_TSV`, `PERU_SOURCE_TSV`, `HUMAN_TSV`, and `CONTIG_DIRS`.

Then **inventory first and stop**:

```bash
./00_inventory_and_preflight.sh
```

Inspect `work/inventory/input_summary.txt`, `source_counts.tsv`, `overlap_summary.tsv` and `contig_coverage.tsv`. Only if the known-source Peru class counts and CAMP cgMLST coverage are defensible should the benchmark continue.

Prepare the clean benchmark and fixed folds:

```bash
./01_prepare_benchmark.sh
```

Run the geography benchmark:

```bash
./10_submit_geography_benchmark.sh
```

Run human predictions:

```bash
./20_submit_human_predictions.sh
```

Run nearest-source cgMLST validation:

```bash
./30_submit_nearest_source.sh
```

Build the Figure 4 population and phylogeny:

```bash
./40_make_tree_ids.sh
./50_submit_coli_pirate_iqtree.sh
```

The production model settings mirror the C. jejuni run: XGBoost, 300 trees, max depth 6, learning rate 0.05, subsample/column-subsample 0.85, 20% training-locus missingness threshold, training-only most-frequent imputation, source-stratified bootstrap, 50 prediction models (seeds 25–74), and a 0.60 filtered-call threshold. Formal benchmark metrics use the unfiltered maximum-probability class.

## Current blocker

The script bundle is ready, but the three paths in `config_coli.sh` are placeholders until the canonical full-CAMP-cgMLST tables are located/rebuilt. Seven-locus MLST/ST/LIN-only exports are not sufficient for SourceRunnerML.

Do **not** run the modelling stages against guessed or partial input files. The first task is to locate the original combined pubMLST/full-profile export, split it into global-source, Peru-source and Peru-human C. coli tables, and run the inventory.

## What not to do yet

Do not add Gubbins, human-as-source models, case/carrier analyses, or extensive sensitivity analyses unless the primary C. coli result gives a reason. The purpose of this analysis is verification/generalisation, not a second full paper inside Figure 4.
