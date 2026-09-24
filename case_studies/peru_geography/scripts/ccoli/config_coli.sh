#!/usr/bin/env bash
# Central configuration for the C. coli verification and tree workflow.
# Edit paths here; the other scripts source this file.

set -euo pipefail

ACCOUNT="${ACCOUNT:-cooperma}"
PARTITION="${PARTITION:-standard}"
PROJECT="${PROJECT:-/xdisk/cooperma/bpascoe/SourceAttribution_Peru}"
ROOT="${ROOT:-$PROJECT/Ccoli_verification}"

# REQUIRED INPUT TABLES — placeholders until the canonical C. coli CAMP cgMLST
# exports are located/rebuilt.
GLOBAL_SOURCE_TSV="${GLOBAL_SOURCE_TSV:-$PROJECT/data_inputs/Ccoli_global_source.tsv}"
PERU_SOURCE_TSV="${PERU_SOURCE_TSV:-$PROJECT/data_inputs/Ccoli_peru_source.tsv}"
HUMAN_TSV="${HUMAN_TSV:-$PROJECT/data_inputs/Ccoli_peru_human.tsv}"
CONTIG_DIRS="${CONTIG_DIRS:-$PROJECT/data_inputs/00.contigs_peru}"

ID_COL="${ID_COL:-id}"
SOURCE_COL="${SOURCE_COL:-source}"
COUNTRY_COL="${COUNTRY_COL:-country}"
SPECIES_COL="${SPECIES_COL:-species}"
SPECIES_VALUE="${SPECIES_VALUE:-C. coli}"
ANALYSIS_SOURCES="${ANALYSIS_SOURCES:-Poultry,Ruminant}"
LOCI_PREFIX="${LOCI_PREFIX:-CAMP}"
LOCI_REGEX="${LOCI_REGEX:-}"
MISSINGNESS="${MISSINGNESS:-0.20}"
GROUP_COL="${GROUP_COL:-auto}"
N_FOLDS="${N_FOLDS:-5}"
SEED="${SEED:-42}"

SRML_PYTHON="${SRML_PYTHON:-/home/u12/bpascoe/miniconda3/envs/srml/bin/python}"
CONDA_SH="${CONDA_SH:-/home/u12/bpascoe/miniconda3/etc/profile.d/conda.sh}"
PROKKA_ENV="${PROKKA_ENV:-prokka}"
PIRATE_ENV="${PIRATE_ENV:-pirate}"
IQTREE_ENV="${IQTREE_ENV:-iqtree}"

WORK="${WORK:-$ROOT/work}"
INVENTORY_DIR="$WORK/inventory"
BENCH_INPUT_DIR="$WORK/benchmark_inputs"
GEO_WORK="$WORK/geography"
HUMAN_WORK="$WORK/human_predictions"
NEAREST_WORK="$WORK/nearest_source"
TREE_WORK="$WORK/tree"
mkdir -p "$INVENTORY_DIR" "$BENCH_INPUT_DIR" "$GEO_WORK" "$HUMAN_WORK" "$NEAREST_WORK" "$TREE_WORK"

MODEL_CPUS="${MODEL_CPUS:-8}"
MODEL_MEM="${MODEL_MEM:-40G}"
MODEL_TIME="${MODEL_TIME:-36:00:00}"
PROKKA_GENOMES_PER_TASK="${PROKKA_GENOMES_PER_TASK:-5}"
ANNOT_MAX_CONCURRENT="${ANNOT_MAX_CONCURRENT:-100}"
GLOBAL_TREE_MAX="${GLOBAL_TREE_MAX:-0}"
