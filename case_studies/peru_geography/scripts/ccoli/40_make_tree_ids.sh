#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/config_coli.sh"
mkdir -p "$TREE_WORK"
args=(--human "$HUMAN_TSV" --peru-source "$PERU_SOURCE_TSV" --global-max "$GLOBAL_TREE_MAX" --out-ids "$TREE_WORK/coli_tree_target_ids.txt" --out-meta "$TREE_WORK/coli_tree_metadata.tsv" --id-col "$ID_COL" --source-col "$SOURCE_COL" --species-col "$SPECIES_COL" --species-value "$SPECIES_VALUE" --seed "$SEED")
if (( GLOBAL_TREE_MAX > 0 )); then args+=(--global-source "$GLOBAL_SOURCE_TSV"); fi
"$SRML_PYTHON" "$HERE/41_make_tree_ids.py" "${args[@]}"
