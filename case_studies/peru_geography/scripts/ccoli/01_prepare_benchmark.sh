#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/config_coli.sh"
"$SRML_PYTHON" "$HERE/01_prepare_coli_benchmark.py" --global-source "$GLOBAL_SOURCE_TSV" --peru-source "$PERU_SOURCE_TSV" --outdir "$BENCH_INPUT_DIR" --id-col "$ID_COL" --source-col "$SOURCE_COL" --country-col "$COUNTRY_COL" --species-col "$SPECIES_COL" --species-value "$SPECIES_VALUE" --sources "$ANALYSIS_SOURCES" --group-col "$GROUP_COL" --folds "$N_FOLDS" --seed "$SEED" --loci-prefix "$LOCI_PREFIX" --loci-regex "$LOCI_REGEX"
