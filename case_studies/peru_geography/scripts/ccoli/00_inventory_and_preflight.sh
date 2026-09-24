#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config_coli.sh"
for f in "$GLOBAL_SOURCE_TSV" "$PERU_SOURCE_TSV" "$HUMAN_TSV"; do [[ -s "$f" ]] || { echo "ERROR: missing/empty input: $f" >&2; exit 1; }; done
[[ -x "$SRML_PYTHON" ]] || { echo "ERROR: SRML python not executable: $SRML_PYTHON" >&2; exit 1; }
unset PYTHON PYTHONPATH || true
"$SRML_PYTHON" - <<'PY'
import sys,numpy,pandas,sklearn,xgboost
print('Python :',sys.executable); print('numpy  :',numpy.__version__); print('pandas :',pandas.__version__); print('sklearn:',sklearn.__version__); print('xgboost:',xgboost.__version__); print('IMPORT TEST PASSED')
PY
"$SRML_PYTHON" "$HERE/00_inventory_coli.py" --global-source "$GLOBAL_SOURCE_TSV" --peru-source "$PERU_SOURCE_TSV" --human "$HUMAN_TSV" --outdir "$INVENTORY_DIR" --id-col "$ID_COL" --source-col "$SOURCE_COL" --country-col "$COUNTRY_COL" --species-col "$SPECIES_COL" --species-value "$SPECIES_VALUE" --loci-prefix "$LOCI_PREFIX" --loci-regex "$LOCI_REGEX" --contig-dirs "$CONTIG_DIRS"
