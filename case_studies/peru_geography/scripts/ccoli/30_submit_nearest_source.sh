#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/config_coli.sh"
GLOBAL="$BENCH_INPUT_DIR/global_nonPeru_clean.tsv"; PERU="$BENCH_INPUT_DIR/peru_known_source_clean.tsv"; mkdir -p "$NEAREST_WORK/logs"
COMBINED="$NEAREST_WORK/combined_sources.tsv"
"$SRML_PYTHON" - "$GLOBAL" "$PERU" "$COMBINED" <<'PY'
import sys,pandas as pd
g=pd.read_csv(sys.argv[1],sep='\t',dtype=str); l=pd.read_csv(sys.argv[2],sep='\t',dtype=str); pd.concat([g,l],ignore_index=True,sort=False).to_csv(sys.argv[3],sep='\t',index=False)
PY
j=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --job-name=CcoliNearest --cpus-per-task=4 --mem=40G --time=24:00:00 --output="$NEAREST_WORK/logs/nearest_%j.out" --error="$NEAREST_WORK/logs/nearest_%j.err" --wrap="'$SRML_PYTHON' '$HERE/31_nearest_source_cgmlst.py' --sources '$COMBINED' --human '$HUMAN_TSV' --out '$NEAREST_WORK/nearest_source_combined.tsv' --id-col '$ID_COL' --source-col source_class --loci-prefix '$LOCI_PREFIX' --loci-regex '$LOCI_REGEX'"); echo "nearest-source job: ${j%%;*}"; echo "Output: $NEAREST_WORK/nearest_source_combined.tsv"
