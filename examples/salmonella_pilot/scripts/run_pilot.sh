#!/usr/bin/env bash
set -euo pipefail

# Salmonella worked-example runner scaffold.
#
# This is intentionally guarded: the general v1.0 wrapper still selects loci
# using one common prefix. Do not run a Salmonella analysis until the selected
# export has a documented compatible prefix, or issue #5 has been implemented.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PILOT_DIR/../.." && pwd)"

TRAIN_FILE="${1:-$PILOT_DIR/data/SALMONELLA_PILOT.train.tsv}"
PREDICT_FILE="${2:-$PILOT_DIR/data/SALMONELLA_PILOT.predict.tsv}"
OUTDIR="${3:-$PILOT_DIR/results}"
LOCI_PREFIX="${4:-${SALMONELLA_LOCI_PREFIX:-}}"

if [[ -z "$LOCI_PREFIX" ]]; then
  cat >&2 <<'EOF'
STOP: Salmonella pilot locus selection has not been fixed yet.

The current full-validation wrapper requires a common --loci_prefix.
Either:
  1. implement/generalise locus selection (GitHub issue #5), or
  2. if the curated pilot export genuinely has one shared locus prefix,
     pass that prefix as argument 4 or SALMONELLA_LOCI_PREFIX.

Example (only if scientifically/data-format appropriate):
  bash examples/salmonella_pilot/scripts/run_pilot.sh TRAIN.tsv PREDICT.tsv results LOCUS_PREFIX
EOF
  exit 2
fi

[[ -s "$TRAIN_FILE" ]] || { echo "Missing training file: $TRAIN_FILE" >&2; exit 1; }
[[ -s "$PREDICT_FILE" ]] || { echo "Missing prediction file: $PREDICT_FILE" >&2; exit 1; }

python "$REPO_ROOT/scripts/sourcerunner_full_validation.py" \
  --train_file "$TRAIN_FILE" \
  --predict_file "$PREDICT_FILE" \
  --output_dir "$OUTDIR" \
  --run_name salmonella_pilot \
  --source_col harmonized_source \
  --keep_sources Poultry,Ruminant,"Wild bird" \
  --loci_prefix "$LOCI_PREFIX" \
  --models random_forest,xgboost,logreg \
  --cv_folds 5 \
  --bootstrap 25 \
  --burn_in 5 \
  --pred_bootstrap 20 \
  --min_confidence 0.60 \
  --cpus 8
