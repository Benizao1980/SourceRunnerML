#!/usr/bin/env bash
set -euo pipefail

# Salmonella worked-example runner scaffold.
#
# This is intentionally guarded: the locus scheme must be audited and explicitly supplied.
# This is a feasibility scaffold rather than a validated Salmonella benchmark.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PILOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$PILOT_DIR/../.." && pwd)"

TRAIN_FILE="${1:-$PILOT_DIR/data/SALMONELLA_PILOT.train.tsv}"
PREDICT_FILE="${2:-$PILOT_DIR/data/SALMONELLA_PILOT.predict.tsv}"
OUTDIR="${3:-$PILOT_DIR/results}"
LOCI_PREFIX="${4:-${SALMONELLA_LOCI_PREFIX:-}}"

LOCI_FILE="${5:-${SALMONELLA_LOCI_FILE:-}}"

if [[ -z "$LOCI_FILE" && -z "$LOCI_PREFIX" ]]; then
  echo "STOP: supply a verified locus file as argument 5, or a verified prefix as argument 4." >&2
  echo "Example: bash examples/salmonella_pilot/scripts/run_pilot.sh TRAIN.tsv PREDICT.tsv results '' loci.txt" >&2
  exit 2
fi
[[ -s "$TRAIN_FILE" ]] || { echo "Missing training file: $TRAIN_FILE" >&2; exit 1; }
[[ -s "$PREDICT_FILE" ]] || { echo "Missing prediction file: $PREDICT_FILE" >&2; exit 1; }

loci_args=()
if [[ -n "$LOCI_FILE" ]]; then
  [[ -s "$LOCI_FILE" ]] || { echo "Missing locus list: $LOCI_FILE" >&2; exit 1; }
  loci_args=(--loci_file "$LOCI_FILE")
else
  loci_args=(--loci_prefix "$LOCI_PREFIX")
fi

python "$REPO_ROOT/scripts/source_runner_preflight.py" \
  --train "$TRAIN_FILE" --predict "$PREDICT_FILE" \
  --source_col harmonized_source --min_source_size 20 \
  "${loci_args[@]}" --outdir "$OUTDIR/preflight"
python "$REPO_ROOT/scripts/sourcerunner_full_validation.py" \
  --train_file "$TRAIN_FILE" \
  --predict_file "$PREDICT_FILE" \
  --output_dir "$OUTDIR" \
  --run_name salmonella_pilot \
  --source_col harmonized_source \
  --keep_sources Poultry,Ruminant,"Wild bird" \
  "${loci_args[@]}" \
  --models random_forest,xgboost,logreg \
  --cv_folds 5 \
  --bootstrap 25 \
  --burn_in 5 \
  --pred_bootstrap 20 \
  --min_confidence 0.60 \
  --cpus 8
