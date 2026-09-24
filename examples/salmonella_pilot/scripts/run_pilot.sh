#!/usr/bin/env bash
set -euo pipefail

# Placeholder runner for the Salmonella worked example.
# Replace paths/source categories after the public pilot dataset has been curated.

TRAIN_FILE="${1:-data/SALMONELLA_PILOT.train.tsv}"
PREDICT_FILE="${2:-data/SALMONELLA_PILOT.predict.tsv}"
OUTDIR="${3:-results}"

python ../../../scripts/sourcerunner_full_validation.py \
  --train_file "$TRAIN_FILE" \
  --predict_file "$PREDICT_FILE" \
  --output_dir "$OUTDIR" \
  --run_name salmonella_pilot \
  --source_col harmonized_source \
  --keep_sources Poultry,Ruminant,"Wild bird" \
  --models random_forest,xgboost,logreg \
  --cv_folds 5 \
  --bootstrap 25 \
  --burn_in 5 \
  --pred_bootstrap 20 \
  --min_confidence 0.60 \
  --cpus 8

# NOTE: locus-prefix handling may need adapting for the selected Salmonella cgMLST export.
# This script is intentionally a placeholder until the pilot input format is fixed.
