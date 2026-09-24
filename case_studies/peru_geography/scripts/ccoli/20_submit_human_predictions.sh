#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/config_coli.sh"
GLOBAL="$BENCH_INPUT_DIR/global_nonPeru_clean.tsv"; PERU="$BENCH_INPUT_DIR/peru_known_source_clean.tsv"; [[ -s "$GLOBAL" && -s "$PERU" && -s "$HUMAN_TSV" ]] || { echo "ERROR: prepare benchmark and check HUMAN_TSV first" >&2; exit 1; }
PARTS="$HUMAN_WORK/parts"; FINAL="$HUMAN_WORK/final"; LOGS="$HUMAN_WORK/logs"; mkdir -p "$PARTS" "$FINAL" "$LOGS"
jobs=()
for spec in 'Global_only_nonPeru:G' 'Peru_local:L' 'Global_plus_Peru:C'; do IFS=: read -r pan short <<< "$spec"; j=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --job-name="CcoliHum_$short" --cpus-per-task="$MODEL_CPUS" --mem="$MODEL_MEM" --time="$MODEL_TIME" --output="$LOGS/${short}_%j.out" --error="$LOGS/${short}_%j.err" --wrap="unset PYTHON PYTHONPATH || true; '$SRML_PYTHON' '$HERE/21_run_human_one.py' --panel '$pan' --global-file '$GLOBAL' --peru-file '$PERU' --human-file '$HUMAN_TSV' --outdir '$PARTS' --id-col '$ID_COL' --missingness '$MISSINGNESS' --cpus '$MODEL_CPUS' --loci-prefix '$LOCI_PREFIX' --loci-regex '$LOCI_REGEX'"); j=${j%%;*}; jobs+=("$j"); echo "$pan: $j"; done
dep=$(IFS=:; echo "${jobs[*]}"); agg=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --dependency="afterok:$dep" --job-name=CcoliHum_agg --cpus-per-task=2 --mem=10G --time=01:00:00 --output="$LOGS/aggregate_%j.out" --error="$LOGS/aggregate_%j.err" --wrap="'$SRML_PYTHON' '$HERE/22_aggregate_human.py' --parts '$PARTS' --outdir '$FINAL'"); echo "aggregation: ${agg%%;*}"; echo "Final: $FINAL"
