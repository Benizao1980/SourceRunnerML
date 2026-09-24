#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/config_coli.sh"
GLOBAL="$BENCH_INPUT_DIR/global_nonPeru_clean.tsv"; PERU="$BENCH_INPUT_DIR/peru_known_source_clean.tsv"; MAN="$BENCH_INPUT_DIR/coli_binary_5fold_manifest.csv"
for f in "$GLOBAL" "$PERU" "$MAN"; do [[ -s "$f" ]] || { echo "ERROR: run ./01_prepare_benchmark.sh first; missing $f" >&2; exit 1; }; done
PARTS="$GEO_WORK/parts"; FINAL="$GEO_WORK/final"; LOGS="$GEO_WORK/logs"; SCRIPTS="$GEO_WORK/slurm_scripts"; mkdir -p "$PARTS" "$FINAL" "$LOGS" "$SCRIPTS"
PREF="$SCRIPTS/preflight.sh"; WORKER="$SCRIPTS/worker.sh"; AGG="$SCRIPTS/aggregate.sh"
cat > "$PREF" <<'EOF'
#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=00:20:00
set -euo pipefail
unset PYTHON PYTHONPATH || true
"$PYTHON_BIN" - <<'PY'
import sys,numpy,pandas,sklearn,xgboost
print('Python',sys.executable); print('imports passed')
PY
for f in "$GLOBAL" "$PERU" "$MAN"; do [[ -s "$f" ]] || exit 1; done
EOF
cat > "$WORKER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
unset PYTHON PYTHONPATH || true
exec "$PYTHON_BIN" "$RUNNER" --panel "$PANEL" --fold "$FOLD" --global-file "$GLOBAL" --peru-file "$PERU" --manifest "$MAN" --outdir "$PARTS" --id-col "$ID_COL" --missingness "$MISSINGNESS" --cpus "$SLURM_CPUS_PER_TASK" --loci-prefix "$LOCI_PREFIX" --loci-regex "$LOCI_REGEX"
EOF
cat > "$AGG" <<'EOF'
#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=01:00:00
set -euo pipefail
unset PYTHON PYTHONPATH || true
exec "$PYTHON_BIN" "$AGGREGATOR" --parts "$PARTS" --outdir "$FINAL"
EOF
chmod +x "$PREF" "$WORKER" "$AGG"
pre=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --job-name=CcoliGeo_pre --output="$LOGS/preflight_%j.out" --error="$LOGS/preflight_%j.err" --export=ALL,PYTHON_BIN="$SRML_PYTHON",GLOBAL="$GLOBAL",PERU="$PERU",MAN="$MAN" "$PREF"); pre=${pre%%;*}; echo "preflight: $pre"
workers=()
submit_one(){ local name=$1 panel=$2 fold=$3; local j; j=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --dependency="afterok:$pre" --job-name="$name" --cpus-per-task="$MODEL_CPUS" --mem="$MODEL_MEM" --time="$MODEL_TIME" --output="$LOGS/${name}_%j.out" --error="$LOGS/${name}_%j.err" --export=ALL,PYTHON_BIN="$SRML_PYTHON",RUNNER="$HERE/11_run_geography_one.py",PANEL="$panel",FOLD="$fold",GLOBAL="$GLOBAL",PERU="$PERU",MAN="$MAN",PARTS="$PARTS",ID_COL="$ID_COL",MISSINGNESS="$MISSINGNESS",LOCI_PREFIX="$LOCI_PREFIX",LOCI_REGEX="$LOCI_REGEX" "$WORKER"); j=${j%%;*}; workers+=("$j"); echo "$name: $j panel=$panel fold=$fold"; }
submit_one CcoliGeo_G Global_only_nonPeru -1
for f in $(seq 0 $((N_FOLDS-1))); do submit_one "CcoliGeo_L$f" Peru_local_OOF "$f"; done
for f in $(seq 0 $((N_FOLDS-1))); do submit_one "CcoliGeo_C$f" Global_plus_Peru_OOF "$f"; done
dep=$(IFS=:; echo "${workers[*]}")
agg=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --dependency="afterok:$dep" --job-name=CcoliGeo_agg --output="$LOGS/aggregate_%j.out" --error="$LOGS/aggregate_%j.err" --export=ALL,PYTHON_BIN="$SRML_PYTHON",AGGREGATOR="$HERE/12_aggregate_geography.py",PARTS="$PARTS",FINAL="$FINAL" "$AGG"); agg=${agg%%;*}
echo "aggregation: $agg"; echo "Final: $FINAL"
