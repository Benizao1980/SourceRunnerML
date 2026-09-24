#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/config_coli.sh"
TARGET_IDS="$TREE_WORK/coli_tree_target_ids.txt"; [[ -s "$TARGET_IDS" ]] || { echo "ERROR: run ./40_make_tree_ids.sh first" >&2; exit 1; }
WORK="$TREE_WORK/PeruCcoli_PIRATE_IQTree"; MANIFEST="$WORK/coli_fasta_manifest.tsv"; ANNOT_ROOT="$WORK/01_prokka"; GFF_DIR="$WORK/01_prokka_gffs"; PIRATE_OUT="$WORK/02_pirate"; IQTREE_OUT="$WORK/03_iqtree"; LOG_DIR="$WORK/logs"; SCRIPT_DIR="$WORK/slurm_scripts"; FINAL_DIR="$WORK/final"; mkdir -p "$WORK" "$ANNOT_ROOT" "$GFF_DIR" "$PIRATE_OUT" "$IQTREE_OUT" "$LOG_DIR" "$SCRIPT_DIR" "$FINAL_DIR"
EXPECTED_N=$(grep -cve '^\s*$' "$TARGET_IDS"); echo "Target C. coli taxa: $EXPECTED_N"
declare -A WANT SEEN; while IFS= read -r id; do [[ -n "$id" ]] && WANT["$id"]=1; done < "$TARGET_IDS"; : > "$MANIFEST"
IFS=':' read -ra DIRS <<< "$CONTIG_DIRS"; shopt -s nullglob
for d in "${DIRS[@]}"; do for fasta in "$d"/*.fasta "$d"/*.fa "$d"/*.fna; do base=$(basename "$fasta"); stem="${base%.*}"; id="${stem%%_*}"; if [[ -n "${WANT[$id]:-}" ]]; then if [[ -n "${SEEN[$id]:-}" ]]; then echo "ERROR: duplicate contig mapping for $id" >&2; exit 1; fi; SEEN[$id]="$fasta"; printf '%s\t%s\n' "$id" "$fasta" >> "$MANIFEST"; fi; done; done; shopt -u nullglob; sort -k1,1 "$MANIFEST" -o "$MANIFEST"
N_MAN=$(wc -l < "$MANIFEST"); if [[ "$N_MAN" -ne "$EXPECTED_N" ]]; then echo "ERROR: mapped $N_MAN/$EXPECTED_N target contigs" >&2; comm -23 <(sort "$TARGET_IDS") <(cut -f1 "$MANIFEST"|sort -u) > "$WORK/missing_fasta_ids.txt" || true; exit 1; fi
PROKKA_SCRIPT="$SCRIPT_DIR/01_prokka_array.sh"; PIRATE_SCRIPT="$SCRIPT_DIR/02_pirate_core.sh"; IQTREE_SCRIPT="$SCRIPT_DIR/03_iqtree.sh"
cat > "$PROKKA_SCRIPT" <<'EOF'
#!/usr/bin/env bash
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=06:00:00
set -euo pipefail
source "$CONDA_SH"; conda activate "$PROKKA_ENV"; export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
START=$((SLURM_ARRAY_TASK_ID*GENOMES_PER_TASK)); END=$((START+GENOMES_PER_TASK-1)); LAST=$((EXPECTED_N-1)); (( END>LAST )) && END=$LAST
for IDX in $(seq "$START" "$END"); do line=$(sed -n "$((IDX+1))p" "$MANIFEST"); IFS=$'\t' read -r ID FASTA <<< "$line"; OUT="$ANNOT_ROOT/$ID"; FINAL_GFF="$GFF_DIR/$ID.gff"; [[ -s "$FINAL_GFF" ]] && continue; rm -rf "$OUT"; prokka "$FASTA" --outdir "$OUT" --prefix "$ID" --locustag "CC${ID}" --kingdom Bacteria --genus Campylobacter --species coli --cpus "${SLURM_CPUS_PER_TASK:-2}" --force; [[ -s "$OUT/$ID.gff" ]] || exit 1; ln -sfn "$OUT/$ID.gff" "$FINAL_GFF"; done
EOF
cat > "$PIRATE_SCRIPT" <<'EOF'
#!/usr/bin/env bash
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=72:00:00
set -euo pipefail
source "$CONDA_SH"; conda activate "$PIRATE_ENV"; export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
N=$(find -L "$GFF_DIR" -maxdepth 1 -type f -name '*.gff'|wc -l); [[ "$N" -eq "$EXPECTED_N" ]] || { echo "ERROR expected $EXPECTED_N GFF, found $N" >&2; exit 1; }
if ! find "$PIRATE_OUT" -maxdepth 3 -type f -name core_alignment.fasta -size +0c | grep -q .; then PIRATE -i "$GFF_DIR" -o "$PIRATE_OUT" -t "${SLURM_CPUS_PER_TASK:-64}" -s '50,60,70,80,90,95,98' --pan-opt '--diamond' -a -z 1; fi
CORE_RAW=$(find "$PIRATE_OUT" -maxdepth 3 -type f -name core_alignment.fasta -size +0c | head -1); [[ -s "$CORE_RAW" ]] || exit 1; CORE="$FINAL_DIR/Ccoli_PIRATE_core_alignment.fasta"; awk '/^>/{h=substr($0,2);sub(/[[:space:]].*/,"",h);sub(/\.gff$/, "", h);print ">"h;next}{print}' "$CORE_RAW" > "$CORE"; [[ $(grep -c '^>' "$CORE") -eq "$EXPECTED_N" ]] || exit 1
EOF
cat > "$IQTREE_SCRIPT" <<'EOF'
#!/usr/bin/env bash
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=72:00:00
set -euo pipefail
source "$CONDA_SH"; conda activate "$IQTREE_ENV"; BIN=''; for c in iqtree2 iqtree3 iqtree; do command -v "$c" >/dev/null 2>&1 && { BIN=$(command -v "$c"); break; }; done; [[ -n "$BIN" ]] || exit 1; CORE="$FINAL_DIR/Ccoli_PIRATE_core_alignment.fasta"; [[ -s "$CORE" ]] || exit 1; mkdir -p "$IQTREE_OUT"; PREFIX="$IQTREE_OUT/Ccoli_PIRATE_core_IQTree"; "$BIN" -s "$CORE" -m MFP -B 1000 -alrt 1000 -T "${SLURM_CPUS_PER_TASK:-32}" -safe -seed 42 --prefix "$PREFIX"; [[ -s "$PREFIX.treefile" ]] || exit 1; cp -f "$PREFIX.treefile" "$FINAL_DIR/Figure4_Ccoli_core_IQTree.treefile"
EOF
chmod +x "$PROKKA_SCRIPT" "$PIRATE_SCRIPT" "$IQTREE_SCRIPT"
TASKS=$(( (EXPECTED_N+PROKKA_GENOMES_PER_TASK-1)/PROKKA_GENOMES_PER_TASK )); (( TASKS<=500 )) || { echo "ERROR: $TASKS array tasks >500" >&2; exit 1; }; MAX=$((TASKS-1))
p=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --job-name=Ccoli_Prokka --array="0-${MAX}%${ANNOT_MAX_CONCURRENT}" --output="$LOG_DIR/prokka_%A_%a.out" --error="$LOG_DIR/prokka_%A_%a.err" --export=ALL,MANIFEST="$MANIFEST",ANNOT_ROOT="$ANNOT_ROOT",GFF_DIR="$GFF_DIR",CONDA_SH="$CONDA_SH",PROKKA_ENV="$PROKKA_ENV",GENOMES_PER_TASK="$PROKKA_GENOMES_PER_TASK",EXPECTED_N="$EXPECTED_N" "$PROKKA_SCRIPT"); p=${p%%;*}
r=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --dependency="afterok:$p" --job-name=Ccoli_PIRATE --output="$LOG_DIR/pirate_%j.out" --error="$LOG_DIR/pirate_%j.err" --export=ALL,MANIFEST="$MANIFEST",GFF_DIR="$GFF_DIR",PIRATE_OUT="$PIRATE_OUT",FINAL_DIR="$FINAL_DIR",CONDA_SH="$CONDA_SH",PIRATE_ENV="$PIRATE_ENV",EXPECTED_N="$EXPECTED_N" "$PIRATE_SCRIPT"); r=${r%%;*}
i=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --dependency="afterok:$r" --job-name=Ccoli_IQTree --output="$LOG_DIR/iqtree_%j.out" --error="$LOG_DIR/iqtree_%j.err" --export=ALL,FINAL_DIR="$FINAL_DIR",IQTREE_OUT="$IQTREE_OUT",CONDA_SH="$CONDA_SH",IQTREE_ENV="$IQTREE_ENV",EXPECTED_N="$EXPECTED_N" "$IQTREE_SCRIPT"); i=${i%%;*}
echo "Prokka: $p [$TASKS tasks]"; echo "PIRATE: $r"; echo "IQ-TREE: $i"; echo "Final tree: $FINAL_DIR/Figure4_Ccoli_core_IQTree.treefile"
