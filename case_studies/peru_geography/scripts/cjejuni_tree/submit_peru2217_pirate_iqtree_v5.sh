#!/usr/bin/env bash
# submit_peru2217_pirate_iqtree_v4.sh
#
# Build a publication-ready core-genome phylogeny for the same 2,217 Peru
# C. jejuni isolates used in the Figure 1 Microreact metadata:
#
#   1. Select contigs by pubMLST id from:
#      peru_human_predict.tsv + peru_contextual_train_3source.tsv
#   2. Annotate each genome with Prokka (SLURM array)
#   3. Run PIRATE with DIAMOND and produce the >=95% core alignment
#   4. Run IQ-TREE 2 with ModelFinder + SH-aLRT + UFBoot
#
# Run this file from a login node:
#   chmod +x submit_peru2217_pirate_iqtree.sh
#   ./submit_peru2217_pirate_iqtree.sh
#
# Uses the dedicated conda environments already present on Junonia:
#   prokka  -> annotation
#   pirate  -> pangenome/core alignment
#   iqtree  -> phylogeny

set -euo pipefail

ACCOUNT="${ACCOUNT:-cooperma}"
PARTITION="${PARTITION:-standard}"
PROJECT="${PROJECT:-/xdisk/cooperma/bpascoe/SourceAttribution_Peru}"
CONTIG_DIR="${CONTIG_DIR:-$PROJECT/data_inputs/00.contigs_peru}"
HUMAN_TSV="${HUMAN_TSV:-$PROJECT/data_inputs/peru_human_predict.tsv}"
SOURCE_TSV="${SOURCE_TSV:-$PROJECT/data_inputs/peru_contextual_train_3source.tsv}"
WORK="${WORK:-$PROJECT/figures/Figure1/Peru2217_PIRATE_IQTree}"
EXPECTED_N="${EXPECTED_N:-2217}"
CONDA_SH="${CONDA_SH:-/home/u12/bpascoe/miniconda3/etc/profile.d/conda.sh}"
PROKKA_ENV="${PROKKA_ENV:-prokka}"
PIRATE_ENV="${PIRATE_ENV:-pirate}"
IQTREE_ENV="${IQTREE_ENV:-iqtree}"
PROKKA_GENOMES_PER_TASK="${PROKKA_GENOMES_PER_TASK:-5}"
ANNOT_MAX_CONCURRENT="${ANNOT_MAX_CONCURRENT:-100}"

MANIFEST="$WORK/peru2217_fasta_manifest.tsv"
TARGET_IDS="$WORK/peru2217_target_ids.txt"
ANNOT_ROOT="$WORK/01_prokka"
GFF_DIR="$WORK/01_prokka_gffs"
PIRATE_OUT="$WORK/02_pirate"
IQTREE_OUT="$WORK/03_iqtree"
LOG_DIR="$WORK/logs"
SCRIPT_DIR="$WORK/slurm_scripts"
FINAL_DIR="$WORK/final"
mkdir -p "$WORK" "$ANNOT_ROOT" "$GFF_DIR" "$PIRATE_OUT" "$IQTREE_OUT" "$LOG_DIR" "$SCRIPT_DIR" "$FINAL_DIR"

for f in "$HUMAN_TSV" "$SOURCE_TSV"; do [[ -s "$f" ]] || { echo "ERROR: missing/empty metadata input: $f" >&2; exit 1; }; done
[[ -d "$CONTIG_DIR" ]] || { echo "ERROR: contig directory does not exist: $CONTIG_DIR" >&2; exit 1; }

extract_id_column () {
    local file="$1"
    awk -F '\t' 'NR==1 {col=0; for (i=1;i<=NF;i++){gsub(/\r/,"",$i); if ($i=="id") col=i} if(!col){print "ERROR: no id column in " FILENAME > "/dev/stderr"; exit 2} next} {gsub(/\r/,"",$col); if($col!="") print $col}' "$file"
}
{
    extract_id_column "$HUMAN_TSV"
    extract_id_column "$SOURCE_TSV"
} | sort -u > "$TARGET_IDS"
N_TARGET=$(wc -l < "$TARGET_IDS" | tr -d ' ')
[[ "$N_TARGET" -eq "$EXPECTED_N" ]] || { echo "ERROR: expected $EXPECTED_N unique target ids, found $N_TARGET" >&2; exit 1; }

declare -A WANT
while IFS= read -r id; do WANT["$id"]=1; done < "$TARGET_IDS"
: > "$MANIFEST"
shopt -s nullglob
for fasta in "$CONTIG_DIR"/*.fasta "$CONTIG_DIR"/*.fa "$CONTIG_DIR"/*.fna; do
    base=$(basename "$fasta"); stem="${base%.*}"; id="${stem%%_*}"
    [[ -n "${WANT[$id]:-}" ]] && printf "%s\t%s\n" "$id" "$fasta" >> "$MANIFEST"
done
shopt -u nullglob
sort -k1,1 "$MANIFEST" -o "$MANIFEST"
N_MANIFEST=$(wc -l < "$MANIFEST" | tr -d ' ')
N_UNIQUE=$(cut -f1 "$MANIFEST" | sort -u | wc -l | tr -d ' ')
if [[ "$N_MANIFEST" -ne "$EXPECTED_N" || "$N_UNIQUE" -ne "$EXPECTED_N" ]]; then
    comm -23 "$TARGET_IDS" <(cut -f1 "$MANIFEST" | sort -u) > "$WORK/missing_fasta_ids.txt" || true
    cut -f1 "$MANIFEST" | sort | uniq -d > "$WORK/duplicate_fasta_ids.txt" || true
    echo "ERROR: FASTA mapping failed; see missing/duplicate id files" >&2; exit 1
fi

echo "Manifest validated: $N_MANIFEST FASTAs for $N_UNIQUE unique ids"

PROKKA_SCRIPT="$SCRIPT_DIR/01_prokka_array.sh"
cat > "$PROKKA_SCRIPT" <<'PROKKA_EOF'
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=06:00:00
set -euo pipefail
source "$CONDA_SH"; conda activate "$PROKKA_ENV"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
START_INDEX=$((SLURM_ARRAY_TASK_ID * GENOMES_PER_TASK)); END_INDEX=$((START_INDEX + GENOMES_PER_TASK - 1)); LAST_INDEX=$((EXPECTED_N - 1)); (( END_INDEX > LAST_INDEX )) && END_INDEX=$LAST_INDEX
for GLOBAL_TASK_ID in $(seq "$START_INDEX" "$END_INDEX"); do
    LINE_NO=$((GLOBAL_TASK_ID + 1)); LINE=$(sed -n "${LINE_NO}p" "$MANIFEST"); IFS=$'\t' read -r ID FASTA <<< "$LINE"
    OUTDIR="$ANNOT_ROOT/$ID"; FINAL_GFF="$GFF_DIR/$ID.gff"; [[ -s "$FINAL_GFF" ]] && continue
    rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
    prokka "$FASTA" --outdir "$OUTDIR" --prefix "$ID" --locustag "CJ${ID}" --kingdom Bacteria --genus Campylobacter --species jejuni --cpus "${SLURM_CPUS_PER_TASK:-2}" --force
    [[ -s "$OUTDIR/$ID.gff" ]] || exit 1; ln -sfn "$OUTDIR/$ID.gff" "$FINAL_GFF"
done
PROKKA_EOF

PIRATE_SCRIPT="$SCRIPT_DIR/02_pirate_core.sh"
cat > "$PIRATE_SCRIPT" <<'PIRATE_EOF'
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=72:00:00
set -euo pipefail
source "$CONDA_SH"; conda activate "$PIRATE_ENV"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N_GFF=$(find -L "$GFF_DIR" -maxdepth 1 -type f -name '*.gff' | wc -l | tr -d ' '); [[ "$N_GFF" -eq "$EXPECTED_N" ]] || { echo "ERROR: expected $EXPECTED_N GFFs, found $N_GFF" >&2; exit 1; }
if ! find "$PIRATE_OUT" -maxdepth 2 -type f -name 'core_alignment.fasta' -size +0c | grep -q .; then
    PIRATE -i "$GFF_DIR" -o "$PIRATE_OUT" -t "${SLURM_CPUS_PER_TASK:-64}" -s "50,60,70,80,90,95,98" --pan-opt "--diamond" -a -z 1
fi
CORE_RAW=$(find "$PIRATE_OUT" -maxdepth 3 -type f -name 'core_alignment.fasta' -size +0c | head -n 1); [[ -n "$CORE_RAW" && -s "$CORE_RAW" ]] || exit 1
CORE_FINAL="$FINAL_DIR/Peru2217_PIRATE_core_alignment.fasta"
awk '/^>/ {h=substr($0,2); sub(/[[:space:]].*$/, "", h); sub(/\.gff$/, "", h); print ">" h; next} {print}' "$CORE_RAW" > "$CORE_FINAL"
N_SEQ=$(grep -c '^>' "$CORE_FINAL"); [[ "$N_SEQ" -eq "$EXPECTED_N" ]] || exit 1
cut -f1 "$MANIFEST" | sort -u > "$FINAL_DIR/expected_tree_ids.txt"; grep '^>' "$CORE_FINAL" | sed 's/^>//; s/[[:space:]].*$//' | sort -u > "$FINAL_DIR/core_alignment_tree_ids.txt"
diff -u "$FINAL_DIR/expected_tree_ids.txt" "$FINAL_DIR/core_alignment_tree_ids.txt" > "$FINAL_DIR/core_alignment_id_check.diff" || exit 1
PIRATE_EOF

IQTREE_SCRIPT="$SCRIPT_DIR/03_iqtree.sh"
cat > "$IQTREE_SCRIPT" <<'IQTREE_EOF'
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
set -euo pipefail
source "$CONDA_SH"; conda activate "$IQTREE_ENV"
IQTREE_BIN="$(command -v iqtree2 || command -v iqtree3 || command -v iqtree)"
CORE="$FINAL_DIR/Peru2217_PIRATE_core_alignment.fasta"; N_SEQ=$(grep -c '^>' "$CORE"); [[ "$N_SEQ" -eq "$EXPECTED_N" ]] || exit 1
mkdir -p "$IQTREE_OUT"; PREFIX="$IQTREE_OUT/Peru2217_PIRATE_core_IQTree"
"$IQTREE_BIN" -s "$CORE" -m MFP -B 1000 -alrt 1000 -T "${SLURM_CPUS_PER_TASK:-32}" -safe -seed 42 --prefix "$PREFIX"
[[ -s "$PREFIX.treefile" ]] || exit 1; cp -f "$PREFIX.treefile" "$FINAL_DIR/Figure1B_Peru2217_core_IQTree.treefile"
IQTREE_EOF
chmod +x "$PROKKA_SCRIPT" "$PIRATE_SCRIPT" "$IQTREE_SCRIPT"

N_ARRAY_TASKS=$(( (EXPECTED_N + PROKKA_GENOMES_PER_TASK - 1) / PROKKA_GENOMES_PER_TASK )); ARRAY_MAX=$((N_ARRAY_TASKS - 1)); (( N_ARRAY_TASKS <= 500 )) || exit 1
PROKKA_RAW=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --job-name=Peru2217_Prokka --array="0-${ARRAY_MAX}%${ANNOT_MAX_CONCURRENT}" --output="$LOG_DIR/prokka_%A_%a.out" --error="$LOG_DIR/prokka_%A_%a.err" --export=ALL,MANIFEST="$MANIFEST",ANNOT_ROOT="$ANNOT_ROOT",GFF_DIR="$GFF_DIR",CONDA_SH="$CONDA_SH",PROKKA_ENV="$PROKKA_ENV",GENOMES_PER_TASK="$PROKKA_GENOMES_PER_TASK",EXPECTED_N="$EXPECTED_N" "$PROKKA_SCRIPT"); PROKKA_JOB="${PROKKA_RAW%%;*}"
PIRATE_RAW=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --dependency="afterok:${PROKKA_JOB}" --job-name=Peru2217_PIRATE --output="$LOG_DIR/pirate_%j.out" --error="$LOG_DIR/pirate_%j.err" --export=ALL,MANIFEST="$MANIFEST",GFF_DIR="$GFF_DIR",PIRATE_OUT="$PIRATE_OUT",FINAL_DIR="$FINAL_DIR",CONDA_SH="$CONDA_SH",PIRATE_ENV="$PIRATE_ENV",EXPECTED_N="$EXPECTED_N" "$PIRATE_SCRIPT"); PIRATE_JOB="${PIRATE_RAW%%;*}"
IQTREE_RAW=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --dependency="afterok:${PIRATE_JOB}" --job-name=Peru2217_IQTree --output="$LOG_DIR/iqtree_%j.out" --error="$LOG_DIR/iqtree_%j.err" --export=ALL,FINAL_DIR="$FINAL_DIR",IQTREE_OUT="$IQTREE_OUT",CONDA_SH="$CONDA_SH",IQTREE_ENV="$IQTREE_ENV",EXPECTED_N="$EXPECTED_N" "$IQTREE_SCRIPT"); IQTREE_JOB="${IQTREE_RAW%%;*}"
echo "Prokka array: $PROKKA_JOB; PIRATE: $PIRATE_JOB; IQ-TREE: $IQTREE_JOB"
