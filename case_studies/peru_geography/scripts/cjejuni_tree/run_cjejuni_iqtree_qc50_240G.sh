#!/usr/bin/env bash
#SBATCH --account=cooperma
#SBATCH --partition=standard
#SBATCH --job-name=PeruCj_IQ_240G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=72:00:00
#SBATCH --output=/xdisk/cooperma/bpascoe/SourceAttribution_Peru/figures/Figure1/Peru2217_PIRATE_IQTree/logs/iqtree_qc50_240G_%j.out
#SBATCH --error=/xdisk/cooperma/bpascoe/SourceAttribution_Peru/figures/Figure1/Peru2217_PIRATE_IQTree/logs/iqtree_qc50_240G_%j.err

set -euo pipefail

ROOT="/xdisk/cooperma/bpascoe/SourceAttribution_Peru/figures/Figure1/Peru2217_PIRATE_IQTree"
FILTERED="$ROOT/final/Peru_Cjejuni_PIRATE_core_alignment_qc50.fasta"
OUTDIR="$ROOT/03_iqtree_qc50"
FINAL="$ROOT/final/Figure1B_Peru_Cjejuni_core_IQTree_qc50.treefile"
PREFIX="$OUTDIR/Peru_Cjejuni_PIRATE_core_IQTree_qc50"

source /home/u12/bpascoe/miniconda3/etc/profile.d/conda.sh
conda activate iqtree
IQTREE_BIN="$(command -v iqtree2 || command -v iqtree3 || command -v iqtree)"

echo "Host: $(hostname)"
echo "SLURM job: ${SLURM_JOB_ID:-NA}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-NA}"
echo "IQ-TREE: $IQTREE_BIN"
echo "Input: $FILTERED"
echo "Prefix: $PREFIX"

[[ -s "$FILTERED" ]] || { echo "ERROR: filtered alignment not found: $FILTERED" >&2; exit 1; }

"$IQTREE_BIN" \
  -s "$FILTERED" \
  -m MFP \
  -B 1000 \
  -alrt 1000 \
  -T "${SLURM_CPUS_PER_TASK}" \
  -safe \
  -seed 42 \
  --prefix "$PREFIX"

[[ -s "$PREFIX.treefile" ]] || { echo "ERROR: IQ-TREE completed without treefile: $PREFIX.treefile" >&2; exit 1; }
cp -f "$PREFIX.treefile" "$FINAL"
echo "IQ-TREE PASSED"
echo "Final tree: $FINAL"
