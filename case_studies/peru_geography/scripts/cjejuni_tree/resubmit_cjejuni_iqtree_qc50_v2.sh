#!/usr/bin/env bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-cooperma}"
PARTITION="${PARTITION:-standard}"
ROOT="${ROOT:-/xdisk/cooperma/bpascoe/SourceAttribution_Peru/figures/Figure1/Peru2217_PIRATE_IQTree}"
CORE_IN="${CORE_IN:-$ROOT/final/Peru2217_PIRATE_core_alignment.fasta}"
OUTDIR="${OUTDIR:-$ROOT/03_iqtree_qc50}"
FINALDIR="${FINALDIR:-$ROOT/final}"
LOGDIR="${LOGDIR:-$ROOT/logs}"
SCRIPT_DIR="${SCRIPT_DIR:-$ROOT/slurm_scripts}"
PYTHON_BIN="${PYTHON_BIN:-/home/u12/bpascoe/miniconda3/envs/srml/bin/python}"
CONDA_SH="${CONDA_SH:-/home/u12/bpascoe/miniconda3/etc/profile.d/conda.sh}"
IQTREE_ENV="${IQTREE_ENV:-iqtree}"
MAX_MISSING="${MAX_MISSING:-0.50}"

mkdir -p "$OUTDIR" "$FINALDIR" "$LOGDIR" "$SCRIPT_DIR"
[[ -s "$CORE_IN" ]] || { echo "ERROR: missing core alignment: $CORE_IN" >&2; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "ERROR: Python not executable: $PYTHON_BIN" >&2; exit 1; }

FILTERED="$FINALDIR/Peru_Cjejuni_PIRATE_core_alignment_qc50.fasta"
QC_TSV="$FINALDIR/Peru_Cjejuni_core_alignment_sequence_qc.tsv"
EXCLUDED="$FINALDIR/Peru_Cjejuni_core_alignment_excluded_qc50.txt"
RETAINED="$FINALDIR/Peru_Cjejuni_core_alignment_retained_qc50.txt"
QC_SCRIPT="$SCRIPT_DIR/04_filter_core_alignment_qc50.sh"
IQ_SCRIPT="$SCRIPT_DIR/05_iqtree_qc50.sh"

cat > "$QC_SCRIPT" <<'QC_EOF'
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=02:00:00
set -euo pipefail

echo "=== C. jejuni core-alignment taxon QC ==="
echo "Host: $(hostname)"
echo "Input: $CORE_IN"
echo "Threshold: <= $MAX_MISSING missing/ambiguous"
rm -f "$FILTERED" "$QC_TSV" "$EXCLUDED" "$RETAINED"

"$PYTHON_BIN" - "$CORE_IN" "$FILTERED" "$QC_TSV" "$EXCLUDED" "$RETAINED" "$MAX_MISSING" <<'PY'
import sys
src, outfa, qctsv, excluded, retained, threshold = sys.argv[1:]
threshold = float(threshold)

def process_record(name, chunks, outfh, qcfh, exfh, refh, expected_len):
    seq = "".join(chunks); L = len(seq)
    if expected_len is None: expected_len = L
    elif L != expected_len: raise SystemExit(f"ERROR: {name} has length {L}, expected {expected_len}")
    good = sum(ch in "ACGTacgt" for ch in seq); missing = L-good; frac = missing/L if L else 1.0
    decision = "retain" if frac <= threshold else "exclude"
    qcfh.write(f"{name}\t{L}\t{good}\t{missing}\t{frac:.6f}\t{decision}\n")
    if decision == "retain":
        refh.write(name+"\n"); outfh.write(f">{name}\n")
        for i in range(0,L,80): outfh.write(seq[i:i+80]+"\n")
    else:
        exfh.write(name+"\n"); print(f"EXCLUDE\t{name}\tmissing={frac:.2%}", flush=True)
    return expected_len, decision

n=n_keep=n_drop=0; expected_len=None; name=None; chunks=[]
with open(src) as infh, open(outfa,"w") as outfh, open(qctsv,"w") as qcfh, open(excluded,"w") as exfh, open(retained,"w") as refh:
    qcfh.write("id\talignment_length\tunambiguous_ACGT\tmissing_or_ambiguous\tmissing_fraction\tdecision\n")
    for raw in infh:
        line=raw.strip()
        if not line: continue
        if line.startswith(">"):
            if name is not None:
                expected_len,decision=process_record(name,chunks,outfh,qcfh,exfh,refh,expected_len); n+=1; n_keep+=decision=="retain"; n_drop+=decision=="exclude"
            name=line[1:].split()[0]; chunks=[]
        else: chunks.append(line)
    if name is not None:
        expected_len,decision=process_record(name,chunks,outfh,qcfh,exfh,refh,expected_len); n+=1; n_keep+=decision=="retain"; n_drop+=decision=="exclude"
if n==0: raise SystemExit("ERROR: no FASTA records found")
print(f"Input taxa   : {n}\nRetained     : {n_keep}\nExcluded     : {n_drop}\nAlignment bp : {expected_len}\nFiltered     : {outfa}\nQC table     : {qctsv}")
PY

N_IN=$(grep -c '^>' "$CORE_IN"); N_OUT=$(grep -c '^>' "$FILTERED"); N_DROP=$(wc -l < "$EXCLUDED" | tr -d ' ')
(( N_OUT + N_DROP == N_IN )) || { echo "ERROR: taxa accounting failed" >&2; exit 1; }
(( N_OUT > 0 )) || { echo "ERROR: no taxa retained" >&2; exit 1; }
echo "QC PASSED"; echo "Input taxa: $N_IN; retained: $N_OUT; excluded: $N_DROP"; cat "$EXCLUDED"
QC_EOF

# Historical note: the original dependent IQ-TREE step below requested 160G and
# OOM-killed during ModelFinder on the 2,194-genome alignment. For a repeat run,
# use run_cjejuni_iqtree_qc50_240G.sh after this QC stage, or increase this block
# to >=240G before submission.
cat > "$IQ_SCRIPT" <<'IQ_EOF'
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=72:00:00
set -euo pipefail
source "$CONDA_SH"; conda activate "$IQTREE_ENV"
IQTREE_BIN="$(command -v iqtree2 || command -v iqtree3 || command -v iqtree)"
[[ -n "$IQTREE_BIN" && -s "$FILTERED" ]] || exit 1
PREFIX="$OUTDIR/Peru_Cjejuni_PIRATE_core_IQTree_qc50"
"$IQTREE_BIN" -s "$FILTERED" -m MFP -B 1000 -alrt 1000 -T "${SLURM_CPUS_PER_TASK:-32}" -safe -seed 42 --prefix "$PREFIX"
[[ -s "$PREFIX.treefile" ]] || exit 1
cp -f "$PREFIX.treefile" "$FINALDIR/Figure1B_Peru_Cjejuni_core_IQTree_qc50.treefile"
IQ_EOF
chmod +x "$QC_SCRIPT" "$IQ_SCRIPT"

echo "Submitting streaming alignment QC..."
QC_JOB=$(sbatch --parsable --account="$ACCOUNT" --partition="$PARTITION" --job-name=PeruCj_QC50 --output="$LOGDIR/core_qc50_%j.out" --error="$LOGDIR/core_qc50_%j.err" --export=ALL,CORE_IN="$CORE_IN",FILTERED="$FILTERED",QC_TSV="$QC_TSV",EXCLUDED="$EXCLUDED",RETAINED="$RETAINED",MAX_MISSING="$MAX_MISSING",PYTHON_BIN="$PYTHON_BIN" "$QC_SCRIPT")
QC_JOB=${QC_JOB%%;*}; echo "QC job: $QC_JOB"
echo "WARNING: historical dependent IQ-TREE stage requests only 160G and is retained for provenance. Prefer the 240G final script for production."