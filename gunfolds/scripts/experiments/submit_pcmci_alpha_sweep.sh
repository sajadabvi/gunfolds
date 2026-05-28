#!/bin/bash
# =============================================================================
# Submit pcmci_alpha_sweep.py as a SLURM array — one task per alpha value.
#
# Decoupled from drasl: this script only runs the PCMCI front-end at multiple
# alpha values so you can pick a good alpha by the Exp 4 composite score
# (0.6 × cross-subject Jaccard + 0.4 × density proximity) without paying the
# multi-day drasl cost. PCMCI at N=53 is ~10–60 s per subject; the full sweep
# completes in a few hours rather than weeks.
#
# Memory/CPU/time are sized for PCMCI alone, not drasl. PCMCI is single-
# threaded, so we ask for 2 CPUs (one for PCMCI, one for OS overhead) and
# memory scales with N: 4 GB (N=10), 8 GB (N=20), 16 GB (N=53).
#
# Usage (run from anywhere; paths are resolved relative to this script):
#   bash submit_pcmci_alpha_sweep.sh 53
#   bash submit_pcmci_alpha_sweep.sh 10
#
# Override alpha grid:
#   ALPHAS="0.005 0.01 0.02 0.05 0.1" bash submit_pcmci_alpha_sweep.sh 53
# =============================================================================

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash submit_pcmci_alpha_sweep.sh <N>   where N in {10, 20, 53}"
    exit 1
fi

N="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/pcmci_alpha_sweep.py"

OUTPUT_DIR="results/pcmci_alpha_sweep_N${N}"
LOG_DIR="logs/pcmci_alpha_sweep_N${N}"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─── SLURM constants ─────────────────────────────────────────────────────────
PARTITION="qTRD"             # CPU partition — no GPU needed for PCMCI
ACCOUNT="psy53c17"
EMAIL="mabavisani@gsu.edu"
CPUS=2

case "$N" in
    10) MEM="4g";  WALLTIME="00:30:00" ;;
    20) MEM="8g";  WALLTIME="02:00:00" ;;
    53) MEM="16g"; WALLTIME="06:00:00" ;;
    *)  echo "Unsupported N=$N (allowed: 10, 20, 53)"; exit 1 ;;
esac

# Default alpha grid (overridable via env)
ALPHAS="${ALPHAS:-0.005 0.01 0.02 0.03 0.05 0.08 0.10}"
read -ra ALPHA_ARR <<< "$ALPHAS"
N_ALPHAS=${#ALPHA_ARR[@]}
LAST_IDX=$((N_ALPHAS - 1))

echo "Submitting PCMCI alpha sweep for N=${N}"
echo "  alphas:    ${ALPHAS}  (${N_ALPHAS} values)"
echo "  partition: ${PARTITION}    mem=${MEM}  cpus=${CPUS}  time=${WALLTIME}"
echo "  output:    ${OUTPUT_DIR}/"

JOB_ID=$(sbatch \
    --parsable \
    -J "pcmci_alpha_N${N}" \
    -p "$PARTITION" \
    -A "$ACCOUNT" \
    --mail-type=FAIL,END \
    --mail-user="$EMAIL" \
    -N 1 -n 1 \
    --cpus-per-task=${CPUS} \
    --mem=${MEM} \
    -t "$WALLTIME" \
    --array=0-${LAST_IDX} \
    -o "${LOG_DIR}/alpha_%a.out" \
    -e "${LOG_DIR}/alpha_%a.err" \
    --wrap "set -e
        export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-${CPUS}}
        . /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
        conda activate multi_v3
        cd \$SLURM_SUBMIT_DIR
        # POSIX-safe alpha pick: --wrap runs under /bin/sh (dash), no bash arrays.
        ALPHA=\$(echo '${ALPHAS}' | awk -v i=\$((SLURM_ARRAY_TASK_ID + 1)) '{print \$i}')
        echo \"task \$SLURM_ARRAY_TASK_ID -> alpha=\$ALPHA\"
        python ${RUNNER} \
            --alpha \$ALPHA \
            --n_components ${N} \
            --output_dir ${OUTPUT_DIR}
    ")

echo ""
echo "Submitted job array ${JOB_ID} (${N_ALPHAS} tasks)"
echo ""
echo "Monitor:"
echo "  squeue -j ${JOB_ID}"
echo "  tail -f ${LOG_DIR}/alpha_0.out"
echo ""
echo "After completion, aggregate:"
echo "  python ${SCRIPT_DIR}/aggregate_pcmci_sweep.py \\"
echo "      --input_dir ${OUTPUT_DIR} --n_components ${N} \\"
echo "      --output_csv ${OUTPUT_DIR}/summary.csv"
