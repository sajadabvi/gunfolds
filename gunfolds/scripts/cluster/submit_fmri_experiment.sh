#!/bin/bash
# =============================================================================
# Submit all fMRI experiment configurations as SLURM array jobs
# =============================================================================
#
# Usage:
#   bash submit_fmri_experiment.sh [N_SUBJECTS]
#
# Example:
#   bash submit_fmri_experiment.sh 310
#
# This script:
#   1. Generates a shared timestamp
#   2. Submits 12 array jobs (one per configuration)
#   3. Prints a summary of all submitted jobs
#
# Configurations:
#   - RASL with N=10,20,53 x SCC={domain,correlation}  =>  6 jobs
#   - PCMCI baseline with N=10,20,53                    =>  3 jobs
#   - GCM baseline with N=10,20,53                      =>  3 jobs
# =============================================================================

set -e

N_SUBJECTS=${1:-310}
LAST_IDX=$((N_SUBJECTS - 1))
TIMESTAMP=$(date +%m%d%Y%H%M%S)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_fmri_large.sh"

# Ensure log directories exist
mkdir -p ./logs ./err ./out

echo "=============================================================="
echo "FMRI EXPERIMENT - BATCH SUBMISSION"
echo "=============================================================="
echo "Timestamp:    $TIMESTAMP"
echo "Subjects:     0-${LAST_IDX} (${N_SUBJECTS} total)"
echo "SLURM script: $SLURM_SCRIPT"
echo "=============================================================="
echo ""

# Track job IDs for the dependency chain (analysis job)
JOB_IDS=()

submit_config() {
    local N_COMP=$1
    local SCC=$2
    local METHOD=$3
    local CONFIG_TAG="N${N_COMP}_${SCC}_${METHOD}"

    # Adjust time and memory based on configuration
    local TIME_LIMIT=7200
    local MEM="4g"
    if [ "$METHOD" = "RASL" ]; then
        if [ "$N_COMP" -ge 53 ]; then
            TIME_LIMIT=28800  # 8h for N=53 RASL
            MEM="16g"
        elif [ "$N_COMP" -ge 20 ]; then
            TIME_LIMIT=14400  # 4h for N=20 RASL
            MEM="8g"
        else
            TIME_LIMIT=7200   # 2h for N=10 RASL
            MEM="8g"
        fi
    fi

    local JOB_ID
    JOB_ID=$(sbatch \
        --array=0-${LAST_IDX} \
        --time=${TIME_LIMIT} \
        --mem=${MEM} \
        --job-name="fmri_${CONFIG_TAG}" \
        "$SLURM_SCRIPT" "$TIMESTAMP" "$N_COMP" "$SCC" "$METHOD" \
        | awk '{print $NF}')

    JOB_IDS+=("$JOB_ID")
    printf "  %-30s  JobID: %s  (time=%ds, mem=%s)\n" "$CONFIG_TAG" "$JOB_ID" "$TIME_LIMIT" "$MEM"
}

echo "Submitting RASL configurations..."
submit_config 10 domain      RASL
submit_config 10 correlation RASL
submit_config 20 domain      RASL
submit_config 20 correlation RASL
submit_config 53 domain      RASL
submit_config 53 correlation RASL

echo ""
echo "Submitting PCMCI baseline configurations..."
submit_config 10 none PCMCI
submit_config 20 none PCMCI
submit_config 53 none PCMCI

echo ""
echo "Submitting GCM baseline configurations..."
submit_config 10 none GCM
submit_config 20 none GCM
submit_config 53 none GCM

echo ""
echo "=============================================================="
echo "SUBMISSION COMPLETE"
echo "=============================================================="
echo "Total jobs submitted: ${#JOB_IDS[@]}"
echo "Shared timestamp:     $TIMESTAMP"
echo "Results will be in:   fbirn_results/$TIMESTAMP/"
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "After all jobs complete, run the analysis:"
echo "  python analyze_fmri_experiment.py --timestamp $TIMESTAMP"
echo "=============================================================="

# Save submission record
RECORD="./logs/submission_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "N_SUBJECTS: $N_SUBJECTS"
    echo "Job IDs: ${JOB_IDS[*]}"
    echo ""
    echo "Configurations:"
    echo "  RASL:  N=10,20,53 x SCC={domain,correlation}"
    echo "  PCMCI: N=10,20,53 x SCC=none"
    echo "  GCM:   N=10,20,53 x SCC=none"
} > "$RECORD"
echo "Submission record: $RECORD"
