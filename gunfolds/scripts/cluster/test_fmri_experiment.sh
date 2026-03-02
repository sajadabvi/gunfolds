#!/bin/bash
# =============================================================================
# TEST fMRI Experiment - Smoke test a few configs before full launch
# =============================================================================
#
# Runs 3 subjects (0, 1, 2) on 4 representative configs to verify:
#   - Data loads correctly
#   - RASL (with domain SCC) works at N=10
#   - PCMCI baseline works at N=10
#   - GCM baseline works at N=10
#   - RASL with correlation SCC works at N=10
#
# This script submits via slurm_fmri_large.sh (the real worker script),
# so it tests the exact same code path as the full experiment.
#
# Usage:
#   cd gunfolds/scripts/real_data
#   bash ../cluster/test_fmri_experiment.sh
#
# After jobs finish, check errors:
#   cat ./err/fmri_error*.err
#   grep 'Status:' ./logs/fmri_task_N*.log
#
# If all 12 task logs say SUCCESS, proceed with the full run:
#   bash ../cluster/submit_fmri_experiment.sh 310
# =============================================================================

set -e

TIMESTAMP="test_$(date +%m%d%Y%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_fmri_large.sh"
PYTHON_SCRIPT="fmri_experiment_large.py"

# Test subjects: first 3
TEST_SUBJECTS="0-2"

# Test configs: one of each method + one extra SCC strategy
TEST_CONFIGS=(
    "10 domain      RASL"
    "10 correlation RASL"
    "10 none        PCMCI"
    "10 none        GCM"
)

# Ensure directories
mkdir -p ./logs ./err ./out

# =============================================================================
# Master log (following jobsubmit standard)
# =============================================================================
LOG_DIR="./logs"
LOG_FILE="$LOG_DIR/test_master_${TIMESTAMP}.log"
{
    echo "Date and Time: $(date)"
    echo "**********************************************************"
    echo "TEST RUN - fMRI Experiment Smoke Test"
    echo "Timestamp:  $TIMESTAMP"
    echo "Subjects:   $TEST_SUBJECTS"
    echo "**********************************************************"
    echo -e "\nContents of this script:"
    cat "$0"
    echo "**********************************************************"
    echo -e "\nContents of $PYTHON_SCRIPT (first 50 lines):"
    head -50 "$PYTHON_SCRIPT"
    echo "..."
    echo "**********************************************************"
    echo -e "\nConfigs to test:"
    for cfg in "${TEST_CONFIGS[@]}"; do
        echo "  $cfg"
    done
    echo "**********************************************************"
} > "$LOG_FILE"

echo "=============================================================="
echo "FMRI EXPERIMENT - SMOKE TEST"
echo "=============================================================="
echo "Timestamp:    $TIMESTAMP"
echo "Subjects:     $TEST_SUBJECTS  (3 subjects only)"
echo "Configs:      ${#TEST_CONFIGS[@]}"
echo "SLURM script: $SLURM_SCRIPT"
echo "Master log:   $LOG_FILE"
echo "=============================================================="
echo ""

JOB_IDS=()

for cfg in "${TEST_CONFIGS[@]}"; do
    read N SCC METHOD <<< "$cfg"
    CONFIG_TAG="N${N}_${SCC}_${METHOD}"

    JOB_ID=$(sbatch \
        --array=${TEST_SUBJECTS} \
        --time=3600 \
        --mem=8g \
        --job-name="test_${CONFIG_TAG}" \
        "$SLURM_SCRIPT" "$TIMESTAMP" "$N" "$SCC" "$METHOD" \
        | awk '{print $NF}')

    JOB_IDS+=("$JOB_ID")
    printf "  %-30s  JobID: %s\n" "$CONFIG_TAG" "$JOB_ID"
done

echo ""
echo "=============================================================="
echo "TEST SUBMITTED  (${#JOB_IDS[@]} array jobs x 3 subjects = $((${#JOB_IDS[@]} * 3)) tasks)"
echo "=============================================================="
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "--- HOW TO CHECK ---"
echo ""
echo "1) Watch progress:"
echo "   squeue -u \$USER"
echo ""
echo "2) After jobs finish, check for errors:"
echo "   cat ./err/fmri_error*.err"
echo ""
echo "3) Check task outcomes:"
echo "   grep 'Status:' ./logs/fmri_task_N*.log"
echo ""
echo "4) Quick pass/fail summary:"
echo "   echo \"SUCCESS: \$(grep -c SUCCESS ./logs/fmri_task_N10_*.log 2>/dev/null) / $((${#JOB_IDS[@]} * 3))\""
echo "   echo \"FAILED:  \$(grep -c FAILED  ./logs/fmri_task_N10_*.log 2>/dev/null) / $((${#JOB_IDS[@]} * 3))\""
echo ""
echo "5) Check result files were created:"
echo "   find fbirn_results/$TIMESTAMP -name 'result.zkl' | wc -l"
echo "   (expect 12 = 4 configs x 3 subjects)"
echo ""
echo "6) If all good, run the full experiment:"
echo "   bash ../cluster/submit_fmri_experiment.sh 310"
echo "=============================================================="

# Append job IDs to master log
{
    echo ""
    echo "Submitted Job IDs: ${JOB_IDS[*]}"
    echo "Submission Time:   $(date)"
} >> "$LOG_FILE"
