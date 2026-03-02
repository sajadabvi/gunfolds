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
# Usage:
#   cd gunfolds/scripts/real_data
#   bash ../cluster/test_fmri_experiment.sh
#
# After jobs finish, check errors:
#   cat ./err/test_fmri_error*.err
#   grep -l "FAILED" ./logs/test_*.log
#   grep -l "SUCCESS" ./logs/test_*.log
#
# If all 12 task logs say SUCCESS, proceed with the full run:
#   bash ../cluster/submit_fmri_experiment.sh 310
# =============================================================================

set -e

TIMESTAMP="test_$(date +%m%d%Y%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
echo "Timestamp:  $TIMESTAMP"
echo "Subjects:   $TEST_SUBJECTS  (3 subjects only)"
echo "Configs:    ${#TEST_CONFIGS[@]}"
echo "Master log: $LOG_FILE"
echo "=============================================================="
echo ""

JOB_IDS=()

for cfg in "${TEST_CONFIGS[@]}"; do
    read N SCC METHOD <<< "$cfg"
    CONFIG_TAG="N${N}_${SCC}_${METHOD}"

    JOB_ID=$(sbatch \
        --array=${TEST_SUBJECTS} \
        -N 1 -n 1 -c 1 \
        --mem=8g \
        -p qTRDGPU \
        -t 3600 \
        -J "test_${CONFIG_TAG}" \
        -e "./err/test_fmri_error_%j_%a_${CONFIG_TAG}.err" \
        -o "./out/test_fmri_out_%j_%a_${CONFIG_TAG}.out" \
        -A psy53c17 \
        --mail-type=ALL \
        --mail-user=mabavisani@gsu.edu \
        --oversubscribe \
        --wrap="
export OMP_NUM_THREADS=1
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo \$HOSTNAME >&2
source /home/users/mabavisani/anaconda3/bin/activate
conda activate multi_v3

SUBJECT_IDX=\$SLURM_ARRAY_TASK_ID
CONFIG_TAG=\"${CONFIG_TAG}\"
TASK_LOG=\"./logs/test_\${CONFIG_TAG}_subj\${SUBJECT_IDX}.log\"

echo \"Task ID:      \$SLURM_ARRAY_TASK_ID\" > \"\$TASK_LOG\"
echo \"Subject:      \$SUBJECT_IDX\"         >> \"\$TASK_LOG\"
echo \"Config:       \$CONFIG_TAG\"           >> \"\$TASK_LOG\"
echo \"Start Time:   \$(date)\"              >> \"\$TASK_LOG\"
echo \"Hostname:     \$HOSTNAME\"             >> \"\$TASK_LOG\"

EXTRA_ARGS=\"\"
if [ \"${METHOD}\" = \"RASL\" ]; then
    EXTRA_ARGS=\"--MAXU 4 --PRIORITY 11112 --selection_mode top_k --top_k 10\"
fi

CMD=\"python fmri_experiment_large.py \\
    --subject_idx \$SUBJECT_IDX \\
    --n_components ${N} \\
    --scc_strategy ${SCC} \\
    --method ${METHOD} \\
    --timestamp ${TIMESTAMP} \\
    \$EXTRA_ARGS\"

echo \"Executing: \$CMD\" >&2
eval \$CMD
EXIT_CODE=\$?

echo \"End Time:   \$(date)\"   >> \"\$TASK_LOG\"
echo \"Exit Code:  \$EXIT_CODE\" >> \"\$TASK_LOG\"
if [ \$EXIT_CODE -eq 0 ]; then
    echo \"Status: SUCCESS\" >> \"\$TASK_LOG\"
else
    echo \"Status: FAILED (exit code \$EXIT_CODE)\" >> \"\$TASK_LOG\"
fi
exit \$EXIT_CODE
" | awk '{print $NF}')

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
echo "   cat ./err/test_fmri_error_*.err"
echo ""
echo "3) Check task outcomes:"
echo "   grep 'Status:' ./logs/test_N*.log"
echo ""
echo "4) Quick pass/fail summary:"
echo "   echo \"SUCCESS: \$(grep -l SUCCESS ./logs/test_N*.log 2>/dev/null | wc -l) / $((${#JOB_IDS[@]} * 3))\""
echo "   echo \"FAILED:  \$(grep -l FAILED  ./logs/test_N*.log 2>/dev/null | wc -l) / $((${#JOB_IDS[@]} * 3))\""
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
