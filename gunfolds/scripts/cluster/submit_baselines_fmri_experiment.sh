#!/bin/bash
# =============================================================================
# Submit ALL baseline methods (GIMME, MVAR, MVGC, FASK) on FBIRN for N=10,13.
# =============================================================================
# Records one binary directed graph per subject in the SAME result.zkl format as
# the refactored RASL/PCMCI runs, under a shared timestamp, so that
#   analysis/refactored_analyze_fmri_experiment.py --timestamp <TS> \
#       --results_root fbirn_results_refactored ...
# analyses baselines side-by-side with RASL/PCMCI.
#
# To compare against an EXISTING RASL/PCMCI run, pass that run's timestamp:
#   TIMESTAMP=06042026120000 bash submit_baselines_fmri_experiment.sh
# Otherwise a fresh timestamp is created.
#
# Per-method execution:
#   FASK  : SLURM array (one subject/task) via slurm_baselines_fask.sh
#   MVGC/ : one job per (N,method)         via slurm_baselines_matlab.sh
#   MVAR    (export -> MATLAB -> collect)
#   GIMME : one job per N  (pooled)        via slurm_baselines_gimme.sh
#
# Usage:
#   bash submit_baselines_fmri_experiment.sh [N_SUBJECTS]
# Env:
#   TIMESTAMP, METHODS ("GIMME MVAR MVGC FASK"), NCOMPS ("10 13"),
#   RESULTS_ROOT (fbirn_results_refactored), MAX_PARALLEL (50)
# =============================================================================

N_SUBJECTS=${1:-311}
LAST_IDX=$((N_SUBJECTS - 1))
TIMESTAMP=${TIMESTAMP:-$(date +%m%d%Y%H%M%S)}
METHODS=${METHODS:-"GIMME MVAR MVGC FASK"}
NCOMPS=${NCOMPS:-"10 13"}
RESULTS_ROOT=${RESULTS_ROOT:-fbirn_results_refactored}
MAX_PARALLEL=${MAX_PARALLEL:-50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ./logs ./err ./out
export RESULTS_ROOT

echo "=============================================================="
echo "BASELINES BATCH SUBMISSION (GIMME/MVAR/MVGC/FASK)"
echo "=============================================================="
echo "Timestamp:   $TIMESTAMP"
echo "Methods:     $METHODS"
echo "N comps:     $NCOMPS"
echo "Subjects:    0-${LAST_IDX} (${N_SUBJECTS})"
echo "Output:      $RESULTS_ROOT/$TIMESTAMP/"
echo "=============================================================="

JOB_IDS=()
submit() {  # echoes job id
    local jid
    jid=$(sbatch "$@" | awk '{print $NF}')
    JOB_IDS+=("$jid")
    echo "$jid"
}

for N in $NCOMPS; do
    for M in $METHODS; do
        case "$M" in
            FASK)
                JID=$(submit --array=0-${LAST_IDX}%${MAX_PARALLEL} \
                    "${SCRIPT_DIR}/slurm_baselines_fask.sh" "$TIMESTAMP" "$N")
                printf "  %-18s array=0-%-4s  JobID: %s\n" "N${N}_FASK" "$LAST_IDX" "$JID" ;;
            MVGC|MVAR)
                JID=$(submit "${SCRIPT_DIR}/slurm_baselines_matlab.sh" "$TIMESTAMP" "$N" "$M")
                printf "  %-18s (matlab)      JobID: %s\n" "N${N}_${M}" "$JID" ;;
            GIMME)
                JID=$(submit "${SCRIPT_DIR}/slurm_baselines_gimme.sh" "$TIMESTAMP" "$N")
                printf "  %-18s (R/pooled)    JobID: %s\n" "N${N}_GIMME" "$JID" ;;
            *)
                echo "  WARN: unknown method '$M' (skipped)" ;;
        esac
    done
done

echo ""
echo "=============================================================="
echo "SUBMISSION COMPLETE"
echo "  Timestamp: $TIMESTAMP"
echo "  Job IDs:   ${JOB_IDS[*]}"
echo ""
echo "After completion, analyse alongside RASL/PCMCI with:"
echo "  python ../analysis/refactored_analyze_fmri_experiment.py \\"
echo "      --timestamp $TIMESTAMP --results_root $RESULTS_ROOT \\"
echo "      --correction fdr --plot"
echo "=============================================================="

RECORD="./logs/baselines_submission_${TIMESTAMP}.log"
{
    echo "Timestamp: $TIMESTAMP"
    echo "Submitted: $(date)"
    echo "Methods:   $METHODS"
    echo "N comps:   $NCOMPS"
    echo "Subjects:  ${N_SUBJECTS}"
    echo "Root:      $RESULTS_ROOT"
    echo "Job IDs:   ${JOB_IDS[*]}"
} > "$RECORD"
echo "Submission record: $RECORD"
