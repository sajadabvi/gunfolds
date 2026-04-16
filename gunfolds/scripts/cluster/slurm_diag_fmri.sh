#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# SLURM diagnostic dashboard for fmri_large array job (3133801)
#
# Usage:
#   bash slurm_diag_fmri.sh              # auto-detect latest fmri_large job
#   bash slurm_diag_fmri.sh 3133801      # explicit job ID
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

JOBID="${1:-}"
MAX_TASK=309

if [[ -z "$JOBID" ]]; then
    JOBID=$(sacct -u "$USER" --name=fmri_large --format=JobID -n -X \
            | tail -1 | tr -d ' ')
    if [[ -z "$JOBID" ]]; then
        echo "ERROR: Could not auto-detect job ID. Pass it explicitly."
        exit 1
    fi
    echo "Auto-detected array job: $JOBID"
fi

echo
echo "================================================================"
echo " 1. QUEUE STATUS  (squeue)"
echo "================================================================"
squeue -j "$JOBID" \
    --format="%.12i %.9P %.22j %.8u %.2t %.12M %.12l %.6D %.4C %.10m %.20R" \
    2>/dev/null | head -60
TOTAL_QUEUE=$(squeue -j "$JOBID" --noheader 2>/dev/null | wc -l)
echo "  (showing first 60 — total in queue: $TOTAL_QUEUE)"

echo
echo "================================================================"
echo " 2. STATE DISTRIBUTION"
echo "================================================================"
sacct -j "$JOBID" -X --format=State -n 2>/dev/null \
    | sort | uniq -c | sort -rn

echo
echo "================================================================"
echo " 3. ACCOUNTING SUMMARY  (completed/failed tasks)"
echo "================================================================"
echo "--- COMPLETED ---"
sacct -j "$JOBID" -X --state=COMPLETED \
    --format="JobID%20,JobName%28,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode" \
    --units=G -n 2>/dev/null | head -20
echo "($(sacct -j "$JOBID" -X --state=COMPLETED -n 2>/dev/null | wc -l) total COMPLETED)"

echo
echo "--- FAILED ---"
sacct -j "$JOBID" -X --state=FAILED \
    --format="JobID%20,JobName%28,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode" \
    --units=G -n 2>/dev/null | head -20
echo "($(sacct -j "$JOBID" -X --state=FAILED -n 2>/dev/null | wc -l) total FAILED)"

echo
echo "--- TIMEOUT ---"
sacct -j "$JOBID" -X --state=TIMEOUT \
    --format="JobID%20,JobName%28,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode" \
    --units=G -n 2>/dev/null | head -20
echo "($(sacct -j "$JOBID" -X --state=TIMEOUT -n 2>/dev/null | wc -l) total TIMEOUT)"

echo
echo "--- OUT_OF_MEMORY ---"
sacct -j "$JOBID" -X --state=OUT_OF_MEMORY \
    --format="JobID%20,JobName%28,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode" \
    --units=G -n 2>/dev/null | head -20
echo "($(sacct -j "$JOBID" -X --state=OUT_OF_MEMORY -n 2>/dev/null | wc -l) total OOM)"

echo
echo "================================================================"
echo " 4. RUNNING TASKS — ELAPSED vs LIMIT"
echo "================================================================"
sacct -j "$JOBID" -X --state=RUNNING \
    --format="JobID%20,JobName%28,Elapsed,Timelimit,ReqMem,AllocCPUS,NodeList%15" \
    -n 2>/dev/null | head -30
TOTAL_RUNNING=$(sacct -j "$JOBID" -X --state=RUNNING -n 2>/dev/null | wc -l)
echo "  ($TOTAL_RUNNING total RUNNING)"

echo
echo "================================================================"
echo " 5. PENDING TASKS"
echo "================================================================"
TOTAL_PENDING=$(squeue -j "$JOBID" --states=PD --noheader 2>/dev/null | wc -l)
echo "  $TOTAL_PENDING tasks pending in queue"
if (( TOTAL_PENDING > 0 )); then
    squeue -j "$JOBID" --states=PD \
        --format="%.12i %.22j %.2t %.12l %R" \
        --noheader 2>/dev/null | head -10
fi

echo
echo "================================================================"
echo " 6. EFFICIENCY SAMPLE  (seff on 10 running tasks)"
echo "================================================================"
running_ids=$(squeue -j "$JOBID" --states=R --format="%i" --noheader 2>/dev/null | head -10)
for jid in $running_ids; do
    echo "--- $jid ---"
    seff "$jid" 2>/dev/null \
        | grep -E "State|CPU Utilized|CPU Efficiency|Memory Utilized|Memory Efficiency|Wall|Cores" \
        || echo "  (seff unavailable)"
    echo
done

echo
echo "================================================================"
echo " 7. MEMORY HIGH-WATER  (sstat on running tasks, sample 10)"
echo "================================================================"
for jid in $running_ids; do
    sstat -j "${jid}.batch" \
        --format="JobID%20,AveCPU,MaxRSS,MaxVMSize" \
        --noheader 2>/dev/null || true
done

echo
echo "================================================================"
echo " 8. OUTPUT / RESULT FILE INVENTORY"
echo "================================================================"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/../real_data"
latest_dir=$(ls -dt "$RESULTS_DIR"/fbirn_results/* 2>/dev/null | head -1)
if [[ -n "$latest_dir" ]]; then
    echo "  Latest results dir: $latest_dir"
    result_count=$(find "$latest_dir" -name "*.npz" -o -name "*.csv" 2>/dev/null | wc -l)
    echo "  Result files found: $result_count"
else
    echo "  (no fbirn_results directory found)"
fi

echo
echo "================================================================"
echo " 9. RECENT stderr TAIL  (last 5 non-empty error files)"
echo "================================================================"
ERR_DIR="$SCRIPT_DIR/../real_data/err"
if [[ -d "$ERR_DIR" ]]; then
    count=0
    for f in $(ls -t "$ERR_DIR"/fmri_error"${JOBID}"-*.err 2>/dev/null); do
        [[ -f "$f" ]] || continue
        sz=$(wc -c < "$f")
        (( sz < 10 )) && continue
        echo "--- $(basename "$f") ($sz bytes) ---"
        tail -8 "$f"
        echo
        count=$((count + 1))
        (( count >= 5 )) && break
    done
    (( count == 0 )) && echo "  (all error files empty or missing)"
else
    echo "  (err/ directory not found at $ERR_DIR)"
fi

echo
echo "================================================================"
echo " 10. RECENT stdout TAIL  (last 5 output files)"
echo "================================================================"
OUT_DIR="$SCRIPT_DIR/../real_data/out"
if [[ -d "$OUT_DIR" ]]; then
    count=0
    for f in $(ls -t "$OUT_DIR"/fmri_out"${JOBID}"-*.out 2>/dev/null); do
        [[ -f "$f" ]] || continue
        sz=$(wc -c < "$f")
        (( sz < 10 )) && continue
        echo "--- $(basename "$f") ($sz bytes) ---"
        tail -8 "$f"
        echo
        count=$((count + 1))
        (( count >= 5 )) && break
    done
    (( count == 0 )) && echo "  (all output files empty or missing)"
else
    echo "  (out/ directory not found at $OUT_DIR)"
fi

echo
echo "================================================================"
echo " 11. LOG FILES"
echo "================================================================"
LOG_DIR="$SCRIPT_DIR/../real_data/logs"
if [[ -d "$LOG_DIR" ]]; then
    master_log=$(ls -t "$LOG_DIR"/fmri_master_${JOBID}*.log 2>/dev/null | head -1)
    if [[ -n "$master_log" ]]; then
        echo "  Master log: $master_log"
        cat "$master_log"
    fi
    echo
    completed_tasks=$(ls "$LOG_DIR"/fmri_task_*_subj*.log 2>/dev/null | wc -l)
    echo "  Task log files: $completed_tasks"
    # Show a few completed ones
    grep -l "Status: SUCCESS" "$LOG_DIR"/fmri_task_*_subj*.log 2>/dev/null | wc -l | \
        xargs -I{} echo "  Successful: {}"
    grep -l "Status: FAILED" "$LOG_DIR"/fmri_task_*_subj*.log 2>/dev/null | wc -l | \
        xargs -I{} echo "  Failed:     {}"
else
    echo "  (logs/ directory not found)"
fi

echo
echo "Done.  $(date)"
