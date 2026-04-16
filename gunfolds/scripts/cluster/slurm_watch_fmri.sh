#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Continuous watch loop for fmri_large array job
#
# Usage:
#   bash slurm_watch_fmri.sh              # 60s interval, auto-detect
#   bash slurm_watch_fmri.sh 3133801 30   # explicit job, 30s interval
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

JOBID="${1:-}"
INTERVAL="${2:-60}"

if [[ -z "$JOBID" ]]; then
    JOBID=$(sacct -u "$USER" --name=fmri_large --format=JobID -n -X \
            | tail -1 | tr -d ' ')
    if [[ -z "$JOBID" ]]; then
        echo "ERROR: Could not auto-detect job ID."
        exit 1
    fi
fi

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  FMRI LARGE WATCH — Job $JOBID — $(date '+%H:%M:%S')  ║"
    echo "║  Refreshing every ${INTERVAL}s — Ctrl-C to stop                    ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo

    # State summary
    echo "── State Distribution ──"
    sacct -j "$JOBID" -X --format=State -n 2>/dev/null \
        | sort | uniq -c | sort -rn
    echo

    # Running count and elapsed range
    echo "── Running Tasks ──"
    running_info=$(squeue -j "$JOBID" --states=R \
        --format="%i %N %M" --noheader 2>/dev/null || true)
    if [[ -n "$running_info" ]]; then
        total_r=$(echo "$running_info" | wc -l)
        min_elapsed=$(echo "$running_info" | awk '{print $3}' | sort | head -1)
        max_elapsed=$(echo "$running_info" | awk '{print $3}' | sort | tail -1)
        echo "  Count: $total_r   Elapsed range: $min_elapsed — $max_elapsed"
        echo "  Nodes:"
        echo "$running_info" | awk '{print $2}' | sort | uniq -c | sort -rn | sed 's/^/    /'
    else
        echo "  (none)"
    fi
    echo

    # Pending
    pend=$(squeue -j "$JOBID" --states=PD --noheader 2>/dev/null | wc -l)
    echo "── Pending: $pend ──"
    echo

    # Memory high-water (sample 5)
    echo "── Memory High-Water (sample 5 running) ──"
    echo "$running_info" 2>/dev/null | head -5 | while read -r jid rest; do
        [[ -z "$jid" ]] && continue
        line=$(sstat -j "${jid}.batch" \
            --format="JobID%22,MaxRSS%14,MaxVMSize%14" \
            --noheader 2>/dev/null | head -1 || true)
        [[ -n "$line" ]] && echo "  $line"
    done
    echo

    # Completed count
    comp=$(sacct -j "$JOBID" -X --state=COMPLETED -n 2>/dev/null | wc -l)
    fail=$(sacct -j "$JOBID" -X --state=FAILED -n 2>/dev/null | wc -l)
    tout=$(sacct -j "$JOBID" -X --state=TIMEOUT -n 2>/dev/null | wc -l)
    echo "── Finished: $comp completed, $fail failed, $tout timeout ──"

    sleep "$INTERVAL"
done
