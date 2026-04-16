#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Live resource monitor for fmri_large SLURM array job
#
# Shows real-time CPU%, memory, and process info for running tasks
# via sstat and ssh to compute nodes.
#
# Usage:
#   bash slurm_live_fmri.sh              # auto-detect
#   bash slurm_live_fmri.sh 3133801      # explicit job ID
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

JOBID="${1:-}"

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
echo " RUNNING TASKS OVERVIEW  ($(date))"
echo "================================================================"
running_info=$(squeue -j "$JOBID" --states=R \
    --format="%i %N %M %l %C %m" --noheader 2>/dev/null || true)

if [[ -z "$running_info" ]]; then
    echo "No tasks currently RUNNING."
    exit 0
fi

TOTAL_RUNNING=$(echo "$running_info" | wc -l)
echo "Total running: $TOTAL_RUNNING"
echo
printf "%-24s %-15s %-14s %-14s %-6s %-10s\n" "JOBID" "NODE" "ELAPSED" "TIMELIMIT" "CPUS" "MEM"
echo "$running_info" | while read -r jid node elapsed tlimit cpus mem; do
    printf "%-24s %-15s %-14s %-14s %-6s %-10s\n" "$jid" "$node" "$elapsed" "$tlimit" "$cpus" "$mem"
done | head -50
(( TOTAL_RUNNING > 50 )) && echo "  ... and $((TOTAL_RUNNING - 50)) more"

echo
echo "================================================================"
echo " MEMORY & CPU via sstat  (sample of 20 running tasks)"
echo "================================================================"
printf "%-24s %14s %14s %14s\n" "JOBID" "AveCPU" "MaxRSS" "MaxVMSize"
echo "$running_info" | head -20 | while read -r jid node elapsed tlimit cpus mem; do
    line=$(sstat -j "${jid}.batch" \
        --format="JobID%24,AveCPU%14,MaxRSS%14,MaxVMSize%14" \
        --noheader 2>/dev/null | head -1 || true)
    [[ -n "$line" ]] && echo "$line"
done

echo
echo "================================================================"
echo " NODE UTILIZATION SUMMARY"
echo "================================================================"
echo "Tasks per node:"
echo "$running_info" | awk '{print $2}' | sort | uniq -c | sort -rn

echo
echo "================================================================"
echo " TOP PROCESSES ON EACH COMPUTE NODE  (ssh + ps)"
echo "================================================================"
nodes=$(echo "$running_info" | awk '{print $2}' | sort -u)
for node in $nodes; do
    tasks_on_node=$(echo "$running_info" | awk -v n="$node" '$2==n' | wc -l)
    echo "--- $node ($tasks_on_node tasks) ---"
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$node" \
        "echo 'Load average:'; uptime; echo; \
         echo 'Memory:'; free -h; echo; \
         echo 'Your processes (top 15 by mem):'; \
         ps -u $USER -o pid,state,%cpu,%mem,etime,rss:12,args --sort=-%mem | head -16" \
        2>/dev/null || echo "  (ssh to $node failed)"
    echo
done

echo
echo "Done.  $(date)"
