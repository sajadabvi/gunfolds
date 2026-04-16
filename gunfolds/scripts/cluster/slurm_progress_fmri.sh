#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Progress + stall detection for fmri_large array job
#
# Checks output logs to estimate how far each task has gotten
# and flags stalled tasks (no output update in >30 min).
#
# Usage:
#   bash slurm_progress_fmri.sh              # auto-detect
#   bash slurm_progress_fmri.sh 3133801      # explicit job ID
#   bash slurm_progress_fmri.sh 3133801 60   # stall threshold = 60 min
# ─────────────────────────────────────────────────────────────────
set -euo pipefail

JOBID="${1:-}"
STALL_THRESHOLD="${2:-30}"   # minutes

if [[ -z "$JOBID" ]]; then
    JOBID=$(sacct -u "$USER" --name=fmri_large --format=JobID -n -X \
            | tail -1 | tr -d ' ')
    if [[ -z "$JOBID" ]]; then
        echo "ERROR: Could not auto-detect job ID."
        exit 1
    fi
    echo "Auto-detected array job: $JOBID"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$SCRIPT_DIR/../real_data/out"
ERR_DIR="$SCRIPT_DIR/../real_data/err"

echo
echo "================================================================"
echo " PROGRESS SUMMARY  (from stdout logs)"
echo "================================================================"

completed=0
running=0
failed=0
timeout=0
pending=0
other=0
no_output=0
stalled=0
now=$(date +%s)

# Collect per-task info into a temp file for reporting
tmpfile=$(mktemp)
trap "rm -f $tmpfile" EXIT

for tid in $(seq 0 309); do
    taskjob="${JOBID}_${tid}"
    state=$(sacct -j "$taskjob" --format=State -n -X 2>/dev/null | head -1 | tr -d ' ')
    [[ -z "$state" ]] && state="UNKNOWN"

    outfile="$OUT_DIR/fmri_out${JOBID}-${tid}.out"
    if [[ -f "$outfile" ]]; then
        nlines=$(wc -l < "$outfile")
        filesize=$(wc -c < "$outfile")
        lastline=$(tail -1 "$outfile" | cut -c1-50)
        mtime=$(stat -c %Y "$outfile" 2>/dev/null || stat -f %m "$outfile" 2>/dev/null)
        age_min=$(( (now - mtime) / 60 ))
    else
        nlines=0
        filesize=0
        lastline="(no output file)"
        age_min=-1
    fi

    is_stalled="no"
    if [[ "$state" == "RUNNING" ]] && (( age_min > STALL_THRESHOLD )); then
        is_stalled="YES"
        stalled=$((stalled + 1))
    fi

    case "$state" in
        COMPLETED*) completed=$((completed + 1)) ;;
        RUNNING*)   running=$((running + 1)) ;;
        FAILED*)    failed=$((failed + 1)) ;;
        TIMEOUT*)   timeout=$((timeout + 1)) ;;
        PENDING*)   pending=$((pending + 1)) ;;
        *)          other=$((other + 1)) ;;
    esac

    (( nlines == 0 )) && no_output=$((no_output + 1))

    echo "$tid|$state|$nlines|$filesize|$age_min|$is_stalled|$lastline" >> "$tmpfile"
done

total=$((completed + running + failed + timeout + pending + other))
echo "  Total tasks : $total / 310"
echo "  COMPLETED   : $completed"
echo "  RUNNING     : $running"
echo "  FAILED      : $failed"
echo "  TIMEOUT     : $timeout"
echo "  PENDING     : $pending"
echo "  OTHER       : $other"
echo "  No output   : $no_output"
echo

echo "================================================================"
echo " STALL DETECTION  (output not updated in >${STALL_THRESHOLD} min)"
echo "================================================================"
if (( stalled > 0 )); then
    echo "  $stalled potentially stalled tasks:"
    printf "  %-6s %-12s %-8s %-10s %-50s\n" "TASK" "STATE" "LINES" "STALE_MIN" "LAST_LINE"
    grep "|YES|" "$tmpfile" | while IFS='|' read -r tid state nlines fsize age stall lastline; do
        printf "  %-6s %-12s %-8s %-10s %-50s\n" "$tid" "$state" "$nlines" "${age}m" "$lastline"
    done
else
    echo "  (none detected)"
fi

echo
echo "================================================================"
echo " RUNNING TASKS — OUTPUT SIZE DISTRIBUTION"
echo "================================================================"
echo "  (sorted by output lines, descending — shows where each task is)"
printf "  %-6s %-8s %-10s %-10s %-50s\n" "TASK" "LINES" "BYTES" "STALE" "LAST_LINE"
grep "|RUNNING" "$tmpfile" | sort -t'|' -k3 -rn | head -30 | \
    while IFS='|' read -r tid state nlines fsize age stall lastline; do
        printf "  %-6s %-8s %-10s %-10s %-50s\n" "$tid" "$nlines" "$fsize" "${age}m" "$lastline"
    done
TOTAL_RUNNING_SHOWN=$(grep -c "|RUNNING" "$tmpfile" 2>/dev/null || echo 0)
(( TOTAL_RUNNING_SHOWN > 30 )) && echo "  ... and $((TOTAL_RUNNING_SHOWN - 30)) more running tasks"

echo
echo "================================================================"
echo " FAILED / TIMED-OUT TASKS — ERROR DETAILS"
echo "================================================================"
grep -E "\|FAILED|\|TIMEOUT" "$tmpfile" | head -20 | \
    while IFS='|' read -r tid state nlines fsize age stall lastline; do
        errfile="$ERR_DIR/fmri_error${JOBID}-${tid}.err"
        echo "--- task $tid ($state) ---"
        if [[ -f "$errfile" ]] && (( $(wc -c < "$errfile") > 0 )); then
            echo "  stderr (last 5 lines):"
            tail -5 "$errfile" | sed 's/^/    /'
        else
            echo "  (no stderr)"
        fi
        echo "  stdout last line: $lastline"
        echo
    done

echo
echo "Done.  $(date)"
