#!/bin/bash
# Status checker for the runtime_scaling experiment.
#
# Cross-references three sources:
#   1. results/runtime_scaling/n{N}_inst{I}.csv  — completed/errored runs
#   2. squeue -u $USER                            — currently queued/running
#   3. nothing of the above                       — missing
#
# Usage:
#   bash check_status.sh           # summary table
#   bash check_status.sh --full    # also list every (N, inst) state
#   bash check_status.sh --watch   # auto-refresh every 30s

set -u

OUTPUT_DIR="results/runtime_scaling"
N_VALUES=(8 10 12 14 18 20 24 30 42 54)
INSTANCES_PER_N=10

FULL=0
WATCH=0
for arg in "$@"; do
    case "$arg" in
        --full)  FULL=1 ;;
        --watch) WATCH=1 ;;
    esac
done

show_status() {
    # Snapshot the squeue once (cheap)
    local squeue_snap
    squeue_snap=$(squeue -u "$USER" -h -o "%j %T" 2>/dev/null || true)

    declare -A STATE_COUNT
    STATE_COUNT[completed]=0
    STATE_COUNT[error]=0
    STATE_COUNT[running]=0
    STATE_COUNT[pending]=0
    STATE_COUNT[missing]=0

    declare -A PER_N_DONE
    declare -A PER_N_ERR
    declare -A PER_N_RUN
    declare -A PER_N_PEND
    declare -A PER_N_MISS

    local detail=""

    for N in "${N_VALUES[@]}"; do
        PER_N_DONE[$N]=0
        PER_N_ERR[$N]=0
        PER_N_RUN[$N]=0
        PER_N_PEND[$N]=0
        PER_N_MISS[$N]=0

        for ((I=0; I<INSTANCES_PER_N; I++)); do
            local job_name="rt_n${N}_i${I}"
            local csv="${OUTPUT_DIR}/n${N}_inst${I}.csv"
            local state="missing"
            local extra=""

            # 1. Live queue state wins over historical CSV.  If a job with this
            #    name is currently queued/running, the CSV (if any) reflects an
            #    earlier run that's about to be overwritten — so report the
            #    in-flight state instead.
            local sq_state
            sq_state=$(echo "$squeue_snap" | awk -v n="$job_name" '$1==n {print $2; exit}')
            if [ -n "$sq_state" ]; then
                case "$sq_state" in
                    R|RUNNING)        state="running"; extra="(in queue: RUNNING)" ;;
                    PD|PENDING)       state="pending"; extra="(in queue: PENDING)" ;;
                    CG|COMPLETING)    state="running"; extra="(in queue: COMPLETING)" ;;
                    *)                state="running"; extra="(in queue: $sq_state)" ;;
                esac
                # Note if there's a historical CSV that will be overwritten.
                if [ -f "$csv" ]; then
                    local prev
                    prev=$(tail -n 1 "$csv" 2>/dev/null | awk -F',' '{print $NF}' | cut -c1-30)
                    extra="$extra  [prev: $prev]"
                fi
            elif [ -f "$csv" ]; then
                # 2. No live job — CSV is authoritative.
                local last
                last=$(tail -n 1 "$csv" 2>/dev/null)
                if echo "$last" | grep -q ',completed$'; then
                    state="completed"
                    # Pull drasl_time and total_time from the row (cols 6 and 7).
                    local drasl_time total_time
                    drasl_time=$(echo "$last" | awk -F',' '{print $6}')
                    total_time=$(echo "$last" | awk -F',' '{print $7}')
                    extra="drasl=${drasl_time}s total=${total_time}s"
                elif echo "$last" | grep -q ',timeout'; then
                    state="error"
                    extra="(timeout)"
                else
                    state="error"
                    extra=$(echo "$last" | awk -F',' '{print $NF}' | cut -c1-60)
                fi
            fi
            # 3. Else: state stays "missing" (no CSV and not in queue).

            # Tally.
            STATE_COUNT[$state]=$(( ${STATE_COUNT[$state]:-0} + 1 ))
            case "$state" in
                completed) PER_N_DONE[$N]=$(( ${PER_N_DONE[$N]} + 1 )) ;;
                error)     PER_N_ERR[$N]=$(( ${PER_N_ERR[$N]} + 1 )) ;;
                running)   PER_N_RUN[$N]=$(( ${PER_N_RUN[$N]} + 1 )) ;;
                pending)   PER_N_PEND[$N]=$(( ${PER_N_PEND[$N]} + 1 )) ;;
                missing)   PER_N_MISS[$N]=$(( ${PER_N_MISS[$N]} + 1 )) ;;
            esac

            if [ "$FULL" = "1" ]; then
                local mark
                case "$state" in
                    completed) mark="✓ completed" ;;
                    error)     mark="✗ error    " ;;
                    running)   mark="▶ running  " ;;
                    pending)   mark="⏳ pending  " ;;
                    missing)   mark="? missing  " ;;
                esac
                detail+=$(printf "  N=%-2d inst=%d  %s  %s\n" "$N" "$I" "$mark" "$extra")
                detail+=$'\n'
            fi
        done
    done

    # ─── Output ───────────────────────────────────────────────────────────
    echo "=== runtime_scaling status @ $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo

    # Per-N table
    printf "  %-4s | %-9s | %-6s | %-7s | %-7s | %-7s\n" \
        "N" "completed" "error" "running" "pending" "missing"
    printf "  -----+-----------+--------+---------+---------+--------\n"
    for N in "${N_VALUES[@]}"; do
        printf "  %-4d | %-9s | %-6s | %-7s | %-7s | %-7s\n" \
            "$N" \
            "${PER_N_DONE[$N]}/$INSTANCES_PER_N" \
            "${PER_N_ERR[$N]}" \
            "${PER_N_RUN[$N]}" \
            "${PER_N_PEND[$N]}" \
            "${PER_N_MISS[$N]}"
    done

    # Grand total
    local total=$(( INSTANCES_PER_N * ${#N_VALUES[@]} ))
    echo
    printf "  TOTAL: %d completed / %d  (errors=%d  running=%d  pending=%d  missing=%d)\n" \
        "${STATE_COUNT[completed]}" "$total" \
        "${STATE_COUNT[error]}" \
        "${STATE_COUNT[running]}" \
        "${STATE_COUNT[pending]}" \
        "${STATE_COUNT[missing]}"

    # Optional full per-instance detail
    if [ "$FULL" = "1" ]; then
        echo
        echo "=== per-instance detail ==="
        echo "$detail"
    fi
}

if [ "$WATCH" = "1" ]; then
    while true; do
        clear
        show_status
        echo
        echo "(refreshing every 30s — Ctrl-C to exit)"
        sleep 30
    done
else
    show_status
fi
