#!/bin/bash
# =============================================================================
# Submit the PER-U split of the drasl runtime-scaling experiment.
#
# For every (n_nodes, instance) pair this submits a 3-stage SLURM pipeline:
#
#   1. ONE  prep  job   — seed→G¹→VAR→BOLD→PCMCI→DD/BD, pickles the drasl input
#                         (light, short walltime).
#   2. MANY solve jobs  — one clingo job per fixed undersampling rate u, each
#                         `afterok` the prep job.  These are the heavy,
#                         embarrassingly-parallel jobs (one per u, all reading
#                         the same prepped input).
#   3. ONE  aggregate   — pools all per-u solutions and keeps the top 30 % by
#                         cost, `afterok` all the solve jobs.
#
# Each pipeline is identified by a UNIQUE run_tag = "${LABEL}_n${N}_inst${I}",
# so arbitrarily many sweeps can coexist in one output dir without collision.
#
# Usage:
#   bash submit_runtime_scaling_per_u.sh
#   U_VALUES="2 3 4 5" LABEL=myexp bash submit_runtime_scaling_per_u.sh
#   FORCE=1 bash submit_runtime_scaling_per_u.sh          # ignore skip guards
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/runtime_scaling_per_u.py"
AGGREGATOR="${SCRIPT_DIR}/aggregate_per_u.py"

OUTPUT_DIR="results/runtime_scaling_per_u"
LOG_DIR="logs/runtime_scaling_per_u"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ─── SLURM constants (match submit_runtime_scaling.sh) ───────────────────────
PARTITION="qTRDGPU"
ACCOUNT="psy53c17"
EMAIL="mabavisani@gsu.edu"
SOLVE_WALLTIME="36:00:00"
PREP_WALLTIME="04:00:00"
AGG_WALLTIME="00:30:00"
SOLVE_CPUS=16          # 15 clingo threads + 1 buffer
PREP_CPUS=4
AGG_CPUS=1
TIMEOUT_HOURS=35       # clingo internal timeout (1h margin under solve walltime)

# Per-N RAM tier for the solve (grounding footprint ~ N²·u).  Reused, with a
# generous cap, for prep too (PCMCI is not memory-heavy but this avoids OOM).
declare -A MEM_BY_N=(
    [8]="8g"   [10]="8g"
    [12]="32g" [14]="32g"
    [18]="64g" [20]="64g"
    [24]="128g" [30]="128g"
    [42]="192g"
    [54]="256g"
)
PREP_MEM="32g"         # PCMCI/BOLD memory is modest and N-insensitive here.

N_VALUES=(8 10 12 14 18 20 24 30 42 54)
INSTANCES_PER_N=10

# Undersampling rates to split across.  Default 2..4 reproduces the combined
# baseline (weighted min rate = 2, MAX_URATE = 4).  Add 5 to extend.
read -r -a U_VALUES <<< "${U_VALUES:-2 3 4}"
LABEL="${LABEL:-rtpu}"
FORCE="${FORCE:-0}"

# Common environment preamble for every wrapped job.
read -r -d '' ENV_PREAMBLE <<'EOF' || true
set -e
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
. /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3
cd $SLURM_SUBMIT_DIR
EOF

RUNNING_NAMES=""
if [ "$FORCE" != "1" ]; then
    RUNNING_NAMES=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
fi

queued() { [ "$FORCE" != "1" ] && echo "$RUNNING_NAMES" | grep -qxF "$1"; }

N_PREP=0; N_SOLVE=0; N_AGG=0
SKIP_GROUP=0

for N in "${N_VALUES[@]}"; do
    MEM="${MEM_BY_N[$N]}"
    for ((I=0; I<INSTANCES_PER_N; I++)); do
        TAG="${LABEL}_n${N}_inst${I}"
        INPUT_PKL="${OUTPUT_DIR}/${TAG}__input.pkl"
        AGG_JSON="${OUTPUT_DIR}/${TAG}__aggregated.json"

        # Skip the whole group if already aggregated.
        if [ "$FORCE" != "1" ] && [ -f "$AGG_JSON" ]; then
            printf "  [skip group] %s  (already aggregated)\n" "$TAG"
            SKIP_GROUP=$((SKIP_GROUP + 1))
            continue
        fi

        # ── Stage 1: prep ────────────────────────────────────────────────────
        # CAN_SOLVE gates stage 2: we only submit solves when the input either
        # already exists (PREP_DEP empty) or we just queued prep this round
        # (PREP_DEP=afterok:id).  If prep is queued from a PRIOR submission its
        # job id is unknown, so we cannot safely chain — skip the group and let
        # the in-flight pipeline finish (or re-run with FORCE=1).
        PREP_DEP=""
        CAN_SOLVE=1
        PREP_NAME="prep_${TAG}"
        if [ "$FORCE" != "1" ] && [ -f "$INPUT_PKL" ]; then
            printf "  [have input] %s\n" "$TAG"
        elif queued "$PREP_NAME"; then
            printf "  [prep queued — skipping group, no chainable id] %s\n" "$TAG"
            CAN_SOLVE=0
        else
            PREP_ID=$(sbatch --parsable \
                -J "$PREP_NAME" -p "$PARTITION" -A "$ACCOUNT" \
                --mail-type=FAIL --mail-user="$EMAIL" \
                -N 1 -n 1 --cpus-per-task=${PREP_CPUS} --mem=${PREP_MEM} \
                -t "$PREP_WALLTIME" \
                -o "${LOG_DIR}/${TAG}__prep.out" -e "${LOG_DIR}/${TAG}__prep.err" \
                --wrap "${ENV_PREAMBLE}
                    python ${RUNNER} prep \
                        --n_nodes ${N} --instance_id ${I} \
                        --run_tag ${TAG} --output_dir ${OUTPUT_DIR}
                ")
            PREP_DEP="--dependency=afterok:${PREP_ID}"
            N_PREP=$((N_PREP + 1))
            printf "  prep  %-22s JobID=%s\n" "$TAG" "$PREP_ID"
        fi

        # ── Stage 2: one solve per u ─────────────────────────────────────────
        if [ "$CAN_SOLVE" != "1" ]; then
            continue
        fi
        SOLVE_IDS=()
        for U in "${U_VALUES[@]}"; do
            SOLVE_NAME="u${U}_${TAG}"
            U_JSON="${OUTPUT_DIR}/${TAG}__u${U}.json"
            if [ "$FORCE" != "1" ] && [ -f "$U_JSON" ] && \
               grep -q '"status": "completed"' "$U_JSON" 2>/dev/null; then
                printf "    [skip] u=%s %s (done)\n" "$U" "$TAG"
                continue
            fi
            if queued "$SOLVE_NAME"; then
                printf "    [queued] u=%s %s\n" "$U" "$TAG"
                continue
            fi
            SID=$(sbatch --parsable \
                -J "$SOLVE_NAME" -p "$PARTITION" -A "$ACCOUNT" \
                --mail-type=FAIL --mail-user="$EMAIL" \
                -N 1 -n 1 --cpus-per-task=${SOLVE_CPUS} --mem=${MEM} \
                -t "$SOLVE_WALLTIME" ${PREP_DEP} \
                -o "${LOG_DIR}/${TAG}__u${U}.out" -e "${LOG_DIR}/${TAG}__u${U}.err" \
                --wrap "${ENV_PREAMBLE}
                    python ${RUNNER} solve \
                        --run_tag ${TAG} --u_value ${U} \
                        --output_dir ${OUTPUT_DIR} \
                        --timeout_hours ${TIMEOUT_HOURS} \
                        --pnum \${SLURM_CPUS_PER_TASK:-15}
                ")
            SOLVE_IDS+=("$SID")
            N_SOLVE=$((N_SOLVE + 1))
            printf "    solve u=%s %-18s JobID=%s\n" "$U" "$TAG" "$SID"
        done

        # ── Stage 3: aggregate (afterok all solves submitted this round) ─────
        if [ "${#SOLVE_IDS[@]}" -gt 0 ]; then
            DEP_LIST=$(IFS=:; echo "${SOLVE_IDS[*]}")
            AGG_ID=$(sbatch --parsable \
                -J "agg_${TAG}" -p "$PARTITION" -A "$ACCOUNT" \
                --mail-type=FAIL --mail-user="$EMAIL" \
                -N 1 -n 1 --cpus-per-task=${AGG_CPUS} --mem=4g \
                -t "$AGG_WALLTIME" --dependency=afterok:${DEP_LIST} \
                -o "${LOG_DIR}/${TAG}__agg.out" -e "${LOG_DIR}/${TAG}__agg.err" \
                --wrap "${ENV_PREAMBLE}
                    python ${AGGREGATOR} \
                        --input_dir ${OUTPUT_DIR} --run_tag ${TAG} --top_frac 0.30
                ")
            N_AGG=$((N_AGG + 1))
            printf "    agg   %-22s JobID=%s  (afterok %s)\n" "$TAG" "$AGG_ID" "$DEP_LIST"
        fi
    done
done

echo ""
echo "=============================================================="
echo "SUBMISSION COMPLETE"
echo "=============================================================="
echo "Tag label:        ${LABEL}"
echo "U values:         ${U_VALUES[*]}"
echo "prep jobs:        ${N_PREP}"
echo "solve jobs:       ${N_SOLVE}"
echo "aggregate jobs:   ${N_AGG}"
echo "skipped groups:   ${SKIP_GROUP}  (already aggregated)"
echo "Output dir:       ${OUTPUT_DIR}/"
echo "Log dir:          ${LOG_DIR}/"
echo ""
echo "Manual re-aggregate (any time):"
echo "  python ${AGGREGATOR} --input_dir ${OUTPUT_DIR}"
echo ""

RECORD="${LOG_DIR}/submission_$(date +%m%d%Y%H%M%S).log"
{
    echo "Submitted: $(date)"
    echo "Label:     ${LABEL}"
    echo "U values:  ${U_VALUES[*]}"
    echo "prep/solve/agg: ${N_PREP}/${N_SOLVE}/${N_AGG}"
} > "$RECORD"
echo "Submission record: $RECORD"
