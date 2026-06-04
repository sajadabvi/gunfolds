#!/bin/bash
# =============================================================================
# Submit the PER-U split of the fMRI RASL experiment across all subjects.
#
# Architecture (mirrors runtime_scaling_per_u, applied to FBIRN subjects):
#   1. prep   array — one task per subject: PCMCI + DD/BD + SCC, pickles input.
#   2. solve  arrays — ONE array PER undersampling rate u; task i solves
#                      subject i at fixed u.  Each solve array depends on the
#                      prep array via `aftercorr` (task i waits for prep task i),
#                      so subject i's solves start as soon as its prep is done.
#   3. agg    job   — after ALL solve arrays finish (`afterany`), pools every
#                      subject's per-u solutions and writes the standard
#                      result.zkl (original format) so downstream analysis is
#                      unchanged.
#
# This is 1 + |U| + 1 array/job submissions (default 6) covering 311 subjects
# × |U| solves — not thousands of individual jobs.
#
# Identity of a subject's pipeline = (TIMESTAMP, config_tag, subject_idx),
# encoded in the output path fbirn_results/<TS>/<config_tag>/subject_<idx>/.
#
# Usage:
#   bash submit_fmri_experiment_per_u.sh [TIMESTAMP]
#   N_SUBJECTS=311 U_VALUES="2 3 4 5" N_COMP=20 SCC=domain MAXP=50 \
#       bash submit_fmri_experiment_per_u.sh
#
# Defaults reproduce the N=20 domain-RASL production config from
# slurm_fmri_large.sh (MAXU 5, PRIORITY 11112, pcmci tau=1 α=0.05 no-FDR,
# GT_density fixed by N, top_k 10).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_DATA_DIR="$(cd "${SCRIPT_DIR}/../real_data" && pwd)"
RUNNER="${REAL_DATA_DIR}/fmri_experiment_per_u.py"
AGGREGATOR="${REAL_DATA_DIR}/aggregate_fmri_per_u.py"

# ─── Config (env-overridable) ────────────────────────────────────────────────
TIMESTAMP="${1:-$(date +%m%d%Y%H%M%S)}"
N_SUBJECTS="${N_SUBJECTS:-311}"
N_COMP="${N_COMP:-20}"
SCC="${SCC:-domain}"
read -r -a U_VALUES <<< "${U_VALUES:-2 3 4 5}"
MAXP="${MAXP:-50}"                 # max concurrent array tasks
RESULTS_ROOT="${RESULTS_ROOT:-fbirn_results}"
CONFIG_TAG="N${N_COMP}_${SCC}_RASL"
LAST_IDX=$((N_SUBJECTS - 1))

# ─── SLURM resources ─────────────────────────────────────────────────────────
PARTITION="${PARTITION:-qTRDGPU}"
ACCOUNT="${ACCOUNT:-psy53c17}"
EMAIL="${EMAIL:-mabavisani@gsu.edu}"
PREP_CPUS=4;  PREP_MEM="32g";  PREP_TIME="04:00:00"
SOLVE_CPUS=15; SOLVE_MEM="160g"; SOLVE_TIME="2-00:00:00"
AGG_CPUS=1;   AGG_MEM="8g";    AGG_TIME="00:30:00"

LOG_DIR="${REAL_DATA_DIR}/logs/fmri_per_u"
mkdir -p "$LOG_DIR" "${REAL_DATA_DIR}/${RESULTS_ROOT}"

# Per-job environment preamble.  cd into real_data so the default data_path
# (../fbirn/fbirn_sz_data.npz) and the fbirn_results/ output root resolve.
read -r -d '' ENV_PREAMBLE <<EOF || true
set -e
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3
cd "${REAL_DATA_DIR}"
EOF

COMMON_ARGS="--timestamp ${TIMESTAMP} --n_components ${N_COMP} --scc_strategy ${SCC} --results_root ${RESULTS_ROOT}"

echo "=============================================================="
echo "FMRI PER-U SUBMISSION"
echo "=============================================================="
echo "Timestamp:   ${TIMESTAMP}"
echo "Config:      ${CONFIG_TAG}"
echo "Subjects:    0-${LAST_IDX}  (${N_SUBJECTS})"
echo "U values:    ${U_VALUES[*]}"
echo "Max parallel:${MAXP} per array"
echo "Results:     ${RESULTS_ROOT}/${TIMESTAMP}/${CONFIG_TAG}/"
echo "=============================================================="

# ─── Stage 1: prep array ─────────────────────────────────────────────────────
PREP_ID=$(sbatch --parsable \
    -J "preppu_${CONFIG_TAG}" -p "$PARTITION" -A "$ACCOUNT" \
    --mail-type=FAIL --mail-user="$EMAIL" \
    --array=0-${LAST_IDX}%${MAXP} \
    -N 1 -n 1 --cpus-per-task=${PREP_CPUS} --mem=${PREP_MEM} -t "$PREP_TIME" \
    -o "${LOG_DIR}/prep_%A_%a.out" -e "${LOG_DIR}/prep_%A_%a.err" \
    --wrap "${ENV_PREAMBLE}
        python ${RUNNER} prep --subject_idx \${SLURM_ARRAY_TASK_ID} \
            ${COMMON_ARGS} --PNUM \${SLURM_CPUS_PER_TASK:-4}
    ")
echo "  prep array      JobID=${PREP_ID}  (array 0-${LAST_IDX}%${MAXP})"

# ─── Stage 2: one solve array per u, each aftercorr the prep array ───────────
SOLVE_IDS=()
for U in "${U_VALUES[@]}"; do
    SID=$(sbatch --parsable \
        -J "u${U}pu_${CONFIG_TAG}" -p "$PARTITION" -A "$ACCOUNT" \
        --mail-type=FAIL --mail-user="$EMAIL" \
        --array=0-${LAST_IDX}%${MAXP} \
        --dependency=aftercorr:${PREP_ID} \
        -N 1 -n 1 --cpus-per-task=${SOLVE_CPUS} --mem=${SOLVE_MEM} -t "$SOLVE_TIME" \
        -o "${LOG_DIR}/u${U}_%A_%a.out" -e "${LOG_DIR}/u${U}_%A_%a.err" \
        --wrap "${ENV_PREAMBLE}
            python ${RUNNER} solve --subject_idx \${SLURM_ARRAY_TASK_ID} \
                ${COMMON_ARGS} --u_value ${U} --PNUM \${SLURM_CPUS_PER_TASK:-15}
        ")
    SOLVE_IDS+=("$SID")
    echo "  solve u=${U} array JobID=${SID}  (aftercorr ${PREP_ID})"
done

# ─── Stage 3: single aggregate job, afterany all solve arrays ────────────────
DEP=$(IFS=:; echo "${SOLVE_IDS[*]}")
AGG_ID=$(sbatch --parsable \
    -J "aggpu_${CONFIG_TAG}" -p "$PARTITION" -A "$ACCOUNT" \
    --mail-type=END,FAIL --mail-user="$EMAIL" \
    -N 1 -n 1 --cpus-per-task=${AGG_CPUS} --mem=${AGG_MEM} -t "$AGG_TIME" \
    --dependency=afterany:${DEP} \
    -o "${LOG_DIR}/agg_%A.out" -e "${LOG_DIR}/agg_%A.err" \
    --wrap "${ENV_PREAMBLE}
        python ${AGGREGATOR} ${COMMON_ARGS}
    ")
echo "  aggregate job   JobID=${AGG_ID}  (afterany ${DEP})"

echo ""
echo "=============================================================="
echo "SUBMITTED:  1 prep + ${#U_VALUES[@]} solve + 1 aggregate array/job"
echo "Monitor:    squeue -u \$USER"
echo "Manual re-aggregate (pools whatever u*.json exist):"
echo "  python ${AGGREGATOR} ${COMMON_ARGS}"
echo "=============================================================="

RECORD="${LOG_DIR}/submission_${TIMESTAMP}.log"
{
    echo "Submitted: $(date)"
    echo "Timestamp: ${TIMESTAMP}   Config: ${CONFIG_TAG}"
    echo "Subjects:  0-${LAST_IDX}   U: ${U_VALUES[*]}"
    echo "prep=${PREP_ID}  solves=${SOLVE_IDS[*]}  agg=${AGG_ID}"
} > "$RECORD"
echo "Submission record: $RECORD"
