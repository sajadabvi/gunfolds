#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -p qTRD
#SBATCH -t 1-00:00:00
#SBATCH -J base_gimme
#SBATCH -e ./err/base_gimme_error%A.err
#SBATCH -o ./out/base_gimme_out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu

# =============================================================================
# GIMME baseline - R bridge, GROUP-LEVEL (pooled). ONE job per N.
# =============================================================================
# GIMME estimates a shared group structure over ALL subjects at once, then frees
# per-subject paths.  Three chained steps:
#   1) python export  -> <WORKDIR>/N<N>/gimme_in/sub_<idx>.csv  (+ gimme_meta.json)
#   2) Rscript        -> <WORKDIR>/N<N>/gimme_out/indivPathEstimates.csv
#   3) python collect -> fbirn_results_refactored/<TS>/N<N>_none_GIMME/subject_*/result.zkl
#
# Requires R with the `gimme` package installed.
#
# Env overrides (optional):
#   WORKDIR       (baselines_work/<TIMESTAMP>)
#   RESULTS_ROOT  (fbirn_results_refactored)
#   R_MODULE      (R)              GIMME_AR (TRUE)
#   GROUPCUTOFF   (0.75)           SUBCUTOFF (0.50)
#
# Usage:
#   sbatch slurm_baselines_gimme.sh <TIMESTAMP> <N_COMP>
# =============================================================================

TIMESTAMP=$1
N_COMP=${2:-10}
WORKDIR=${WORKDIR:-baselines_work/$TIMESTAMP}
RESULTS_ROOT=${RESULTS_ROOT:-fbirn_results_refactored}
R_MODULE=${R_MODULE:-R}
GIMME_AR=${GIMME_AR:-TRUE}
GROUPCUTOFF=${GROUPCUTOFF:-0.75}
SUBCUTOFF=${SUBCUTOFF:-0.50}

if [ -z "$TIMESTAMP" ]; then
    echo "Error: TIMESTAMP not provided"
    echo "Usage: sbatch slurm_baselines_gimme.sh <TIMESTAMP> <N_COMP>"
    exit 1
fi

mkdir -p ./logs ./err ./out

export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3

cd $SLURM_SUBMIT_DIR
scontrol update jobid=$SLURM_JOB_ID name="base_gimme_N${N_COMP}" 2>/dev/null

GIMME_IN="$WORKDIR/N${N_COMP}/gimme_in"
GIMME_OUT="$WORKDIR/N${N_COMP}/gimme_out"

echo "==========================================="
echo "GIMME baseline (R bridge, pooled)  N=$N_COMP  TS=$TIMESTAMP"
echo "  in=$GIMME_IN  out=$GIMME_OUT  results_root=$RESULTS_ROOT"
echo "==========================================="

set -e

# ---- 1) export per-subject CSVs ----
python baselines_fmri_experiment.py \
    --method GIMME --stage export \
    --n_components $N_COMP --timestamp $TIMESTAMP \
    --results_root $RESULTS_ROOT --workdir "$WORKDIR"

# ---- 2) R: pooled gimme over the folder ----
module load $R_MODULE 2>/dev/null || true
Rscript baselines_run_gimme.R "$GIMME_IN" "$GIMME_OUT" "$GIMME_AR" "$GROUPCUTOFF" "$SUBCUTOFF"

# ---- 3) collect gimme output into result.zkl ----
python baselines_fmri_experiment.py \
    --method GIMME --stage collect \
    --n_components $N_COMP --timestamp $TIMESTAMP \
    --results_root $RESULTS_ROOT --workdir "$WORKDIR"

echo "DONE GIMME N=$N_COMP -> $RESULTS_ROOT/$TIMESTAMP/N${N_COMP}_none_GIMME/"
