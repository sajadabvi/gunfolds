#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH -p qTRD
#SBATCH -t 0-04:00:00
#SBATCH -J base_mat
#SBATCH -e ./err/base_mat_error%A.err
#SBATCH -o ./out/base_mat_out%A.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu

# =============================================================================
# MVGC / MVAR baseline - MATLAB bridge, ONE job per (N, METHOD).
# =============================================================================
# Three chained steps (all subjects loop inside MATLAB; tiny per-subject cost):
#   1) python export  -> <WORKDIR>/N<N>/input.mat
#   2) MATLAB         -> <WORKDIR>/N<N>/<METHOD>/sig_<s>.mat  (A_src_tgt)
#   3) python collect -> fbirn_results_refactored/<TS>/N<N>_none_<METHOD>/subject_*/result.zkl
#
# MVAR is self-contained MATLAB (OLS VAR + Wald test) and needs no toolbox.
# MVGC needs the MVGC toolbox (Barnett & Seth) on the MATLAB path -> set
# MVGC_TOOLBOX to its root (it runs addpath(genpath(.)); startup).
#
# Env overrides (optional):
#   WORKDIR        (baselines_work/<TIMESTAMP>)
#   RESULTS_ROOT   (fbirn_results_refactored)
#   MATLAB_BIN     (matlab)            MATLAB_MODULE (matlab)
#   MVGC_TOOLBOX   ($HOME/MVGC)        ALPHA (0.05)   MOMAX (5)   P (1)
#
# Usage:
#   sbatch slurm_baselines_matlab.sh <TIMESTAMP> <N_COMP> <METHOD:MVGC|MVAR>
# =============================================================================

TIMESTAMP=$1
N_COMP=${2:-10}
METHOD=${3:-MVGC}
WORKDIR=${WORKDIR:-baselines_work/$TIMESTAMP}
RESULTS_ROOT=${RESULTS_ROOT:-fbirn_results_refactored}
MATLAB_BIN=${MATLAB_BIN:-matlab}
MATLAB_MODULE=${MATLAB_MODULE:-matlab}
MVGC_TOOLBOX=${MVGC_TOOLBOX:-$HOME/MVGC}
ALPHA=${ALPHA:-0.05}
MOMAX=${MOMAX:-5}
P=${P:-1}

if [ -z "$TIMESTAMP" ]; then
    echo "Error: TIMESTAMP not provided"
    echo "Usage: sbatch slurm_baselines_matlab.sh <TIMESTAMP> <N_COMP> <METHOD:MVGC|MVAR>"
    exit 1
fi
if [ "$METHOD" != "MVGC" ] && [ "$METHOD" != "MVAR" ]; then
    echo "Error: METHOD must be MVGC or MVAR (got '$METHOD')"; exit 1
fi

mkdir -p ./logs ./err ./out

export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3

cd $SLURM_SUBMIT_DIR
scontrol update jobid=$SLURM_JOB_ID name="base_${METHOD}_N${N_COMP}" 2>/dev/null

echo "==========================================="
echo "$METHOD baseline (MATLAB bridge)  N=$N_COMP  TS=$TIMESTAMP"
echo "  workdir=$WORKDIR  results_root=$RESULTS_ROOT"
echo "==========================================="

set -e

# ---- 1) export per-subject data to a single MATLAB-readable .mat ----
python baselines_fmri_experiment.py \
    --method $METHOD --stage export \
    --n_components $N_COMP --timestamp $TIMESTAMP \
    --results_root $RESULTS_ROOT --workdir "$WORKDIR"

# ---- 2) MATLAB: compute per-subject significance matrices ----
module load $MATLAB_MODULE 2>/dev/null || true
if [ "$METHOD" = "MVGC" ]; then
    MCMD="try, addpath(genpath('${MVGC_TOOLBOX}')); startup; catch e, disp(e.message); end; \
          baselines_mvgc('${WORKDIR}', ${N_COMP}, ${ALPHA}, ${MOMAX}); exit"
else
    MCMD="baselines_mvar('${WORKDIR}', ${N_COMP}, ${ALPHA}, ${P}); exit"
fi
echo "Running MATLAB: $MCMD"
$MATLAB_BIN -nodisplay -nosplash -nodesktop -r "$MCMD"

# ---- 3) collect MATLAB output into result.zkl ----
python baselines_fmri_experiment.py \
    --method $METHOD --stage collect \
    --n_components $N_COMP --timestamp $TIMESTAMP \
    --results_root $RESULTS_ROOT --workdir "$WORKDIR"

echo "DONE $METHOD N=$N_COMP -> $RESULTS_ROOT/$TIMESTAMP/N${N_COMP}_none_${METHOD}/"
