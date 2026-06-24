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
MATLAB_MODULE=${MATLAB_MODULE:-matlab/R2023a}   # `module avail matlab` to list
# Absolute fallback path used if neither $MATLAB_BIN nor the module resolve.
MATLAB_ABS=${MATLAB_ABS:-/sysapps/ubuntu-applications/matlab/MATLAB_R2023a/bin/matlab}
# Headless nodes lack X11 client libs (libXt.so.6 ...) that MATLAB loads even
# under -batch. We supply them from a small conda env (see baselines_README).
# Build once:  conda create -n xlibs -c conda-forge xorg-libxt xorg-libxext \
#   xorg-libxmu xorg-libxtst xorg-libxrandr xorg-libxfixes xorg-libxcursor \
#   xorg-libxinerama xorg-libxi xorg-libxrender xorg-libxcomposite \
#   xorg-libxdamage xorg-libxscrnsaver xorg-libsm xorg-libice libxcb
MATLAB_XLIB_DIR=${MATLAB_XLIB_DIR:-$HOME/anaconda3/envs/xlibs/lib}
MVGC_TOOLBOX=${MVGC_TOOLBOX:-$HOME/MVGC1}   # lcbarnett/MVGC1 clone
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

# NOTE: do NOT overwrite MODULEPATH here -- the matlab/R modules live in a
# different tree (/sysapps/ubuntu-applications/...) than the legacy
# /apps/Compilers tree, so clobbering MODULEPATH hides `matlab`.
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
# (a) supply X11 client libs MATLAB needs even under -batch (headless node).
if [ -d "$MATLAB_XLIB_DIR" ]; then
    export LD_LIBRARY_PATH="$MATLAB_XLIB_DIR:$LD_LIBRARY_PATH"
    echo "Added X libs to LD_LIBRARY_PATH: $MATLAB_XLIB_DIR"
else
    echo "WARN: MATLAB_XLIB_DIR '$MATLAB_XLIB_DIR' not found; matlab may fail to" >&2
    echo "      load libXt.so.6. Build it (see header) or set MATLAB_XLIB_DIR." >&2
fi
# (b) resolve the matlab binary: $MATLAB_BIN on PATH -> module -> absolute path.
# (The module only puts matlab on PATH; with the X libs above the absolute
#  binary works headlessly too, so we don't depend on module-in-batch.)
MATLAB_RUN="$MATLAB_BIN"
if ! command -v "$MATLAB_RUN" >/dev/null 2>&1; then
    command -v module >/dev/null 2>&1 || {
        for f in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh \
                 /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash \
                 /usr/share/Modules/init/bash; do
            [ -f "$f" ] && source "$f" && break
        done; }
    module load "$MATLAB_MODULE" 2>/dev/null || true
fi
command -v "$MATLAB_RUN" >/dev/null 2>&1 || MATLAB_RUN="$MATLAB_ABS"
if ! command -v "$MATLAB_RUN" >/dev/null 2>&1; then
    echo "ERROR: no matlab found (tried '$MATLAB_BIN', module '$MATLAB_MODULE', '$MATLAB_ABS')." >&2
    echo "       Set MATLAB_ABS to the absolute matlab path (module load matlab/R2023a; command -v matlab)." >&2
    exit 2
fi
echo "Using MATLAB: $MATLAB_RUN"
if [ "$METHOD" = "MVGC" ]; then
    MCMD="try, addpath(genpath('${MVGC_TOOLBOX}')); startup; catch e, disp(e.message); end; \
          baselines_mvgc('${WORKDIR}', ${N_COMP}, ${ALPHA}, ${MOMAX}); exit"
else
    MCMD="baselines_mvar('${WORKDIR}', ${N_COMP}, ${ALPHA}, ${P}); exit"
fi
echo "Running MATLAB: $MCMD"
"$MATLAB_RUN" -nodisplay -nosplash -nodesktop -batch "$MCMD"

# ---- 3) collect MATLAB output into result.zkl ----
python baselines_fmri_experiment.py \
    --method $METHOD --stage collect \
    --n_components $N_COMP --timestamp $TIMESTAMP \
    --results_root $RESULTS_ROOT --workdir "$WORKDIR"

echo "DONE $METHOD N=$N_COMP -> $RESULTS_ROOT/$TIMESTAMP/N${N_COMP}_none_${METHOD}/"
