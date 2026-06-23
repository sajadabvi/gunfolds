#!/bin/bash

#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16g
#SBATCH -p qTRD
#SBATCH -t 0-02:00:00
#SBATCH -J base_fask
#SBATCH -e ./err/base_fask_error%A-%a.err
#SBATCH -o ./out/base_fask_out%A-%a.out
#SBATCH -A psy53c17
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mabavisani@gsu.edu

# =============================================================================
# FASK baseline - ONE subject per array task (py-tetrad / jpype + tetrad jar)
# =============================================================================
# Mirrors refactored_slurm_fmri_large.sh but runs the FASK baseline, writing the
# single-solution result.zkl under
#   fbirn_results_refactored/<TIMESTAMP>/N<N>_none_FASK/subject_<idx>/result.zkl
# (same format as PCMCI/GCM -> analysed by refactored_analyze_fmri_experiment.py).
#
# Env overrides (optional):
#   TETRAD_JAR     (resources/tetrad-current.jar)
#   PYTETRAD_PATH  ($HOME/tread/py-tetrad)
#   FASK_ALPHA     (0.05)
#   RESULTS_ROOT   (fbirn_results_refactored)
#
# Usage:
#   sbatch --array=0-310%50 slurm_baselines_fask.sh <TIMESTAMP> [N_COMP]
# =============================================================================

TIMESTAMP=$1
N_COMP=${2:-10}
RESULTS_ROOT=${RESULTS_ROOT:-fbirn_results_refactored}
TETRAD_JAR=${TETRAD_JAR:-resources/tetrad-current.jar}
PYTETRAD_PATH=${PYTETRAD_PATH:-$HOME/tread/py-tetrad}
FASK_ALPHA=${FASK_ALPHA:-0.05}

if [ -z "$TIMESTAMP" ]; then
    echo "Error: TIMESTAMP not provided"
    echo "Usage: sbatch --array=0-310%50 slurm_baselines_fask.sh <TIMESTAMP> [N_COMP]"
    exit 1
fi

SUBJECT_IDX=$SLURM_ARRAY_TASK_ID
mkdir -p ./logs ./err ./out

export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
echo "Activating conda environment..." >&2
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3

cd $SLURM_SUBMIT_DIR
scontrol update jobid=$SLURM_JOB_ID name="base_fask_N${N_COMP}_${SUBJECT_IDX}" 2>/dev/null

echo "==========================================="
echo "FASK baseline  subject=$SUBJECT_IDX  N=$N_COMP"
echo "  jar=$TETRAD_JAR  pytetrad=$PYTETRAD_PATH  alpha=$FASK_ALPHA"
echo "==========================================="

python baselines_fmri_experiment.py \
    --method FASK --stage run \
    --subject_idx $SUBJECT_IDX \
    --n_components $N_COMP \
    --timestamp $TIMESTAMP \
    --results_root $RESULTS_ROOT \
    --tetrad_jar "$TETRAD_JAR" \
    --pytetrad_path "$PYTETRAD_PATH" \
    --fask_alpha $FASK_ALPHA

exit $?
