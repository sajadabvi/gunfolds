#!/bin/bash
# run_missing_rasl.sh
# Submit a fast PARALLEL array job for the refactored-RASL subjects that are
# MISSING a result.zkl, writing into the SAME timestamp folder so the run
# completes to 311. Same refactored config as the existing subjects.
#
# Idempotent: it only submits subjects that currently lack a result.zkl, so
# re-running it after completion submits nothing.
#
# Defaults target N=10 domain RASL (subjects 0-142 missing on the cluster).
# For N=13 instead:   TS=06052026010734 N_COMP=13 GT_VAL=32 bash run_missing_rasl.sh
#
# Run from:  .../gunfolds/gunfolds/scripts/real_data
set -uo pipefail

# ---- experiment config (must match the original run) ----
N_COMP="${N_COMP:-10}"
SCC="${SCC:-domain}"
METHOD="${METHOD:-RASL}"
TS="${TS:-06052026012518}"             # N=10 timestamp; N=13 -> 06052026010734
GT_MODE="${GT_MODE:-fixed}"
GT_VAL="${GT_VAL:-35}"                  # N=10 fixed GT_density; N=13 -> 32
NSUBJ="${NSUBJ:-311}"                   # total subjects (indices 0..NSUBJ-1)
OPTIM="${OPTIM:-optN}"                  # 'optN' = full near-optimal set (posterior);
                                        # 'opt'  = single optimum (fast, 1 sol/subj, degenerate band)
SLURM_SCRIPT="${SLURM_SCRIPT:-../cluster/refactored_slurm_fmri_large.sh}"

# ---- resources: RASL N=10 is ~1 min/subj, so run up to THROTTLE at once ----
PART="${PART:-qTRD}"                    # CPU partition; use qTRDGPU if qTRD is full
CPUS="${CPUS:-8}"
MEM="${MEM:-32g}"
WALL="${WALL:-02:00:00}"
THROTTLE="${THROTTLE:-100}"

CONFIG_TAG="N${N_COMP}_${SCC}_${METHOD}"
DIR="fbirn_results_refactored/${TS}/${CONFIG_TAG}"

if [ ! -f "$SLURM_SCRIPT" ]; then
  echo "ERROR: slurm script not found at $SLURM_SCRIPT (run from real_data/)"; exit 1
fi

# ---- find subjects with no result.zkl ----
MISSING=""
for i in $(seq 0 $((NSUBJ-1))); do
  f=$(printf '%s/subject_%04d/result.zkl' "$DIR" "$i")
  [ -f "$f" ] || MISSING="${MISSING}${MISSING:+,}$i"
done
NMISS=$(printf '%s' "$MISSING" | tr ',' '\n' | grep -c . || true)

echo "=============================================================="
echo " Config:      $CONFIG_TAG   timestamp=$TS   gt=$GT_MODE $GT_VAL"
echo " Output dir:  $DIR"
echo " Present:     $(find "$DIR" -name result.zkl 2>/dev/null | wc -l | tr -d ' ') / $NSUBJ"
echo " Missing:     $NMISS subjects"
echo " Indices:     $MISSING"
echo " Optim mode:  $OPTIM"
echo " Resources:   $PART  cpus=$CPUS mem=$MEM wall=$WALL  throttle=%$THROTTLE"
echo "=============================================================="
if [ "$OPTIM" = "opt" ]; then
  echo " NOTE: opt mode returns ~1 solution/subject (the MAP graph). Those subjects"
  echo "       get a DEGENERATE band — no multi-solution posterior, no under-"
  echo "       determination biomarkers — so they are NOT comparable to optN"
  echo "       subjects for the posterior/biomarker analysis. Use it to FINISH"
  echo "       hard/dense subjects that optN cannot enumerate in time."
  echo "=============================================================="
fi
if [ "$NMISS" -eq 0 ]; then
  echo "Nothing missing — run already complete. Exiting."; exit 0
fi

CMD=(sbatch
  --array="${MISSING}%${THROTTLE}"
  --partition="$PART" --time="$WALL" --cpus-per-task="$CPUS" --mem="$MEM"
  --export="ALL,OPTIM=${OPTIM}"
  --job-name="rfmri_${CONFIG_TAG}_${OPTIM}_fill"
  "$SLURM_SCRIPT" "$TS" "$N_COMP" "$SCC" "$METHOD" "$GT_MODE" "$GT_VAL")

echo "Submitting:"; echo "   ${CMD[*]}"; echo
"${CMD[@]}"
echo
echo "Track:   squeue -u \$USER | grep _fill"
echo "Verify:  bash progress.sh   (or re-run this script — it will report 0 missing when done)"
