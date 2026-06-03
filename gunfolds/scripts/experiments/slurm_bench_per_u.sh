#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=64g
#SBATCH -p qTRDGPU
#SBATCH -A psy53c17
#SBATCH -t 06:00:00
#SBATCH -J bench_per_u
#SBATCH -o logs/bench_per_u_%j.out
#SBATCH -e logs/bench_per_u_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mabavisani@gsu.edu
# =============================================================================
# Run the controlled A/B benchmark (combined-u vs per-u split) on the SAME
# graphs, under a 6-hour walltime.
#
# Args (all optional, positional):
#   $1 N_NODES      graph size               (default 14; must be in SCC_COMPOSITION)
#   $2 N_GRAPHS     number of graphs/instances (default 5)
#   $3 TIMEOUT_SEC  per-solve cap in seconds (default 3600 = 1h per solve)
#
# The benchmark writes results INCREMENTALLY (one CSV row per graph, flushed),
# so even if the 6-hour walltime kills the job mid-sweep the completed graphs
# are preserved in bench_n<N>.csv.
#
# Submit:
#   cd gunfolds/scripts/experiments
#   sbatch slurm_bench_per_u.sh                 # N=14, 5 graphs, 1h/solve
#   sbatch slurm_bench_per_u.sh 18 5 5400        # N=18, 5 graphs, 90min/solve
#   sbatch --mem=160g slurm_bench_per_u.sh 20 3  # bump RAM for large N
# =============================================================================

set -e

N_NODES=${1:-14}
N_GRAPHS=${2:-5}
TIMEOUT_SEC=${3:-3600}
PNUM=$(( ${SLURM_CPUS_PER_TASK:-16} - 1 ))   # leave one core as buffer

mkdir -p logs

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
source /home/users/mabavisani/anaconda3/etc/profile.d/conda.sh
conda activate multi_v3
cd "$SLURM_SUBMIT_DIR"

echo "=============================================================="
echo "A/B benchmark (combined-u vs per-u split)"
echo "  N_NODES=${N_NODES}  N_GRAPHS=${N_GRAPHS}  per-solve timeout=${TIMEOUT_SEC}s"
echo "  pnum=${PNUM}  walltime=6h  node=$SLURM_NODELIST"
echo "  start: $(date)"
echo "=============================================================="

python bench_per_u_vs_combined.py \
    --n_nodes "$N_NODES" \
    --n_graphs "$N_GRAPHS" \
    --pnum "$PNUM" \
    --timeout_sec "$TIMEOUT_SEC" \
    --out "bench_n${N_NODES}.csv"

echo "=============================================================="
echo "done: $(date)   ->  bench_n${N_NODES}.csv"
echo "=============================================================="
