# Cluster Computing Guide

**Version**: 2.0.0  
**Last Updated**: December 2025

---

## Getting Started with SLURM

### Cluster Access

1. **Get an account**: Contact your cluster administrator
2. **SSH to login node**:

```bash
ssh username@cluster.university.edu
```

3. **Set up environment**:

```bash
# Load required modules (cluster-specific)
module load python/3.9
module load numpy
module load scipy

# Or use conda/virtualenv
module load anaconda
conda activate gunfolds_env
```

4. **Install gunfolds**:

```bash
pip install --user gunfolds
# or from source
git clone https://github.com/your-org/gunfolds.git
cd gunfolds
pip install --user -e .
```

5. **Verify installation**:

```bash
python -c "import gunfolds; print(gunfolds.__version__)"
```
---

## Submitting Jobs

### Method 1: Using Gunfolds Built-in SLURM Support

Most gunfolds scripts have built-in SLURM support:

```bash
# Automatic SLURM job creation and submission
python benchmarks/time_undersampling.py \
    --method RASL \
    --slurm \
    --submit \
    --nodes 5,10,15 \
    --samples 100,500,1000 \
    --partition normal \
    --time 02:00:00 \
    --mem 16G
```

**What happens**:
1. Script generates SLURM job script
2. Automatically submits job to scheduler
3. Prints job ID for monitoring
4. Results saved to specified output directory

### Method 2: Manual Job Script Creation

For more control, create a SLURM job script manually:

#### Example: benchmark_rasl.sh

```bash
#!/bin/bash
#SBATCH --job-name=rasl_benchmark
#SBATCH --output=logs/rasl_%A_%a.out
#SBATCH --error=logs/rasl_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-100

# Load environment
module load python/3.9
source ~/envs/gunfolds/bin/activate

# Move to scratch directory
cd $SCRATCH/gunfolds_experiments

# Run gunfolds with task ID
python benchmarks/benchmark_runner.py \
    --method RASL \
    --nodes 10 \
    --samples 500 \
    --run-id $SLURM_ARRAY_TASK_ID \
    --output-dir results/rasl/task_${SLURM_ARRAY_TASK_ID}/

echo "Job $SLURM_ARRAY_TASK_ID completed on $(hostname)"
```

#### Submit the job:

```bash
sbatch benchmark_rasl.sh
```

### SLURM Directives Explained

| Directive | Description | Example |
|-----------|-------------|---------|
| `--job-name` | Name for your job | `--job-name=my_experiment` |
| `--output` | stdout file path | `--output=logs/%j.out` |
| `--error` | stderr file path | `--error=logs/%j.err` |
| `--partition` | Queue/partition | `--partition=normal` |
| `--time` | Max wall time | `--time=02:00:00` (2 hours) |
| `--mem` | Memory per node | `--mem=16G` |
| `--cpus-per-task` | CPU cores | `--cpus-per-task=8` |
| `--nodes` | Number of nodes | `--nodes=1` |
| `--array` | Job array | `--array=1-100` |
| `--mail-user` | Email for notifications | `--mail-user=me@edu` |
| `--mail-type` | When to email | `--mail-type=END,FAIL` |

### Job Array Variables

| Variable | Description |
|----------|-------------|
| `$SLURM_ARRAY_TASK_ID` | Current array task ID (1, 2, 3, ...) |
| `$SLURM_ARRAY_JOB_ID` | Parent job ID |
| `$SLURM_JOB_ID` | Full job ID (parent_id + task_id) |

---

## Monitoring and Managing Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# Output:
#   JOBID   PARTITION   NAME     USER   ST  TIME  NODES
#   123456  normal      rasl_    user   R   0:15  1
#   123457  normal      rasl_    user   PD  0:00  1

# ST (status):
#   PD = Pending (waiting for resources)
#   R  = Running
#   CG = Completing
#   CD = Completed
#   F  = Failed
#   CA = Cancelled
```

### Detailed Job Info

```bash
# Detailed info for specific job
scontrol show job 123456

# Estimated start time for pending job
squeue --start -j 123456
```

### Cancel Jobs

```bash
# Cancel specific job
scancel 123456

# Cancel all your jobs
scancel -u $USER

# Cancel specific array tasks
scancel 123456_10  # Cancel task 10
scancel 123456_[1-50]  # Cancel tasks 1-50
```

### View Job History

```bash
# Jobs from last 7 days
sacct -u $USER

# Detailed info with custom format
sacct -j 123456 --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS

# Filter by date
sacct --starttime 2025-12-01 --endtime 2025-12-02
```

### Monitor Job Output

```bash
# Watch output file in real-time
tail -f logs/rasl_123456_1.out

# Check for errors
grep -i error logs/rasl_123456_*.err

# Count completed tasks
grep -c "completed" logs/rasl_123456_*.out
```

---

## Collecting Results

### Method 1: Automatic Collection

Use gunfolds' result analyzer:

```bash
# Collect and analyze all results
python analysis/result_analyzer.py \
    --data-type simulation \
    --results-dir $SCRATCH/gunfolds_experiments/results/ \
    --output summary_results.csv
```

### Method 2: Manual Collection

```bash
# Copy results from scratch to home directory
mkdir -p ~/results/rasl_experiment
cp -r $SCRATCH/gunfolds_experiments/results/* ~/results/rasl_experiment/

# Compress for download
cd ~/results
tar -czf rasl_experiment.tar.gz rasl_experiment/

# Download to local machine (from local terminal)
scp username@cluster.edu:~/results/rasl_experiment.tar.gz .
```

### Partial Result Collection

```bash
# Collect results as jobs complete (don't wait for all)
while true; do
    python analysis/result_analyzer.py \
        --data-type simulation \
        --results-dir $SCRATCH/results/ \
        --output partial_results_$(date +%Y%m%d_%H%M%S).csv
    sleep 3600  # Check every hour
done
```

---



## Common Use Cases

### Use Case 1: Large Benchmark Study

**Scenario**: Compare 6 methods × 3 node counts × 5 sample sizes × 100 repetitions = 9,000 runs

#### Step 1: Create Job Script

```bash
#!/bin/bash
#SBATCH --job-name=benchmark_study
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-9000
#SBATCH --mail-user=me@edu
#SBATCH --mail-type=END,FAIL

module load python/3.9
source ~/envs/gunfolds/bin/activate
cd $SCRATCH/gunfolds_benchmark

# Use task ID to select configuration
python benchmarks/benchmark_runner.py \
    --method all \
    --config-id $SLURM_ARRAY_TASK_ID \
    --output-dir results/
```

#### Step 2: Submit

```bash
mkdir -p logs
sbatch benchmark_study.sh
```

#### Step 3: Monitor

```bash
# Check progress
watch -n 60 'squeue -u $USER | grep benchmark'

# Count completed
grep -c "completed" logs/benchmark_*.out
```

#### Step 4: Collect Results

```bash
python analysis/result_analyzer.py \
    --data-type simulation \
    --results-dir $SCRATCH/gunfolds_benchmark/results/ \
    --output benchmark_summary.csv
```

---

### Use Case 2: Hyperparameter Grid Search

**Scenario**: VAR hyperparameter tuning with 4×3×5×4 = 240 configurations

#### Using Gunfolds Built-in SLURM Support

```bash
python experiments/var_hyperparameter_tuning.py \
    --config tuning_config.yaml \
    --slurm \
    --submit \
    --partition normal \
    --time 00:30:00 \
    --mem 4G \
    --output-dir results/hyperparameter_tuning/
```

This automatically:
- Creates job array with 240 tasks
- Submits to SLURM
- Saves results to specified directory

---

### Use Case 3: Real Data Analysis (fMRI)

**Scenario**: Analyze 50 fMRI subjects, each takes ~30 minutes

#### Job Script: fmri_analysis.sh

```bash
#!/bin/bash
#SBATCH --job-name=fmri_analysis
#SBATCH --output=logs/fmri_%A_%a.out
#SBATCH --error=logs/fmri_%A_%a.err
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --array=1-50

module load python/3.9
source ~/envs/gunfolds/bin/activate

# Get subject ID from array task ID
SUBJECT_ID=$(printf "sub-%03d" $SLURM_ARRAY_TASK_ID)

# Run analysis for this subject
python real_data/fmri_experiment.py \
    --subject $SUBJECT_ID \
    --input-dir $SCRATCH/fmri_data/ \
    --output-dir $SCRATCH/fmri_results/${SUBJECT_ID}/

echo "Completed analysis for $SUBJECT_ID"
```

#### Submit and Monitor

```bash
sbatch fmri_analysis.sh

# Monitor progress
watch -n 60 'sacct -j JOBID --format=JobID,State | grep -c COMPLETED'
```

---


## Advanced Topics

### 1. Job Dependencies

Run jobs in sequence:

```bash
# Submit first job
JOB1=$(sbatch --parsable job1.sh)

# Submit second job (depends on first completing)
sbatch --dependency=afterok:$JOB1 job2.sh
```

### 2. GPU Jobs

For deep learning methods:

```bash
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --partition=gpu

module load cuda/11.0
python deep_learning_causal_discovery.py
```

### 3. Interactive Jobs

For debugging:

```bash
# Request interactive session
srun --pty --mem=8G --time=01:00:00 bash

# Now you're on a compute node
python benchmarks/benchmark_runner.py --method RASL
```

### 4. Array Job with Variable Resources

Different tasks need different resources:

```bash
#!/bin/bash
#SBATCH --array=1-10

# Task 1-5 need less memory, 6-10 need more
if [ $SLURM_ARRAY_TASK_ID -le 5 ]; then
    export MEMORY="8G"
else
    export MEMORY="32G"
fi

# Note: Can't change SLURM allocation, but can adjust code behavior
python experiment.py --task-id $SLURM_ARRAY_TASK_ID
```

**Better approach**: Submit separate job arrays for different resource needs

---

## Quick Reference

### Common Commands

```bash
# Submit job
sbatch job_script.sh

# Check queue
squeue -u $USER

# Cancel job
scancel JOBID

# Job history
sacct -u $USER

# Detailed job info
scontrol show job JOBID

# Cluster info
sinfo

# Check disk quota
quota -s

# Interactive session
srun --pty --mem=8G bash
```

### Useful SLURM Variables

```bash
$SLURM_JOB_ID            # Job ID
$SLURM_ARRAY_TASK_ID     # Array task ID
$SLURM_ARRAY_JOB_ID      # Array job ID
$SLURM_CPUS_PER_TASK     # Number of CPUs
$SLURM_MEM_PER_NODE      # Memory per node
$SLURM_JOB_NAME          # Job name
$SLURM_SUBMIT_DIR        # Directory where sbatch was called
```

---

## Resources

- **SLURM Documentation**: https://slurm.schedmd.com/
