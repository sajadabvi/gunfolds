# Cluster Computing Guide

**Version**: 2.0.0  
**Last Updated**: December 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Getting Started with SLURM](#getting-started-with-slurm)
3. [Submitting Jobs](#submitting-jobs)
4. [Monitoring and Managing Jobs](#monitoring-and-managing-jobs)
5. [Collecting Results](#collecting-results)
6. [Best Practices](#best-practices)
7. [Common Use Cases](#common-use-cases)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### Why Use a Cluster?

Gunfolds experiments can be computationally intensive:

- **Large parameter grids**: 1000+ configurations to test
- **Many repetitions**: 100+ runs per configuration for statistical significance
- **Time-consuming methods**: RASL can take minutes to hours per graph
- **Memory requirements**: Large graphs (n > 20) may need 32+ GB RAM

**Solution**: Distribute computation across cluster nodes

### Supported Cluster Systems

- **SLURM** (Simple Linux Utility for Resource Management)
- **PBS/Torque** (basic support)
- **SGE** (Sun Grid Engine) - experimental

This guide focuses on **SLURM**, the most common system.

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

### Key Concepts

| Term | Description |
|------|-------------|
| **Login node** | Where you SSH in; for editing, not computation |
| **Compute node** | Where jobs actually run |
| **Job** | A computational task submitted to the scheduler |
| **Queue/Partition** | Group of nodes with similar resources |
| **Allocation** | Resources assigned to your job (CPUs, memory, time) |
| **Job array** | Multiple similar jobs with one submission |

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

## Best Practices

### 1. Use Scratch Space

**Don't run experiments in home directory** (limited space, not optimized for I/O)

```bash
# Set up scratch directory
mkdir -p $SCRATCH/gunfolds_experiments
cd $SCRATCH/gunfolds_experiments

# Copy data to scratch
cp -r ~/data/input_files .

# Run experiments from scratch
python benchmarks/benchmark_runner.py \
    --input-dir $SCRATCH/gunfolds_experiments/input_files \
    --output-dir $SCRATCH/gunfolds_experiments/results
```

**Important**: Scratch space is often **temporary** (auto-deleted after 30-90 days). Copy results to home directory when done.

### 2. Request Appropriate Resources

**Don't over-request** (wastes resources, longer queue times):

```bash
# Bad: Over-requesting
#SBATCH --mem=128G  # Only using 8GB
#SBATCH --time=24:00:00  # Finishes in 1 hour

# Good: Appropriate request
#SBATCH --mem=16G
#SBATCH --time=02:00:00
```

**How to estimate**:
1. Run a few jobs with more resources than needed
2. Check actual usage with `sacct -j JOBID --format=MaxRSS,Elapsed`
3. Adjust requests accordingly

### 3. Use Job Arrays for Repetitions

**Don't submit 100 individual jobs**:

```bash
# Bad:
for i in {1..100}; do
    sbatch experiment_${i}.sh  # 100 separate jobs
done
```

**Use job arrays instead**:

```bash
# Good:
#SBATCH --array=1-100
python experiment.py --run-id $SLURM_ARRAY_TASK_ID
```

### 4. Save Intermediate Results

Long jobs should save checkpoints:

```python
# In your Python script
import pickle

# Save checkpoint every 10 iterations
if iteration % 10 == 0:
    with open(f'checkpoint_{iteration}.pkl', 'wb') as f:
        pickle.dump(state, f)
```

Resume from checkpoint if job fails:

```python
# Check for existing checkpoint
checkpoint_file = 'checkpoint_latest.pkl'
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        state = pickle.load(f)
    start_iteration = state['iteration']
else:
    start_iteration = 0
```

### 5. Log Important Information

```python
import sys
import socket

print(f"Job started on: {socket.gethostname()}", file=sys.stderr)
print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'N/A')}", file=sys.stderr)
print(f"Array Task ID: {os.environ.get('SLURM_ARRAY_TASK_ID', 'N/A')}", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)

# At end of job
print(f"Job completed successfully at: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
```

### 6. Test Locally First

Before large-scale cluster submission:

```bash
# Test one instance locally on login node (quick check)
python benchmarks/benchmark_runner.py \
    --method RASL \
    --nodes 5 \
    --samples 100 \
    --output test_output/

# Submit small test job (1-5 tasks)
#SBATCH --array=1-5
# ... (rest of script)
```

Verify:
- Script runs without errors
- Output format is correct
- Resource usage is reasonable

Then scale up to full experiment.

### 7. Email Notifications

Get notified when jobs complete/fail:

```bash
#SBATCH --mail-user=your_email@university.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
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

### Use Case 4: Time-Critical Deadline

**Scenario**: Need results by tomorrow, have access to high-priority queue

```bash
#!/bin/bash
#SBATCH --job-name=urgent_experiment
#SBATCH --partition=priority  # High-priority queue
#SBATCH --qos=high  # Quality of Service
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --array=1-100
#SBATCH --mail-user=me@edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT

# Enable parallel processing
export OMP_NUM_THREADS=8

module load python/3.9
source ~/envs/gunfolds/bin/activate

python benchmarks/benchmark_runner.py \
    --method RASL \
    --nodes 15 \
    --samples 1000 \
    --run-id $SLURM_ARRAY_TASK_ID \
    --parallel 8 \
    --output-dir $SCRATCH/urgent_results/
```

**Note**: Check with your cluster admin about priority queue access and costs.

---

## Troubleshooting

### Problem 1: Job Stays in Pending (PD) State

**Possible causes**:

1. **Insufficient resources**: Requested more CPUs/memory than available

```bash
# Check available resources
sinfo

# Adjust request to fit available nodes
#SBATCH --mem=16G  # instead of 128G
```

2. **Partition limits**: Exceeded max jobs for partition

```bash
# Check partition limits
scontrol show partition normal

# Wait for some jobs to complete, or use different partition
```

3. **Dependency not met**: Job has dependencies that haven't completed

```bash
# Check dependencies
scontrol show job JOBID | grep Dependency
```

---

### Problem 2: Job Fails Immediately

**Check error log**:

```bash
cat logs/job_JOBID.err
```

**Common issues**:

1. **Module not loaded**:

```bash
# Add to job script:
module load python/3.9
```

2. **Environment not activated**:

```bash
# Add to job script:
source ~/envs/gunfolds/bin/activate
```

3. **Wrong working directory**:

```bash
# Add to job script:
cd $SCRATCH/gunfolds_experiments
```

4. **Input file not found**:

```bash
# Copy input files to scratch before running
cp -r ~/data/inputs $SCRATCH/
```

---

### Problem 3: Job Killed (Out of Memory)

**Check actual memory usage**:

```bash
sacct -j JOBID --format=JobID,MaxRSS,ReqMem

# MaxRSS = actual memory used
# ReqMem = requested memory
```

**Solution**: Increase memory request

```bash
# In job script:
#SBATCH --mem=32G  # increased from 16G
```

---

### Problem 4: Job Timeout

**Check elapsed time**:

```bash
sacct -j JOBID --format=JobID,Elapsed,Timelimit,State
```

**Solutions**:

1. **Increase time limit**:

```bash
#SBATCH --time=04:00:00  # increased from 02:00:00
```

2. **Optimize code** (reduce computation time)

3. **Use checkpointing** (restart from saved state)

---

### Problem 5: Results Not Saved

**Possible causes**:

1. **Script crashed before saving**: Add intermediate saves

2. **Wrong output directory**: Print output path in logs

```python
import sys
print(f"Saving results to: {output_dir}", file=sys.stderr)
```

3. **Scratch space full**: Check disk quota

```bash
# Check quota
quota -s

# Clean up old files
find $SCRATCH -type f -mtime +30 -delete
```

---

### Problem 6: Inconsistent Results Across Tasks

**Possible cause**: Random seed not set per task

**Solution**: Use task ID as seed

```python
import numpy as np
task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 1))
np.random.seed(task_id)
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
- **Your Cluster Docs**: (check your institution's wiki)
- **Gunfolds GitHub**: https://github.com/your-org/gunfolds
- **Support**: Contact your cluster administrator or HPC support

---

**Version**: 2.0.0  
**Last Updated**: December 2, 2025  
**Maintainer**: Gunfolds Development Team
