# GCM Parallel Processing on GSU URSA Cluster

## Overview

Run GCM analysis on multiple subjects in parallel on the **GSU URSA cluster** using SLURM array jobs with the **qTRDGPU partition**.

## 🔧 Cluster Configuration

The scripts are configured for:
- **Cluster:** GSU URSA
- **Partition:** `qTRDGPU`
- **Account:** `psy53c17`
- **Email:** `mabavisani@gsu.edu`
- **Environment:** Conda (`multi_v3`)

## 🚀 Quick Start

### Step 1: Submit Job

```bash
# Make scripts executable (if not already done)
chmod +x submit_gcm_parallel.sh slurm_gcm_parallel.sh

# Submit 50 subjects with alpha=0.05, 10 concurrent
./submit_gcm_parallel.sh -n 50 -A 50 -c 10
```

### Step 2: Monitor Progress

```bash
# Use the auto-generated progress check script
./logs/check_gcm_progress_<TIMESTAMP>.sh

# Or manually check
squeue -u mabavisani | grep gcm
ls gcm_roebroeck/<TIMESTAMP>/csv/*Adj_labeled.csv | wc -l

# Watch mode (auto-updates)
python analyze_gcm_results.py -t <TIMESTAMP> --watch --expected-subjects 50
```

### Step 3: Analyze Results

```bash
# After all jobs complete
python analyze_gcm_results.py -t <TIMESTAMP> --aggregate --plot
```

---

## 📋 Detailed Configuration

### SLURM Settings (in `slurm_gcm_parallel.sh`)

```bash
#SBATCH -N 1                    # 1 node
#SBATCH -n 1                    # 1 task
#SBATCH -c 1                    # 1 CPU per task
#SBATCH --mem=4g                # 4GB memory per job
#SBATCH -p qTRDGPU              # qTRDGPU partition
#SBATCH -t 7200                 # 2 hours max
#SBATCH -A psy53c17             # Account
#SBATCH --mail-type=ALL         # Email notifications
#SBATCH --mail-user=mabavisani@gsu.edu
#SBATCH --oversubscribe         # Allow oversubscription
```

### Environment Setup

The scripts automatically:
1. Set `MODULEPATH` for the cluster
2. Activate Anaconda3
3. Activate `multi_v3` conda environment
4. Set OpenMP threads to 1

**Location:** `/home/users/mabavisani/anaconda3`

---

## 📁 Directory Structure

### Input/Output

```
gunfolds/scripts/                # Working directory
├── gcm_roebroeck/
│   └── MMDDYYYYHHMMSS/         # Shared timestamp folder
│       ├── csv/
│       │   ├── subj0_*.csv
│       │   ├── subj1_*.csv
│       │   └── ...
│       ├── figures/
│       └── run_metadata.txt
│
├── logs/                        # Task logs
│   ├── gcm_master_job_*.log
│   ├── gcm_task_*.log
│   ├── latest_timestamp.txt
│   ├── latest_job_id.txt
│   └── check_gcm_progress_*.sh
│
├── out/                         # SLURM stdout
│   └── gcm_out<JOB_ID>-<TASK>.out
│
└── err/                         # SLURM stderr
    └── gcm_error<JOB_ID>-<TASK>.err
```

---

## 🎯 Usage Examples

### Example 1: Standard Run (50 subjects)

```bash
# Submit
./submit_gcm_parallel.sh -n 50 -A 50 -c 10

# Note the timestamp from output
# Example: 12262025103045

# Monitor
./logs/check_gcm_progress_12262025103045.sh

# Or watch continuously
python analyze_gcm_results.py -t 12262025103045 --watch --expected-subjects 50
```

### Example 2: Stricter Alpha (0.01)

```bash
./submit_gcm_parallel.sh -n 50 -A 10 -c 10
```

### Example 3: Large Dataset (100 subjects)

```bash
./submit_gcm_parallel.sh -n 100 -A 50 -c 20
```

### Example 4: Test Run (5 subjects)

```bash
# Test with just 5 subjects first
./submit_gcm_parallel.sh -n 5 -A 50 -c 2
```

---

## 🔍 Monitoring Commands

### Check Job Status

```bash
# Check your jobs
squeue -u mabavisani

# Check specific job
squeue -j <JOB_ID>

# Detailed job info
scontrol show job <JOB_ID>
```

### Check Progress

```bash
# Count completed subjects
ls gcm_roebroeck/<TIMESTAMP>/csv/*Adj_labeled.csv | wc -l

# Watch progress (updates every 30 seconds)
watch -n 30 'ls gcm_roebroeck/<TIMESTAMP>/csv/*Adj_labeled.csv | wc -l'

# Use generated progress script
./logs/check_gcm_progress_<TIMESTAMP>.sh
```

### View Logs

```bash
# View SLURM output for subject 0
tail -f out/gcm_out<JOB_ID>-0.out

# View SLURM errors
tail -f err/gcm_error<JOB_ID>-0.err

# View task logs
tail -f logs/gcm_task_0.log

# View master log
cat logs/gcm_master_job_<JOB_ID>.log
```

### Check for Failures

```bash
# Show failed/cancelled jobs
sacct -j <JOB_ID> --format=JobID,State,ExitCode | grep -v COMPLETED

# Find error messages
grep -i error err/gcm_error*.err

# Show job statistics
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS
```

---

## 🛠️ Troubleshooting

### Issue: Job Fails Immediately

**Check:**
```bash
# View error output
cat err/gcm_error<JOB_ID>-0.err

# Check if conda environment exists
source /home/users/mabavisani/anaconda3/bin/activate
conda env list | grep multi_v3
```

**Common Causes:**
- Conda environment not activated
- Python script not found
- Module path incorrect

### Issue: Jobs Time Out

**Solution:** Increase time limit in `slurm_gcm_parallel.sh`:
```bash
#SBATCH -t 10800  # 3 hours instead of 2
```

### Issue: Out of Memory

**Solution:** Increase memory in `slurm_gcm_parallel.sh`:
```bash
#SBATCH --mem=8g  # 8GB instead of 4GB
```

### Issue: Too Many Jobs in Queue

**Check queue limits:**
```bash
squeue -u mabavisani | wc -l
```

**Reduce concurrent jobs:**
```bash
./submit_gcm_parallel.sh -n 50 -A 50 -c 5  # Only 5 concurrent
```

### Issue: Resubmit Failed Subjects

**Find failed subjects:**
```bash
sacct -j <JOB_ID> --format=JobID,State | grep FAILED | awk -F'_' '{print $2}'
```

**Resubmit specific subjects:**
```bash
# Example: resubmit subjects 10, 25, 30
sbatch --array=10,25,30 slurm_gcm_parallel.sh <TIMESTAMP> 50
```

---

## 📊 Resource Usage

### Per-Subject Resources

- **Time:** ~30-60 minutes (depends on data size, alpha, bootstrap iterations)
- **Memory:** ~2-4 GB
- **CPU:** 1 core

### Recommendations

| Total Subjects | Concurrent | Total Time | Peak Memory |
|---------------|------------|------------|-------------|
| 50            | 10         | ~1 hour    | 40 GB       |
| 100           | 20         | ~1 hour    | 80 GB       |
| 50            | 5          | ~2 hours   | 20 GB       |

---

## 🎓 Advanced Usage

### Modify Parameters

Edit `slurm_gcm_parallel.sh` to adjust:

```bash
# Time limit (in seconds)
#SBATCH -t 10800  # 3 hours

# Memory per job
#SBATCH --mem=8g  # 8 GB

# CPUs per job (for parallel bootstrap)
#SBATCH -c 2      # 2 CPUs
```

### Run Multiple Alpha Values

```bash
# Alpha = 0.05
./submit_gcm_parallel.sh -n 50 -A 50 -c 10
# Note timestamp: TS1

# Alpha = 0.01  
./submit_gcm_parallel.sh -n 50 -A 10 -c 10
# Note timestamp: TS2

# Alpha = 0.001
./submit_gcm_parallel.sh -n 50 -A 1 -c 10
# Note timestamp: TS3

# Compare after completion
python analyze_gcm_results.py -t TS1 --plot
python analyze_gcm_results.py -t TS2 --plot
python analyze_gcm_results.py -t TS3 --plot
```

### Job Dependencies

```bash
# Submit second job after first completes
JOB1=$(./submit_gcm_parallel.sh -n 50 -A 50 -c 10 | grep "Job ID:" | awk '{print $3}')
sbatch --dependency=afterok:$JOB1 --array=0-49 slurm_gcm_parallel.sh <NEW_TIMESTAMP> 10
```

### Email Notifications

Email will be sent:
- When job begins
- When job ends
- If job fails

To disable:
```bash
# Edit slurm_gcm_parallel.sh
#SBATCH --mail-type=NONE  # No emails
# or
#SBATCH --mail-type=END   # Only when complete
```

---

## 📝 Logging Details

### Master Log
- Created once per array job
- Location: `logs/gcm_master_job_<JOB_ID>.log`
- Contains: Job configuration, script contents

### Task Logs
- One per subject
- Location: `logs/gcm_task_<TASK_ID>.log`
- Contains: Start time, end time, exit status

### SLURM Logs
- Stdout: `out/gcm_out<JOB_ID>-<TASK>.out`
- Stderr: `err/gcm_error<JOB_ID>-<TASK>.err`
- Contains: Python output, error messages

---

## ✅ Verification Checklist

Before running on full dataset:

- [ ] Test with 5 subjects: `./submit_gcm_parallel.sh -n 5 -c 2`
- [ ] Check conda environment: `conda activate multi_v3`
- [ ] Verify data file exists: `ls ./fbirn/fbirn_sz_data.npz`
- [ ] Check account access: `sacctmgr show assoc user=mabavisani`
- [ ] Monitor test job completion
- [ ] Verify output files created
- [ ] Check for errors in logs

---

## 🔗 Cluster-Specific Notes

### GSU URSA Cluster

**Partition:** `qTRDGPU`
- GPU partition but can run CPU jobs
- Typically less congested
- Good for array jobs

**Account:** `psy53c17`
- Psychology department allocation
- Check remaining hours: `sbalance`

**Storage:**
- Home directory: `/home/users/mabavisani/`
- Scratch space: Use if needed for large temporary files

**Network:**
- Access via SSH
- Use VPN if off-campus

---

## 📚 Quick Reference

### Submit Job
```bash
./submit_gcm_parallel.sh -n <NUM_SUBJECTS> -A <ALPHA> -c <CONCURRENT>
```

### Check Status
```bash
squeue -u mabavisani
```

### Monitor Progress
```bash
./logs/check_gcm_progress_<TIMESTAMP>.sh
```

### View Logs
```bash
tail -f out/gcm_out*.out
```

### Analyze Results
```bash
python analyze_gcm_results.py -t <TIMESTAMP> --aggregate --plot
```

---

## 🆘 Getting Help

### Cluster Issues
- Email: `ursa-admin@gsu.edu`
- Wiki: Check GSU URSA documentation

### Script Issues
- Check logs in `err/` and `logs/` directories
- Verify conda environment
- Test single subject first

### Data Issues
- Verify input file exists
- Check file permissions
- Ensure sufficient disk space

---

## 📖 Related Files

- **`submit_gcm_parallel.sh`** - Submission script (GSU URSA configured)
- **`slurm_gcm_parallel.sh`** - Worker script (GSU URSA configured)
- **`gcm_on_ICA.py`** - Main GCM analysis
- **`analyze_gcm_results.py`** - Results analysis
- **`GCM_PARALLEL_GUIDE.md`** - General parallel guide
- **`slurm_hyperparameter_tuning.sh`** - Template reference

---

**Ready to run on GSU URSA cluster! 🚀**

Test with a small number of subjects first, then scale up.

