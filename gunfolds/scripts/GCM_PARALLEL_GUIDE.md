# GCM Parallel Processing with SLURM - Complete Guide

## Overview

Run GCM analysis on multiple subjects in parallel using SLURM array jobs. All subjects write to the **same timestamped folder**, ensuring synchronized results.

## 🚀 Quick Start

### Step 1: Submit Parallel Job

```bash
# Make submit script executable
chmod +x submit_gcm_parallel.sh

# Submit job for 50 subjects with alpha=0.05
./submit_gcm_parallel.sh -n 50 -A 50 -c 10
```

This will:
- Create a timestamped folder (e.g., `gcm_roebroeck/12262025103045/`)
- Submit SLURM array job for subjects 0-49
- Run 10 subjects concurrently

### Step 2: Monitor Progress

```bash
# Check job status (use Job ID from submit output)
squeue -j <JOB_ID>

# Count completed subjects
ls gcm_roebroeck/12262025103045/csv/*Adj_labeled.csv | wc -l

# Watch for new results (auto-updates every 30 seconds)
python analyze_gcm_results.py -t 12262025103045 --watch --expected-subjects 50
```

### Step 3: Analyze Results

```bash
# After all jobs complete
python analyze_gcm_results.py -t 12262025103045 --aggregate --plot
```

---

## 📁 Files Created

### SLURM Scripts

1. **`submit_gcm_parallel.sh`** - Main submission script
   - Generates shared timestamp
   - Creates output directory
   - Submits array job

2. **`slurm_gcm_parallel.sh`** - Array job script
   - Runs one subject per array task
   - Uses shared timestamp from submission

### Output Structure

```
gcm_roebroeck/
└── 12262025103045/              # Shared timestamp folder
    ├── csv/
    │   ├── subj0_*_Adj_labeled.csv    # From array task 0
    │   ├── subj1_*_Adj_labeled.csv    # From array task 1
    │   ├── ...
    │   ├── subj49_*_Adj_labeled.csv   # From array task 49
    │   ├── group_edge_hits.csv        # Created by analyze_gcm_results.py
    │   ├── group_edge_rate.csv        # Created by analyze_gcm_results.py
    │   └── group_Fdiff_mean.csv       # Created by analyze_gcm_results.py
    ├── figures/
    │   ├── subj0_*_gcm.png
    │   ├── subj1_*_gcm.png
    │   ├── ...
    │   └── group_edge_frequency.png   # Created by analyze_gcm_results.py
    ├── run_metadata.txt          # Job submission info
    └── analysis_plots/           # Created by analyze_gcm_results.py --plot
        └── *.pdf

logs/
├── gcm_<JOB_ID>_0.out          # STDOUT for subject 0
├── gcm_<JOB_ID>_0.err          # STDERR for subject 0
├── gcm_<JOB_ID>_1.out          # STDOUT for subject 1
├── ...
├── latest_timestamp.txt         # Most recent timestamp
└── latest_job_id.txt            # Most recent job ID
```

---

## 🎯 Key Features

### Synchronized Timestamp
- **Problem:** Each array job starts at slightly different times
- **Solution:** Timestamp generated BEFORE job submission
- **Result:** All subjects write to same folder

### Thread-Safe Directory Creation
- Multiple jobs can create directories simultaneously
- `os.makedirs(..., exist_ok=True)` prevents conflicts

### Progressive Aggregation
- Analyze results as they arrive with `--watch` mode
- No need to wait for all jobs to complete

---

## 📋 Detailed Usage

### Submit Script Options

```bash
./submit_gcm_parallel.sh [OPTIONS]

Options:
  -A, --alpha ALPHA           Alpha value x1000 (default: 50 = 0.05)
  -n, --num-subjects N        Number of subjects (default: 100)
  -c, --concurrent N          Number of concurrent jobs (default: 20)
  -h, --help                  Show help message
```

**Examples:**

```bash
# 50 subjects, alpha=0.05, 10 concurrent
./submit_gcm_parallel.sh -n 50 -A 50 -c 10

# 100 subjects, alpha=0.01, 20 concurrent
./submit_gcm_parallel.sh -n 100 -A 10 -c 20

# 30 subjects, alpha=0.001, 5 concurrent
./submit_gcm_parallel.sh -n 30 -A 1 -c 5
```

### Analysis Options

```bash
python analyze_gcm_results.py [OPTIONS]

New Options for Parallel Runs:
  --aggregate                 Force aggregation from subject files
  --expected-subjects N       Expected number of subjects
  --watch                     Watch mode: continuously update
  --watch-interval N          Seconds between updates (default: 30)
```

**Examples:**

```bash
# Basic aggregation (after jobs complete)
python analyze_gcm_results.py -t 12262025103045 --aggregate --plot

# Watch mode: monitor as jobs run
python analyze_gcm_results.py -t 12262025103045 --watch --expected-subjects 50

# Watch with custom interval
python analyze_gcm_results.py -t 12262025103045 --watch --expected-subjects 100 --watch-interval 60
```

---

## 🔄 Complete Workflow

### 1. Submit Jobs

```bash
# Submit 50 subjects, alpha=0.05, 10 concurrent
./submit_gcm_parallel.sh -n 50 -A 50 -c 10

# Note the output:
# Job submitted successfully!
# Job ID: 12345678
# Timestamp: 12262025103045
```

### 2. Monitor in Real-Time

**Option A: Watch mode (auto-updates)**
```bash
python analyze_gcm_results.py -t 12262025103045 --watch --expected-subjects 50
```

**Option B: Manual checks**
```bash
# Check SLURM queue
squeue -j 12345678

# Count completed subjects
ls gcm_roebroeck/12262025103045/csv/*Adj_labeled.csv | wc -l

# View recent log
tail -f logs/gcm_12345678_0.out
```

### 3. Final Analysis (after completion)

```bash
# Aggregate and plot
python analyze_gcm_results.py -t 12262025103045 --aggregate --plot

# With threshold
python analyze_gcm_results.py -t 12262025103045 --aggregate --threshold 0.3 --plot
```

---

## 🛠️ Troubleshooting

### Issue: Jobs fail with "directory not found"

**Cause:** Submit script didn't create directories properly

**Solution:**
```bash
# Manually create directory structure
TIMESTAMP=12262025103045
mkdir -p gcm_roebroeck/$TIMESTAMP/csv
mkdir -p gcm_roebroeck/$TIMESTAMP/figures
```

### Issue: Some subjects missing

**Check which subjects completed:**
```bash
ls gcm_roebroeck/12262025103045/csv/*Adj_labeled.csv | grep -oP 'subj\K[0-9]+' | sort -n
```

**Resubmit failed subjects:**
```bash
# Example: resubmit subject 25
sbatch --array=25 slurm_gcm_parallel.sh 12262025103045 50
```

### Issue: "No results found" during analysis

**Cause:** Wrong timestamp or no subjects completed yet

**Solution:**
```bash
# Check available timestamps
ls gcm_roebroeck/

# Check if any subjects completed
ls gcm_roebroeck/12262025103045/csv/ | head
```

### Issue: Analysis shows fewer subjects than expected

**Check job status:**
```bash
# Show failed/completed jobs
sacct -j 12345678 --format=JobID,State,ExitCode | grep -v COMPLETED
```

**Check error logs:**
```bash
# Find errors in log files
grep -i error logs/gcm_12345678_*.err
```

---

## 📊 Monitoring Tips

### Real-Time Progress

```bash
# Terminal 1: Watch job queue
watch -n 10 'squeue -j 12345678 | tail -20'

# Terminal 2: Count completed subjects
watch -n 30 'ls gcm_roebroeck/12262025103045/csv/*Adj_labeled.csv 2>/dev/null | wc -l'

# Terminal 3: Watch analysis
python analyze_gcm_results.py -t 12262025103045 --watch --expected-subjects 50
```

### Progress Script

Create `check_progress.sh`:
```bash
#!/bin/bash
TIMESTAMP=$1
EXPECTED=$2

COMPLETED=$(ls gcm_roebroeck/$TIMESTAMP/csv/*Adj_labeled.csv 2>/dev/null | wc -l)
PERCENT=$(echo "scale=1; $COMPLETED * 100 / $EXPECTED" | bc)

echo "Progress: $COMPLETED/$EXPECTED ($PERCENT%)"
```

Usage:
```bash
chmod +x check_progress.sh
./check_progress.sh 12262025103045 50
```

---

## 🔧 Customization

### Modify SLURM Resources

Edit `slurm_gcm_parallel.sh`:

```bash
#SBATCH --time=02:00:00          # Time per subject
#SBATCH --cpus-per-task=1        # CPUs per subject
#SBATCH --mem=4G                 # Memory per subject
```

Adjust based on your needs:
- **More alpha values tested:** Increase `--time`
- **Larger datasets:** Increase `--mem`
- **Parallel bootstrap:** Increase `--cpus-per-task`

### Modify Concurrent Jobs

In submit script or manually:
```bash
# Run 30 subjects, max 5 concurrent
./submit_gcm_parallel.sh -n 30 -c 5

# Run all subjects, max 50 concurrent
./submit_gcm_parallel.sh -n 100 -c 50
```

### Custom Watch Interval

```bash
# Check every minute
python analyze_gcm_results.py -t 12262025103045 --watch --watch-interval 60

# Check every 10 seconds (for testing)
python analyze_gcm_results.py -t 12262025103045 --watch --watch-interval 10
```

---

## 💡 Best Practices

### 1. **Test with Few Subjects First**

```bash
# Test with 5 subjects
./submit_gcm_parallel.sh -n 5 -A 50 -c 2
```

### 2. **Use Appropriate Concurrency**

- Too many: Queue overload, slower overall
- Too few: Underutilizes resources
- **Recommended:** 10-20 concurrent jobs

### 3. **Monitor Early**

Start watch mode immediately after submission:
```bash
./submit_gcm_parallel.sh -n 50 -A 50 -c 10
# Get timestamp from output
python analyze_gcm_results.py -t <TIMESTAMP> --watch --expected-subjects 50 &
```

### 4. **Save Metadata**

The submit script automatically saves:
- Timestamp
- Job ID
- Parameters used
- Submission time

Check: `gcm_roebroeck/<TIMESTAMP>/run_metadata.txt`

### 5. **Check Logs for Errors**

```bash
# Find any errors
grep -i error logs/gcm_*.err

# Check failed jobs
sacct -j <JOB_ID> --format=JobID,State,ExitCode | grep FAILED
```

---

## 🎓 Advanced Usage

### Resubmit Specific Subjects

```bash
# Subjects 10, 15, 20 failed - resubmit only those
TIMESTAMP=12262025103045
sbatch --array=10,15,20 slurm_gcm_parallel.sh $TIMESTAMP 50
```

### Run Multiple Alpha Values

```bash
# Alpha = 0.05
./submit_gcm_parallel.sh -n 50 -A 50 -c 10
# Timestamp: 12262025103045

# Alpha = 0.01
./submit_gcm_parallel.sh -n 50 -A 10 -c 10
# Timestamp: 12262025104530

# Alpha = 0.001
./submit_gcm_parallel.sh -n 50 -A 1 -c 10
# Timestamp: 12262025105215

# Compare results
python analyze_gcm_results.py -t 12262025103045 --plot
python analyze_gcm_results.py -t 12262025104530 --plot
python analyze_gcm_results.py -t 12262025105215 --plot
```

### Dependency Chain

Run alpha=0.01 only after alpha=0.05 completes:
```bash
# Submit first job
JOB1=$(sbatch submit_gcm_parallel.sh -n 50 -A 50 -c 10 | awk '{print $4}')

# Submit second job dependent on first
sbatch --dependency=afterok:$JOB1 submit_gcm_parallel.sh -n 50 -A 10 -c 10
```

---

## 📈 Performance Tips

### Optimal Concurrency

```bash
# Check cluster limits
sinfo
scontrol show config | grep MaxArraySize

# Adjust concurrent jobs based on availability
./submit_gcm_parallel.sh -n 100 -c 20  # Good for busy cluster
./submit_gcm_parallel.sh -n 100 -c 50  # Good for idle cluster
```

### Batch Size vs. Wall Time

- **Smaller batches (10-20):** Better for busy clusters
- **Larger batches (50+):** Faster on idle clusters
- **Consider:** Job startup overhead vs. parallel benefit

---

## 🔍 Verification

### Check All Subjects Completed

```bash
TIMESTAMP=12262025103045
EXPECTED=50

# Count files
FOUND=$(ls gcm_roebroeck/$TIMESTAMP/csv/*Adj_labeled.csv 2>/dev/null | wc -l)

if [ $FOUND -eq $EXPECTED ]; then
    echo "✓ All $EXPECTED subjects completed"
else
    echo "✗ Only $FOUND of $EXPECTED completed"
    # Find missing subjects
    for i in $(seq 0 $((EXPECTED-1))); do
        if [ ! -f "gcm_roebroeck/$TIMESTAMP/csv/subj${i}_*_Adj_labeled.csv" ]; then
            echo "  Missing: subject $i"
        fi
    done
fi
```

### Validate Results

```bash
# Check group files were created
ls -lh gcm_roebroeck/$TIMESTAMP/csv/group_*.csv

# Check all subjects have same nodes
head -1 gcm_roebroeck/$TIMESTAMP/csv/subj*_Adj_labeled.csv | sort -u

# Should see only one unique header
```

---

## 📚 Summary

### Key Points

1. ✅ **Shared Timestamp** - All subjects write to same folder
2. ✅ **Thread-Safe** - Multiple jobs can run simultaneously
3. ✅ **Progressive Analysis** - Monitor results as they arrive
4. ✅ **Auto-Aggregation** - Combine results automatically
5. ✅ **Watch Mode** - Real-time progress tracking

### Quick Commands

```bash
# Submit
./submit_gcm_parallel.sh -n 50 -A 50 -c 10

# Monitor
python analyze_gcm_results.py -t <TIMESTAMP> --watch --expected-subjects 50

# Analyze
python analyze_gcm_results.py -t <TIMESTAMP> --aggregate --plot
```

---

## Related Files

- `submit_gcm_parallel.sh` - Job submission helper
- `slurm_gcm_parallel.sh` - SLURM array job script
- `gcm_on_ICA.py` - Main GCM analysis (modified for parallel)
- `analyze_gcm_results.py` - Analysis tool (enhanced for parallel)
- `GCM_QUICK_START.md` - Serial execution guide

---

Happy parallel processing! 🚀

