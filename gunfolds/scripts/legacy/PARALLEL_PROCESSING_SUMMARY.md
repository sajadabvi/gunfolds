# Parallel Processing Summary - GCM with SLURM

## Overview

Complete parallel processing setup for GCM analysis using SLURM array jobs with synchronized timestamped results.

---

## ✅ What Was Added

### 1. SLURM Scripts

**`submit_gcm_parallel.sh`** - Main submission script
- Generates shared timestamp BEFORE job submission
- Creates output directory structure
- Submits SLURM array job with timestamp
- Saves run metadata
- Command-line options for subjects, alpha, concurrency

**`slurm_gcm_parallel.sh`** - Array job worker script
- Runs one subject per array task
- Uses timestamp passed from submit script
- Handles SLURM environment variables
- Logs to separate files per subject

### 2. Modified `gcm_on_ICA.py`

**Thread-safe directory creation:**
- `os.makedirs(..., exist_ok=True)` prevents conflicts
- Multiple jobs can create same directories simultaneously

**Enhanced timestamp handling:**
- Accepts `-t` flag to use existing timestamp
- Checks if directory already exists
- Suitable for parallel execution

**Better documentation:**
- Comments explain parallel usage
- Notes about aggregation after parallel runs

### 3. Enhanced `analyze_gcm_results.py`

**New Features:**

1. **Aggregation from Individual Files**
   - `--aggregate`: Force aggregation from subject CSVs
   - `aggregate_subject_results()`: Combine per-subject results
   - `save_aggregated_results()`: Save group-level files

2. **Progress Tracking**
   - `--expected-subjects N`: Track completion progress
   - Shows `X/Y subjects (Z%)`
   - Warnings for incomplete data

3. **Watch Mode**
   - `--watch`: Continuously monitor for new results
   - `--watch-interval N`: Seconds between checks
   - Auto-exits when all subjects complete
   - Perfect for monitoring running jobs

4. **Robust Loading**
   - Handles missing group files (common in parallel runs)
   - Falls back to aggregation automatically
   - Works with partial results

### 4. Documentation

**`GCM_PARALLEL_GUIDE.md`** - Complete parallel processing guide
- Step-by-step instructions
- Troubleshooting section
- Monitoring tips
- Advanced usage examples

**`PARALLEL_PROCESSING_SUMMARY.md`** - This file
- Quick overview
- Key concepts
- Usage examples

---

## 🔑 Key Concepts

### Synchronized Timestamp

**Problem:** Array jobs start at different times, would create different timestamps

**Solution:**
1. Submit script generates timestamp FIRST
2. Creates directory with that timestamp
3. Passes timestamp to ALL array jobs
4. All jobs write to SAME folder

```bash
# In submit_gcm_parallel.sh
TIMESTAMP=$(date '+%m%d%Y%H%M%S')
mkdir -p gcm_roebroeck/$TIMESTAMP/csv

# Pass to array job
sbatch --array=0-49 slurm_gcm_parallel.sh $TIMESTAMP
```

### Thread-Safe Operations

**Problem:** Multiple jobs create directories simultaneously

**Solution:**
```python
# In ensure_dirs()
os.makedirs(fig_dir, exist_ok=True)  # Won't error if exists
os.makedirs(csv_dir, exist_ok=True)
```

### Progressive Aggregation

**Problem:** Want to see results before all jobs finish

**Solution:**
```bash
# Watch mode continuously aggregates as results arrive
python analyze_gcm_results.py -t <TIMESTAMP> --watch --expected-subjects 50
```

---

## 📋 Quick Usage

### Submit Jobs

```bash
# 50 subjects, alpha=0.05, 10 concurrent
./submit_gcm_parallel.sh -n 50 -A 50 -c 10
```

### Monitor Progress

```bash
# Watch mode (auto-updates every 30 seconds)
python analyze_gcm_results.py -t <TIMESTAMP> --watch --expected-subjects 50

# Manual check
ls gcm_roebroeck/<TIMESTAMP>/csv/*Adj_labeled.csv | wc -l
```

### Analyze Results

```bash
# After completion
python analyze_gcm_results.py -t <TIMESTAMP> --aggregate --plot
```

---

## 🎯 Workflow Diagram

```
submit_gcm_parallel.sh
    ↓
1. Generate TIMESTAMP
    ↓
2. Create gcm_roebroeck/TIMESTAMP/
    ↓
3. Submit Array Job
    ├─→ Task 0: Subject 0 → subj0_*.csv
    ├─→ Task 1: Subject 1 → subj1_*.csv
    ├─→ Task 2: Subject 2 → subj2_*.csv
    └─→ Task N: Subject N → subjN_*.csv
         ↓
    All write to SAME TIMESTAMP folder
         ↓
analyze_gcm_results.py --aggregate
    ↓
1. Load all subj*_Adj_labeled.csv
2. Aggregate into group_edge_*.csv
3. Generate group plot
4. Create analysis_plots/
```

---

## 🔧 Components

### File Structure

```
gunfolds/scripts/
├── submit_gcm_parallel.sh       ← Submit helper (NEW)
├── slurm_gcm_parallel.sh        ← Array job worker (NEW)
├── gcm_on_ICA.py                ← Main analysis (MODIFIED)
├── analyze_gcm_results.py       ← Analysis tool (ENHANCED)
├── GCM_PARALLEL_GUIDE.md        ← Detailed guide (NEW)
└── PARALLEL_PROCESSING_SUMMARY.md ← This file (NEW)
```

### Output Structure

```
gcm_roebroeck/
└── 12262025103045/              ← Shared timestamp
    ├── csv/
    │   ├── subj0_*.csv          ← From array task 0
    │   ├── subj1_*.csv          ← From array task 1
    │   ├── ...
    │   ├── group_edge_hits.csv  ← Aggregated by analysis
    │   └── group_edge_rate.csv  ← Aggregated by analysis
    ├── figures/
    │   ├── subj0_*.png
    │   ├── ...
    │   └── group_edge_frequency.png
    ├── analysis_plots/
    │   └── *.pdf
    └── run_metadata.txt

logs/
├── gcm_<JOB_ID>_0.out
├── gcm_<JOB_ID>_0.err
├── ...
├── latest_timestamp.txt
└── latest_job_id.txt
```

---

## 💡 Usage Examples

### Example 1: Basic Parallel Run

```bash
# Submit 50 subjects
./submit_gcm_parallel.sh -n 50 -A 50 -c 10
# Note timestamp: 12262025103045

# Monitor in real-time
python analyze_gcm_results.py -t 12262025103045 --watch --expected-subjects 50

# Final analysis (after completion)
python analyze_gcm_results.py -t 12262025103045 --aggregate --plot
```

### Example 2: Multiple Alpha Values

```bash
# Run with different alpha values in parallel
./submit_gcm_parallel.sh -n 50 -A 50 -c 10  # alpha=0.05
./submit_gcm_parallel.sh -n 50 -A 10 -c 10  # alpha=0.01
./submit_gcm_parallel.sh -n 50 -A 1 -c 10   # alpha=0.001

# Each gets its own timestamp, analyze separately
python analyze_gcm_results.py -t <TIMESTAMP1> --aggregate --plot
python analyze_gcm_results.py -t <TIMESTAMP2> --aggregate --plot
python analyze_gcm_results.py -t <TIMESTAMP3> --aggregate --plot
```

### Example 3: Resubmit Failed Subjects

```bash
# Check which subjects completed
ls gcm_roebroeck/12262025103045/csv/*Adj_labeled.csv | grep -oP 'subj\K[0-9]+'

# Subjects 10, 25, 30 failed - resubmit
sbatch --array=10,25,30 slurm_gcm_parallel.sh 12262025103045 50

# Reaggregate after completion
python analyze_gcm_results.py -t 12262025103045 --aggregate --plot
```

---

## 🔍 Verification

### Check All Subjects Completed

```bash
TIMESTAMP=12262025103045
EXPECTED=50
FOUND=$(ls gcm_roebroeck/$TIMESTAMP/csv/*Adj_labeled.csv 2>/dev/null | wc -l)

echo "$FOUND of $EXPECTED subjects completed"
```

### Verify Aggregation

```bash
# Check group files created
ls -lh gcm_roebroeck/12262025103045/csv/group_*.csv

# Check subjects count in group files
python -c "
import pandas as pd
hits = pd.read_csv('gcm_roebroeck/12262025103045/csv/group_edge_hits.csv', index_col=0)
print(f'Max hits: {hits.max().max()} (should equal number of subjects)')
"
```

---

## 🆚 Comparison: Serial vs Parallel

### Serial (Original)

```bash
# One process, all subjects
python gcm_on_ICA.py
```

**Pros:**
- Simple
- Auto-aggregation
- No SLURM needed

**Cons:**
- Slow for many subjects
- Single failure = restart all
- No parallelization

**Time:** ~2 min/subject × 50 subjects = **100 minutes**

### Parallel (New)

```bash
# Multiple processes, distributed
./submit_gcm_parallel.sh -n 50 -c 10
```

**Pros:**
- Fast (10x speedup with 10 concurrent)
- Independent subjects
- Restart only failed jobs
- Progress monitoring

**Cons:**
- Requires SLURM
- Manual aggregation
- More complex setup

**Time:** ~2 min/subject ÷ 10 concurrent = **10 minutes**

---

## 🛠️ Troubleshooting

### Issue: "Timestamp directory not found"

**Cause:** Submit script failed to create directory

**Fix:**
```bash
mkdir -p gcm_roebroeck/<TIMESTAMP>/csv
mkdir -p gcm_roebroeck/<TIMESTAMP>/figures
```

### Issue: Some subjects not completing

**Check job status:**
```bash
squeue -j <JOB_ID>
sacct -j <JOB_ID> --format=JobID,State,ExitCode
```

**Check error logs:**
```bash
grep -i error logs/gcm_*_<TASK_ID>.err
```

**Resubmit:**
```bash
sbatch --array=<FAILED_ID> slurm_gcm_parallel.sh <TIMESTAMP> <ALPHA>
```

### Issue: "No results found" in analysis

**Verify files exist:**
```bash
ls gcm_roebroeck/<TIMESTAMP>/csv/
```

**Check permissions:**
```bash
ls -la gcm_roebroeck/<TIMESTAMP>/
```

---

## 📚 Best Practices

1. **Test Small First**
   ```bash
   ./submit_gcm_parallel.sh -n 5 -c 2  # Test with 5 subjects
   ```

2. **Use Watch Mode**
   ```bash
   ./submit_gcm_parallel.sh -n 50 -c 10
   python analyze_gcm_results.py -t <TIMESTAMP> --watch --expected-subjects 50 &
   ```

3. **Save Metadata**
   - Always check `run_metadata.txt`
   - Note timestamp for reproducibility

4. **Monitor Logs**
   ```bash
   tail -f logs/gcm_*.out
   ```

5. **Verify Completion**
   ```bash
   # Before final analysis
   ls gcm_roebroeck/<TIMESTAMP>/csv/*Adj_labeled.csv | wc -l
   ```

---

## 🎓 Advanced Tips

### Resource Optimization

**Adjust SLURM parameters:**
```bash
# In slurm_gcm_parallel.sh
#SBATCH --time=02:00:00    # Increase if subjects timeout
#SBATCH --mem=4G           # Increase if OOM errors
#SBATCH --cpus-per-task=1  # Increase for parallel bootstrap
```

### Job Dependencies

**Chain jobs:**
```bash
JOB1=$(./submit_gcm_parallel.sh -n 50 -A 50 | grep "Job ID" | awk '{print $3}')
sbatch --dependency=afterok:$JOB1 <next_analysis_script>
```

### Bulk Submission

**Multiple parameter sets:**
```bash
for alpha in 50 10 5 1; do
    ./submit_gcm_parallel.sh -n 50 -A $alpha -c 10
    sleep 5  # Avoid overwhelming scheduler
done
```

---

## 📊 Performance Metrics

### Expected Speedup

| Concurrent Jobs | Sequential Time | Parallel Time | Speedup |
|----------------|-----------------|---------------|---------|
| 1 (serial)     | 100 min         | 100 min       | 1x      |
| 5              | 100 min         | 20 min        | 5x      |
| 10             | 100 min         | 10 min        | 10x     |
| 20             | 100 min         | 5 min         | 20x     |
| 50             | 100 min         | 2 min         | 50x     |

*Assumes 2 min per subject, 50 total subjects*

### Overhead

- **Job startup:** ~10-30 seconds per job
- **File I/O contention:** Minimal with separate output files
- **Aggregation:** ~1-5 seconds for 100 subjects

---

## ✨ Summary

### Key Achievements

✅ **Shared Timestamp** - All jobs write to same folder
✅ **Thread-Safe** - No race conditions
✅ **Watch Mode** - Real-time monitoring
✅ **Auto-Aggregation** - Combines results automatically
✅ **Progress Tracking** - Know when jobs complete
✅ **Flexible** - Works with any number of subjects
✅ **Robust** - Handle failures gracefully

### Files Modified/Created

| File | Status | Purpose |
|------|--------|---------|
| `submit_gcm_parallel.sh` | NEW | Submit array jobs |
| `slurm_gcm_parallel.sh` | NEW | Run single subject |
| `gcm_on_ICA.py` | MODIFIED | Thread-safe, timestamp support |
| `analyze_gcm_results.py` | ENHANCED | Aggregation, watch mode |
| `GCM_PARALLEL_GUIDE.md` | NEW | Detailed documentation |
| `PARALLEL_PROCESSING_SUMMARY.md` | NEW | This summary |

---

## 🔗 Related Documentation

- **`GCM_QUICK_START.md`** - Serial execution guide
- **`GCM_PARALLEL_GUIDE.md`** - Detailed parallel guide (READ THIS!)
- **`GCM_TIMESTAMPED_SUMMARY.md`** - Timestamping overview

---

**Ready to run in parallel! 🚀**

For detailed instructions, see `GCM_PARALLEL_GUIDE.md`

