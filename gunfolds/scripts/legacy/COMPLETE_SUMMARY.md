# Complete Hyperparameter Tuning Suite - Final Summary

## What You Have Now

A complete hyperparameter tuning framework with **two execution modes**:

### Mode 1: Local/Sequential Execution
For running on your local machine or testing smaller subsets

### Mode 2: Cluster/Parallel Execution  
For running all 3125 configurations in parallel on SLURM cluster ⭐

---

## All Files Created

### Core Scripts (11 files)

#### Local Execution:
1. **`VAR_hyperparameter_tuning.py`** - Local tuning (sequential or subset)
2. **`VAR_analyze_hyperparameters.py`** - Analysis and visualization
3. **`VAR_generate_report.py`** - Report generation
4. **`test_hyperparameter_tuning.py`** - Quick test script
5. **`run_hyperparameter_tuning.sh`** - Wrapper for local execution

#### Cluster Execution:
6. **`VAR_hyperparameter_single_job.py`** - Single job for cluster
7. **`slurm_hyperparameter_tuning.sh`** - SLURM submission script
8. **`VAR_collect_parallel_results.py`** - Collect parallel results

#### Documentation (8 files):
9. **`VAR_HYPERPARAMETER_TUNING_README.md`** - Detailed docs
10. **`HYPERPARAMETER_TUNING_SUMMARY.md`** - Complete guide
11. **`QUICK_START.txt`** - Quick reference (local)
12. **`CLUSTER_HYPERPARAMETER_GUIDE.md`** - Cluster detailed guide
13. **`CLUSTER_QUICK_START.txt`** - Quick reference (cluster)
14. **`COMPLETE_SUMMARY.md`** - This file

---

## Quick Decision Guide

### Should I Use Local or Cluster Execution?

| Factor | Local | Cluster |
|--------|-------|---------|
| **Number of configs** | < 20 | All 3125 |
| **Time** | 1-2 hours | 1-3 hours (parallel) |
| **Setup complexity** | Easy | Moderate |
| **Best for** | Testing, subset search | Exhaustive search |

---

## Quick Start: Local Execution

```bash
cd /Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts

# Test
python test_hyperparameter_tuning.py

# Run subset mode (~20 configs)
./run_hyperparameter_tuning.sh --subset

# Results auto-appear in: VAR_ruben/hyperparameter_tuning/
```

**Time:** 1-2 hours  
**Best for:** Quick exploration, finding good configurations

---

## Quick Start: Cluster Execution

```bash
# 1. Upload to cluster
scp VAR_hyperparameter_single_job.py user@cluster:/path/
scp slurm_hyperparameter_tuning.sh user@cluster:/path/
scp VAR_collect_parallel_results.py user@cluster:/path/

# 2. On cluster
ssh user@cluster
cd /path
mkdir -p err out logs VAR_ruben/hyperparameter_tuning/individual_jobs

# 3. Test first (IMPORTANT!)
sbatch --array=1-10 slurm_hyperparameter_tuning.sh

# 4. Submit all (after test succeeds)
sbatch --array=1-3125 slurm_hyperparameter_tuning.sh

# 5. Collect (after completion)
python VAR_collect_parallel_results.py -n 3 -u 3
```

**Time:** 1-3 hours (all 3125 configs in parallel)  
**Best for:** Exhaustive search, finding absolute best configuration

---

## What Gets Tested

All possible combinations of priorities `[p1, p2, p3, p4, p5]` where each priority is 1-5:

- **Total combinations:** 5^5 = 3,125
- **Original config:** `[4, 2, 5, 3, 1]` (from VAR_for_ruben_nets.py line 152)
- **Each config tested on:** 5 batches (for statistical averaging)

### Example Priority Configurations:
```python
[1, 1, 1, 1, 1]  # All equal priority
[5, 4, 3, 2, 1]  # Descending importance
[5, 5, 5, 5, 5]  # All maximum priority
[4, 2, 5, 3, 1]  # Original configuration
```

---

## Output and Results

### After Local Execution:
```
VAR_ruben/hyperparameter_tuning/
├── priority_tuning_net3_u3_TIMESTAMP.csv       # Results table
├── priority_tuning_net3_u3_TIMESTAMP.zkl       # Full data
├── priority_tuning_net3_u3_TIMESTAMP_report.md # Report
├── f1_comparison.png                            # Plots
├── combined_f1.png
├── priority_heatmap.png
├── precision_recall.png
└── summary_statistics.csv
```

### After Cluster Execution:
```
VAR_ruben/hyperparameter_tuning/
├── individual_jobs/
│   ├── job_0001_net3_u3.csv  # 3125 individual CSVs
│   └── job_0001_net3_u3.zkl  # 3125 individual ZKLs
├── priority_tuning_net3_u3_TIMESTAMP_parallel.csv  # Combined
├── priority_tuning_net3_u3_TIMESTAMP_parallel.zkl
└── failed_jobs_net3_u3_TIMESTAMP.txt (if any)
```

Then download and analyze locally:
```bash
scp cluster:/path/priority_tuning_*_parallel.csv .
python VAR_analyze_hyperparameters.py -f priority_tuning_*_parallel.csv
python VAR_generate_report.py -f priority_tuning_*_parallel.csv
```

---

## Key Metrics

Results are ranked by **Combined F1 Score**:

```
Combined F1 = (Orientation F1 + Adjacency F1 + Cycle F1) / 3
```

Where:
- **Orientation F1**: Edge direction accuracy
- **Adjacency F1**: Edge presence accuracy
- **Cycle F1**: 2-cycle (feedback loop) accuracy

Higher is better! (0.0 = worst, 1.0 = perfect)

---

## Recommended Workflow

### For Most Users:
1. **Start Local** - Run subset mode to explore
2. **Review top configs** - See which patterns work best
3. **Optional: Run cluster** - If you want exhaustive search

### For Thorough Research:
1. **Test locally** - Verify everything works
2. **Run cluster** - Submit all 3125 in parallel
3. **Comprehensive analysis** - Use all visualization tools

---

## Documentation Quick Links

| Document | Purpose | Read This If... |
|----------|---------|-----------------|
| `QUICK_START.txt` | Local quick ref | You want to start immediately (local) |
| `CLUSTER_QUICK_START.txt` | Cluster quick ref | You want to start immediately (cluster) |
| `HYPERPARAMETER_TUNING_SUMMARY.md` | Complete guide | You want full understanding |
| `CLUSTER_HYPERPARAMETER_GUIDE.md` | Cluster details | You're running on cluster |
| `VAR_HYPERPARAMETER_TUNING_README.md` | Technical docs | You need API details |

---

## Examples by Use Case

### Use Case 1: "I want to quickly find better priorities"
```bash
./run_hyperparameter_tuning.sh --subset
# Takes 1-2 hours, tests 20 configs
```

### Use Case 2: "I want to test everything exhaustively"
```bash
# On cluster:
sbatch --array=1-3125 slurm_hyperparameter_tuning.sh
python VAR_collect_parallel_results.py -n 3 -u 3
# Takes 1-3 hours (parallel), tests all 3125 configs
```

### Use Case 3: "I want to test custom priorities"
Edit `VAR_hyperparameter_tuning.py`, add your priorities to `get_priority_combinations()`, then:
```bash
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5
```

### Use Case 4: "I want to tune for different network/undersampling"
```bash
# Local:
./run_hyperparameter_tuning.sh --subset --net 2 --undersampling 2

# Cluster (edit slurm_hyperparameter_tuning.sh first):
# Change NET=2, UNDERSAMPLING=2, then submit
```

---

## Validation Checklist

Before you run hyperparameter tuning, verify:

### Local Execution:
- [ ] Data files exist: `~/DataSets_Feedbacks/8_VAR_simulation/net3/u3/txtSTD/data{1-5}.txt`
- [ ] Dependencies installed: `pip install tqdm pandas numpy matplotlib seaborn`
- [ ] Test passes: `python test_hyperparameter_tuning.py`

### Cluster Execution:
- [ ] Scripts uploaded to cluster
- [ ] Conda environment `multi_v3` exists
- [ ] Directories created: `err/`, `out/`, `logs/`, `VAR_ruben/`
- [ ] SLURM script configured (account, partition, memory, time)
- [ ] Test run succeeds: `sbatch --array=1-10 slurm_hyperparameter_tuning.sh`

---

## Resource Requirements

### Local (Subset Mode):
- **Time:** 1-2 hours
- **CPU:** 8 cores recommended
- **Memory:** 4-8 GB
- **Disk:** 1 GB

### Cluster (Full Mode):
- **Time:** 1-3 hours (parallel)
- **CPU:** 3,125 CPU-hours total
- **Memory:** 2 GB per job
- **Disk:** 150 GB total

---

## Troubleshooting

### Common Issues:

**"Data file not found"**
- Check: `~/DataSets_Feedbacks/8_VAR_simulation/net{NET}/u{UNDERSAMPLING}/txtSTD/`

**"Import error"**
- Install: `pip install tqdm pandas numpy matplotlib seaborn tigramite`

**"No results generated"**
- Run test script: `python test_hyperparameter_tuning.py`
- Check PCMCI is working properly

**Cluster jobs fail**
- Check conda environment exists
- Verify data paths are correct
- Test with `--array=1-10` first

**Collection says "not enough jobs"**
- Some failures are OK: `python VAR_collect_parallel_results.py --min_jobs 2500`
- Or resubmit failed jobs only

---

## Next Steps After Tuning

1. **Review results report** - Open the `*_report.md` file
2. **Check visualizations** - Look at all the `.png` plots
3. **Identify best config** - Note the top-ranked priorities
4. **Update your code** - Replace line 152 in `VAR_for_ruben_nets.py`:
   ```python
   priorities = [4, 2, 5, 3, 1]  # Original
   priorities = [5, 3, 4, 2, 1]  # Your best config from tuning
   ```
5. **Run your analysis** - Use the optimized priorities for your research

---

## Support

- **Local execution questions:** See `QUICK_START.txt` and `HYPERPARAMETER_TUNING_SUMMARY.md`
- **Cluster execution questions:** See `CLUSTER_QUICK_START.txt` and `CLUSTER_HYPERPARAMETER_GUIDE.md`
- **Technical details:** See `VAR_HYPERPARAMETER_TUNING_README.md`
- **Test first:** Run `python test_hyperparameter_tuning.py`

---

## Final Notes

✅ **All scripts are ready to use**  
✅ **No linter errors**  
✅ **Comprehensive documentation**  
✅ **Both local and cluster modes supported**  
✅ **Automatic analysis and visualization**  

**You're all set!** Choose your execution mode and start tuning. Good luck finding the optimal priority configuration! 🚀

---

## Quick Command Reference

```bash
# LOCAL - Test
python test_hyperparameter_tuning.py

# LOCAL - Run subset
./run_hyperparameter_tuning.sh --subset

# CLUSTER - Upload
scp VAR_hyperparameter_single_job.py user@cluster:/path/
scp slurm_hyperparameter_tuning.sh user@cluster:/path/
scp VAR_collect_parallel_results.py user@cluster:/path/

# CLUSTER - Submit
sbatch --array=1-3125 slurm_hyperparameter_tuning.sh

# CLUSTER - Collect
python VAR_collect_parallel_results.py -n 3 -u 3

# ANALYZE (either mode)
python VAR_analyze_hyperparameters.py -f RESULTS.csv
python VAR_generate_report.py -f RESULTS.csv
```

