# Complete Delta Hyperparameter Tuning Suite - Final Summary

## What You Have Now

A complete delta hyperparameter tuning framework with **two execution modes**:

### Mode 1: Local/Sequential Execution
For running on your local machine or testing smaller delta ranges

### Mode 2: Cluster/Parallel Execution  
For running comprehensive delta searches in parallel on SLURM cluster ⭐

---

## What is Delta and Why Tune It?

### The Problem

When DRASL returns multiple solutions, you need to decide:
- Use only the lowest-cost solution?
- Use a few solutions near the minimum cost?
- Average across many solutions?

**Traditional approach (VAR_for_ruben_nets.py lines 166-175):**
- Iterate through ALL solutions
- Compute F1 for each with ground truth
- Pick the single best one

**Problem:** This uses ground truth (not available in real analysis) and picks only one solution.

### The Solution: Delta-Based Selection

**New approach:**
1. Sort all solutions by cost (lower is better)
2. Find minimum cost: `min_cost`
3. Select solutions where: `cost <= min_cost + delta`
4. Compute F1 for each selected solution
5. Report **average F1** across selected solutions

**Benefit:** No ground truth needed, averages multiple good solutions, delta is tunable!

### What Delta Means

- **Delta = 0**: Only select the single minimum-cost solution
- **Delta = 1,000**: Select solutions within 1,000 of minimum cost
- **Delta = 10,000**: Select solutions within 10,000 of minimum cost
- **Delta = ∞**: Select all solutions

**Goal:** Find the delta that maximizes average F1 score!

---

## All Files Created

### Core Scripts (5 files)

#### Local Execution:
1. **`VAR_delta_tuning.py`** - Local tuning (sequential)
2. **`VAR_analyze_delta.py`** - Analysis and visualization

#### Cluster Execution:
3. **`VAR_delta_single_job.py`** - Single job for cluster
4. **`slurm_delta_tuning.sh`** - SLURM submission script
5. **`VAR_collect_delta_results.py`** - Collect parallel results

#### Documentation (3 files):
6. **`DELTA_TUNING_README.md`** - Detailed technical docs
7. **`DELTA_QUICK_START.txt`** - Quick reference
8. **`DELTA_COMPLETE_SUMMARY.md`** - This file

---

## Quick Decision Guide

### Should I Use Local or Cluster Execution?

| Factor | Local | Cluster |
|--------|-------|---------|
| **Number of delta values** | < 20 | 20-100 |
| **Time** | 2-3 hours | 1-2 hours (parallel) |
| **Setup complexity** | Easy | Moderate |
| **Best for** | Testing, quick search | Comprehensive search |

---

## Quick Start: Local Execution

```bash
cd /Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts

# Test mode (5 delta values, ~30 minutes)
python VAR_delta_tuning.py -n 3 -u 3 --num_batches 5 --test_mode

# Full range (21 delta values, ~2-3 hours)
python VAR_delta_tuning.py -n 3 -u 3 --num_batches 5

# Analyze results
python VAR_analyze_delta.py -f VAR_ruben/delta_tuning/delta_tuning_net3_u3_*.csv

# Results auto-appear in: VAR_ruben/delta_tuning/
```

**Time:** 30 min (test) to 2-3 hours (full)  
**Best for:** Quick exploration, finding good delta range

---

## Quick Start: Cluster Execution

```bash
# 1. Calculate number of jobs
# Formula: N = (DELTA_MAX - DELTA_MIN) / DELTA_STEP + 1
# Example: (100000 - 0) / 5000 + 1 = 21 jobs

# 2. Upload to cluster
scp VAR_delta_single_job.py user@cluster:/path/
scp slurm_delta_tuning.sh user@cluster:/path/
scp VAR_collect_delta_results.py user@cluster:/path/

# 3. On cluster
ssh user@cluster
cd /path
mkdir -p err out logs VAR_ruben/delta_tuning/individual_jobs

# 4. Edit slurm_delta_tuning.sh
# Set: DELTA_MIN=0, DELTA_MAX=100000, DELTA_STEP=5000

# 5. Test first (IMPORTANT!)
sbatch --array=1-10 slurm_delta_tuning.sh

# 6. Submit all (after test succeeds)
sbatch --array=1-21 slurm_delta_tuning.sh

# 7. Collect (after completion)
python VAR_collect_delta_results.py -n 3 -u 3 --expected_jobs 21

# 8. Download and analyze
scp cluster:/path/VAR_ruben/delta_tuning/delta_tuning_*_parallel.csv .
python VAR_analyze_delta.py -f delta_tuning_*_parallel.csv
```

**Time:** 1-2 hours (all jobs in parallel)  
**Best for:** Comprehensive search, finding absolute best delta

---

## What Gets Tested

Different delta values determine solution selection threshold:

### Default Configuration:
- **Delta range:** 0 to 100,000
- **Step size:** 5,000
- **Total values:** 21 delta values
- **Batches per value:** 5 (for statistical averaging)

### Example Delta Values:
```python
[0, 5000, 10000, 15000, 20000, ..., 100000]
```

### Test Mode:
```python
[0, 1000, 5000, 10000, 20000]  # Only 5 values for quick testing
```

### Customizable:
```bash
python VAR_delta_tuning.py --delta_min 0 --delta_max 50000 --delta_step 2500
# Tests: [0, 2500, 5000, 7500, ..., 50000]
```

---

## Output and Results

### After Local Execution:
```
VAR_ruben/delta_tuning/
├── delta_tuning_net3_u3_TIMESTAMP.csv           # Results table
├── delta_tuning_net3_u3_TIMESTAMP.zkl           # Full data
├── delta_vs_f1_analysis.png                     # Main analysis plot
├── f1_comparison.png                            # Top configs comparison
├── combined_f1_top10.png                        # Top 10 bar chart
├── solutions_vs_f1.png                          # Solutions relationship
└── summary_statistics.csv                       # Statistical summary
```

### After Cluster Execution:
```
VAR_ruben/delta_tuning/
├── individual_jobs/
│   ├── job_0001_net3_u3.csv                    # 21 individual CSVs
│   ├── job_0001_net3_u3.zkl                    # 21 individual ZKLs
│   └── ...
├── delta_tuning_net3_u3_TIMESTAMP_parallel.csv  # Combined results
├── delta_tuning_net3_u3_TIMESTAMP_parallel.zkl  # Combined data
└── failed_jobs_net3_u3_TIMESTAMP.txt (if any)
```

Then analyze locally:
```bash
python VAR_analyze_delta.py -f delta_tuning_*_parallel.csv
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

Additional metrics:
- **Avg Solutions Selected**: How many solutions are averaged
- **Avg Solutions Total**: Total solutions available from DRASL

Higher F1 is better! (0.0 = worst, 1.0 = perfect)

---

## Understanding Results

### Example Output:

```
BEST CONFIGURATION:
Delta: 15000
Combined F1: 0.8542
  - Orientation F1: 0.8234
  - Adjacency F1: 0.8912
  - Cycle F1: 0.8480
Avg solutions selected: 12.4/45.2
```

### Interpretation:

- **Delta = 15,000** is optimal
- Selects ~12 solutions on average (out of ~45 total)
- These 12 solutions have costs within 15,000 of the minimum
- Averaging their F1 scores gives 0.8542

### Trade-offs:

| Delta Value | Solutions Selected | Pros | Cons |
|-------------|-------------------|------|------|
| 0 | 1 (minimum cost) | Fast, simple | May miss good solutions |
| 5,000 | ~5 solutions | Focused averaging | Still limited |
| 15,000 | ~12 solutions | Good balance ⭐ | - |
| 50,000 | ~30 solutions | More averaging | May include poor solutions |
| 100,000 | All solutions | Maximum averaging | Includes low-quality solutions |

**Key insight:** There's a sweet spot where averaging improves performance without including too many poor solutions!

---

## Recommended Workflow

### For Most Users:
1. **Test locally** - Run test mode (30 min) to verify setup
2. **Run cluster** - Submit full range in parallel (1-2 hours)
3. **Analyze results** - Find optimal delta
4. **Update code** - Use optimal delta in your analysis

### For Quick Testing:
1. **Test mode only** - Run local test mode (30 min)
2. **Review results** - See which delta range looks promising
3. **Optional: Refine** - Run more focused search if needed

### For Thorough Research:
1. **Broad search** - Run delta 0-100k, step 10k (11 values)
2. **Identify range** - See where optimum is (e.g., around 15k)
3. **Narrow search** - Run delta 10k-20k, step 1k (11 values)
4. **Final analysis** - Use refined optimal delta

---

## Visualizations Generated

### 1. Delta vs F1 Analysis (4 subplots)
- **Combined F1 vs Delta**: Main relationship
- **Individual F1 metrics**: Orientation, Adjacency, Cycle
- **Solutions selected**: How many solutions at each delta
- **Precision vs Recall**: Trade-off colored by delta

### 2. F1 Comparison (Bar Chart)
- Top 10 configurations
- Side-by-side bars for three F1 metrics
- Easy comparison of performance

### 3. Combined F1 Top 10 (Horizontal Bars)
- Ranked by combined F1
- Best configuration highlighted
- Quick identification of optimal delta

### 4. Solutions vs F1 (2 subplots)
- **Absolute count**: Solutions selected vs F1
- **Percentage**: % of solutions vs F1
- Shows relationship between averaging and performance

All plots saved as high-resolution PNG files (300 DPI).

---

## Documentation Quick Links

| Document | Purpose | Read This If... |
|----------|---------|-----------------|
| `DELTA_QUICK_START.txt` | Quick reference | You want to start immediately |
| `DELTA_TUNING_README.md` | Technical docs | You need detailed information |
| `DELTA_COMPLETE_SUMMARY.md` | Comprehensive guide | You want full understanding (this file) |

---

## Examples by Use Case

### Use Case 1: "I want to quickly test if delta tuning helps"
```bash
python VAR_delta_tuning.py -n 3 -u 3 --test_mode
# Takes 30 min, tests 5 delta values
```

### Use Case 2: "I want to find the optimal delta comprehensively"
```bash
# On cluster:
sbatch --array=1-21 slurm_delta_tuning.sh
python VAR_collect_delta_results.py -n 3 -u 3
# Takes 1-2 hours (parallel), tests 21 delta values
```

### Use Case 3: "I want to fine-tune after initial search"
```bash
# First run finds optimal around 15000
# Fine-tune with smaller steps:
python VAR_delta_tuning.py --delta_min 10000 --delta_max 20000 --delta_step 1000
```

### Use Case 4: "I want to test delta for different networks"
```bash
python VAR_delta_tuning.py -n 1 -u 2 --num_batches 5
python VAR_delta_tuning.py -n 2 -u 3 --num_batches 5
python VAR_delta_tuning.py -n 3 -u 3 --num_batches 5
# Optimal delta may differ by network!
```

---

## Validation Checklist

Before you run delta tuning, verify:

### Local Execution:
- [ ] Data files exist: `~/DataSets_Feedbacks/8_VAR_simulation/net3/u3/txtSTD/data{1-5}.txt`
- [ ] Dependencies installed: `pip install tqdm pandas numpy matplotlib seaborn tigramite`
- [ ] Test mode works: `python VAR_delta_tuning.py --test_mode`

### Cluster Execution:
- [ ] Scripts uploaded to cluster
- [ ] Conda environment `multi_v3` exists
- [ ] Directories created: `err/`, `out/`, `logs/`, `VAR_ruben/delta_tuning/`
- [ ] SLURM script configured (DELTA_MIN, DELTA_MAX, DELTA_STEP)
- [ ] Number of jobs calculated: N = (MAX - MIN) / STEP + 1
- [ ] Test run succeeds: `sbatch --array=1-10 slurm_delta_tuning.sh`

---

## Resource Requirements

### Local (Test Mode):
- **Time:** 30 minutes
- **CPU:** 8 cores recommended
- **Memory:** 4-8 GB
- **Disk:** 500 MB

### Local (Full Range, 21 values):
- **Time:** 2-3 hours
- **CPU:** 8 cores recommended
- **Memory:** 4-8 GB
- **Disk:** 1 GB

### Cluster (21 jobs):
- **Time:** 1-2 hours (parallel)
- **CPU:** 21 jobs × 1 core = 21 CPU-hours
- **Memory:** 2 GB per job
- **Disk:** 50 MB per job, ~1 GB total

---

## Troubleshooting

### Common Issues:

**"Data file not found"**
- Check: `~/DataSets_Feedbacks/8_VAR_simulation/net{NET}/u{UNDERSAMPLING}/txtSTD/`
- Verify network and undersampling values are correct

**"Import error"**
- Install: `pip install tqdm pandas numpy matplotlib seaborn tigramite`
- Verify gunfolds package is installed

**"No results generated"**
- Check PCMCI is working: test mode should complete
- Verify data quality and format

**Cluster jobs fail**
- Check conda environment exists: `conda env list`
- Verify data paths are correct on cluster
- Test with `--array=1-5` first

**Collection says "no files found"**
- Check: `ls VAR_ruben/delta_tuning/individual_jobs/`
- Some failures OK: use `--min_jobs` to set threshold
- Resubmit failed jobs if needed

**All delta values have similar F1**
- Delta range may be too narrow or too wide
- Try different min/max/step values
- Check if DRASL is returning diverse solutions

---

## Next Steps After Tuning

1. **Review results report** - Look at generated plots
2. **Check summary statistics** - Review `summary_statistics.csv`
3. **Identify best delta** - Note the optimal value
4. **Update your code** - Implement delta-based selection in `VAR_for_ruben_nets.py`
5. **Run your analysis** - Use optimized delta for research

### Code Update Example

Replace lines 166-175 in `VAR_for_ruben_nets.py`:

```python
# OLD CODE (uses ground truth to pick single best):
max_f1_score = 0
for answer in r_estimated:
    res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
    rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
    curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])
    if curr_f1 > max_f1_score:
        max_f1_score = curr_f1
        max_answer = answer
res_rasl = bfutils.num2CG(max_answer[0][0], len(network_GT))

# NEW CODE (uses optimized delta, no ground truth needed):
OPTIMAL_DELTA = 15000  # From tuning results

# Extract solutions with costs
solutions_with_costs = []
for answer in r_estimated:
    graph_num = answer[0][0]
    undersampling = answer[0][1]
    cost = answer[1]
    solutions_with_costs.append((graph_num, undersampling, cost))

# Sort by cost and select within delta
solutions_with_costs.sort(key=lambda x: x[2])
min_cost = solutions_with_costs[0][2]
selected_solutions = [s for s in solutions_with_costs if s[2] <= min_cost + OPTIMAL_DELTA]

# Compute F1 for selected solutions (still using GT for validation)
f1_scores = []
for graph_num, undersampling, cost in selected_solutions:
    res_rasl = bfutils.num2CG(graph_num, len(network_GT))
    rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
    f1_scores.append(rasl_sol)

# Average the metrics
avg_orientation_f1 = np.mean([s['orientation']['F1'] for s in f1_scores])
avg_adjacency_f1 = np.mean([s['adjacency']['F1'] for s in f1_scores])
avg_cycle_f1 = np.mean([s['cycle']['F1'] for s in f1_scores])

# For final output, you might pick the middle solution or use all
# This example uses the solution closest to median cost:
median_idx = len(selected_solutions) // 2
res_rasl = bfutils.num2CG(selected_solutions[median_idx][0], len(network_GT))
```

---

## Comparison: Delta vs Priorities Tuning

| Aspect | Priorities Tuning | Delta Tuning |
|--------|-------------------|--------------|
| **What it tunes** | RASL constraint weights (5 values) | Solution selection threshold (1 value) |
| **Search space** | 3,125 combinations (5^5) | ~20-100 values (depends on range) |
| **Affects** | Which solutions DRASL finds | Which solutions we average |
| **Time (cluster)** | 2-4 hours (3125 jobs) | 1-2 hours (20-100 jobs) |
| **Complexity** | Higher (many combinations) | Lower (single parameter) |
| **When to use** | Optimize DRASL search itself | Optimize result aggregation |
| **Can combine?** | Yes! Tune priorities first, then delta with optimal priorities |

### Combined Workflow

1. **First:** Tune priorities (comprehensive)
   - Find optimal priority configuration
   - Takes 2-4 hours on cluster

2. **Second:** Tune delta (with optimal priorities)
   - Fix priorities to optimal values
   - Find optimal delta
   - Takes 1-2 hours on cluster

3. **Result:** Fully optimized system!

---

## Advanced Features

### 1. Non-Uniform Delta Spacing

Edit `get_delta_values()` in `VAR_delta_tuning.py`:

```python
def get_delta_values(args):
    # Custom list with non-uniform spacing
    return [0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
```

### 2. Multi-Network Testing

```bash
# Test same delta range on multiple networks
for net in 1 2 3; do
    for u in 2 3; do
        python VAR_delta_tuning.py -n $net -u $u --num_batches 5
    done
done
```

### 3. Adaptive Delta Search

Run coarse search first, then refine:

```bash
# Step 1: Coarse search (10k steps)
python VAR_delta_tuning.py --delta_step 10000

# Results show optimal around 20k
# Step 2: Fine search (1k steps)
python VAR_delta_tuning.py --delta_min 10000 --delta_max 30000 --delta_step 1000
```

### 4. Statistical Analysis

The framework averages over multiple batches for robust results:
- Default: 5 batches
- More batches = more robust, but slower
- Fewer batches = faster, but less reliable

```bash
# More robust (slower)
python VAR_delta_tuning.py --num_batches 10

# Faster (less robust)
python VAR_delta_tuning.py --num_batches 3
```

---

## Support

- **Quick start:** See `DELTA_QUICK_START.txt`
- **Technical details:** See `DELTA_TUNING_README.md`
- **Test first:** Run test mode before full execution
- **Check logs:** Look in `err/` directory for errors (cluster)

---

## Final Notes

✅ **All scripts are ready to use**  
✅ **No linter errors** 
✅ **Comprehensive documentation**  
✅ **Both local and cluster modes supported**  
✅ **Automatic analysis and visualization**  
✅ **Works standalone or combined with priorities tuning**

**You're all set!** Choose your execution mode and start tuning delta. Good luck finding the optimal configuration! 🚀

---

## Quick Command Reference

```bash
# LOCAL - Test mode (5 delta values)
python VAR_delta_tuning.py -n 3 -u 3 --test_mode

# LOCAL - Full range (21 delta values)
python VAR_delta_tuning.py -n 3 -u 3 --num_batches 5

# CLUSTER - Calculate jobs
# N = (DELTA_MAX - DELTA_MIN) / DELTA_STEP + 1

# CLUSTER - Upload
scp VAR_delta_single_job.py user@cluster:/path/
scp slurm_delta_tuning.sh user@cluster:/path/
scp VAR_collect_delta_results.py user@cluster:/path/

# CLUSTER - Submit
sbatch --array=1-N slurm_delta_tuning.sh

# CLUSTER - Collect
python VAR_collect_delta_results.py -n 3 -u 3

# ANALYZE (either mode)
python VAR_analyze_delta.py -f RESULTS.csv
```

---

**End of Complete Summary**

