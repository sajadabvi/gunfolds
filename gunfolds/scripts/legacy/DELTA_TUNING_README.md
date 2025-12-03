# Delta Hyperparameter Tuning for DRASL Solution Selection

This set of scripts performs hyperparameter tuning for the **delta parameter** used in selecting DRASL solutions based on cost.

## Overview

### What is Delta?

When DRASL returns multiple solutions, we need to decide which ones to consider. The **delta parameter** determines solution selection based on cost:

1. Sort all solutions by cost (lower is better)
2. Find the minimum cost: `min_cost`
3. Select all solutions where: `cost <= min_cost + delta`
4. Compute F1 scores for selected solutions
5. Report the **average F1 score** across selected solutions

### Goal

Find the optimal delta value that **maximizes the average F1 score** across selected solutions.

### Key Insight

- **Delta = 0**: Only select the single lowest-cost solution
- **Small delta**: Select a few solutions near the minimum cost
- **Large delta**: Select many solutions (potentially all of them)
- **Optimal delta**: Balances solution quality and diversity

## Files in This Framework

### Core Scripts

1. **VAR_delta_tuning.py** - Local/sequential execution
2. **VAR_delta_single_job.py** - Single job for cluster (SLURM)
3. **slurm_delta_tuning.sh** - SLURM submission script
4. **VAR_collect_delta_results.py** - Collect parallel results
5. **VAR_analyze_delta.py** - Analysis and visualization

### Documentation

6. **DELTA_TUNING_README.md** - This file (technical details)
7. **DELTA_QUICK_START.txt** - Quick reference guide
8. **DELTA_COMPLETE_SUMMARY.md** - Comprehensive guide

## Usage

### Option 1: Local Execution (Sequential)

Best for: Testing, small delta ranges, understanding the system

```bash
# Test mode (5 delta values)
python VAR_delta_tuning.py -n 3 -u 3 --num_batches 5 --test_mode

# Custom range
python VAR_delta_tuning.py -n 3 -u 3 --num_batches 5 \
    --delta_min 0 --delta_max 50000 --delta_step 2500

# Default (0 to 100000, step 5000)
python VAR_delta_tuning.py -n 3 -u 3 --num_batches 5
```

**Arguments:**
- `-n, --NET`: Network number (default: 3)
- `-u, --UNDERSAMPLING`: Undersampling rate (default: 3)
- `--num_batches`: Number of batches to average over (default: 5)
- `--delta_min`: Minimum delta value (default: 0)
- `--delta_max`: Maximum delta value (default: 100000)
- `--delta_step`: Step size for delta values (default: 5000)
- `--test_mode`: Test with only 5 delta values
- `-p, --PNUM`: Number of CPUs to use

**Time:** Depends on number of delta values (typically 1-3 hours for 20 values)

### Option 2: Cluster Execution (Parallel)

Best for: Comprehensive search, many delta values, faster results

#### Step 1: Calculate Number of Jobs

```bash
# Formula: N = (DELTA_MAX - DELTA_MIN) / DELTA_STEP + 1

# Example: delta_min=0, delta_max=100000, delta_step=5000
# N = (100000 - 0) / 5000 + 1 = 21 jobs
```

#### Step 2: Edit SLURM Script

Edit `slurm_delta_tuning.sh` and set:
```bash
DELTA_MIN=0
DELTA_MAX=100000
DELTA_STEP=5000
NET=3
UNDERSAMPLING=3
NUM_BATCHES=5
```

#### Step 3: Upload to Cluster

```bash
scp VAR_delta_single_job.py user@cluster:/path/
scp slurm_delta_tuning.sh user@cluster:/path/
scp VAR_collect_delta_results.py user@cluster:/path/
```

#### Step 4: Submit Jobs

```bash
ssh user@cluster
cd /path

# Create directories
mkdir -p err out logs VAR_ruben/delta_tuning/individual_jobs

# Test first (10 jobs)
sbatch --array=1-10 slurm_delta_tuning.sh

# Check results
ls VAR_ruben/delta_tuning/individual_jobs/

# Submit all (after test succeeds)
sbatch --array=1-21 slurm_delta_tuning.sh  # Replace 21 with your N
```

#### Step 5: Collect Results

After all jobs complete:

```bash
# On cluster
python VAR_collect_delta_results.py -n 3 -u 3 --expected_jobs 21

# Download to local machine
scp cluster:/path/VAR_ruben/delta_tuning/delta_tuning_*_parallel.csv .
```

**Time:** 1-2 hours (all jobs run in parallel)

### Analyzing Results

After running tuning (local or cluster):

```bash
python VAR_analyze_delta.py -f VAR_ruben/delta_tuning/delta_tuning_net3_u3_TIMESTAMP.csv
```

This generates:
- **delta_vs_f1_analysis.png** - How F1 changes with delta
- **f1_comparison.png** - Top configurations comparison
- **combined_f1_top10.png** - Top 10 delta values
- **solutions_vs_f1.png** - Relationship between solutions and performance
- **summary_statistics.csv** - Statistical summary

## Understanding Results

### Key Metrics

1. **Combined F1**: Average of orientation, adjacency, and cycle F1 scores
   - Primary metric for ranking delta values
   - Range: 0.0 (worst) to 1.0 (perfect)

2. **Orientation F1**: Accuracy of edge directions
3. **Adjacency F1**: Accuracy of edge presence
4. **Cycle F1**: Accuracy of 2-cycles (feedback loops)

5. **Avg Solutions Selected**: How many solutions are included
6. **Avg Solutions Total**: Total solutions available

### Interpreting Results

**Good Configuration:**
- High combined F1 score
- Reasonable number of solutions selected (not just 1, not all)
- Balanced precision and recall

**Trade-offs:**
- Very small delta → Only 1 solution → May miss good solutions
- Very large delta → All solutions → May include poor solutions
- Optimal delta → Sweet spot for averaging

### Example Output

```
BEST CONFIGURATION:
Delta: 15000
Combined F1: 0.8542
  - Orientation F1: 0.8234
  - Adjacency F1: 0.8912
  - Cycle F1: 0.8480
Avg solutions selected: 12.4/45.2
```

This means: Delta of 15,000 selects ~12 solutions on average (out of ~45 total), achieving a combined F1 of 0.8542.

## Implementation in Your Code

After finding the optimal delta, update your analysis code:

```python
# Current approach (in VAR_for_ruben_nets.py lines 166-175):
max_f1_score = 0
for answer in r_estimated:
    res_rasl = bfutils.num2CG(answer[0][0], len(network_GT))
    rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
    curr_f1 = (rasl_sol['orientation']['F1']) + (rasl_sol['adjacency']['F1']) + (rasl_sol['cycle']['F1'])
    if curr_f1 > max_f1_score:
        max_f1_score = curr_f1
        max_answer = answer

# New approach (with optimized delta):
OPTIMAL_DELTA = 15000  # From tuning results

# Extract solutions with costs
solutions_with_costs = []
for answer in r_estimated:
    graph_num = answer[0][0]
    undersampling = answer[0][1]
    cost = answer[1]
    solutions_with_costs.append((graph_num, undersampling, cost))

# Sort by cost
solutions_with_costs.sort(key=lambda x: x[2])
min_cost = solutions_with_costs[0][2]

# Select solutions within delta
selected_solutions = [s for s in solutions_with_costs if s[2] <= min_cost + OPTIMAL_DELTA]

# Compute average F1 across selected solutions
f1_scores = []
for graph_num, undersampling, cost in selected_solutions:
    res_rasl = bfutils.num2CG(graph_num, len(network_GT))
    rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=include_selfloop)
    f1_scores.append(rasl_sol)

# Average metrics
avg_orientation_f1 = np.mean([s['orientation']['F1'] for s in f1_scores])
avg_adjacency_f1 = np.mean([s['adjacency']['F1'] for s in f1_scores])
avg_cycle_f1 = np.mean([s['cycle']['F1'] for s in f1_scores])
```

## Technical Details

### Data Requirements

- Data files: `~/DataSets_Feedbacks/8_VAR_simulation/net{NET}/u{UNDERSAMPLING}/txtSTD/data{1-N}.txt`
- Format: Tab-delimited time series data
- Generated by: VAR simulation with specified network and undersampling

### Dependencies

```bash
pip install numpy pandas matplotlib seaborn tqdm tigramite
```

Also requires:
- gunfolds package (with all modules)
- PCMCI for causal discovery
- Clingo for DRASL solver

### Priorities (Fixed)

The priorities used in DRASL are fixed at the original values:
```python
priorities = [4, 2, 5, 3, 1]
```

If you want to tune both delta and priorities, you would need to:
1. First tune priorities (using VAR_hyperparameter_tuning.py)
2. Then tune delta with optimal priorities

### Resource Requirements

**Local Execution:**
- Time: 2-4 hours for 20 delta values
- CPU: 8 cores recommended
- Memory: 4-8 GB
- Disk: ~1 GB

**Cluster Execution:**
- Time: 1-2 hours (parallel)
- CPU: N jobs × 1 core each
- Memory: 2 GB per job
- Disk: ~50 MB per job

## Troubleshooting

### "Data file not found"
- Check path: `~/DataSets_Feedbacks/8_VAR_simulation/net{NET}/u{UNDERSAMPLING}/txtSTD/`
- Ensure data files exist for specified network and undersampling

### "No solutions returned by DRASL"
- Check PCMCI is finding a valid graph
- Verify priorities are set correctly
- Check data quality and preprocessing

### "All F1 scores are similar"
- Delta range may be too narrow or too wide
- Try different delta_min, delta_max, delta_step values
- Consider the cost distribution of your solutions

### Cluster jobs fail
- Verify conda environment `multi_v3` exists
- Check data paths on cluster
- Test with small array first: `sbatch --array=1-5 slurm_delta_tuning.sh`

### Collection fails
- Check individual job outputs: `ls VAR_ruben/delta_tuning/individual_jobs/`
- Look at SLURM error logs: `cat err/error*.err`
- Some failures OK: use `--min_jobs` to set lower threshold

## Comparison with Priorities Tuning

| Aspect | Priorities Tuning | Delta Tuning |
|--------|-------------------|--------------|
| **What it tunes** | RASL constraint weights | Solution selection threshold |
| **Number of configs** | 3,125 (5^5 combinations) | ~20-100 (depends on range) |
| **Affects** | Which solutions DRASL finds | Which solutions we average |
| **Time (cluster)** | 2-4 hours | 1-2 hours |
| **When to use** | Optimize DRASL search | Optimize result aggregation |
| **Can combine?** | Yes! Tune priorities first, then delta |

## Advanced Usage

### Fine-Grained Search

After initial tuning, narrow down to fine-grained search:

```bash
# Initial: broad search
python VAR_delta_tuning.py -n 3 -u 3 --delta_min 0 --delta_max 100000 --delta_step 10000

# Results show optimal around 15000
# Fine-grained: narrow search
python VAR_delta_tuning.py -n 3 -u 3 --delta_min 10000 --delta_max 20000 --delta_step 1000
```

### Different Networks

Test delta on different networks and undersampling rates:

```bash
python VAR_delta_tuning.py -n 1 -u 2 --num_batches 5
python VAR_delta_tuning.py -n 2 -u 3 --num_batches 5
python VAR_delta_tuning.py -n 3 -u 3 --num_batches 5
```

Delta optimal for one network may not be optimal for others.

### Custom Delta Lists

For non-uniform spacing, edit `get_delta_values()` in `VAR_delta_tuning.py`:

```python
def get_delta_values(args):
    # Custom list
    return [0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000, 100000]
```

## Support

For issues, questions, or additional documentation:
- See `DELTA_QUICK_START.txt` for quick reference
- See `DELTA_COMPLETE_SUMMARY.md` for comprehensive guide
- Check error logs in `err/` directory (cluster)
- Verify all dependencies are installed

## Citation

If you use this delta tuning framework in your research, please cite the underlying methods:
- DRASL/RASL solver from the gunfolds package
- PCMCI algorithm from tigramite package

---

**Version:** 1.0  
**Date:** 2025-11-26  
**Author:** Multi-causal discovery research group

