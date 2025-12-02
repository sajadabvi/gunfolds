# Hyperparameter Tuning Guide

This guide consolidates information from:
- VAR_HYPERPARAMETER_TUNING_README.md
- HYPERPARAMETER_TUNING_SUMMARY.md
- COMPLETE_SUMMARY.md
- QUICK_START.txt

## Overview

The RASL algorithm uses 5 priority values `[p1, p2, p3, p4, p5]` to weight different constraints during optimization. Each priority can be 1-5 (5 = highest priority). Different combinations significantly affect performance.

## Quick Start

### Step 1: Test Your Setup

```bash
cd /path/to/gunfolds/scripts
python test_hyperparameter_tuning.py
```

Verifies:
- All dependencies installed
- Data files exist
- Basic functionality works

### Step 2: Run Hyperparameter Tuning

**Option A: Subset Mode (Recommended)**
```bash
./run_hyperparameter_tuning.sh --subset
```
- Tests ~20 curated configurations
- Takes 1-2 hours
- Good for finding optimal parameters

**Option B: Default Mode (Quick Test)**
```bash
./run_hyperparameter_tuning.sh
```
- Tests 5 representative configurations
- Takes 15-30 minutes
- Good for testing workflow

**Option C: Full Search (Exhaustive)**
```bash
./run_hyperparameter_tuning.sh --all
```
- Tests all 3125 possible configurations
- Takes several days!
- Use cluster execution instead (see below)

### Step 3: Analyze Results

Results automatically saved to: `VAR_ruben/hyperparameter_tuning/`

View generated files:
- `priority_tuning_net3_u3_TIMESTAMP.csv` - Results table
- `priority_tuning_net3_u3_TIMESTAMP_report.md` - Full report
- `f1_comparison.png` - F1 score charts
- `combined_f1.png` - Combined scores
- `priority_heatmap.png` - Priority patterns
- `precision_recall.png` - P-R analysis

## Execution Modes

### Local Execution (Sequential)

**Advantages:**
- Easy setup
- Good for testing small subsets
- No cluster access needed

**Disadvantages:**
- Slow for exhaustive search
- Only uses one machine

**Usage:**
```bash
# Subset mode (~20 configs × 5 batches)
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5 --test_subset

# Custom selection
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5 --priorities "[5,5,5,5,5]" "[3,3,3,3,3]"
```

### Cluster Execution (Parallel)

**Advantages:**
- Can run all 3125 configurations
- Completes in 1-3 hours (parallel)
- Efficient resource usage

**Disadvantages:**
- Requires SLURM cluster access
- More complex setup

**Usage:**

1. **Submit Array Job:**
```bash
sbatch --array=1-3125 slurm_hyperparameter_tuning.sh
```

2. **Monitor Progress:**
```bash
squeue -u $USER
```

3. **Collect Results:**
```bash
python VAR_collect_parallel_results.py --output-dir VAR_ruben/hyperparameter_tuning/parallel_TIMESTAMP
```

4. **Analyze:**
```bash
python experiments/var_analyzer.py --analysis-type hyperparameters -f VAR_ruben/hyperparameter_tuning/parallel_TIMESTAMP/collected_results.csv
```

## Command-Line Arguments

### Main Tuning Script

```bash
python VAR_hyperparameter_tuning.py [OPTIONS]
```

**Options:**
- `-n, --NET`: Network number (default: 3)
- `-u, --UNDERSAMPLING`: Undersampling rate (default: 3)
- `--num_batches`: Number of batches to average over (default: 5)
- `-p, --PNUM`: Number of CPUs to use
- `--test_subset`: Test curated subset of ~20 combinations
- `--test_all`: Test all 3125 possible combinations
- `--priorities`: Specify custom priority configurations

### Analysis Script

```bash
python experiments/var_analyzer.py --analysis-type hyperparameters -f RESULTS.csv [OPTIONS]
```

**Options:**
- `-f, --file`: CSV file with results (required)
- `-o, --output-dir`: Output directory for plots (default: same as input)

## Understanding Results

### Priority Configurations

Format: `[p1, p2, p3, p4, p5]`

Example: `[5, 3, 1, 3, 2]`
- `p1=5`: Highest priority constraint
- `p2=3`: Medium-high priority
- `p3=1`: Lowest priority
- etc.

### Metrics

**Orientation:**
- Measures correctness of causal direction (A→B vs A←B)
- Most important for causal inference

**Adjacency:**
- Measures presence of edges (ignoring direction)
- Important for network structure

**Cycle:**
- Measures detection of 2-cycles (feedback loops)
- Important for identifying bidirectional relationships

**Combined F1:**
- Average of all three metrics
- Overall performance indicator

### Plots

**F1 Comparison:**
- Bar chart comparing all three metrics
- Shows trade-offs between metrics

**Combined F1:**
- Ranked configurations by overall performance
- Quick identification of best settings

**Priority Heatmap:**
- Shows which priority values work best at each position
- Reveals patterns (e.g., p1 should be high)

**Precision-Recall:**
- Shows trade-off between false positives and false negatives
- Colored by F1 score

## Best Practices

### 1. Start Small

Don't jump to full search immediately:

```bash
# Day 1: Test with default configs
./run_hyperparameter_tuning.sh

# Day 2: Try subset mode
./run_hyperparameter_tuning.sh --subset

# Day 3: If needed, run full search on cluster
```

### 2. Choose Appropriate Network

- **Network 3**: Good balanced starting point
- **Network 1**: Simplest, fastest
- **Network 6**: Most complex, slowest

### 3. Batch Averaging

- Use `--num_batches 5` minimum for stable results
- More batches = more reliable, but slower
- Default 5 is a good balance

### 4. Cluster Considerations

**Before submitting large jobs:**
- Test single job first: `--array=1`
- Verify data paths are correct
- Check output directory has space

**Resource limits:**
- Memory: ~4GB per job sufficient
- Time: 1-2 hours per job
- CPUs: 4 cores recommended

### 5. Result Interpretation

**High precision, low recall:**
- Too conservative
- Missing true edges
- Try increasing p1 (edge priority)

**Low precision, high recall:**
- Too liberal
- Too many false positives
- Try increasing p2 (penalty priority)

## Troubleshooting

### Issue: Script hangs or runs very slowly

**Solution:**
- Check `--PNUM` setting (use all available cores)
- Verify data files aren't corrupted
- Try smaller subset first

### Issue: Out of memory errors

**Solution:**
- Reduce batch size
- Run on machine with more RAM
- Use cluster execution

### Issue: Inconsistent results across runs

**Solution:**
- Increase `--num_batches` (more averaging)
- Check for random seed issues
- Verify data preprocessing is consistent

### Issue: All configurations perform poorly

**Solution:**
- Verify ground truth networks are correct
- Check data quality
- Try different undersampling rate (`-u`)

### Issue: Cluster jobs fail silently

**Solution:**
- Check SLURM output logs
- Verify paths in submission script
- Test single job with `--array=1` first

## Advanced Usage

### Custom Priority Search

Test specific configurations:

```bash
python VAR_hyperparameter_tuning.py \
    --priorities "[5,5,5,5,5]" "[5,3,1,3,2]" "[3,3,3,3,3]" \
    --num_batches 10
```

### Different Networks

Compare across networks:

```bash
for net in 1 2 3 4 5 6; do
    python VAR_hyperparameter_tuning.py -n $net --test_subset
done
```

### Analyzing Subsets

Analyze only orientation performance:

```bash
python experiments/var_analyzer.py --analysis-type orientation -f results.csv
```

## References

- Original RASL paper: [citation]
- Priority parameter theory: [citation]
- Experimental design: See `METHODS_COMPARISON_SUMMARY.md` in legacy docs
