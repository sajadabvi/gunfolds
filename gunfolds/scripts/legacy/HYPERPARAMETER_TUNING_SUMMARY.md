# Hyperparameter Tuning Suite - Complete Summary

## What You Have

I've created a complete hyperparameter tuning framework for optimizing RASL priority parameters. Here's what was created:

### Main Scripts

1. **VAR_hyperparameter_tuning.py** - Core hyperparameter search script
   - Tests different priority combinations [p1, p2, p3, p4, p5]
   - Each priority can be 1-5 (5 = highest priority)
   - Runs multiple batches (1-5) for statistical averaging
   - Saves results in CSV and ZKL formats

2. **VAR_analyze_hyperparameters.py** - Analysis and visualization script
   - Generates 4 plots: F1 comparison, combined F1, priority heatmap, precision-recall
   - Creates summary statistics table
   - Analyzes priority patterns

3. **VAR_generate_report.py** - Report generation script
   - Creates comprehensive Markdown report
   - Includes top configurations, statistics, and recommendations
   - Full results table in appendix

4. **test_hyperparameter_tuning.py** - Quick test script
   - Verifies setup and dependencies
   - Runs quick test with 2 priorities × 2 batches
   - Validates data files exist

5. **run_hyperparameter_tuning.sh** - Convenient wrapper script
   - Easy command-line interface
   - Automatically runs analysis and reporting after tuning
   - Supports test, default, subset, and full modes

### Documentation

6. **VAR_HYPERPARAMETER_TUNING_README.md** - Complete usage guide
7. **HYPERPARAMETER_TUNING_SUMMARY.md** - This file

---

## Quick Start

### Step 1: Test Your Setup
```bash
cd /Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts
python test_hyperparameter_tuning.py
```

This will verify:
- All dependencies are installed
- Data files exist
- Basic functionality works

### Step 2: Run Hyperparameter Tuning

**Option A: Using the Shell Script (Recommended)**
```bash
# Quick test (2 configs × 2 batches)
./run_hyperparameter_tuning.sh --test

# Default mode (5 representative configs × 5 batches)
./run_hyperparameter_tuning.sh

# Subset mode (~20 curated configs × 5 batches)
./run_hyperparameter_tuning.sh --subset

# Custom settings
./run_hyperparameter_tuning.sh --subset --net 3 --undersampling 3 --batches 5 --pnum 8
```

**Option B: Direct Python Execution**
```bash
# Default mode
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5

# Subset mode
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5 --test_subset

# All combinations (WARNING: 3125 configs - very slow!)
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5 --test_all
```

### Step 3: View Results

Results are automatically analyzed if you use the shell script. Otherwise:

```bash
# Analyze results (creates plots)
python VAR_analyze_hyperparameters.py -f VAR_ruben/hyperparameter_tuning/priority_tuning_net3_u3_TIMESTAMP.csv

# Generate report (creates markdown document)
python VAR_generate_report.py -f VAR_ruben/hyperparameter_tuning/priority_tuning_net3_u3_TIMESTAMP.csv
```

---

## Understanding Priority Parameters

The RASL algorithm uses 5 priorities to weight different optimization constraints:

```python
priorities = [p1, p2, p3, p4, p5]
```

- **Values**: Each can be 1-5 (5 = highest priority, 1 = lowest priority)
- **Equal values**: Constraints are weighted equally
- **Original**: `[4, 2, 5, 3, 1]` (from VAR_for_ruben_nets.py line 152)

### Example Configurations

```python
[4, 2, 5, 3, 1]  # Original - mixed priorities
[1, 1, 1, 1, 1]  # All equal - balanced approach
[5, 4, 3, 2, 1]  # Descending - priority hierarchy
[5, 5, 5, 5, 5]  # All high - maximize all constraints
[5, 1, 1, 1, 1]  # Focus on first constraint only
```

---

## Test Modes

### 1. Test Mode (Recommended First)
```bash
./run_hyperparameter_tuning.sh --test
```
- Tests: 2 configurations
- Batches: 2
- Time: ~2-5 minutes
- Purpose: Verify everything works

### 2. Default Mode
```bash
./run_hyperparameter_tuning.sh
```
- Tests: 5 representative configurations
- Batches: 5
- Time: ~10-30 minutes
- Purpose: Quick exploration

### 3. Subset Mode
```bash
./run_hyperparameter_tuning.sh --subset
```
- Tests: ~20 curated configurations
- Batches: 5
- Time: ~1-2 hours
- Purpose: Thorough search of promising configurations

### 4. Full Mode (Use with Caution!)
```bash
./run_hyperparameter_tuning.sh --all
```
- Tests: 3,125 configurations (5^5)
- Batches: 5
- Time: ~several days
- Purpose: Exhaustive search (only if you have time and resources)

---

## Output Files

After running hyperparameter tuning, you'll get:

### In `VAR_ruben/hyperparameter_tuning/`:

1. **priority_tuning_net3_u3_TIMESTAMP.csv**
   - Main results table
   - Sortable by any metric
   - Easy to open in Excel/Numbers

2. **priority_tuning_net3_u3_TIMESTAMP.zkl**
   - Full results with metadata
   - Python-readable format

3. **priority_tuning_net3_u3_TIMESTAMP_report.md**
   - Comprehensive markdown report
   - Includes executive summary, statistics, recommendations

4. **f1_comparison.png**
   - Bar chart comparing F1 scores across metrics

5. **combined_f1.png**
   - Combined F1 scores for top configurations

6. **priority_heatmap.png**
   - Shows how each priority position/value affects performance

7. **precision_recall.png**
   - Precision-recall scatter plots for all three metrics

8. **summary_statistics.csv**
   - Statistical summary of all metrics

---

## Interpreting Results

### Key Metrics

1. **Combined F1** (primary metric)
   - Average of orientation_F1, adjacency_F1, and cycle_F1
   - Higher is better (0.0 to 1.0)
   - Use this to rank overall performance

2. **Orientation F1**
   - How well edge directions are recovered
   - Important for causal inference

3. **Adjacency F1**
   - How well edge presence (ignoring direction) is recovered
   - Important for connectivity structure

4. **Cycle F1**
   - How well 2-cycles (bidirectional edges) are recovered
   - Important for feedback loops

### Reading the CSV

```csv
priorities,p1,p2,p3,p4,p5,combined_F1,orientation_F1,adjacency_F1,cycle_F1,...
"(4, 2, 5, 3, 1)",4,2,5,3,1,0.8523,0.8912,0.8456,0.8201,...
```

- **priorities**: The tested configuration
- **p1-p5**: Individual priority values (for easy filtering)
- **combined_F1**: Overall score (use for ranking)
- Remaining columns: Detailed metrics

### Example Analysis

"Which priority configuration gives best orientation F1?"
```python
import pandas as pd
df = pd.read_csv('results.csv')
best_orientation = df.loc[df['orientation_F1'].idxmax()]
print(best_orientation['priorities'])
```

---

## Workflow Example

Complete workflow from start to finish:

```bash
# 1. Navigate to scripts directory
cd /Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts

# 2. Test setup
python test_hyperparameter_tuning.py
# Expected: ✓ ALL TESTS PASSED

# 3. Run hyperparameter tuning (subset mode)
./run_hyperparameter_tuning.sh --subset --net 3 --undersampling 3 --batches 5
# This will automatically:
#   - Run tuning
#   - Generate plots
#   - Create report

# 4. View results
open VAR_ruben/hyperparameter_tuning/priority_tuning_net3_u3_*_report.md
open VAR_ruben/hyperparameter_tuning/*.png

# 5. Use best configuration in your main analysis
# Copy the best priorities from the report into VAR_for_ruben_nets.py line 152
```

---

## Customization

### Test Different Networks
```bash
./run_hyperparameter_tuning.sh --subset --net 1 --undersampling 2
./run_hyperparameter_tuning.sh --subset --net 4 --undersampling 3
```

### Use More/Fewer CPUs
```bash
./run_hyperparameter_tuning.sh --subset --pnum 16  # Use 16 CPUs
./run_hyperparameter_tuning.sh --subset --pnum 4   # Use 4 CPUs
```

### Test Custom Priority Sets

Edit `VAR_hyperparameter_tuning.py` and add to `get_priority_combinations()`:

```python
if test_subset:
    priority_sets = [
        [4, 2, 5, 3, 1],  # Original
        [5, 5, 5, 5, 5],  # Your custom config
        [2, 3, 4, 2, 1],  # Another custom config
        # ... add more
    ]
```

---

## Troubleshooting

### "Data file not found"
- Ensure data exists at: `~/DataSets_Feedbacks/8_VAR_simulation/net{NET}/u{UNDERSAMPLING}/txtSTD/`
- Check that you have files `data1.txt` through `data{num_batches}.txt`

### "Import error"
- Install missing packages: `pip install tqdm pandas numpy matplotlib seaborn tigramite`
- Ensure gunfolds is properly installed

### "No results"
- Check that PCMCI is producing valid estimates
- Try reducing `--num_batches` to 2 or 3 for faster testing
- Verify data quality

### Script takes too long
- Start with `--test` mode
- Use `--subset` instead of `--all`
- Reduce `--num_batches` to 2 or 3
- Increase `--pnum` to use more CPUs

---

## Files Created

All files are in: `/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/`

```
VAR_hyperparameter_tuning.py              # Main tuning script
VAR_analyze_hyperparameters.py            # Analysis script
VAR_generate_report.py                    # Report generator
test_hyperparameter_tuning.py             # Test script
run_hyperparameter_tuning.sh              # Wrapper script
VAR_HYPERPARAMETER_TUNING_README.md       # Detailed README
HYPERPARAMETER_TUNING_SUMMARY.md          # This file
```

---

## Next Steps

1. **✓ Run the test**: `./run_hyperparameter_tuning.sh --test`
2. **✓ Try subset mode**: `./run_hyperparameter_tuning.sh --subset`
3. **✓ Review results**: Check the generated report and plots
4. **✓ Update main script**: Use best priorities in `VAR_for_ruben_nets.py`
5. **Optional**: Run full search with `--all` if you have computational resources

---

## Questions?

- See detailed usage: `VAR_HYPERPARAMETER_TUNING_README.md`
- Run help: `./run_hyperparameter_tuning.sh --help`
- Check test output: `python test_hyperparameter_tuning.py`

**Good luck with your hyperparameter tuning!** 🚀

