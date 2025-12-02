# fMRI Experiment Quick Start Guide

## 🚀 Running the Experiment

### Basic Usage

```bash
cd /path/to/gunfolds/gunfolds/scripts
python fmri_experiment.py
```

This uses default settings:
- Selection mode: `top_k`
- Top K: 10 solutions
- Results saved to: `fbirn_results/MMDDYYYYHHMMSS/`

### Custom Parameters

**Select top 15 solutions:**
```bash
python fmri_experiment.py --selection_mode top_k --top_k 15
```

**Use delta threshold (1.9x min cost):**
```bash
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 1.9
```

**Use delta threshold (2.5x min cost):**
```bash
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 2.5
```

---

## 📁 Output Structure

Each run creates a **timestamped folder**:

```
fbirn_results/
└── 12262025103045/              # Format: MMDDYYYYHHMMSS
    ├── combined/                # All subjects combined
    │   ├── all_solutions_info.zkl
    │   ├── selection_params.zkl
    │   ├── group_graph_gt_combined.pdf
    │   ├── group_graph_weighted_combined.pdf
    │   └── group_edge_counts_combined.*
    │
    ├── group_0/                 # Group 0 subjects
    │   ├── subjects/
    │   │   ├── rasl_subj_0000_sol_01_grp_0.pdf
    │   │   ├── rasl_subj_0000_sol_02_grp_0.pdf
    │   │   └── ... (K solutions per subject)
    │   ├── solutions/
    │   │   └── solutions_info_0.zkl
    │   ├── group_graph_gt_0.pdf
    │   ├── group_graph_weighted_0.pdf
    │   └── group_edge_counts_0.*
    │
    └── group_1/                 # Group 1 subjects
        └── ... (same structure as group_0)
```

**Timestamp Format:** MMDDYYYYHHMMSS
- Example: `12262025103045` = December 26, 2025 at 10:30:45

---

## 🔍 Analyzing Results

### Option 1: Analyze Most Recent Run (Auto-detect)

```bash
python analyze_saved_solutions.py
```

Automatically finds and analyzes the most recent timestamped folder.

### Option 2: Analyze Specific Run

```bash
# Using timestamp
python analyze_saved_solutions.py -t 12262025103045

# Using full path
python analyze_saved_solutions.py -d fbirn_results/12262025103045/combined
```

### Option 3: Generate Plots

```bash
# Auto-detect most recent + plots
python analyze_saved_solutions.py --plot

# Specific timestamp + plots
python analyze_saved_solutions.py -t 12262025103045 --plot
```

### Option 4: Analyze Specific Group

```bash
# Group 0 from most recent run
python analyze_saved_solutions.py -g 0 --plot

# Group 1 from specific timestamp
python analyze_saved_solutions.py -t 12262025103045 -g 1 --plot
```

### Option 5: Compare with Ground Truth

```bash
python analyze_saved_solutions.py --compare_gt
```

---

## 📊 Analysis Outputs

The analysis script generates:

### Console Output
- Cost distribution statistics
- Per-subject summary
- Group comparisons
- Ground truth metrics (if requested)

### CSV Files
- `analysis_summary.csv` - Per-subject statistics
- `ground_truth_comparison.csv` - F1 scores vs GT (if requested)

### Plot Files (with `--plot`)
- `cost_distribution_all.pdf` / `.png`
- `cost_distribution_by_group.pdf` / `.png`
- `cost_boxplot_by_group.pdf` / `.png`

---

## 💡 Tips

### Finding Your Results

After running the experiment, note the timestamp printed:
```
Saving results to: fbirn_results/12262025103045
Timestamp: 12262025103045
```

Use this timestamp for analysis:
```bash
python analyze_saved_solutions.py -t 12262025103045 --plot
```

### Comparing Different Runs

Run with different parameters:
```bash
# Run 1: Top 10
python fmri_experiment.py --selection_mode top_k --top_k 10

# Run 2: Top 20
python fmri_experiment.py --selection_mode top_k --top_k 20

# Run 3: Delta 1.5x
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 1.5

# Run 4: Delta 2.0x
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 2.0
```

Each run creates a separate timestamped folder, preserving all results.

### Loading Results in Python

```python
from gunfolds.utils import zickle as zkl
import os

# Find most recent results
results_dir = "fbirn_results"
timestamps = sorted([d for d in os.listdir(results_dir) 
                    if d.isdigit() and len(d) == 14], reverse=True)
latest = timestamps[0]

# Load solutions
solutions = zkl.load(f'fbirn_results/{latest}/combined/all_solutions_info.zkl')
params = zkl.load(f'fbirn_results/{latest}/combined/selection_params.zkl')

print(f"Analyzing run from: {latest}")
print(f"Selection mode: {params['selection_mode']}")
print(f"Number of subjects: {len(solutions)}")
```

---

## 🆘 Common Issues

### Issue: "No results found in fbirn_results/"

**Solution:** Make sure you've run the experiment first:
```bash
python fmri_experiment.py
```

### Issue: "Solutions file not found"

**Solution:** Check the timestamp. List available runs:
```bash
ls fbirn_results/
```

### Issue: Want to analyze older run

**Solution:** Specify the timestamp explicitly:
```bash
python analyze_saved_solutions.py -t <TIMESTAMP> --plot
```

---

## 📚 More Information

- **Detailed usage guide:** `SOLUTION_SELECTION_GUIDE.md`
- **Technical changes:** `FMRI_SELECTION_CHANGES.md`
- **Delta tuning methodology:** `VAR_delta_tuning.py`

---

## Example Workflow

```bash
# 1. Run experiment with default settings
python fmri_experiment.py

# Note the timestamp from output, e.g., 12262025103045

# 2. Analyze results with plots
python analyze_saved_solutions.py -t 12262025103045 --plot

# 3. Compare with ground truth
python analyze_saved_solutions.py -t 12262025103045 --compare_gt

# 4. Analyze specific group
python analyze_saved_solutions.py -t 12262025103045 -g 0 --plot

# 5. Run again with different parameters
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 2.0

# 6. Compare results from both runs
python analyze_saved_solutions.py -t 12262025103045 --plot
python analyze_saved_solutions.py -t <NEW_TIMESTAMP> --plot
```

