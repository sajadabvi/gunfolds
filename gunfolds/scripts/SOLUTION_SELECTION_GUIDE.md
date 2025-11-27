# Solution Selection Guide for fMRI Experiment

## Overview

The `fmri_experiment.py` script has been updated to support **cost-based solution selection** instead of using ground truth network for selection. This enables unbiased analysis where the selection of candidate graphs does not depend on knowing the true network structure.

## Two Selection Modes

### 1. Top-K Selection (Default)

Select the top K solutions with the **lowest cost**.

**Usage:**
```bash
python fmri_experiment.py --selection_mode top_k --top_k 10
```

**Parameters:**
- `--selection_mode top_k`: Activates top-K selection mode
- `--top_k N`: Number of solutions to keep (default: 10)

**How it works:**
1. All solutions from RASL are sorted by their cost (ascending)
2. The top K solutions with lowest cost are selected
3. These solutions are saved for further analysis

---

### 2. Delta Threshold Selection

Select **all solutions** where `cost <= min_cost * delta_multiplier`.

This approach is inspired by the delta tuning methodology in `VAR_delta_tuning.py`.

**Usage:**
```bash
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 1.9
```

**Parameters:**
- `--selection_mode delta_threshold`: Activates delta threshold mode
- `--delta_multiplier X`: Multiplier for the minimum cost threshold (default: 1.9)

**How it works:**
1. All solutions from RASL are sorted by their cost
2. The minimum cost is identified: `min_cost`
3. Threshold is calculated: `threshold = min_cost * delta_multiplier`
4. All solutions with `cost <= threshold` are selected
5. These solutions are saved for further analysis

**Example with delta=1.9:**
- If min_cost = 1000
- Threshold = 1000 * 1.9 = 1900
- All solutions with cost <= 1900 are kept

---

## Output Files

The script now saves detailed solution information for further analysis:

### Per-Group Files
```
fbirn_results/
└── MMDDYYYYHHMMSS/              # Timestamped folder (e.g., 12262025103045)
    ├── group_0/
    │   ├── subjects/          # Individual subject PDFs (as before)
    │   ├── solutions/
    │   │   └── solutions_info_0.zkl   # Detailed solution data for group 0
    │   ├── group_edge_counts_0.npz    # Edge counts (as before)
    │   ├── group_edge_counts_0.csv    # Edge counts CSV (as before)
    │   └── group_graph_*.pdf          # Group graphs (as before)
    └── group_1/
        └── ... (same structure)
```

### Combined Files
```
fbirn_results/
└── MMDDYYYYHHMMSS/              # Timestamped folder
    └── combined/
        ├── all_solutions_info.zkl      # All solutions from all subjects
        ├── selection_params.zkl        # Parameters used for selection
        ├── group_edge_counts_combined.npz
        ├── group_edge_counts_combined.csv
        └── group_graph_*.pdf
```

---

## Solution Info Structure

Each `.zkl` file contains structured information:

```python
# Example structure in solutions_info_0.zkl
[
    {
        'subject_id': 0,
        'group': 0,
        'num_solutions': 10,
        'solutions': [
            {
                'solution_idx': 1,
                'cost': 1234.5,
                'undersampling': (2,),
                'graph': {...}  # CG dictionary
            },
            ...
        ]
    },
    ...
]

# selection_params.zkl structure
{
    'selection_mode': 'top_k' or 'delta_threshold',
    'top_k': 10,                    # if mode = 'top_k'
    'delta_multiplier': 1.9,        # if mode = 'delta_threshold'
    'timestamp': '2025-11-26 10:30:00'
}
```

---

## Loading and Analyzing Saved Solutions

Example script to load and analyze saved solutions:

```python
from gunfolds.utils import zickle as zkl
import numpy as np

# Load all solutions (replace timestamp with your run's timestamp)
timestamp = "12262025103045"  # Example: Dec 26, 2025, 10:30:45
all_solutions = zkl.load(f'fbirn_results/{timestamp}/combined/all_solutions_info.zkl')
selection_params = zkl.load(f'fbirn_results/{timestamp}/combined/selection_params.zkl')

print(f"Selection mode: {selection_params['selection_mode']}")

# Analyze cost distribution
all_costs = []
for subject_info in all_solutions:
    for sol in subject_info['solutions']:
        all_costs.append(sol['cost'])

print(f"Number of solutions: {len(all_costs)}")
print(f"Cost range: [{min(all_costs):.2f}, {max(all_costs):.2f}]")
print(f"Mean cost: {np.mean(all_costs):.2f}")
print(f"Median cost: {np.median(all_costs):.2f}")

# Group-specific analysis
for subject_info in all_solutions:
    subject_id = subject_info['subject_id']
    group = subject_info['group']
    num_sols = subject_info['num_solutions']
    costs = [s['cost'] for s in subject_info['solutions']]
    
    print(f"Subject {subject_id} (Group {group}): {num_sols} solutions, "
          f"costs: [{min(costs):.2f}, {max(costs):.2f}]")
```

---

## Example Runs

### Example 1: Select top 10 solutions by cost
```bash
python fmri_experiment.py --selection_mode top_k --top_k 10
```

### Example 2: Use delta threshold with multiplier 1.5
```bash
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 1.5
```

### Example 3: More conservative delta threshold (2.5x)
```bash
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 2.5
```

### Example 4: Top 20 solutions
```bash
python fmri_experiment.py --selection_mode top_k --top_k 20
```

---

## 📂 Output Directory Structure

Each run creates a **timestamped folder** in `fbirn_results/`:

```
fbirn_results/
├── 12262025103045/    # Run from Dec 26, 2025, 10:30:45
│   ├── combined/
│   ├── group_0/
│   └── group_1/
├── 12262025154520/    # Run from Dec 26, 2025, 15:45:20
│   ├── combined/
│   ├── group_0/
│   └── group_1/
└── ...
```

This allows you to:
- **Compare different runs** with different parameters
- **Preserve all results** without overwriting
- **Track when experiments were performed**

---

## 🔍 Analyzing Results

### Option 1: Auto-detect most recent results
```bash
# Automatically analyzes the most recent timestamped folder
python analyze_saved_solutions.py --plot
```

### Option 2: Specify timestamp
```bash
# Analyze specific run by timestamp
python analyze_saved_solutions.py -t 12262025103045 --plot
```

### Option 3: Specify full directory path
```bash
# Specify exact directory
python analyze_saved_solutions.py -d fbirn_results/12262025103045/combined --plot
```

### Option 4: Analyze specific group
```bash
# Analyze group 0 from most recent run
python analyze_saved_solutions.py -g 0 --plot

# Analyze group 1 from specific timestamp
python analyze_saved_solutions.py -t 12262025103045 -g 1 --plot
```

---

## Key Changes from Previous Version

### Before:
- Used ground truth network (`network_GT`) to compute F1 scores
- Selected top 75% of solutions based on F1 score
- **Problem:** Selection was biased by knowledge of true network

### After:
- Uses **cost** from RASL solver (no ground truth needed for selection)
- Two flexible selection modes: top-K or delta threshold
- Saves all solution details (cost, undersampling, graphs) for post-hoc analysis
- Ground truth is still used for:
  - Computing GT_density (RASL parameter)
  - SCC members (if `--SCCMEMBERS f`)
  - Final evaluation metrics (if using `run_analysis`)

---

## Recommended Workflow

1. **Run experiment with both modes:**
   ```bash
   # Top-K approach
   python fmri_experiment.py --selection_mode top_k --top_k 10
   
   # Delta threshold approach
   python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 1.9
   ```

2. **Compare results:**
   - Load the saved solutions from both runs
   - Compare the cost distributions
   - Analyze the diversity of selected solutions
   - Evaluate against ground truth (if available) for validation

3. **Fine-tune parameters:**
   - If top-K: adjust K based on computational budget
   - If delta: adjust multiplier based on cost distribution analysis

---

## Notes

- The ground truth network is **no longer used for solution selection**
- Selection is now based purely on the cost metric from RASL
- All selected solutions are saved with their costs for transparency
- You can post-process the saved solutions with any custom analysis
- The delta_multiplier value of 1.9 is based on findings from VAR_delta_tuning.py

---

## Questions?

For more details on the delta tuning approach, see:
- `VAR_delta_tuning.py` - Delta hyperparameter tuning script
- `DELTA_TUNING_README.md` - Comprehensive delta tuning documentation

