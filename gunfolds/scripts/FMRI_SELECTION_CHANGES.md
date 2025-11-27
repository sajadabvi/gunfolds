# Changes to fMRI Experiment Solution Selection

## Summary

Modified `fmri_experiment.py` to use **cost-based solution selection** instead of ground-truth F1 scoring. This enables unbiased graph selection that doesn't rely on knowing the true network structure.

## What Changed

### 1. Modified `select_top_solutions()` Function

**Before:**
```python
def select_top_solutions(r_estimated, network_GT, include_selfloop):
    # Computed F1 scores against ground truth
    # Kept top 75% by F1
```

**After:**
```python
def select_top_solutions(r_estimated, n_nodes, selection_mode='top_k', k=10, delta_multiplier=1.9):
    # Two modes:
    # 1. 'top_k': Select top k by lowest cost
    # 2. 'delta_threshold': Select all where cost <= min_cost * delta_multiplier
```

**Key changes:**
- No longer requires `network_GT` for selection
- No longer computes F1 scores for selection
- Selection based purely on cost from RASL
- Returns list of (cost, graph, undersampling) tuples

---

### 2. Updated `RASL_subject()` Function

**Before:**
```python
def RASL_subject(ts_2d, args, network_GT, include_selfloop):
    # ... runs RASL ...
    kept = select_top_solutions(r_estimated, network_GT, include_selfloop)
    return [res for _, res in kept]
```

**After:**
```python
def RASL_subject(ts_2d, args, network_GT, include_selfloop, 
                 selection_mode='top_k', top_k=10, delta_multiplier=1.9):
    # ... runs RASL ...
    kept = select_top_solutions(r_estimated, n_nodes, selection_mode, top_k, delta_multiplier)
    res_cgs = [res for _, res, _ in kept]
    return res_cgs, kept  # Returns both graphs and full info
```

**Key changes:**
- Added selection mode parameters
- Returns both graphs and full solution info (with costs)
- Selection independent of ground truth

---

### 3. Enhanced `run_all_subjects()` Function

**Before:**
```python
def run_all_subjects(args, network_GT, include_selfloop):
    # ... processed subjects ...
    res_list = RASL_subject(ts_2d, args, network_GT, include_selfloop)
    # Saved only plots
```

**After:**
```python
def run_all_subjects(args, network_GT, include_selfloop,
                     selection_mode='top_k', top_k=10, delta_multiplier=1.9):
    # ... processed subjects ...
    res_list, solution_info = RASL_subject(ts_2d, args, network_GT, include_selfloop,
                                            selection_mode, top_k, delta_multiplier)
    # Saves plots AND detailed solution info
```

**Key additions:**
- Saves all solution info to `.zkl` files:
  - Per-group: `group_X/solutions/solutions_info_X.zkl`
  - Combined: `combined/all_solutions_info.zkl`
- Saves selection parameters: `combined/selection_params.zkl`
- Solution info includes: subject_id, group, costs, undersampling, graphs

---

### 4. Updated `RASL()` and `run_analysis()` Functions

Both updated to accept and pass through selection parameters.

---

### 5. Added Command-Line Arguments

```python
parser.add_argument("--selection_mode", default="top_k", 
                    choices=['top_k', 'delta_threshold'])
parser.add_argument("--top_k", default=10, type=int)
parser.add_argument("--delta_multiplier", default=1.9, type=float)
```

---

## Usage Examples

### Top-K Selection (Default)
```bash
python fmri_experiment.py --selection_mode top_k --top_k 10
```
Selects the 10 solutions with lowest cost.

### Delta Threshold Selection
```bash
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 1.9
```
Selects all solutions where `cost <= 1.9 * min_cost`.

---

## Output Structure

### New Timestamped Directory Structure

Each run creates a timestamped folder to preserve results:

```
fbirn_results/
└── MMDDYYYYHHMMSS/                     # NEW: Timestamped folder (e.g., 12262025103045)
    ├── combined/
    │   ├── all_solutions_info.zkl          # NEW: All solution details
    │   ├── selection_params.zkl            # NEW: Selection parameters used
    │   └── ... (existing files)
    └── group_X/
        ├── solutions/
        │   └── solutions_info_X.zkl        # NEW: Group X solution details
        └── ... (existing files)
```

**Benefits:**
- Compare different runs without overwriting
- Track when experiments were performed
- Preserve all parameter variations

### Solution Info Format

```python
# In solutions_info_X.zkl:
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
                'graph': {...}
            },
            ...
        ]
    },
    ...
]
```

---

## Analysis Tools

### New Script: `analyze_saved_solutions.py`

Analyzes saved solution data:

```bash
# Basic analysis (auto-detects most recent run)
python analyze_saved_solutions.py

# Analyze specific timestamp
python analyze_saved_solutions.py -t 12262025103045 --plot

# Analyze specific directory
python analyze_saved_solutions.py -d fbirn_results/12262025103045/combined --plot

# Compare against ground truth
python analyze_saved_solutions.py --compare_gt

# Analyze specific group
python analyze_saved_solutions.py -g 0 --plot
```

**Features:**
- Cost distribution statistics
- Per-subject summary
- Group comparisons
- Plots (histograms, boxplots)
- Ground truth comparison (optional)

---

## Key Benefits

1. **Unbiased Selection**: No longer uses ground truth for solution selection
2. **Flexible Methods**: Two selection approaches (top-K and delta threshold)
3. **Transparent**: All solution costs saved for post-hoc analysis
4. **Reproducible**: Selection parameters saved with results
5. **Backward Compatible**: Can still use ground truth for final evaluation

---

## Migration Notes

### If You Were Using the Old Version:

**Old code:**
```python
kept = select_top_solutions(r_estimated, network_GT, include_selfloop)
res_list = [res for _, res in kept]
```

**New code (top-10):**
```python
kept = select_top_solutions(r_estimated, n_nodes, selection_mode='top_k', k=10)
res_list = [res for _, res, _ in kept]
```

**New code (delta threshold):**
```python
kept = select_top_solutions(r_estimated, n_nodes, selection_mode='delta_threshold', delta_multiplier=1.9)
res_list = [res for _, res, _ in kept]
```

---

## Validation

To validate that the new selection produces reasonable results:

1. Run with both selection modes
2. Use `analyze_saved_solutions.py --compare_gt` to check F1 scores
3. Compare cost distributions between groups
4. Verify that selected solutions are diverse and reasonable

---

## References

- **Inspiration**: `VAR_delta_tuning.py` (delta threshold methodology)
- **Documentation**: `SOLUTION_SELECTION_GUIDE.md` (detailed usage guide)
- **Analysis**: `analyze_saved_solutions.py` (post-hoc analysis tool)

---

## Questions?

See `SOLUTION_SELECTION_GUIDE.md` for detailed usage examples and best practices.

