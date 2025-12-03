# GCM Timestamped Results - Summary of Changes

## Overview

Modified `gcm_on_ICA.py` and created `analyze_gcm_results.py` to save Granger Causality Mapping (GCM) results in **timestamped folders**, allowing multiple runs to be preserved without overwriting.

---

## Changes Made

### 1. `gcm_on_ICA.py` - Main Script

**Modified:**
- Added `datetime` import for timestamp generation
- Changed `out_base` to be dynamically generated with timestamp
- Added command-line arguments for timestamp control
- Added run parameters saving for reproducibility

**Before:**
```python
out_base = "./gcm_roebroeck"
```

**After:**
```python
out_base_root = "./gcm_roebroeck"
if args.no_timestamp:
    out_base = out_base_root
else:
    timestamp = datetime.now().strftime('%m%d%Y%H%M%S')
    out_base = os.path.join(out_base_root, timestamp)
```

**New Arguments:**
- `-t / --timestamp`: Use existing timestamp directory
- `--no-timestamp`: Disable timestamped directories (legacy mode)
- Changed `-S / --subject` default from 1 to None (runs all subjects by default)

**New Feature:**
- Saves `run_params.csv` with all analysis parameters for reproducibility

---

### 2. `analyze_gcm_results.py` - New Analysis Tool

**Created new script** with features:
- Auto-detects most recent timestamped folder
- Loads and analyzes group-level GCM results
- Generates comprehensive statistics
- Creates multiple visualization plots
- Supports threshold-based edge filtering

**Key Features:**
```bash
# Auto-detect most recent
python analyze_gcm_results.py

# Specify timestamp
python analyze_gcm_results.py -t 12262025103045

# Generate plots with threshold
python analyze_gcm_results.py --plot --threshold 0.3
```

**Outputs:**
- Console statistics (edge frequencies, node degrees, top edges)
- `analysis_summary.txt` - Text summary
- Multiple plots (histogram, heatmaps, threshold comparison)

---

### 3. Documentation Created

**New Files:**
- `GCM_QUICK_START.md` - Quick reference guide
- `GCM_TIMESTAMPED_SUMMARY.md` - This file (comprehensive summary)

---

## Directory Structure

### Old Structure (before changes)
```
gcm_roebroeck/
├── csv/
│   └── ... (per-subject and group CSVs)
└── figures/
    └── ... (per-subject and group plots)
```
❌ **Problem:** Each run overwrites previous results

### New Structure (after changes)
```
gcm_roebroeck/
├── 12262025103045/      # Run 1: Alpha = 0.05
│   ├── csv/
│   │   ├── subj0_*_Adj_labeled.csv
│   │   ├── subj1_*_Adj_labeled.csv
│   │   ├── ... (all subjects)
│   │   ├── group_edge_hits.csv
│   │   ├── group_edge_rate.csv
│   │   └── group_Fdiff_mean.csv
│   ├── figures/
│   │   ├── subj0_*_gcm.png
│   │   ├── subj1_*_gcm.png
│   │   ├── ... (all subjects)
│   │   └── group_edge_frequency.png
│   ├── analysis_plots/      # Created by analyze_gcm_results.py
│   │   ├── edge_frequency_histogram.pdf
│   │   ├── edge_frequency_heatmap.pdf
│   │   ├── fdiff_heatmap.pdf
│   │   └── threshold_comparison.pdf
│   ├── run_params.csv       # Run parameters
│   └── analysis_summary.txt # Analysis summary
│
├── 12262025110530/      # Run 2: Alpha = 0.01
│   └── ... (same structure)
│
└── 12272025091030/      # Run 3: Different day
    └── ... (same structure)
```
✅ **Benefits:**
- All runs preserved
- Easy comparison between alpha values
- Track when experiments were performed
- No accidental overwrites

---

## Timestamp Format

**Format:** `MMDDYYYYHHMMSS`
- MM: Month (01-12)
- DD: Day (01-31)
- YYYY: Year (4 digits)
- HH: Hour (00-23)
- MM: Minute (00-59)
- SS: Second (00-59)

**Examples:**
- `12262025103045` = December 26, 2025 at 10:30:45 AM
- `01012026000000` = January 1, 2026 at midnight
- `06152025143000` = June 15, 2025 at 2:30:00 PM

---

## Usage Examples

### Running GCM Analysis

```bash
# Run all subjects (default alpha = 0.05)
python gcm_on_ICA.py
# Output: gcm_roebroeck/12262025103045/

# Run with stricter alpha (0.01)
python gcm_on_ICA.py -A 10
# Output: gcm_roebroeck/12262025110530/

# Run single subject
python gcm_on_ICA.py -S 5
# Output: gcm_roebroeck/12262025112145/

# Add subject to existing run
python gcm_on_ICA.py -S 10 -t 12262025103045
# Output: Added to gcm_roebroeck/12262025103045/

# Disable timestamping (old behavior)
python gcm_on_ICA.py --no-timestamp
# Output: gcm_roebroeck/
```

### Analyzing Results

```bash
# Analyze most recent run (auto-detect)
python analyze_gcm_results.py

# Analyze specific run with plots
python analyze_gcm_results.py -t 12262025103045 --plot

# Apply threshold to focus on consistent edges
python analyze_gcm_results.py -t 12262025103045 --threshold 0.5 --plot

# Specify full directory path
python analyze_gcm_results.py -d gcm_roebroeck/12262025103045 --plot
```

---

## Benefits

### 1. **Preservation**
- All runs automatically preserved
- No manual renaming needed
- No risk of overwriting important results

### 2. **Comparison**
- Easily compare different alpha values
- Test robustness across parameter settings
- Identify consistent vs. parameter-dependent edges

### 3. **Tracking**
- Timestamp shows exactly when analysis was run
- `run_params.csv` stores all analysis parameters
- Full reproducibility

### 4. **Organization**
- Chronological ordering
- Clear separation of different experiments
- Easy to find specific runs

---

## Analysis Tool Features

### Statistical Outputs

1. **Run Parameters** - All settings used for the analysis
2. **Edge Frequency Statistics:**
   - Total possible edges
   - Edges above threshold
   - Mean, median, max frequencies
3. **Top Edges** - 10 most frequent edges with hit counts
4. **Per-Node Statistics** - In/out degree for each brain region
5. **F-statistic Summary** - Mean, max, min Fdiff values

### Visualization Plots

1. **Edge Frequency Histogram:**
   - Distribution of edge frequencies across subjects
   - Shows consistency of detected edges

2. **Edge Frequency Heatmap:**
   - Matrix view of all edge frequencies
   - Color-coded by frequency
   - Threshold-aware display

3. **F-statistic Heatmap:**
   - Mean F-statistic differences
   - Shows strength of causal relationships

4. **Threshold Comparison:**
   - Network density vs. threshold
   - Number of edges vs. threshold
   - Helps choose appropriate threshold

---

## Migration Notes

### For Existing Scripts

If you have scripts that load from `gcm_roebroeck/csv/`:

**Old code:**
```python
edge_rate = pd.read_csv('gcm_roebroeck/csv/group_edge_rate.csv', index_col=0)
```

**New code (specify timestamp):**
```python
timestamp = "12262025103045"
edge_rate = pd.read_csv(f'gcm_roebroeck/{timestamp}/csv/group_edge_rate.csv', index_col=0)
```

**New code (auto-detect latest):**
```python
import os
results_dir = "gcm_roebroeck"
timestamps = sorted([d for d in os.listdir(results_dir) 
                    if d.isdigit() and len(d) == 14], reverse=True)
latest = timestamps[0]
edge_rate = pd.read_csv(f'gcm_roebroeck/{latest}/csv/group_edge_rate.csv', index_col=0)
```

### For Existing Results

Old results in `gcm_roebroeck/` won't interfere with new timestamped runs. You can:
- Keep them as-is
- Manually move them into a timestamped folder
- Delete them if no longer needed

---

## Comparison with fMRI Experiment

Both scripts now use the same timestamping approach:

| Feature | fMRI Experiment | GCM Analysis |
|---------|----------------|--------------|
| Base directory | `fbirn_results/` | `gcm_roebroeck/` |
| Timestamp format | MMDDYYYYHHMMSS | MMDDYYYYHHMMSS |
| Auto-detection | ✓ | ✓ |
| Analysis tool | `analyze_saved_solutions.py` | `analyze_gcm_results.py` |
| Parameter saving | `selection_params.zkl` | `run_params.csv` |
| Legacy mode | N/A | `--no-timestamp` flag |

---

## Implementation Details

### Timestamp Generation
```python
from datetime import datetime
timestamp = datetime.now().strftime('%m%d%Y%H%M%S')
out_base = os.path.join(out_base_root, timestamp)
```

### Auto-detection Logic
```python
def find_latest_results():
    """Find the most recent timestamped results folder"""
    base_dir = "gcm_roebroeck"
    subdirs = []
    for item in os.listdir(base_dir):
        if os.path.isdir(os.path.join(base_dir, item)) and \
           item.isdigit() and len(item) == 14:
            subdirs.append(item)
    subdirs.sort(reverse=True)
    return os.path.join(base_dir, subdirs[0]) if subdirs else None
```

### Parameters Saved
```python
params_df = pd.DataFrame({
    'parameter': ['n_subjects', 'components', 'tr_sec', 'n_cycles', 
                  'remove_linear', 'alpha', 'pmax', 'n_boot', 
                  'surr_mode', 'block_len', 'seed'],
    'value': [S, ','.join(map(str, comp_idx)), tr_sec, n_cycles, 
              remove_linear, alpha, pmax, n_boot, surr_mode, 
              block_len, seed]
})
params_df.to_csv(os.path.join(out_base, "run_params.csv"), index=False)
```

---

## Validation

### Check Timestamp Creation
```bash
# Run analysis
python gcm_on_ICA.py

# Should print:
# Saving results to: gcm_roebroeck/12262025103045
# Timestamp: 12262025103045
```

### Verify Directory Structure
```bash
ls gcm_roebroeck/
# Should show: 12262025103045/

ls gcm_roebroeck/12262025103045/
# Should show: csv/ figures/ run_params.csv
```

### Test Auto-detection
```bash
python analyze_gcm_results.py
# Should print:
# Auto-detected most recent results: gcm_roebroeck/12262025103045
```

---

## FAQ

**Q: Can I still use the old non-timestamped mode?**
A: Yes! Use the `--no-timestamp` flag:
```bash
python gcm_on_ICA.py --no-timestamp
```

**Q: What happens if I run the script at the exact same second?**
A: Extremely unlikely. If it happens, results would add to the same folder. Consider the aggregation feature if needed.

**Q: Can I add subjects to an existing timestamped run?**
A: Yes! Use the `-t` flag with the timestamp:
```bash
python gcm_on_ICA.py -S 10 -t 12262025103045
```

**Q: How do I compare results across different alpha values?**
A: Run multiple times with different `-A` values, then analyze each:
```bash
python gcm_on_ICA.py -A 50  # alpha = 0.05
python gcm_on_ICA.py -A 10  # alpha = 0.01
python gcm_on_ICA.py -A 5   # alpha = 0.005

python analyze_gcm_results.py -t <TIMESTAMP1> --plot
python analyze_gcm_results.py -t <TIMESTAMP2> --plot
python analyze_gcm_results.py -t <TIMESTAMP3> --plot
```

**Q: Can I rename the timestamped folders?**
A: Yes, but then use `-d` with full path instead of `-t` with timestamp.

**Q: How do I delete old results?**
A: Simply delete the timestamped folders:
```bash
rm -rf gcm_roebroeck/12262025103045/
```

---

## Related Files

- `gcm_on_ICA.py` - Main GCM analysis script (modified)
- `analyze_gcm_results.py` - Analysis tool (new)
- `GCM_QUICK_START.md` - Quick start guide (new)
- `roebroeck_gcm.py` - Core GCM implementation (unchanged)

---

## Example Workflow

```bash
# 1. Run GCM with default alpha (0.05)
python gcm_on_ICA.py
# Note timestamp: 12262025103045

# 2. Run with stricter alpha (0.01)
python gcm_on_ICA.py -A 10
# Note timestamp: 12262025110530

# 3. Analyze first run
python analyze_gcm_results.py -t 12262025103045 --plot

# 4. Analyze second run
python analyze_gcm_results.py -t 12262025110530 --plot

# 5. Compare edge frequencies
python
>>> import pandas as pd
>>> rate1 = pd.read_csv('gcm_roebroeck/12262025103045/csv/group_edge_rate.csv', index_col=0)
>>> rate2 = pd.read_csv('gcm_roebroeck/12262025110530/csv/group_edge_rate.csv', index_col=0)
>>> diff = rate1 - rate2
>>> print("Edges more frequent with alpha=0.05:")
>>> print(diff[diff > 0].stack().sort_values(ascending=False))
```

---

## Conclusion

The timestamped approach provides:
- ✓ Automatic preservation of all results
- ✓ Easy comparison across parameters
- ✓ Full reproducibility
- ✓ No risk of overwriting
- ✓ Comprehensive analysis tools
- ✓ Backward compatibility (with `--no-timestamp`)

All changes are production-ready and fully documented! 🎉

