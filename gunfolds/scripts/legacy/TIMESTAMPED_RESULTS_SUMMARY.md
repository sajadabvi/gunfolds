# Timestamped Results - Summary of Changes

## Overview

Modified `fmri_experiment.py` and `analyze_saved_solutions.py` to save results in **timestamped folders**, allowing multiple runs to be preserved without overwriting.

---

## Changes Made

### 1. `fmri_experiment.py`

**Modified:** `run_all_subjects()` function

**Before:**
```python
root_dir = "fbirn_results"
```

**After:**
```python
timestamp = datetime.now().strftime('%m%d%Y%H%M%S')
root_dir = os.path.join("fbirn_results", timestamp)
```

**Result:**
- Each run creates a unique folder: `fbirn_results/MMDDYYYYHHMMSS/`
- Prints timestamp to console for reference
- All subdirectories (combined/, group_0/, etc.) created inside timestamped folder

---

### 2. `analyze_saved_solutions.py`

**Added:**
- New argument: `-t/--timestamp` to specify which run to analyze
- `find_latest_results()` function to auto-detect most recent run
- Auto-detection logic in `main()` function

**New Usage Options:**
```bash
# Auto-detect most recent
python analyze_saved_solutions.py

# Specify timestamp
python analyze_saved_solutions.py -t 12262025103045

# Specify full path (still works)
python analyze_saved_solutions.py -d fbirn_results/12262025103045/combined
```

**Key Features:**
- **Auto-detection:** If no arguments provided, analyzes most recent timestamped folder
- **Backward compatible:** Old `-d` flag still works with full paths
- **Flexible:** Can specify timestamp or full path

---

### 3. Documentation Updates

**Created:**
- `FMRI_QUICK_START.md` - Quick reference guide

**Updated:**
- `SOLUTION_SELECTION_GUIDE.md` - Added timestamped directory info
- `FMRI_SELECTION_CHANGES.md` - Updated output structure section

---

## Directory Structure

### Old Structure (before changes)
```
fbirn_results/
├── combined/
├── group_0/
└── group_1/
```
❌ **Problem:** Each run overwrites previous results

### New Structure (after changes)
```
fbirn_results/
├── 12262025103045/      # Run 1: Dec 26, 2025, 10:30:45
│   ├── combined/
│   ├── group_0/
│   └── group_1/
├── 12262025154520/      # Run 2: Dec 26, 2025, 15:45:20
│   ├── combined/
│   ├── group_0/
│   └── group_1/
└── 12272025091030/      # Run 3: Dec 27, 2025, 09:10:30
    ├── combined/
    ├── group_0/
    └── group_1/
```
✅ **Benefits:**
- Preserve all runs
- Compare different parameter settings
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
- `01012026235959` = January 1, 2026 at 11:59:59 PM
- `07042025120000` = July 4, 2025 at 12:00:00 PM

---

## Usage Examples

### Running Experiments

```bash
# Run 1: Top 10 solutions
python fmri_experiment.py --selection_mode top_k --top_k 10
# Output: fbirn_results/12262025103045/

# Run 2: Top 20 solutions
python fmri_experiment.py --selection_mode top_k --top_k 20
# Output: fbirn_results/12262025110532/

# Run 3: Delta threshold 1.9x
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 1.9
# Output: fbirn_results/12262025112145/

# Run 4: Delta threshold 2.5x
python fmri_experiment.py --selection_mode delta_threshold --delta_multiplier 2.5
# Output: fbirn_results/12262025114820/
```

### Analyzing Results

```bash
# Analyze most recent run (auto-detect)
python analyze_saved_solutions.py --plot

# Analyze specific run
python analyze_saved_solutions.py -t 12262025103045 --plot

# Analyze specific group from specific run
python analyze_saved_solutions.py -t 12262025110532 -g 0 --plot

# Compare with ground truth
python analyze_saved_solutions.py -t 12262025112145 --compare_gt
```

---

## Benefits

### 1. **Preservation**
- All runs are preserved automatically
- No need to manually rename folders
- No risk of accidentally overwriting important results

### 2. **Comparison**
- Easily compare different parameter settings
- Run multiple experiments in sequence
- Analyze trade-offs between approaches

### 3. **Tracking**
- Timestamp shows exactly when experiment was run
- Can correlate with lab notebooks or other records
- Easy to identify latest results

### 4. **Organization**
- Clear chronological ordering
- All timestamped folders sort naturally
- Easy to find specific runs

---

## Migration Notes

### For Existing Scripts

If you have scripts that load from `fbirn_results/combined/`:

**Old code:**
```python
solutions = zkl.load('fbirn_results/combined/all_solutions_info.zkl')
```

**New code (specify timestamp):**
```python
timestamp = "12262025103045"
solutions = zkl.load(f'fbirn_results/{timestamp}/combined/all_solutions_info.zkl')
```

**New code (auto-detect latest):**
```python
import os
results_dir = "fbirn_results"
timestamps = sorted([d for d in os.listdir(results_dir) 
                    if d.isdigit() and len(d) == 14], reverse=True)
latest = timestamps[0]
solutions = zkl.load(f'fbirn_results/{latest}/combined/all_solutions_info.zkl')
```

### For Existing Results

Old results in `fbirn_results/` won't interfere with new timestamped runs. You can:
- Keep them as-is
- Manually move them into a timestamped folder
- Delete them if no longer needed

---

## Implementation Details

### Timestamp Generation
```python
from datetime import datetime
timestamp = datetime.now().strftime('%m%d%Y%H%M%S')
```

### Auto-detection Logic
```python
def find_latest_results():
    """Find the most recent timestamped results folder"""
    base_dir = "fbirn_results"
    if not os.path.exists(base_dir):
        return None
    
    # List all subdirectories that look like timestamps (14 digits)
    subdirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 14:
            subdirs.append((item, item_path))
    
    if not subdirs:
        return None
    
    # Sort by timestamp (most recent first)
    subdirs.sort(reverse=True)
    return subdirs[0][1]  # Return path to most recent
```

---

## Validation

### Check Timestamp Creation
```bash
# Run experiment
python fmri_experiment.py

# Should print:
# Saving results to: fbirn_results/12262025103045
# Timestamp: 12262025103045
```

### Verify Directory Structure
```bash
ls fbirn_results/
# Should show: 12262025103045/

ls fbirn_results/12262025103045/
# Should show: combined/ group_0/ group_1/
```

### Test Auto-detection
```bash
python analyze_saved_solutions.py
# Should print:
# Auto-detected most recent results: fbirn_results/12262025103045
```

---

## FAQ

**Q: Can I still use the old `-d` flag with full paths?**
A: Yes! It's backward compatible:
```bash
python analyze_saved_solutions.py -d fbirn_results/12262025103045/combined
```

**Q: What if I run two experiments at the exact same second?**
A: Extremely unlikely (need to start within 1 second). If it happens, the second run will either add to the same folder or fail. Consider adding milliseconds if this is a concern.

**Q: Can I rename the timestamped folders?**
A: Yes, but then you'll need to use the `-d` flag with the full path instead of `-t` with timestamp.

**Q: How do I delete old results?**
A: Simply delete the timestamped folders you don't need:
```bash
rm -rf fbirn_results/12262025103045/
```

**Q: Can I run experiments in parallel?**
A: Yes! Each run creates a unique timestamped folder, so parallel runs won't interfere with each other.

---

## Related Files

- `fmri_experiment.py` - Main experiment script
- `analyze_saved_solutions.py` - Analysis tool
- `FMRI_QUICK_START.md` - Quick start guide
- `SOLUTION_SELECTION_GUIDE.md` - Comprehensive guide
- `FMRI_SELECTION_CHANGES.md` - Technical changes documentation

