# Timestamped Results - Complete Index

## Overview

Two analysis pipelines now use timestamped directory structures for organized, non-destructive result storage:

1. **fMRI RASL Experiment** (`fmri_experiment.py`)
2. **GCM Analysis** (`gcm_on_ICA.py`)

Both save results in format: `base_directory/MMDDYYYYHHMMSS/`

---

## Quick Reference

### fMRI RASL Experiment

| Item | Details |
|------|---------|
| **Script** | `fmri_experiment.py` |
| **Base Directory** | `fbirn_results/` |
| **Analysis Tool** | `analyze_saved_solutions.py` |
| **Documentation** | `FMRI_QUICK_START.md` |

**Run:**
```bash
python fmri_experiment.py --selection_mode top_k --top_k 10
```

**Analyze:**
```bash
python analyze_saved_solutions.py --plot
```

---

### GCM Analysis

| Item | Details |
|------|---------|
| **Script** | `gcm_on_ICA.py` |
| **Base Directory** | `gcm_roebroeck/` |
| **Analysis Tool** | `analyze_gcm_results.py` |
| **Documentation** | `GCM_QUICK_START.md` |

**Run:**
```bash
python gcm_on_ICA.py
```

**Analyze:**
```bash
python analyze_gcm_results.py --plot
```

---

## Common Timestamp Format

**Format:** `MMDDYYYYHHMMSS`

**Example:** `12262025103045` = December 26, 2025 at 10:30:45

Both pipelines use the exact same timestamp format for consistency.

---

## Directory Structures

### fMRI RASL Results
```
fbirn_results/
└── MMDDYYYYHHMMSS/
    ├── combined/
    │   ├── all_solutions_info.zkl
    │   ├── selection_params.zkl
    │   ├── analysis_summary.csv
    │   └── *.pdf (graphs)
    ├── group_0/
    │   ├── subjects/*.pdf
    │   ├── solutions/*.zkl
    │   └── *.pdf (group graphs)
    └── group_1/
        └── ... (same as group_0)
```

### GCM Results
```
gcm_roebroeck/
└── MMDDYYYYHHMMSS/
    ├── csv/
    │   ├── subj*_Adj_labeled.csv
    │   ├── group_edge_hits.csv
    │   ├── group_edge_rate.csv
    │   └── group_Fdiff_mean.csv
    ├── figures/
    │   ├── subj*_gcm.png
    │   └── group_edge_frequency.png
    ├── analysis_plots/
    │   └── *.pdf (analysis visualizations)
    ├── run_params.csv
    └── analysis_summary.txt
```

---

## Documentation Files

### fMRI RASL Experiment
- **`FMRI_QUICK_START.md`** - Quick start guide
- **`SOLUTION_SELECTION_GUIDE.md`** - Comprehensive usage guide
- **`FMRI_SELECTION_CHANGES.md`** - Technical changes
- **`TIMESTAMPED_RESULTS_SUMMARY.md`** - Timestamping summary

### GCM Analysis
- **`GCM_QUICK_START.md`** - Quick start guide
- **`GCM_TIMESTAMPED_SUMMARY.md`** - Comprehensive summary

### This File
- **`TIMESTAMPED_RESULTS_INDEX.md`** - You are here!

---

## Common Workflow

### 1. Run Analysis
```bash
# fMRI RASL
python fmri_experiment.py --selection_mode top_k --top_k 10

# GCM
python gcm_on_ICA.py
```

### 2. Note the Timestamp
Both scripts print:
```
Saving results to: <directory>/12262025103045
Timestamp: 12262025103045
```

### 3. Analyze Results (Auto-detect)
```bash
# fMRI RASL
python analyze_saved_solutions.py --plot

# GCM
python analyze_gcm_results.py --plot
```

### 4. Analyze Specific Run
```bash
# fMRI RASL
python analyze_saved_solutions.py -t 12262025103045 --plot

# GCM
python analyze_gcm_results.py -t 12262025103045 --plot
```

---

## Common Features

### Both Pipelines Support:

✅ **Auto-timestamping** - Automatic unique folder creation
✅ **Auto-detection** - Analysis tools find most recent run
✅ **Manual specification** - Use `-t TIMESTAMP` for specific runs
✅ **Parameter saving** - All settings saved for reproducibility
✅ **Multiple runs** - No overwriting, all results preserved
✅ **Visualization** - Comprehensive plotting with `--plot`

---

## Analysis Tools Comparison

| Feature | fMRI (`analyze_saved_solutions.py`) | GCM (`analyze_gcm_results.py`) |
|---------|-------------------------------------|--------------------------------|
| Auto-detect latest | ✓ | ✓ |
| Specify timestamp | `-t` | `-t` |
| Specify directory | `-d` | `-d` |
| Generate plots | `--plot` | `--plot` |
| Group-specific | `-g 0` or `-g 1` | N/A |
| Threshold | N/A | `--threshold 0.3` |
| Ground truth comparison | `--compare_gt` | N/A |

---

## Timestamp Management

### Finding Available Timestamps

**fMRI:**
```bash
ls fbirn_results/
```

**GCM:**
```bash
ls gcm_roebroeck/
```

### Loading Most Recent in Python

**fMRI:**
```python
import os
from gunfolds.utils import zickle as zkl

results_dir = "fbirn_results"
timestamps = sorted([d for d in os.listdir(results_dir) 
                    if d.isdigit() and len(d) == 14], reverse=True)
latest = timestamps[0]

solutions = zkl.load(f'fbirn_results/{latest}/combined/all_solutions_info.zkl')
params = zkl.load(f'fbirn_results/{latest}/combined/selection_params.zkl')
```

**GCM:**
```python
import os
import pandas as pd

results_dir = "gcm_roebroeck"
timestamps = sorted([d for d in os.listdir(results_dir) 
                    if d.isdigit() and len(d) == 14], reverse=True)
latest = timestamps[0]

edge_rate = pd.read_csv(f'gcm_roebroeck/{latest}/csv/group_edge_rate.csv', index_col=0)
params = pd.read_csv(f'gcm_roebroeck/{latest}/run_params.csv')
```

### Comparing Multiple Runs

```python
import pandas as pd

# Load two different runs
timestamp1 = "12262025103045"
timestamp2 = "12262025110530"

# fMRI: Compare selection modes
from gunfolds.utils import zickle as zkl
params1 = zkl.load(f'fbirn_results/{timestamp1}/combined/selection_params.zkl')
params2 = zkl.load(f'fbirn_results/{timestamp2}/combined/selection_params.zkl')
print(f"Run 1: {params1['selection_mode']}")
print(f"Run 2: {params2['selection_mode']}")

# GCM: Compare alpha values
params1 = pd.read_csv(f'gcm_roebroeck/{timestamp1}/run_params.csv')
params2 = pd.read_csv(f'gcm_roebroeck/{timestamp2}/run_params.csv')
alpha1 = float(params1[params1['parameter'] == 'alpha']['value'].iloc[0])
alpha2 = float(params2[params2['parameter'] == 'alpha']['value'].iloc[0])
print(f"Run 1 alpha: {alpha1}")
print(f"Run 2 alpha: {alpha2}")
```

---

## Cleaning Up Old Results

### Delete Specific Timestamp
```bash
# fMRI
rm -rf fbirn_results/12262025103045/

# GCM
rm -rf gcm_roebroeck/12262025103045/
```

### Keep Only Most Recent N Runs

**Bash script example:**
```bash
#!/bin/bash
# keep_recent.sh - Keep only 5 most recent runs

BASE_DIR="fbirn_results"  # or "gcm_roebroeck"
KEEP_N=5

cd "$BASE_DIR" || exit
ls -1 | grep -E '^[0-9]{14}$' | sort -r | tail -n +$((KEEP_N + 1)) | xargs rm -rf
```

### Archive Old Results
```bash
# Create archive of runs older than 30 days
find fbirn_results/ -maxdepth 1 -type d -mtime +30 -name '[0-9]*' -exec tar -czf old_fmri_runs.tar.gz {} +
find gcm_roebroeck/ -maxdepth 1 -type d -mtime +30 -name '[0-9]*' -exec tar -czf old_gcm_runs.tar.gz {} +
```

---

## Troubleshooting

### Issue: "No results found"

**fMRI:**
```bash
# Check for results
ls fbirn_results/
# If empty, run experiment first
python fmri_experiment.py
```

**GCM:**
```bash
# Check for results
ls gcm_roebroeck/
# If empty, run analysis first
python gcm_on_ICA.py
```

### Issue: Cannot find specific timestamp

**Solution:** List available timestamps and verify spelling
```bash
ls fbirn_results/    # or gcm_roebroeck/
# Use exact timestamp shown
```

### Issue: Want to compare parameter settings

**Solution:** Check saved parameters
```bash
# fMRI
python -c "from gunfolds.utils import zickle as zkl; print(zkl.load('fbirn_results/12262025103045/combined/selection_params.zkl'))"

# GCM
cat gcm_roebroeck/12262025103045/run_params.csv
```

---

## Best Practices

### 1. **Always Note the Timestamp**
Save the timestamp from console output for future reference.

### 2. **Document Your Experiments**
Keep a lab notebook with timestamps and parameter choices:
```
12262025103045 - fMRI top_k=10 (baseline)
12262025110530 - fMRI delta=1.9 (testing delta)
12262025114820 - GCM alpha=0.05 (default)
12262025120015 - GCM alpha=0.01 (stricter)
```

### 3. **Use Consistent Naming**
If manually organizing, use descriptive names:
```
fbirn_results/
├── 12262025103045_baseline_top10/
├── 12262025110530_delta19/
└── 12262025120000_delta25/
```

### 4. **Regular Cleanup**
Archive or delete old runs to save disk space.

### 5. **Verify Parameters**
Always check saved parameters before comparing runs.

---

## Integration with Workflow

### Example: Complete Analysis Pipeline

```bash
#!/bin/bash
# complete_analysis.sh - Run both fMRI and GCM analyses

echo "=== Starting fMRI RASL Analysis ==="
python fmri_experiment.py --selection_mode top_k --top_k 10
FMRI_TIMESTAMP=$(ls -1t fbirn_results/ | head -1)
echo "fMRI results: fbirn_results/$FMRI_TIMESTAMP"

echo "=== Analyzing fMRI Results ==="
python analyze_saved_solutions.py -t $FMRI_TIMESTAMP --plot

echo "=== Starting GCM Analysis ==="
python gcm_on_ICA.py
GCM_TIMESTAMP=$(ls -1t gcm_roebroeck/ | head -1)
echo "GCM results: gcm_roebroeck/$GCM_TIMESTAMP"

echo "=== Analyzing GCM Results ==="
python analyze_gcm_results.py -t $GCM_TIMESTAMP --plot

echo "=== Complete! ==="
echo "fMRI: fbirn_results/$FMRI_TIMESTAMP"
echo "GCM: gcm_roebroeck/$GCM_TIMESTAMP"
```

---

## Summary

Both analysis pipelines now feature:

✅ **Organized** - Timestamped folders for each run
✅ **Reproducible** - Parameters saved automatically
✅ **Comparable** - Easy to compare across settings
✅ **Safe** - No overwriting of previous results
✅ **Documented** - Comprehensive guides available
✅ **Automated** - Auto-detection of latest runs
✅ **Flexible** - Manual specification when needed

All changes are production-ready! 🎉

---

## Quick Links

### Getting Started
- fMRI: Read `FMRI_QUICK_START.md`
- GCM: Read `GCM_QUICK_START.md`

### Detailed Information
- fMRI: Read `SOLUTION_SELECTION_GUIDE.md`
- GCM: Read `GCM_TIMESTAMPED_SUMMARY.md`

### Technical Details
- fMRI Changes: `FMRI_SELECTION_CHANGES.md`
- Timestamping: `TIMESTAMPED_RESULTS_SUMMARY.md`

---

**Last Updated:** December 2025

