# GCM (Granger Causality Mapping) Quick Start Guide

## 🚀 Running the GCM Analysis

### Basic Usage (All Subjects with Timestamped Output)

```bash
cd /path/to/gunfolds/gunfolds/scripts
python gcm_on_ICA.py
```

This uses default settings:
- Runs on **all subjects**
- Alpha: 0.05 (50/1000)
- Results saved to: `gcm_roebroeck/MMDDYYYYHHMMSS/`

### Run Specific Subject

```bash
# Run only subject 5
python gcm_on_ICA.py -S 5

# Run subject 10 with different alpha
python gcm_on_ICA.py -S 10 -A 10  # alpha = 0.01
```

### Custom Alpha Level

```bash
# Alpha = 0.05 (default)
python gcm_on_ICA.py -A 50

# Alpha = 0.01
python gcm_on_ICA.py -A 10

# Alpha = 0.001
python gcm_on_ICA.py -A 1
```

### Use Existing Timestamp (Add Subjects)

```bash
# Run additional subjects into existing timestamped folder
python gcm_on_ICA.py -S 15 -t 12262025103045
```

### Disable Timestamping (Old Behavior)

```bash
# Save directly to gcm_roebroeck/ without timestamp
python gcm_on_ICA.py --no-timestamp
```

---

## 📁 Output Structure

Each run creates a **timestamped folder**:

```
gcm_roebroeck/
└── 12262025103045/              # Format: MMDDYYYYHHMMSS
    ├── csv/                     # CSV data files
    │   ├── subj0_IC25,26,35,44,45,46_Fdiff_labeled.csv
    │   ├── subj0_IC25,26,35,44,45,46_Pdiff_labeled.csv
    │   ├── subj0_IC25,26,35,44,45,46_Adj_labeled.csv
    │   ├── subj1_IC25,26,35,44,45,46_*.csv
    │   ├── ... (one set per subject)
    │   ├── group_edge_hits.csv       # Aggregated edge counts
    │   ├── group_edge_rate.csv       # Aggregated edge frequencies
    │   └── group_Fdiff_mean.csv      # Mean F-statistics
    │
    ├── figures/                 # Visualization files
    │   ├── subj0_IC25,26,35,44,45,46_gcm.png
    │   ├── subj1_IC25,26,35,44,45,46_gcm.png
    │   ├── ... (one per subject)
    │   └── group_edge_frequency.png  # Group-level graph
    │
    └── run_params.csv           # Run parameters for reference
```

**Timestamp Format:** MMDDYYYYHHMMSS
- Example: `12262025103045` = December 26, 2025 at 10:30:45

---

## 🔍 Analyzing Results

### Option 1: Analyze Most Recent Run (Auto-detect)

```bash
python analyze_gcm_results.py
```

Automatically finds and analyzes the most recent timestamped folder.

### Option 2: Analyze Specific Run

```bash
# Using timestamp
python analyze_gcm_results.py -t 12262025103045

# Using full path
python analyze_gcm_results.py -d gcm_roebroeck/12262025103045
```

### Option 3: Generate Plots

```bash
# Auto-detect most recent + plots
python analyze_gcm_results.py --plot

# Specific timestamp + plots
python analyze_gcm_results.py -t 12262025103045 --plot
```

### Option 4: Apply Edge Frequency Threshold

```bash
# Only show edges with frequency > 30%
python analyze_gcm_results.py --threshold 0.3 --plot

# Only show edges with frequency > 50%
python analyze_gcm_results.py -t 12262025103045 --threshold 0.5 --plot
```

---

## 📊 Analysis Outputs

The analysis script generates:

### Console Output
- Run parameters
- Number of subjects analyzed
- Edge frequency statistics
- Top 10 most frequent edges
- Per-node degree statistics
- F-statistic differences

### Text Files
- `analysis_summary.txt` - Summary statistics

### Plot Files (with `--plot`)
- `edge_frequency_histogram.pdf` / `.png` - Distribution of edge frequencies
- `edge_frequency_heatmap.pdf` / `.png` - Matrix view of edge frequencies
- `fdiff_heatmap.pdf` / `.png` - F-statistic differences
- `threshold_comparison.pdf` / `.png` - Network density vs threshold
- All saved to: `gcm_roebroeck/TIMESTAMP/analysis_plots/`

---

## 💡 Tips

### Finding Your Results

After running GCM, note the timestamp printed:
```
Saving results to: gcm_roebroeck/12262025103045
Timestamp: 12262025103045
```

Use this timestamp for analysis:
```bash
python analyze_gcm_results.py -t 12262025103045 --plot
```

### Comparing Different Alpha Values

Run with different alpha levels:
```bash
# Run 1: Alpha = 0.05
python gcm_on_ICA.py -A 50

# Run 2: Alpha = 0.01
python gcm_on_ICA.py -A 10

# Run 3: Alpha = 0.001
python gcm_on_ICA.py -A 1
```

Each run creates a separate timestamped folder, preserving all results.

### Aggregating Existing CSVs

If you already have per-subject CSV files in a directory:
```bash
python gcm_on_ICA.py --aggregate-csvs /path/to/csv/directory
```

### Loading Results in Python

```python
import pandas as pd
import os

# Find most recent results
results_dir = "gcm_roebroeck"
timestamps = sorted([d for d in os.listdir(results_dir) 
                    if d.isdigit() and len(d) == 14], reverse=True)
latest = timestamps[0]

# Load group-level edge rates
edge_rate = pd.read_csv(f'gcm_roebroeck/{latest}/csv/group_edge_rate.csv', index_col=0)
edge_hits = pd.read_csv(f'gcm_roebroeck/{latest}/csv/group_edge_hits.csv', index_col=0)

print(f"Analyzing run from: {latest}")
print(f"Edge rate matrix shape: {edge_rate.shape}")
print("\nTop edges:")
print(edge_rate.stack().sort_values(ascending=False).head(10))
```

---

## 🔬 Understanding the Output

### Per-Subject Files

For each subject, three CSV files are saved:

1. **`*_Adj_labeled.csv`**: Binary adjacency matrix (0/1)
   - 1 = Significant Granger causality detected
   - 0 = No significant causality

2. **`*_Fdiff_labeled.csv`**: F-statistic differences
   - Higher values = stronger evidence for causality
   - Positive values indicate direction of causality

3. **`*_Pdiff_labeled.csv`**: P-value differences
   - Lower values = more significant
   - Used for statistical thresholding

### Group-Level Files

1. **`group_edge_hits.csv`**: Count of subjects with each edge
   - Values range from 0 to N (number of subjects)
   - Shows consistency across subjects

2. **`group_edge_rate.csv`**: Fraction of subjects with each edge
   - Values range from 0.0 to 1.0
   - 0.7 means 70% of subjects have that edge

3. **`group_Fdiff_mean.csv`**: Mean F-statistic across subjects
   - Average strength of causal relationships
   - Higher = stronger average effect

---

## 🆘 Common Issues

### Issue: "No results found in gcm_roebroeck/"

**Solution:** Make sure you've run the GCM analysis first:
```bash
python gcm_on_ICA.py
```

### Issue: Want to analyze older run

**Solution:** Specify the timestamp explicitly:
```bash
ls gcm_roebroeck/  # List available timestamps
python analyze_gcm_results.py -t <TIMESTAMP> --plot
```

### Issue: Need to add more subjects to existing run

**Solution:** Use the same timestamp:
```bash
python gcm_on_ICA.py -S 20 -t 12262025103045
```

### Issue: Different alpha values between runs

**Solution:** Each run should use a consistent alpha. Compare runs:
```bash
# Check run parameters
cat gcm_roebroeck/12262025103045/run_params.csv
cat gcm_roebroeck/12262025110530/run_params.csv
```

---

## 📈 Example Workflow

```bash
# 1. Run GCM on all subjects with default settings
python gcm_on_ICA.py

# Note the timestamp from output, e.g., 12262025103045

# 2. Analyze results with plots
python analyze_gcm_results.py -t 12262025103045 --plot

# 3. Check different thresholds
python analyze_gcm_results.py -t 12262025103045 --threshold 0.3 --plot
python analyze_gcm_results.py -t 12262025103045 --threshold 0.5 --plot

# 4. Run again with stricter alpha
python gcm_on_ICA.py -A 10  # alpha = 0.01

# 5. Compare results
python analyze_gcm_results.py -t <NEW_TIMESTAMP> --plot

# 6. Load into Python for custom analysis
python
>>> import pandas as pd
>>> edge_rate = pd.read_csv('gcm_roebroeck/12262025103045/csv/group_edge_rate.csv', index_col=0)
>>> print(edge_rate)
```

---

## 🧠 Brain Regions Analyzed

The script analyzes 6 ICA components from the FBIRN dataset:

| Index | Region      | Full Name                                |
|-------|-------------|------------------------------------------|
| 25    | rPPC        | Right Posterior Parietal Cortex          |
| 26    | rFIC        | Right Frontal Insular Cortex             |
| 35    | rDLPFC      | Right Dorsolateral Prefrontal Cortex     |
| 44    | ACC         | Anterior Cingulate Cortex                |
| 45    | PCC         | Posterior Cingulate Cortex               |
| 46    | VMPFC       | Ventromedial Prefrontal Cortex           |

These regions are part of key brain networks involved in:
- Attention and executive function (rDLPFC, rPPC)
- Salience and interoception (rFIC, ACC)
- Default mode network (PCC, VMPFC)

---

## 📚 Parameters Explained

### Alpha (`-A` / `--alpha`)
- **Type:** Integer (multiplied by 1000)
- **Default:** 50 (= 0.05)
- **Description:** Significance level for detecting causal relationships
- **Example:** `-A 10` sets alpha = 0.01 (stricter)

### Subject (`-S` / `--subject`)
- **Type:** Integer
- **Default:** None (runs all subjects)
- **Description:** Subject index to analyze (0-based)
- **Example:** `-S 5` runs only subject 5

### Timestamp (`-t` / `--timestamp`)
- **Type:** String (14 digits)
- **Default:** Auto-generated
- **Description:** Use existing timestamp folder
- **Example:** `-t 12262025103045`

### No Timestamp (`--no-timestamp`)
- **Type:** Flag
- **Description:** Disable timestamped directories (old behavior)
- **Use case:** Legacy compatibility

---

## 🔗 Related Files

- `gcm_on_ICA.py` - Main GCM analysis script
- `analyze_gcm_results.py` - Analysis and visualization tool
- `roebroeck_gcm.py` - Core GCM implementation (imported)

---

## ✨ What's New (Timestamped Version)

### Before:
- Results saved to `gcm_roebroeck/`
- Each run overwrites previous results
- Hard to compare different parameter settings

### After:
- Results saved to `gcm_roebroeck/MMDDYYYYHHMMSS/`
- All runs preserved automatically
- Easy comparison across alpha values
- Auto-detection of most recent run
- Comprehensive analysis tools

---

## 🎯 Best Practices

1. **Always check the timestamp** after running
2. **Save the timestamp** for reproducibility
3. **Compare multiple alpha values** to assess robustness
4. **Use thresholds** to focus on consistent edges
5. **Inspect per-subject variability** using individual CSVs
6. **Visualize with plots** for easier interpretation

---

For more information on Granger causality methodology, see the Roebroeck et al. papers on GCM.

