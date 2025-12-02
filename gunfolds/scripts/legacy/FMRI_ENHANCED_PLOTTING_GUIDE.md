# Enhanced fMRI RASL Plotting Guide

## Overview

The `plot_fmri_enhanced.py` script creates publication-quality visualizations of fMRI RASL results using the same style as GCM plots, with network color coding and multiple threshold views.

---

## ✨ Features

### 🎨 Visual Improvements
- **Color-coded by brain networks:**
  - 🔴 **Red:** CEN (rPPC, rDLPFC)
  - 🔵 **Blue:** Salience (rFIC, ACC)
  - 🟢 **Green:** DMN (PCC, VMPFC)
- **Multiple threshold views** (50%, 60%, 70%, 80%, 90%)
- **Publication-quality** (300 DPI, PDF + PNG)
- **Clear legends** and annotations
- **Group comparisons** (side-by-side)

### 📊 Plot Types
1. **Network plots** at various thresholds
2. **Top connections bar charts**
3. **Enhanced heatmaps** with annotations
4. **Group comparison plots** (Group 0 vs Group 1)

---

## 🚀 Usage

### Basic Usage
```bash
python plot_fmri_enhanced.py -t <TIMESTAMP>
```

### Example
```bash
python plot_fmri_enhanced.py -t 11262025164900
```

### Custom Thresholds
```bash
# Default: 50%, 60%, 70%, 80%, 90%
python plot_fmri_enhanced.py -t 11262025164900 --thresholds 0.5 0.6 0.7 0.8 0.9

# Focus on very strong connections only
python plot_fmri_enhanced.py -t 11262025164900 --thresholds 0.7 0.8 0.9
```

### Specific Groups Only
```bash
# Only combined
python plot_fmri_enhanced.py -t 11262025164900 --groups combined

# Only group comparisons
python plot_fmri_enhanced.py -t 11262025164900 --groups 0 1

# All (default)
python plot_fmri_enhanced.py -t 11262025164900 --groups combined 0 1
```

---

## 📁 Output Structure

```
fbirn_results/11262025164900/
├── combined/enhanced_plots/
│   ├── network_thresh_50.png/pdf
│   ├── network_thresh_60.png/pdf
│   ├── network_thresh_70.png/pdf    ⭐ Key finding
│   ├── network_thresh_80.png/pdf    ⭐ Strongest connection
│   ├── heatmap_enhanced.png/pdf
│   └── top_connections.png/pdf
│
├── group_0/enhanced_plots/
│   └── ... (same structure)
│
├── group_1/enhanced_plots/
│   └── ... (same structure)
│
└── group_comparison/
    ├── comparison_thresh_50.png/pdf
    ├── comparison_thresh_60.png/pdf
    ├── comparison_thresh_70.png/pdf  ⭐ Shows group differences
    ├── comparison_thresh_80.png/pdf
    └── comparison_thresh_90.png/pdf
```

---

## 🔍 Key Findings from Your Results

### 💡 **Strongest Connection (89.7% frequency!):**
**VMPFC → rFIC**
- Present in **90% of solutions!**
- **DMN → Salience Network**
- Unidirectional (VMPFC drives rFIC)

This is the **dominant feature** of your RASL causal network!

### Top 5 Connections:
1. **VMPFC → rFIC: 89.7%** ⭐⭐⭐
2. **rDLPFC → ACC: 67.3%**
3. **PCC → rFIC: 64.1%**
4. **VMPFC → PCC: 62.6%**
5. **VMPFC → rPPC: 59.1%**

### Key Pattern:
**VMPFC (DMN) appears to drive other networks!**
- VMPFC → rFIC (Salience): 89.7%
- VMPFC → rDLPFC (CEN): 56.7%
- VMPFC → PCC (DMN): 62.6%
- VMPFC → rPPC (CEN): 59.1%

---

## 🆚 Comparison: RASL vs GCM

| Method | Strongest Connection | Strength | Direction |
|--------|---------------------|----------|-----------|
| **GCM** | rFIC ↔ rDLPFC | 58% | Bidirectional |
| **RASL** | VMPFC → rFIC | 90% | Unidirectional |

### Different Methods, Different Insights:

**GCM (Granger Causality):**
- Identifies **rFIC-rDLPFC** as core hub
- Bidirectional coupling
- Focus on executive-salience interaction

**RASL (Causal Structure Learning):**
- Identifies **VMPFC as major driver**
- Drives salience network (rFIC)
- Suggests DMN may initiate network switching

---

## 🧠 Interpretation

### RASL Findings Suggest:
1. **VMPFC (DMN) is a causal driver**, not just a passive default network
2. **VMPFC → rFIC**: DMN may trigger salience network activation
3. **rDLPFC → ACC**: CEN activates salience monitoring
4. **High overall connectivity**: Most edges >50% frequency

### Comparison with Sridharan et al.:
- ⚠️ **Different emphasis:** RASL shows VMPFC as driver (not just rFIC)
- ⚠️ **Alternative model:** DMN → Salience → CEN cascade?
- ✓ **Consistent:** High interconnectivity between all three networks

### Important Note:
RASL and GCM measure different aspects:
- **GCM:** Time-series Granger causality (temporal precedence)
- **RASL:** Causal graph structure (structural causation)
- **Both valid:** Capture different types of causal relationships

---

## 📊 Group Differences

### At 70% Threshold:
- **Both groups:** VMPFC → rFIC (very strong, ~90%)
- **Both groups:** Same dominant connection
- **Similarity:** Core network structure preserved across groups

### Interpretation:
- Core causal architecture is **robust across clinical groups**
- VMPFC → rFIC connection is **universal** (not group-specific)
- Suggests this is a fundamental brain network property

---

## 🎯 Best Plots for Different Purposes

### For Presentations:
- **`network_thresh_70.png`** - Shows cleanest view of strongest connections
- **`top_connections.png`** - Quantitative summary
- **`comparison_thresh_70.png`** - Group differences

### For Papers:
- **Main Figure:** `network_thresh_50.png` (comprehensive)
- **Supplement:** `heatmap_enhanced.png` (complete matrix)
- **Table:** Data from `top_connections.png`

### For Exploration:
- Start with `network_thresh_50.png` (moderate threshold)
- Increase threshold to focus on strongest connections
- Use `heatmap_enhanced.png` for complete overview

---

## 💡 Usage Tips

### Quick Regeneration
```bash
# Regenerate all plots
python plot_fmri_enhanced.py -t 11262025164900

# Only combined
python plot_fmri_enhanced.py -t 11262025164900 --groups combined

# Only groups
python plot_fmri_enhanced.py -t 11262025164900 --groups 0 1
```

### Custom Thresholds for Your Data
Since your mean frequency is ~57%, good thresholds are:
```bash
python plot_fmri_enhanced.py -t 11262025164900 --thresholds 0.5 0.6 0.7 0.8
```

### For Future Runs
```bash
# Run fMRI experiment
python fmri_experiment.py --selection_mode top_k --top_k 10
# Get timestamp: 12282025091234

# Generate plots
python plot_fmri_enhanced.py -t 12282025091234
```

---

## 🔧 Customization

### Change Node Colors
Edit `plot_fmri_enhanced.py`:
```python
network_colors = {
    'rPPC': '#E64B35',    # CEN - Red
    'rDLPFC': '#E64B35',  # CEN - Red
    'rFIC': '#4DBBD5',    # Salience - Blue
    'ACC': '#4DBBD5',     # Salience - Blue
    'PCC': '#00A087',     # DMN - Green
    'VMPFC': '#00A087'    # DMN - Green
}
```

### Adjust Figure Size
```python
fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
```

### Change Number of Top Connections
```python
plot_top_connections(edge_rate, n_top=20, ...)  # Show top 20
```

---

## 📈 Summary Statistics

From your run:
```
Combined (all subjects):
  Total solutions: 2496
  Mean edge frequency: 57.04%
  Max edge frequency: 89.74% (VMPFC → rFIC)
  
Group 0 (Controls):
  Total solutions: 1215
  Mean edge frequency: 57.63%
  Max edge frequency: 89.71%
  
Group 1 (Patients):
  Total solutions: 1281
  Mean edge frequency: 56.49%
  Max edge frequency: 89.77%
```

**Key Observation:** Very similar across groups - core network structure preserved!

---

## 🔗 Related Scripts

- **`fmri_experiment.py`** - Main RASL analysis
- **`analyze_saved_solutions.py`** - Basic analysis tool
- **`plot_fmri_enhanced.py`** - Enhanced plotting (this script)
- **`plot_gcm_enhanced.py`** - GCM enhanced plotting (similar style)

---

## ✅ What Was Created

For your results in `11262025164900`:

### Combined Plots (All Subjects)
- ✅ 5 network plots at different thresholds
- ✅ Enhanced heatmap
- ✅ Top 15 connections chart

### Group 0 Plots (Controls)
- ✅ 5 network plots
- ✅ Enhanced heatmap  
- ✅ Top 15 connections chart

### Group 1 Plots (Patients)
- ✅ 5 network plots
- ✅ Enhanced heatmap
- ✅ Top 15 connections chart

### Group Comparisons
- ✅ 5 side-by-side comparison plots

**Total: 40 high-quality visualization files (20 PNG + 20 PDF)**

---

**All enhanced plots are now available! Much cleaner and publication-ready! 🎨✨**

