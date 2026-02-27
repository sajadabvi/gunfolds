# Enhanced Plotting - Quick Reference

## 📊 Quick Commands

### For GCM Results
```bash
python plot_gcm_enhanced.py -t <TIMESTAMP>
```

### For fMRI RASL Results
```bash
python plot_fmri_enhanced.py -t <TIMESTAMP>
```

---

## 🎨 What Gets Created

### GCM Plots
```
gcm_roebroeck/TIMESTAMP/enhanced_plots/
├── network_circular_thresh_*.png/pdf    (4 thresholds: 0%, 30%, 40%, 50%)
├── network_spring_thresh_*.png/pdf      (alternative layout)
├── top_connections.png/pdf               (bar chart)
└── heatmap_enhanced.png/pdf              (matrix view)
```

### fMRI RASL Plots
```
fbirn_results/TIMESTAMP/
├── combined/enhanced_plots/
│   ├── network_thresh_*.png/pdf         (5 thresholds: 50%-90%)
│   ├── top_connections.png/pdf
│   └── heatmap_enhanced.png/pdf
├── group_0/enhanced_plots/
│   └── ... (same as combined)
├── group_1/enhanced_plots/
│   └── ... (same as combined)
└── group_comparison/
    └── comparison_thresh_*.png/pdf      (5 thresholds)
```

---

## 🎯 Key Findings at a Glance

### GCM (Temporal Causality)
**Strongest:** rFIC ↔ rDLPFC (58%, bidirectional)
- Salience ↔ Executive reciprocal coupling
- Temporal information flow

### fMRI RASL (Structural Causality)
**Strongest:** VMPFC → rFIC (89.7%, unidirectional)
- DMN → Salience structural pathway
- VMPFC as causal driver

### Sridharan et al. (2008)
**Emphasis:** rFIC → DLPFC (event-triggered)
- rFIC switches networks
- Event-driven activation

---

## 🔄 When to Use Each Plot

### Network Plots (Circular/Spring)
**Use for:**
- Main figures in presentations
- Showing network topology
- Highlighting key connections

**Best Thresholds:**
- **GCM:** 30-40% (balanced view)
- **RASL:** 70-80% (cleanest view)

### Top Connections Bar Chart
**Use for:**
- Quantitative comparisons
- Creating tables
- Showing rankings

### Heatmap
**Use for:**
- Supplementary materials
- Complete connectivity matrix
- Checking all pairwise connections

### Group Comparison (RASL only)
**Use for:**
- Clinical studies
- Group differences
- Side-by-side comparison

---

## 💡 Customization

### Different Thresholds
```bash
# GCM - focus on strong connections
python plot_gcm_enhanced.py -t 11272025173313 --thresholds 0.35 0.4 0.45 0.5

# RASL - focus on very strong
python plot_fmri_enhanced.py -t 11262025164900 --thresholds 0.7 0.8 0.9
```

### Specific Groups Only
```bash
# Only combined
python plot_fmri_enhanced.py -t 11262025164900 --groups combined

# Only clinical groups
python plot_fmri_enhanced.py -t 11262025164900 --groups 0 1
```

---

## 📈 Statistical Summary

### GCM (11272025173313):
- Subjects: 310
- Mean frequency: 34.02%
- Max: 58.06% (rFIC ↔ rDLPFC)
- Edges >50%: 2 (both rFIC ↔ rDLPFC)

### RASL (11262025164900):
- Subjects: 310
- Solutions: 2496 total
- Mean frequency: 57.04%
- Max: 89.74% (VMPFC → rFIC)
- Edges >70%: 1 (VMPFC → rFIC)
- Edges >80%: 1 (VMPFC → rFIC)

---

## 🎨 Color Scheme (Both Methods)

Consistent across all plots:

| Network | Color | Regions |
|---------|-------|---------|
| **CEN** | 🔴 Red (#E64B35) | rPPC, rDLPFC |
| **Salience** | 🔵 Blue (#4DBBD5) | rFIC, ACC |
| **DMN** | 🟢 Green (#00A087) | PCC, VMPFC |

### Connection Colors (Bar Charts):
- 🟣 Purple: Salience → CEN
- 🟠 Orange: Salience → DMN
- 🔴 Red: CEN connections
- 🟢 Green: DMN connections
- 🔵 Blue: Within Salience
- 🟤 Brown: CEN → DMN

---

## 📁 File Locations

### Your Current Results:

**GCM:**
```
gcm_roebroeck/11272025173313/enhanced_plots/
```

**RASL:**
```
fbirn_results/11262025164900/combined/enhanced_plots/
fbirn_results/11262025164900/group_0/enhanced_plots/
fbirn_results/11262025164900/group_1/enhanced_plots/
fbirn_results/11262025164900/group_comparison/
```

---

## 🔧 Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `plot_gcm_enhanced.py` | GCM visualization | Timestamp | Network plots, heatmap, bar chart |
| `plot_fmri_enhanced.py` | RASL visualization | Timestamp | Network plots, heatmap, bar chart, comparisons |

---

## 📝 Documentation

| File | Content |
|------|---------|
| `ENHANCED_PLOTTING_GUIDE.md` | GCM plotting guide |
| `FMRI_ENHANCED_PLOTTING_GUIDE.md` | RASL plotting guide |
| `METHODS_COMPARISON_SUMMARY.md` | Comprehensive comparison |
| `PLOTTING_QUICK_REFERENCE.md` | This file |

---

## ✨ Quick Tips

1. **Start with default thresholds** - they're optimized for each method
2. **Use circular layout** - cleaner for presentations
3. **Include bar charts** - great for quantitative claims
4. **Group comparisons** (RASL) - essential for clinical papers
5. **Both PNG and PDF** - automatically generated

---

## 🚀 Workflow

```bash
# 1. Run analysis
python gcm_on_ICA.py              # or
python fmri_experiment.py

# 2. Generate enhanced plots
python plot_gcm_enhanced.py -t <TIMESTAMP>    # or
python plot_fmri_enhanced.py -t <TIMESTAMP>

# 3. Select best plots for your needs
# - Presentations: network_thresh_70 or 80
# - Papers: network_thresh_50 or 60 + top_connections
# - Supplement: heatmap_enhanced
```

---

**All your visualizations are publication-ready! 🎉**

