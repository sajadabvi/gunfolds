# Enhanced GCM Plotting Guide

## Overview

The `plot_gcm_enhanced.py` script creates publication-quality visualizations of GCM results with better aesthetics, color coding by brain networks, and multiple threshold views.

---

## 🎨 What's Improved

### Before (Original):
- ❌ Shows ALL edges (cluttered)
- ❌ No color coding by functional networks
- ❌ Hard to see key connections
- ❌ Single view only
- ❌ Basic aesthetics

### After (Enhanced):
- ✅ **Color-coded by brain networks:**
  - 🔴 **Red:** CEN (Central Executive) - rPPC, rDLPFC
  - 🔵 **Blue:** Salience Network - rFIC, ACC
  - 🟢 **Green:** DMN (Default Mode) - PCC, VMPFC
- ✅ **Multiple threshold views** (0%, 30%, 40%, 50%)
- ✅ **Two layout styles** (circular + spring)
- ✅ **Top connections bar chart**
- ✅ **Enhanced heatmap** with annotations
- ✅ **Publication-quality** (high DPI, both PDF and PNG)
- ✅ **Clearer edge visualization** with proper arrows

---

## 🚀 Usage

### Basic Usage
```bash
python plot_gcm_enhanced.py -t <TIMESTAMP>
```

### Custom Thresholds
```bash
# Default: 0%, 30%, 40%, 50%
python plot_gcm_enhanced.py -t 11272025173313 --thresholds 0.0 0.3 0.4 0.5

# Only show strong connections
python plot_gcm_enhanced.py -t 11272025173313 --thresholds 0.4 0.5 0.6
```

---

## 📁 Output Files

All plots saved to: `gcm_roebroeck/<TIMESTAMP>/enhanced_plots/`

### Network Plots (Multiple Thresholds)

**Circular Layout:**
- `network_circular_thresh_0.png/pdf` - All edges (threshold > 0%)
- `network_circular_thresh_30.png/pdf` - Edges > 30%
- `network_circular_thresh_40.png/pdf` - Edges > 40%
- `network_circular_thresh_50.png/pdf` - Edges > 50%

**Spring Layout:**
- `network_spring_thresh_0.png/pdf` - All edges
- `network_spring_thresh_30.png/pdf` - Edges > 30%
- `network_spring_thresh_40.png/pdf` - Edges > 40%
- `network_spring_thresh_50.png/pdf` - Edges > 50%

### Summary Plots

- `top_connections.png/pdf` - Bar chart of top 15 connections
- `heatmap_enhanced.png/pdf` - Matrix view with annotations

**All files available in both PNG (for viewing) and PDF (for publication).**

---

## 🎨 Visualization Types

### 1. **Circular Network (Threshold-based)**

**Features:**
- Nodes arranged in circle
- Color-coded by brain network
- Shows only edges above threshold
- Bidirectional edges curved
- Edge thickness = connection strength
- Legend explains color coding

**Best for:**
- Showing overall network topology
- Comparing different thresholds
- Highlighting key connections

**Thresholds:**
- **0%:** All edges (may be cluttered)
- **30%:** Moderate-strong connections ⭐ **Recommended**
- **40%:** Strong connections only
- **50%:** Very strong connections (only rFIC ↔ rDLPFC)

### 2. **Spring Layout Network**

**Features:**
- Force-directed layout
- Strongly connected nodes cluster together
- Same color coding as circular
- Alternative perspective

**Best for:**
- Identifying clusters
- Seeing connectivity hubs
- Different visual perspective

### 3. **Top Connections Bar Chart**

**Features:**
- Top 15 strongest connections
- Color-coded by connection type:
  - 🟣 Purple: Salience → CEN
  - 🟠 Orange: Salience → DMN
  - 🔴 Red: CEN connections
  - 🟢 Green: DMN connections
  - 🔵 Blue: Within Salience

**Best for:**
- Quantitative comparison
- Identifying strongest pathways
- Publication tables

### 4. **Enhanced Heatmap**

**Features:**
- Complete matrix view
- Better color scheme (blue → yellow → red)
- Annotations on strong connections (>40%)
- Grid for clarity
- Diagonal masked (no self-loops)

**Best for:**
- Complete overview
- Checking all pairwise connections
- Spotting patterns in connectivity

---

## 📊 Interpreting the Results

### Network at 50% Threshold

Shows **only the most consistent connections** (present in >50% of subjects):

- **rFIC ↔ rDLPFC** (58% both directions)
  - **Strongest connection in the entire network**
  - Bidirectional coupling
  - Salience Network ↔ Central Executive Network

**Interpretation:** This is the **core hub** of the network.

### Network at 30-40% Threshold

Shows **moderate-to-strong connections:**

Key patterns visible:
1. **Salience → DMN:**
   - ACC → PCC: 37%
   - ACC → VMPFC: 36%
   - rFIC → VMPFC: 34%
   
2. **CEN → DMN:**
   - rDLPFC → VMPFC: 37%
   - rDLPFC → PCC: 34%
   
3. **Within DMN:**
   - PCC ↔ VMPFC: 39% (both directions)
   - Strong internal DMN connectivity

### Top Connections Chart

**Key Insights:**
1. **rFIC ↔ rDLPFC dominates** (58% both directions)
2. **DMN regions highly connected** (PCC ↔ VMPFC: 39%)
3. **Salience → DMN present** (ACC → DMN: 36-37%)
4. **CEN → DMN present** (rDLPFC → DMN: 34-37%)

---

## 🎯 Comparing with Sridharan et al. (2008)

### What Your Plots Show:

**At 50% Threshold (Most Robust):**
- ✅ rFIC ↔ rDLPFC bidirectional coupling (58%)
- ⚠️ **Bidirectional**, not unidirectional as paper emphasized

**At 30% Threshold (Moderate Connections):**
- ✅ rFIC connects to both CEN and DMN
- ✅ ACC (salience partner) → DMN regions
- ⚠️ Many other connections present (complex network)

### Key Finding:
The **rFIC ↔ rDLPFC** connection is by far the **dominant feature**, appearing in **58% of 310 subjects** - a very robust finding that validates the paper's emphasis on rFIC's role, but suggests **reciprocal interaction** rather than pure one-way control.

---

## 💡 Usage Tips

### For Presentations

Use **threshold=40% or 50%** for cleaner slides:
```bash
python plot_gcm_enhanced.py -t 11272025173313 --thresholds 0.4 0.5
```

Shows only the most important connections.

### For Papers

Include multiple views:
1. **Circular network (30%)** - Main figure showing network structure
2. **Top connections bar chart** - Quantitative supplement
3. **Heatmap** - Complete matrix in supplement

### For Exploration

Start with threshold=30%:
```bash
python plot_gcm_enhanced.py -t 11272025173313 --thresholds 0.3
```

Then adjust based on what you see.

---

## 🔧 Customization

### Change Thresholds

Edit command line:
```bash
# Focus on very strong only
python plot_gcm_enhanced.py -t 11272025173313 --thresholds 0.45 0.5 0.55

# More granular range
python plot_gcm_enhanced.py -t 11272025173313 --thresholds 0.2 0.25 0.3 0.35 0.4
```

### Change Node Colors

Edit `plot_gcm_enhanced.py`:
```python
network_colors = {
    'rPPC': '#YOUR_COLOR',    # CEN
    'rDLPFC': '#YOUR_COLOR',  # CEN
    'rFIC': '#YOUR_COLOR',    # Salience
    'ACC': '#YOUR_COLOR',     # Salience
    'PCC': '#YOUR_COLOR',     # DMN
    'VMPFC': '#YOUR_COLOR'    # DMN
}
```

### Change Figure Size

Edit `plot_gcm_enhanced.py`:
```python
fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')  # Larger
```

---

## 📈 Summary Statistics

The script automatically prints:
```
Total subjects: 310
Mean edge frequency: 34.02%
Max edge frequency: 58.06%
Edges > 30%: 21
Edges > 40%: 2  
Edges > 50%: 2  (both are rFIC ↔ rDLPFC!)
```

---

## 🔄 Regenerating for New Results

Whenever you run a new GCM analysis:

```bash
# Run GCM analysis
./submit_gcm_parallel.sh -n 310 -A 50 -c 20
# Get timestamp: 12282025091234

# Wait for completion or use watch mode
python analyze_gcm_results.py -t 12282025091234 --watch --expected-subjects 310

# Generate enhanced plots
python plot_gcm_enhanced.py -t 12282025091234
```

---

## 📚 Files

- **`plot_gcm_enhanced.py`** - Enhanced plotting script
- **Original:** `gcm_on_ICA.py` - Still generates basic plots
- **Analysis:** `analyze_gcm_results.py` - Statistics and basic plots

---

## ✨ Key Improvements Summary

| Feature | Original | Enhanced |
|---------|----------|----------|
| Network Color Coding | ❌ | ✅ (by brain system) |
| Multiple Thresholds | ❌ | ✅ (4 levels) |
| Two Layout Styles | ❌ | ✅ (circular + spring) |
| Top Connections Chart | ❌ | ✅ |
| Enhanced Heatmap | ❌ | ✅ |
| High Resolution | ❌ | ✅ (300 DPI) |
| PDF + PNG | ❌ | ✅ |
| Legend | ❌ | ✅ |
| Publication Ready | ❌ | ✅ |

---

**All enhanced plots are now available in:**
`gcm_roebroeck/11272025173313/enhanced_plots/`

Enjoy your beautiful visualizations! 🎨

