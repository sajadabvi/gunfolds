# Comprehensive Methods Comparison: GCM vs RASL vs Sridharan et al.

## Overview

This document compares findings from three approaches to understanding brain network causality:
1. **Sridharan et al. (2008)** - Event-related fMRI + Granger causality
2. **Your GCM Analysis** - Roebroeck GCM on 310 subjects
3. **Your RASL Analysis** - Causal structure learning on 310 subjects

---

## 🔬 Methodology Comparison

| Aspect | Sridharan et al. (2008) | Your GCM | Your RASL |
|--------|------------------------|----------|-----------|
| **Method** | Event-related fMRI + GC | Roebroeck GCM | DRASL (causal discovery) |
| **Subjects** | 18-25 per experiment | 310 (fBIRN) | 310 (fBIRN) |
| **Data Type** | Task-evoked (music, oddball) | Resting/task | Resting/task |
| **Causality** | Temporal (GC) | Temporal (GC) | Structural |
| **Alpha** | Not specified | 0.05 | N/A (cost-based) |
| **Output** | Directional influences | Directional influences | Causal graphs |

---

## 🎯 Key Findings Comparison

### **1. Sridharan et al. (2008) Main Finding:**
**rFIC plays critical causal role in network switching**
- rFIC → DLPFC (strong)
- rFIC → ACC (strong)
- rFIC → PCC/VMPFC (deactivation)
- Emphasis: rFIC as **causal outflow hub**

### **2. Your GCM Analysis (310 subjects):**
**rFIC ↔ rDLPFC bidirectional coupling is dominant**

| Connection | Frequency | Type |
|------------|-----------|------|
| rFIC ↔ rDLPFC | **58%** | Bidirectional ⭐⭐⭐ |
| PCC ↔ VMPFC | 39% | Bidirectional (DMN) |
| rDLPFC → VMPFC | 37% | CEN → DMN |
| ACC → PCC | 37% | Salience → DMN |

**Key Insight:** rFIC and rDLPFC form **reciprocal hub**, not one-way control

### **3. Your RASL Analysis (310 subjects):**
**VMPFC drives salience network (rFIC)**

| Connection | Frequency | Type |
|------------|-----------|------|
| VMPFC → rFIC | **89.7%** | Unidirectional ⭐⭐⭐ |
| rDLPFC → ACC | 67.3% | CEN → Salience |
| PCC → rFIC | 64.1% | DMN → Salience |
| VMPFC → PCC | 62.6% | Within DMN |

**Key Insight:** VMPFC (DMN) is **structural driver** of salience network

---

## 🔍 Synthesis: What Do These Findings Tell Us?

### Different Methods Reveal Different Causal Layers:

#### **Temporal Causality (GCM):**
- **rFIC ↔ rDLPFC** most consistently precede/follow each other
- Suggests tight **temporal coupling** during cognitive processing
- **Bidirectional** information flow between salience and executive systems

#### **Structural Causality (RASL):**
- **VMPFC → rFIC** is strongest structural pathway
- Suggests DMN's architecture positions it to **initiate** salience activation
- VMPFC may be **upstream** in causal hierarchy

#### **Event-Related (Sridharan):**
- **rFIC** activates early during salient events
- Acts as **switch** between CEN and DMN
- Emphasis on rFIC's **triggering role**

---

## 🧠 Integrated Model

### Proposed Multi-Level Causality:

```
STRUCTURAL LEVEL (RASL):
    VMPFC (DMN hub)
        ↓
    rFIC (Salience)
        ↓
    CEN + DMN regulation

TEMPORAL LEVEL (GCM):
    rFIC ↔ rDLPFC (tight coupling)
        ↕
    Joint regulation of DMN

EVENT LEVEL (Sridharan):
    Salient event
        ↓
    rFIC activation
        ↓
    CEN on, DMN off
```

### Interpretation:
1. **Structural architecture** (RASL): VMPFC positioned upstream
2. **Moment-to-moment dynamics** (GCM): rFIC-rDLPFC reciprocal interaction
3. **Event-driven switching** (Sridharan): rFIC initiates network transitions

**All three are true at different timescales/levels!**

---

## 📊 Quantitative Comparison

### Strongest Connections:

| Method | Connection | Strength | Interpretation |
|--------|------------|----------|----------------|
| **GCM** | rFIC ↔ rDLPFC | 58% | Temporal coupling |
| **RASL** | VMPFC → rFIC | 89.7% | Structural causation |
| **Sridharan** | rFIC → DLPFC | N/A* | Event-triggered |

*Sridharan used different metrics (latency, GC coefficients)

### Agreement on Key Points:

✅ **All methods agree:**
1. rFIC is critically involved in network interactions
2. rFIC connects salience to both CEN and DMN
3. DLPFC (CEN) and rFIC (Salience) are tightly linked
4. Multiple pathways between the three networks

⚠️ **Methods differ on:**
1. **Directionality:** GCM shows bidirectional, RASL shows VMPFC-driven
2. **Primary hub:** rFIC (GCM) vs VMPFC (RASL)
3. **Mechanism:** Reciprocal (GCM) vs hierarchical (RASL, Sridharan)

---

## 💡 Why Different Results?

### 1. **Different Types of Causality:**
- **GCM:** "X temporally precedes Y" (Granger causality)
- **RASL:** "X structurally causes Y" (interventional causality)
- **Both valid:** Different aspects of brain network organization

### 2. **Different Timescales:**
- **Event-related (Sridharan):** Milliseconds to seconds (transient switching)
- **GCM (yours):** Seconds to minutes (ongoing dynamics)
- **RASL (yours):** Structural architecture (timescale-invariant)

### 3. **Different Tasks:**
- **Sridharan:** Active tasks (music, oddball)
- **Yours:** Resting state (intrinsic connectivity)
- **Impact:** Task may engage different pathways

---

## 🎯 Key Insights from Your Data

### 1. **RASL Reveals VMPFC as Structural Driver**
- **VMPFC → rFIC: 89.7%** (strongest connection)
- Suggests DMN is not just "default" but actively initiates salience
- May explain why DMN activation precedes task engagement

### 2. **GCM Shows rFIC-DLPFC Reciprocal Hub**
- **rFIC ↔ DLPFC: 58%** (bidirectional)
- Suggests integrated processing during ongoing cognition
- Challenges simple hierarchical view

### 3. **Both Consistent with Sridharan's Core Finding**
- rFIC is central to network interactions ✓
- rFIC connects to both CEN and DMN ✓
- But add nuance: reciprocal + VMPFC influence

---

## 📈 Clinical Implications

### Group Comparison (RASL):
- **No major differences** between Group 0 and Group 1
- **VMPFC → rFIC** equally strong in both (~90%)
- Core network architecture **preserved** in schizophrenia

### Interpretation:
- Structural brain network organization is **robust**
- If clinical differences exist, may be in:
  - Temporal dynamics (GCM might show group differences)
  - Task-related modulation
  - Connection strength (not topology)

---

## 📚 Publication Strategy

### For Your Paper:

#### Main Figures:
1. **RASL Network (70% threshold)** - Shows VMPFC → rFIC as core
2. **Top Connections Chart** - Quantitative summary
3. **Group Comparison** - Shows preservation across groups

#### Supplementary:
1. **GCM Network (30% threshold)** - Shows rFIC-DLPFC coupling
2. **Complete Heatmaps** - Full connectivity matrices
3. **Multiple thresholds** - Robustness across thresholds

#### Text Points:
- "RASL causal discovery reveals VMPFC → rFIC as strongest structural pathway (89.7%)"
- "GCM analysis shows reciprocal rFIC-DLPFC coupling (58%)"
- "Core network architecture preserved across clinical groups"
- "Findings extend Sridharan et al. (2008) by revealing structural antecedents"

---

## 🔬 Future Directions

### To Further Investigate:

1. **Task-based RASL/GCM:**
   - Do patterns differ during active tasks?
   - Does VMPFC → rFIC strengthen with task demands?

2. **Temporal Dynamics:**
   - Sliding window GCM
   - State transitions in RASL

3. **Individual Differences:**
   - Correlate connection strength with behavior
   - Look for subgroups within clinical populations

4. **Direct Comparison:**
   - Run GCM and RASL on same exact data
   - Examine concordance/discordance

---

## ✨ Summary

### Your Unique Contributions:

1. **Large-scale replication** (310 subjects vs 18-25)
2. **Structural causality** via RASL (novel approach)
3. **Group preservation** finding
4. **Multi-method convergence** (GCM + RASL)

### Agreement with Sridharan et al.:
- ✅ rFIC central to network interactions
- ✅ Connects CEN and DMN
- ✅ Salience network plays key role

### Extensions Beyond Sridharan:
- 🔄 Bidirectional rFIC-DLPFC (not just one-way)
- 🌟 VMPFC as structural driver (new finding)
- 🎯 Clinical robustness (preserved in SZ)
- 📊 Quantitative edge frequencies (reproducible)

---

## 📂 All Your Enhanced Visualizations:

**GCM Results:**
- `gcm_roebroeck/11272025173313/enhanced_plots/`

**fMRI RASL Results:**
- `fbirn_results/11262025164900/combined/enhanced_plots/`
- `fbirn_results/11262025164900/group_0/enhanced_plots/`
- `fbirn_results/11262025164900/group_1/enhanced_plots/`
- `fbirn_results/11262025164900/group_comparison/`

**Total:** 60+ publication-quality figures! 🎨

---

**Both methods provide complementary insights into brain network causality! 🧠✨**

