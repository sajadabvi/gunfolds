# RASL FDR-Corrected Results vs. Two Decades of fMRI Literature: Agreement and Disagreement

## Overview

This report systematically compares the **28 FDR-corrected significant directed edges** identified by RASL on the FBIRN dataset (310 subjects, 10 NeuroMark ICA components) against the consensus findings from approximately 20 years of fMRI functional connectivity research in schizophrenia (SZ) vs. healthy controls (HC), as synthesised in the deep-research literature review.

A fundamental methodological difference shapes every comparison below: **RASL produces directed causal edges** (source → target, with group-level significance), whereas the literature is overwhelmingly based on **undirected functional connectivity** (correlation, partial correlation, ICA-FNC). Where RASL identifies A → B as SZ > HC, the literature may report "increased FC between A and B" without directional attribution. This asymmetry means that agreements are especially noteworthy (RASL's directed findings are consistent with undirected literature), while some apparent disagreements may reflect the added information content of directionality rather than genuine contradiction.

---

## 1. Points of Agreement

### 1.1 Thalamic Hyperconnectivity with Sensorimotor and Sensory Cortex

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| PoCG → Thalamus (SZ > HC, edge #11) | Increased thalamo–somatosensory/motor FC is the single most replicated finding in SZ connectivity research (Woodward 2012; Cheng 2015, P ≈ 10⁻¹⁸; Anticevic 2013; Damaraju 2014) |
| CalcarineG → Thalamus (SZ > HC, edge #2) | Thalamic hyperconnectivity with visual networks documented in both static and dynamic ICA analyses (Damaraju 2014) |

RASL's directed edges show cortex driving the thalamus; the literature reports the same regions as excessively coupled (albeit without direction). The convergence is strong: RASL finds the thalamus receiving excessive input from sensorimotor (PoCG) and visual (CalcarineG) cortices in SZ, matching the literature's "thalamo–sensory hyperconnectivity" signature.

### 1.2 Thalamic Gating Failure Model

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| Four sources converge on thalamus in SZ: CalcarineG, STG, PoCG, Caudate → Thalamus (edges #2, 6, 11, 5) | Thalamic gating failure is a central model: the thalamus is overwhelmed and cannot filter/relay information (Woodward 2012; Anticevic 2014) |

RASL provides a richer causal architecture for the gating failure model than correlational studies alone. The literature posits gating failure; RASL specifies **four directed input channels** (visual, auditory, somatosensory, basal ganglia) that each independently bombard the thalamus more in SZ. This is a direct confirmation with added mechanistic detail.

### 1.3 Auditory Cortex (STG) Dysfunction

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| STG → ACC (SZ > HC, #9), STG → PCC (SZ > HC, #10), STG → Thalamus (SZ > HC, #6), STG → PoCG (mixed, #7), STG → CalcarineG (SZ > HC, #8) | Auditory network abnormalities are robust across multiple levels: within-network hypoconnectivity, altered coupling with salience/self-referential systems, and fronto-temporal integration deficits (Li 2019; Dong 2018) |

Both RASL and the literature identify the auditory cortex as a major locus of dysconnectivity in SZ. RASL's finding that STG is the single most prolific source of aberrant outflow (5 targets) is consistent with the literature's emphasis on auditory network dysfunction as central to perceptual distortions and hallucinations in SZ.

### 1.4 DMN Within-Network Disruption

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| ACC → PCC (HC > SZ, #23): weakened intra-DMN coupling | Within-network DMN hypoconnectivity is among the most consistent meta-analytic findings, involving medial prefrontal/ACC and posteromedial/PCC hubs (Dong 2018; O'Neill 2018; Doucet 2020; Li 2019) |
| ACC → IPL (HC > SZ, #22), PCC → IPL (HC > SZ, #26): weakened DMN-to-CC coupling | DMN-to-frontoparietal/CC hypoconnectivity is documented, especially in chronic samples |

RASL's HC > SZ directed edges between DMN nodes (ACC, PCC) and between DMN and cognitive control (IPL) directly match the literature's replicated finding of within-DMN and DMN-to-executive hypoconnectivity in SZ.

### 1.5 Salience Network Switching Dysfunction

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| Insula → ACC (SZ > HC, #19), Insula → PCC (SZ > HC, #20) | Triple-network model (Menon 2011): salience network dysfunction disrupts switching between CEN and DMN. Literature reports altered salience-to-DMN/CEN coupling (O'Neill 2018; Dong 2018) |
| IPL → Insula (SZ > HC, #16): CC drives salience network excessively | Between-network dysconnectivity involving salience/ventral-attention with DMN and frontoparietal systems is emphasised in meta-analyses |

RASL provides the first **directed** evidence of salience-to-DMN aberrant coupling in this dataset. The Insula → {ACC, PCC} edges directly operationalise the triple-network model's prediction that the salience network's switching mechanism is "stuck on" in SZ. The literature has proposed this mechanism from undirected FC; RASL confirms the specific direction (salience → DMN, not the reverse).

### 1.6 Cerebellar Disconnection

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| Thalamus → CB (HC > SZ, #27): reduced thalamo-cerebellar outflow in SZ | Cerebellar involvement appears in multiple modalities: thalamocortical studies, task ICA, dynamic modularity, connectome stability deficits (Anticevic 2013; Kaufmann 2018; Gifford 2020) |
| ACC → CB (HC > SZ, #24): reduced DMN → cerebellar in SZ | Cognitive dysmetria hypothesis (Andreasen 1998) and CTC meta-analytic work (2025) emphasise integrated cerebello-thalamo-cortical circuitry disruption |
| CB → IPL (HC > SZ, #28): reduced cerebellar → CC in SZ | Reduced connectome fingerprint stability in cerebellar networks in SZ (Kaufmann 2018) |

RASL's three cerebellar edges paint a picture of progressive cerebellar disconnection: reduced input from the thalamus and DMN, and reduced output to cognitive control. This matches the literature's cerebellar dysmetria framework, CHR-conversion thalamo-cerebellar hypoconnectivity (Anticevic; Hedges g ≈ 0.88), and the expanded CTC circuit emphasis in recent meta-analytic work.

### 1.7 Striatal/Basal Ganglia Involvement

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| Caudate → Thalamus (SZ > HC, #5): excessive basal-ganglia-to-thalamic drive | Corticostriatal dysconnectivity is state-dependent and treatment-modulated (Sarpal 2015); striatal/putamen–sensory abnormalities appear in dynamic states (Damaraju 2014) |

Both sources implicate striatal-thalamic circuitry in SZ. RASL's directed edge (Caudate → Thalamus) adds to the thalamic convergence picture and is compatible with the literature's emphasis on cortico-striatal-thalamic loop dysfunction.

### 1.8 Cognitive Control Network (IPL) Abnormalities

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| IPL → ACC (SZ > HC, #17), IPL → PCC (SZ > HC, #18): CC drives DMN excessively | Frontoparietal control network abnormalities are consistent in network-level meta-analyses (Kim 2009; O'Neill 2018) |
| IPL → PoCG (SZ > HC, #15): CC drives sensorimotor excessively | Between-network dysconnectivity between executive and sensory systems documented |

RASL's finding that IPL (cognitive control) excessively drives DMN and sensorimotor targets in SZ aligns with the literature's documentation of frontoparietal-to-DMN boundary failures and impaired network segregation.

### 1.9 Reduced Network Segregation in SZ

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| 16 of 28 edges are SZ > HC (excessive cross-network driving) | Graph-theory meta-analyses show reduced local organisation (clustering, g ≈ −0.56) and reduced small-worldness (g ≈ −0.65) in SZ (Kambeitz 2016) |
| Extensive between-network edges in SZ (visual → subcortical, auditory → DMN, CC → salience, salience → DMN) | Early ICA-FNC work showed "less specialised / more entangled" large-scale coupling in SZ (Jafri 2008) |

RASL's pattern of widespread cross-network directed edges in SZ is the circuit-level manifestation of the literature's graph-theoretic finding that SZ shows reduced local segregation. The "entangled networks" interpretation from the literature is given specific directed architecture by RASL.

### 1.10 Sensorimotor–Auditory Coupling

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| STG → PoCG (mixed, #7), PoCG → STG (HC > SZ, #12) | Sensorimotor–auditory coupling abnormalities are documented; sensory–sensory hypoconnectivity in dynamic states (Damaraju 2014) |

RASL reveals an asymmetry: STG → PoCG is mixed but PoCG → STG is reduced in SZ, suggesting a directional imbalance in auditory-sensorimotor integration. The literature's "sensory–sensory hypoconnectivity" is refined by RASL into a specific directional deficit (sensorimotor-to-auditory feedback weakened).

---

## 2. Points of Disagreement or Tension

### 2.1 Direction of Thalamocortical Effects: Cortex → Thalamus vs. Thalamus → Cortex

| RASL FDR Finding | Literature Emphasis |
|------------------|---------------------|
| Sensory/motor cortex **→ Thalamus** (SZ > HC): cortical sources drive the thalamus | Literature frames the pattern as **thalamo–sensory** hyperconnectivity, implying thalamic output to sensory cortex is increased |

**Nature of tension:** The literature's most replicated signature is framed as "increased thalamo–sensorimotor FC" — implying the thalamus projects excessively to sensory cortices. RASL reverses the arrow: **cortex → thalamus**, with the thalamus as a convergence *target* rather than the source. This is not necessarily a contradiction — undirected FC cannot distinguish direction — but it challenges the common narrative interpretation. If RASL is correct, the "thalamic gating failure" model should be reframed: the problem is not that the thalamus over-broadcasts to sensory cortex, but that sensory cortex overwhelms the thalamus from multiple directions, impairing its relay/filter function.

**Assessment:** This is a **reframing** rather than a contradiction, but it carries significant theoretical implications for intervention targets.

### 2.2 Missing Thalamo-Prefrontal Hypoconnectivity

| RASL FDR Finding | Literature Consensus |
|------------------|---------------------|
| No significant edge between Thalamus and any prefrontal region | Reduced thalamo–prefrontal (especially mediodorsal thalamus–DLPFC/mPFC) connectivity is the other half of the canonical thalamocortical signature (Woodward 2012; Cheng 2015; Anticevic 2013) |

**Nature of tension:** The literature's thalamocortical model has two arms: (1) increased thalamo-sensory and (2) decreased thalamo-prefrontal. RASL captures arm (1) robustly but shows no significant edge for arm (2). This likely reflects the **10-component ICA parcellation**: the NeuroMark N=10 decomposition may not isolate a dedicated prefrontal/DLPFC component. ACC and PCC are included, but these are midline DMN nodes, not lateral prefrontal cortex. The absence of a DLPFC component means this canonical finding **cannot be tested** rather than being refuted.

**Assessment:** A **parcellation limitation** rather than a true disagreement. N=20 or N=53 component analyses might recover this signature.

### 2.3 Early-Course Prefrontal Hyperconnectivity

| RASL FDR Finding | Literature Evidence |
|------------------|---------------------|
| No evidence of prefrontal hyperconnectivity; DMN edges (ACC, PCC) predominantly show HC > SZ (weakened in SZ) | Early-course, unmedicated SZ shows robust prefrontal hyperconnectivity (d ≈ 0.84) with partial normalisation longitudinally (Anticevic 2015) |

**Nature of tension:** The FBIRN dataset likely includes predominantly chronic, medicated patients. The literature clearly documents that early-course, unmedicated patients can show the **opposite** pattern — prefrontal hyperconnectivity — which partially normalises with treatment. RASL's findings of weakened DMN connections are consistent with the chronic profile but would not be expected to generalise to early-course cohorts.

**Assessment:** An **illness-stage and medication confound** rather than a methodological disagreement. This is explicitly flagged in the literature as one of the clearest explanations for why "hyper-" and "hypo-connectivity" can both be true.

### 2.4 Salience Network: Hypoconnectivity (Literature) vs. Hyperconnectivity (RASL)

| RASL FDR Finding | Literature Meta-Analytic Results |
|------------------|---------------------|
| Insula → ACC (SZ > HC, #19), Insula → PCC (SZ > HC, #20): salience network **drives** DMN **more** in SZ | Meta-analyses emphasise **hypoconnectivity** within salience-related circuitry and between salience/ventral-attention and DMN/CEN (Dong 2018; O'Neill 2018) |

**Nature of tension:** This is a genuine interpretive conflict. The literature's meta-analyses consistently report salience-related **hypo**connectivity in SZ, while RASL finds the insula **excessively driving** DMN targets. Possible reconciliations: (a) undirected hypoconnectivity (reduced correlation) could coexist with increased directed causal influence if the coupling becomes less synchronous but more driving; (b) RASL captures between-network directed influence, while hypoconnectivity meta-analyses aggregate within-network cohesion and between-network correlation; (c) first-episode meta-analyses report salience hyperconnectivity to sensory regions, suggesting the direction of salience abnormality depends on the target network.

**Assessment:** A **genuine tension** that highlights the difference between correlational FC (lower synchrony = hypoconnectivity) and directed causal influence (more driving = hyperconnectivity). RASL's directed edges may be capturing an aspect of salience dysfunction invisible to undirected methods.

### 2.5 Visual Cortex as a Primary Pathological Broadcaster

| RASL FDR Finding | Literature Emphasis |
|------------------|---------------------|
| CalcarineG → {Caudate, Thalamus, PoCG, CB} (all SZ > HC): visual cortex is a pathological "broadcaster" with 4 outgoing edges | Visual network findings are present but secondary; the literature focuses on thalamo-visual hyperconnectivity (thalamus as the driver) and visual network stability deficits, not visual cortex as a causal source |

**Nature of tension:** RASL elevates the visual cortex (CalcarineG) to one of the most dysregulated sources in SZ, with 4 significant outgoing edges. The literature acknowledges visual network involvement but does not emphasise it as a primary driver. This may be a novel RASL finding, or it may be an artefact of the ICA decomposition (a single CalcarineG component absorbing variance from multiple visual-related processes).

**Assessment:** A **potential novel finding** that goes beyond the current literature. If replicated, it would extend the sensory broadcasting model beyond STG to include visual cortex as a second major source of aberrant outflow.

### 2.6 Auditory Network: Hypoconnectivity (Literature) vs. Excessive Outflow (RASL)

| RASL FDR Finding | Literature Meta-Analytic Results |
|------------------|---------------------|
| STG shows 5 outgoing edges (SZ > HC for 4 of them): auditory cortex is the most prolific aberrant source | Meta-analyses report **hypoconnectivity within the auditory network** (Li 2019; Dong 2018) |

**Nature of tension:** Similar to the salience network tension (§2.4). The literature's "auditory hypoconnectivity" refers to reduced within-network cohesion (e.g., reduced STG–STG or STG–insula synchrony). RASL's finding of excessive STG outflow to **other networks** (thalamus, DMN, visual, sensorimotor) is a between-network phenomenon. These can coexist: a network can lose internal coherence (within-network hypoconnectivity) while simultaneously sending aberrant signals to other networks (between-network directed hyperconnectivity).

**Assessment:** **Compatible rather than contradictory** when within- vs. between-network levels are distinguished, but the framing differs substantially. RASL's emphasis on STG as a source of excessive between-network causal influence is not a standard interpretation in the literature.

### 2.7 Hippocampal Findings Absent from RASL

| RASL FDR Finding | Literature Evidence |
|------------------|---------------------|
| No hippocampal component in the N=10 ICA parcellation; no hippocampal edges | Hippocampal FC abnormalities are well-supported, especially with subregion analyses, medication-naïve designs, and treatment-response prediction (longitudinal studies) |

**Nature of tension:** Like the prefrontal gap (§2.2), this is a parcellation limitation. The N=10 NeuroMark decomposition does not include a hippocampal component, so RASL cannot address a substantial body of SZ FC literature.

**Assessment:** A **parcellation gap**, not a disagreement. Hippocampal findings are beyond the reach of the current analysis.

### 2.8 Dynamic State Dependence Not Captured

| RASL FDR Finding | Literature Emphasis |
|------------------|---------------------|
| Static group-level edge counts (28 significant edges across all subjects) | Dynamic FC reveals state-dependent abnormalities: some subcortical effects appear only in specific states; SZ spends less time in strongly integrated states (Damaraju 2014; Kottaram 2019; Gifford 2020) |

**Nature of tension:** RASL's analysis aggregates across the entire time series and all subjects. The literature strongly emphasises that some SZ connectivity abnormalities are **state-dependent** — visible only in certain dynamic states and obscured by static averaging. RASL's 28 edges may therefore be a conservative subset: additional edges may be significant in specific dynamic states but wash out in the aggregate.

**Assessment:** A **methodological limitation** of the current RASL analysis framework, not a disagreement with the findings themselves. Future RASL analyses incorporating dynamic windowing could potentially reveal additional state-dependent edges.

### 2.9 Cerebellar–DMN Coupling Direction

| RASL FDR Finding | Literature Finding |
|------------------|---------------------|
| ACC → CB (HC > SZ, #24): DMN-to-cerebellar connectivity is **reduced** in SZ | A seed-based study reported **increased** cerebellar–DMN connectivity in first-episode drug-naïve SZ |

**Nature of tension:** RASL shows the DMN (ACC) sends less to the cerebellum in SZ, while at least one study reports increased cerebellar–DMN FC in first-episode patients. Possible reconciliation: (a) illness stage — first-episode patients may show compensatory cerebellar–DMN hyperconnectivity that reverses in chronic illness; (b) direction — the literature's undirected "increased cerebellar–DMN FC" could be cerebellum → DMN (increased) co-occurring with DMN → cerebellum (decreased), which is consistent with RASL's directed finding.

**Assessment:** Likely an **illness-stage effect**, but directionality differences cannot be ruled out.

### 2.10 Global Signal and Motion Confounds

| RASL FDR Analysis | Literature Warnings |
|------------------|---------------------|
| Edge-count based analysis on undersampled causal graphs; no explicit discussion of global signal regression or motion artifact handling | GSR can attenuate clinically meaningful global variance differences in SZ; motion produces systematic distance-dependent artifacts; these are flagged as major determinants of FC results (Yang 2014; Power 2012) |

**Nature of tension:** The literature emphasises that preprocessing choices (especially GSR and motion handling) can qualitatively shift group-level FC conclusions. The RASL FDR report does not discuss whether or how these confounds were addressed in the upstream ICA decomposition or subject-level preprocessing. If the FBIRN preprocessing did not adequately control motion, some of RASL's "excessive connectivity" edges could partially reflect motion artifacts (motion tends to inflate short-range and reduce long-range FC).

**Assessment:** A **methodological caveat** that applies to any FBIRN-derived analysis. Not a disagreement with RASL's findings per se, but the literature's emphasis on these confounds means RASL's results should be interpreted with awareness of the preprocessing pipeline.

---

## 3. Summary Table

| # | Domain | RASL FDR Finding | Literature Position | Verdict |
|---|--------|-----------------|---------------------|---------|
| 1 | Thalamo-sensory/motor coupling | Sensory/motor cortex → Thalamus (SZ > HC) | Thalamo-sensory hyperconnectivity (SZ > HC) | **Agreement** (same regions; RASL adds direction) |
| 2 | Thalamic gating failure | 4 cortical/subcortical sources converge on thalamus | Thalamic gating failure model well-established | **Agreement** (RASL enriches the model) |
| 3 | STG/auditory dysfunction | STG is most prolific aberrant source (5 targets) | Auditory network robustly implicated | **Agreement** (magnitude and centrality align) |
| 4 | Intra-DMN hypoconnectivity | ACC → PCC weakened in SZ | Meta-analytic consensus on DMN hypoconnectivity | **Agreement** |
| 5 | Salience switching dysfunction | Insula → ACC, PCC (SZ > HC) | Triple-network model predicts salience dysfunction | **Agreement** (RASL provides directed mechanism) |
| 6 | Cerebellar disconnection | 3 cerebellar edges all HC > SZ | CTC circuit disruption documented | **Agreement** |
| 7 | Striatal-thalamic loops | Caudate → Thalamus (SZ > HC) | Corticostriatal dysconnectivity documented | **Agreement** |
| 8 | Frontoparietal CC abnormalities | IPL drives DMN and sensorimotor excessively in SZ | FPN abnormalities in meta-analyses | **Agreement** |
| 9 | Reduced network segregation | 16 cross-network SZ > HC edges | Reduced small-worldness and clustering in SZ | **Agreement** |
| 10 | Thalamocortical arrow direction | Cortex → Thalamus | Framed as thalamus → cortex | **Tension** (reframing, not contradiction) |
| 11 | Thalamo-prefrontal hypoconnectivity | Not tested (no PFC component) | Most replicated finding | **Gap** (parcellation limitation) |
| 12 | Early-course hyperconnectivity | Not observed (likely chronic sample) | Robust in early-course unmedicated SZ | **Stage-dependent difference** |
| 13 | Salience hypo- vs. hyperconnectivity | Insula drives DMN more (SZ > HC) | Salience hypoconnectivity in meta-analyses | **Genuine tension** |
| 14 | Visual cortex as broadcaster | CalcarineG has 4 outgoing SZ > HC edges | Visual cortex not emphasised as a driver | **Potential novel finding** |
| 15 | Auditory within-network hypo | STG shows excessive between-network outflow | Within-auditory hypoconnectivity | **Different levels** (within vs. between) |
| 16 | Hippocampal findings | Not testable (no component) | Well-supported | **Gap** (parcellation limitation) |
| 17 | Dynamic state dependence | Static analysis only | State-dependent effects emphasised | **Methodological limitation** |
| 18 | Cerebellar–DMN coupling | Reduced DMN → CB in SZ | Increased cerebellar–DMN FC in FEP | **Stage-dependent tension** |
| 19 | GSR/motion confounds | Not explicitly addressed | Major determinants of FC results | **Methodological caveat** |

---

## 4. Conclusions

### Agreements Outweigh Disagreements

Of the 19 comparison points, **9 show clear agreement**, **3 represent parcellation gaps** (not disagreements), **3 reflect illness-stage or methodological limitations**, and only **4 represent genuine tensions** or novel findings. The overall picture is one of strong convergence: RASL's FDR-corrected directed causal edges recover the major circuit-level findings from 20 years of undirected FC research — thalamic gating failure, DMN hypoconnectivity, auditory cortex dysfunction, cerebellar disconnection, and salience switching dysfunction — while adding directional specificity that the correlational literature cannot provide.

### RASL's Unique Contributions Beyond the Literature

1. **Directional architecture of thalamic gating failure.** Rather than a single "increased thalamo-sensory FC" label, RASL identifies four distinct cortical/subcortical sources that each independently bombard the thalamus, specifying the causal direction as cortex → thalamus.

2. **Directed salience-to-DMN mechanism.** The Insula → {ACC, PCC} edges provide the first directed evidence for the triple-network model's switching dysfunction in this dataset.

3. **Visual cortex as a pathological broadcaster.** CalcarineG's 4 outgoing SZ > HC edges are not prominently discussed in the literature and may represent a novel finding.

4. **STG's between-network directed hub role.** While auditory dysfunction is well-documented, RASL's finding that STG is the single most prolific source of between-network aberrant causal influence (5 targets) goes beyond the literature's within-network hypoconnectivity framing.

5. **Cerebellar disconnection with full directional circuit.** The three cerebellar edges specify which directions are weakened (thalamus → CB, ACC → CB, CB → IPL), giving directed architecture to the cognitive dysmetria hypothesis.

### Key Tensions Requiring Further Investigation

1. **Salience hypo- vs. hyperconnectivity** (§2.4) may be resolved by distinguishing within-network coherence (correlational, likely reduced) from between-network directed influence (causal, possibly increased). This distinction could be tested with future analyses that separate intra- and inter-network RASL edges.

2. **Early-course hyperconnectivity** (§2.3) requires applying RASL to first-episode, medication-naïve cohorts to determine whether the directed edge pattern reverses.

3. **Visual cortex broadcaster** (§2.5) requires replication in independent datasets and at higher ICA dimensionalities to confirm this is not a parcellation artefact.

---

## References

- Anticevic, A., et al. (2013). Thalamic overconnectivity with sensory-motor cortex and underconnectivity with prefrontal–striatal–cerebellar regions. *NeuroImage*, etc.
- Anticevic, A., et al. (2014). Characterizing thalamo-cortical disturbances in schizophrenia and bipolar illness. *Cerebral Cortex*, 24(12), 3116–3130.
- Andreasen, N. C., et al. (1998). "Cognitive dysmetria" as an integrative theory of schizophrenia. *Schizophrenia Bulletin*, 24(2), 203–218.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *JRSS-B*, 57(1), 289–300.
- Cheng, W., et al. (2015). Brain-wide association study pooling 415 SZ / 405 HC. P ≈ 10⁻¹⁸ for thalamo-somatosensory.
- Damaraju, E., et al. (2014). Static and dynamic FNC: thalamus–sensory hyperconnectivity; state-dependent subcortical effects.
- Dong, D., et al. (2018). MKDA meta-analysis: within- and between-network dysconnectivity mapping.
- Doucet, G. E., et al. (2020). ALE meta-analysis of intra-DMN cohesion.
- Du, Y., et al. (2020). NeuroMark: An automated and adaptive ICA based pipeline. *NeuroImage: Clinical*, 28, 102375.
- Gifford, G., et al. (2020). Multilayer community detection; flexibility in cerebellar/subcortical/frontoparietal.
- Kambeitz, J., et al. (2016). Graph-theory meta-analysis: reduced local organization (g ≈ −0.56) and small-worldness (g ≈ −0.65).
- Kaufmann, T., et al. (2018). Connectome fingerprint stability: reduced in SZ across subnetworks.
- Kottaram, A., et al. (2019). HMM on RSNs: reduced DMN/executive occupancy in SZ.
- Li, S., et al. (2019). Meta-analysis (2,588 SZ / 2,567 HC): hypoconnectivity within auditory, CC, DMN, and somatomotor.
- Menon, V. (2011). Large-scale brain networks and psychopathology: a unifying triple network model. *Trends in Cognitive Sciences*, 15(10), 483–506.
- O'Neill, A., et al. (2018). FEP meta-analysis: DMN/SN/CEN seed-based differences.
- Power, J. D., et al. (2012). Motion artifact characterization in rsFC.
- Sarpal, D. K., et al. (2015). Longitudinal striatal FC and treatment response.
- Whitfield-Gabrieli, S., et al. (2009). Hyperactivity and hyperconnectivity of the default network in schizophrenia. *PNAS*, 106(4), 1279–1284.
- Woodward, N. D., et al. (2012). Thalamocortical dysconnectivity in schizophrenia. *AJP*, 169(10), 1092–1099.
- Yang, G. J., et al. (2014). Global signal power/variance in SZ; GSR impact on rGBC.
