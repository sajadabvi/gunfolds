# RASL Results vs. 20-Year fMRI Literature: Agreements and Disagreements

This report systematically compares the findings in
**rasl_vs_pcmci_results.md** (our RASL N10_domain_RASL analysis on FBIRN, 310 subjects)
against the comprehensive literature review in
**deep-research-healthy_vs_schz.md** (covering ~2006--2026 fMRI connectivity studies in schizophrenia).

---

## Part 1: Points of Agreement

### 1.1 Thalamocortical Dysconnectivity -- The Central Finding

| RASL result | Literature consensus |
|-------------|---------------------|
| CalcarineG &rarr; Thalamus significantly more frequent in SZ (Circuit 1) | "Thalamus has emerged as a cross-method and cross-sample connectivity hub in SZ, with unusually consistent directionality: thalamo-prefrontal weakening and thalamo-sensory strengthening" |

This is the strongest point of convergence. The literature identifies the thalamo-sensory hyperconnectivity motif as the single most replicated finding across 20 years of SZ fMRI research, supported by Woodward (2012, N=139), Anticevic (2013, N=180), Cheng (2015, N=820), Damaraju (2014, N=314), and the clinical high-risk conversion study (Anticevic, N=397). RASL recovers this signature with CalcarineG &rarr; Thalamus as one of its 15 significant edges. Notably, the CalcarineG &rarr; Thalamus edge is also the **only** significant edge found by PCMCI, underscoring how robust this signal is -- even a method that does not model undersampling can barely detect it.

**Strength of agreement:** Very strong. The literature calls this a "benchmark validity target" for any connectivity pipeline.

### 1.2 Visual-to-Subcortical Hyperconnectivity in SZ

| RASL result | Literature consensus |
|-------------|---------------------|
| CalcarineG &rarr; Caudate (SZ > HC, significant) | "Thalamic overconnectivity with auditory, motor, and **visual** networks" (Damaraju 2014) |
| CalcarineG &rarr; PoCG (SZ > HC, significant) | "Increased thalamus--primary somatosensory cortex connectivity (P &asymp; 10^-18)" (Cheng 2015) |

RASL extends the thalamocortical finding to include a Caudate (basal ganglia) target and a sensorimotor target, both driven by visual cortex. The literature supports this broader subcortical-sensory hyperconnectivity pattern. Cheng (2015) found the thalamus--somatosensory link as the single most significant aberration in an 820-subject BWAS, and Damaraju (2014) documented thalamic hyperconnectivity with visual networks specifically. RASL confirms that the visual cortex is a key source node in this circuit.

**Strength of agreement:** Strong. RASL adds directional specificity (see Section 2.1 for the nuance).

### 1.3 Default Mode Network Hypoconnectivity

| RASL result | Literature consensus |
|-------------|---------------------|
| ACC &rarr; PCC reduced in SZ (HC > SZ, significant) | "Within-network DMN hypoconnectivity is common in both chronic and early-psychosis samples, often involving medial prefrontal/anterior cingulate and posteromedial hubs" |
| ACC &rarr; IPL reduced in SZ (HC > SZ, significant) | "Hub-level DMN dysconnectivity in anteromedial/posteromedial cortex" (Doucet 2020, ALE meta-analysis of 70 rs-fMRI studies) |
| PCC &rarr; IPL reduced in SZ (HC > SZ, significant) | "DMN subregion differences... relatively greater posterior cingulate/precuneus effects in HC" (Garrity 2007) |

RASL finds three edges constituting reduced *outgoing* influence from the DMN (ACC and PCC) in SZ. This maps precisely onto the literature's most consistent cortex-level finding: DMN within-network hypoconnectivity, particularly involving the anteromedial (ACC) and posteromedial (PCC) hubs. The Doucet (2020) transdiagnostic ALE meta-analysis (70 studies, 2789 patients, 3002 HC) specifically identified these two hubs as the locus of schizophrenia-specific DMN dysconnectivity.

RASL adds directional information: the *outflow* from ACC and PCC to other regions is specifically weakened, suggesting that these DMN hubs fail to coordinate downstream targets in SZ, rather than simply having reduced co-activation.

**Strength of agreement:** Very strong. Both identify ACC and PCC as the critical DMN nodes affected in SZ.

### 1.4 Task-Positive / Task-Negative Network Segregation Failure

| RASL result | Literature consensus |
|-------------|---------------------|
| IPL &rarr; Insula (SZ > HC, significant) | "SZ involves hypoconnectivity within salience-related circuitry... and between salience/ventral attention and DMN/frontoparietal systems" |
| IPL &rarr; ACC (SZ > HC, significant) | "DMN component showed more consistent connectivity with other components in SZ" (Jafri 2008) |
| IPL &rarr; PCC (SZ > HC, significant) | "Impaired cognition reflects not only reduced executive recruitment but also abnormal interaction between task-positive and task-negative systems" |

RASL finds that the Inferior Parietal Lobule (a cognitive control / task-positive hub) exerts *more* causal influence on default-mode and salience nodes in SZ. The literature describes this as a failure of task-positive / task-negative network segregation. Jafri (2008) found that the DMN showed "more consistent connectivity with other components in SZ," supporting an early "less specialised / more entangled" interpretation that RASL confirms with directional, edge-level specificity.

The convergence is notable: where the literature sees blurred network boundaries, RASL pinpoints the IPL as a specific source of aberrant cross-network influence.

**Strength of agreement:** Strong. The mechanism (network segregation failure) matches; RASL identifies the specific directional pathway.

### 1.5 Auditory Network Abnormalities and Hallucination Relevance

| RASL result | Literature consensus |
|-------------|---------------------|
| STG &rarr; ACC (SZ > HC, significant) | "Hypoconnectivity within the auditory network... altered coupling between auditory components and salience/self-referential systems" (Li 2019, meta-analysis, 2588 SZ / 2567 HC) |
| STG &rarr; PCC (SZ > HC, significant) | "Modulating speech-perception--speech-production coupling can alter symptom expression" (STG neurofeedback study) |
| STG &rarr; PoCG (mixed, significant) | Fronto-temporal dysconnectivity involving sensorimotor regions documented in NBS work |

RASL finds increased causal outflow from the Superior Temporal Gyrus (auditory) to DMN regions in SZ. The literature frames auditory network abnormalities as one of the most robust connectivity signatures, with Li (2019) meta-analytically confirming hypoconnectivity *within* auditory networks and altered *between*-network coupling. The literature also specifically links STG dysconnectivity to auditory hallucinations, noting that modulating STG--IFG coupling affects hallucination symptoms.

RASL's finding of *increased* STG &rarr; DMN influence in SZ is particularly interesting in the context of hallucinations: it suggests a causal mechanism by which auditory processing intrudes on self-referential processing, potentially contributing to the misattribution of internal speech as external.

**Strength of agreement:** Strong. Both identify STG as a key node with altered between-network coupling in SZ.

### 1.6 Cerebellar Dysconnectivity

| RASL result | Literature consensus |
|-------------|---------------------|
| ACC &rarr; CB reduced in SZ (HC > SZ, significant) | "Cerebellar involvement appears at multiple levels: as a task-positive ICA network differing in SZ, as a stability deficit, as part of dynamic modular abnormalities, and as a locus of DMN coupling differences" |

RASL finds that the ACC &rarr; CB edge is weaker in SZ, indicating reduced default-mode outflow to the cerebellum. The literature supports cerebellar involvement through Andreasen's (1998) "cognitive dysmetria" hypothesis, through the NeuroMark study (Du et al., 2020) showing reduced subcortical-cerebellar coupling, and through dynamic modularity work showing increased thalamus flexibility involving cerebellar communities (Gifford 2020). The literature review explicitly notes a 2025 meta-analytic effort expanding emphasis on integrated cerebello-thalamo-cortical (CTC) circuitry, confirming the continuing relevance of this finding.

**Strength of agreement:** Moderate to strong. The general cerebellar involvement is well-supported; the specific ACC &rarr; CB edge adds novel directional detail.

### 1.7 Insula Involvement in Dysconnectivity

| RASL result | Literature consensus |
|-------------|---------------------|
| IPL &rarr; Insula (SZ > HC, significant) | "SZ involves hypoconnectivity within salience-related circuitry (e.g., involving anterior cingulate and putamen/insula nodes)" |

Both identify the Insula as a key node in SZ dysconnectivity. The literature frames the Insula as part of the salience / ventral-attention network and reports within-network hypoconnectivity. RASL finds *increased* IPL &rarr; Insula influence in SZ, which represents aberrant between-network (CC &rarr; Salience) connectivity rather than within-network effects. This is not contradictory (within-salience hypo can coexist with cross-network hyper) but reflects a different level of analysis.

**Strength of agreement:** Moderate. The node is the same; the network-level framing differs but is compatible.

---

## Part 2: Points of Disagreement or Tension

### 2.1 Direction of Causal Influence in Thalamocortical Circuits

| RASL result | Literature framing |
|-------------|-------------------|
| CalcarineG &rarr; Thalamus (cortex drives thalamus, SZ > HC) | "Thalamo-sensory hyperconnectivity" -- typically framed as thalamus driving sensory cortex |

This is the most consequential tension between the two documents. The literature consistently frames the thalamocortical finding as the **thalamus** excessively driving sensory/sensorimotor cortex -- a "gating failure" where the thalamic relay floods cortex with unfiltered sensory input. Woodward (2012) used cortical seeds projected to thalamic targets, and Cheng (2015) identified "thalamus--primary somatosensory cortex" as the hub link, both implying thalamus as the source of pathology.

RASL finds the **reverse direction**: it is the visual cortex (CalcarineG) that excessively drives the Thalamus and Caudate in SZ. This suggests a *cortico-fugal* model where sensory cortex sends excessive top-down feedback to subcortical structures, rather than a thalamic gating failure per se.

**Implications:** This disagreement may not be a true contradiction. The literature relies on undirected FC (correlation-based), which cannot distinguish cortex &rarr; thalamus from thalamus &rarr; cortex. What the literature calls "thalamo-sensory hyperconnectivity" is actually "increased covariance between thalamus and sensory cortex" -- agnostic to direction. RASL, by modelling directed causal graphs and undersampling, can resolve this ambiguity. If RASL's directional finding is correct, it would significantly reframe the thalamocortical dysconnectivity model: the problem may originate in cortex (excessive sensory cortical output) rather than in thalamus (failed gating). This is a **novel and potentially important contribution** that the correlation-based literature cannot address.

**Severity of disagreement:** Low as a factual conflict (undirected FC is consistent with either direction), but high in interpretive significance (the causal model of thalamocortical pathology would be reframed).

### 2.2 Missing Prefrontal Cortex -- The Other Half of the Thalamocortical Signature

| RASL result | Literature consensus |
|-------------|---------------------|
| No prefrontal component in the 10-node model | "Reduced prefrontal--thalamic connectivity (mediodorsal/anterior thalamic nuclei targets)" is one of the two pillars of the thalamocortical model |

The literature's thalamocortical signature is explicitly **bidirectional**: reduced thalamo-prefrontal coupling *alongside* increased thalamo-sensory coupling. RASL's 10-component model includes no prefrontal cortex (no DLPFC, mPFC, or SFG). The ACC serves as the nearest proxy, and RASL does find reduced ACC outgoing connections in SZ, but this is framed as DMN hypoconnectivity rather than thalamo-prefrontal dysconnectivity.

**Implication:** RASL can confirm the sensory/subcortical half of the thalamocortical motif but is structurally unable to test the prefrontal half. This is a limitation of the 10-component resolution, not of the RASL method itself. Future analyses at N=20 or N=53 (which include prefrontal components) could address this.

**Severity of disagreement:** This is an absence rather than a contradiction. It represents an incomplete test of the literature's consensus model.

### 2.3 Graph Density: Redistribution vs. Reduction

| RASL result | Literature tendency |
|-------------|-------------------|
| HC and SZ have nearly identical overall graph density (diff = 0.007) | "Widespread hypoconnectivity" in chronic SZ across multiple networks suggests globally sparser graphs |

RASL finds that HC and SZ graphs have essentially the same density -- they differ in *which* edges are present, not *how many*. The literature, particularly the large meta-analyses (Li 2019; Dong 2018), emphasises "diffuse hypoconnectivity" -- reduced connectivity across auditory, cognitive-control, DMN, self-referential, and somatomotor networks. This framing implies that SZ graphs should be globally sparser.

**Possible reconciliation:** The literature's "hypoconnectivity" is measured as reduced correlation *strength*, while RASL's density is based on binary edge presence (is there a directed causal influence or not?). A weaker-but-still-present edge registers as hypoconnectivity in correlation analyses but still counts as "present" in RASL's binary framework. Additionally, RASL simultaneously uncovers new SZ > HC edges (e.g., CalcarineG outflow, IPL outflow, STG outflow) that offset the HC > SZ edges, yielding similar total density but different topology. This reframes the SZ connectivity profile as a **rewiring** rather than a **depletion** -- a potentially more nuanced model.

**Severity of disagreement:** Moderate. The conceptual framing differs (rewiring vs. depletion), but the underlying data may be compatible once the difference between binary causal edges and continuous correlation strength is accounted for.

### 2.4 Salience Network -- Hypo vs. Hyper

| RASL result | Literature consensus |
|-------------|---------------------|
| IPL &rarr; Insula: SZ > HC (more cognitive-control input to insula in SZ) | "SZ involves hypoconnectivity within salience-related circuitry (e.g., involving anterior cingulate and putamen/insula nodes)" |

The literature meta-analytically finds salience-network *hypo*connectivity in SZ -- both within the salience network and between salience and other networks. RASL finds *increased* causal influence from IPL to Insula in SZ. These point in opposite directions.

**Possible reconciliation:** The literature's within-salience hypoconnectivity refers to reduced coupling *among* salience-network nodes (e.g., insula--ACC covariance). RASL's finding is a *between-network* effect: cognitive control (IPL) driving salience (Insula). These can coexist -- the insula may receive more *external* (cross-network) input while having less *internal* (within-network) coherence. In fact, this pattern would be consistent with the "less specialised / more entangled" model from Jafri (2008): reduced within-network cohesion plus increased between-network coupling.

However, the first-episode psychosis meta-analysis also reports salience *hypoconnectivity with DMN/CEN*, which is the opposite of RASL's IPL &rarr; Insula hyperconnectivity. This specific tension is harder to reconcile and may reflect differences between first-episode and chronic samples (FBIRN likely contains mostly chronic patients).

**Severity of disagreement:** Moderate. Within-network vs. between-network distinctions partially resolve it, but the between-network direction still differs from the FEP meta-analysis.

### 2.5 Hippocampal and Striatal Circuits -- Absent from RASL

| RASL result | Literature consensus |
|-------------|---------------------|
| Not testable (no hippocampal or striatal component) | "Hippocampal FC abnormalities are supported by medication-naive and longitudinal designs"; "Symptom improvement associated with increased striatal FC to ACC/DLPFC/hippocampus/insula" |

The literature highlights hippocampal subregion dysconnectivity (linked to treatment response) and corticostriatal dysconnectivity (linked to symptom improvement under treatment) as important circuits. RASL's 10-component model includes Caudate (basal ganglia/striatal) but not hippocampus. The Caudate finding (CalcarineG &rarr; Caudate, SZ > HC) partially addresses the striatal dimension but does not capture the specific cortico-striatal treatment-response circuit.

**Severity of disagreement:** Absence, not contradiction. A limitation of spatial resolution.

### 2.6 Dynamic Connectivity -- Untested by RASL

| RASL result | Literature consensus |
|-------------|---------------------|
| Static group comparison only | "A central lesson of dFC in SZ is that mean/static FC can obscure abnormalities that are state dependent" |

The literature strongly emphasises that dynamic functional connectivity reveals group differences hidden in static averages. RASL's current analysis is a static group comparison: it pools all solutions across subjects and tests group differences in edge frequency. It does not analyse temporal dynamics, state occupancy, or switching behaviour.

**Implication:** Some of RASL's 15 significant edges may be more prominent in specific dynamic states, and additional edges might emerge from a state-stratified analysis. The literature warns that static analyses systematically undercount abnormalities.

**Severity of disagreement:** Methodological gap rather than contradiction. RASL's static analysis is conservative, potentially missing state-dependent effects.

### 2.7 Undersampling Rate as a Biomarker -- Novel, Unvalidated

| RASL result | Literature consensus |
|-------------|---------------------|
| HC and SZ differ in effective causal timescale (*p* < 0.001) | No literature precedent |

RASL's finding that HC and SZ have significantly different undersampling rates has **no parallel** in the fMRI connectivity literature. The literature does not model undersampling and therefore cannot speak to this. This is a genuinely novel finding, but it is also completely unvalidated by external evidence. It could reflect (a) a true difference in the temporal scale of causal dynamics, (b) an artefact of differential motion or signal quality affecting RASL's optimisation, or (c) a confound from medication effects on hemodynamic timing.

**Severity of disagreement:** Not a disagreement per se, but an unvalidated claim that the literature cannot corroborate or refute.

---

## Summary Table

| Topic | Agreement / Disagreement | Severity | Notes |
|-------|------------------------|----------|-------|
| Thalamocortical hyperconnectivity (sensory--subcortical) | **Agree** | -- | Strongest convergence |
| DMN hypoconnectivity (ACC, PCC hubs) | **Agree** | -- | Matches meta-analytic consensus |
| Task-positive / task-negative segregation failure | **Agree** | -- | RASL adds directional detail via IPL |
| Auditory--DMN aberrant coupling | **Agree** | -- | Hallucination-relevant |
| Cerebellar dysconnectivity | **Agree** | -- | Consistent with CTC model |
| Insula involvement | **Agree** (partially) | -- | Same node, different network level |
| **Direction of thalamocortical influence** | **Tension** | High interpretive significance | RASL: cortex &rarr; thalamus; Literature: thalamus &rarr; cortex (but undirected) |
| **Prefrontal component absent** | **Absence** | Moderate | Cannot test thalamo-PFC hypoconnectivity |
| **Density: rewiring vs. depletion** | **Disagree** (framing) | Moderate | Binary edges vs. continuous correlation |
| **Salience network direction** | **Disagree** (partial) | Moderate | Within-network hypo vs. between-network hyper |
| **Hippocampal / striatal circuits** | **Absence** | Low | Resolution limitation |
| **Dynamic connectivity** | **Untested** | Moderate | Static analysis only |
| **Undersampling rate biomarker** | **Novel / unvalidated** | Low | No literature precedent |

---

## Conclusions

**The RASL results on 10 NeuroMark components recover the major signatures of the 20-year fMRI literature on schizophrenia**, including thalamocortical dysconnectivity, DMN hypoconnectivity, task-positive/task-negative segregation failure, auditory-network aberrations, and cerebellar involvement. This convergence is remarkable given that RASL operates on directed, binary causal graphs derived from a 10-node parcellation, while the literature relies on undirected, continuous functional connectivity across diverse parcellation scales and analytic methods.

**The most important tension is directional.** Where the literature assumes (without evidence of direction) that the thalamus drives sensory cortex excessively in SZ, RASL finds the reverse: sensory cortex drives the thalamus. If confirmed, this would meaningfully reframe the causal model of thalamocortical pathology in schizophrenia -- from a subcortical gating failure to a cortical feedback excess. This directional resolution is only possible because RASL models causal structure with undersampling correction, a capability that correlation-based methods lack.

**The primary gaps** are spatial (no prefrontal or hippocampal components at N=10) and temporal (no dynamic-state analysis). Both can be addressed in future work with higher-resolution component sets and dynamic extensions of the RASL framework.
