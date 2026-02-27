# Results Section: Method Validation and Comparative Analysis

## Validation of Causal Discovery Methods on fMRI Data

---

### GCM Analysis Validates Against Published Findings

To establish the validity of our causal discovery framework, we first implemented Granger Causality Mapping (GCM) following the methodology of Roebroeck et al. (2005) and applied it to the same ICA components analyzed in Sridharan et al. (2008). We analyzed resting-state fMRI data from 310 subjects in the fBIRN dataset, focusing on six key brain regions: right posterior parietal cortex (rPPC), right fronto-insular cortex (rFIC), right dorsolateral prefrontal cortex (rDLPFC), anterior cingulate cortex (ACC), posterior cingulate cortex (PCC), and ventromedial prefrontal cortex (VMPFC). These regions span three major functional networks: the Central Executive Network (CEN: rPPC, rDLPFC), the Salience Network (SN: rFIC, ACC), and the Default Mode Network (DMN: PCC, VMPFC).

Our GCM analysis (Figure X) revealed a pattern of directed influences that showed strong agreement with Sridharan et al.'s key findings. First, we confirmed the central role of rFIC in network interactions, with rFIC showing significant Granger-causal relationships to regions in both the CEN and DMN. Specifically, we observed robust bidirectional coupling between rFIC and rDLPFC (58% of subjects), the strongest connection in our GCM analysis. Second, consistent with Sridharan et al.'s emphasis on rFIC's connectivity to both CEN and DMN, we found that rFIC exhibited directed influences to VMPFC (34% of subjects) and PCC (30% of subjects) within the DMN. Third, the ACC, which Sridharan et al. identified as a key partner with rFIC in the salience network, showed directed influences to DMN regions (ACC → PCC: 37%, ACC → VMPFC: 36%), supporting the salience network's role in regulating default mode activity.

However, our large-sample GCM analysis (N=310) revealed an important extension to the original findings: the rFIC-rDLPFC connection was equally strong in both directions (rFIC → rDLPFC: 58%, rDLPFC → rFIC: 58%), suggesting reciprocal temporal coupling rather than purely unidirectional control. This bidirectional pattern may reflect the larger sample size and resting-state data, revealing sustained interactive processing that complements the event-driven, unidirectional influences emphasized in task-based paradigms.

### RASL Reveals Complementary Structural Causal Architecture  

Having validated our implementation with GCM, we next applied our novel Restricted Anytime Structure Learner (RASL) method to the same ICA components and subject population. Unlike GCM, which identifies temporal precedence relationships through time-series analysis, RASL employs constraint-based causal structure learning to infer the underlying directed acyclic graph that best explains the observed statistical dependencies. This complementary approach can reveal structural causal relationships that may not be apparent from temporal analyses alone.

The RASL analysis (Figure Y) identified a strikingly different but theoretically coherent pattern of causal relationships. The strongest and most consistent structural connection was VMPFC → rFIC (89.7% of candidate solutions), representing a causal pathway from the DMN hub to the salience network. This finding was remarkably robust, appearing in nearly 90% of the 2,496 causal graphs identified across all subjects (310 subjects × ~8 solutions per subject). Additional strong structural connections included rDLPFC → ACC (67.3%), PCC → rFIC (64.1%), and VMPFC → PCC (62.6%), suggesting that VMPFC acts as a central driver with directed influences to multiple target regions across different functional networks.

### Methodological Complementarity: Temporal vs. Structural Causality

Direct comparison of GCM and RASL results revealed both important similarities and a key mechanistic difference (Figure Z). Both methods converged on several fundamental observations: (1) all three functional networks (CEN, SN, DMN) showed high interconnectivity with directed influences between networks; (2) the salience network (rFIC, ACC) occupied a central position with connections to both CEN and DMN, consistent with its proposed switching role; and (3) DMN regions (PCC, VMPFC) were not simply passive but exhibited directed causal influences to other networks.

The critical difference between methods lay in the identity of the primary network hub. GCM identified rFIC ↔ rDLPFC as the dominant temporal relationship (58%, bidirectional), emphasizing the tight coupling between salience monitoring and executive control during ongoing neural dynamics. In contrast, RASL identified VMPFC → rFIC as the dominant structural pathway (89.7%, unidirectional), suggesting that the DMN hub is positioned upstream in the causal hierarchy and may structurally enable or constrain salience network activation. 

This complementarity can be understood by recognizing that GCM captures which regions' activity temporally precedes others (relevant for moment-to-moment information flow), while RASL infers which regions causally influence others at the level of network architecture (relevant for understanding how anatomical and functional constraints shape possible information routes). The strong VMPFC → rFIC structural connection revealed by RASL suggests that the default mode network's architecture positions it to initiate or gate salience responses, while the bidirectional rFIC ↔ rDLPFC temporal coupling revealed by GCM indicates that once engaged, salience and executive systems interact reciprocally on a moment-to-moment basis.

### Clinical Invariance of Core Network Architecture

An important finding from the RASL analysis was the preservation of core causal architecture across clinical groups. Comparing healthy controls (Group 0, N=180) and schizophrenia patients (Group 1, N=130), we found that the VMPFC → rFIC connection was equally dominant in both groups (Group 0: 89.7%, Group 1: 89.8%), with no qualitative differences in network topology at the 70% threshold (Figure W). This invariance suggests that the fundamental structural organization of large-scale brain networks remains intact in schizophrenia, indicating that potential clinical differences may manifest in the dynamics of network engagement rather than in the basic causal architecture itself.

### Implications for Understanding Brain Network Causality

These results demonstrate that different causal discovery methods, when applied to the same neural data, can reveal complementary aspects of brain network organization. Our findings validate the use of both temporal (GCM) and structural (RASL) approaches for understanding brain network causality and highlight the importance of considering multiple levels of causal organization—from structural constraints to temporal dynamics—when investigating large-scale brain networks. The discovery of VMPFC as a structural driver through RASL, alongside the temporal rFIC-DLPFC coupling revealed by GCM, suggests a multi-level causal architecture where structural positioning enables certain temporal interaction patterns. This framework reconciles apparently contradictory findings in the literature by recognizing that different methodologies access different causal relationships, all of which contribute to our understanding of how the brain coordinates activity across distributed networks.

---

## Figure Captions

**Figure X: GCM Analysis Validates Key Findings from Sridharan et al. (2008)**
Granger Causality Mapping results from 310 subjects showing directed temporal influences between six key brain regions. (Left) Network diagram displaying connections present in >30% of subjects. Node colors indicate functional networks: red = Central Executive Network (CEN), blue = Salience Network, green = Default Mode Network (DMN). Edge thickness indicates frequency across subjects. The dominant bidirectional coupling between rFIC and rDLPFC (58%, thick black edges) replicates and extends the rFIC-DLPFC connectivity emphasized by Sridharan et al., while revealing its reciprocal nature in resting-state data. (Right) Quantitative ranking of the top 15 Granger-causal connections. Colors indicate connection types: purple = Salience→CEN, orange = Salience→DMN, red = CEN connections, green = DMN connections. The strong presence of rFIC and ACC connections to both CEN and DMN regions confirms the salience network's role in mediating between these systems.

**Figure Y: RASL Identifies VMPFC as Primary Structural Driver**
Causal structure learning results using RASL applied to the same 310 subjects and brain regions. (Top) Network diagram at >50% threshold showing dense interconnectivity across all networks, with 28 directed edges surviving this stringent criterion. (Bottom) Top 15 strongest structural causal connections ranked by frequency across 2,496 candidate solutions. The VMPFC → rFIC connection (89.7%, green bar) emerges as by far the most consistent structural pathway, indicating that the DMN hub (VMPFC) is positioned upstream of the salience network (rFIC) in the brain's causal architecture. This finding complements temporal analyses by revealing the structural constraints that shape possible information flow patterns.

**Figure Z: Complementary Insights from GCM and RASL**
Side-by-side comparison of network organization revealed by temporal (GCM, left) and structural (RASL, right) causal discovery methods. (Top) Network diagrams at conservative thresholds (GCM: 50%, RASL: 70%) highlight the dominant causal relationship identified by each method. GCM identifies bidirectional rFIC ↔ rDLPFC temporal coupling (58%) as the primary feature, while RASL identifies unidirectional VMPFC → rFIC structural causation (89.7%). (Bottom) Comparison bar charts showing method-specific rankings. Both methods identify high interconnectivity between networks (similarity), but differ in the primary hub (difference): GCM emphasizes Salience-CEN reciprocal dynamics, while RASL reveals DMN as structural initiator.

**Figure W: Clinical Invariance of Network Architecture**  
Comparison of RASL-derived networks between healthy controls (Group 0, N=180, left) and schizophrenia patients (Group 1, N=130, right) at 80% threshold. Both groups show identical topology with a single dominant connection: VMPFC → rFIC (Group 0: 89.7%, Group 1: 89.8%). This preservation of core structural architecture across clinical populations suggests that potential schizophrenia-related differences may lie in the dynamics of network engagement or connection strength modulation rather than fundamental reorganization of causal pathways.

---

## Statistical Details for Methods Section

**GCM Analysis Parameters:**
- Alpha level: 0.05
- Maximum lag: p_max = 8
- Bootstrap surrogates: n = 200
- Block length: 16 TRs
- Surrogate mode: block permutation
- Significance threshold: FDR-corrected at α = 0.05

**RASL Analysis Parameters:**
- Selection mode: top-k
- Solutions per subject: k = 10
- Total solutions analyzed: 2,496 (310 subjects × ~8 solutions)
- Cost-based selection: no ground truth used
- Maximum undersampling: u_max = 4

**Subjects:**
- Dataset: fBIRN schizophrenia study
- Total: N = 310 (Group 0: 180, Group 1: 130)
- ICA components: 6 regions (indices 25, 26, 35, 44, 45, 46)
- TR: 2.0 seconds

---

## Key Statistics for Results Text

### GCM Results:
- Mean edge frequency: 34.02%
- Strongest connection: rFIC ↔ rDLPFC (58% both directions)
- Connections >50%: 2 (both rFIC ↔ rDLPFC)
- Connections >30%: 21 edges

### RASL Results:
- Mean edge frequency: 57.04%
- Strongest connection: VMPFC → rFIC (89.7%)
- Connections >80%: 1 (VMPFC → rFIC)
- Connections >70%: 1 (VMPFC → rFIC)
- Connections >50%: 28 edges

### Agreement Points with Sridharan et al. (2008):
1. ✓ rFIC central to network interactions
2. ✓ rFIC → rDLPFC connection confirmed (58%)
3. ✓ rFIC connects to both CEN and DMN
4. ✓ ACC (salience partner) → DMN regions (36-37%)
5. ✓ Salience network mediates between CEN and DMN

### Novel Findings:
1. rFIC ↔ rDLPFC bidirectional (not unidirectional)
2. VMPFC identified as primary structural driver (89.7%)
3. Core architecture preserved across clinical groups

---

## Draft Text Snippets

### For emphasizing validation:
"Our GCM implementation showed strong concordance with the seminal findings of Sridharan et al. (2008), successfully replicating the central role of rFIC in mediating between CEN and DMN (rFIC → rDLPFC: 58%, rFIC → DMN: 30-34%)."

### For emphasizing complementarity:
"While GCM revealed rFIC ↔ rDLPFC reciprocal coupling as the dominant temporal relationship (58%), RASL identified VMPFC → rFIC as the primary structural pathway (89.7%), demonstrating that temporal and structural causal analyses access distinct but complementary aspects of brain network organization."

### For emphasizing robustness:
"The remarkably high frequency of the VMPFC → rFIC connection (89.7% of solutions across 310 subjects) suggests this represents a fundamental architectural feature of large-scale brain networks, preserved across both clinical groups."

### For emphasizing novel insight:
"Critically, RASL's identification of VMPFC as a structural driver extends previous frameworks that emphasized rFIC as the primary control hub, suggesting a hierarchical architecture where DMN positioning enables downstream salience and executive network engagement."

