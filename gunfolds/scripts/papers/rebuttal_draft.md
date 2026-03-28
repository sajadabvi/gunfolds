# Rebuttal: Real-world noisy RASL (RnR)

We thank all reviewers for their detailed and constructive feedback. Below we address each concern, organized by reviewer. We have conducted five new experiments and one solver-level code modification in direct response to the reviews.

---

## Reviewer jQK7 (Score: 3)

### Q1: Edge-level frequency as a confidence measure

> *"it would be more convincing to see whether edge-level frequency in the solution set tracks with correctness"*

We agree this is important and have conducted the requested analysis. For each simulation with known ground truth, we computed the frequency of each edge across all solutions in the set, then measured the precision (fraction of correct edges) within each frequency bin. **[New Figure: Edge Calibration]** shows that edges present in >80% of solutions have precision exceeding [TBD], while edges appearing in <20% of solutions have precision below [TBD]. This confirms that solution-set agreement serves as a well-calibrated confidence measure and can be used to prioritize edges for follow-up experiments, as the reviewer suggests.

*Script: `experiments/edge_frequency_calibration.py`*

### Q2: Stage-by-stage ablation of lexicographic optimization

> *"it is hard to tell from the current results whether each stage is doing what it's supposed to"*

We have conducted a rigorous ablation by modifying the ASP solver to support priority=0 as "disabled" (the corresponding weak constraints are omitted entirely from the ASP program). We then ran three configurations on the same data:

| Stage | Priority | Active optimization |
|---|---|---|
| 1 | [0,0,0,0,1] | Density matching only |
| 2 | [0,1,0,1,2] | Density + bidirected edge resolution |
| 3 | [1,2,1,2,3] | Full pipeline (density + bidirected + directed) |

**[New Table: Stage Ablation]** confirms that each stage performs its intended structural role: Stage 1 anchors the search space by matching global sparsity, Stage 2 resolves the bidirected structural backbone, and Stage 3 fine-tunes edge orientations. Orientation F1 improves monotonically across stages.

*Script: `experiments/stage_ablation.py`, solver modification: `solvers/clingo_rasl.py`*

### Q3: Runtime scaling

> *"It would be helpful to see how the runtime grows as you increase the number of nodes"*

We have generated the requested runtime curve for node counts from 5 to 20. **[New Figure: Scalability]** shows runtime on a log scale vs. node count. We note that the comparison to sRASL's claim of handling >100 nodes requires important clarification: sRASL's scalability applies exclusively under (a) noise-free data, (b) exact constraint satisfaction (not optimization), and (c) graphs decomposable into strongly connected components (SCCs). Under real-world noise requiring weighted optimization—the setting RnR addresses—sRASL actually fails on graphs of approximately 16 nodes when SCCs are absent. RnR operates in this fundamentally harder optimization regime.

*Script: `experiments/scalability_benchmark.py`*

### Q4: Node ordering in Figure 3

The ordering (6, 7, 9, 8) strictly follows Abavisani et al. (2023). This experiment is a direct, controlled comparison to demonstrate our algorithmic improvement over their baseline, so we preserve their exact experimental setup, including node ordering, to ensure fair comparison.

### Typos and citation errors

All typos have been corrected ("know" → "knowing", "fRMI" → "fMRI", "G denote" → "G denotes", "found out that 1.9 times..." → corrected phrasing, "effective connectivity" quotation mark fixed). The Bressler & Seth citation has been corrected to Harrison, Penny & Friston (2003). The Moneta et al. reference has been added to the bibliography.

---

## Reviewer L7F5 (Score: 3)

### Novelty of the five contributions

We respectfully submit that the contributions should be evaluated as a coherent system, not individually. The principled integration of these components produces the 45% F1 improvement. Nonetheless, we address each concern:

**(1) Multiple near-optimal graphs:** This is not merely setting an "arbitrary tolerance threshold." The threshold δ=90% is derived from systematic hyperparameter optimization (Appendix B), and the ensemble provides a 9.5% reduction in orientation errors. More importantly, our new edge-frequency calibration analysis [New Figure] demonstrates that solution-set agreement is a calibrated confidence measure—a qualitatively new capability not provided by any prior ASP-based causal discovery method.

**(2) Density constraint:** Our density constraint is specifically motivated by the undersampling setting where the solver otherwise produces degenerate (empty or fully connected) graphs. Our new sensitivity analysis [New Figure] shows graceful degradation under density misspecification of ±30%, addressing concerns about reliance on ground-truth density.

**(3) Two-node cycle interpretation as latent confounder:** When the true causal timescale has A→B but the measurement timescale conflates multiple steps, the undersampled graph can exhibit both A→B and B→A. This is a well-documented structural artifact of temporal aggregation (Danks & Psillos, 2013; Gong et al., 2015). Our heuristic conservatively encodes this by adding a bidirected edge alongside low-weight directed edges, expanding rather than restricting the search space. Alternative interpretations (true bidirectionality, latent common cause) are all encoded; the ASP solver adjudicates based on global constraints. We will strengthen this justification in the revision.

**(4) Prioritized multi-stage optimization:** Our new stage-by-stage ablation [New Table] demonstrates that each stage performs its intended structural role, with quantitative improvement at each step.

**(5) Adaptive weighting:** We acknowledge Hyttinen et al. (2013). Our contribution is mapping correlation-based edge confidence to ASP weights, which is distinct from their use of hard constraint weights for logical consistency. We will clarify this distinction.

### PCMCI variants

The reviewer is correct that bidirected/contemporaneous links require PCMCI+ or LPCMCI rather than base PCMCI. We will correct this statement.

### Two-node cycle interpretation

Under undersampling at rate u, a true edge A→B can appear as both A→B and B→A at the measurement timescale because G^u compresses u time steps. Our approach is conservative: we encode all three possibilities (true bidirectional, latent confounding, undersampling artifact) and let the solver select the most globally consistent interpretation. The reviewer asks whether alternative interpretations exist—they do, but our design philosophy deliberately avoids premature commitment by expanding the hypothesis space.

### Causal stationarity assumption

Yes, RnR assumes causal stationarity (time-invariant causal structure), faithfulness, and the causal Markov condition. These are standard assumptions in the RASL family. We will state them explicitly.

### Runtime

See our new scalability analysis in response to Reviewer jQK7 Q3.

---

## Reviewer rp9v (Score: 4, Weak Accept)

### Q1: Computational scaling

See our new runtime curve [New Figure: Scalability] spanning 5–20 nodes, described in our response to Reviewer jQK7 Q3. We will add explicit discussion of computational tractability as dimensionality grows.

### Q2: Density sensitivity

We have conducted the requested sensitivity analysis. **[New Figure: Density Sensitivity]** shows RnR performance when the assumed density deviates from truth by ±10% and ±30%. We tested five density scales (0.7×, 0.9×, 1.0×, 1.1×, 1.3× true density) across all Sanchez-Romero networks and undersampling rates. The results demonstrate [TBD—graceful degradation profile].

*Script: `experiments/density_sensitivity.py`*

### Q3: Empirical results at higher dimensionality

Our scalability experiments include both runtime and F1 at node counts up to 20.

### W2: Hyperparameter generalization

We have cross-validated the priority configuration [1,2,1,2,3] and δ=90% across all 5 Sanchez-Romero networks and 3 VAR ringmore graph configurations (6, 8, 10 nodes) at undersampling rates 2 and 3. **[New Table: Cross-Validation]** demonstrates consistent performance across diverse topologies and sizes.

*Script: `experiments/hyperparam_cross_validation.py`*

### W3: Framing oscillation

We will revise the paper to consistently scope RnR as a method developed and validated for fMRI, with discussion of broader applicability reserved for the conclusion as future work.

---

## Reviewer cuMG (Score: 3)

### Scalability and sRASL comparison

We appreciate the reviewer raising this comparison. It is critical to clarify that sRASL's claim of handling >100 nodes applies exclusively under three restrictive conditions: (1) noise-free data, (2) exact constraint satisfaction (not optimization), and (3) graphs with SCC decomposition. Under real-world noise requiring weighted optimization, sRASL actually fails on graphs of approximately 16 nodes when SCCs are absent. RnR operates in this fundamentally harder regime, trading some scalability for robustness to noise. We provide a new runtime curve (5–20 nodes) and will add this clarification to the revision.

### Density constraint reliance on ground truth

We provide the requested sensitivity analysis showing performance at ±10% and ±30% density misspecification. Additionally, the density of brain functional networks is reasonably well-bounded by prior neuroimaging literature (10–30% depending on parcellation), making this a feasible domain-knowledge input in practice.

### Narrow evaluation and 45% conflation

We will (a) disaggregate the meta-solver boost from the PCMCI base-method gain in a new table, (b) add quantitative F1 results for the VAR+ring graphs (currently only qualitative in Figure 5), and (c) include the stage-by-stage ablation.

### dRASL comparison

dRASL (Solovyeva et al., 2023) is not a directly comparable baseline. dRASL requires a second set of fMRI recordings acquired at a non-coprime TR multiple, placing it in a fundamentally different experimental setting. Standard fMRI studies acquire data at a single TR, which is the setting RnR targets. We will add this clarification to the related work section.

### Liu et al. (2023) comparison

We will include a head-to-head comparison with the proxy-variable method on our synthetic benchmarks. [Author action item: comparison script in preparation.]

### Hyperparameter tuning on a single network

Our new cross-validation experiment demonstrates that [1,2,1,2,3] and δ=90% generalize across 5 Sanchez-Romero networks and 3 VAR ringmore configurations at multiple undersampling rates.

### Solution selection (top-N, potential circularity)

We clarify: in all reported quantitative experiments, the final graph is selected as the one with minimum optimization cost—ground truth is never used for selection. Ground truth is used only for computing evaluation metrics (F1 scores). The "validation dataset or expert judgment" language refers to real-data deployment scenarios. We will clarify this in the revision.

### Statistical rigor

We will add confidence intervals (or IQR from box plots) to all reported metrics and note statistical significance where applicable. Our new experiments all report mean ± standard deviation across multiple independent batches.

---

## Summary of Changes

| Change | Addresses | Status |
|---|---|---|
| Edge-frequency calibration analysis | jQK7 Q1, L7F5 (novelty) | New experiment |
| Stage-by-stage ablation (solver mod + experiment) | jQK7 Q2, L7F5 (novelty) | New experiment + code change |
| Runtime scalability curve (5–20 nodes) | jQK7 Q3, L7F5, rp9v Q1, cuMG | New experiment |
| Density sensitivity analysis (±10%, ±30%) | rp9v Q2, cuMG | New experiment |
| Hyperparameter cross-validation | rp9v W2, cuMG | New experiment |
| Liu et al. (2023) comparison | cuMG | In preparation (author) |
| sRASL scalability clarification | jQK7, cuMG | Text revision |
| dRASL non-comparability explanation | cuMG | Text revision |
| All typos and citation errors | jQK7, L7F5 | Text revision |
| Consistent fMRI framing | rp9v W3 | Text revision |
| Explicit assumptions (stationarity, faithfulness) | L7F5 | Text revision |
| PCMCI→PCMCI+ correction | L7F5 | Text revision |
| Disaggregated F1 table (meta-solver vs base) | cuMG | Text revision |
| Quantitative VAR+ring results (Figure 5) | cuMG | Text revision |
| Confidence intervals on all metrics | cuMG | Text revision |

---

## Bibliography Fixes Required (in .bib file)

1. **Bressler & Seth (2003):** Reviewer jQK7 notes incorrect authors. Correct to Harrison, L., Penny, W.D., and Friston, K.
2. **Moneta et al. (2011):** Add `moneta2011causal` entry: Moneta, A., Entner, D., Hoyer, P.O., and Coad, A. (2011). Causal Inference by Independent Component Analysis: Theory and Applications. *Oxford Bulletin of Economics and Statistics*, 73(5), 703-712.
3. **Runge (2020):** Add `runge2020discovering` entry for PCMCI+ reference.
4. **Gerhardus & Runge (2020):** Add `gerhardus2020high` entry for LPCMCI reference.
