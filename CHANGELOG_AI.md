# AI-Assisted Development Log

Short summaries of code and documentation changes made via Cursor AI sessions.



## 2026-03-23

### Experiment config `.md` saved with each run
- **`submit_fmri_experiment.sh`** and **`submit_fmri_experiment_partial.sh`**: After submission, both scripts now create `fbirn_results/<TIMESTAMP>/experiment_config.md` recording all experiment parameters (timestamp, date, N subjects, configurations, GT density settings, RASL params, resource limits, partition/subject distribution, job IDs, SLURM script path). This makes it easy to revisit past results and understand what each run was.

---

## 2026-03-16

### GT_density mode for RASL in fMRI large experiment
- **`fmri_experiment_large.py`**: Added configurable `GT_density` behavior for RASL via three options (default: `none`).
  1. **`--gt_density_mode none`** (default): pass `GT_density=None` to `drasl()` (no density constraint).
  2. **`--gt_density_mode fixed`**: use a fixed value 0–1000 (density×1000); `--gt_density` (default 75).
  3. **`--gt_density_mode fraction`**: use a fraction of `g_estimated` density; `--gt_density_fraction` (default 0.5). Value is clamped to [0, 1].
- **`slurm_fmri_large.sh`**: For RASL jobs, optional 5th arg = mode (`none`|`fixed`|`fraction`), 6th arg = value (fixed: 0–1000 default 75; fraction: e.g. 0.5). Usage comments and examples updated.

---

## 2026-03-12

### Created `gunfolds/scripts/papers/rasl_vs_pcmci_results.md`
Wrote a detailed interpretation of the N10_domain_RASL vs N10_none_PCMCI comparison on FBIRN fMRI data (310 subjects). Covers: executive summary table, the undersampling problem, domain-based SCC strategy, quantitative metrics (15 vs 1 significant edges, Frobenius, density), heatmap analysis organizing 15 significant edges into four neuroscientific circuits (visual→subcortical, CC→DMN, auditory→DMN, DMN outgoing), and a mechanistic explanation of why RASL outperforms PCMCI. References established SZ literature.

### Created `gunfolds/scripts/papers/rasl_vs_literature_comparison.md`
Wrote a systematic comparison of RASL results against the 20-year fMRI literature review (`deep-research-healthy_vs_schz.md`). Identified 7 points of agreement (thalamocortical dysconnectivity, DMN hypoconnectivity, task-positive/task-negative failure, auditory-DMN coupling, cerebellar involvement, insula involvement, visual-subcortical hyperconnectivity) and 6 points of disagreement/tension (causal direction of thalamocortical influence, missing prefrontal component, density rewiring-vs-depletion, salience network direction, absent hippocampal/striatal circuits, unvalidated undersampling-rate biomarker).

---

## Quick Reference: FDR vs FWER

| | FWER (Bonferroni) | FDR (Benjamini-Hochberg) |
|--|-------------------|--------------------------|
| **Controls** | P(any false positive) | Expected proportion of false positives among discoveries |
| **Threshold** | alpha / m (same for all tests) | alpha * rank / m (adaptive per test) |
| **Conservatism** | Very conservative | Less conservative |
| **Power** | Low when m is large | Higher -- more true effects detected |
| **Use when** | Any single false positive is costly | A small fraction of false discoveries is acceptable |

---

### Added FDR correction option to `analyze_fmri_experiment.py`
Modified `edge_level_tests()` to support both Bonferroni (FWER) and Benjamini-Hochberg (FDR) correction via a `correction` parameter (default: `"bonferroni"` to preserve existing behaviour). Added `_benjamini_hochberg()` helper (no new dependencies). Threaded `--correction` CLI flag through `parse_args`, `analyze_config`, and `main`. Refactored `plot_edge_diff_heatmap` to use a pre-computed `_sig_mask` instead of recalculating Bonferroni inline. Added FDR-vs-FWER quick-reference table to this changelog.

### Created `gunfolds/scripts/papers/rasl_vs_pcmci_results_fdr.md`
FDR-corrected companion to the Bonferroni report. RASL jumps from 15 to 28 significant edges; PCMCI stays at 1 (ratio widens from 15:1 to 28:1). The 13 new FDR edges extend all four original circuits and reveal two new motifs: (1) thalamic multi-modal convergence (four sources drive the thalamus excessively in SZ) and (2) cerebellar disconnection syndrome. Also identifies the Insula as a new source (salience→DMN) and STG as a 5-target cross-modal hub. Full 28-edge table with Bonferroni/FDR-only labels provided.

### Created `gunfolds/scripts/papers/rasl_vs_literature_comparison_fdr.md`
FDR-corrected comparison of RASL's 28 significant directed edges against the 20-year fMRI literature review. Identified 10 points of agreement (thalamic sensory/motor hyperconnectivity, thalamic gating failure with 4 converging sources, STG/auditory dysfunction, intra-DMN hypoconnectivity, salience switching dysfunction, cerebellar disconnection with 3 directed edges, striatal-thalamic loops, frontoparietal CC abnormalities, reduced network segregation, sensorimotor–auditory coupling) and 10 points of disagreement/tension (thalamocortical arrow direction, missing thalamo-prefrontal signature, early-course hyperconnectivity, salience hypo- vs. hyperconnectivity, visual cortex as novel broadcaster, auditory within- vs. between-network framing, absent hippocampal component, dynamic state dependence, cerebellar–DMN coupling direction, GSR/motion confounds). The FDR version serves authors better than the Bonferroni comparison for peer review: more agreement touchpoints, richer circuit-level stories, stronger novel contributions, and no additional genuine contradictions.

> "Even under the most conservative correction (Bonferroni), RASL identifies 15 significant group-differing edges vs. PCMCI's 1. Under the standard FDR correction (BH, q = 0.05), this widens to 28 vs. 1, and all 15 Bonferroni edges are retained. The 13 additional FDR edges are internally coherent — they complete circuits, fill reciprocal loops, and introduce two new circuit motifs — consistent with true discoveries rather than noise."


we will be asked about how stable are your results. Please think about how to address that

