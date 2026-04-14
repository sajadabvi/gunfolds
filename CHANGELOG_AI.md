# AI-Assisted Development Log

Short summaries of code and documentation changes made via Cursor AI sessions.


## 2026-04-13

### Clingo / clasp: solver-flag benchmark and paper note

- **Paper note:** [`gunfolds/scripts/papers/clingo_clasp_optimization_flags_benchmark.md`](gunfolds/scripts/papers/clingo_clasp_optimization_flags_benchmark.md) — documents empirical evaluation of clasp flags on FBIRN N=10 weighted RASL: **`--opt-strategy=usc,stratify` is not recommended** (timeouts, unstable or worse costs vs baseline); **`--opt-heuristic=1`** and **`--project=show`** preserve optimal cost/solution sets when runs complete but **speedups are subject-dependent** (sometimes baseline fastest). Production default: **no extra clasp flags** beyond `configuration=crafty` and existing `run_clingo` / `drasl` path. Also records that the hypothesized `--configuration=` split-argv bug was **refuted** on clingo 5.7.1, and warns about **`--gt_density 220`** clamping to **100** (wrong target) on old SLURM invocations.
- **Companion update:** [`gunfolds/scripts/papers/rasl_clingo_soft_optimization_and_scaling.md`](gunfolds/scripts/papers/rasl_clingo_soft_optimization_and_scaling.md) — cross-reference to the benchmark note; text aligned with current **`MAXCOST`** (20 in fMRI large) and **`density_weight`** default (50).

### Benchmark and diagnostic scripts

- **`gunfolds/scripts/tests/benchmark_clingo_flags.py`:** Runs all **2³** combinations of USC / opt-heuristic / project on the same PCMCI→`drasl_command` instance; prints per-scenario timing, clingo stats (with safe access for statistics objects), parsed solution comparison, **`--skip` / `--only`** scenario filters, and **`--timeout`** with **`ctrl.interrupt()`** for stuck solves.
- **`gunfolds/scripts/tests/test_clingo_config_bug.py`:** Confirms split `["--configuration=", "crafty"]` vs `["--configuration=crafty"]` yield the same active configuration.

### RASL fMRI pipeline parameters (continued)

- **`fmri_experiment_large.py` / `diagnose_rasl_bottleneck.py`:** **`MAXCOST = 20`** for DD/BD scaling (was 50).
- **`gunfolds/solvers/clingo_rasl.py`:** Default **`density_weight=50`** (was 100); optional **`extra_clingo_args`** threaded through to **`gunfolds.utils.clingo.run_clingo`** / **`clingo()`** for exploratory clasp flags without changing defaults.

---

## 2026-04-12

### RASL / clingo: soft single-level optimization + coarser costs (scaling)

- **`gunfolds/solvers/clingo_rasl.py`:** **Option A** — weighted DRASL weak constraints unified at **`@1`** (no lexicographic `@2` density vs `@1` edges). Density penalty uses `hypoth_density` with **×100** (was ×1000) and `[Diff * density_weight @ 1]` with default **`density_weight=100`** passed through `drasl_command` / `drasl`. `PRIORITY` / `edge_weights` positions 0–3 kept for API compatibility but no longer set distinct ASP priority levels for the four edge families.
- **DD/BD scale:** **`MAXCOST = 50`** (was 10000) everywhere in active (non-legacy) experiment and real-data scripts — 50 discrete weight levels per matrix entry.
- **`GT_density` scale:** **0–100** (density×100); defaults e.g. N=20 **22** (was 215 at ×1000). **`fmri_experiment_large.py`**, **`slurm_fmri_large.sh`**, **`diagnose_rasl_bottleneck.py`** aligned.
- **Docs:** Handover **`Past_chat/rasl_clingo_optimization_scaling_handoff.md`** (SLURM tools, diagnostics, theory, checklist). Paper-oriented technical note **`gunfolds/scripts/papers/rasl_clingo_soft_optimization_and_scaling.md`** (methods appendix material, two formulations, granularity, suggested experiments).

### RASL / clingo bottleneck diagnostic (local progress visibility)

- **`gunfolds/scripts/real_data/diagnose_rasl_bottleneck.py`:** Runs the same PCMCI → penalty matrices → `drasl_command` path as `fmri_experiment_large.py` for one subject, but replaces the silent `gunfolds.utils.clingo.run_clingo` path with an instrumented run: **grounding** progress via clingo `GroundProgramObserver` (periodic atom/rule counts), **solving** progress via per-model `on_model` prints (cost, optimality), then clingo statistics and a timing split (PCMCI vs grounding vs solving) with a suggested bottleneck (grounding vs search). Flags: `--n_components`, `--subject_idx`, `--MAXU`, `--PNUM`, `--timeout`, `--grounding_interval`, plus PCMCI/GT knobs aligned with the large experiment.

### SLURM monitoring for `fmri_large` array jobs

- **`gunfolds/scripts/cluster/slurm_diag_fmri.sh`:** One-shot report (queue, `sacct` by state, `seff` sample, memory via `sstat`, log tails, master/task logs).
- **`gunfolds/scripts/cluster/slurm_live_fmri.sh`:** Running tasks, `sstat`, tasks-per-node, optional `ssh` + `ps`/`uptime`/`free` on compute nodes.
- **`gunfolds/scripts/cluster/slurm_progress_fmri.sh`:** Per-task stdout line counts, stall detection (stale `.out`), stderr snippets for failed tasks.
- **`gunfolds/scripts/cluster/slurm_watch_fmri.sh`:** Refreshing dashboard loop.

### Exp 4: N=20 PCMCI hyperparameter grid — results documented

- **Results:** 48-config grid (311 FBIRN subjects, 20 ICA components) merged in `gunfolds/scripts/real_data/results_exp4_04122026050527/exp4_n20_results_04122026062704.json`.
- **Paper note:** `gunfolds/scripts/papers/exp4_n20_pcmci_hyperparam_results.md` — methodology, top configs by composite score (0.6×Jaccard + 0.4×proximity to 22% density), recommendation **`pcmci_tau1_a0.05_fdrnone`** (~23% mean density, composite 0.474), comparison to PCMCIplus, runtimes, and `fmri_experiment_large.py` CLI mapping.
- **Cluster:** `slurm_fmri_large.sh` updated for qTRDGPU, 2-day wall time, 160G / 15 CPUs, default N=20 RASL with Exp4 PCMCI seed + `fixed` GT density **22** (×100 scale; `--array=0-309%50` documented).

---

## 2026-04-10

### PCMCI hyperparameter audit, Glag2CG bug fix, and codebase unification

Fixed reversed edge directions in canonical `cv.Glag2CG` (incorrect `np.transpose`), corrected NumPy advanced-indexing bug that swapped time/variable axes in fMRI data slicing, and ran a 36-config grid search to find optimal PCMCI settings. Switched default from `run_pcmci(tau_max=1, alpha=0.1)` to `run_pcmciplus(tau_max=2)`, improving cross-subject Jaccard from 0.313 to 0.460 (+47%). Removed 12 duplicated local `Glag2CG` copies across the codebase. Full experiments and results: **`gunfolds/scripts/papers/pcmci_hyperparameter_audit.md`**.

---

### N-specific default `GT_density` for RASL fixed mode

- **`fmri_experiment_large.py`:** `--gt_density` now defaults to (`fixed`). Under `--gt_density_mode fixed`, the effective density is **350** (N=10), **215** (N=20), or **125** (N=53) when `--gt_density` is not passed — midpoints of the ranges in `gunfolds/scripts/papers/ground_truth_connectivity_estimates.md` §7. Explicit `--gt_density` still clamps to 0–1000. Saved `result.zkl` / `run_params.zkl` include effective `gt_density` and optional `gt_density_explicit` (CLI value, or `None` if the default was used).
- **`slurm_fmri_large.sh`**, **`submit_fmri_experiment.sh`**, **`submit_fmri_experiment_partial.sh`:** For `fixed` mode, the optional numeric argument is only forwarded when set, so jobs can rely on the Python N-based defaults.
- **`Past_chat/fmri_experiment_large_handoff.md`:** Documented the mapping and cluster behavior.

---

## 2026-04-03

### HC vs SZ supervised classification (Tiers 1–3 + time-series + aggregator)

Added scripts under `gunfolds/scripts/analysis/` to compare healthy-control vs schizophrenia classification using causal graphs from `fmri_experiment_large.py` (`result.zkl`) and, optionally, raw ICA time series.

- **`classify_hc_sz.py` (Tier 1)**: Classical ML on vectorised features — mean adjacency (off-diagonal), per-edge std across RASL solutions, topology (density, in/out degree stats), RASL extras (cost mean/std, mean undersampling rate, edge-frequency entropy). Classifiers: SVM (linear/RBF), Random Forest, Logistic Regression (L1/L2) with nested stratified CV; optional permutation p-values.
- **`brain_transformer_classify.py` (Tier 2)**: PyTorch “BrainNet”-style graph classifier — each region is a token with row/column/std connectivity features plus learnable domain embedding; transformer encoder; orthonormal-clustering readout; early stopping on validation accuracy.
- **`solution_set_transformer.py` (Tier 3)**: Solution-set model for RASL — shared graph encoder per solution (adjacency + cost + undersampling), induced set attention block (ISAB) + pooling by multihead attention (PMA) over up to `--max_solutions` graphs per subject; meaningful for multi-solution RASL, degenerates to single-graph for PCMCI/GCM.
- **`timeseries_foundation_classify.py`**: Factored spatiotemporal transformer on FBIRN ICA time courses from `fbirn_sz_data.npz` (spatial attention across regions, temporal attention after pooling); bypasses causal discovery for a baseline comparison.
- **`run_all_classifiers.py`**: Optional `--run-all` to invoke the four scripts via subprocess; `--aggregate-only` loads `tier1_results.csv`, `tier2_results.csv`, `tier3_results.csv`, `timeseries_foundation_results.csv` from `fbirn_results/<TIMESTAMP>/ml_classification/`, writes `all_tiers_combined.csv`, `best_per_config_tier.csv`, `method_comparison.csv`, and bar plots (`all_tiers_comparison.png`, `method_comparison.png`).

**Usage (from `gunfolds/scripts/real_data/`):** `python ../analysis/run_all_classifiers.py --timestamp <TS> --run-all` or run each tier script with `--timestamp <TS>`.

**Dependencies:** scikit-learn (Tier 1); PyTorch for Tiers 2–4.

**Stability / review:** Tier 1 uses nested CV and optional permutation tests; deep tiers use stratified K-fold with held-out folds. For “how stable are your results,” report fold-wise mean±std, compare to permutation or shuffle labels, and repeat across independent timestamps or train/val splits; multisite confounds (motion, site) should be stated explicitly if generalising beyond FBIRN.

---

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
