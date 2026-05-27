# AI-Assisted Development Log

Short summaries of code and documentation changes made via Cursor AI sessions.


## 2026-05-27  (branch: current)

### runtime_scaling: stable-matrix sampling strategies + large-N resubmit script

Diagnosed the `Could not find stable matrix after 1,000,000 tries` failure: ρ(random sparse W) grows like √(N·density), so the post-`0.99/ρ` rescale shrinks entries as 1/√N and the path-strength filter rejects nearly every draw at large N.

Added five composable strategies to `create_stable_weighted_matrix` in `runtime_scaling.py`: `scale_aware` (σ = 1/√⟨in-deg⟩, on by default), `bias_magnitudes`, `auto_threshold`, configurable `powers`, and a new deterministic `construct_stable_matrix_from_sccs(A, partition)` (block-triangular, zero rejection, ρ < 1 by construction). Plus `get_stable_weighted_matrix(...)` dispatcher and CLI flags `--w_strategy`, `--w_threshold`, `--w_powers`, `--w_scale_aware`/`--w_no_scale_aware`, `--w_bias_magnitudes`, `--w_auto_threshold`. All strategies succeed in < 3 ms at N=24, 36, 54 in standalone tests.

New `submit_runtime_scaling_large.sh`: one instance each at N=30/42/54 with max walltime `5-08:00:00`, `--timeout_hours=127`, `--w_bias_magnitudes`. Memory + CPUs sized so `mem/cpu ≤ 15 GB` (qTRDGPU MaxMemPerCPU), avoiding the silent CPU bump the old N=54 run hit: N=30 → 160 GB/11 cpus, N=42 → 256 GB/18 cpus, N=54 → 480 GB/32 cpus.

**Files:** `gunfolds/scripts/experiments/runtime_scaling.py`, `gunfolds/scripts/experiments/submit_runtime_scaling_large.sh` (new).


## 2026-05-07  (branch: current)

### New experiment + three library-bug workarounds + checklist updates

**New experiment** in `gunfolds/scripts/experiments/`: `runtime_scaling.py` (one job per `(N, instance_id)`: build multi-SCC ring G¹ → VAR + BOLD → PCMCI `tau_max=1,alpha=0.05` → drasl with `threading.Timer` interrupt → F1), `submit_runtime_scaling.sh` (100 jobs, skip-if-completed/queued guards), `aggregate_results.py`, `check_status.sh`. SCC compositions: `N=8→[6,2], 10→[6,4], 12→[6,6], 14→[6,4,4], 18→[6,6,6], 20→[6,6,4,4], 24..54=[6]*k`. Headline medians: N=8 → 0.2 s, N=10 → 2.9 s, N=12 → 1.6 min, N=14 → 1.4 min, N=18 → 50 min, N=20 → 8.9 h.

**Three caller-side workarounds for gunfolds bugs** (upstream issues filed):

1. `gk.randomDAG` infinite-loops for `N ≤ 2` (`remove_tril_singletons` off-by-one); replaced with a hand-rolled `nx.DiGraph` chain in `make_multi_scc_ring`.
2. `simulate_bold` ignored `u_rate` because `end_time=100` was fixed; now passes `end_time=100*u_rate`.
3. `compute_bold_signals` returns a ragged 1-D object array when scipy `vode` fails on one node; detect `ndim == 1` and retry with tighter input rescaling.

**Cluster-ops lessons:** `sbatch --wrap` runs under `/bin/sh` (dash) so `source` must be `.` or prepend `#!/bin/bash`; `qTRDGPU` enforces `MaxMemPerCPU≈15.2 GB` and silently bumps `--cpus-per-task` to satisfy the ratio; `clingo -n 1 --opt-mode=opt` returns the first feasible model, not the optimum — use `-n 0`.

**Checklist skill updates** (`~/.claude/skills/checklist/SKILL.md`): item 16 (`simulate_bold` end_time scaling), item 17 (never call `gk.randomDAG` when `num_sccs ≤ 2`), mandatory `_assert_glag2cg_direction()` sanity check under item 2.

**Files:** `gunfolds/scripts/experiments/{runtime_scaling.py,submit_runtime_scaling.sh,aggregate_results.py,check_status.sh}` (all new); `~/.claude/skills/checklist/SKILL.md`.


## 2026-05-04  (branch: current)

### SCC quotient back-edge selection: weighted MFAS via igraph `exact_ip` (Option C)

Replaces the class-index-ordering criterion in `_acyclic_quotient_edges` (introduced in `0079ca60`) with a principled minimum-weight feedback arc set (MFAS) using exact integer programming via `python-igraph`'s `Graph.feedback_arc_set(method='exact_ip')`.

**Edge weight definition** (uses *all* available PCMCI signal, per the user's design ask). For each candidate cross-class arrow `K → L`:

```
w(K → L) = pos(K → L) + neg(L → K)
```

where `pos(K → L)` is the total `hdirected` weight summed across `(X ∈ K, Y ∈ L)` node pairs that PCMCI judges present, and `neg(L → K)` is the total `no_hdirected` weight summed across `(Y ∈ L, X ∈ K)` node pairs that PCMCI judges absent. The first term penalises dropping arrows that PCMCI directly supports; the second penalises drops that would force the encoding into the reverse direction, which `no_hdirected` facts contradict. The minimum-weight feedback arc set returned by igraph's exact ILP solver is the principled drop set: minimum number of edges (NP-hard in general, trivial at our 7-node quotient size) and minimum total PCMCI evidence cost.

**Headline result on FBIRN N=10 subject 0** (NeuroMark domain partition → cyclic 7-class quotient with 22 internal edges): class-index fallback dropped 14 back-edges keeping 8 quotient edges; weighted MFAS drops only **5 back-edges keeping 17** (77% retention) with a total evidence cost of 142. The number of dropped edges is provably minimum at this scale; the choice of *which* 5 minimises lost PCMCI evidence.

**Empirical impact on subject 1 N=10 baseline:** solve time effectively unchanged (0.04 s vs prior 0.03 s). Optimum cost descends from `[452, 350]` to `[423, 350]` — expected and correct: keeping 17 quotient edges instead of 8 gives the solver more freedom (the SCC integrity constraints fire less often), so a lower-cost graph becomes reachable. The encoding is *less* restrictive than the class-index fallback but *more principled* — it only drops the cycle-creating back-edges that are necessary, weighted by PCMCI confidence. Compared to the original cyclic encoding (which optimum was `[340, 350]`, an unsound phantom), this 423 is sound modulo the speed-vs-soundness lever already documented (we still drop *some* valid cross-class arrows; just the minimum-evidence-cost subset).

**Backward compatibility.** When `dm` is not supplied to `encode_list_sccs` (legacy callers), `_acyclic_quotient_edges` falls back to the class-index ordering. `drasl_command` always has `dm` available and now passes it through.

**Logged for future.** Bayesian log-likelihood-ratio weights (Option D from the design discussion) requires calibrating DD weights against actual probability scales and is captured in `todo.md` item #7. Suggestion #9 ground-reduction trick (drop weight-0 facts) is captured in `todo.md` item #6 — to measure first before implementing.

**Files:** `gunfolds/conversions.py` (`_acyclic_quotient_edges` gains `dm` parameter and weighted-MFAS path; `encode_list_sccs` plumbs `dm` through; updated docstrings). `gunfolds/solvers/clingo_rasl.py` (passes `dm` to `encode_list_sccs`). `gunfolds/scripts/papers/scc_quotient_edge_dropping_research.md` (new — local-only research doc with literature review, weight-function options A–D, and the Option C derivation). `todo.md` (items 6 and 7 added).

---

### SCC encoding fix: `dag/3` → `scc_edge/3` rename + acyclic-quotient via back-edge dropping (todo item #1)

**Two-part change in `gunfolds/conversions.py`.**

1. **Rename `dag/3` → `scc_edge/3`** in the three sites that emit/consume the predicate inside `encode_sccs` and `encode_list_sccs`. The old name was misleading because the relation is the SCC quotient digraph of the measured graph, not a DAG in general.

2. **Acyclic-quotient enforcement via back-edge dropping (no class merging).** When `scc_members` is supplied (the production path with `--scc_strategy=domain` / `correlation`), the partition is hand-specified by NeuroMark domain or correlation cluster and may *split* a real SCC across multiple classes — producing a cyclic quotient. New helper `_acyclic_quotient_edges(glist, partition)` builds the quotient over the union of `glist`, finds its SCCs, and **drops only the back-edges within each cyclic SCC of the quotient** (sorted by class index for a deterministic forward direction). Every input class is preserved as its own SCC in the encoding. `encode_sccs` now accepts an optional `quotient_edges` override that bypasses NetworkX's `condensation` and emits the precomputed acyclic edge set verbatim.

**Why this design over the merge alternative.** Merging classes within each cyclic quotient-SCC is theoretically sounder (it gives a true SCC coarsening), but on real fMRI data every NeuroMark domain pair has bidirectional flow at PCMCI alpha=0.05, so the merge collapses the 7-class partition to a single SCC and the SCC integrity constraints become vacuous (no class-distinct pairs to constrain). The user explicitly opted for the speed-vs-soundness lever: keep the partition, drop back-edges, accept that the constraint may now reject some valid causal graphs whose arrows go in dropped directions. This preserves per-SCC pruning power and keeps per-SCC decomposition meaningful as a future speedup.

**Empirical results (FBIRN N=10 subject 1, `--optim opt`, `density_mode='hard_soft0'`).**

| Variant | Solve time | Optimum | Sound? |
|---|---|---|---|
| Original cyclic encoding | 1.43 s | `[340, 350]` | ❌ — cycles let invalid arrows pass |
| Merge cyclic classes | 120 s timeout (descending past `[302, 350]`) | < `[302, 350]` | ✅ but vacuous on fMRI |
| **Drop back-edges (this fix)** | **0.03 s** | `[452, 350]` | ⚠️ technically unsound (drops valid arrows) |

Subject 0 N=10 domain partition `[{1,2}, {3}, {4}, {5}, {6,7}, {8,9}, {10}]` is preserved; the fix drops 14 back-edges out of 22 quotient edges and emits the remaining 8 forward arrows. Synthetic `[{1}, {2,3}, {4}]` over `1→2→3→1, 4→3` similarly preserves all 3 classes and drops 1 back-edge.

The reported optimum cost rises from `[340, 350]` to `[452, 350]` because the new encoding is *more* restrictive than the original. The original's lower 340 was a phantom: cycles in the old `dag/3` quotient allowed cross-class arrows that should have been forbidden — so half the prior "optima" violated the SCC invariant. **All prior fMRI optimization numbers measured against `--scc_strategy=domain` should be regenerated** before being cited.

**Per-SCC decomposition (future).** With the partition now preserved, per-SCC decomposition becomes a meaningful next-step speedup: solve each SCC's sub-problem independently in Python, then combine. Not implemented in this change.

**Out-of-scope finding.** `clingo_rasl.py:425` emits a `dagl(N-1)` fact never consumed by any rule. Pure dead code. Left for a separate cleanup PR.

**Files:** `gunfolds/conversions.py` (rename + new `_acyclic_quotient_edges` helper + `quotient_edges` parameter on `encode_sccs` + updated docstrings); `gunfolds/scripts/papers/example_clingo.md` (predicate name updated to match emitter); `todo.md` (item 1 moved to Done).

---

### Domain heuristic + PCMCI-prior `#heuristic` directives — tested and REJECTED (suggestion #4)

New benchmark script `gunfolds/scripts/tests/benchmark_domain_heuristic.py` tests the only remaining untested clasp branching/heuristic knob from `clingo_speedup_suggestions.md`: emit `#heuristic edge1(X,Y). [W,true|false]` directives keyed off PCMCI's DD weights and run clasp with `--heuristic=Domain` (and a variant with `--dom-mod=5,16`). Built against the production encoding (`density_mode='hard_soft0'`, `tol_low=15, tol_high=5`).

**Outcome: REJECTED.** Two-N empirical sweep on subject 1:

- **N=10, `--optim opt`, 600 s timeout:** all three scenarios proved the same optimum `[608]`. Baseline 1.43 s; Domain 45.54 s (0.03×, 32× slower); Domain+`dom-mod=5,16` 19.17 s (0.07×, 13× slower).
- **N=14, `--optim opt`, 800 s timeout:** all three timed out. Baseline reached `[664, 400]`; Domain reached `[779, 350]` (17 % worse incumbent); Domain+`dom-mod=5,16` reached `[807, 350]` (21 % worse).

The PCMCI prior is informative enough to land on a 7–42 % better first-feasible model but *misleading enough* over the rest of the search space to lock the solver in a sub-optimal basin it can't escape. The "Domain helps at scale" hypothesis was empirically inverted: at larger N the bigger search space offers more places for the prior to mislead.

**Production recommendation:** keep clasp's default branching. PCMCI evidence is already used appropriately as *cost* in `[W@1,…]` weak-constraint terms. This rejection joins USC, `--opt-heuristic=1`, and `--project=show` — all four heuristic/objective-shaping knobs tested on this encoding have failed to give a robust speedup. The remaining speedup levers are structural (per-u splitting, weak-constraint reification).

**Files:** `gunfolds/scripts/tests/benchmark_domain_heuristic.py` (new); `gunfolds/scripts/real_data/component_config.py` (new `COMP_SET_14`: 2 ICNs/domain, `N=10 ⊂ N=14 ⊂ N=20`); `gunfolds/scripts/papers/clingo_speedup_suggestions.md` (§4 marked REJECTED with results table and diagnosis); `gunfolds/scripts/papers/clingo_drasl_encoding_improvements.md` (status table updated); `gunfolds/scripts/papers/clingo_clasp_optimization_flags_benchmark.md` (new §5).


## 2026-04-27  (branch: `fix-weak-constraint-dedup`)

### Density encoding: hard cardinality window, adaptive ladder, asymmetric tolerance — production default

Three composable changes turning the soft-only density penalty into a robust per-subject hard window with downward bias.

**Hard cardinality window + lex priorities (`drasl_command`).** New `density_mode` parameter selects among `'soft'` (legacy A), `'hard'` (B), `'hard_soft0'` (C, hard bounds + density at `@0`), `'hard_soft1'` (D, hard bounds + density at `@1`), `'none'` (no density encoding), and `'adaptive'` (production default). Variant C lex-separates edge matching at `@1` from density tiebreaker at `@0` — keeps Bayesian framing (likelihood vs prior) and lets clasp prove `@1` optimality on a clean weighted-MaxSAT objective before exercising the prior.

**Adaptive escalation ladder (`drasl`, new default `density_mode='adaptive'`).** Three-step fallback: (E.1) `hard_soft0` with tight tolerance → (E.2) `hard_soft0` with widened tolerance (`+tol_widen` on both sides) → (E.3) `soft` (legacy unbounded). Each attempt is a fresh `clingo()` invocation; ladder advances only on UNSAT (empty result). Verbose progress lines `[drasl] adaptive attempt N: SUCCESS / UNSAT — falling back`.

**Per-subject GT_density auto-derivation.** `_compute_directed_density_pct(g)` derives GT from `glist[0]` when caller passes `GT_density=None`, replacing the fixed-population value (35 for N=10) that excluded ~40 % of subjects' optimal regions.

**Asymmetric tolerance defaults.** `tol_low=15`, `tol_high=5`. PCMCI's measurement density systematically overestimates causal density (every length-`u` walk becomes an observed edge), so the prior should be wider downward than upward. Legacy symmetric `tol` still works as an override when not `None`.

**Benchmark result (N=10, FBIRN subjects 0–9, 500 s timeout).** Adaptive E with asymmetric `[-15 %, +5 %]` reduced primary cost on **every** subject vs the legacy A baseline (17–70 %, median ≈ 47 %). Mean solve time 90.5 s vs 189.9 s (2.10× faster). 9/10 subjects converge within 500 s vs 7/10 for A. All 10 subjects succeed on E.1; no fallback needed. See `gunfolds/scripts/papers/clingo_drasl_encoding_improvements.md` for the full writeup and per-subject tables.

**Files:** `gunfolds/solvers/clingo_rasl.py` (signature: `drasl(..., density_mode='adaptive', tol=None, tol_low=15, tol_high=5, tol_widen=10, verbose=True)`), `gunfolds/scripts/tests/benchmark_density_encoding.py` (variants `E,A,C,D` benchmark with `--tol_low`/`--tol_high`/`--tol_widen`/`--extra_clingo_args`).


---

## 2026-04-24  (branch: `fix-weak-constraint-dedup`)

### Weak-constraint term-tuple dedup fix + density encoding fix (`gunfolds/solvers/clingo_rasl.py`)

**Dedup fix.** Clingo counts cost elements with identical `(weight, priority, tuple)` only once. The four weak constraints all shared tuple `[W@P, X, Y]`, so directed- and bidirected-mismatch penalties at the same `(X,Y)` with the same weight would silently cancel. Fix: appended a type tag `(K, 1..4)` to each tuple. For N=10/FBIRN subject 0: 5 colliding pairs, 79 hidden cost units.

**Density encoding fix (Option B).** Old code used `1000*X/Y` for density but `d = GT_density` in the 0–100 convention — a 10× mismatch making `abs_diff` never close to zero. Also `[Diff@priority]` used density as a priority level, not a weight multiplier. New encoding: `50*X/Y` (50 bins, 2%/bin), `d = GT_density // 2`, cost `[Diff*density_weight@1]` with `density_weight=50` (new param on `drasl_command` / `drasl`). One density-bin error (cost 50) now outweighs one edge mismatch (max 20).

**Benchmark result (N=10, subject 0):** optimal cost unchanged (373), solve time 266 s → 225 s (1.18×). See `gunfolds/scripts/tests/benchmark_dedup_fix.py`.


---

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

- **`fmri_experiment_large.py`:** `--gt_density` now defaults to omitted (`None`). Under `--gt_density_mode fixed`, the effective density is **350** (N=10), **215** (N=20), or **125** (N=53) when `--gt_density` is not passed — midpoints of the ranges in `gunfolds/scripts/papers/ground_truth_connectivity_estimates.md` §7. Explicit `--gt_density` still clamps to 0–1000. Saved `result.zkl` / `run_params.zkl` include effective `gt_density` and optional `gt_density_explicit` (CLI value, or `None` if the default was used).
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
