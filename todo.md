# TODO

A running list of things to investigate or implement when time allows. Add to the top, push completed items to the bottom (or strike through).

---

## Open

### 7. Bayesian evidence-ratio weights for the SCC quotient MFAS (Option D)

**Where:** `_acyclic_quotient_edges` in `gunfolds/conversions.py`. Currently uses Option C from the design discussion — additive `w(K → L) = pos(K → L) + neg(L → K)` weights with `igraph.Graph.feedback_arc_set(method='exact_ip')`.

**Idea.** Replace the additive weight with a log-likelihood-ratio weight:

```
w(K → L) = log[ pos(K → L) / pos(L → K) ]
```

(or a Bayes-factor variant summed across node pairs, or a smoothed version with a small prior to handle the `pos(L → K) = 0` case).

**Why it might be better.** The additive Option C treats the DD weights as additive evidence units; the Bayesian ratio treats them as log-odds. If we ever calibrate DD weights so they're closer to actual log-likelihood-ratios — e.g. by mapping `|partial_correlation_t_statistic|` to a proper p-value-derived score — then Option D becomes the principled choice and Option C becomes a heuristic surrogate.

**Why it's not implemented today.** The current DD recipe `|A_norm| × MAXCOST` is a heuristic transform of partial-correlation magnitudes, not a calibrated probability. Treating these as log-odds without calibration risks distorting the MFAS objective in subtle ways (especially with the `log(0)` degenerate case requiring a small prior). The additive Option C is robust to scaling and works without calibration.

**Action items.**

- Calibrate DD weights against partial-correlation t-statistics (or against a held-out subject's known SCC structure). Validate that the log-ratio form yields meaningful weights.
- Implement as a `weight_strategy` parameter on `_acyclic_quotient_edges` (`'asymmetry'` for current Option C, `'log_ratio'` for Option D).
- A/B test on FBIRN N=10 / N=14: does Option D give lower-evidence-cost drops than Option C? Does it change the encoding's optimum cost on real subjects?

**Reference:** logged in chat 2026-05-04 alongside the implementation of Option C. Detailed write-up in `gunfolds/scripts/papers/scc_quotient_edge_dropping_research.md` (local, gitignored).

---

### 6. Suggestion #9 — drop weight-0 `hdirected` / `no_hdirected` (and bidirected) facts at grounding time

**Where:** `glist2str` (or the helper it calls) in `gunfolds/conversions.py`, and any other site that emits `hdirected/no_hdirected/hbidirected/no_hbidirected` facts based on `dm`/`bdm` matrices. Item from §9 of [`gunfolds/scripts/papers/clingo_speedup_suggestions.md`](gunfolds/scripts/papers/clingo_speedup_suggestions.md).

**Idea.** A weak-constraint instance with weight 0 contributes nothing to the cost regardless of whether it's satisfied or violated, but clingo still grounds the rule and tracks it. For node pairs `(X, Y)` where PCMCI has no signal at all (both `DD[X-1, Y-1] == 0` and `BD[X-1, Y-1] == 0`), the four ground facts are dead weight that bloats the program without affecting the objective.

**Conservative version (recommended first step).** Skip emission of all four families for `(X, Y, K)` whenever `dm[K][X-1, Y-1] == 0` AND `bdm[K][X-1, Y-1] == 0`. Zero risk — these facts genuinely contribute nothing.

**Aggressive variants** (each progressively more aggressive, each requires measurement before adopting):

- **#9-A** — also skip an *individual* family when its weight is 0, even if the other family has nonzero weight. E.g. emit only `hbidirected(...)` if `DD = 0` but `BD > 0`.
- **#9-B** — threshold-based: skip facts with weight `< T` for some cutoff `T` (e.g. 2). Bigger savings, but starts dropping faintly informative constraints; needs a knob to tune.

**Why not done today.** Mostly because we have not actually *measured* whether weight-0 facts are common in our PCMCI output. For FBIRN N=10 with `tau_max=1` and `alpha=0.05`, the DD ranges we have observed are roughly `[4, 20]` and BD `[6, 20]` — no zeros at all in the runs we have looked at. The trick may save 0% on real fMRI runs, in which case it is not worth the implementation. Synthetic / longer-tau runs may behave differently.

**Note about the user's "drop both `hdirected` and `no_hdirected`" recall.** In single-graph mode, only one of `{hdirected(X, Y, W, K), no_hdirected(X, Y, W, K)}` is ever emitted per `(X, Y, K)` — picked by whether `g_estimated[X][Y]` has the directed edge. They are mutually exclusive. Suggestion #9 is *not* about that case; it's about the `DD = BD = 0` case where all four families would carry weight 0. The mutually-exclusive-pair case only arises in multi-graph DRASL where graphs disagree, which we are not currently using.

**Action items.**

- First, *measure*: count how often `DD == 0` AND `BD == 0` per `(X, Y)` pair in production fMRI runs. If never, do not implement.
- If common enough to matter (say >5% of pairs), implement the conservative version: a one-line guard in `glist2str` that skips both pairs of facts when both DD and BD are 0.
- Consider the aggressive variants only after seeing what the conservative version saves.

**Reference:** logged in chat 2026-05-04. Tracking item from `clingo_speedup_suggestions.md` §9 bullet 2.

---

### 5. Checkpoint + warm-restart for long clingo runs that hit timeouts

**Where:** wrapper around the existing solve loop in `gunfolds/scripts/tests/benchmark_domain_heuristic.py` (and any production caller of `drasl()` that has a wall-time budget). Likely lives as a small standalone script `gunfolds/scripts/tests/clingo_with_checkpoint.py` so the checkpoint logic stays orthogonal to other benchmarks.

**Idea.** When a clingo solve hits a timeout (or a cluster preemption), we currently throw away the best incumbent and the next attempt starts from cost ∞. The descent through high-cost models took ~50–60 s on N=14 subject 1 (out of an 800 s budget that timed out without proving optimum). Save the incumbent on every new best, and on resume start a fresh clingo session that knows the previous best as an upper bound. Standard "warm-restart" pattern from MaxSAT competition.

**Two ingredients (clasp supports both natively).**

1. **Cost upper bound** — pass `--opt-bound=C` (or per-priority `--opt-bound=C1,C2` for the `[edge@1, density@0]` vector). Clasp prunes any partial assignment whose lower bound exceeds it. **One scalar per priority level, no model state needed.** Always safe.
2. **Branching seed** (optional, opt-in) — emit `#heuristic edge1(X,Y). [W,true|false]` directives derived from the incumbent atoms, run clasp with `--heuristic=Domain`. The first model in the resumed session typically lands at or near the previous incumbent in the first few decisions.

The cost-bound part alone (#1) is the unambiguous win and should be the default. The branching seed (#2) is the same mechanism rejected as suggestion #4 — but used here from a *known feasible incumbent* rather than the noisy PCMCI prior, which is qualitatively different. Worth re-testing as an opt-in flag, with the awareness that it could re-introduce the "biases backtracking, hurts proof" failure mode in a different costume.

**What is NOT possible.** Clasp does not serialize internal solver state — clause database, conflict trail, restart phase, VSIDS scores. Each new `Control()` is a fresh search. "Resume from checkpoint" really means "warm-start hint to a fresh search," not "pick up where you left off." This means the proof-phase budget after warm restart is a fresh budget (not cumulative). The win is purely on the descent: skip the time spent finding any model worse than the incumbent.

**Hard guard rails.**

- **Never include the incumbent's edges as hard constraints** (`:- not edge1(X,Y).` for each true edge). If the true optimum requires removing one of those edges, you have made it unreachable and silently sub-optimal. Hard-pinning is a tempting shortcut and breaks soundness — only use the soft Domain heuristic.
- The cost bound must be passed correctly for the *full* priority vector. Passing `--opt-bound=778` when the cost is `[778, 350]` is ambiguous and may be interpreted as a single-priority bound. Always pass the comma-separated form matching the vector length.
- Atomic checkpoint writes (write to `checkpoint.json.tmp`, then `os.replace`) so a SIGTERM mid-write doesn't corrupt the file.

**Design sketch.**

```
class CheckpointWriter:
    def __init__(self, path):
        self.path = path
        self.best_cost = None
    def maybe_save(self, cost, atoms):
        if self.best_cost is None or cost < self.best_cost:
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"cost": list(cost), "atoms": atoms,
                           "ts": time.time()}, f)
            os.replace(tmp, self.path)
            self.best_cost = cost

# In the solve loop:
ckpt = CheckpointWriter(args.checkpoint_path)
for model in handle:
    cost = list(model.cost)
    atoms = [str(a) for a in model.symbols(shown=True)]
    ckpt.maybe_save(cost, atoms)

# On resume:
def resume_args(checkpoint_path, with_heuristic_seed=False):
    data = json.load(open(checkpoint_path))
    cost = data["cost"]
    bound = ",".join(str(c - 1) for c in cost)   # strict improvement
    extra = [f"--opt-bound={bound}"]
    program_addendum = ""
    if with_heuristic_seed:
        extra.append("--heuristic=Domain")
        true_edges = parse_edge1_atoms(data["atoms"])
        program_addendum = build_heuristic_block_from_incumbent(
            true_edges, n_nodes)
    return extra, program_addendum
```

**CLI flags to add to a new `clingo_with_checkpoint.py`.**

- `--checkpoint_path PATH` (default `<output_dir>/checkpoint.json`)
- `--resume_from PATH` — read cost + atoms, set `--opt-bound`, optionally seed.
- `--warm_seed_from_checkpoint` (flag, default off) — also emit Domain-heuristic seed from the incumbent atoms.
- `--auto_resume_chain N` — run N consecutive sessions of `--timeout` each, automatically chaining checkpoint between them. Useful for cluster jobs with strict per-job time limits.

**Open design questions to resolve when implementing.**

- **Per-priority bound semantics.** `--opt-bound=778,349` enforces strict improvement on edge cost AND on density. Sometimes desirable (force progress on both axes), sometimes not (we want to keep density and just improve edge cost). Reasonable default: pass the bound from the previous best with `-1` only on the highest priority (`@1`), keep `@0` at the previous value (allow same density). Worth measuring both.
- **Where to store checkpoints in production.** Same directory as benchmark logs is fine for one-off experiments. For SLURM jobs, the SCRATCH dir convention. Should match whatever pattern existing `fmri_experiment_large.py` already uses.
- **Should `drasl()` itself learn to write checkpoints, or only the orchestration layer?** Lean toward orchestration only — `drasl_command` should stay a pure encoding builder, and checkpointing is solver-loop concern.

**Action items.**

1. Implement cost-bound-only first as a standalone script. Measure on N=14 subject 1: does a 200 s session followed by a 200 s warm-resume reach a better incumbent than a single 400 s fresh session?
2. If (1) shows a win, add the optional Domain-heuristic seed as a flag. A/B test whether seeding helps further or re-introduces the suggestion #4 failure pattern.
3. If both win, integrate with the SLURM job scripts so cluster preemption automatically chains checkpoints.

**Reference:** raised in chat 2026-05-04 after the GPU-acceleration discussion.

---

### 4. GPU acceleration of the clingo optimization — revisit when GPU SAT/MaxSAT tooling matures

**Where:** the clingo solving step inside `drasl()` / `drasl_command` (currently parallelised via clasp's `-t N,split` or `-t N,compete` across CPU threads).

**Question.** Today clingo runs on CPUs. A single subject at N=14 already pegs ~10 cores for tens of seconds with diminishing returns past ~16 threads. Could we plug in a GPU somewhere to get the kind of 10–100× wall-time win that GPUs deliver for, e.g., neural network training?

**Current assessment (2026-05-04): no clear path with present-day tooling, revisit later.**

The hot loop inside clasp is CDCL — Conflict-Driven Clause Learning. Each step is: pick a branching variable, propagate via watched literals (pointer-chasing through linked lists), on conflict analyse and learn a clause, backtrack to the right level, repeat. Per-step work is tiny, memory access is highly irregular, and every step depends on the previous one. The `-t N,split` / `compete` modes are *coarse-grained* parallelism: each thread runs an independent CDCL solver and they share learned clauses. They do not parallelise the per-step work.

GPUs need (a) regular memory access, (b) high arithmetic intensity, (c) the same operation over millions of elements. CDCL is the opposite on all three axes. Branch divergence inside a warp would be near-total. A small academic literature exists on GPU SAT solvers (e.g. *clauSPaR*, *ParaFROST-GPU*); they generally lose to a single modern CPU solver on industrial benchmarks. Running 10 000 GPU-resident solvers in portfolio mode tends to net out as 10 000 solvers each at 1/1000th of CPU speed.

**What this argument does NOT cover (and why it's worth revisiting later).**

- The argument applies to *complete* CDCL/Branch-and-Bound. *Incomplete* GPU MaxSAT solvers based on parallel local search (simulated annealing, parallel tempering) are an active research area and could in principle scale. They would not produce optimality proofs but might match clingo's "best feasible" within seconds.
- NVIDIA's *cuOpt* (GPU-accelerated optimisation, currently focused on routing/ILP) keeps adding solver families. If they ever add weighted MaxSAT or pseudo-boolean optimisation, plugging it in becomes worth measuring.
- Tensor-network / GBP-style approaches to constraint satisfaction map naturally to GPUs and have shown promise on structured problems. The DRASL encoding is highly structured (per-undersampling, per-SCC), so it could be a good candidate.
- Quantum-inspired annealers (D-Wave, Fujitsu Digital Annealer, NEC SX-Aurora) are a separate hardware path; they target QUBO and weighted MaxSAT directly. Worth re-evaluating every 12–18 months.

**What we *would* implement first if GPU acceleration ever made sense.**

- **GPU PCMCI.** Partial correlation tests are dense linear algebra and well-suited to GPUs. This is independent of the clingo question and would speed up the whole pipeline; trivial to prototype with `cupy`.
- **GPU brute-force as an oracle for small N (N ≤ 6).** A CUDA kernel could enumerate all `2^(N²)` candidate edge sets in minutes and serve as ground truth for verifying the clingo solver. Useful for correctness validation, not a production solver.

**What we should NOT do.**

- Port CDCL to CUDA. Would take weeks and produce a slower solver.
- Switch to a GPU-only ILP encoding without measuring against the existing CPU clingo first.

**Higher-priority alternatives that should land first.**

- **Per-SCC Python decomposition** (item 3 below). Expected 10–100× wall-time win on N≥14, ≈1–2 days of engineering. This dominates anything GPU work could plausibly deliver and should be done before any GPU exploration.
- **Cluster job parallelism.** One subject per cluster node with `-t 64,compete` gives linear scaling across subjects with no code change.
- **Tighter density tolerance** (`tol_low=8, tol_high=3`). Cheap configuration tweak that prunes the cardinality search by ~2–5× on hard subjects.

**Action items.**

- Re-evaluate this entry every ~18 months or whenever a credible GPU MaxSAT / weighted-PB solver ships (NVIDIA cuOpt updates, academic releases, etc.).
- Before spending serious time on a GPU port, confirm that per-SCC decomposition (item 3) and density-tolerance tuning are already in production.
- If a candidate GPU solver appears, validate first on the small-N brute-force oracle to make sure it returns the same optimum as clingo.

**Reference:** raised in chat 2026-05-04 after delivering the SCC encoding fix on branch `scc-edge-acyclic-quotient`.

---

### 3. Per-SCC Python-level decomposition of the DRASL optimization

**Where:** Orchestration layer above `drasl_command` / `drasl` in `gunfolds/solvers/clingo_rasl.py`, plus the existing SCC-aware encoding in `gunfolds/conversions.py` (where `_acyclic_quotient_edges` already preserves the partition).

**Idea.** Once the user-supplied SCC partition is preserved as distinct classes (delivered in commit `0079ca60` on branch `scc-edge-acyclic-quotient`), we can split the global optimization into one independent sub-problem per SCC and solve them in parallel. Each sub-problem is exponentially smaller in the candidate-edge space, so the combined wall time should drop by another order of magnitude beyond the in-encoding pruning win we already measured (47× on N=10 subject 1).

**Decomposition sketch.**

1. Split nodes by SCC class: ``n_classes`` sub-problems, each over the nodes inside one class.
2. Build a per-class measurement view: restrict ``g_estimated``, ``DD``, and ``BD`` to the rows/columns of nodes inside the class.
3. Solve each per-class DRASL instance independently (parallel processes / threads), each finds its own optimum causal sub-graph.
4. Merge: the combined causal graph is the union of within-class edges from each sub-problem plus the *forward* cross-class edges that are witnessed in `scc_edge` (these are determined by the DAG order, not optimization variables).

**Open design questions.**

- **Cross-SCC `hdirected` / `hbidirected` weights.** Each one is a measurement-scale fact about a pair (X, Y) where X and Y can be in different classes. Naively, this couples two sub-problems. Options: (a) treat cross-class measurement weights as a separate small post-processing step that decides whether to include each cross-class edge, given the within-class solutions; (b) include them as boundary constraints in both sub-problems and re-solve if they disagree; (c) ignore them inside sub-problems and recover them from the post-merge graph using the existing `bfutils.undersample` machinery. Option (a) is cleanest if the DAG-order constraint already determines most cross-class edges.

- **Undersampling closure across classes.** The encoding's `directed(X, Y, L)` rule chains length-L paths through *any* nodes — including nodes in different classes. Splitting by class breaks this chain. Probably fine for within-class subproblems (they only enumerate paths through their own nodes) but cross-class paths need separate handling. Verify on a small N=10 case.

- **Density window decomposition.** The hard `[GT-tol_low, GT+tol_high]` cardinality constraint is over the *whole* graph. We need either a per-class proportional sub-budget or a single post-merge feasibility check. Per-class proportional is simpler but may over-constrain on uneven class sizes.

- **`MAXU` and `u/2` choice.** The undersampling rate is a single global decision. Either fix it across all sub-problems (probably what we want) or treat it as a top-level loop with K sub-problems per u-value.

**Why it matters.**

- Per-SCC independence is a structural decomposition that VSIDS / clasp branching cannot exploit on its own — it requires Python-side orchestration.
- If each sub-problem is small enough (say, ≤4 nodes per SCC at N=14), each one solves in well under a second even with `optN`. Combined wall time stays bounded as N grows, instead of exponential.
- This is the single biggest speedup lever we have not yet pulled. It complements (not replaces) all the encoding-level fixes.

**Action items.**

- Prototype on a 2-SCC synthetic case first (e.g. two disjoint 5-node graphs joined by one cross-edge) to validate the decomposition recovers the same optimum as the monolithic solve.
- Then apply to N=14 fMRI subject 1 (same subject we benchmarked, where SCC pruning now actually fires) and compare wall time and optimum cost vs. the monolithic baseline.
- If validated, add a `decompose_by_scc=True` flag to `drasl()` and benchmark across N=10/14/20.

**Reference:** spun out from chat 2026-05-04 ("Reading 2: Python-level decomposition") after delivering the in-encoding SCC fix on branch `scc-edge-acyclic-quotient`.

---

### 2. Why does clingo report `choices = 0`, `conflicts = 0`, `restarts = 0` in every run?

**Where:** `gunfolds/scripts/tests/benchmark_density_encoding.py` (the `_sg(...)` stats extraction in `run_variant`), and any other script that reads `ctrl.statistics["solving"]["solvers"]…`.

**Symptom:** Every single benchmark run prints `choices=0  conflicts=0  restarts=0`, even when the solver clearly did substantial work (35+ improving models found, optimality proven, etc.). These are core CDCL solver counters — seeing zero is impossible if solving happened.

**Hypothesis:** The stats extraction path is wrong for multi-threaded runs. With `-t {pnum},split`, clasp arranges the stats tree as:

```
solving:
  solvers:
    choices:    <aggregate sum across threads>
    conflicts:  <aggregate>
    restarts:   <aggregate>
    0:                       ← per-thread node
      choices: ...
      ...
    1: { ... }
    ...
```

The current code does `_sg(solvers, 0, _sg(solvers, "0", {}))` then reads `.choices` from that — likely throwing an exception silently caught by `_sg`, returning the `0` default. The aggregate counters at `solving.solvers.choices` (one level up from the per-thread node) are probably what we want.

**Why it matters:**

- Without correct counters we cannot empirically attribute speed differences between variants A/C/D/E to grounding size vs search-tree size vs conflict learning.
- USC's failure mode (high conflicts, slow bound improvement) is invisible.
- Plateau-enumeration cost (many choices, ~0 conflicts) is invisible.
- Any future paper claiming "X is faster because Y" needs these numbers to back it up.

**Action items:**

- Dump the full stats tree from one run to see where choices/conflicts actually live:
  ```python
  import json
  def _dump(s):
      if hasattr(s, "keys"):
          return {k: _dump(s[k]) for k in s.keys()}
      if hasattr(s, "__len__") and not isinstance(s, str):
          try: return [_dump(s[i]) for i in range(len(s))]
          except Exception: return str(s)
      return s
  print(json.dumps(_dump(ctrl.statistics), indent=2)[:5000])
  ```
- Replace the per-thread lookup with the aggregate path: `solvers.choices`, `solvers.conflicts`, `solvers.restarts`.
- Optionally also report per-thread breakdown by iterating numeric keys under `solvers`.
- Re-run the existing benchmarks once fixed and back-fill the proper numbers in `gunfolds/scripts/papers/clingo_drasl_encoding_improvements.md`.

**Reference:** flagged in the analytical review of the 10-subject Variant E run (2026-04-27) — explicitly noted as "unfixed" in §4.7 of the encoding-improvements paper.

---

## Done

### 1. Investigate the cycle in `dag/3` facts emitted by `encode_list_sccs` — **resolved 2026-05-04: rename + acyclic-quotient via back-edge dropping**

**Root cause.** `encode_sccs` (in `gunfolds/conversions.py`) calls
`networkx.algorithms.components.condensation(G, scc=SCCS)`. NetworkX's
`condensation` requires `scc` to be a *partition* of the nodes, but does
**not** enforce that the elements are actually strongly connected. When
`scc_members` comes from `--scc_strategy=domain` / `correlation`, the
partition may *split* a real SCC across multiple classes, and the
"condensation" is a generic quotient digraph with cycles — making the SCC
integrity constraints reject some valid cross-class arrows and accept some
invalid ones.

**Design decision: drop back-edges, keep all classes.** Two ways to make the
quotient acyclic:

1. **Merge** any classes that lie in the same SCC of the quotient
   (theoretically sound; restores a true SCC coarsening).
2. **Drop back-edges** within each cyclic SCC of the quotient (technically
   unsound — may reject valid arrows in dropped directions — but preserves
   the user's intended class granularity).

Approach (1) collapses the 7-class NeuroMark domain partition to a single
SCC on real fMRI data because every domain pair has bidirectional flow at
PCMCI alpha=0.05; the constraint then becomes vacuous and all SCC pruning
power is lost. The user explicitly chose approach (2) to preserve per-class
pruning and keep per-SCC decomposition meaningful as a future speedup.

**Fix applied.**

1. Renamed predicate `dag/3` → `scc_edge/3` (3 sites in
   `gunfolds/conversions.py`).
2. Added new helper `_acyclic_quotient_edges(glist, partition)` — builds
   the quotient over the union of `glist`, finds its SCCs, picks a
   deterministic within-SCC class order (sorted by class index), and
   filters edges so only forward arrows survive. Returns
   `(triples, n_dropped)`.
3. Added a `quotient_edges` override parameter to `encode_sccs` so callers
   can bypass `condensation` and emit a precomputed acyclic edge set.
4. `encode_list_sccs` now precomputes the acyclic edges via the helper and
   passes per-graph slices to `encode_sccs`; prints a one-line stdout
   report when back-edges are actually dropped.
5. Docstrings updated on all three functions.
6. `gunfolds/scripts/papers/example_clingo.md` updated to match the
   emitter.

**Empirical verification.**

| Test | Input | Output |
|---|---|---|
| Synthetic | partition `[{1}, {2,3}, {4}]` over `1→2, 2→3, 3→1, 4→3` | preserved 3 classes; dropped 1 back-edge; `scc_edge` acyclic |
| Real fMRI subject 0 N=10 | domain partition `[{1,2}, {3}, {4}, {5}, {6,7}, {8,9}, {10}]` | preserved 7 classes; dropped 14 back-edges; 8 forward `scc_edge` facts; acyclic |
| Real fMRI subject 1 N=10 baseline | `--optim opt` | **0.03 s** to optimum `[452, 350]` (was 1.43 s for `[340, 350]` under unsound cyclic encoding) |

The new optimum (452) is *higher* than the prior 340 because the new
encoding is more restrictive: cycles in the old quotient allowed cross-class
arrows that should have been forbidden, so the prior 340 was a phantom
optimum produced by an unsound encoding. **All prior fMRI optimization
numbers measured against `--scc_strategy=domain` should be regenerated
before being cited.**

**Future work.** Per-SCC decomposition (solve each class's sub-problem
independently in Python, then combine) is now meaningful since the
partition is preserved. Not implemented in this change.

**Out-of-scope finding (separate cleanup):** `clingo_rasl.py:425` emits a
`dagl(N-1)` fact that is never consumed by any rule in the project — pure
dead code. Should be deleted in a separate change.
