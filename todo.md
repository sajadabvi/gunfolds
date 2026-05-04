# TODO

A running list of things to investigate or implement when time allows. Add to the top, push completed items to the bottom (or strike through).

---

## Open

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
