# TODO

A running list of things to investigate or implement when time allows. Add to the top, push completed items to the bottom (or strike through).

---

## Open

### 1. Investigate the cycle in `dag/3` facts emitted by `encode_list_sccs`

**Where:** `gunfolds/conversions.py` (the `encode_list_sccs` function) → ASP facts emitted into the base command built by `drasl_command`.

**Symptom:** The `dag(K, L, gnum)` facts are supposed to encode an SCC-level DAG (the name says so), but the emitted facts contain cycles. Concrete example observed in a single-subject N=10 run:

```
dag(0, 2, 1).
dag(2, 5, 1).
dag(5, 0, 1).      ← closes a 3-cycle 0 → 2 → 5 → 0
dag(0, 1, 1).
dag(5, 1, 1).
```

If the relation is a true DAG (as the name suggests), this is a bug in `encode_list_sccs`. If it is actually a transitive-closure / reachability relation (and the name is misleading), the constraint logic in `drasl_command` that consumes it still works — but the name should change and the docstring should clarify.

**Why it matters:** the constraint

```clingo
:- directed(X,Y,U), scc(X,K), scc(Y,L), K != L,
   sccsize(L,Z), Z > 1, not dag(K,L,N), u(U,N).
```

uses `not dag(K,L,N)` as a NAF guard. If `dag` is a true DAG, this rule excludes "back-edges" relative to a topological order. If `dag` is reachability, it excludes any cross-SCC edge to a non-reachable SCC. The two semantics give different optimal graphs in some cases.

**Action items:**

- Read `encode_list_sccs` and confirm whether the cycle is intended.
- If unintended → fix the function and re-benchmark on FBIRN N=10 to confirm cost numbers don't shift.
- If intended → rename the relation (e.g. `scc_reach/3`) or at minimum add a docstring/comment in both `encode_list_sccs` and the `drasl_command` rule that consumes it.

**Reference:** noticed during the N=10 / `selfloop=None` benchmark on 2026-04-28 while reviewing the base ASP command for subject 0.

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

*(nothing yet)*
