"""
Benchmark: weak-constraint term-tuple dedup fix.

Compares two ASP encodings on the same PCMCI-derived fMRI input:

  OLD  :~ directed(X,Y,L), no_hdirected(X,Y,W,K), ..., u(L,K). [W@P, X, Y]
  NEW  :~ directed(X,Y,L), no_hdirected(X,Y,W,K), ..., u(L,K). [W@P, X, Y, K, 1]
  (and analogously for the other three weak constraints)

The OLD encoding silently deduplicates any two cost elements that share
(weight, priority, X, Y) — even when they come from different constraint
types (e.g. directed-mismatch and bidirected-mismatch at the same node pair
with the same weight).  The NEW encoding appends a type tag (1..4) and the
graph index K so every source is unique.

What we check:
  1. Cost disagreement  — old vs new report different optimal costs,
                          confirming the bug was live.
  2. Collision count    — how many (X,Y) pairs had matching weights that
                          triggered the dedup.
  3. Solution difference — the optimal *graph* returned may differ once the
                           cost landscape is corrected.
  4. Timing             — whether the fix changes solve time.

Usage:
  # Quick: N=10, subject 0
  python benchmark_dedup_fix.py --n_components 10 --subject_idx 0

  # Multiple subjects
  python benchmark_dedup_fix.py --n_components 10 --subject_idx 0,1,2 --timeout 300

  # Override data path
  python benchmark_dedup_fix.py --data_path /path/to/fbirn_sz_data.npz
"""

import os
import sys
import time
import argparse
import threading
from datetime import datetime
from string import Template

import numpy as np
import clingo as clngo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "real_data"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gunfolds.utils import bfutils
from gunfolds import conversions as cv
from gunfolds.solvers.clingo_rasl import drasl_command
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.utils import graphkit as gk
from gunfolds.conversions import drasl_jclingo2g

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from component_config import get_comp_indices, get_comp_names, get_scc_members

CLINGO_LIMIT = 64
MAXCOST = 20
DEFAULT_GT_DENSITY_BY_N = {10: 35, 20: 22, 53: 13}


# ─────────────────────────────────────────────────────────────────────────────
# Build OLD encoding (buggy term tuples) inline for direct comparison
# ─────────────────────────────────────────────────────────────────────────────

_OLD_WEAK_TEMPLATE = Template("""
    {edge1(X,Y)} :- node(X), node(Y).
    directed(X, Y, 1) :- edge1(X, Y).
    directed(X, Y, L) :- directed(X, Z, L-1), edge1(Z, Y), L <= U, u(U, _).
    bidirected(X, Y, U) :- directed(Z, X, L), directed(Z, Y, L), node(X;Y;Z), X < Y, L < U, u(U, _).

    :~ directed(X, Y, L), no_hdirected(X, Y, W, K), node(X;Y), u(L, K). [W@$$p,X,Y]
    :~ bidirected(X, Y, L), no_hbidirected(X, Y, W, K), node(X;Y), u(L, K), X < Y. [W@$$p,X,Y]
    :~ not directed(X, Y, L), hdirected(X, Y, W, K), node(X;Y), u(L, K). [W@$$p,X,Y]
    :~ not bidirected(X, Y, L), hbidirected(X, Y, W, K), node(X;Y), u(L, K), X < Y. [W@$$p,X,Y]

""")


def _old_weak_program(priority=1):
    """Return the OLD (buggy) weak-constraint block at given priority."""
    return _OLD_WEAK_TEMPLATE.substitute(p=priority)


def build_command_with_custom_weak(g_list, max_urate, dm, bdm, GT_density,
                                   scc, scc_members, weak_program_str,
                                   selfloop=False, density_weight=50):
    """
    Re-assemble an ASP command using a caller-supplied weak-constraint block.
    Mirrors drasl_command() but lets us swap in the old or new weak section.
    """
    from gunfolds.conversions import (clingo_preamble, numbered_g2wclingo,
                                      encode_list_sccs)

    assert len({len(g) for g in g_list}) == 1

    n = len(g_list)
    command = clingo_preamble(g_list[0])

    if GT_density is not None:
        command += f"#const d = {GT_density}. "
        command += ("countedge1(C):- C = #count { edge1(X, Y): "
                    "edge1(X, Y), node(X), node(Y)}. ")
        command += "countfull(C):- C = n*n. "
        command += ("hypoth_density(D) :- D = 100*X/Y, "
                    "countfull(Y), countedge1(X). ")
        command += "abs_diff(Diff) :- hypoth_density(D), Diff = |D - d|. "
        command += f":~ abs_diff(Diff). [Diff*{density_weight}@1] "

    if scc and scc_members:
        command += encode_list_sccs(g_list, scc_members)

    command += f"dagl({len(g_list[0])-1}). "

    for count, (g, D, B) in enumerate(zip(g_list, dm, bdm)):
        command += numbered_g2wclingo(
            g, count + 1,
            directed_weights_matrix=D.astype('int'),
            bidirected_weights_matrix=B.astype('int')) + ' '

    command += 'uk(1..'+str(max_urate)+').' + ' '

    from gunfolds.solvers.clingo_rasl import drate
    command += drate(max_urate, 1, weighted=True) + ' '

    command += weak_program_str

    # drasl convergence / non-empty rules (copied from drasl_program)
    command += """
    pastfork(X,Y,L) :- directed(Z, X, K), directed(Z, Y, K), node(X;Y;Z), X < Y, K < L-1, uk(K), u(L, _).
    notequal(L-1,L) :- bidirected(X,Y,L), not pastfork(X,Y,L), node(X;Y), X < Y, u(L, _).
    notequal(K,L) :- directed(X, Y, K), not directed(X, Y, L), node(X;Y), K<L, uk(K), u(L,_).
    notequal(K,L) :- not directed(X, Y, K), directed(X, Y, L), node(X;Y), K<L, uk(K), u(L,_).
    :- not notequal(K,L), K<L, uk(K), u(L,_).
    nonempty(L) :- directed(X, Y, L), u(L,_).
    nonempty(L) :- bidirected(X, Y, L), u(L,_).
    :- not nonempty(L), u(L,_).
    """

    if not selfloop:
        command += ":-  edge1(X, X), node(X). "
    else:
        command += ":- not edge1(X, X), node(X). "

    command += ":- u(L,A), u(T,B), not T=L, A<B. "
    command += ":- not edge1(_, _). "
    command += "#show edge1/2. "
    command += "#show u/2."

    return command.encode().replace(b"\n", b" ")


# ─────────────────────────────────────────────────────────────────────────────
# Grounding observer
# ─────────────────────────────────────────────────────────────────────────────

class GroundingObserver:
    def __init__(self):
        self.atoms = 0
        self.rules = 0
        self.weight_rules = 0
        self.minimize_stmts = 0

    def init_program(self, incremental): pass
    def begin_step(self): pass
    def end_step(self): pass
    def rule(self, choice, head, body): self.rules += 1
    def weight_rule(self, choice, head, lower_bound, body): self.weight_rules += 1
    def minimize(self, priority, literals): self.minimize_stmts += 1
    def project(self, atoms): pass
    def output_atom(self, symbol, atom): self.atoms += 1
    def output_term(self, symbol, condition): pass
    def external(self, atom, value): pass
    def assume(self, literals): pass
    def heuristic(self, atom, type_, bias, priority, condition): pass
    def acyc_edge(self, node_u, node_v, condition): pass
    def theory_term_number(self, term_id, number): pass
    def theory_term_string(self, term_id, name): pass
    def theory_term_compound(self, term_id, name_id_or_type, arguments): pass
    def theory_element(self, element_id, terms, condition): pass
    def theory_atom(self, atom_id_or_zero, term_id, elements): pass
    def theory_atom_with_guard(self, atom_id_or_zero, term_id, elements,
                               operator_id, right_hand_side_id): pass

    def summary(self):
        return {"atoms": self.atoms, "rules": self.rules,
                "weight_rules": self.weight_rules,
                "minimize_stmts": self.minimize_stmts}


# ─────────────────────────────────────────────────────────────────────────────
# Run one scenario
# ─────────────────────────────────────────────────────────────────────────────

def _sg(obj, key, default=0):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError):
        return getattr(obj, 'get', lambda k, d: d)(key, default)


def run_scenario(command, label, pnum, capsize, optim, timeout,
                 configuration, n_nodes):
    """Run clingo and return timing + solution stats."""
    clingo_args = [
        "--warn=no-atom-undefined",
        f"--configuration={configuration}",
        "-t", f"{int(pnum)},split",
        "-n", str(capsize),
    ]

    print(f"\n{'#'*70}", flush=True)
    print(f"  SCENARIO: {label}", flush=True)
    print(f"{'#'*70}", flush=True)

    ctrl = clngo.Control(clingo_args)
    ctrl.configuration.solve.opt_mode = optim

    observer = GroundingObserver()
    ctrl.register_observer(observer)

    ctrl.add("base", [], command.decode())
    t0 = time.time()
    ctrl.ground([("base", [])])
    t_ground = time.time() - t0
    gs = observer.summary()
    print(f"  Grounded in {t_ground:.2f}s  "
          f"atoms={gs['atoms']:,}  rules={gs['rules']:,}  "
          f"weight_rules={gs['weight_rules']:,}  "
          f"minimize={gs['minimize_stmts']:,}", flush=True)

    t0 = time.time()
    models = []
    best_cost = None
    timed_out = False
    timer = None

    if timeout and timeout > 0:
        def _interrupt():
            nonlocal timed_out
            timed_out = True
            ctrl.interrupt()
        timer = threading.Timer(timeout, _interrupt)
        timer.start()

    try:
        with ctrl.solve(yield_=True, async_=True) as handle:
            for model in handle:
                cost = model.cost
                atoms = [str(a) for a in model.symbols(shown=True)]
                models.append((atoms, cost))
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    elapsed = time.time() - t0
                    print(f"  [t={elapsed:.1f}s] NEW BEST cost={cost}",
                          flush=True)
    finally:
        if timer is not None:
            timer.cancel()

    t_solve = time.time() - t0

    stats = ctrl.statistics
    summary = _sg(stats, "summary", {})
    costs = _sg(summary, "costs", [])
    models_stats = _sg(summary, "models", {})
    n_optimal = int(_sg(models_stats, "optimal", 0))
    n_enumerated = int(_sg(models_stats, "enumerated", 0))
    times = _sg(summary, "times", {})
    t_clingo = _sg(times, "total", 0)
    solving = _sg(stats, "solving", {})
    solvers = _sg(solving, "solvers", {})
    solver0 = _sg(solvers, 0, _sg(solvers, "0", {}))
    choices = int(_sg(solver0, "choices", 0))
    conflicts = int(_sg(solver0, "conflicts", 0))

    # Parse solutions
    solutions = []
    if models:
        parsed = {(drasl_jclingo2g(m[0]), sum(m[1])) for m in models}
        for (gnum, u), cost_val in parsed:
            res_cg = bfutils.num2CG(gnum, n_nodes)
            solutions.append((cost_val, gnum, u, gk.density(res_cg)))
        solutions.sort()

    print(f"  Solve: {t_solve:.2f}s  clingo_total: {t_clingo:.2f}s  "
          f"models_returned={len(models)}  enumerated={n_enumerated}  "
          f"optimal={n_optimal}  timed_out={timed_out}", flush=True)
    print(f"  Best cost (clingo): {list(costs)}", flush=True)
    print(f"  Choices={choices:,}  Conflicts={conflicts:,}", flush=True)
    if solutions:
        print(f"  Top-5 solutions: "
              f"{[(s[0], s[2], round(s[3],3)) for s in solutions[:5]]}",
              flush=True)

    return {
        "label": label,
        "t_ground": round(t_ground, 3),
        "t_solve": round(t_solve, 3),
        "t_clingo": round(t_clingo, 3),
        "n_models": len(models),
        "n_enumerated": n_enumerated,
        "n_optimal": n_optimal,
        "best_cost": list(costs),
        "choices": choices,
        "conflicts": conflicts,
        "ground_atoms": gs["atoms"],
        "ground_rules": gs["rules"],
        "ground_weight_rules": gs["weight_rules"],
        "ground_minimize": gs["minimize_stmts"],
        "solutions": solutions,
        "timed_out": timed_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Collision analysis: count (X,Y) pairs whose dir and bidir weights collide
# ─────────────────────────────────────────────────────────────────────────────

def analyze_collisions(DD, BD, g_estimated, n_nodes):
    """
    For each X<Y pair, determine what weight each side of the mismatch would
    carry (depending on whether the edge is in the measured graph or not) and
    count how many pairs have matching weights across constraint types.

    Returns a list of (X, Y, w_dir, w_bidir, type_d, type_b) for each collision.
    """
    adj = cv.graph2adj(g_estimated)   # N x N, 1 if directed edge
    badj = cv.graph2badj(g_estimated) # N x N symmetric, 1 if bidirected

    collisions = []
    for x in range(n_nodes):
        for y in range(n_nodes):
            # directed side: if adj[x,y]==1 → no_hdirected weight = DD[x,y]
            #                if adj[x,y]==0 → hdirected weight    = DD[x,y]
            w_dir = int(DD[x, y])
            type_d = "no_hdir" if adj[x, y] == 1 else "hdir_missing"

            if x >= y:
                continue  # bidirected only for x < y

            # bidirected side (x < y): if badj[x,y]==1 → no_hbidirected = BD[x,y]
            #                          if badj[x,y]==0 → hbidirected     = BD[x,y]
            w_bidir = int(BD[x, y])
            type_b = "no_hbidir" if badj[x, y] == 1 else "hbidir_missing"

            if w_dir == w_bidir and w_dir > 0:
                collisions.append((x + 1, y + 1, w_dir, w_bidir,
                                   type_d, type_b))
    return collisions


# ─────────────────────────────────────────────────────────────────────────────
# Per-subject driver
# ─────────────────────────────────────────────────────────────────────────────

def process_subject(args, data, labels, subject_idx, comp_indices,
                    gt_density, n_nodes):
    ts_2d = data[subject_idx][:, comp_indices]
    print(f"\n  Subject {subject_idx}: shape {ts_2d.shape}, "
          f"group={int(labels[subject_idx])}", flush=True)

    # PCMCI
    t0 = time.time()
    dataframe = pp.DataFrame(ts_2d)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(
        tau_max=args.pcmci_tau_max, pc_alpha=None,
        alpha_level=args.pcmci_alpha, fdr_method=args.pcmci_fdr)
    g_estimated, A, B = cv.Glag2CG(results)
    t_pcmci = time.time() - t0
    print(f"  PCMCI done in {t_pcmci:.2f}s  "
          f"density={gk.density(g_estimated):.3f}", flush=True)

    # SCC
    import networkx as nx
    scc_members = get_scc_members(
        args.scc_strategy, comp_indices, ts_2d, max_cluster_size=8)
    use_scc = scc_members is not None

    # Distance matrices
    a_max = np.abs(A).max()
    b_max = np.abs(B).max()
    DD = (np.abs((np.abs(A / a_max if a_max > 0 else A)
                  + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
    BD = (np.abs((np.abs(B / b_max if b_max > 0 else B)
                  + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)
    print(f"  DD range [{DD.min()}, {DD.max()}]  "
          f"BD range [{BD.min()}, {BD.max()}]", flush=True)

    # Collision analysis
    collisions = analyze_collisions(DD, BD, g_estimated, n_nodes)
    n_xy_pairs = n_nodes * (n_nodes - 1) // 2
    print(f"\n  === Collision analysis (X<Y pairs) ===", flush=True)
    print(f"  Total X<Y pairs: {n_xy_pairs}", flush=True)
    print(f"  Pairs with matching dir/bidir weights (potential dedup): "
          f"{len(collisions)}", flush=True)
    if collisions:
        hidden_cost = sum(c[2] for c in collisions)
        print(f"  Hidden cost if all collide: {hidden_cost}", flush=True)
        print(f"  First 10 collisions (X, Y, weight, dir_type, bidir_type):",
              flush=True)
        for c in collisions[:10]:
            print(f"    ({c[0]},{c[1]})  w={c[2]}  "
                  f"dir=[{c[4]}]  bidir=[{c[5]}]", flush=True)

    urate = min(args.MAXU, 3 * n_nodes + 1)

    # ── Build OLD command (buggy term tuples) ──
    old_weak = _old_weak_program(priority=1)
    cmd_old = build_command_with_custom_weak(
        [g_estimated], urate, [DD], [BD],
        GT_density=gt_density, scc=use_scc, scc_members=scc_members,
        weak_program_str=old_weak, selfloop=False, density_weight=50)

    # ── Build NEW command (fixed term tuples, from patched drasl_command) ──
    cmd_new = drasl_command(
        [g_estimated], max_urate=urate, weighted=True,
        scc=use_scc, scc_members=scc_members,
        dm=[DD], bdm=[BD],
        GT_density=gt_density, selfloop=False, density_weight=50)

    print(f"\n  ASP sizes — old: {len(cmd_old):,} bytes  "
          f"new: {len(cmd_new):,} bytes", flush=True)

    pnum = min(CLINGO_LIMIT, get_process_count(1))

    results_old = run_scenario(
        cmd_old, label="OLD (buggy term tuples)",
        pnum=pnum, capsize=args.capsize, optim="optN",
        timeout=args.timeout, configuration="crafty", n_nodes=n_nodes)

    results_new = run_scenario(
        cmd_new, label="NEW (fixed term tuples)",
        pnum=pnum, capsize=args.capsize, optim="optN",
        timeout=args.timeout, configuration="crafty", n_nodes=n_nodes)

    return {
        "subject": subject_idx,
        "collisions": collisions,
        "n_xy_pairs": n_xy_pairs,
        "old": results_old,
        "new": results_new,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark weak-constraint term-tuple dedup fix")
    p.add_argument("--n_components", type=int, default=10,
                   choices=[10, 20, 53])
    p.add_argument("--subject_idx", type=str, default="0",
                   help="Comma-separated subject indices, e.g. '0,1,2'")
    p.add_argument("--data_path", type=str,
                   default="../fbirn/fbirn_sz_data.npz")
    p.add_argument("--scc_strategy", type=str, default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--MAXU", type=int, default=5)
    p.add_argument("--gt_density", type=int, default=None)
    p.add_argument("--timeout", type=int, default=0,
                   help="Per-scenario timeout in seconds (0 = no limit)")
    p.add_argument("--capsize", type=int, default=0,
                   help="Max models to return (0 = unlimited)")
    p.add_argument("--pcmci_tau_max", type=int, default=1)
    p.add_argument("--pcmci_alpha", type=float, default=0.05)
    p.add_argument("--pcmci_fdr", default="none")
    args = p.parse_args()

    subject_indices = [int(x.strip()) for x in args.subject_idx.split(",")]
    gt_density = (args.gt_density
                  if args.gt_density is not None
                  else DEFAULT_GT_DENSITY_BY_N.get(args.n_components))

    print("=" * 70, flush=True)
    print("WEAK-CONSTRAINT TERM-TUPLE DEDUP FIX BENCHMARK", flush=True)
    print("=" * 70, flush=True)
    print(f"  Time:         {datetime.now()}", flush=True)
    print(f"  N:            {args.n_components}", flush=True)
    print(f"  Subjects:     {subject_indices}", flush=True)
    print(f"  MAXCOST:      {MAXCOST}", flush=True)
    print(f"  GT density:   {gt_density}", flush=True)
    print(f"  MAXU:         {args.MAXU}", flush=True)
    print(f"  Timeout:      {args.timeout}s" if args.timeout else
          f"  Timeout:      none", flush=True)
    print("=" * 70, flush=True)

    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "real_data",
                                 data_path)
    npzfile = np.load(data_path)
    data = npzfile["data"]
    labels_key = "labels" if "labels" in npzfile.files else "label"
    labels = npzfile[labels_key]
    comp_indices = get_comp_indices(args.n_components)
    n_nodes = len(comp_indices)

    all_results = []
    for s_idx in subject_indices:
        print(f"\n{'='*70}", flush=True)
        print(f"  SUBJECT {s_idx}", flush=True)
        print(f"{'='*70}", flush=True)
        r = process_subject(args, data, labels, s_idx, comp_indices,
                            gt_density, n_nodes)
        all_results.append(r)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n\n{'='*90}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"  N={args.n_components}  MAXCOST={MAXCOST}  "
          f"GT_density={gt_density}  MAXU={args.MAXU}", flush=True)
    print(f"{'='*90}", flush=True)

    hdr = (f"  {'Subj':>4s}  {'Collisions':>10s}  {'Hidden%':>8s}  "
           f"{'OLD cost':>12s}  {'NEW cost':>12s}  {'Cost diff':>10s}  "
           f"{'OLD t(s)':>9s}  {'NEW t(s)':>9s}  {'Speedup':>8s}  "
           f"{'Same soln?':>10s}")
    print(hdr, flush=True)
    print("  " + "-" * (len(hdr) - 2), flush=True)

    for r in all_results:
        s = r["subject"]
        n_col = len(r["collisions"])
        hidden_pct = (100.0 * n_col / r["n_xy_pairs"]
                      if r["n_xy_pairs"] else 0)

        old_c = r["old"]["best_cost"]
        new_c = r["new"]["best_cost"]
        old_sum = sum(old_c) if old_c else None
        new_sum = sum(new_c) if new_c else None
        diff_str = (str(new_sum - old_sum)
                    if old_sum is not None and new_sum is not None else "N/A")

        t_old = r["old"]["t_solve"]
        t_new = r["new"]["t_solve"]
        speedup = (t_old / t_new if t_new > 0 else float("nan"))

        # Do optimal solutions match?
        old_sols = set((s2[1], s2[2]) for s2 in r["old"]["solutions"]
                       if r["old"]["solutions"]
                       and s2[0] == r["old"]["solutions"][0][0])
        new_sols = set((s2[1], s2[2]) for s2 in r["new"]["solutions"]
                       if r["new"]["solutions"]
                       and s2[0] == r["new"]["solutions"][0][0])
        same = "YES" if old_sols == new_sols else "NO"

        print(f"  {s:>4d}  {n_col:>10d}  {hidden_pct:>7.1f}%  "
              f"{str(old_c):>12s}  {str(new_c):>12s}  {diff_str:>10s}  "
              f"{t_old:>9.2f}  {t_new:>9.2f}  {speedup:>7.2f}x  "
              f"{same:>10s}",
              flush=True)

    # ── Per-subject solution detail ───────────────────────────────────────────
    print(f"\n{'='*90}", flush=True)
    print("SOLUTION DETAIL", flush=True)
    print(f"{'='*90}", flush=True)

    for r in all_results:
        s = r["subject"]
        print(f"\n  Subject {s}:", flush=True)
        for variant in ("old", "new"):
            res = r[variant]
            sols = res["solutions"]
            if not sols:
                print(f"    {variant.upper():4s}  NO SOLUTIONS  "
                      f"{'(T/O)' if res['timed_out'] else ''}", flush=True)
                continue
            min_cost = sols[0][0]
            opt_set = [(sv[1], sv[2]) for sv in sols if sv[0] == min_cost]
            print(f"    {variant.upper():4s}  cost={min_cost}  "
                  f"{len(opt_set)} optimal graphs  "
                  f"top-3: {[(sv[0], sv[1], round(sv[3], 3)) for sv in sols[:3]]}",
                  flush=True)

        old_sols = set((sv[1], sv[2]) for sv in r["old"]["solutions"]
                       if r["old"]["solutions"]
                       and sv[0] == r["old"]["solutions"][0][0])
        new_sols = set((sv[1], sv[2]) for sv in r["new"]["solutions"]
                       if r["new"]["solutions"]
                       and sv[0] == r["new"]["solutions"][0][0])
        if old_sols != new_sols:
            only_old = old_sols - new_sols
            only_new = new_sols - old_sols
            shared = old_sols & new_sols
            print(f"    *** SOLUTION SET CHANGED: "
                  f"{len(shared)} shared, "
                  f"{len(only_old)} only in OLD, "
                  f"{len(only_new)} only in NEW ***", flush=True)
        else:
            print(f"    Solution set UNCHANGED after fix.", flush=True)

        if r["collisions"]:
            print(f"    Collisions detected: {len(r['collisions'])} pairs "
                  f"with matching weights.  "
                  f"Total hidden cost: {sum(c[2] for c in r['collisions'])}",
                  flush=True)
        else:
            print(f"    No weight collisions detected for this subject.",
                  flush=True)

    print(f"\nDone at {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
