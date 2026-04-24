"""
Benchmark all 8 combinations of three clingo solver flags on real fMRI data.

Flags tested:
  A) --opt-strategy=usc,stratify   (core-guided optimization vs default branch-and-bound)
  B) --opt-heuristic=1             (optimization-aware branching vs default VSIDS)
  C) --project=show                (project enumeration to shown atoms vs full models)

All 2^3 = 8 combinations are run on the SAME grounded ASP program (PCMCI +
distance matrices are computed once, the ASP command is built once, and then
each scenario gets its own clingo Control instance).

Uses the same instrumentation as diagnose_rasl_bottleneck.py: grounding
observer, per-model callbacks, and full clingo statistics.

Usage (on cluster or local machine with clingo installed):

  # Quick: N=10, subject 0 (should finish in minutes total)
  python benchmark_clingo_flags.py --n_components 10 --subject_idx 0

  # Heavier: N=20, subject 0 (may take hours for some combos)
  python benchmark_clingo_flags.py --n_components 20 --subject_idx 0 --timeout 3600

  # Limit threads
  python benchmark_clingo_flags.py --n_components 10 --subject_idx 0 --PNUM 4
"""

import os
import sys
import time
import argparse
import itertools
import threading
from datetime import datetime, timedelta

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

from gunfolds.scripts.real_data.component_config import (
    get_comp_indices, get_comp_names, get_scc_members,
)

CLINGO_LIMIT = 64
MAXCOST = 20
DEFAULT_GT_DENSITY_BY_N = {10: 35, 20: 22, 53: 13}


# ─────────────────────────────────────────────────────────────────────────────
# Grounding observer (same as diagnose_rasl_bottleneck.py)
# ─────────────────────────────────────────────────────────────────────────────

class GroundingObserver:
    def __init__(self, report_interval=5.0):
        self.atoms = 0
        self.rules = 0
        self.weight_rules = 0
        self.minimize_stmts = 0
        self._start = time.time()
        self._last_report = self._start
        self._interval = report_interval

    def _maybe_report(self):
        now = time.time()
        if now - self._last_report >= self._interval:
            elapsed = now - self._start
            print(f"    [GROUNDING {elapsed:8.1f}s]  atoms={self.atoms:,}  "
                  f"rules={self.rules:,}  weight_rules={self.weight_rules:,}  "
                  f"minimize={self.minimize_stmts}", flush=True)
            self._last_report = now

    def init_program(self, incremental): pass
    def begin_step(self): pass
    def end_step(self): pass
    def rule(self, choice, head, body):
        self.rules += 1; self._maybe_report()
    def weight_rule(self, choice, head, lower_bound, body):
        self.weight_rules += 1; self._maybe_report()
    def minimize(self, priority, literals):
        self.minimize_stmts += 1
    def project(self, atoms): pass
    def output_atom(self, symbol, atom):
        self.atoms += 1; self._maybe_report()
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
        elapsed = time.time() - self._start
        return {
            "elapsed_s": round(elapsed, 2),
            "atoms": self.atoms,
            "rules": self.rules,
            "weight_rules": self.weight_rules,
            "minimize_stmts": self.minimize_stmts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Run one scenario
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(command, scenario_label, extra_args, capsize, configuration,
                 pnum, optim, timeout, grounding_interval, n_nodes=10):
    """
    Run clingo with a specific set of extra flags and return timing + stats
    plus the parsed solution set (cost, graph number, undersampling).
    """
    base_args = [
        "--warn=no-atom-undefined",
        f"--configuration={configuration}",
        "-t", f"{int(pnum)},split",
        "-n", str(capsize),
    ]
    clingo_args = base_args + extra_args

    print(f"\n{'#'*70}", flush=True)
    print(f"  SCENARIO: {scenario_label}", flush=True)
    print(f"{'#'*70}", flush=True)
    print(f"  clingo args: {clingo_args}", flush=True)
    print(f"  optim={optim}  capsize={capsize}  threads={pnum}", flush=True)

    ctrl = clngo.Control(clingo_args)
    ctrl.configuration.solve.opt_mode = optim

    observer = GroundingObserver(report_interval=grounding_interval)
    ctrl.register_observer(observer)

    # Phase 1: Add program
    t0 = time.time()
    ctrl.add("base", [], command.decode())
    t_add = time.time() - t0

    # Phase 2: Grounding
    print(f"\n  [GROUNDING]...", flush=True)
    t0 = time.time()
    ctrl.ground([("base", [])])
    t_ground = time.time() - t0
    ground_stats = observer.summary()
    print(f"    Done in {t_ground:.2f}s  "
          f"(atoms={ground_stats['atoms']:,}  rules={ground_stats['rules']:,}  "
          f"weight_rules={ground_stats['weight_rules']:,})", flush=True)

    # Phase 3: Solving (with real timeout via ctrl.interrupt())
    timeout_str = f"{timeout}s" if timeout else "none"
    print(f"\n  [SOLVING] (optim={optim}, timeout={timeout_str})...", flush=True)
    t0 = time.time()
    models = []
    model_count = 0
    best_cost = None
    timed_out = False

    timer = None
    if timeout and timeout > 0:
        def _interrupt():
            nonlocal timed_out
            timed_out = True
            print(f"\n    *** TIMEOUT ({timeout}s) — interrupting solver ***",
                  flush=True)
            ctrl.interrupt()
        timer = threading.Timer(timeout, _interrupt)
        timer.start()

    try:
        with ctrl.solve(yield_=True, async_=True) as handle:
            for model in handle:
                model_count += 1
                cost = model.cost
                optimality = model.optimality_proven
                atoms = [str(a) for a in model.symbols(shown=True)]
                models.append((atoms, cost))

                elapsed = time.time() - t0
                improved = ""
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    improved = " ** NEW BEST **"

                if model_count <= 20 or optimality or improved:
                    cost_str = f"cost={cost}" if cost else "cost=[]"
                    print(f"    [MODEL #{model_count} at {elapsed:.1f}s]  "
                          f"{cost_str}  optimal={optimality}  "
                          f"atoms={len(atoms)}{improved}", flush=True)

                if model_count == 21:
                    print(f"    ... (suppressing further model-by-model output, "
                          f"will report summary) ...", flush=True)
    finally:
        if timer is not None:
            timer.cancel()

    t_solve = time.time() - t0
    if timed_out:
        print(f"    Solver interrupted after {t_solve:.1f}s  "
              f"(best-so-far cost: {best_cost})", flush=True)

    # Phase 4: Statistics
    def _sg(obj, key, default=0):
        """Safe get from clingo statistics objects (support [] and .get)."""
        if obj is None:
            return default
        try:
            return obj[key]
        except (KeyError, IndexError, TypeError):
            if hasattr(obj, 'get'):
                return obj.get(key, default)
            return default

    stats = ctrl.statistics
    summary = _sg(stats, "summary", {})
    solving = _sg(stats, "solving", {})
    solvers = _sg(solving, "solvers", {})

    costs = _sg(summary, "costs", [])
    models_stats = _sg(summary, "models", {})
    n_optimal = _sg(models_stats, "optimal", 0)
    n_enumerated = _sg(models_stats, "enumerated", 0)

    times = _sg(summary, "times", {})
    total_time = _sg(times, "total", 0)
    solve_time = _sg(times, "solve", 0)
    sat_time = _sg(times, "sat", 0)
    unsat_time = _sg(times, "unsat", 0)

    solver0 = _sg(solvers, 0, _sg(solvers, "0", {}))
    choices = _sg(solver0, "choices", 0)
    conflicts = _sg(solver0, "conflicts", 0)
    restarts = _sg(solver0, "restarts", 0)

    print(f"\n  [RESULT]", flush=True)
    print(f"    Grounding:     {t_ground:10.2f}s", flush=True)
    print(f"    Solving:       {t_solve:10.2f}s", flush=True)
    print(f"    clingo total:  {total_time:10.2f}s  "
          f"(solve={solve_time:.2f}  sat={sat_time:.2f}  "
          f"unsat={unsat_time:.2f})", flush=True)
    print(f"    Models:        enumerated={n_enumerated}  "
          f"optimal={n_optimal}  returned={model_count}", flush=True)
    print(f"    Best cost:     {costs}", flush=True)
    print(f"    Solver stats:  choices={choices:,.0f}  "
          f"conflicts={conflicts:,.0f}  restarts={restarts:,.0f}", flush=True)

    # Phase 5: Parse solutions
    solutions = []
    if models:
        r_estimated = {(drasl_jclingo2g(m[0]), sum(m[1])) for m in models}
        for answer in r_estimated:
            graph_num = answer[0][0]
            undersampling = answer[0][1]
            cost = answer[1]
            res_cg = bfutils.num2CG(graph_num, n_nodes)
            solutions.append((cost, graph_num, undersampling,
                              gk.density(res_cg)))
        solutions.sort(key=lambda x: x[0])
        print(f"    Unique solutions: {len(solutions)}  "
              f"(top-5 costs: {[s[0] for s in solutions[:5]]})", flush=True)

    return {
        "scenario": scenario_label,
        "extra_args": extra_args,
        "t_ground": round(t_ground, 3),
        "t_solve": round(t_solve, 3),
        "clingo_total": round(total_time, 3),
        "clingo_solve": round(solve_time, 3),
        "clingo_sat": round(sat_time, 3),
        "clingo_unsat": round(unsat_time, 3),
        "n_models_returned": model_count,
        "n_enumerated": int(n_enumerated),
        "n_optimal": int(n_optimal),
        "best_cost": costs,
        "choices": int(choices),
        "conflicts": int(conflicts),
        "restarts": int(restarts),
        "ground_atoms": ground_stats["atoms"],
        "ground_rules": ground_stats["rules"],
        "solutions": solutions,
        "timed_out": timed_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark 8 combinations of clingo solver flags on fMRI data")
    p.add_argument("--n_components", type=int, default=10, choices=[10, 20, 53])
    p.add_argument("--subject_idx", type=int, default=0)
    p.add_argument("--data_path", type=str, default="../fbirn/fbirn_sz_data.npz",
                   help="Path to fbirn_sz_data.npz (relative to real_data/)")
    p.add_argument("--scc_strategy", type=str, default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--PNUM", type=int,
                   default=int(min(CLINGO_LIMIT, get_process_count(1))))
    p.add_argument("--MAXU", type=int, default=5)
    p.add_argument("--PRIORITY", type=str, default="11112")
    p.add_argument("--gt_density", type=int, default=None,
                   help="Override GT_density (0-100); defaults to literature value for N")
    p.add_argument("--timeout", type=int, default=0,
                   help="Per-scenario timeout in seconds (0=no limit)")
    p.add_argument("--capsize", type=int, default=0,
                   help="Max models to return (0=unlimited)")
    p.add_argument("--pcmci_method", default="pcmci")
    p.add_argument("--pcmci_tau_max", type=int, default=1)
    p.add_argument("--pcmci_alpha", type=float, default=0.05)
    p.add_argument("--pcmci_fdr", default="none")
    p.add_argument("--grounding_interval", type=float, default=5.0)
    p.add_argument("--skip", type=str, default="",
                   help="Comma-separated scenario numbers to skip (e.g. '0,2,4,6')")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated scenario numbers to run (e.g. '1,3,5,7')")
    args = p.parse_args()

    skip_set = set(int(x) for x in args.skip.split(",") if x.strip())
    only_set = set(int(x) for x in args.only.split(",") if x.strip())

    gt_density = args.gt_density
    if gt_density is None:
        gt_density = DEFAULT_GT_DENSITY_BY_N.get(args.n_components)

    print("=" * 70, flush=True)
    print("CLINGO SOLVER FLAGS BENCHMARK", flush=True)
    print("=" * 70, flush=True)
    print(f"  Time:           {datetime.now()}", flush=True)
    print(f"  N components:   {args.n_components}", flush=True)
    print(f"  Subject index:  {args.subject_idx}", flush=True)
    print(f"  SCC strategy:   {args.scc_strategy}", flush=True)
    print(f"  MAXU:           {args.MAXU}", flush=True)
    print(f"  MAXCOST:        {MAXCOST}", flush=True)
    print(f"  GT density:     {gt_density}", flush=True)
    print(f"  Clingo threads: {args.PNUM}", flush=True)
    print(f"  Timeout/scen:   {args.timeout}s" if args.timeout else
          f"  Timeout/scen:   none", flush=True)
    print(f"  Capsize:        {args.capsize}", flush=True)
    print(f"  PCMCI:          {args.pcmci_method} tau_max={args.pcmci_tau_max}"
          f" alpha={args.pcmci_alpha}", flush=True)
    print("=" * 70, flush=True)

    # ── Load data ──
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "real_data",
                                 data_path)
    print(f"\n[SETUP] Loading data from {data_path}...", flush=True)
    npzfile = np.load(data_path)
    data = npzfile["data"]
    labels_key = "labels" if "labels" in npzfile.files else "label"
    labels = npzfile[labels_key]
    print(f"  Data shape: {data.shape}", flush=True)

    comp_indices = get_comp_indices(args.n_components)
    comp_names = get_comp_names(comp_indices)
    n_nodes = len(comp_indices)

    s = args.subject_idx
    ts_2d = data[s][:, comp_indices]
    print(f"  Subject {s}: shape {ts_2d.shape}, group={int(labels[s])}",
          flush=True)

    # ── PCMCI ──
    print(f"\n[SETUP] Running PCMCI...", flush=True)
    t0 = time.time()
    dataframe = pp.DataFrame(ts_2d)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    if args.pcmci_method == "pcmciplus":
        results = pcmci.run_pcmciplus(
            tau_max=args.pcmci_tau_max, pc_alpha=0.01,
            fdr_method=args.pcmci_fdr)
    else:
        results = pcmci.run_pcmci(
            tau_max=args.pcmci_tau_max, pc_alpha=None,
            alpha_level=args.pcmci_alpha, fdr_method=args.pcmci_fdr)
    g_estimated, A, B = cv.Glag2CG(results)
    t_pcmci = time.time() - t0
    density = gk.density(g_estimated)
    n_dir = sum(1 for n in g_estimated for t in g_estimated[n]
                if g_estimated[n][t] in (1, 3))
    n_bidir = sum(1 for n in g_estimated for t in g_estimated[n]
                  if g_estimated[n][t] in (2, 3) and n < t)
    print(f"  PCMCI done in {t_pcmci:.2f}s: {n_nodes} nodes, "
          f"{n_dir} directed, {n_bidir} bidirected, density={density:.3f}",
          flush=True)

    # ── SCC ──
    import networkx as nx
    scc_members = get_scc_members(
        args.scc_strategy, comp_indices, ts_2d, max_cluster_size=8)
    use_scc = scc_members is not None
    if args.scc_strategy == "estimated":
        scc_members = list(
            nx.strongly_connected_components(gk.graph2nx(g_estimated)))
        use_scc = True
    if use_scc and scc_members:
        print(f"  SCCs ({len(scc_members)} groups): "
              f"{[len(s) for s in scc_members]}", flush=True)

    # ── Distance matrices ──
    a_max = np.abs(A).max()
    b_max = np.abs(B).max()
    if a_max > 0:
        DD = (np.abs((np.abs(A / a_max) + (cv.graph2adj(g_estimated) - 1))
                     * MAXCOST)).astype(int)
    else:
        DD = (np.abs((cv.graph2adj(g_estimated) - 1) * MAXCOST)).astype(int)
    if b_max > 0:
        BD = (np.abs((np.abs(B / b_max) + (cv.graph2badj(g_estimated) - 1))
                     * MAXCOST)).astype(int)
    else:
        BD = (np.abs((cv.graph2badj(g_estimated) - 1) * MAXCOST)).astype(int)
    print(f"  DD range: [{DD.min()}, {DD.max()}]  "
          f"BD range: [{BD.min()}, {BD.max()}]", flush=True)

    # ── Build ASP program (once) ──
    priority = [int(c) for c in args.PRIORITY]
    urate = min(args.MAXU, (3 * n_nodes + 1))
    command = drasl_command(
        [g_estimated], max_urate=urate, weighted=True,
        scc=use_scc, scc_members=scc_members,
        dm=[DD], bdm=[BD], edge_weights=priority,
        GT_density=gt_density, selfloop=False,
    )
    print(f"  ASP program: {len(command):,} bytes", flush=True)

    # ── Define 8 scenarios ──
    FLAG_A = ("--opt-strategy=usc,stratify", "usc")
    FLAG_B = ("--opt-heuristic=1", "optheur")
    FLAG_C = ("--project=show", "proj")

    flags = [FLAG_A, FLAG_B, FLAG_C]
    flag_labels = ["usc,stratify", "opt-heuristic=1", "project=show"]

    all_results = []

    for combo in itertools.product([False, True], repeat=3):
        parts = []
        extra_args = []
        for on, (arg, short) in zip(combo, flags):
            if on:
                parts.append(short)
                extra_args.append(arg)
        label = "+".join(parts) if parts else "baseline"

        scenario_num = sum(v << i for i, v in enumerate(combo))
        label = f"S{scenario_num}: {label}"

        if only_set and scenario_num not in only_set:
            print(f"\n  [SKIP] {label} (not in --only)", flush=True)
            continue
        if scenario_num in skip_set:
            print(f"\n  [SKIP] {label} (in --skip)", flush=True)
            continue

        result = run_scenario(
            command=command,
            scenario_label=label,
            extra_args=extra_args,
            capsize=args.capsize,
            configuration="crafty",
            pnum=args.PNUM,
            optim="optN",
            timeout=args.timeout,
            grounding_interval=args.grounding_interval,
            n_nodes=n_nodes,
        )
        all_results.append(result)

    # ── Summary table ──
    print(f"\n\n{'='*90}", flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"  N={args.n_components}  subject={args.subject_idx}  "
          f"MAXCOST={MAXCOST}  GT_density={gt_density}  "
          f"MAXU={args.MAXU}  threads={args.PNUM}", flush=True)
    print(f"{'='*90}", flush=True)

    header = (f"{'Scenario':<35s} {'Solve(s)':>10s} {'Total(s)':>10s} "
              f"{'Models':>8s} {'Optimal':>8s} {'Cost':>12s} "
              f"{'Choices':>10s} {'Conflicts':>10s} {'Note':>6s}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for r in all_results:
        cost_str = str(r["best_cost"]) if r["best_cost"] else "[]"
        if len(cost_str) > 12:
            cost_str = cost_str[:11] + "~"
        note = "T/O" if r.get("timed_out") else ""
        print(f"{r['scenario']:<35s} "
              f"{r['t_solve']:>10.2f} "
              f"{r['clingo_total']:>10.2f} "
              f"{r['n_models_returned']:>8d} "
              f"{r['n_optimal']:>8d} "
              f"{cost_str:>12s} "
              f"{r['choices']:>10,d} "
              f"{r['conflicts']:>10,d} "
              f"{note:>6s}",
              flush=True)

    print(f"\n{'='*90}", flush=True)

    # ── Flag-by-flag effect ──
    baseline = all_results[0]
    print(f"\nFlag-by-flag speedup vs baseline (solve time):", flush=True)
    for r in all_results[1:]:
        if r["t_solve"] > 0 and baseline["t_solve"] > 0:
            speedup = baseline["t_solve"] / r["t_solve"]
            print(f"  {r['scenario']:<35s}  {speedup:>6.2f}x", flush=True)
        else:
            print(f"  {r['scenario']:<35s}  N/A", flush=True)

    # ── Verify solutions agree ──
    print(f"\nSOLUTION COMPARISON ACROSS SCENARIOS", flush=True)
    print(f"{'-'*90}", flush=True)

    # Compare optimal costs
    costs_set = set()
    for r in all_results:
        c = tuple(r["best_cost"]) if r["best_cost"] else ()
        costs_set.add(c)
    if len(costs_set) == 1:
        print(f"  Optimal cost:  ALL SAME  {all_results[0]['best_cost']}",
              flush=True)
    else:
        print(f"  WARNING: DIFFERENT optimal costs!", flush=True)
        for r in all_results:
            print(f"    {r['scenario']}: {r['best_cost']}", flush=True)

    # Compare solution sets (graph_num, undersampling) at optimal cost
    print(f"\n  Per-scenario solution details:", flush=True)
    baseline_graphs = None
    for r in all_results:
        sols = r["solutions"]
        n_sol = len(sols)
        if n_sol == 0:
            print(f"    {r['scenario']:<35s}  NO SOLUTIONS", flush=True)
            continue

        min_cost = sols[0][0]
        optimal_sols = [(s[1], s[2]) for s in sols if s[0] == min_cost]
        graph_set = set(optimal_sols)

        top5 = sols[:min(5, n_sol)]
        top5_str = ", ".join(
            f"(cost={s[0]} g={s[1]} u={s[2]} d={s[3]:.3f})"
            for s in top5
        )
        print(f"    {r['scenario']:<35s}  {n_sol:>4d} solutions  "
              f"{len(optimal_sols):>3d} optimal  top-5: {top5_str}",
              flush=True)

        if baseline_graphs is None:
            baseline_graphs = graph_set
        else:
            if graph_set == baseline_graphs:
                print(f"      -> optimal set MATCHES baseline", flush=True)
            else:
                only_here = graph_set - baseline_graphs
                only_base = baseline_graphs - graph_set
                shared = graph_set & baseline_graphs
                print(f"      -> DIFFERS from baseline: "
                      f"{len(shared)} shared, "
                      f"{len(only_here)} only here, "
                      f"{len(only_base)} only in baseline", flush=True)

    print(f"\nDone at {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
