"""
Benchmark all 6 clingo solver configurations on real fMRI data.

Configurations tested:
  - frumpy  : Conservative defaults
  - jumpy   : Aggressive defaults
  - tweety  : Defaults geared towards ASP problems
  - handy   : Defaults geared towards large problems
  - crafty  : Defaults geared towards crafted problems (current default)
  - trendy  : Defaults geared towards industrial problems

Optionally also tests 'auto' (clingo picks per-problem) and 'many' (portfolio
of multiple strategies in parallel).

All configurations are run on the SAME grounded ASP program (PCMCI + distance
matrices computed once, ASP command built once; each scenario gets a fresh
clingo Control).

Usage:
  # Quick: N=10, subject 0
  python benchmark_clingo_configs.py --n_components 10 --subject_idx 0

  # With timeout per config (recommended for N=20)
  python benchmark_clingo_configs.py --n_components 20 --subject_idx 0 --timeout 600

  # Test on multiple subjects for robustness
  python benchmark_clingo_configs.py --n_components 10 --subject_idx 0,1,2

  # Only run specific configs
  python benchmark_clingo_configs.py --n_components 10 --subject_idx 0 --only crafty,jumpy,trendy
"""

import os
import sys
import time
import argparse
import threading
from datetime import datetime

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

ALL_CONFIGS = ["frumpy", "jumpy", "tweety", "handy", "crafty", "trendy",
               "auto", "many"]


# ─────────────────────────────────────────────────────────────────────────────
# Grounding observer
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
                  f"rules={self.rules:,}  weight_rules={self.weight_rules:,}",
                  flush=True)
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
        }


# ─────────────────────────────────────────────────────────────────────────────
# Run one configuration
# ─────────────────────────────────────────────────────────────────────────────

def _sg(obj, key, default=0):
    """Safe get from clingo statistics objects."""
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError):
        if hasattr(obj, 'get'):
            return obj.get(key, default)
        return default


def run_config(command, config_name, pnum, capsize, optim, timeout,
               grounding_interval, n_nodes):
    """Run clingo with a specific configuration and return timing + solutions."""

    clingo_args = [
        "--warn=no-atom-undefined",
        f"--configuration={config_name}",
        "-t", f"{int(pnum)},split",
        "-n", str(capsize),
    ]

    print(f"\n{'#'*70}", flush=True)
    print(f"  CONFIG: {config_name}", flush=True)
    print(f"{'#'*70}", flush=True)
    print(f"  threads={pnum}  optim={optim}  capsize={capsize}  "
          f"timeout={timeout if timeout else 'none'}", flush=True)

    ctrl = clngo.Control(clingo_args)
    ctrl.configuration.solve.opt_mode = optim

    observer = GroundingObserver(report_interval=grounding_interval)
    ctrl.register_observer(observer)

    # Grounding
    t0 = time.time()
    ctrl.add("base", [], command.decode())
    t_add = time.time() - t0

    print(f"\n  [GROUNDING]...", flush=True)
    t0 = time.time()
    ctrl.ground([("base", [])])
    t_ground = time.time() - t0
    ground_stats = observer.summary()
    print(f"    Done in {t_ground:.2f}s  "
          f"(atoms={ground_stats['atoms']:,}  rules={ground_stats['rules']:,})",
          flush=True)

    # Solving with timeout
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
            print(f"\n    *** TIMEOUT ({timeout}s) — interrupting ***",
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

                if model_count <= 10 or optimality or improved:
                    cost_str = f"cost={cost}" if cost else "cost=[]"
                    print(f"    [MODEL #{model_count} at {elapsed:.1f}s]  "
                          f"{cost_str}  optimal={optimality}  "
                          f"atoms={len(atoms)}{improved}", flush=True)

                if model_count == 11:
                    print(f"    ... (suppressing further output) ...",
                          flush=True)
    finally:
        if timer is not None:
            timer.cancel()

    t_solve = time.time() - t0
    if timed_out:
        print(f"    Interrupted after {t_solve:.1f}s  "
              f"(best-so-far: {best_cost})", flush=True)

    # Statistics
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

    # Time to reach optimum (first model with optimality_proven or best cost)
    t_to_best = None
    for i, (atoms, cost) in enumerate(models):
        if best_cost is not None and sum(cost) == sum(costs) if costs else False:
            t_to_best = None
            break

    print(f"\n  [RESULT]", flush=True)
    print(f"    Grounding:     {t_ground:10.2f}s", flush=True)
    print(f"    Solving:       {t_solve:10.2f}s", flush=True)
    print(f"    clingo total:  {total_time:10.2f}s  "
          f"(solve={solve_time:.2f}  sat={sat_time:.2f}  "
          f"unsat={unsat_time:.2f})", flush=True)
    print(f"    Models:        enumerated={n_enumerated}  "
          f"optimal={n_optimal}  returned={model_count}", flush=True)
    print(f"    Best cost:     {costs}", flush=True)

    # Parse solutions
    solutions = []
    if models:
        r_estimated = {(drasl_jclingo2g(m[0]), sum(m[1])) for m in models}
        for answer in r_estimated:
            graph_num = answer[0][0]
            undersampling = answer[0][1]
            cost_val = answer[1]
            res_cg = bfutils.num2CG(graph_num, n_nodes)
            solutions.append((cost_val, graph_num, undersampling,
                              gk.density(res_cg)))
        solutions.sort(key=lambda x: x[0])
        print(f"    Unique solutions: {len(solutions)}", flush=True)

    return {
        "config": config_name,
        "t_ground": round(t_ground, 3),
        "t_solve": round(t_solve, 3),
        "clingo_total": round(total_time, 3),
        "clingo_solve": round(solve_time, 3),
        "clingo_unsat": round(unsat_time, 3),
        "n_models_returned": model_count,
        "n_enumerated": int(n_enumerated),
        "n_optimal": int(n_optimal),
        "best_cost": costs,
        "choices": int(choices),
        "conflicts": int(conflicts),
        "solutions": solutions,
        "timed_out": timed_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Build ASP command for one subject
# ─────────────────────────────────────────────────────────────────────────────

def build_asp_command(data, labels, subject_idx, comp_indices, comp_names,
                      args, gt_density):
    """Run PCMCI and build the ASP command for one subject."""
    n_nodes = len(comp_indices)
    ts_2d = data[subject_idx][:, comp_indices]

    print(f"\n  Subject {subject_idx}: shape {ts_2d.shape}, "
          f"group={int(labels[subject_idx])}", flush=True)

    # PCMCI
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
    print(f"  PCMCI done in {t_pcmci:.2f}s: density={density:.3f}", flush=True)

    # SCC
    import networkx as nx
    scc_members = get_scc_members(
        args.scc_strategy, comp_indices, ts_2d, max_cluster_size=8)
    use_scc = scc_members is not None
    if args.scc_strategy == "estimated":
        scc_members = list(
            nx.strongly_connected_components(gk.graph2nx(g_estimated)))
        use_scc = True

    # Distance matrices
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

    # ASP command
    priority = [int(c) for c in args.PRIORITY]
    urate = min(args.MAXU, (3 * n_nodes + 1))
    command = drasl_command(
        [g_estimated], max_urate=urate, weighted=True,
        scc=use_scc, scc_members=scc_members,
        dm=[DD], bdm=[BD], edge_weights=priority,
        GT_density=gt_density, selfloop=False,
    )
    print(f"  ASP program: {len(command):,} bytes", flush=True)
    return command


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark clingo configurations on fMRI data")
    p.add_argument("--n_components", type=int, default=10, choices=[10, 20, 53])
    p.add_argument("--subject_idx", type=str, default="0",
                   help="Comma-separated subject indices (e.g. '0,1,2')")
    p.add_argument("--data_path", type=str, default="../fbirn/fbirn_sz_data.npz")
    p.add_argument("--scc_strategy", type=str, default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--PNUM", type=int,
                   default=int(min(CLINGO_LIMIT, get_process_count(1))))
    p.add_argument("--MAXU", type=int, default=5)
    p.add_argument("--PRIORITY", type=str, default="11112")
    p.add_argument("--gt_density", type=int, default=None)
    p.add_argument("--timeout", type=int, default=0,
                   help="Per-config timeout in seconds (0=no limit)")
    p.add_argument("--capsize", type=int, default=0)
    p.add_argument("--pcmci_method", default="pcmci")
    p.add_argument("--pcmci_tau_max", type=int, default=1)
    p.add_argument("--pcmci_alpha", type=float, default=0.05)
    p.add_argument("--pcmci_fdr", default="none")
    p.add_argument("--grounding_interval", type=float, default=5.0)
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated config names to run (e.g. 'crafty,jumpy,trendy')")
    args = p.parse_args()

    subject_indices = [int(x.strip()) for x in args.subject_idx.split(",")]
    gt_density = args.gt_density
    if gt_density is None:
        gt_density = DEFAULT_GT_DENSITY_BY_N.get(args.n_components)

    configs = ALL_CONFIGS
    if args.only:
        configs = [c.strip() for c in args.only.split(",")]

    print("=" * 70, flush=True)
    print("CLINGO CONFIGURATION BENCHMARK", flush=True)
    print("=" * 70, flush=True)
    print(f"  Time:           {datetime.now()}", flush=True)
    print(f"  N components:   {args.n_components}", flush=True)
    print(f"  Subjects:       {subject_indices}", flush=True)
    print(f"  Configs:        {configs}", flush=True)
    print(f"  MAXCOST:        {MAXCOST}", flush=True)
    print(f"  GT density:     {gt_density}", flush=True)
    print(f"  Threads:        {args.PNUM}", flush=True)
    print(f"  Timeout/config: {args.timeout}s" if args.timeout else
          f"  Timeout/config: none", flush=True)
    print("=" * 70, flush=True)

    # Load data
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "real_data",
                                 data_path)
    npzfile = np.load(data_path)
    data = npzfile["data"]
    labels_key = "labels" if "labels" in npzfile.files else "label"
    labels = npzfile[labels_key]
    comp_indices = get_comp_indices(args.n_components)
    comp_names = get_comp_names(comp_indices)
    n_nodes = len(comp_indices)

    # Per-subject, per-config results
    all_results = {}  # subject_idx -> list of result dicts

    for s_idx in subject_indices:
        print(f"\n{'='*70}", flush=True)
        print(f"  SUBJECT {s_idx}", flush=True)
        print(f"{'='*70}", flush=True)

        command = build_asp_command(
            data, labels, s_idx, comp_indices, comp_names, args, gt_density)

        subject_results = []
        for config_name in configs:
            result = run_config(
                command=command,
                config_name=config_name,
                pnum=args.PNUM,
                capsize=args.capsize,
                optim="optN",
                timeout=args.timeout,
                grounding_interval=args.grounding_interval,
                n_nodes=n_nodes,
            )
            result["subject"] = s_idx
            subject_results.append(result)

        all_results[s_idx] = subject_results

    # ── Summary tables ──
    print(f"\n\n{'='*100}", flush=True)
    print("CONFIGURATION BENCHMARK SUMMARY", flush=True)
    print(f"{'='*100}", flush=True)

    for s_idx in subject_indices:
        results = all_results[s_idx]
        print(f"\n  Subject {s_idx}:", flush=True)

        header = (f"  {'Config':<12s} {'Solve(s)':>10s} {'Total(s)':>10s} "
                  f"{'Models':>8s} {'Optimal':>8s} {'#Solns':>8s} "
                  f"{'Cost':>12s} {'Choices':>10s} {'Conflicts':>10s} "
                  f"{'Note':>6s}")
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)

        for r in results:
            cost_str = str(r["best_cost"]) if r["best_cost"] else "[]"
            if len(cost_str) > 12:
                cost_str = cost_str[:11] + "~"
            note = "T/O" if r.get("timed_out") else ""
            n_sol = len(r["solutions"])
            print(f"  {r['config']:<12s} "
                  f"{r['t_solve']:>10.2f} "
                  f"{r['clingo_total']:>10.2f} "
                  f"{r['n_models_returned']:>8d} "
                  f"{r['n_optimal']:>8d} "
                  f"{n_sol:>8d} "
                  f"{cost_str:>12s} "
                  f"{r['choices']:>10,d} "
                  f"{r['conflicts']:>10,d} "
                  f"{note:>6s}",
                  flush=True)

    # ── Aggregate across subjects ──
    if len(subject_indices) > 1:
        print(f"\n\n{'='*100}", flush=True)
        print("AGGREGATE (mean solve time across subjects)", flush=True)
        print(f"{'='*100}", flush=True)

        header = f"  {'Config':<12s} {'Mean Solve':>12s} {'Min Solve':>12s} {'Max Solve':>12s} {'All Optimal?':>14s}"
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)

        for ci, config_name in enumerate(configs):
            solve_times = []
            all_optimal = True
            for s_idx in subject_indices:
                r = all_results[s_idx][ci]
                solve_times.append(r["t_solve"])
                if r.get("timed_out") or r["n_optimal"] == 0:
                    all_optimal = False
            mean_t = np.mean(solve_times)
            min_t = np.min(solve_times)
            max_t = np.max(solve_times)
            opt_str = "yes" if all_optimal else "NO"
            print(f"  {config_name:<12s} "
                  f"{mean_t:>12.2f} "
                  f"{min_t:>12.2f} "
                  f"{max_t:>12.2f} "
                  f"{opt_str:>14s}",
                  flush=True)

    # ── Solution comparison ──
    print(f"\n\nSOLUTION COMPARISON", flush=True)
    print(f"{'-'*100}", flush=True)
    for s_idx in subject_indices:
        results = all_results[s_idx]
        print(f"\n  Subject {s_idx}:", flush=True)

        costs_set = set()
        for r in results:
            c = tuple(r["best_cost"]) if r["best_cost"] else ()
            costs_set.add(c)
        if len(costs_set) == 1:
            print(f"    Optimal cost: ALL SAME {results[0]['best_cost']}",
                  flush=True)
        else:
            print(f"    WARNING: DIFFERENT optimal costs!", flush=True)
            for r in results:
                note = " (T/O)" if r.get("timed_out") else ""
                print(f"      {r['config']:<12s}: {r['best_cost']}{note}",
                      flush=True)

        baseline_graphs = None
        for r in results:
            sols = r["solutions"]
            if not sols:
                print(f"    {r['config']:<12s}: NO SOLUTIONS", flush=True)
                continue
            min_cost = sols[0][0]
            optimal_set = set((s[1], s[2]) for s in sols if s[0] == min_cost)
            n_opt = len(optimal_set)
            note = " (T/O)" if r.get("timed_out") else ""
            print(f"    {r['config']:<12s}: {len(sols):>4d} total, "
                  f"{n_opt} at best cost={min_cost}{note}", flush=True)

            if baseline_graphs is None:
                baseline_graphs = optimal_set
            elif optimal_set != baseline_graphs:
                shared = len(optimal_set & baseline_graphs)
                print(f"      -> DIFFERS from {results[0]['config']}: "
                      f"{shared} shared", flush=True)

    print(f"\nDone at {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
