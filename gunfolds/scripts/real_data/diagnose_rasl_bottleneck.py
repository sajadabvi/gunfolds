"""
Diagnostic script to find where RASL+clingo spends its time on N=20 fMRI data.

Reproduces the exact same pipeline as fmri_experiment_large.py for ONE subject
but instruments every phase with detailed timing and clingo progress reporting:

  1. PCMCI estimation phase       (usually fast, ~seconds)
  2. ASP program construction     (fast)
  3. Clingo GROUNDING phase       (can be very slow — reports atom/rule counts)
  4. Clingo SOLVING phase         (can be very slow — reports each model found)
  5. Solution parsing             (fast)

Clingo progress:
  - Grounding: Uses a GroundProgramObserver to count atoms and rules as they
    are generated, printing periodic updates.
  - Solving:   Uses on_model callback to report each model/cost in real time.
  - After solving, dumps full clingo statistics (search tree size, conflicts,
    propagations, etc.)

Usage (run locally or on the cluster):

  # Quick test with 10-node graph (should finish in minutes)
  python diagnose_rasl_bottleneck.py --n_components 10 --subject_idx 0

  # Full 20-node diagnosis (the real bottleneck)
  python diagnose_rasl_bottleneck.py --n_components 20 --subject_idx 0

  # Override MAXU to test if lowering it helps
  python diagnose_rasl_bottleneck.py --n_components 20 --subject_idx 0 --MAXU 3

  # Use fewer CPUs locally
  python diagnose_rasl_bottleneck.py --n_components 20 --subject_idx 0 --PNUM 4

  # Set a timeout (seconds) to abort if grounding+solving exceeds it
  python diagnose_rasl_bottleneck.py --n_components 20 --subject_idx 0 --timeout 600
"""

import os
import sys
import time
import json
import argparse
import threading
from datetime import datetime, timedelta

import numpy as np
import clingo as clngo

sys.path.insert(0, os.path.dirname(__file__))
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


# ─────────────────────────────────────────────────────────────────────────────
# Grounding observer — reports progress during ctrl.ground()
# ─────────────────────────────────────────────────────────────────────────────

class GroundingObserver:
    """
    Implements clingo's GroundProgramObserver to track what happens during
    grounding (atom creation, rule generation).  Prints periodic updates.
    """
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
            print(f"  [GROUNDING {elapsed:8.1f}s]  atoms={self.atoms:,}  "
                  f"rules={self.rules:,}  weight_rules={self.weight_rules:,}  "
                  f"minimize={self.minimize_stmts}", flush=True)
            self._last_report = now

    def init_program(self, incremental):
        pass

    def begin_step(self):
        pass

    def end_step(self):
        pass

    def rule(self, choice, head, body):
        self.rules += 1
        self._maybe_report()

    def weight_rule(self, choice, head, lower_bound, body):
        self.weight_rules += 1
        self._maybe_report()

    def minimize(self, priority, literals):
        self.minimize_stmts += 1

    def project(self, atoms):
        pass

    def output_atom(self, symbol, atom):
        self.atoms += 1
        self._maybe_report()

    def output_term(self, symbol, condition):
        pass

    def external(self, atom, value):
        pass

    def assume(self, literals):
        pass

    def heuristic(self, atom, type_, bias, priority, condition):
        pass

    def acyc_edge(self, node_u, node_v, condition):
        pass

    def theory_term_number(self, term_id, number):
        pass

    def theory_term_string(self, term_id, name):
        pass

    def theory_term_compound(self, term_id, name_id_or_type, arguments):
        pass

    def theory_element(self, element_id, terms, condition):
        pass

    def theory_atom(self, atom_id_or_zero, term_id, elements):
        pass

    def theory_atom_with_guard(self, atom_id_or_zero, term_id, elements,
                               operator_id, right_hand_side_id):
        pass

    def summary(self):
        elapsed = time.time() - self._start
        print(f"  [GROUNDING DONE in {elapsed:.1f}s]  "
              f"atoms={self.atoms:,}  rules={self.rules:,}  "
              f"weight_rules={self.weight_rules:,}  "
              f"minimize={self.minimize_stmts}", flush=True)
        return {
            "elapsed_s": round(elapsed, 2),
            "atoms": self.atoms,
            "rules": self.rules,
            "weight_rules": self.weight_rules,
            "minimize_stmts": self.minimize_stmts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented clingo runner
# ─────────────────────────────────────────────────────────────────────────────

def run_clingo_instrumented(command, capsize=0, configuration="crafty",
                            pnum=1, optim="optN", timeout=0,
                            grounding_report_interval=5.0):
    """
    Replacement for gunfolds.utils.clingo.run_clingo with full progress
    reporting during both grounding and solving.
    """

    clingo_args = [
        "--warn=no-atom-undefined",
        f"--configuration={configuration}",
        "-t", f"{int(pnum)},split",
        "-n", str(capsize),
    ]

    print(f"\n{'='*70}", flush=True)
    print(f"CLINGO LAUNCH", flush=True)
    print(f"  capsize={capsize}  config={configuration}  "
          f"threads={pnum}  optim={optim}  timeout={timeout}", flush=True)
    print(f"  ASP program size: {len(command):,} bytes", flush=True)
    print(f"{'='*70}", flush=True)

    ctrl = clngo.Control(clingo_args)
    ctrl.configuration.solve.opt_mode = optim

    # Register grounding observer
    observer = GroundingObserver(report_interval=grounding_report_interval)
    ctrl.register_observer(observer)

    # ── Phase 1: Add program ──
    t0 = time.time()
    print(f"\n[PHASE] Adding ASP program...", flush=True)
    ctrl.add("base", [], command.decode())
    t_add = time.time() - t0
    print(f"  Done in {t_add:.2f}s", flush=True)

    # ── Phase 2: Grounding ──
    print(f"\n[PHASE] GROUNDING (this can be slow for large N)...", flush=True)
    t0 = time.time()
    ctrl.ground([("base", [])])
    t_ground = time.time() - t0
    ground_stats = observer.summary()

    # ── Phase 3: Solving ──
    print(f"\n[PHASE] SOLVING (optim={optim}, searching for optimal models)...",
          flush=True)
    t0 = time.time()
    models = []
    model_count = 0
    best_cost = None

    solve_kwargs = {"yield_": True, "async_": True}

    with ctrl.solve(**solve_kwargs) as handle:
        for model in handle:
            model_count += 1
            cost = model.cost
            optimality = model.optimality_proven
            atoms = [str(a) for a in model.symbols(shown=True)]
            models.append((atoms, cost))

            elapsed = time.time() - t0

            cost_str = f"cost={cost}" if cost else "cost=[]"
            improved = ""
            if best_cost is None or cost < best_cost:
                best_cost = cost
                improved = " ** NEW BEST **"

            print(f"  [MODEL #{model_count} at {elapsed:.1f}s]  "
                  f"{cost_str}  optimal={optimality}  "
                  f"atoms={len(atoms)}{improved}", flush=True)

    t_solve = time.time() - t0

    # ── Phase 4: Statistics ──
    print(f"\n{'='*70}", flush=True)
    print(f"CLINGO FINISHED", flush=True)
    print(f"{'='*70}", flush=True)

    stats = ctrl.statistics
    summary = stats.get("summary", {})
    solving = stats.get("solving", {})
    solve_stats_list = solving.get("solvers", [])

    costs = summary.get("costs", [])
    n_optimal = summary.get("models", {}).get("optimal", 0)
    n_enumerated = summary.get("models", {}).get("enumerated", 0)

    times = summary.get("times", {})
    total_time = times.get("total", 0)
    solve_time = times.get("solve", 0)
    model_time = times.get("sat", 0)
    unsat_time = times.get("unsat", 0)

    print(f"\n  Timing breakdown:", flush=True)
    print(f"    Program add:  {t_add:10.2f}s", flush=True)
    print(f"    Grounding:    {t_ground:10.2f}s", flush=True)
    print(f"    Solving:      {t_solve:10.2f}s", flush=True)
    print(f"    clingo total: {total_time:10.2f}s  "
          f"(solve={solve_time:.2f}  sat={model_time:.2f}  "
          f"unsat={unsat_time:.2f})", flush=True)

    print(f"\n  Models:", flush=True)
    print(f"    Enumerated: {n_enumerated}", flush=True)
    print(f"    Optimal:    {n_optimal}", flush=True)
    print(f"    Best cost:  {costs}", flush=True)

    if solve_stats_list:
        solver0 = solve_stats_list[0] if solve_stats_list else {}
        print(f"\n  Solver stats (thread 0):", flush=True)
        for key in ["choices", "conflicts", "restarts", "models"]:
            val = solver0.get(key, "?")
            if isinstance(val, float):
                val = f"{val:,.0f}"
            print(f"    {key:20s}: {val}", flush=True)

    print(f"\n  Grounding stats:", flush=True)
    for k, v in ground_stats.items():
        print(f"    {k:20s}: {v:,}" if isinstance(v, int) else
              f"    {k:20s}: {v}", flush=True)

    return models, costs, {
        "t_add": t_add,
        "t_ground": t_ground,
        "t_solve": t_solve,
        "ground_stats": ground_stats,
        "n_models": model_count,
        "n_optimal": n_optimal,
        "best_cost": costs,
        "clingo_total_time": total_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PCMCI wrapper with timing
# ─────────────────────────────────────────────────────────────────────────────

def run_pcmci_timed(ts_2d, pcmci_method="pcmci", tau_max=1,
                    alpha_level=0.05, pc_alpha=0.01, fdr_method="none"):
    print(f"\n[PHASE] PCMCI estimation "
          f"(method={pcmci_method}, tau_max={tau_max}, "
          f"alpha={alpha_level}, fdr={fdr_method})...", flush=True)
    t0 = time.time()

    dataframe = pp.DataFrame(ts_2d)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)

    if pcmci_method == "pcmciplus":
        results = pcmci.run_pcmciplus(
            tau_max=tau_max, pc_alpha=pc_alpha, fdr_method=fdr_method)
    else:
        results = pcmci.run_pcmci(
            tau_max=tau_max, pc_alpha=None,
            alpha_level=alpha_level, fdr_method=fdr_method)

    g_estimated, A, B = cv.Glag2CG(results)
    elapsed = time.time() - t0

    n_nodes = len(g_estimated)
    n_dir = sum(1 for n in g_estimated for t in g_estimated[n]
                if g_estimated[n][t] in (1, 3))
    n_bidir = sum(1 for n in g_estimated for t in g_estimated[n]
                  if g_estimated[n][t] in (2, 3) and n < t)
    density = gk.density(g_estimated)

    print(f"  Done in {elapsed:.2f}s", flush=True)
    print(f"  Estimated graph: {n_nodes} nodes, "
          f"{n_dir} directed edges, {n_bidir} bidirected edges, "
          f"density={density:.3f}", flush=True)

    return g_estimated, A, B, elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Diagnose RASL/clingo bottleneck on fMRI data")
    p.add_argument("--n_components", type=int, default=20, choices=[10, 20, 53])
    p.add_argument("--subject_idx", type=int, default=0)
    p.add_argument("--data_path", type=str, default="../fbirn/fbirn_sz_data.npz")
    p.add_argument("--scc_strategy", type=str, default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--PNUM", type=int,
                   default=int(min(CLINGO_LIMIT, get_process_count(1))))
    p.add_argument("--MAXU", type=int, default=5)
    p.add_argument("--PRIORITY", type=str, default="11112")
    p.add_argument("--gt_density", type=int, default=220)
    p.add_argument("--timeout", type=int, default=0,
                   help="Abort clingo after this many seconds (0=no limit)")
    p.add_argument("--pcmci_method", default="pcmci")
    p.add_argument("--pcmci_tau_max", type=int, default=1)
    p.add_argument("--pcmci_alpha", type=float, default=0.05)
    p.add_argument("--pcmci_fdr", default="none")
    p.add_argument("--grounding_interval", type=float, default=5.0,
                   help="Seconds between grounding progress reports")
    args = p.parse_args()

    print("=" * 70)
    print("RASL / CLINGO BOTTLENECK DIAGNOSTIC")
    print("=" * 70)
    print(f"  Time:           {datetime.now()}")
    print(f"  N components:   {args.n_components}")
    print(f"  Subject index:  {args.subject_idx}")
    print(f"  SCC strategy:   {args.scc_strategy}")
    print(f"  MAXU:           {args.MAXU}")
    print(f"  PRIORITY:       {args.PRIORITY}")
    print(f"  GT density:     {args.gt_density}")
    print(f"  Clingo threads: {args.PNUM}")
    print(f"  Timeout:        {args.timeout}s" if args.timeout else
          f"  Timeout:        none")
    print(f"  PCMCI:          {args.pcmci_method} tau_max={args.pcmci_tau_max}"
          f" alpha={args.pcmci_alpha}")
    print("=" * 70)

    # ── Load data ──
    print(f"\n[PHASE] Loading data from {args.data_path}...", flush=True)
    t0 = time.time()
    npzfile = np.load(args.data_path)
    data = npzfile["data"]
    labels = npzfile.get("labels", npzfile.get("label"))
    print(f"  Data shape: {data.shape}  (loaded in {time.time()-t0:.2f}s)",
          flush=True)

    comp_indices = get_comp_indices(args.n_components)
    comp_names = get_comp_names(comp_indices)
    n_nodes = len(comp_indices)

    s = args.subject_idx
    ts_2d = data[s][:, comp_indices]
    print(f"  Subject {s}: time series shape {ts_2d.shape}, "
          f"group={int(labels[s])}", flush=True)

    # ── Step 1: PCMCI ──
    g_estimated, A, B, t_pcmci = run_pcmci_timed(
        ts_2d,
        pcmci_method=args.pcmci_method,
        tau_max=args.pcmci_tau_max,
        alpha_level=args.pcmci_alpha,
        fdr_method=args.pcmci_fdr,
    )

    # ── Step 2: SCC setup ──
    print(f"\n[PHASE] SCC setup (strategy={args.scc_strategy})...", flush=True)
    import networkx as nx
    scc_members = get_scc_members(
        args.scc_strategy, comp_indices, ts_2d, max_cluster_size=8)
    use_scc = scc_members is not None

    if args.scc_strategy == "estimated":
        scc_members = list(
            nx.strongly_connected_components(gk.graph2nx(g_estimated)))
        use_scc = True

    if use_scc and scc_members:
        print(f"  SCCs ({len(scc_members)} groups):", flush=True)
        for i, scc in enumerate(scc_members):
            node_names = [comp_names[n-1] for n in scc]
            print(f"    SCC {i}: size={len(scc)} nodes={sorted(scc)} "
                  f"({', '.join(node_names)})", flush=True)
    else:
        print(f"  No SCC decomposition (treating all {n_nodes} nodes as one)",
              flush=True)

    # ── Step 3: Build distance matrices ──
    print(f"\n[PHASE] Building distance penalty matrices...", flush=True)
    MAXCOST = 10000
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
    print(f"  DD shape: {DD.shape}  range: [{DD.min()}, {DD.max()}]",
          flush=True)
    print(f"  BD shape: {BD.shape}  range: [{BD.min()}, {BD.max()}]",
          flush=True)

    # ── Step 4: Build ASP program (same as drasl_command) ──
    print(f"\n[PHASE] Building ASP program...", flush=True)
    t0 = time.time()
    priority = [int(c) for c in args.PRIORITY]
    urate = min(args.MAXU, (3 * n_nodes + 1))

    command = drasl_command(
        [g_estimated],
        max_urate=urate,
        weighted=True,
        scc=use_scc,
        scc_members=scc_members,
        dm=[DD],
        bdm=[BD],
        edge_weights=priority,
        GT_density=args.gt_density,
        selfloop=False,
    )
    t_cmd = time.time() - t0
    print(f"  ASP program built in {t_cmd:.2f}s "
          f"({len(command):,} bytes)", flush=True)

    # ── Step 5: Instrumented clingo run ──
    models, costs, clingo_stats = run_clingo_instrumented(
        command,
        capsize=0,
        configuration="crafty",
        pnum=args.PNUM,
        optim="optN",
        timeout=args.timeout,
        grounding_report_interval=args.grounding_interval,
    )

    # ── Step 6: Parse results ──
    print(f"\n[PHASE] Parsing results...", flush=True)
    t0 = time.time()
    if models:
        r_estimated = {(drasl_jclingo2g(m[0]), sum(m[1])) for m in models}
        solutions = []
        for answer in r_estimated:
            graph_num = answer[0][0]
            undersampling = answer[0][1]
            cost = answer[1]
            res_cg = bfutils.num2CG(graph_num, n_nodes)
            solutions.append((cost, res_cg, undersampling))
        solutions.sort(key=lambda x: x[0])
        top_k = solutions[:10]
        print(f"  {len(solutions)} total solutions, showing top {len(top_k)}:",
              flush=True)
        for i, (cost, cg, u) in enumerate(top_k):
            d = gk.density(cg)
            print(f"    #{i+1}: cost={cost}  u={u}  density={d:.3f}",
                  flush=True)
    else:
        print(f"  No solutions found!", flush=True)
    t_parse = time.time() - t0

    # ── Final summary ──
    total = t_pcmci + clingo_stats["t_ground"] + clingo_stats["t_solve"]
    print(f"\n{'='*70}")
    print(f"TIMING SUMMARY")
    print(f"{'='*70}")
    print(f"  PCMCI:         {t_pcmci:10.2f}s  "
          f"({100*t_pcmci/total:.1f}%)" if total > 0 else "")
    print(f"  ASP build:     {t_cmd:10.2f}s")
    print(f"  GROUNDING:     {clingo_stats['t_ground']:10.2f}s  "
          f"({100*clingo_stats['t_ground']/total:.1f}%)" if total > 0 else "")
    print(f"  SOLVING:       {clingo_stats['t_solve']:10.2f}s  "
          f"({100*clingo_stats['t_solve']/total:.1f}%)" if total > 0 else "")
    print(f"  Parse:         {t_parse:10.2f}s")
    print(f"  {'─'*40}")
    print(f"  TOTAL:         {total:10.2f}s  "
          f"({str(timedelta(seconds=int(total)))})")
    print(f"{'='*70}")

    bottleneck = "GROUNDING" if clingo_stats["t_ground"] > clingo_stats["t_solve"] else "SOLVING"
    print(f"\n  >> BOTTLENECK: {bottleneck} phase <<")

    if bottleneck == "GROUNDING":
        print(f"\n  The grounding phase is expanding the ASP rules into a "
              f"propositional program.")
        print(f"  With {n_nodes} nodes and MAXU={urate}, the number of "
              f"ground rules grows combinatorially.")
        print(f"  Consider:")
        print(f"    - Reducing MAXU (try --MAXU 3)")
        print(f"    - Using SCC decomposition (--scc_strategy domain)")
        print(f"    - Reducing n_components")
    else:
        print(f"\n  The solving phase is searching through the space of "
              f"possible graphs.")
        print(f"  With weighted optimization (optN), clingo must first find "
              f"the optimal cost,")
        print(f"  then enumerate all models with that cost.")
        print(f"  Consider:")
        print(f"    - Setting a timeout (--timeout 3600 for 1 hour)")
        print(f"    - Reducing MAXU to shrink the search space")
        print(f"    - Using capsize > 0 to limit returned models")
        print(f"    - Checking if GT_density constraint is too tight/loose")

    print(f"\nDone at {datetime.now()}")


if __name__ == "__main__":
    main()
