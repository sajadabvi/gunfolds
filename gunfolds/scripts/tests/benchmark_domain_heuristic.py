"""
Benchmark suggestion #4 — Domain heuristic with PCMCI-prior #heuristic directives.

The idea (from clingo_speedup_suggestions.md, suggestion 4):
  VSIDS branching is blind to PCMCI's prior over which edges are likely
  present/absent. Domain heuristic (--heuristic=Domain) lets us bias the
  *initial* branching polarity of every edge1/2 atom according to the PCMCI
  confidence already encoded in hdirected/no_hdirected weights:

      #heuristic edge1(X,Y). [W,true]  :- hdirected(X,Y,W,1).
      #heuristic edge1(X,Y). [W,false] :- no_hdirected(X,Y,W,1).

  The expected payoff is reaching a low-cost feasible model in the first few
  decisions, which tightens the bound for branch-and-bound and dramatically
  shortens the optimality proof.

This is a strictly different test from --opt-heuristic=1 (already tested in
benchmark_clingo_flags.py): --opt-heuristic=1 only biases branching after a
model is found; --heuristic=Domain controls *initial* polarity from the start
and only fires when #heuristic directives are present in the program.

Three scenarios are compared on identical input:

  S0: baseline                       — no #heuristic, default heuristic
  S1: #heuristic + --heuristic=Domain
  S2: #heuristic + --heuristic=Domain + --dom-mod=5,16
        (5 = level/init-true polarity, 16 = scope=show — apply only to shown
         atoms, i.e. edge1/2 and u/2)

Usage (mirrors benchmark_clingo_flags.py):

  python benchmark_domain_heuristic.py --n_components 10 --subject_idx 0
  python benchmark_domain_heuristic.py --n_components 20 --subject_idx 0 --timeout 3600
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
DEFAULT_GT_DENSITY_BY_N = {10: 35, 14: 30, 20: 22, 53: 13}


# ─────────────────────────────────────────────────────────────────────────────
# Grounding observer (mirrors benchmark_clingo_flags.py)
# ─────────────────────────────────────────────────────────────────────────────

class GroundingObserver:
    def __init__(self, report_interval=5.0):
        self.atoms = 0
        self.rules = 0
        self.weight_rules = 0
        self.minimize_stmts = 0
        self.heuristic_stmts = 0
        self._start = time.time()
        self._last_report = self._start
        self._interval = report_interval

    def _maybe_report(self):
        now = time.time()
        if now - self._last_report >= self._interval:
            elapsed = now - self._start
            print(f"    [GROUNDING {elapsed:8.1f}s]  atoms={self.atoms:,}  "
                  f"rules={self.rules:,}  weight_rules={self.weight_rules:,}  "
                  f"heuristic={self.heuristic_stmts}  "
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
    def heuristic(self, atom, type_, bias, priority, condition):
        self.heuristic_stmts += 1
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
            "heuristic_stmts": self.heuristic_stmts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Build the #heuristic block from PCMCI weights (DD, BD)
# ─────────────────────────────────────────────────────────────────────────────

def build_heuristic_block(DD, BD, n_nodes, k=1):
    """
    Emit #heuristic directives biasing edge1/2 polarity by PCMCI confidence.

    The drasl ASP encoding emits:
      hdirected(X,Y,W,K)    when g_estimated has a directed edge X->Y
      no_hdirected(X,Y,W,K) when it does not
      hbidirected/...       analogous for bidirected edges

    W is the PCMCI-derived weight (0..MAXCOST); higher W = stronger evidence.
    edge1/2 in the encoding is the candidate causal-scale directed edge.
    Bidirected at causal scale is not directly choosable (it emerges from
    pairs of forward edges into a common past node), so we only bias edge1/2.

    For each ordered pair (X,Y), we use whichever has higher confidence —
    presence weight (from hdirected) or absence weight (from no_hdirected) —
    as the magnitude, and set the polarity accordingly.
    """
    # gunfolds is 1-indexed
    lines = []

    # Domain heuristic polarity biases. The bias magnitude is the PCMCI weight;
    # ties broken by the heuristic's own priority field (we leave it as the
    # default by giving every directive the same modifier, matching the
    # encoding's MAXCOST scale).
    lines.append("% PCMCI-prior #heuristic block (suggestion #4)")
    for x in range(1, n_nodes + 1):
        for y in range(1, n_nodes + 1):
            # gunfolds disables self-loops via selfloop=False above; skip too
            if x == y:
                continue
            w_pres = int(DD[x - 1, y - 1])
            w_abs = MAXCOST - w_pres  # complementary weight for the absence bias
            # Presence bias: when PCMCI gives high evidence for X->Y, emit
            #   #heuristic edge1(X,Y). [W,true]
            # Absence bias: complementary directive with [W,false]
            # We always emit both — clasp will use the stronger one as the
            # initial polarity.
            if w_pres > 0:
                lines.append(
                    f"#heuristic edge1({x},{y}). [{w_pres},true]")
            if w_abs > 0:
                lines.append(
                    f"#heuristic edge1({x},{y}). [{w_abs},false]")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Run one scenario
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(command, scenario_label, extra_args, capsize, configuration,
                 pnum, optim, timeout, grounding_interval, n_nodes,
                 heuristic_block=""):
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
    print(f"  optim={optim}  capsize={capsize}  threads={pnum}  "
          f"heuristic_block_chars={len(heuristic_block):,}", flush=True)

    ctrl = clngo.Control(clingo_args)
    ctrl.configuration.solve.opt_mode = optim

    observer = GroundingObserver(report_interval=grounding_interval)
    ctrl.register_observer(observer)

    # Phase 1: Add program
    t0 = time.time()
    program = command.decode() + "\n" + heuristic_block
    ctrl.add("base", [], program)
    t_add = time.time() - t0

    # Phase 2: Grounding
    print(f"\n  [GROUNDING]...", flush=True)
    t0 = time.time()
    ctrl.ground([("base", [])])
    t_ground = time.time() - t0
    ground_stats = observer.summary()
    print(f"    Done in {t_ground:.2f}s  "
          f"(atoms={ground_stats['atoms']:,}  rules={ground_stats['rules']:,}  "
          f"weight_rules={ground_stats['weight_rules']:,}  "
          f"heuristic={ground_stats['heuristic_stmts']:,})", flush=True)

    # Phase 3: Solving
    timeout_str = f"{timeout}s" if timeout else "none"
    print(f"\n  [SOLVING] (optim={optim}, timeout={timeout_str})...", flush=True)
    t0 = time.time()
    models = []
    model_count = 0
    best_cost = None
    timed_out = False
    first_feasible_time = None
    first_feasible_cost = None

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
                if first_feasible_time is None:
                    first_feasible_time = elapsed
                    first_feasible_cost = cost

                if model_count <= 20 or optimality or improved:
                    cost_str = f"cost={cost}" if cost else "cost=[]"
                    print(f"    [MODEL #{model_count} at {elapsed:.1f}s]  "
                          f"{cost_str}  optimal={optimality}  "
                          f"atoms={len(atoms)}{improved}", flush=True)

                if model_count == 21:
                    print(f"    ... (suppressing further model-by-model "
                          f"output) ...", flush=True)
    finally:
        if timer is not None:
            timer.cancel()

    t_solve = time.time() - t0

    # Statistics
    def _sg(obj, key, default=0):
        if obj is None:
            return default
        try:
            return obj[key]
        except (KeyError, IndexError, TypeError):
            if hasattr(obj, 'get'):
                return obj.get(key, default)
            return default

    stats = ctrl.statistics
    summary_s = _sg(stats, "summary", {})
    solving = _sg(stats, "solving", {})
    solvers = _sg(solving, "solvers", {})
    costs = _sg(summary_s, "costs", [])
    models_stats = _sg(summary_s, "models", {})
    n_optimal = _sg(models_stats, "optimal", 0)
    n_enumerated = _sg(models_stats, "enumerated", 0)
    times = _sg(summary_s, "times", {})
    total_time = _sg(times, "total", 0)
    solve_time = _sg(times, "solve", 0)

    solver0 = _sg(solvers, 0, _sg(solvers, "0", {}))
    choices = _sg(solver0, "choices", 0)
    conflicts = _sg(solver0, "conflicts", 0)
    restarts = _sg(solver0, "restarts", 0)

    print(f"\n  [RESULT]", flush=True)
    print(f"    Grounding:        {t_ground:10.2f}s", flush=True)
    print(f"    Solving:          {t_solve:10.2f}s", flush=True)
    print(f"    1st feasible at:  {first_feasible_time}  "
          f"cost={first_feasible_cost}", flush=True)
    print(f"    Models:           enum={n_enumerated} optimal={n_optimal} "
          f"returned={model_count}", flush=True)
    print(f"    Best cost:        {costs}", flush=True)
    print(f"    Solver stats:     choices={choices:,.0f}  "
          f"conflicts={conflicts:,.0f}  restarts={restarts:,.0f}", flush=True)

    # Parse solutions
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

    return {
        "scenario": scenario_label,
        "extra_args": extra_args,
        "t_ground": round(t_ground, 3),
        "t_solve": round(t_solve, 3),
        "clingo_total": round(total_time, 3),
        "clingo_solve": round(solve_time, 3),
        "n_models_returned": model_count,
        "n_enumerated": int(n_enumerated),
        "n_optimal": int(n_optimal),
        "best_cost": list(costs) if costs else [],
        "first_feasible_time": first_feasible_time,
        "first_feasible_cost": list(first_feasible_cost)
            if first_feasible_cost else [],
        "choices": int(choices),
        "conflicts": int(conflicts),
        "restarts": int(restarts),
        "ground_atoms": ground_stats["atoms"],
        "ground_rules": ground_stats["rules"],
        "ground_heuristic": ground_stats["heuristic_stmts"],
        "solutions": solutions,
        "timed_out": timed_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark Domain heuristic + PCMCI-prior #heuristic block")
    p.add_argument("--n_components", type=int, default=10,
                   choices=[10, 14, 20, 53])
    p.add_argument("--subject_idx", type=int, default=0)
    p.add_argument("--data_path", type=str, default="../fbirn/fbirn_sz_data.npz")
    p.add_argument("--scc_strategy", type=str, default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--PNUM", type=int,
                   default=int(min(CLINGO_LIMIT, get_process_count(1))))
    p.add_argument("--MAXU", type=int, default=5)
    p.add_argument("--PRIORITY", type=str, default="11112")
    p.add_argument("--gt_density", type=int, default=None)
    p.add_argument("--timeout", type=int, default=0)
    p.add_argument("--capsize", type=int, default=0)
    p.add_argument("--pcmci_method", default="pcmci")
    p.add_argument("--pcmci_tau_max", type=int, default=1)
    p.add_argument("--pcmci_alpha", type=float, default=0.05)
    p.add_argument("--pcmci_fdr", default="none")
    p.add_argument("--grounding_interval", type=float, default=5.0)
    p.add_argument("--density_mode", type=str, default="hard_soft0",
                   choices=["soft", "hard", "hard_soft0", "hard_soft1", "none"],
                   help="Production default: hard_soft0 (hard window @ none + "
                        "density tiebreaker @0, edge matching @1)")
    p.add_argument("--tol_low", type=int, default=15,
                   help="Density tolerance below GT_density (PCMCI overestimates)")
    p.add_argument("--tol_high", type=int, default=5,
                   help="Density tolerance above GT_density")
    p.add_argument("--optim", type=str, default="optN",
                   choices=["opt", "optN"],
                   help="opt = find one optimum and stop; optN = enumerate all "
                        "optima (slower but matches production)")
    p.add_argument("--only", type=str, default="",
                   help="Comma-separated scenario numbers to run (e.g. '0,1')")
    args = p.parse_args()

    only_set = set(int(x) for x in args.only.split(",") if x.strip())

    gt_density = args.gt_density
    if gt_density is None:
        gt_density = DEFAULT_GT_DENSITY_BY_N.get(args.n_components)

    print("=" * 70, flush=True)
    print("DOMAIN-HEURISTIC BENCHMARK (suggestion #4)", flush=True)
    print("=" * 70, flush=True)
    print(f"  Time:           {datetime.now()}", flush=True)
    print(f"  N components:   {args.n_components}", flush=True)
    print(f"  Subject index:  {args.subject_idx}", flush=True)
    print(f"  SCC strategy:   {args.scc_strategy}", flush=True)
    print(f"  MAXU:           {args.MAXU}", flush=True)
    print(f"  MAXCOST:        {MAXCOST}", flush=True)
    print(f"  GT density:     {gt_density}", flush=True)
    print(f"  Clingo threads: {args.PNUM}", flush=True)
    print(f"  Timeout/scen:   "
          f"{(str(args.timeout) + 's') if args.timeout else 'none'}",
          flush=True)
    print("=" * 70, flush=True)

    # Load data
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
    n_nodes = len(comp_indices)

    s = args.subject_idx
    ts_2d = data[s][:, comp_indices]
    print(f"  Subject {s}: shape {ts_2d.shape}, group={int(labels[s])}",
          flush=True)

    # PCMCI
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
    if use_scc and scc_members:
        print(f"  SCCs ({len(scc_members)} groups): "
              f"{[len(sc) for sc in scc_members]}", flush=True)

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
    print(f"  DD range: [{DD.min()}, {DD.max()}]  "
          f"BD range: [{BD.min()}, {BD.max()}]", flush=True)

    # Build ASP program (once)
    priority = [int(c) for c in args.PRIORITY]
    urate = min(args.MAXU, (3 * n_nodes + 1))
    # Match production encoding: hard density window + density tiebreaker at @0
    # (lex below edge matching at @1), asymmetric downward-biased tolerance
    # (PCMCI overestimates causal-scale density). See CHANGELOG_AI.md lines
    # 12-22. With density_mode='soft' the benchmark would collapse density and
    # edge cost into one @1 term, giving a single-scalar objective.
    command = drasl_command(
        [g_estimated], max_urate=urate, weighted=True,
        scc=use_scc, scc_members=scc_members,
        dm=[DD], bdm=[BD], edge_weights=priority,
        GT_density=gt_density, selfloop=False,
        density_mode=args.density_mode,
        tol=None, tol_low=args.tol_low, tol_high=args.tol_high,
    )
    print(f"  ASP program: {len(command):,} bytes", flush=True)

    # Heuristic block (used by S1 and S2)
    heuristic_block = build_heuristic_block(DD, BD, n_nodes)
    n_directives = heuristic_block.count("#heuristic")
    print(f"  #heuristic directives: {n_directives}  "
          f"({len(heuristic_block):,} bytes)", flush=True)

    # Define scenarios
    scenarios = [
        ("S0: baseline (current)", [], ""),
        ("S1: #heuristic + Domain",
         ["--heuristic=Domain"], heuristic_block),
        ("S2: #heuristic + Domain + dom-mod=5,16",
         ["--heuristic=Domain", "--dom-mod=5,16"], heuristic_block),
    ]

    all_results = []
    for i, (label, extra_args, hb) in enumerate(scenarios):
        if only_set and i not in only_set:
            print(f"\n  [SKIP] {label} (not in --only)", flush=True)
            continue
        result = run_scenario(
            command=command,
            scenario_label=label,
            extra_args=extra_args,
            capsize=args.capsize,
            configuration="crafty",
            pnum=args.PNUM,
            optim=args.optim,
            timeout=args.timeout,
            grounding_interval=args.grounding_interval,
            n_nodes=n_nodes,
            heuristic_block=hb,
        )
        all_results.append(result)

    # Summary
    print(f"\n\n{'='*100}", flush=True)
    print("DOMAIN-HEURISTIC BENCHMARK SUMMARY", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"  N={args.n_components}  subject={args.subject_idx}  "
          f"MAXCOST={MAXCOST}  GT_density={gt_density}  "
          f"MAXU={args.MAXU}  threads={args.PNUM}", flush=True)
    print(f"{'='*100}", flush=True)

    header = (f"{'Scenario':<42s} {'Solve(s)':>10s} {'1stFeas(s)':>11s} "
              f"{'1stCost':>10s} {'BestCost':>12s} {'Choices':>10s} "
              f"{'Conflicts':>10s} {'Note':>5s}")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in all_results:
        cost_str = str(r["best_cost"]) if r["best_cost"] else "[]"
        if len(cost_str) > 12:
            cost_str = cost_str[:11] + "~"
        ff_t = (f"{r['first_feasible_time']:.2f}"
                if r['first_feasible_time'] is not None else "N/A")
        ff_c = (str(r['first_feasible_cost'])[:10]
                if r['first_feasible_cost'] else "[]")
        note = "T/O" if r.get("timed_out") else ""
        print(f"{r['scenario']:<42s} "
              f"{r['t_solve']:>10.2f} "
              f"{ff_t:>11s} "
              f"{ff_c:>10s} "
              f"{cost_str:>12s} "
              f"{r['choices']:>10,d} "
              f"{r['conflicts']:>10,d} "
              f"{note:>5s}", flush=True)

    print(f"\n{'='*100}", flush=True)

    # Speedup vs baseline
    if len(all_results) > 1 and all_results[0]["t_solve"] > 0:
        baseline = all_results[0]
        print(f"\nSpeedup vs baseline (S0):", flush=True)
        for r in all_results[1:]:
            if r["t_solve"] > 0:
                speedup = baseline["t_solve"] / r["t_solve"]
                print(f"  {r['scenario']:<42s}  solve {speedup:>5.2f}x",
                      flush=True)
            ff_b = baseline['first_feasible_time']
            ff_r = r['first_feasible_time']
            if ff_b and ff_r and ff_r > 0:
                ff_speedup = ff_b / ff_r
                print(f"    1st-feasible speedup: {ff_speedup:>5.2f}x  "
                      f"({ff_b:.2f}s -> {ff_r:.2f}s)", flush=True)

    # Optimal cost agreement
    costs_set = set()
    for r in all_results:
        c = tuple(r["best_cost"]) if r["best_cost"] else ()
        costs_set.add(c)
    if len(costs_set) == 1:
        print(f"\n  Optimal cost: ALL SAME  {all_results[0]['best_cost']}",
              flush=True)
    else:
        print(f"\n  WARNING: differing optimal costs across scenarios",
              flush=True)
        for r in all_results:
            print(f"    {r['scenario']}: {r['best_cost']}", flush=True)

    print(f"\nDone at {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
