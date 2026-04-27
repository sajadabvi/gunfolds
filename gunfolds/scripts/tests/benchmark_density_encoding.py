"""
Benchmark five density-encoding strategies for weighted DRASL.

Variants
--------
  A  current    : soft density penalty at @1, mixed with edge-matching cost.
                  No cardinality bounds — any graph density is reachable.
  B  hard_only  : hard cardinality bounds [d_lo, d_hi], no soft penalty.
                  Density completely removed from objective.
  C  hard+soft0 : hard bounds + residual soft density at @0 (lower priority
                  than edge matching). Edge matching solved first, density
                  breaks ties among equally-good edge solutions.
  D  hard+soft1 : hard bounds + soft density at @1 (same priority as edges).
                  Density and edge matching compete in one objective.
  E  adaptive   : (production default) per-subject GT_density derived from
                  the input graph; tries C with tol, then C with tol_widen,
                  then falls back to A. Combines pruning speedup with the
                  robustness of A on subjects whose causal density falls
                  outside the tight window.

Scientific question being tested:
  - Does restricting density to a hard window speed up the solver?
  - Does the priority level of the residual soft penalty affect the optimal
    solution (i.e., do C and D return different graphs)?
  - Is the hard+soft separation (C) faster than current (A)?

Usage
-----
  # Quick: N=10, subject 0
  python benchmark_density_encoding.py --n_components 10 --subject_idx 0

  # Multiple subjects, custom tolerance
  python benchmark_density_encoding.py --n_components 10 --subject_idx 0,1,2 --tol 5

  # Only run C and D to compare them directly
  python benchmark_density_encoding.py --n_components 10 --subject_idx 0 --variants C,D

  # With timeout (recommended for N=20)
  python benchmark_density_encoding.py --n_components 20 --subject_idx 0 --timeout 600
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


# ─────────────────────────────────────────────────────────────────────────────
# Density encoding builders
# ─────────────────────────────────────────────────────────────────────────────

def density_edge_bounds(GT_density, N, tol_pct):
    """
    Convert GT_density (0-100 scale) and tolerance into edge-count bounds
    suitable for hard cardinality constraints.

    Uses the same integer floor as #count in the ASP program.
    Returns (d_lo, d_hi) as edge counts.
    """
    n_sq = N * N
    d_lo = max(0, int((GT_density - tol_pct) * n_sq / 100))
    d_hi = min(n_sq, int((GT_density + tol_pct) * n_sq / 100) + 1)
    return d_lo, d_hi


def build_density_block(variant, GT_density, N, density_weight, tol_pct):
    """
    Return the ASP snippet that encodes density for the given variant.
    The snippet is appended to the base command (which has no density encoding).

    The density scale matches drasl_command: bins of 2%, so d_bins = GT_density//2
    and hypoth_density = 50 * edge_count / N².  abs_diff is in bin units.

    variant : one of 'A', 'B', 'C', 'D'.  Variant 'E' is built by the
              orchestration loop, not here, since it is a sequence of
              attempts with different (mode, tol) combinations.
    """
    d_bins = GT_density // 2   # same quantisation as drasl_command
    n_sq = N * N
    d_lo, d_hi = density_edge_bounds(GT_density, N, tol_pct)

    # Atoms shared by all variants that need the density value
    shared = (
        f"#const d = {d_bins}. "
        f"countedge1(C) :- C = #count{{edge1(X,Y) : edge1(X,Y), node(X), node(Y)}}. "
        f"countfull(C) :- C = n*n. "
        f"hypoth_density(D) :- D = 50*X/Y, countfull(Y), countedge1(X). "
        f"abs_diff(Diff) :- hypoth_density(D), Diff = |D - d|. "
    )

    hard_bounds = (
        f"#const d_lo = {d_lo}. "
        f"#const d_hi = {d_hi}. "
        f":- countedge1(K), K < d_lo. "
        f":- countedge1(K), K > d_hi. "
    )

    if variant == 'A':
        # Current: soft penalty at @1, no hard bounds
        return shared + f":~ abs_diff(Diff). [Diff*{density_weight}@1] "

    elif variant == 'B':
        # Hard bounds only, no soft penalty, no density in objective
        return shared + hard_bounds

    elif variant == 'C':
        # Hard bounds + soft density at @0 (below edge matching)
        return shared + hard_bounds + f":~ abs_diff(Diff). [Diff*{density_weight}@0] "

    elif variant == 'D':
        # Hard bounds + soft density at @1 (same level as edges)
        return shared + hard_bounds + f":~ abs_diff(Diff). [Diff*{density_weight}@1] "

    else:
        raise ValueError(f"Unknown variant: {variant!r}")


def build_command_for_variant(base_command_bytes, variant, GT_density, N,
                               density_weight, tol_pct):
    """
    Append the variant-specific density block to a base ASP command that
    was built without any density encoding (GT_density=None).
    """
    base_str = base_command_bytes.decode()
    density_str = build_density_block(variant, GT_density, N, density_weight, tol_pct)
    full_str = base_str + " " + density_str
    return full_str.encode().replace(b"\n", b" ")


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
        return {
            "elapsed_s": round(time.time() - self._start, 2),
            "atoms": self.atoms,
            "rules": self.rules,
            "weight_rules": self.weight_rules,
            "minimize_stmts": self.minimize_stmts,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Run one variant
# ─────────────────────────────────────────────────────────────────────────────

def _sg(obj, key, default=0):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError):
        if hasattr(obj, 'get'):
            return obj.get(key, default)
        return default


def run_variant(command, variant_label, pnum, capsize, optim, timeout,
                grounding_interval, n_nodes, configuration="crafty",
                extra_clingo_args=None):
    clingo_args = [
        "--warn=no-atom-undefined",
        f"--configuration={configuration}",
        "-t", f"{int(pnum)},split",
        "-n", str(capsize),
    ]
    if extra_clingo_args:
        clingo_args.extend(extra_clingo_args)

    print(f"\n{'#'*70}", flush=True)
    print(f"  VARIANT: {variant_label}", flush=True)
    print(f"{'#'*70}", flush=True)
    print(f"  optim={optim}  capsize={capsize}  threads={pnum}  "
          f"timeout={timeout if timeout else 'none'}  "
          f"extra={extra_clingo_args or []}", flush=True)

    ctrl = clngo.Control(clingo_args)
    ctrl.configuration.solve.opt_mode = optim

    observer = GroundingObserver(report_interval=grounding_interval)
    ctrl.register_observer(observer)

    t0 = time.time()
    ctrl.add("base", [], command.decode())
    ctrl.ground([("base", [])])
    t_ground = time.time() - t0
    ground_stats = observer.summary()
    print(f"  [GROUNDING] Done in {t_ground:.2f}s  "
          f"(atoms={ground_stats['atoms']:,}  rules={ground_stats['rules']:,}  "
          f"weight_rules={ground_stats['weight_rules']:,}  "
          f"minimize_stmts={ground_stats['minimize_stmts']:,})", flush=True)

    print(f"\n  [SOLVING]...", flush=True)
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
            print(f"\n    *** TIMEOUT ({timeout}s) ***", flush=True)
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
                    cost_str = f"cost={list(cost)}"
                    print(f"    [#{model_count} at {elapsed:.1f}s]  "
                          f"{cost_str}  optimal={optimality}{improved}", flush=True)
                elif model_count == 11:
                    print(f"    ... (suppressing per-model output) ...", flush=True)
    finally:
        if timer is not None:
            timer.cancel()

    t_solve = time.time() - t0

    stats = ctrl.statistics
    summary  = _sg(stats, "summary", {})
    solving  = _sg(stats, "solving", {})
    solvers  = _sg(solving, "solvers", {})
    costs    = _sg(summary, "costs", [])
    mstats   = _sg(summary, "models", {})
    n_optimal    = _sg(mstats, "optimal", 0)
    n_enumerated = _sg(mstats, "enumerated", 0)
    times    = _sg(summary, "times", {})
    total_time   = _sg(times, "total", 0)
    solve_time   = _sg(times, "solve", 0)
    sat_time     = _sg(times, "sat", 0)
    unsat_time   = _sg(times, "unsat", 0)
    solver0  = _sg(solvers, 0, _sg(solvers, "0", {}))
    choices   = _sg(solver0, "choices", 0)
    conflicts = _sg(solver0, "conflicts", 0)
    restarts  = _sg(solver0, "restarts", 0)

    print(f"\n  [RESULT]  solve={t_solve:.2f}s  "
          f"clingo_total={total_time:.2f}s  "
          f"(sat={sat_time:.2f}  unsat={unsat_time:.2f})", flush=True)
    print(f"    models: enumerated={n_enumerated}  optimal={n_optimal}  "
          f"returned={model_count}", flush=True)
    print(f"    cost vector: {list(costs)}", flush=True)
    print(f"    choices={choices:,.0f}  conflicts={conflicts:,.0f}  "
          f"restarts={restarts:,.0f}", flush=True)
    if timed_out:
        print(f"    *** TIMED OUT — results are best-so-far ***", flush=True)

    solutions = []
    if models:
        r_estimated = {(drasl_jclingo2g(m[0]), tuple(m[1])) for m in models}
        for answer in r_estimated:
            graph_num      = answer[0][0]
            undersampling  = answer[0][1]
            cost_tuple     = answer[1]
            res_cg = bfutils.num2CG(graph_num, n_nodes)
            solutions.append({
                "cost_vec":     cost_tuple,
                "cost_primary": cost_tuple[0] if cost_tuple else 0,
                "graph_num":    graph_num,
                "undersampling": undersampling,
                "density":      gk.density(res_cg),
            })
        solutions.sort(key=lambda x: x["cost_vec"])
        print(f"    unique solutions: {len(solutions)}  "
              f"top-3 costs: {[s['cost_vec'] for s in solutions[:3]]}", flush=True)

    return {
        "variant":        variant_label,
        "t_ground":       round(t_ground, 3),
        "t_solve":        round(t_solve, 3),
        "clingo_total":   round(total_time, 3),
        "clingo_solve":   round(solve_time, 3),
        "clingo_sat":     round(sat_time, 3),
        "clingo_unsat":   round(unsat_time, 3),
        "n_returned":     model_count,
        "n_enumerated":   int(n_enumerated),
        "n_optimal":      int(n_optimal),
        "best_cost":      list(costs),
        "choices":        int(choices),
        "conflicts":      int(conflicts),
        "restarts":       int(restarts),
        "ground_atoms":   ground_stats["atoms"],
        "ground_rules":   ground_stats["rules"],
        "ground_minimize": ground_stats["minimize_stmts"],
        "solutions":      solutions,
        "timed_out":      timed_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Benchmark density encoding strategies A/B/C/D for DRASL")
    p.add_argument("--n_components", type=int, default=10, choices=[10, 20, 53])
    p.add_argument("--subject_idx", type=str, default="0",
                   help="Comma-separated subject indices")
    p.add_argument("--data_path", type=str,
                   default="../fbirn/fbirn_sz_data.npz")
    p.add_argument("--scc_strategy", type=str, default="domain",
                   choices=["domain", "correlation", "estimated", "none"])
    p.add_argument("--PNUM", type=int,
                   default=int(min(CLINGO_LIMIT, get_process_count(1))))
    p.add_argument("--MAXU", type=int, default=5)
    p.add_argument("--PRIORITY", type=str, default="11112")
    p.add_argument("--gt_density", type=int, default=None,
                   help="Override GT_density (0-100). If omitted, GT_density is "
                        "computed per-subject from the directed-edge density of "
                        "the PCMCI graph (production default).")
    p.add_argument("--tol", type=int, default=5,
                   help="Density tolerance in percentage points for hard bounds "
                        "(variants B/C/D and the first attempt of variant E).")
    p.add_argument("--tol_widen", type=int, default=10,
                   help="Wider tolerance used by variant E's second attempt "
                        "before falling back to soft-only.")
    p.add_argument("--density_weight", type=int, default=50,
                   help="Weight multiplier on abs_diff for soft density penalty")
    p.add_argument("--timeout", type=int, default=0,
                   help="Per-variant timeout in seconds (0=no limit)")
    p.add_argument("--capsize", type=int, default=0)
    p.add_argument("--pcmci_method", default="pcmci")
    p.add_argument("--pcmci_tau_max", type=int, default=1)
    p.add_argument("--pcmci_alpha", type=float, default=0.05)
    p.add_argument("--pcmci_fdr", default="none")
    p.add_argument("--grounding_interval", type=float, default=5.0)
    p.add_argument("--variants", type=str, default="E,A,C,D",
                   help="Comma-separated variants to run. Default 'E,A,C,D' "
                        "puts the production adaptive variant first, then "
                        "the legacy A and the explicit C/D for comparison.")
    p.add_argument("--configuration", type=str, default="crafty")
    p.add_argument("--extra_clingo_args", nargs=argparse.REMAINDER, default=[],
                   help="Extra clingo flags appended verbatim to every variant. "
                        "Must be the LAST argument on the command line. "
                        "Example: --extra_clingo_args --opt-strategy=usc,stratify --opt-heuristic=1")
    args = p.parse_args()

    subject_indices = [int(x.strip()) for x in args.subject_idx.split(",")]
    variants = [v.strip().upper() for v in args.variants.split(",")]
    gt_density_override = args.gt_density

    extra_clingo_args = args.extra_clingo_args  # already a list from REMAINDER

    print("=" * 72, flush=True)
    print("DENSITY ENCODING BENCHMARK", flush=True)
    print("=" * 72, flush=True)
    print(f"  Time:           {datetime.now()}", flush=True)
    print(f"  N components:   {args.n_components}", flush=True)
    print(f"  Subjects:       {subject_indices}", flush=True)
    print(f"  Variants:       {variants}", flush=True)
    print(f"  Tol (±%):       {args.tol}  (widen={args.tol_widen} for E)", flush=True)
    print(f"  Density weight: {args.density_weight}", flush=True)
    print(f"  MAXCOST:        {MAXCOST}", flush=True)
    print(f"  Timeout/var:    {args.timeout}s" if args.timeout else
          f"  Timeout/var:    none", flush=True)
    print(f"  Configuration:  {args.configuration}", flush=True)
    print(f"  Extra clingo:   {extra_clingo_args or '(none)'}", flush=True)
    print("=" * 72, flush=True)

    VARIANT_DESCRIPTIONS = {
        'A': "soft @1 only (current, no hard bounds)",
        'B': f"hard bounds only [GT±{args.tol}%], no soft penalty",
        'C': f"hard bounds [GT±{args.tol}%] + soft density @0 (lex below edges)",
        'D': f"hard bounds [GT±{args.tol}%] + soft density @1 (same as edges)",
        'E': f"adaptive (production): C@tol={args.tol} → C@tol={args.tol_widen} → A",
    }

    # Load data
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "..", "real_data",
                                 data_path)
    npzfile  = np.load(data_path)
    data     = npzfile["data"]
    labels_key = "labels" if "labels" in npzfile.files else "label"
    labels   = npzfile[labels_key]
    comp_indices = get_comp_indices(args.n_components)
    n_nodes  = len(comp_indices)

    all_results = {}   # subject_idx -> list of result dicts

    for s_idx in subject_indices:
        print(f"\n{'='*72}", flush=True)
        print(f"  SUBJECT {s_idx}  (group={int(labels[s_idx])})", flush=True)
        print(f"{'='*72}", flush=True)

        ts_2d = data[s_idx][:, comp_indices]
        print(f"  Time-series shape: {ts_2d.shape}", flush=True)

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

        # Per-subject GT_density.  Production behaviour: derive from this
        # subject's PCMCI graph rather than using a fixed population value,
        # so the hard cardinality bounds do not exclude subjects whose
        # actual causal density differs from the population mean.
        if gt_density_override is not None:
            gt_density = gt_density_override
            gt_source = "user-override"
        else:
            gt_density = int(round(100.0 * density))
            gt_source = "per-subject from PCMCI"
        print(f"  PCMCI done in {t_pcmci:.2f}s: density={density:.3f}  "
              f"GT_density={gt_density}  ({gt_source})", flush=True)

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
        DD = (np.abs((np.abs(A / a_max) + (cv.graph2adj(g_estimated) - 1))
                     * MAXCOST)).astype(int) if a_max > 0 else \
             (np.abs((cv.graph2adj(g_estimated) - 1) * MAXCOST)).astype(int)
        BD = (np.abs((np.abs(B / b_max) + (cv.graph2badj(g_estimated) - 1))
                     * MAXCOST)).astype(int) if b_max > 0 else \
             (np.abs((cv.graph2badj(g_estimated) - 1) * MAXCOST)).astype(int)

        priority = [int(c) for c in args.PRIORITY]
        urate    = min(args.MAXU, 3 * n_nodes + 1)

        # Build base ASP command WITHOUT any density encoding.
        # Each variant appends its own density block.  Use
        # density_mode='none' to suppress the auto-density that
        # drasl_command otherwise emits when GT_density is None.
        base_command = drasl_command(
            [g_estimated], max_urate=urate, weighted=True,
            scc=use_scc, scc_members=scc_members,
            dm=[DD], bdm=[BD], edge_weights=priority,
            GT_density=None,
            density_mode='none',
            selfloop=False,
        )
        print(f"  Base ASP program (no density): {len(base_command):,} bytes",
              flush=True)

        # Compute and print the hard-bound edge counts for information
        d_lo, d_hi = density_edge_bounds(gt_density, n_nodes, args.tol)
        print(f"  Hard bounds (B/C/D): d_lo={d_lo} edges, d_hi={d_hi} edges  "
              f"(GT={gt_density}%, tol=±{args.tol}%, N={n_nodes})", flush=True)

        # Detect colliding (X,Y) pairs where DD[x,y] == BD[x,y]
        # for x<y (only pairs where both directed and bidirected constraints fire)
        collisions = []
        for xi in range(n_nodes):
            for yi in range(xi + 1, n_nodes):
                if DD[xi, yi] == BD[xi, yi] and DD[xi, yi] > 0:
                    collisions.append((xi + 1, yi + 1, DD[xi, yi]))
        print(f"  Weight collisions (DD==BD, x<y, w>0): {len(collisions)}  "
              f"(expected cost loss with old tuples: "
              f"~{sum(c[2] for c in collisions)} units)", flush=True)

        subject_results = []
        for v in variants:
            desc = VARIANT_DESCRIPTIONS.get(v, v)

            if v == 'E':
                # Adaptive: try C@tol, then C@tol_widen, then A.  Track
                # cumulative time and which attempt succeeded so the
                # summary table reflects what production would experience.
                print(f"\n  ========== Variant E (adaptive production) ==========",
                      flush=True)
                attempts = [
                    ('C', args.tol,       f"E.1 hard+soft0 @ tol=±{args.tol}%"),
                    ('C', args.tol_widen, f"E.2 hard+soft0 @ tol=±{args.tol_widen}% (widened)"),
                    ('A', None,           "E.3 soft fallback (no bounds)"),
                ]
                cumulative_solve = 0.0
                cumulative_total = 0.0
                final_result = None
                final_attempt_label = None
                for attempt_idx, (sub_v, this_tol, label) in enumerate(attempts, 1):
                    print(f"\n  --- {label} ---", flush=True)
                    use_tol = this_tol if this_tol is not None else args.tol
                    cmd = build_command_for_variant(
                        base_command, sub_v, gt_density, n_nodes,
                        args.density_weight, use_tol,
                    )
                    print(f"  Full command: {len(cmd):,} bytes", flush=True)
                    attempt_result = run_variant(
                        command=cmd,
                        variant_label=f"Variant E (attempt {attempt_idx}: {label})",
                        pnum=args.PNUM,
                        capsize=args.capsize,
                        optim="optN",
                        timeout=args.timeout,
                        grounding_interval=args.grounding_interval,
                        n_nodes=n_nodes,
                        configuration=args.configuration,
                        extra_clingo_args=extra_clingo_args or None,
                    )
                    cumulative_solve += attempt_result["t_solve"]
                    cumulative_total += attempt_result["clingo_total"]
                    has_solutions = (
                        attempt_result["solutions"] and
                        not all(s.get("cost_primary") == float("inf")
                                for s in attempt_result["solutions"])
                        and any(c not in ([float("inf")], (float("inf"),))
                                for c in [attempt_result["best_cost"]])
                    )
                    if has_solutions:
                        final_result = attempt_result
                        final_attempt_label = label
                        print(f"  -> Variant E succeeded at attempt {attempt_idx}: "
                              f"{label}", flush=True)
                        break
                    else:
                        print(f"  -> attempt {attempt_idx} UNSAT/empty, escalating",
                              flush=True)

                if final_result is None:
                    # Even soft fallback returned nothing (extreme corner case).
                    final_result = attempt_result
                    final_attempt_label = "all attempts failed"

                # Override the cumulative timing fields so the summary
                # reflects the *total* cost of running E end-to-end.
                final_result["t_solve"]      = round(cumulative_solve, 3)
                final_result["clingo_total"] = round(cumulative_total, 3)
                final_result["adaptive_attempt"] = final_attempt_label
                final_result["subject"]      = s_idx
                final_result["variant_key"]  = v
                subject_results.append(final_result)
                continue

            print(f"\n  Building command for Variant {v}: {desc}", flush=True)
            cmd = build_command_for_variant(
                base_command, v, gt_density, n_nodes,
                args.density_weight, args.tol,
            )
            print(f"  Full command: {len(cmd):,} bytes", flush=True)

            result = run_variant(
                command=cmd,
                variant_label=f"Variant {v}: {desc}",
                pnum=args.PNUM,
                capsize=args.capsize,
                optim="optN",
                timeout=args.timeout,
                grounding_interval=args.grounding_interval,
                n_nodes=n_nodes,
                configuration=args.configuration,
                extra_clingo_args=extra_clingo_args or None,
            )
            result["subject"] = s_idx
            result["variant_key"] = v
            subject_results.append(result)

        all_results[s_idx] = subject_results

    # ── Summary tables ──────────────────────────────────────────────────────
    print(f"\n\n{'='*90}", flush=True)
    print("DENSITY ENCODING BENCHMARK — SUMMARY", flush=True)
    print(f"{'='*90}", flush=True)

    for s_idx in subject_indices:
        results = all_results[s_idx]
        print(f"\n  Subject {s_idx}:", flush=True)

        header = (f"  {'Variant':<10s} {'Solve(s)':>10s} {'Total(s)':>10s} "
                  f"{'Models':>8s} {'Optimal':>8s} {'#Sols':>7s} "
                  f"{'CostVec':>20s} {'Choices':>10s} {'Conflicts':>10s} "
                  f"{'Minimize':>9s} {'Note':>5s}")
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)

        for r in results:
            cost_str = str(r["best_cost"])
            if len(cost_str) > 20:
                cost_str = cost_str[:19] + "~"
            note = "T/O" if r.get("timed_out") else ""
            n_sol = len(r["solutions"])
            print(f"  {r['variant_key']:<10s} "
                  f"{r['t_solve']:>10.2f} "
                  f"{r['clingo_total']:>10.2f} "
                  f"{r['n_returned']:>8d} "
                  f"{r['n_optimal']:>8d} "
                  f"{n_sol:>7d} "
                  f"{cost_str:>20s} "
                  f"{r['choices']:>10,d} "
                  f"{r['conflicts']:>10,d} "
                  f"{r['ground_minimize']:>9d} "
                  f"{note:>5s}",
                  flush=True)

        # Speedup vs A (if A was run)
        a_result = next((r for r in results if r["variant_key"] == "A"), None)
        if a_result and a_result["t_solve"] > 0:
            print(f"\n  Speedup vs Variant A:", flush=True)
            for r in results:
                if r["variant_key"] != "A" and r["t_solve"] > 0:
                    speedup = a_result["t_solve"] / r["t_solve"]
                    print(f"    Variant {r['variant_key']}: {speedup:.2f}x",
                          flush=True)

    # ── Solution comparison ──────────────────────────────────────────────────
    print(f"\n\n{'='*90}", flush=True)
    print("SOLUTION COMPARISON", flush=True)
    print(f"{'='*90}", flush=True)

    for s_idx in subject_indices:
        results = all_results[s_idx]
        print(f"\n  Subject {s_idx}:", flush=True)

        # Compare primary (first-level) costs
        print(f"  {'Variant':<10s}  {'Primary cost (@1)':>18s}  "
              f"{'Secondary cost (@0)':>20s}  {'# optimal sols':>15s}", flush=True)
        print(f"  " + "-" * 70, flush=True)
        for r in results:
            bv = r["best_cost"]
            c1 = bv[0] if len(bv) > 0 else "?"
            c0 = bv[1] if len(bv) > 1 else "n/a"
            n_opt = r["n_optimal"]
            note = " (T/O)" if r.get("timed_out") else ""
            print(f"  {r['variant_key']:<10s}  {str(c1):>18s}  "
                  f"{str(c0):>20s}  {n_opt:>15d}{note}", flush=True)

        # Compare optimal solution sets (graph, undersampling) at best primary cost
        print(f"\n  Optimal solution sets (graph_num, u) per variant:", flush=True)
        reference_set = None
        reference_key = None
        for r in results:
            sols = r["solutions"]
            if not sols:
                print(f"    Variant {r['variant_key']}: NO SOLUTIONS", flush=True)
                continue
            # primary cost is cost_vec[0]
            best_primary = min(s["cost_primary"] for s in sols)
            opt_set = frozenset(
                (s["graph_num"], s["undersampling"])
                for s in sols if s["cost_primary"] == best_primary
            )
            densities = [s["density"] for s in sols
                         if s["cost_primary"] == best_primary]
            avg_d = np.mean(densities) if densities else float("nan")
            top3 = sols[:3]
            top3_str = "  ".join(
                f"g={s['graph_num']} u={s['undersampling']} "
                f"d={s['density']:.3f} cost={s['cost_vec']}"
                for s in top3
            )
            print(f"    Variant {r['variant_key']}: "
                  f"{len(opt_set)} solutions at primary_cost={best_primary}  "
                  f"avg_density={avg_d:.3f}", flush=True)
            print(f"      top-3: {top3_str}", flush=True)

            if reference_set is None:
                reference_set = opt_set
                reference_key = r["variant_key"]
            else:
                shared  = len(opt_set & reference_set)
                only_r  = len(opt_set - reference_set)
                only_ref = len(reference_set - opt_set)
                if opt_set == reference_set:
                    print(f"      -> optimal set IDENTICAL to Variant {reference_key}",
                          flush=True)
                else:
                    print(f"      -> DIFFERS from Variant {reference_key}: "
                          f"{shared} shared, {only_r} only here, "
                          f"{only_ref} only in {reference_key}", flush=True)

    # ── Multi-subject aggregate ──────────────────────────────────────────────
    if len(subject_indices) > 1:
        print(f"\n\n{'='*90}", flush=True)
        print("AGGREGATE ACROSS SUBJECTS (mean solve time)", flush=True)
        print(f"{'='*90}", flush=True)
        header = (f"  {'Variant':<10s} {'Mean Solve':>12s} "
                  f"{'Min Solve':>12s} {'Max Solve':>12s}")
        print(header, flush=True)
        print("  " + "-" * (len(header) - 2), flush=True)
        for v in variants:
            times = [r["t_solve"]
                     for s_idx in subject_indices
                     for r in all_results[s_idx]
                     if r["variant_key"] == v]
            if times:
                print(f"  {v:<10s} "
                      f"{np.mean(times):>12.2f} "
                      f"{np.min(times):>12.2f} "
                      f"{np.max(times):>12.2f}", flush=True)

    print(f"\nDone at {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
