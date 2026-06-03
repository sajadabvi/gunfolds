"""
Per-u (undersampling-rate) split of the drasl runtime-scaling experiment.

Motivation
----------
The monolithic drasl encoding lets clingo search *all* undersampling rates
``u ∈ 1..max_urate`` inside a single solve.  Splitting that search — one
clingo job per fixed ``u`` — was flagged in the wiki as "the biggest single
win": the jobs are independent (embarrassingly parallel on SLURM), each job
grounds a smaller ``uk(1..k)`` program, and the per-rate solution sets are an
exact partition of the original search space (each causal graph has a unique
minimal undersampling rate), so pooling them afterwards loses nothing.

This script implements the **prep-once / solve-per-u** architecture:

    prep   — run the deterministic pipeline (seed → SCC-ring G¹ → VAR → BOLD
             → PCMCI → DD/BD weight matrices) ONCE per (n_nodes, instance) and
             pickle the shared drasl input.  Running it once (rather than
             re-deriving it inside every u-job) guarantees all u-jobs solve
             against bit-for-bit identical inputs, so their clingo costs are
             directly comparable, and avoids paying for PCMCI N times.

    solve  — load that pickled input and run clingo with
             ``drasl_command(..., fix_urate=u_value)``.  Saves the FULL scored
             solution list (every model clingo reported, each with its graph,
             undersampling rate, and cost) to a per-u JSON file.

Aggregation across the per-u JSONs (pool, sort by cost, keep top 30 %) lives
in the separate ``aggregate_per_u.py`` — see that script.

Unique tags
-----------
Every logical run is identified by a ``--run_tag``.  All files for one run
share that tag:  ``<tag>__input.pkl`` (prep) and ``<tag>__u<k>.json`` (solve).
The aggregator groups strictly by tag, so distinct runs never bleed into one
another no matter how many are launched.  The SLURM submitter derives a unique
tag per (label, n_nodes, instance).

Usage
-----
    # 1. prep once per instance
    python runtime_scaling_per_u.py prep \
        --n_nodes 14 --instance_id 3 \
        --run_tag rtpu_n14_inst3 --output_dir results/runtime_scaling_per_u/

    # 2. one solve per undersampling rate (parallel jobs)
    python runtime_scaling_per_u.py solve \
        --run_tag rtpu_n14_inst3 --u_value 2 \
        --output_dir results/runtime_scaling_per_u/ --timeout_hours 35
    #    ... repeat for --u_value 3, 4, ...

    # 3. aggregate (separate script)
    python aggregate_per_u.py --input_dir results/runtime_scaling_per_u/ \
        --run_tag rtpu_n14_inst3

Convention deviations vs the gunfolds checklist are inherited verbatim from
runtime_scaling.py (selfloop=False, PCMCI tau_max=1 / fdr 'none',
density_mode='hard_soft0') — this experiment exists to measure the per-u
speedup on the *same* encoding as the combined-search baseline.
"""

import argparse
import json
import os
import pickle
import sys
import time
import threading
import traceback
from datetime import datetime

import numpy as np

import clingo as clngo

# Make the repo importable when invoked as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..', '..')
sys.path.insert(0, _ROOT)

# Light imports only — the `solve` path must NOT drag in tigramite/BOLD.
from gunfolds.conversions import drasl_jclingo2g
from gunfolds.solvers.clingo_rasl import drasl_command
from gunfolds.utils.calc_procs import get_process_count


# ─────────────────────────────────────────────────────────────────────────────
# clingo constants — mirror runtime_scaling.py / benchmark_domain_heuristic.py.
# Redefined locally (rather than imported from runtime_scaling) so the `solve`
# subcommand stays dependency-light: it never imports the PCMCI/BOLD pipeline.
# ─────────────────────────────────────────────────────────────────────────────
CLINGO_LIMIT = 64
CAPSIZE = 0            # 0 = unlimited improving-model reports; with opt-mode=opt
                       # clingo runs full BnB and reports every improving model.

INPUT_SUFFIX = "__input.pkl"


def input_path(output_dir, run_tag):
    return os.path.join(output_dir, f"{run_tag}{INPUT_SUFFIX}")


def solve_path(output_dir, run_tag, u_value):
    return os.path.join(output_dir, f"{run_tag}__u{u_value}.json")


# ═════════════════════════════════════════════════════════════════════════════
# PREP — deterministic pipeline, run once per (n_nodes, instance)
# ═════════════════════════════════════════════════════════════════════════════

def cmd_prep(args):
    # Heavy pipeline imports are deferred to here so that `solve` jobs (which
    # only need clingo) never import tigramite.  We reuse runtime_scaling.py's
    # helpers verbatim to guarantee the prepped input is identical to the
    # combined-search baseline's input.
    from gunfolds.scripts.experiments.runtime_scaling import (
        derive_seed, make_multi_scc_ring, verify_partition,
        get_stable_weighted_matrix, simulate_var, simulate_bold,
        SCC_COMPOSITION, MAXCOST, PRIORITY, DENSITY_MODE,
        TOL_LOW, TOL_HIGH, MAX_URATE, SSIZE, NOISE,
    )
    import random
    from gunfolds import conversions as cv
    from gunfolds.utils import graphkit as gk
    import tigramite.data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr

    os.makedirs(args.output_dir, exist_ok=True)
    out_pkl = input_path(args.output_dir, args.run_tag)

    if args.n_nodes not in SCC_COMPOSITION:
        raise ValueError(f"n_nodes={args.n_nodes} not in SCC_COMPOSITION "
                         f"{sorted(SCC_COMPOSITION)}")
    scc_sizes = SCC_COMPOSITION[args.n_nodes]

    meta = {
        "run_tag": args.run_tag,
        "n_nodes": args.n_nodes,
        "instance_id": args.instance_id,
        "scc_sizes": scc_sizes,
        "status": "",
    }

    overall_t0 = time.perf_counter()
    print("=" * 70, flush=True)
    print(f"PREP  run_tag={args.run_tag}  n_nodes={args.n_nodes}  "
          f"instance={args.instance_id}", flush=True)
    print(f"  scc_sizes={scc_sizes}  start {datetime.now()}", flush=True)
    print("=" * 70, flush=True)

    try:
        # ── Step 1: deterministic seed ───────────────────────────────────────
        seed = derive_seed(args.n_nodes, args.instance_id)
        np.random.seed(seed)
        random.seed(seed)
        print(f"[1/5] seed = {seed}", flush=True)

        # ── Step 2: ground-truth graph G¹ ────────────────────────────────────
        gt_g, partition = make_multi_scc_ring(
            scc_sizes, dens=0.5, dag_degree=1, max_cross_connections=2)
        verify_partition(gt_g, partition, max_scc_size=6)
        n_nodes_actual = len(gt_g)
        assert n_nodes_actual == args.n_nodes, \
            f"Built {n_nodes_actual} nodes but expected {args.n_nodes}"
        n_edges = sum(len(nbrs) for nbrs in gt_g.values())
        n_extra_edges = n_edges - sum(scc_sizes)
        print(f"[2/5] G¹: nodes={n_nodes_actual}, edges={n_edges}, "
              f"partition_sizes={[len(p) for p in partition]}", flush=True)

        # ── Step 3: VAR + BOLD ───────────────────────────────────────────────
        bold_t0 = time.perf_counter()
        A = cv.graph2adj(gt_g)
        powers_tuple = tuple(int(p) for p in args.w_powers.split(',')
                             if p.strip()) if args.w_powers else ()
        W = get_stable_weighted_matrix(
            A, partition=partition, strategy=args.w_strategy,
            threshold=args.w_threshold, powers=powers_tuple,
            scale_aware=args.w_scale_aware, bias_magnitudes=args.w_bias_magnitudes,
            auto_threshold=args.w_auto_threshold,
        )
        var_data = simulate_var(W, ssize=args.ssize * args.u_rate, noise=args.noise)
        bold_data = simulate_bold(var_data, u_rate=args.u_rate)
        bold_time_sec = round(time.perf_counter() - bold_t0, 4)
        print(f"[3/5] BOLD: shape={bold_data.shape}  ({bold_time_sec:.2f}s)",
              flush=True)

        # ── Step 4: PCMCI ────────────────────────────────────────────────────
        pcmci_t0 = time.perf_counter()
        ts_2d = bold_data.T
        assert ts_2d.shape[1] == n_nodes_actual, \
            f"axis swap detected: ts_2d shape {ts_2d.shape}"
        dataframe = pp.DataFrame(ts_2d)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
        pcmci_results = pcmci.run_pcmci(
            tau_max=1, pc_alpha=None, alpha_level=0.05, fdr_method="none")
        g_estimated, A_pc, B_pc = cv.Glag2CG(pcmci_results)
        pcmci_time_sec = round(time.perf_counter() - pcmci_t0, 4)
        print(f"[4/5] PCMCI: g_est nodes={len(g_estimated)}  "
              f"({pcmci_time_sec:.2f}s)", flush=True)

        # ── Step 5: DD / BD weight matrices (canonical, MAXCOST=20) ──────────
        a_max = np.abs(A_pc).max()
        b_max = np.abs(B_pc).max()
        if a_max > 0:
            DD = (np.abs((np.abs(A_pc / a_max) +
                          (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
        else:
            DD = (np.abs((cv.graph2adj(g_estimated) - 1) * MAXCOST)).astype(int)
        if b_max > 0:
            BD = (np.abs((np.abs(B_pc / b_max) +
                          (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)
        else:
            BD = (np.abs((cv.graph2badj(g_estimated) - 1) * MAXCOST)).astype(int)

        urate_max = min(MAX_URATE, 3 * n_nodes_actual + 1)
        gt_density = max(1, int(round(100.0 * gk.density(gt_g))))
        print(f"[5/5] drasl input ready  urate_max={urate_max}  "
              f"gt_density={gt_density}  DD[{DD.min()},{DD.max()}]  "
              f"BD[{BD.min()},{BD.max()}]", flush=True)

        # ── Persist EVERYTHING the per-u solves + aggregation need ───────────
        payload = {
            # identity
            "run_tag": args.run_tag,
            "n_nodes": n_nodes_actual,
            "instance_id": args.instance_id,
            "seed": seed,
            "scc_sizes": scc_sizes,
            # graphs
            "gt_g": gt_g,                 # ground truth (for F1 in aggregation)
            "partition": partition,       # SCC members (scc_members arg)
            "g_estimated": g_estimated,   # PCMCI estimate (drasl input graph)
            # weights + drasl_command kwargs (so every u-job rebuilds the
            # IDENTICAL command except for fix_urate)
            "DD": DD, "BD": BD,
            "GT_density": gt_density,
            "urate_max": urate_max,
            "priority": list(PRIORITY),
            "density_mode": DENSITY_MODE,
            "tol_low": TOL_LOW, "tol_high": TOL_HIGH,
            "selfloop": False,
            "maxcost": MAXCOST,
            # provenance
            "n_edges": n_edges,
            "n_extra_edges": n_extra_edges,
            "bold_time_sec": bold_time_sec,
            "pcmci_time_sec": pcmci_time_sec,
            "w_strategy": args.w_strategy,
            "ssize": args.ssize, "noise": args.noise, "u_rate": args.u_rate,
        }
        with open(out_pkl, "wb") as f:
            pickle.dump(payload, f)
        meta["status"] = "completed"
        print(f"\nWrote {out_pkl}", flush=True)

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n!! PREP ERROR: {exc}\n{tb}", flush=True)
        meta["status"] = f"error: {str(exc)[:200]}"
        # Surface failure to the scheduler (so afterok dependents don't run).
        _write_status_json(args.output_dir, args.run_tag, "prep", meta,
                           time.perf_counter() - overall_t0)
        sys.exit(1)

    _write_status_json(args.output_dir, args.run_tag, "prep", meta,
                       time.perf_counter() - overall_t0)


def _write_status_json(output_dir, run_tag, phase, meta, elapsed):
    meta = dict(meta)
    meta["phase"] = phase
    meta["elapsed_sec"] = round(elapsed, 3)
    p = os.path.join(output_dir, f"{run_tag}__{phase}_status.json")
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)


# ═════════════════════════════════════════════════════════════════════════════
# SOLVE — one clingo job for a single fixed undersampling rate
# ═════════════════════════════════════════════════════════════════════════════

def _solve_collect_all(command, pnum, timeout_sec):
    """
    Build clingo.Control, add+ground+solve under a threading.Timer interrupt,
    collecting EVERY model clingo reports (opt mode → all improving models),
    each with its full cost vector.  Mirrors runtime_scaling.run_drasl_with_timeout
    but keeps all solutions instead of only the best.

    Returns (drasl_time_sec, models, timed_out) where models is a list of
    (atoms:list[str], cost:list[int]).
    """
    base_args = [
        "--warn=no-atom-undefined",
        "--configuration=crafty",
        "-t", f"{int(pnum)},split",
        "-n", str(CAPSIZE),
    ]
    ctrl = clngo.Control(base_args)
    ctrl.configuration.solve.opt_mode = "opt"

    t0 = time.perf_counter()
    ctrl.add("base", [], command.decode())
    ctrl.ground([("base", [])])

    models = []
    timed_out = False
    timer = None
    if timeout_sec and timeout_sec > 0:
        elapsed_so_far = time.perf_counter() - t0
        remaining = max(1.0, timeout_sec - elapsed_so_far)

        def _interrupt():
            nonlocal timed_out
            timed_out = True
            print(f"    *** TIMEOUT ({timeout_sec}s) — interrupting solver ***",
                  flush=True)
            ctrl.interrupt()
        timer = threading.Timer(remaining, _interrupt)
        timer.start()

    try:
        with ctrl.solve(yield_=True, async_=True) as handle:
            for model in handle:
                cost = list(model.cost)
                atoms = [str(a) for a in model.symbols(shown=True)]
                models.append((atoms, cost))
    finally:
        if timer is not None:
            timer.cancel()

    return time.perf_counter() - t0, models, timed_out


def cmd_solve(args):
    os.makedirs(args.output_dir, exist_ok=True)
    in_pkl = input_path(args.output_dir, args.run_tag)
    if not os.path.exists(in_pkl):
        print(f"!! SOLVE ERROR: prepped input not found: {in_pkl}\n"
              f"   Run `prep` for run_tag={args.run_tag} first.", flush=True)
        sys.exit(1)

    with open(in_pkl, "rb") as f:
        P = pickle.load(f)

    u = args.u_value
    out_json = solve_path(args.output_dir, args.run_tag, u)
    timeout_sec = int(args.timeout_hours * 3600)
    pnum = args.pnum

    result = {
        "run_tag": args.run_tag,
        "n_nodes": P["n_nodes"],
        "instance_id": P["instance_id"],
        "u": u,
        "pnum": pnum,
        "timeout_sec": timeout_sec,
        "status": "",
        "timed_out": False,
        "drasl_time_sec": float("nan"),
        "n_solutions": 0,
        "solutions": [],
    }

    overall_t0 = time.perf_counter()
    print("=" * 70, flush=True)
    print(f"SOLVE run_tag={args.run_tag}  u={u}  n_nodes={P['n_nodes']}  "
          f"instance={P['instance_id']}", flush=True)
    print(f"  pnum={pnum}  timeout={timeout_sec}s  start {datetime.now()}",
          flush=True)
    print("=" * 70, flush=True)

    try:
        if u > P["urate_max"]:
            print(f"  [note] u_value={u} exceeds prepped urate_max="
                  f"{P['urate_max']}; solving anyway (fix_urate drives uk(1..{u})).",
                  flush=True)

        # Rebuild the IDENTICAL drasl_command as the combined baseline, except
        # the undersampling rate is forced to exactly u via fix_urate.
        command = drasl_command(
            [P["g_estimated"]],
            max_urate=P["urate_max"],
            weighted=True,
            scc=True,
            scc_members=P["partition"],
            dm=[P["DD"]], bdm=[P["BD"]],
            edge_weights=P["priority"],
            GT_density=P["GT_density"],
            selfloop=P["selfloop"],
            density_mode=P["density_mode"],
            tol=None, tol_low=P["tol_low"], tol_high=P["tol_high"],
            fix_urate=u,
        )
        print(f"  drasl: ASP {len(command):,} bytes  fix_urate={u}", flush=True)

        drasl_time_sec, models, timed_out = _solve_collect_all(
            command, pnum, timeout_sec)
        result["drasl_time_sec"] = round(drasl_time_sec, 4)
        result["timed_out"] = timed_out

        # Parse + score EVERY model.  This is the `scored` list described in the
        # spec: each entry is a (graph, undersampling-rate, cost) triple.  We
        # keep the full cost vector (lexicographic priority levels) AND its sum
        # (cost_total) so the aggregator can rank either way; cost_total matches
        # runtime_scaling.py's `sum(cost)` convention.
        scored = []
        for atoms, cost in models:
            parsed = drasl_jclingo2g(atoms)
            if isinstance(parsed, set):
                parsed = next(iter(parsed))
            graph_num = int(parsed[0])
            u_tuple = [int(x) for x in parsed[1]]
            cost_total = int(sum(cost)) if cost else 0
            scored.append({
                "graph_num": graph_num,
                "u": u,                  # the forced rate (== u_tuple entries)
                "u_parsed": u_tuple,
                "cost_vector": [int(c) for c in cost],
                "cost_total": cost_total,
            })
        # Sort by least cost (the basis for the later top-30% aggregation).
        scored.sort(key=lambda s: s["cost_total"])

        result["n_solutions"] = len(scored)
        result["solutions"] = scored
        result["status"] = "timeout" if timed_out else "completed"
        best = scored[0]["cost_total"] if scored else None
        print(f"  done: {drasl_time_sec:.2f}s  n_solutions={len(scored)}  "
              f"best_cost={best}  timed_out={timed_out}", flush=True)

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n!! SOLVE ERROR: {exc}\n{tb}", flush=True)
        result["status"] = f"error: {str(exc)[:200]}"

    result["total_time_sec"] = round(time.perf_counter() - overall_t0, 4)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_json}  status={result['status']}  "
          f"n_solutions={result['n_solutions']}", flush=True)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _add_prep_pipeline_args(p):
    """Stable-matrix strategy flags — copied from runtime_scaling.py so prep
    reproduces the baseline's VAR/BOLD behaviour exactly."""
    p.add_argument("--ssize", type=int, default=2500)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--u_rate", type=int, default=2,
                   help="VAR→BOLD undersampling used during DATA GENERATION "
                        "(not the drasl search rate). Mirrors runtime_scaling.")
    p.add_argument("--w_strategy", choices=['sample', 'construct'], default='sample')
    p.add_argument("--w_threshold", type=float, default=0.0)
    p.add_argument("--w_powers", type=str, default="")
    p.add_argument("--w_scale_aware", action='store_true', default=True)
    p.add_argument("--w_no_scale_aware", dest='w_scale_aware', action='store_false')
    p.add_argument("--w_bias_magnitudes", action='store_true', default=False)
    p.add_argument("--w_auto_threshold", action='store_true', default=False)


def main():
    parser = argparse.ArgumentParser(
        description="Per-u split of the drasl runtime-scaling experiment "
                    "(prep-once / solve-per-u).")
    sub = parser.add_subparsers(dest="mode", required=True)

    # prep
    pp_ = sub.add_parser("prep", help="Run the pipeline once; pickle drasl input.")
    pp_.add_argument("--n_nodes", type=int, required=True)
    pp_.add_argument("--instance_id", type=int, required=True)
    pp_.add_argument("--run_tag", type=str, required=True,
                     help="Unique tag grouping this run's prep + all u-solves.")
    pp_.add_argument("--output_dir", type=str,
                     default="results/runtime_scaling_per_u/")
    _add_prep_pipeline_args(pp_)
    pp_.set_defaults(func=cmd_prep)

    # solve
    sp = sub.add_parser("solve", help="Solve clingo for a single fixed u.")
    sp.add_argument("--run_tag", type=str, required=True,
                    help="Must match the run_tag used in `prep`.")
    sp.add_argument("--u_value", type=int, required=True,
                    help="The single undersampling rate this job fixes.")
    sp.add_argument("--output_dir", type=str,
                    default="results/runtime_scaling_per_u/")
    sp.add_argument("--timeout_hours", type=float, default=35.0,
                    help="clingo internal timeout (1h margin under 36h walltime).")
    sp.add_argument("--pnum", type=int,
                    default=int(min(CLINGO_LIMIT, get_process_count(1))))
    sp.set_defaults(func=cmd_solve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
