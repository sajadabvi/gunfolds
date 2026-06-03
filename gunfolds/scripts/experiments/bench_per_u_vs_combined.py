"""
Controlled A/B benchmark: combined-u clingo vs per-u-split clingo, on the
SAME graphs.

For each of K deterministic graphs (one N, instances 0..K-1) we build the drasl
input ONCE (seed → SCC-ring G¹ → VAR → BOLD → PCMCI → DD/BD, exactly as
runtime_scaling.py), then solve that identical input two ways:

  A) combined : one clingo run with uk(1..max_urate)         (the old way)
  B) per-u    : one clingo run per fixed u (fix_urate=u), then pool the
                solutions.  Reported two ways —
                  critical = max_u solve_time   (rates run in parallel on SLURM)
                  total    = sum_u solve_time   (CPU booked / serial)

Both conditions use the IDENTICAL clingo configuration (opt mode, threads,
timeout) and the IDENTICAL drasl encoding except for the u restriction, so any
time difference is attributable to combined-vs-split alone.

We also compare the best (lowest sum-cost) solution each side finds — they
should agree (or the split finds an equal/lower sum; see the lex-vs-sum note in
aggregate_per_u.py).

Usage:
    python bench_per_u_vs_combined.py --n_nodes 12 --n_graphs 5 \
        --timeout_sec 1800 --pnum 8 --out bench_n12.csv
"""

import argparse
import csv
import os
import sys
import time
import threading

import numpy as np
import random

import clingo as clngo

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..', '..')
sys.path.insert(0, _ROOT)

from gunfolds.conversions import drasl_jclingo2g
from gunfolds.solvers.clingo_rasl import drasl_command
from gunfolds import conversions as cv
from gunfolds.utils import graphkit as gk

# Reuse the exact pipeline + constants from the runtime-scaling experiment so
# the inputs are identical to the real benchmark.
from gunfolds.scripts.experiments.runtime_scaling import (
    derive_seed, make_multi_scc_ring, verify_partition,
    get_stable_weighted_matrix, simulate_var, simulate_bold,
    SCC_COMPOSITION, MAXCOST, PRIORITY, DENSITY_MODE, TOL_LOW, TOL_HIGH,
    MAX_URATE, SSIZE, NOISE,
)

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr


# ─────────────────────────────────────────────────────────────────────────────
# Build the drasl input for one (n_nodes, instance) — mirrors runtime_scaling.
# ─────────────────────────────────────────────────────────────────────────────

def build_input(n_nodes, instance_id, ssize, u_rate_gen):
    seed = derive_seed(n_nodes, instance_id)
    np.random.seed(seed); random.seed(seed)
    scc_sizes = SCC_COMPOSITION[n_nodes]
    gt_g, partition = make_multi_scc_ring(scc_sizes, dens=0.5, dag_degree=1,
                                          max_cross_connections=2)
    verify_partition(gt_g, partition, max_scc_size=6)
    A = cv.graph2adj(gt_g)
    W = get_stable_weighted_matrix(A, partition=partition, strategy='sample',
                                   threshold=0.0, powers=(),
                                   scale_aware=True, bias_magnitudes=False,
                                   auto_threshold=False)
    var_data = simulate_var(W, ssize=ssize * u_rate_gen, noise=NOISE)
    bold = simulate_bold(var_data, u_rate=u_rate_gen)
    dataframe = pp.DataFrame(bold.T)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05, fdr_method="none")
    g_est, A_pc, B_pc = cv.Glag2CG(res)

    a_max = np.abs(A_pc).max(); b_max = np.abs(B_pc).max()
    if a_max > 0:
        DD = (np.abs((np.abs(A_pc / a_max) + (cv.graph2adj(g_est) - 1)) * MAXCOST)).astype(int)
    else:
        DD = (np.abs((cv.graph2adj(g_est) - 1) * MAXCOST)).astype(int)
    if b_max > 0:
        BD = (np.abs((np.abs(B_pc / b_max) + (cv.graph2badj(g_est) - 1)) * MAXCOST)).astype(int)
    else:
        BD = (np.abs((cv.graph2badj(g_est) - 1) * MAXCOST)).astype(int)

    urate = min(MAX_URATE, 3 * len(g_est) + 1)
    gt_density = max(1, int(round(100.0 * gk.density(gt_g))))
    return dict(g_est=g_est, DD=DD, BD=BD, partition=partition,
                gt_density=gt_density, urate=urate, n_nodes=len(g_est))


def make_command(P, fix_urate=None):
    return drasl_command(
        [P["g_est"]], max_urate=P["urate"], weighted=True, scc=True,
        scc_members=P["partition"], dm=[P["DD"]], bdm=[P["BD"]],
        edge_weights=PRIORITY, GT_density=P["gt_density"], selfloop=False,
        density_mode=DENSITY_MODE, tol=None, tol_low=TOL_LOW, tol_high=TOL_HIGH,
        fix_urate=fix_urate,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Identical clingo solve for both conditions (opt mode, threads, timeout).
# ─────────────────────────────────────────────────────────────────────────────

def solve(command, pnum, timeout_sec):
    """Return (elapsed_sec, models, timed_out).  models: list of (cost_sum,
    graph_num, u_tuple)."""
    base_args = ["--warn=no-atom-undefined", "--configuration=crafty",
                 "-t", f"{int(pnum)},split", "-n", "0"]
    ctrl = clngo.Control(base_args)
    ctrl.configuration.solve.opt_mode = "opt"
    t0 = time.perf_counter()
    ctrl.add("base", [], command.decode())
    ctrl.ground([("base", [])])

    models, timed_out, timer = [], False, None
    if timeout_sec and timeout_sec > 0:
        remaining = max(1.0, timeout_sec - (time.perf_counter() - t0))
        def _interrupt():
            nonlocal timed_out
            timed_out = True
            ctrl.interrupt()
        timer = threading.Timer(remaining, _interrupt); timer.start()
    try:
        with ctrl.solve(yield_=True, async_=True) as h:
            for m in h:
                cost = sum(m.cost) if m.cost else 0
                parsed = drasl_jclingo2g([str(a) for a in m.symbols(shown=True)])
                if isinstance(parsed, set):
                    parsed = next(iter(parsed))
                models.append((int(cost), int(parsed[0]),
                               tuple(int(x) for x in parsed[1])))
    finally:
        if timer is not None:
            timer.cancel()
    return time.perf_counter() - t0, models, timed_out


def best_cost(models):
    return min((c for c, _, _ in models), default=None)


# ─────────────────────────────────────────────────────────────────────────────

def fmt(t):
    if t is None:
        return "   --   "
    if t < 60:
        return f"{t:8.2f}s"
    if t < 3600:
        return f"{t/60:8.2f}m"
    return f"{t/3600:8.2f}h"


def main():
    ap = argparse.ArgumentParser(description="A/B benchmark: combined-u vs per-u split.")
    ap.add_argument("--n_nodes", type=int, default=12, choices=sorted(SCC_COMPOSITION))
    ap.add_argument("--n_graphs", type=int, default=5, help="instances 0..n_graphs-1")
    ap.add_argument("--pnum", type=int, default=8, help="clingo threads (same for A and B)")
    ap.add_argument("--timeout_sec", type=int, default=1800,
                    help="per-solve timeout (applies to the combined run and to "
                         "EACH per-u run)")
    ap.add_argument("--ssize", type=int, default=SSIZE)
    ap.add_argument("--u_rate_gen", type=int, default=2, help="data-gen undersampling")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rates = list(range(2, MAX_URATE + 1))  # weighted: u starts at 2
    print(f"Benchmark  N={args.n_nodes}  graphs={args.n_graphs}  "
          f"u-rates(split)={rates}  pnum={args.pnum}  timeout={args.timeout_sec}s")
    print("=" * 96)

    rows = []
    for inst in range(args.n_graphs):
        print(f"\n[graph {inst}] building input (seed from N={args.n_nodes}, inst={inst}) …",
              flush=True)
        P = build_input(args.n_nodes, inst, args.ssize, args.u_rate_gen)
        print(f"          n={P['n_nodes']}  urate={P['urate']}  gt_density={P['gt_density']}",
              flush=True)

        # A) combined
        ta, ma, toa = solve(make_command(P, fix_urate=None), args.pnum, args.timeout_sec)
        print(f"  A combined : {fmt(ta)}  models={len(ma):5d}  best={best_cost(ma)}"
              f"{'  TIMEOUT' if toa else ''}", flush=True)

        # B) per-u
        per_u_times, pool, any_to = {}, [], False
        for u in rates:
            tu, mu, tou = solve(make_command(P, fix_urate=u), args.pnum, args.timeout_sec)
            per_u_times[u] = tu
            pool.extend(mu)
            any_to = any_to or tou
            print(f"    B u={u}    : {fmt(tu)}  models={len(mu):5d}  best={best_cost(mu)}"
                  f"{'  TIMEOUT' if tou else ''}", flush=True)
        crit = max(per_u_times.values())
        tot = sum(per_u_times.values())
        b_best = best_cost(pool)

        sp_crit = ta / crit if crit else None
        sp_tot = ta / tot if tot else None
        rows.append(dict(inst=inst, n=P['n_nodes'], combined=ta, a_to=toa,
                         crit=crit, total=tot, b_to=any_to,
                         a_best=best_cost(ma), b_best=b_best,
                         sp_crit=sp_crit, sp_tot=sp_tot,
                         per_u=dict(per_u_times)))
        verdict = "split faster" if (sp_crit and sp_crit > 1) else "combined faster"
        print(f"  => combined {fmt(ta)}  |  split crit {fmt(crit)} (sum {fmt(tot)})  "
              f"|  crit speedup {sp_crit:.2f}x  [{verdict}]"
              f"{'  best-cost MISMATCH' if best_cost(ma) != b_best else ''}", flush=True)

    # ── summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print(f"{'graph':>5} {'combined':>10} {'split_crit':>11} {'split_sum':>10} "
          f"{'crit_x':>7} {'sum_x':>7}  {'best A/B':>14}")
    print("-" * 96)
    for r in rows:
        bm = "match" if r["a_best"] == r["b_best"] else f"{r['a_best']}/{r['b_best']}"
        sc = f"{r['sp_crit']:.2f}" if r["sp_crit"] else "--"
        ss = f"{r['sp_tot']:.2f}" if r["sp_tot"] else "--"
        print(f"{r['inst']:>5} {fmt(r['combined']):>10} {fmt(r['crit']):>11} "
              f"{fmt(r['total']):>10} {sc:>7} {ss:>7}  {bm:>14}")
    import statistics as stt
    crit_x = [r["sp_crit"] for r in rows if r["sp_crit"]]
    if crit_x:
        faster = sum(1 for x in crit_x if x > 1)
        print("-" * 96)
        print(f"median critical-path speedup: {stt.median(crit_x):.2f}x   "
              f"({faster}/{len(crit_x)} graphs faster split)   "
              f">1 = split wins on wall-clock")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["inst", "n_nodes", "combined_sec", "combined_timeout",
                        "split_critical_sec", "split_total_sec", "split_timeout",
                        "a_best_cost", "b_best_cost", "speedup_critical",
                        "speedup_total"])
            for r in rows:
                w.writerow([r["inst"], r["n"], f"{r['combined']:.4f}", r["a_to"],
                            f"{r['crit']:.4f}", f"{r['total']:.4f}", r["b_to"],
                            r["a_best"], r["b_best"],
                            f"{r['sp_crit']:.4f}" if r["sp_crit"] else "",
                            f"{r['sp_tot']:.4f}" if r["sp_tot"] else ""])
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
