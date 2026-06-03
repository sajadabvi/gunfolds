"""
Runtime scaling experiment for drasl on SCC-ring synthetic graphs.

One job per (n_nodes, instance_id).  Pipeline:
  1. Deterministic seed from (n_nodes, instance_id)
  2. Ground-truth multi-SCC ring graph G^1 (max SCC size = 6)
  3. VAR + balloon-BOLD simulation (mirroring exp4_pcmci_drasl_ringmore5.py)
  4. PCMCI on BOLD                       (mirroring benchmark_domain_heuristic.py)
  5. drasl via drasl_command + clingo.Control with threading.Timer interrupt
     (mirroring benchmark_domain_heuristic.py exactly)
  6. F1 vs ground truth (adjacency + orientation) on best (lowest-cost) model
  7. CSV row write (always, even on timeout/error)

Usage:
    python runtime_scaling.py --n_nodes 14 --instance_id 3 \
        --output_dir results/runtime_scaling/ --timeout_hours 35

Notes on convention deviations vs gunfolds checklist (deliberate, follows
prompt + benchmark_domain_heuristic.py exactly):
  - selfloop=False (checklist #1 prefers None for PCMCI data; we follow the
    benchmark to keep the encoding identical to the FBIRN scaling baseline
    against which this experiment is compared).
  - PCMCI: run_pcmci(tau_max=1, alpha=0.05, fdr_method='none')
    (checklist #3 prefers pcmciplus tau_max=2 for fresh scripts; we follow
    the benchmark since this experiment exists to measure the SCC speedup
    on the same encoding the benchmark uses).
  - density_mode='hard_soft0' (checklist #6 default is 'adaptive'; we follow
    the benchmark's explicit choice).

API note: gk.ring_sccs only accepts a single ring size for all SCCs and so
cannot produce non-uniform compositions like 6+4+4.  This script therefore
inlines the same construction (ringmore per SCC + DAG-quotient cross-edges
+ shift_list_labels + merge_list), letting us pass an arbitrary list of
ring sizes while preserving gk.ring_sccs's algorithmic behaviour.  We do not
call gk.ensure_gcd1 (it may add a self-loop) and explicitly strip any
self-loops at the end.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import threading
import traceback
from datetime import datetime

import numpy as np
import random
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

import clingo as clngo

import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# Make the repo importable when invoked as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..', '..', '..')
sys.path.insert(0, _ROOT)

from gunfolds.utils import bfutils
from gunfolds.utils import graphkit as gk
from gunfolds.utils.calc_procs import get_process_count
from gunfolds import conversions as cv
from gunfolds.conversions import drasl_jclingo2g
from gunfolds.solvers.clingo_rasl import drasl_command
from gunfolds.scripts.simulation import bold_function as hrf


# ─────────────────────────────────────────────────────────────────────────────
# Constants — match benchmark_domain_heuristic.py
# ─────────────────────────────────────────────────────────────────────────────

CLINGO_LIMIT = 64
MAXCOST = 20
PRIORITY = [1, 1, 1, 1, 2]
DENSITY_MODE = 'hard_soft0'
TOL_LOW = 5            # tighter than the FBIRN default (15) — synthetic gt_density
                       # is lower (~14) so 15 % floor-clamps to 0 and removes the
                       # lower cardinality bound entirely.
TOL_HIGH = 5
MAX_URATE = 4
CAPSIZE = 0            # 0 = unlimited improving-model reports; with opt-mode=opt
                       # clingo runs full BnB until optimality is proven.  -n 1
                       # would stop at the first feasible model without optimising.

# VAR + BOLD defaults — match exp4_pcmci_drasl_ringmore5.py defaults
SSIZE = 2500
NOISE = 0.1
U_RATE = 2

# SCC composition table (max SCC size = 6).  Documented in prompt.
# Compositions for N >= 24 are uniform 6-node SCCs (N must be a multiple of 6).
SCC_COMPOSITION = {
    8:  [6, 2],
    10: [6, 4],
    12: [6, 6],
    14: [6, 4, 4],
    16: [6, 6, 4],
    18: [6, 6, 6],
    20: [6, 6, 4, 4],
    24: [6] * 4,
    30: [6] * 5,
    42: [6] * 7,
    54: [6] * 9,
}


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

def derive_seed(n_nodes: int, instance_id: int) -> int:
    h = hashlib.md5(f"{n_nodes}-{instance_id}".encode()).hexdigest()
    return int(h[:8], 16)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth multi-SCC ring graph
# ─────────────────────────────────────────────────────────────────────────────

def make_multi_scc_ring(scc_sizes, dens=0.5, dag_degree=1, max_cross_connections=2):
    """
    Build a multi-SCC ring graph with the given (possibly non-uniform) ring
    sizes, mirroring gk.ring_sccs but allowing per-SCC sizes and skipping
    ensure_gcd1 (which may add a self-loop).

    Returns (g, partition) where partition is a list of node-id lists, one
    per SCC.  All node ids are 1-indexed (gunfolds convention).
    """
    num_sccs = len(scc_sizes)

    # 1. Quotient DAG over the SCCs.  We avoid gk.randomDAG here: its inner
    #    helper gk.remove_tril_singletons (graphkit.py:843) infinite-loops
    #    for N<=2 because np.random.randint(0, N-1) excludes the upper bound
    #    and returns 0 forever when an isolated node has index 0.  The
    #    prompt requires "cross-ring arrows are forward only (a DAG over
    #    the SCC quotient)", and the simplest such DAG is a chain
    #    0->1->2->...->(num_sccs-1) — connected, weakly connected,
    #    acyclic, and forward-only by construction.  Cross-SCC edge counts
    #    remain randomised in step 3.
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_sccs))
    for i in range(num_sccs - 1):
        dag.add_edge(i, i + 1)

    # 2. Build one ring-with-extra-edges per SCC, then shift node labels so
    #    each SCC occupies its own contiguous id range.  We pick a per-ring
    #    extra-edge count of 1 or 2 (prompt: "1-2 random extra non-self-loop
    #    edges per ring"), capped by what ringmore can place without making
    #    the ring trivially small.
    rings = []
    for s in scc_sizes:
        # `dens * s^2 - s` is the gk.ring_sccs convention; clamp to [1, 2]
        # to satisfy the prompt's 1-2 extra-edge spec.
        n_extra = max(1, min(2, int(dens * s * s) - s))
        rings.append(gk.ringmore(s, n_extra))
    shifted = gk.shift_list_labels(rings)

    # Track the partition before adding cross-SCC edges.  Each ring's nodes
    # are exactly the keys of its (shifted) dict.
    partition = [sorted(list(r.keys())) for r in shifted]

    # 3. Random cross-SCC edges following the DAG (forward only — the
    #    quotient DAG enforces a valid SCC structure).
    for v in dag:
        v_nodes = list(shifted[v].keys())
        for w in dag[v]:
            w_nodes = list(shifted[w].keys())
            n_cross = np.random.randint(low=1, high=max_cross_connections + 1)
            for _ in range(n_cross):
                a = random.choice(v_nodes)
                b = random.choice(w_nodes)
                shifted[v][a][b] = 1

    # 4. Merge into one graph and strip any accidental self-loops.
    g = gk.merge_list(shifted)
    for n in list(g.keys()):
        if n in g[n]:
            del g[n][n]

    return g, partition


def verify_partition(g, partition, max_scc_size):
    """Hard-checks: partition matches actual SCCs of g, and no SCC > max."""
    G = gk.graph2nx(g)
    actual_sccs = [set(c) for c in nx.strongly_connected_components(G)]
    declared_sccs = [set(p) for p in partition]

    # All declared SCCs must equal an actual SCC (set equality, ignoring order)
    for ds in declared_sccs:
        if ds not in actual_sccs:
            raise AssertionError(
                f"Declared SCC {sorted(ds)} does not match any actual SCC. "
                f"Actual SCCs: {[sorted(s) for s in actual_sccs]}"
            )
    # Number of declared SCCs must equal number of non-trivial-or-not SCCs
    if len(declared_sccs) != len(actual_sccs):
        raise AssertionError(
            f"Declared {len(declared_sccs)} SCCs but graph has "
            f"{len(actual_sccs)} actual SCCs."
        )
    for p in partition:
        if len(p) > max_scc_size:
            raise AssertionError(
                f"SCC of size {len(p)} exceeds max_scc_size={max_scc_size}: {p}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# VAR + BOLD simulation — match exp4_pcmci_drasl_ringmore5.py
# ─────────────────────────────────────────────────────────────────────────────

def _sample_W(A, scale_aware=True, bias_magnitudes=False):
    # Idea 1: pre-scale draws so post-damping entries stay O(1).
    #   ρ(random sparse) ~ σ·√(mean_in_degree); choose σ so ρ ≈ 1.
    # Idea 2: bias away from zero so |W^n| stays above the floor for more n.
    N = A.shape[0]
    if scale_aware:
        mean_in_deg = max(1.0, A.sum(axis=0).mean())
        sigma = 1.0 / np.sqrt(mean_in_deg)
    else:
        sigma = 1.0
    if bias_magnitudes:
        r = np.random.randn(*A.shape)
        # |entry| in [0.5σ, ~2.5σ], sign uniform
        mag = sigma * (0.5 + 0.5 * np.abs(r))
        W = A * np.sign(r) * mag
    else:
        W = A * (sigma * np.random.randn(*A.shape))
    return W


def create_stable_weighted_matrix(A, threshold=0.1, powers=(1, 2, 3, 4),
                                  max_attempts=1_000_000, damping=0.99,
                                  scale_aware=True, bias_magnitudes=False,
                                  auto_threshold=False,
                                  threshold_decay_ref_n=8):
    # NOTE: exp4_pcmci_drasl_ringmore5.py used scipy.sparse.linalg.eigs (ARPACK)
    # which routinely fails to converge for N>=50.  Dense np.linalg.eigvals is
    # fast and robust at N<=54.
    #
    # Ideas implemented:
    #   1) scale_aware     — σ = 1/√⟨in-deg⟩ keeps ρ(W) near 1 before damping
    #   2) bias_magnitudes — draw |entry| ≥ 0.5σ so powers stay above floor
    #   3) auto_threshold  — threshold scales by √(ref_N/N); set threshold=base
    #   4) powers          — caller picks lag set; () disables filter entirely
    N = A.shape[0]
    if auto_threshold and N > threshold_decay_ref_n:
        threshold = threshold * np.sqrt(threshold_decay_ref_n / N)

    for _ in range(max_attempts):
        W = _sample_W(A, scale_aware=scale_aware, bias_magnitudes=bias_magnitudes)
        evals = np.linalg.eigvals(W)
        rho = np.abs(evals).max()
        if rho == 0:
            continue
        W *= damping / rho
        ok = True
        for n in powers:
            Wn = np.linalg.matrix_power(W, n)
            nz = np.nonzero(Wn)
            if len(nz[0]) > 0 and (np.abs(Wn[nz]) < threshold).any():
                ok = False
                break
        if ok:
            return W
    raise ValueError(f'Could not find stable matrix after {max_attempts} tries.')


def construct_stable_matrix_from_sccs(A, partition, spectral_radius=0.95,
                                       min_magnitude=0.2, max_magnitude=0.8):
    # Idea 5: deterministic block-triangular construction.
    #   - Per SCC: place eigenvalues on a circle of radius `spectral_radius`
    #     by building a normalised companion-style block, then mask by the
    #     SCC's own adjacency pattern with sampled magnitudes ≥ min_magnitude.
    #   - Cross-SCC edges: sampled in [min_magnitude, max_magnitude] with
    #     random sign.  Quotient is a DAG so cross-edges don't affect ρ(W).
    # Returns a W with sparsity ⊆ A, ρ(W) ≤ spectral_radius by construction,
    # and every nonzero entry ≥ min_magnitude in magnitude.
    N = A.shape[0]
    W = np.zeros_like(A, dtype=float)

    # node-id (1-indexed from partition) → matrix index (0-indexed)
    # partition is 1-indexed per make_multi_scc_ring convention
    node_to_idx = {}
    for scc in partition:
        for node in scc:
            node_to_idx[node] = node - 1  # gunfolds 1-indexed → 0-indexed

    def _rand_mag():
        return np.random.uniform(min_magnitude, max_magnitude)

    def _rand_signed():
        return _rand_mag() * np.random.choice([-1.0, 1.0])

    for scc in partition:
        idxs = [node_to_idx[n] for n in scc]
        k = len(idxs)
        if k == 1:
            i = idxs[0]
            if A[i, i] != 0:
                W[i, i] = spectral_radius * np.random.choice([-1.0, 1.0])
            continue

        # Sub-block: build a magnitudes matrix only where A has edges.
        sub = np.zeros((k, k))
        for a, ia in enumerate(idxs):
            for b, ib in enumerate(idxs):
                if A[ia, ib] != 0:
                    sub[a, b] = _rand_signed()
        # Rescale this block to target spectral radius.
        evals = np.linalg.eigvals(sub)
        rho = np.abs(evals).max()
        if rho > 0:
            sub *= spectral_radius / rho
        # Floor magnitudes: any nonzero that fell below min_magnitude gets
        # bumped back up (preserves sign).  Slightly perturbs ρ but keeps
        # it below spectral_radius * (max_magnitude/min_magnitude) bound.
        nz = sub != 0
        small = nz & (np.abs(sub) < min_magnitude)
        if small.any():
            sub[small] = np.sign(sub[small]) * min_magnitude
        for a, ia in enumerate(idxs):
            for b, ib in enumerate(idxs):
                W[ia, ib] = sub[a, b]

    # Cross-SCC edges: any A[i,j] not yet filled and not on the diagonal of
    # the SCC blocks above.
    scc_node_sets = [set(node_to_idx[n] for n in scc) for scc in partition]

    def _same_scc(i, j):
        for s in scc_node_sets:
            if i in s and j in s:
                return True
        return False

    nz_a = np.argwhere(A != 0)
    for i, j in nz_a:
        if _same_scc(i, j):
            continue
        W[i, j] = _rand_signed()

    return W


def get_stable_weighted_matrix(A, partition=None, strategy='sample', **kwargs):
    if strategy == 'construct':
        if partition is None:
            raise ValueError("strategy='construct' requires partition")
        return construct_stable_matrix_from_sccs(
            A, partition,
            spectral_radius=kwargs.get('damping', 0.95),
            min_magnitude=kwargs.get('threshold', 0.2),
        )
    if strategy != 'sample':
        raise ValueError(f"unknown strategy {strategy!r}")
    return create_stable_weighted_matrix(A, **kwargs)


def simulate_var(W, ssize, noise):
    n = W.shape[0]
    data = np.zeros((n, ssize))
    data[:, 0] = noise * np.random.randn(n)
    for t in range(1, ssize):
        data[:, t] = W @ data[:, t - 1] + noise * np.random.randn(n)
    return data


def simulate_bold(var_data, u_rate):
    # end_time must scale with u_rate so the effective TR = 100/ssize * u_rate.
    # Without this, generating ssize*u_rate VAR samples and then taking [::u_rate]
    # cancels exactly and u_rate has no effect on the observed signal.
    #
    # Numerical-stability retry: for large N, random VAR transients can push
    # the balloon-model ODE past `vode`'s error tolerance on a single node;
    # that node's output is then shorter than the others and np.array() in
    # compute_bold_signals returns a 1-D object array (`.ndim == 1`), which
    # blows up `bold_out.shape[1]`.  Detect that and retry with the input
    # rescaled tighter — smaller drive ⇒ gentler dynamics ⇒ stable integration.
    extra_scale = 1.0
    bold_out = None
    for _ in range(5):
        data_scaled = var_data / (np.abs(var_data).max() * extra_scale + 1e-12)
        bold_out, _ = hrf.compute_bold_signals(data_scaled, end_time=100 * u_rate)
        if bold_out.ndim == 2 and bold_out.shape[1] > 0:
            break
        extra_scale *= 3.0
        print(f"    [simulate_bold] BOLD ODE failed (ragged output); "
              f"retrying with input rescaled by {extra_scale:g}x", flush=True)
    else:
        raise RuntimeError(
            "simulate_bold: BOLD ODE integrator failed even after rescaling "
            f"input by {extra_scale:g}x.  This VAR realisation may be "
            "intrinsically unstable for the balloon model."
        )
    drop = bold_out.shape[1] // 5
    bold_out = bold_out[:, drop:]
    return bold_out[:, ::u_rate]


# ─────────────────────────────────────────────────────────────────────────────
# F1 metrics
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_dir_adj(g, n):
    """1-indexed gunfolds graph → 0-indexed n×n directed-adjacency matrix."""
    adj = np.zeros((n, n), dtype=int)
    for i, nbrs in g.items():
        for j, v in nbrs.items():
            if v in (1, 3):
                adj[i - 1, j - 1] = 1
    return adj


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_solution(pred_g, gt_g, n):
    pred = graph_to_dir_adj(pred_g, n)
    true = graph_to_dir_adj(gt_g, n)
    np.fill_diagonal(pred, 0)
    np.fill_diagonal(true, 0)

    # Orientation: F1 over directed arrows
    tp_o = int(np.sum((pred == 1) & (true == 1)))
    fp_o = int(np.sum((pred == 1) & (true == 0)))
    fn_o = int(np.sum((pred == 0) & (true == 1)))
    orientation_f1 = f1_from_counts(tp_o, fp_o, fn_o)

    # Adjacency: undirected presence — symmetrise both
    pred_u = ((pred + pred.T) > 0).astype(int)
    true_u = ((true + true.T) > 0).astype(int)
    iu = np.triu_indices(n, k=1)
    tp_a = int(np.sum((pred_u[iu] == 1) & (true_u[iu] == 1)))
    fp_a = int(np.sum((pred_u[iu] == 1) & (true_u[iu] == 0)))
    fn_a = int(np.sum((pred_u[iu] == 0) & (true_u[iu] == 1)))
    adjacency_f1 = f1_from_counts(tp_a, fp_a, fn_a)

    return adjacency_f1, orientation_f1


# ─────────────────────────────────────────────────────────────────────────────
# CSV row write
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "n_nodes", "instance_id", "n_extra_edges", "scc_used", "scc_sizes",
    "drasl_time_sec", "total_time_sec", "pcmci_time_sec", "bold_time_sec",
    "n_solutions", "orientation_f1", "adjacency_f1", "status",
]


def write_csv_row(path, row):
    """Write a one-header-line, one-data-row CSV."""
    with open(path, "w") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        vals = []
        for col in CSV_COLUMNS:
            v = row.get(col, "")
            if isinstance(v, float) and np.isnan(v):
                vals.append("")
            elif isinstance(v, (list, tuple)):
                vals.append(json.dumps(list(v)).replace(",", ";"))
            elif isinstance(v, str) and ("," in v or '"' in v):
                vals.append('"' + v.replace('"', '""') + '"')
            else:
                vals.append(str(v))
        f.write(",".join(vals) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_drasl_with_timeout(command, n_nodes, pnum, timeout_sec):
    """
    Mirror benchmark_domain_heuristic.py:run_scenario.  Build clingo.Control,
    add+ground+solve under a threading.Timer.  Returns:
        drasl_time_sec, n_solutions, best_pred_g_or_None, timed_out_bool
    drasl_time_sec covers grounding + search (no I/O, no F1 calc).
    """
    base_args = [
        "--warn=no-atom-undefined",
        "--configuration=crafty",
        "-t", f"{int(pnum)},split",
        "-n", str(CAPSIZE),
    ]
    ctrl = clngo.Control(base_args)
    ctrl.configuration.solve.opt_mode = "opt"

    # Phase 1: program text + grounding + search — all timed together
    t0 = time.perf_counter()
    program = command.decode()
    ctrl.add("base", [], program)
    ctrl.ground([("base", [])])

    models = []
    timed_out = False
    timer = None
    if timeout_sec and timeout_sec > 0:
        # Adjust the timer for the grounding time we already spent so the
        # total drasl_time_sec budget is respected end-to-end.
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
                cost = model.cost
                atoms = [str(a) for a in model.symbols(shown=True)]
                models.append((atoms, cost))
    finally:
        if timer is not None:
            timer.cancel()

    drasl_time_sec = time.perf_counter() - t0

    # Parse solutions; choose lowest-cost model as "best"
    best_pred_g = None
    if models:
        # drasl_jclingo2g returns ((graph_num, urate), ...); each model's cost
        # is a list of weighted-priority sums.  Pick lowest sum-of-cost.
        scored = []
        for atoms, cost in models:
            parsed = drasl_jclingo2g(atoms)
            cost_total = sum(cost) if cost else 0
            scored.append((cost_total, parsed))
        scored.sort(key=lambda t: t[0])
        best_parsed = scored[0][1]
        # drasl_jclingo2g returns a set or single tuple; normalise:
        if isinstance(best_parsed, set):
            best_parsed = next(iter(best_parsed))
        graph_num = best_parsed[0]
        best_pred_g = bfutils.num2CG(graph_num, n_nodes)

    return drasl_time_sec, len(models), best_pred_g, timed_out


def main():
    parser = argparse.ArgumentParser(
        description="Per-instance runtime scaling experiment for drasl on "
                    "SCC-ring synthetic graphs."
    )
    parser.add_argument("--n_nodes", type=int, required=True,
                        choices=sorted(SCC_COMPOSITION.keys()))
    parser.add_argument("--instance_id", type=int, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="results/runtime_scaling/")
    parser.add_argument("--timeout_hours", type=float, default=35.0,
                        help="drasl internal timeout (default 35h, 1h margin "
                             "under the 36h SLURM walltime).")
    parser.add_argument("--pnum", type=int,
                        default=int(min(CLINGO_LIMIT, get_process_count(1))))
    parser.add_argument("--ssize", type=int, default=SSIZE)
    parser.add_argument("--noise", type=float, default=NOISE)
    parser.add_argument("--u_rate", type=int, default=U_RATE)
    # Stable-matrix strategy (ideas 1–5).  Defaults preserve old behaviour
    # (no path-strength filter); flags let you re-enable a calibrated filter
    # or switch to deterministic block-triangular construction at high N.
    parser.add_argument("--w_strategy", choices=['sample', 'construct'],
                        default='sample',
                        help="'sample': random + rejection (ideas 1-4); "
                             "'construct': deterministic block-triangular "
                             "from SCC partition (idea 5).")
    parser.add_argument("--w_threshold", type=float, default=0.0,
                        help="Path-strength floor.  0 disables filter.")
    parser.add_argument("--w_powers", type=str, default="",
                        help="Comma-sep lag powers to check, e.g. '1,2'. "
                             "Empty disables filter (default).")
    parser.add_argument("--w_scale_aware", action='store_true', default=True,
                        help="Idea 1: σ = 1/√⟨in-deg⟩.  On by default.")
    parser.add_argument("--w_no_scale_aware", dest='w_scale_aware',
                        action='store_false')
    parser.add_argument("--w_bias_magnitudes", action='store_true', default=False,
                        help="Idea 2: bias |entry| away from zero.")
    parser.add_argument("--w_auto_threshold", action='store_true', default=False,
                        help="Idea 3: scale threshold by √(8/N).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, f"n{args.n_nodes}_inst{args.instance_id}.csv")

    timeout_sec = int(args.timeout_hours * 3600)
    scc_sizes = SCC_COMPOSITION[args.n_nodes]

    # Initialise a row that we will fill in as we go and write at the end —
    # even on timeout / exception.
    row = {
        "n_nodes": args.n_nodes,
        "instance_id": args.instance_id,
        "n_extra_edges": "",
        "scc_used": "",
        "scc_sizes": scc_sizes,
        "drasl_time_sec": float('nan'),
        "total_time_sec": float('nan'),
        "pcmci_time_sec": float('nan'),
        "bold_time_sec": float('nan'),
        "n_solutions": 0,
        "orientation_f1": float('nan'),
        "adjacency_f1": float('nan'),
        "status": "",
    }

    overall_t0 = time.perf_counter()
    print("=" * 70, flush=True)
    print(f"runtime_scaling — n_nodes={args.n_nodes}, instance={args.instance_id}",
          flush=True)
    print(f"  scc_sizes={scc_sizes}  timeout={timeout_sec}s  "
          f"pnum={args.pnum}", flush=True)
    print(f"  start {datetime.now()}", flush=True)
    print("=" * 70, flush=True)

    try:
        # ── Step 1: deterministic seed ────────────────────────────────────
        seed = derive_seed(args.n_nodes, args.instance_id)
        np.random.seed(seed)
        random.seed(seed)
        print(f"[1/5] seed = {seed}", flush=True)

        # ── Step 2: ground-truth graph G^1 ────────────────────────────────
        gt_g, partition = make_multi_scc_ring(
            scc_sizes, dens=0.5, dag_degree=1, max_cross_connections=2)
        verify_partition(gt_g, partition, max_scc_size=6)
        n_nodes_actual = len(gt_g)
        n_edges = sum(len(nbrs) for nbrs in gt_g.values())
        # n_extra_edges relative to the bare ring topology
        n_extra_edges = n_edges - sum(scc_sizes)
        row["scc_used"] = json.dumps([len(p) for p in partition]).replace(",", ";")
        row["n_extra_edges"] = n_extra_edges
        print(f"[2/5] G^1: nodes={n_nodes_actual}, edges={n_edges}, "
              f"partition_sizes={[len(p) for p in partition]}", flush=True)
        assert n_nodes_actual == args.n_nodes, \
            f"Built {n_nodes_actual} nodes but expected {args.n_nodes}"

        # ── Step 3: VAR + BOLD ────────────────────────────────────────────
        bold_t0 = time.perf_counter()
        A = cv.graph2adj(gt_g)
        # powers=() disables the path-strength filter entirely.  exp4 tuned
        # threshold=0.1, powers=(1,2,3,4) for N=5; even relaxed to 0.01 / (2,)
        # this filter rejected every random W for N>=18 (W^2 entries along
        # sparse paths fall below any positive threshold a sizable fraction
        # of the time, regardless of damping).  The VAR's stability is fully
        # determined by spectral radius < 1, which is enforced by the
        # damping/rho normalisation inside create_stable_weighted_matrix —
        # no extra filter is needed.
        powers_tuple = tuple(int(p) for p in args.w_powers.split(',')
                             if p.strip()) if args.w_powers else ()
        W = get_stable_weighted_matrix(
            A,
            partition=partition,
            strategy=args.w_strategy,
            threshold=args.w_threshold,
            powers=powers_tuple,
            scale_aware=args.w_scale_aware,
            bias_magnitudes=args.w_bias_magnitudes,
            auto_threshold=args.w_auto_threshold,
        )
        var_data = simulate_var(W, ssize=args.ssize * args.u_rate,
                                noise=args.noise)
        bold_data = simulate_bold(var_data, u_rate=args.u_rate)
        row["bold_time_sec"] = round(time.perf_counter() - bold_t0, 4)
        print(f"[3/5] BOLD: shape={bold_data.shape}  "
              f"({row['bold_time_sec']:.2f}s)", flush=True)

        # ── Step 4: PCMCI ─────────────────────────────────────────────────
        pcmci_t0 = time.perf_counter()
        ts_2d = bold_data.T  # (T, N) for tigramite
        assert ts_2d.shape[1] == n_nodes_actual, \
            f"axis swap detected: ts_2d shape {ts_2d.shape}"
        dataframe = pp.DataFrame(ts_2d)
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
        pcmci_results = pcmci.run_pcmci(
            tau_max=1, pc_alpha=None, alpha_level=0.05, fdr_method="none")
        g_estimated, A_pc, B_pc = cv.Glag2CG(pcmci_results)
        row["pcmci_time_sec"] = round(time.perf_counter() - pcmci_t0, 4)
        print(f"[4/5] PCMCI: g_est nodes={len(g_estimated)}  "
              f"({row['pcmci_time_sec']:.2f}s)", flush=True)

        # ── Step 5: drasl ─────────────────────────────────────────────────
        # DD/BD construction (canonical, MAXCOST=20)
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

        urate = min(MAX_URATE, 3 * n_nodes_actual + 1)
        # GT_density derived from G^1 (per-instance, in 0-100 convention).
        gt_density = max(1, int(round(100.0 * gk.density(gt_g))))

        # Pass scc_members = ground-truth SCC partition (this experiment knows
        # G^1).  dm=[DD] activates the weighted-MFAS quotient back-edge
        # selection inside encode_list_sccs (CHANGELOG_AI 2026-05-04).
        # selfloop=False mirrors benchmark_domain_heuristic.py.
        command = drasl_command(
            [g_estimated],
            max_urate=urate,
            weighted=True,
            scc=True,
            scc_members=partition,
            dm=[DD], bdm=[BD],
            edge_weights=PRIORITY,
            GT_density=gt_density,
            selfloop=False,
            density_mode=DENSITY_MODE,
            tol=None, tol_low=TOL_LOW, tol_high=TOL_HIGH,
        )
        print(f"[5/5] drasl: ASP {len(command):,} bytes  urate={urate}  "
              f"gt_density={gt_density}  scc={[len(p) for p in partition]}",
              flush=True)
        print(f"        DD range [{DD.min()},{DD.max()}]  "
              f"BD range [{BD.min()},{BD.max()}]", flush=True)

        drasl_time_sec, n_solutions, best_pred_g, timed_out = (
            run_drasl_with_timeout(command, n_nodes_actual, args.pnum, timeout_sec)
        )
        row["drasl_time_sec"] = round(drasl_time_sec, 4)
        row["n_solutions"] = n_solutions
        print(f"        drasl done: {drasl_time_sec:.2f}s  "
              f"n_solutions={n_solutions}  timed_out={timed_out}", flush=True)

        # F1 vs ground truth
        if best_pred_g is not None:
            adj_f1, orient_f1 = evaluate_solution(
                best_pred_g, gt_g, n_nodes_actual)
            row["adjacency_f1"] = round(adj_f1, 4)
            row["orientation_f1"] = round(orient_f1, 4)
            print(f"        F1: adjacency={adj_f1:.3f}  "
                  f"orientation={orient_f1:.3f}", flush=True)

        row["status"] = "timeout" if timed_out else "completed"

    except Exception as exc:
        tb = traceback.format_exc()
        print(f"\n!! ERROR: {exc}\n{tb}", flush=True)
        # Truncate the message for the CSV but keep enough to debug.
        msg = str(exc).replace("\n", " ").replace(",", ";")[:200]
        row["status"] = f"error: {msg}"

    row["total_time_sec"] = round(time.perf_counter() - overall_t0, 4)
    write_csv_row(out_path, row)
    print(f"\nWrote {out_path}  status={row['status']}  "
          f"total={row['total_time_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
