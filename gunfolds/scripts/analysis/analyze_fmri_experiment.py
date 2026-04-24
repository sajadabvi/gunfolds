"""
Cross-configuration analysis for the fMRI RASL / PCMCI / GCM experiment.

Aggregates per-subject results produced by fmri_experiment_large.py (SLURM
or local), computes HC-vs-SZ separability metrics for every configuration,
and generates comparison plots + a summary table.

Usage:
    python analyze_fmri_experiment.py --timestamp 03012026120000
    python analyze_fmri_experiment.py --timestamp 03012026120000 --plot
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from collections import defaultdict

from gunfolds.utils import zickle as zkl
from gunfolds import conversions as cv

_REAL_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "real_data")
if _REAL_DATA_DIR not in sys.path:
    sys.path.insert(0, _REAL_DATA_DIR)
from gunfolds.scripts.real_data.component_config import (
    INDEX_TO_DOMAIN, get_comp_indices,
)


# ---------------------------------------------------------------------------
# Domain boundaries and parameter formatting helpers
# ---------------------------------------------------------------------------

def get_domain_boundaries(comp_indices):
    """
    Compute network domain group boundaries for heatmap dividers.

    Given an ordered list of 0-based component indices, identifies where the
    NeuroMark functional domain changes (SC, AU, SM, VI, CC, DM, CB).

    Returns
    -------
    boundaries : list[int]
        Positions where the domain changes (used for divider lines).
    domain_labels : list[str]
        Domain name for each contiguous group.
    domain_centers : list[float]
        Center index for each domain group (for label placement).
    """
    if not comp_indices:
        return [], [], []

    domains = [INDEX_TO_DOMAIN.get(idx, "?") for idx in comp_indices]
    boundaries = []
    group_start = 0
    groups = []

    for i in range(1, len(domains)):
        if domains[i] != domains[i - 1]:
            boundaries.append(i)
            groups.append((group_start, i, domains[i - 1]))
            group_start = i
    groups.append((group_start, len(domains), domains[-1]))

    domain_labels = [g[2] for g in groups]
    domain_centers = [(g[0] + g[1] - 1) / 2.0 for g in groups]
    return boundaries, domain_labels, domain_centers


def _build_params_parts(analysis_args, run_params=None):
    """Build a list of parameter key=value strings for plot annotation."""
    parts = []
    if run_params:
        gt_mode = run_params.get("gt_density_mode")
        if gt_mode is not None:
            if gt_mode == "fixed":
                val = run_params.get("gt_density", "?")
                parts.append(f"GT Density: fixed ({val}/1000)")
            elif gt_mode == "fraction":
                frac = run_params.get("gt_density_fraction", "?")
                parts.append(f"GT Density: fraction ({frac})")
            else:
                parts.append(f"GT Density: {gt_mode}")
        sel = run_params.get("selection_mode")
        if sel:
            k = run_params.get("top_k", "?")
            parts.append(f"Selection: {sel} (k={k})")
    if analysis_args is not None:
        corr = ("Bonferroni" if analysis_args.correction == "bonferroni"
                else "FDR (BH)")
        parts.append(f"Correction: {corr}")
        parts.append(f"\u03b1 = {analysis_args.alpha}")
    return parts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze fMRI experiment results across configurations."
    )
    p.add_argument("--timestamp", required=True,
                   help="Shared timestamp of the experiment batch")
    p.add_argument("--results_root", default="fbirn_results",
                   help="Root directory containing timestamped results")
    p.add_argument("--data_path", default="../fbirn/fbirn_sz_data.npz",
                   help="Path to fbirn_sz_data.npz (for label info)")
    p.add_argument("--plot", action="store_true",
                   help="Generate comparison plots")
    p.add_argument("--alpha", default=0.05, type=float,
                   help="Significance level for edge-level tests")
    p.add_argument("--correction", default="bonferroni",
                   choices=["bonferroni", "fdr"],
                   help="Multiple-comparison correction: bonferroni (FWER) or fdr (Benjamini-Hochberg)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def discover_configs(root_dir):
    """
    Find all configuration directories under root_dir.
    Each config dir looks like N10_domain_RASL/ and contains subject_*/result.zkl.
    """
    configs = []
    if not os.path.isdir(root_dir):
        return configs
    for name in sorted(os.listdir(root_dir)):
        config_path = os.path.join(root_dir, name)
        if not os.path.isdir(config_path):
            continue
        # Expect pattern N<num>_<scc>_<method>
        parts = name.split("_")
        if len(parts) < 3 or not parts[0].startswith("N"):
            continue
        configs.append(name)
    return configs


def load_subject_results(config_dir):
    """
    Load all subject result.zkl files from a config directory.

    Returns list of dicts (one per subject).
    """
    results = []
    pattern = os.path.join(config_dir, "subject_*", "result.zkl")
    files = sorted(glob.glob(pattern))
    for f in files:
        try:
            info = zkl.load(f)
            results.append(info)
        except Exception as e:
            print(f"  Warning: failed to load {f}: {e}")
    return results


def cg_to_adj_binary(cg):
    A = cv.graph2adj(cg)
    A = (A > 0).astype(int)
    np.fill_diagonal(A, 0)
    return A


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_group_edge_matrices(subject_results):
    """
    For each subject, aggregate all solution adjacency matrices into
    per-group edge frequency matrices.

    Returns
    -------
    group_counts : dict[int, ndarray]  (group -> NxN int matrix)
    group_n_subjects : dict[int, int]
    group_n_solutions : dict[int, int]
    n_nodes : int
    """
    if not subject_results:
        return {}, {}, {}, 0

    first = subject_results[0]
    # Determine N from the first solution's graph
    first_sol = first["solutions"][0]
    if "adj" in first_sol:
        n_nodes = len(first_sol["adj"])
    else:
        cg = first_sol["graph"]
        n_nodes = len(cg)

    group_counts = defaultdict(lambda: np.zeros((n_nodes, n_nodes), dtype=int))
    group_n_subjects = defaultdict(int)
    group_n_solutions = defaultdict(int)

    for info in subject_results:
        grp = info["group"]
        group_n_subjects[grp] += 1
        for sol in info["solutions"]:
            if "adj" in sol:
                adj = np.array(sol["adj"], dtype=int)
            else:
                adj = cg_to_adj_binary(sol["graph"])
            group_counts[grp] += adj
            group_n_solutions[grp] += 1

    return dict(group_counts), dict(group_n_subjects), dict(group_n_solutions), n_nodes


def _benjamini_hochberg(pvals_flat, alpha):
    """Return a boolean mask of the same length indicating BH-FDR significance."""
    m = len(pvals_flat)
    if m == 0:
        return np.array([], dtype=bool)
    order = np.argsort(pvals_flat)
    sorted_p = pvals_flat[order]
    thresholds = alpha * np.arange(1, m + 1) / m
    # largest rank where p <= threshold
    below = np.where(sorted_p <= thresholds)[0]
    if len(below) == 0:
        return np.zeros(m, dtype=bool)
    max_rank = below[-1]
    sig = np.zeros(m, dtype=bool)
    sig[order[:max_rank + 1]] = True
    return sig


def edge_level_tests(group_counts, group_n_solutions, alpha=0.05,
                     correction="bonferroni"):
    """
    For each directed edge (i->j), test whether its frequency differs
    between groups using a chi-squared or Fisher exact test.

    Parameters
    ----------
    correction : str
        "bonferroni" (FWER) or "fdr" (Benjamini-Hochberg FDR).

    Returns
    -------
    n_sig : int          (number of significant edges after correction)
    pvalues : ndarray    (NxN matrix of raw p-values, NaN where untestable)
    sig_mask : ndarray   (NxN boolean matrix, True where significant)
    """
    groups = sorted(group_counts.keys())
    if len(groups) < 2:
        return 0, None, None

    g0, g1 = groups[0], groups[1]
    c0 = group_counts[g0]
    c1 = group_counts[g1]
    n0 = group_n_solutions[g0]
    n1 = group_n_solutions[g1]
    N = c0.shape[0]

    pvalues = np.full((N, N), np.nan)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            a = int(c0[i, j])
            b = int(c1[i, j])
            c = n0 - a
            d = n1 - b
            table = np.array([[a, b], [c, d]])
            if table.sum() == 0:
                continue
            try:
                if table.min() < 5:
                    _, p = fisher_exact(table)
                else:
                    _, p, _, _ = chi2_contingency(table, correction=True)
                pvalues[i, j] = p
            except Exception:
                pass

    sig_mask = np.zeros((N, N), dtype=bool)
    valid = ~np.isnan(pvalues) & (np.eye(N) == 0)
    valid_p = pvalues[valid]

    if correction == "fdr":
        sig_flat = _benjamini_hochberg(valid_p, alpha)
    else:
        threshold = alpha / max(valid_p.size, 1)
        sig_flat = valid_p < threshold

    sig_mask[valid] = sig_flat
    n_sig = int(sig_mask.sum())

    return n_sig, pvalues, sig_mask


def frobenius_distance(group_counts, group_n_solutions):
    """
    Frobenius norm of the difference in edge-frequency matrices between
    two groups, where frequency = count / n_solutions.
    """
    groups = sorted(group_counts.keys())
    if len(groups) < 2:
        return 0.0
    g0, g1 = groups[0], groups[1]
    freq0 = group_counts[g0].astype(float) / max(group_n_solutions[g0], 1)
    freq1 = group_counts[g1].astype(float) / max(group_n_solutions[g1], 1)
    return float(np.linalg.norm(freq0 - freq1, "fro"))


def density_difference(group_counts, group_n_solutions):
    """Mean graph density difference between groups."""
    groups = sorted(group_counts.keys())
    if len(groups) < 2:
        return 0.0
    densities = {}
    for g in groups:
        freq = group_counts[g].astype(float) / max(group_n_solutions[g], 1)
        N = freq.shape[0]
        np.fill_diagonal(freq, 0)
        densities[g] = freq.sum() / (N * (N - 1)) if N > 1 else 0.0
    g0, g1 = groups[0], groups[1]
    return abs(densities[g0] - densities[g1])


def undersampling_rate_summary(subject_results):
    """
    For RASL configs, summarize the distribution of undersampling rates
    across solutions and groups.
    """
    rates_by_group = defaultdict(list)
    for info in subject_results:
        grp = info["group"]
        for sol in info["solutions"]:
            usamp = sol.get("undersampling")
            if usamp is not None:
                if isinstance(usamp, (tuple, list)):
                    rates_by_group[grp].append(usamp[0] if usamp else 1)
                else:
                    rates_by_group[grp].append(int(usamp))
    return dict(rates_by_group)


def undersampling_group_test(rates_by_group):
    """
    Mann-Whitney U test: do HC and SZ have different undersampling rates?
    Returns (U statistic, p-value) or (None, None) if not applicable.
    """
    groups = sorted(rates_by_group.keys())
    if len(groups) < 2:
        return None, None
    r0 = rates_by_group[groups[0]]
    r1 = rates_by_group[groups[1]]
    if len(r0) < 2 or len(r1) < 2:
        return None, None
    try:
        stat, p = mannwhitneyu(r0, r1, alternative="two-sided")
        return float(stat), float(p)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Analysis per configuration
# ---------------------------------------------------------------------------

def analyze_config(config_name, config_dir, alpha=0.05, correction="bonferroni"):
    """
    Analyze one configuration. Returns a dict of metrics.
    """
    subject_results = load_subject_results(config_dir)
    if not subject_results:
        print(f"  {config_name}: no subject results found, skipping.")
        return None

    n_loaded = len(subject_results)
    group_counts, group_n_subj, group_n_sol, n_nodes = \
        compute_group_edge_matrices(subject_results)

    n_sig, pvalues, sig_mask = edge_level_tests(
        group_counts, group_n_sol, alpha=alpha, correction=correction,
    )
    frob = frobenius_distance(group_counts, group_n_sol)
    dens_diff = density_difference(group_counts, group_n_sol)

    # Parse config tag
    parts = config_name.split("_")
    n_comp = int(parts[0][1:])
    scc_strategy = parts[1]
    method = parts[2]

    result = {
        "config": config_name,
        "method": method,
        "n_components": n_comp,
        "scc_strategy": scc_strategy,
        "n_subjects_loaded": n_loaded,
        "n_nodes": n_nodes,
        "n_sig_edges": n_sig,
        "frobenius_dist": frob,
        "density_diff": dens_diff,
        "group_n_subjects": dict(group_n_subj),
        "group_n_solutions": dict(group_n_sol),
    }

    # Undersampling analysis (RASL only)
    if method == "RASL":
        rates = undersampling_rate_summary(subject_results)
        u_stat, u_pval = undersampling_group_test(rates)
        result["usamp_U_stat"] = u_stat
        result["usamp_p_value"] = u_pval
        if rates:
            for g, r in sorted(rates.items()):
                result[f"usamp_mean_grp{g}"] = float(np.mean(r))
                result[f"usamp_median_grp{g}"] = float(np.median(r))
    else:
        result["usamp_U_stat"] = None
        result["usamp_p_value"] = None

    # Store raw data for plotting
    result["_group_counts"] = group_counts
    result["_group_n_sol"] = group_n_sol
    result["_pvalues"] = pvalues
    result["_sig_mask"] = sig_mask

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison_bar(summary_df, metric, ylabel, title, outpath,
                        analysis_args=None, run_params=None):
    """Bar chart comparing a metric across all configurations."""
    fig, ax = plt.subplots(figsize=(max(10, len(summary_df) * 0.8), 6))
    colors = []
    for _, row in summary_df.iterrows():
        if row["method"] == "RASL":
            colors.append("#2196F3")
        elif row["method"] == "PCMCI":
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")

    x = range(len(summary_df))
    bars = ax.bar(x, summary_df[metric], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["config"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="RASL"),
        Patch(facecolor="#FF9800", label="PCMCI"),
        Patch(facecolor="#4CAF50", label="GCM"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    params_parts = _build_params_parts(analysis_args, run_params)
    if params_parts:
        fig.suptitle("  |  ".join(params_parts), fontsize=8, style="italic",
                     y=0.99,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                               edgecolor="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_edge_diff_heatmap(result, comp_names, outpath, comp_indices=None,
                           analysis_args=None, run_params=None):
    """
    Heatmap of (HC_freq - SZ_freq) for one configuration, with NeuroMark
    network domain dividers (SC, AU, SM, VI, CC, DM, CB) and parameter
    annotation.
    """
    group_counts = result["_group_counts"]
    group_n_sol = result["_group_n_sol"]
    groups = sorted(group_counts.keys())
    if len(groups) < 2:
        return

    freq0 = group_counts[groups[0]].astype(float) / max(group_n_sol[groups[0]], 1)
    freq1 = group_counts[groups[1]].astype(float) / max(group_n_sol[groups[1]], 1)
    diff = freq0 - freq1

    N = diff.shape[0]
    if comp_names is None or len(comp_names) != N:
        comp_names = [str(i) for i in range(N)]

    fig, ax = plt.subplots(figsize=(max(10, N * 0.45), max(9, N * 0.4)))
    vmax = max(abs(diff.min()), abs(diff.max()), 0.01)
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    tick_size = max(4, 8 - N // 10)
    ax.set_xticklabels(comp_names, rotation=90, fontsize=tick_size)
    ax.set_yticklabels(comp_names, fontsize=tick_size)
    ax.set_xlabel("Target", labelpad=25)
    ax.set_ylabel("Source", labelpad=35)
    title_main = (f"{result['config']}: Group {groups[0]} \u2212 Group {groups[1]} "
                  "edge freq")
    ax.set_title(title_main, pad=12)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Mark significant edges
    sig_mask = result.get("_sig_mask")
    if sig_mask is not None:
        for i in range(N):
            for j in range(N):
                if sig_mask[i, j]:
                    ax.plot(j, i, "k*", markersize=max(3, 8 - N // 10))

    # --- Network domain dividers ---
    if comp_indices is not None and len(comp_indices) == N:
        boundaries, domain_labels, domain_centers = get_domain_boundaries(
            comp_indices)
        if boundaries:
            for b in boundaries:
                ax.axhline(y=b - 0.5, color="black", linewidth=2, zorder=5)
                ax.axvline(x=b - 0.5, color="black", linewidth=2, zorder=5)

            for edge in (-0.5, N - 0.5):
                ax.axhline(y=edge, color="black", linewidth=2, zorder=5)
                ax.axvline(x=edge, color="black", linewidth=2, zorder=5)

            dom_fs = max(7, min(11, 12 - N // 15))
            max_label_chars = max(len(str(c)) for c in comp_names)
            y_offset = -(tick_size * max_label_chars * 0.6 + 12)
            x_offset = -(tick_size * max_label_chars * 0.7 + 12)
            yax_trans = ax.get_yaxis_transform()
            xax_trans = ax.get_xaxis_transform()

            for label, center in zip(domain_labels, domain_centers):
                ax.annotate(label, xy=(0, center), xycoords=yax_trans,
                            xytext=(y_offset, 0), textcoords="offset points",
                            fontsize=dom_fs, fontweight="bold",
                            ha="center", va="center", clip_on=False)
                ax.annotate(label, xy=(center, 0), xycoords=xax_trans,
                            xytext=(0, x_offset), textcoords="offset points",
                            fontsize=dom_fs, fontweight="bold",
                            ha="center", va="top", clip_on=False)

    # --- Parameter annotation ---
    params_parts = _build_params_parts(analysis_args, run_params)
    if params_parts:
        fig.suptitle("  |  ".join(params_parts), fontsize=8, style="italic",
                     y=0.99,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                               edgecolor="gray", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root_dir = os.path.join(args.results_root, args.timestamp)

    if not os.path.isdir(root_dir):
        print(f"Error: results directory not found: {root_dir}")
        sys.exit(1)

    print("=" * 80)
    print("FMRI EXPERIMENT - CROSS-CONFIGURATION ANALYSIS")
    print("=" * 80)
    correction_label = "Bonferroni (FWER)" if args.correction == "bonferroni" else "Benjamini-Hochberg (FDR)"
    print(f"Timestamp:    {args.timestamp}")
    print(f"Results root: {root_dir}")
    print(f"Correction:   {correction_label}")
    print()

    configs = discover_configs(root_dir)
    if not configs:
        print("No configuration directories found.")
        sys.exit(1)

    print(f"Found {len(configs)} configurations: {configs}\n")

    comp_names_map = {}
    comp_indices_map = {}
    run_params_map = {}

    all_results = []
    for cfg in configs:
        cfg_dir = os.path.join(root_dir, cfg)
        print(f"Analyzing {cfg}...")
        result = analyze_config(cfg, cfg_dir, alpha=args.alpha,
                                correction=args.correction)
        if result is not None:
            all_results.append(result)
            if cfg not in comp_names_map:
                subj_files = sorted(glob.glob(
                    os.path.join(cfg_dir, "subject_*", "result.zkl")
                ))
                if subj_files:
                    try:
                        info = zkl.load(subj_files[0])
                        if "comp_names" in info:
                            comp_names_map[cfg] = info["comp_names"]
                        if "comp_indices" in info:
                            comp_indices_map[cfg] = info["comp_indices"]
                        rp = {}
                        for key in ("gt_density_mode", "gt_density",
                                    "gt_density_fraction", "selection_mode",
                                    "top_k"):
                            if key in info:
                                rp[key] = info[key]
                        if rp:
                            run_params_map[cfg] = rp
                    except Exception:
                        pass
            if cfg not in comp_indices_map:
                try:
                    n_comp = int(cfg.split("_")[0][1:])
                    comp_indices_map[cfg] = get_comp_indices(n_comp)
                except Exception:
                    pass

    if not all_results:
        print("No results to analyze.")
        sys.exit(1)

    # Build summary table
    summary_rows = []
    for r in all_results:
        row = {
            "config": r["config"],
            "method": r["method"],
            "n_components": r["n_components"],
            "scc_strategy": r["scc_strategy"],
            "n_subjects": r["n_subjects_loaded"],
            "n_sig_edges": r["n_sig_edges"],
            "frobenius_dist": round(r["frobenius_dist"], 4),
            "density_diff": round(r["density_diff"], 4),
        }
        if r["usamp_p_value"] is not None:
            row["usamp_p_value"] = round(r["usamp_p_value"], 6)
        else:
            row["usamp_p_value"] = ""
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Save summary CSV (per-correction subfolder so FDR and Bonferroni don't overwrite)
    out_dir = os.path.join(root_dir, "analysis", args.correction)
    os.makedirs(out_dir, exist_ok=True)
    summary_csv = os.path.join(out_dir, "config_comparison.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")

    # Ranking
    print("\n" + "=" * 80)
    print("RANKING BY GROUP SEPARABILITY")
    print("=" * 80)
    ranked = summary_df.sort_values("n_sig_edges", ascending=False)
    print("\nBy significant edges (more = better separation):")
    for i, (_, row) in enumerate(ranked.iterrows(), 1):
        print(f"  {i}. {row['config']:30s}  sig_edges={row['n_sig_edges']}"
              f"  frob={row['frobenius_dist']:.4f}")

    print("\nBy Frobenius distance (larger = more different group graphs):")
    ranked_frob = summary_df.sort_values("frobenius_dist", ascending=False)
    for i, (_, row) in enumerate(ranked_frob.iterrows(), 1):
        print(f"  {i}. {row['config']:30s}  frob={row['frobenius_dist']:.4f}"
              f"  sig_edges={row['n_sig_edges']}")

    # Key comparison: for each N, RASL vs baselines
    print("\n" + "=" * 80)
    print("RASL vs BASELINES (per N)")
    print("=" * 80)
    for n_comp in sorted(summary_df["n_components"].unique()):
        subset = summary_df[summary_df["n_components"] == n_comp]
        print(f"\n  N = {n_comp}:")
        for _, row in subset.iterrows():
            tag = f"    {row['method']:6s} ({row['scc_strategy']:12s})"
            print(f"{tag}  sig_edges={row['n_sig_edges']}"
                  f"  frob={row['frobenius_dist']:.4f}"
                  f"  density_diff={row['density_diff']:.4f}")

    # Plots
    if args.plot:
        print("\n" + "=" * 80)
        print("GENERATING PLOTS")
        print("=" * 80)

        common_run_params = None
        if run_params_map:
            common_run_params = dict(next(iter(run_params_map.values())))

        plot_comparison_bar(
            summary_df, "n_sig_edges",
            "Number of Significant Edges",
            "HC vs SZ: Significant Edge Differences by Configuration",
            os.path.join(out_dir, "bar_sig_edges.png"),
            analysis_args=args, run_params=common_run_params,
        )
        print(f"  Saved: {out_dir}/bar_sig_edges.png")

        plot_comparison_bar(
            summary_df, "frobenius_dist",
            "Frobenius Distance",
            "HC vs SZ: Edge Frequency Frobenius Distance by Configuration",
            os.path.join(out_dir, "bar_frobenius.png"),
            analysis_args=args, run_params=common_run_params,
        )
        print(f"  Saved: {out_dir}/bar_frobenius.png")

        plot_comparison_bar(
            summary_df, "density_diff",
            "Density Difference",
            "HC vs SZ: Graph Density Difference by Configuration",
            os.path.join(out_dir, "bar_density_diff.png"),
            analysis_args=args, run_params=common_run_params,
        )
        print(f"  Saved: {out_dir}/bar_density_diff.png")

        # Per-config heatmaps
        heatmap_dir = os.path.join(out_dir, "heatmaps")
        for r in all_results:
            cfg_name = r["config"]
            cnames = comp_names_map.get(cfg_name)
            cindices = comp_indices_map.get(cfg_name)
            rparams = run_params_map.get(cfg_name)
            plot_edge_diff_heatmap(
                r, cnames,
                os.path.join(heatmap_dir, f"diff_{cfg_name}.png"),
                comp_indices=cindices,
                analysis_args=args,
                run_params=rparams,
            )
        print(f"  Saved heatmaps: {heatmap_dir}/")

    # Save detailed results for downstream use
    detailed = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if not k.startswith("_")}
        detailed.append(entry)
    zkl.save(detailed, os.path.join(out_dir, "detailed_results.zkl"))
    print(f"\nSaved detailed results: {out_dir}/detailed_results.zkl")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
