"""
Unified comparison: run all classification tiers and produce a combined
results table + plots for the paper.

Collects results from:
  - Tier 1: Classical ML (classify_hc_sz.py)
  - Tier 2: BrainNet Transformer (brain_transformer_classify.py)
  - Tier 3: Solution-Set Transformer (solution_set_transformer.py)
  - Time-Series Foundation Model (timeseries_foundation_classify.py)

Can either run each tier as a subprocess or aggregate pre-computed CSV
results.

Usage:
    # Aggregate existing results (fast)
    python run_all_classifiers.py --timestamp 03232026175230 --aggregate-only

    # Run everything end-to-end
    python run_all_classifiers.py --timestamp 03232026175230 --run-all
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified classifier comparison for HC vs SZ.")
    p.add_argument("--timestamp", required=True)
    p.add_argument("--results_root", default="fbirn_results")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--data_path", default=None,
                   help="Path to fbirn_sz_data.npz (for time-series model)")
    p.add_argument("--run-all", action="store_true",
                   help="Run all tier scripts before aggregating")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Only aggregate existing CSV results")
    p.add_argument("--n_components_ts", type=int, nargs="+", default=[10, 20],
                   help="Component sizes for time-series model")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Running tier scripts
# ---------------------------------------------------------------------------

def run_tier(script_name, timestamp, extra_args=None):
    """Run a tier script as a subprocess."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    if not os.path.isfile(script_path):
        print(f"  Warning: {script_path} not found, skipping")
        return False

    cmd = [sys.executable, script_path, "--timestamp", timestamp]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  Warning: {script_name} exited with code {result.returncode}")
        return False
    return True


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def load_tier_results(ml_dir):
    """Load all tier CSV results from the ml_classification directory."""
    files = {
        "tier1_results.csv": "Tier1_ClassicalML",
        "tier2_results.csv": "Tier2_GraphTransformer",
        "tier3_results.csv": "Tier3_SolutionSetTransformer",
        "timeseries_foundation_results.csv": "TimeSeries_Foundation",
    }

    frames = []
    for fname, tier_label in files.items():
        path = os.path.join(ml_dir, fname)
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df["tier"] = tier_label
            frames.append(df)
            print(f"  Loaded {fname}: {len(df)} rows")
        else:
            print(f"  Not found: {fname}")

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def build_comparison_table(df):
    """Build the main comparison: best classifier per (config, tier)."""
    key_cols = ["config", "tier", "method", "n_components", "scc_strategy"]
    available = [c for c in key_cols if c in df.columns]

    if "accuracy_mean" not in df.columns:
        return df

    grouped = df.loc[df.groupby(["config", "tier"])["accuracy_mean"].idxmax()]
    show_cols = available + [
        "classifier", "accuracy_mean", "accuracy_std",
        "auc_mean", "auc_std",
    ]
    show_cols = [c for c in show_cols if c in grouped.columns]
    return grouped[show_cols].sort_values(
        ["n_components", "config", "tier"]).reset_index(drop=True)


def build_method_comparison(df):
    """
    For each n_components, compare methods (RASL vs PCMCI vs GCM vs
    TimeSeries) using the best classifier for each.
    """
    if "method" not in df.columns or "accuracy_mean" not in df.columns:
        return None

    rows = []
    for n_comp in sorted(df["n_components"].unique()):
        sub = df[df["n_components"] == n_comp]
        for method in sub["method"].unique():
            msub = sub[sub["method"] == method]
            best_idx = msub["accuracy_mean"].idxmax()
            best = msub.loc[best_idx]
            rows.append({
                "n_components": n_comp,
                "method": method,
                "best_classifier": best.get("classifier", ""),
                "tier": best.get("tier", ""),
                "config": best.get("config", ""),
                "accuracy": best["accuracy_mean"],
                "accuracy_std": best.get("accuracy_std", 0),
                "auc": best.get("auc_mean", 0),
                "auc_std": best.get("auc_std", 0),
            })
    return pd.DataFrame(rows).sort_values(
        ["n_components", "accuracy"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tier_comparison(df, out_dir):
    """Bar chart: accuracy by tier for each config."""
    if df is None or df.empty:
        return

    configs = df["config"].unique()
    tiers = df["tier"].unique()
    tier_colors = {
        "Tier1_ClassicalML": "#2196F3",
        "Tier2_GraphTransformer": "#FF9800",
        "Tier3_SolutionSetTransformer": "#4CAF50",
        "TimeSeries_Foundation": "#9C27B0",
    }

    fig, ax = plt.subplots(figsize=(max(12, len(configs) * 2), 6))
    x = np.arange(len(configs))
    width = 0.8 / max(len(tiers), 1)

    for i, tier in enumerate(tiers):
        tier_data = df[df["tier"] == tier]
        accs = []
        stds = []
        for cfg in configs:
            row = tier_data[tier_data["config"] == cfg]
            if len(row) > 0:
                accs.append(row["accuracy_mean"].values[0])
                stds.append(row.get("accuracy_std", pd.Series([0])).values[0])
            else:
                accs.append(0)
                stds.append(0)
        color = tier_colors.get(tier, "#888888")
        offset = (i - len(tiers) / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, yerr=stds, label=tier,
               color=color, edgecolor="black", linewidth=0.5, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("HC vs SZ Classification: All Tiers Comparison")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "all_tiers_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_method_comparison(method_df, out_dir):
    """Bar chart: best accuracy per method, grouped by n_components."""
    if method_df is None or method_df.empty:
        return

    method_colors = {
        "RASL": "#2196F3",
        "PCMCI": "#FF9800",
        "GCM": "#4CAF50",
        "TimeSeries": "#9C27B0",
    }

    n_comp_values = sorted(method_df["n_components"].unique())
    methods = method_df["method"].unique()

    fig, ax = plt.subplots(figsize=(max(8, len(n_comp_values) * 3), 6))
    x = np.arange(len(n_comp_values))
    width = 0.8 / max(len(methods), 1)

    for i, method in enumerate(methods):
        msub = method_df[method_df["method"] == method]
        accs = []
        stds = []
        for nc in n_comp_values:
            row = msub[msub["n_components"] == nc]
            if len(row) > 0:
                accs.append(row["accuracy"].values[0])
                stds.append(row["accuracy_std"].values[0])
            else:
                accs.append(0)
                stds.append(0)
        color = method_colors.get(method, "#888888")
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width, yerr=stds, label=method,
                       color=color, edgecolor="black", linewidth=0.5, capsize=3)
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{acc:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={nc}" for nc in n_comp_values], fontsize=10)
    ax.set_ylabel("Best Accuracy (across all classifiers)")
    ax.set_title("HC vs SZ: RASL vs Baselines (Best Classifier per Method)")
    ax.legend(loc="upper right")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "method_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root_dir = os.path.join(args.results_root, args.timestamp)
    if not os.path.isdir(root_dir):
        alt = os.path.join("..", "real_data", args.results_root, args.timestamp)
        if os.path.isdir(alt):
            root_dir = alt
        else:
            print(f"Error: {root_dir} not found")
            sys.exit(1)

    ml_dir = args.output_dir or os.path.join(root_dir, "ml_classification")
    os.makedirs(ml_dir, exist_ok=True)

    print("=" * 80)
    print("UNIFIED CLASSIFIER COMPARISON — HC vs SZ")
    print("=" * 80)
    print(f"Timestamp: {args.timestamp}")
    print(f"Output:    {ml_dir}")
    print()

    # Optionally run all tier scripts first
    if args.run_all:
        extra = []
        if args.configs:
            extra.extend(["--configs"] + args.configs)

        run_tier("classify_hc_sz.py", args.timestamp, extra)
        run_tier("brain_transformer_classify.py", args.timestamp, extra)
        run_tier("solution_set_transformer.py", args.timestamp, extra)

        ts_extra = []
        if args.data_path:
            ts_extra.extend(["--data_path", args.data_path])
        ts_extra.extend(["--n_components"] +
                        [str(n) for n in args.n_components_ts])
        run_tier("timeseries_foundation_classify.py", args.timestamp, ts_extra)

    # Aggregate results
    print("\n" + "=" * 80)
    print("AGGREGATING RESULTS")
    print("=" * 80)

    combined_df = load_tier_results(ml_dir)
    if combined_df is None or combined_df.empty:
        print("\nNo results found. Run with --run-all first, or ensure "
              "tier CSV files exist in the ml_classification directory.")
        sys.exit(1)

    # Save combined CSV
    combined_csv = os.path.join(ml_dir, "all_tiers_combined.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nCombined CSV: {combined_csv}")

    # Best per config+tier
    comparison = build_comparison_table(combined_df)
    print("\n" + "=" * 80)
    print("BEST CLASSIFIER PER CONFIG & TIER")
    print("=" * 80)
    with pd.option_context("display.max_columns", 20, "display.width", 120):
        print(comparison.to_string(index=False))

    comp_csv = os.path.join(ml_dir, "best_per_config_tier.csv")
    comparison.to_csv(comp_csv, index=False)

    # Method comparison
    method_df = build_method_comparison(combined_df)
    if method_df is not None and not method_df.empty:
        print("\n" + "=" * 80)
        print("BEST ACCURACY PER METHOD (across all tiers)")
        print("=" * 80)
        print(method_df.to_string(index=False))
        method_csv = os.path.join(ml_dir, "method_comparison.csv")
        method_df.to_csv(method_csv, index=False)

    # Key finding: does RASL outperform baselines?
    if method_df is not None:
        print("\n" + "=" * 80)
        print("KEY FINDING: RASL vs BASELINES")
        print("=" * 80)
        for n_comp in sorted(method_df["n_components"].unique()):
            sub = method_df[method_df["n_components"] == n_comp]
            rasl_rows = sub[sub["method"] == "RASL"]
            if not rasl_rows.empty:
                rasl_best = rasl_rows.iloc[0]["accuracy"]
                baselines = sub[sub["method"] != "RASL"]
                if not baselines.empty:
                    baseline_best = baselines["accuracy"].max()
                    diff = rasl_best - baseline_best
                    symbol = ">" if diff > 0 else ("<" if diff < 0 else "=")
                    print(f"  N={n_comp}: RASL ({rasl_best:.3f}) "
                          f"{symbol} best baseline ({baseline_best:.3f})  "
                          f"[delta={diff:+.3f}]")

    # Plots
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)
    plot_tier_comparison(comparison, ml_dir)
    plot_method_comparison(method_df, ml_dir)

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
