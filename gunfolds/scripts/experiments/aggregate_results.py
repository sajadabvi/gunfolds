"""
Aggregate per-instance CSVs from the runtime-scaling experiment and print
the headline "N → solve time" table.

Reads:   <input_dir>/n*_inst*.csv  (one row each)
Writes:  <input_dir>/runtime_scaling_master.csv
Prints:  the headline table (median + IQR + completion rate per N)

Usage:
    python aggregate_results.py --input_dir results/runtime_scaling/
    python aggregate_results.py --input_dir results/runtime_scaling/ --plot
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


N_VALUES = [8, 10, 12, 14, 18, 20, 24, 30, 42, 54]
INSTANCES_PER_N = 10


def fmt_time(seconds):
    """Human-readable: s for ≤60, m for 60-3600, h for >3600."""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "  --   "
    if seconds <= 60:
        return f"{seconds:6.2f}s"
    if seconds <= 3600:
        return f"{seconds / 60:6.2f}m"
    return f"{seconds / 3600:6.2f}h"


def fmt_iqr(lo, hi):
    if lo is None or np.isnan(lo) or hi is None or np.isnan(hi):
        return "[  --   ,   --   ]"
    return f"[{fmt_time(lo).strip()}, {fmt_time(hi).strip()}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="results/runtime_scaling/")
    p.add_argument("--plot", action="store_true",
                   help="Save runtime_scaling.svg with per-instance markers + "
                        "median highlighted (log-y vs n_nodes).")
    args = p.parse_args()

    pattern = os.path.join(args.input_dir, "n*_inst*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matching {pattern}")
        return

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [WARN] could not read {f}: {e}")
    if not frames:
        print("No CSVs successfully read.")
        return

    master = pd.concat(frames, ignore_index=True)
    master_path = os.path.join(args.input_dir, "runtime_scaling_master.csv")
    master.to_csv(master_path, index=False)
    print(f"Wrote {master_path}  ({len(master)} rows)\n")

    # ── Headline table ───────────────────────────────────────────────────
    header = (
        f"{'n_nodes':>7s}  {'median(drasl_time_sec)':>22s}  "
        f"{'IQR':>26s}  {'completion_rate':>17s}"
    )
    print(header)
    print("-" * len(header))

    minmax_lines = []
    for N in N_VALUES:
        rows = master[master["n_nodes"] == N]
        completed = rows[rows["status"] == "completed"]
        total = len(rows) if len(rows) else INSTANCES_PER_N

        if len(completed) == 0:
            print(f"{N:>7d}  {'  --   ':>22s}  "
                  f"{'[  --   ,   --   ]':>26s}  "
                  f"{f'0/{total}':>17s}")
            continue

        times = completed["drasl_time_sec"].astype(float).dropna().values
        med = float(np.median(times))
        q1 = float(np.percentile(times, 25))
        q3 = float(np.percentile(times, 75))
        comp = f"{len(completed)}/{total}"

        print(f"{N:>7d}  {fmt_time(med):>22s}  "
              f"{fmt_iqr(q1, q3):>26s}  {comp:>17s}")

        mn = float(np.min(times)); mx = float(np.max(times))
        minmax_lines.append(
            f"  N={N:>2d}  min={fmt_time(mn).strip():>9s}  "
            f"max={fmt_time(mx).strip():>9s}  n_completed={len(completed)}"
        )

    if minmax_lines:
        print("\nPer-N min/max:")
        for line in minmax_lines:
            print(line)

    # ── Optional plot ────────────────────────────────────────────────────
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for N in N_VALUES:
            rows = master[(master["n_nodes"] == N) & (master["status"] == "completed")]
            times = rows["drasl_time_sec"].astype(float).dropna().values
            if len(times) == 0:
                continue
            ax.scatter([N] * len(times), times, alpha=0.4, color="steelblue",
                       s=30, label="instance" if N == N_VALUES[0] else None)
            ax.scatter([N], [np.median(times)], marker="D", s=80,
                       color="darkorange", edgecolor="black",
                       label="median" if N == N_VALUES[0] else None,
                       zorder=5)

        ax.set_yscale("log")
        ax.set_xlabel("n_nodes")
        ax.set_ylabel("drasl_time_sec (log)")
        ax.set_xticks(N_VALUES)
        ax.set_title("drasl runtime vs n_nodes (per-instance + median)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="upper left")
        plot_path = os.path.join(args.input_dir, "runtime_scaling.svg")
        fig.tight_layout()
        fig.savefig(plot_path, format="svg")
        print(f"\nPlot: {plot_path}")


if __name__ == "__main__":
    main()
