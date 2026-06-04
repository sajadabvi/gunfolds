"""
Compare per-u split solve time vs the old combined (single-job) drasl runtime.

Apples-to-apples on the SOLVER time only (both measured as drasl_time_sec =
grounding + search, excluding PCMCI/BOLD/IO):

  combined      = old runtime_scaling_master.csv  drasl_time_sec   (one job
                  searched all rates u=1..max_urate together)
  per-u split   = for each (n_nodes, instance), the per-rate u*.json files.
                  - critical path = max_u drasl_time_sec  (rates run in
                    parallel, so wall-clock of the solve phase ≈ the slowest u)
                  - total work    = sum_u drasl_time_sec  (CPU booked)

A speedup ratio = combined / critical_path.  >1 means the split was faster
(wall-clock); <1 means it was slower.

NOTE: this ignores the one-time prep (PCMCI/BOLD) job, which the split adds in
front of the solves but which the combined run also paid inside its pipeline.
Add it back if you want end-to-end latency rather than solver-only.

Usage:
    python compare_per_u_vs_combined.py \
        --old_csv results/runtime_scaling/runtime_scaling_master.csv \
        --per_u_dir results/runtime_scaling_per_u --label rtpu
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics as st


def load_old(old_csv):
    """(n_nodes, instance_id) -> combined drasl_time_sec (completed only)."""
    old = {}
    with open(old_csv) as f:
        for row in csv.DictReader(f):
            if row.get("status") != "completed":
                continue
            try:
                n = int(row["n_nodes"]); i = int(row["instance_id"])
                t = float(row["drasl_time_sec"])
            except (ValueError, KeyError):
                continue
            old[(n, i)] = t
    return old


def load_per_u(per_u_dir, label):
    """(n,i) -> {u: (drasl_time_sec, status)} parsed from <label>_n*_inst*__u*.json."""
    pat = os.path.join(per_u_dir, f"{label}_n*_inst*__u*.json")
    groups = {}
    rx = re.compile(rf"{re.escape(label)}_n(\d+)_inst(\d+)__u(\d+)\.json$")
    for fp in glob.glob(pat):
        m = rx.search(os.path.basename(fp))
        if not m:
            continue
        n, i, u = int(m.group(1)), int(m.group(2)), int(m.group(3))
        try:
            with open(fp) as fh:
                d = json.load(fh)
        except Exception:
            continue
        t = d.get("drasl_time_sec")
        groups.setdefault((n, i), {})[u] = (t, d.get("status", ""))
    return groups


def fmt(t):
    if t is None:
        return "   --   "
    if t < 60:
        return f"{t:7.2f}s"
    if t < 3600:
        return f"{t/60:7.2f}m"
    return f"{t/3600:7.2f}h"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old_csv", default="results/runtime_scaling/runtime_scaling_master.csv")
    p.add_argument("--per_u_dir", default="results/runtime_scaling_per_u")
    p.add_argument("--label", default="rtpu")
    p.add_argument("--out", default=None, help="Optional CSV path for the per-instance table.")
    args = p.parse_args()

    old = load_old(args.old_csv)
    groups = load_per_u(args.per_u_dir, args.label)
    if not groups:
        print(f"No per-u files matching {args.label}_n*_inst*__u*.json in {args.per_u_dir}")
        return

    rows = []
    for (n, i) in sorted(groups):
        per_u = groups[(n, i)]
        done = {u: t for u, (t, s) in per_u.items() if t is not None and s == "completed"}
        incomplete = [u for u, (t, s) in per_u.items() if s != "completed"]
        crit = max(done.values()) if done else None
        tot = sum(done.values()) if done else None
        comb = old.get((n, i))
        ratio = (comb / crit) if (comb is not None and crit) else None
        rows.append({
            "n": n, "inst": i, "combined": comb, "n_u": len(per_u),
            "critical": crit, "total": tot, "ratio": ratio,
            "incomplete": incomplete,
        })

    # ── per-instance table ───────────────────────────────────────────────────
    print(f"{'N':>3} {'inst':>4} {'combined':>9} {'split_crit':>11} "
          f"{'split_sum':>10} {'speedup':>8}  notes")
    print("-" * 64)
    for r in rows:
        note = ""
        if r["combined"] is None:
            note = "no combined baseline"
        elif r["incomplete"]:
            note = f"u{r['incomplete']} not completed"
        sp = f"{r['ratio']:6.2f}x" if r["ratio"] else "   --  "
        flag = ""
        if r["ratio"]:
            flag = "  ✓faster" if r["ratio"] > 1 else "  ✗slower"
        print(f"{r['n']:>3} {r['inst']:>4} {fmt(r['combined']):>9} "
              f"{fmt(r['critical']):>11} {fmt(r['total']):>10} {sp:>8}{flag}  {note}")

    # ── per-N summary (instances with both baselines + all u done) ───────────
    print("\nPer-N summary (instances comparable on both sides):")
    print(f"{'N':>3} {'#cmp':>5} {'med_combined':>13} {'med_split_crit':>15} "
          f"{'med_speedup':>12} {'faster/slower':>14}")
    print("-" * 70)
    comparable = [r for r in rows if r["ratio"] is not None and not r["incomplete"]]
    by_n = {}
    for r in comparable:
        by_n.setdefault(r["n"], []).append(r)
    tot_fast = tot_slow = 0
    for n in sorted(by_n):
        rs = by_n[n]
        med_c = st.median([r["combined"] for r in rs])
        med_k = st.median([r["critical"] for r in rs])
        med_sp = st.median([r["ratio"] for r in rs])
        fast = sum(1 for r in rs if r["ratio"] > 1)
        slow = len(rs) - fast
        tot_fast += fast; tot_slow += slow
        print(f"{n:>3} {len(rs):>5} {fmt(med_c):>13} {fmt(med_k):>15} "
              f"{med_sp:>10.2f}x {f'{fast}/{slow}':>14}")

    print("-" * 70)
    if comparable:
        all_sp = [r["ratio"] for r in comparable]
        print(f"\nOVERALL: {len(comparable)} comparable instances  |  "
              f"median speedup {st.median(all_sp):.2f}x  |  "
              f"{tot_fast} faster, {tot_slow} slower")
        print("(speedup = combined_solve_time / per-u_critical_path; "
              ">1 = split won on wall-clock)")
    else:
        print("\nNo instances with both a combined baseline AND all u-solves completed yet.")

    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n_nodes", "instance_id", "combined_sec",
                        "split_critical_sec", "split_total_sec", "speedup",
                        "n_u_files", "incomplete_u"])
            for r in rows:
                w.writerow([r["n"], r["inst"], r["combined"], r["critical"],
                            r["total"], r["ratio"], r["n_u"],
                            ";".join(map(str, r["incomplete"]))])
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
