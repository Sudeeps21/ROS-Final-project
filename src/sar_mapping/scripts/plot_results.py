#!/usr/bin/env python3
"""
plot_results.py — Member D, Task 6
Reads results/runs.csv, prints summary statistics (mean ± std) per scenario,
and generates publication-quality comparison bar charts saved to results/.

Usage:
  python3 scripts/plot_results.py [path/to/runs.csv]
"""

import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/runs.csv"
    if not os.path.exists(csv_path):
        # Look in workspace root if running from ws root
        if os.path.exists(os.path.join("sar_drone_ws", csv_path)):
            csv_path = os.path.join("sar_drone_ws", csv_path)
        else:
            print(f"Error: results file '{csv_path}' not found.")
            sys.exit(1)

    print(f"Loading experiment results from: {csv_path}")

    # Parse CSV
    records = []
    with open(csv_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) <= 1:
        print("No run data found in CSV file.")
        return

    header = [h.strip() for h in lines[0].split(",")]
    col_idx = {name: i for i, name in enumerate(header)}

    scenarios = defaultdict(lambda: defaultdict(list))

    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        scen = parts[col_idx["scenario"]]
        try:
            scenarios[scen]["recall"].append(float(parts[col_idx["recall"]]))
            scenarios[scen]["precision"].append(float(parts[col_idx["precision"]]))
            scenarios[scen]["mean_loc_error"].append(float(parts[col_idx["mean_loc_error_m"]]))
            scenarios[scen]["ttfd"].append(float(parts[col_idx["time_to_first_det_s"]]))
            scenarios[scen]["coverage"].append(float(parts[col_idx["coverage_pct"]]))
            scenarios[scen]["mission_time"].append(float(parts[col_idx["mission_time_s"]]))
        except (ValueError, KeyError) as e:
            print(f"Warning: skipped malformed row: {line} ({e})")

    # Console Summary Table
    print("\n" + "=" * 90)
    print(f"{'Scenario':<16} | {'Runs':<4} | {'Recall (%)':<14} | {'Precision (%)':<14} | {'Loc Err (m)':<12} | {'Coverage (%)':<12}")
    print("-" * 90)

    summary_stats = {}
    for scen, metrics in scenarios.items():
        n = len(metrics["recall"])
        r_m, r_s = np.mean(metrics["recall"]) * 100, np.std(metrics["recall"]) * 100
        p_m, p_s = np.mean(metrics["precision"]) * 100, np.std(metrics["precision"]) * 100
        e_m, e_s = np.mean(metrics["mean_loc_error"]), np.std(metrics["mean_loc_error"])
        c_m, c_s = np.mean(metrics["coverage"]), np.std(metrics["coverage"])

        summary_stats[scen] = {
            "recall": (r_m, r_s),
            "precision": (p_m, p_s),
            "loc_err": (e_m, e_s),
            "coverage": (c_m, c_s),
        }

        print(f"{scen:<16} | {n:<4} | {r_m:5.1f} ± {r_s:4.1f}% | {p_m:5.1f} ± {p_s:4.1f}% | {e_m:4.2f} ± {e_s:4.2f} m | {c_m:5.1f} ± {c_s:4.1f}%")
    print("=" * 90 + "\n")

    # Generate Bar Chart Figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Autonomous SAR Drone — Performance by Scenario", fontsize=14, fontweight="bold")

    scen_names = list(summary_stats.keys())
    x = np.arange(len(scen_names))
    width = 0.55

    # 1. Recall
    ax = axes[0, 0]
    vals = [summary_stats[s]["recall"][0] for s in scen_names]
    errs = [summary_stats[s]["recall"][1] for s in scen_names]
    ax.bar(x, vals, width, yerr=errs, capsize=5, color="#2ecc71", edgecolor="black")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Victim Recall Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(scen_names)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # 2. Precision
    ax = axes[0, 1]
    vals = [summary_stats[s]["precision"][0] for s in scen_names]
    errs = [summary_stats[s]["precision"][1] for s in scen_names]
    ax.bar(x, vals, width, yerr=errs, capsize=5, color="#3498db", edgecolor="black")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Detection Precision")
    ax.set_xticks(x)
    ax.set_xticklabels(scen_names)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # 3. Localization Error
    ax = axes[1, 0]
    vals = [summary_stats[s]["loc_err"][0] for s in scen_names]
    errs = [summary_stats[s]["loc_err"][1] for s in scen_names]
    ax.bar(x, vals, width, yerr=errs, capsize=5, color="#e74c3c", edgecolor="black")
    ax.set_ylabel("Mean Error (meters)")
    ax.set_title("Ground Localization Error")
    ax.set_xticks(x)
    ax.set_xticklabels(scen_names)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # 4. Coverage
    ax = axes[1, 1]
    vals = [summary_stats[s]["coverage"][0] for s in scen_names]
    errs = [summary_stats[s]["coverage"][1] for s in scen_names]
    ax.bar(x, vals, width, yerr=errs, capsize=5, color="#f39c12", edgecolor="black")
    ax.set_ylabel("Coverage Area (%)")
    ax.set_title("Search Area Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(scen_names)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(csv_path))
    out_img = os.path.join(out_dir, "summary_metrics.png")
    plt.savefig(out_img, dpi=200)
    plt.close()
    print(f"Summary charts saved to: {out_img}")


if __name__ == "__main__":
    main()

