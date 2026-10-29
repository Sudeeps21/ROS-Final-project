#!/usr/bin/env python3
"""
plot_path.py  — Member B, Task 6
Compare executed path (from rosbag or live odom CSV) vs planned lawnmower path.

Usage options:

  A) From a rosbag (ros2 bag):
     ros2 bag record -o mission_bag /odom /mission/waypoints
     # ... fly the mission ...
     python3 scripts/plot_path.py --bag mission_bag --world open_field

  B) From the JSON odom log written by mission_node (if --log-path is given):
     python3 scripts/plot_path.py --log odom_log.json --world sparse_trees

  C) Recompute planned path from params (no bag needed):
     python3 scripts/plot_path.py --replan-only

Output: saves plot to results/path_comparison_<world>.png and shows it.
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")   # headless-safe; change to "TkAgg" for interactive window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Allow importing planner without a full ROS install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sar_mission"))
from planner import lawnmower_waypoints


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Plot mission path vs planned path")
    p.add_argument("--bag",          default=None,  help="Path to ros2 bag directory")
    p.add_argument("--log",          default=None,  help="Path to JSON odom log file")
    p.add_argument("--world",        default="open_field",
                   choices=["open_field", "sparse_trees", "dense_trees"])
    p.add_argument("--area-x",       type=float, default=60.0)
    p.add_argument("--area-y",       type=float, default=60.0)
    p.add_argument("--altitude",     type=float, default=10.0)
    p.add_argument("--hfov",         type=float, default=1.2)
    p.add_argument("--overlap",      type=float, default=0.30)
    p.add_argument("--replan-only",  action="store_true", help="Just plot the planned path")
    p.add_argument("--out-dir",      default="results")
    p.add_argument("--show",         action="store_true", help="Show interactive window")
    return p.parse_args()


# ── Load executed path from bag ───────────────────────────────────────────────
def load_from_bag(bag_path: str):
    """Extract /odom poses from a ros2 bag using rosbag2_py."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from nav_msgs.msg import Odometry
    except ImportError:
        print("[plot_path] rosbag2_py not available. Install ros-jazzy-ros2bag.")
        return [], []

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    xs, ys = [], []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic == "/odom":
            msg = deserialize_message(data, Odometry)
            xs.append(msg.pose.pose.position.x)
            ys.append(msg.pose.pose.position.y)

    return xs, ys


# ── Load from JSON log ────────────────────────────────────────────────────────
def load_from_json(log_path: str):
    with open(log_path) as f:
        records = json.load(f)
    xs = [r["x"] for r in records]
    ys = [r["y"] for r in records]
    return xs, ys


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(args, planned_wps, exec_xs, exec_ys):
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # Search area boundary
    half_x, half_y = args.area_x / 2.0, args.area_y / 2.0
    rect = mpatches.Rectangle(
        (-half_x, -half_y), args.area_x, args.area_y,
        linewidth=1.5, edgecolor="#e94560", facecolor="none", linestyle="--",
        label="Search area boundary", zorder=2,
    )
    ax.add_patch(rect)

    # Planned path
    px = [w[0] for w in planned_wps]
    py = [w[1] for w in planned_wps]
    ax.plot(px, py, color="#0f3460", linewidth=1.5, zorder=3)
    ax.scatter(px, py, color="#0f3460", s=25, zorder=4)
    # Planned path legend proxy
    plan_line = mpatches.Patch(color="#0f3460", label=f"Planned path ({len(planned_wps)} WPs)")

    # Waypoint numbers
    for i, (x, y) in enumerate(zip(px, py)):
        if i % max(1, len(px) // 12) == 0:
            ax.text(x + 0.5, y + 0.5, str(i), color="#e2e2e2", fontsize=6, zorder=6)

    # Executed path
    if exec_xs:
        ax.plot(exec_xs, exec_ys, color="#e94560", linewidth=1.2,
                alpha=0.85, zorder=5, label=f"Executed path ({len(exec_xs)} pts)")
        ax.scatter(exec_xs[0],  exec_ys[0],  marker="^", color="#00b4d8", s=80, zorder=7, label="Start")
        ax.scatter(exec_xs[-1], exec_ys[-1], marker="s", color="#f4a261", s=80, zorder=7, label="End")

        # Path error stats
        min_len = min(len(exec_xs), len(px))
        if min_len > 0:
            errs = [math.sqrt((exec_xs[i] - px[i])**2 + (exec_ys[i] - py[i])**2)
                    for i in range(min_len)]
            mean_err = np.mean(errs)
            max_err  = np.max(errs)
            ax.set_title(
                f"Path comparison — {args.world}\n"
                f"Mean tracking error: {mean_err:.2f} m   Max: {max_err:.2f} m",
                color="#e2e2e2", fontsize=13, pad=12,
            )
        else:
            ax.set_title(f"Planned lawnmower path — {args.world}", color="#e2e2e2", fontsize=13)
    else:
        ax.set_title(f"Planned lawnmower path — {args.world}", color="#e2e2e2", fontsize=13)

    # Home marker
    ax.scatter(0, 0, marker="*", color="#ffd166", s=200, zorder=8, label="Home")

    # Legend / labels
    handles, labels = ax.get_legend_handles_labels()
    if exec_xs:
        handles.append(plan_line)
        labels.append(plan_line.get_label())
    else:
        handles.append(plan_line)
        labels.append(plan_line.get_label())
    ax.legend(handles, labels, facecolor="#0d1117", labelcolor="#e2e2e2",
              fontsize=9, loc="upper right")

    ax.set_xlabel("X [m]", color="#e2e2e2")
    ax.set_ylabel("Y [m]", color="#e2e2e2")
    ax.tick_params(colors="#e2e2e2")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333366")
    ax.set_aspect("equal")
    ax.grid(color="#333366", linewidth=0.5, alpha=0.5)

    # Camera footprint width annotation
    fp_w = 2.0 * args.altitude * math.tan(args.hfov / 2.0)
    strip = fp_w * (1.0 - args.overlap)
    info = (f"Alt={args.altitude:.0f} m  FOV={math.degrees(args.hfov):.0f}°  "
            f"FP={fp_w:.1f} m  Overlap={args.overlap:.0%}  Strip={strip:.1f} m")
    ax.text(0.01, 0.01, info, transform=ax.transAxes,
            color="#aaaaaa", fontsize=7, verticalalignment="bottom")

    plt.tight_layout()

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"path_comparison_{args.world}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot_path] Saved: {out_path}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # Build planned path
    planned_wps = lawnmower_waypoints(
        area_x=args.area_x,
        area_y=args.area_y,
        altitude=args.altitude,
        hfov_rad=args.hfov,
        overlap=args.overlap,
    )
    print(f"[plot_path] Planned path: {len(planned_wps)} waypoints")

    exec_xs, exec_ys = [], []

    if not args.replan_only:
        if args.bag:
            exec_xs, exec_ys = load_from_bag(args.bag)
            print(f"[plot_path] Loaded {len(exec_xs)} odom points from bag")
        elif args.log:
            exec_xs, exec_ys = load_from_json(args.log)
            print(f"[plot_path] Loaded {len(exec_xs)} odom points from JSON log")
        else:
            print("[plot_path] No --bag or --log given. Plotting planned path only.")

    plot(args, planned_wps, exec_xs, exec_ys)


if __name__ == "__main__":
    main()
