#!/usr/bin/env python3
"""
evaluator_node.py — Member D, Task 5
Evaluates detection & localization performance against Gazebo ground truth.
When /mission/state transitions to DONE, matches confirmed humans to ground truth,
computes metrics (recall, precision, mean localization error, time to first detection,
coverage, mission time), logs a summary to console, and appends a row to results/runs.csv.

Subscriptions:
  /mission/state     std_msgs/String           Mission state machine
  /humans/estimated  geometry_msgs/PoseArray   Confirmed human detections
  /mission/coverage  std_msgs/Float32          Latest coverage percentage
"""

import csv
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float32, String


class EvaluatorNode(Node):
    def __init__(self):
        super().__init__("evaluator_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("world", "open_field")
        self.declare_parameter("ground_truth_file", "")
        self.declare_parameter("match_threshold_m", 2.5)
        self.declare_parameter("results_path", "results/runs.csv")
        self.declare_parameter("run_id", "")

        self._world = self.get_parameter("world").value
        self._gt_file = self.get_parameter("ground_truth_file").value
        self._match_threshold = float(self.get_parameter("match_threshold_m").value)
        self._results_path = self.get_parameter("results_path").value
        self._run_id = self.get_parameter("run_id").value

        if not self._run_id:
            self._run_id = time.strftime("%Y%m%d_%H%M%S")

        # ── Load Ground Truth ─────────────────────────────────────────────────
        self._ground_truth: List[Dict[str, float]] = []
        if not self._gt_file or not os.path.exists(self._gt_file):
            try:
                from ament_index_python.packages import get_package_share_directory
                fallback_path = os.path.join(
                    get_package_share_directory("sar_world"),
                    "config",
                    f"humans_ground_truth_{self._world}.yaml"
                )
                if os.path.exists(fallback_path):
                    self._gt_file = fallback_path
            except Exception:
                pass

        if self._gt_file and os.path.exists(self._gt_file):
            self._load_ground_truth(self._gt_file)
        else:
            self.get_logger().warn(
                f"Ground truth file '{self._gt_file}' not found."
            )

        # ── State ─────────────────────────────────────────────────────────────
        self._latest_estimates: List[Tuple[float, float]] = []
        self._latest_coverage: float = 0.0
        self._mission_start_time: Optional[float] = None
        self._time_to_first_det: Optional[float] = None
        self._evaluated: bool = False

        # ── Subscriptions ─────────────────────────────────────────────────────
        self._state_sub = self.create_subscription(
            String, "/mission/state", self._state_cb, 10
        )
        self._humans_sub = self.create_subscription(
            PoseArray, "/humans/estimated", self._humans_cb, 10
        )
        self._coverage_sub = self.create_subscription(
            Float32, "/mission/coverage", self._coverage_cb, 10
        )

        self.get_logger().info(
            f"EvaluatorNode initialized for world='{self._world}', "
            f"match_thresh={self._match_threshold}m, output='{self._results_path}'"
        )

    def _load_ground_truth(self, filepath: str):
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            self._ground_truth = data.get("humans", [])
            self.get_logger().info(
                f"Loaded {len(self._ground_truth)} ground truth humans from {filepath}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to read ground truth file {filepath}: {e}")

    def _humans_cb(self, msg: PoseArray):
        now = time.monotonic()
        self._latest_estimates = [(p.position.x, p.position.y) for p in msg.poses]

        if len(self._latest_estimates) > 0 and self._time_to_first_det is None:
            if self._mission_start_time is not None:
                self._time_to_first_det = max(0.0, now - self._mission_start_time)
            else:
                self._time_to_first_det = 0.0
            self.get_logger().info(
                f"First human confirmed! Time to first detection: {self._time_to_first_det:.1f}s"
            )

    def _coverage_cb(self, msg: Float32):
        self._latest_coverage = float(msg.data)

    def _state_cb(self, msg: String):
        state = msg.data.strip().upper()
        now = time.monotonic()

        if state == "TAKEOFF" and self._mission_start_time is None:
            self._mission_start_time = now
            self.get_logger().info("Mission started (TAKEOFF). Timer started.")

        elif state == "DONE" and not self._evaluated:
            self._evaluated = True
            mission_time = (now - self._mission_start_time) if self._mission_start_time else 0.0
            self._compute_and_log_metrics(mission_time)

    def _compute_and_log_metrics(self, mission_time: float):
        n_gt = len(self._ground_truth)
        n_est = len(self._latest_estimates)

        # Greedy matching between ground truth and estimates
        matched_gt = set()
        matched_est = set()
        localization_errors = []

        # Build pair distances
        pairs = []
        for i, gt in enumerate(self._ground_truth):
            for j, est in enumerate(self._latest_estimates):
                dist = math.hypot(gt["x"] - est[0], gt["y"] - est[1])
                pairs.append((dist, i, j))

        pairs.sort(key=lambda x: x[0])

        for dist, i, j in pairs:
            if dist > self._match_threshold:
                break
            if i not in matched_gt and j not in matched_est:
                matched_gt.add(i)
                matched_est.add(j)
                localization_errors.append(dist)

        tp = len(matched_gt)
        fp = n_est - tp
        fn = n_gt - tp

        recall = (tp / n_gt) if n_gt > 0 else 1.0
        precision = (tp / n_est) if n_est > 0 else (1.0 if n_gt == 0 else 0.0)
        mean_err = float(np.mean(localization_errors)) if localization_errors else 0.0
        ttfd = self._time_to_first_det if self._time_to_first_det is not None else 0.0

        # Console Summary
        self.get_logger().info("=" * 60)
        self.get_logger().info(f" MISSION EVALUATION SUMMARY — {self._world} (Run {self._run_id})")
        self.get_logger().info("=" * 60)
        self.get_logger().info(f" Ground Truth Humans : {n_gt}")
        self.get_logger().info(f" Confirmed Estimates : {n_est}")
        self.get_logger().info(f" True Positives (TP) : {tp}")
        self.get_logger().info(f" False Positives (FP): {fp}")
        self.get_logger().info(f" False Negatives (FN): {fn}")
        self.get_logger().info(f" Recall              : {recall * 100:.1f} %")
        self.get_logger().info(f" Precision           : {precision * 100:.1f} %")
        self.get_logger().info(f" Mean Loc Error (m)  : {mean_err:.2f} m")
        self.get_logger().info(f" Time to 1st Det (s) : {ttfd:.1f} s")
        self.get_logger().info(f" Final Coverage      : {self._latest_coverage:.1f} %")
        self.get_logger().info(f" Total Mission Time  : {mission_time:.1f} s")
        self.get_logger().info("=" * 60)

        # Append to CSV
        self._write_results_csv(
            world=self._world,
            run_id=self._run_id,
            recall=recall,
            precision=precision,
            mean_loc_error=mean_err,
            time_to_first_det=ttfd,
            coverage=self._latest_coverage,
            mission_time=mission_time,
        )

    def _write_results_csv(
        self, world: str, run_id: str, recall: float, precision: float,
        mean_loc_error: float, time_to_first_det: float,
        coverage: float, mission_time: float
    ):
        os.makedirs(os.path.dirname(os.path.abspath(self._results_path)), exist_ok=True)
        file_exists = os.path.isfile(self._results_path)

        with open(self._results_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(self._results_path) == 0:
                writer.writerow([
                    "scenario", "run_id", "recall", "precision",
                    "mean_loc_error_m", "time_to_first_det_s",
                    "coverage_pct", "mission_time_s"
                ])
            writer.writerow([
                world,
                run_id,
                f"{recall:.4f}",
                f"{precision:.4f}",
                f"{mean_loc_error:.3f}",
                f"{time_to_first_det:.1f}",
                f"{coverage:.1f}",
                f"{mission_time:.1f}",
            ])
        self.get_logger().info(f"Results appended to {self._results_path}")


def main(args=None):
    rclpy.init(args=args)
    node = EvaluatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
