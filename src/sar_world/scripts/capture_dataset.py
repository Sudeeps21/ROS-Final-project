#!/usr/bin/env python3
"""
capture_dataset.py  — Member A, Task 6
Subscribe to /camera/image_raw and save frames to sar_world/dataset/.

Usage (while flying with teleop):
  ros2 run sar_world capture_dataset

Or equivalently:
  python3 scripts/capture_dataset.py

The node saves every N-th frame (configurable) as a JPEG into:
  dataset/frames/  — raw frames
  dataset/annot/   — (empty dir; for Member C to populate with CVAT/Roboflow labels)

Press Ctrl+C to stop. A summary is printed on exit.

Parameters (via ROS 2 params or env):
  ~output_dir   : absolute path to output dir (default: <pkg_share>/dataset)
  ~save_every   : save 1 out of every N frames (default: 3, giving ~5 Hz saves at 15 Hz camera)
  ~max_frames   : stop after saving this many frames (default: 350)
"""

import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class DatasetCapture(Node):
    def __init__(self):
        super().__init__("dataset_capture")

        # Parameters
        default_out = os.path.join(
            os.path.dirname(__file__), "..", "dataset"
        )
        self.declare_parameter("output_dir", default_out)
        self.declare_parameter("save_every", 3)
        self.declare_parameter("max_frames", 350)

        self.output_dir = self.get_parameter("output_dir").value
        self.save_every = self.get_parameter("save_every").value
        self.max_frames = self.get_parameter("max_frames").value

        # Directories
        self.frames_dir = os.path.join(self.output_dir, "frames")
        self.annot_dir  = os.path.join(self.output_dir, "annot")
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.annot_dir,  exist_ok=True)

        self.bridge = CvBridge()
        self._counter = 0       # frames received
        self._saved   = 0       # frames saved

        self.sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self._image_cb,
            10,
        )

        self.get_logger().info(
            f"DatasetCapture ready. Saving every {self.save_every} frames "
            f"(max {self.max_frames}) → {self.frames_dir}"
        )

    # ── callback ──────────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        if self._saved >= self.max_frames:
            return

        self._counter += 1
        if self._counter % self.save_every != 0:
            return

        # Convert to OpenCV BGR
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        # Filename: frame_<stamp_sec>_<stamp_nsec>.jpg
        stamp = msg.header.stamp
        fname = f"frame_{stamp.sec:010d}_{stamp.nanosec:09d}.jpg"
        fpath = os.path.join(self.frames_dir, fname)
        cv2.imwrite(fpath, cv_img, [cv2.IMWRITE_JPEG_QUALITY, 92])

        self._saved += 1
        if self._saved % 50 == 0:
            self.get_logger().info(f"Saved {self._saved}/{self.max_frames} frames")

        if self._saved >= self.max_frames:
            self.get_logger().info(
                f"Reached max frames ({self.max_frames}). Shutting down capture node."
            )
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DatasetCapture()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        saved = node._saved
        node.destroy_node()
        print(f"\n[capture_dataset] Done. Saved {saved} frames to {node.frames_dir}")
        print(f"[capture_dataset] Hand the 'dataset/' folder to Member C for labeling.")


if __name__ == "__main__":
    main()
