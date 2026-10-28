#!/usr/bin/env python3
"""
record_demo.py  — Member A, Task 7
Record a short screen capture of Gazebo + RViz side-by-side as a fallback demo clip.

Requirements:
  pip install opencv-python
  System: ffmpeg or kazam must be installed for screen recording.

This script:
  1. Sends the drone a hover command (enable + zero cmd_vel) for a few seconds.
  2. Uses ffmpeg to record the screen for 30 seconds into dataset/demo_hover.mp4.
  3. Stops recording and prints the output path.

Usage:
  # In one terminal: ros2 launch sar_world world.launch.py
  # In another:      python3 scripts/record_demo.py

Note: adjust DISPLAY if needed (e.g., :0 for local, :1 for virtual display).
"""

import os
import subprocess
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist


RECORD_DURATION = 30  # seconds
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "demo_hover.mp4"
)
DISPLAY = os.environ.get("DISPLAY", ":0")


class HoverPublisher(Node):
    """Enable drone and publish zero velocity to make it hover in place."""

    def __init__(self):
        super().__init__("demo_hover_publisher")
        self.enable_pub = self.create_publisher(Bool, "/drone/enable", 1)
        self.vel_pub    = self.create_publisher(Twist, "/cmd_vel", 1)
        self.timer = self.create_timer(0.1, self._publish)
        self.get_logger().info("HoverPublisher: enabling drone and holding hover...")

    def _publish(self):
        # Enable drone
        enable_msg = Bool()
        enable_msg.data = True
        self.enable_pub.publish(enable_msg)

        # Zero velocity → hover
        twist = Twist()
        self.vel_pub.publish(twist)


def record_screen(output_path: str, duration: int):
    """Use ffmpeg to capture the screen."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",                        # overwrite if exists
        "-video_size", "1920x1080",
        "-framerate", "15",
        "-f", "x11grab",
        "-i", DISPLAY,
        "-t", str(duration),
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        output_path,
    ]
    print(f"[record_demo] Starting screen capture: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[record_demo] Recording saved to: {output_path}")
    else:
        print(f"[record_demo] ffmpeg error:\n{result.stderr}")
        print("[record_demo] Fallback: install kazam or OBS and record manually.")


def main():
    rclpy.init()
    node = HoverPublisher()

    # Start recording in a background thread
    rec_thread = threading.Thread(
        target=record_screen,
        args=(OUTPUT_FILE, RECORD_DURATION),
        daemon=True,
    )
    rec_thread.start()

    # Spin for the duration of the recording
    start = time.time()
    while (time.time() - start) < (RECORD_DURATION + 2):
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()
    rec_thread.join(timeout=5)
    print("[record_demo] Done.")


if __name__ == "__main__":
    main()
