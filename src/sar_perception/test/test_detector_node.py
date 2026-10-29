#!/usr/bin/env python3
"""
test_detector_node.py  — Member C, Task 5
Unit test: feed a folder of images through DetectorNode and verify output message format.

Run (no Gazebo needed):
  # Terminal 1 — start the image publisher stub:
  ros2 run sar_perception image_folder_publisher.py \
      --ros-args -p source:=/path/to/frames -p loop:=false

  # Terminal 2 — run this test:
  python3 test/test_detector_node.py --images /path/to/frames --timeout 30

Or as a proper ROS test:
  colcon test --packages-select sar_perception

The test verifies:
  1. /detections is published within timeout_sec of /camera/image_raw being live.
  2. Each Detection2DArray message:
     a. has the same header stamp as the source image (within 10 ms tolerance).
     b. each detection has class_id == 'person'.
     c. bounding box size_x and size_y are positive.
     d. score is in [0, 1].
  3. Publish rate >= 8 Hz (measured over a window of 20 messages).
"""

import sys
import time
import threading
import argparse
from collections import deque
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
import cv2


PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"


class DetectorTester(Node):
    def __init__(self, timeout_sec: float = 30.0):
        super().__init__("detector_tester")
        self._timeout_sec    = timeout_sec
        self._results        = []   # list of (test_name, passed, message)
        self._received_dets  = []   # Detection2DArray messages received
        self._recv_times     = deque(maxlen=30)   # wall-clock times of received dets
        self._last_img_stamp = None
        self._bridge         = CvBridge()

        self._img_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self._det_sub = self.create_subscription(
            Detection2DArray, "/detections", self._det_cb, 10
        )

        self.get_logger().info(
            f"DetectorTester ready. Timeout={timeout_sec:.0f}s. "
            "Waiting for /detections..."
        )

    def _det_cb(self, msg: Detection2DArray):
        now = time.monotonic()
        self._recv_times.append(now)
        self._received_dets.append(msg)

        # Test 2a: stamp matches source image stamp
        if self._last_img_stamp is not None:
            img_t = self._last_img_stamp.sec + self._last_img_stamp.nanosec * 1e-9
            det_t = msg.header.stamp.sec   + msg.header.stamp.nanosec   * 1e-9
            diff_ms = abs(img_t - det_t) * 1000.0
            stamp_ok = diff_ms < 10.0
            self._results.append((
                f"msg {len(self._received_dets)}: stamp matches image",
                stamp_ok,
                f"diff={diff_ms:.1f} ms",
            ))

        # Test 2b/c/d: per-detection checks
        for i, det in enumerate(msg.detections):
            for hyp in det.results:
                class_ok = hyp.hypothesis.class_id == "person"
                score_ok = 0.0 <= hyp.hypothesis.score <= 1.0
                self._results.append((
                    f"msg {len(self._received_dets)} det {i}: class_id=='person'",
                    class_ok, f"got '{hyp.hypothesis.class_id}'",
                ))
                self._results.append((
                    f"msg {len(self._received_dets)} det {i}: score in [0,1]",
                    score_ok, f"got {hyp.hypothesis.score:.3f}",
                ))
            bbox_ok = det.bbox.size_x > 0 and det.bbox.size_y > 0
            self._results.append((
                f"msg {len(self._received_dets)} det {i}: bbox positive size",
                bbox_ok, f"size=({det.bbox.size_x:.1f}, {det.bbox.size_y:.1f})",
            ))

    def publish_images_from_folder(self, folder: Path, rate_hz: float = 15.0):
        """Publish images from folder to /camera/image_raw in a background thread."""
        files = sorted(
            list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        )
        if not files:
            self.get_logger().error(f"No images in {folder}")
            return

        dt = 1.0 / rate_hz

        def _pub_loop():
            for fpath in files:
                img = cv2.imread(str(fpath))
                if img is None:
                    continue
                msg = self._bridge.cv2_to_imgmsg(img, encoding="bgr8")
                msg.header.stamp    = self.get_clock().now().to_msg()
                msg.header.frame_id = "X3/camera_link"
                self._last_img_stamp = msg.header.stamp
                self._img_pub.publish(msg)
                time.sleep(dt)

        t = threading.Thread(target=_pub_loop, daemon=True)
        t.start()
        return t

    def run_tests(self, images_path: Path) -> bool:
        """Run all tests and return True if all pass."""

        # Start publishing images
        pub_thread = self.publish_images_from_folder(images_path)

        # Test 1: receive at least one detection within timeout
        deadline = time.monotonic() + self._timeout_sec
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            if len(self._received_dets) >= 1:
                break

        first_det_ok = len(self._received_dets) >= 1
        self._results.insert(0, (
            "Test 1: /detections published within timeout",
            first_det_ok,
            f"received {len(self._received_dets)} messages",
        ))

        if not first_det_ok:
            return self._print_results()

        # Collect more messages for rate test (Test 3)
        collect_deadline = time.monotonic() + 10.0
        while time.monotonic() < collect_deadline and len(self._received_dets) < 20:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Test 3: rate >= 8 Hz
        if len(self._recv_times) >= 2:
            times = list(self._recv_times)
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            avg_hz = 1.0 / (sum(intervals) / len(intervals)) if intervals else 0.0
        else:
            avg_hz = 0.0

        rate_ok = avg_hz >= 8.0
        self._results.insert(1, (
            f"Test 3: publish rate >= 8 Hz",
            rate_ok,
            f"measured {avg_hz:.1f} Hz over {len(self._recv_times)} messages",
        ))

        return self._print_results()

    def _print_results(self) -> bool:
        print("\n" + "="*60)
        print(" DETECTOR NODE TEST RESULTS")
        print("="*60)
        all_pass = True
        for name, passed, msg in self._results:
            icon = PASS if passed else FAIL
            print(f"  {icon}  {name}  — {msg}")
            if not passed:
                all_pass = False
        print("="*60)
        total = len(self._results)
        passed = sum(1 for _, p, _ in self._results if p)
        print(f"  {passed}/{total} tests passed.")
        return all_pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images",  required=True, help="Folder of test images")
    ap.add_argument("--timeout", type=float, default=30.0)
    args, ros_args = ap.parse_known_args()

    rclpy.init(args=ros_args)
    node = DetectorTester(timeout_sec=args.timeout)
    ok = node.run_tests(Path(args.images))
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
