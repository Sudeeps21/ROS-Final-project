#!/usr/bin/env python3
"""
detector_node.py  — Member C, Tasks 1 & 2
YOLOv8n human detector node.

Subscriptions:
  /camera/image_raw   sensor_msgs/Image      live camera feed from Gazebo (or stub)

Publications:
  /detections         vision_msgs/Detection2DArray   person bounding boxes + scores
  /detections/image   sensor_msgs/Image              annotated image for RViz

Parameters (set via detector_params.yaml or CLI):
  model_path    : path to .pt model file (default: yolov8n.pt, auto-downloaded)
  conf          : confidence threshold (default: 0.35)
  iou           : NMS IoU threshold (default: 0.50)
  imgsz         : inference image size (default: 640)
  device        : 'cpu' | 'cuda:0' | 'mps' (default: 'cpu')
  max_rate_hz   : max publish rate; frames are skipped if inference is slower (default: 15.0)

"Done when": /detections publishes at >= 8 Hz on CPU hardware.
"""

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
)


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")

        # ── Parameters ──────────────────────────────────────────────────────
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("conf",        0.35)
        self.declare_parameter("iou",         0.50)
        self.declare_parameter("imgsz",       640)
        self.declare_parameter("device",      "cpu")
        self.declare_parameter("max_rate_hz", 15.0)

        model_path   = self.get_parameter("model_path").value
        self._conf   = float(self.get_parameter("conf").value)
        self._iou    = float(self.get_parameter("iou").value)
        self._imgsz  = int(self.get_parameter("imgsz").value)
        self._device = self.get_parameter("device").value
        max_rate_hz  = float(self.get_parameter("max_rate_hz").value)
        self._min_interval = 1.0 / max_rate_hz   # minimum seconds between inferences

        # ── Load YOLO model ──────────────────────────────────────────────────
        self.get_logger().info(
            f"Loading YOLO model: {model_path}  device={self._device}  "
            f"conf={self._conf}  imgsz={self._imgsz}"
        )
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
            # Warm-up pass so first inference isn't slow
            dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
            self._model.predict(
                dummy,
                classes=[0],      # COCO class 0 = person
                conf=self._conf,
                iou=self._iou,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )
            self.get_logger().info("YOLO model loaded and warmed up.")
        except ImportError:
            self.get_logger().error(
                "ultralytics not installed! Run: pip install ultralytics"
            )
            raise
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        # ── cv_bridge ────────────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── State ────────────────────────────────────────────────────────────
        self._last_infer_time = 0.0
        self._total_received  = 0
        self._total_processed = 0
        self._total_detections= 0

        # ── Publishers ───────────────────────────────────────────────────────
        self._det_pub  = self.create_publisher(Detection2DArray, "/detections",       10)
        self._img_pub  = self.create_publisher(Image,            "/detections/image", 10)

        # ── Subscriber ───────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self._img_sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_cb, sensor_qos
        )

        # ── Stats timer (every 10 s) ─────────────────────────────────────────
        self.create_timer(10.0, self._log_stats)

        self.get_logger().info(
            f"DetectorNode ready. Subscribed to /camera/image_raw. "
            f"max_rate={max_rate_hz:.0f} Hz"
        )

    # ── Image callback ────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        self._total_received += 1

        # Rate limiting: skip frames if inference is slower than camera
        now = time.monotonic()
        if (now - self._last_infer_time) < self._min_interval:
            return
        self._last_infer_time = now
        self._total_processed += 1

        # Convert to OpenCV BGR
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge error: {e}")
            return

        # ── Run inference ────────────────────────────────────────────────────
        t0 = time.monotonic()
        results = self._model.predict(
            cv_img,
            classes=[0],          # person only
            conf=self._conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        infer_ms = (time.monotonic() - t0) * 1000.0

        # ── Build Detection2DArray ────────────────────────────────────────────
        det_array = Detection2DArray()
        det_array.header = msg.header   # copy stamp from source image (required!)

        annotated = cv_img.copy()

        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # xyxy in pixel coords
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_score      = float(box.conf[0])

                det = Detection2D()
                det.header = msg.header

                # Bounding box centre + size
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w  = x2 - x1
                h  = y2 - y1

                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.size_x            = w
                det.bbox.size_y            = h

                # Hypothesis
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = "person"
                hyp.hypothesis.score    = conf_score
                det.results.append(hyp)

                det_array.detections.append(det)
                self._total_detections += 1

                # Annotate image
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 80), 2)
                label = f"person {conf_score:.2f}"
                cv2.putText(annotated, label, (int(x1), max(int(y1) - 8, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1, cv2.LINE_AA)

        # Add inference time overlay
        cv2.putText(annotated, f"{infer_ms:.0f} ms", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 80), 1, cv2.LINE_AA)

        # Publish
        self._det_pub.publish(det_array)
        ann_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        ann_msg.header = msg.header
        self._img_pub.publish(ann_msg)

    # ── Periodic stats ────────────────────────────────────────────────────────
    def _log_stats(self):
        if self._total_processed == 0:
            self.get_logger().info("DetectorNode: no frames processed yet.")
            return
        self.get_logger().info(
            f"Frames: recv={self._total_received}  "
            f"processed={self._total_processed}  "
            f"detections={self._total_detections}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
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
