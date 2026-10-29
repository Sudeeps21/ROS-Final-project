#!/usr/bin/env python3
"""
image_folder_publisher.py  — Member C, Stub
Publish images from a folder or video file to /camera/image_raw at 15 Hz.
Use this to develop and tune the detector without Gazebo.

Usage:
  # Publish from folder:
  ros2 run sar_perception image_folder_publisher.py \
      --ros-args -p source:=/path/to/frames -p rate_hz:=15.0 -p loop:=true

  # Publish from video file:
  ros2 run sar_perception image_folder_publisher.py \
      --ros-args -p source:=/path/to/video.mp4

Parameters:
  source    : folder path or video file path
  rate_hz   : publish rate (default: 15.0)
  loop      : loop when folder is exhausted (default: true)
  frame_id  : camera frame id (default: X3/camera_link)
"""

import os
import glob
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageFolderPublisher(Node):
    def __init__(self):
        super().__init__("image_folder_publisher")

        self.declare_parameter("source",   "")
        self.declare_parameter("rate_hz",  15.0)
        self.declare_parameter("loop",     True)
        self.declare_parameter("frame_id", "X3/camera_link")

        source   = self.get_parameter("source").value
        rate_hz  = float(self.get_parameter("rate_hz").value)
        self._loop     = self.get_parameter("loop").value
        self._frame_id = self.get_parameter("frame_id").value

        if not source:
            self.get_logger().error("Parameter 'source' is required!")
            raise ValueError("source parameter not set")

        self._bridge = CvBridge()
        self._pub    = self.create_publisher(Image, "/camera/image_raw", 10)
        self._dt     = 1.0 / rate_hz

        source_path = Path(source)
        self._is_video = source_path.is_file() and source_path.suffix in (
            ".mp4", ".avi", ".mov", ".mkv"
        )

        if self._is_video:
            self._cap    = cv2.VideoCapture(str(source_path))
            self._frames = None
            self.get_logger().info(f"Video source: {source_path}")
        else:
            # Folder of images
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            files = []
            for ext in exts:
                files.extend(sorted(glob.glob(os.path.join(source, ext))))
            if not files:
                self.get_logger().error(f"No images found in {source}")
                raise FileNotFoundError(source)
            self._frames  = files
            self._idx     = 0
            self._cap     = None
            self.get_logger().info(
                f"Folder source: {source}  ({len(files)} images)  loop={self._loop}"
            )

        self._timer = self.create_timer(self._dt, self._tick)
        self.get_logger().info(f"Publishing at {rate_hz:.0f} Hz → /camera/image_raw")

    def _tick(self):
        if self._is_video:
            self._publish_video_frame()
        else:
            self._publish_folder_frame()

    def _publish_video_frame(self):
        ret, frame = self._cap.read()
        if not ret:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return
            else:
                self.get_logger().info("Video finished.")
                rclpy.shutdown()
                return
        self._publish(frame)

    def _publish_folder_frame(self):
        if self._idx >= len(self._frames):
            if self._loop:
                self._idx = 0
            else:
                self.get_logger().info("All images published.")
                rclpy.shutdown()
                return

        path = self._frames[self._idx]
        self._idx += 1
        frame = cv2.imread(path)
        if frame is None:
            return
        self._publish(frame)

    def _publish(self, frame):
        msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        self._pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ImageFolderPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
