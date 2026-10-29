#!/usr/bin/env python3
"""
fake_inputs.py — Member D stub
Generates synthetic /odom, /camera/camera_info, and /detections
by projecting known ground truth humans into the moving camera.
Used to verify GeolocatorNode math standalone without running Gazebo.

Usage:
  ros2 run sar_mapping fake_inputs.py
"""

import math
import time
from typing import List, Tuple

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert RPY (rad) to quaternion (x, y, z, w)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return (qx, qy, qz, qw)


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


class FakeInputsNode(Node):
    def __init__(self):
        super().__init__("fake_inputs_node")

        # Fixed Known Humans (ground z=0)
        self._ground_truth_humans = [
            (5.0, 8.0, 0.0),
            (-10.0, -12.0, 0.0),
            (18.0, 3.0, 0.0),
            (-3.0, 5.0, 0.0),
        ]

        # Camera intrinsics
        self._width = 640
        self._height = 480
        self._fx = 457.0
        self._fy = 457.0
        self._cx = 320.0
        self._cy = 240.0

        # Mount parameters
        self._mount_xyz = np.array([0.1, 0.0, -0.05], dtype=np.float64)
        self._pitch = 1.04719755  # 60 deg

        cp = math.cos(self._pitch)
        sp = math.sin(self._pitch)
        self._R_sens_body = np.array([
            [cp, 0.0, -sp],
            [0.0, 1.0, 0.0],
            [sp, 0.0,  cp]
        ], dtype=np.float64)

        # Publishers
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self._cam_info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", 10)
        self._det_pub = self.create_publisher(Detection2DArray, "/detections", 10)

        self._start_time = time.monotonic()
        self._timer = self.create_timer(1.0 / 15.0, self._step)  # 15 Hz

        self.get_logger().info(
            f"FakeInputsNode publishing simulated drone + detections for {len(self._ground_truth_humans)} humans."
        )

    def _step(self):
        t = time.monotonic() - self._start_time
        stamp = self.get_clock().now().to_msg()

        # Circular trajectory: Radius 12 m at altitude 10 m, completing a circle in 35 s
        radius = 12.0
        omega = 2.0 * math.pi / 35.0
        x = radius * math.cos(omega * t)
        y = radius * math.sin(omega * t)
        z = 10.0

        yaw = omega * t + math.pi / 2.0  # Tangent to circle
        qx, qy, qz, qw = euler_to_quat(0.0, 0.0, yaw)

        # 1. Publish /odom
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self._odom_pub.publish(odom)

        # 2. Publish /camera/camera_info
        ci = CameraInfo()
        ci.header.stamp = stamp
        ci.header.frame_id = "camera_optical_frame"
        ci.width = self._width
        ci.height = self._height
        ci.k = [self._fx, 0.0, self._cx, 0.0, self._fy, self._cy, 0.0, 0.0, 1.0]
        self._cam_info_pub.publish(ci)

        # 3. Project ground humans into camera and publish /detections
        R_wb = quat_to_rot_matrix(qx, qy, qz, qw)
        p_drone = np.array([x, y, z], dtype=np.float64)
        p_cam = p_drone + R_wb @ self._mount_xyz

        det_array = Detection2DArray()
        det_array.header.stamp = stamp
        det_array.header.frame_id = "camera_optical_frame"

        for hx, hy, hz in self._ground_truth_humans:
            p_human = np.array([hx, hy, hz], dtype=np.float64)
            d_world = p_human - p_cam
            d_body = R_wb.T @ d_world
            d_sens = self._R_sens_body @ d_body

            # Sensor to optical:
            # X_sens = Z_opt, Y_sens = -X_opt, Z_sens = -Y_opt
            # => Z_opt = X_sens, X_opt = -Y_sens, Y_opt = -Z_sens
            z_opt = d_sens[0]
            x_opt = -d_sens[1]
            y_opt = -d_sens[2]

            if z_opt > 0.5:
                u = self._cx + self._fx * (x_opt / z_opt)
                v = self._cy + self._fy * (y_opt / z_opt)

                if 0 <= u < self._width and 0 <= v < self._height:
                    # Human foot is at (u, v)
                    bbox_h = 40.0
                    bbox_w = 20.0
                    u_center = u
                    v_center = v - bbox_h / 2.0

                    det = Detection2D()
                    det.header.stamp = stamp
                    det.bbox.center.position.x = float(u_center)
                    det.bbox.center.position.y = float(v_center)
                    det.bbox.size_x = bbox_w
                    det.bbox.size_y = bbox_h

                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = "person"
                    hyp.hypothesis.score = 0.92
                    det.results.append(hyp)

                    det_array.detections.append(det)

        if det_array.detections:
            self._det_pub.publish(det_array)


def main(args=None):
    rclpy.init(args=args)
    node = FakeInputsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

