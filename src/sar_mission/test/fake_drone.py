#!/usr/bin/env python3
"""
fake_drone.py  — Member B, Stub (test/)
A minimal kinematic drone simulator that integrates /cmd_vel into a fake /odom.

Use this to test the full mission_node state machine WITHOUT Gazebo.
Swap this out for the real Gazebo bridge (Member A) once ready.

Physics model:
  - Integrates world-frame linear velocity from /cmd_vel into position.
  - Integrates angular.z into yaw.
  - Publishes nav_msgs/Odometry on /odom at 50 Hz.
  - Responds to /drone/enable (Bool): when false, damps all velocity to zero.

Usage (terminal 1):
  ros2 run sar_mission fake_drone.py

Usage (terminal 2):
  ros2 run sar_mission mission_node.py
  ros2 service call /mission/start std_srvs/srv/Trigger
"""

import math
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import rclpy.time


class FakeDrone(Node):
    def __init__(self):
        super().__init__("fake_drone")

        # Declare parameters
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", 0.0)
        self.declare_parameter("start_z", 0.1)   # small offset so drone isn't buried

        rate_hz    = self.get_parameter("publish_rate_hz").value
        self._x    = self.get_parameter("start_x").value
        self._y    = self.get_parameter("start_y").value
        self._z    = self.get_parameter("start_z").value
        self._yaw  = 0.0
        self._enabled = False

        # Commanded velocities (world frame)
        self._vx = 0.0
        self._vy = 0.0
        self._vz = 0.0
        self._vyaw = 0.0

        # Subscriptions
        self.create_subscription(Twist, "/cmd_vel",       self._cmd_cb,    10)
        self.create_subscription(Bool,  "/drone/enable",  self._enable_cb, 10)

        # Publishers
        self._odom_pub = self.create_publisher(Odometry, "/odom", 10)

        # Timer
        self._dt = 1.0 / rate_hz
        self.create_timer(self._dt, self._tick)

        self.get_logger().info(
            f"FakeDrone ready at ({self._x:.1f}, {self._y:.1f}, {self._z:.1f}). "
            "Waiting for /drone/enable..."
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────
    def _cmd_cb(self, msg: Twist):
        """Accept world-frame velocity commands."""
        if not self._enabled:
            return
        self._vx   = msg.linear.x
        self._vy   = msg.linear.y
        self._vz   = msg.linear.z
        self._vyaw = msg.angular.z

    def _enable_cb(self, msg: Bool):
        self._enabled = msg.data
        if not self._enabled:
            # Damp to zero on disable
            self._vx = self._vy = self._vz = self._vyaw = 0.0
        self.get_logger().info(f"Drone {'enabled' if self._enabled else 'disabled'}")

    # ── Integration tick ───────────────────────────────────────────────────────
    def _tick(self):
        dt = self._dt
        if self._enabled:
            self._x   += self._vx   * dt
            self._y   += self._vy   * dt
            self._z   += self._vz   * dt
            self._yaw += self._vyaw * dt
            self._z    = max(0.0, self._z)  # can't go underground

        self._publish_odom()

    def _publish_odom(self):
        msg = Odometry()
        now = self.get_clock().now().to_msg()
        msg.header.stamp    = now
        msg.header.frame_id = "world"
        msg.child_frame_id  = "X3/base_link"

        msg.pose.pose.position.x = self._x
        msg.pose.pose.position.y = self._y
        msg.pose.pose.position.z = self._z

        # Convert yaw to quaternion
        qw = math.cos(self._yaw / 2.0)
        qz = math.sin(self._yaw / 2.0)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        msg.twist.twist.linear.x  = self._vx
        msg.twist.twist.linear.y  = self._vy
        msg.twist.twist.linear.z  = self._vz
        msg.twist.twist.angular.z = self._vyaw

        self._odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeDrone()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
