#!/usr/bin/env python3
"""
mission_node.py  — Member B, Tasks 1, 3, 4, 5
Autonomous flight controller and mission state machine for the SAR drone.

State machine:
  IDLE  →  TAKEOFF  →  SEARCH  →  RETURN  →  LAND  →  DONE

Subscriptions:
  /odom            nav_msgs/Odometry       drone pose (world frame)

Publications:
  /cmd_vel         geometry_msgs/Twist     velocity commands (world frame)
  /drone/enable    std_msgs/Bool           arm/disarm the motors
  /mission/state   std_msgs/String         current state name
  /mission/waypoints nav_msgs/Path         planned lawnmower path (for RViz)

Services:
  /mission/start   std_srvs/Trigger        begin the mission
  /mission/abort   std_srvs/Trigger        emergency abort → RETURN

Parameters (loaded from mission.yaml via launch):
  area_size_x, area_size_y, area_center, altitude,
  speed_horizontal, speed_vertical, overlap, home,
  geofence_margin, waypoint_timeout, altitude_floor
"""

import math
import time
from collections import deque
from enum import Enum, auto
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from sar_mission.planner import lawnmower_waypoints, waypoints_to_ros_path


# ── State machine enum ────────────────────────────────────────────────────────
class MissionState(Enum):
    IDLE    = auto()
    TAKEOFF = auto()
    SEARCH  = auto()
    RETURN  = auto()
    LAND    = auto()
    DONE    = auto()


STATE_NAMES = {
    MissionState.IDLE:    "IDLE",
    MissionState.TAKEOFF: "TAKEOFF",
    MissionState.SEARCH:  "SEARCH",
    MissionState.RETURN:  "RETURN",
    MissionState.LAND:    "LAND",
    MissionState.DONE:    "DONE",
}

# ── Tuning constants ──────────────────────────────────────────────────────────
CTRL_HZ          = 20.0      # controller loop rate [Hz]
POS_KP_HORIZ     = 0.6       # P-gain, horizontal [1/s]
POS_KP_VERT      = 0.8       # P-gain, vertical   [1/s]
YAW_KP           = 1.2       # P-gain, yaw rate   [rad/s per rad]
WP_ACCEPT_RADIUS = 0.8       # waypoint reached when within this [m]
LAND_ACCEPT_ALT  = 0.25      # consider landed when z < this [m]
TAKEOFF_LOITER   = 2.0       # seconds to hover at altitude before searching


# ── Waypoint type ─────────────────────────────────────────────────────────────
Waypoint = Tuple[float, float, float]   # (x, y, z) world frame


class MissionNode(Node):
    """Full autonomous mission controller."""

    def __init__(self):
        super().__init__("mission_node")

        # ── Declare parameters (overridden by mission.yaml) ────────────────
        self.declare_parameter("area_size_x",       60.0)
        self.declare_parameter("area_size_y",       60.0)
        self.declare_parameter("area_center",       [0.0, 0.0])
        self.declare_parameter("altitude",          10.0)
        self.declare_parameter("speed_horizontal",  2.0)
        self.declare_parameter("speed_vertical",    1.0)
        self.declare_parameter("overlap",           0.30)
        self.declare_parameter("home",              [0.0, 0.0])
        self.declare_parameter("geofence_margin",   5.0)
        self.declare_parameter("waypoint_timeout",  30.0)
        self.declare_parameter("altitude_floor",    1.5)
        self.declare_parameter("hfov_rad",          1.2)

        self._load_params()

        # ── State ──────────────────────────────────────────────────────────
        self._state          = MissionState.IDLE
        self._current_pose   = None          # (x, y, z, yaw) latest
        self._odom_history   = deque(maxlen=200)   # for logging
        self._wp_index       = 0
        self._waypoints: List[Waypoint] = []
        self._wp_timer_start = 0.0
        self._takeoff_reach_time: Optional[float] = None
        self._mission_start_time = 0.0

        # ── Publishers ─────────────────────────────────────────────────────
        latch_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._cmd_vel_pub   = self.create_publisher(Twist,  "/cmd_vel",           10)
        self._enable_pub    = self.create_publisher(Bool,   "/drone/enable",       10)
        self._state_pub     = self.create_publisher(String, "/mission/state",      latch_qos)
        self._waypoints_pub = self.create_publisher(Path,   "/mission/waypoints",  latch_qos)

        # ── Subscribers ────────────────────────────────────────────────────
        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, 10
        )

        # ── Services ───────────────────────────────────────────────────────
        self._start_srv = self.create_service(
            Trigger, "/mission/start", self._start_cb
        )
        self._abort_srv = self.create_service(
            Trigger, "/mission/abort", self._abort_cb
        )

        # ── Control loop timer ─────────────────────────────────────────────
        self._ctrl_timer = self.create_timer(1.0 / CTRL_HZ, self._ctrl_loop)

        self._publish_state()
        self.get_logger().info(
            f"MissionNode ready. Call /mission/start to begin. "
            f"Area={self._area_x}x{self._area_y} m, altitude={self._altitude} m"
        )

    # ── Parameter loading ──────────────────────────────────────────────────────
    def _load_params(self):
        self._area_x    = self.get_parameter("area_size_x").value
        self._area_y    = self.get_parameter("area_size_y").value
        center          = self.get_parameter("area_center").value
        self._center    = (float(center[0]), float(center[1]))
        self._altitude  = self.get_parameter("altitude").value
        self._spd_h     = self.get_parameter("speed_horizontal").value
        self._spd_v     = self.get_parameter("speed_vertical").value
        self._overlap   = self.get_parameter("overlap").value
        home            = self.get_parameter("home").value
        self._home      = (float(home[0]), float(home[1]))
        self._geo_margin= self.get_parameter("geofence_margin").value
        self._wp_timeout= self.get_parameter("waypoint_timeout").value
        self._alt_floor = self.get_parameter("altitude_floor").value
        self._hfov      = self.get_parameter("hfov_rad").value

    # ── Odometry callback ──────────────────────────────────────────────────────
    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = self._quat_to_yaw(q.x, q.y, q.z, q.w)
        self._current_pose = (p.x, p.y, p.z, yaw)
        self._odom_history.append({
            "t": self.get_clock().now().nanoseconds * 1e-9,
            "x": p.x, "y": p.y, "z": p.z,
            "state": STATE_NAMES[self._state],
        })

    # ── Service callbacks ─────────────────────────────────────────────────────
    def _start_cb(self, request, response):
        if self._state != MissionState.IDLE:
            response.success = False
            response.message = f"Cannot start: already in state {STATE_NAMES[self._state]}"
            return response

        if self._current_pose is None:
            response.success = False
            response.message = "No odometry received yet. Is the drone running?"
            return response

        self.get_logger().info("Mission START received.")
        self._mission_start_time = self.get_clock().now().nanoseconds * 1e-9
        self._build_and_publish_path()
        self._set_state(MissionState.TAKEOFF)
        self._enable_drone(True)

        response.success = True
        response.message = "Mission started. State: TAKEOFF"
        return response

    def _abort_cb(self, request, response):
        if self._state in (MissionState.LAND, MissionState.DONE):
            response.success = False
            response.message = "Mission already complete."
            return response

        self.get_logger().warn("Mission ABORT received — transitioning to RETURN.")
        self._set_state(MissionState.RETURN)
        response.success = True
        response.message = "Abort acknowledged. Returning home."
        return response

    # ── Path planning ──────────────────────────────────────────────────────────
    def _build_and_publish_path(self):
        raw_wps = lawnmower_waypoints(
            area_x=self._area_x,
            area_y=self._area_y,
            center=self._center,
            altitude=self._altitude,
            hfov_rad=self._hfov,
            overlap=self._overlap,
        )
        self._waypoints = raw_wps
        self._wp_index  = 0

        path_msg = waypoints_to_ros_path(raw_wps, frame_id="world")
        now = self.get_clock().now().to_msg()
        path_msg.header.stamp = now
        for ps in path_msg.poses:
            ps.header.stamp = now
        self._waypoints_pub.publish(path_msg)

        self.get_logger().info(
            f"Lawnmower path: {len(raw_wps)} waypoints published on /mission/waypoints"
        )

    # ── Main control loop ──────────────────────────────────────────────────────
    def _ctrl_loop(self):
        if self._current_pose is None:
            return

        x, y, z, yaw = self._current_pose

        # ── Safety checks (run in all states except IDLE/DONE) ─────────────
        if self._state not in (MissionState.IDLE, MissionState.DONE):
            # Altitude floor
            if z < self._alt_floor and self._state not in (MissionState.TAKEOFF, MissionState.LAND):
                self.get_logger().error(
                    f"Altitude floor breached! z={z:.2f} < {self._alt_floor}. Aborting."
                )
                self._set_state(MissionState.RETURN)
                return

            # Geofence
            half_x = self._area_x / 2.0 + self._geo_margin
            half_y = self._area_y / 2.0 + self._geo_margin
            cx, cy = self._center
            if abs(x - cx) > half_x or abs(y - cy) > half_y:
                self.get_logger().error(
                    f"Geofence breached! ({x:.1f}, {y:.1f}). Aborting."
                )
                self._set_state(MissionState.RETURN)
                return

        # ── State dispatch ─────────────────────────────────────────────────
        if self._state == MissionState.IDLE:
            return

        elif self._state == MissionState.TAKEOFF:
            self._do_takeoff(x, y, z)

        elif self._state == MissionState.SEARCH:
            self._do_search(x, y, z, yaw)

        elif self._state == MissionState.RETURN:
            self._do_return(x, y, z)

        elif self._state == MissionState.LAND:
            self._do_land(x, y, z)

        elif self._state == MissionState.DONE:
            self._publish_zero_cmd()

    # ── TAKEOFF ────────────────────────────────────────────────────────────────
    def _do_takeoff(self, x: float, y: float, z: float):
        target = (self._home[0], self._home[1], self._altitude)
        vx, vy, vz = self._position_p_controller(x, y, z, *target)
        self._publish_cmd(vx, vy, vz, 0.0)

        # Reached takeoff altitude?
        if abs(z - self._altitude) < 0.5:
            if self._takeoff_reach_time is None:
                self._takeoff_reach_time = self.get_clock().now().nanoseconds * 1e-9
                self.get_logger().info(f"Reached takeoff altitude {z:.2f} m. Hovering {TAKEOFF_LOITER:.1f} s...")

            elapsed = self.get_clock().now().nanoseconds * 1e-9 - self._takeoff_reach_time
            if elapsed >= TAKEOFF_LOITER:
                self.get_logger().info("Loiter complete → SEARCH")
                self._wp_index = 0
                self._wp_timer_start = self.get_clock().now().nanoseconds * 1e-9
                self._set_state(MissionState.SEARCH)
        else:
            self._takeoff_reach_time = None   # reset if we drift below

    # ── SEARCH ─────────────────────────────────────────────────────────────────
    def _do_search(self, x: float, y: float, z: float, yaw: float):
        if self._wp_index >= len(self._waypoints):
            self.get_logger().info("All waypoints visited → RETURN")
            self._set_state(MissionState.RETURN)
            return

        target = self._waypoints[self._wp_index]
        tx, ty, tz = target

        # Waypoint timeout
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._wp_timer_start > self._wp_timeout:
            self.get_logger().warn(
                f"Waypoint {self._wp_index} timed out ({self._wp_timeout:.0f} s). Skipping."
            )
            self._advance_waypoint()
            return

        vx, vy, vz = self._position_p_controller(x, y, z, tx, ty, tz)

        # Yaw to point along travel direction
        vyaw = self._yaw_towards(x, y, tx, ty, yaw)

        self._publish_cmd(vx, vy, vz, vyaw)

        # Check if waypoint reached
        dist_h = math.sqrt((x - tx) ** 2 + (y - ty) ** 2)
        dist_v = abs(z - tz)
        if dist_h < WP_ACCEPT_RADIUS and dist_v < 0.5:
            self.get_logger().info(
                f"WP {self._wp_index}/{len(self._waypoints) - 1} reached "
                f"({tx:.1f}, {ty:.1f}, {tz:.1f})"
            )
            self._advance_waypoint()

    def _advance_waypoint(self):
        self._wp_index += 1
        self._wp_timer_start = self.get_clock().now().nanoseconds * 1e-9

    # ── RETURN ─────────────────────────────────────────────────────────────────
    def _do_return(self, x: float, y: float, z: float):
        target = (self._home[0], self._home[1], self._altitude)
        vx, vy, vz = self._position_p_controller(x, y, z, *target)
        self._publish_cmd(vx, vy, vz, 0.0)

        dist_h = math.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2)
        if dist_h < WP_ACCEPT_RADIUS and abs(z - self._altitude) < 0.5:
            self.get_logger().info("Home position reached → LAND")
            self._set_state(MissionState.LAND)

    # ── LAND ───────────────────────────────────────────────────────────────────
    def _do_land(self, x: float, y: float, z: float):
        # Descend at speed_vertical towards z=0
        target = (self._home[0], self._home[1], 0.0)
        vx, vy, vz = self._position_p_controller(x, y, z, *target)
        # Slow down near ground
        if z < 1.0:
            vz = max(vz, -0.3)
        self._publish_cmd(vx, vy, vz, 0.0)

        if z < LAND_ACCEPT_ALT:
            self.get_logger().info(f"Landed (z={z:.2f} m). Mission DONE.")
            self._publish_zero_cmd()
            self._enable_drone(False)
            self._set_state(MissionState.DONE)
            mission_time = self.get_clock().now().nanoseconds * 1e-9 - self._mission_start_time
            self.get_logger().info(f"Total mission time: {mission_time:.1f} s")

    # ── P-controller ───────────────────────────────────────────────────────────
    def _position_p_controller(
        self,
        cx: float, cy: float, cz: float,
        tx: float, ty: float, tz: float,
    ) -> Tuple[float, float, float]:
        """
        Proportional position controller.
        Returns (vx, vy, vz) clamped to configured max speeds.
        """
        ex = tx - cx
        ey = ty - cy
        ez = tz - cz

        vx = POS_KP_HORIZ * ex
        vy = POS_KP_HORIZ * ey
        vz = POS_KP_VERT  * ez

        # Clamp horizontal speed
        v_h = math.sqrt(vx**2 + vy**2)
        if v_h > self._spd_h:
            scale = self._spd_h / v_h
            vx *= scale
            vy *= scale

        # Clamp vertical speed
        vz = max(-self._spd_v, min(self._spd_v, vz))

        return vx, vy, vz

    def _yaw_towards(
        self, cx: float, cy: float, tx: float, ty: float, current_yaw: float
    ) -> float:
        """Compute yaw rate (rad/s) to point towards the target waypoint."""
        dx = tx - cx
        dy = ty - cy
        dist = math.sqrt(dx**2 + dy**2)
        if dist < 0.5:
            return 0.0
        desired_yaw = math.atan2(dy, dx)
        err = self._wrap_angle(desired_yaw - current_yaw)
        return max(-1.5, min(1.5, YAW_KP * err))

    # ── Publishers ─────────────────────────────────────────────────────────────
    def _publish_cmd(self, vx: float, vy: float, vz: float, vyaw: float):
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.linear.y  = float(vy)
        msg.linear.z  = float(vz)
        msg.angular.z = float(vyaw)
        self._cmd_vel_pub.publish(msg)

    def _publish_zero_cmd(self):
        self._publish_cmd(0.0, 0.0, 0.0, 0.0)

    def _enable_drone(self, enable: bool):
        msg = Bool()
        msg.data = enable
        self._enable_pub.publish(msg)
        self.get_logger().info(f"Drone {'enabled' if enable else 'disabled'}.")

    def _set_state(self, new_state: MissionState):
        old = STATE_NAMES[self._state]
        self._state = new_state
        new = STATE_NAMES[new_state]
        self.get_logger().info(f"State: {old} → {new}")
        self._publish_state()

    def _publish_state(self):
        msg = String()
        msg.data = STATE_NAMES[self._state]
        self._state_pub.publish(msg)

    # ── Utilities ──────────────────────────────────────────────────────────────
    @staticmethod
    def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        return math.atan2(siny, cosy)

    @staticmethod
    def _wrap_angle(a: float) -> float:
        while a >  math.pi: a -= 2.0 * math.pi
        while a < -math.pi: a += 2.0 * math.pi
        return a

    def get_odom_log(self):
        """Return the recorded odometry history (for offline path plotting)."""
        return list(self._odom_history)


def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
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
