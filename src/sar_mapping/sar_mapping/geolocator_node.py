#!/usr/bin/env python3
"""
geolocator_node.py — Member D, Tasks 1, 2, 4
Projects 2D human detections to 3D world coordinates, performs tracking/clustering,
publishes confirmed estimates + markers, and estimates survey coverage.

Subscriptions:
  /detections         vision_msgs/Detection2DArray   2D detections from Member C
  /odom               nav_msgs/Odometry              Drone pose in world frame
  /camera/camera_info sensor_msgs/CameraInfo         Camera intrinsics

Publications:
  /humans/estimated   geometry_msgs/PoseArray        Confirmed human positions (world frame)
  /humans/markers     visualization_msgs/MarkerArray RViz markers (cylinders + labels)
  /mission/coverage   std_msgs/Float32               Coverage percentage (0–100%)
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import ColorRGBA, Float32, Header
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster


@dataclass
class OdomRecord:
    stamp_sec: float
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class HumanCandidate:
    candidate_id: int
    x: float
    y: float
    z: float = 0.0
    count: int = 1
    scores: List[float] = field(default_factory=list)
    last_seen_sec: float = 0.0

    @property
    def avg_score(self) -> float:
        return float(np.mean(self.scores)) if self.scores else 0.0


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert unit quaternion to 3x3 rotation matrix."""
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1e-9:
        return np.eye(3)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def slerp(q1: Tuple[float, float, float, float],
          q2: Tuple[float, float, float, float],
          t: float) -> Tuple[float, float, float, float]:
    """Spherical linear interpolation between two quaternions (x, y, z, w)."""
    dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]
    q2_mod = q2
    if dot < 0.0:
        dot = -dot
        q2_mod = (-q2[0], -q2[1], -q2[2], -q2[3])

    if dot > 0.9995:
        res = [q1[i] + t * (q2_mod[i] - q1[i]) for i in range(4)]
        norm = math.sqrt(sum(x * x for x in res))
        return tuple(x / norm for x in res)

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    sin_theta = math.sin(theta)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    res = [s0 * q1[i] + s1 * q2_mod[i] for i in range(4)]
    return tuple(res)


class GeolocatorNode(Node):
    def __init__(self):
        super().__init__("geolocator_node")

        # ── Declare Parameters ────────────────────────────────────────────────
        self.declare_parameter("mount_xyz", [0.1, 0.0, -0.05])
        self.declare_parameter("mount_rpy", [0.0, 1.04719755, 0.0])  # pitch = 60°
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("fx", 457.0)
        self.declare_parameter("fy", 457.0)
        self.declare_parameter("cx", 320.0)
        self.declare_parameter("cy", 240.0)

        self.declare_parameter("merge_dist_m", 1.5)
        self.declare_parameter("confirm_count", 3)
        self.declare_parameter("max_range_m", 40.0)
        self.declare_parameter("ground_z", 0.0)

        self.declare_parameter("area_size_x", 60.0)
        self.declare_parameter("area_size_y", 60.0)
        self.declare_parameter("area_center", [0.0, 0.0])
        self.declare_parameter("coverage_grid_res", 1.0)
        self.declare_parameter("coverage_step_frames", 5)

        # ── Read Parameters ───────────────────────────────────────────────────
        self._mount_xyz = np.array(self.get_parameter("mount_xyz").value, dtype=np.float64)
        mount_rpy = self.get_parameter("mount_rpy").value
        self._pitch = float(mount_rpy[1])

        self._width = int(self.get_parameter("camera_width").value)
        self._height = int(self.get_parameter("camera_height").value)
        self._fx = float(self.get_parameter("fx").value)
        self._fy = float(self.get_parameter("fy").value)
        self._cx = float(self.get_parameter("cx").value)
        self._cy = float(self.get_parameter("cy").value)

        self._merge_dist = float(self.get_parameter("merge_dist_m").value)
        self._confirm_count = int(self.get_parameter("confirm_count").value)
        self._max_range = float(self.get_parameter("max_range_m").value)
        self._ground_z = float(self.get_parameter("ground_z").value)

        self._area_x = float(self.get_parameter("area_size_x").value)
        self._area_y = float(self.get_parameter("area_size_y").value)
        self._area_center = self.get_parameter("area_center").value
        self._grid_res = float(self.get_parameter("coverage_grid_res").value)
        self._cov_step = int(self.get_parameter("coverage_step_frames").value)

        # ── Precompute Camera Mount Rotations ─────────────────────────────────
        # Optical to Camera Sensor (FLU): X_sens = Z_opt, Y_sens = -X_opt, Z_sens = -Y_opt
        self._R_sens_opt = np.array([
            [ 0.0,  0.0, 1.0],
            [-1.0,  0.0, 0.0],
            [ 0.0, -1.0, 0.0]
        ], dtype=np.float64)

        # Pitch around body Y-axis (positive pitch angles drone camera down)
        cp = math.cos(self._pitch)
        sp = math.sin(self._pitch)
        self._R_body_sens = np.array([
            [ cp, 0.0,  sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0,  cp]
        ], dtype=np.float64)

        # Body from Optical
        self._R_body_opt = self._R_body_sens @ self._R_sens_opt

        # ── State Buffers ─────────────────────────────────────────────────────
        self._odom_buffer: deque[OdomRecord] = deque(maxlen=200)
        self._candidates: List[HumanCandidate] = []
        self._next_id: int = 1
        self._frame_count: int = 0

        # Coverage grid setup
        self._grid_nx = int(math.ceil(self._area_x / self._grid_res))
        self._grid_ny = int(math.ceil(self._area_y / self._grid_res))
        self._coverage_grid = np.zeros((self._grid_nx, self._grid_ny), dtype=bool)

        # ── Subscriptions & Publishers ────────────────────────────────────────
        sensor_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self._odom_sub = self.create_subscription(
            Odometry, "/odom", self._odom_cb, sensor_qos
        )
        self._cam_info_sub = self.create_subscription(
            CameraInfo, "/camera/camera_info", self._camera_info_cb, 5
        )
        self._det_sub = self.create_subscription(
            Detection2DArray, "/detections", self._detections_cb, 10
        )

        self._humans_pub = self.create_publisher(PoseArray, "/humans/estimated", 10)
        self._markers_pub = self.create_publisher(MarkerArray, "/humans/markers", 10)
        self._coverage_pub = self.create_publisher(Float32, "/mission/coverage", 10)
        self._tf_broadcaster = TransformBroadcaster(self)

        # 5 Hz visualization timer so RViz markers stay continuously active
        self._viz_timer = self.create_timer(0.2, self._timer_viz_cb)

        self.get_logger().info(
            f"GeolocatorNode initialized. Mount pitch={self._pitch:.3f} rad, "
            f"merge_dist={self._merge_dist}m, confirm_count={self._confirm_count}"
        )

    def _timer_viz_cb(self):
        self._publish_estimates_and_markers(self.get_clock().now().to_msg())

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _camera_info_cb(self, msg: CameraInfo):
        if msg.k[0] > 0.0:
            self._fx = float(msg.k[0])
            self._fy = float(msg.k[4])
            self._cx = float(msg.k[2])
            self._cy = float(msg.k[5])
            self._width = int(msg.width)
            self._height = int(msg.height)

    def _odom_cb(self, msg: Odometry):
        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        record = OdomRecord(
            stamp_sec=stamp_sec,
            x=pos.x, y=pos.y, z=pos.z,
            qx=ori.x, qy=ori.y, qz=ori.z, qw=ori.w,
        )
        self._odom_buffer.append(record)

        self._frame_count += 1
        if self._frame_count % self._cov_step == 0:
            self._update_coverage(record)

        # Broadcast world -> base_link TF so RViz fixed frame 'world' is satisfied
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.translation.x = pos.x
        tf_msg.transform.translation.y = pos.y
        tf_msg.transform.translation.z = pos.z
        tf_msg.transform.rotation = ori
        self._tf_broadcaster.sendTransform(tf_msg)

    def _get_interpolated_odom(self, target_sec: float) -> Optional[OdomRecord]:
        if not self._odom_buffer:
            return None

        # If target is newer or older than buffer extremes
        if target_sec <= self._odom_buffer[0].stamp_sec:
            return self._odom_buffer[0]
        if target_sec >= self._odom_buffer[-1].stamp_sec:
            return self._odom_buffer[-1]

        # Binary or linear scan
        buf = list(self._odom_buffer)
        for i in range(len(buf) - 1):
            t0 = buf[i].stamp_sec
            t1 = buf[i + 1].stamp_sec
            if t0 <= target_sec <= t1:
                dt = t1 - t0
                alpha = (target_sec - t0) / dt if dt > 1e-6 else 0.0
                o0, o1 = buf[i], buf[i + 1]

                ix = (1.0 - alpha) * o0.x + alpha * o1.x
                iy = (1.0 - alpha) * o0.y + alpha * o1.y
                iz = (1.0 - alpha) * o0.z + alpha * o1.z
                iq = slerp((o0.qx, o0.qy, o0.qz, o0.qw), (o1.qx, o1.qy, o1.qz, o1.qw), alpha)

                return OdomRecord(
                    stamp_sec=target_sec,
                    x=ix, y=iy, z=iz,
                    qx=iq[0], qy=iq[1], qz=iq[2], qw=iq[3],
                )
        return self._odom_buffer[-1]

    def _detections_cb(self, msg: Detection2DArray):
        det_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        odom = self._get_interpolated_odom(det_sec)
        if odom is None:
            return

        new_confirmed = False
        for det in msg.detections:
            # Extract foot point (bottom-center of bbox)
            u = det.bbox.center.position.x
            v = det.bbox.center.position.y + det.bbox.size_y / 2.0

            score = 0.5
            if det.results:
                score = float(det.results[0].hypothesis.score)

            pos_world = self._project_pixel_to_ground(u, v, odom)
            if pos_world is not None:
                is_conf = self._add_observation(pos_world[0], pos_world[1], score, det_sec)
                if is_conf:
                    new_confirmed = True

        self._publish_estimates_and_markers(msg.header.stamp)

    def _project_pixel_to_ground(self, u: float, v: float, odom: OdomRecord) -> Optional[Tuple[float, float, float]]:
        """Project pixel coordinate (u, v) to ground plane (z = ground_z) in world frame."""
        # 1. Optical ray
        r_opt = np.array([
            (u - self._cx) / self._fx,
            (v - self._cy) / self._fy,
            1.0,
        ], dtype=np.float64)

        # 2. Drone body to world rotation
        R_wb = quat_to_rot_matrix(odom.qx, odom.qy, odom.qz, odom.qw)

        # 3. Ray in world frame
        r_world = R_wb @ (self._R_body_opt @ r_opt)
        norm = np.linalg.norm(r_world)
        if norm < 1e-6:
            return None
        r_world /= norm

        # Ray must point down towards ground
        if r_world[2] >= -0.01:
            return None

        # 4. Camera position in world frame
        p_drone = np.array([odom.x, odom.y, odom.z], dtype=np.float64)
        p_cam = p_drone + R_wb @ self._mount_xyz

        # Must be above ground
        dz = p_cam[2] - self._ground_z
        if dz <= 0.0:
            return None

        # Ray parameter lambda
        lam = - dz / r_world[2]
        if lam <= 0.0 or lam > self._max_range:
            return None

        gx = p_cam[0] + lam * r_world[0]
        gy = p_cam[1] + lam * r_world[1]
        gz = self._ground_z

        return (float(gx), float(gy), float(gz))

    # ── Tracking & Clustering ─────────────────────────────────────────────────
    def _add_observation(self, x: float, y: float, score: float, stamp_sec: float) -> bool:
        """Add detection to nearest candidate or spawn new candidate."""
        best_cand: Optional[HumanCandidate] = None
        best_dist = float("inf")

        for cand in self._candidates:
            dist = math.hypot(cand.x - x, cand.y - y)
            if dist < best_dist:
                best_dist = dist
                best_cand = cand

        if best_cand is not None and best_dist < self._merge_dist:
            # Update running average
            n = best_cand.count
            best_cand.x = (best_cand.x * n + x) / (n + 1)
            best_cand.y = (best_cand.y * n + y) / (n + 1)
            best_cand.count += 1
            best_cand.scores.append(score)
            best_cand.last_seen_sec = stamp_sec
            return best_cand.count == self._confirm_count  # newly confirmed
        else:
            cand = HumanCandidate(
                candidate_id=self._next_id,
                x=x, y=y, z=self._ground_z,
                count=1,
                scores=[score],
                last_seen_sec=stamp_sec,
            )
            self._next_id += 1
            self._candidates.append(cand)
            return False

    # ── Coverage Estimation ───────────────────────────────────────────────────
    def _update_coverage(self, odom: OdomRecord):
        # 4 image corners
        corners = [
            (0.0, 0.0),
            (float(self._width), 0.0),
            (float(self._width), float(self._height)),
            (0.0, float(self._height)),
        ]
        ground_pts = []
        for u, v in corners:
            pt = self._project_pixel_to_ground(u, v, odom)
            if pt is not None:
                ground_pts.append((pt[0], pt[1]))

        if len(ground_pts) == 4:
            # Rasterize polygon into coverage grid
            min_x = min(p[0] for p in ground_pts)
            max_x = max(p[0] for p in ground_pts)
            min_y = min(p[1] for p in ground_pts)
            max_y = max(p[1] for p in ground_pts)

            x0 = self._area_center[0] - self._area_x / 2.0
            y0 = self._area_center[1] - self._area_y / 2.0

            i_min = max(0, int((min_x - x0) / self._grid_res))
            i_max = min(self._grid_nx - 1, int((max_x - x0) / self._grid_res))
            j_min = max(0, int((min_y - y0) / self._grid_res))
            j_max = min(self._grid_ny - 1, int((max_y - y0) / self._grid_res))

            if i_min <= i_max and j_min <= j_max:
                for i in range(i_min, i_max + 1):
                    for j in range(j_min, j_max + 1):
                        self._coverage_grid[i, j] = True

        pct = float(np.count_nonzero(self._coverage_grid)) / float(self._grid_nx * self._grid_ny) * 100.0
        cov_msg = Float32()
        cov_msg.data = min(100.0, pct)
        self._coverage_pub.publish(cov_msg)

    # ── Visualization ─────────────────────────────────────────────────────────
    def _publish_estimates_and_markers(self, stamp):
        confirmed = [c for c in self._candidates if c.count >= self._confirm_count]
        unconfirmed = [c for c in self._candidates if c.count < self._confirm_count]

        # 1. PoseArray (confirmed victims)
        pa = PoseArray()
        pa.header.stamp = stamp
        pa.header.frame_id = "world"

        for c in confirmed:
            pose = Pose()
            pose.position.x = c.x
            pose.position.y = c.y
            pose.position.z = 0.05
            pose.orientation.w = 1.0
            pa.poses.append(pose)
        self._humans_pub.publish(pa)

        # 2. MarkerArray
        ma = MarkerArray()

        # Search area boundary (cyan line on ground)
        area_m = Marker()
        area_m.header.stamp = stamp
        area_m.header.frame_id = "world"
        area_m.ns = "search_boundary"
        area_m.id = 0
        area_m.type = Marker.LINE_STRIP
        area_m.action = Marker.ADD
        area_m.scale.x = 0.25
        area_m.color = ColorRGBA(r=0.0, g=0.85, b=1.0, a=0.9)
        half_x = self._area_x / 2.0
        half_y = self._area_y / 2.0
        for px, py in [
            (-half_x, -half_y), (half_x, -half_y),
            (half_x, half_y), (-half_x, half_y), (-half_x, -half_y)
        ]:
            pt = Point()
            pt.x, pt.y, pt.z = px, py, 0.05
            area_m.points.append(pt)
        ma.markers.append(area_m)

        # Unconfirmed candidates (instant amber/yellow detection points on grid)
        for c in unconfirmed:
            # Ground spot point
            pt_m = Marker()
            pt_m.header.stamp = stamp
            pt_m.header.frame_id = "world"
            pt_m.ns = "detected_points"
            pt_m.id = c.candidate_id + 50000
            pt_m.type = Marker.SPHERE
            pt_m.action = Marker.ADD
            pt_m.pose.position.x = c.x
            pt_m.pose.position.y = c.y
            pt_m.pose.position.z = 0.25
            pt_m.pose.orientation.w = 1.0
            pt_m.scale.x = 0.6
            pt_m.scale.y = 0.6
            pt_m.scale.z = 0.5
            pt_m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.85)  # Amber
            ma.markers.append(pt_m)

            # Ground ring
            disc = Marker()
            disc.header.stamp = stamp
            disc.header.frame_id = "world"
            disc.ns = "detected_rings"
            disc.id = c.candidate_id + 60000
            disc.type = Marker.CYLINDER
            disc.action = Marker.ADD
            disc.pose.position.x = c.x
            disc.pose.position.y = c.y
            disc.pose.position.z = 0.05
            disc.pose.orientation.w = 1.0
            disc.scale.x = 1.0
            disc.scale.y = 1.0
            disc.scale.z = 0.05
            disc.color = ColorRGBA(r=1.0, g=0.85, b=0.0, a=0.5)
            ma.markers.append(disc)

            # Label
            lbl = Marker()
            lbl.header.stamp = stamp
            lbl.header.frame_id = "world"
            lbl.ns = "candidate_labels"
            lbl.id = c.candidate_id + 70000
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = c.x
            lbl.pose.position.y = c.y
            lbl.pose.position.z = 0.9
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.45
            lbl.color = ColorRGBA(r=1.0, g=0.9, b=0.2, a=1.0)
            lbl.text = f"Detected ({c.avg_score:.0%})"
            ma.markers.append(lbl)

        # Confirmed victims (prominent red/green target disc, 3D cylinder, pin, text)
        for c in confirmed:
            # 1. Ground target disc on grid
            disc = Marker()
            disc.header.stamp = stamp
            disc.header.frame_id = "world"
            disc.ns = "victim_grid_target"
            disc.id = c.candidate_id + 20000
            disc.type = Marker.CYLINDER
            disc.action = Marker.ADD
            disc.pose.position.x = c.x
            disc.pose.position.y = c.y
            disc.pose.position.z = 0.06
            disc.pose.orientation.w = 1.0
            disc.scale.x = 1.8
            disc.scale.y = 1.8
            disc.scale.z = 0.08
            disc.color = ColorRGBA(r=1.0, g=0.15, b=0.15, a=0.75)  # Bright red target disk
            ma.markers.append(disc)

            # 2. Standing body cylinder
            cyl = Marker()
            cyl.header.stamp = stamp
            cyl.header.frame_id = "world"
            cyl.ns = "confirmed_humans"
            cyl.id = c.candidate_id
            cyl.type = Marker.CYLINDER
            cyl.action = Marker.ADD
            cyl.pose.position.x = c.x
            cyl.pose.position.y = c.y
            cyl.pose.position.z = 0.9  # half cylinder height
            cyl.pose.orientation.w = 1.0
            cyl.scale.x = 0.6
            cyl.scale.y = 0.6
            cyl.scale.z = 1.8
            cyl.color = ColorRGBA(r=0.1, g=0.95, b=0.2, a=0.9)  # Vivid green
            ma.markers.append(cyl)

            # 3. Head pin / sphere
            head = Marker()
            head.header.stamp = stamp
            head.header.frame_id = "world"
            head.ns = "victim_head_pins"
            head.id = c.candidate_id + 30000
            head.type = Marker.SPHERE
            head.action = Marker.ADD
            head.pose.position.x = c.x
            head.pose.position.y = c.y
            head.pose.position.z = 1.95
            head.pose.orientation.w = 1.0
            head.scale.x = 0.5
            head.scale.y = 0.5
            head.scale.z = 0.5
            head.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)  # Red pin
            ma.markers.append(head)

            # 4. Text label
            lbl = Marker()
            lbl.header.stamp = stamp
            lbl.header.frame_id = "world"
            lbl.ns = "human_labels"
            lbl.id = c.candidate_id + 10000
            lbl.type = Marker.TEXT_VIEW_FACING
            lbl.action = Marker.ADD
            lbl.pose.position.x = c.x
            lbl.pose.position.y = c.y
            lbl.pose.position.z = 2.4
            lbl.pose.orientation.w = 1.0
            lbl.scale.z = 0.65
            lbl.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            lbl.text = f"VICTIM #{c.candidate_id} ({c.avg_score:.0%})"
            ma.markers.append(lbl)

        self._markers_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = GeolocatorNode()
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

