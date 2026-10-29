"""
planner.py  — Member B, Task 2
Lawnmower (boustrophedon) path generator.

Given:
  area_x, area_y  : search area dimensions [m]
  center          : (cx, cy) world-frame center of the area
  altitude        : survey altitude [m]
  hfov_rad        : camera horizontal FOV [rad]
  vfov_rad        : camera vertical FOV [rad]  (computed from hfov and aspect if not given)
  overlap         : fractional row overlap [0–1], default 0.30
  heading         : 'x' (rows along X axis) or 'y' (rows along Y axis)

Returns:
  List of (x, y, z) waypoints in world frame.

Also publishes as nav_msgs/Path on /mission/waypoints if used as a ROS node component.
"""

import math
from typing import List, Tuple

import numpy as np


# ── Helper types ──────────────────────────────────────────────────────────────
Waypoint = Tuple[float, float, float]   # (x, y, z) world frame


def compute_footprint_width(altitude: float, hfov_rad: float) -> float:
    """Ground footprint width [m] for the given altitude and horizontal FOV."""
    return 2.0 * altitude * math.tan(hfov_rad / 2.0)


def compute_footprint_height(altitude: float, vfov_rad: float) -> float:
    """Ground footprint height [m] for the given altitude and vertical FOV."""
    return 2.0 * altitude * math.tan(vfov_rad / 2.0)


def lawnmower_waypoints(
    area_x: float = 60.0,
    area_y: float = 60.0,
    center: Tuple[float, float] = (0.0, 0.0),
    altitude: float = 10.0,
    hfov_rad: float = 1.2,
    aspect: float = 640.0 / 480.0,
    overlap: float = 0.30,
    heading: str = "x",
) -> List[Waypoint]:
    """
    Generate boustrophedon (lawnmower) waypoints.

    heading='x' : rows run along X; drone sweeps in Y direction.
    heading='y' : rows run along Y; drone sweeps in X direction.

    Returns list of (x, y, z) waypoints including the home position at start/end.
    """
    vfov_rad = 2.0 * math.atan(math.tan(hfov_rad / 2.0) / aspect)

    # Footprint dimensions on the ground at survey altitude
    fp_width  = compute_footprint_width(altitude, hfov_rad)   # across-track
    fp_height = compute_footprint_height(altitude, vfov_rad)  # along-track

    cx, cy = center

    if heading == "x":
        # Rows run parallel to X axis. Step in Y direction.
        strip_width = fp_width * (1.0 - overlap)   # distance between row centres
        half_area   = area_y / 2.0
        y_start     = cy - half_area + fp_width / 2.0
        y_end       = cy + half_area - fp_width / 2.0
        n_rows      = max(1, math.ceil((y_end - y_start) / strip_width) + 1)

        x_left  = cx - area_x / 2.0
        x_right = cx + area_x / 2.0

        waypoints: List[Waypoint] = []
        for i in range(n_rows):
            y = y_start + i * strip_width
            y = min(y, y_end)  # clamp
            if i % 2 == 0:
                waypoints.append((x_left,  y, altitude))
                waypoints.append((x_right, y, altitude))
            else:
                waypoints.append((x_right, y, altitude))
                waypoints.append((x_left,  y, altitude))

    else:  # heading == 'y'
        strip_width = fp_height * (1.0 - overlap)
        half_area   = area_x / 2.0
        x_start     = cx - half_area + fp_height / 2.0
        x_end       = cx + half_area - fp_height / 2.0
        n_rows      = max(1, math.ceil((x_end - x_start) / strip_width) + 1)

        y_bottom = cy - area_y / 2.0
        y_top    = cy + area_y / 2.0

        waypoints = []
        for i in range(n_rows):
            x = x_start + i * strip_width
            x = min(x, x_end)
            if i % 2 == 0:
                waypoints.append((x, y_bottom, altitude))
                waypoints.append((x, y_top,    altitude))
            else:
                waypoints.append((x, y_top,    altitude))
                waypoints.append((x, y_bottom, altitude))

    return waypoints


def waypoints_to_ros_path(waypoints: List[Waypoint], frame_id: str = "world"):
    """
    Convert list of (x,y,z) waypoints to a nav_msgs/Path message.
    Import this function inside a ROS node to avoid import-time ROS dependency.
    """
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    import rclpy

    path = Path()
    path.header.frame_id = frame_id

    for (x, y, z) in waypoints:
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.position.z = z
        ps.pose.orientation.w = 1.0  # identity quaternion
        path.poses.append(ps)

    return path


if __name__ == "__main__":
    # Quick sanity check — print waypoints for default config
    wps = lawnmower_waypoints()
    print(f"Generated {len(wps)} waypoints")
    for i, wp in enumerate(wps):
        print(f"  WP {i:03d}: x={wp[0]:+7.2f}  y={wp[1]:+7.2f}  z={wp[2]:.1f}")

    # Camera footprint info
    fp_w = compute_footprint_width(10.0, 1.2)
    print(f"\nCamera footprint width at 10 m: {fp_w:.2f} m")
    print(f"Row spacing (30% overlap): {fp_w * 0.70:.2f} m")
