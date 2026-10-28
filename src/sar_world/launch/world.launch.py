"""
world.launch.py  — Member A
Launches Gazebo Harmonic with the selected SAR world and starts ros_gz_bridge.

Usage:
  ros2 launch sar_world world.launch.py world:=open_field headless:=false
  ros2 launch sar_world world.launch.py world:=sparse_trees headless:=true
  ros2 launch sar_world world.launch.py world:=dense_trees

Arguments:
  world    : open_field | sparse_trees | dense_trees   (default: open_field)
  headless : true | false                              (default: false)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node


def generate_launch_description():
    pkg_sar_world = get_package_share_directory("sar_world")

    # ── Launch arguments ──────────────────────────────────────────────────────
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="open_field",
        description="World variant to load: open_field | sparse_trees | dense_trees",
        choices=["open_field", "sparse_trees", "dense_trees"],
    )

    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo without GUI (true for batch experiments)",
    )

    world_lc = LaunchConfiguration("world")
    headless_lc = LaunchConfiguration("headless")

    # ── Path to selected world SDF ─────────────────────────────────────────
    world_sdf = PathJoinSubstitution([pkg_sar_world, "worlds",
                                      PythonExpression(["'", world_lc, "' + '.sdf'"])])

    # ── Bridge config ──────────────────────────────────────────────────────
    bridge_yaml = os.path.join(pkg_sar_world, "config", "bridge.yaml")

    # ── Environment: point GZ_SIM_RESOURCE_PATH at our models dir ─────────
    set_gz_resource = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=os.path.join(pkg_sar_world, "models"),
    )

    # ── Gazebo with GUI (default) ─────────────────────────────────────────
    gz_sim_gui = ExecuteProcess(
        cmd=["gz", "sim", world_sdf],
        output="screen",
        condition=UnlessCondition(headless_lc),
    )

    # ── Gazebo headless ────────────────────────────────────────────────────
    gz_sim_headless = ExecuteProcess(
        cmd=["gz", "sim", "-s", world_sdf],  # -s = server-only, no GUI
        output="screen",
        condition=IfCondition(headless_lc),
    )

    # ── ros_gz_bridge ──────────────────────────────────────────────────────
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        arguments=["--ros-args", "--params-file", bridge_yaml],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # ── ros_gz_image (re-publishes compressed camera if needed) ───────────
    # This node is optional; it translates gz Image → sensor_msgs/Image
    # The parameter_bridge above already handles it, but included for clarity.

    return LaunchDescription([
        set_gz_resource,
        world_arg,
        headless_arg,
        gz_sim_gui,
        gz_sim_headless,
        bridge,
    ])
