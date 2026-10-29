"""
full_mission.launch.py  — sar_bringup (top-level, owned by Member B)
Starts the complete SAR pipeline:
  A → Gazebo world + bridge
  B → Mission / flight controller node
  C → Human detector (YOLO) node
  D → Geolocator, evaluator, RViz

Usage:
  ros2 launch sar_bringup full_mission.launch.py world:=sparse_trees
  ros2 launch sar_bringup full_mission.launch.py world:=open_field headless:=true rviz:=false

NOTE TO TEAM:
  Each member contributes their own IncludeLaunchDescription snippet below.
  Do NOT remove another member's section. Coordinate via group chat before changing
  shared args (world, headless, use_sim_time).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    # ── Common args ────────────────────────────────────────────────────────
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="open_field",
        description="World variant: open_field | sparse_trees | dense_trees",
    )
    headless_arg = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run Gazebo headless (true for batch experiments)",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Launch RViz2 visualization",
    )

    world_lc    = LaunchConfiguration("world")
    headless_lc = LaunchConfiguration("headless")
    rviz_lc     = LaunchConfiguration("rviz")

    # ════════════════════════════════════════════════════════════════════════
    # MEMBER A — Simulation World + Bridge
    # ════════════════════════════════════════════════════════════════════════
    sar_world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("sar_world"),
                "launch",
                "world.launch.py",
            )
        ),
        launch_arguments={
            "world":    world_lc,
            "headless": headless_lc,
        }.items(),
    )

    # ════════════════════════════════════════════════════════════════════════
    # MEMBER B — Flight Controller + Mission Node
    # ════════════════════════════════════════════════════════════════════════
    sar_mission_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("sar_mission"),
                "launch", "mission.launch.py",
            )
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    # ════════════════════════════════════════════════════════════════════════
    # MEMBER C — Human Detector
    # ════════════════════════════════════════════════════════════════════════
    sar_perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("sar_perception"),
                "launch", "detector.launch.py",
            )
        ),
        launch_arguments={"use_sim_time": "true"}.items(),
    )

    # ════════════════════════════════════════════════════════════════════════
    # MEMBER D — Geolocator + Evaluator + RViz
    # ════════════════════════════════════════════════════════════════════════
    sar_mapping_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("sar_mapping"),
                "launch", "mapping.launch.py",
            )
        ),
        launch_arguments={
            "world": world_lc,
            "use_sim_time": "true",
            "rviz": rviz_lc,
        }.items(),
    )

    # ── Assemble ─────────────────────────────────────────────────────────────
    return LaunchDescription([
        world_arg,
        headless_arg,
        rviz_arg,

        # A: world first (no delay)
        sar_world_launch,

        # B: mission controller — small delay to let Gazebo settle
        TimerAction(period=5.0, actions=[sar_mission_launch]),

        # C: detector — after bridge is up
        TimerAction(period=6.0, actions=[sar_perception_launch]),

        # D: mapping — after detector
        TimerAction(period=7.0, actions=[sar_mapping_launch]),
    ])
