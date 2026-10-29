"""
mapping.launch.py — Member D
Launches the geolocator node, evaluator node, and RViz2.

Usage:
  ros2 launch sar_mapping mapping.launch.py world:=open_field
  ros2 launch sar_mapping mapping.launch.py world:=sparse_trees rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg_sar_mapping = get_package_share_directory("sar_mapping")
    pkg_sar_world = get_package_share_directory("sar_world")

    mapping_yaml = os.path.join(pkg_sar_mapping, "config", "mapping_params.yaml")
    rviz_config = os.path.join(pkg_sar_mapping, "rviz", "sar.rviz")

    # ── Arguments ─────────────────────────────────────────────────────────────
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="open_field",
        description="World variant: open_field | sparse_trees | dense_trees",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use Gazebo sim time (/clock)",
    )
    rviz_arg = DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Start RViz2 visualization",
    )

    world_lc = LaunchConfiguration("world")
    use_sim_time_lc = LaunchConfiguration("use_sim_time")
    rviz_lc = LaunchConfiguration("rviz")

    # Ground truth file path dynamically based on world arg
    gt_file = PathJoinSubstitution([
        pkg_sar_world, "config",
        PythonExpression(["'humans_ground_truth_' + '", world_lc, "' + '.yaml'"])
    ])

    # ── Geolocator Node ───────────────────────────────────────────────────────
    geolocator_node = Node(
        package="sar_mapping",
        executable="geolocator_node.py",
        name="geolocator_node",
        output="screen",
        parameters=[
            mapping_yaml,
            {"use_sim_time": use_sim_time_lc},
        ],
    )

    # ── Evaluator Node ────────────────────────────────────────────────────────
    evaluator_node = Node(
        package="sar_mapping",
        executable="evaluator_node.py",
        name="evaluator_node",
        output="screen",
        parameters=[
            mapping_yaml,
            {
                "world": world_lc,
                "ground_truth_file": gt_file,
                "use_sim_time": use_sim_time_lc,
            },
        ],
    )

    # ── RViz2 Node ────────────────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": use_sim_time_lc}],
        condition=IfCondition(rviz_lc),
        output="screen",
    )

    return LaunchDescription([
        world_arg,
        use_sim_time_arg,
        rviz_arg,
        geolocator_node,
        evaluator_node,
        rviz_node,
    ])

