"""
detector.launch.py — Member C
Launches the YOLO detector node with parameters from detector_params.yaml.

Usage:
  ros2 launch sar_perception detector.launch.py
  ros2 launch sar_perception detector.launch.py use_sim_time:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_sar_perception = get_package_share_directory("sar_perception")
    detector_params_yaml = os.path.join(pkg_sar_perception, "config", "detector_params.yaml")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use Gazebo sim time (/clock)",
    )
    use_sim_time_lc = LaunchConfiguration("use_sim_time")

    detector_node = Node(
        package="sar_perception",
        executable="detector_node.py",
        name="detector_node",
        output="screen",
        parameters=[
            detector_params_yaml,
            {"use_sim_time": use_sim_time_lc},
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        detector_node,
    ])

