"""
mission.launch.py  — Member B
Launches the mission_node with parameters from mission.yaml.

Usage:
  ros2 launch sar_mission mission.launch.py                      # uses Gazebo /odom
  ros2 launch sar_mission mission.launch.py fake_drone:=true     # use fake_drone stub

After launch, start the mission:
  ros2 service call /mission/start std_srvs/srv/Trigger
  ros2 service call /mission/abort std_srvs/srv/Trigger
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_sar_mission  = get_package_share_directory("sar_mission")
    pkg_sar_bringup  = get_package_share_directory("sar_bringup")

    mission_yaml = os.path.join(pkg_sar_bringup, "config", "mission.yaml")

    # ── Args ──────────────────────────────────────────────────────────────────
    fake_drone_arg = DeclareLaunchArgument(
        "fake_drone",
        default_value="false",
        description="Launch fake_drone stub instead of relying on Gazebo",
    )
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use Gazebo sim time (/clock)",
    )

    fake_drone_lc    = LaunchConfiguration("fake_drone")
    use_sim_time_lc  = LaunchConfiguration("use_sim_time")

    # ── Fake drone node (only when fake_drone:=true) ───────────────────────────
    fake_drone_node = Node(
        package="sar_mission",
        executable="fake_drone.py",
        name="fake_drone",
        output="screen",
        parameters=[{"use_sim_time": False}],  # fake drone uses wall clock
        condition=IfCondition(fake_drone_lc),
    )

    # ── Mission node ───────────────────────────────────────────────────────────
    mission_node = Node(
        package="sar_mission",
        executable="mission_node.py",
        name="mission_node",
        output="screen",
        parameters=[
            mission_yaml,
            {"use_sim_time": use_sim_time_lc},
        ],
        remappings=[],
    )

    return LaunchDescription([
        fake_drone_arg,
        use_sim_time_arg,
        fake_drone_node,
        # Small delay so fake_drone has time to start publishing /odom
        TimerAction(period=1.0, actions=[mission_node]),
    ])
