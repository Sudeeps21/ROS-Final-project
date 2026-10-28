# SAR Drone — Autonomous Human Search & Detection (ROS 2 / Gazebo Harmonic)

**Stack:** ROS 2 Jazzy · Gazebo Harmonic · `ros_gz_bridge` · YOLOv8 · RViz2  
**Team:** 4 members | **Package assignments:** A=`sar_world` · B=`sar_mission` · C=`sar_perception` · D=`sar_mapping`

---

## Quick Start

```bash
# 1. Source ROS 2
source /opt/ros/jazzy/setup.bash

# 2. Build
cd sar_drone_ws
colcon build --symlink-install
source install/setup.bash

# 3. Launch single world (Member A — world + bridge)
ros2 launch sar_world world.launch.py world:=open_field

# 4. Full pipeline (after all members integrate)
ros2 launch sar_bringup full_mission.launch.py world:=sparse_trees
ros2 launch sar_bringup full_mission.launch.py world:=open_field headless:=true

# 5. Start mission
ros2 service call /mission/start std_srvs/srv/Trigger
```

---

## Package Overview

| Package | Owner | Purpose |
|---|---|---|
| `sar_world` | **Member A** | Gazebo worlds, X3 drone SDF, `ros_gz_bridge` config, dataset capture |
| `sar_mission` | Member B | Lawnmower planner, P-controller, state machine |
| `sar_perception` | Member C | YOLOv8 detector node |
| `sar_mapping` | Member D | Geolocation, tracking, RViz, metrics CSV |
| `sar_bringup` | All (B owns) | Top-level launch + shared config |

---

## Key Topics (Interface Contract)

See [`drone_human_detection_project.md`](../drone_human_detection_project.md) Section 3 for the full contract.

| Topic | Type | Direction |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Gazebo → C, D |
| `/odom` | `nav_msgs/Odometry` | Gazebo → B, D |
| `/cmd_vel` | `geometry_msgs/Twist` | B → Gazebo |
| `/detections` | `vision_msgs/Detection2DArray` | C → D |
| `/mission/state` | `std_msgs/String` | B → D |
| `/humans/estimated` | `geometry_msgs/PoseArray` | D → RViz |

---

## World Scenarios

| Scenario | World arg | Humans | Description |
|---|---|---|---|
| S1 | `open_field` | 6 | Baseline, no occlusion |
| S2 | `sparse_trees` | 6 | Partial occlusion |
| S3 | `dense_trees` | 8 | Heavy occlusion |

Ground truth files: `sar_world/config/humans_ground_truth_<world>.yaml`

---

## Dataset Capture (Member A, Task 6)

After launching the world, fly with teleop and run:
```bash
ros2 run sar_world capture_dataset.py
```
Saves ~300 frames to `sar_world/dataset/frames/`. Hand to Member C by hour 2.

---

## Git Workflow

- Branch per member: `a/world`, `b/mission`, `c/perception`, `d/mapping`
- Only touch your own package
- Shared config changes (`mission.yaml`, `camera_mount.yaml`) → group chat first
- Merge to `main` at integration checkpoints only
