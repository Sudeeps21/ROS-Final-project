# Autonomous Drone Search & Human Detection in Gazebo

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS_2-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Gazebo Harmonic](https://img.shields.io/badge/Gazebo-Harmonic-orange.svg)](https://gazebosim.org/docs/harmonic/)
[![YOLOv8](https://img.shields.io/badge/Detector-YOLOv8n-green.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-Apache_2.0-lightgrey.svg)](LICENSE)

An autonomous quadrotor search and rescue (SAR) simulation system built on **ROS 2 Jazzy** and **Gazebo Harmonic**. The drone performs autonomous takeoff, flies an optimized **lawnmower (boustrophedon) survey grid** over an outdoor search area, runs real-time **YOLOv8 human detection** on its onboard downward-tilted camera, projects 2D detections into **3D world ground coordinates**, clusters and tracks victims, visualizes targets as 3D markers on the RViz map grid, and automatically evaluates search metrics against ground truth.

---

## Table of Contents

- [1. System Architecture](#1-system-architecture)
- [2. Features](#2-features)
- [3. Interface Contract](#3-interface-contract)
- [4. Repository Structure](#4-repository-structure)
- [5. Prerequisites & Installation](#5-prerequisites--installation)
- [6. Quick Start & Execution](#6-quick-start--execution)
- [7. Viewing Drone POV in Gazebo & RViz](#7-viewing-drone-pov-in-gazebo--rviz)
- [8. Evaluation & Batch Experiments](#8-evaluation--batch-experiments)
- [9. Scenarios & Benchmark Metrics](#9-scenarios--benchmark-metrics)
- [10. Troubleshooting](#10-troubleshooting)

---

## 1. System Architecture

```
                 +--------------------------------+
                 |          Gazebo World          |  (sar_world)
                 |   X3 Drone + Downward Camera   |
                 |   Terrain, Humans, Obstacles   |
                 +---------------+----------------+
                                 |  ros_gz_bridge
   /camera/image_raw  /camera/camera_info  /odom  /clock      ^ /cmd_vel
          |                  |               |                |
          v                  |               |         +------+-------+
 +----------------+          |               +-------> |  Flight ctrl |  (sar_mission)
 | Human detector |          |               |         |  + Lawnmower |
 |     YOLOv8     |  (sar_   |               |         +------+-------+
 +-------+--------+   percep)|               |                |
         | /detections       |               |                | /mission/state
         v                   v               v                v
 +--------------------------------------------------------------------+
 |           Geolocation + Tracking + Grid Markers + Evaluation       |  (sar_mapping)
 +--------------------------------------------------------------------+
         |                      |                     |
 /humans/estimated       /humans/markers       /mission/coverage    results/runs.csv
 (PoseArray -> RViz)   (MarkerArray -> RViz)   (Float32 -> RViz)    (Benchmark Metrics)
```

---

## 2. Features

- **Autonomous Flight & Coverage Planning:**
  - Automated state machine: `IDLE` → `TAKEOFF` → `SEARCH` → `RETURN` → `LAND` → `DONE`.
  - Analytical boustrophedon (lawnmower) path generation over configurable areas (default $60 \times 60\text{ m}$) with 30% row overlap based on camera FOV and altitude ($10\text{ m}$).
  - Closed-loop P position and yaw controller clamping speeds ($2\text{ m/s}$ horizontal, $1\text{ m/s}$ vertical).
  - Safety geofencing, waypoint timeout skipping, and `/mission/abort` trigger service.

- **Real-Time AI Perception:**
  - Lightweight YOLOv8n detector tuned for aerial person detection on CPU/GPU.
  - Subscribes to `/camera/image_raw` ($640 \times 480$ @ 15 Hz) and publishes standard `vision_msgs/Detection2DArray` with confidence scores.
  - Generates annotated visualization stream on `/detections/image`.

- **Geolocation & Target Tracking:**
  - Projects bottom-center foot point $(u, v)$ through camera intrinsics $K$, camera mount rotation ($60^\circ$ pitch down), and SLERP-interpolated drone odometry to ground plane $z = 0$.
  - Spatial clustering merges observations within $1.5\text{ m}$ running average; confirms victims after $\ge 2$ detections.
  - Dynamic footprint polygon projection rasterizes survey coverage onto a $1\text{ m}$ grid over the search boundary.

- **Grid Visualization & RViz Markers:**
  - **Ground Target Discs:** $1.8\text{ m}$ red target disks flat on the ground grid ($z = 0.06\text{ m}$) marking exact victim positions.
  - **Instant Spotting:** Raw detections instantly display as amber ground markers on the grid.
  - **3D Mannequins & Pins:** Standing cylinders ($1.8\text{ m}$) with red pin spheres ($z = 1.95\text{ m}$).
  - **Floating Billboards:** Text labels displaying `VICTIM #[ID] (92%)`.
  - **PoseArray Poses:** Red flat directional poses on `/humans/estimated`.

- **Automated Performance Evaluation:**
  - Automatically compares confirmed positions to ground-truth coordinates on `/mission/state == DONE`.
  - Computes Recall, Precision, Mean Localization Error, Time to First Detection, Coverage %, and Mission Duration.
  - Logs results to CSV and generates publication-grade bar charts and summary tables.

---

## 3. Interface Contract

All packages communicate strictly over standard ROS 2 message types:

| Topic / Service | ROS 2 Type | Direction | Purpose |
|---|---|---|---|
| `/clock` | `rosgraph_msgs/msg/Clock` | Gazebo → All | Simulation clock (`use_sim_time:=true`) |
| `/camera/image_raw` | `sensor_msgs/msg/Image` | Gazebo → Perception, Mapping | RGB image ($640 \times 480$, 15 Hz) |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | Gazebo → Mapping | Camera focal length & optical center |
| `/odom` | `nav_msgs/msg/Odometry` | Gazebo → Mission, Mapping | Drone world-frame position & orientation |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | Mission → Gazebo | Velocity commands ($v_x, v_y, v_z, \omega_z$) |
| `/drone/enable` | `std_msgs/msg/Bool` | Mission → Gazebo | Motor arming/disarming |
| `/detections` | `vision_msgs/msg/Detection2DArray` | Perception → Mapping | 2D bounding boxes & person confidence |
| `/detections/image` | `sensor_msgs/msg/Image` | Perception → RViz | Annotated video stream with bounding boxes |
| `/mission/start` | `std_srvs/srv/Trigger` | User → Mission | Starts autonomous mission |
| `/mission/abort` | `std_srvs/srv/Trigger` | User → Mission | Aborts search and returns drone to home |
| `/mission/state` | `std_msgs/msg/String` | Mission → Mapping | State machine (`TAKEOFF`, `SEARCH`, etc.) |
| `/mission/waypoints` | `nav_msgs/msg/Path` | Mission → RViz | Planned boustrophedon flight path |
| `/mission/coverage` | `std_msgs/msg/Float32` | Mapping → Evaluator, User | Survey coverage percentage ($0 - 100\%$) |
| `/humans/estimated` | `geometry_msgs/msg/PoseArray` | Mapping → RViz | Confirmed human positions on grid |
| `/humans/markers` | `visualization_msgs/msg/MarkerArray` | Mapping → RViz | 3D cylinders, ground discs, pins, labels |

### Coordinate & Mount Conventions
- **World Frame:** `world`, ENU (East-North-Up), $z$ up, ground at $z = 0$.
- **Camera Mount:** Translation $[0.1, 0.0, -0.05]\text{ m}$ relative to `base_link`, pitched down by $60^\circ$ ($1.047\text{ rad}$) around the body $Y$ axis.
- **Flight Defaults:** Altitude $10\text{ m}$, cruising speed $2\text{ m/s}$, search area $60 \times 60\text{ m}$ centered at $[0, 0]$.

---

## 4. Repository Structure

```
sar_drone_ws/
├── src/
│   ├── sar_world/               # Simulation world, X3 drone model & bridge
│   │   ├── config/              # bridge.yaml, camera_mount.yaml, ground truth YAMLs
│   │   ├── launch/              # world.launch.py (Gazebo Harmonic + bridge)
│   │   ├── models/              # Drone meshes and SDF assets
│   │   ├── scripts/             # capture_dataset.py, record_demo.py
│   │   └── worlds/              # open_field.sdf, sparse_trees.sdf, dense_trees.sdf
│   ├── sar_mission/             # Flight controller & mission planner
│   │   ├── launch/              # mission.launch.py
│   │   ├── sar_mission/         # mission_node.py, planner.py
│   │   ├── scripts/             # plot_path.py
│   │   └── test/                # fake_drone.py (kinematic stub)
│   ├── sar_perception/          # YOLOv8 human detection
│   │   ├── config/              # detector_params.yaml
│   │   ├── launch/              # detector.launch.py
│   │   ├── sar_perception/      # detector_node.py
│   │   ├── scripts/             # evaluate_detector.py
│   │   └── test/                # test_detector_node.py, image_folder_publisher.py
│   ├── sar_mapping/             # Geolocation, clustering, RViz & evaluation
│   │   ├── config/              # mapping_params.yaml
│   │   ├── launch/              # mapping.launch.py
│   │   ├── rviz/                # sar.rviz
│   │   ├── sar_mapping/         # geolocator_node.py, evaluator_node.py
│   │   ├── scripts/             # run_experiments.sh, plot_results.py
│   │   └── test/                # fake_inputs.py, test_geolocator.py
│   └── sar_bringup/             # Top-level orchestration & mission config
│       ├── config/              # mission.yaml
│       └── launch/              # full_mission.launch.py
├── results/                     # runs.csv, summary_metrics.png
└── README.md
```

---

## 5. Prerequisites & Installation

### Requirements
- **OS:** Ubuntu 24.04 LTS (Noble Numbat)
- **ROS 2:** Jazzy Jalisco (`ros-jazzy-desktop`)
- **Simulator:** Gazebo Harmonic (`gz-sim8`)
- **Bridge:** `ros-jazzy-ros-gz-bridge`, `ros-jazzy-ros-gz-image`, `ros-jazzy-vision-msgs`

### 1. Install ROS & Gazebo Dependencies
```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-image \
  ros-jazzy-vision-msgs \
  ros-jazzy-cv-bridge
```

### 2. Python ML Dependencies
Install PyTorch (CPU) and Ultralytics with versions compatible with ROS 2 Jazzy's OpenCV bridge:
```bash
pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install --break-system-packages "numpy<2.0,>=1.26" "opencv-python<=4.9.0.80" ultralytics
```

### 3. Clone and Build Workspace
```bash
cd ~/ROS\ Final\ Project/sar_drone_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 6. Quick Start & Execution

### 1. Launch Complete Pipeline (GUI + RViz)
Launch Gazebo Harmonic, the ROS-Gazebo bridge, the YOLO detector, flight controller, geolocator, and RViz:
```bash
cd ~/ROS\ Final\ Project/sar_drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch sar_bringup full_mission.launch.py world:=open_field
```

*World options: `world:=open_field` | `world:=sparse_trees` | `world:=dense_trees`*

### 2. Trigger Autonomous Search
Once Gazebo and RViz are open and the nodes are initialized, open a second terminal:
```bash
cd ~/ROS\ Final\ Project/sar_drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 service call /mission/start std_srvs/srv/Trigger
```

### 3. Emergency Abort (Optional)
To cancel the mission and command the drone to return home immediately:
```bash
ros2 service call /mission/abort std_srvs/srv/Trigger
```

---

## 7. Viewing Drone POV in Gazebo & RViz

### In RViz
- The **Annotated Camera** panel displays the real-time onboard camera stream with YOLOv8 bounding box annotations, class labels, and confidence scores.
- The **3D Viewport** displays:
  - Planned flight path (yellow polyline)
  - Drone pose and trajectory (blue odometry path)
  - Detected victims on the grid (red target rings, standing cylinders, and confidence text billboards)

### In Gazebo Sim
1. **Onboard Camera Feed:** Click the three dots menu (`⋮`) in the top-right corner → Select **Image Display** → Enter topic `/camera`.
2. **Follow Drone (Chase Cam):** In the right **Entity Tree**, click **`X3`** → Click the **Follow** tool icon in the top toolbar to lock the 3D scene camera onto the drone.

---

## 8. Evaluation & Batch Experiments

### Automated Benchmark Suite (3 Scenarios × 3 Repeats)
To run the automated headless experiment suite across all three world environments:
```bash
cd ~/ROS\ Final\ Project/sar_drone_ws
bash src/sar_mapping/scripts/run_experiments.sh
```
This script:
1. Iterates over `open_field`, `sparse_trees`, and `dense_trees` (3 runs each).
2. Spawns headless simulation, arms the drone, runs the survey, and detects targets.
3. Automatically evaluates results when `/mission/state` reaches `DONE`.
4. Appends metrics to `results/runs.csv`.
5. Invokes `plot_results.py` to generate summary tables and bar charts.

### Plotting Saved Results
To re-generate summary charts from an existing CSV:
```bash
python3 src/sar_mapping/scripts/plot_results.py results/runs.csv
```
Summary figure will be saved to `results/summary_metrics.png`.

---

## 9. Scenarios & Benchmark Metrics

### World Environments

| Scenario | World Arg | Human Victims | Environment Description |
|:---:|:---|:---:|:---|
| **S1** | `open_field` | 6 | Baseline flat terrain with rocks; zero tree canopy occlusion |
| **S2** | `sparse_trees` | 6 | Scattered oak and pine trees; partial visual occlusion |
| **S3** | `dense_trees` | 8 | Dense forest canopy and underbrush; heavy occlusion testing recall limits |

### Evaluation Metrics Defined

- **Victim Recall (%):** $\frac{\text{True Positives}}{\text{Total Ground Truth Humans}} \times 100$
- **Detection Precision (%):** $\frac{\text{True Positives}}{\text{Confirmed Detections}} \times 100$
- **Mean Localization Error ($\text{m}$):** Average Euclidean 2D distance between matched estimated positions and ground-truth coordinates ($< 2.5\text{ m}$ threshold).
- **Time to First Detection ($\text{s}$):** Elapsed seconds from takeoff until the first victim is confirmed.
- **Search Area Coverage (%):** Percentage of $60 \times 60\text{ m}$ grid covered by the projected camera footprint.
- **Mission Time ($\text{s}$):** Total duration from takeoff to return and landing.

---

## 10. Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| `No odometry received yet` | Gazebo started paused | Ensure Gazebo is running (`-r` flag in launch, or click the bottom-left orange play button). |
| `No Image` in RViz | Simulation paused or detector warmup | Ensure Gazebo is unpaused. YOLO takes ~2 seconds to warm up on first run. |
| `Fixed Frame [world] does not exist` | TF tree missing | Fixed in latest build: `geolocator_node.py` broadcasts `world -> base_link` TF from odometry. |
| `Couldn't parse params file` | YAML missing `ros__parameters` | Verify `mission.yaml` and `mapping_params.yaml` use standard ROS 2 `/**: ros__parameters:` format. |
| `Joint rotor_0_joint could not be found` | Incomplete drone model in SDF | Resolved: SDF files use complete X3 quadrotor model definition with 4 rotor joints. |

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
