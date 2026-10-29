#!/bin/bash
# run_experiments.sh — Member D, Task 6
# Runs 3 scenarios x 3 repeats in headless mode, waits for mission DONE,
# collects metrics into results/runs.csv, and plots the results.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "=========================================================="
echo " Starting Autonomous Drone SAR Experiment Suite"
echo " Workspace: $WS_DIR"
echo "=========================================================="

source /opt/ros/jazzy/setup.bash
source "$WS_DIR/install/setup.bash"

WORLDS=("open_field" "sparse_trees" "dense_trees")
REPEATS=3

for world in "${WORLDS[@]}"; do
  for run in $(seq 1 $REPEATS); do
    echo "----------------------------------------------------------"
    echo " Starting World: $world | Run: $run / $REPEATS"
    echo "----------------------------------------------------------"

    # Launch full mission headless without RViz
    ros2 launch sar_bringup full_mission.launch.py \
      world:="$world" \
      headless:=true \
      rviz:=false &
    LAUNCH_PID=$!

    # Wait for mission to initialize and arm
    sleep 8

    # Trigger mission start
    echo "Triggering /mission/start service..."
    ros2 service call /mission/start std_srvs/srv/Trigger || true

    # Monitor /mission/state until DONE or timeout (e.g. 180s)
    TIMEOUT=180
    ELAPSED=0
    DONE=false

    while [ $ELAPSED -lt $TIMEOUT ]; do
      STATE=$(ros2 topic echo /mission/state --once 2>/dev/null | grep "data:" | awk '{print $2}' | tr -d '"')
      if [ "$STATE" == "DONE" ]; then
        echo "Mission reached state DONE!"
        DONE=true
        sleep 3
        break
      fi
      sleep 2
      ELAPSED=$((ELAPSED + 2))
    done

    if [ "$DONE" = false ]; then
      echo "Warning: Run timed out after ${TIMEOUT}s before reaching DONE."
    fi

    # Kill launch processes
    kill -SIGINT $LAUNCH_PID 2>/dev/null || true
    sleep 4
    # Kill any lingering gz or ros processes
    pkill -f "gz sim" 2>/dev/null || true
    pkill -f "ros_gz_bridge" 2>/dev/null || true
    sleep 2
  done
done

echo "=========================================================="
echo " Experiments completed! Generating plots..."
echo "=========================================================="
python3 "$SCRIPT_DIR/plot_results.py" "$WS_DIR/results/runs.csv"

