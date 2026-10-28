cmake_minimum_required(VERSION 3.8)
project(sar_world)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# Install all resource directories
install(DIRECTORY
  worlds
  models
  config
  launch
  dataset
  DESTINATION share/${PROJECT_NAME}
)

# Install Python scripts as executable ROS 2 nodes
install(PROGRAMS
  scripts/capture_dataset.py
  scripts/record_demo.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
