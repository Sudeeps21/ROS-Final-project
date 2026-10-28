cmake_minimum_required(VERSION 3.8)
project(sar_mission)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python package (sar_mission/)
ament_python_install_package(${PROJECT_NAME})

# Install launch, config, scripts
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

# Install executable nodes
install(PROGRAMS
  sar_mission/mission_node.py
  sar_mission/planner.py
  test/fake_drone.py
  scripts/plot_path.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
