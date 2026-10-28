cmake_minimum_required(VERSION 3.8)
project(sar_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  sar_perception/detector_node.py
  test/image_folder_publisher.py
  scripts/evaluate_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
