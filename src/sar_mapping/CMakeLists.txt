cmake_minimum_required(VERSION 3.8)
project(sar_mapping)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY launch config rviz DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  sar_mapping/geolocator_node.py
  sar_mapping/evaluator_node.py
  scripts/plot_results.py
  test/fake_inputs.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

