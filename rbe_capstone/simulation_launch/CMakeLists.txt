cmake_minimum_required(VERSION 3.8)
project(simulation_launch)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch rviz maps
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
