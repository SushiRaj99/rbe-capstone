cmake_minimum_required(VERSION 3.8)
project(rl_pipeline)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

ament_export_dependencies(rosidl_default_runtime)

install(DESTINATION share/${PROJECT_NAME})

install(PROGRAMS
  scripts/placeholder
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
