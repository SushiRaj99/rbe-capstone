cmake_minimum_required(VERSION 3.8)
project(potr_navigation)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SwitchPlanner.srv"
  "srv/SetParamPreset.srv"
)

ament_export_dependencies(rosidl_default_runtime)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/planner_controller.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME planner_controller
)

ament_package()
