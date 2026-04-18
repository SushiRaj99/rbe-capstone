cmake_minimum_required(VERSION 3.8)
project(potr_navigation)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EpisodeMetrics.msg"
  "msg/StepMetrics.msg"
  "srv/SwitchPlanner.srv"
  "srv/SetParamPreset.srv"
  "srv/GetMetrics.srv"
  "srv/SetRawParams.srv"
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

install(PROGRAMS
  scripts/metrics_tracker.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME metrics_tracker
)

install(PROGRAMS
  scripts/episode_runner.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME episode_runner
)

ament_package()
