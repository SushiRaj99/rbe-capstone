cmake_minimum_required(VERSION 3.8)
project(simulation_launch)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

ament_export_dependencies(rosidl_default_runtime)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/SendGoalToNav2.action"
)

install(DIRECTORY launch rviz maps config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/diff_drive_model.py
  scripts/lidar_model.py
  scripts/goal_manager.py
  scripts/nav2_goal_client.py
  scripts/sim_clock.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
