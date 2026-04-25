#!/usr/bin/env python3
"""
simulation_bringup.launch.py  —  rl_pipeline package
=====================================================
Shared infrastructure launch file included by both train.launch.py and
eval.launch.py.  Starts the entire simulation stack from scratch so that
nothing needs to be running beforehand.

Startup sequence (all timers are relative to launch start):

  T = 0 s   Layer 1 — Robot description (robot_state_publisher),
                       static map→odom TF broadcaster, sim_clock

  T = 1 s   Layer 2 — Simulation physics: diff_drive_model, lidar_model

  T = 2 s   Layer 3 — Nav2 map server  +  its dedicated lifecycle manager
                       (kept separate so /map_server/load_map stays available
                       independently of the navigation lifecycle state)

  T = 2 s   Layer 4 — Nav2 navigation stack (controller_server, planner_server,
                       behavior_server, bt_navigator, velocity_smoother)
                       + its lifecycle manager (autostart=True)

  T = 8 s   Layer 5 — goal_manager  +  planner_config_manager
                       (Nav2 lifecycle activation takes ~3-5 s; 8 s gives a
                       comfortable margin before these nodes try to connect)

The rl_backbone.py process is NOT started here — it is started by the
calling file (train.launch.py / eval.launch.py) at T=15 s.

-----------------------------------------------------------------------
FILE/PATH ASSUMPTIONS — verify before first use:
-----------------------------------------------------------------------
  simulation_launch/config/nav2_params.yaml
    Required Nav2 parameter file.  At minimum it must declare these params
    so that planner_config_manager can call /controller_server/set_parameters:

        controller_server:
          ros__parameters:
            controller_plugins: ["FollowPath"]
            FollowPath:
              plugin: "dwb_core::DWBLocalPlanner"
              max_vel_x:          0.26
              min_speed_xy:       0.0
              GoalAlign.scale:    24.0
              PathAlign.scale:    32.0
              GoalDist.scale:     24.0
              PathDist.scale:     32.0
              # ... other required DWB / costmap params

  simulation_launch/rviz/nav2_view.rviz
    RViz2 config file.  Only loaded when use_rviz:=true.
    Rename the path constant _DEFAULT_RVIZ_CFG below if yours differs.

  turtlebot4_description  (ROS2 package)
    Must be installed.  Provides the URDF via:
      urdf/standard/turtlebot4.urdf.xacro
    If your URDF lives elsewhere, update the robot_description_content
    substitution in Layer 1.

  map→odom static TF
    A static identity transform is broadcast here.  This is valid when
    diff_drive_model resets its internal odometry to match each new start
    pose on /simulation/set_pose.  If diff_drive_model accumulates odom
    across resets (i.e. odom≠map after the first reset), you will need to
    have diff_drive_model publish a dynamic map→odom TF and remove the
    static_transform_publisher node below to avoid TF conflicts.

  executable names in simulation_launch/setup.py
    The Node() calls for diff_drive_model, lidar_model, sim_clock, and
    goal_manager assume these entry-point names in simulation_launch's
    setup.py console_scripts:
        diff_drive_model = simulation_launch.scripts.diff_drive_model:main
        lidar_model      = simulation_launch.scripts.lidar_model:main
        sim_clock        = simulation_launch.scripts.sim_clock:main
        goal_manager     = simulation_launch.scripts.goal_manager:main
    Adjust the executable= arguments below if yours differ.
"""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo, TimerAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import (
    Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

# ---------------------------------------------------------------------------
# Resolved at parse time — cheaper than substitutions for paths that don't
# depend on launch args.
# ---------------------------------------------------------------------------
_SIM_PKG  = get_package_share_directory('simulation_launch')
_RL_PKG   = get_package_share_directory('rl_pipeline')

_DEFAULT_NAV2_PARAMS = os.path.join(_SIM_PKG, 'config', 'nav2_dwb_params.yaml')
_DEFAULT_INITIAL_MAP = os.path.join(_SIM_PKG, 'maps', 'warehouse', 'warehouse.yaml')
_DEFAULT_RVIZ_CFG    = os.path.join(_SIM_PKG, 'rviz', 'mac_is_a_pain_in_the_ass.rviz')


def generate_launch_description():

    # -----------------------------------------------------------------------
    # Launch arguments
    # -----------------------------------------------------------------------
    planner_arg = DeclareLaunchArgument(
        'planner', default_value='dwb',
        description='Local planner to tune: dwb | mppi'
    )
    nav2_params_arg = DeclareLaunchArgument(
        'nav2_params_file', default_value=_DEFAULT_NAV2_PARAMS,
        description='Absolute path to the Nav2 params YAML file'
    )
    initial_map_arg = DeclareLaunchArgument(
        'initial_map', default_value=_DEFAULT_INITIAL_MAP,
        description=(
            'Absolute path to the initial .yaml map loaded at startup. '
            'The RL pipeline hot-swaps maps at episode boundaries via '
            '/map_server/load_map, so any valid map works here.'
        )
    )
    max_episode_steps_arg = DeclareLaunchArgument(
        'max_episode_steps', default_value='500',
        description='Maximum timesteps per episode before truncation'
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description=(
            'Launch RViz2 for visualization. '
            'Recommended for eval; leave false during training to reduce overhead.'
        )
    )
    speed_factor_arg = DeclareLaunchArgument(
        'speed_factor',
        default_value='1.0',
        description='Simulation speed multiplier (e.g. 3.0 = 3x real time)',
    )

    # -----------------------------------------------------------------------
    # Layer 1  (T = 0 s)
    # Robot state publisher, static map->odom TF, sim clock
    # -----------------------------------------------------------------------

    # TurtleBot4 URDF via xacro.  Command() is evaluated lazily at launch
    # time (not at parse time), so missing xacro is caught with a clean error.
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([
            FindPackageShare('turtlebot4_description'),
            'urdf', 'standard', 'turtlebot4.urdf.xacro',
        ])
    ])

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'robot_description': robot_description_content,
        }]
    )

    # Static identity transform: map -> odom.
    # In this simulation, odom and map are treated as coincident frames
    # (perfect odometry assumption).  diff_drive_model is expected to
    # publish the odom -> base_link transform.
    # See module docstring for caveats on multi-episode pose resets.
    static_map_odom_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_to_odom',
        # args: x y z  roll pitch yaw  parent_frame  child_frame
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': True}]
    )

    sim_clock_node = Node(
        package='simulation_launch',
        executable='sim_clock.py',
        name='sim_clock',
        output='screen',
        parameters=[{'speed_factor': LaunchConfiguration('speed_factor')}]
    )

    layer_1 = GroupAction([
        sim_clock_node,
        robot_state_publisher_node,
        static_map_odom_tf_node,
    ])

    # -----------------------------------------------------------------------
    # Layer 2  (T = 1 s)
    # Simulation physics nodes — depend on sim_clock being up first
    # -----------------------------------------------------------------------

    diff_drive_model_node = Node(
        package='simulation_launch',
        executable='diff_drive_model.py',
        name='diff_drive_model',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    lidar_model_node = Node(
        package='simulation_launch',
        executable='lidar_model.py',
        name='lidar_model',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    layer_2 = TimerAction(period=1.0, actions=[
        diff_drive_model_node,
        lidar_model_node,
    ])

    # -----------------------------------------------------------------------
    # Layer 3  (T = 2 s)
    # Nav2 map server + its own dedicated lifecycle manager.
    #
    # map_server is kept in a separate lifecycle manager from the navigation
    # stack so that /map_server/load_map (called per-episode by
    # planner_config_manager) remains independently accessible regardless
    # of whether the navigation lifecycle is mid-transition.
    # -----------------------------------------------------------------------

    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            LaunchConfiguration('nav2_params_file'),
            {
                'use_sim_time':   True,
                'yaml_filename':  LaunchConfiguration('initial_map'),
            }
        ]
    )

    map_lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart':    True,
            'node_names':   ['map_server'],
        }]
    )

    layer_3 = TimerAction(period=4.0, actions=[
        map_server_node,
        map_lifecycle_manager_node,
    ])

    # -----------------------------------------------------------------------
    # Layer 4  (T = 2 s)
    # Nav2 navigation stack + its lifecycle manager.
    #
    # All nodes share nav2_params_file.  The lifecycle manager uses
    # autostart=True so it configures and activates each node as soon
    # as that node's lifecycle services become available — no manual
    # activation step required.
    #
    # velocity_smoother is included because Nav2 Jazzy's default BT publishes
    # to /cmd_vel_smoothed (which velocity_smoother re-publishes as /cmd_vel).
    # If your nav2_params.yaml routes /cmd_vel directly from bt_navigator,
    # remove velocity_smoother from this layer and from node_names below.
    # -----------------------------------------------------------------------

    """controller_server_node = Node(
        package='nav2_controller',
        executable='controller_server',
        name='controller_server',
        output='screen',
        parameters=[
            LaunchConfiguration('nav2_params_file'),
            {'use_sim_time': True}
        ]
    )

    planner_server_node = Node(
        package='nav2_planner',
        executable='planner_server',
        name='planner_server',
        output='screen',
        parameters=[
            LaunchConfiguration('nav2_params_file'),
            {'use_sim_time': True}
        ]
    )

    behavior_server_node = Node(
        package='nav2_behaviors',
        executable='behavior_server',
        name='behavior_server',
        output='screen',
        parameters=[
            LaunchConfiguration('nav2_params_file'),
            {'use_sim_time': True}
        ]
    )

    bt_navigator_node = Node(
        package='nav2_bt_navigator',
        executable='bt_navigator',
        name='bt_navigator',
        output='screen',
        parameters=[
            LaunchConfiguration('nav2_params_file'),
            {'use_sim_time': True}
        ]
    )

    velocity_smoother_node = Node(
        package='nav2_velocity_smoother',
        executable='velocity_smoother',
        name='velocity_smoother',
        output='screen',
        parameters=[
            LaunchConfiguration('nav2_params_file'),
            {'use_sim_time': True}
        ]
    )

    nav2_lifecycle_manager_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'autostart':    True,
            # Activation order matters: costmap servers before their consumers.
            # controller_server owns local_costmap; planner_server owns global_costmap.
            'node_names': [
                'controller_server',
                'planner_server',
                'behavior_server',
                'bt_navigator',
                'velocity_smoother',
            ],
        }]
    )


    layer_4 = TimerAction(period=2.0, actions=[
        controller_server_node,
        planner_server_node,
        behavior_server_node,
        bt_navigator_node,
        velocity_smoother_node,
        nav2_lifecycle_manager_node,
    ])"""
    layer_4 = TimerAction(period=4.0, actions=[
    IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'), 'launch', 'navigation_launch.py'
            ])
        ]),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file':  LaunchConfiguration('nav2_params_file'),
        }.items(),
    )
])

    # -----------------------------------------------------------------------
    # Layer 5  (T = 8 s)
    # GoalManager + PlannerConfigManager
    #
    # Why T=8 s?  Nav2 lifecycle activation starts at T=2 s.  Activating
    # 5 nodes in sequence takes ~3-5 s, so all nodes are ACTIVE by ~T=7 s.
    # GoalManager.__init__ calls client.wait_for_server() (blocking, no
    # timeout) on bt_navigator's NavigateToPose action server — the 1-second
    # buffer past expected activation prevents it from blocking indefinitely
    # if Nav2 is slightly slow on a loaded machine.
    # -----------------------------------------------------------------------

    goal_manager_node = Node(
        package='simulation_launch',
        executable='goal_manager.py',
        name='goal_manager',
        output='screen',
        emulate_tty=True,
        parameters=[{'use_sim_time': True}]
    )

    planner_config_manager_node = Node(
        package='rl_pipeline',
        executable='planner_config_manager',
        name='planner_config_manager',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'use_sim_time':      True,
            'planner_type':      LaunchConfiguration('planner'),
            'max_episode_steps': LaunchConfiguration('max_episode_steps'),
        }]
    )

    layer_5 = TimerAction(period=8.0, actions=[
        goal_manager_node,
        planner_config_manager_node,
    ])

    # -----------------------------------------------------------------------
    # Optional RViz2  (enabled with use_rviz:=true)
    # -----------------------------------------------------------------------

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', _DEFAULT_RVIZ_CFG],
        parameters=[{'use_sim_time': True}],
        condition=IfCondition(LaunchConfiguration('use_rviz'))
    )

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    return LaunchDescription([
        planner_arg,
        nav2_params_arg,
        initial_map_arg,
        max_episode_steps_arg,
        use_rviz_arg,
        speed_factor_arg,
        LogInfo(msg='[simulation_bringup] Starting Layer 1: robot description, static TF, sim clock'),
        layer_1,
        LogInfo(msg='[simulation_bringup] Starting Layer 2 (T+1s): diff_drive_model, lidar_model'),
        layer_2,
        LogInfo(msg='[simulation_bringup] Starting Layer 3+4 (T+2s): map_server, Nav2 navigation stack'),
        layer_3,
        layer_4,
        LogInfo(msg='[simulation_bringup] Starting Layer 5 (T+8s): goal_manager, planner_config_manager'),
        layer_5,
        rviz_node,
    ])
