from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Robot description
    robot_description = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([
            FindPackageShare('turtlebot4_description'),
            'urdf', 'standard', 'turtlebot4.urdf.xacro',
        ])
    ])

    rviz_config = PathJoinSubstitution([
        FindPackageShare('simulation_launch'),
        'rviz', 'tb4_with_lasers.rviz',
    ])

    map_name = DeclareLaunchArgument(
        'map_name',
        default_value='mixed',
        description='Map name',
    )

    speed_factor_arg = DeclareLaunchArgument(
        'speed_factor',
        default_value='1.0',
        description='Simulation speed multiplier (e.g. 3.0 = 3x real time)',
    )

    map_yaml = PathJoinSubstitution([
        FindPackageShare('simulation_launch'),
        'maps',
        LaunchConfiguration('map_name'),
        PythonExpression(["'", LaunchConfiguration('map_name'), "' + '.yaml'"]),
    ])

    goal_config_arg = DeclareLaunchArgument(
        'goal_config',
        description='Goal config name (without .yaml) in simulation_launch/config/',
    )
    goal_config = PathJoinSubstitution([
        FindPackageShare('simulation_launch'),
        'config',
        PythonExpression(["'", LaunchConfiguration('goal_config'), "' + '.yaml'"]),
    ])

    use_sim_time = {'use_sim_time': True}

    # Nav2 stack + planner controller from potr_navigation
    potr_nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('potr_navigation'),
                'launch',
                'nav2_bringup.launch.py',
            ])
        ]),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': 'true',
        }.items(),
    )

    return LaunchDescription([
        map_name,
        speed_factor_arg,
        goal_config_arg,

        # Simulation clock (drives use_sim_time for all nodes)
        Node(
            package='simulation_launch',
            executable='sim_clock.py',
            name='sim_clock',
            output='screen',
            parameters=[{'speed_factor': LaunchConfiguration('speed_factor')}],
        ),

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}, use_sim_time],
        ),

        # Simulated laser scanner
        Node(
            package='simulation_launch',
            executable='lidar_model.py',
            name='lidar_model',
            output='screen',
            parameters=[use_sim_time],
        ),

        # 'Perfect' localization (static map -> odom TF)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom',
            arguments=['-4', '0', '0', '0', '0', '0', 'map', 'odom'],
            parameters=[use_sim_time],
        ),

        # Differential drive model (odom -> base_link TF + /odom topic)
        Node(
            package='simulation_launch',
            executable='diff_drive_model.py',
            name='diff_drive_model',
            output='screen',
            parameters=[use_sim_time],
        ),

        # Nav2 stack + planner controller
        potr_nav2,

        # Map server (separate from Nav2's internal one, for lidar_model access)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_yaml}, use_sim_time],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['map_server'],
            }, use_sim_time],
        ),

        # Episode runner (goal manager + metrics lifecycle + run orchestration)
        Node(
            package='potr_navigation',
            executable='episode_runner',
            name='episode_runner',
            output='screen',
            parameters=[ParameterFile(goal_config, allow_substs=True), use_sim_time],
        ),

        # Metrics tracker
        Node(
            package='potr_navigation',
            executable='metrics_tracker',
            name='metrics_tracker',
            output='screen',
            parameters=[use_sim_time],
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
            parameters=[use_sim_time],
        ),
    ])
