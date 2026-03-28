from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
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

    map_yaml = PathJoinSubstitution([
        FindPackageShare('simulation_launch'),
        'maps',
        LaunchConfiguration('map_name'),
        PythonExpression(["'", LaunchConfiguration('map_name'), "' + '.yaml'"]),
    ])

    goal_config_path = PathJoinSubstitution([
        FindPackageShare('simulation_launch'),
        'config',
        'goalpoints.yaml',
    ])
    goal_config_arg = DeclareLaunchArgument(
        'goal_config',
        default_value=goal_config_path,
        description='Goal point definition (path to YAML or dictionary of parameters)',
    )

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
        }.items(),
    )

    return LaunchDescription([
        map_name,
        goal_config_arg,

        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),

        # Simulated laser scanner
        Node(
            package='simulation_launch',
            executable='lidar_model.py',
            name='lidar_model',
            output='screen',
        ),

        # 'Perfect' localization (static map -> odom TF)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom',
            arguments=['-4', '0', '0', '0', '0', '0', 'map', 'odom'],
        ),

        # Differential drive model (odom -> base_link TF + /odom topic)
        Node(
            package='simulation_launch',
            executable='diff_drive_model.py',
            name='diff_drive_model',
            output='screen',
        ),

        # Nav2 stack + planner controller
        potr_nav2,

        # Map server (separate from Nav2's internal one, for lidar_model access)
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_yaml}],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['map_server'],
            }],
        ),

        # Goal manager + client
        Node(
            package='simulation_launch',
            executable='goal_manager.py',
            name='goal_manager',
            output='screen',
        ),
        Node(
            package='simulation_launch',
            executable='nav2_goal_client.py',
            name='nav2_goal_client',
            output='screen',
            parameters=[goal_config_path],
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
