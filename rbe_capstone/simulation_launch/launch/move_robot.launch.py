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
        default_value='warehouse',
        description='Map name'
    )

    map_yaml = PathJoinSubstitution([
        FindPackageShare('simulation_launch'),
        'maps',
        LaunchConfiguration('map_name'),
        PythonExpression(["'", LaunchConfiguration('map_name'), "' + '.yaml'"]),
    ])

    # Nav2 bringup
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'bringup_launch.py'
            ])
        ]),
        launch_arguments={
            'map': map_yaml,
            'use_sim_time': 'false',
        }.items()
    )

    # Goal point configuration:
    goal_config_path = PathJoinSubstitution([
        FindPackageShare('simulation_launch'),
        'config',
        'goalpoints.yaml',
    ])
    goal_config_arg = DeclareLaunchArgument(
        'goal_config',
        default_value=goal_config_path,
        description='Goal point definition (path to YAML or dictionary of parameters)'
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

        # laser scanner model for perception
        Node(
            package='simulation_launch',
            executable='lidar_model.py',
            name='lidar_model',
            output='screen',
        ),

        # Localization via AMCL
        #Node(
        #    package='nav2_amcl',
        #    executable='amcl',
        #    name='amcl',
        #    output='screen',
        #    parameters=[{
        #        'use_sim_time': False,
        #        'scan_topic': 'scan',
        #        'base_frame_id': 'base_link',
        #        'odom_frame_id': 'odom',
        #        'global_fram_id': 'map',
        #    }],
        #),

        # 'Perfect' Localization
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        ),

        # Differential drive model node to map odom to base_link
        Node(
            package='simulation_launch',
            executable='diff_drive_model.py',
            name='diff_drive_model',
            output='screen',
        ),

        # Nav2 stack (planner, controller, BT, etc.)
        nav2_bringup,

        # Load the map
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'yaml_filename': map_yaml}],
        ),

        # Enable map server by default, because apparantly that's something you have to do now
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{
                'autostart': True,
                'node_names': ['map_server'], #, 'amcl']
            }],
        ),

        # Add Goal Manager to bring up the send_goal_to_nav2 action server
        Node(
            package='simulation_launch',
            executable='goal_manager.py',
            name='goal_manager',
            output='screen',
        ),

        # Add a Goal Client to use the send_goal_to_nav2 action server
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