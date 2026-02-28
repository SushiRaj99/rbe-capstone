from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Process the TurtleBot4 xacro URDF into a robot_description string
    robot_description = Command([
        FindExecutable(name='xacro'), ' ',
        PathJoinSubstitution([
            FindPackageShare('turtlebot4_description'),
            'urdf', 'standard', 'turtlebot4.urdf.xacro',
        ])
    ])

    rviz_config = PathJoinSubstitution([
        FindPackageShare('simulation_launch'),
        'rviz', 'turtlebot4.rviz',
    ])

    map_name = DeclareLaunchArgument(
        'map_name',
        default_value='warehouse',
        description='Name of the map to load (must exist as maps/<name>/<name>.yaml)',
    )

    map_yaml = PathJoinSubstitution([
        FindPackageShare('simulation_launch'),
        'maps',
        LaunchConfiguration('map_name'),
        PythonExpression(["'", LaunchConfiguration('map_name'), "' + '.yaml'"]),
    ])

    return LaunchDescription([
        map_name,

        # Publishes /tf and /robot_description from the URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description}],
        ),
        # Static TF (TEMPORARY) - TODO, make an actual odometry source and drop this
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        ),
        # Static TF (TEMPORARY) - TODO, make an actual odometry source and drop this
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_base_link',
            arguments=['0', '0', '0', '0', '0', '0', 'odom', 'base_link'],
        ),

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
                'node_names': ['map_server'],
            }],
        ),
        # RViz2 with our config
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config],
        ),
    ])
