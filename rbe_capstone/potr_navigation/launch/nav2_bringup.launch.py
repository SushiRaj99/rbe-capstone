from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = PathJoinSubstitution([
        FindPackageShare('potr_navigation'),
        'config',
        'nav2_params.yaml',
    ])

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('nav2_bringup'),
                'launch',
                'navigation_launch.py',
            ])
        ]),
        launch_arguments={
            'use_sim_time': 'false',
            'params_file': params_file,
        }.items(),
    )

    planner_controller = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='potr_navigation',
                executable='planner_controller',
                name='planner_controller',
                output='screen',
                respawn=True,
                respawn_delay=2.0,
            ),
        ]
    )

    return LaunchDescription([
        nav2_bringup,
        planner_controller,
    ])