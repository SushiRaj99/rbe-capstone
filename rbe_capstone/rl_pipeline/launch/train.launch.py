#!/usr/bin/env python3
"""
train.launch.py  —  rl_pipeline package
========================================
Fully self-contained training launch.  Includes simulation_bringup.launch.py
to start the entire stack (robot description, simulation physics, Nav2, and
ROS2 bridge nodes), then launches rl_backbone.py as a standalone PPO training
process at T=15 s once everything is confirmed active.

Complete startup timeline:
  T =  0 s  robot_state_publisher, static map→odom TF, sim_clock
  T =  1 s  diff_drive_model, lidar_model
  T =  2 s  map_server, map_lifecycle_manager,
             controller_server, planner_server, behavior_server,
             bt_navigator, velocity_smoother, nav2_lifecycle_manager
  T =  8 s  goal_manager, planner_config_manager
  T = 15 s  rl_backbone.py  --mode train  (PPO training loop)

Example usage:
  # Defaults (500k steps, dwb planner):
  ros2 launch rl_pipeline train.launch.py

  # Custom hyperparameters:
  ros2 launch rl_pipeline train.launch.py \\
      steps:=1000000 lr:=1e-4 net_arch:='[512,512]' \\
      log_dir:=/data/rl_logs ckpt_dir:=/data/rl_ckpts

  # Override map and Nav2 params:
  ros2 launch rl_pipeline train.launch.py \\
      initial_map:=/path/to/my_map.yaml \\
      nav2_params_file:=/path/to/my_nav2_params.yaml

  # Enable RViz (not recommended during training — adds overhead):
  ros2 launch rl_pipeline train.launch.py use_rviz:=true
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription,
    LogInfo, TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

_RL_PKG    = get_package_share_directory('rl_pipeline')
_SIM_PKG   = get_package_share_directory('simulation_launch')

_BRINGUP   = os.path.join(_RL_PKG, 'launch', 'simulation_bringup.launch.py')
#_RL_BACKBONE = os.path.join(_RL_PKG, 'scripts', 'rl_backbone.py')

_DEFAULT_NAV2_PARAMS = os.path.join(_SIM_PKG, 'config', 'nav2_dwb_params.yaml')
_DEFAULT_INITIAL_MAP = os.path.join(_SIM_PKG, 'maps', 'warehouse', 'warehouse.yaml')


def generate_launch_description():

    # -----------------------------------------------------------------------
    # Launch arguments — infrastructure (forwarded to simulation_bringup)
    # -----------------------------------------------------------------------
    planner_arg = DeclareLaunchArgument(
        'planner', default_value='dwb',
        description='Local planner to tune: dwb | mppi'
    )
    nav2_params_arg = DeclareLaunchArgument(
        'nav2_params_file', default_value=_DEFAULT_NAV2_PARAMS,
        description='Absolute path to Nav2 params YAML'
    )
    initial_map_arg = DeclareLaunchArgument(
        'initial_map', default_value=_DEFAULT_INITIAL_MAP,
        description='Absolute path to initial map .yaml (RL swaps maps per-episode)'
    )
    max_episode_steps_arg = DeclareLaunchArgument(
        'max_episode_steps', default_value='2000',
        description='Maximum timesteps per episode before truncation'
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='Launch RViz2 (not recommended during training)'
    )

    # -----------------------------------------------------------------------
    # Launch arguments — PPO training hyperparameters
    # -----------------------------------------------------------------------
    steps_arg = DeclareLaunchArgument(
        'steps', default_value='20000',
        description='Total environment steps for PPO training'
    )
    log_dir_arg = DeclareLaunchArgument(
        'log_dir', default_value=os.path.expanduser('~/ws/src/rbe_capstone/rl_pipeline/rl_logs'),
        description='Directory for TensorBoard logs and SB3 Monitor CSVs'
    )
    ckpt_dir_arg = DeclareLaunchArgument(
        'ckpt_dir', default_value=os.path.expanduser('~/ws/src/rbe_capstone/rl_pipeline/rl_checkpoints'),
        description='Directory where model checkpoints (.zip) are saved'
    )
    net_arch_arg = DeclareLaunchArgument(
        'net_arch', default_value='[512, 512]',
        description='JSON list of hidden layer widths for PPO actor and critic MLPs'
    )
    lr_arg = DeclareLaunchArgument(
        'lr', default_value='1e-4',
        description='PPO learning rate (scientific notation supported, e.g. "1e-4")'
    )
    ro_buffer_arg = DeclareLaunchArgument(
        'ro_buffer', default_value='256',
        description='PPO rollout buffer size (n_steps per policy update)'
    )
    batch_size_arg = DeclareLaunchArgument(
        'batch_size', default_value='16',
        description='PPO mini-batch size for gradient updates'
    )
    epochs_arg = DeclareLaunchArgument(
        'epochs', default_value='10',
        description='PPO optimization epochs per rollout'
    )

    # -----------------------------------------------------------------------
    # Include simulation_bringup.launch.py — starts everything up to and
    # including planner_config_manager at T=8 s.
    # -----------------------------------------------------------------------
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(_BRINGUP),
        launch_arguments={
            'planner':           LaunchConfiguration('planner'),
            'nav2_params_file':  LaunchConfiguration('nav2_params_file'),
            'initial_map':       LaunchConfiguration('initial_map'),
            'max_episode_steps': LaunchConfiguration('max_episode_steps'),
            'use_rviz':          LaunchConfiguration('use_rviz'),
        }.items()
    )

    # -----------------------------------------------------------------------
    # rl_backbone.py — standalone PPO training process (T = 15 s)
    #
    # Why T=15 s?  simulation_bringup layers complete at T=8 s.  An extra
    # 7 s gives planner_config_manager time to register its /rl/reset
    # service and for goal_manager to finish its wait_for_server() call
    # against bt_navigator.  Reduce to ~12 s on fast machines once you've
    # confirmed the stack comes up reliably.
    # -----------------------------------------------------------------------
    rl_train_process = ExecuteProcess(
        cmd=[
            #sys.executable, _RL_BACKBONE,
            '/opt/.venv/bin/python',
            '-m', 'rl_pipeline.rl_backbone',
            '--mode',       'train',
            '--planner',    LaunchConfiguration('planner'),
            '--steps',      LaunchConfiguration('steps'),
            '--log-dir',    LaunchConfiguration('log_dir'),
            '--ckpt-dir',   LaunchConfiguration('ckpt_dir'),
            '--net-arch',   LaunchConfiguration('net_arch'),
            '--lr',         LaunchConfiguration('lr'),
            '--ro-buffer',  LaunchConfiguration('ro_buffer'),
            '--batch-size', LaunchConfiguration('batch_size'),
            '--epochs',     LaunchConfiguration('epochs'),
        ],
        output='screen',
        emulate_tty=True,
    )

    delayed_train = TimerAction(period=15.0, actions=[rl_train_process])

    # -----------------------------------------------------------------------
    # Assemble
    # -----------------------------------------------------------------------
    return LaunchDescription([
        # Infrastructure args
        planner_arg,
        nav2_params_arg,
        initial_map_arg,
        max_episode_steps_arg,
        use_rviz_arg,
        # Training args
        steps_arg,
        log_dir_arg,
        ckpt_dir_arg,
        net_arch_arg,
        lr_arg,
        ro_buffer_arg,
        batch_size_arg,
        epochs_arg,
        # Banner
        LogInfo(msg=[
            '\n========== rl_pipeline | TRAINING MODE ==========\n'
            '  planner   : ', LaunchConfiguration('planner'), '\n'
            '  steps     : ', LaunchConfiguration('steps'), '\n'
            '  lr        : ', LaunchConfiguration('lr'), '\n'
            '  log_dir   : ', LaunchConfiguration('log_dir'), '\n'
            '  ckpt_dir  : ', LaunchConfiguration('ckpt_dir'), '\n'
            '  rl_backbone starts at T+15 s\n'
            '=================================================\n'
        ]),
        # Full stack bringup
        bringup,
        # Training process
        delayed_train,
    ])
