#!/usr/bin/env python3
"""
train.launch.py  —  rl_pipeline package
========================================
Launches the full RL training pipeline on top of an already-running Nav2 /
simulation stack.  Specifically it brings up:

  1. planner_config_manager  (ROS2 node) — bridges the RL env to Nav2 by:
        • publishing /rl/observation and /rl/status
        • subscribing to /rl/action and forwarding denormalized params to
          /controller_server/set_parameters
        • serving /rl/reset to handle per-episode map swaps + pose resets
        • acting as action client to GoalManager (send_goal_to_nav2)

  2. rl_backbone.py  (standalone Python process, NOT a ROS2 node) — runs
        the Stable-Baselines3 PPO training loop via the Nav2GymEnv Gymnasium
        wrapper.  It is launched with ExecuteProcess because it owns its own
        rclpy init/shutdown and argparse CLI.

Pre-requisites (must be running before this launch file is invoked):
  • Nav2 stack  : bt_navigator, controller_server, planner_server,
                  local_costmap, global_costmap, recoveries, velocity_smoother
  • map_server + nav2_lifecycle_manager
  • goal_manager           (simulation_launch package)
  • diff_drive_model       (simulation_launch package)
  • lidar_model            (simulation_launch package)
  • sim_clock              (simulation_launch package)

Example usage:
  ros2 launch rl_pipeline train.launch.py
  ros2 launch rl_pipeline train.launch.py planner:=dwb steps:=1000000
  ros2 launch rl_pipeline train.launch.py log_dir:=/data/rl_logs ckpt_dir:=/data/rl_ckpts
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ---------------------------------------------------------------------------
# Resolve rl_backbone.py path at launch-file parse time.
# The script is expected to be installed to
#   install/rl_pipeline/share/rl_pipeline/scripts/rl_backbone.py
# which is the default location when setup.py data_files includes:
#   (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py'))
# If your setup.py installs scripts elsewhere, adjust this path accordingly.
# ---------------------------------------------------------------------------
_PKG_SHARE = get_package_share_directory('rl_pipeline')
_RL_BACKBONE = os.path.join(_PKG_SHARE, 'scripts', 'rl_backbone.py')


def generate_launch_description():

    # -----------------------------------------------------------------------
    # Launch arguments
    # -----------------------------------------------------------------------

    # -- Shared --
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='dwb',
        description='Local planner type to tune.  Choices: dwb | mppi'
    )

    # -- planner_config_manager node --
    max_steps_arg = DeclareLaunchArgument(
        'max_episode_steps',
        default_value='500',
        description='Maximum timesteps per episode before truncation'
    )

    # -- PPO / training hyper-parameters --
    steps_arg = DeclareLaunchArgument(
        'steps',
        default_value='500000',
        description='Total environment steps for PPO training'
    )
    log_dir_arg = DeclareLaunchArgument(
        'log_dir',
        default_value=os.path.expanduser('~/rl_logs'),
        description='Directory for TensorBoard logs and SB3 Monitor CSVs'
    )
    ckpt_dir_arg = DeclareLaunchArgument(
        'ckpt_dir',
        default_value=os.path.expanduser('~/rl_checkpoints'),
        description='Directory where model checkpoints (.zip) are saved'
    )
    net_arch_arg = DeclareLaunchArgument(
        'net_arch',
        default_value='[256, 256]',
        description='JSON list defining hidden layer widths for PPO actor and critic MLPs'
    )
    lr_arg = DeclareLaunchArgument(
        'lr',
        default_value='3e-4',
        description='PPO learning rate (string form to support scientific notation, e.g. "1e-3")'
    )
    ro_buffer_arg = DeclareLaunchArgument(
        'ro_buffer',
        default_value='2048',
        description='PPO rollout buffer size (n_steps per policy update)'
    )
    batch_size_arg = DeclareLaunchArgument(
        'batch_size',
        default_value='64',
        description='PPO mini-batch size for gradient updates'
    )
    epochs_arg = DeclareLaunchArgument(
        'epochs',
        default_value='10',
        description='Number of PPO optimization epochs per rollout'
    )

    # -----------------------------------------------------------------------
    # 1) planner_config_manager ROS2 node
    # -----------------------------------------------------------------------
    planner_config_manager_node = Node(
        package='rl_pipeline',
        executable='planner_config_manager',   # entry point name from setup.py
        name='planner_config_manager',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'planner_type':        LaunchConfiguration('planner'),
            'max_episode_steps':   LaunchConfiguration('max_episode_steps'),
            # All per-episode params (goal_id, start_x/y/yaw, goal_x/y/yaw,
            # xy_tolerance, yaw_tolerance, map_filepath) are injected
            # dynamically at runtime by RLBridgeNode.call_reset() via the
            # /planner_config_manager/set_parameters service — no need to
            # set static defaults here beyond what the node already declares.
        }],
    )

    # -----------------------------------------------------------------------
    # 2) rl_backbone.py  —  standalone PPO training process
    #    Delayed by 2 s to allow planner_config_manager to finish its own
    #    startup and service/action registrations before the bridge tries
    #    to connect.
    # -----------------------------------------------------------------------
    rl_train_process = ExecuteProcess(
        cmd=[
            sys.executable, _RL_BACKBONE,
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

    # Delay the training process start to give the ROS2 node time to come up
    delayed_train = TimerAction(period=2.0, actions=[rl_train_process])

    # -----------------------------------------------------------------------
    # Assemble LaunchDescription
    # -----------------------------------------------------------------------
    return LaunchDescription([
        # Launch arguments
        planner_arg,
        max_steps_arg,
        steps_arg,
        log_dir_arg,
        ckpt_dir_arg,
        net_arch_arg,
        lr_arg,
        ro_buffer_arg,
        batch_size_arg,
        epochs_arg,
        # Startup banner
        LogInfo(msg=[
            '\n\n========== rl_pipeline | TRAINING MODE ==========\n'
            '  planner        : ', LaunchConfiguration('planner'), '\n'
            '  total steps    : ', LaunchConfiguration('steps'), '\n'
            '  log dir        : ', LaunchConfiguration('log_dir'), '\n'
            '  checkpoint dir : ', LaunchConfiguration('ckpt_dir'), '\n'
            '=================================================\n'
        ]),
        # Nodes / processes
        planner_config_manager_node,
        delayed_train,
    ])
