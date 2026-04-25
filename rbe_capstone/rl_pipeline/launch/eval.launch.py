#!/usr/bin/env python3
"""
eval.launch.py  —  rl_pipeline package
=======================================
Fully self-contained evaluation launch.  Includes simulation_bringup.launch.py
to start the entire stack, then launches rl_backbone.py in eval mode at T=15 s.

Evaluation runs two sequential back-to-back passes over the same episode seed:
  1) evaluate_model()    — frozen PPO policy from loaded checkpoint
  2) evaluate_baseline() — Nav2 default YAML params, no RL action published

Complete startup timeline:
  T =  0 s  robot_state_publisher, static map→odom TF, sim_clock
  T =  1 s  diff_drive_model, lidar_model
  T =  2 s  map_server, map_lifecycle_manager,
             controller_server, planner_server, behavior_server,
             bt_navigator, velocity_smoother, nav2_lifecycle_manager
  T =  8 s  goal_manager, planner_config_manager
  T = 15 s  rl_backbone.py  --mode eval  (model eval then baseline eval)

Example usage:
  # Minimal — model path is required:
  ros2 launch rl_pipeline eval.launch.py \\
      model:=~/rl_checkpoints/ppo_nav2_dwb_final.zip

  # More episodes + RViz for visualization:
  ros2 launch rl_pipeline eval.launch.py \\
      model:=~/rl_checkpoints/ppo_nav2_dwb_500k.zip \\
      episodes:=50 eval_seed:=42 use_rviz:=true

  # Override infrastructure:
  ros2 launch rl_pipeline eval.launch.py \\
      model:=~/rl_checkpoints/ppo_nav2_dwb_final.zip \\
      nav2_params_file:=/path/to/my_nav2_params.yaml \\
      initial_map:=/path/to/my_map.yaml
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
from launch.substitutions import LaunchConfiguration

_RL_PKG    = get_package_share_directory('rl_pipeline')
_SIM_PKG   = get_package_share_directory('simulation_launch')

_BRINGUP     = os.path.join(_RL_PKG, 'launch', 'simulation_bringup.launch.py')
#_RL_BACKBONE = os.path.join(_RL_PKG, 'scripts', 'rl_backbone.py')

_DEFAULT_NAV2_PARAMS = os.path.join(_SIM_PKG, 'config', 'nav2_dwb_params.yaml')
_DEFAULT_INITIAL_MAP = os.path.join(_SIM_PKG, 'maps', 'warehouse', 'warehouse.yaml')


def generate_launch_description():

    # -----------------------------------------------------------------------
    # Launch arguments — infrastructure (forwarded to simulation_bringup)
    # -----------------------------------------------------------------------
    planner_arg = DeclareLaunchArgument(
        'planner', default_value='dwb',
        description='Planner the model was trained on: dwb | mppi'
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
        description='Should match the value used during training'
    )
    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='Launch RViz2 for visualization (recommended for eval)'
    )

    # -----------------------------------------------------------------------
    # Launch arguments — evaluation specific
    # -----------------------------------------------------------------------
    model_arg = DeclareLaunchArgument(
        'model', default_value='',
        description=(
            '[REQUIRED] Absolute path to a saved PPO checkpoint (.zip). '
            'rl_backbone.py will exit immediately if this is empty.'
        )
    )
    episodes_arg = DeclareLaunchArgument(
        'episodes', default_value='200',
        description='Number of evaluation episodes (applied to both RL model and baseline runs)'
    )
    eval_seed_arg = DeclareLaunchArgument(
        'eval_seed', default_value='27',
        description=(
            'RNG seed for episode sampling. Using the same seed for both runs '
            'ensures the RL model and baseline see identical episode sequences.'
        )
    )
    results_arg = DeclareLaunchArgument(
        'results', default_value='',
        description=(
            'Optional path to write evaluation results .json (e.g. ./eval_results.json)'
        )
    )

    # -----------------------------------------------------------------------
    # Include simulation_bringup.launch.py
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
    # rl_backbone.py — evaluation process (T = 15 s)
    # -----------------------------------------------------------------------
    rl_eval_process = ExecuteProcess(
        cmd=[
            #sys.executable, _RL_BACKBONE,
            '/opt/.venv/bin/python',
            '-m', 'rl_pipeline.rl_backbone',
            '--mode',       'eval',
            '--planner',    LaunchConfiguration('planner'),
            '--model',      LaunchConfiguration('model'),
            '--episodes',   LaunchConfiguration('episodes'),
            '--eval-seed',  LaunchConfiguration('eval_seed'),
            '--results',    LaunchConfiguration('results'),
        ],
        output='screen',
        emulate_tty=True,
    )

    delayed_eval = TimerAction(period=15.0, actions=[rl_eval_process])

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
        # Evaluation args
        model_arg,
        episodes_arg,
        eval_seed_arg,
        results_arg,
        # Banner
        LogInfo(msg=[
            '\n========== rl_pipeline | EVALUATION MODE ==========\n'
            '  planner  : ', LaunchConfiguration('planner'), '\n'
            '  model    : ', LaunchConfiguration('model'), '\n'
            '  episodes : ', LaunchConfiguration('episodes'), '\n'
            '  seed     : ', LaunchConfiguration('eval_seed'), '\n'
            '  Two sequential eval runs will execute back-to-back:\n'
            '    1) RL model   (deterministic, loaded from checkpoint)\n'
            '    2) Baseline   (Nav2 YAML defaults, no RL)\n'
            '  rl_backbone starts at T+15 s\n'
            '====================================================\n'
        ]),
        # Full stack bringup
        bringup,
        # Evaluation process
        delayed_eval,
    ])
