#!/usr/bin/env python3
"""
eval.launch.py  —  rl_pipeline package
=======================================
Launches the RL evaluation pipeline on top of an already-running Nav2 /
simulation stack.  Specifically it brings up:

  1. planner_config_manager  (ROS2 node) — same bridge node used during
        training; kept alive here so the frozen policy can still interact
        with Nav2 for deterministic rollouts.

  2. rl_backbone.py  (standalone Python process) — runs in '--mode eval',
        which executes two sequential evaluations:
          a) evaluate_model()    — runs the loaded PPO checkpoint
             deterministically for --episodes episodes.
          b) evaluate_baseline() — runs the same episodes using the Nav2
             default parameters (no RL involvement) for a side-by-side
             comparison.
        Both sets of results are printed to stdout and, if --results is
        supplied, merged into a single JSON file.

        NOTE: --results is a planned argument that is not yet declared in
        rl_backbone.py's build_parser() (args.results will raise
        AttributeError).  Add the following line to build_parser() before
        using that flag:
          p.add_argument('--results', type=str, default=None,
                         help='Path to JSON file for saving eval results')

Pre-requisites (must be running before this launch file is invoked):
  • Nav2 stack  : bt_navigator, controller_server, planner_server,
                  local_costmap, global_costmap, recoveries, velocity_smoother
  • map_server + nav2_lifecycle_manager
  • goal_manager           (simulation_launch package)
  • diff_drive_model       (simulation_launch package)
  • lidar_model            (simulation_launch package)
  • sim_clock              (simulation_launch package)

Example usage:
  # Evaluate a specific checkpoint (model arg is required for eval mode):
  ros2 launch rl_pipeline eval.launch.py \\
      model:=~/rl_checkpoints/ppo_nav2_dwb_final.zip

  # Evaluate with more episodes and save results to JSON:
  ros2 launch rl_pipeline eval.launch.py \\
      model:=~/rl_checkpoints/ppo_nav2_dwb_final.zip \\
      episodes:=50 \\
      eval_seed:=42 \\
      results:=~/eval_results.json

  # Evaluate an MPPI checkpoint:
  ros2 launch rl_pipeline eval.launch.py \\
      planner:=mppi \\
      model:=~/rl_checkpoints/ppo_nav2_mppi_final.zip
"""

import os
import sys

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ---------------------------------------------------------------------------
# Resolve rl_backbone.py path — mirrors the logic in train.launch.py.
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
        description='Local planner type that the model was trained on.  Choices: dwb | mppi'
    )

    # -- planner_config_manager node --
    max_steps_arg = DeclareLaunchArgument(
        'max_episode_steps',
        default_value='500',
        description='Maximum timesteps per episode before truncation (should match training value)'
    )

    # -- Evaluation-specific --
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='',
        description=(
            '[REQUIRED] Absolute path to a saved PPO model checkpoint (.zip). '
            'rl_backbone.py will error out at startup if this is empty.'
        )
    )
    episodes_arg = DeclareLaunchArgument(
        'episodes',
        default_value='10',
        description='Number of evaluation episodes to run (applies to both RL model and baseline)'
    )
    eval_seed_arg = DeclareLaunchArgument(
        'eval_seed',
        default_value='28',
        description=(
            'RNG seed for episode configuration sampling.  Using the same seed '
            'ensures the RL model and baseline see identical episode sequences.'
        )
    )
    results_arg = DeclareLaunchArgument(
        'results',
        default_value='',
        description=(
            'Optional path to a .json file where evaluation results will be saved. '
            'Requires --results to be added to build_parser() in rl_backbone.py first '
            '(see module docstring in this file).'
        )
    )

    # -----------------------------------------------------------------------
    # 1) planner_config_manager ROS2 node
    # -----------------------------------------------------------------------
    planner_config_manager_node = Node(
        package='rl_pipeline',
        executable='planner_config_manager',
        name='planner_config_manager',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'planner_type':      LaunchConfiguration('planner'),
            'max_episode_steps': LaunchConfiguration('max_episode_steps'),
        }],
    )

    # -----------------------------------------------------------------------
    # 2) rl_backbone.py  —  standalone evaluation process
    #    Conditionally appends --results only when the arg is non-empty.
    #    Because ROS2 launch substitutions can't do conditional list building
    #    neatly, we build the base command and rely on the empty-string default
    #    of args.results in rl_backbone.py (once that arg is added to the parser)
    #    to be a no-op when results saving is not requested.
    # -----------------------------------------------------------------------
    rl_eval_process = ExecuteProcess(
        cmd=[
            sys.executable, _RL_BACKBONE,
            '--mode',       'eval',
            '--planner',    LaunchConfiguration('planner'),
            '--model',      LaunchConfiguration('model'),
            '--episodes',   LaunchConfiguration('episodes'),
            '--eval-seed',  LaunchConfiguration('eval_seed'),
            # '--results',  LaunchConfiguration('results'),
            # ^ Uncomment the line above after adding '--results' to
            #   build_parser() in rl_backbone.py.
        ],
        output='screen',
        emulate_tty=True,
    )

    # Allow planner_config_manager to register its services before the eval
    # process attempts to connect.
    delayed_eval = TimerAction(period=2.0, actions=[rl_eval_process])

    # -----------------------------------------------------------------------
    # Assemble LaunchDescription
    # -----------------------------------------------------------------------
    return LaunchDescription([
        # Launch arguments
        planner_arg,
        max_steps_arg,
        model_arg,
        episodes_arg,
        eval_seed_arg,
        results_arg,
        # Startup banner
        LogInfo(msg=[
            '\n\n========== rl_pipeline | EVALUATION MODE ==========\n'
            '  planner  : ', LaunchConfiguration('planner'), '\n'
            '  model    : ', LaunchConfiguration('model'), '\n'
            '  episodes : ', LaunchConfiguration('episodes'), '\n'
            '  seed     : ', LaunchConfiguration('eval_seed'), '\n'
            '====================================================\n'
            '  Two evaluations will run back-to-back:\n'
            '    1) RL model  (deterministic policy from loaded checkpoint)\n'
            '    2) Baseline  (Nav2 default params, no RL)\n'
            '====================================================\n'
        ]),
        # Nodes / processes
        planner_config_manager_node,
        delayed_eval,
    ])
