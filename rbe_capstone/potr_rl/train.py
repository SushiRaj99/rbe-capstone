#!/usr/bin/env python3
import argparse
import os
import time
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env

from potr_rl.env import PotrNavEnv
from potr_rl.callbacks import LivePlotCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action-mode', default='continuous',
                        choices=['discrete', 'continuous'])
    parser.add_argument('--planner', default='MPPI', choices=['MPPI', 'DWB'])
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--action-freq', type=int, default=25,
                        help='Odom ticks per gym step (25=~2.5s, 50=~5s). '
                             'Lower = more steps/hour. Use >=50 for MPPI.')
    parser.add_argument('--save-dir', default='.',
                        help='Directory to save the policy zip and plot')
    parser.add_argument('--plot', default=None,
                        help='Override path for the training plot (default: auto-named alongside policy)')
    parser.add_argument('--check', action='store_true',
                        help='Run env checker before training')
    args = parser.parse_args()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_name  = f'{args.planner.lower()}_{args.action_mode}_{timestamp}'
    save_path = os.path.join(args.save_dir, run_name)
    plot_path = args.plot or os.path.join(args.save_dir, f'{run_name}_metrics.png')

    env = PotrNavEnv(action_mode=args.action_mode, planner=args.planner,
                     action_frequency=args.action_freq)

    if args.check:
        print('Running environment checker...')
        check_env(env, warn=True)
        print('Environment check passed.')

    if args.action_mode == 'discrete':
        model = PPO('MlpPolicy', env, verbose=1, n_steps=128, batch_size=64)
    else:
        model = SAC('MlpPolicy', env, verbose=1, learning_starts=200)

    plot_cb = LivePlotCallback(save_path=plot_path, update_every=8, verbose=1)

    print(f'Training {model.__class__.__name__} '
          f'({args.action_mode}, {args.planner}) '
          f'for {args.timesteps} steps...')
    print(f'Run name : {run_name}')
    print(f'Policy    {save_path}.zip')
    print(f'Plot      {plot_path}')
    model.learn(total_timesteps=args.timesteps, callback=plot_cb)
    model.save(save_path)
    print(f'Model saved  {save_path}.zip')
    env.close()


if __name__ == '__main__':
    main()
