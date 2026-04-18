#!/usr/bin/env python3
import argparse
import os
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env

from potr_rl.env import PotrNavEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action-mode', default='continuous',
                        choices=['discrete', 'continuous'])
    parser.add_argument('--planner', default='MPPI', choices=['MPPI', 'DWB'])
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--save', default='potr_policy')
    parser.add_argument('--check', action='store_true',
                        help='Run env checker before training')
    args = parser.parse_args()

    env = PotrNavEnv(action_mode=args.action_mode, planner=args.planner)

    if args.check:
        print('Running environment checker...')
        check_env(env, warn=True)
        print('Environment check passed.')

    # PPO works for both discrete and continuous; SAC is continuous-only.
    # Each step() call covers action_frequency odom ticks (~5s at 50 ticks/10Hz).
    # A 60s episode yields ~12 steps, so n_steps=128 covers ~10 episodes per
    # PPO update — enough for a stable gradient estimate.
    if args.action_mode == 'discrete':
        model = PPO('MlpPolicy', env, verbose=1, n_steps=128, batch_size=64)
    else:
        # learning_starts: steps of random exploration before first update.
        # At 50 ticks/step, 20 steps ≈ 2 full episodes of random exploration.
        model = SAC('MlpPolicy', env, verbose=1, learning_starts=20)

    print(f'Training {model.__class__.__name__} '
          f'({args.action_mode}, {args.planner}) '
          f'for {args.timesteps} steps...')
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save)
    print(f'Model saved to {args.save}.zip')
    env.close()


if __name__ == '__main__':
    main()
