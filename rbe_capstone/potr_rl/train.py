#!/usr/bin/env python3
import argparse
import os
import time
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_checker import check_env

from potr_rl.env import PotrNavEnv
from potr_rl.callbacks import LivePlotCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action-mode', default='continuous',
                        choices=['discrete', 'continuous'])
    parser.add_argument('--planner', default='MPPI', choices=['MPPI', 'DWB'])
    parser.add_argument('--timesteps', type=int, default=100_000)
    parser.add_argument('--action-freq', type=int, default=10,
                        help='Odom ticks per gym step (10=~1s, 25=~2.5s, 50=~5s). '
                             'Lower = more decisions per episode. Use >=50 for MPPI.')
    parser.add_argument('--save-dir', default='.',
                        help='Directory to save the policy zip and plot')
    parser.add_argument('--plot', default=None,
                        help='Override path for the training plot (default: auto-named alongside policy)')
    parser.add_argument('--check', action='store_true',
                        help='Run env checker before training')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint .zip to resume from. If a '
                             '<stem>.replay.pkl sits next to it the replay '
                             'buffer is loaded too; otherwise SAC starts with '
                             'an empty buffer and re-does warmup.')
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
        if args.resume:
            model = PPO.load(args.resume, env=env)
            print(f'Resumed PPO from {args.resume}')
        else:
            model = PPO('MlpPolicy', env, verbose=1, n_steps=128, batch_size=64)
    else:
        # target_entropy at 1/4 of SB3's default (-n_act) keeps exploration
        # alive much longer — helps the policy find deltas from preset 1 baseline.
        n_act = env.action_space.shape[0]
        if args.resume:
            model = SAC.load(args.resume, env=env)
            print(f'Resumed SAC from {args.resume}')
            stem = args.resume[:-4] if args.resume.endswith('.zip') else args.resume
            replay_path = f'{stem}.replay.pkl'
            if os.path.exists(replay_path):
                model.load_replay_buffer(replay_path)
                print(f'Loaded replay buffer from {replay_path} '
                      f'({model.replay_buffer.size()} transitions)')
            else:
                # No buffer on disk. SAC's learning_starts check is against
                # num_timesteps, which is restored from the checkpoint and
                # already past 200, so updates would otherwise begin on a
                # near-empty buffer and degrade the loaded policy. Defer
                # updates by `warmup` env steps to rebuild a usable buffer.
                warmup = 500
                model.learning_starts = model.num_timesteps + warmup
                print(f'No replay buffer at {replay_path} — '
                      f'deferring gradient updates for {warmup} env steps '
                      f'to rebuild a warmup buffer.')
        else:
            model = SAC(
                'MlpPolicy', env, verbose=1,
                learning_starts=200,
                # Tightened from -0.25·n_act to -0.5·n_act so the policy
                # commits harder once it's found a tune. The looser value
                # kept exploration alive long enough to find a basin, but
                # prevented the policy from settling into a specific config.
                target_entropy=-0.5 * n_act,
            )

    plot_cb = LivePlotCallback(save_path=plot_path, update_every=8, verbose=1)
    ckpt_cb = CheckpointCallback(
        save_freq=2000,
        save_path=args.save_dir,
        name_prefix=f'{run_name}_ckpt',
        save_replay_buffer=(args.action_mode != 'discrete'),
    )
    callback = CallbackList([plot_cb, ckpt_cb])

    print(f'Training {model.__class__.__name__} '
          f'({args.action_mode}, {args.planner}) '
          f'for {args.timesteps} steps...')
    print(f'Run name : {run_name}')
    print(f'Policy    {save_path}.zip')
    print(f'Plot      {plot_path}')

    try:
        # Don't reset num_timesteps when resuming — our warmup offset is
        # relative to the loaded counter, and SB3's default (reset=True)
        # would zero it out, pushing learning_starts far out of reach and
        # producing a run with zero gradient updates.
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            reset_num_timesteps=(args.resume is None),
        )
    except KeyboardInterrupt:
        print('\nInterrupted — saving current weights before exit.')
    finally:
        model.save(save_path)
        print(f'Model saved  {save_path}.zip')
        if args.action_mode != 'discrete':
            replay_path = f'{save_path}.replay.pkl'
            model.save_replay_buffer(replay_path)
            print(f'Replay buffer saved  {replay_path}')
        env.close()


if __name__ == '__main__':
    main()
