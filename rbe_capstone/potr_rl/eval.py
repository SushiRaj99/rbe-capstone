#!/usr/bin/env python3
import argparse
import numpy as np
from stable_baselines3 import SAC, PPO

from potr_rl.env import PotrNavEnv
from potr_rl.params import CONTINUOUS_PARAMS, PARAM_RANGES, DISCRETE_CONFIGS


def decode_continuous_action(action):
    parts = []
    for i, name in enumerate(CONTINUOUS_PARAMS):
        lo, hi = PARAM_RANGES[name]
        val = lo + (float(np.clip(action[i], -1.0, 1.0)) + 1.0) / 2.0 * (hi - lo)
        parts.append(f'{name}={val:.3f}')
    return '  '.join(parts)


def run_episodes(env, action_fn, label, n_episodes, action_mode):
    """Run n_episodes using action_fn(obs) -> action.  Returns list of result dicts."""
    print(f'\n=== {label} ===')
    results = []
    ep = 0
    obs, _ = env.reset()
    ep_reward = 0.0

    while ep < n_episodes:
        action = action_fn(obs)

        if action_mode == 'discrete':
            planner, preset = DISCRETE_CONFIGS[int(action)]
            print(f'  [{ep + 1}] action={int(action)} ({planner} preset {preset})')
        else:
            print(f'  [{ep + 1}] {decode_continuous_action(action)}')

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if terminated or truncated:
            ep_metrics = info.get('episode_metrics')
            if ep_metrics:
                reached = ep_metrics.goal_reached
                t       = ep_metrics.total_time
                dist    = ep_metrics.total_distance
                cols    = ep_metrics.collision_count
                print(f'       done: goal_reached={reached}  time={t:.1f}s  '
                      f'dist={dist:.1f}m  collisions={cols}  '
                      f'reward={ep_reward:.1f}\n')
                results.append({
                    'goal_reached': reached,
                    'total_time':   t,
                    'total_dist':   dist,
                    'collisions':   cols,
                    'reward':       ep_reward,
                })
            else:
                print(f'       truncated  reward={ep_reward:.1f}\n')
                results.append({
                    'goal_reached': False,
                    'total_time':   None,
                    'total_dist':   None,
                    'collisions':   None,
                    'reward':       ep_reward,
                })
            ep += 1
            ep_reward = 0.0
            if ep < n_episodes:
                obs, _ = env.reset()

    return results


def print_summary(label, results):
    n = len(results)
    n_reached  = sum(r['goal_reached'] for r in results)
    avg_reward = np.mean([r['reward'] for r in results])
    times      = [r['total_time'] for r in results if r['total_time'] is not None]
    dists      = [r['total_dist'] for r in results if r['total_dist'] is not None]
    cols       = [r['collisions'] for r in results if r['collisions'] is not None]
    print(f'  {label}')
    print(f'    Goals reached : {n_reached}/{n} ({100 * n_reached / n:.0f}%)')
    print(f'    Avg reward    : {avg_reward:.1f}')
    print(f'    Avg time      : {np.mean(times):.1f}s' if times else '    Avg time      : n/a')
    print(f'    Avg distance  : {np.mean(dists):.1f}m' if dists else '    Avg distance  : n/a')
    print(f'    Avg collisions: {np.mean(cols):.1f}' if cols else '    Avg collisions: n/a')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?',
                        help='Path to saved model (without .zip). '
                             'Omit to run baseline only.')
    parser.add_argument('--action-mode', default='continuous',
                        choices=['discrete', 'continuous'])
    parser.add_argument('--planner', default='MPPI', choices=['MPPI', 'DWB'])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--baseline-preset', type=int, default=1, choices=[1, 2],
                        help='Preset to use for the baseline run (default: 1)')
    parser.add_argument('--no-baseline', action='store_true',
                        help='Skip the baseline run and only evaluate the model')
    args = parser.parse_args()

    if args.model is None and args.no_baseline:
        parser.error('Nothing to run: provide a model path or remove --no-baseline')

    env = PotrNavEnv(action_mode=args.action_mode, planner=args.planner)

    all_results = {}

    # ------------------------------------------------------------------
    # Baseline — fixed preset, action does not change with observation
    # ------------------------------------------------------------------
    if not args.no_baseline:
        if args.action_mode == 'discrete':
            # Find the discrete action index that matches the requested preset
            # for the selected planner
            baseline_action = next(
                i for i, (p, pr) in enumerate(DISCRETE_CONFIGS)
                if p == args.planner and pr == args.baseline_preset
            )
            baseline_fn = lambda obs: baseline_action
        else:
            # Map preset midpoints to normalised [-1,1] action
            # Preset 1 → use lower half of each param range
            # Preset 2 → use upper half of each param range
            scale = -0.5 if args.baseline_preset == 1 else 0.5
            fixed_action = np.full(len(CONTINUOUS_PARAMS), scale, dtype=np.float32)
            baseline_fn = lambda obs: fixed_action

        label = f'Baseline ({args.planner} preset {args.baseline_preset})'
        all_results['baseline'] = run_episodes(
            env, baseline_fn, label, args.episodes, args.action_mode,
        )

    # ------------------------------------------------------------------
    # Trained policy
    # ------------------------------------------------------------------
    if args.model is not None:
        if args.action_mode == 'discrete':
            model = PPO.load(args.model, env=env)
        else:
            model = SAC.load(args.model, env=env)

        policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]
        all_results['policy'] = run_episodes(
            env, policy_fn, f'Trained policy ({args.model})',
            args.episodes, args.action_mode,
        )

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 50)
    print(f'COMPARISON  ({args.episodes} episodes each)')
    print('=' * 50)
    if 'baseline' in all_results:
        print_summary(f'Baseline ({args.planner} preset {args.baseline_preset})',
                      all_results['baseline'])
    if 'policy' in all_results:
        print_summary(f'Trained policy  ({args.model})', all_results['policy'])

    env.close()


if __name__ == '__main__':
    main()
