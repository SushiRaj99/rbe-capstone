#!/usr/bin/env python3
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC, PPO

from potr_rl.env import PotrNavEnv, decode_param, encode_param
from potr_rl.params import PLANNER_PARAM_RANGES, PLANNER_BASELINES, DISCRETE_CONFIGS


def decode_continuous_action(action, param_ranges, baselines):
    parts = []
    for i, name in enumerate(param_ranges):
        val = decode_param(name, action[i], param_ranges, baselines)
        parts.append(f'{name}={val:.3f}')
    return '  '.join(parts)


def run_episodes(env, action_fn, label, n_episodes, action_mode, param_ranges=None, baselines=None):
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
            print(f'  [{ep + 1}] {decode_continuous_action(action, param_ranges, baselines)}')

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if terminated or truncated:
            ep_metrics = info.get('episode_metrics')
            if ep_metrics:
                reached = ep_metrics.goal_reached
                t = ep_metrics.total_time
                dist = ep_metrics.total_distance
                cols = ep_metrics.collision_count
                print(f'       done: goal_reached={reached}  time={t:.1f}s  dist={dist:.1f}m  collisions={cols}  reward={ep_reward:.1f}\n')
                results.append({
                    'goal_reached': reached,
                    'total_time': t,
                    'total_dist': dist,
                    'collisions': cols,
                    'reward': ep_reward,
                })
            else:
                print(f'       truncated  reward={ep_reward:.1f}\n')
                results.append({
                    'goal_reached': False,
                    'total_time': None,
                    'total_dist': None,
                    'collisions': None,
                    'reward': ep_reward,
                })
            ep += 1
            ep_reward = 0.0
            if ep < n_episodes:
                obs, _ = env.reset()

    return results


def save_eval_plot(all_results, save_path, planner):
    series = []
    if 'baseline' in all_results:
        series.append(('Baseline', all_results['baseline'], 'slategray'))
    if 'policy' in all_results:
        series.append(('Policy', all_results['policy'], 'steelblue'))
    if not series:
        return

    n_episodes = len(series[0][1])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f'{planner} evaluation - {n_episodes} episodes each', fontsize=13)
    names = [s[0] for s in series]
    rng = np.random.default_rng(0)

    # Success rate
    ax = axes[0, 0]
    rates = [100.0 * sum(r['goal_reached'] for r in s[1]) / len(s[1]) for s in series]
    bars = ax.bar(names, rates, color=[s[2] for s in series])
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 1, f'{rate:.0f}%', ha='center', fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Goal reached (%)')
    ax.set_title('Success rate')
    ax.grid(True, alpha=0.3, axis='y')

    # Reward per episode
    ax = axes[0, 1]
    for i, (_, results, color) in enumerate(series):
        rewards = [r['reward'] for r in results]
        xs = i + rng.uniform(-0.12, 0.12, len(rewards))
        goal = [r['goal_reached'] for r in results]
        ax.scatter([x for x, g in zip(xs, goal) if g], [r for r, g in zip(rewards, goal) if g], color='seagreen', alpha=0.7, s=40)
        ax.scatter([x for x, g in zip(xs, goal) if not g], [r for r, g in zip(rewards, goal) if not g], color='firebrick', alpha=0.7, s=40)
        ax.plot([i - 0.3, i + 0.3], [np.mean(rewards)] * 2, color=color, linewidth=2)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(names)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Episode reward')
    ax.set_title('Reward per episode  (green=goal, red=fail)')
    ax.grid(True, alpha=0.3, axis='y')

    # Time to goal (successes only)
    ax = axes[1, 0]
    for i, (_, results, color) in enumerate(series):
        times = [r['total_time'] for r in results if r['goal_reached'] and r['total_time'] is not None]
        if times:
            xs = i + rng.uniform(-0.12, 0.12, len(times))
            ax.scatter(xs, times, color=color, alpha=0.7, s=40)
            ax.plot([i - 0.3, i + 0.3], [np.mean(times)] * 2, color=color, linewidth=2)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(names)
    ax.set_ylabel('Time to goal (s)')
    ax.set_title('Time to goal (successes only)')
    ax.grid(True, alpha=0.3, axis='y')

    # Collision count
    ax = axes[1, 1]
    for i, (_, results, color) in enumerate(series):
        cols = [r['collisions'] for r in results if r['collisions'] is not None]
        if cols:
            xs = i + rng.uniform(-0.12, 0.12, len(cols))
            ax.scatter(xs, cols, color=color, alpha=0.7, s=40)
            ax.plot([i - 0.3, i + 0.3], [np.mean(cols)] * 2, color=color, linewidth=2)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(names)
    ax.set_ylabel('Collision steps')
    ax.set_title('Collisions per episode')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f'Eval plot saved  {save_path}')


def print_summary(label, results):
    n = len(results)
    n_reached = sum(r['goal_reached'] for r in results)
    avg_reward = np.mean([r['reward'] for r in results])
    times = [r['total_time'] for r in results if r['total_time'] is not None]
    dists = [r['total_dist'] for r in results if r['total_dist'] is not None]
    cols = [r['collisions'] for r in results if r['collisions'] is not None]
    print(f'  {label}')
    print(f'    Goals reached : {n_reached}/{n} ({100 * n_reached / n:.0f}%)')
    print(f'    Avg reward    : {avg_reward:.1f}')
    print(f'    Avg time      : {np.mean(times):.1f}s' if times else '    Avg time      : n/a')
    print(f'    Avg distance  : {np.mean(dists):.1f}m' if dists else '    Avg distance  : n/a')
    print(f'    Avg collisions: {np.mean(cols):.1f}' if cols else '    Avg collisions: n/a')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', nargs='?', help='Path to saved model (without .zip). Omit to run baseline only.')
    parser.add_argument('--action-mode', default='continuous', choices=['discrete', 'continuous'])
    parser.add_argument('--planner', default='MPPI', choices=['MPPI', 'DWB'])
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--baseline-preset', type=int, default=1, choices=[1, 2])
    parser.add_argument('--no-baseline', action='store_true')
    parser.add_argument('--plot', default=None)
    args = parser.parse_args()

    if args.model is None and args.no_baseline:
        parser.error('Nothing to run: provide a model path or remove --no-baseline')

    env = PotrNavEnv(action_mode=args.action_mode, planner=args.planner)
    param_ranges = PLANNER_PARAM_RANGES[args.planner]
    baselines = PLANNER_BASELINES[args.planner]

    all_results = {}

    if not args.no_baseline:
        if args.action_mode == 'discrete':
            baseline_action = next(i for i, (p, pr) in enumerate(DISCRETE_CONFIGS) if p == args.planner and pr == args.baseline_preset)
            baseline_fn = lambda obs: baseline_action
        else:
            names = list(param_ranges.keys())
            fixed_action = np.array([encode_param(n, baselines[n], param_ranges, baselines) for n in names], dtype=np.float32)
            if args.baseline_preset == 2:
                fixed_action = fixed_action + 0.3
            baseline_fn = lambda obs: fixed_action

        label = f'Baseline ({args.planner} preset {args.baseline_preset})'
        all_results['baseline'] = run_episodes(env, baseline_fn, label, args.episodes, args.action_mode, param_ranges=param_ranges, baselines=baselines)

    if args.model is not None:
        if args.action_mode == 'discrete':
            model = PPO.load(args.model, env=env)
        else:
            model = SAC.load(args.model, env=env)

        policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]
        all_results['policy'] = run_episodes(env, policy_fn, f'Trained policy ({args.model})', args.episodes, args.action_mode, param_ranges=param_ranges, baselines=baselines)

    print('\n' + '=' * 50)
    print(f'COMPARISON  ({args.episodes} episodes each)')
    print('=' * 50)
    if 'baseline' in all_results:
        print_summary(f'Baseline ({args.planner} preset {args.baseline_preset})', all_results['baseline'])
    if 'policy' in all_results:
        print_summary(f'Trained policy  ({args.model})', all_results['policy'])

    plot_path = args.plot or (f'{args.model}_eval.png' if args.model else f'{args.planner.lower()}_preset{args.baseline_preset}_eval.png')
    save_eval_plot(all_results, plot_path, args.planner)

    env.close()


if __name__ == '__main__':
    main()
