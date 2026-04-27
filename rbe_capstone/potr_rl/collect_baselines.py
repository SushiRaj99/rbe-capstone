#!/usr/bin/env python3
"""Collect per-goal_id mean baseline times for the env's terminal time-delta reward."""
import argparse
import json
import os
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np

from potr_rl.env import PotrNavEnv, encode_param
from potr_rl.params import PLANNER_PARAM_RANGES, PLANNER_BASELINES, DISCRETE_CONFIGS


def collect_times(env: PotrNavEnv, baseline_fn: Callable, n_episodes: int, action_mode: str) -> Dict[str, List[float]]:
    """
    Run the baseline action repeatedly and record the time-to-goal for each
    successful episode, grouped by goal_id.

    Inputs:
        env: PotrNavEnv instance
        baseline_fn: callable(obs) -> action that returns the fixed baseline action
        n_episodes: total episodes to run (episode_runner cycles its goal list)
        action_mode: 'discrete' or 'continuous'

    Returns:
        Dict mapping goal_id -> list of total_time (s) for successful episodes.
    """
    times_by_goal = defaultdict(list)
    obs, _ = env.reset()
    ep_reward = 0.0
    ep = 0

    while ep < n_episodes:
        action = baseline_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if terminated or truncated:
            em = info.get('episode_metrics')
            if em and em.goal_reached:
                times_by_goal[em.goal_id].append(float(em.total_time))
                print(f'  [{ep + 1}] goal_id={em.goal_id}  time={em.total_time:.2f}s  reward={ep_reward:.1f}')
            elif em:
                print(f'  [{ep + 1}] goal_id={em.goal_id}  FAILED  reward={ep_reward:.1f}')
            else:
                print(f'  [{ep + 1}] truncated  reward={ep_reward:.1f}')
            ep += 1
            ep_reward = 0.0
            if ep < n_episodes:
                obs, _ = env.reset()

    return times_by_goal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Output JSON path for per-goal_id mean baseline times.')
    parser.add_argument('--action-mode', default='continuous', choices=['discrete', 'continuous'])
    parser.add_argument('--planner', default='DWB', choices=['MPPI', 'DWB'])
    parser.add_argument('--preset', type=int, default=1, choices=[1, 2])
    parser.add_argument('--episodes', type=int, default=60, help='Total episodes to run. episode_runner cycles the goalpoints list, so set >> len(list) for even coverage.')
    args = parser.parse_args()

    env = PotrNavEnv(action_mode=args.action_mode, planner=args.planner)
    param_ranges = PLANNER_PARAM_RANGES[args.planner]
    baselines = PLANNER_BASELINES[args.planner]

    if args.action_mode == 'discrete':
        baseline_action = next(i for i, (p, pr) in enumerate(DISCRETE_CONFIGS) if p == args.planner and pr == args.preset)
        baseline_fn = lambda obs: baseline_action
    else:
        names = list(param_ranges.keys())
        fixed_action = np.array([encode_param(n, baselines[n], param_ranges, baselines) for n in names], dtype=np.float32)
        baseline_fn = lambda obs: fixed_action

    print(f'\n=== Collecting baselines ({args.planner} preset {args.preset}, {args.episodes} episodes) ===')
    times_by_goal = collect_times(env, baseline_fn, args.episodes, args.action_mode)

    summary = {goal_id: float(np.mean(ts)) for goal_id, ts in times_by_goal.items()}

    print('\nPer-goal_id summary:')
    for goal_id in sorted(times_by_goal.keys()):
        ts = times_by_goal[goal_id]
        print(f'  {goal_id}: mean={np.mean(ts):.2f}s  std={np.std(ts):.2f}s  n={len(ts)}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f'\nSaved {len(summary)} baseline times to {args.output}')

    env.close()


if __name__ == '__main__':
    main()
