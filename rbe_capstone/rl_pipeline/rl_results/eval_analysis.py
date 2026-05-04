#!/usr/bin/env python3
import numpy as np
import json
from typing import NamedTuple, Tuple

class Comparison(NamedTuple):
    rl: float
    base: float

class PerEpisodeDelta(NamedTuple):
    mean: float
    std: float

def analyze_json_results(filepath: str) -> Tuple[Comparison, Comparison, Comparison, Comparison, PerEpisodeDelta, PerEpisodeDelta]:
    data = None
    with open(filepath) as file:
        data = json.load(file)
    if data is not None:
        rl_success = np.array([x == 'goal_reached' for x in data['rl_model']['outcomes']])
        rl_collisions = np.array([x == 'collision' for x in data['rl_model']['outcomes']])
        rl_timeouts = np.array([x == 'timeout' for x in data['rl_model']['outcomes']])
        rl_stucks = np.array([x == 'stuck' for x in data['rl_model']['outcomes']])
        rl_rewards = np.array(data['rl_model']['rewards'])
        rl_steps = np.array(data['rl_model']['lengths'])
        base_success = np.array([x == 'goal_reached' for x in data['baseline']['outcomes']])
        base_collisions = np.array([x == 'collision' for x in data['baseline']['outcomes']])
        base_timeouts = np.array([x == 'timeout' for x in data['baseline']['outcomes']])
        base_stucks = np.array([x == 'stuck' for x in data['baseline']['outcomes']])
        base_rewards = np.array(data['baseline']['rewards'])
        base_steps = np.array(data['baseline']['lengths'])
        shared_success = np.logical_and(rl_success, base_success)
        # Outcome analysis:
        rl_success_rate = float(np.sum(rl_success)/data['rl_model']['n_episodes'])
        rl_collision_rate = float(np.sum(rl_collisions)/data['rl_model']['n_episodes'])
        rl_timeout_rate = float(np.sum(rl_timeouts)/data['rl_model']['n_episodes'])
        rl_stuck_rate = float(np.sum(rl_stucks)/data['rl_model']['n_episodes'])
        base_success_rate = float(np.sum(base_success)/data['baseline']['n_episodes'])
        base_collision_rate = float(np.sum(base_collisions)/data['baseline']['n_episodes'])
        base_timeout_rate = float(np.sum(base_timeouts)/data['baseline']['n_episodes'])
        base_stuck_rate = float(np.sum(base_stucks)/data['baseline']['n_episodes'])
        # Reward anaylsis:
        rl_mean_reward = float(np.mean(rl_rewards))
        rl_std_reward = float(np.std(rl_rewards))
        rl_mean_reward_success = float(np.mean(rl_rewards[rl_success]))
        rl_std_reward_success = float(np.std(rl_rewards[rl_success]))
        base_mean_reward = float(np.mean(base_rewards))
        base_std_reward = float(np.std(base_rewards))
        base_mean_reward_success = float(np.mean(base_rewards[base_success]))
        base_std_reward_success = float(np.std(base_rewards[base_success]))
        mean_reward_deltas = float(np.mean(rl_rewards[shared_success] - base_rewards[shared_success]))
        std_reward_deltas = float(np.std(rl_rewards[shared_success] - base_rewards[shared_success]))
        # Time analysis:
        rl_mean_time = float(np.mean(rl_steps))
        rl_std_time = float(np.std(rl_steps))
        rl_mean_time_success = float(np.mean(rl_steps[rl_success]))
        rl_std_time_success = float(np.std(rl_steps[rl_success]))
        base_mean_time = float(np.mean(base_steps))
        base_std_time = float(np.std(base_steps))
        base_mean_time_success = float(np.mean(base_steps[base_success]))
        base_std_time_success = float(np.std(base_steps[base_success]))
        mean_time_deltas = float(np.mean(rl_steps[shared_success] - base_steps[shared_success]))
        std_time_deltas = float(np.std(rl_steps[shared_success]) - np.std(base_steps[shared_success]))
        # Print results:
        print(f"Results for {filepath}...")
        print(
            f"RL Model:\n\t{rl_success_rate = }\n\t{rl_collision_rate = }\n\t{rl_timeout_rate = }\n\t{rl_stuck_rate = }"
            f"\n\t{rl_mean_reward = }\n\t{rl_std_reward = }\n\t{rl_mean_reward_success = }\n\t{rl_std_reward_success = }"
            f"\n\t{rl_mean_time = }\n\t{rl_std_time = }\n\t{rl_mean_time_success = }\n\t{rl_std_time_success = }"
        )
        print(
            f"Baseline Model:\n\t{base_success_rate = }\n\t{base_collision_rate = }\n\t{base_timeout_rate = }\n\t{base_stuck_rate = }"
            f"\n\t{base_mean_reward = }\n\t{base_std_reward = }\n\t{base_mean_reward_success = }\n\t{base_std_reward_success = }"
            f"\n\t{base_mean_time = }\n\t{base_std_time = }\n\t{base_mean_time_success = }\n\t{base_std_time_success = }"
        )
        print(
            f"Per Episode Deltas:\n\t{mean_reward_deltas = }\n\t{std_reward_deltas = }\n\t{mean_time_deltas = }\n\t{std_time_deltas = }\n"
        )
        return (
            Comparison(rl_success_rate, base_success_rate),
            Comparison(rl_collision_rate, base_collision_rate),
            Comparison(rl_timeout_rate, base_timeout_rate),
            Comparison(rl_stuck_rate, base_stuck_rate),
            PerEpisodeDelta(mean_reward_deltas, std_reward_deltas),
            PerEpisodeDelta(mean_time_deltas, std_time_deltas)
        )

if __name__ == "__main__":
    comp_succ1, comp_coll1, comp_tout1, comp_stuk1, delta_r1, delta_t1 = analyze_json_results('./eval_results_10000_steps_200eps_seed27_with_length_run1.json')
    comp_succ2, comp_coll2, comp_tout2, comp_stuk2, delta_r2, delta_t2 = analyze_json_results('./eval_results_10000_steps_200eps_seed27_with_length_run2.json')
    comp_succ3, comp_coll3, comp_tout3, comp_stuk3, delta_r3, delta_t3 = analyze_json_results('./eval_results_10000_steps_200eps_seed27_with_length_run3.json')
    comp_succ4, comp_coll4, comp_tout4, comp_stuk4, delta_r4, delta_t4 = analyze_json_results('./eval_results_10000_steps_200eps_seed27_with_length_run4.json')
    print(
        "Overall Results:\n\t                      Baseline        RL"
        f"\n\tMean Success Rate:     {float(np.mean(np.array([comp_succ1.base, comp_succ2.base, comp_succ3.base, comp_succ4.base]))):.5f}"
        f"   {float(np.mean(np.array([comp_succ1.rl, comp_succ2.rl, comp_succ3.rl, comp_succ4.rl]))):.5f}"
        f"\n\tMean Collision Rate:   {float(np.mean(np.array([comp_coll1.base, comp_coll2.base, comp_coll3.base, comp_coll4.base]))):.5f}"
        f"   {float(np.mean(np.array([comp_coll1.rl, comp_coll2.rl, comp_coll3.rl, comp_coll4.rl]))):.5f}"
        f"\n\tMean Timeout Rate:     {float(np.mean(np.array([comp_tout1.base, comp_tout2.base, comp_tout3.base, comp_tout4.base]))):.5f}"
        f"   {float(np.mean(np.array([comp_tout1.rl, comp_tout2.rl, comp_tout3.rl, comp_tout4.rl]))):.5f}"
        f"\n\tMean Stuck Rate:       {float(np.mean(np.array([comp_stuk1.base, comp_stuk2.base, comp_stuk3.base, comp_stuk4.base]))):.5f}"
        f"   {float(np.mean(np.array([comp_stuk1.rl, comp_stuk2.rl, comp_stuk3.rl, comp_stuk4.rl]))):.5f}"
        f"\n\n\tAvg Mean Reward Difference per Episode (RL - Baseline) for Shared Successes: {float(np.mean(np.array([delta_r1.mean, delta_r2.mean, delta_r3.mean, delta_r4.mean])))}"
        f"\n\tAvg Std Reward Difference per Episode (RL - Baseline) for Shared Successes: {float(np.mean(np.array([delta_r1.std, delta_r2.std, delta_r3.std, delta_r4.std])))}"
        f"\n\tAvg Mean Time Difference per Episode (RL - Baseline) for Shared Successes: {float(np.mean(np.array([delta_t1.mean, delta_t2.mean, delta_t3.mean, delta_t4.mean])))}"
        f"\n\tAvg Difference of Std Time for Shared Successes (RL - Baseline): {float(np.mean(np.array([delta_t1.std, delta_t2.std, delta_t3.std, delta_t4.std])))}"
    )