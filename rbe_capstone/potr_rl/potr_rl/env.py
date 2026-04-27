#!/usr/bin/env python3
import json
import queue
import threading

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger

from potr_navigation.msg import StepMetrics, EpisodeMetrics
from potr_navigation.srv import SwitchPlanner, SetParamPreset, SetRawParams

from potr_rl.params import (
    DISCRETE_CONFIGS, PLANNER_PARAM_RANGES, PLANNER_BASELINES,
    OBS_LOW, OBS_HIGH, REWARD, ACTION_EMA_ALPHA, DWB_META_REFERENCE,
)


# Log decode for params with a wide range (action=0 hits geometric mean).
LOG_DECODED_PARAMS = {'obstacle_scale'}


def load_baseline_times(path):
    # Returns {goal_id: mean_baseline_time_s}. Empty dict disables the time-delta term.
    if not path:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        print(f'Warning: failed to load baseline times from {path}; time-delta term disabled')
        return {}


def decode_param(name, action, ranges, baselines):
    lo, hi = ranges[name]
    base = baselines[name]
    a = float(np.clip(action, -1.0, 1.0))
    if name in LOG_DECODED_PARAMS:
        if a >= 0:
            return base * (hi / base) ** a
        return base * (base / lo) ** a
    if a >= 0:
        return base + a * (hi - base)
    return base + a * (base - lo)


def encode_param(name, value, ranges, baselines):
    lo, hi = ranges[name]
    base = baselines[name]
    if name in LOG_DECODED_PARAMS:
        if value >= base:
            return np.log(value / base) / np.log(hi / base)
        return np.log(value / base) / np.log(base / lo)
    if value >= base:
        return (value - base) / (hi - base)
    return (value - base) / (base - lo)


class BridgeNode(Node):
    def __init__(self, obs_queue):
        super().__init__('potr_rl_bridge')
        self.obs_queue = obs_queue
        self.cb = ReentrantCallbackGroup()

        self.create_subscription(StepMetrics, '/potr_navigation/step_metrics', self.step_metrics_cb, 10, callback_group=self.cb)
        self.create_subscription(EpisodeMetrics, '/potr_navigation/episode_metrics', self.episode_metrics_cb, 10, callback_group=self.cb)

        self.start_episode_client = self.create_client(Trigger, '/potr_navigation/start_episode', callback_group=self.cb)
        self.cancel_episode_client = self.create_client(Trigger, '/potr_navigation/cancel_episode', callback_group=self.cb)
        self.switch_planner_client = self.create_client(SwitchPlanner, '/potr_navigation/switch_planner', callback_group=self.cb)
        self.set_preset_client = self.create_client(SetParamPreset, '/potr_navigation/set_param_preset', callback_group=self.cb)
        self.set_raw_params_client = self.create_client(SetRawParams, '/potr_navigation/set_raw_params', callback_group=self.cb)

    def step_metrics_cb(self, msg):
        self.obs_queue.put(('step', msg))

    def episode_metrics_cb(self, msg):
        self.obs_queue.put(('done', msg))

    def call_sync(self, client, request, label, timeout=10.0):
        event = threading.Event()
        result = [None]

        def cb(future):
            result[0] = future.result()
            event.set()

        client.call_async(request).add_done_callback(cb)
        event.wait(timeout=timeout)

        res = result[0]
        if res is None:
            self.get_logger().warn(f'{label} timed out')
        elif not res.success:
            self.get_logger().warn(f'{label} failed: {res.message}')
        return res


class PotrNavEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, action_mode='continuous', planner='MPPI', action_frequency=50, step_timeout=30.0, baseline_times_path=None):
        super().__init__()

        self.action_mode = action_mode
        self.planner = planner
        self.action_frequency = action_frequency
        self.step_timeout = step_timeout
        self.baseline_time_by_goal = load_baseline_times(baseline_times_path)

        self.param_ranges = PLANNER_PARAM_RANGES[planner]
        self.baselines = PLANNER_BASELINES[planner]

        n_act = len(self.param_ranges)
        act_bounds = np.ones(n_act, dtype=np.float32)
        base_lo = np.full(OBS_LOW.shape, -1.0, dtype=np.float32)
        base_hi = np.full(OBS_HIGH.shape, 1.0, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_lo, -act_bounds]),
            high=np.concatenate([base_hi, act_bounds]),
            dtype=np.float32,
        )
        self.obs_span = (OBS_HIGH - OBS_LOW).astype(np.float32)

        if action_mode == 'discrete':
            self.action_space = spaces.Discrete(len(DISCRETE_CONFIGS))
        else:
            self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)

        self.obs_queue = queue.Queue()
        self.last_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.prev_dist = 0.0
        self.smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_smoothed = np.zeros(self.action_space.shape, dtype=np.float32)
        self.ep_switches = 0
        self.ep_delta_sum = 0.0
        self.total_switches = 0

        # Per-episode mean of the raw (clipped, pre-EMA) action - diagnostic for what
        # the policy actually wants to output before smoothing.
        self.action_names = list(self.param_ranges.keys())
        self.ep_action_sum = np.zeros(n_act, dtype=np.float32)
        self.ep_action_count = 0

        # Per-tick trace: linear velocity vs smoothed/raw max_linear_vel cap.
        # Used by eval.py --trace to diagnose velocity-dip sources (policy vs DWB vs smoother).
        self.last_raw_action = np.zeros(n_act, dtype=np.float32)
        self.tick_trace_v = []
        self.tick_trace_cap_smoothed = []
        self.tick_trace_cap_raw = []

        rclpy.init()
        self.node = BridgeNode(self.obs_queue)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

        if not self.node.switch_planner_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('switch_planner service not available after 10s')
        req = SwitchPlanner.Request()
        req.planner_name = self.planner
        self.node.call_sync(self.node.switch_planner_client, req, label=f'initial switch_planner({self.planner})')

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_smoothed = np.zeros(self.action_space.shape, dtype=np.float32)
        self.ep_switches = 0
        self.ep_delta_sum = 0.0
        self.ep_action_sum[:] = 0.0
        self.ep_action_count = 0
        self.last_raw_action[:] = 0.0
        self.tick_trace_v.clear()
        self.tick_trace_cap_smoothed.clear()
        self.tick_trace_cap_raw.clear()

        self.ep_r_progress = 0.0
        self.ep_r_pathdev = 0.0
        self.ep_r_angvel = 0.0
        self.ep_r_proximity = 0.0
        self.ep_r_slow = 0.0
        self.ep_collision_steps = 0
        self.ep_step_count = 0
        self.ep_last_distance = 0.0

        try:
            while True:
                self.obs_queue.get_nowait()
        except queue.Empty:
            pass

        self.node.call_sync(self.node.start_episode_client, Trigger.Request(), label='start_episode')

        obs, raw_dist = self.wait_for_step_obs()
        self.prev_dist = raw_dist
        self.last_obs = obs
        return obs, {}

    def step(self, action):
        self.apply_action(action)

        total_reward = 0.0
        ticks = 0

        while ticks < self.action_frequency:
            try:
                tag, data = self.obs_queue.get(timeout=self.step_timeout)
            except queue.Empty:
                self.node.get_logger().warn('Step timeout - truncating')
                return self.last_obs, total_reward, False, True, self.episode_info(
                    terminal_reward=0.0, termination='truncated', goal_reached=False,
                )

            if tag == 'done':
                ep = data
                terminal = REWARD['goal_bonus'] if ep.goal_reached else REWARD['fail_penalty']
                # Per-goal_id time delta centers the reward on zero across episodes.
                if ep.goal_reached:
                    baseline_time = self.baseline_time_by_goal.get(ep.goal_id)
                    if baseline_time is not None:
                        terminal += REWARD['time_delta_weight'] * (float(baseline_time) - float(ep.total_time))
                total_reward += terminal
                info = self.episode_info(
                    terminal_reward=terminal,
                    termination=('goal' if ep.goal_reached else 'fail'),
                    goal_reached=bool(ep.goal_reached),
                )
                info['episode_metrics'] = ep
                return self.last_obs, total_reward, True, False, info

            obs, raw_dist = self.step_metrics_to_obs(data)
            total_reward += self.compute_step_reward(data)
            self.record_tick_trace(data)
            self.prev_dist = raw_dist
            self.last_obs = obs
            ticks += 1

            # Collision = unrecoverable failure: cancel the Nav2 goal so episode_runner
            # transitions to S_GETTING_METRICS, drain until we see the matching 'done',
            # then return with fail_penalty applied.
            if data.collision:
                self.node.call_sync(self.node.cancel_episode_client, Trigger.Request(), label='cancel_episode (collision)')
                while True:
                    try:
                        tag, data = self.obs_queue.get(timeout=self.step_timeout)
                    except queue.Empty:
                        self.node.get_logger().warn('Timeout waiting for done after collision cancel')
                        break
                    if tag == 'done':
                        break
                terminal = REWARD['fail_penalty']
                total_reward += terminal
                info = self.episode_info(
                    terminal_reward=terminal,
                    termination='fail',
                    goal_reached=False,
                )
                if tag == 'done':
                    info['episode_metrics'] = data
                return self.last_obs, total_reward, True, False, info

        info = {
            'param_switches': self.ep_switches,
            'mean_delta': self.ep_delta_sum / max(self.ep_switches, 1),
            'total_switches': self.total_switches,
        }
        return self.last_obs, total_reward, False, False, info

    def close(self):
        self.executor.shutdown()
        rclpy.shutdown()

    def episode_info(self, *, terminal_reward, termination, goal_reached):
        info = {
            'param_switches': self.ep_switches,
            'mean_delta': self.ep_delta_sum / max(self.ep_switches, 1),
            'total_switches': self.total_switches,
            'r_progress': self.ep_r_progress,
            'r_pathdev': self.ep_r_pathdev,
            'r_angvel': self.ep_r_angvel,
            'r_proximity': self.ep_r_proximity,
            'r_slow': self.ep_r_slow,
            'r_terminal': terminal_reward,
            'collision_frac': self.ep_collision_steps / max(self.ep_step_count, 1),
            'final_distance': self.ep_last_distance,
            'termination': termination,
            'goal_reached': goal_reached,
        }
        if self.action_mode != 'discrete' and self.ep_action_count > 0:
            info['action_mean'] = (self.ep_action_sum / self.ep_action_count).tolist()
            info['action_names'] = self.action_names
        if self.action_mode != 'discrete' and self.tick_trace_v:
            info['trace'] = {
                'v': list(self.tick_trace_v),
                'cap_smoothed': list(self.tick_trace_cap_smoothed),
                'cap_raw': list(self.tick_trace_cap_raw),
            }
        return info

    def apply_action(self, action):
        if self.action_mode == 'discrete':
            planner, preset = DISCRETE_CONFIGS[int(action)]
            req = SwitchPlanner.Request()
            req.planner_name = planner
            self.node.call_sync(self.node.switch_planner_client, req, label=f'switch_planner({planner})')
            req2 = SetParamPreset.Request()
            req2.preset = preset
            self.node.call_sync(self.node.set_preset_client, req2, label=f'set_param_preset({preset})')
            return

        action = np.clip(action, -1.0, 1.0)
        self.last_raw_action = action.copy()
        self.ep_action_sum += action
        self.ep_action_count += 1
        # EMA smoothing damps rapid swings before they hit Nav2.
        new_smoothed = ACTION_EMA_ALPHA * self.smoothed_action + (1.0 - ACTION_EMA_ALPHA) * action
        delta = float(np.sum(np.abs(new_smoothed - self.prev_smoothed)))
        self.prev_smoothed = self.smoothed_action.copy()
        self.smoothed_action = new_smoothed
        self.ep_switches += 1
        self.ep_delta_sum += delta
        self.total_switches += 1

        names = list(self.param_ranges.keys())
        values = [decode_param(n, self.smoothed_action[i], self.param_ranges, self.baselines) for i, n in enumerate(names)]
        # Expand meta-params (e.g. path_weight) into the multiple ROS params they scale.
        ros_names, ros_values = self.expand_meta_params(names, values)
        req = SetRawParams.Request()
        req.names = ros_names
        req.values = ros_values
        self.node.call_sync(self.node.set_raw_params_client, req, label='set_raw_params')

    def expand_meta_params(self, names, values):
        ros_names = []
        ros_values = []
        for name, val in zip(names, values):
            mapping = DWB_META_REFERENCE.get(name)
            if mapping is None:
                ros_names.append(name)
                ros_values.append(val)
                continue
            for ros_name, reference_value in mapping.items():
                ros_names.append(ros_name)
                ros_values.append(reference_value * val)
        return ros_names, ros_values

    def wait_for_step_obs(self):
        while True:
            try:
                tag, data = self.obs_queue.get(timeout=self.step_timeout)
            except queue.Empty:
                self.node.get_logger().warn('Timed out waiting for first StepMetrics')
                return self.last_obs.copy(), self.prev_dist
            if tag == 'step':
                return self.step_metrics_to_obs(data)

    def step_metrics_to_obs(self, msg):
        path_costs = np.array([msg.path_cost_near, msg.path_cost_mid, msg.path_cost_far], dtype=np.float32)
        state = np.array([
            msg.distance_to_goal,
            msg.heading_error_to_goal,
            msg.linear_velocity,
            msg.angular_velocity,
            msg.path_deviation,
        ], dtype=np.float32)
        base = np.clip(np.concatenate([path_costs, state]), OBS_LOW, OBS_HIGH)
        base_norm = 2.0 * (base - OBS_LOW) / self.obs_span - 1.0
        obs = np.concatenate([base_norm, self.smoothed_action.astype(np.float32)])
        return obs, float(msg.distance_to_goal)

    def compute_step_reward(self, msg):
        raw_dist = float(msg.distance_to_goal)
        progress = self.prev_dist - raw_dist
        proximity = max(0.0, float(msg.path_cost_near) - 30.0) / 100.0

        # Penalize slow cruise far from goal in open obs (path_cost_near low).
        # Doesn't fire for legitimate slowdowns at corners/inflation/near-goal.
        slow_when_safe = float(
            msg.linear_velocity < 0.4
            and msg.distance_to_goal > 1.5
            and msg.path_cost_near < 30
        )

        r_progress = REWARD['progress'] * progress
        r_pathdev = REWARD['path_dev'] * msg.path_deviation
        r_angvel = REWARD['ang_vel'] * msg.angular_velocity
        r_proximity = REWARD['proximity'] * proximity
        r_collision = REWARD['collision'] * float(msg.collision)
        r_slow = REWARD['slow_pace'] * slow_when_safe

        self.ep_r_progress += r_progress
        self.ep_r_pathdev += r_pathdev
        self.ep_r_angvel += r_angvel
        self.ep_r_proximity += r_proximity
        self.ep_r_slow += r_slow
        self.ep_collision_steps += int(msg.collision)
        self.ep_step_count += 1
        self.ep_last_distance = raw_dist

        return float(r_progress + r_pathdev + r_angvel + r_proximity + r_collision + r_slow)

    def record_tick_trace(self, msg):
        # Per-tick trace for eval velocity-dip diagnosis. linear_velocity always logs;
        # cap lines fall back to the yaml-fixed value when max_linear_vel isn't in the action space.
        if self.action_mode == 'discrete':
            return
        self.tick_trace_v.append(float(msg.linear_velocity))
        if 'max_linear_vel' in self.action_names:
            i = self.action_names.index('max_linear_vel')
            self.tick_trace_cap_smoothed.append(
                decode_param('max_linear_vel', float(self.smoothed_action[i]), self.param_ranges, self.baselines)
            )
            self.tick_trace_cap_raw.append(
                decode_param('max_linear_vel', float(self.last_raw_action[i]), self.param_ranges, self.baselines)
            )
        else:
            # max_linear_vel is fixed at preset-1 yaml value (0.8 m/s for current setup).
            self.tick_trace_cap_smoothed.append(0.8)
            self.tick_trace_cap_raw.append(0.8)
