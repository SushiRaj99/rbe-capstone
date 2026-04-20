#!/usr/bin/env python3
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
    OBS_LOW, OBS_HIGH, REWARD, ACTION_EMA_ALPHA,
)


# Parameters whose range spans an order of magnitude or more — linear decode
# would give wildly uneven coverage of the useful regime. Log decode maps
# action uniformly over log(value), so action=0 lands on the geometric mean
# of (lo, hi) and each unit of action corresponds to a fixed cost *ratio*.
LOG_DECODED_PARAMS = {'obstacle_scale'}


def decode_param(name: str, action: float, ranges: dict, baselines: dict) -> float:
    """Decode a normalized action in [-1, 1] to a real param value.
    action=-1 → lo, action=+1 → hi. Linear by default, log for params listed
    in LOG_DECODED_PARAMS.
    """
    lo, hi = ranges[name]
    a = float(np.clip(action, -1.0, 1.0))
    if name in LOG_DECODED_PARAMS:
        return lo * (hi / lo) ** ((a + 1.0) / 2.0)
    return lo + (a + 1.0) / 2.0 * (hi - lo)


def encode_param(name: str, value: float, ranges: dict) -> float:
    """Inverse of decode_param — given a real param value, return the action
    in [-1, 1] that produces it. Used by eval.py to build a baseline action
    vector that decodes exactly to preset 1's values.
    """
    lo, hi = ranges[name]
    if name in LOG_DECODED_PARAMS:
        return 2.0 * np.log(value / lo) / np.log(hi / lo) - 1.0
    return 2.0 * (value - lo) / (hi - lo) - 1.0


class BridgeNode(Node):
    """ROS2 node that runs in a background thread and feeds the gym env."""

    def __init__(self, obs_queue):
        super().__init__('potr_rl_bridge')
        self.obs_queue = obs_queue
        self.cb = ReentrantCallbackGroup()

        self.create_subscription(
            StepMetrics, '/potr_navigation/step_metrics',
            self.step_metrics_cb, 10, callback_group=self.cb,
        )
        self.create_subscription(
            EpisodeMetrics, '/potr_navigation/episode_metrics',
            self.episode_metrics_cb, 10, callback_group=self.cb,
        )

        self.start_episode_client = self.create_client(
            Trigger, '/potr_navigation/start_episode', callback_group=self.cb,
        )
        self.switch_planner_client = self.create_client(
            SwitchPlanner, '/potr_navigation/switch_planner', callback_group=self.cb,
        )
        self.set_preset_client = self.create_client(
            SetParamPreset, '/potr_navigation/set_param_preset', callback_group=self.cb,
        )
        self.set_raw_params_client = self.create_client(
            SetRawParams, '/potr_navigation/set_raw_params', callback_group=self.cb,
        )

    def step_metrics_cb(self, msg):
        self.obs_queue.put(('step', msg))

    def episode_metrics_cb(self, msg):
        self.obs_queue.put(('done', msg))

    def call_sync(self, client, request, timeout=10.0):
        """Call a service from the main thread while rclpy spins in background."""
        event = threading.Event()
        result = [None]

        def cb(future):
            result[0] = future.result()
            event.set()

        future = client.call_async(request)
        future.add_done_callback(cb)
        event.wait(timeout=timeout)
        return result[0]


class PotrNavEnv(gym.Env):
    """Gymnasium env that tunes Nav2 planner parameters via RL over the POTR simulation."""

    metadata = {'render_modes': []}

    def __init__(self, action_mode='continuous', planner='MPPI',
                 action_frequency=50, step_timeout=30.0):
        super().__init__()

        self.action_mode      = action_mode
        self.planner          = planner
        self.action_frequency = action_frequency
        self.step_timeout     = step_timeout

        self.param_ranges = PLANNER_PARAM_RANGES[planner]
        self.baselines    = PLANNER_BASELINES[planner]

        n_act = len(self.param_ranges)
        act_bounds = np.ones(n_act, dtype=np.float32)
        # Base obs is normalised to [-1, 1] in step_metrics_to_obs; smoothed
        # action is already in [-1, 1].
        base_lo = np.full(OBS_LOW.shape,  -1.0, dtype=np.float32)
        base_hi = np.full(OBS_HIGH.shape,  1.0, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([base_lo, -act_bounds]),
            high=np.concatenate([base_hi,  act_bounds]),
            dtype=np.float32,
        )
        self._obs_span = (OBS_HIGH - OBS_LOW).astype(np.float32)

        if action_mode == 'discrete':
            self.action_space = spaces.Discrete(len(DISCRETE_CONFIGS))
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32,
            )

        self.obs_queue       = queue.Queue()
        self.last_obs        = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.prev_dist       = 0.0
        self._last_raw_dist  = 0.0  # set by step_metrics_to_obs; read by reset()
        self.smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_smoothed   = np.zeros(self.action_space.shape, dtype=np.float32)
        self.ep_switches     = 0
        self.ep_delta_sum    = 0.0
        self.total_switches  = 0

        rclpy.init()
        self.node     = BridgeNode(self.obs_queue)
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(
            target=self.executor.spin, daemon=True,
        )
        self.spin_thread.start()

        # Tell planner_controller which shared-name vocabulary to use. Without
        # this it defaults to 'MPPI' and every continuous-mode set_raw_params
        # call with a DWB-only name (e.g. goal_align_scale) silently fails its
        # name lookup, meaning the policy has zero effect on the controller.
        if not self.node.switch_planner_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('switch_planner service not available after 10s')
        req = SwitchPlanner.Request()
        req.planner_name = self.planner
        self.call_service(
            self.node.switch_planner_client, req,
            label=f'initial switch_planner({self.planner})',
        )

    # Gymnasium API functions
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_smoothed   = np.zeros(self.action_space.shape, dtype=np.float32)
        self.ep_switches     = 0
        self.ep_delta_sum    = 0.0

        self.ep_r_progress   = 0.0
        self.ep_r_pathdev    = 0.0
        self.ep_r_angvel     = 0.0
        self.ep_r_collision  = 0.0
        self.ep_r_time       = 0.0
        self.ep_collision_steps = 0
        self.ep_step_count   = 0
        self.ep_last_distance = 0.0

        while not self.obs_queue.empty():
            try:
                self.obs_queue.get_nowait()
            except queue.Empty:
                break

        res = self.node.call_sync(
            self.node.start_episode_client, Trigger.Request(),
        )
        if res is None or not res.success:
            msg = res.message if res else 'timeout'
            self.node.get_logger().warn(f'start_episode failed: {msg}')

        obs = self.wait_for_step_obs()
        self.prev_dist = self._last_raw_dist
        self.last_obs  = obs
        return obs, {}

    def step(self, action):
        self.apply_action(action)

        total_reward = 0.0
        ticks = 0

        while ticks < self.action_frequency:
            try:
                tag, data = self.obs_queue.get(timeout=self.step_timeout)
            except queue.Empty:
                self.node.get_logger().warn('Step timeout — truncating')
                return self.last_obs, total_reward, False, True, self.episode_info(
                    terminal_reward=0.0, termination='truncated', goal_reached=False,
                )

            if tag == 'done':
                ep: EpisodeMetrics = data
                terminal = (
                    REWARD['goal_bonus'] if ep.goal_reached else REWARD['fail_penalty']
                )
                total_reward += terminal
                info = self.episode_info(
                    terminal_reward=terminal,
                    termination=('goal' if ep.goal_reached else 'fail'),
                    goal_reached=bool(ep.goal_reached),
                )
                info['episode_metrics'] = ep
                return self.last_obs, total_reward, True, False, info

            obs = self.step_metrics_to_obs(data)
            total_reward += self.compute_step_reward(data, obs)
            self.prev_dist = float(data.distance_to_goal)
            self.last_obs  = obs
            ticks += 1

        info = {
            'param_switches': self.ep_switches,
            'mean_delta':     self.ep_delta_sum / max(self.ep_switches, 1),
            'total_switches': self.total_switches,
        }
        return self.last_obs, total_reward, False, False, info

    def episode_info(self, *, terminal_reward: float, termination: str,
                     goal_reached: bool) -> dict:
        return {
            'param_switches':  self.ep_switches,
            'mean_delta':      self.ep_delta_sum / max(self.ep_switches, 1),
            'total_switches':  self.total_switches,
            'r_progress':      self.ep_r_progress,
            'r_pathdev':       self.ep_r_pathdev,
            'r_angvel':        self.ep_r_angvel,
            'r_collision':     self.ep_r_collision,
            'r_time':          self.ep_r_time,
            'r_terminal':      terminal_reward,
            'collision_frac':  self.ep_collision_steps / max(self.ep_step_count, 1),
            'final_distance':  self.ep_last_distance,
            'termination':     termination,
            'goal_reached':    goal_reached,
        }

    def close(self):
        self.executor.shutdown()
        rclpy.shutdown()

    def call_service(self, client, req, *, label: str):
        """call_sync wrapper that surfaces service failures as log warnings.
        Previously we ignored the response, which let silent server-side errors
        (e.g. wrong-planner name-lookup failures) masquerade as a working pipe.
        """
        res = self.node.call_sync(client, req)
        if res is None:
            self.node.get_logger().warn(f'{label} timed out')
        elif not res.success:
            self.node.get_logger().warn(f'{label} failed: {res.message}')
        return res

    def apply_action(self, action):
        if self.action_mode == 'discrete':
            planner, preset = DISCRETE_CONFIGS[int(action)]
            req = SwitchPlanner.Request()
            req.planner_name = planner
            self.call_service(
                self.node.switch_planner_client, req,
                label=f'switch_planner({planner})',
            )
            req2 = SetParamPreset.Request()
            req2.preset = preset
            self.call_service(
                self.node.set_preset_client, req2,
                label=f'set_param_preset({preset})',
            )
        else:
            action = np.clip(action, -1.0, 1.0)
            # EMA smoothing: damps rapid swings before they hit Nav2/MPPI
            new_smoothed = (ACTION_EMA_ALPHA * self.smoothed_action
                            + (1.0 - ACTION_EMA_ALPHA) * action)
            delta = float(np.sum(np.abs(new_smoothed - self.prev_smoothed)))
            self.prev_smoothed   = self.smoothed_action.copy()
            self.smoothed_action = new_smoothed
            self.ep_switches    += 1
            self.ep_delta_sum   += delta
            self.total_switches += 1
            names  = list(self.param_ranges.keys())
            values = [decode_param(name, self.smoothed_action[i],
                                   self.param_ranges, self.baselines)
                      for i, name in enumerate(names)]
            req = SetRawParams.Request()
            req.names  = names
            req.values = values
            self.call_service(
                self.node.set_raw_params_client, req,
                label='set_raw_params',
            )

    def wait_for_step_obs(self):
        while True:
            try:
                tag, data = self.obs_queue.get(timeout=self.step_timeout)
            except queue.Empty:
                self.node.get_logger().warn('Timed out waiting for first StepMetrics')
                return self.last_obs.copy()
            if tag == 'step':
                # step_metrics_to_obs sets self._last_raw_dist as a side effect
                # so reset() can seed prev_dist without re-parsing the msg.
                return self.step_metrics_to_obs(data)

    def step_metrics_to_obs(self, msg: StepMetrics) -> np.ndarray:
        self._last_raw_dist = float(msg.distance_to_goal)
        path_costs = np.array([
            msg.path_cost_near,
            msg.path_cost_mid,
            msg.path_cost_far,
        ], dtype=np.float32)
        state = np.array([
            msg.distance_to_goal,
            msg.heading_error_to_goal,
            msg.linear_velocity,
            msg.angular_velocity,
            msg.path_deviation,
        ], dtype=np.float32)
        base = np.clip(np.concatenate([path_costs, state]), OBS_LOW, OBS_HIGH)
        base_norm = 2.0 * (base - OBS_LOW) / self._obs_span - 1.0
        return np.concatenate([base_norm, self.smoothed_action.astype(np.float32)])

    def compute_step_reward(self, msg: StepMetrics, obs: np.ndarray) -> float:
        raw_dist    = float(msg.distance_to_goal)
        progress    = self.prev_dist - raw_dist
        r_progress  = REWARD['progress']  * progress
        r_pathdev   = REWARD['path_dev']  * msg.path_deviation
        r_angvel    = REWARD['ang_vel']   * msg.angular_velocity
        r_collision = REWARD['collision'] * float(msg.collision)
        r_time      = REWARD['time_step']

        self.ep_r_progress      += r_progress
        self.ep_r_pathdev       += r_pathdev
        self.ep_r_angvel        += r_angvel
        self.ep_r_collision     += r_collision
        self.ep_r_time          += r_time
        self.ep_collision_steps += int(msg.collision)
        self.ep_step_count      += 1
        self.ep_last_distance    = raw_dist

        return float(r_progress + r_pathdev + r_angvel + r_collision + r_time)
