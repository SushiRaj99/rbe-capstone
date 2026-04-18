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
    DISCRETE_CONFIGS, PLANNER_PARAM_RANGES,
    OBS_LOW, OBS_HIGH, REWARD, N_LIDAR_RAYS, ACTION_EMA_ALPHA,
)


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

        n_act = len(self.param_ranges)
        act_bounds = np.ones(n_act, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate([OBS_LOW,  -act_bounds]),
            high=np.concatenate([OBS_HIGH,  act_bounds]),
            dtype=np.float32,
        )

        if action_mode == 'discrete':
            self.action_space = spaces.Discrete(len(DISCRETE_CONFIGS))
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32,
            )

        self.obs_queue       = queue.Queue()
        self.last_obs        = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.prev_dist       = 0.0
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

    # Gymnasium API functions
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.smoothed_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_smoothed   = np.zeros(self.action_space.shape, dtype=np.float32)
        self.ep_switches     = 0
        self.ep_delta_sum    = 0.0

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
        self.prev_dist = float(obs[N_LIDAR_RAYS])   # index 18: distance_to_goal
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
                return self.last_obs, total_reward, False, True, {}

            if tag == 'done':
                ep: EpisodeMetrics = data
                total_reward += (
                    REWARD['goal_bonus'] if ep.goal_reached else REWARD['fail_penalty']
                )
                info = {
                    'episode_metrics': ep,
                    'param_switches':  self.ep_switches,
                    'mean_delta':      self.ep_delta_sum / max(self.ep_switches, 1),
                    'total_switches':  self.total_switches,
                }
                return self.last_obs, total_reward, True, False, info

            obs = self.step_metrics_to_obs(data)
            total_reward += self.compute_step_reward(data, obs)
            self.prev_dist = float(obs[N_LIDAR_RAYS])   # index 18: distance_to_goal
            self.last_obs  = obs
            ticks += 1

        info = {
            'param_switches': self.ep_switches,
            'mean_delta':     self.ep_delta_sum / max(self.ep_switches, 1),
            'total_switches': self.total_switches,
        }
        return self.last_obs, total_reward, False, False, info

    def close(self):
        self.executor.shutdown()
        rclpy.shutdown()

    def apply_action(self, action):
        if self.action_mode == 'discrete':
            planner, preset = DISCRETE_CONFIGS[int(action)]
            req = SwitchPlanner.Request()
            req.planner_name = planner
            self.node.call_sync(self.node.switch_planner_client, req)
            req2 = SetParamPreset.Request()
            req2.preset = preset
            self.node.call_sync(self.node.set_preset_client, req2)
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
            values = []
            for i, name in enumerate(names):
                lo, hi = self.param_ranges[name]
                values.append(float(lo + (self.smoothed_action[i] + 1.0) / 2.0 * (hi - lo)))
            req = SetRawParams.Request()
            req.names  = names
            req.values = values
            self.node.call_sync(self.node.set_raw_params_client, req)

    def wait_for_step_obs(self):
        while True:
            try:
                tag, data = self.obs_queue.get(timeout=self.step_timeout)
            except queue.Empty:
                self.node.get_logger().warn('Timed out waiting for first StepMetrics')
                return self.last_obs.copy()
            if tag == 'step':
                return self.step_metrics_to_obs(data)

    def step_metrics_to_obs(self, msg: StepMetrics) -> np.ndarray:
        lidar = np.array(msg.lidar_rays, dtype=np.float32)
        state = np.array([
            msg.distance_to_goal,
            msg.heading_error_to_goal,
            msg.linear_velocity,
            msg.angular_velocity,
            msg.clearance,
            msg.path_deviation,
            float(msg.collision),
        ], dtype=np.float32)
        base = np.clip(np.concatenate([lidar, state]), OBS_LOW, OBS_HIGH)
        return np.concatenate([base, self.smoothed_action.astype(np.float32)])

    def compute_step_reward(self, msg: StepMetrics, obs: np.ndarray) -> float:
        progress = self.prev_dist - float(obs[N_LIDAR_RAYS])  # index 18: distance_to_goal
        r = (
            REWARD['progress']  * progress
            + REWARD['path_dev']  * msg.path_deviation
            + REWARD['ang_vel']   * msg.angular_velocity
            + REWARD['collision'] * float(msg.collision)
        )
        return float(r)
