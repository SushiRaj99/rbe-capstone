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
    DISCRETE_CONFIGS, CONTINUOUS_PARAMS, PARAM_RANGES,
    OBS_LOW, OBS_HIGH, REWARD,
)


class _BridgeNode(Node):
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
    """
    Gymnasium environment that bridges stable-baselines3 to the POTR Nav2 simulation.

    Each call to step(action) applies new planner parameters then runs for
    action_frequency odom ticks before returning, accumulating reward over
    that block.  This gives the agent the ability to adapt parameters as the
    robot moves while giving MPPI enough time between resets to build a stable
    trajectory rollout buffer.

    At the default action_frequency=50 and 10 Hz odom, parameters are updated
    every ~5 sim seconds.  MPPI typically recovers from a reset within 0.1–0.5s,
    leaving ~4.5s of uninterrupted planning per block.

    Parameters
    ----------
    action_mode : 'discrete' | 'continuous'
        'discrete'   — action is an integer index into DISCRETE_CONFIGS.
        'continuous' — action is a float32 vector in [-1, 1] mapped to physical
                       parameter ranges defined in params.py.
    planner : 'MPPI' | 'DWB'
        Active planner for continuous mode (ignored in discrete mode).
    action_frequency : int
        Odom ticks between parameter updates.  50 (~5s) is a safe minimum for
        MPPI.  Lower values increase adaptability but risk the robot stalling
        if MPPI does not recover before the next reset.
    step_timeout : float
        Seconds to wait for a single StepMetrics message before truncating.
    """

    metadata = {'render_modes': []}

    def __init__(self, action_mode='continuous', planner='MPPI',
                 action_frequency=50, step_timeout=30.0):
        super().__init__()

        self.action_mode      = action_mode
        self.planner          = planner
        self.action_frequency = action_frequency
        self.step_timeout     = step_timeout

        self.observation_space = spaces.Box(
            low=OBS_LOW, high=OBS_HIGH, dtype=np.float32,
        )

        if action_mode == 'discrete':
            self.action_space = spaces.Discrete(len(DISCRETE_CONFIGS))
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0,
                shape=(len(CONTINUOUS_PARAMS),),
                dtype=np.float32,
            )

        self._obs_queue = queue.Queue()
        self._last_obs  = np.zeros(len(OBS_LOW), dtype=np.float32)
        self._prev_dist = 0.0

        rclpy.init()
        self._node     = _BridgeNode(self._obs_queue)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin, daemon=True,
        )
        self._spin_thread.start()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        while not self._obs_queue.empty():
            try:
                self._obs_queue.get_nowait()
            except queue.Empty:
                break

        res = self._node.call_sync(
            self._node.start_episode_client, Trigger.Request(),
        )
        if res is None or not res.success:
            msg = res.message if res else 'timeout'
            self._node.get_logger().warn(f'start_episode failed: {msg}')

        # Wait for first observation — robot is in the settle window, stationary.
        # The first step() call will apply parameters before the nav goal is sent.
        obs = self._wait_for_step_obs()
        self._prev_dist = float(obs[0])
        self._last_obs  = obs
        return obs, {}

    def step(self, action):
        # Apply parameters now.  For the first block of each episode this happens
        # during the settle window (robot stationary, MPPI idle) so the reset is
        # harmless.  For subsequent blocks the robot is moving; MPPI resets and
        # recovers within ~0.5s, causing a brief velocity dip but not a stall.
        self._apply_action(action)

        total_reward = 0.0
        ticks = 0

        while ticks < self.action_frequency:
            try:
                tag, data = self._obs_queue.get(timeout=self.step_timeout)
            except queue.Empty:
                self._node.get_logger().warn('Step timeout — truncating')
                return self._last_obs, total_reward, False, True, {}

            if tag == 'done':
                ep: EpisodeMetrics = data
                total_reward += (
                    REWARD['goal_bonus'] if ep.goal_reached else REWARD['fail_penalty']
                )
                return self._last_obs, total_reward, True, False, {'episode_metrics': ep}

            obs = self._step_metrics_to_obs(data)
            total_reward += self._compute_step_reward(data, obs)
            self._prev_dist = float(obs[0])
            self._last_obs  = obs
            ticks += 1

        return self._last_obs, total_reward, False, False, {}

    def close(self):
        self._executor.shutdown()
        rclpy.shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_action(self, action):
        if self.action_mode == 'discrete':
            planner, preset = DISCRETE_CONFIGS[int(action)]
            req = SwitchPlanner.Request()
            req.planner_name = planner
            self._node.call_sync(self._node.switch_planner_client, req)
            req2 = SetParamPreset.Request()
            req2.preset = preset
            self._node.call_sync(self._node.set_preset_client, req2)
        else:
            action = np.clip(action, -1.0, 1.0)
            names  = CONTINUOUS_PARAMS
            values = []
            for i, name in enumerate(names):
                lo, hi = PARAM_RANGES[name]
                values.append(float(lo + (action[i] + 1.0) / 2.0 * (hi - lo)))
            req = SetRawParams.Request()
            req.names  = list(names)
            req.values = values
            self._node.call_sync(self._node.set_raw_params_client, req)

    def _wait_for_step_obs(self):
        while True:
            try:
                tag, data = self._obs_queue.get(timeout=self.step_timeout)
            except queue.Empty:
                self._node.get_logger().warn('Timed out waiting for first StepMetrics')
                return self._last_obs.copy()
            if tag == 'step':
                return self._step_metrics_to_obs(data)

    def _step_metrics_to_obs(self, msg: StepMetrics) -> np.ndarray:
        obs = np.array([
            msg.distance_to_goal,
            msg.heading_error_to_goal,
            msg.linear_velocity,
            msg.angular_velocity,
            msg.clearance,
            msg.path_deviation,
            float(msg.collision),
        ], dtype=np.float32)
        return np.clip(obs, OBS_LOW, OBS_HIGH)

    def _compute_step_reward(self, msg: StepMetrics, obs: np.ndarray) -> float:
        progress = self._prev_dist - float(obs[0])
        r = (
            REWARD['progress']  * progress
            + REWARD['path_dev']  * msg.path_deviation
            + REWARD['ang_vel']   * msg.angular_velocity
            + REWARD['collision'] * float(msg.collision)
        )
        return float(r)
