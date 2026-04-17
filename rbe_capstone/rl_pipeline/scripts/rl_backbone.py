#!/usr/bin/env python3
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Empty
from rcl_interfaces.srv import SetParameters

import rl_pipeline.pipeline_utils as putils

from typing import Optional, Dict, Tuple
import threading    # shouldn't be problematic for rclpy since the threading should only occur inside nodes within a MultiThreadedExecutor
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces    # TODO - might be missing this package and may need to update Dockerfile
from stable_baselines3 import PPO

class RLBridgeNode(Node):
    def __init__(self):
        super().__init__('rl_bridge')
        self.cb_group = ReentrantCallbackGroup()
        # Create a thread-safe observation buffer
        self.obs_lock = threading.Lock()
        self.obs_event = threading.Event()
        self.latest_obs: Optional[np.ndarray] = None
        self.status: str = 'idle'
        # Configure subscribers:
        self.create_subscription(Float32MultiArray, '/rl/observation', self.collect_observation, 10, callback_group=self.cb_group)
        self.create_subscription(String, '/rl/status', self.collect_status, 10, callback_group=self.cb_group)
        # Configure publishers:
        self.action_pub = self.create_publisher(Float32MultiArray, '/rl/action', 10)
        # Configure service clients:
        self.reset_client = self.create_client(Empty, '/rl/reset', callback_group=self.cb_group)
        self.config_param_client = self.create_client(SetParameters, '/planner_config_manager/set_parameters', callback_group=self.cb_group)

    def collect_observation(self):
        # TODO - callback method for locking/unlocking and storing observation (state) vector
        pass

    def collect_status(self):
        # TODO - callback method for storing episode status (based on planner_config_manager's interraction with the goal_manager)
        pass

    def wait_for_observation(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        # TODO - method to use within the gym environment to block until a fresh observation arrives on /rl/observation topic
        pass

    def get_status(self) -> str:
        # TODO - getter for the gym environment
        pass

    def call_reset(self, config: Dict) -> bool:
        # TODO - method for the gym environment to call the /rl/reset service from within its native reset() method
        pass

    def apply_action(self, action: np.ndarray) -> None:
        # Publish the raw (normalized) PPO action vector to /rl/action. After the action is published, it's the 
        # responsibilty of the subscriber to denormalize from [-1, 1] -> [0, 1] -> [lo, hi] according to the associated
        # parameter spec in pipeline_utils and call the controller_server/set_parameters service accordingly:
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)

class Nav2GymEnv(gym.Env):
    
    metadata = {'render_modes': []}     # NOTE - recommended class attribute that may be needed for compatibility with SB3 (TODO - confirm whether this is needed)
    
    def __init__(self, planner_type: str = 'dwb'):
        super().__init__()
        self.planner_type = planner_type
        self.previous_dist_to_goal = np.inf     # used for dense reward
        self.curr_goal_x = np.inf               # used for dense reward
        self.curr_goal_y = np.inf               # used for dense reward
        # Define environment spaces:
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(putils.OBSERVATION_DIMS,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=-(putils.NUM_ACTIONS,), dtype=np.float32)
        # Configure "bridge" to ROS2:
        if not rclpy.ok():
            rclpy.init()
        self.bridge = RLBridgeNode()
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.bridge)
        self.ros_thread = threading.Thread(target=self.executor.spin, daemon=True, name='rl_ros_spin')
        self.ros_thread.start()
        self.bridge.get_logger().info(
            f"Nav2GymEnv initialized  (planner={planner_type}, "
            f"obs_dim={putils.OBSERVATION_DIMS}, action_dim={putils.NUM_ACTIONS})"
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        # Perform uniform random sample for a (map, pose) pair from the flattened list of episode configs:
        rng = np.random.default_rng(seed)
        config = putils.EPISODE_MAP_CONFIGS[int(rng.integers(len(putils.EPISODE_MAP_CONFIGS)))]
        # Pre-compute initial distance to goal for the dense reward for progress:
        self.previous_dist_to_goal = np.sqrt(
            (config['goal_x'] - config['start_x'])**2 + (config['goal_y'] - config['start_y'])**2
        )
        self.curr_goal_x = config['goal_x']
        self.curr_goal_y = config['goal_y']
        # 'Propagate' reset to ROS2 simulation (map swap + pose reset + new goal) and wait for the first 
        # observation from the new episode:
        self.bridge.call_reset(config)
        raw_observation = self.bridge.wait_for_observation(timeout=10.0)    # allow good chunk of time for map load
        if raw_observation is None:
            self.bridge.get_logger().warn("Observation timeout after reset - returning blank observation.")
            raw_observation = np.zeros((putils.OBSERVATION_DIMS,), dtype=np.float32)
        return self.normalize_observation(raw_observation), {'map': config.get('map_filepath', '')}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0.0
        terminated, truncated = False, True
        info = {'status': 'observation_timeout'}
        # Publish the raw action to /rl/action so the subscribing node can handle the 
        # denormalization and controller_server update:
        self.bridge.apply_action(action)
        # Wait for the next observation, then update reward and check done flags:
        raw_observation = self.bridge.wait_for_observation(timeout=2.0)
        if raw_observation is None:
            self.bridge.get_logger().warn("Observation timeout during step - returning blank observation.")
            observation = np.zeros((putils.OBSERVATION_DIMS,), dtype=np.float32)
        else:
            observation = self.normalize_observation(raw_observation)
            status = self.bridge.get_status()
            reward, info = self.compute_reward(raw_observation, status)
            terminated = status in ['goal_reached', 'collision']
            truncated = 'timeout' in status     # should cover both the 'observation_timeout' and standard 'timeout' cases
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.executor.shutdown(wait=False)
        rclpy.try_shutdown()

    def normalize_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        # The raw observation vector will need to be normalized to [-1, 1] for the policy (actor) network:
        obs = raw_obs.copy()
        # LIDAR ranges [0, range_max] -> [0, 1]:
        obs[:(2*putils.N_LIDAR_RAYS)][0::2] = np.clip(obs[:(2*putils.N_LIDAR_RAYS)][0::2] / putils.MAX_LIDAR_RANGE, 0.0, 1.0)
        # LIDAR bearings [0, 2*pi] -> [0, 1]:
        obs[:(2*putils.N_LIDAR_RAYS)][1::2] = np.clip(obs[:(2*putils.N_LIDAR_RAYS)][1::2] / (2*np.pi), 0.0, 1.0)
        # Forward velocity [-MAX_VX, MAX_VX] -> [-1, 1]:
        obs[(2*putils.N_LIDAR_RAYS) + 1] = np.clip(obs[(2*putils.N_LIDAR_RAYS) + 1] / putils.MAX_VX, -1.0, 1.0)
        # Steering velocity [-MAX_VX, MAX_VX] -> [-1, 1]:
        obs[(2*putils.N_LIDAR_RAYS) + 2] = np.clip(obs[(2*putils.N_LIDAR_RAYS) + 2] / putils.MAX_WX, -1.0, 1.0)
        # X distance to goal [-MAX_GOAL_DIST, MAX_GOAL_DIST] -> [-1, 1]:
        obs[(2*putils.N_LIDAR_RAYS) + 3] = np.clip(obs[(2*putils.N_LIDAR_RAYS) + 3] / putils.MAX_GOAL_DIST, -1.0, 1.0)
        # Y distance to goal [-MAX_GOAL_DIST, MAX_GOAL_DIST] -> [-1, 1]:
        obs[(2*putils.N_LIDAR_RAYS) + 4] = np.clip(obs[(2*putils.N_LIDAR_RAYS) + 4] / putils.MAX_GOAL_DIST, -1.0, 1.0)
        # Bearing to goal [-pi, pi] -> [-1, 1]
        obs[(2*putils.N_LIDAR_RAYS) + 5] = np.clip(obs[(2*putils.N_LIDAR_RAYS) + 5] / np.pi, -1.0, 1.0)
        return obs.astype(np.float32)

    def compute_reward(self, raw_obs: np.ndarray, status: str) -> Tuple[float, dict]:
        # Hybrid sparse + dense reward signal:
        reward = 0.0
        info   = {'status': status}
        # Sparse terminal rewards:
        if status == 'goal_reached':
            reward += putils.R_GOAL_REACHED
            info['reward'] = reward
        elif status == 'collision':
            reward += putils.R_COLLISION
            info['reward'] = reward
        else:
            # Dense step penalty encourages finding shorter, more direct paths:
            reward += putils.R_STEP_PENALTY
            # Dense progress reward is positive when closing distance to goal:
            curr_x = float(raw_obs[(2*putils.N_LIDAR_RAYS) + 3])
            curr_y = float(raw_obs[(2*putils.N_LIDAR_RAYS) + 4])
            curr_dist = float((self.curr_goal_x - curr_x)**2 + (self.curr_goal_y - curr_y)**2)
            progress = self.previous_dist_to_goal - curr_dist
            reward += putils.R_PROGRESS_SCALE * progress
            self.previous_dist_to_goal = curr_dist
            info['dist_to_goal'] = curr_dist
            # Dense smoothness penalty discourages unnecessary or jerky rotation:
            wz  = float(raw_obs[(2*putils.N_LIDAR_RAYS) + 2])
            reward += putils.R_SMOOTH_PENALTY * abs(wz)
            # Dense proximity penalty encourages maintaining clearance from walls:
            min_lidar_range = float(np.min(raw_obs[:(2*putils.N_LIDAR_RAYS)][0::2]))
            if min_lidar_range < putils.MIN_LIDAR_RANGE:
                reward += putils.R_PROXIMITY_SCALE * (1.0 - min_lidar_range)
            info['reward'] = reward
            info['progress'] = progress
            info['min_lidar'] = min_lidar
        return reward, info

def train():
    # TODO - need to define an entrypoint for instantiating the Nav2GymEnv in a training mode
    pass

def evaluate():
    # TODO - need to define an entrypoint for instantiating the Nav2GymEnv in an evaluation mode with loadable checkpoint
    pass

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='RL-based Nav2 local planner parameter tuner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--mode', choices=['train', 'eval'], default='train', help='Run training or evaluation of a saved model')
    p.add_argument('--planner',  choices=['dwb', 'mppi'],   default='dwb', help='Which Nav2 local planner to tune')
    p.add_argument('--steps',    type=int, default=500_000, help='Total environment steps for training')
    p.add_argument('--model',    type=str, default=None, help='Path to saved .zip model (required for --mode eval)')
    p.add_argument('--episodes', type=int, default=10, help='Number of evaluation episodes')
    p.add_argument('--log-dir',  type=str, default='./rl_logs', help='Directory for TensorBoard logs and Monitor CSVs')
    p.add_argument('--ckpt-dir', type=str, default='./rl_checkpoints', help='Directory for model checkpoints')
    return p

if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.mode == 'train':
        train(
            # TODO - placeholder
        )
    elif args.mode == 'eval':
        if args.model is None:
            build_parser().error("--model is required for --mode eval")
        evaluate(
            # TODO - placeholder
        )