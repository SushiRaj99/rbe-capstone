#!/usr/bin/env python3
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import Empty
from rcl_interfaces.srv import SetParameters

import rl_pipeline.pipeline_utils as putils

from typing import Optional, Dict, Tuple, List
import threading    # shouldn't be problematic for rclpy since the threading should only occur inside nodes within a MultiThreadedExecutor
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env

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

    def collect_observation(self, msg: Float32MultiArray) -> None:
        with self.obs_lock:
            self.latest_obs = np.array(msg.data, dtype=np.float32)
        self.obs_event.set()    # unblocks any calling method using RLBridgeNode.wait_for_observation() (e.g. Nav2GymEnv.step() or Nav2GymEnv.reset())

    def collect_status(self, msg: String) -> None:
        self.status = msg.data

    def wait_for_observation(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        obs = None
        self.obs_event.clear()  # effectively blocks logic until RLBridgeNode.collect_observation() sets the lock, indicating that a fresh observation was collected
        if not self.obs_event.wait(timeout=timeout):
            self.get_logger().warn("Timeout waiting for observation.")
        with self.obs_lock:
            obs = self.latest_obs.copy() if self.latest_obs is not None else None
        return obs

    def get_status(self) -> str:
        return self.status

    def call_reset(self, config: Dict) -> bool:
        # Pushes episode configuration to the planner_config_manager and calls its rl/reset service. This method effectively 
        # allows the gym environment to call the /rl/reset service from within its native reset() method:
        episode_params = [
            putils.make_rclparam_string('goal_id', config['goal_id']),
            putils.make_rclparam_double('start_x', config['start_x']),
            putils.make_rclparam_double('start_y', config['start_y']),
            putils.make_rclparam_double('start_yaw', config['start_yaw']),
            putils.make_rclparam_double('goal_x', config['goal_x']),
            putils.make_rclparam_double('goal_y', config['goal_y']),
            putils.make_rclparam_double('goal_yaw', config['goal_yaw']),
            putils.make_rclparam_double('xy_tolerance', config['xy_tolerance']),
            putils.make_rclparam_double('yaw_tolerance', config['yaw_tolerance']),
            putils.make_rclparam_string('map_filepath', config['map_filepath'])
        ]
        success = False
        # First, ensure that the episode configuration is updated:
        if self.config_param_client.wait_for_service(timeout_sec=3.0):
            success = True
            request = SetParameters.Request()
            request.parameters = episode_params
            future = self.config_param_client.call_async(request)
            putils.spin_wait_for_future(future, timeout=3.0)
        else:
            self.get_logger().warn(f"planner_config_node/set_parameters not available, so episode configuration for {config['goal_id']} will not be applied.")
        # Finally, trigger the actual episode reset:
        if not self.reset_client.wait_for_service(timeout_sec=3.0):
            success = False
            self.get_logger().error("/rl/reset service not available.")
        elif (success):
            future = self.reset_client.call_async(Empty.Request())
            putils.spin_wait_for_future(future, timeout=5.0)
        return success

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
        self.episode_cnt = 0
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
        config['goal_id'] = f"episode_{self.episode_cnt}"   # the EPISODE_MAP_CONFIGS intentionally doesn't have a 'goal_id' parameter
        self.episode_cnt += 1
        # Pre-compute initial distance to goal for the dense reward for progress:
        self.previous_dist_to_goal = np.sqrt(
            (config['goal_x'] - config['start_x'])**2 + (config['goal_y'] - config['start_y'])**2
        )
        self.curr_goal_x = config['goal_x']
        self.curr_goal_y = config['goal_y']
        # 'Propagate' reset to ROS2 simulation (map swap + pose reset + new goal) and wait for the first 
        # observation from the new episode:
        raw_observation = None
        reset_applied = self.bridge.call_reset(config)
        if (reset_applied): raw_observation = self.bridge.wait_for_observation(timeout=10.0)    # allow good chunk of time for map load
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
                reward += putils.R_PROXIMITY_SCALE * (putils.MIN_LIDAR_RANGE - min_lidar_range)
            info['reward'] = reward
            info['progress'] = progress
            info['min_lidar'] = min_lidar
        return reward, info

def train(
    planner_type: str = 'dwb',
    num_steps: int = 500_000,
    net_arch: List[int] = [256, 256],
    lr: float = 3e-4,
    rollout_buffer: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    log_dir: str = './rl_logs',
    checkpoint_dir: str = './rl_checkpoints'
) -> None:
    # Train a PPO agent with Stable-Baselines3 and the Monitor wrapper, which logs per-episode reward and length to csv files 
    # to log_dir (Tensorboard should be able to support csv files).
    env = Monitor(Nav2GymEnv(planner_type=planner_type), log_dir)
    check_env(env, warn=True)
    checkpoint_cb = CheckpointCallback(
        save_freq = min(10_000, int(0.02*num_steps)),
        save_path = checkpoint_dir,
        name_prefix = f"ppo_nav2_{planner_type}",
        verbose = 1
    )
    policy_kwargs = dict(
        # NOTE - the default activation functions between hidden layers for PPO is torch.nn.Tanh() in an MlpPolicy. If you want to switch 
        # this to ReLU, add 'activation_fn=torch.nn.ReLU' to the policy_kwargs dictionary.
        net_arch=dict(pi=net_arch.copy(), vf=net_arch.copy())
    )
    model = PPO(
        policy = "MlpPolicy",
        env = env,
        learning_rate = lr,
        n_steps = rollout_buffer,
        batch_size = batch_size,
        n_epochs = n_epochs,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = 0.20,
        ent_coef = 0.01,
        vf_coef = 0.50,
        policy_kwargs = policy_kwargs,
        verbose = 1,
        tensorboard_log = log_dir
    )
    print(
        f"Starting PPO training\n"
        f"\tplanner    : {planner_type}\n"
        f"\ttotal_steps: {total_steps:,}\n"
        f"\tlog_dir    : {log_dir}\n"
        f"\tcheckpoints: {checkpoint_dir}"
    )
    model.learn(total_timesteps=num_steps, callback=checkpoint_cb)
    save_path = f"{checkpoint_dir}/ppo_nav2_{planner_type}_final"
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}.zip")
    env.close()

def evaluate_model(
    model_path: str,
    planner_type: str = 'dwb',
    num_episodes: int = 10
) -> None:
    # TODO - need to define an entrypoint for instantiating the Nav2GymEnv with the PPO agent model loaded from a checkpoint in evaluation (deterministic) mode. 
    # The rewards per episode should be printed or logged for comparing against the standard planner.
    env = Nav2GymEnv(planner_type=planner_type)
    model = PPO.load(model_path, env=env)
    # TODO - need to fill in the blanks here...
    pass

def evaluate_baseline(
    planner_type: str = 'dwb',
    num_episodes: int = 10
) -> None:
    # TODO - need to define entrypoint for instantiating the Nav2GymEnv to operate with a standard planner. The rewards per episode should be printed or logged 
    # for comparing against the "RL augmented" planner. 
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
        evaluate_model(
            # TODO - placeholder
        )
        evaluate_baseline(
            # TODO - placeholder
        )