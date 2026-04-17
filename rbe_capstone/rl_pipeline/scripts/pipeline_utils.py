#!/usr/bin/env python3
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rclpy.task import Future
import time, os     # using this module shouldn't be problematic if all nodes are using MultiThreadedExecutors

# Since Stable-Baselines3's PPO agent applies tanh activation to the output layer, actions 
# must be normalized/de-normalized when interfacing between the RL pipeline and Nav2 stack. 
# This will be handled with a list of tuples characterized by: 
#   (controller_server.param_name, min_value, max_value):
DWB_PARAM_BOUNDS = [
    # Velocity limits
    ("FollowPath.max_vel_x",           0.05,  0.50),
    ("FollowPath.min_speed_xy",         0.00,  0.20),
    # DWB critic scales — higher scale = stronger influence on path selection
    ("FollowPath.GoalAlign.scale",      1.00, 50.00),
    ("FollowPath.PathAlign.scale",      1.00, 50.00),
    ("FollowPath.GoalDist.scale",       1.00, 50.00),
    ("FollowPath.PathDist.scale",       1.00, 50.00),
]
PLANNER_PARAM_BOUNDS = {"dwb": DWB_PARAM_BOUNDS}    # placeholder to add MPPI later

# Constants for the state vector and action space:
N_LIDAR_RAYS = 18                                   # 360° / 20° downsample = 18 rays
N_STATE_VARS = 5                                    # vx, wz, dist_to_goal_x, dist_to_goal_y, bearing_to_goal
OBSERVATION_DIMS = 2*N_LIDAR_RAYS + N_STATE_VARS    # 40 for now
NUM_ACTIONS = 6                                     # tunable planner parameters — must match len(PLANNER_PARAM_BOUNDS[planner])

# Episode configuration:
MAPS_DIR = os.path.expanduser('~/ws/src/simulation_launch/maps')
MAP_CONFIGS = { # TODO - the start and goal point configs within each map are placeholders that need to be updated!
    os.path.join(MAPS_DIR, 'corridors', 'corridors.yaml'): [
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=1.0, goal_y=1.0, goal_yaw=1.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=2.0, goal_y=2.0, goal_yaw=2.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=3.0, goal_y=3.0, goal_yaw=3.0)
    ],
    os.path.join(MAPS_DIR, 'dense_clutter', 'dense_clutter.yaml'): [
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=1.0, goal_y=1.0, goal_yaw=1.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=2.0, goal_y=2.0, goal_yaw=2.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=3.0, goal_y=3.0, goal_yaw=3.0)
    ],
    os.path.join(MAPS_DIR, 'mixed', 'mixed.yaml'): [
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=1.0, goal_y=1.0, goal_yaw=1.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=2.0, goal_y=2.0, goal_yaw=2.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=3.0, goal_y=3.0, goal_yaw=3.0)
    ],
    os.path.join(MAPS_DIR, 'narrow_aisles', 'narrow_aisles.yaml'): [
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=1.0, goal_y=1.0, goal_yaw=1.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=2.0, goal_y=2.0, goal_yaw=2.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=3.0, goal_y=3.0, goal_yaw=3.0)
    ],
    os.path.join(MAPS_DIR, 'open_cluttered', 'open_cluttered.yaml'): [
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=1.0, goal_y=1.0, goal_yaw=1.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=2.0, goal_y=2.0, goal_yaw=2.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=3.0, goal_y=3.0, goal_yaw=3.0)
    ],
    os.path.join(MAPS_DIR, 'warehouse', 'warehouse.yaml'): [
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=1.0, goal_y=1.0, goal_yaw=1.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=2.0, goal_y=2.0, goal_yaw=2.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=3.0, goal_y=3.0, goal_yaw=3.0)
    ],
    os.path.join(MAPS_DIR, 'wide_aisles', 'wide_aisles.yaml'): [
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=1.0, goal_y=1.0, goal_yaw=1.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=2.0, goal_y=2.0, goal_yaw=2.0),
        dict(start_x=0.0, start_y=0.0, start_yaw=0.0, goal_x=3.0, goal_y=3.0, goal_yaw=3.0)
    ],
}
EPISODE_MAP_CONFIGS = [ 
    # This performs in place expansion of each pose dictionary within a given map in order to add the 
    # respective map_filepath to a new dictionary representing the unique episode configuration. So this implies 
    # that when this flattened episode configuration list is sampled on a uniform distribution, maps with more 
    # poses are more likely to be loaded on a per episode basis: 
    {**poses, 'map_filepath': map_filepath} for map_filepath, pose_list in MAP_CONFIGS.items() for poses in pose_list
]

# Max values used for PPO input/output normalization:   # TODO - using a bunch of placeholder values for now
MAX_LIDAR_RANGE = 10.0  # meters (must match lidar_model range_max)
MIN_LIDAR_RANGE = 1.0   # meters (used to penalize ranges approaching collision)  # TODO - may need to ensure collision_monitor is disabled for this
MAX_VX = 0.50           # m/s
MAX_WZ = 1.00           # rad/s
MAX_GOAL_DIST = 15.0    # meters (assumed upper bound on goal distance)

# Reward scaling:   # TODO - these are not set in stone
R_GOAL_REACHED = +100.0
R_COLLISION = -100.0
R_STEP_PENALTY = -0.01
R_PROGRESS_SCALE = +10.0  # reward = R_PROGRESS_SCALE * (prev_dist - curr_dist)
R_SMOOTH_PENALTY = -0.01  # penalty = R_SMOOTH_PENALTY * |wz|
R_PROXIMITY_SCALE = -1.0  # penalty = R_PROXIMITY_SCALE * max(0, 1 - min_lidar)

# Helper function to generate an RCL PARAMETER_DOUBLE from a provided name and value:
def make_rclparam_double(name: str, value: float) -> Parameter:
    rclparam = Parameter()
    rclparam.name = name
    rclparam.value = ParameterValue()
    rclparam.value.type = ParameterType.PARAMETER_DOUBLE
    rclparam.value.double_value = float(value)  # additional type protection
    return rclparam

# Helper function to spin-wait for a ROS2 future from a non-executuor thread without blocking 
# waiting node's executor:
def spin_wait_for_future(future: Future, timeout: float = 2.0):
    timeout_nanos = int(1e9 * timeout)
    timelimit = rclpy.time.Time().nanoseconds + timeout_nanos
    while not future.done() and rclpy.time.Time().nanoseconds < timelimit:
        time.sleep(0.005)