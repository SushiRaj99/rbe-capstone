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
OBSERVATION_DIMS = 2*N_LIDAR_RAYS + N_STATE_VARS    # 41 for now
NUM_ACTIONS = 6                                     # tunable planner parameters — must match len(PLANNER_PARAM_BOUNDS[planner])

# Episode configuration:
MAPS_DIR = os.path.expanduser('~/ws/src/rbe_capstone/simulation_launch/maps')
MAP_CONFIGS = {

    os.path.join(MAPS_DIR, 'mixed', 'mixed.yaml'): [
        dict(start_x=-4.00, start_y=0.00, start_yaw=0.0, goal_x=4.27, goal_y=-8.70, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-4.00, start_y=0.00, start_yaw=0.0, goal_x=-11.37, goal_y=8.41, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=2.10, start_y=-6.08, start_yaw=0.0, goal_x=-11.37, goal_y= 8.41, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=4.14, start_y=6.50, start_yaw=0.0, goal_x=-2.04, goal_y=-7.39, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-2.04, start_y=-7.39, start_yaw=0.0, goal_x=-1.92, goal_y=7.39, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-1.92, start_y=7.39, start_yaw=0.0, goal_x=-8.12, goal_y=-8.05, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=3.13, start_y=-9.71, start_yaw=0.0, goal_x=-5.62, goal_y= 8.41, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-11.37, start_y=8.41, start_yaw=0.0, goal_x=3.13, goal_y=-9.71, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=4.27, start_y=-8.70, start_yaw=0.0, goal_x=-11.37, goal_y=8.41, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5), 
        dict(start_x=-4.00, start_y=0.00, start_yaw=0.0, goal_x=4.14, goal_y=6.50, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-8.12, start_y=-8.05, start_yaw=0.0, goal_x=4.14, goal_y=6.50, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-5.62, start_y=8.41, start_yaw=0.0, goal_x=2.10, goal_y=-6.08, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5)
    ],
    os.path.join(MAPS_DIR, 'narrow_aisles', 'narrow_aisles.yaml'): [
        dict(start_x=-3.30, start_y=-2.17, start_yaw=0.0, goal_x=-3.52, goal_y=7.03, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-3.30, start_y=-2.17, start_yaw=0.0, goal_x=-3.52, goal_y=7.03, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-3.30, start_y=-2.17, start_yaw=0.0, goal_x=6.80, goal_y=8.67, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-3.30, start_y=-2.17, start_yaw=0.0, goal_x=-3.52, goal_y=7.03, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=0.72, start_y=-6.51, start_yaw=0.0, goal_x=0.72, goal_y=8.07, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=4.96, start_y=-7.82, start_yaw=0.0, goal_x=-3.52, goal_y=7.03, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=6.80, start_y=8.67, start_yaw=0.0, goal_x=4.96, goal_y=-7.82, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-7.66, start_y=0.75, start_yaw=0.0, goal_x=3.84, goal_y=-3.73, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-3.45, start_y=8.22, start_yaw=0.0, goal_x=0.72, goal_y=-6.51, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-5.30, start_y=2.88, start_yaw=0.0, goal_x=6.80, goal_y=8.67, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-1.40, start_y=3.40, start_yaw=0.0, goal_x=4.96, goal_y=-7.82, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-3.15, start_y=0.40, start_yaw=0.0, goal_x=0.72, goal_y=8.07, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=0.72, start_y=8.07, start_yaw=0.0, goal_x=-7.66, goal_y=0.75, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x= 3.84, start_y=-3.73, start_yaw=0.0, goal_x=-3.45, goal_y=8.22, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
    ],
    os.path.join(MAPS_DIR, 'dense_clutter', 'dense_clutter.yaml'): [
        dict(start_x=-5.75, start_y=-6.75, start_yaw=0.0, goal_x=4.65, goal_y=3.20, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-6.47, start_y=4.74, start_yaw=0.0, goal_x=5.34, goal_y=-6.58, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=5.34, start_y=-6.58, start_yaw=0.0, goal_x=-6.47, goal_y=4.74, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-4.66, start_y=3.64, start_yaw=0.0, goal_x=2.95, goal_y=-4.68, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=1.06, start_y=4.63, start_yaw=0.0, goal_x=-5.75, goal_y=-6.75, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=1.11, start_y=-5.14, start_yaw=0.0, goal_x=-4.66, goal_y=3.64, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=0.99, start_y=-4.81, start_yaw=0.0, goal_x=4.65, goal_y=3.20, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-6.78, start_y=-1.83, start_yaw=0.0, goal_x=4.65, goal_y=3.20, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=3.16, start_y=-0.63, start_yaw=0.0, goal_x=-6.47, goal_y=4.74, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-5.07, start_y=3.78, start_yaw=0.0, goal_x=5.34, goal_y=-6.58, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=1.67, start_y=0.11, start_yaw=0.0, goal_x=-6.47, goal_y=4.74, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-5.98, start_y=-6.67, start_yaw=0.0, goal_x=5.97, goal_y=5.86, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=5.97, start_y=5.86, start_yaw=0.0, goal_x=-6.68, goal_y=-3.44, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=2.40, start_y= -2.02, start_yaw=0.0, goal_x=-6.08, goal_y=-0.57, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-2.30, start_y=-4.80, start_yaw=0.0, goal_x=4.65, goal_y=3.20, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-6.47, start_y=4.74, start_yaw=0.0, goal_x=2.40, goal_y=-2.02, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-6.08, start_y=-0.57, start_yaw=0.0, goal_x=5.34, goal_y=-6.58, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-1.50, start_y=-4.56, start_yaw=0.0, goal_x=5.97, goal_y=5.86, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-5.98, start_y=-6.67, start_yaw=0.0, goal_x=-1.51, goal_y=5.90, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=4.65, start_y=3.20, start_yaw=0.0, goal_x=-6.47, goal_y=4.74, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-1.51, start_y=5.90, start_yaw=0.0, goal_x=2.95, goal_y=-4.68, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-1.51, start_y=5.90, start_yaw=0.0, goal_x=5.34, goal_y=-6.58, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-6.68, start_y=-3.44, start_yaw=0.0, goal_x=5.34, goal_y= -6.58, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=3.16, start_y=-0.63, start_yaw=0.0, goal_x=5.97, goal_y=5.86, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=-2.30, start_y=-4.80, start_yaw=0.0, goal_x=-1.51, goal_y=5.90, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5),
        dict(start_x=1.67, start_y=0.11, start_yaw=0.0, goal_x=-6.08, goal_y=-0.57, goal_yaw=0.0, xy_tolerance=0.35, yaw_tolerance=0.5)
    ]
}
EPISODE_MAP_CONFIGS = [ 
    # This performs in place expansion of each pose dictionary within a given map in order to add the 
    # respective map_filepath to a new dictionary representing the unique episode configuration. So this implies 
    # that when this flattened episode configuration list is sampled on a uniform distribution, maps with more 
    # poses are more likely to be loaded on a per episode basis: 
    {**poses, 'map_filepath': map_filepath} for map_filepath, pose_list in MAP_CONFIGS.items() for poses in pose_list
]

# Collision Threshold (defines min lidar distance that triggers a 'collision')
LIDAR_COLLISION_RANGE = 0.10

# Max values used for PPO input/output normalization:   # TODO - using a bunch of placeholder values for now
MAX_LIDAR_RANGE = 10.0  # meters (must match lidar_model range_max)
MIN_LIDAR_RANGE = 1.0   # meters (used to penalize ranges approaching LIDAR_COLLISION RANGE)  # TODO - may need to ensure collision_monitor is disabled for this
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

def make_rclparam_string(name: str, value: str) -> Parameter:
    rclparam = Parameter()
    rclparam.name = name
    rclparam.value = ParameterValue()
    rclparam.value.type = ParameterType.PARAMETER_STRING
    rclparam.value.string_value = str(value)
    return rclparam

# Helper function to spin-wait for a ROS2 future from a non-executuor thread without blocking 
# waiting node's executor:
def spin_wait_for_future(future: Future, timeout: float = 2.0):
    timeout_nanos = int(1e9 * timeout)
    timelimit = rclpy.time.Time().nanoseconds + timeout_nanos
    while not future.done() and rclpy.time.Time().nanoseconds < timelimit:
        time.sleep(0.005)