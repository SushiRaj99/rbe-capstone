#!/usr/bin/env python3
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rclpy.task import Future
import time     # using this module shouldn't be problematic if all nodes are using MultiThreadedExecutors

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

N_LIDAR_RAYS = 18                                   # 360° / 20° downsample = 18 rays
N_STATE_VARS = 5                                    # vx, wz, dist_to_goal_x, dist_to_goal_y, bearing_to_goal
OBSERVATION_DIMS = 2*N_LIDAR_RAYS + N_STATE_VARS    # 40 for now
NUM_ACTIONS = 6                                     # tunable planner parameters — must match len(PLANNER_PARAM_BOUNDS[planner])

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