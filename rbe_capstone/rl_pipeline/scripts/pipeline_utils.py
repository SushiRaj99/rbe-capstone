#!/usr/bin/env python3
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rclpy.task import Future

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