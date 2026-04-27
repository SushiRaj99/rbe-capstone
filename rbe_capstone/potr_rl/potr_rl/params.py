"""Action / observation / reward configuration for the SAC agent."""
from dataclasses import dataclass

import numpy as np

# Costmap convention: 0 = free, ~30 = inflation begins, 99 = inscribed, 254 = lethal.
MAX_COSTMAP_COST = 254.0

# Clip path-cost observations at the inscribed boundary. Anything above 100 is
# already "blocked" from the policy's point of view, so wider dynamic range
# would only waste resolution on a distinction the policy can't act on.
PATH_COST_OBS_CLIP = 100.0

# Discrete action space: each int picks a (planner, preset) pair.
DISCRETE_CONFIGS = [
    ('DWB', 1),
    ('DWB', 2),
    ('MPPI', 1),
    ('MPPI', 2),
]

# Continuous action ranges. Param names must match SHARED_PARAM_MAP in
# planner_controller.py so the bridge knows how to translate them into the
# right ROS parameter names at runtime.
MPPI_PARAM_RANGES = {
    'max_linear_vel': (0.1, 1.5),
    'max_angular_vel': (0.3, 2.5),
    'linear_accel': (0.5, 3.5),
    'angular_accel': (0.5, 4.0),
}

DWB_PARAM_RANGES = {
    # Linear speed cap. Upper bound matches max_speed_xy in nav2_params.yaml.
    'max_linear_vel': (0.4, 1.5),
    'obstacle_scale': (0.005, 0.1),
    # path_weight and goal_weight are meta-params: each one multiplicatively
    # scales two of DWB's four critic scales together. path_align and path_dist
    # are paired because they encode the same intent (stay on the global plan);
    # goal_align and goal_dist are paired for the same reason. This halves the
    # action-space dimensionality vs. tuning all four scales independently.
    'path_weight': (0.7, 1.5),
    'goal_weight': (0.7, 1.5),
}

# Reference values for path_weight=1.0 / goal_weight=1.0. These match preset_1's
# hand-tuned scales from dwb_params.yaml; the policy's chosen weight is then
# multiplied through to the four ROS params at apply-action time.
DWB_META_REFERENCE = {
    'path_weight': {
        'path_align_scale': 32.0,
        'path_dist_scale': 32.0,
    },
    'goal_weight': {
        'goal_align_scale': 20.0,
        'goal_dist_scale': 20.0,
    },
}

MPPI_BASELINES = {
    'max_linear_vel': 0.8,
    'max_angular_vel': 1.2,
    'linear_accel': 1.5,
    'angular_accel': 2.0,
}

DWB_BASELINES = {
    'max_linear_vel': 0.8,
    'obstacle_scale': 0.05,
    'path_weight': 1.0,
    'goal_weight': 1.0,
}

PLANNER_PARAM_RANGES = {
    'MPPI': MPPI_PARAM_RANGES,
    'DWB': DWB_PARAM_RANGES,
}

PLANNER_BASELINES = {
    'MPPI': MPPI_BASELINES,
    'DWB': DWB_BASELINES,
}

# Observation bounds: 3 path-cost windows (near/mid/far) + 5 state dims
# (distance_to_goal, heading_error, linear_velocity, angular_velocity, path_deviation).
path_cost_low = np.zeros(3, dtype=np.float32)
path_cost_high = np.full(3, PATH_COST_OBS_CLIP, dtype=np.float32)
state_low = np.array([0.0, -np.pi, 0.0, 0.0, 0.0], dtype=np.float32)
state_high = np.array([50.0, np.pi, 2.0, 5.0, 5.0], dtype=np.float32)

OBS_LOW = np.concatenate([path_cost_low, state_low])
OBS_HIGH = np.concatenate([path_cost_high, state_high])

@dataclass
class RewardConfig:
    """
    Reward weights. The terminal time_delta_weight term adds
    weight * (baseline_time[goal_id] - actual_time) at episode end, which
    centers the time signal around zero across episodes instead of letting
    it grow with path length.
    """
    progress: float = 1.0
    path_dev: float = -0.02
    ang_vel: float = -0.01
    proximity: float = -0.1
    collision: float = -2.0
    # Per-step penalty for cruising slowly when there's no good reason - i.e.
    # below 0.4 m/s while well away from the goal and not in inflation. This
    # rewards pushing speed in open stretches but does not punish legitimate
    # slowdowns at corners or near the goal.
    slow_pace: float = -0.5
    goal_bonus: float = 100.0
    fail_penalty: float = -50.0
    time_delta_weight: float = 2.0


REWARD = RewardConfig()

# EMA smoothing factor on the action vector. Damps rapid swings before they
# reach Nav2 so the controller doesn't see whiplash parameter changes.
ACTION_EMA_ALPHA: float = 0.3
