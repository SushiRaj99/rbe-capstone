import numpy as np

MAX_COSTMAP_COST = 254.0

# Clip path-cost obs at inscribed; costs above this are all "blocked" and wasting
# obs dynamic range on a lethal/non-lethal distinction the policy can't use.
PATH_COST_OBS_CLIP = 100.0

# (planner, preset) indexed by action int
DISCRETE_CONFIGS = [
    ('DWB', 1),
    ('DWB', 2),
    ('MPPI', 1),
    ('MPPI', 2),
]

# Continuous action ranges - names must match SHARED_PARAM_MAP in planner_controller.py
MPPI_PARAM_RANGES = {
    'max_linear_vel': (0.1, 1.5),
    'max_angular_vel': (0.3, 2.5),
    'linear_accel': (0.5, 3.5),
    'angular_accel': (0.5, 4.0),
}

DWB_PARAM_RANGES = {
    # Speed cap - the only kinematic dim the policy meaningfully exercises.
    # Upper bound matches max_speed_xy in nav2_params.yaml boot.
    'max_linear_vel': (0.4, 1.5),
    'obstacle_scale': (0.005, 0.1),
    # Meta-params: multiplicative scaling on the four DWB critic scales, since
    # path_align/path_dist and goal_align/goal_dist track the same intent within
    # each pair. Halves the action-space dim count vs. tuning all four separately.
    'path_weight': (0.7, 1.5),
    'goal_weight': (0.7, 1.5),
}

# What path_weight=1.0 / goal_weight=1.0 decode to in the four actual ROS params.
# Matches preset_1's hand-tuned values from dwb_params.yaml.
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

# Observation bounds: 3 path-cost + 5 state dims
path_cost_low = np.zeros(3, dtype=np.float32)
path_cost_high = np.full(3, PATH_COST_OBS_CLIP, dtype=np.float32)
state_low = np.array([0.0, -np.pi, 0.0, 0.0, 0.0], dtype=np.float32)
state_high = np.array([50.0, np.pi, 2.0, 5.0, 5.0], dtype=np.float32)

OBS_LOW = np.concatenate([path_cost_low, state_low])
OBS_HIGH = np.concatenate([path_cost_high, state_high])

# time_delta_weight replaces the old per-tick time_step: terminal reward adds
# weight * (baseline_time[goal_id] - actual_time) so the signal is centered on
# zero across episodes instead of scaling with path length.
REWARD = {
    'progress': 1.0,
    'path_dev': -0.02,
    'ang_vel': -0.01,
    'proximity': -0.1,
    'collision': -2.0,
    # Per-step penalty for being slow when there's no good reason - cruising at
    # below 0.4 m/s while still well away from the goal and not in inflation.
    # Encourages the policy to push speed in open stretches without punishing
    # legitimate slowdowns at corners or near goal.
    'slow_pace': -0.5,
    'goal_bonus': 100.0,
    'fail_penalty': -50.0,
    'time_delta_weight': 2.0,
}

ACTION_EMA_ALPHA = 0.3
