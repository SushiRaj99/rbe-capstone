import numpy as np

MAX_COSTMAP_COST = 254.0

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
    'max_linear_vel': (0.1, 1.5),
    'max_angular_vel': (0.3, 2.5),
    'linear_accel': (0.5, 3.5),
    'angular_accel': (0.5, 4.0),
    'goal_align_scale': (5.0, 40.0),
    'path_align_scale': (5.0, 40.0),
    'goal_dist_scale': (5.0, 40.0),
    'path_dist_scale': (5.0, 40.0),
    'obstacle_scale': (0.005, 0.1),
}

MPPI_BASELINES = {
    'max_linear_vel': 0.8,
    'max_angular_vel': 1.2,
    'linear_accel': 1.5,
    'angular_accel': 2.0,
}

DWB_BASELINES = {
    'max_linear_vel': 0.8,
    'max_angular_vel': 1.2,
    'linear_accel': 1.5,
    'angular_accel': 2.0,
    'goal_align_scale': 20.0,
    'path_align_scale': 32.0,
    'goal_dist_scale': 20.0,
    'path_dist_scale': 32.0,
    'obstacle_scale': 0.05,
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
path_cost_high = np.full(3, MAX_COSTMAP_COST, dtype=np.float32)
state_low = np.array([0.0, -np.pi, 0.0, 0.0, 0.0], dtype=np.float32)
state_high = np.array([50.0, np.pi, 2.0, 5.0, 5.0], dtype=np.float32)

OBS_LOW = np.concatenate([path_cost_low, state_low])
OBS_HIGH = np.concatenate([path_cost_high, state_high])

REWARD = {
    'progress': 1.0,
    'path_dev': -0.02,
    'ang_vel': -0.01,
    'proximity': -0.1,
    'time_step': -0.2,
    'goal_bonus': 100.0,
    'fail_penalty': -50.0,
}

ACTION_EMA_ALPHA = 0.3
