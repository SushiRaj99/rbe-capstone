import numpy as np

# Lidar constants — must match metrics_tracker.py
N_LIDAR_RAYS    = 18    # one ray every 20° around 360°
MAX_LIDAR_RANGE = 10.0  # metres

# Discrete action space: (planner, preset) indexed by action int
DISCRETE_CONFIGS = [
    ('DWB',  1),
    ('DWB',  2),
    ('MPPI', 1),
    ('MPPI', 2),
]

# Continuous action space — names must match SHARED_PARAM_MAP in planner_controller.py
# Action vector normalised to [-1, 1]: val = lo + (a + 1) / 2 * (hi - lo)
MPPI_PARAM_RANGES = {
    'max_linear_vel':  (0.1, 1.5),   # m/s
    'max_angular_vel': (0.3, 2.5),   # rad/s
    'linear_accel':    (0.5, 3.5),   # m/s²
    'angular_accel':   (0.5, 4.0),   # rad/s²
}

DWB_PARAM_RANGES = {
    'max_linear_vel':   (0.1, 1.5),   # m/s
    'max_angular_vel':  (0.3, 2.5),   # rad/s
    'linear_accel':     (0.50, 3.50),  # m/s²
    'angular_accel':    (0.50, 4.00),  # rad/s²
    'goal_align_scale': (1.0,  50.0),  # DWB GoalAlign critic weight
    'path_align_scale': (1.0,  50.0),  # DWB PathAlign critic weight
    'goal_dist_scale':  (1.0,  50.0),  # DWB GoalDist critic weight
    'path_dist_scale':  (1.0,  50.0),  # DWB PathDist critic weight
}

PLANNER_PARAM_RANGES = {
    'MPPI': MPPI_PARAM_RANGES,
    'DWB':  DWB_PARAM_RANGES,
}

# Legacy aliases
PARAM_RANGES      = MPPI_PARAM_RANGES
CONTINUOUS_PARAMS = list(MPPI_PARAM_RANGES.keys())

# Observation space  [N_LIDAR_RAYS + 7 scalar fields = 25 dims]
lidar_low  = np.zeros(N_LIDAR_RAYS, dtype=np.float32)
lidar_high = np.full(N_LIDAR_RAYS, MAX_LIDAR_RANGE, dtype=np.float32)
state_low  = np.array([0.0,  -np.pi, 0.0, 0.0, 0.0,  0.0, 0.0], dtype=np.float32)
state_high = np.array([50.0,  np.pi, 2.0, 5.0, 10.0, 5.0, 1.0], dtype=np.float32)

OBS_LOW  = np.concatenate([lidar_low,  state_low])
OBS_HIGH = np.concatenate([lidar_high, state_high])
# indices 0–17: lidar ranges; 18–24: dist_to_goal, heading_err, lin_vel, ang_vel, clearance, path_dev, collision
# indices 25+: smoothed action (planner-dependent length, appended in env.py)

# Reward weights
REWARD = {
    'progress':     1.0,    # per-metre closer to goal
    'path_dev':    -0.1,    # per-metre lateral deviation
    'ang_vel':     -0.05,   # per rad/s (penalise excessive spinning)
    'collision':   -10.0,   # per step where collision=True
    'goal_bonus':   100.0,  # terminal: goal reached
    'fail_penalty': -50.0,  # terminal: episode ended without reaching goal
}

# EMA smoothing on continuous actions — policy sees raw, Nav2 sees smoothed (prevents MPPI reset storms)
ACTION_EMA_ALPHA = 0.7
