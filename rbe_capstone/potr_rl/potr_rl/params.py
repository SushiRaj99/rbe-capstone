import numpy as np

# Path-cost features come from metrics_tracker.compute_path_costs — three
# arc-length windows along the global plan, each carrying the *max* Nav2
# costmap value observed in that window. 253 = lethal, 0 = free.
MAX_COSTMAP_COST = 254.0

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
    'goal_align_scale': (5.0,  40.0),  # DWB GoalAlign critic weight (Nav2 default 20-32)
    'path_align_scale': (5.0,  40.0),  # DWB PathAlign critic weight
    'goal_dist_scale':  (5.0,  40.0),  # DWB GoalDist critic weight
    'path_dist_scale':  (5.0,  40.0),  # DWB PathDist critic weight
    'obstacle_scale':   (0.005, 0.1),  # DWB BaseObstacle critic weight (preset 1 = 0.05)
}

# Preset-1 baselines.  Action = 0 maps to these values so the policy can always
# reproduce the hand-tuned preset; action ± 1 interpolates toward the range ends.
MPPI_BASELINES = {
    'max_linear_vel':  0.8,
    'max_angular_vel': 1.2,
    'linear_accel':    1.5,
    'angular_accel':   2.0,
}

DWB_BASELINES = {
    'max_linear_vel':   0.8,
    'max_angular_vel':  1.2,
    'linear_accel':     1.5,
    'angular_accel':    2.0,
    'goal_align_scale': 20.0,
    'path_align_scale': 32.0,
    'goal_dist_scale':  20.0,
    'path_dist_scale':  32.0,
    'obstacle_scale':   0.05,
}

PLANNER_PARAM_RANGES = {
    'MPPI': MPPI_PARAM_RANGES,
    'DWB':  DWB_PARAM_RANGES,
}

PLANNER_BASELINES = {
    'MPPI': MPPI_BASELINES,
    'DWB':  DWB_BASELINES,
}

# Legacy aliases
PARAM_RANGES      = MPPI_PARAM_RANGES
CONTINUOUS_PARAMS = list(MPPI_PARAM_RANGES.keys())

# Observation space (raw, pre-normalization)  [3 path-cost + 5 state = 8 dims]
# The lidar was dropped — it's robot-centric while the policy's job is
# path-centric ("is my planned route blocked?"). The three path_cost_* samples
# answer that directly by sampling the costmap along the upcoming plan.
path_cost_low  = np.zeros(3, dtype=np.float32)
path_cost_high = np.full(3, MAX_COSTMAP_COST, dtype=np.float32)
state_low  = np.array([0.0,  -np.pi, 0.0, 0.0, 0.0], dtype=np.float32)
state_high = np.array([50.0,  np.pi, 2.0, 5.0, 5.0], dtype=np.float32)

OBS_LOW  = np.concatenate([path_cost_low,  state_low])
OBS_HIGH = np.concatenate([path_cost_high, state_high])
# indices 0–2: path_cost_near, path_cost_mid, path_cost_far
# indices 3–7: dist_to_goal, heading_err, lin_vel, ang_vel, path_dev
# indices 8+:  smoothed action (planner-dependent length, appended in env.py)
# Note: env.py normalises the base (indices 0–7) to [-1, 1] using OBS_LOW/HIGH
# before returning; the smoothed-action suffix is already in [-1, 1] by construction.

# Reward weights — per-tick terms accumulate ~1200× per episode at action_freq=10.
# Shaped so that episode *duration* (which the policy can influence via
# max_vel/accel) dominates the variable part of the signal. The proximity term
# replaced a binary collision penalty so the policy gets a continuous gradient
# for "stay out of the inflation halo," not just "don't hit the wall."
REWARD = {
    'progress':     1.0,    # per-metre closer to goal
    'path_dev':    -0.02,   # per-metre lateral deviation per tick (small — guardrail only)
    'ang_vel':     -0.01,   # per rad/s per tick (small — guardrail only)
    'proximity':   -0.1,    # graduated: multiplied by max(0, path_cost_near-30)/100 per tick
    'time_step':   -0.2,    # per tick — dominant per-episode term; rewards faster completions
    'goal_bonus':   100.0,  # terminal: goal reached
    'fail_penalty': -50.0,  # terminal: episode ended without reaching goal
}

# EMA smoothing on continuous actions — policy sees raw, Nav2 sees smoothed (prevents MPPI reset storms).
# alpha is the weight on the *previous* smoothed action, so lower = more responsive.
# 0.3 means 70% of each new policy action reaches DWB immediately; still enough
# filtering to avoid reset storms, but not so much that the credit-assignment
# signal is buried under the previous step's action.
ACTION_EMA_ALPHA = 0.3
