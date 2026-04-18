import numpy as np

# ---------------------------------------------------------------------------
# Discrete action space
# ---------------------------------------------------------------------------
# Each entry is (planner, preset).  The index into this list is the action.
DISCRETE_CONFIGS = [
    ('DWB',  1),
    ('DWB',  2),
    ('MPPI', 1),
    ('MPPI', 2),
]

# ---------------------------------------------------------------------------
# Continuous action space
# ---------------------------------------------------------------------------
# Logical parameter names (must match keys in SHARED_PARAM_MAP in
# planner_controller.py) and their (min, max) physical ranges.
PARAM_RANGES = {
    'max_linear_vel':  (0.1, 1.5),   # m/s
    'max_angular_vel': (0.3, 2.5),   # rad/s
    'linear_accel':    (0.5, 3.5),   # m/s²
    'angular_accel':   (0.5, 4.0),   # rad/s²
}
CONTINUOUS_PARAMS = list(PARAM_RANGES.keys())

# Action vector is normalised to [-1, 1]; map to physical values with:
#   val = lo + (a + 1) / 2 * (hi - lo)
PARAM_LOW  = np.array([v[0] for v in PARAM_RANGES.values()], dtype=np.float32)
PARAM_HIGH = np.array([v[1] for v in PARAM_RANGES.values()], dtype=np.float32)

# ---------------------------------------------------------------------------
# Observation space bounds  [7 fields matching StepMetrics.msg]
# ---------------------------------------------------------------------------
OBS_LOW  = np.array([0.0,  -np.pi, 0.0, 0.0, 0.0,  0.0, 0.0], dtype=np.float32)
OBS_HIGH = np.array([50.0,  np.pi, 2.0, 5.0, 10.0, 5.0, 1.0], dtype=np.float32)
# Fields: dist_to_goal, heading_err, lin_vel, ang_vel, clearance, path_dev, collision

# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------
REWARD = {
    'progress':     1.0,    # per-metre closer to goal
    'path_dev':    -0.1,    # per-metre lateral deviation
    'ang_vel':     -0.05,   # per rad/s (penalise excessive spinning)
    'collision':   -10.0,   # per step where collision=True
    'goal_bonus':   100.0,  # terminal: goal reached
    'fail_penalty': -50.0,  # terminal: episode ended without reaching goal
}
