# Per-step observation published by metrics_tracker at every odom tick

float64[18] lidar_rays            # 18 evenly-spaced range readings (0°…340°, step 20°), clipped to range_max (m)
float64 distance_to_goal          # Euclidean distance to current goal (m)
float64 heading_error_to_goal     # Signed angle error toward goal (rad, -pi to pi)
float64 linear_velocity           # Current linear speed magnitude (m/s)
float64 angular_velocity          # Current angular velocity magnitude (rad/s)
float64 clearance                 # Instantaneous min obstacle distance from /scan (m)
float64 path_deviation            # Lateral deviation from global /plan (m)
bool    collision                 # True if lethal costmap cost at robot footprint this tick
float64 goal_x_map                # Current goal x in map frame (m) — 0 if no active goal
float64 goal_y_map                # Current goal y in map frame (m)
