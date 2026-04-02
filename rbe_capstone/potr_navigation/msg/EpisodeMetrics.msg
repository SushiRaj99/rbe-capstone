# Per-episode path quality metrics produced by metrics_tracker.py

# Episode context
string planner               # Planner used (DWB or MPPI)
int32  preset                # Parameter preset index
string map_name              # Map used for this episode
string goal_id               # Goal identifier
bool   goal_reached          # Whether the robot reached the goal
float64 start_x              # Spawn position in map frame (m)
float64 start_y
float64 start_yaw            # (rad)
float64 goal_x               # Goal position in map frame (m)
float64 goal_y
float64 goal_yaw             # (rad)

# Safety
float64 min_clearance          # Minimum obstacle clearance recorded from /scan (m)

# Accuracy
float64 mean_path_deviation    # Mean lateral deviation from global path /plan (m)
float64 max_path_deviation     # Max lateral deviation from global path /plan (m)

# Efficiency
float64 total_time             # Episode duration (s)
float64 total_distance         # Total distance traveled (m)

# Smoothness
float64 velocity_variance      # Variance of linear speed over episode (m/s)^2
float64 heading_variance       # Variance of angular velocity magnitude over episode (rad/s)^2

# Collision
int32 collision_count          # Number of timesteps where lethal cost entered the robot footprint
