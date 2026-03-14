# POT-r Design Document

### Planner Parameters Optimization Through Reinforcement Learning

Also known as POT nobody wants to smoke

### Goals

* Learn a policy that maps site/map features to optimal planner parameter sets
* Support both DWB and MPPI as candidate planners
* Evaluate performance against five metrics: path length, smoothness, travel time, clearance, and average velocity
* Demonstrate generalization across multiple warehouse map layouts---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     RL Training Loop                     │
│                                                         │
│   ┌──────────────┐     action      ┌─────────────────┐  │
│   │   RL Agent   │ ──────────────► │  Gym Env Wrapper│  │
│   │   (PPO/SB3)  │ ◄────────────── │                 │  │
│   └──────────────┘   obs + reward  └────────┬────────┘  │
│                                             │           │
└─────────────────────────────────────────────┼───────────┘
                                              │
                              ┌───────────────▼───────────────┐
                              │       ROS2 Simulation          │
                              │                               │
                              │  ┌──────────┐  ┌──────────┐  │
                              │  │  Nav2    │  │ Map      │  │
                              │  │ Planner  │  │ Server   │  │
                              │  │(DWB/MPPI)│  │ (.pgm)   │  │
                              │  └────┬─────┘  └──────────┘  │
                              │       │                       │
                              │  ┌────▼─────────────────┐    │
                              │  │  TurtleBot4 (URDF)   │    │
                              │  │  /odom /cmd_vel       │    │
                              │  │  /local_costmap       │    │
                              │  └──────────────────────┘    │
                              │                               │
                              │  ┌───────────────────────┐   │
                              │  │  Trajectory Evaluator  │   │
                              │  │  → reward, metrics     │   │
                              │  └───────────────────────┘   │
                              └───────────────────────────────┘
```

### Key Design Principle

The RL agent never directly controls the robot. It selects a **parameter configuration** for the planner before each episode, then the planner runs normally. The agent observes the outcome and updates its policy accordingly. This keeps the RL problem well-scoped and allows the existing Nav2 stack to remain unchanged.

---

## Simulation Environment

### Nav2 Planner Integration

Both DWB and MPPI will be configured as Nav2 local planner plugins. Their tunable parameters will be exposed via the ROS2 parameter server so they can be set programmatically between episodes without restarting any nodes.

**DWB parameters of interest:**

* `max_vel_x`, `min_vel_x`, `max_vel_theta`
* `acc_lim_x`, `acc_lim_theta`
* Critic weights: `PathAlignCritic.scale`, `GoalCritic.scale`, `ObstacleFootprintCritic.scale`

**MPPI parameters of interest:**

* `vx_max`, `vy_max`, `wz_max`
* `AX_MAX`, `AY_MAX`
* `temperature`, `gamma`

### Episode Lifecycle

Each RL training episode follows this sequence:

```
1. Select map from training set
2. Sample valid start and goal poses
3. Agent observes map features → selects parameter set (action)
4. Set planner parameters via ROS2 parameter server
5. Send NavigateToPose goal via Nav2 action client
6. Trajectory evaluator records /odom, /cmd_vel, /local_costmap
7. Episode ends on: goal reached | collision | timeout
8. Compute reward from trajectory metrics
9. Return (observation, reward, done, info) to Gymnasium
```

### Map Feature Extraction

The observation fed to the RL agent should encode the structural characteristics of the current map. These features are computed from the `.pgm` occupancy grid at the start of each episode:

| Feature                  | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `obstacle_density`     | Fraction of occupied cells in the map                        |
| `mean_corridor_width`  | Average free-space width along the planned path              |
| `open_area_ratio`      | Fraction of map that is large open space vs. narrow passages |
| `path_length_estimate` | Euclidean distance from start to goal                        |
| `path_complexity`      | Ratio of A* path length to straight-line distance            |
| `goal_heading`         | Direction of goal relative to start (normalized)             |

These 6–8 scalar features form the observation vector. No raw image data is fed to the agent.

### ROS2 Package Structure

```
rbe_capstone/
├── simulation_launch/
├── rl_env/
│   └── rl_env/
│       ├── planner_param_env.py    # gym.Env subclass
│       ├── map_features.py         # .pgm → observation vector
│       ├── trajectory_evaluator.py # metrics from ROS2 topics
│       ├── reward.py               # reward computation
│       └── param_sets.py           # named DWB/MPPI configs
└── training/                   # NEW — not a ROS package
    ├── train.py
    ├── evaluate.py
    └── configs/
        └── ppo_config.yaml
```

---

## Trajectory Evaluation

The trajectory evaluator is a ROS2 node that subscribes to live topics during a navigation episode and produces a structured result when the episode ends.

### Metrics

| Metric                  | Source             | Computation                                          |
| ----------------------- | ------------------ | ---------------------------------------------------- |
| **Path Length**   | `/odom`          | Cumulative Euclidean distance between pose samples   |
| **Travel Time**   | Episode timer      | Wall time from goal send to goal reached/failed      |
| **Smoothness**    | `/odom`          | Variance of heading angle changes over trajectory    |
| **Min Clearance** | `/local_costmap` | Minimum distance to any occupied cell during episode |
| **Avg Velocity**  | `/cmd_vel`       | Mean linear velocity magnitude over episode          |
| **Goal Success**  | Nav2 action result | Boolean — did the robot reach the goal              |

### Reward Function

The reward combines all five evaluation metrics into a single scalar. Weights are tunable and should be adjusted based on experimental results.

```python
def compute_reward(result: EpisodeResult) -> float:
    # Binary success signal
    r_success  = 10.0 if result.reached_goal else -10.0

    # Safety: penalize proximity to obstacles (threshold: 0.3m)
    r_safety   = -max(0, 0.3 - result.min_clearance) * 20.0

    # Efficiency: penalize path length relative to global plan
    r_length   = -(result.path_length / result.global_plan_length)

    # Time: small penalty per timestep
    r_time     = -result.time_elapsed * 0.01

    # Smoothness: penalize heading variance
    r_smooth   = -result.heading_variance * 0.5

    return r_success + r_safety + r_length + r_time + r_smooth
```

> **Note:** For initial development, start with only `r_success + r_safety` to verify the training loop converges before adding complexity.

---

## Reinforcement Learning Design

### Problem Formulation

| Component             | Definition                                        |
| --------------------- | ------------------------------------------------- |
| **Agent**       | PPO policy network                                |
| **Environment** | Gymnasium wrapper around ROS2 sim                 |
| **State**       | Map feature vector (6–8 floats)                  |
| **Action**      | Discrete index into parameter set library         |
| **Reward**      | Weighted combination of trajectory metrics        |
| **Episode**     | One navigation attempt (start → goal or failure) |

### Action Space

For the initial implementation, the action space is  **discrete** : the agent selects one of N pre-defined parameter configurations. This keeps the problem tractable for early training and makes results interpretable.

```python
# Phase 1: Discrete (start here)
action_space = gym.spaces.Discrete(N_PARAM_SETS)

# Phase 2: Continuous (future work)
action_space = gym.spaces.Box(low=param_mins, high=param_maxs)
```

Suggested initial parameter sets (5–8 configs) should span the space from aggressive (fast, lower clearance) to conservative (slow, wide clearance), with a few intermediate options.

### Observation Space

```python
observation_space = gym.spaces.Box(
    low=0.0, high=1.0,
    shape=(8,),          # normalized map features
    dtype=np.float32
)
```

All features should be normalized to [0, 1] before being fed to the network.

### Network Architecture

A simple MLP is sufficient for this state space size. Following the architecture validated in Wong et al. (2024) for a similar path-selection problem:

```
Input (8) → Dense(64, ReLU) → Dense(64, ReLU) → Output (N actions)
```

Stable-Baselines3's default `MlpPolicy` implements this automatically.

### Training

```python
from stable_baselines3 import PPO
from rl_env import PlannerParamEnv

env = PlannerParamEnv(
    maps=["warehouse_a", "warehouse_b", "narrow_corridor"],
    planner="mppi"
)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    verbose=1,
    tensorboard_log="./tb_logs/"
)

model.learn(total_timesteps=100_000)
model.save("potr_policy_v1")
```

---

## Why PPO

Proximal Policy Optimization is the recommended algorithm for POT-r for several reasons that align specifically with this project's constraints.

### Sample Efficiency vs. Stability Tradeoff

Each training episode requires running a full robot navigation simulation — typically several seconds of wall time even in a lightweight 2D sim. This means sample efficiency matters: we cannot afford to run millions of episodes like some model-free RL methods require. PPO strikes a good balance here. Unlike vanilla policy gradient methods which throw away data after each update, PPO reuses collected rollouts for multiple gradient update steps via its clipped surrogate objective, extracting more learning signal per episode.

### The Clipping Mechanism Prevents Training Instability

PPO's defining feature is its clipped objective function:

```
L_CLIP(θ) = E[ min( r_t(θ) * A_t,  clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]
```

where `r_t(θ)` is the ratio of new to old policy probability. This prevents any single update from changing the policy too drastically, which is important here because our reward signal will be noisy — small differences in map layout or planner initialization can produce meaningfully different trajectories. Without clipping, noisy rewards can cause catastrophic policy updates that are hard to recover from.

### Discrete Action Spaces Are a Natural Fit

Our initial action space is discrete (select parameter set 0–N). PPO handles discrete action spaces natively via a categorical distribution over actions, with no modifications needed. Algorithms like SAC or TD3 are designed for continuous action spaces and require additional complexity (e.g., Gumbel-softmax tricks) to work with discrete actions.

### Strong Baseline in Related Work

Two papers directly relevant to this project both used PPO successfully in robot navigation optimization contexts:

* **Ting et al. (2025)** used PPO with a dense reward function to reduce training time by 52% over standard DRL for mobile robot path planning
* **Wong et al. (2024)** used PPO to select optimal path combinations for multi-robot navigation, achieving up to 9.84% conflict reduction with a simple 2-layer MLP

Both cases involved a meta-decision layer over existing planners — structurally identical to what POT-r is doing — which gives strong prior evidence that PPO will work well here.

### Stable-Baselines3 Integration

SB3's PPO implementation is battle-tested, well-documented, and integrates directly with the Gymnasium interface we are building. It also includes built-in TensorBoard logging, callback support for checkpointing and early stopping, and vectorized environment support for parallelizing episodes if needed in later stages.

### Path to Continuous Actions

If Phase 2 extends the action space to continuous parameter values (e.g., directly outputting `max_vel_x`, `temperature`, etc.), PPO can accommodate this with no algorithmic changes — simply swap to a `Box` action space and SB3 handles the rest by switching to a Gaussian policy. This makes PPO a low-risk starting point with a clear upgrade path.

---

## Implementation Roadmap

### Phase 1 — Navigation Foundation

*Target: March 13*

* [ ] Integrate Nav2 DWB and MPPI as active local planners
* [ ] Add active odometry source (`/odom` publishing)
* [ ] Implement programmatic `NavigateToPose` goal sending via action client
* [ ] Verify end-to-end: robot navigates from A to B in RViz2
* [ ] Add `stable-baselines3` and `gymnasium` to `capstone_ros_layer` Dockerfile

### Phase 2 — Evaluation & Gym Wrapper

*Target: March 20*

* [ ] Implement `trajectory_evaluator.py` — subscribes to `/odom`, `/cmd_vel`, `/local_costmap` and produces `EpisodeResult`
* [ ] Implement `map_features.py` — loads `.pgm` and computes observation vector
* [ ] Implement `param_sets.py` — define 5–8 named DWB/MPPI configurations
* [ ] Implement `planner_param_env.py` — full Gymnasium `Env` subclass
* [ ] Validate env with dummy `_run_episode` returning random results
* [ ] Add 2–3 additional warehouse maps with varying geometry

### Phase 3 — RL Training

*Target: March 27*

* [ ] Wire real simulation into Gymnasium env
* [ ] Run initial PPO training (start with `r_success + r_safety` only)
* [ ] Confirm training loop converges (reward trending upward in TensorBoard)
* [ ] Add full reward function terms incrementally
* [ ] Checkpoint best policy

### Phase 4 — Evaluation & Analysis

*Target: April 25*

* [ ] Evaluate trained policy against baseline (default Nav2 parameters) across all maps
* [ ] Generate per-metric comparison tables (path length, clearance, time, smoothness)
* [ ] Test generalization on a held-out map not seen during training
* [ ] Ablation study: compare reward-only-success vs. full reward function
* [ ] (Stretch) Compare discrete vs. continuous action space

---

## 9. Open Questions & Risks

| Item                               | Notes                                                                                                                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Episode speed**            | Each episode = one full Nav2 navigation run. If episodes are too slow, training will be impractical. Mitigation: keep maps small, set aggressive timeouts, explore SB3 vectorized envs.        |
| **Parameter server latency** | Setting ROS2 parameters between episodes via `rclpy`may introduce non-trivial latency. Test early and batch parameter sets if needed.                                                        |
| **Reward sparsity**          | If the robot fails most early episodes, the reward signal will be nearly all `-10.0`. Mitigation: use curriculum learning (start with easy maps/short distances) as described in Ting et al. |
| **Map diversity**            | Generalizing across map layouts requires sufficient variety in the training set. At minimum: one open floor, one dense warehouse, one narrow corridor map.                                     |
| **Continuous action space**  | Moving to continuous parameters in Phase 2 significantly expands the search space. Only pursue if discrete results are strong and time allows.                                                 |
| **Planner determinism**      | Nav2 planners may produce slightly different trajectories across runs with identical parameters due to timing. This adds noise to the reward signal — worth quantifying early.                |
