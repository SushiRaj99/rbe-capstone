# rl_pipeline — Architecture & Flow Documentation

> **Project:** POT-r (Parameter-Optimizing Tuner via RL)  
> **Stack:** ROS2 Jazzy · Nav2 · Stable-Baselines3 PPO · Gymnasium  
> **Packages:** `rl_pipeline` · `simulation_launch`

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Node & Interface Reference](#2-node--interface-reference)
3. [Training Mode Flow](#3-training-mode-flow)
4. [Episode Lifecycle](#4-episode-lifecycle)
5. [Observation & Action Pipeline](#5-observation--action-pipeline)
6. [Evaluation Mode Flow](#6-evaluation-mode-flow)
7. [Launch File Reference](#7-launch-file-reference)
8. [TODOs](#8-todos)

---

## 1. System Architecture

The pipeline has two distinct execution contexts that communicate over ROS2 topics and services:

- **Standalone Python process** — `rl_backbone.py` owns the SB3 training loop and spins a lightweight `RLBridgeNode` in a background thread to interface with ROS2.
- **ROS2 nodes** — `planner_config_manager` (`rl_pipeline`) and `goal_manager` (`simulation_launch`) integrate with Nav2.

```mermaid
graph TD
    subgraph PROC["Standalone Python Process - rl_backbone.py"]
        PPO["SB3 PPO Agent - MlpPolicy"]
        ENV["Nav2GymEnv - gymnasium.Env"]
        BRIDGE["RLBridgeNode - background thread"]
        PPO -->|"action array"| ENV
        ENV -->|"normalized obs + reward"| PPO
        ENV <-->|"method calls"| BRIDGE
    end

    subgraph RL["rl_pipeline ROS2 Node"]
        PCM["PlannerConfigManager"]
    end

    subgraph SIM["simulation_launch ROS2 Node"]
        GM["GoalManager"]
    end

    subgraph NAV2["Nav2 Stack"]
        CS["controller_server"]
        BT["bt_navigator"]
        MS["map_server"]
        LC["local_costmap"]
        GC["global_costmap"]
    end

    subgraph SIMNODES["Simulation Nodes"]
        DD["diff_drive_model"]
        LM["lidar_model"]
    end

    BRIDGE -->|"pub /rl/action"| PCM
    PCM -->|"pub /rl/observation"| BRIDGE
    PCM -->|"pub /rl/status"| BRIDGE
    BRIDGE -->|"svc /rl/reset"| PCM
    BRIDGE -->|"svc /planner_config_manager/set_parameters"| PCM

    PCM -->|"svc /controller_server/set_parameters"| CS
    PCM -->|"svc /map_server/load_map"| MS
    PCM -->|"svc /local_costmap/clear_entirely_local_costmap"| LC
    PCM -->|"svc /global_costmap/clear_entirely_global_costmap"| GC

    PCM -->|"action send_goal_to_nav2"| GM
    GM -->|"action navigate_to_pose"| BT

    LM -->|"sub /scan"| PCM
    DD -->|"sub /odom"| PCM
    PCM -->|"pub /simulation/set_pose"| DD
    BT -->|"/cmd_vel"| DD
```

---

## 2. Node & Interface Reference

### `PlannerConfigManager`  (`rl_pipeline/scripts/planner_config_manager.py`)

| Interface | Type | Direction | Purpose |
|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | **sub** | Builds downsampled (range, bearing) LiDAR feature vector |
| `/odom` | `nav_msgs/Odometry` | **sub** | Extracts `vx`, `wz` for state vector |
| `/rl/action` | `std_msgs/Float32MultiArray` | **sub** | Receives normalized PPO action `[-1,1]^6`, denormalizes, forwards to Nav2 |
| `/rl/observation` | `std_msgs/Float32MultiArray` | **pub** | Publishes raw state vector at 20 Hz |
| `/rl/status` | `std_msgs/String` | **pub** | Publishes episode status: `idle` / `running` / `goal_reached` / `collision` / `timeout` |
| `/simulation/set_pose` | `geometry_msgs/Pose2D` | **pub** | Snaps diff_drive_model to episode start pose |
| `/rl/reset` | `std_srvs/Empty` | **svc server** | Triggers episode reset (map load, pose snap, costmap clear, new Nav2 goal) |
| `/planner_config_manager/set_parameters` | `rcl_interfaces/SetParameters` | **svc server** (auto) | Receives episode config from `RLBridgeNode.call_reset()` |
| `/controller_server/set_parameters` | `rcl_interfaces/SetParameters` | **svc client** | Applies denormalized DWB params every step |
| `/map_server/load_map` | `nav2_msgs/LoadMap` | **svc client** | Hot-swaps map at episode start |
| `/local_costmap/clear_entirely_local_costmap` | `nav2_msgs/ClearEntireCostmap` | **svc client** | Clears stale costmap data |
| `/global_costmap/clear_entirely_global_costmap` | `nav2_msgs/ClearEntireCostmap` | **svc client** | Clears stale costmap data |
| `send_goal_to_nav2` | `simulation_launch/SendGoalToNav2` | **action client** | Dispatches goal to `GoalManager` |

**Key ROS2 parameters (configurable at launch):**

| Parameter | Default | Description |
|---|---|---|
| `planner_type` | `'dwb'` | Which planner's param bounds to use |
| `max_episode_steps` | `500` | Step limit before truncation |
| `collision_threshold` | `0.10 m` | Min LiDAR range triggering collision |
| `lidar_downsample_n` | `18` | Number of (range, bearing) pairs in state |
| `xy_tolerance` | `0.25 m` | Goal proximity threshold |
| `yaw_tolerance` | `0.40 rad` | Goal heading threshold |

---

### `GoalManager`  (`simulation_launch/scripts/goal_manager.py`)

| Interface | Type | Direction | Purpose |
|---|---|---|---|
| `navigate_to_pose` | `nav2_msgs/NavigateToPose` | **action client** | Forwards goal to Nav2 BT Navigator |
| `send_goal_to_nav2` | `simulation_launch/SendGoalToNav2` | **action server** | Accepts goals from `PlannerConfigManager` |
| `/bt_navigator/get_state` | `lifecycle_msgs/GetState` | **svc client** | Polls Nav2 lifecycle state before transmitting goals |

---

### `RLBridgeNode`  (class inside `rl_backbone.py`, runs in background thread)

| Interface | Type | Direction | Purpose |
|---|---|---|---|
| `/rl/observation` | `std_msgs/Float32MultiArray` | **sub** | Buffers latest raw observation; unblocks `wait_for_observation()` |
| `/rl/status` | `std_msgs/String` | **sub** | Caches latest episode status string |
| `/rl/action` | `std_msgs/Float32MultiArray` | **pub** | Publishes raw PPO action each step |
| `/rl/reset` | `std_srvs/Empty` | **svc client** | Triggers episode reset on `PlannerConfigManager` |
| `/planner_config_manager/set_parameters` | `rcl_interfaces/SetParameters` | **svc client** | Pushes episode config before calling `/rl/reset` |

---

## 3. Training Mode Flow

```mermaid
flowchart TD
    A["ros2 launch rl_pipeline train.launch.py"] --> B

    subgraph LAUNCH["ROS2 Launch"]
        B["Start PlannerConfigManager node"] --> C
        C["Wait 2 s via TimerAction"] --> D
        D["Start rl_backbone.py process"]
    end

    D --> E

    subgraph TRAIN["rl_backbone.py - train()"]
        E["Create Nav2GymEnv"] --> F
        F["Wrap with SB3 Monitor"] --> G
        G["check_env - SB3 sanity checks"] --> H
        H["Init PPO MlpPolicy with 256x256 nets"] --> I
        I["model.learn - total_timesteps = N"]
    end

    I --> J

    subgraph ROLLOUT["Per-Rollout Loop - repeats until N steps reached"]
        J["Collect rollout_buffer steps via Nav2GymEnv.step"] --> K
        K{"Checkpoint interval?"}
        K -->|"Yes"| L["CheckpointCallback saves .zip to ckpt_dir"]
        L --> M
        K -->|"No"| M
        M["PPO gradient update - n_epochs x mini-batches"] --> J
    end

    J -->|"N steps reached"| N["Save final model ppo_nav2_dwb_final.zip"]
    N --> O["Done"]
```

---

## 4. Episode Lifecycle

This diagram covers what happens inside a single Gymnasium `reset()` followed by the `step()` loop through to a terminal condition, corresponding to one navigation attempt by the TurtleBot.

```mermaid
flowchart TD
    A["Nav2GymEnv.reset()"] --> B
    B["Sample random map + start_pose + goal_pose from EPISODE_MAP_CONFIGS"] --> C
    C["RLBridgeNode.call_reset - push episode params then call /rl/reset service"] --> D

    subgraph PCM_RESET["PlannerConfigManager.reset_config()"]
        D["load_map via /map_server/load_map"] --> E
        E["reset_diff_drive via pub /simulation/set_pose"] --> F
        F["clear_costmaps - local and global"] --> G
        G["send_nav2_goal via send_goal_to_nav2 action"] --> H
        H["episode_status = running, episode_step = 0"]
    end

    H --> I["wait_for_observation - 10 s timeout for first obs"]
    I --> J["return normalized obs and info dict"]

    J --> K

    subgraph STEP_LOOP["Step Loop - repeats each PPO timestep"]
        K["Nav2GymEnv.step with action"] --> L
        L["bridge.apply_action - pub /rl/action"] --> M

        subgraph PROC_ACTION["PlannerConfigManager.process_action()"]
            M["Denormalize: tanh output to physical param range"] --> N
            N["/controller_server/set_parameters - updates DWB scales and velocity limits"]
        end

        N --> O["bridge.wait_for_observation - 2 s timeout"]

        subgraph PUB_OBS["PlannerConfigManager.publish_observation() - 20 Hz timer"]
            O --> P["build_observation - 18 LiDAR pairs + goal deltas + velocities"]
            P --> Q["check_terminal - collision, goal_reached, or timeout"]
            Q --> R["pub /rl/observation and /rl/status"]
        end

        R --> S["compute_reward - sparse terminal + dense step rewards"]
        S --> T{"Terminal?"}
        T -->|"goal_reached or collision"| U["terminated = True"]
        T -->|"timeout"| V["truncated = True"]
        T -->|"running"| K
    end

    U --> W["return obs, reward, terminated, truncated, info"]
    V --> W
```

---

## 5. Observation & Action Pipeline

### State Vector Layout  (dim = 41)

```
Index  0  : LiDAR range   - sector 0  (closest return in 0 to 20 deg arc)
Index  1  : LiDAR bearing - sector 0
Index  2  : LiDAR range   - sector 1  (closest return in 20 to 40 deg arc)
Index  3  : LiDAR bearing - sector 1
       ...
Index 34  : LiDAR range   - sector 17
Index 35  : LiDAR bearing - sector 17
Index 36  : delta-x  to goal  (robot frame)
Index 37  : delta-y  to goal  (robot frame)
Index 38  : delta-yaw to goal (wrapped to -pi to pi)
Index 39  : vx  forward velocity
Index 40  : wz  steering velocity
```

### Normalization for PPO Input

| Component | Raw range | Normalized range |
|---|---|---|
| LiDAR ranges | `[0, range_max]` | `[0, 1]` clipped |
| LiDAR bearings | `[0, 2π]` | `[0, 1]` |
| delta-x, delta-y to goal | `[-15 m, 15 m]` | `[-1, 1]` |
| delta-yaw to goal | `[-π, π]` | `[-1, 1]` |
| Forward velocity vx | `[-0.5, 0.5 m/s]` | `[-1, 1]` |
| Steering velocity wz | `[-1.0, 1.0 rad/s]` | `[-1, 1]` |

### Action Vector Layout  (dim = 6) — DWB planner

PPO outputs actions through a `tanh` layer, so all values are in `[-1, 1]`.
`PlannerConfigManager.process_action()` maps them back to physical units:

| Index | Parameter | Min | Max |
|---|---|---|---|
| 0 | `FollowPath.max_vel_x` | 0.05 m/s | 0.50 m/s |
| 1 | `FollowPath.min_speed_xy` | 0.00 m/s | 0.20 m/s |
| 2 | `FollowPath.GoalAlign.scale` | 1.0 | 50.0 |
| 3 | `FollowPath.PathAlign.scale` | 1.0 | 50.0 |
| 4 | `FollowPath.GoalDist.scale` | 1.0 | 50.0 |
| 5 | `FollowPath.PathDist.scale` | 1.0 | 50.0 |

Denormalization formula: `real_val = clip(norm_val, 0, 1) * (hi - lo) + lo`
where `norm_val = (ppo_output + 1) / 2`

### Reward Function

```
R_total = R_terminal + R_step_penalty + R_progress + R_smooth + R_proximity

  R_terminal     = +100   if goal_reached
                 = -100   if collision

  R_step_penalty = -0.01  (every non-terminal step)

  R_progress     = +10 * (prev_dist_to_goal - curr_dist_to_goal)
                   (positive when closing on goal)

  R_smooth       = -0.01 * |wz|
                   (penalizes unnecessary rotation)

  R_proximity    = -1.0 * max(0, 1 - min_lidar_range)
                   (active only when min range < 1.0 m)
```

---

## 6. Evaluation Mode Flow

Running `eval.launch.py` executes two sequential back-to-back evaluations over identical episode sequences (controlled by `--eval-seed`) for a fair comparison.

```mermaid
flowchart TD
    A["ros2 launch rl_pipeline eval.launch.py model:=path/to/checkpoint.zip"] --> B

    subgraph LAUNCH["ROS2 Launch"]
        B["Start PlannerConfigManager node"] --> C
        C["Wait 2 s"] --> D
        D["Start rl_backbone.py with --mode eval"]
    end

    D --> E

    subgraph EVAL_MODEL["evaluate_model() - frozen PPO policy"]
        E["Load PPO checkpoint via PPO.load"] --> F
        F["Nav2GymEnv.reset with seed"] --> G
        G["model.predict - deterministic=True"] --> H
        H["env.step with action"] --> I
        I{"Done?"}
        I -->|"No"| G
        I -->|"Yes"| J["Log: reward, steps, outcome"]
        J --> K{"More episodes?"}
        K -->|"Yes"| F
        K -->|"No"| L["summarize_eval_results - print + optional JSON save"]
    end

    L --> M

    subgraph EVAL_BASE["evaluate_baseline() - Nav2 default params, no RL"]
        M["Nav2GymEnv.reset with same seed"] --> N
        N["bridge.wait_for_observation - no action published, Nav2 uses YAML defaults"] --> O
        O["compute_reward"] --> P
        P{"Done?"}
        P -->|"No"| N
        P -->|"Yes"| Q["Log: reward, steps, outcome"]
        Q --> R{"More episodes?"}
        R -->|"Yes"| M
        R -->|"No"| S["summarize_eval_results - print + optional JSON save"]
    end

    S --> T["Evaluation complete - results printed to stdout"]
```

---

## 7. Launch File Reference

### `train.launch.py`

| Argument | Default | Description |
|---|---|---|
| `planner` | `dwb` | Local planner to tune (`dwb` or `mppi`) |
| `max_episode_steps` | `500` | Step limit before episode truncation |
| `steps` | `500000` | Total PPO training steps |
| `log_dir` | `~/rl_logs` | TensorBoard logs + Monitor CSVs |
| `ckpt_dir` | `~/rl_checkpoints` | Model checkpoint output directory |
| `net_arch` | `[256, 256]` | JSON list of hidden layer widths |
| `lr` | `3e-4` | PPO learning rate |
| `ro_buffer` | `2048` | Rollout buffer size (n_steps) |
| `batch_size` | `64` | PPO mini-batch size |
| `epochs` | `10` | PPO optimization epochs per rollout |

```bash
# Minimal
ros2 launch rl_pipeline train.launch.py

# Custom hyper-params
ros2 launch rl_pipeline train.launch.py \
    steps:=1000000 lr:=1e-4 net_arch:='[512,512]' \
    log_dir:=/data/rl_logs ckpt_dir:=/data/rl_ckpts
```

### `eval.launch.py`

| Argument | Default | Description |
|---|---|---|
| `planner` | `dwb` | Planner the model was trained on |
| `max_episode_steps` | `500` | Should match value used during training |
| `model` | *(empty — **required**)* | Path to `.zip` PPO checkpoint |
| `episodes` | `10` | Episodes per evaluation run |
| `eval_seed` | `28` | RNG seed — keep the same as training for fair comparison |
| `results` | *(empty)* | Optional `.json` output path (requires parser fix — see §8) |

```bash
# Minimal (model arg is required)
ros2 launch rl_pipeline eval.launch.py \
    model:=~/rl_checkpoints/ppo_nav2_dwb_final.zip

# Full options
ros2 launch rl_pipeline eval.launch.py \
    model:=~/rl_checkpoints/ppo_nav2_dwb_500k.zip \
    episodes:=50 eval_seed:=42
```

---

## 8. TODOs

### TODO — Map start/goal coordinates are placeholders

All entries in `MAP_CONFIGS` inside `pipeline_utils.py` use identical placeholder coordinates. These must be replaced with valid poses for each map before training begins, otherwise every episode will start in the same corner regardless of the map loaded.

### TODO — MPPI action space is undefined

`PLANNER_PARAM_BOUNDS` only has an entry for `dwb`. The `mppi` key is listed as a placeholder. Exposing MPPI parameters and updating `NUM_ACTIONS` will be required before `--planner mppi` can be used.
