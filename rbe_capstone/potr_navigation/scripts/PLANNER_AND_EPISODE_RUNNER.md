# Planner Controller & Episode Runner

Two ROS2 nodes that collaborate to benchmark Nav2 controllers:

- **`planner_controller.py`** — swaps between DWB and MPPI controllers at runtime and pushes parameter updates to `/controller_server` and `/velocity_smoother`.
- **`episode_runner.py`** — drives a state machine that loops over `(planner, preset) × episode` combinations, respawns the robot, sends Nav2 goals, and collects metrics.

---

## 1. `planner_controller.py`

### Purpose
A thin control-plane node. It does **not** plan anything itself; it just holds the currently-active `(planner, preset)` pair and translates high-level requests into ROS parameter updates on the running `controller_server` + `velocity_smoother`.

### Services exposed
| Service | Type | Behavior |
|---|---|---|
| `/potr_navigation/switch_planner` | `SwitchPlanner` | Sets `self.planner` (DWB / MPPI) and reapplies current preset. |
| `/potr_navigation/set_param_preset` | `SetParamPreset` | Sets `self.preset` (1 / 2) and reapplies full parameter set. |
| `/potr_navigation/set_raw_params` | `SetRawParams` | Writes a partial set of shared-name params (e.g. `max_linear_vel`) directly. Used by the RL policy. |

### Published topic
- `/potr_navigation/current_planner_params` — latched `std_msgs/String` JSON blob `{planner, preset, values}`. Late subscribers still see the current state.

### Parameter mapping
`SHARED_PARAM_MAP` maps shared logical names (`max_linear_vel`, `goal_align_scale`, …) to per-planner ROS names (`FollowPathDWB.max_vel_x` for DWB vs `FollowPath.vx_max` for MPPI). This lets the RL agent speak one vocabulary regardless of which controller is loaded.

### Apply flow
`apply_params()` reloads the YAMLs and pushes **both** the shared params and the planner-specific plugin block to `controller_server`, then syncs `max_velocity`/`min_velocity` on `velocity_smoother` so acceleration limits don't clip the new speed envelope. The `critics` key is skipped — updating it live crashes the controller.

### Diagram

```mermaid
flowchart TD
    A[Client: ros2 service call] --> B{Which service?}
    B -->|switch_planner| C[set self.planner]
    B -->|set_param_preset| D[set self.preset]
    B -->|set_raw_params| E[Partial update of shared params]

    C --> F[apply_params]
    D --> F

    F --> G[Load shared_params.yaml<br/>+ dwb_params.yaml or mppi_params.yaml]
    G --> H[Build param dict via<br/>SHARED_PARAM_MAP + plugin namespace]
    H --> I[send_params → /controller_server/set_parameters]
    I --> J[Sync /velocity_smoother/set_parameters<br/>max_velocity / min_velocity]
    J --> K[Update latest_shared_values]

    E --> L[Map shared names to ROS names<br/>for current planner]
    L --> M[send_params → /controller_server]
    M --> N{max_lin or max_ang changed?}
    N -->|yes| O[Sync velocity_smoother]
    N -->|no| P[skip]
    O --> K
    P --> K

    K --> Q[publish_params_snapshot<br/>latched JSON on current_planner_params]
```

---

## 2. `episode_runner.py`

### Purpose
Runs benchmark episodes end-to-end. For each `(planner, preset)` in `RUN_CONFIGS` and each episode loaded from the `episodes` parameter (a JSON list of `{map, start, goal}`), it:

1. Tells `planner_controller` to switch planner + preset.
2. Switches the map if the episode uses a different one.
3. Teleports the robot to `start` via `/simulation/set_pose` (map→odom converted).
4. Waits `SETTLE_SECS` for physics to settle.
5. Resets metrics, publishes the goal pose, sends a `NavigateToPose` action goal.
6. Waits for Nav2 to finish, fetches metrics, publishes `EpisodeMetrics`.

Two operating modes, selected by the `rl_mode` parameter:

- **Benchmark mode** (`rl_mode=False`): loops through all configs × episodes autonomously.
- **RL mode** (`rl_mode=True`): idles in `S_WAITING_FOR_RL` until an external caller hits `/potr_navigation/start_episode`. Episode index wraps so training can run indefinitely.

### Secondary role: manual goal action server
Independent of the run loop, the node also hosts an `SendGoalToNav2` action server. `manage_send_goal` forwards the request to Nav2 while `monitor_goal` (10 Hz) watches TF for position/yaw error and declares success when tolerances are hit (bypassing Nav2's own goal-reached criterion if needed).

### Pose jittering
When `randomize_poses=True`, `jittered_episode()` adds uniform ±`jitter_xy` and ±`jitter_yaw` noise to both start and goal for domain randomization. Default is off because several hand-tuned goalpoints sit close to obstacles.

### Diagram

```mermaid
stateDiagram-v2
    [*] --> INIT
    INIT --> DONE: episodes param empty / bad JSON
    INIT --> WAITING_FOR_RL: rl_mode=true
    INIT --> SWITCHING_PLANNER: rl_mode=false

    WAITING_FOR_RL --> START_EPISODE: /start_episode service call

    SWITCHING_PLANNER --> START_EPISODE: switch_planner + set_preset callbacks done

    START_EPISODE --> SWITCHING_PLANNER: episode_index exhausted,<br/>advance run_index
    START_EPISODE --> DONE: run_index exhausted
    START_EPISODE --> MAP_SWITCHING: episode map != current map
    START_EPISODE --> SETTLING: same map → just respawn

    MAP_SWITCHING --> SETTLING: LoadMap response +<br/>respawn published

    SETTLING --> GOAL_ACTIVE: settle timer elapsed<br/>→ reset_metrics +<br/>send NavigateToPose goal

    GOAL_ACTIVE --> GETTING_METRICS: Nav2 result received<br/>(SUCCEEDED / ABORTED / ...)

    GETTING_METRICS --> WAITING_FOR_RL: rl_mode=true<br/>(wrap episode_index)
    GETTING_METRICS --> START_EPISODE: rl_mode=false<br/>(episode_index++)

    DONE --> [*]
```

### Run-loop timing
Three timers keep the node alive:

| Timer | Period | Role |
|---|---|---|
| `check_nav2_active` | 0.1 s | Polls `/bt_navigator/get_state`; gates the run loop on Nav2 being `active`. |
| `run_loop_tick` | 0.5 s | Advances the state machine above. |
| `monitor_goal` | 0.1 s | Drives the `SendGoalToNav2` action server (TF-based error feedback). |

### Key external dependencies
- **`planner_controller`**: `/potr_navigation/switch_planner`, `/potr_navigation/set_param_preset`.
- **`metrics_tracker`**: `/potr_navigation/reset_metrics`, `/potr_navigation/get_metrics`, and subscribes to `/potr_navigation/current_goal` for distance/heading error.
- **Nav2**: `navigate_to_pose` action, `/map_server/load_map`, `/bt_navigator/get_state`.
- **Simulator**: `/simulation/set_pose` (Pose2D in odom frame — note the `map_to_odom_x/y` offset).

---

## How they fit together

```mermaid
sequenceDiagram
    participant ER as episode_runner
    participant PC as planner_controller
    participant CS as controller_server
    participant VS as velocity_smoother
    participant MT as metrics_tracker
    participant N2 as Nav2 (bt_navigator)
    participant SIM as simulator

    ER->>PC: SwitchPlanner(DWB)
    PC->>CS: SetParameters (DWB plugin + shared)
    PC->>VS: SetParameters (max/min velocity)
    PC-->>ER: success
    ER->>PC: SetParamPreset(1)
    PC->>CS: SetParameters (preset 1 values)
    PC->>VS: SetParameters
    PC-->>ER: success

    alt different map
        ER->>N2: LoadMap
    end
    ER->>SIM: /simulation/set_pose (Pose2D)
    Note over ER: wait SETTLE_SECS

    ER->>MT: reset_metrics
    ER->>MT: publish /current_goal
    ER->>N2: NavigateToPose goal
    N2-->>ER: result (SUCCEEDED / ABORTED / ...)
    ER->>MT: get_metrics
    MT-->>ER: EpisodeMetrics
    ER->>ER: publish /episode_metrics
```
