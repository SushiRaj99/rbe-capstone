# RL Training — How It Works

This document explains the reinforcement learning layer added on top of the POTR
navigation stack.  It is written for someone who understands the ROS2 simulation
but is new to gymnasium and stable-baselines3.

---

## Big Picture

The goal is to have an RL agent automatically discover good parameter values for
the DWB or MPPI local planner.  Instead of a human picking presets, the agent
runs the robot through navigation episodes, observes how well the robot is doing,
and adjusts planner parameters to maximise a reward signal.

```
┌─────────────────────────────────────────────────────────────────┐
│  Python training process                                         │
│                                                                  │
│   SB3 algorithm (PPO / SAC)                                      │
│        │  action (param values)                                  │
│        ▼                                                         │
│   PotrNavEnv  ◄──── step_metrics (every odom tick) ────┐        │
│        │                                               │        │
│        │  ROS2 service calls                    ROS2 topics     │
│        ▼                                               │        │
│   _BridgeNode  ─────────────────────────────────────►  │        │
│   (background thread)                                  │        │
└────────────────────────────────────────────────────────┼────────┘
                                                         │
         ROS2 network (same machine or DDS)              │
                                                         │
┌────────────────────────────────────────────────────────┼────────┐
│  Simulation process                                     │        │
│                                                         │        │
│   episode_runner  ──► nav2 goal ──► robot moves ──► odom        │
│        │                                               │        │
│        └─► /current_goal ──► metrics_tracker ──► /step_metrics ─┘
│                                      │
│                                      └─► /episode_metrics (end of episode)
└─────────────────────────────────────────────────────────────────┘
```

---

## What Is Gymnasium?

Gymnasium (formerly OpenAI Gym) is a standard Python interface for RL
environments.  Every environment—whether it is an Atari game, a robot arm, or
our navigation stack—exposes exactly the same three things:

| Concept | What it means here |
|---|---|
| **Observation space** | The set of numbers the agent can see each step |
| **Action space** | The set of numbers the agent can output each step |
| **Reward** | A single float the environment returns after each action |

Any SB3 algorithm can train on any gymnasium-compatible environment without
knowing anything about what is inside it.  The environment is a black box.

### The two methods every gym env must implement

```
observation, info = env.reset()
```
Called at the start of each episode.  Resets the environment to a fresh state
and returns the first observation.

```
observation, reward, terminated, truncated, info = env.step(action)
```
Called once per timestep.  The agent sends an action; the environment advances
one step, and returns:
- `observation` — what the agent sees now
- `reward` — how good that step was (positive = good, negative = bad)
- `terminated` — True if the episode ended naturally (goal reached, crash)
- `truncated` — True if the episode ended due to a timeout
- `info` — optional dictionary of extra data for logging

---

## The Observation Vector

The agent sees 7 numbers every time the robot's odometry updates (~10 Hz):

| Index | Field | Units | Description |
|---|---|---|---|
| 0 | `distance_to_goal` | m | Euclidean distance from robot to current goal |
| 1 | `heading_error_to_goal` | rad | Signed angle between robot heading and direction to goal (−π to π) |
| 2 | `linear_velocity` | m/s | Current forward speed magnitude |
| 3 | `angular_velocity` | rad/s | Current turn rate magnitude |
| 4 | `clearance` | m | Distance to nearest obstacle from lidar |
| 5 | `path_deviation` | m | Lateral distance from robot to the global plan |
| 6 | `collision` | 0 or 1 | 1 if the costmap shows a lethal cell at the robot's position |

These values come from `metrics_tracker.py`, which publishes them as a
`StepMetrics` ROS message on `/potr_navigation/step_metrics` every odom tick.
The bridge node subscribes to this topic and places each message into a Python
`queue.Queue`, where `env.step()` picks it up.

The observation space is declared as a `gymnasium.spaces.Box`, which is just a
bounded multi-dimensional array.  Each field has a defined minimum and maximum
so that SB3 can normalise inputs if needed.

---

## The Action Space

Two modes are supported, selected when constructing the environment.

### Discrete mode (`action_mode='discrete'`)

```python
env = PotrNavEnv(action_mode='discrete')
```

The agent outputs a single integer (0–3) selecting one of four fixed configs:

| Action | Planner | Preset |
|---|---|---|
| 0 | DWB | 1 (Safe/Accurate) |
| 1 | DWB | 2 (Efficient) |
| 2 | MPPI | 1 (Safe/Accurate) |
| 3 | MPPI | 2 (Efficient) |

The environment calls the `/potr_navigation/switch_planner` and
`/potr_navigation/set_param_preset` ROS services to apply the selection.

This is the simpler option.  The agent is essentially learning which preset works
best for each situation it observes.

### Continuous mode (`action_mode='continuous'`)

```python
env = PotrNavEnv(action_mode='continuous', planner='MPPI')
```

The agent outputs a vector of 4 floats, each in the range [−1, 1].  These are
mapped to physical parameter values:

| Action dim | Parameter | Physical range |
|---|---|---|
| 0 | `max_linear_vel` | 0.1 – 1.5 m/s |
| 1 | `max_angular_vel` | 0.3 – 2.5 rad/s |
| 2 | `linear_accel` | 0.5 – 3.5 m/s² |
| 3 | `angular_accel` | 0.5 – 4.0 rad/s² |

The normalisation formula is:

```
physical_value = low + (action + 1) / 2 × (high − low)
```

So action = −1 gives the minimum value, action = 0 gives the midpoint, and
action = +1 gives the maximum.  SB3 algorithms work better when actions live in
[−1, 1] than when they use raw physical units with very different scales.

The environment calls the `/potr_navigation/set_raw_params` ROS service, which
was added specifically for this purpose in `planner_controller.py`.  It accepts
a list of parameter names and values, maps them to the correct ROS parameter
names for the active planner, and pushes them to the Nav2 controller server.

---

## The Reward Function

The reward is computed inside `env.step()` for every timestep, and once more at
episode termination.

### Per-step reward

```
reward = +1.0  × (prev_distance_to_goal − current_distance_to_goal)   # progress
       − 0.1   × path_deviation          # penalise straying from the global plan
       − 0.05  × angular_velocity        # penalise excessive spinning
       − 10.0  × collision               # heavy per-step collision penalty
```

The **progress** term is the most important.  It is positive when the robot moves
closer to the goal and negative when it moves away.  This gives the agent a
dense, continuous signal at every timestep rather than waiting until the end.

### Terminal reward

When the episode ends (the `EpisodeMetrics` message arrives), an additional
one-time reward is applied:

```
+100.0   if goal_reached == True
 −50.0   if the episode ended without reaching the goal
```

All weights are defined in `potr_rl/params.py` under `REWARD` and can be tuned
without touching the environment code.

---

## The ROS2 ↔ Python Bridge

This is the trickiest part of the design.  The problem is a mismatch between two
programming models:

- **SB3 / gymnasium** is fully synchronous: `step()` is called, does work, and
  returns.  The training loop blocks until it returns.
- **ROS2** is fully asynchronous: nodes register callbacks that fire whenever
  messages or service responses arrive.  There is no built-in way to "wait" for
  something from a non-ROS thread.

### Solution: background spin thread + queue

```
Main thread (SB3)                Background thread (rclpy)
─────────────────                ─────────────────────────
env.step(action)                 executor.spin()  ← runs forever
  │                                │
  │  call_async(set_params)        │  StepMetrics callback fires
  │    └─ threading.Event.wait() ◄─┘    obs_queue.put(('step', msg))
  │                                │
  │  obs_queue.get()  ◄────────────┘  EpisodeMetrics callback fires
  │    blocks until                     obs_queue.put(('done', msg))
  │    next message arrives
  │
  └─ returns (obs, reward, done, ...)
```

The background thread is started in `PotrNavEnv.__init__()`:

```python
self._executor = MultiThreadedExecutor()
self._executor.add_node(self._node)
self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
self._spin_thread.start()
```

`daemon=True` means this thread is automatically killed when the main process
exits, so there is no cleanup needed.

### Synchronous service calls from the main thread

When `env.step()` needs to call a ROS service (e.g. to apply new parameters),
it cannot just call `client.call_async()` and return — it needs to wait for the
response.  This is done with a `threading.Event`:

```python
def call_sync(self, client, request, timeout=10.0):
    event = threading.Event()
    result = [None]

    def cb(future):
        result[0] = future.result()
        event.set()          # unblocks the main thread

    future = client.call_async(request)
    future.add_done_callback(cb)
    event.wait(timeout=timeout)   # main thread blocks here
    return result[0]
```

The ROS2 executor (running on the background thread) processes the future and
calls `cb`, which stores the result and sets the event.  The main thread wakes
up and continues.

### Episode end detection

Two different message types arrive on the queue:

- `('step', StepMetrics)` — a regular timestep during the episode
- `('done', EpisodeMetrics)` — signals the episode has ended

`env.step()` checks the tag of whatever it pops from the queue:

```python
tag, data = self._obs_queue.get(timeout=30.0)

if tag == 'done':
    # episode over — return terminal reward and terminated=True
    ...
else:
    # normal step — compute step reward and return terminated=False
    ...
```

---

## Episode Lifecycle

### How `env.reset()` works

1. Drains any leftover messages from the previous episode out of the queue.
2. Calls `/potr_navigation/start_episode` (a std_srvs/Trigger service on
   `episode_runner`).
3. `episode_runner` respawns the robot at the episode's start position, waits
   2 seconds for physics to settle, then sends a `NavigateToPose` goal to Nav2
   and publishes the goal position to `/potr_navigation/current_goal`.
4. `metrics_tracker` receives the goal position and starts computing
   `distance_to_goal` in every `StepMetrics` message it publishes.
5. `env.reset()` blocks on `obs_queue.get()` until the first `StepMetrics`
   arrives, then returns that as the initial observation.

### How `env.step()` works

1. Applies the action (calls planner parameter services synchronously).
2. Blocks on `obs_queue.get()` waiting for the next message.
3. If it is a `('step', ...)` message: computes reward, returns `terminated=False`.
4. If it is a `('done', ...)` message: applies the terminal reward from
   `EpisodeMetrics.goal_reached`, returns `terminated=True`.
5. SB3 sees `terminated=True` and calls `env.reset()` to start the next episode.

### How `episode_runner` handles RL mode

When `rl_mode=true` is set as a ROS parameter, `episode_runner` behaves
differently from its default benchmark mode:

- After completing each episode it goes to `WAITING_FOR_RL` instead of
  immediately starting the next one.
- It wraps the episode index (instead of advancing through `RUN_CONFIGS`) so
  training can run indefinitely.
- The planner/preset are **not** set by `episode_runner` — that is the RL
  agent's job via its actions.

See `doc/episode_runner_state_machine.md` for the full state diagram.

---

## Training Entry Point

`train.py` is the script you run to start training:

```bash
python3 train.py --action-mode continuous --planner MPPI --timesteps 100000
```

| Argument | Default | Description |
|---|---|---|
| `--action-mode` | `continuous` | `discrete` or `continuous` |
| `--planner` | `MPPI` | Active planner in continuous mode |
| `--timesteps` | `100000` | Total environment steps to train for |
| `--save` | `potr_policy` | Output filename for the saved model |
| `--check` | off | Run gymnasium's built-in env checker before training |

**Discrete mode** uses PPO (Proximal Policy Optimisation), which handles both
discrete and continuous actions.

**Continuous mode** uses SAC (Soft Actor-Critic), which is generally more sample
efficient for continuous action spaces because it also maximises entropy
(encouraging the agent to explore).

The trained model is saved as `<name>.zip` and can be loaded later for
evaluation:

```python
from stable_baselines3 import SAC
from potr_rl.env import PotrNavEnv

env   = PotrNavEnv(action_mode='continuous', planner='MPPI')
model = SAC.load('potr_policy', env=env)

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

---

---

## Running It — Step by Step

### Prerequisites

Everything runs in two separate terminals.  Both need the ROS2 workspace sourced.

```bash
# Run this at the start of every terminal session
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash   # adjust path to your workspace
```

Install the Python RL dependencies once (outside the ROS2 workspace):

```bash
pip install stable-baselines3 gymnasium
```

---

### Step 1 — Enable RL mode in the config

Open `rbe_capstone/simulation_launch/config/goalpoints_episode.yaml` and add
`rl_mode: true` under `ros__parameters`:

```yaml
episode_runner:
  ros__parameters:
    default_map: 'mixed'
    map_to_odom_x: -4.0
    map_to_odom_y:  0.0
    rl_mode: true          # ← add this line
    episodes: >
      [ ... ]
```

This puts `episode_runner` into the mode where it waits for `start_episode`
service calls rather than running all benchmark configs automatically.

> **To switch back to benchmark mode**, remove or set `rl_mode: false` and
> rebuild.  Everything else stays the same.

Rebuild after any config or code change:

```bash
cd ~/ros2_ws    # your colcon workspace root
colcon build --packages-select potr_navigation simulation_launch potr_rl
```

---

### Step 2 — Launch the simulation (Terminal 1)

```bash
ros2 launch simulation_launch potr_robot.launch.py
```

Optional arguments:

```bash
# Run at 3× real time to speed up training
ros2 launch simulation_launch potr_robot.launch.py speed_factor:=3.0

# Use a different starting map
ros2 launch simulation_launch potr_robot.launch.py map_name:=office
```

**What you should see:**
- RViz opens showing the map and the robot at the origin
- Terminal output from `episode_runner`: `RL mode: 5 episodes, waiting for start_episode calls`
- The robot does **not** move yet — it is waiting for the training script

---

### Step 3 — Start training (Terminal 2)

```bash
cd rbe_capstone/potr_rl
python3 train.py --action-mode continuous --planner MPPI --timesteps 50000
```

**What you should see in Terminal 2:**

```
Training SAC (continuous, MPPI) for 50000 steps...
---------------------------------
| rollout/            |         |
|    ep_len_mean      | 312     |
|    ep_rew_mean      | -23.4   |
| time/               |         |
|    total_timesteps  | 1024    |
---------------------------------
```

SB3 prints a summary every few thousand steps.  `ep_rew_mean` is the key
number — it should trend upward as the agent learns better parameters.

**What you should see in Terminal 1 / RViz:**
- The robot spawns at the episode start position
- Nav2 starts planning and the robot begins moving toward the goal
- After reaching the goal (or timing out), the robot respawns and a new episode begins
- This cycle repeats automatically for the entire training run

When training finishes:

```
Model saved to potr_policy.zip
```

---

### Step 4 — Verify the data stream (optional but useful)

While the simulation is running, open extra terminals to inspect what is
flowing through the system.

**Watch the per-step observation stream** (fires ~10 Hz while the robot moves):

```bash
ros2 topic echo /potr_navigation/step_metrics
```

Example output:
```
distance_to_goal: 8.42
heading_error_to_goal: 0.15
linear_velocity: 0.48
angular_velocity: 0.12
clearance: 1.83
path_deviation: 0.04
collision: false
```

**Watch episode summaries** (fires once per episode):

```bash
ros2 topic echo /potr_navigation/episode_metrics
```

Example output:
```
planner: MPPI
preset: 1
map_name: mixed
goal_id: ep_1
goal_reached: true
total_time: 41.2
total_distance: 18.7
min_clearance: 0.31
mean_path_deviation: 0.08
collision_count: 0
```

**Check that the agent is sending parameter changes** (continuous mode):

```bash
ros2 service list | grep potr
# should include /potr_navigation/set_raw_params

ros2 topic hz /potr_navigation/step_metrics
# should show ~10 Hz while robot is moving
```

---

### Step 5 — Evaluate the trained model

Once you have a saved `potr_policy.zip`, you can run the agent without further
training to see how it performs.  Create a short script `eval.py`:

```python
from stable_baselines3 import SAC
from potr_rl.env import PotrNavEnv

env   = PotrNavEnv(action_mode='continuous', planner='MPPI')
model = SAC.load('potr_policy', env=env)

obs, _ = env.reset()
episode_rewards = []
current_reward = 0.0

for _ in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    current_reward += reward
    if terminated or truncated:
        ep = info.get('episode_metrics')
        if ep:
            print(f'Episode done — goal_reached={ep.goal_reached}  '
                  f'time={ep.total_time:.1f}s  '
                  f'collisions={ep.collision_count}  '
                  f'reward={current_reward:.1f}')
        current_reward = 0.0
        obs, _ = env.reset()

env.close()
```

```bash
python3 eval.py
```

`deterministic=True` tells SB3 to use the mean of the learned policy rather
than sampling — this gives consistent, repeatable behaviour and is what you
would use in a real deployment.

---

### Step 6 — Monitor training with TensorBoard (optional)

Pass a log directory to `train.py` to enable TensorBoard logging.  First, edit
the model constructor in `train.py`:

```python
model = SAC('MlpPolicy', env, verbose=1, tensorboard_log='./tb_logs/')
```

Then during or after training:

```bash
tensorboard --logdir ./tb_logs/
# open http://localhost:6006 in a browser
```

Useful plots to watch:
- `rollout/ep_rew_mean` — average reward per episode (should rise)
- `rollout/ep_len_mean` — average steps per episode (shorter = more efficient)
- `train/actor_loss`, `train/critic_loss` — should decrease and stabilise

---

### Benchmark mode (no RL)

If you want to run the original four-config benchmark (DWB/MPPI × preset 1/2)
without any RL involvement, set `rl_mode: false` (or omit it) in
`goalpoints_episode.yaml` and launch normally:

```bash
ros2 launch simulation_launch potr_robot.launch.py
```

`episode_runner` will automatically cycle through all four configs and all five
episodes, publishing an `EpisodeMetrics` message at the end of each one.
Monitor results with:

```bash
ros2 topic echo /potr_navigation/episode_metrics
```

---

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Robot does not move after launch | `rl_mode=true` and training script not started | Start `train.py` or set `rl_mode: false` |
| `start_episode` service not found | `episode_runner` not yet started or crashed | Check Terminal 1 for errors |
| `ep_rew_mean` stuck at very negative values | Robot colliding or never reaching goals | Check clearance in step_metrics; try lower speed_factor |
| Timestamp mismatch warnings from collision_monitor | `use_sim_time` not set consistently | Ensure `nav2_bringup.launch.py` has `'use_sim_time': 'true'` |
| Training hangs at first `reset()` | `step_metrics` not publishing | Check `metrics_tracker` is running; check `/odom` topic is active |

---

## File Summary

| File | Role |
|---|---|
| `potr_rl/potr_rl/env.py` | `PotrNavEnv` — the gymnasium environment and ROS2 bridge |
| `potr_rl/potr_rl/params.py` | Action/observation space bounds, param ranges, reward weights |
| `potr_rl/train.py` | Training entry point (PPO or SAC via SB3) |
| `potr_navigation/msg/StepMetrics.msg` | Per-step observation message published by `metrics_tracker` |
| `potr_navigation/srv/SetRawParams.srv` | Service for the RL agent to push raw parameter values |
| `potr_navigation/scripts/metrics_tracker.py` | Publishes `StepMetrics` every odom tick; tracks current goal |
| `potr_navigation/scripts/planner_controller.py` | Handles `set_raw_params` service for continuous mode |
| `potr_navigation/scripts/episode_runner.py` | Episode lifecycle; exposes `start_episode` service in RL mode |
