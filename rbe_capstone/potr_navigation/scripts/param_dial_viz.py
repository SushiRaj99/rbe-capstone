#!/usr/bin/env python3
"""Live gauge-style viewer for the current Nav2 planner params.

Subscribes to /potr_navigation/current_planner_params (std_msgs/String, JSON
payload published by planner_controller.py) and draws one horizontal gauge per
param, with the preset-1 baseline marked as a gray tick for reference.

Run (needs an interactive matplotlib backend — do NOT run inside a headless
Docker):
    ros2 run potr_navigation param_dial_viz
"""
import json
import threading

import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import String


PARAM_RANGES = {
    'MPPI': {
        'max_linear_vel':  (0.1, 1.5),
        'max_angular_vel': (0.3, 2.5),
        'linear_accel':    (0.5, 3.5),
        'angular_accel':   (0.5, 4.0),
    },
    'DWB': {
        'max_linear_vel':   (0.1, 1.5),
        'max_angular_vel':  (0.3, 2.5),
        'linear_accel':     (0.50, 3.50),
        'angular_accel':    (0.50, 4.00),
        'goal_align_scale': (5.0,  40.0),
        'path_align_scale': (5.0,  40.0),
        'goal_dist_scale':  (5.0,  40.0),
        'path_dist_scale':  (5.0,  40.0),
        'obstacle_scale':   (0.005, 0.1),
    },
}

BASELINES = {
    'MPPI': {
        'max_linear_vel':  0.8,
        'max_angular_vel': 1.2,
        'linear_accel':    1.5,
        'angular_accel':   2.0,
    },
    'DWB': {
        'max_linear_vel':   0.8,
        'max_angular_vel':  1.2,
        'linear_accel':     1.5,
        'angular_accel':    2.0,
        'goal_align_scale': 20.0,
        'path_align_scale': 32.0,
        'goal_dist_scale':  20.0,
        'path_dist_scale':  32.0,
        'obstacle_scale':   0.05,
    },
}


class ParamDialNode(Node):
    def __init__(self):
        super().__init__('param_dial_viz')
        self.latest = None
        # Match planner_controller's latched publisher so we get the most recent
        # snapshot as soon as we connect, not just future updates.
        latched_qos = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(
            String, '/potr_navigation/current_planner_params',
            self.on_params, latched_qos,
        )

    def on_params(self, msg):
        try:
            self.latest = json.loads(msg.data)
        except Exception as e:
            self.get_logger().warn(f'Bad params msg: {e}')


def draw(ax, snapshot):
    ax.clear()
    if snapshot is None:
        ax.text(0.5, 0.5, 'waiting for planner_controller...',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    planner   = snapshot.get('planner', 'DWB')
    preset    = snapshot.get('preset',  1)
    values    = snapshot.get('values',  {})
    ranges    = PARAM_RANGES.get(planner, {})
    baselines = BASELINES.get(planner, {})
    names = list(ranges.keys())
    n = len(names)

    for i, name in enumerate(names):
        y  = n - 1 - i
        lo, hi = ranges[name]
        span   = hi - lo
        base   = baselines.get(name, lo)
        val    = values.get(name)
        nbase  = (base - lo) / span
        nval   = (val  - lo) / span if val is not None else None

        # track
        ax.plot([0, 1], [y, y],
                color='lightgray', linewidth=12, solid_capstyle='round',
                zorder=1)
        # preset-1 baseline tick
        ax.plot([nbase], [y],
                marker='|', color='dimgray',
                markersize=22, markeredgewidth=2, zorder=3)

        if nval is not None:
            colour = 'steelblue' if nval >= nbase else 'darkorange'
            ax.plot([0, nval], [y, y],
                    color=colour, linewidth=12, solid_capstyle='round',
                    zorder=2)
            ax.plot([nval], [y],
                    marker='o', color=colour, markersize=12,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=4)

        ax.text(-0.02, y, name, ha='right', va='center', fontsize=10)
        label = f'{val:.3f}' if val is not None else '—'
        ax.text(1.02, y, label, ha='left', va='center',
                fontsize=10, family='monospace')
        ax.text(0.0, y - 0.38, f'{lo:g}', ha='center',
                fontsize=7, color='gray')
        ax.text(1.0, y - 0.38, f'{hi:g}', ha='center',
                fontsize=7, color='gray')

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_title(f'{planner}  (preset {preset} baseline = |)', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    rclpy.init()
    node = ParamDialNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.canvas.manager.set_window_title('POTR planner params')

    try:
        while rclpy.ok() and plt.fignum_exists(fig.number):
            draw(ax, node.latest)
            fig.canvas.draw_idle()
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
