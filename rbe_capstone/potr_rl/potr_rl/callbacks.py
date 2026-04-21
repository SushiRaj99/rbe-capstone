#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

WINDOW = 10


class LivePlotCallback(BaseCallback):
    def __init__(self, save_path='training_metrics.png', update_every=8, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.update_every = update_every

        self.ep_rewards = []
        self.ep_lengths = []
        self.ent_coefs = []
        self.ep_switches = []
        self.ep_deltas = []

        self.ep_r_progress = []
        self.ep_r_pathdev = []
        self.ep_r_angvel = []
        self.ep_r_proximity = []
        self.ep_r_time = []
        self.ep_r_terminal = []

        self.ep_goal = []
        self.ep_fail = []
        self.ep_trunc = []

        self.ep_collision_frac = []
        self.ep_final_distance = []

        self.n_episodes = 0

    def _on_step(self):
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self.n_episodes += 1

            if self.model.ep_info_buffer:
                info = self.model.ep_info_buffer[-1]
                self.ep_rewards.append(float(info['r']))
                self.ep_lengths.append(float(info['l']))

            ep_info = self.locals.get('infos', [{}])[0]
            if 'param_switches' in ep_info:
                self.ep_switches.append(int(ep_info['param_switches']))
            if 'mean_delta' in ep_info:
                self.ep_deltas.append(float(ep_info['mean_delta']))

            if 'r_progress' in ep_info:
                self.ep_r_progress.append(float(ep_info['r_progress']))
                self.ep_r_pathdev.append(float(ep_info['r_pathdev']))
                self.ep_r_angvel.append(float(ep_info['r_angvel']))
                self.ep_r_proximity.append(float(ep_info['r_proximity']))
                self.ep_r_time.append(float(ep_info.get('r_time', 0.0)))
                self.ep_r_terminal.append(float(ep_info['r_terminal']))
                self.ep_collision_frac.append(float(ep_info['collision_frac']))
                self.ep_final_distance.append(float(ep_info['final_distance']))
                reason = ep_info.get('termination', 'truncated')
                self.ep_goal.append(1 if reason == 'goal' else 0)
                self.ep_fail.append(1 if reason == 'fail' else 0)
                self.ep_trunc.append(1 if reason == 'truncated' else 0)

            vals = self.model.logger.name_to_value
            if 'train/ent_coef' in vals:
                self.ent_coefs.append(float(vals['train/ent_coef']))

            if self.n_episodes % self.update_every == 0:
                self.save_plot()

        return True

    def on_training_end(self):
        self.save_plot()

    def rolling(self, data):
        kernel = np.ones(WINDOW) / WINDOW
        return np.convolve(data, kernel, mode='valid')

    def save_plot(self):
        if not self.ep_rewards:
            return

        has_ent = bool(self.ent_coefs)
        has_switches = bool(self.ep_switches)
        has_breakdown = bool(self.ep_r_progress)
        has_termination = bool(self.ep_goal)
        has_extras = bool(self.ep_collision_frac)
        n_plots = 2 + int(has_termination) + int(has_breakdown) + int(has_extras) + int(has_ent) + int(has_switches)

        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.2 * n_plots))
        fig.suptitle(f'Training - {self.n_episodes} episodes  ({self.num_timesteps} steps)', fontsize=13)

        episodes = np.arange(1, len(self.ep_rewards) + 1)

        # Reward
        ax = axes[0]
        ax.plot(episodes, self.ep_rewards, alpha=0.25, color='steelblue', linewidth=0.8, label='raw')
        if len(self.ep_rewards) >= WINDOW:
            ax.plot(np.arange(WINDOW, len(self.ep_rewards) + 1), self.rolling(self.ep_rewards), color='steelblue', linewidth=2, label=f'{WINDOW}-ep mean')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylabel('Episode reward')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Episode length
        ax = axes[1]
        ax.plot(episodes, self.ep_lengths, alpha=0.25, color='darkorange', linewidth=0.8, label='raw')
        if len(self.ep_lengths) >= WINDOW:
            ax.plot(np.arange(WINDOW, len(self.ep_lengths) + 1), self.rolling(self.ep_lengths), color='darkorange', linewidth=2, label=f'{WINDOW}-ep mean')
        ax.set_ylabel('Episode length (blocks)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        next_ax = 2

        # Termination
        if has_termination:
            ax = axes[next_ax]
            next_ax += 1
            ep = np.arange(1, len(self.ep_goal) + 1)
            if len(self.ep_goal) >= WINDOW:
                xs = np.arange(WINDOW, len(self.ep_goal) + 1)
                ax.plot(xs, self.rolling(self.ep_goal), color='seagreen', linewidth=1.8, label='goal')
                ax.plot(xs, self.rolling(self.ep_fail), color='firebrick', linewidth=1.8, label='fail')
                ax.plot(xs, self.rolling(self.ep_trunc), color='slategray', linewidth=1.8, label='truncated')
            else:
                ax.plot(ep, self.ep_goal, color='seagreen', alpha=0.6, label='goal')
                ax.plot(ep, self.ep_fail, color='firebrick', alpha=0.6, label='fail')
                ax.plot(ep, self.ep_trunc, color='slategray', alpha=0.6, label='truncated')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Termination rate')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Reward breakdown
        if has_breakdown:
            ax = axes[next_ax]
            next_ax += 1
            components = [
                ('progress', self.ep_r_progress, 'steelblue'),
                ('path_dev', self.ep_r_pathdev, 'goldenrod'),
                ('ang_vel', self.ep_r_angvel, 'purple'),
                ('proximity', self.ep_r_proximity, 'firebrick'),
                ('time', self.ep_r_time, 'gray'),
                ('terminal', self.ep_r_terminal, 'seagreen'),
            ]
            for name, series, color in components:
                if len(series) >= WINDOW:
                    xs = np.arange(WINDOW, len(series) + 1)
                    ax.plot(xs, self.rolling(series), color=color, linewidth=1.6, label=name)
                else:
                    ax.plot(np.arange(1, len(series) + 1), series, color=color, alpha=0.6, label=name)
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_ylabel('Reward contribution')
            ax.legend(loc='upper right', fontsize=8, ncol=5)
            ax.grid(True, alpha=0.3)

        # Collision fraction + final distance
        if has_extras:
            ax = axes[next_ax]
            next_ax += 1
            ep = np.arange(1, len(self.ep_collision_frac) + 1)
            ax.plot(ep, self.ep_collision_frac, color='firebrick', alpha=0.25, linewidth=0.8)
            if len(self.ep_collision_frac) >= WINDOW:
                ax.plot(np.arange(WINDOW, len(self.ep_collision_frac) + 1), self.rolling(self.ep_collision_frac), color='firebrick', linewidth=1.8, label='collision frac')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Collision fraction', color='firebrick')
            ax.tick_params(axis='y', labelcolor='firebrick')
            ax.grid(True, alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(ep, self.ep_final_distance, color='darkcyan', alpha=0.25, linewidth=0.8)
            if len(self.ep_final_distance) >= WINDOW:
                ax2.plot(np.arange(WINDOW, len(self.ep_final_distance) + 1), self.rolling(self.ep_final_distance), color='darkcyan', linewidth=1.8, label='final dist (m)')
            ax2.set_ylabel('Final distance to goal (m)', color='darkcyan')
            ax2.tick_params(axis='y', labelcolor='darkcyan')

        # Entropy coef
        if has_ent:
            ax = axes[next_ax]
            next_ax += 1
            ax.plot(np.arange(1, len(self.ent_coefs) + 1), self.ent_coefs, color='mediumseagreen', linewidth=1.5)
            ax.set_ylabel('ent_coef')
            ax.grid(True, alpha=0.3)

        # Param switches + delta
        if has_switches:
            ax = axes[next_ax]
            next_ax += 1
            sw_ep = np.arange(1, len(self.ep_switches) + 1)
            ax.bar(sw_ep, self.ep_switches, color='slateblue', alpha=0.6, label='switches / episode')
            if self.ep_deltas:
                ax2 = ax.twinx()
                ax2.plot(np.arange(1, len(self.ep_deltas) + 1), self.ep_deltas, color='tomato', linewidth=1.5, label='mean delta (smoothed)')
                ax2.set_ylabel('mean |delta action|', color='tomato')
                ax2.tick_params(axis='y', labelcolor='tomato')
            ax.set_ylabel('param switches', color='slateblue')
            ax.tick_params(axis='y', labelcolor='slateblue')
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Episode')

        plt.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(self.save_path)), exist_ok=True)
        fig.savefig(self.save_path, dpi=120)
        plt.close(fig)
        if self.verbose:
            print(f'Plot saved -> {self.save_path}')
