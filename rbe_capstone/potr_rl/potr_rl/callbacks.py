#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless — no display needed in Docker
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

_WINDOW = 10  # rolling mean window size


class LivePlotCallback(BaseCallback):
    """
    Saves a training-progress PNG after every `update_every` episodes and
    once more at the end of training.  Works headless inside Docker — open
    the file and refresh it to monitor progress.
    """

    def __init__(self, save_path: str = 'training_metrics.png',
                 update_every: int = 8, verbose: int = 0):
        super().__init__(verbose)
        self.save_path    = save_path
        self.update_every = update_every

        self._ep_rewards:   list = []
        self._ep_lengths:   list = []
        self._ent_coefs:    list = []
        self._ep_switches:  list = []   # param switches per episode
        self._ep_deltas:    list = []   # mean smoothed action delta per episode

        self._ep_r_progress:  list = []
        self._ep_r_pathdev:   list = []
        self._ep_r_angvel:    list = []
        self._ep_r_proximity: list = []
        self._ep_r_time:      list = []
        self._ep_r_terminal:  list = []

        self._ep_goal:        list = []
        self._ep_fail:        list = []
        self._ep_trunc:       list = []

        self._ep_collision_frac: list = []
        self._ep_final_distance: list = []

        self._n_episodes    = 0

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [False])
        if dones[0]:
            self._n_episodes += 1

            if self.model.ep_info_buffer:
                info = self.model.ep_info_buffer[-1]
                self._ep_rewards.append(float(info['r']))
                self._ep_lengths.append(float(info['l']))

            ep_info = self.locals.get('infos', [{}])[0]
            if 'param_switches' in ep_info:
                self._ep_switches.append(int(ep_info['param_switches']))
            if 'mean_delta' in ep_info:
                self._ep_deltas.append(float(ep_info['mean_delta']))

            if 'r_progress' in ep_info:
                self._ep_r_progress.append(float(ep_info['r_progress']))
                self._ep_r_pathdev.append(float(ep_info['r_pathdev']))
                self._ep_r_angvel.append(float(ep_info['r_angvel']))
                self._ep_r_proximity.append(float(ep_info['r_proximity']))
                self._ep_r_time.append(float(ep_info.get('r_time', 0.0)))
                self._ep_r_terminal.append(float(ep_info['r_terminal']))
                self._ep_collision_frac.append(float(ep_info['collision_frac']))
                self._ep_final_distance.append(float(ep_info['final_distance']))
                reason = ep_info.get('termination', 'truncated')
                self._ep_goal.append(1 if reason == 'goal' else 0)
                self._ep_fail.append(1 if reason == 'fail' else 0)
                self._ep_trunc.append(1 if reason == 'truncated' else 0)

            vals = self.model.logger.name_to_value
            if 'train/ent_coef' in vals:
                self._ent_coefs.append(float(vals['train/ent_coef']))

            if self._n_episodes % self.update_every == 0:
                self._save_plot()

        return True

    def on_training_end(self) -> None:
        self._save_plot()

    # ------------------------------------------------------------------

    def _rolling(self, data: list) -> np.ndarray:
        kernel = np.ones(_WINDOW) / _WINDOW
        return np.convolve(data, kernel, mode='valid')

    def _save_plot(self) -> None:
        if not self._ep_rewards:
            return

        has_ent        = bool(self._ent_coefs)
        has_switches   = bool(self._ep_switches)
        has_breakdown  = bool(self._ep_r_progress)
        has_termination = bool(self._ep_goal)
        has_extras     = bool(self._ep_collision_frac)
        n_plots = (
            2
            + int(has_termination)
            + int(has_breakdown)
            + int(has_extras)
            + int(has_ent)
            + int(has_switches)
        )
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3.2 * n_plots))
        fig.suptitle(f'Training — {self._n_episodes} episodes  '
                     f'({self.num_timesteps} steps)', fontsize=13)

        episodes = np.arange(1, len(self._ep_rewards) + 1)

        # Reward
        ax = axes[0]
        ax.plot(episodes, self._ep_rewards,
                alpha=0.25, color='steelblue', linewidth=0.8, label='raw')
        if len(self._ep_rewards) >= _WINDOW:
            ax.plot(np.arange(_WINDOW, len(self._ep_rewards) + 1),
                    self._rolling(self._ep_rewards),
                    color='steelblue', linewidth=2,
                    label=f'{_WINDOW}-ep mean')
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_ylabel('Episode reward')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Episode length
        ax = axes[1]
        ax.plot(episodes, self._ep_lengths,
                alpha=0.25, color='darkorange', linewidth=0.8, label='raw')
        if len(self._ep_lengths) >= _WINDOW:
            ax.plot(np.arange(_WINDOW, len(self._ep_lengths) + 1),
                    self._rolling(self._ep_lengths),
                    color='darkorange', linewidth=2,
                    label=f'{_WINDOW}-ep mean')
        ax.set_ylabel('Episode length (blocks)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        next_ax = 2

        # Termination breakdown (rolling fraction of goal/fail/truncated)
        if has_termination:
            ax = axes[next_ax]; next_ax += 1
            ep = np.arange(1, len(self._ep_goal) + 1)
            if len(self._ep_goal) >= _WINDOW:
                xs = np.arange(_WINDOW, len(self._ep_goal) + 1)
                ax.plot(xs, self._rolling(self._ep_goal),
                        color='seagreen',  linewidth=1.8, label='goal')
                ax.plot(xs, self._rolling(self._ep_fail),
                        color='firebrick', linewidth=1.8, label='fail')
                ax.plot(xs, self._rolling(self._ep_trunc),
                        color='slategray', linewidth=1.8, label='truncated')
            else:
                ax.plot(ep, self._ep_goal,  color='seagreen',  alpha=0.6, label='goal')
                ax.plot(ep, self._ep_fail,  color='firebrick', alpha=0.6, label='fail')
                ax.plot(ep, self._ep_trunc, color='slategray', alpha=0.6, label='truncated')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Termination rate')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Reward component breakdown (rolling means)
        if has_breakdown:
            ax = axes[next_ax]; next_ax += 1
            components = [
                ('progress',  self._ep_r_progress,  'steelblue'),
                ('path_dev',  self._ep_r_pathdev,   'goldenrod'),
                ('ang_vel',   self._ep_r_angvel,    'purple'),
                ('proximity', self._ep_r_proximity, 'firebrick'),
                ('time',      self._ep_r_time,      'gray'),
                ('terminal',  self._ep_r_terminal,  'seagreen'),
            ]
            for name, series, color in components:
                if len(series) >= _WINDOW:
                    xs = np.arange(_WINDOW, len(series) + 1)
                    ax.plot(xs, self._rolling(series),
                            color=color, linewidth=1.6, label=name)
                else:
                    ax.plot(np.arange(1, len(series) + 1), series,
                            color=color, alpha=0.6, label=name)
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.set_ylabel('Reward contribution')
            ax.legend(loc='upper right', fontsize=8, ncol=5)
            ax.grid(True, alpha=0.3)

        # Collision fraction + final distance (twin axis)
        if has_extras:
            ax = axes[next_ax]; next_ax += 1
            ep = np.arange(1, len(self._ep_collision_frac) + 1)
            ax.plot(ep, self._ep_collision_frac,
                    color='firebrick', alpha=0.25, linewidth=0.8)
            if len(self._ep_collision_frac) >= _WINDOW:
                ax.plot(np.arange(_WINDOW, len(self._ep_collision_frac) + 1),
                        self._rolling(self._ep_collision_frac),
                        color='firebrick', linewidth=1.8, label='collision frac')
            ax.set_ylim(-0.05, 1.05)
            ax.set_ylabel('Collision fraction', color='firebrick')
            ax.tick_params(axis='y', labelcolor='firebrick')
            ax.grid(True, alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(ep, self._ep_final_distance,
                     color='darkcyan', alpha=0.25, linewidth=0.8)
            if len(self._ep_final_distance) >= _WINDOW:
                ax2.plot(np.arange(_WINDOW, len(self._ep_final_distance) + 1),
                         self._rolling(self._ep_final_distance),
                         color='darkcyan', linewidth=1.8, label='final dist (m)')
            ax2.set_ylabel('Final distance to goal (m)', color='darkcyan')
            ax2.tick_params(axis='y', labelcolor='darkcyan')

        # Entropy coefficient
        if has_ent:
            ax = axes[next_ax]; next_ax += 1
            ax.plot(np.arange(1, len(self._ent_coefs) + 1), self._ent_coefs,
                    color='mediumseagreen', linewidth=1.5)
            ax.set_ylabel('ent_coef')
            ax.grid(True, alpha=0.3)

        # Parameter switch count + mean delta per episode
        if has_switches:
            ax = axes[next_ax]; next_ax += 1
            sw_ep = np.arange(1, len(self._ep_switches) + 1)
            ax.bar(sw_ep, self._ep_switches, color='slateblue', alpha=0.6,
                   label='switches / episode')
            if self._ep_deltas:
                ax2 = ax.twinx()
                ax2.plot(np.arange(1, len(self._ep_deltas) + 1), self._ep_deltas,
                         color='tomato', linewidth=1.5, label='mean Δ (smoothed)')
                ax2.set_ylabel('mean |Δ action|', color='tomato')
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
            print(f'Plot saved → {self.save_path}')
