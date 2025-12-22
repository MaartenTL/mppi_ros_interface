# mppi_online_plot.py
import matplotlib.pyplot as plt
from collections import deque
import numpy as np
import os

class OnlineMppiPlotter:
    def __init__(self, max_history=200,
                 save_dir=None,
                 save_every=1,
                 file_prefix="mppi_diag"):
        self.max_history = max_history

        # --- NEW: saving options ---
        self.save_dir = save_dir
        self.save_every = int(save_every)
        self.file_prefix = file_prefix

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        # ----------------------------

        self.t_hist = deque(maxlen=max_history)
        self.u_throttle_hist = deque(maxlen=max_history)
        self.u_steer_hist    = deque(maxlen=max_history)
        self.du_throttle_hist = deque(maxlen=max_history)
        self.du_steer_hist    = deque(maxlen=max_history)
        self.Neff_hist        = deque(maxlen=max_history)
        self.cost_min_hist    = deque(maxlen=max_history)

        self._last_u0 = None
        self._step    = 0

        plt.ion()
        # self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 12))
        self.fig, self.ax_seq = plt.subplots(1, 1, figsize=(12, 12))

        # self.ax_u = self.axes[0, 0]
        # self.ax_duNeff = self.axes[0, 1]
        # self.ax_cloud = self.axes[1, 0]
        # self.ax_hist = self.axes[1, 1]
        # self.ax_seq = self.axes[0, 1]


    def update(self, u0, Neff, cost_min, u0_samples=None, weights=None, t=None, mean_u=None, best_u=None, filt_u=None):
        """
        u0:          np.array shape (2,) -> [th, st]
        Neff:        float
        cost_min:    float
        u0_samples:  np.array shape (K, 2) or None
        weights:     np.array shape (K,) or None
        t:           step index or time
        """
        self._step += 1
        if t is None:
            t = self._step

        u_throttle = float(u0[0])
        u_steer    = float(u0[1])

        if self._last_u0 is None:
            du_throttle = 0.0
            du_steer    = 0.0
        else:
            du_throttle = u_throttle - self._last_u0[0]
            du_steer    = u_steer    - self._last_u0[1]
        self._last_u0 = np.array([u_throttle, u_steer])

        self.t_hist.append(t)
        self.u_throttle_hist.append(u_throttle)
        self.u_steer_hist.append(u_steer)
        self.du_throttle_hist.append(du_throttle)
        self.du_steer_hist.append(du_steer)
        self.Neff_hist.append(Neff)
        self.cost_min_hist.append(cost_min)

        # self._plot_controls()
        # self._plot_du_and_Neff()
        # self._plot_cloud(u0_samples, weights, u0)
        # self._plot_hist(u0_samples, weights, u0)
        self._plot_action_sequence(mean_u, best_u, filt_u)

        self.fig.tight_layout()

        if self.save_dir is not None and (t % self.save_every == 0):
            fname = f"{self.file_prefix}_step_{t:05d}.png"
            fpath = os.path.join(self.save_dir, fname)
            # Make sure the canvas is up to date
            self.fig.canvas.draw()
            self.fig.savefig(fpath, dpi=150, bbox_inches="tight")

        plt.pause(0.001)

    def _plot_controls(self):
        self.ax_u.clear()
        self.ax_u.set_title("u0 vs time")
        self.ax_u.plot(self.t_hist, self.u_throttle_hist, label="th")
        self.ax_u.plot(self.t_hist, self.u_steer_hist,    label="st")
        self.ax_u.set_xlabel("step")
        self.ax_u.set_ylabel("command")
        self.ax_u.grid(True)
        self.ax_u.legend()

    def _plot_du_and_Neff(self):
        # Clear main axis
        self.ax_duNeff.clear()

        # --- REMOVE OLD SECONDARY AXES IF THEY EXIST ---
        if hasattr(self, "ax_duNeff_twin"):
            try:
                self.ax_duNeff_twin.remove()
            except:
                pass
            self.ax_duNeff_twin = None

        self.ax_duNeff.set_title("|Δu0| and N_eff")

        # primary plot (Δu0)
        self.ax_duNeff.plot(self.t_hist, np.abs(self.du_throttle_hist), label="|Δth|")
        self.ax_duNeff.plot(self.t_hist, np.abs(self.du_steer_hist), label="|Δst|")
        self.ax_duNeff.set_xlabel("step")
        self.ax_duNeff.set_ylabel("|Δu0|")
        self.ax_duNeff.grid(True)

        # secondary y-axis
        self.ax_duNeff_twin = self.ax_duNeff.twinx()
        self.ax_duNeff_twin.plot(self.t_hist, self.Neff_hist, linestyle="--", color="tab:green")
        self.ax_duNeff_twin.set_ylabel("N_eff")

        # merge legends
        lines1, labels1 = self.ax_duNeff.get_legend_handles_labels()
        lines2 = [plt.Line2D([], [], linestyle="--", color="tab:green")]
        labels2 = ["N_eff"]
        self.ax_duNeff.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    def _plot_cloud(self, u0_samples, weights, u0):
        self.ax_cloud.clear()
        self.ax_cloud.set_title("Sample cloud (current step)")
        self.ax_cloud.set_xlabel("steering st")
        self.ax_cloud.set_ylabel("throttle th")
        self.ax_cloud.grid(True)

        if u0_samples is not None and weights is not None:
            u0_samples = np.asarray(u0_samples)
            weights    = np.asarray(weights)
            if weights.size > 0:
                w_norm = weights / (weights.max() + 1e-12)
                sizes = 10 + 90 * w_norm
                self.ax_cloud.scatter(
                    u0_samples[:, 1],  # steering on x
                    u0_samples[:, 0],  # throttle on y
                    s=sizes,
                    alpha=0.5,
                )
        self.ax_cloud.scatter(u0[1], u0[0], c="red", marker="x", s=100, label="u0 exec")
        self.ax_cloud.legend()

    def _plot_hist(self, u0_samples, weights, u0):
        self.ax_hist.clear()

        # --- REMOVE OLD SECONDARY AXES IF THEY EXIST ---
        if hasattr(self, "ax_hist_twin"):
            try:
                self.ax_hist_twin.remove()
            except:
                pass
            self.ax_hist_twin = None

        self.ax_hist.set_title("Weighted histograms (current step)")
        self.ax_hist.grid(True)

        if u0_samples is None or weights is None:
            self.ax_hist.text(0.5, 0.5, "No sample data", ha="center", va="center")
            return

        u0_samples = np.asarray(u0_samples)
        weights = np.asarray(weights)

        if u0_samples.size == 0:
            self.ax_hist.text(0.5, 0.5, "Empty samples", ha="center", va="center")
            return

        throttle = u0_samples[:, 0]
        steer = u0_samples[:, 1]

        # Primary histogram
        self.ax_hist.hist(throttle, bins=30, weights=weights, alpha=0.5, label="th")
        self.ax_hist.axvline(u0[0], linestyle="--")
        self.ax_hist.set_xlabel("th")

        # Secondary histogram on x-axis (top)
        self.ax_hist_twin = self.ax_hist.twiny()
        self.ax_hist_twin.hist(steer, bins=30, weights=weights, alpha=0.3, label="st", color="green")
        self.ax_hist_twin.axvline(u0[1], linestyle="--", color="green")
        self.ax_hist_twin.set_xlabel("st")

        # Collect legend entries from both axes
        handles1, labels1 = self.ax_hist.get_legend_handles_labels()
        handles2, labels2 = self.ax_hist_twin.get_legend_handles_labels()

        # Put the combined legend on the main axis
        self.ax_hist.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    def _plot_action_sequence(self, mean_action, best_traj, filter_traj):
        ax = self.ax_seq
        ax.clear()
        ax.set_title("Action sequence (horizon)")
        ax.set_xlabel("timestep")
        ax.set_ylabel("action value")
        ax.grid(True)

        T = mean_action.shape[0]

        # Plot mean sequence
        ax.plot(range(T), mean_action[:,0], label="mean throttle")
        ax.plot(range(T), mean_action[:,1], label="mean steering")

        # Plot best trajectory
        # ax.plot(range(T), best_traj[:,0], 'r--', alpha=0.6, label="best throttle")
        # ax.plot(range(T), best_traj[:,1], 'g--', alpha=0.6, label="best steering")

        # Plot filtered trajectory
        # ax.plot(range(T), best_traj[:,0], 'r--', alpha=0.6, label="best throttle")
        # ax.plot(range(T), filter_traj[:,1], 'g', alpha=0.6, label="filtered steering")

        ax.legend()
