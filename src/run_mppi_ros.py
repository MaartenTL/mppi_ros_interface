#!/usr/bin/env python3
from typing import Optional

import rospy
import yaml
import torch
import os
import time
import subprocess
import csv
import datetime

import matplotlib.pyplot as plt


from mppi_torch.mppi import MPPIPlanner
from simulator_ros import SimulatorROS
from functions_for_controllers import find_s_of_closest_point_on_global_path, produce_track,produce_marker_array_rviz, produce_marker_rviz, steer_angle_2_command
import numpy as np
from dynamics import Kinematic_Bicycle, Dynamic_Bicycle, SVGP, RateAugmentedDynamics, DynLimRateAugmentedDynamics
from mppi_online_plot import OnlineMppiPlotter
# noinspection PyUnresolvedReferences
from tf.transformations import euler_from_quaternion
from path_track_definitions import generate_track, generate_path_data
# noinspection PyUnresolvedReferences
from dynamic_reconfigure.client import Client
# noinspection PyUnresolvedReferences
from std_msgs.msg import String, Float32, MultiArrayDimension, Float32MultiArray, Int32
from collections import deque
import json
# noinspection PyUnresolvedReferences
from geometry_msgs.msg import PointStamped, Point

from dart_dynamic_models import model_functions,load_SVGPModel_actuator_dynamics_analytic

mf = model_functions()


abs_path = os.path.dirname(os.path.abspath(__file__))
# noinspection PyUnresolvedReferences
from visualization_msgs.msg import MarkerArray, Marker

import numpy as np
import os
import time

class RunRolloutLogger:
    """
    Collect MPPI rollout weights + related controls per control step, then flush once per run.
    Writes a single compressed npz file per run for fast offline analysis.
    """
    def __init__(self, root_dir: str, car_number: int, max_steps: int = 5000, downsample_K: int = None):
        self.root_dir = root_dir
        self.car_number = car_number
        self.max_steps = int(max_steps)
        self.downsample_K = downsample_K  # e.g. 500 to limit file size, or None for full
        os.makedirs(self.root_dir, exist_ok=True)
        self.reset(run_id=0, meta=None)

    def reset(self, run_id: int, meta: Optional[dict]):
        self.run_id = int(run_id)
        self.meta = meta or {}
        self.t0 = time.time()

        # Step-wise arrays stored as Python lists, then stacked at flush.
        self.steps = []
        self.u0 = []
        self.u0_samples = []
        self.weights = []
        self.costs = []
        self.cost_min = []
        self.mean_u = []
        self.best_u = []

    def _maybe_downsample(self, u0_samples: np.ndarray, w: np.ndarray, costs: Optional[np.ndarray]):
        if self.downsample_K is None:
            return u0_samples, w, costs

        K = u0_samples.shape[0]
        if K <= self.downsample_K:
            return u0_samples, w, costs

        # Keep top-N by weight + random remainder (stable stats, smaller files)
        N_top = min(100, self.downsample_K // 2)
        top_idx = np.argsort(-w)[:N_top]

        remaining = self.downsample_K - top_idx.size
        all_idx = np.arange(K)
        mask = np.ones(K, dtype=bool)
        mask[top_idx] = False
        rest_idx = all_idx[mask]
        rand_idx = np.random.choice(rest_idx, size=remaining, replace=False) if remaining > 0 else np.array([], dtype=int)

        sel = np.concatenate([top_idx, rand_idx])
        return u0_samples[sel], w[sel], (costs[sel] if costs is not None else None)

    def log_step(
        self,
        step_idx: int,
        u0: np.ndarray,                 # shape (2,)
        u0_samples: np.ndarray,         # shape (K,2)
        weights: np.ndarray,            # shape (K,)
        costs: Optional[np.ndarray],       # shape (K,) or None
        cost_min: float,
        mean_u: Optional[np.ndarray],      # shape (T,2) or None
        best_u: Optional[np.ndarray]  # shape (T,2) or None
    ):
        if len(self.steps) >= self.max_steps:
            return

        # Ensure float32 to shrink files
        u0 = np.asarray(u0, dtype=np.float32)
        u0_samples = np.asarray(u0_samples, dtype=np.float32)
        weights = np.asarray(weights, dtype=np.float32)
        if costs is not None:
            costs = np.asarray(costs, dtype=np.float32)

        # Normalize weights defensively
        s = float(weights.sum())
        if s > 0:
            weights = weights / s

        u0_samples, weights, costs = self._maybe_downsample(u0_samples, weights, costs)

        self.steps.append(int(step_idx))
        self.u0.append(u0)
        self.u0_samples.append(u0_samples)
        self.weights.append(weights)
        self.cost_min.append(float(cost_min))
        self.costs.append(costs if costs is not None else np.array([], dtype=np.float32))

        # Optional: store mean/best once per step (can be big; keep if you need it)
        if mean_u is not None:
            self.mean_u.append(np.asarray(mean_u, dtype=np.float32))
        else:
            self.mean_u.append(np.array([], dtype=np.float32))

        if best_u is not None:
            self.best_u.append(np.asarray(best_u, dtype=np.float32))
        else:
            self.best_u.append(np.array([], dtype=np.float32))



    def flush(self):
        if len(self.steps) == 0:
            return None

        run_dir = os.path.join(self.root_dir, f"car{self.car_number}")
        os.makedirs(run_dir, exist_ok=True)

        out_path = os.path.join(run_dir, f"run_{self.run_id:03d}.npz")

        np.savez_compressed(
            out_path,
            meta=self.meta,
            wall_time_s=float(time.time() - self.t0),
            steps=np.array(self.steps, dtype=np.int32),
            u0=np.stack(self.u0, axis=0),  # [S,2]

            # Ragged arrays saved as object arrays (each step can have different K if you downsample)
            u0_samples=np.array(self.u0_samples, dtype=object),
            weights=np.array(self.weights, dtype=object),
            costs=np.array(self.costs, dtype=object),

            cost_min=np.array(self.cost_min, dtype=np.float32),
            mean_u=np.array(self.mean_u, dtype=object),
            best_u=np.array(self.best_u, dtype=object),
        )
        return out_path




class StateDisturber:
    def __init__(self,
                 sigma_pos=0.1,        # m
                 sigma_yaw_deg=0.05,     # deg
                 sigma_v=0.05,          # m/s for vx, vy
                 sigma_omega=0.02,      # rad/s
                 bias_yaw_deg=0.0,      # constant heading bias
                 seed=42,
                 latency_steps=0,        # integer delay in control steps
                 dropout_prob=0.0,       # chance to drop a sample
                 outlier_prob=0.0,       # chance to inject an outlier
                 ):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.sigma = np.array([sigma_pos, sigma_pos,
                               np.deg2rad(sigma_yaw_deg),
                               sigma_v, sigma_v, sigma_omega], dtype=float)
        self.bias = np.array([0.0, 0.0,
                              np.deg2rad(bias_yaw_deg),
                              0.0, 0.0, 0.0], dtype=float)
        self.latency_steps = int(latency_steps)
        self.buffer = deque(maxlen=max(1, self.latency_steps+1))
        self.dropout_prob = float(dropout_prob)
        self.outlier_prob = float(outlier_prob)


    def _wrap_angle(self, a):
        return np.arctan2(np.sin(a), np.cos(a))

    def disturb(self, state_tensor):
        """
        state_tensor: torch.Tensor shape [1, 6] (x, y, yaw, vx, vy, omega)
        returns: disturbed torch.Tensor [1, 6]
        """
        s = state_tensor[0].detach().cpu().numpy().astype(float)   # [6]

        if self.rng.random() < self.dropout_prob:
            # hold last sample (simple dropout model)
            if len(self.buffer) > 0:
                s_noisy = self.buffer[-1].copy()
            else:
                s_noisy = s.copy()
        else:
            noise = self.rng.normal(0.0, self.sigma)
            s_noisy = s + self.bias + noise

            # occasional outlier
            if self.rng.random() < self.outlier_prob:
                s_noisy[:2] += self.rng.normal(0.0, 0.5, size=2)   # 0.5 m spike
            # wrap yaw
            s_noisy[2] = self._wrap_angle(s_noisy[2])

        # simple sanity: clamp ridiculous speeds so MPPI doesn't explode
        s_noisy[3] = float(np.clip(s_noisy[3], -8.0, 8.0))  # vx
        s_noisy[4] = float(np.clip(s_noisy[4], -8.0, 8.0))  # vy
        s_noisy[5] = float(np.clip(s_noisy[5], -10.0, 10.0))# omega

        # latency: push to buffer, pop delayed
        self.buffer.append(s_noisy)
        if self.latency_steps > 0 and len(self.buffer) > self.latency_steps:
            s_noisy = self.buffer[-(self.latency_steps+1)]
        return torch.tensor(s_noisy, dtype=state_tensor.dtype, device=state_tensor.device).unsqueeze(0)

class MPPIWeights:
    def __init__(self):

        self.ter_q_v = 0.0 # self.q_v * 0.0 # 1.0
        self.ter_q_vy = 0.0 # self.q_vy * 0.0
        self.ter_q_omega = 0.0 # self.q_omega * 0.0 #0.5
        self.ter_q_lat = 0.0 # self.q_lat * 0.0 # 10.0
        self.ter_q_lag = 0.0 # self.q_lag * 0.0 # 2.0

        self.q_vy = 0.0  # 0.0

        self.q_lat = 0.0 # 0.5
        self.q_lag = 0.0 # 2.0

        self.q_head = 0.1 # 0.1
        self.q_v = 0.1 # 0.1
        self.q_omega = 0.0 # 0.0
        self.q_pos = 1.0 # 1.0

        self.ter_q_head = 0.0 # self.q_head * 0.0 # 1.0
        self.ter_q_pos = 0.0 # self.q_pos * 0.0

        # self.q_lat = 0.1 # 0.5
        # self.q_lag = 0.1 # 2.0
        # self.q_head = 0.0 # 0.1
        # self.q_v = 0.0 # 0.1
        #
        # self.q_omega = 0.01 # 0.0
        # self.q_pos = 0.0 # 1.0
        #
        # self.ter_q_head = 1.0 # self.q_head * 0.0 # 1.0
        # self.ter_q_pos = 1.0 # self.q_pos * 0.0



        self.q_u_throttle = 0.01 #0.01  #5 # 0.01
        self.q_du_throttle = 0.01 # 0.01  # 2 # 0.01

        self.q_u_steering = 0.01 # 0.2 # 0.1
        self.q_du_steering = 0.0


        self.w_lane = 100.0 # 100.0 # 100.0 # 100.0  # weight for lane penalty

        self.w_opponent = 1000.0 # 300.0 # weight for opponent

        self.q_ref_beta  = 0.0 # 0.1    # sideslip proxy: atan2(vy,vx) - yaw
        self.q_ref_omega = 0.01    # yaw-rate penalty
        self.q_ref_v     = 0.0    # speed tracking to V_target
        self.q_ref_lag   = 0.1
        self.q_ref_lat   = 0.1
        self.q_ref_lane  = 100.0  # keep reference inside lane

        self.q_ref_ter_head = 1.0
        self.q_ref_ter_pos = 1.0

        # temperature for reference weights (higher => smoother reference)
        self.lambda_ref  = 0.5
        self.M = 0 # Number of frozen timesteps close to vehicle
        self.ref_alpha = 1.0 # 1.0 take previous reference, 0.0 agressive
        self.ref_v_multiplier = 1.5


class ROSObjective:
    def __init__(self, track_choice, N, dt, V_target, use_obstacle, troubleshoot, use_ref_generator, device="cpu"):

        self.s_vals_global_path,\
        self.x_vals_global_path,\
        self.y_vals_global_path,\
        self.s_4_local_path,\
        self.x_4_local_path,\
        self.y_4_local_path,\
        self.dx_ds, self.dy_ds, self.d2x_ds2, self.d2y_ds2,\
        self.k_vals_global_path,\
        self.k_4_local_path = generate_path_data(track_choice)

        self.V_target = V_target  # target velocity

        self.troubleshoot = troubleshoot

        self.weight   =  MPPIWeights()

        self.N = N
        self.time_horizon = N * dt
        self.dt = dt
        self.previous_path_index = 0  # initial index for closest point in global path
        self.n_points_kernelized = 41  # number of points in the kernelized path (41 for reference)

        self.nu = 2  # control dimension: [steering, throttle]
        self.nx = 6  # state dimension: [x, y, yaw, vx, vy]

        self.current_state = None  # will be set by the simulator

        self.device = device
        self.counter = 0  # to keep track of which reference point to use

        self.rviz_local_path_pub = rospy.Publisher(
            f'rviz_local_path', MarkerArray, queue_size=1
        )

        self.path_len = len(self.s_vals_global_path)
        self.wrap_low = int(0.20 * self.path_len)  # “low” band near 0
        self.wrap_high = int(0.80 * self.path_len)  # “high” band near end
        self.lap_count = 0
        self.lap_pub = rospy.Publisher("lap_count", Int32, queue_size=1, latch=True)
        self.lap_start_time = time.time()
        self.lap_time_pub = rospy.Publisher("lap_time", Float32, queue_size=1)

        self.ref_yaw_cos = 0.0
        self.ref_yaw_sin = 0.0

        self.lane_width = 1.0 # 0.60  # m; you can overwrite via ROS param or GUI later
        self.lane_margin = 0.05
        # m; inner safety buffer

        self.left_lane_pub = rospy.Publisher('left_lane', MarkerArray, queue_size=1)
        self.right_lane_pub = rospy.Publisher('right_lane', MarkerArray, queue_size=1)

        self.obs_s_now = 200.0  # current arc-length of obstacle
        self.obs_v = 1.0  # obstacle speed along track [m/s], for example
        self.obs_c_seq = None  # will hold [T, 2] centres
        self.obs_yaw_seq = None  # will hold [T] yaw angles (rad)

        self.obstacle_c = None
        self.obstacle_Q = torch.tensor([[0, 0],
                          [0, 0]],
                         device=device, dtype=torch.float32)
        self.obstacle_margin = 0.1

        if use_obstacle == "static":
            self.use_obstacle = True
            self.use_dyn_obstacle = False
        elif use_obstacle == "dynamic":
            self.use_obstacle = False
            self.use_dyn_obstacle = True
        else:
            self.use_obstacle = False
            self.use_dyn_obstacle = False

        self.obst_a, self.obst_b = 0.4, 0.15  # ellipse semi-axes (m): x'-axis=a, y'-axis=b
        self.obs_pub = rospy.Publisher('obstacles', MarkerArray, queue_size=1, latch=True)

        self.rviz_ref_pub = rospy.Publisher('rviz_rollout_reference', MarkerArray, queue_size=1)
        self.rviz_gen_ref_pub = rospy.Publisher('rviz_rollout_generator_reference', MarkerArray, queue_size=1)

        # --- rollout-based lagged reference ---
        self.use_rollout_reference = use_ref_generator
        self.rollout_ref_valid = False

        self.X0_con_plan = None  # torch [N,4] = [x,y,yaw,V]
        self.X0_gen_plan = None  # torch [N,4] = [x,y,yaw,V]
        self.ref_yaw_cos_rollout_prev = None
        self.ref_yaw_sin_rollout_prev = None

        self.rollout_ref_alpha = self.weight.ref_alpha  # 0.0 = use only new, 0.7 = heavy inertia

    def _softplus_hinge(self, x: torch.Tensor, sharp=25.0):
        # ≈ max(0, x) but smooth; larger 'sharp' -> steeper wall
        return torch.log1p(torch.exp(sharp * x)) / sharp

    def wrap_angle(self, x: torch.Tensor) -> torch.Tensor:
        # wrap into [-π, π]
        return torch.atan2(torch.sin(x), torch.cos(x))

    def compute_running_cost(self, state: torch.Tensor):
        """
        state: Tensor[K,5] = [x, y, yaw, vx, vy]
        self.nav_goal: Tensor[4] = [ref_x, ref_y, ref_yaw, V_target]
        """
        # unpack
        x, y, yaw, vx, vy, omega = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4], state[:, 5]

        ref_x, ref_y, ref_yaw, V_target = self.get_current_goal()  # each a scalar tensor

        # errors
        dx = x - ref_x
        dy = y - ref_y

        lag_err = dx * self.ref_yaw_cos[self.counter-1] + dy * self.ref_yaw_sin[self.counter-1]
        lat_err = -dx * self.ref_yaw_sin[self.counter-1] + dy * self.ref_yaw_cos[self.counter-1]
        head_err = self.wrap_angle(yaw - ref_yaw)
        speed = torch.sqrt(vx ** 2 + vy ** 2)
        speed_err = speed - V_target
        pos_err = dx ** 2 + dy ** 2


        # cost = (self.weight.q_lat * lat_err ** 2
        #          + self.weight.q_lag * lag_err ** 2
        #         + self.weight.q_head * head_err ** 2
        #         + self.weight.q_v * speed_err ** 2
        #         + self.weight.q_vy * vy ** 2
        #         + self.weight.q_omega * omega ** 2)

        cost = (self.weight.q_pos * pos_err #** 2
                + self.weight.q_head * head_err ** 2
                + self.weight.q_v * speed_err ** 2
                + self.weight.q_vy * vy ** 2
                + self.weight.q_omega * omega ** 2
                + self.weight.q_lag * lag_err ** 2
                + self.weight.q_lat * lat_err ** 2)



        half_w = self.lane_width * 0.5
        excess = torch.abs(lat_err) - (half_w - self.lane_margin)

        c_lane = torch.clamp(self.weight.w_lane * self._softplus_hinge(excess), 0, 300)
        cost = cost + c_lane

        # throttle = state[:,6]
        # steering = state[:,7]
        #
        # control_cost = self.weight.q_u_throttle * throttle ** 2 + self.weight.q_u_steering * steering ** 2
        #
        # cost = cost + control_cost

        if self.use_obstacle:
            c_opponent = self.obstacle_cost(x, y)

            cost += c_opponent

        # if self.use_obstacle:
        #     idx = max(self.counter - 1, 0)
        #     c_t = self.obs_c_seq[idx]  # shape [2]
        #     c_opponent = self._obstacle_cost(x, y, c=c_t)
        #     cost += c_opponent

        if self.use_dyn_obstacle:
            idx = max(self.counter - 1, 0)
            c_t = self.obs_c_seq[idx]  # [2]
            yaw_t = self.obs_yaw_seq[idx]  # scalar tensor
            c_opponent = self.dyn_obstacle_cost(x, y, c=c_t, yaw=yaw_t)
            cost += c_opponent



        return cost

    def terminal_costs(self, states: torch.Tensor, actions: torch.Tensor):

        x, y, yaw, vx, vy, omega = states[:, -1, 0], states[:, -1, 1], states[:, -1, 2], states[:, -1, 3], states[:, -1, 4], states[:, -1, 5]
        ref_x = self.X0_con_plan[-1,0]
        ref_y = self.X0_con_plan[-1,1]
        ref_yaw = self.X0_con_plan[-1,2]
        V_target = self.X0_con_plan[-1,3]

        dx = x - ref_x
        dy = y - ref_y

        # lag_err = dx * self.ref_yaw_cos[-1] + dy * self.ref_yaw_sin[-1]
        lat_err = -dx * self.ref_yaw_sin[-1] + dy * self.ref_yaw_cos[-1]
        head_err = self.wrap_angle(yaw - ref_yaw)
        speed = torch.sqrt(vx ** 2 + vy ** 2)
        speed_err = speed - V_target

        pos_err = dx ** 2 + dy ** 2

        # terminal_cost = (self.weight.ter_q_lat * lat_err ** 2
        #         + self.weight.ter_q_lag * lag_err ** 2
        #         + self.weight.ter_q_head * head_err ** 2
        #         + self.weight.ter_q_v * speed_err ** 2
        #         + self.weight.ter_q_vy * vy ** 2
        #         + self.weight.ter_q_omega * omega ** 2)

        terminal_cost = (self.weight.ter_q_pos * pos_err # ** 2
                + self.weight.ter_q_head * head_err ** 2
                + self.weight.ter_q_v * speed_err ** 2
                + self.weight.ter_q_vy * vy ** 2
                + self.weight.ter_q_omega * omega ** 2)

        steer_seq = actions[:, :, 1]  # [K, T]
        dsteer = (steer_seq[:, 1:] - steer_seq[:, :-1]) / self.dt  # [K, T-1], rate [rad/s]


        throttle_seq = actions[:, :, 0]
        dthrottle = (throttle_seq[:, 1:] - throttle_seq[:, :-1]) / self.dt

        rate_cost = self.weight.q_du_steering * torch.sum(dsteer ** 2, dim=1) + self.weight.q_du_throttle * torch.sum(dthrottle ** 2, dim=1)  # [K]

        control_cost = self.weight.q_u_throttle * torch.sum(actions[:,:,0] ** 2,1) + self.weight.q_u_steering * torch.sum(actions[:,:,1] ** 2,1) + rate_cost

        half_w = self.lane_width * 0.5
        excess_T = torch.abs(lat_err) - (half_w - self.lane_margin)
        c_lane_T = torch.clamp((1.5 * self.weight.w_lane) * self._softplus_hinge(excess_T), 0, 300)

        terminal_cost += control_cost + c_lane_T

        if self.use_obstacle:
            c_opponent_T = self.obstacle_cost(x, y)

            terminal_cost += c_opponent_T

        if self.use_dyn_obstacle:
            c_T = self.obs_c_seq[-1]
            c_opponent_T = self.obstacle_cost(x, y, c=c_T)
            terminal_cost += c_opponent_T


        if self.use_rollout_reference:

            # Freeze trajecotry close to vehicle

            M = self.weight.M

            x_ref, y_ref, w1, J1 = self._compute_reference_from_rollouts(states)

            dx = x_ref[1:] - x_ref[:-1]
            dy = y_ref[1:] - y_ref[:-1]
            yaw_ref = torch.zeros_like(x_ref)
            yaw_ref[:-1] = torch.atan2(dy, dx)
            yaw_ref[-1] = yaw_ref[-2]

            if self.X0_gen_plan is not None and self.rollout_ref_valid:
                X0_rollout_new = self.X0_gen_plan.clone()
                X0_rollout_new[M:, 0] = x_ref[M:]
                X0_rollout_new[M:, 1] = y_ref[M:]
                X0_rollout_new[M:, 2] = yaw_ref[M:]
                X0_rollout_new[M:, 3] = float(self.V_target)

                self.X0_gen_plan = self.X0_track # setting this so the reference stays anchored


                a = float(self.rollout_ref_alpha)
                X0_rollout_new[M:, 0] = a * self.X0_gen_plan[M:, 0] + (1 - a) * X0_rollout_new[M:, 0]
                X0_rollout_new[M:, 1] = a * self.X0_gen_plan[M:, 1] + (1 - a) * X0_rollout_new[M:, 1]

                # recompute yaw after filtering positions
                dx = X0_rollout_new[1:, 0] - X0_rollout_new[:-1, 0]
                dy = X0_rollout_new[1:, 1] - X0_rollout_new[:-1, 1]
                yaw_ref = torch.zeros((self.N,), device=states.device)
                yaw_ref[:-1] = torch.atan2(dy, dx)
                yaw_ref[-1] = yaw_ref[-2]
                X0_rollout_new[:, 2] = yaw_ref

                self.X0_gen_plan = X0_rollout_new.detach()[1:,:]
                # self.X0_gen_plan = X0_rollout_new.detach()
            else:

                self.X0_gen_plan = self.X0_track


            self.ref_yaw_cos_gen_plan = torch.cos(self.X0_gen_plan[:, 2])
            self.ref_yaw_sin_gen_plan = torch.sin(self.X0_gen_plan[:, 2])

        return terminal_cost

    def _publish_top_rollouts_rviz(self, states_top: torch.Tensor, ns="top_rollouts_cost1", base_id=9100):
        """
        states_top: [M, T, 6]
        """
        ma = MarkerArray()
        M, T, _ = states_top.shape

        for j in range(M):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = rospy.Time.now()
            m.ns = ns
            m.id = base_id + j
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.scale.x = 0.015
            m.color.r = 0.0
            m.color.g = 0.7
            m.color.b = 1.0
            m.color.a = 0.5
            m.lifetime = rospy.Duration(0.0)

            x = states_top[j, :, 0].detach().cpu().numpy()
            y = states_top[j, :, 1].detach().cpu().numpy()
            for xi, yi in zip(x, y):
                p = Point()
                p.x = float(xi)
                p.y = float(yi)
                p.z = 0.03
                m.points.append(p)

            ma.markers.append(m)

        self.rviz_ref_pub.publish(ma)

    def _compute_reference_from_rollouts(self, states: torch.Tensor) -> tuple:
        """
        Build a reference trajectory from rollouts using a separate Cost 1.
        states: [K, T, 6] with [x,y,yaw,vx,vy,omega]
        returns: (x_ref_hat[T], y_ref_hat[T], w1[K], J1[K])
        """
        K, T, _ = states.shape

        x = states[:, :, 0]
        y = states[:, :, 1]
        yaw = states[:, :, 2]
        vx = states[:, :, 3]
        vy = states[:, :, 4]
        omega = states[:, :, 5]


        M = self.weight.M

        if self.X0_gen_plan is not None:
            x[:,:M] = self.X0_gen_plan.clone()[:M,0]
            y[:,:M] = self.X0_gen_plan.clone()[:M, 1]
            yaw[:,:M] = self.X0_gen_plan.clone()[:M, 2]

        # speed
        speed = torch.sqrt(vx ** 2 + vy ** 2)  # [K,T]
        v_err = speed - self.V_target

        # sideslip proxy: velocity direction vs heading
        v_dir = torch.atan2(vy, vx)  # [K,T]
        beta = self.wrap_angle(v_dir - yaw)  # [K,T]

        # if self.X0_gen_plan is not None:
        #     ref_x = self.X0_gen_plan[:, 0].unsqueeze(0)  # [1,T]
        #     ref_y = self.X0_gen_plan[:, 1].unsqueeze(0)  # [1,T]
        #     cy = self.ref_yaw_cos.unsqueeze(0)  # [1,T]
        #     sy = self.ref_yaw_sin.unsqueeze(0)  # [1,T]
        #
        # else:
        ref_x = self.X0_track[:, 0].unsqueeze(0)  # [1,T]
        ref_y = self.X0_track[:, 1].unsqueeze(0)  # [1,T]
        cy = self.ref_yaw_cos_track.unsqueeze(0)  # [1,T]
        sy = self.ref_yaw_sin_track.unsqueeze(0)  # [1,T]

        ref_yaw = self.X0_track[:, 2].unsqueeze(0)

        dx = x - ref_x
        dy = y - ref_y
        lat_err = -dx * sy + dy * cy  # [K,T]
        lag_err = dx * cy + dy * sy  # [K,T]
        pos_err = dx ** 2 + dy ** 2

        head_err = self.wrap_angle(yaw - ref_yaw)

        half_w = 0.5 * self.lane_width
        excess = torch.abs(lat_err) - (half_w - self.lane_margin)  # [K,T]
        c_lane = torch.clamp(self.weight.q_ref_lane * self._softplus_hinge(excess), 0, 300)  # [K,T]

        stage_cost1 = (
                self.weight.q_ref_lag * (lag_err ** 2) +
                self.weight.q_ref_lat * (lat_err ** 2) +
                self.weight.q_ref_v * (v_err ** 2) +
                c_lane +
                self.weight.q_ref_omega * (omega ** 2) +
                self.weight.q_ref_beta * (beta ** 2)
        )  # [K,T]

        if self.use_dyn_obstacle:
            # obstacle sequence is [T,2] and yaw is [T]
            stage_cost1 += self.obstacle_cost_batched(
                x, y, c=self.obs_c_seq, yaw=self.obs_yaw_seq
            )
        elif self.use_obstacle:

            stage_cost1 += self.obstacle_cost_batched(x, y)  # static c/Q


        J1 = torch.sum(stage_cost1, dim=1)  # [K]
        # weights for reference (softmax of -J/lambda)

        J1 +=   ( self.weight.q_ref_ter_pos * (pos_err[:,-1] ** 2) +
                self.weight.q_ref_ter_head * (head_err[:,-1] ** 2))

        lam = float(self.weight.lambda_ref)
        J1_shift = J1 - torch.min(J1)
        w1 = torch.softmax(-J1_shift / max(lam, 1e-6), dim=0)  # [K]
        
        # idxs = torch.argsort(J1)[:5]
        # self._publish_top_rollouts_rviz(states[idxs], ns="top_rollouts_cost1", base_id=9100)

        # build reference as weighted mean of positions
        x_ref_hat = torch.sum(w1.view(K, 1) * x, dim=0)  # [T]
        y_ref_hat = torch.sum(w1.view(K, 1) * y, dim=0)  # [T]

        return x_ref_hat, y_ref_hat, w1, J1

    def reachable_prefix_by_vmax(self, x: torch.Tensor, y: torch.Tensor, v_target: float, dt: float, N: int):
        """
        Return points along the input polyline that are reachable within N steps
        assuming max progress per step is ds = v_target*dt.

        Output length: N+1 points at arc-lengths 0, ds, 2ds, ..., N*ds,
        clamped to the available polyline length.
        """
        ds = float(v_target) * float(dt)

        # cumulative arc-length of input polyline
        dx = x[1:] - x[:-1]
        dy = y[1:] - y[:-1]
        seg = torch.sqrt(dx * dx + dy * dy) + 1e-6

        s = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)
        s[1:] = torch.cumsum(seg, dim=0)

        L = s[-1]  # total polyline length
        S_max = N * ds

        # target distances along the polyline (reachable progress)
        s_grid = torch.arange(N, device=x.device, dtype=x.dtype) * ds
        s_grid = torch.clamp(s_grid, 0.0, min(L, S_max))

        # interpolate on the polyline
        idx = torch.searchsorted(s, s_grid)
        idx = torch.clamp(idx, 1, x.shape[0] - 1)

        s0 = s[idx - 1]
        s1 = s[idx]
        w = (s_grid - s0) / (s1 - s0 + 1e-6)

        x_out = x[idx - 1] + w * (x[idx] - x[idx - 1])
        y_out = y[idx - 1] + w * (y[idx] - y[idx - 1])

        reachable_all = (L <= S_max + 1e-6)

        if reachable_all:
            x_out = x
            y_out = y
        return x_out, y_out

    def get_current_goal(self):
        if self.counter == 0:
            cs = self.current_state[0].detach().to('cpu')  # [x, y, yaw, vx, vy]
            xy = cs[:2].numpy()
            x_vals_global_path = self.x_vals_global_path
            y_vals_global_path = self.y_vals_global_path
            s_vals_global_path = self.s_vals_global_path

            estimated_ds = self.current_state[0, 3] * self.dt

            prev_idx = self.previous_path_index
            # ------ HIGH LEVEL SOLVER ------

            self.s, self.current_path_index = find_s_of_closest_point_on_global_path(
                xy,
                s_vals_global_path, x_vals_global_path,
                y_vals_global_path, self.previous_path_index, estimated_ds)

            self.previous_path_index = self.current_path_index

            # Check which lap the DART simualator is on
            if (self.current_path_index < prev_idx and
                    prev_idx >= self.wrap_high and
                    self.current_path_index <= self.wrap_low):

                lap_time = time.time() - self.lap_start_time
                if self.lap_count > 0:
                    self.lap_time_pub.publish(lap_time)
                    rospy.loginfo(f"[logger_node] Lap time: {lap_time}")

                self.lap_start_time = time.time()


                self.lap_count += 1
                self.lap_pub.publish(Int32(self.lap_count))
                rospy.loginfo(
                    f"[mppi_ros] Lap {self.lap_count} detected (index {prev_idx} → {self.current_path_index}).")

            if self.use_rollout_reference:
                # Obtain new reference track for this timestep of MPPI
                self.X0_track = torch.tensor(self.produce_X0_global(
                    V_target=self.V_target * self.weight.ref_v_multiplier,
                    N=self.N,
                    s_start=self.s
                ), dtype= torch.float32, device = self.device)

                Ds_forward = self.V_target * self.weight.ref_v_multiplier * self.time_horizon
                Ds_back = 0.0

                self.ref_yaw_cos_track = torch.cos(self.X0_track[:, 2])
                self.ref_yaw_sin_track = torch.sin(self.X0_track[:, 2])

            else:

                # Obtain new reference track for this timestep of MPPI
                self.X0_con_plan = torch.tensor(self.produce_X0_global(
                    V_target=self.V_target,
                    N=self.N,
                    s_start=self.s
                ), dtype=torch.float32, device=self.device)

                Ds_forward = self.V_target * self.time_horizon
                Ds_back = 0.0

                self.ref_yaw_cos = torch.cos(self.X0_con_plan[:, 2])
                self.ref_yaw_sin = torch.sin(self.X0_con_plan[:, 2])


            if self.use_dyn_obstacle:
                self.build_obstacle_seq()

            mask = (self.s_4_local_path >= self.s - Ds_back) & \
                   (self.s_4_local_path <= self.s + Ds_forward)
            idxs = np.nonzero(mask)[0]

            local_x = self.x_4_local_path[idxs]
            local_y = self.y_4_local_path[idxs]

            rgba = [0.0, 1.0, 0.0, 0.8]  # bright green
            marker_type = 4  # sphere list or LINE_STRIP
            marray = produce_marker_array_rviz(local_x, local_y, rgba, marker_type)
            self.rviz_local_path_pub.publish(marray)

            dx = np.gradient(local_x)
            dy = np.gradient(local_y)
            heading = np.arctan2(dy, dx)

            half_w = self.lane_width * 0.5
            # left/right offset vectors (normal to path)
            nx = -np.sin(heading)
            ny = np.cos(heading)

            x_left = local_x + half_w * nx
            y_left = local_y + half_w * ny
            x_right = local_x - half_w * nx
            y_right = local_y - half_w * ny

            rgba = [57.0, 81.0, 100.0, 1.0]
            marker_type = 4
            left_msg = produce_marker_array_rviz(x_left, y_left, rgba, marker_type)
            right_msg = produce_marker_array_rviz(x_right, y_right, rgba, marker_type)
            self.left_lane_pub.publish(left_msg)
            self.right_lane_pub.publish(right_msg)

        # objective = self.X_track[self.counter, :]  # "cpu") # Faster on cpu

        if self.use_rollout_reference:

            self.ref_yaw_cos = torch.cos(self.X0_track[:, 2])
            self.ref_yaw_sin = torch.sin(self.X0_track[:, 2])

            if self.X0_gen_plan is not None:
                if self.counter == 0:



                    if self.X0_gen_plan.shape[0] == self.N-1:

                        self.X0_gen_plan = torch.cat(
                            [self.X0_gen_plan, self.X0_track[-1, :].unsqueeze(0)],
                            dim=0
                        )
                        

                    ds = float(self.V_target) * float(self.dt)  # desired spacing per step)



                    x_ref_hat, y_ref_hat = self.reachable_prefix_by_vmax(self.X0_gen_plan[:, 0], self.X0_gen_plan[:, 1],
                                                                         self.V_target, self.dt, self.N)

                    dx = x_ref_hat[1:] - x_ref_hat[:-1]
                    dy = y_ref_hat[1:] - y_ref_hat[:-1]
                    yaw_ref_hat = torch.zeros_like(x_ref_hat)
                    yaw_ref_hat[:-1] = torch.atan2(dy, dx)
                    yaw_ref_hat[-1] = yaw_ref_hat[-2]

                    self.X0_con_plan = torch.zeros((self.N, 4), device=self.device, dtype=torch.float32)
                    self.X0_con_plan[:, 0] = x_ref_hat
                    self.X0_con_plan[:, 1] = y_ref_hat
                    self.X0_con_plan[:, 2] = yaw_ref_hat
                    self.X0_con_plan[:, 3] = float(self.V_target)

                    self.ref_yaw_cos = torch.cos(self.X0_con_plan[:, 2])
                    self.ref_yaw_sin = torch.sin(self.X0_con_plan[:, 2])

                    self.rollout_ref_valid = True


                    self._publish_gen_ref_path_rviz(self.X0_gen_plan[:, 0], self.X0_gen_plan[:, 1],
                                                    ns="rollout_ref_lagged", mid=9002)
                    self._publish_ref_path_rviz(self.X0_con_plan[:, 0], self.X0_con_plan[:, 1],
                                                ns="rollout_ref_lagged", mid=9002)

            else:
                self.X0_con_plan = self.X0_track



        objective = self.X0_con_plan[self.counter,:]



        self.counter += 1
        return objective

    def produce_X0_global(self, V_target, N, s_start):
        """
        Build a global‐frame reference of length N starting at arc‐length s_start.
        Returns an (N×4) array: [x, y, yaw, V_target].
        """
        # 1) decide the look‐ahead in meters
        total_horizon = V_target* self.time_horizon
        # create N+1 sample points from s_start → s_start + total_horizon
        s_refs = np.linspace(s_start,
                             s_start + total_horizon,
                             N + 1)

        # 2) wrap around if you exceed the end of your path
        s_max = self.s_vals_global_path[-1]
        s_refs = np.mod(s_refs, s_max)

        # 3) interpolate global x,y
        x_refs = np.interp(s_refs,
                           self.s_vals_global_path,
                           self.x_vals_global_path)
        y_refs = np.interp(s_refs,
                           self.s_vals_global_path,
                           self.y_vals_global_path)

        # 4) compute heading by finite‐difference
        #    d/ds of (x,y) gives unit‐direction
        dx_ds = np.gradient(x_refs, s_refs)
        dy_ds = np.gradient(y_refs, s_refs)
        yaw_refs = np.arctan2(dy_ds, dx_ds)

        # 5) pack into X0 (skip the first point, that's "now")
        X0 = np.zeros((N, 4))
        X0[:, 0] = x_refs[1:]  # x
        X0[:, 1] = y_refs[1:]  # y
        X0[:, 2] = yaw_refs[1:]  # heading
        X0[:, 3] = V_target  # constant speed

        return X0

    def _publish_gen_ref_path_rviz(self, x_ref: torch.Tensor, y_ref: torch.Tensor, ns="ref_from_rollouts", mid=9001):
        """
        Publish the reference as a LINE_STRIP.
        x_ref, y_ref: [T] tensors
        """
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0

        # line width
        m.scale.x = 0.03

        # color
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 0.95

        m.lifetime = rospy.Duration(0.0)

        # fill points
        x_np = x_ref.detach().cpu().numpy()
        y_np = y_ref.detach().cpu().numpy()
        for xi, yi in zip(x_np, y_np):
            p = Point()
            p.x = float(xi)
            p.y = float(yi)
            p.z = 0.05
            m.points.append(p)

        ma.markers.append(m)
        self.rviz_gen_ref_pub.publish(ma)

    def _publish_ref_path_rviz(self, x_ref: torch.Tensor, y_ref: torch.Tensor, ns="ref_from_rollouts", mid=9001):
        """
        Publish the reference as a LINE_STRIP.
        x_ref, y_ref: [T] tensors
        """
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = rospy.Time.now()
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0

        # line width
        m.scale.x = 0.03

        # color
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 0.95

        m.lifetime = rospy.Duration(0.0)

        # fill points
        x_np = x_ref.detach().cpu().numpy()
        y_np = y_ref.detach().cpu().numpy()
        for xi, yi in zip(x_np, y_np):
            p = Point()
            p.x = float(xi)
            p.y = float(yi)
            p.z = 0.05
            m.points.append(p)

        ma.markers.append(m)
        self.rviz_ref_pub.publish(ma)


    def compute_expected_cost(self, state: torch.Tensor):
        x = [tensor[0,0].item() for tensor in state]
        y = [tensor[0,1].item() for tensor in state]
        yaw = [tensor[0,2].item() for tensor in state]
        vx = [tensor[0,3].item() for tensor in state]
        vy = [tensor[0,4].item() for tensor in state]
        omega = [tensor[0,5].item() for tensor in state]

        # cs = self.current_state[0].detach().to('cpu')  # [x, y, yaw, vx, vy]
        # xy = cs[:2].numpy()
        # self.current_state = [x[0],y[0],yaw[0]]

        x_vals_global_path = self.x_vals_global_path
        y_vals_global_path = self.y_vals_global_path
        s_vals_global_path = self.s_vals_global_path

        estimated_ds = vx[0] * self.dt

        prev_idx = self.previous_path_index

        self.s, self.current_path_index = find_s_of_closest_point_on_global_path(
            [x[0],y[0]],
            s_vals_global_path, x_vals_global_path,
            y_vals_global_path, self.previous_path_index, estimated_ds)

        self.previous_path_index = self.current_path_index

        # Obtain new reference track for this timestep of MPPI
        self.X0 = torch.tensor(self.produce_X0_global(
            V_target=self.V_target,
            N=self.N,
            s_start=self.s
        ), dtype=torch.float32, device=self.device)

        self.ref_yaw_cos = torch.cos(self.X0[:, 2])
        self.ref_yaw_sin = torch.sin(self.X0[:, 2])

        ref_x = self.X0[:,0]
        ref_y = self.X0[:,1]
        ref_yaw = self.X0[:,2]
        V_target = self.V_target


        dx = torch.tensor(x[1:]) - ref_x


        dy = torch.tensor(y[1:]) - ref_y
        lag_err = dx * self.ref_yaw_cos + dy * self.ref_yaw_sin
        lat_err = -dx * self.ref_yaw_sin + dy * self.ref_yaw_cos
        head_err = self.wrap_angle(torch.tensor(yaw[1:]) - ref_yaw)

        speed = torch.sqrt(torch.tensor(vx[1:]) ** 2 + torch.tensor(vy[1:]) ** 2)
        speed_err = speed - V_target

        pos_err = dx ** 2 + dy ** 2


        # expected_cost = (self.weight.q_lat * lat_err ** 2
        #         + self.weight.q_lag * lag_err ** 2
        #         + self.weight.q_head * head_err ** 2
        #         + self.weight.q_v * speed_err ** 2
        #         + self.weight.q_vy * torch.Tensor(vy[1:]) ** 2
        #         + self.weight.q_omega * torch.Tensor(omega[1:]) ** 2)

        expected_cost = (self.weight.q_pos * pos_err # ** 2
                + self.weight.q_head * head_err ** 2
                + self.weight.q_v * speed_err ** 2
                + self.weight.q_vy * torch.Tensor(vy[1:]) ** 2
                + self.weight.q_omega * torch.Tensor(omega[1:]) ** 2)

        return expected_cost, self.weight, lat_err, lag_err, head_err, speed_err, torch.Tensor(vy), torch.Tensor(omega), pos_err

    def dyn_obstacle_cost(self, xk, yk, c=None, yaw=None):
        """
        xk, yk: shape [K], positions of all rollouts at current time step.
        c:     shape [2] tensor with obstacle centre.
        yaw:   scalar tensor or float, obstacle heading [rad].
        """
        p = torch.stack([xk, yk], dim=1)  # [K,2]

        # Fallbacks for backward-compatibility
        if c is None:
            c = self.obstacle_c  # [2]
        if yaw is None:
            # old behaviour: use fixed global Q
            d = p - c
            phi = torch.sum((d @ self.obstacle_Q) * d, dim=1) - 1.0
        else:
            # --- new behaviour: oriented ellipse using (a,b,yaw) ---
            if not torch.is_tensor(yaw):
                yaw = torch.tensor(yaw, dtype=torch.float32, device=xk.device)

            d = p - c  # [K,2] in world frame

            cy = torch.cos(yaw)
            sy = torch.sin(yaw)
            # Rotation from obstacle frame to world: R
            # we want world -> obstacle, so use R^T
            R_T = torch.stack([
                torch.stack([cy, sy]),
                torch.stack([-sy, cy]),
            ], dim=0)  # [2,2]

            d_body = d @ R_T  # [K,2], coordinates in obstacle frame

            inv_a2 = 1.0 / (self.obst_a ** 2)
            inv_b2 = 1.0 / (self.obst_b ** 2)

            # ellipse equation in body frame
            phi = (d_body[:, 0] ** 2) * inv_a2 + (d_body[:, 1] ** 2) * inv_b2 - 1.0

        # Same smooth wall as before
        arg = (-phi) + self.obstacle_margin
        c_obs = self.weight.w_opponent * self._softplus_hinge(arg)

        self._publish_obstacles_rviz()

        return torch.clamp(c_obs, 0.0, 1e6)

    def obstacle_cost(self, xk, yk, c=None):
        """
        xk, yk: shape [K], positions of all rollouts at current time step.
        returns: cost per rollout [K]
        """
        p = torch.stack([xk, yk], dim=1)  # [K,2]
        if c is None:
            c = self.obstacle_c  # fallback to static
        # c should be shape [2], broadcast automatically
        d = p - c
        phi = torch.sum((d @ self.obstacle_Q) * d, dim=1) - 1.0
        arg = (-phi) + self.obstacle_margin
        c_obs = self.weight.w_opponent * self._softplus_hinge(arg)

        return torch.clamp(c_obs, 0.0, 1e6)

    def obstacle_cost_batched(self, x, y, c=None, Q=None, yaw=None):
        """
        x,y: [K] or [K,T]
        c:
          - static: [2]
          - time-varying: [T,2] (broadcasted across K)
        yaw:
          - optional time-varying yaw [T] for oriented ellipse (preferred),
            or scalar for static orientation.
        returns:
          - [K] if x is [K]
          - [K,T] if x is [K,T]
        """
        if c is None:
            c = self.obstacle_c
        if Q is None:
            Q = self.obstacle_Q

        # Build p = [...,2]
        p = torch.stack([x, y], dim=-1)

        # Broadcast centre to p shape
        if c.ndim == 1:  # [2]
            c_ = c
        else:  # [T,2] -> [1,T,2] to broadcast over K
            c_ = c.unsqueeze(0)

        d = p - c_  # [...,2]

        phi = torch.sum((d @ Q) * d, dim=-1) - 1.0
        arg = (-phi) + self.obstacle_margin
        c_obs = self.weight.w_opponent * self._softplus_hinge(arg) #* 100
        return torch.clamp(c_obs, 0.0, 1e6)

    def build_obstacle_seq(self):
        # horizon duration
        total_horizon = self.time_horizon  # = N * dt

        # time grid for the horizon
        t_grid = np.arange(self.N) * self.dt  # [0, dt, 2dt, ...]

        # arc length along the path for the obstacle
        s_obs = self.obs_s_now + self.obs_v * t_grid

        s_max = self.s_vals_global_path[-1]
        s_obs = np.mod(s_obs, s_max)

        x_obs = np.interp(s_obs, self.s_vals_global_path, self.x_vals_global_path)
        y_obs = np.interp(s_obs, self.s_vals_global_path, self.y_vals_global_path)

        # 2) compute heading of the obstacle along the path
        #    same trick as produce_X0_global: derivative wrt s → direction
        dx_ds_obs = np.gradient(x_obs, s_obs)
        dy_ds_obs = np.gradient(y_obs, s_obs)
        yaw_obs = np.arctan2(dy_ds_obs, dx_ds_obs)  # [T]

        lateral_offset = 0.5

        self.obs_c_seq = torch.stack(
            [torch.tensor(x_obs, dtype=torch.float32, device=self.device),
             torch.tensor(y_obs, dtype=torch.float32, device=self.device)],
            dim=1  # [T, 2]
        )

        self.obs_yaw_seq = torch.tensor(yaw_obs, dtype=torch.float32, device=self.device)  # [T]

        # self._publish_obstacles_rviz()

    def _publish_obstacles_rviz(self):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.type = Marker.CYLINDER  # flat cylinder as an ellipse footprint
        m.id = 1
        m.pose.position.x = float(self.obs_c_seq[0,0].item())
        m.pose.position.y = float(self.obs_c_seq[0,1].item())
        m.pose.position.z = 0.05

        # yaw -> quaternion (z-rotation)
        yaw = float(self.obs_yaw_seq[0].item())
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = sy
        m.pose.orientation.w = cy

        m.scale.x = 2.0 * self.obst_a # X diameter
        m.scale.y = 2.0 * self.obst_b  # Y diameter
        m.scale.z = 0.1  # thin "puck"

        m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.15, 0.15, 0.6
        m.lifetime = rospy.Duration(0)  # persistent
        ma.markers.append(m)
        self.obs_pub.publish(ma)


def reset_sim(x, y, theta, model_choice, *,
              actuator_dynamics=False, disturbance=False):
    # Set pose + options and raise the reset flag
    dr_client.update_configuration({
        "reset_state_x": float(x),
        "reset_state_y": float(y),
        "reset_state_theta": float(theta),
        "dynamic_model_choice": int(model_choice),
        "actuator_dynamics": bool(actuator_dynamics),
        "disturbance": bool(disturbance),
        "reset_state": True,     # rising edge triggers the callback
    })
    rospy.sleep(0.05)            # short pulse
    # Drop the flag so you can trigger it again later
    dr_client.update_configuration({"reset_state": False})

    sim.send_control(torch.tensor([0.0, 0.0]))

def publish_meta():
    meta = {
        "mppi_model": base_dynamics.__class__.__name__,
        "sim_model": model_choice,
        "track_choice": track_choice,
        "dt": CONFIG["dt"],
        "mppi": CONFIG["mppi"],
        "V_target": obj.V_target,
        "vel_mode": CONFIG["vel_mode"],
        "obstacle": CONFIG["use_obstacle"],
        "mode": CONFIG["mode"],
    }
    meta_pub.publish(json.dumps(meta))

def kill_rosbag(name):
    rospy.logwarn("Stopping recorder by killing node: %s", name)
    try:
        subprocess.call(['rosnode', 'kill', name])  # sends SIGINT → bag closes cleanly
    except Exception as e:
        rospy.logerr("Failed to kill %s: %s", name, e)

rviz_controls_pub = rospy.Publisher(
            f"/rviz_best_traj",
            MarkerArray,
            queue_size=1
        )

def reset_run():
    global planner, obj, last_throttle, last_steer, track_choice, nx, global_path_message, sim_dynamic, run_counter, dynamics

    rospy.logwarn("[mppi_ros] Max laps reached. Resetting for next run...")

    # 0) send zero command while resetting
    try:
        sim.send_control(torch.tensor([0.0, 0.0], dtype=torch.float32))
    except Exception:
        pass
    # 2) reset objective "time" and lap bookkeeping
    # obj.counter = 0
    # obj.previous_path_index = 0
    # obj.lap_count = 0
    # obj.lap_start_time = time.time()
    #
    # # publish latched lap_count reset so logger re-arms

    #
    # # (optional) reset dynamic obstacle progress
    # obj.obs_s_now = 200.0

    # # 3) reset integrator memory for s-mppi variants
    # if nx == 8:
    #     last_throttle = 0.0
    #     last_steer = 0.0

    # if CONFIG["mode"] == "mppi" and cfg["update_lambda"] == False and cfg["lambda_"] == 0.005:
    #     cfg["lambda_"] = 0.05
    #     rospy.loginfo("MPPI with 0.05 temperature -----------------------------------------")
    #
    # elif CONFIG["mode"] == "mppi" and cfg["update_lambda"] == False and cfg["lambda_"] == 0.05 and cfg["update_cov"] == False:
    #     cfg["lambda_"] = 0.5
    #     rospy.loginfo("MPPI with 0.5 temperature -----------------------------------------")
    #
    # elif CONFIG["mode"] == "mppi" and cfg["update_lambda"] == False and cfg["lambda_"] == 0.5 and cfg["update_cov"] == False:
    #     cfg["lambda_"] = 5
    #     rospy.loginfo("MPPI with 5.0 temperature -----------------------------------------")
    #
    # # elif CONFIG["mode"] == "mppi" and cfg["update_lambda"] == False and cfg["lambda_"] == 0.05 and cfg["update_cov"] == False:
    # #     cfg["update_lambda"] = True
    # #     rospy.loginfo("MPPI with dynamic temperature -----------------------------------------")
    #
    # elif CONFIG["mode"] == "mppi" and cfg["update_lambda"] == True and cfg["update_cov"] == False:
    #     cfg["update_cov"] = True
    #     rospy.loginfo("MPPI with dynamic temperature and dynamic covariance -----------------------------------------")
    #
    # elif CONFIG["mode"] == "mppi" and cfg["update_lambda"] == True and cfg["update_cov"] == True:
    #     cfg["update_lambda"] = False
    #     rospy.loginfo("MPPI with dynamic covariance -----------------------------------------")
    # else:
    #     sim_dynamic = True
    #     cfg["update_cov"] = False
    #     cfg["lambda_"] = 0.005
    #     rospy.loginfo("Starting dynamic simulation loop -----------------------------------------")

    # if run_counter == 4:
    #
    #     run_counter = 0
    #
    #     if CONFIG["mode"] == "mppi" and cfg["update_cov"] == False and cfg["pre_filter"] == "off":
    #         CONFIG["mode"] = "s_mppi"
    #         rospy.loginfo("S-MPPI run set ready -----------------------------------------")
    #
    #         dynamics = RateAugmentedDynamics(
    #             base_dyn=base_dynamics,
    #             dt=CONFIG["dt"],
    #             th_min=th_min,
    #             th_max=th_max,
    #             steer_min=steer_min,
    #             steer_max=steer_max,
    #             device=CONFIG["device"],
    #         )
    #         nx = 8
    #
    #         last_throttle = 0.0
    #         last_steer = 0.0
    #
    #         cfg["u_min"] = [-0.5, -3.0]
    #         cfg["u_max"] = [0.5, 3.0]
    #         cfg["noise_sigma"] = [[0.2, 0.0], [0., 1.0]]
    #
    #     elif CONFIG["mode"] == "s_mppi":
    #         CONFIG["mode"] = "s_mppi_dyn_lim"
    #
    #         dynamics = DynLimRateAugmentedDynamics(
    #             base_dyn=base_dynamics,
    #             dt=CONFIG["dt"],
    #             th_min=th_min,
    #             th_max=th_max,
    #             steer_min=steer_min,
    #             steer_max=steer_max,
    #             device=CONFIG["device"],
    #         )
    #
    #         nx = 8
    #
    #         last_throttle = 0.0
    #         last_steer = 0.0
    #
    #         cfg["u_min"] = [-0.5, -3.0]
    #         cfg["u_max"] = [0.5, 3.0]
    #         cfg["noise_sigma"] = [[0.2, 0.0], [0., 1.0]]
    #
    #         rospy.loginfo("S-MPPI with dynamic limits run set ready -----------------------------------------")
    #
    #     elif CONFIG["mode"] == "s_mppi_dyn_lim":
    #         CONFIG["mode"] = "mppi"
    #         cfg["update_cov"] = True
    #
    #         cfg["u_min"] = [0.0, -1.0]
    #         cfg["u_max"] = [1.0, 1.0]
    #         cfg["noise_sigma"] = [[0.3, 0.0], [0., 0.5]]
    #
    #         rospy.loginfo("MPPI with dynamic variance run set ready -----------------------------------------")
    #
    #     elif cfg["update_cov"] == True:
    #         cfg["update_cov"] = False
    #         cfg["pre_filter"] = "lowpass"
    #         cfg["noise_sigma"] = [[0.1, 0.0], [0., 0.2]]
    #
    #         rospy.loginfo("MPPI with lowpass filtering run set ready -----------------------------------------")
    #
    #     elif cfg["pre_filter"] == "lowpass" and CONFIG["track"] == "simple_smooth" and CONFIG["v_target"] == 1.5:
    #         CONFIG["v_target"] = 2.5
    #         cfg["pre_filter"] = "off"
    #
    #     elif cfg["pre_filter"] == "lowpass" and CONFIG["track"] == "simple_smooth" and CONFIG["v_target"] == 2.5:
    #         track_choice = "racetrack_vicon"
    #         obj.s_vals_global_path, \
    #             obj.x_vals_global_path, \
    #             obj.y_vals_global_path, \
    #             obj.s_4_local_path, \
    #             obj.x_4_local_path, \
    #             obj.y_4_local_path, \
    #             obj.dx_ds, obj.dy_ds, obj.d2x_ds2, obj.d2y_ds2, \
    #             obj.k_vals_global_path, \
    #             obj.k_4_local_path = generate_path_data(track_choice)
    #
    #         rospy.loginfo("Racetrack started with low velocity -----------------------------------------")
    #         CONFIG["track"] = track_choice
    #         CONFIG["v_target"] = 1.5
    #         cfg["pre_filter"] = "off"
    #
    #     elif cfg["pre_filter"] == "lowpass" and CONFIG["track"] == "racetrack_vicon" and CONFIG["v_target"] == 1.5:
    #         rospy.loginfo("Racetrack started with high velocity -----------------------------------------")
    #         CONFIG["v_target"] = 2.5
    #         cfg["pre_filter"] = "off"
    #
    #     elif cfg["pre_filter"] == "lowpass" and CONFIG["track"] == "racetrack_vicon" and CONFIG["v_target"] == 2.5:
    #         track_choice = "simple_smooth"
    #
    #         rospy.loginfo("Dynamic simulation model started -----------------------------------------")
    #         CONFIG["track"] = track_choice
    #         CONFIG["v_target"] = 1.5
    #         cfg["pre_filter"] = "off"
    #         sim_dynamic = True
    #
    #     elif cfg["pre_filter"] == "lowpass" and CONFIG["track"] == "racetrack_vicon" and CONFIG["v_target"] == 2.5 and sim_dynamic == True:
    #
    #         os._exit(0)
    #         # track_choice = "simple_smooth"
    #         #
    #         # CONFIG["track"] = track_choice
    #         # CONFIG["v_target"] = 1.5
    #         # cfg["pre_filter"] = "off"
    #         # sim_dynamic = False
    # else:
    #     run_counter += 1
    #
    # if run_counter == 1:
    #
    #     run_counter = 0
    #
    #     if CONFIG["track"] == "simple_smooth" and CONFIG["v_target"] == 1.5:
    #         rospy.loginfo("High velocity simple track started -----------------------------------------")
    #         CONFIG["v_target"] = 2.5
    #
    #     elif CONFIG["track"] == "simple_smooth" and CONFIG["v_target"] == 2.5:
    #         rospy.loginfo("Low velocity race track started -----------------------------------------")
    #         track_choice = "racetrack_vicon"
    #         CONFIG["track"] = track_choice
    #         CONFIG["v_target"] = 1.5
    #
    #     elif CONFIG["track"] == "racetrack_vicon" and CONFIG["v_target"] == 1.5:
    #         rospy.loginfo("High velocity race track started -----------------------------------------")
    #         track_choice = "racetrack_vicon"
    #
    #         CONFIG["v_target"] = 2.5
    #
    #     elif CONFIG["track"] == "racetrack_vicon" and CONFIG["v_target"] == 2.5 and sim_dynamic == False:
    #         rospy.loginfo("Dynamic simulation model started -----------------------------------------")
    #         track_choice = "simple_smooth"
    #         CONFIG["track"] = track_choice
    #         CONFIG["v_target"] = 1.5
    #         sim_dynamic = True
    #
    #     else:
    #         rospy.loginfo("Ending loop -----------------------------------------")
    #         os.exit(0)
    #
    # else:
    #     run_counter += 1

    # if run_counter == 0:
    #     cfg["noise_sigma"] = [[0.1, 0.0], [0., 0.5]]
    #
    # if run_counter == 1:
    #     cfg["noise_sigma"] = [[0.2, 0.0], [0., 0.5]]
    #
    # if run_counter == 2:
    #     cfg["noise_sigma"] = [[0.3, 0.0], [0., 0.5]]
    #
    # if run_counter == 3:
    #     cfg["noise_sigma"] = [[0.2, 0.0], [0., 0.3]]
    #
    # if run_counter == 4:
    #     cfg["noise_sigma"] = [[0.1, 0.0], [0., 0.5]]
    #
    # if run_counter == 5:
    #     if cfg["lambda_"] == 0.05:
    #         cfg["lambda_"] = 0.1
    #     elif cfg["lambda_"] == 0.1:
    #         cfg["lambda_"] = 0.25
    #     elif cfg["lambda_"] == 0.25:
    #         cfg["lambda_"] = 0.5
    #     elif cfg["lambda_"] == 0.5:
    #         cfg["lambda_"] = 1.0
    #     elif cfg["lambda_"] == 1.0:
    #         cfg["lambda_"] = 2.0
    #     elif cfg["lambda_"] == 2.0:
    #         cfg["lambda_"] = 5.0
    #     elif cfg["lambda_"] == 5.0:
    #         cfg["lambda_"] = 10.0
    #     elif cfg["lambda_"] == 10.0:
    #         cfg["lambda_"] = 20.0
    #     elif cfg["lambda_"] == 20.0:
    #         cfg["lambda_"] = 50.0
    #
    #     cfg["noise_sigma"] = [[0.1, 0.0], [0., 0.3]]
    #     run_counter = -1

    # run_counter += 1

    if run_counter == 4:

        run_counter = 0
        
        if CONFIG["track"] == "simple_smooth":
            track_choice = "racetrack_vicon"
            obj.s_vals_global_path, \
                obj.x_vals_global_path, \
                obj.y_vals_global_path, \
                obj.s_4_local_path, \
                obj.x_4_local_path, \
                obj.y_4_local_path, \
                obj.dx_ds, obj.dy_ds, obj.d2x_ds2, obj.d2y_ds2, \
                obj.k_vals_global_path, \
                obj.k_4_local_path = generate_path_data(track_choice)


            CONFIG["track"] = track_choice
            CONFIG["v_target"] = 2.5
            CONFIG["mode"] = "s_mppi"
            dynamics = RateAugmentedDynamics(
                base_dyn=base_dynamics,
                dt=CONFIG["dt"],
                th_min=th_min,
                th_max=th_max,
                steer_min=steer_min,
                steer_max=steer_max,
                device=CONFIG["device"],
            )

            last_throttle = 0.0
            last_steer = 0.0

            rospy.loginfo("Racetrack started with high velocity -----------------------------------------")

        elif CONFIG["mode"] == "s_mppi_dyn_lim" and CONFIG["track"] == "racetrack_vicon" and CONFIG["v_target"] == 1.5:
            rospy.loginfo("Racetrack started with high velocity -----------------------------------------")
            CONFIG["v_target"] = 2.5
            CONFIG["mode"] = "s_mppi"
            dynamics = RateAugmentedDynamics(
                base_dyn=base_dynamics,
                dt=CONFIG["dt"],
                th_min=th_min,
                th_max=th_max,
                steer_min=steer_min,
                steer_max=steer_max,
                device=CONFIG["device"],
            )

            last_throttle = 0.0
            last_steer = 0.0

        elif CONFIG["mode"] == "s_mppi_dyn_lim" and CONFIG["track"] == "racetrack_vicon" and CONFIG["v_target"] == 2.5:
            track_choice = "simple_smooth"

            rospy.loginfo("Dynamic simulation model started -----------------------------------------")
            CONFIG["mode"] = "s_mppi"
            dynamics = RateAugmentedDynamics(
                base_dyn=base_dynamics,
                dt=CONFIG["dt"],
                th_min=th_min,
                th_max=th_max,
                steer_min=steer_min,
                steer_max=steer_max,
                device=CONFIG["device"],
            )

            last_throttle = 0.0
            last_steer = 0.0

        elif cfg["pre_filter"] == "lowpass" and CONFIG["track"] == "racetrack_vicon" and CONFIG["v_target"] == 2.5 and sim_dynamic == True:

            os._exit(0)

    else:
        run_counter += 1

    if CONFIG["env"] == "sim" and sim_dynamic:
        if CONFIG["track"] == "racetrack_vicon_2":
            reset_sim(x=0.0, y=-2.5, theta=0.0, model_choice=2)
        else:
            reset_sim(x=-1.5, y=-2.5, theta=0.0, model_choice=2)
    elif CONFIG["env"] == "sim":
        if CONFIG["track"] == "racetrack_vicon_2":
            reset_sim(x=0.0, y=-2.5, theta=0.0, model_choice=1)
        else:
            reset_sim(x=-1.5, y=-2.5, theta=0.0, model_choice=1)

    if CONFIG["mode"] == "FM_mppi":
        ref_gen = True
    else:
        ref_gen = False

    obj = ROSObjective(track_choice, CONFIG["mppi"]["horizon"], CONFIG["dt"], CONFIG["v_target"],
                       CONFIG["use_obstacle"], CONFIG["troubleshoot"], ref_gen, device=CONFIG["device"])

    # 4) hard reset MPPI internal memory by recreating the planner (simplest + robust)
    planner = MPPIPlanner(
        cfg=cfg,
        nx=nx,
        dynamics=dynamics.step,
        running_cost=obj.compute_running_cost,
    )
    planner.terminal_state_cost = obj.terminal_costs

    global_path_message = sim.generate_track(obj.x_vals_global_path, obj.y_vals_global_path, obj.s_vals_global_path)

    # 5) re-publish meta (helps logger distinguish runs if you encode run-id later)
    publish_meta()
    obj.lap_pub.publish(Int32(0))
    # small settle time to allow sim / topics to update
    rospy.sleep(2.0)


def publish_path(U: np.array, state):
    """
    Visualize the full MPPI-mean control sequence by rolling out
    through the dynamics.step dynamics.

    U: shape (T,2) array of [throttle, steering].
    Requires self.dynamics(state, action, t) → (next_state, _).
    """
    # 1) Setup MarkerArray
    ma = MarkerArray()
    now = rospy.Time.now()
    clear = Marker()
    clear.action = Marker.DELETEALL
    ma.markers.append(clear)

    T = U.shape[0]

    pts = []
    p = Point(x=state[0, 0].item(), y=state[0, 1].item(), z=0.0)
    pts.append(p)
    states = []
    for t in range(T):
        state, _ = sim.vizdynamics.step(state, torch.tensor([[U[t, 0], U[t, 1]]], dtype=torch.float32), t)
        x = state[0, 0].item()
        y = state[0, 1].item()
        p = Point(x=x, y=y, z=0.0)
        pts.append(p)
        states.append(state)


    m = Marker()
    m.header.frame_id = "map"
    m.header.stamp = now
    m.ns = f"trajectory"
    m.id = 0
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD

    # visual style
    m.scale.x = 0.02  # line width
    m.color.r = 1.0
    m.color.g = 0.0
    m.color.b = 0.0
    m.color.a = 1.0  # fully opaque

    m.points = pts
    ma.markers.append(m)

    # 3) publish
    rviz_controls_pub.publish(ma)

    return states

if __name__ == "__main__":
    dist = StateDisturber()

    meta_pub = rospy.Publisher("mppi_meta", String, queue_size=1, latch=True)
    rospy.init_node("mppi_ros_node")
    car_number = rospy.get_param("~car_number", 1)

    comptime_publisher = rospy.Publisher('comptime_' + str(car_number), Float32, queue_size=1)

    action_publisher = rospy.Publisher('mppi_action', Float32MultiArray, queue_size=1)
    # action_publisher = rospy.Publisher('mppi_action_sim', Float32MultiArray, queue_size=1)

    mppi_roll_pub = rospy.Publisher("mppi_rollouts", String, queue_size=10)
    cum_expected_cost = 0.0

    dyn_temp_pub = rospy.Publisher("dyn_temp", Float32MultiArray, queue_size=1)

    dyn_cov_pub = rospy.Publisher("dyn_cov", Float32MultiArray, queue_size=1)

    # 1) Load your existing YAML config
    CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))
    cfg = CONFIG["mppi"]

    if CONFIG["sim_model"] == "kinematic":
        model_choice = 1
        sim_dynamic = False
    elif CONFIG["sim_model"] == "dynamic":
        model_choice = 2
        sim_dynamic = True
    elif CONFIG["sim_model"] == "SVGP":
        model_choice = 3
        sim_dynamic = False
    elif CONFIG["sim_model"] == "SVGP_wet":
        model_choice = 4

    # 2) Create ROS “simulator”
    sim = SimulatorROS(car_number, CONFIG["env"])

    if CONFIG["mppi_model"] == "kinematic":
        base_dynamics = Kinematic_Bicycle(dt=CONFIG["dt"], device=CONFIG["device"])
    elif CONFIG["mppi_model"] == "dynamic":
        base_dynamics = Dynamic_Bicycle(dt=CONFIG["dt"], device=CONFIG["device"])
    elif CONFIG["mppi_model"] == "SVGP":
        base_dynamics = SVGP(dt=CONFIG["dt"], device=CONFIG["device"])
    else:
        raise ValueError("Unknown mppi_model in config")

    th_min = CONFIG["throttle_min"]
    th_max = CONFIG["throttle_max"]
    steer_min = CONFIG["steering_min"]
    steer_max = CONFIG["steering_max"]

    if CONFIG["mode"] == "s_mppi":
        dynamics = RateAugmentedDynamics(
            base_dyn=base_dynamics,
            dt=CONFIG["dt"],
            th_min=th_min,
            th_max=th_max,
            steer_min=steer_min,
            steer_max=steer_max,
            device=CONFIG["device"],
        )
        nx = 8

        last_throttle = 0.0
        last_steer = 0.0

    elif CONFIG["mode"] == "s_mppi_dyn_lim":
        dynamics = DynLimRateAugmentedDynamics(
            base_dyn=base_dynamics,
            dt=CONFIG["dt"],
            th_min=th_min,
            th_max=th_max,
            steer_min=steer_min,
            steer_max=steer_max,
            device=CONFIG["device"],
        )
        nx = 8

        last_throttle = 0.0
        last_steer = 0.0
    else:
        dynamics = base_dynamics
        nx = 6

    track_choice = CONFIG["track"]

    sim.dynamics = dynamics
    sim.vel_mode = CONFIG["vel_mode"]

    if CONFIG["mode"] == "FM_mppi":
        ref_gen = True
    else:
        ref_gen = False

    run_counter = 0
    # 3) Objective
    obj = ROSObjective(track_choice, CONFIG["mppi"]["horizon"], CONFIG["dt"], CONFIG["v_target"], CONFIG["use_obstacle"], CONFIG["troubleshoot"], ref_gen, device=CONFIG["device"])

    global_path_message = sim.generate_track(obj.x_vals_global_path, obj.y_vals_global_path, obj.s_vals_global_path)

    # 4) Planner
    planner = MPPIPlanner(
        cfg=cfg,
        nx=nx,
        dynamics=dynamics.step,
        running_cost=obj.compute_running_cost,
    )

    planner.terminal_state_cost = obj.terminal_costs
    step_idx = 0

    # rate = rospy.Rate(1/CONFIG["dt"])

    rate = rospy.Rate(1/0.5)

    counter = 0
    global_path_message_rate = 5  # publish 1 every 5 control loops

    # Reset the simulator
    if CONFIG["env"] == "sim":
        dr_client = Client("/dart_simulator_node", timeout=2.0)
        if CONFIG["track"] == "racetrack_vicon_2":
            reset_sim(x=0.0, y=-2.5, theta=0.0, model_choice=model_choice)
        else:
            reset_sim(x=-1.5, y=-2.5, theta=0.0, model_choice=model_choice)


    publish_meta()

    if CONFIG["use_obstacle"] == "static":
        obj.obstacle_Q, obj.obstacle_c = sim.get_obstacle()


    dt = CONFIG["dt"]
    first = 1

    if CONFIG["troubleshoot"] == 1:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = f"/home/maarten/Documents/Thesis/log_Dart/troubleshoot/1plots_{timestamp}"
        mppi_plotter = OnlineMppiPlotter(
            max_history=200,
            save_dir=plot_dir,
            save_every=1,  # or e.g. 5 if you want every 5th step
            file_prefix=f"car{car_number}"
        )
        rospy.loginfo(f"[MPPI online plot] saving plots to {plot_dir}")

    # --- logging rollouts/weights for offline Neff plots ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rollout_log_dir = f"/home/maarten/Documents/Thesis/log_Dart/mppi_rollouts_{timestamp}"

    run_id = 0
    rollout_logger = RunRolloutLogger(
        root_dir=rollout_log_dir,
        car_number=car_number,
        max_steps=10000,
        downsample_K=800  # set None if you truly want all K; 500-1500 is usually enough
    )

    state_est_pub = rospy.Publisher(
        f"state_est_{car_number}",
        Float32MultiArray,
        queue_size=10
    )

    def current_meta_dict():
        return {
            "mppi_model": base_dynamics.__class__.__name__,
            "sim_model": model_choice,
            "track_choice": track_choice,
            "dt": CONFIG["dt"],
            "mppi": CONFIG["mppi"],
            "V_target": obj.V_target,
            "vel_mode": CONFIG["vel_mode"],
            "obstacle": CONFIG["use_obstacle"],
            "mode": CONFIG["mode"],
        }


    # initialize meta for run 0
    rollout_logger.reset(run_id=run_id, meta=current_meta_dict())

    ctrl_step = 0  # define before while loop

    if CONFIG["track"] == "straight_line":
        # obj.lap_count = 1
        obj.lap_pub.publish(Int32(1))


    while not rospy.is_shutdown():
        # making sure that while waiting the actions are zero
        # sim.send_control(torch.tensor([0.0, 0.0]))
        # if CONFIG["troubleshoot"] == 1:
        #if first == 0:
        #   input("Press Enter to run the next MPPI iteration or ctrl-c to quit")
        #   # first = 10
        #else:
        #   first = first - 1
        # rospy.loginfo("Starting next MPPI iteration...")

        state = sim.get_current_state(dynamics._device)

        if CONFIG["disturbance"] == 1:
            state_est = dist.disturb(state)
        else:
            state_est = state

        s_est = state_est[0].detach().cpu().numpy()

        msg = Float32MultiArray()
        msg.data = s_est.astype(np.float32).tolist()
        state_est_pub.publish(msg)

        obj.current_state = state_est  # update the current state in the objective, which updates the goal
        obj.counter = 0

        # Sending the current position and expected positions of the other agent to the objective
        if CONFIG["use_obstacle"] == "dynamic":
            obj.obs_s_now = (obj.obs_s_now + obj.obs_v * dt) % obj.s_vals_global_path[-1]
        elif CONFIG["use_obstacle"] == "agent":
            obj.obs_c_seq = sim.agent_center
            obj.obs_yaw_seq = sim.agent_yaw



        if state is None:
            rospy.loginfo(f"Waiting for first /vicon/jetracer{car_number} message…")
            rate.sleep()
            continue


        start_time = time.time()
        if nx == 8:
            th_tensor = torch.tensor([[last_throttle]], dtype=state_est.dtype, device=state_est.device)
            st_tensor = torch.tensor([[last_steer]], dtype=state_est.dtype, device=state_est.device)
            state_control = torch.cat([state_est, th_tensor, st_tensor], dim=1)

            with torch.inference_mode():
                action = planner.command(state_control)

            dth = action[0,0].item()
            dst = action[0,1].item()

            last_throttle = np.clip(last_throttle + dth * dt, th_min, th_max)
            last_steer = np.clip(last_steer + dst * dt, steer_min, steer_max)

            throttle = last_throttle
            steer = last_steer

        else:
            with torch.inference_mode():
                action = planner.command(state_est)

                throttle = np.clip(action[0,0].item(), th_min, th_max)
                steer = np.clip(action[0,1].item(),steer_min, steer_max)


        steering_angle = mf.steering_2_steering_angle(
            steer,
            mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        )

        delta_max = mf.steering_2_steering_angle_actual(1.0, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self,
                                                        mf.e_s_self)
        delta_min = mf.steering_2_steering_angle_actual(-1.0, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self,
                                                        mf.e_s_self)

        steering_angle = np.clip(steering_angle, delta_min, delta_max)

        transformed_steer = mf.steering_angle_2_steering_command(
            steering_angle,
            mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self,
            steer_min, steer_max
        )

        sim.send_control(torch.tensor([throttle,transformed_steer], dtype=torch.float32))
        #sim.send_control(torch.tensor([throttle, steer], dtype=torch.float32))
        
        # sim.send_control(torch.tensor([throttle, 0], dtype=torch.float32))

        # If dynamic temperature is one publish the temperature and eta
        if CONFIG["mppi"]["update_lambda"]:
            temp_msg = Float32MultiArray()
            temp_msg.data = [planner.beta, planner.eta.item()]
            dyn_temp_pub.publish(temp_msg)
            print(f"temperature: {planner.beta}")

        cov_msg = Float32MultiArray()
        cov_msg.data = planner.cov_action.numpy()
        dyn_cov_pub.publish(cov_msg)

        action_msg = Float32MultiArray()
        action_msg.data = action.reshape(-1).tolist() # list(action.to(torch.float32))
        action_publisher.publish(action_msg)

        # print(f"action: {action}")


        # troubleshoot_states = publish_path(action.detach(), sim)
        # if CONFIG["troubleshoot"] == 1:
        #     writer.writerow(["current state (estimate)", state_est, "planned throttle", action[:,0], "planned steering", action[:,1], "planned/expected states", troubleshoot_states])
        #     # writer.writerow([])
        #     # writer.writerow(planner.actions.detach())
        #
        #     csvfile.flush()

        elapsed_time = time.time() - start_time
        if elapsed_time > CONFIG["dt"]:

            rospy.loginfo(f"MPPI computation time: {elapsed_time:.4f} seconds")

        # best_u = planner.best_traj.detach().cpu().numpy()
        # publish_path(best_u, state_est)

        # this is just to republish global path message every now and then
        if counter > global_path_message_rate:
            sim.rviz_global_path_publisher.publish(global_path_message)
            counter = 0  # reset counter

        # update counter
        counter = counter + 1

        elapsed_time = time.time() - start_time
        comptime_publisher.publish(elapsed_time)
        # print(f"Elapsed MPPI computation time: {elapsed_time}")

        # plt.figure()
        # for k in range(min(200, planner.perturbed_action.shape[0])):  # limit to avoid clutter
        #     plt.plot(planner.perturbed_action[k, :, 0].cpu(), color='gray', alpha=0.25)
        # plt.plot(planner.mean_action[:, 0].cpu(), 'r', linewidth=2, label='Mean action')
        # plt.plot(planner.best_traj[:, 0].cpu(), 'g--', linewidth=2, label='Best trajectory')
        # plt.xlabel('Timestep')
        # plt.ylabel('Throttle Rate')
        # plt.title('Sampled Throttle Trajectories')
        # plt.legend()
        # plt.show()
        #
        # plt.figure()
        # for k in range(min(200, planner.perturbed_action.shape[0])):  # limit to avoid clutter
        #     plt.plot(planner.perturbed_action[k, :, 1].cpu(), color='gray', alpha=0.25)
        # plt.plot(planner.mean_action[:, 1].cpu(), 'r', linewidth=2, label='Mean action')
        # plt.plot(planner.best_traj[:, 1].cpu(), 'g--', linewidth=2, label='Best trajectory')
        # plt.xlabel('Timestep')
        # plt.ylabel('Steering Rate')
        # plt.title('Sampled Steering Trajectories')
        # plt.legend()
        # plt.show()


        if CONFIG["troubleshoot"] == 1: # and first < 10:
            # --- MPPI online diagnostics / plotting ---
            step_idx += 1

            try:
                # 1) executed first command u0 (rate space)
                #    action_rate might be shape (T,2) or (1,2) depending on u_per_command
                if action.ndim == 2:
                    # u0 = action[0].detach().cpu().numpy()    # first time step
                    
                    u0 = np.array([throttle, steer])
                else:
                    # u0 = action.detach().cpu().numpy()       # already (2,)
                    
                    u0 = np.array([throttle, steer])

                # 2) sample cloud for first step
                u0_samples = None
                weights_np = None
                Neff = 0.0
                cost_min = 0.0

                # Only proceed if MPPI has done a rollout
                if hasattr(planner, "perturbed_action") and planner.perturbed_action is not None:
                    # K x T x nu
                    pa = planner.perturbed_action
                    if pa.ndim == 3 and pa.shape[1] > 0:
                        u0_samples = pa[:, 0, :].detach().cpu().numpy()   # K x nu

                        # cost_min from cost_total (exists in both modes)
                        if hasattr(planner, "cost_total") and planner.cost_total is not None:
                            cost_min = float(torch.min(planner.cost_total).item())

                        # 3) weights
                        if planner.mppi_mode == "simple":
                            # weights already normalised in planner.omega
                            if hasattr(planner, "omega") and planner.omega is not None:
                                w = planner.omega.detach()
                                w = w / torch.sum(w)
                            else:
                                w = None
                        else:
                            # halton-spline: reconstruct from total_costs and beta
                            if hasattr(planner, "total_costs") and planner.total_costs is not None:
                                total_costs = planner.total_costs.detach()
                                beta = planner.beta
                                w = torch.exp((-1.0 / beta) * total_costs)
                                w = w / torch.sum(w)
                            else:
                                w = None

                        if w is not None:
                            weights_np = w.cpu().numpy()
                            # Optional subsampling for huge K
                            K = u0_samples.shape[0]
                            # if K > 2000:
                            #     N_top = 100
                            #     M_rand = 300
                            #
                            #     # For halton we have total_costs; for simple we only have cost_total
                            #     if hasattr(planner, "total_costs") and planner.total_costs is not None:
                            #         costs_for_sort = planner.total_costs.detach()
                            #     else:
                            #         costs_for_sort = planner.cost_total.detach()
                            #
                            #     _, top_idx = torch.topk(-costs_for_sort, N_top)  # lowest cost
                            #     top_idx = top_idx.cpu().numpy()
                            #
                            #     all_idx = np.arange(K)
                            #     mask = np.ones(K, dtype=bool)
                            #     mask[top_idx] = False
                            #     rest_idx = all_idx[mask]
                            #     if rest_idx.size > 0:
                            #         M_rand = min(M_rand, rest_idx.size)
                            #         rand_idx = np.random.choice(rest_idx, size=M_rand, replace=False)
                            #         sel_idx = np.concatenate([top_idx, rand_idx])
                            #     else:
                            #         sel_idx = top_idx
                            #
                            #     u0_samples = u0_samples[sel_idx]
                            #     weights_np = weights_np[sel_idx]

                            # Effective sample size
                            Neff = float(1.0 / np.sum(weights_np**2))
                            # print(Neff)

                mean_u = action  # (T,2)
                best_u = planner.best_traj.detach().cpu().numpy()  # (T,2)

                # 4) update plot if we have something
                # if mppi_plotter is not None and u0_samples is not None and weights_np is not None:
                    # mppi_plotter.update(
                    #    u0=u0,
                    #    Neff=Neff,
                    #    cost_min=cost_min,
                    #    u0_samples=u0_samples,
                    #    weights=weights_np,
                    #    t=step_idx,
                    #    mean_u=mean_u,
                    #    best_u=best_u,
                    #    filt_u=planner.best_filtered,
                    #    sample_costs=planner.cost_total.detach().cpu().numpy()
                    #)

            except Exception as e:
                rospy.logwarn(f"[MPPI online plot] error: {e}")

            # if step_idx % 20 == 0:
            #      input("Paused — press Enter...")
            
            # AFTER you computed u0_samples and weights_np (and cost_min, mean_u, best_u)
            if (ctrl_step > 50) and (u0_samples is not None) and (weights_np is not None):
                # sample costs: pick what you actually have available
                sample_costs_np = None
                if hasattr(planner, "cost_total") and planner.cost_total is not None:
                    sample_costs_np = planner.cost_total.detach().cpu().numpy()
                elif hasattr(planner, "total_costs") and planner.total_costs is not None:
                    sample_costs_np = planner.total_costs.detach().cpu().numpy()
                        #     # mean_u: action is torch; store the full mean plan if you want
                mean_u_np = None
                try:
                    mean_u_np = mean_u.detach().cpu().numpy() if torch.is_tensor(mean_u) else np.asarray(mean_u)
                except Exception:
                    mean_u_np = None
            
                rollout_logger.log_step(
                    step_idx=ctrl_step,
                    u0=np.array([throttle, steer], dtype=np.float32),
                    u0_samples=u0_samples,  # Kx2
                    weights=weights_np,  # K
                    costs=sample_costs_np,  # K or None
                    cost_min=cost_min,
                    mean_u=mean_u_np,  # Tx2 or None
                    best_u=best_u  # already numpy Tx2 in your code
                )
            
                if ctrl_step == 200:
                    try:
                        out = rollout_logger.flush()
                        if out is not None:
                            rospy.loginfo(f"[mppi_ros] Saved rollout log: {out}")
                    except Exception as e:
                        rospy.logwarn(f"[mppi_ros] Failed to flush rollout log: {e}")


        ctrl_step += 1



        if CONFIG["viz_rollouts"] == 1:
            # get the candidate trajectories NOT SHOWING THESE AS IT TAKES A LOT OF COMPUTATIONAL TIME
            rollouts = planner.states.detach()
            # publish them
            sim.publish_rollouts(rollouts)
        #

        if obj.lap_count > CONFIG["laps"]:

            reset_run()
            continue
                
        rate.sleep()


