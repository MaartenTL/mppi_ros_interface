#!/usr/bin/env python3
from typing import Optional

import rospy
import threading
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from std_msgs.msg import Float32                # adjust if your control topics use a different type
from tf.transformations import euler_from_quaternion
from functions_for_controllers import find_s_of_closest_point_on_global_path, produce_track,produce_marker_array_rviz, produce_marker_rviz, steer_angle_2_command
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
import tf
import time
from collections import deque
import torch
from dart_dynamic_models import model_functions,load_SVGPModel_actuator_dynamics_analytic
import numpy as np
import sys
import importlib.resources
from dynamics import Kinematic_Bicycle, Dynamic_Bicycle, SVGP, RateAugmentedDynamics
import matplotlib.pyplot as plt


mf = model_functions()

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "axes.grid": True,

    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.titlesize": 20,
    "legend.fontsize": 14,
})



import numpy as np
import matplotlib.pyplot as plt
import os

# If OnlineMppiPlotter is in mppi_online_plot.py:
from mppi_online_plot import OnlineMppiPlotter

def compute_neff(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    s = w.sum()
    if s <= 0:
        return float("nan")
    w = w / s
    return float(1.0 / np.sum(w**2))

def replay_npz(npz_path: str, out_dir: Optional[str] = None, every: int = 1, max_steps: Optional[int] = None):
    data = np.load(npz_path, allow_pickle=True)

    # What you saved in RunRolloutLogger.flush()
    steps      = data["steps"]               # int array [S]
    u0         = data["u0"]                  # float array [S,2]
    u0_samples = data["u0_samples"]          # object array length S, each (K,2)
    weights    = data["weights"]             # object array length S, each (K,)
    costs      = data["costs"]               # object array length S, each (K,) or empty
    cost_min   = data["cost_min"]            # float array [S]
    mean_u     = data["mean_u"]              # object array length S, each (T,2) or empty
    best_u     = data["best_u"]              # object array length S, each (T,2) or empty

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    # Configure plotter like your online version
    plotter = OnlineMppiPlotter(
        max_history=200,
        save_dir=out_dir,           # if not None, it will save pngs
        save_every=1,               # we’ll control saving by calling update every N
        file_prefix=os.path.splitext(os.path.basename(npz_path))[0],
    )

    # If you want interactive viewing:
    plt.ion()

    S = len(steps)
    if max_steps is not None:
        S = min(S, int(max_steps))

    for i in range(0, S, every):
        w_i = np.asarray(weights[i], dtype=np.float64) if weights[i].size else None
        u0s_i = np.asarray(u0_samples[i], dtype=np.float64) if u0_samples[i].size else None

        # input("Press Enter to run the next MPPI iteration or ctrl-c to quit")

        # costs[i] may be empty array if you didn’t have costs that step
        costs_i = None
        if costs[i] is not None and np.asarray(costs[i]).size > 0:
            costs_i = np.asarray(costs[i], dtype=np.float64)

        Neff_i = compute_neff(w_i) if w_i is not None else float("nan")

        # mean_u / best_u are saved as object entries; handle empties
        mean_u_i = None
        if mean_u[i] is not None and np.asarray(mean_u[i]).size > 0:
            mean_u_i = np.asarray(mean_u[i], dtype=np.float64)

        best_u_i = None
        if best_u[i] is not None and np.asarray(best_u[i]).size > 0:
            best_u_i = np.asarray(best_u[i], dtype=np.float64)

        # Call the SAME update() as online
        plotter.update(
            u0=u0[i],                        # executed [th, st]
            Neff=Neff_i,
            cost_min=float(cost_min[i]),
            u0_samples=u0s_i,
            weights=w_i,
            t=int(steps[i]),
            mean_u=mean_u_i,
            best_u=best_u_i,
            filt_u=None,
            sample_costs=costs_i,            # needed for cost-vs-throttle plots
        )

    # Keep window open if interactive
    plt.ioff()
    plt.show()

#
# if __name__ == "__main__":
#     npz_path = "/home/maarten/Documents/Thesis/log_Dart/mppi_rollouts_h15/car1/run_000.npz"
#     out_dir  = "/home/maarten/Documents/Thesis/log_Dart/mppi_rollouts_h15"
#
#     # every=1 replays every step; every=5 is much faster
#     replay_npz(npz_path, out_dir=out_dir, every=5, max_steps=2000)

#
# if __name__ == "__main__":
#     model = Kinematic_Bicycle(0.1, "cpu")
#
#     xs = []
#     ys = []
#
#     vx_test = torch.tensor([
#         -1e-2, -1e-3, -1e-4, 0.0, 1e-4, 1e-3, 1e-2
#     ])
#
#     # fixed throttle values
#     thr_pos = torch.full_like(vx_test, 0.1)
#     thr_neg = torch.full_like(vx_test, -0.1)
#     thr_abs = torch.full_like(vx_test, 0.1)
#
#     F_pos = mf.motor_force(thr_pos, vx_test, mf.a_m_self, mf.b_m_self, mf.c_m_self)
#     F_neg = mf.motor_force(thr_neg, vx_test, mf.a_m_self, mf.b_m_self, mf.c_m_self)
#     F_abs = mf.motor_force(thr_abs, vx_test, mf.a_m_self, mf.b_m_self, mf.c_m_self)
#
#     print("vx       :", vx_test)
#     print("F(thr=+):", F_pos)
#     print("F(thr=-):", F_neg)
#     print("F(|thr|):", F_abs)
#
#
#
#
#     for i in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0]:
#         states, actions = model.step(
#             torch.tensor([[0.0, 0.0, np.deg2rad(90), 2.0, 0.0, 0.0]]),
#             torch.tensor([[0.5, i]]),
#             0
#         )
#
#         x, y, yaw, vx, vy, w = states.unbind(dim=1)
#         xs.append(x.item())
#         ys.append(y.item())

        # ys.append(i)

        # desired_steering_angle = mf.steering_2_steering_angle(
        #     i,
        #     mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        # )
        #
        # delta_max = mf.steering_2_steering_angle_actual(1.0, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self,
        #                                                 mf.e_s_self)
        # delta_min = mf.steering_2_steering_angle_actual(-1.0, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self,
        #                                                 mf.e_s_self)
        #
        # desired_steering_angle = np.clip(desired_steering_angle, delta_min, delta_max)
        #
        # transformed_steer = mf.steering_angle_2_steering_command(
        #     desired_steering_angle,
        #     mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self,
        #     -1,1
        # )
        #
        # final_steer_angle = mf.steering_2_steering_angle_actual(
        #     transformed_steer,
        #     mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        # )
        #
        # else_steer_angle = mf.steering_2_steering_angle_actual(
        #     i,
        #     mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        # )
        #
        # print(f"steer: {i},steering angle: {desired_steering_angle:.4f} transformed steering: {transformed_steer:.4f}"
        #       f", angle after transform: {final_steer_angle:.4f}, else: {else_steer_angle:.4f}")

    #
    #
    # ax.plot(u, angle, linewidth=2)
    #
    # ax.set_xlabel("Steering command", fontsize=20)
    # ax.set_ylabel("Steering angle [rad]", fontsize=20)
    #
    # ax.tick_params(axis='both', which='major', labelsize=14)
    #
    # ax.grid(True)
    # plt.tight_layout()
    # plt.axis("equal")
    # plt.show()
    #
    # plt.figure()
    #
    # plt.scatter(xs, np.zeros_like(xs), s=60)
    #
    # plt.xlabel("Lateral displacement x [m]", fontsize=20)
    # plt.yticks([])  # remove y-axis entirely
    # # plt.xlim(-0.015, 0.015)
    #
    # ax = plt.gca()
    # ax.tick_params(axis="x", which="major", labelsize=16)
    # plt.axis("equal")
    # plt.grid(True, axis="x")
    # plt.tight_layout()
    # plt.show()

    #
    # plt.figure()
    # plt.scatter(xs, ys)
    # plt.xlabel("lateral displacement [m]", fontsize=20)
    # plt.ylabel("steering command [-]", fontsize=20)
    # plt.xlim(0, 0.015)
    # plt.ylim(-1.0, 1.0)
    # plt.grid(True)
    # plt.show()


import argparse
import os
import re
import numpy as np
import pandas as pd
import yaml

from dart_dynamic_models import model_functions
from path_track_definitions import generate_path_data

mf = model_functions()

# ---------------------------------------------------------------------
# CONFIG / PATHS
# ---------------------------------------------------------------------
BASE_DIR = "/home/maarten/Documents/Thesis/log_Dart"
abs_path = os.path.dirname(os.path.abspath(__file__))
CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))

# ---------------------------------------------------------------------
# SMALL UTILITIES
# ---------------------------------------------------------------------
def ensure_csv_ext(s: str) -> str:
    return s if s.lower().endswith(".csv") else s + ".csv"

def resolve_path(tok: str) -> str:
    tok = ensure_csv_ext(tok)
    return tok if os.path.isabs(tok) else os.path.join(BASE_DIR, tok)

def clean_label(path_or_tok: str) -> str:
    b = os.path.splitext(os.path.basename(path_or_tok))[0]
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", b)

def build_run_id(labels):
    return "__vs__".join(labels)

def nanrms(x):
    x = np.asarray(x, float)
    return np.sqrt(np.nanmean(x**2))

def pct(x, p):
    x = np.asarray(x, float)
    return np.nanpercentile(x, p)

def area_l1(signal, t):
    s = np.asarray(signal, float)
    tt = np.asarray(t, float)
    m = np.isfinite(s) & np.isfinite(tt)
    if m.sum() < 2:
        return np.nan
    s = s[m]; tt = tt[m]
    dt = np.diff(tt, prepend=tt[0])
    dt[0] = 0.0
    return np.sum(np.abs(s) * dt)

def area_l2(signal, t):
    s = np.asarray(signal, float)
    tt = np.asarray(t, float)
    m = np.isfinite(s) & np.isfinite(tt)
    if m.sum() < 2:
        return np.nan
    s = s[m]; tt = tt[m]
    dt = np.diff(tt, prepend=tt[0]); dt[0] = 0.0
    return np.sum((s**2) * dt)

def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m]; b = b[m]
    if len(a) < 5:
        return np.nan
    a = a - np.mean(a)
    b = b - np.mean(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def total_variation(y):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    y = y[m]
    if len(y) < 2:
        return np.nan
    return float(np.sum(np.abs(np.diff(y))))

def tv_per_second(y, t):
    tv = total_variation(y)
    T = float(t[-1] - t[0]) if len(t) else np.nan
    return tv / T if T and T > 0 else np.nan

def step_stats(y, eps=1e-6):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    y = y[m]
    if len(y) < 2:
        return np.nan, np.nan, np.nan
    dy = np.diff(y)
    n_steps = int(np.sum(np.abs(dy) > eps))
    mean_step = float(np.mean(np.abs(dy[np.abs(dy) > eps]))) if n_steps else 0.0
    max_step = float(np.max(np.abs(dy))) if len(dy) else np.nan
    return n_steps, mean_step, max_step

def _mavg(y, k):
    y = np.asarray(y, float)
    if k <= 1 or k >= len(y):
        return y
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, np.ones(k) / k, mode="valid")

def _diff(y, t):
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    out = np.full_like(y, np.nan, float)
    if len(y) >= 3:
        out[:] = np.gradient(y, t)
    return out

def _count_peaks_simple(t, y, thr, min_sep_s=0.10, smooth_k=5):
    y = _mavg(y, smooth_k)
    n = 0
    last_t = -1e9
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and abs(y[i]) >= thr:
            if t[i] - last_t >= min_sep_s:
                n += 1
                last_t = t[i]
    dur = float(t[-1] - t[0]) if len(t) else np.nan
    return n, (n / dur if dur and dur > 0 else 0.0)

def _burst_metrics(t, y, thr):
    t = np.asarray(t, float)
    y = np.abs(np.asarray(y, float))
    if len(t) < 2:
        return 0, np.nan, np.nan, np.nan
    mask = y > thr
    if not np.any(mask):
        return 0, 0.0, 0.0, 0.0

    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(edges == +1)
    ends = np.flatnonzero(edges == -1) - 1

    T = t[-1] - t[0]
    n_bursts = 0
    time_above = 0.0
    longest = 0.0
    area = 0.0

    for s, e in zip(starts, ends):
        if e <= s:
            continue
        n_bursts += 1
        t0, t1 = t[s], t[e]
        time_above += (t1 - t0)
        longest = max(longest, t1 - t0)

        tt = t[s:e + 1]
        yy = y[s:e + 1] - thr
        dt = np.diff(tt, prepend=tt[0]); dt[0] = 0.0
        area += float(np.sum(np.maximum(yy, 0.0) * dt))

    frac = (time_above / T) if T > 0 else np.nan
    return n_bursts, frac, longest, area

# ---------------------------------------------------------------------
# TRACK ERROR COMPUTATION
# ---------------------------------------------------------------------
def nearest_path_indices(xs, ys, x_path, y_path):
    xy_path = np.vstack([x_path, y_path]).T
    xy = np.vstack([xs, ys]).T
    return np.argmin(((xy[:, None, :] - xy_path[None, :, :])**2).sum(-1), axis=1)

def compute_track_errors(df, track_choice):
    (_, x_path, y_path,
     _, _, _,
     dx_ds, dy_ds, _, _,
     _, _) = generate_path_data(track_choice)

    idx = nearest_path_indices(df["x"].values, df["y"].values, x_path, y_path)
    x_ref = x_path[idx]; y_ref = y_path[idx]
    psi_ref = np.arctan2(dy_ds[idx], dx_ds[idx])

    dx = df["x"].values - x_ref
    dy = df["y"].values - y_ref
    lag_err = dx * np.cos(psi_ref) + dy * np.sin(psi_ref)
    lat_err = -dx * np.sin(psi_ref) + dy * np.cos(psi_ref)
    pos_err = np.sqrt(lag_err**2 + lat_err**2)

    df["lag_err"] = lag_err
    df["lat_err"] = lat_err
    df["pos_err"] = pos_err
    return df

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    # Your logs use "t" for time; require core pose fields
    df = df.dropna(subset=["t", "x", "y", "yaw"]).copy()
    if len(df) < 3:
        raise ValueError(f"Too few valid samples after dropna in {csv_path}")

    t0 = df["t"].iloc[0]
    df["time"] = df["t"] - t0

    # Robust speed + beta typing
    if "speed" in df.columns:
        df["speed"] = df["speed"].astype(float)
    else:
        df["speed"] = np.hypot(df.get("vx", 0.0), df.get("vy", 0.0))

    if "beta" in df.columns:
        df["beta"] = df["beta"].astype(float)

    track_choice = df["track_choice"].iloc[0]
    df = compute_track_errors(df, track_choice)
    return df

# ---------------------------------------------------------------------
# RUN SUMMARY (NO PLOTTING)
# ---------------------------------------------------------------------
def summarise_run(df, csv_path):
    t = df["time"].to_numpy(float)
    T = float(t[-1] - t[0]) if len(t) else np.nan
    dt = np.diff(t, prepend=t[0]); dt[0] = 0.0
    mean_dt = float(np.mean(dt)) if len(dt) else np.nan
    n = int(len(df))

    mean_comp_time = float(np.nanmean(df["comp_time"].to_numpy(float))) if "comp_time" in df.columns else np.nan

    # speed
    if "speed" in df.columns and df["speed"].notna().any():
        sp = df["speed"].to_numpy(float)
        if np.nanmax(sp) > 200:  # sanity guard
            sp = np.hypot(df["vx"], df["vy"])
    else:
        sp = np.hypot(df["vx"], df["vy"])

    dist = float(np.nansum(sp * dt))
    speed_mean = float(np.nanmean(sp))
    speed_max = float(np.nanmax(sp))
    speed_p95 = float(pct(sp, 95))

    # errors
    lat = df["lat_err"].to_numpy(float)
    lag = df["lag_err"].to_numpy(float)
    pos = df["pos_err"].to_numpy(float)

    lat_mean_abs = float(np.nanmean(np.abs(lat)))
    lat_rms = float(nanrms(lat))
    lat_p95 = float(pct(np.abs(lat), 95))
    lat_max_abs = float(np.nanmax(np.abs(lat)))

    lag_mean_abs = float(np.nanmean(np.abs(lag)))
    lag_rms = float(nanrms(lag))

    pos_mean = float(np.nanmean(pos))
    pos_rms = float(nanrms(pos))
    pos_p95 = float(pct(pos, 95))
    pos_max = float(np.nanmax(pos))

    int_abs_lat = float(area_l1(lat, t))
    int_pos = float(area_l1(pos, t))
    abs_lat_per_s = float(int_abs_lat / T) if T > 0 else np.nan
    pos_per_s = float(int_pos / T) if T > 0 else np.nan

    # time in band
    thresholds = [0.05, 0.10, 0.20, 0.50]
    tib = {}
    for th in thresholds:
        mask = np.isfinite(lat) & (np.abs(lat) < th)
        tib[th] = float(np.sum(dt[mask])) / T if T > 0 else np.nan

    # controls
    thr = df["throttle"].to_numpy(float) if "throttle" in df.columns else np.full(n, np.nan)
    ste = df["steering"].to_numpy(float) if "steering" in df.columns else np.full(n, np.nan)

    ste_angle = mf.steering_2_steering_angle(
        ste, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
    )
    dste = _diff(ste_angle, t)

    thr_mean = float(np.nanmean(thr))
    thr_l1 = float(area_l1(thr, t))
    ste_rms = float(nanrms(ste_angle))
    ste_rate_energy = float(area_l2(dste, t))

    # beta if available
    if "beta" in df.columns:
        beta_deg = np.rad2deg(df["beta"].to_numpy(float))
        beta_mean_abs = float(np.nanmean(np.abs(beta_deg)))
        beta_p95 = float(pct(np.abs(beta_deg), 95))
        beta_max = float(np.nanmax(np.abs(beta_deg)))
    else:
        beta_mean_abs = beta_p95 = beta_max = np.nan

    # lap time if available
    if "lap_time" in df.columns and "max_laps" in df.columns and np.nansum(df["lap_time"]) > 0.0:
        mean_lap_time = float(np.nansum(df["lap_time"]) / float(df["max_laps"].iloc[0]))
    else:
        mean_lap_time = np.nan

    omega = df["omega"].to_numpy(float) if "omega" in df.columns else np.full(n, np.nan)

    # instability metrics
    omega_thr = 1.0
    peak_thr = 0.8
    steer_rate_thr = 80.0  # deg/s

    npk_o, nps_o = _count_peaks_simple(t, omega, thr=peak_thr, min_sep_s=0.10, smooth_k=5)
    nb_o, frac_o, long_o, area_o = _burst_metrics(t, omega, thr=omega_thr)
    domega = _diff(omega, t)
    omega_max_steepness = float(np.nanmax(np.abs(domega))) if np.any(np.isfinite(domega)) else np.nan

    # steering peaks/bursts
    steer_is_rad = np.nanmax(np.abs(ste_angle)) < 3.5
    peak_thr_steer = (np.deg2rad(2.5) if steer_is_rad else 2.5)
    npk_s, nps_s = _count_peaks_simple(t, ste_angle, thr=peak_thr_steer, min_sep_s=0.10, smooth_k=5)

    dste_deg = np.rad2deg(_diff(ste_angle, t)) if steer_is_rad else _diff(ste_angle, t)
    nb_sr, frac_sr, long_sr, area_sr = _burst_metrics(t, dste_deg, thr=steer_rate_thr)
    steer_rate_max = float(np.nanmax(np.abs(dste_deg))) if np.any(np.isfinite(dste_deg)) else np.nan

    # rates / jerks
    dthr = _diff(thr, t)
    ddthr = _diff(dthr, t)
    ddste = _diff(_diff(ste_angle, t), t)
    ddste_deg = np.rad2deg(ddste) if steer_is_rad else ddste

    # TV / step stats
    thr_tv_s = tv_per_second(thr, t)
    ste_tv_s = tv_per_second(ste, t)

    thr_nsteps, thr_meanstep, thr_maxstep = step_stats(thr, eps=1e-3)
    ste_nsteps, ste_meanstep, ste_maxstep = step_stats(ste, eps=1e-3)

    out = {
        "file": csv_path,
        "group": "",          # filled by caller
        "run_idx": np.nan,    # filled by caller

        "mppi_model": df["mppi_model"].iloc[0] if "mppi_model" in df.columns else "",
        "sim_model": df["sim_model"].iloc[0] if "sim_model" in df.columns else "",
        "track_choice": df["track_choice"].iloc[0] if "track_choice" in df.columns else "",
        "dt": df["dt"].iloc[0] if "dt" in df.columns else np.nan,
        "mean_comp_time": mean_comp_time,

        "duration_s": T,
        "samples": n,
        "mean_dt_s": mean_dt,
        "mean_lap_time": mean_lap_time,

        "distance_m": dist,
        "speed_mean": speed_mean,
        "speed_max": speed_max,
        "speed_p95": speed_p95,

        "lat_mean_abs_m": lat_mean_abs,
        "lat_rms_m": lat_rms,
        "lat_p95_m": lat_p95,
        "lat_max_abs_m": lat_max_abs,

        "lag_mean_abs_m": lag_mean_abs,
        "lag_rms_m": lag_rms,

        "pos_mean_m": pos_mean,
        "pos_rms_m": pos_rms,
        "pos_p95_m": pos_p95,
        "pos_max_m": pos_max,

        "int_abs_lat_m_s": int_abs_lat,
        "int_pos_m_s": int_pos,
        "abs_lat_per_s_m": abs_lat_per_s,
        "pos_err_per_s_m": pos_per_s,

        "tib_<5cm": tib[0.05],
        "tib_<10cm": tib[0.10],
        "tib_<20cm": tib[0.20],
        "tib_<50cm": tib[0.50],

        "thr_mean": thr_mean,
        "int_abs_thr": thr_l1,
        "ste_rms": ste_rms,
        "ste_rate_energy": ste_rate_energy,

        "beta_mean_abs_deg": beta_mean_abs,
        "beta_p95_abs_deg": beta_p95,
        "beta_max_deg": beta_max,

        "omega_n_peaks": npk_o,
        "omega_peaks_per_s": nps_o,
        "omega_unstable_bursts": nb_o,
        "omega_unstable_frac": frac_o,
        "omega_unstable_longest_s": long_o,
        "omega_unstable_area": area_o,
        "omega_max_steepness": omega_max_steepness,

        "steer_n_peaks": npk_s,
        "steer_peaks_per_s": nps_s,
        "steer_rate_unstable_bursts": nb_sr,
        "steer_rate_unstable_frac": frac_sr,
        "steer_rate_unstable_longest_s": long_sr,
        "steer_rate_unstable_area": area_sr,
        "steer_rate_max": steer_rate_max,

        "thr_rate_rms": nanrms(dthr),
        "thr_jerk_rms": nanrms(ddthr),
        "ste_rate_rms_deg_s": nanrms(dste_deg),
        "ste_jerk_rms_deg_s2": nanrms(ddste_deg),

        "thr_tv_per_s": thr_tv_s,
        "ste_tv_per_s": ste_tv_s,

        "thr_n_steps": thr_nsteps,
        "thr_mean_step": thr_meanstep,
        "thr_max_step": thr_maxstep,
        "ste_n_steps": ste_nsteps,
        "ste_mean_step": ste_meanstep,
        "ste_max_step": ste_maxstep,

        "corr_dthr_dste": corr(_diff(thr, t), dste_deg),
        "corr_thr_ste": corr(thr, ste),
    }
    return out

# ---------------------------------------------------------------------
# PREFIX EXPANSION
# ---------------------------------------------------------------------
_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)$")

def expand_token_to_runs(tok: str, nruns: int):
    """
    If tok ends in _<number>, treat as an explicit single run.
    Else, expand to tok_1 ... tok_<nruns>.
    """
    m = _SUFFIX_RE.match(tok)
    if m:
        return [(tok, int(m.group("idx")))]
    return [(f"{tok}_{i}", i) for i in range(1, nruns + 1)]

import os
import numpy as np
import matplotlib.pyplot as plt

import os
import numpy as np
import matplotlib.pyplot as plt

def _load_time_speed(csv_path: str):
    df = load_and_prepare(csv_path)
    t = df["time"].to_numpy(float)
    v = df["speed"].to_numpy(float)
    m = np.isfinite(t) & np.isfinite(v)
    t = t[m]; v = v[m]
    if len(t) < 3:
        raise ValueError(f"Too few valid (time,speed) samples in {csv_path}")
    # ensure monotonic time (just in case)
    order = np.argsort(t)
    return t[order], v[order]

def _interp_to_grid(t, v, t_grid):
    # numpy.interp requires increasing x; we ensured that above
    # outside range -> we will only query within [t[0], t[-1]] anyway
    return np.interp(t_grid, t, v)

def plot_group_speed_over_time(groups: dict, base_dir: str = BASE_DIR,
                               dt_grid: float = 0.05,  # choose what makes sense for your logger rate
                               title: str = "Ensemble-average speed over time",
                               show_std: bool = True):
    """
    groups:
      { "Group A": ["run1.csv","run2.csv","run3.csv"], "Group B": [...], "Group C": [...] }
    For each group: interpolate each run speed to common time grid, then mean over runs.
    """

    fig, ax = plt.subplots(figsize=(9, 4.8))
    missing = []

    for gname, run_list in groups.items():
        series = []
        durations = []

        # 1) Load all runs
        for p in run_list:
            p_full = p if os.path.isabs(p) else os.path.join(base_dir, p)
            if not os.path.exists(p_full):
                missing.append(p_full)
                continue

            t, v = _load_time_speed(p_full)
            series.append((t, v))
            durations.append(t[-1])

        if len(series) == 0:
            print(f"[WARN] group '{gname}': no runs loaded.")
            continue

        # 2) Common grid: only up to shortest run (so everyone has data)
        t_end = float(np.min(durations))
        if t_end <= 0:
            print(f"[WARN] group '{gname}': non-positive duration.")
            continue

        t_grid = np.arange(0.0, t_end, dt_grid)

        # 3) Interpolate each run to grid
        V = []
        for (t, v) in series:
            V.append(_interp_to_grid(t, v, t_grid))
        V = np.vstack(V)  # shape (n_runs, n_time)

        # 4) Mean + std across runs at each time
        v_mean = np.nanmean(V, axis=0)
        v_std  = np.nanstd(V, axis=0, ddof=1) if V.shape[0] > 1 else np.zeros_like(v_mean)

        ax.plot(t_grid, v_mean, linewidth=2, label=f"{gname} (n={V.shape[0]})")

        if show_std and V.shape[0] > 1:
            ax.fill_between(t_grid, v_mean - v_std, v_mean + v_std, alpha=0.4)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    # ax.set_title(title)
    ax.grid(True)
    # ax.legend()
    # plt.tight_layout()
    plt.show()

    if missing:
        print(f"\n[WARN] Missing {len(missing)} files:")
        for p in missing[:20]:
            print("  ", p)
        if len(missing) > 20:
            print("  ...")

# if __name__ == "__main__":
#     # --- Fill these in manually (3 groups x 3 runs) ---
#     GROUPS = {
#         # "Params A": [
#         #     "/home/maarten/Documents/Thesis/log_Dart/1dart_nosteer_s500.csv",
#         #     "/home/maarten/Documents/Thesis/log_Dart/1dart_nosteer_s400.csv",
#         #     "/home/maarten/Documents/Thesis/log_Dart/1dart_nosteer_s500_2.csv",
#         #     "/home/maarten/Documents/Thesis/log_Dart/1dart_nosteer_s500_3.csv",
#         #     "/home/maarten/Documents/Thesis/log_Dart/1dart_nosteer_s500_4.csv",
#         # ],
#         "Params B": [
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_s500.csv",
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_s400.csv",
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_s500_2.csv",
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_s500_3.csv",
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_s500_4.csv",
#         ],
#         "Params C": [
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_005temp_s500.csv",
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_005temp_s400.csv",
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_s500_005temp_2.csv",
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_s500_005temp_3.csv",
#             "/home/maarten/Documents/Thesis/log_Dart/1dart_steer_s500_005temp_4.csv",
#         ],
#     }
#
#     plot_group_speed_over_time(GROUPS, base_dir=BASE_DIR, dt_grid=0.05, show_std=True)
#
import numpy as np
import matplotlib.pyplot as plt

def steering_mapping_np(u, a_s, b_s, c_s, d_s, e_s):
    w = 0.5 * (np.tanh(30 * (u + c_s)) + 1.0)
    angle1 = b_s * np.tanh(a_s * (u + c_s))
    angle2 = d_s * np.tanh(e_s * (u + c_s))
    return (w) * angle1 + (1-w) * angle2, w, angle1, angle2 # w * angle1 + (1 - w) * angle2

u = np.linspace(-1.0, 1.0, 2000)

# steering angle curve --from fitting on vicon data
a_s = 1.392930030822754
b_s = 0.36576229333877563
c_s = 0.0
d_s = 0.5147881507873535
e_s = 1.0230425596237183

angle, w, angle1, angle2 = steering_mapping_np(
    u, a_s, b_s, c_s, d_s, e_s
)

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(u, angle, linewidth=2)

ax.set_xlabel("Steering command [-]", fontsize=20)
ax.set_ylabel("Steering angle [rad]", fontsize=20)

ax.tick_params(axis='both', which='major', labelsize=14)

ax.grid(True)
plt.tight_layout()
plt.axis("equal")
plt.show()
#
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--csv", nargs="+", required=True,
#                     help="One or more experiment prefixes (e.g. exteval_simple_kinkin_v15_mppi) "
#                          "or explicit run names (e.g. ..._3).")
#     ap.add_argument("--nruns", type=int, default=5, help="How many numbered runs per prefix (default: 5).")
#     ap.add_argument("--out", default=None,
#                     help="Output summary CSV path. Default: <BASE_DIR>/summaries/<run_id>/compare.csv")
#     ap.add_argument("--strict", action="store_true",
#                     help="If set, missing CSV files cause an error. Otherwise they are skipped with a warning.")
#     args = ap.parse_args()
#
#     # Expand all tokens into concrete run files
#     expanded = []  # list of dicts with group, run_idx, path
#     for tok in args.csv:
#         for run_tok, run_idx in expand_token_to_runs(tok, args.nruns):
#             path = resolve_path(run_tok)
#             expanded.append({
#                 "group": tok,     # group is the *original token* you provided
#                 "run_idx": run_idx,
#                 "path": path,
#             })
#
#     # Build default out path
#     labels = [clean_label(x) for x in args.csv]
#     run_id = build_run_id(labels) if len(labels) <= 4 else "compare_many_groups"
#     default_root = os.path.join(BASE_DIR, "summaries", run_id)
#     out_csv = args.out if args.out else os.path.join(default_root, "compare.csv")
#     os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#
#     summaries = []
#     missing = []
#
#     for item in expanded:
#         p = item["path"]
#         if not os.path.exists(p):
#             missing.append(p)
#             if args.strict:
#                 raise FileNotFoundError(f"Missing CSV: {p}")
#             continue
#
#         df = load_and_prepare(p)
#         row = summarise_run(df, p)
#         row["group"] = item["group"]
#         row["run_idx"] = item["run_idx"]
#         summaries.append(row)
#
#     if not summaries:
#         raise RuntimeError("No CSVs were processed (all missing or empty after filtering).")
#
#     summ_df = pd.DataFrame(summaries)
#
#     # Column order: keep key identifiers first
#     preferred = [
#         "group", "run_idx", "file",
#         "mppi_model", "sim_model", "track_choice", "dt", "mean_comp_time",
#         "duration_s", "samples", "mean_dt_s", "mean_lap_time",
#         "distance_m", "speed_mean", "speed_max", "speed_p95",
#         "lat_mean_abs_m", "lat_rms_m", "lat_p95_m", "lat_max_abs_m",
#         "lag_mean_abs_m", "lag_rms_m",
#         "pos_mean_m", "pos_rms_m", "pos_p95_m", "pos_max_m",
#         "int_abs_lat_m_s", "abs_lat_per_s_m", "int_pos_m_s", "pos_err_per_s_m",
#         "tib_<5cm", "tib_<10cm", "tib_<20cm", "tib_<50cm",
#         "thr_mean", "int_abs_thr", "ste_rms", "ste_rate_energy",
#         "beta_mean_abs_deg", "beta_p95_abs_deg", "beta_max_deg",
#         "omega_n_peaks", "omega_peaks_per_s", "omega_unstable_bursts", "omega_unstable_frac",
#         "omega_unstable_longest_s", "omega_unstable_area", "omega_max_steepness",
#         "steer_n_peaks", "steer_peaks_per_s",
#         "steer_rate_unstable_bursts", "steer_rate_unstable_frac",
#         "steer_rate_unstable_longest_s", "steer_rate_unstable_area", "steer_rate_max",
#         "thr_rate_rms", "thr_jerk_rms", "ste_rate_rms_deg_s", "ste_jerk_rms_deg_s2",
#         "thr_tv_per_s", "ste_tv_per_s",
#         "thr_n_steps", "thr_mean_step", "thr_max_step",
#         "ste_n_steps", "ste_mean_step", "ste_max_step",
#         "corr_dthr_dste", "corr_thr_ste",
#     ]
#     cols = [c for c in preferred if c in summ_df.columns] + [c for c in summ_df.columns if c not in preferred]
#     summ_df = summ_df[cols]
#
#     summ_df.to_csv(out_csv, index=False)
#
#     print(f"Saved: {out_csv}")
#     if missing:
#         print(f"\nWarning: skipped {len(missing)} missing files (use --strict to error).")
#         for p in missing[:10]:
#             print(f"  missing: {p}")
#         if len(missing) > 10:
#             print(f"  ... ({len(missing) - 10} more)")
#
# if __name__ == "__main__":
#     main()

