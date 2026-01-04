#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
from matplotlib.patches import Ellipse
import yaml

from dart_dynamic_models import model_functions,load_SVGPModel_actuator_dynamics_analytic
import sys
import importlib.resources

mf = model_functions()


from path_track_definitions import generate_path_data

abs_path = os.path.dirname(os.path.abspath(__file__))
mpl.rcParams.update({
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35
})
BASE_DIR = "/home/maarten/Documents/Thesis/log_Dart"

# obstacle footprint (must match ROSObjective.obst_a/obst_b)
OBST_A = 0.5   # [m] semi-axis along length
OBST_B = 0.25  # [m] semi-axis along width

CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))

def _add_direction_arrows(ax, xs, ys, num=3, color=None, frac_len=0.05, z=3):
    """Place `num` arrows along polyline (xs, ys).
    frac_len is the arrow length as a fraction of plot span."""

    xs = np.asarray(xs); ys = np.asarray(ys)
    if len(xs) < 3:  # need a bit of data
        return
    # evenly spaced indices (skip endpoints)
    idxs = np.linspace(0, len(xs)-1, num+2, dtype=int)[1:-1]
    # tangent via gradient
    dx = np.gradient(xs); dy = np.gradient(ys)
    # arrow length in data units based on axes span
    xspan = max(ax.get_xlim()) - min(ax.get_xlim())
    yspan = max(ax.get_ylim()) - min(ax.get_ylim())
    L = max(xspan, yspan)
    base = frac_len * L

    ux = dx[idxs]; uy = dy[idxs]
    n = np.hypot(ux, uy) + 1e-9
    ux = base * (ux / n); uy = base * (uy / n)

    ax.quiver(xs[idxs], ys[idxs], ux, uy,
              angles='xy', scale_units='xy', scale=1,
              width=0.003, headwidth=3, headlength=5, headaxislength=4,
              color=color, zorder=z, label='_nolegend_')

def nearest_path_indices(xs, ys, x_path, y_path):
    xy_path = np.vstack([x_path, y_path]).T
    xy = np.vstack([xs, ys]).T
    return np.argmin(((xy[:, None, :] - xy_path[None, :, :])**2).sum(-1), axis=1)

def nanrms(x):
    x = np.asarray(x, float)
    return np.sqrt(np.nanmean(x**2))

def pct(x, p):
    x = np.asarray(x, float)
    return np.nanpercentile(x, p)

def area_l1(signal, t):
    # ∫ |signal| dt, robust to uneven dt and NaNs
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
    # ∫ signal^2 dt
    s = np.asarray(signal, float)
    tt = np.asarray(t, float)
    m = np.isfinite(s) & np.isfinite(tt)
    if m.sum() < 2:
        return np.nan
    s = s[m]; tt = tt[m]
    dt = np.diff(tt, prepend=tt[0]); dt[0] = 0.0
    return np.sum((s**2) * dt)

def derive(signal, t):
    s = np.asarray(signal, float)
    tt = np.asarray(t, float)
    m = np.isfinite(s) & np.isfinite(tt)
    out = np.full_like(s, np.nan, dtype=float)
    if m.sum() >= 3:
        out[m] = np.gradient(s[m], tt[m])
    return out

def compute_track_errors(df, track_choice):
    # Load path and tangents
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

def summarise_run(df, csv_path):
    # Time base
    t = df["time"].to_numpy()
    T = float(t[-1] - t[0]) if len(t) else np.nan
    dt = np.diff(t, prepend=t[0]); dt[0] = 0.0
    mean_dt = float(np.mean(dt)) if len(dt) else np.nan
    n = int(len(df))
    mean_comp_time = sum(df["comp_time"])/len(df["comp_time"])

    # Kinematics
    if "speed" in df.columns and df["speed"].notna().any():
        sp = df["speed"].to_numpy(float)
        # If speed accidentally stored as v^2, try to detect (rare): fallback to recompute if negative or too large
        if np.nanmax(sp) > 200:  # sanity guard
            sp = np.hypot(df["vx"], df["vy"])
    else:
        sp = np.hypot(df["vx"], df["vy"])
    dist = np.nansum(sp * dt)
    speed_mean = float(np.nanmean(sp))
    speed_max  = float(np.nanmax(sp))
    speed_p95  = float(pct(sp, 95))

    # Errors
    lat = df["lat_err"].to_numpy(float)
    lag = df["lag_err"].to_numpy(float)
    pos = df["pos_err"].to_numpy(float)

    lat_mean_abs = float(np.nanmean(np.abs(lat)))
    lat_rms      = float(nanrms(lat))
    lat_p95      = float(pct(np.abs(lat), 95))
    lat_max_abs  = float(np.nanmax(np.abs(lat)))

    lag_mean_abs = float(np.nanmean(np.abs(lag)))
    lag_rms      = float(nanrms(lag))
    pos_mean     = float(np.nanmean(pos))
    pos_rms      = float(nanrms(pos))
    pos_p95      = float(pct(pos, 95))
    pos_max      = float(np.nanmax(pos))

    int_abs_lat = float(area_l1(lat, t))       # m·s
    int_pos     = float(area_l1(pos, t))       # m·s
    abs_lat_per_s = float(int_abs_lat / T) if T > 0 else np.nan
    pos_per_s     = float(int_pos / T) if T > 0 else np.nan

    # Time in band (good tracking)
    thresholds = [0.05, 0.10, 0.20, 0.50]  # meters
    tib = {}
    for th in thresholds:
        mask = np.isfinite(lat) & (np.abs(lat) < th)
        tib[th] = float(np.sum(dt[mask])) / T if T > 0 else np.nan

    # Controls
    thr = df["throttle"].to_numpy(float) if "throttle" in df.columns else np.full(n, np.nan)
    ste = df["steering"].to_numpy(float) if "steering" in df.columns else np.full(n, np.nan)

    ste_angle = mf.steering_2_steering_angle(ste, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self)
    dste = derive(ste_angle, t)

    thr_mean = float(np.nanmean(thr))
    thr_l1   = float(area_l1(thr, t))
    ste_rms  = float(nanrms(ste_angle))
    # ste_energy = float(area_l2(ste_angle, t))       # ∫ steer^2 dt
    ste_rate_energy = float(area_l2(dste, t)) # ∫ (d/dt steer)^2 dt

    # Sideslip if available
    if "beta" in df.columns:
        beta = df["beta"].to_numpy(float)
        beta_deg = np.rad2deg(beta)
        beta_mean_abs = float(np.nanmean(np.abs(beta_deg)))
        beta_p95 = float(pct(np.abs(beta_deg), 95))
        beta_max = float(np.nanmax(np.abs(beta_deg)))
    else:
        beta_mean_abs = beta_p95 = beta_max = np.nan

    if sum(df["lap_time"]) > 0.0:
        mean_lap_time = sum(df["lap_time"]) / (df["max_laps"][0])  # Divide by number of laps done
    else:
        mean_lap_time = 0.0

    omega = df["omega"].to_numpy(float) if "omega" in df.columns else np.full(n, np.nan)

    # thresholds (tune once)
    omega_thr = 1.0     # rad/s deemed "unstable"
    peak_thr  = 0.8     # rad/s for counting peaks
    steer_rate_thr = 80.0  # deg/s

    # yaw-rate peaks & bursts
    npk_o, nps_o = _count_peaks_simple(t, omega, thr=peak_thr, min_sep_s=0.10, smooth_k=5)
    nb_o, frac_o, long_o, area_o = _burst_metrics(t, omega, thr=omega_thr)
    domega = _diff(omega, t)
    omega_max_steepness = float(np.nanmax(np.abs(domega))) if np.any(np.isfinite(domega)) else np.nan

    # steering peaks & bursts (derivative-based bursts)
    if "steering" in df.columns:
        steering = ste_angle
        dste = _diff(steering, t)
        # try to detect radians vs degrees for the *value* peak count
        steer_is_rad = np.nanmax(np.abs(steering)) < 3.5
        peak_thr_steer = (np.deg2rad(2.5) if steer_is_rad else 2.5)
        npk_s, nps_s = _count_peaks_simple(t, steering, thr=peak_thr_steer, min_sep_s=0.10, smooth_k=5)

        # for bursts, use steering rate in deg/s
        dste_deg = np.rad2deg(dste) if steer_is_rad else dste
        nb_sr, frac_sr, long_sr, area_sr = _burst_metrics(t, dste_deg, thr=steer_rate_thr)
        steer_rate_max = float(np.nanmax(np.abs(dste_deg))) if np.any(np.isfinite(dste_deg)) else np.nan
    else:
        npk_s = nps_s = nb_sr = frac_sr = long_sr = area_sr = steer_rate_max = np.nan


    out = {
        "file": csv_path, "mppi_model": df["mppi_model"][0], "sim_model": df["sim_model"][0],
        "track_choice": df["track_choice"][0], "dt": df["dt"][0], "mean_comp_time": mean_comp_time,
        "duration_s": T, "samples": n, "mean_dt_s": mean_dt, "mean_lap_time": mean_lap_time,
        "distance_m": dist, "speed_mean": speed_mean, "speed_max": speed_max,
        "lat_mean_abs_m": lat_mean_abs, "lat_rms_m": lat_rms, "lat_max_abs_m": lat_max_abs,
        "lag_mean_abs_m": lag_mean_abs, "lag_rms_m": lag_rms,
        "pos_mean_m": pos_mean, "pos_rms_m": pos_rms, "pos_max_m": pos_max,
        "int_abs_lat_m_s": int_abs_lat, "int_pos_m_s": int_pos,
        "abs_lat_per_s_m": abs_lat_per_s, "pos_err_per_s_m": pos_per_s,
        "tib_<5cm": tib[0.05], "tib_<10cm": tib[0.10], "tib_<20cm": tib[0.20], "tib_<50cm": tib[0.50],
        "thr_mean": thr_mean, "int_abs_thr": thr_l1,
        "ste_rms": ste_rms,  "ste_rate_energy": ste_rate_energy,
        "beta_mean_abs_deg": beta_mean_abs, "beta_max_deg": beta_max,
    }

    out.update({
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
    })
    ste_angle = mf.steering_2_steering_angle(ste, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self)

    # Decide if in rad or deg:
    ste_is_rad = np.nanmax(np.abs(ste_angle)) < 3.5
    if ste_is_rad:
        ste_plot = np.rad2deg(ste_angle)  # for nicer units
    else:
        ste_plot = ste_angle

    dthr = _diff(thr, t)
    ddthr = _diff(dthr, t)

    dste = _diff(ste_angle, t)
    ddste = _diff(dste, t)

    # Convert to consistent units for steering jerk
    dste_deg = np.rad2deg(dste) if ste_is_rad else dste
    ddste_deg = np.rad2deg(ddste) if ste_is_rad else ddste

    out.update({
        "thr_rate_rms": nanrms(dthr),
        "thr_jerk_rms": nanrms(ddthr),
        "ste_rate_rms_deg_s": nanrms(dste_deg),
        "ste_jerk_rms_deg_s2": nanrms(ddste_deg),
    })

    thr_tv_s = tv_per_second(thr, t)
    ste_tv_s = tv_per_second(ste, t)

    thr_nsteps, thr_meanstep, thr_maxstep = step_stats(thr, eps=1e-3)
    ste_nsteps, ste_meanstep, ste_maxstep = step_stats(ste, eps=1e-3)

    out.update({
        "thr_tv_per_s": thr_tv_s,
        "ste_tv_per_s": ste_tv_s,
        "thr_n_steps": thr_nsteps,
        "thr_mean_step": thr_meanstep,
        "thr_max_step": thr_maxstep,
        "ste_n_steps": ste_nsteps,
        "ste_mean_step": ste_meanstep,
        "ste_max_step": ste_maxstep,
    })

    out["corr_dthr_dste"] = corr(_diff(thr, t), dste_deg)
    out["corr_thr_ste"] = corr(thr, ste)

    return out


def corr(a, b):
    a = np.asarray(a, float);
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a = a[m];
    b = b[m]
    if len(a) < 5:
        return np.nan
    a = a - np.mean(a);
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


# obstacle footprint (must match ROSObjective.obst_a / obst_b)
OBST_A = 0.5   # [m] semi-axis along obstacle "length"
OBST_B = 0.25  # [m] semi-axis along obstacle "width"

# when is obstacle "relevant" in XY plot?
OBS_DIST_THRESH = 1.0      # [m]
OBS_MAX_ELLIPSES = 10      # max ellipses per run

# lane parameters (keep consistent with ROSObjective)
LANE_WIDTH   = 1.0         # [m]
LANE_MARGIN  = 0.05        # [m]
W_LANE       = 100.0       # same w_lane as in objective
OBST_MARGIN  = 0.05        # obstacle_margin in objective
W_OPPONENT   = 300.0       # same w_opponent as in objective

def softplus_hinge(x, sharp=25.0):
    x = np.asarray(x, float)
    return np.log1p(np.exp(sharp * x)) / sharp

def plot_one(df, label=None):

    # XY
    # plt.figure("XY"); plt.plot(df["x"], df["y"], label=label);
    # plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Trajectory comparison")
    # plt.legend()

    # # XY
    # plt.figure("XY")
    # ax = plt.gca()
    # ax.plot(df["x"], df["y"], label=label)

    # --- dynamic obstacle overlay (if present) ---
    # if {"obstacle_x", "obstacle_y", "obstacle_yaw"}.issubset(df.columns):
    #     obs_x   = df["obstacle_x"].to_numpy(float)
    #     obs_y   = df["obstacle_y"].to_numpy(float)
    #     obs_yaw = df["obstacle_yaw"].to_numpy(float)
    #
    #     # path of the obstacle centre (optional dashed line)
    #     ax.plot(obs_x, obs_y, "--", alpha=0.4,
    #             label=(f"obstacle path {label}" if label else "obstacle path"))
    #
    #     # distance car–obstacle (logged or recomputed)
    #     if "obstacle_dist" in df.columns:
    #         dist = df["obstacle_dist"].to_numpy(float)
    #     else:
    #         dx = df["x"].to_numpy(float) - obs_x
    #         dy = df["y"].to_numpy(float) - obs_y
    #         dist = np.hypot(dx, dy)
    #
    #     # indices where car is close to obstacle
    #     close_idxs = np.where(np.isfinite(dist) & (dist <= OBS_DIST_THRESH))[0]
    #
    #     if close_idxs.size > 0:
    #         # downsample so we don't draw at every timestep
    #         if close_idxs.size > OBS_MAX_ELLIPSES:
    #             step = int(np.ceil(close_idxs.size / OBS_MAX_ELLIPSES))
    #             close_idxs = close_idxs[::step]
    #
    #         for i in close_idxs:
    #             cx = obs_x[i]
    #             cy = obs_y[i]
    #             yaw = obs_yaw[i]
    #             if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(yaw):
    #                 continue
    #
    #             angle_deg = np.rad2deg(yaw)
    #             e = Ellipse(
    #                 (cx, cy),
    #                 width=2.0 * OBST_A,
    #                 height=2.0 * OBST_B,
    #                 angle=angle_deg,
    #                 fill=True,
    #                 alpha=0.35,
    #                 edgecolor="none"
    #             )
    #             ax.add_patch(e)
    #
    # ax.set_aspect("equal", adjustable="box")
    # plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Trajectory comparison")
    # plt.legend()

    # XY
    plt.figure("XY"); plt.plot(df["x"], df["y"], label=label);
    plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Trajectory comparison")
    plt.legend()

    # --- Closeness to obstacle and lane ---
    if "obstacle_dist" in df.columns:
        t   = df["time"].to_numpy(float)
        lat = df["lat_err"].to_numpy(float)

        half_w = 0.5 * LANE_WIDTH

        # distance to each lane boundary in a "clearance" sense:
        # positive = inside lane, negative = outside
        lane_clearance = half_w - np.abs(lat)   # [m]

        plt.figure("Closeness")
        plt.plot(t, df["obstacle_dist"].to_numpy(float),
                 label=f"obstacle_dist {label}" if label else "obstacle_dist")
        plt.plot(t, lane_clearance,
                 label=f"lane_clearance {label}" if label else "lane_clearance")

        plt.axhline(0.0, color="k", linestyle="--", linewidth=1, label="_nolegend_")
        plt.xlabel("time [s]")
        plt.ylabel("distance [m]")
        plt.title("Closeness to obstacle and lanes")
        plt.legend()

    # --- Lane & obstacle cost signals over time ---
    has_obs = {"obstacle_x", "obstacle_y", "obstacle_yaw"}.issubset(df.columns)
    if has_obs:
        t   = df["time"].to_numpy(float)
        lat = df["lat_err"].to_numpy(float)

        # Lane cost: w_lane * softplus_hinge(|lat_err| - (half_w - margin))
        half_w = 0.5 * LANE_WIDTH
        excess = np.abs(lat) - (half_w - LANE_MARGIN)
        lane_cost = W_LANE * softplus_hinge(excess)
        lane_cost = np.clip(lane_cost, 0.0, 300.0)  # same cap as in code

        # Obstacle cost: ellipse-based, matching _obstacle_cost
        x = df["x"].to_numpy(float)
        y = df["y"].to_numpy(float)
        ox = df["obstacle_x"].to_numpy(float)
        oy = df["obstacle_y"].to_numpy(float)
        oyaw = df["obstacle_yaw"].to_numpy(float)

        dx = x - ox
        dy = y - oy

        # transform into obstacle frame
        cy = np.cos(oyaw)
        sy = np.sin(oyaw)
        # body frame: R^T * d
        x_loc =  cy * dx + sy * dy
        y_loc = -sy * dx + cy * dy

        phi = (x_loc / OBST_A)**2 + (y_loc / OBST_B)**2 - 1.0  # ellipse implicit
        arg = -phi + OBST_MARGIN
        obstacle_cost = W_OPPONENT * softplus_hinge(arg)
        obstacle_cost = np.clip(obstacle_cost, 0.0, 1e6)

        plt.figure("Cost components")
        plt.plot(t, lane_cost, label=f"lane_cost {label}" if label else "lane_cost")
        plt.plot(t, obstacle_cost,  label=f"obstacle_cost {label}"  if label else "obstacle_cost")
        plt.xlabel("time [s]")
        plt.ylabel("cost")
        plt.title("Lane & obstacle cost over time")
        plt.legend()

    # Speed & vy
    plt.figure("Speed (vx)")
    plt.plot(df["time"], np.hypot(df["vx"], df["vy"]), label=f"speed {label}" if label else "speed")
    # if "vx est" in df:  plt.plot(df["time"], np.hypot(df["vx est"], df["vy est"]), label=f"(est) speed {label}" if label else "(est) speed")
    plt.xlabel("time [s]"); plt.ylabel("[m/s]"); plt.title("Speed"); plt.legend()

    plt.figure("vy")
    plt.plot(df["time"], df["vy"], label=f"vy {label}" if label else "vy")
    # if "vy est" in df:  plt.plot(df["time"], df["vy est"], label=f"(est) vy {label}" if label else "est vy")
    plt.xlabel("time [s]"); plt.ylabel("[m/s]"); plt.title("Lateral Velocity"); plt.legend()

    # Yaw rate & beta
    plt.figure("Omega")
    plt.plot(df["time"], df["omega"], label=f"omega {label}" if label else "omega")
    plt.xlabel("time [s]"); plt.ylabel("yaw rate [rad/s]"); plt.title("Yaw rate"); plt.legend()

    plt.figure("Beta")
    plt.plot(df["time"], np.rad2deg(df["beta"]), label=f"beta_deg {label}" if label else "beta_deg")
    plt.xlabel("time [s]"); plt.ylabel("sideslip [deg]"); plt.title("Sideslip"); plt.legend()

    # Controls
    plt.figure("Throttle")
    plt.plot(df["time"], df["throttle"], label=f"throttle {label}" if label else "throttle", drawstyle="steps-post")
    plt.xlabel("time [s]"); plt.ylabel("throttle []"); plt.title("Throttle"); plt.legend()


    t = df["time"].to_numpy(float)
    ste = df["steering"].to_numpy(float)

    ste_angle = mf.steering_2_steering_angle(ste, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self)

    # Decide if in rad or deg:
    ste_is_rad = np.nanmax(np.abs(ste_angle)) < 3.5
    if ste_is_rad:
        ste_plot = np.rad2deg(ste_angle)  # for nicer units
    else:
        ste_plot = ste_angle
        
    ste_plot = ste

    plt.figure("Steering")
    plt.plot(df["time"], ste_plot, label=f"steering {label}" if label else "steering", drawstyle="steps-post")
    plt.xlabel("time [s]"); plt.ylabel("steering []"); plt.title("Steering"); plt.legend()

    # Rates
    if "throttle rate" in df.columns and "mode" == "s-mppi":
        plt.figure("Throttle Rate")
        plt.plot(df["time"], df["throttle rate"], label=f"throttle_rate {label}" if label else "throttle_rate", drawstyle="steps-post")
        plt.xlabel("time [s]"); plt.ylabel("throttle rate []"); plt.title("Throttle Rate"); plt.legend()

    else:
        dthr = _diff(df["throttle"],t)

        plt.figure("Throttle Rate")
        plt.plot(df["time"], dthr, label=f"throttle_rate {label}" if label else "throttle_rate", drawstyle="steps-post")
        plt.xlabel("time [s]"); plt.ylabel("throttle rate []"); plt.title("Throttle Rate"); plt.legend()

    if "steering rate" in df.columns and "mode" == "s-mppi":
        plt.figure("Steering Rate")
        plt.plot(df["time"], df["steering rate"], label=f"steering_rate {label}" if label else "steering_rate", drawstyle="steps-post")
        plt.xlabel("time [s]"); plt.ylabel("steering rate []"); plt.title("Steering Rate"); plt.legend()

        # dste_deg = .... has to be thought of a way to represent steering rate unitless to deg/s

    else:
        # Steering rate (deg/s)
        dste = _diff(ste_plot, t)
        dste_deg = np.rad2deg(dste) if ste_is_rad else dste

        plt.figure("Steering Rate")
        plt.plot(df["time"], dste, label=f"steering_rate {label}" if label else "steering_rate", drawstyle="steps-post")
        plt.xlabel("time [s]"); plt.ylabel("steering rate []]"); plt.title("Steering Rate"); plt.legend()


    # Errors
    plt.figure("Lateral Error")
    plt.plot(df["time"], df["lat_err"], label=f"lat {label}" if label else "lat")
    plt.xlabel("time [s]"); plt.ylabel("error [m]"); plt.title("Lateral Error vs time"); plt.legend()

    plt.figure("Lag Error")
    plt.plot(df["time"], df["lag_err"], label=f"lag {label}" if label else "lag")
    plt.xlabel("time [s]"); plt.ylabel("error [m]"); plt.title("Lag Error vs time"); plt.legend()

    plt.figure("Pos Error")
    plt.plot(df["time"], df["pos_err"], label=f"pos {label}" if label else "pos")
    plt.xlabel("time [s]"); plt.ylabel("error [m]"); plt.title("Positional Error vs time"); plt.legend()

    t = df["time"].to_numpy(float)
    dt = np.diff(t, prepend=t[0]); dt[0] = 0.0

    lat = df["lat_err"].to_numpy(float)
    pos = df["pos_err"].to_numpy(float)

    cum_abs_lat = np.cumsum(np.abs(lat))  # ∫|lat| dt
    cum_pos     = np.cumsum(pos)          # ∫||pos|| dt
    cum_lat2    = np.cumsum((lat**2))     # ∫lat^2 dt (optional)

    plt.figure("Cumulative Error (abs)")
    plt.plot(t, cum_abs_lat, label=f"∫|lat| dt — {label}")
    plt.plot(t, cum_pos,     label=f"∫pos dt — {label}")
    plt.xlabel("time [s]"); plt.ylabel("m"); plt.title("Cumulative Errors")
    plt.legend()

    simple_fft_plot(df["time"], ste_plot, label, title="Steering FFT")
    simple_fft_plot(df["time"], dste_deg, label, title="Steering rate FFT")

    if "temperature" in df.columns:
        plt.figure("Temperature")
        plt.plot(t, df["temperature"], label=f"temperature {label}" if label else "temperature")
        plt.xlabel("time [s]"); plt.ylabel("temperature [-]"); plt.title("Temperature over time"); plt.legend()

        plt.figure("Eta")
        plt.plot(t, df["eta"], label=f"eta {label}" if label else "eta")
        #plt.axhline(CONFIG["mppi"]["eta_u_bound"], color="k", linestyle="--", linewidth=1, label="upper bound eta")
        #plt.axhline(CONFIG["mppi"]["eta_l_bound"], color="k", linestyle="--", linewidth=1, label="lower bound eta")
        plt.xlabel("time [s]"); plt.ylabel("eta [-]"); plt.title("Eta over time"); plt.legend()


def simple_fft_plot(t, y,label, title="FFT"):
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # Remove NaNs
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    if len(t) < 4:
        return

    # Use uniform dt assumption (approx ok)
    dt = np.mean(np.diff(t))
    y0 = y - np.mean(y)  # remove DC

    # FFT
    Y = np.fft.rfft(y0)
    freqs = np.fft.rfftfreq(len(y0), d=dt)
    mag = np.abs(Y)

    plt.figure(title)
    plt.plot(freqs, mag, label=f"frequency {label}" if label else "frequency")
    plt.xlabel("frequency [Hz]")
    plt.ylabel("magnitude")
    plt.title(title)
    plt.xlim(0, 10)  # only low frequencies where your control acts
    plt.legend()


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["t","x","y","yaw"]).copy()

    if len(df["t"]) == []:
        print(df["t"])

    t0 = df["t"].iloc[0]; df["time"] = df["t"] - t0
    # Fill or compute speed/beta robustly
    if "speed" in df.columns:
        df["speed"] = df["speed"].astype(float)
    else:
        df["speed"] = np.hypot(df.get("vx", 0.0), df.get("vy", 0.0))
    if "beta" in df.columns:
        df["beta"] = df["beta"].astype(float)

    # Compute errors against track
    track_choice = df["track_choice"][0]
    df = compute_track_errors(df, track_choice)
    return df

def ensure_csv_ext(s: str) -> str:
    return s if s.lower().endswith(".csv") else s + ".csv"

def resolve_path(tok: str) -> str:
    tok = ensure_csv_ext(tok)
    return tok if os.path.isabs(tok) else os.path.join(BASE_DIR, tok)

def clean_label(path: str) -> str:
    """basename without extension, safe for folder names"""
    b = os.path.splitext(os.path.basename(path))[0]
    # replace anything odd with underscore (optional)
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", b)

def build_run_id(labels):
    """Join one or more labels to form the folder name"""
    return "__vs__".join(labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more CSV *filenames* or absolute paths (extension optional).")
    ap.add_argument("--out", help="Summary CSV path (default: summaries/filename/compare.csv).")
    ap.add_argument("--save_figs", help="Folder to save figures (default: summaries/filename/figs/).")
    args = ap.parse_args()


    # Resolve all paths and labels
    csv_paths = [resolve_path(tok) for tok in args.csv]
    labels = [clean_label(p) for p in csv_paths]

    if len(labels) > 4:
        run_id = "1too_long_make_own_name"
    else:
        run_id = build_run_id(labels)

    # Per-run output root: <BASE_DIR>/summaries/<run_id>/
    default_root = os.path.join(BASE_DIR, "summaries", run_id)
    fig_dir = args.save_figs if args.save_figs else os.path.join(default_root, "figs")
    out_csv = args.out if args.out else os.path.join(default_root, "compare.csv")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    summaries = []
    first = True

    for csv_path_i in args.csv:
        print(csv_path_i)
        if csv_path_i.lower().endswith(".csv"):
            pass
        else:
            csv_path_i += ".csv"


        csv_path = csv_path_i if os.path.isabs(csv_path_i) else os.path.join(BASE_DIR, csv_path_i)
        df = load_and_prepare(csv_path)
        summaries.append(summarise_run(df, csv_path))

        label = os.path.splitext(os.path.basename(csv_path))[0]
        plot_one(df, label=label)


    # Pretty table to stdout
    summ_df = pd.DataFrame(summaries)
    # Order cols (nice reading)
    preferred = [
        "file","mppi_model","sim_model","track_choice","dt","mean_comp_time",
        "duration_s","samples","mean_dt_s","mean_lap_time",
        "distance_m","speed_mean","speed_max",
        "lat_mean_abs_m","lat_rms_m","lat_max_abs_m",
        "lag_mean_abs_m","lag_rms_m",
        "pos_mean_m","pos_rms_m","pos_max_m",
        "int_abs_lat_m_s","abs_lat_per_s_m","int_pos_m_s","pos_err_per_s_m",
        "tib_<5cm","tib_<10cm","tib_<20cm",
        "thr_mean","int_abs_thr","ste_rms", "ste_rate_energy",
        "beta_mean_abs_deg","beta_max_deg"
    ]
    cols = [c for c in preferred if c in summ_df.columns] + [c for c in summ_df.columns if c not in preferred]
    summ_df = summ_df[cols]

    # Load track once
    (_, x_path, y_path,
     _, _, _,
     dx_ds, dy_ds, _, _,
     _, _) = generate_path_data(df["track_choice"][0])

    # XY
    plt.figure("XY");
    plt.plot(x_path, y_path, linestyle="--", linewidth=1, label="track")
    ax = plt.gca()
    # Add arrows: first set limits so span is known
    ax.relim();
    ax.autoscale_view()

    # three arrows on the **track** (grey)
    _add_direction_arrows(ax, x_path, y_path, num=2, color="0.0", frac_len=0.06, z=3)

    # os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    summ_df.to_csv(out_csv, index=False)
    print(f"\nSaved summary to: {out_csv}")

    # os.makedirs(args.save_figs, exist_ok=True)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        fig.tight_layout()
        ax = fig.get_axes()[0]
        fig.savefig(os.path.join(fig_dir, f"{ax.get_title()}.png"), dpi=150)
    print(f"Saved figures to: {fig_dir}")
    plt.show()

def _mavg(y, k):
    """Simple moving average with odd window k."""
    y = np.asarray(y, float)
    if k <= 1 or k >= len(y): return y
    if k % 2 == 0: k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    c = np.convolve(ypad, np.ones(k)/k, mode="valid")
    return c

def _diff(y, t):
    y = np.asarray(y, float); t = np.asarray(t, float)
    out = np.full_like(y, np.nan, float)
    if len(y) >= 3:
        out[:] = np.gradient(y, t)
    return out

def _count_peaks_simple(t, y, thr, min_sep_s=0.10, smooth_k=5):
    """
    Count local maxima where y[i-1] < y[i] > y[i+1] AND |y[i]| >= thr.
    Debounce peaks that are closer than min_sep_s.
    Returns (n_peaks, peaks_per_s).
    """
    y = _mavg(y, smooth_k)
    n = 0
    last_t = -1e9
    for i in range(1, len(y)-1):
        if y[i] > y[i-1] and y[i] > y[i+1] and abs(y[i]) >= thr:
            if t[i] - last_t >= min_sep_s:
                n += 1
                last_t = t[i]
    dur = float(t[-1] - t[0]) if len(t) else np.nan
    return n, (n/dur if dur and dur>0 else 0.0)

def _burst_metrics(t, y, thr):
    """
    Treat |y|>thr as 'unstable'. Return (#bursts, frac_time, longest_s, area_above_thr).
    """
    t = np.asarray(t, float); y = np.abs(np.asarray(y, float))
    if len(t) < 2:
        return 0, np.nan, np.nan, np.nan
    mask = y > thr
    if not np.any(mask):
        return 0, 0.0, 0.0, 0.0

    # contiguous regions
    edges = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(edges == +1)
    ends   = np.flatnonzero(edges == -1) - 1

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
        tt = t[s:e+1]; yy = y[s:e+1] - thr
        dt = np.diff(tt, prepend=tt[0]); dt[0] = 0.0
        area += float(np.sum(np.maximum(yy, 0.0) * dt))

    frac = (time_above / T) if T > 0 else np.nan
    return n_bursts, frac, longest, area


if __name__ == "__main__":
    main()
