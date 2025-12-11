#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import re

from path_track_definitions import generate_path_data

mpl.rcParams.update({
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35
})
BASE_DIR = "/home/maarten/Documents/Thesis/log_Dart"

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


def plot_and_save_per_run(df, label, save_dir, x_path, y_path):
    os.makedirs(save_dir, exist_ok=True)

    # 1) XY with track
    fig = plt.figure(figsize=(6,6))
    ax = plt.gca()
    plt.plot(x_path, y_path, linestyle="--", linewidth=1, label="track")
    h, = plt.plot(df["x"], df["y"], label=label)
    plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title(f"Trajectory: {label}")
    plt.legend()

    # Add arrows: first set limits so span is known
    fig.canvas.draw_idle()
    ax.relim(); ax.autoscale_view()
    print("pre test arrows")
    # three arrows on the **track** (grey)
    _add_direction_arrows(ax, x_path, y_path, num=3, color="0.5", frac_len=0.06, z=3)


    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_xy.png"), dpi=150)
    plt.close(fig)

    # 2) Speed & vy
    fig = plt.figure()
    plt.plot(df["time"], np.hypot(df["vx"], df["vy"]), label="speed")
    if "vy" in df: plt.plot(df["time"], df["vy"], label="vy")
    plt.xlabel("timestep"); plt.ylabel("[m/s]"); plt.title(f"Speed & vy: {label}"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_speed_vy.png"), dpi=150)
    plt.close(fig)

    # 3) Yaw rate and sideslip
    fig = plt.figure()
    plt.plot(df["time"], df["omega"], label="omega [rad/s]")
    plt.xlabel("timestep"); plt.ylabel("omega [rad/s]"); plt.title(f"Yaw rate {label}"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_omega.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(df["time"], np.rad2deg(df["beta"]), label="beta [deg]")
    plt.xlabel("timestep"); plt.ylabel("beta [deg]"), plt.title(f"Sideslip: {label}"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_beta.png"), dpi=150)
    plt.close(fig)

    # 4) Controls
    fig = plt.figure()
    plt.plot(df["time"], df["throttle"], label="throttle", drawstyle="steps-post")
    plt.xlabel("timestep"); plt.ylabel("throttle []"); plt.title(f"Throttle: {label}"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_controls.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(df["time"], df["steering"], label="steering", drawstyle="steps-post")
    plt.xlabel("timestep"); plt.ylabel("steering [deg]"); plt.title(f"Steering: {label}"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_controls.png"), dpi=150)
    plt.close(fig)

    # Rates
    if df["throttle rate"]:
        fig = plt.figure()
        plt.plot(df["time"], df["throttle rate"], label="throttle_rate", drawstyle="steps-post")
        plt.xlabel("timestep"); plt.ylabel("throttle rate []"); plt.title(f"Throttle Rate: {label}"); plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{label}_controls.png"), dpi=150)
        plt.close(fig)

    if df["steering rate"]:
        fig = plt.figure()
        plt.plot(df["time"], df["steering rate"], label="steering_rate", drawstyle="steps-post")
        plt.xlabel("timestep"); plt.ylabel("steering rate [deg/s]"); plt.title(f"Steering Rate: {label}"); plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{label}_controls.png"), dpi=150)
        plt.close(fig)

    # 5) Errors
    fig = plt.figure()
    plt.plot(df["time"], df["lat_err"], label="lat [m]")
    plt.xlabel("timestep"); plt.ylabel("error [m]"); plt.title("Lateral Error vs time"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_errors.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(df["time"], df["lag_err"], label="lag [m]")
    plt.xlabel("timestep"); plt.ylabel("error [m]"); plt.title("Lag Error vs time"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_errors.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(df["time"], df["pos_err"], label="pos [m]")
    plt.xlabel("timestep"); plt.ylabel("error [m]"); plt.title("Positional Error vs time"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_errors.png"), dpi=150)
    plt.close(fig)

    t = df["time"].to_numpy(float)
    dt = np.diff(t, prepend=t[0]); dt[0] = 0.0

    lat = df["lat_err"].to_numpy(float)
    pos = df["pos_err"].to_numpy(float)

    cum_abs_lat = np.cumsum(np.abs(lat) * dt)  # ∫|lat| dt
    cum_pos     = np.cumsum(pos * dt)          # ∫||pos|| dt
    cum_lat2    = np.cumsum((lat**2) * dt)     # ∫lat^2 dt (optional)

    plt.figure("Cumulative Error (abs)")
    plt.plot(t, cum_abs_lat, label=f"∫|lat| dt — {label}")
    plt.plot(t, cum_pos,     label=f"∫pos dt — {label}")
    plt.xlabel("timestep"); plt.ylabel("m·s"); plt.title("Cumulative Errors")
    plt.legend()

    # Optional RMS trend over time (nice for seeing convergence)
    rms_lat = np.sqrt(cum_lat2 / (t - t[0] + 1e-9))
    fig = plt.figure()
    plt.plot(t, rms_lat, label=f"RMS(lat) — {label}")
    plt.xlabel("timestep"); plt.ylabel("m"); plt.title("RMS Lateral Error over time"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_errors.png"), dpi=150)
    plt.close(fig)

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
    # dste = derive(ste, t)

    thr_mean = float(np.nanmean(thr))
    thr_l1   = float(area_l1(thr, t))
    ste_rms  = float(nanrms(ste))
    # ste_energy = float(area_l2(ste, t))       # ∫ steer^2 dt
    # ste_rate_energy = float(area_l2(dste, t)) # ∫ (d/dt steer)^2 dt

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
        steering = ste
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
        "ste_rms": ste_rms,  # "ste_rate_energy": ste_rate_energy,
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
    return out

def plot_one(df, label=None):

    # XY
    plt.figure("XY"); plt.plot(df["x"], df["y"], label=label);
    plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Trajectory comparison")
    plt.legend()

    # Speed & vy
    plt.figure("Speed (vx)")
    plt.plot(np.hypot(df["vx"], df["vy"]), label=f"speed {label}" if label else "speed")
    plt.xlabel("timestep"); plt.ylabel("[m/s]"); plt.title("Speed"); plt.legend()

    plt.figure("vy")
    plt.plot(df["vy"], label=f"vy {label}" if label else "vy")
    plt.xlabel("timestep"); plt.ylabel("[m/s]"); plt.title("Lateral Velocity"); plt.legend()

    # Yaw rate & beta
    plt.figure("Omega")
    plt.plot(df["omega"], label=f"omega {label}" if label else "omega")
    plt.xlabel("timestep"); plt.ylabel("yaw rate [rad/s]"); plt.title("Yaw rate"); plt.legend()

    plt.figure("Beta")
    plt.plot(np.rad2deg(df["beta"]), label=f"beta_deg {label}" if label else "beta_deg")
    plt.xlabel("timestep"); plt.ylabel("sideslip [deg/s]"); plt.title("Sideslip"); plt.legend()

    # Controls
    plt.figure("Throttle")
    plt.plot(df["throttle"], label=f"throttle {label}" if label else "throttle", drawstyle="steps-post")
    plt.xlabel("timestep"); plt.ylabel("throttle []"); plt.title("Throttle"); plt.legend()

    plt.figure("Steering")
    plt.plot(df["steering"], label=f"steering {label}" if label else "steering", drawstyle="steps-post")
    plt.xlabel("timestep"); plt.ylabel("steering [deg]"); plt.title("Steering"); plt.legend()

    # Rates
    if "throttle rate" in df.columns:
        plt.figure("Throttle Rate")
        plt.plot(df["throttle rate"], label=f"throttle_rate {label}" if label else "throttle_rate", drawstyle="steps-post")
        plt.xlabel("timestep"); plt.ylabel("throttle rate []"); plt.title("Throttle Rate"); plt.legend()

    if "steering rate" in df.columns:
        plt.figure("Steering Rate")
        plt.plot(df["steering rate"], label=f"steering_rate {label}" if label else "steering_rate", drawstyle="steps-post")
        plt.xlabel("timestep"); plt.ylabel("steering rate [deg/s]"); plt.title("Steering Rate"); plt.legend()

    # Errors
    plt.figure("Lateral Error")
    plt.plot(df["lat_err"], label=f"lat {label}" if label else "lat")
    plt.xlabel("timestep"); plt.ylabel("error [m]"); plt.title("Lateral Error vs time"); plt.legend()

    plt.figure("Lag Error")
    plt.plot(df["lag_err"], label=f"lag {label}" if label else "lag")
    plt.xlabel("timestep"); plt.ylabel("error [m]"); plt.title("Lag Error vs time"); plt.legend()

    plt.figure("Pos Error")
    plt.plot(df["pos_err"], label=f"pos {label}" if label else "pos")
    plt.xlabel("timestep"); plt.ylabel("error [m]"); plt.title("Positional Error vs time"); plt.legend()

    t = df["time"].to_numpy(float)
    dt = df["dt"][0]

    print(dt)

    lat = df["lat_err"].to_numpy(float)
    pos = df["pos_err"].to_numpy(float)

    cum_abs_lat = np.cumsum(np.abs(lat) * dt)  # ∫|lat| dt
    cum_pos     = np.cumsum(pos * dt)          # ∫||pos|| dt
    cum_lat2    = np.cumsum((lat**2) * dt)     # ∫lat^2 dt (optional)

    plt.figure("Cumulative Error (abs)")
    plt.plot(cum_abs_lat, label=f"∫|lat| dt — {label}")
    plt.plot(cum_pos,     label=f"∫pos dt — {label}")
    plt.xlabel("timestep"); plt.ylabel("m·s"); plt.title("Cumulative Errors")
    plt.legend()

    # Optional RMS trend over time (nice for seeing convergence)
    rms_lat = np.sqrt(cum_lat2 / (t - t[0] + 1e-9))
    plt.figure("RMS Lat (trend)")
    plt.plot(t, rms_lat, label=f"RMS(lat) — {label}")
    plt.xlabel("timestep"); plt.ylabel("m"); plt.title("RMS Lateral Error over time"); plt.legend()


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
        "thr_mean","int_abs_thr","ste_rms", # "ste_rate_energy",
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
