#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from path_track_definitions import generate_path_data

mpl.rcParams.update({
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35
})
BASE_DIR = "/home/maarten/Documents/Thesis/log_Dart"

def plot_and_save_per_run(df, label, save_dir, x_path, y_path):
    os.makedirs(save_dir, exist_ok=True)

    # 1) XY with track
    fig = plt.figure(figsize=(6,6))
    plt.plot(x_path, y_path, linestyle="--", linewidth=1, label="track")
    h, = plt.plot(df["x"], df["y"], label=label)
    plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title(f"Trajectory: {label}")
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_xy.png"), dpi=150)
    plt.close(fig)

    # 2) Speed & vy
    fig = plt.figure()
    plt.plot(df["time"], np.hypot(df["vx"], df["vy"]), label="speed")
    if "vy" in df: plt.plot(df["time"], df["vy"], label="vy")
    plt.xlabel("time [s]"); plt.ylabel("[m/s]"); plt.title(f"Speed & vy: {label}"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_speed_vy.png"), dpi=150)
    plt.close(fig)

    # 3) Yaw rate and sideslip
    fig = plt.figure()
    plt.plot(df["time"], df["omega"], label="omega [rad/s]")
    if "beta" in df: plt.plot(df["time"], np.rad2deg(df["beta"]), label="beta [deg]")
    plt.xlabel("time [s]"); plt.title(f"Yaw rate & sideslip: {label}"); plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{label}_omega_beta.png"), dpi=150)
    plt.close(fig)

    # 4) Controls
    if "throttle" in df and "steering" in df:
        fig = plt.figure()
        plt.plot(df["time"], df["throttle"], label="throttle")
        plt.plot(df["time"], df["steering"], label="steering")
        plt.xlabel("time [s]"); plt.ylabel("command"); plt.title(f"Controls: {label}"); plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{label}_controls.png"), dpi=150)
        plt.close(fig)

    # 5) Errors
    fig = plt.figure()
    plt.plot(df["time"], df["lat_err"], label="lat [m]")
    plt.plot(df["time"], df["lag_err"], label="lag [m]")
    plt.plot(df["time"], df["pos_err"], label="pos [m]")
    plt.xlabel("time [s]"); plt.ylabel("error [m]"); plt.title(f"Tracking errors: {label}"); plt.legend()
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
    sum_comp_sat = sum(df["comp_sat"])

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
    thresholds = [0.05, 0.10, 0.20]  # meters
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

    return {
        "file": csv_path, "mppi_model": df["mppi_model"][0], "sim_model": df["sim_model"][0],
        "track_choice": df["track_choice"][0], "dt": df["dt"][0],"comp_sat": sum_comp_sat,
        "duration_s": T, "samples": n, "mean_dt_s": mean_dt,
        "distance_m": dist, "speed_mean": speed_mean, "speed_max": speed_max,
        "lat_mean_abs_m": lat_mean_abs, "lat_rms_m": lat_rms, "lat_max_abs_m": lat_max_abs,
        "lag_mean_abs_m": lag_mean_abs, "lag_rms_m": lag_rms,
        "pos_mean_m": pos_mean, "pos_rms_m": pos_rms, "pos_max_m": pos_max,
        "int_abs_lat_m_s": int_abs_lat, "int_pos_m_s": int_pos,
        "abs_lat_per_s_m": abs_lat_per_s, "pos_err_per_s_m": pos_per_s,
        "tib_<5cm": tib[0.05], "tib_<10cm": tib[0.10], "tib_<20cm": tib[0.20],
        "thr_mean": thr_mean, "int_abs_thr": thr_l1,
        "ste_rms": ste_rms, # "ste_rate_energy": ste_rate_energy,
        "beta_mean_abs_deg": beta_mean_abs, "beta_max_deg": beta_max,
    }

def plot_one(df, label=None):

    # XY
    plt.figure("XY"); plt.plot(df["x"], df["y"], label=label);
    plt.axis("equal"); plt.xlabel("x [m]"); plt.ylabel("y [m]"); plt.title("Trajectory comparison")
    plt.legend()

    # Speed & vy
    plt.figure("Speed/vy")
    plt.plot(df["time"], np.hypot(df["vx"], df["vy"]), label=f"speed {label}" if label else "speed")
    if "vy" in df: plt.plot(df["time"], df["vy"], label=f"vy {label}" if label else "vy")
    plt.xlabel("time [s]"); plt.ylabel("[m/s]"); plt.title("Speed and Lateral Velocity"); plt.legend()
    # Yaw rate & beta
    plt.figure("Omega/Beta")
    plt.plot(df["time"], df["omega"], label=f"omega {label}" if label else "omega")
    if "beta" in df: plt.plot(df["time"], np.rad2deg(df["beta"]), label=f"beta_deg {label}" if label else "beta_deg")
    plt.xlabel("time [s]"); plt.ylabel("yaw rate / sideslip"); plt.title("Yaw rate and Sideslip"); plt.legend()
    # Controls
    if "throttle" in df and "steering" in df:
        plt.figure("Controls")
        plt.plot(df["time"], df["throttle"], label=f"throttle {label}" if label else "throttle")
        plt.plot(df["time"], df["steering"], label=f"steering {label}" if label else "steering")
        plt.xlabel("time [s]"); plt.ylabel("command"); plt.title("Control Inputs"); plt.legend()
    # Errors
    plt.figure("Errors")
    plt.plot(df["time"], df["lat_err"], label=f"lat {label}" if label else "lat")
    plt.plot(df["time"], df["lag_err"], label=f"lag {label}" if label else "lag")
    plt.plot(df["time"], df["pos_err"], label=f"pos {label}" if label else "pos")
    plt.xlabel("time [s]"); plt.ylabel("error [m]"); plt.title("Tracking error vs time"); plt.legend()

def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)

    df = df.dropna(subset=["t","x","y","yaw"]).copy()
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True,
                    help="One or more CSV *filenames* or absolute paths (extension optional).")
    ap.add_argument("--out", default="/home/maarten/Documents/Thesis/log_Dart/summaries/compare.csv", help="Summary CSV path (default: summaries/compare.csv).")
    ap.add_argument("--save-figs", default="/home/maarten/Documents/Thesis/log_Dart/summaries/figs", help="Folder to save figures (default: figs/).")
    args = ap.parse_args()


    summaries = []
    first = True

    for csv_path_i in args.csv:
        print(csv_path_i)
        if csv_path_i.lower().endswith(".csv"):
            return
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
        "file","mppi_model","sim_model","track_choice","dt","comp_sat",
        "duration_s","samples","mean_dt_s",
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

    # Print a compact view
    with pd.option_context("display.max_columns", None, "display.width", 160, "display.precision", 4):
        print("\n=== Run Summary ===")
        print(summ_df.to_string(index=False))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summ_df.to_csv(args.out, index=False)
    print(f"\nSaved summary to: {args.out}")

    os.makedirs(args.save_figs, exist_ok=True)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_figs, f"fig_{num}.png"), dpi=150)
    print(f"Saved figures to: {args.save_figs}")
    plt.show()

if __name__ == "__main__":
    main()
