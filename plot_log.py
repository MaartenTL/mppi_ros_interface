#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import matplotlib as mpl
from path_track_definitions import generate_path_data
import numpy as np

mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["grid.alpha"] = 0.35

BASE_DIR = "/home/maarten/Documents/Thesis/log_Dart"

def nearest_path_indices(xs, ys, x_path, y_path):
    # Fast nearest-neighbour via coarse binning; for small paths, simple argmin is fine.
    xy_path = np.vstack([x_path, y_path]).T
    xy = np.vstack([xs, ys]).T
    # Simple and robust: argmin over all points (opt: KDTree if very long):
    idx = np.argmin(((xy[:, None, :] - xy_path[None, :, :])**2).sum(-1), axis=1)
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/home/maarten/Documents/Thesis/log_Dart/dart_log_car1_20250825_200657.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    # Drop rows until we have timestamps & pose
    df = df.dropna(subset=["t","x","y","yaw"])

    t0 = df["t"].iloc[0]
    df["time"] = df["t"] - t0
    df["speed"] = df["speed"].fillna(0.0)
    df["beta_deg"] = np.rad2deg(df["beta"].fillna(0.0))

    # 1) Load path data (use the same track_choice you run in MPPI)
    track_choice = "racetrack_vicon_2"  # <- keep in sync with run_mppi_ros.py
    (_, x_path, y_path,
     _, _, _,
     dx_ds, dy_ds, _, _,
     _, _) = generate_path_data(track_choice)

    # 2) For each sample find nearest path index
    idx = nearest_path_indices(df["x"].values, df["y"].values, x_path, y_path)

    # 3) Reference values at those indices
    x_ref = x_path[idx]
    y_ref = y_path[idx]
    psi_ref = np.arctan2(dy_ds[idx], dx_ds[idx])

    # 4) Errors
    dx = df["x"].values - x_ref
    dy = df["y"].values - y_ref
    lag_err = dx * np.cos(psi_ref) + dy * np.sin(psi_ref)
    lat_err = -dx * np.sin(psi_ref) + dy * np.cos(psi_ref)
    pos_err = np.sqrt(lag_err ** 2 + lat_err ** 2)

    df["lag_err"] = lag_err
    df["lat_err"] = lat_err
    df["pos_err"] = pos_err

    # 5) Cumulative & RMS style metrics
    t = df["time"].values
    dt = np.diff(t, prepend=t[0])  # first dt=0, fine for cumulative sum

    df["cum_abs_lat"] = np.cumsum(np.abs(lat_err) * dt)  # ∫|lat| dt
    df["cum_pos_err"] = np.cumsum(pos_err * dt)  # ∫||e|| dt

    # Optional RMS over time
    df["rms_pos_err"] = np.sqrt(np.cumsum(pos_err ** 2 * dt) / (t - t[0] + 1e-9))

    # —— Plots ——
    # 1) XY trajectory
    plt.figure()
    plt.plot(df["x"], df["y"])
    plt.axis("equal")
    plt.title("Trajectory (x,y)")
    plt.xlabel("x [m]"); plt.ylabel("y [m]")

    # 2) Speed & vy
    plt.figure()
    plt.plot(df["time"], df["speed"], label="speed")
    plt.plot(df["time"], df["vy"], label="vy")
    plt.xlabel("time [s]"); plt.ylabel("[m/s]")
    plt.title("Speed and Lateral Velocity")
    plt.legend()

    # 3) Yaw rate and sideslip
    plt.figure()
    plt.plot(df["time"], df["omega"], label="omega [rad/s]")
    plt.plot(df["time"], df["beta_deg"], label="beta [deg]")
    plt.xlabel("time [s]"); plt.ylabel("yaw rate / sideslip")
    plt.title("Yaw rate and Sideslip")
    plt.legend()

    # 4) Controls
    plt.figure()
    plt.plot(df["time"], df["throttle"], label="throttle")
    plt.plot(df["time"], df["steering"], label="steering")
    plt.xlabel("time [s]"); plt.ylabel("command")
    plt.title("Control Inputs")
    plt.legend()

    # Errors vs time
    plt.figure()
    plt.plot(df["time"], df["lat_err"], label="lateral error [m]")
    plt.plot(df["time"], df["lag_err"], label="lag error [m]")
    plt.plot(df["time"], df["pos_err"], label="pos error [m]")
    plt.xlabel("time [s]");
    plt.ylabel("error [m]")
    plt.title("Tracking error vs time");
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.35)

    # Cumulative metrics
    plt.figure()
    plt.plot(df["time"], df["cum_abs_lat"], label="∫ |lat| dt [m·s]")
    plt.plot(df["time"], df["cum_pos_err"], label="∫ ||pos err|| dt [m·s]")
    plt.xlabel("time [s]");
    plt.ylabel("cumulative")
    plt.title("Cumulative error");
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.35)

    # RMS (gives you a single trend that settles)
    plt.figure()
    plt.plot(df["time"], df["rms_pos_err"])
    plt.xlabel("time [s]");
    plt.ylabel("RMS pos error [m]")
    plt.title("RMS position error over time")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)\

    # plt.figure()
    # plt.plot(df["time"], df["lat_err"], label="actual lat err")
    # plt.plot(df["time"], df["sel_lat"], label="MPPI predicted lat err (t+1)")
    # plt.legend();
    # plt.xlabel("time [s]");
    # plt.ylabel("m");
    # plt.title("Actual vs MPPI-predicted lateral error");
    # plt.grid(True, which="both", linestyle="--", alpha=0.35)
    #
    # plt.figure()
    # plt.plot(df["time"], df["sel_cum_cost"], label="cum expected cost")
    # plt.plot(df["time"], df["cum_abs_lat"], label="∫|lat| dt")
    # plt.legend();
    # plt.title("Expected cost vs realised tracking integral");
    # plt.grid(True, which="both", linestyle="--", alpha=0.35)
    #
    # plt.figure()
    # plt.plot(df["time"], df["roll_lat_p50"], label="rollout p50 lat (t+1)")
    # plt.fill_between(df["time"], df["roll_lat_p10"], df["roll_lat_p90"], alpha=0.2, label="p10–p90")
    # plt.plot(df["time"], df["lat_err"], label="actual lat err", linewidth=1)
    # plt.legend();
    # plt.title("Rollout lateral error spread vs actual");
    # plt.grid(True, which="both", linestyle="--", alpha=0.35)
    #
    # plt.figure()
    # plt.plot(df["time"], df["roll_mean_cost"], label="mean rollout cost (t+1)")
    # plt.plot(df["time"], df["roll_min_cost"], label="min rollout cost")
    # plt.plot(df["time"], df["roll_max_cost"], label="max rollout cost")
    # plt.legend();
    # plt.title("Rollout cost stats at each control step");
    # plt.grid(True, which="both", linestyle="--", alpha=0.35)

    plt.show()

if __name__ == "__main__":
    main()
