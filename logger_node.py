#!/usr/bin/env python3
import rospy
import csv
import os
from geometry_msgs.msg import PoseWithCovarianceStamped
import math
from threading import Lock
import datetime
from std_msgs.msg import Float32, Float32MultiArray, Int32, String
import json
import numpy as np

def quat_to_yaw(qx, qy, qz, qw):
    # yaw from quaternion
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

class DARTLogger:
    def __init__(self):
        self.seen = {"pose": False, "throttle": False, "steering": False, "comp": False}
        self.ready = False

        car = rospy.get_param("~car_number", 1)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = rospy.get_param("~outfile", f"/home/maarten/Documents/Thesis/log_Dart/dart_log_car{car}_{timestamp}.csv")
        rate_hz = rospy.get_param("~rate", 20) # equal to timestep from simulator

        self.lock = Lock()
        self.data = {
            "t": None, "x": None, "y": None, "yaw": None,
            "vx": None, "vy": None, "omega": None,
            "throttle": None, "steering": None,
            "comp_time": None, "lap_time": 0.0,
            "max_laps": None,
        }

        rospy.Subscriber(f"/vicon/jetracer{car}", PoseWithCovarianceStamped, self.cb_pose, queue_size=50)
        rospy.Subscriber(f"/vx_{car}", Float32, self.cb_vx, queue_size=100)
        rospy.Subscriber(f"/vy_{car}", Float32, self.cb_vy, queue_size=100)
        rospy.Subscriber(f"/omega_{car}", Float32, self.cb_omega, queue_size=100)
        rospy.Subscriber(f"/throttle_{car}", Float32, self.cb_throttle, queue_size=100)
        rospy.Subscriber(f"/steering_{car}", Float32, self.cb_steer, queue_size=100)
        rospy.Subscriber(f"/comptime_{car}", Float32, self.cb_comp, queue_size=100)


        self.meta = { "mppi": "unknown", "mppi_model": "unknown", "sim_model": "unknown", "track_choice": "unknown", "dt": float('nan'), "V_target": float('nan')}
        rospy.Subscriber("mppi_meta", String, self.cb_meta, queue_size=1)

        self.outfile = out
        self.csvfile = open(self.outfile, "w", newline="")
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow([
            "mppi_config","mppi_model", "sim_model", "track_choice", "dt","V_target","max_laps", "lap_time",
            "t", "x", "y", "yaw", "vx", "vy", "omega", "throttle", "steering", "speed", "beta",
            "comp_time",
            # # selected rollout prediction:
            # "sel_ref_x", "sel_ref_y", "sel_ref_yaw",
            # "sel_x", "sel_y", "sel_yaw", "sel_vx", "sel_vy", "sel_omega",
            # "sel_lat", "sel_lag", "sel_speed_err", "sel_cost", "sel_cum_cost",
            # # rollout stats at t=0 across K:
            # "roll_mean_cost", "roll_min_cost", "roll_max_cost",
            # "roll_lat_mean", "roll_lat_p10", "roll_lat_p50", "roll_lat_p90",
            # "roll_lag_mean", "roll_lag_p10", "roll_lag_p50", "roll_lag_p90",
            # "roll_spderr_mean", "roll_spderr_p10", "roll_spderr_p50", "roll_spderr_p90",
            # "roll_vy_mean", "roll_vy_p10", "roll_vy_p50", "roll_vy_p90",
            # "roll_w_mean", "roll_w_p10", "roll_w_p50", "roll_w_p90"
        ])
        self.csvfile.flush()

        self.timer = rospy.Timer(rospy.Duration(1.0/float(rate_hz)), self.flush_row)
        rospy.loginfo(f"[logger_node] Writing to {self.outfile} @ {rate_hz} Hz")

        # rospy.Subscriber("mppi_debug/selected", Float32MultiArray, self.cb_mppi_selected, queue_size=100)
        # rospy.Subscriber("mppi_debug/rollouts", Float32MultiArray, self.cb_mppi_rollouts, queue_size=100)
        # # ...
        # self.data.update({
        #     "mppi_sel": None,
        #     "mppi_roll": None
        # })

        self.max_laps = rospy.get_param("~laps_to_log", 5)
        self.logging_enabled = True
        rospy.Subscriber("lap_count", Int32, self.cb_lap_count, queue_size=1)

        self.meta_received = False
        self.meta_written_once = False

        rospy.Subscriber("lap_time", Float32, self.cb_lap_time, queue_size=1)
        self.new_lap_time = False

    def cb_lap_time(self, msg):
        with self.lock:
            self.data["lap_time"] = msg.data
            rospy.loginfo(f"[logger_node] Lap time: {msg.data}")
            self.new_lap_time = True


    def cb_pose(self, msg):
        with self.lock:
            self.seen["pose"] = True
            self.data["t"] = msg.header.stamp.to_sec()
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self.data["x"] = p.x
            self.data["y"] = p.y
            self.data["yaw"] = quat_to_yaw(q.x, q.y, q.z, q.w)

    def cb_vx(self, msg):
        with self.lock: self.data["vx"] = float(msg.data)
    def cb_vy(self, msg):
        with self.lock: self.data["vy"] = float(msg.data)
    def cb_omega(self, msg):
        with self.lock: self.data["omega"] = float(msg.data)
    def cb_throttle(self, msg):
        with self.lock:
            self.seen["throttle"] = True
            self.data["throttle"] = float(msg.data)
    def cb_steer(self, msg):
        with self.lock:
            self.seen["steering"] = True
            self.data["steering"] = float(msg.data)

    def cb_comp(self, msg):
        with self.lock:
            self.seen["comp"] = True
            self.data["comp_time"] = float(msg.data)

    def cb_meta(self, msg):
        try:
            d = json.loads(msg.data)
            with self.lock:
                for k in self.meta:
                    if k in d:
                        self.meta[k] = d[k]
                self.meta_received = True
        except Exception as e:
            rospy.logwarn(f"[logger_node] meta parse failed: {e}")

    def cb_lap_count(self, msg):
        try:
            lap = int(msg.data)
        except Exception:
            lap = int(float(msg.data))
        if lap >= self.max_laps and self.logging_enabled:
            self.logging_enabled = False
            rospy.loginfo(f"[logger_node] Received lap {lap} ≥ {self.max_laps}. Stopping logging.")
            try:
                self.timer.shutdown()
            except Exception:
                pass
            try:
                self.csvfile.flush()
                self.csvfile.close()
            except Exception:
                pass

    def flush_row(self, _):
        with self.lock:

            if not self.logging_enabled:
                return

            if not self.ready:
                if self.seen["pose"] and (self.seen["throttle"] or self.seen["steering"]) and self.seen["comp"]:
                    self.ready = True
                else:
                    return

            if self.meta_received and not self.meta_written_once:
                sim_model = self.meta["sim_model"]

                if sim_model == 1:
                    sim_model = "kinematic"
                elif sim_model == 2:
                    sim_model = "dynamic"
                elif sim_model == 3:
                    sim_model = "SVGP"
                elif sim_model == 4:
                    sim_model = "SVGP_slippery"

                meta_vals = [

                    self.meta["mppi"],
                    self.meta["mppi_model"], sim_model,
                    self.meta["track_choice"], self.meta["dt"],
                    self.meta["V_target"],self.max_laps,
                ]
                self.meta_written_once = True
            else:
                # write blanks after the first time
                meta_vals = ["", "", "", "", "", "",""]

            if self.new_lap_time:
                self.new_lap_time = False
            else:
                self.data["lap_time"] = 0.0

            row = [
                *meta_vals, self.data["lap_time"],
                self.data["t"], self.data["x"], self.data["y"], self.data["yaw"],
                self.data["vx"], self.data["vy"], self.data["omega"],
                self.data["throttle"], self.data["steering"],math.hypot(self.data["vx"], self.data["vy"]),
                math.atan2(self.data["vy"], self.data["vx"]),
                self.data["comp_time"],
                #speed, beta,
                # selected block (skip t at index 0)
                # *sel_f[1:4],  # ref_x, ref_y, ref_yaw
                # *sel_f[4:10],  # predicted next state x..omega
                # *sel_f[10:15],  # lat, lag, speed_err, immediate cost, cum cost
                # # rollout stats (skip t at index 0):
                # *roll_f[1:]
            ]
            self.writer.writerow(row)
            self.csvfile.flush()

    # def cb_mppi_selected(self, msg):
    #     with self.lock:
    #         self.data["mppi_sel"] = list(msg.data)
    #
    # def cb_mppi_rollouts(self, msg):
    #     with self.lock:
    #         self.data["mppi_roll"] = list(msg.data)

    def __del__(self):
        try:
            self.csvfile.close()
        except:
            pass

if __name__ == "__main__":
    rospy.init_node("dart_logger_node")


    DARTLogger()
    rospy.spin()
