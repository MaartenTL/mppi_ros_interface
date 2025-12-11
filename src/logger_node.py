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
from dynamics import Kinematic_Bicycle, Dynamic_Bicycle, RateAugmentedDynamics
from simulator_ros import SimulatorROS
from run_mppi_ros import ROSObjective
import yaml
import torch
from visualization_msgs.msg import MarkerArray


abs_path = os.path.dirname(os.path.abspath(__file__))

def quat_to_yaw(qx, qy, qz, qw):
    # yaw from quaternion
    siny_cosp = 2.0*(qw*qz + qx*qy)
    cosy_cosp = 1.0 - 2.0*(qy*qy + qz*qz)
    return math.atan2(siny_cosp, cosy_cosp)

class DARTLogger:
    def __init__(self, laps, env):
        self.seen = {"pose": False, "throttle": False, "steering": False, "comp": False}
        self.ready = False
        self.event_driven = True
        self.lap = 0.0
        self.total_expected_cost = 0.0
        self.last_cmd_time = 0.0

        car = rospy.get_param("~car_number", 1)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.env = env
        if env == "sim":
            out = rospy.get_param("~outfile", f"/home/maarten/Documents/Thesis/log_Dart/1dart_log_car{car}_{timestamp}.csv")
        elif env == "lab":
            out = rospy.get_param("~outfile", f"/home/maarten/Documents/Thesis/log_Dart/1lab_log_car{car}_{timestamp}.csv")
        rate_hz = rospy.get_param("~rate", 20) # equal to timestep from simulator

        self.lock = Lock()
        self.data = {
            "t": None, "x": None, "y": None, "yaw": None,
            "vx": None, "vy": None, "omega": None,
            "vx_est": None, "vy_est": None, "omega_est": None,
            "throttle": None, "steering": None,
            "comp_time": None, "cmd_time": None, "lap_time": 0.0,
            "max_laps": None, "expected_cost": 0.0,
            "throttle_rate": 0.0, "steering_rate": 0.0,
            "obstacle_x": None, "obstacle_y": None, "obstacle_yaw": None, "obstacle_dist": None,
        }
        self.costs = {
            "lat_cost": 0.0, "lag_cost": 0.0, "heading_cost": 0.0,
            "speed_cost": 0.0, "vy_cost": 0.0, "omega_cost": 0.0,
        }

        rospy.Subscriber(f"/vicon/jetracer{car}", PoseWithCovarianceStamped, self.cb_pose, queue_size=50)
        rospy.Subscriber(f"/vx_{car}", Float32, self.cb_vx, queue_size=100)
        rospy.Subscriber(f"/vy_{car}", Float32, self.cb_vy, queue_size=100)
        rospy.Subscriber(f"/omega_{car}", Float32, self.cb_omega, queue_size=100)
        rospy.Subscriber(f"/throttle_{car}", Float32, self.cb_throttle, queue_size=100)
        rospy.Subscriber(f"/steering_{car}", Float32, self.cb_steer, queue_size=100)
        rospy.Subscriber(f"/comptime_{car}", Float32, self.cb_comp, queue_size=100)

        rospy.Subscriber(f"/vx_est_{car}", Float32, self.cb_vx_est, queue_size=100)
        rospy.Subscriber(f"/vy_est_{car}", Float32, self.cb_vy_est, queue_size=100)
        rospy.Subscriber(f"/omega_est_{car}", Float32, self.cb_omega_est, queue_size=100)

        rospy.Subscriber("/obstacles", MarkerArray, self.cb_obstacle, queue_size=1)

        self.meta = { "mppi": "unknown","mode": "unknown", "mppi_model": "unknown", "sim_model": "unknown", "track_choice": "unknown", "dt": float('nan'), "V_target": float('nan'), "vel_mode": "unknown", "obstacle": float('nan')}
        rospy.Subscriber("mppi_meta", String, self.cb_meta, queue_size=1)

        self.outfile = out
        self.csvfile = open(self.outfile, "w", newline="")
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow([
            "mppi_config","mppi_model", "sim_model", "track_choice", "dt","V_target", "max_laps","Vel_estimator","Use_obstacle",
            "weights", "lap_time", "total expected cost",
            "t", "x", "y", "yaw", "vx", "vy", "omega","vx est", "vy est", "omega est",
            "throttle", "steering","throttle rate", "steering rate", "speed", "beta",
            "comp_time","cmd_time", "lat cost", "lag cost", "heading cost", "speed cost", "vy cost", "omega cost",
            "obstacle_x", "obstacle_y", "obstacle_yaw","obstacle_dist"
        ])

        self.max_laps = laps
        self.logging_enabled = True
        rospy.Subscriber("lap_count", Int32, self.cb_lap_count, queue_size=1)

        self.meta_received = False
        self.meta_written_once = False
        self.new_lap_time = False
        rospy.Subscriber("lap_time", Float32, self.cb_lap_time, queue_size=1)


        rospy.Subscriber("/mppi_action", Float32MultiArray, self.cb_action, queue_size=1)
        self.sim = SimulatorROS(car, 2)



        self.csvfile.flush()


        self.trigger_on_raw_controls = rospy.get_param("~trigger_on_raw_controls", False)

        if not self.event_driven:
            self.timer = rospy.Timer(rospy.Duration(1.0 / float(rate_hz)), lambda _: self.write_row())
            rospy.loginfo(f"[logger_node] Writing to {self.outfile} @ {rate_hz} Hz (timer)")
        else:
            self.timer = None
            rospy.loginfo(f"[logger_node] Writing to {self.outfile} on NEW CONTROL events")



    def cb_action(self, msg):
        # This function publishes the path generated by the proposed action
        with self.lock:

            now = rospy.get_time()
            if self.last_cmd_time is None:
                self.data["cmd_time"] = 0.0
            else:
                self.data["cmd_time"] = now - self.last_cmd_time
            self.last_cmd_time = now


            if self.meta_written_once:
                T = self.meta["mppi"]["horizon"] #self.meta.hor  # or read from a ROS param
                NU = 2
                action = np.array(msg.data, dtype=np.float32).reshape(T, NU)
                self.sim.x = self.data["x"]
                self.sim.y = self.data["y"]
                self.sim.yaw = self.data["yaw"]
                self.sim.vx = self.data["vx"]
                self.sim.vy = self.data["vy"]
                self.sim.omega = self.data["omega"]

                self.sim.publish_path(action)
                # self.data["throttle_rate"] = action[0,0].item()
                # self.data["steering_rate"] = action[0,1].item()

                # rospy.loginfo(f"[MPPI] control sequence: {action}")

                expected_cost, weight, lat_err, lag_err, head_err, speed_err, vy, omega = self.obj.compute_expected_cost(self.sim.states)
                total_expected_cost = sum(expected_cost)
                self.data["expected_cost"] = total_expected_cost.item()

                # rospy.loginfo(f"[MPPI logger] lag error: {lag_err}")
                total_lat_cost = sum(weight.q_lat * lat_err ** 2)
                total_lag_cost = sum(weight.q_lag * lag_err ** 2)
                total_heading_cost = sum(weight.q_head * head_err ** 2)
                total_speed_cost = sum(weight.q_v * speed_err ** 2)
                total_vy_cost = sum(weight.q_vy * vy ** 2)
                total_omega_cost = sum(weight.q_omega * omega ** 2)

                self.costs["lat_cost"] = total_lat_cost.item()
                self.costs["lag_cost"] = total_lag_cost.item()
                self.costs["heading_cost"] = total_heading_cost.item()
                self.costs["speed_cost"] = total_speed_cost.item()
                self.costs["vy_cost"] = total_vy_cost.item()
                self.costs["omega_cost"] = total_omega_cost.item()



            # OUTSIDE the lock: event-driven write
            if self.event_driven:
                # print("[MPPI] logging new row -------------------------------------------------")
                self.flush_row()


    def cb_lap_time(self, msg):
        with self.lock:
            self.data["lap_time"] = msg.data
            # rospy.loginfo(f"[logger_node] Lap time: {msg.data}")
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

    def cb_obstacle(self, msg: MarkerArray):
        with self.lock:
            if not msg.markers:
                return
            m = msg.markers[0]
            p = m.pose.position
            q = m.pose.orientation
            self.data["obstacle_x"] = p.x
            self.data["obstacle_y"] = p.y
            self.data["obstacle_yaw"] = quat_to_yaw(q.x, q.y, q.z, q.w)


    def cb_vx(self, msg):
        with self.lock: self.data["vx"] = float(msg.data)
    def cb_vy(self, msg):
        with self.lock: self.data["vy"] = float(msg.data)
    def cb_omega(self, msg):
        with self.lock: self.data["omega"] = float(msg.data)

    def cb_vx_est(self, msg):
        with self.lock: self.data["vx_est"] = float(msg.data)

    def cb_vy_est(self, msg):
        with self.lock: self.data["vy_est"] = float(msg.data)

    def cb_omega_est(self, msg):
        with self.lock: self.data["omega_est"] = float(msg.data)

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
            print("meta received")
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
            self.lap = int(msg.data)
        except Exception:
            self.lap = int(float(msg.data))


    def write_row(self):
        with self.lock:
            self.flush_row()

    def flush_row(self):
        if not self.logging_enabled:
            return
        if not self.ready:
            if self.lap >= 1.0:
                if self.seen["pose"] and (self.seen["throttle"] or self.seen["steering"]): # and self.seen["comp"]:
                    self.ready = True
                    print("start logging now -------------------------------------------------")
                else:
                    return
            else:
                return

        if self.meta_received and not self.meta_written_once:
            if self.env == "lab":
                sim_model = "lab"
            else:
                sim_model = self.meta["sim_model"]

            if sim_model == 1:
                sim_model = "kinematic"
            elif sim_model == 2:
                sim_model = "dynamic"
            elif sim_model == 3:
                sim_model = "SVGP"
            elif sim_model == 4:
                sim_model = "SVGP_slippery"



            if self.meta["mppi_model"] == "Kinematic_Bicycle":
                base_dynamics = Kinematic_Bicycle(self.meta["dt"], device="cpu")
            elif self.meta["mppi_model"] == "Dynamic_Bicycle":
                base_dynamics = Dynamic_Bicycle(self.meta["dt"], device="cpu")

            th_min = CONFIG["throttle_min"]
            th_max = CONFIG["throttle_max"]
            steer_min = CONFIG["steering_min"]
            steer_max = CONFIG["steering_max"]

            # dynamics = RateAugmentedDynamics(
            #     base_dyn=base_dynamics,
            #     dt=self.meta["dt"],
            #     th_min=th_min,
            #     th_max=th_max,
            #     steer_min=steer_min,
            #     steer_max=steer_max,
            #     device=CONFIG["device"],
            # )

            dynamics = base_dynamics

            self.sim.dynamics = dynamics

            # self.obj = ROSObjective(self.meta["track_choice"],self.meta["mppi"]["horizon"], self.meta["dt"], self.meta["V_target"],0,0, "cpu")
            self.obj = ROSObjective(self.meta["track_choice"], self.meta["mppi"]["horizon"], self.meta["dt"],self.meta["V_target"], 0, 0, "cpu")

            meta_vals = [
                self.meta["mppi"],
                self.meta["mode"],
                self.meta["mppi_model"], sim_model,
                self.meta["track_choice"], self.meta["dt"],
                self.meta["V_target"],self.max_laps,
                self.meta["vel_mode"],self.meta["obstacle"],
            ]
            weights = self.obj.weight.__dict__
            self.meta_written_once = True
        else:
            # write blanks after the first time
            meta_vals = ["", "", "", "", "", "","","",""]
            weights = None

        if self.new_lap_time:
            self.new_lap_time = False
        else:
            self.data["lap_time"] = 0.0

            # --- distance to obstacle ---
        if (self.data["x"] is not None and self.data["y"] is not None and
                self.data["obstacle_x"] is not None and self.data["obstacle_y"] is not None):
            self.data["obstacle_dist"] = math.hypot(
                self.data["x"] - self.data["obstacle_x"],
                self.data["y"] - self.data["obstacle_y"]
            )
        else:
            self.data["obstacle_dist"] = float('nan')

        row = [
            *meta_vals, weights, self.data["lap_time"],self.data["expected_cost"],
            self.data["t"], self.data["x"], self.data["y"], self.data["yaw"],
            self.data["vx"], self.data["vy"], self.data["omega"],
            self.data["vx_est"], self.data["vy_est"], self.data["omega_est"],
            self.data["throttle"], self.data["steering"],self.data["throttle_rate"], self.data["steering_rate"],
            math.hypot(self.data["vx"], self.data["vy"]),
            math.atan2(self.data["vy"], self.data["vx"]),
            self.data["comp_time"],self.data["cmd_time"],self.costs["lat_cost"],self.costs["lag_cost"], self.costs["heading_cost"],
            self.costs["speed_cost"], self.costs["vy_cost"], self.costs["omega_cost"],
            self.data["obstacle_x"], self.data["obstacle_y"], self.data["obstacle_yaw"],
            self.data["obstacle_dist"]
        ]
        self.writer.writerow(row)


        self.csvfile.flush()

        if self.lap > self.max_laps and self.logging_enabled:
            self.logging_enabled = False
            rospy.loginfo(f"[logger_node] Received lap {self.lap} > {self.max_laps}. Stopping logging.")
            try:
                self.timer.shutdown()
            except Exception:
                pass
            try:
                self.csvfile.flush()
                self.csvfile.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.csvfile.close()
        except:
            pass

if __name__ == "__main__":
    rospy.init_node("dart_logger_node")

    CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))
    laps = CONFIG["laps"]
    env = CONFIG["env"]
    DARTLogger(laps, env)
    rospy.spin()
