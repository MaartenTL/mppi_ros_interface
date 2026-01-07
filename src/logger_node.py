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
from dynamics import Kinematic_Bicycle, Dynamic_Bicycle, RateAugmentedDynamics, DynLimRateAugmentedDynamics
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
        
        self.num_runs = 0

        self.run_idx = 0
        self.phase = "RUNNING"

        self.car = rospy.get_param("~car_number", 1)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.env = env
        if env == "sim":
            out = rospy.get_param("~outfile", f"/home/maarten/Documents/Thesis/log_Dart/1dart_log_car{self.car}_{timestamp}.csv")
        elif env == "lab":
            out = rospy.get_param("~outfile", f"/home/maarten/Documents/Thesis/log_Dart/1lab_log_car{self.car}_{timestamp}.csv")
            
            self.num_runs = 2
            
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
            "temperature": None, "eta": None,"covariance": None, "scale": None,
        }
        self.costs = {
            "lat_cost": 0.0, "lag_cost": 0.0, "heading_cost": 0.0,
            "speed_cost": 0.0, "vy_cost": 0.0, "omega_cost": 0.0,
            "poss_cost": 0.0,
        }

        rospy.Subscriber(f"/vicon/jetracer{self.car}", PoseWithCovarianceStamped, self.cb_pose, queue_size=50)
        rospy.Subscriber(f"/vx_{self.car}", Float32, self.cb_vx, queue_size=100)
        rospy.Subscriber(f"/vy_{self.car}", Float32, self.cb_vy, queue_size=100)
        rospy.Subscriber(f"/omega_{self.car}", Float32, self.cb_omega, queue_size=100)
        rospy.Subscriber(f"/throttle_{self.car}", Float32, self.cb_throttle, queue_size=100)
        rospy.Subscriber(f"/steering_{self.car}", Float32, self.cb_steer, queue_size=100)
        rospy.Subscriber(f"/comptime_{self.car}", Float32, self.cb_comp, queue_size=100)

        rospy.Subscriber(f"/vx_est_{self.car}", Float32, self.cb_vx_est, queue_size=100)
        rospy.Subscriber(f"/vy_est_{self.car}", Float32, self.cb_vy_est, queue_size=100)
        rospy.Subscriber(f"/omega_est_{self.car}", Float32, self.cb_omega_est, queue_size=100)

        rospy.Subscriber("/obstacles", MarkerArray, self.cb_obstacle, queue_size=1)

        rospy.Subscriber("/dyn_temp", Float32MultiArray, self.cb_dyn_temp, queue_size=1)

        rospy.Subscriber("/dyn_cov", Float32MultiArray, self.cb_dyn_cov, queue_size=1)

        self.meta = { "mppi": "unknown","mode": "unknown", "mppi_model": "unknown", "sim_model": "unknown", "track_choice": "unknown", "dt": float('nan'), "V_target": float('nan'), "vel_mode": "unknown", "obstacle": float('nan')}
        rospy.Subscriber("mppi_meta", String, self.cb_meta, queue_size=1)

        self._open_new_csv()

        self.max_laps = laps
        self.logging_enabled = True
        rospy.Subscriber("lap_count", Int32, self.cb_lap_count, queue_size=1)

        self.meta_received = False
        self.meta_written_once = False
        self.new_lap_time = False
        rospy.Subscriber("lap_time", Float32, self.cb_lap_time, queue_size=1)


        rospy.Subscriber("/mppi_action", Float32MultiArray, self.cb_action, queue_size=1)
        self.sim = SimulatorROS(self.car, 2)



        self.csvfile.flush()


        self.trigger_on_raw_controls = rospy.get_param("~trigger_on_raw_controls", False)

        if not self.event_driven:
            self.timer = rospy.Timer(rospy.Duration(1.0 / float(rate_hz)), lambda _: self.write_row())
            rospy.loginfo(f"[logger_node] Writing to {self.outfile} @ {rate_hz} Hz (timer)")
        else:
            self.timer = None
            rospy.loginfo(f"[logger_node] Writing to {self.outfile} on NEW CONTROL events")



    def _make_outfile(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Keep your existing naming scheme, just add run index
        prefix = "dart" if self.env == "sim" else "lab"
        return f"/home/maarten/Documents/Thesis/log_Dart/1{prefix}_log_car{self.car}_{timestamp}_run{self.run_idx:02d}.csv"

    def _open_new_csv(self):
        # Close previous file if open
        try:
            if hasattr(self, "csvfile") and self.csvfile:
                self.csvfile.flush()
                self.csvfile.close()
        except Exception:
            pass

        self.outfile = self._make_outfile()
        self.csvfile = open(self.outfile, "w", newline="")
        self.writer = csv.writer(self.csvfile)

        # Write header (copy your existing header exactly)
        self.writer.writerow([
            "mppi_config","mppi_type", "mppi_model", "sim_model", "track_choice", "dt","V_target", "max_laps","Vel_estimator","Use_obstacle",
            "weights", "lap_time", "total expected cost",
            "t", "x", "y", "yaw", "vx", "vy", "omega","vx est", "vy est", "omega est",
            "throttle", "steering","throttle rate", "steering rate", "speed", "beta",
            "comp_time","cmd_time", "lat cost", "lag cost", "heading cost", "speed cost", "vy cost", "omega cost","poss cost",
            "obstacle_x", "obstacle_y", "obstacle_yaw","obstacle_dist","temperature","eta","covariance","scale"
        ])
        self.csvfile.flush()
        rospy.loginfo(f"[logger_node] Opened new log file: {self.outfile}")

    def _reset_for_next_run(self):
        # Reset all “first row / meta / readiness” gating so the next run logs cleanly
        self.ready = False
        self.seen = {"pose": False, "throttle": False, "steering": False, "comp": False}

        self.meta_received = False
        self.meta_written_once = False
        self.new_lap_time = False
        self.logging_enabled = True
        self.phase = "RUNNING"

        self.last_cmd_time = 0.0
        self.data["expected_cost"] = 0.0
        self.data["lap_time"] = 0.0
        self.data["throttle_rate"] = 0.0
        self.data["steering_rate"] = 0.0
        self.data["temperature"] = None
        self.data["eta"] = None
        self.data["covariance"] = None
        self.data["scale"] = None

        # Optional: clear costs
        for k in self.costs:
            self.costs[k] = 0.0

    def cb_dyn_cov(self, msg):
        with self.lock:
            self.data["covariance"] = msg.data
            # self.data["scale"] = msg.data[1]

    def cb_dyn_temp(self, msg):
        with self.lock:
            self.data["temperature"] = msg.data[0]
            self.data["eta"] = msg.data[1]

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


                self.sim.throttle = self.data["throttle"]
                self.sim.steering = self.data["steering"]
                self.sim.publish_path(action)

                if CONFIG["mode"] == "s_mppi" or CONFIG["mode"] == "s_mppi_dyn_lim":
                    self.data["throttle_rate"] = action[0,0].item()
                    self.data["steering_rate"] = action[0,1].item()

                expected_cost, weight, lat_err, lag_err, head_err, speed_err, vy, omega, pos_err = self.obj.compute_expected_cost(self.sim.states)
                total_expected_cost = sum(expected_cost)
                self.data["expected_cost"] = total_expected_cost.item()

                # rospy.loginfo(f"[MPPI logger] lag error: {lag_err}")
                total_lat_cost = sum(weight.q_lat * lat_err ** 2)
                total_lag_cost = sum(weight.q_lag * lag_err ** 2)
                total_heading_cost = sum(weight.q_head * head_err ** 2)
                total_speed_cost = sum(weight.q_v * speed_err ** 2)
                total_vy_cost = sum(weight.q_vy * vy ** 2)
                total_omega_cost = sum(weight.q_omega * omega ** 2)
                total_poss_cost = sum(weight.q_pos * pos_err ** 2)

                self.costs["lat_cost"] = total_lat_cost.item()
                self.costs["lag_cost"] = total_lag_cost.item()
                self.costs["heading_cost"] = total_heading_cost.item()
                self.costs["speed_cost"] = total_speed_cost.item()
                self.costs["vy_cost"] = total_vy_cost.item()
                self.costs["omega_cost"] = total_omega_cost.item()
                self.costs["poss_cost"] = total_poss_cost.item()



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

            if CONFIG["mode"] == "s_mppi":
                dynamics = RateAugmentedDynamics(
                    base_dyn=base_dynamics,
                    dt=self.meta["dt"],
                    th_min=th_min,
                    th_max=th_max,
                    steer_min=steer_min,
                    steer_max=steer_max,
                    device=CONFIG["device"],
                )

                self.sim.mode = "s_mppi"

            elif  CONFIG["mode"] == "s_mppi_dyn_lim":
                dynamics = DynLimRateAugmentedDynamics(
                    base_dyn=base_dynamics,
                    dt=self.meta["dt"],
                    th_min=th_min, 
                    th_max=th_max,
                    steer_min=steer_min,
                    steer_max=steer_max,
                    device=CONFIG["device"],
                )

                self.sim.mode = "s_mppi"

            else:
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
            meta_vals = ["", "", "", "", "", "","","","",""]
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
            self.costs["speed_cost"], self.costs["vy_cost"], self.costs["omega_cost"],self.costs["poss_cost"],
            self.data["obstacle_x"], self.data["obstacle_y"], self.data["obstacle_yaw"],
            self.data["obstacle_dist"], self.data["temperature"], self.data["eta"],
            self.data["covariance"], self.data["scale"],
        ]
        self.writer.writerow(row)


        self.csvfile.flush()

        # if self.lap > self.max_laps and self.logging_enabled:
        #     self.logging_enabled = False
        #     rospy.loginfo(f"[logger_node] Received lap {self.lap} > {self.max_laps}. Stopping logging.")
        #     try:
        #         self.timer.shutdown()
        #     except Exception:
        #         pass
        #     try:
        #         self.csvfile.flush()
        #         self.csvfile.close()
        #     except Exception:
        #         pass

        if self.lap > self.max_laps and self.phase == "RUNNING":
            rospy.loginfo(f"[logger_node] Lap {self.lap} > {self.max_laps}. Rotating log and waiting for reset...")

            # Move to next run file immediately, but do NOT log until lap counter resets
            self.phase = "WAIT_RESET"
            self.logging_enabled = False

            self.run_idx += 1
            # Ensure outfile param does not pin you to one file forever:
            # If you launch with _outfile:=..., you likely want to ignore it after run 0.
            # Force new auto-named file for subsequent runs:
            rospy.set_param("~outfile", self._make_outfile())

            if self.run_idx > self.num_runs:
                self.logging_enabled = False
                rospy.loginfo(f"[logger_node] Number of runs exceeded {self.run_idx}. Stopping logging.")
                try:
                    self.timer.shutdown()
                except Exception:
                    pass
                try:
                    self.csvfile.flush()
                    self.csvfile.close()
                except Exception:
                    pass

            else:

                # rospy.sleep(2)
                self._open_new_csv()
                self._reset_for_next_run()
            return

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
