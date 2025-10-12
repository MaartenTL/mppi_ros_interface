#!/usr/bin/env python3
import rospy
import yaml
import torch
import os
import time
import subprocess

from mppi_torch.mppi import MPPIPlanner
from simulator_ros import SimulatorROS
from functions_for_controllers import find_s_of_closest_point_on_global_path, produce_track,produce_marker_array_rviz, produce_marker_rviz, steer_angle_2_command
import numpy as np
from dynamics import Kinematic_Bicycle, Dynamic_Bicycle
from tf.transformations import euler_from_quaternion
from path_track_definitions import generate_track, generate_path_data
from dynamic_reconfigure.client import Client
from std_msgs.msg import String, Float32, MultiArrayDimension, Float32MultiArray, Int32
from collections import deque
import json

abs_path = os.path.dirname(os.path.abspath(__file__))

from visualization_msgs.msg import MarkerArray


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
        self.q_lat = 0.5 # 1.0
        self.q_lag = 1.5 # 1.0
        self.q_head = 0.25 # 1.0
        self.q_v = 0.1 # 0.0
        self.q_vy = 0.2 # 0.0
        self.q_omega = 0.0 # 0.0

        self.ter_q_lat = self.q_lat * 10.0 # 10.0
        self.ter_q_lag = self.q_lag * 10.0 # 2.0
        self.ter_q_head = self.q_head * 10.0 # 1.0
        self.ter_q_v = self.q_v * 10.0 # 1.0
        self.ter_q_vy = self.q_vy * 10.0
        self.ter_q_omega = self.q_omega * 10.0 #0.5

        self.q_u_throttle = 0.01
        self.q_u_steering = 0.01

        self.w_lane =  100.0  # 100.0  # weight for lane penalty


class ROSObjective:
    def __init__(self, track_choice, N, dt, V_target, config, device="cpu"):

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

        # self.max_laps = config["laps"]
        # self.stopped_log = False

        self.ref_yaw_cos = 0.0
        self.ref_yaw_sin = 0.0

        self.lane_width = 1.0 # 0.60  # m; you can overwrite via ROS param or GUI later
        self.lane_margin = 0.05  # m; inner safety buffer

        self.left_lane_pub = rospy.Publisher('left_lane', MarkerArray, queue_size=1)
        self.right_lane_pub = rospy.Publisher('right_lane', MarkerArray, queue_size=1)

    def _softplus_hinge(self, x: torch.Tensor, sharp=25.0):
        # ≈ max(0, x) but smooth; larger 'sharp' -> steeper wall
        return torch.log1p(torch.exp(sharp * x)) / sharp

    def terminal_costs(self, states: torch.Tensor, actions: torch.Tensor):

        x, y, yaw, vx, vy, omega = states[:, -1, 0], states[:, -1, 1], states[:, -1, 2], states[:, -1, 3], states[:, -1, 4], states[:, -1, 5]
        ref_x = self.X0[-1,0]
        ref_y = self.X0[-1,1]
        ref_yaw = self.X0[-1,2]
        V_target = self.X0[-1,3]

        dx = x - ref_x
        dy = y - ref_y

        lag_err = dx * self.ref_yaw_cos[-1] + dy * self.ref_yaw_sin[-1]
        lat_err = -dx * self.ref_yaw_sin[-1] + dy * self.ref_yaw_cos[-1]
        head_err = self.wrap_angle(yaw - ref_yaw)
        speed = torch.sqrt(vx ** 2 + vy ** 2)
        speed_err = speed - V_target

        terminal_cost = (self.weight.ter_q_lat * lat_err ** 2
                + self.weight.ter_q_lag * lag_err ** 2
                + self.weight.ter_q_head * head_err ** 2
                + self.weight.ter_q_v * speed_err ** 2
                + self.weight.ter_q_vy * vy ** 2
                + self.weight.ter_q_omega * omega ** 2)

        # ADD CONTROL INPUT COSTS

        half_w = self.lane_width * 0.5
        excess_T = torch.abs(lat_err) - (half_w - self.lane_margin)
        c_lane_T = torch.clamp((1.5 * self.weight.w_lane) * self._softplus_hinge(excess_T,), 0, 300)

        terminal_cost = terminal_cost + c_lane_T
        return terminal_cost

    def wrap_angle(self, x: torch.Tensor) -> torch.Tensor:
        # wrap into [-π, π]
        return torch.atan2(torch.sin(x), torch.cos(x))

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
        # ------ HIGH LEVEL SOLVER ------

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


        dx = torch.tensor(x) - ref_x
        dy = torch.tensor(y) - ref_y
        lag_err = dx * self.ref_yaw_cos + dy * self.ref_yaw_sin
        lat_err = -dx * self.ref_yaw_sin + dy * self.ref_yaw_cos
        head_err = self.wrap_angle(torch.tensor(yaw) - ref_yaw)

        speed = torch.sqrt(torch.tensor(vx) ** 2 + torch.tensor(vy) ** 2)
        speed_err = speed - V_target

        expected_cost = (self.weight.q_lat * lat_err ** 2
                + self.weight.q_lag * lag_err ** 2
                + self.weight.q_head * head_err ** 2
                + self.weight.q_v * speed_err ** 2
                + self.weight.q_vy * torch.Tensor(vy) ** 2
                + self.weight.q_omega * torch.Tensor(omega) ** 2)

        return expected_cost, self.weight, lat_err, lag_err, head_err, speed_err, torch.Tensor(vy), torch.Tensor(omega)

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

        cost = (self.weight.q_lat * lat_err ** 2
                + self.weight.q_lag * lag_err ** 2
                + self.weight.q_head * head_err ** 2
                + self.weight.q_v * speed_err ** 2
                + self.weight.q_vy * vy ** 2
                + self.weight.q_omega * omega ** 2)

        half_w = self.lane_width * 0.5
        excess = torch.abs(lat_err) - (half_w - self.lane_margin)

        c_lane = torch.clamp(self.weight.w_lane * self._softplus_hinge(excess), 0, 300)
        cost = cost + c_lane

        return cost


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

                self.lap_start_time = time.time()


                self.lap_count += 1
                self.lap_pub.publish(Int32(self.lap_count))
                rospy.loginfo(
                    f"[mppi_ros] Lap {self.lap_count} detected (index {prev_idx} → {self.current_path_index}).")

                # if self.lap_count > self.max_laps and not self.stopped_log:
                #     self.stopped_log = True
                #     kill_rosbag(rospy.get_param("~bag_node", "/bag_recorder"))
                #     rospy.loginfo("Maximum number of laps reached, rosbag recording stopped")




            # Obtain new reference track for this timestep of MPPI
            self.X0 = torch.tensor(self.produce_X0_global(
                V_target=self.V_target,
                N=self.N,
                s_start=self.s
            ), dtype= torch.float32, device = self.device)

            Ds_forward = 1.2 * self.V_target * self.time_horizon
            Ds_back = 0.0

            mask = (self.s_4_local_path >= self.s - Ds_back) & \
                   (self.s_4_local_path <= self.s + Ds_forward)
            idxs = np.nonzero(mask)[0]

            local_x = self.x_4_local_path[idxs]
            local_y = self.y_4_local_path[idxs]

            rgba = [0.0, 1.0, 0.0, 0.8]  # bright green
            marker_type = 4  # sphere list or LINE_STRIP
            marray = produce_marker_array_rviz(local_x, local_y, rgba, marker_type)
            self.rviz_local_path_pub.publish(marray)


            self.ref_yaw_cos = torch.cos(self.X0[:, 2])
            self.ref_yaw_sin = torch.sin(self.X0[:, 2])

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

        objective = self.X0[self.counter, :] # "cpu") # Faster on cpu
        self.counter += 1
        return objective

    def produce_X0_global(self, V_target, N, s_start):
        """
        Build a global‐frame reference of length N starting at arc‐length s_start.
        Returns an (N×4) array: [x, y, yaw, V_target].
        """
        # 1) decide the look‐ahead in meters
        total_horizon = self.V_target * self.time_horizon
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

def reset_sim(x=-1.5, y=-2.2, theta=0.0, model_choice=1, *,
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
        "mppi_model": dynamics.__class__.__name__,
        "sim_model": model_choice,
        "track_choice": track_choice,
        "dt": CONFIG["dt"],
        "mppi": CONFIG["mppi"],
        "V_target": obj.V_target,
    }
    meta_pub.publish(json.dumps(meta))

def kill_rosbag(name):
    rospy.logwarn("Stopping recorder by killing node: %s", name)
    try:
        subprocess.call(['rosnode', 'kill', name])  # sends SIGINT → bag closes cleanly
    except Exception as e:
        rospy.logerr("Failed to kill %s: %s", name, e)

if __name__ == "__main__":

    dist = StateDisturber()

    meta_pub = rospy.Publisher("mppi_meta", String, queue_size=1, latch=True)
    rospy.init_node("mppi_ros_node")
    car_number = rospy.get_param("~car_number", 1)

    comptime_publisher = rospy.Publisher('comptime_' + str(car_number), Float32, queue_size=1)
    action_publisher = rospy.Publisher('mppi_action', Float32MultiArray, queue_size=1)

    mppi_roll_pub = rospy.Publisher("mppi_rollouts", String, queue_size=10)
    cum_expected_cost = 0.0

    # 1) Load your existing YAML config
    CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))
    cfg = CONFIG["mppi"]

    if CONFIG["sim_model"] == "kinematic":
        model_choice = 1
    elif CONFIG["sim_model"] == "dynamic":
        model_choice = 2
    elif CONFIG["sim_model"] == "SVGP":
        model_choice = 3
    elif CONFIG["sim_model"] == "SVGP_wet":
        model_choice = 4

    # 2) Create ROS “simulator”
    sim = SimulatorROS(car_number)

    if CONFIG["mppi_model"] == "kinematic":
        dynamics = Kinematic_Bicycle(dt=CONFIG["dt"], device=CONFIG["device"])
    elif CONFIG["mppi_model"] == "dynamic":
        dynamics = Dynamic_Bicycle(dt=CONFIG["dt"], device=CONFIG["device"])

    track_choice = CONFIG["track"]

    #sim.s_vals_global_path = s_vals_global_path
    #sim.x_vals_global_path = x_vals_global_path
    #sim.y_vals_global_path = y_vals_global_path
    #sim.global_path_message = global_path_message

    sim.dynamics = dynamics
    sim.vizdynamics = dynamics


    # 3) Objective
    obj = ROSObjective(track_choice, CONFIG["mppi"]["horizon"], CONFIG["dt"], CONFIG["v_target"], CONFIG, device=CONFIG["device"])

    global_path_message = sim.generate_track(obj.x_vals_global_path, obj.y_vals_global_path)

    # 4) Planner
    planner = MPPIPlanner(
        cfg=cfg,
        nx=6,
        dynamics=dynamics.step,
        running_cost=obj.compute_running_cost,
    )

    planner.terminal_state_cost = obj.terminal_costs

    rate = rospy.Rate(1/CONFIG["dt"])  #

    counter = 0
    global_path_message_rate = 5  # publish 1 every 5 control loops

    # Reset the simulator
    dr_client = Client("/dart_simulator_node", timeout=2.0)

    reset_sim(x=-1.5, y=-2.2, theta=0.0, model_choice=model_choice)
    publish_meta()
    stopped = False

    # compile_mode = "reduce-overhead"
    # try:
    #     dynamics.step = torch.compile(dynamics.step, mode=compile_mode, fullgraph=True)
    #     obj.compute_running_cost = torch.compile(obj.compute_running_cost, mode=compile_mode, fullgraph=True)
    #     print("[compile] enabled:", compile_mode)
    # except Exception as e:
    #     print("[compile] disabled:", e)

    while not rospy.is_shutdown():

        # making sure that while waiting the actions are zero
        # sim.send_control(torch.tensor([0.0, 0.0]))
        # input("Press Enter to run the next MPPI iteration or ctrl-c to quit")

        state = sim.get_current_state(dynamics._device)

        if CONFIG["disturbance"] == 1:
            state_est = dist.disturb(state)
        else:
            state_est = state


        obj.current_state = state_est  # update the current state in the objective, which updates the goal
        obj.counter = 0

        if state is None:
            rospy.loginfo("Waiting for first /vicon/jetracer1 message…")
            rate.sleep()
            continue

        start_time = time.time()
        with torch.inference_mode(): # don't need to keep track of the gradients
            # 5) Compute MPPI action
            action = planner.command(state_est)

        elapsed_time = time.time() - start_time
        if elapsed_time > CONFIG["dt"]:

            print(f"MPPI computation time: {elapsed_time:.4f} seconds")

        if elapsed_time > 0.2:
            print("sending 0 as input due to high computation time (0.2 seconds)")
            sim.send_control(torch.tensor([0.0, 0.0]))

        else:
            # 6) Send to vehicle
            sim.send_control(action[0])

        # get the candidate trajectories NOT SHOWING THESE AS IT TAKES A LOT OF COMPUTATIONAL TIME
        rollouts = planner.states.detach()
        # publish them
        sim.publish_rollouts(rollouts)


        # print(f"Elapsed MPPI computation time: {elapsed_time}")
        comptime_publisher.publish(elapsed_time)

        action_msg = Float32MultiArray()
        action_msg.data = action.reshape(-1).tolist() # list(action.to(torch.float32))
        action_publisher.publish(action_msg)
        # publish entire control input
        # T = CONFIG["mppi"]["horizon"]
        # NU = 2
        # disp_action = np.array(action_msg.data, dtype=np.float32).reshape(T, NU)
        # sim.publish_path(action.detach())

        # this is just to republish global path message every now and then
        if counter > global_path_message_rate:
            sim.rviz_global_path_publisher.publish(global_path_message)
            counter = 0  # reset counter

        # update counter
        counter = counter + 1


        rate.sleep()


