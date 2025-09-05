#!/usr/bin/env python3
import rospy
import yaml
import torch
import os
import time

from mppi_torch.mppi import MPPIPlanner
from simulator_ros import SimulatorROS
from functions_for_controllers import find_s_of_closest_point_on_global_path, produce_track,produce_marker_array_rviz, produce_marker_rviz, steer_angle_2_command
import numpy as np
from dynamics import Kinematic_Bicycle, Dynamic_Bicycle
from tf.transformations import euler_from_quaternion
from path_track_definitions import generate_track, generate_path_data
from dynamic_reconfigure.client import Client
from std_msgs.msg import String, Float32, MultiArrayDimension, Float32MultiArray, Int32
import json


abs_path = os.path.dirname(os.path.abspath(__file__))

from visualization_msgs.msg import MarkerArray

class MPPIWeights:
    def __init__(self):
        self.q_lat = 5.0
        self.q_lag = 0.2
        self.q_head = 0.1
        self.q_v = 0.1
        self.q_vy = 0.0
        self.q_omega = 0.0

        self.q_u_throttle = 1.0
        self.q_u_steering = 1.0

class ROSObjective:
    def __init__(self, track_choice, N, dt, V_target, device="cpu"):

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

        self.ref_yaw_cos = 0.0
        self.ref_yaw_sin = 0.0


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

if __name__ == "__main__":
    meta_pub = rospy.Publisher("mppi_meta", String, queue_size=1, latch=True)
    rospy.init_node("mppi_ros_node")
    car_number = rospy.get_param("~car_number", 1)

    comptime_publisher = rospy.Publisher('comptime_' + str(car_number), Float32, queue_size=1)
    action_publisher = rospy.Publisher('mppi_action', Float32MultiArray, queue_size=1)

    # mppi_roll_pub = rospy.Publisher("mppi_rollouts", String, queue_size=10)
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

    # DEMO ARENA TRACK 8x14
    track_choice = "racetrack_vicon_2"

    #sim.s_vals_global_path = s_vals_global_path
    #sim.x_vals_global_path = x_vals_global_path
    #sim.y_vals_global_path = y_vals_global_path
    #sim.global_path_message = global_path_message

    sim.dynamics = dynamics


    # 3) Objective
    obj = ROSObjective(track_choice, CONFIG["mppi"]["horizon"], CONFIG["dt"], CONFIG["v_target"], device=CONFIG["device"])

    global_path_message = sim.generate_track(obj.x_vals_global_path, obj.y_vals_global_path)

    # 4) Planner
    planner = MPPIPlanner(
        cfg=cfg,
        nx=6,
        dynamics=dynamics.step,
        running_cost=obj.compute_running_cost,
    )

    rate = rospy.Rate(1/CONFIG["dt"])  #

    counter = 0
    global_path_message_rate = 5  # publish 1 every 5 control loops

    # Reset the simulator
    dr_client = Client("/dart_simulator_node", timeout=2.0)

    reset_sim(x=-1.5, y=-2.2, theta=0.0, model_choice=model_choice)
    publish_meta()

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
        obj.current_state = state  # update the current state in the objective, which updates the goal
        obj.counter = 0

        if state is None:
            rospy.loginfo("Waiting for first /vicon/jetracer1 message…")
            rate.sleep()
            continue

        start_time = time.time()
        with torch.inference_mode(): # don't need to keep track of the gradients
            # 5) Compute MPPI action
            action = planner.command(state)

        # 6) Send to vehicle
        sim.send_control(action[0])


        # # get the candidate trajectories NOT SHOWING THESE AS IT TAKES A LOT OF COMPUTATIONAL TIME
        # rollouts = planner.states.detach()
        # # publish them
        # sim.publish_rollouts(rollouts)

        action_msg = Float32MultiArray()
        action_msg.data = action.reshape(-1).tolist() # list(action.to(torch.float32))
        action_publisher.publish(action_msg)

        # publish entire control input (NOW DONE IN LOGGER NODE)
        # sim.publish_path(action.detach())

        # this is just to republish global path message every now and then
        if counter > global_path_message_rate:
            sim.rviz_global_path_publisher.publish(global_path_message)
            counter = 0  # reset counter

        # update counter
        counter = counter + 1

        elapsed_time = time.time() - start_time
        # print(f"Elapsed MPPI computation time: {elapsed_time}")
        comptime_publisher.publish(elapsed_time)
        if elapsed_time > 0.05:

            print(f"MPPI computation time: {elapsed_time:.4f} seconds")
        rate.sleep()

