#!/usr/bin/env python3
import rospy
import yaml
import torch
import math
import os
import sys

from mppi_torch.mppi import MPPIPlanner
from simulator_ros import SimulatorROS
from functions_for_controllers import find_s_of_closest_point_on_global_path, produce_track,produce_marker_array_rviz, produce_marker_rviz, steer_angle_2_command
import numpy as np
from dynamics import OmnidirectionalPointRobotDynamics, Kinematic_Bicycle
from tf.transformations import euler_from_quaternion
from path_track_definitions import generate_track, generate_path_data

abs_path = os.path.dirname(os.path.abspath(__file__))

from visualization_msgs.msg import MarkerArray



class ROSObjective:
    def __init__(self, track_choice, N, dt, device="cpu"):

        self.s_vals_global_path,\
        self.x_vals_global_path,\
        self.y_vals_global_path,\
        self.s_4_local_path,\
        self.x_4_local_path,\
        self.y_4_local_path,\
        self.dx_ds, self.dy_ds, self.d2x_ds2, self.d2y_ds2,\
        self.k_vals_global_path,\
        self.k_4_local_path = generate_path_data(track_choice)

        self.V_target = 1.0  # target velocity
        self.q_lat   = 3.0
        self.q_lag   =  1.0
        self.q_head  =  1.0
        self.q_v     =  1.0

        self.q_u     = 1.0

        self.N = N
        self.time_horizon = N * dt
        self.dt = dt
        self.previous_path_index = 0  # initial index for closest point in global path
        self.n_points_kernelized = 41  # number of points in the kernelized path (41 for reference)

        self.nu = 2  # control dimension: [steering, throttle]
        self.nx = 5  # state dimension: [x, y, yaw, vx, vy]

        self.current_state = None  # will be set by the simulator

        self.device = device
        self.counter = 0  # to keep track of which reference point to use

        self.rviz_local_path_pub = rospy.Publisher(
            f'~rviz_local_path_{rospy.get_name()}', MarkerArray, queue_size=1
        )

    def wrap_angle(self, x: torch.Tensor) -> torch.Tensor:
        # wrap into [-π, π]
        return torch.atan2(torch.sin(x), torch.cos(x))

    def compute_running_cost(self, state: torch.Tensor):
        """
        state: Tensor[K,5] = [x, y, yaw, vx, vy]
        self.nav_goal: Tensor[4] = [ref_x, ref_y, ref_yaw, V_target]
        """
        # unpack
        x, y, yaw, vx, vy = state[:, 0], state[:, 1], state[:, 2], state[:, 3], state[:, 4]
        ref_x, ref_y, ref_yaw, V_target = self.get_current_goal()  # each a scalar tensor

        # errors
        dx = x - ref_x
        dy = y - ref_y

        lag_err = dx * torch.cos(ref_yaw) + dy * torch.sin(ref_yaw)
        lat_err = -dx * torch.sin(ref_yaw) + dy * torch.cos(ref_yaw)
        head_err = self.wrap_angle(yaw - ref_yaw)
        speed = torch.sqrt(vx ** 2 + vy ** 2)
        speed_err = speed - V_target



        cost = (self.q_lat * lat_err ** 2
                + self.q_lag * lag_err ** 2
                + self.q_head * head_err ** 2
                + self.q_v * speed_err ** 2)
        return cost


    def get_current_goal(self):
        x_vals_global_path = self.x_vals_global_path
        y_vals_global_path = self.y_vals_global_path
        s_vals_global_path = self.s_vals_global_path

        # produce Chebyshev coefficients that represent local path
        Ds_forward = 1.2 * self.V_target * self.time_horizon #  self.dtt * self.high_level_solver_generator_obj.N
        Ds_back = 0.0 # this is the length of the path that is behind the car
        estimated_ds = self.current_state[0,3] * self.dt


        s, self.current_path_index = find_s_of_closest_point_on_global_path(
            np.array([self.current_state[0,0].item(),self.current_state[0,1].item()]),
            s_vals_global_path, x_vals_global_path,
            y_vals_global_path, self.previous_path_index, estimated_ds)

        self.s = s
        # print(f"s: {s}")
        # print(f"path index: {self.current_path_index}")
        self.previous_path_index = self.current_path_index
        # ------ HIGH LEVEL SOLVER ------


        if self.counter == 0:
            # x_y_yaw_state = self.current_state[0, 0:3]  # [x, y, yaw]
            # labels_k, local_path_length = self.produce_ylabels_4_local_kernelized_path(s, Ds_back, Ds_forward)
            # pos_x_init_rot, pos_y_init_rot, yaw_init_rot, xyyaw_ref_path = self.relative_xyyaw_to_current_path(
            #     x_y_yaw_state)  # current car state relative to current path index
            #
            # self.X0 = self.produce_X0(self.V_target, local_path_length, labels_k)

            self.X0 = self.produce_X0_global(
                V_target=self.V_target,
                N=self.N,
                s_start=self.s
            )

        if self.counter == 0:
            Ds_forward = 1.2 * self.V_target * self.time_horizon
            Ds_back = 0.0

            mask = (self.s_4_local_path >= s - Ds_back) & \
                   (self.s_4_local_path <= s + Ds_forward)
            idxs = np.nonzero(mask)[0]

            local_x = self.x_4_local_path[idxs]
            local_y = self.y_4_local_path[idxs]

            # Use your existing produce_marker_array_rviz utility:
            rgba = [0.0, 1.0, 0.0, 0.8]  # bright green
            marker_type = 4  # sphere list or LINE_STRIP
            marray = produce_marker_array_rviz(local_x, local_y, rgba, marker_type)
            self.rviz_local_path_pub.publish(marray)


        objective = torch.tensor(self.X0[self.counter, :])

        self.counter = self.counter + 1
        return objective


    def produce_ylabels_4_local_kernelized_path(self,s,Ds_back,Ds_forward):
        #extract indexes of local path

        mask = (self.s_4_local_path >= s - Ds_back) & (self.s_4_local_path <= s + Ds_forward)
        local_path_length = Ds_back + Ds_forward
        # Extract the indexes where the condition is true
        indexes = np.where(mask)[0]
        s_data_points =  self.s_4_local_path[indexes]  # This will have the local path parametrized starting from 0
        k_data_points = self.k_4_local_path[indexes]

        # resample the data points to have a fixed number of points
        n = self.n_points_kernelized

        s_data_points_fit = np.linspace(s_data_points[0], s_data_points[-1], n)
        labels_k = np.interp(s_data_points_fit, s_data_points, k_data_points)
        return labels_k, local_path_length

    def relative_xyyaw_to_current_path(self, x_y_yaw_state):
        # evaluate current reference path and derivatives needed for initial conditions
        # find corresponding index for s on s_4_local path
        current_path_index_on_4_local_path = np.argmin(np.abs(self.s_4_local_path - self.s))
        x_ref_path = self.x_4_local_path[current_path_index_on_4_local_path]
        y_ref_path = self.y_4_local_path[current_path_index_on_4_local_path]
        dx_ds_ref_path = self.dx_ds[current_path_index_on_4_local_path]
        dy_ds_ref_path = self.dy_ds[current_path_index_on_4_local_path]
        # evaluate heading angle
        heading_angle_path = np.arctan2(dy_ds_ref_path, dx_ds_ref_path)

        # self.local_path_ref_x,self.local_path_ref_y, self.local_path_rot_angle
        # apply shift to the x y position of the car
        pos_x_0 = x_y_yaw_state[0].item() - x_ref_path
        pos_y_0 = x_y_yaw_state[1].item() - y_ref_path
        # now rotate to have the first point aligned with the x axis
        pos_x_init_rot = pos_x_0 * np.cos(heading_angle_path) + pos_y_0 * np.sin(heading_angle_path)
        pos_y_init_rot = -pos_x_0 * np.sin(heading_angle_path) + pos_y_0 * np.cos(heading_angle_path)

        # apply rotation to yaw
        yaw_init_rot = x_y_yaw_state[2].item() - heading_angle_path

        # this is needed to keep the yaw angle from going over 2 pi
        if yaw_init_rot > np.pi:
            yaw_init_rot -= 2 * np.pi

        elif yaw_init_rot < -np.pi:
            yaw_init_rot += 2 * np.pi

        #
        xyyaw_ref_path = [x_ref_path, y_ref_path, heading_angle_path]

        return pos_x_init_rot, pos_y_init_rot, yaw_init_rot, xyyaw_ref_path

    def produce_X0(self,V_target,local_path_length,labels_k_params):
        # Initial guess for state trajectory
        X0_array = np.zeros((self.N,4))
        # assign initial guess for the states by forward euler integration on th ereference path

        # refinement for first guess needs to be higher because the forward euler is a bit lame
        N_0 = 1000

        s_0_vec = np.linspace(0, 0 + V_target * 1.5, N_0+1)

        # interpolate to get kurvature values
        normalized_s_4_kernel_path = np.linspace(0.0, 1.0, self.n_points_kernelized)

        s_star_0 = s_0_vec / local_path_length # normalize s
        k_0_vals = np.interp(s_star_0, normalized_s_4_kernel_path, labels_k_params)
        x_ref_0 = np.zeros(N_0+1)
        y_ref_0 = np.zeros(N_0+1)
        ref_heading_0 = np.zeros(N_0+1)
        dt = self.time_horizon / N_0
        u_yaw_rate_0 = np.zeros(N_0+1)
        for i in range(1,N_0+1):
            x_ref_0[i] = x_ref_0[i-1] + V_target * dt * np.cos(ref_heading_0[i-1])
            y_ref_0[i] = y_ref_0[i-1] + V_target * dt * np.sin(ref_heading_0[i-1])
            ref_heading_0[i] = ref_heading_0[i-1] + k_0_vals[i-1] * V_target * dt

            u_yaw_rate_0[i-1] = (ref_heading_0[i] - ref_heading_0[i-1] )/ dt

        # now down sample to the N points
        s_0_vec = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), s_0_vec)
        x_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), x_ref_0)
        y_ref_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), y_ref_0)
        ref_heading_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), ref_heading_0)
        u_yaw_rate_0 = np.interp(np.linspace(0,1,self.N+1), np.linspace(0,1,N_0+1), u_yaw_rate_0)

        # assign values to the array
        #X0_array[:,0] = u_yaw_rate_0
        #X0_array[:,1] = np.zeros(self.N+1) # slack variable should be zero
        #X0_array[:,2] = x_ref_0
        #X0_array[:,3] = y_ref_0
        #X0_array[:,4] = ref_heading_0
        #X0_array[:,5] = s_0_vec
        #X0_array[:,6] = x_ref_0
        #X0_array[:,7] = y_ref_0
        #X0_array[:,8] = ref_heading_0
        X0_array[:, 0] = x_ref_0[1:13] #+ self.current_state[0, 0].item()  # x position
        X0_array[:, 1] = y_ref_0[1:13] - 2.2 #+ self.current_state[0, 1].item()  # y position
        X0_array[:, 2] = ref_heading_0[1:13]
        X0_array[:, 3] = V_target * np.ones(self.N)

        return X0_array

    def produce_X0_global(self, V_target, N, s_start):
        """
        Build a global‐frame reference of length N starting at arc‐length s_start.
        Returns an (N×4) array: [x, y, yaw, V_target].
        """
        # 1) decide the look‐ahead in meters
        total_horizon = self.time_horizon  # e.g. N*dt
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
        # # evaluate this for longitudinal controller coordination
        # # adding delay compensation by projecting the position of the robot into the future
        # # delay = 0.165  # [s]
        # # robot_position[0] = robot_position[0] + np.cos(robot_theta) * self.v * delay
        # # robot_position[1] = robot_position[1] + np.sin(robot_theta) * self.v * delay
        # # robot_theta = robot_theta + self.w * delay
        #
        # # measure the closest point on the global path, returning the respective s parameter and its index
        #
        # # update index along the path to know where to search in next iteration
        #
        # # x_closest_point = x_vals_global_path[self.current_path_index]
        # # y_closest_point = y_vals_global_path[self.current_path_index]
        #
        # # plot closest point on the reference path
        # # rgba = [255.0, 0.0, 0.0, 0.6]
        # # marker_type = 2
        # # scale = 0.05
        # # closest_point_message = produce_marker_rviz(x_closest_point, y_closest_point, rgba, marker_type, scale)
        # # self.rviz_closest_point_on_path.publish(closest_point_message)
        #
        # # ----------------------------------------
        # L = 0.175  # length of vehicle [m]
        # # if self.v > 1.0:
        # # 	look_ahead_dist = 1 + (self.v-1)*2
        # # else:
        # look_ahead_dist = 0.6  # look ahead distance on path [m]
        #
        # # account for path running out
        # if s + look_ahead_dist > s_vals_global_path[-1]:  # look ahead is beyond the path length
        #     look_ahead_s = s + look_ahead_dist - s_vals_global_path[-1]
        # else:
        #     look_ahead_s = s + look_ahead_dist
        #
        # Px = np.interp(look_ahead_s, s_vals_global_path, x_vals_global_path)
        # Py = np.interp(look_ahead_s, s_vals_global_path, y_vals_global_path)
        #
        # # dx = self.x_vals_global_path[self.current_path_index + 10] - self.x_vals_global_path[self.current_path_index]
        # # dy = self.y_vals_global_path[self.current_path_index + 10] - self.y_vals_global_path[self.current_path_index]
        # # ref_heading = np.arctan2(dy, dx)
        #
        # # objective = np.array([Px, Py, ref_heading[0], self.V_target])
        # # objective = torch.Tensor(objective) # faster when first numpy.array is made


if __name__ == "__main__":
    rospy.init_node("mppi_ros_node")
    car_number = rospy.get_param("~car_number", 1)

    # 1) Load your existing YAML config
    CONFIG = yaml.safe_load(open(f"{abs_path}/config.yaml"))
    cfg = CONFIG["mppi"]

    dynamics = Kinematic_Bicycle(
        dt=CONFIG["dt"], device=CONFIG["device"]
    )

    # 2) Create ROS “simulator”
    sim = SimulatorROS(car_number)

    # DEMO ARENA TRACK 8x14
    track_choice = "racetrack_vicon_2"

    #sim.s_vals_global_path = s_vals_global_path
    #sim.x_vals_global_path = x_vals_global_path
    #sim.y_vals_global_path = y_vals_global_path
    #sim.global_path_message = global_path_message

    sim.dynamics = dynamics

    # 3) Objective
    obj = ROSObjective(track_choice, CONFIG["mppi"]["horizon"], CONFIG["dt"], device="cpu")

    global_path_message = sim.generate_track(obj.x_vals_global_path, obj.y_vals_global_path)

    # 4) Planner
    planner = MPPIPlanner(
        cfg=cfg,
        nx=5,
        dynamics=dynamics.step,
        running_cost=obj.compute_running_cost,
    )

    rate = rospy.Rate(1/CONFIG["dt"])  #

    counter = 0
    global_path_message_rate = 5  # publish 1 every 5 control loops

    while not rospy.is_shutdown():
        # making sure that while waiting the actions are zero
        #sim.send_control(torch.tensor([0.0, 0.0]))
        #input("Press Enter to run the next MPPI iteration or ctrl-c to quit")

        state = sim.get_current_state()
        obj.current_state = state  # update the current state in the objective, which updates the goal
        obj.counter = 0

        if state is None:
            rospy.loginfo("Waiting for first /vicon/jetracer1 message…")
            rate.sleep()
            continue

        # 5) Compute MPPI action
        action = planner.command(state)
        # get the candidate trajectories
        rollouts = planner.states.detach().cpu().numpy()
        #print(f"perturbed_action: {planner.perturbed_action}")
        # publish them
        sim.publish_rollouts(rollouts)

        # publish entire control input
        sim.publish_path(action.detach().cpu())

        # 6) Send to vehicle
        sim.send_control(action[0])
        # this is just to republish global path message every now and then
        if counter > global_path_message_rate:
            sim.rviz_global_path_publisher.publish(global_path_message)
            counter = 0  # reset counter

        # update counter
        counter = counter + 1

        rate.sleep()

