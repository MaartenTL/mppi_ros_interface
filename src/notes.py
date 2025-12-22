#!/usr/bin/env python3

import rospy
import threading
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from std_msgs.msg import Float32                # adjust if your control topics use a different type
from tf.transformations import euler_from_quaternion
from functions_for_controllers import find_s_of_closest_point_on_global_path, produce_track,produce_marker_array_rviz, produce_marker_rviz, steer_angle_2_command
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
import tf
import time
from collections import deque
import torch
from dart_dynamic_models import model_functions,load_SVGPModel_actuator_dynamics_analytic
import numpy as np
import sys
import importlib.resources
from dynamics import Kinematic_Bicycle, Dynamic_Bicycle, SVGP, RateAugmentedDynamics
import matplotlib.pyplot as plt


mf = model_functions()


if __name__ == "__main__":
    model = Kinematic_Bicycle(0.1, "cpu")

    xs = []
    ys = []

    for i in [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8, -1.0]:
        states, actions = model.step(
            torch.tensor([[0.0, 0.0, np.deg2rad(90), 1.0, 0.0, 0.0]]),
            torch.tensor([[0.5, i]]),
            0
        )

        x, y, yaw, vx, vy, w = states.unbind(dim=1)
        xs.append(x.item())
        ys.append(y.item())

        desired_steering_angle = mf.steering_2_steering_angle(
            i,
            mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        )

        delta_max = mf.steering_2_steering_angle_actual(1.0, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self,
                                                        mf.e_s_self)
        delta_min = mf.steering_2_steering_angle_actual(-1.0, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self,
                                                        mf.e_s_self)

        desired_steering_angle = np.clip(desired_steering_angle, delta_min, delta_max)

        transformed_steer = mf.steering_angle_2_steering_command(
            desired_steering_angle,
            mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self,
            -1,1
        )

        final_steer_angle = mf.steering_2_steering_angle_actual(
            transformed_steer,
            mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        )

        else_steer_angle = mf.steering_2_steering_angle_actual(
            i,
            mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        )

        print(f"steer: {i},steering angle: {desired_steering_angle:.4f} transformed steering: {transformed_steer:.4f}"
              f", angle after transform: {final_steer_angle:.4f}, else: {else_steer_angle:.4f}")


    plt.figure()
    plt.scatter(xs, ys)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.show()




