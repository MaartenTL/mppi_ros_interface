#!/usr/bin/env python3

import rospy
import threading
import torch
from geometry_msgs.msg import PoseWithCovarianceStamped, Point
from std_msgs.msg import Float32                # adjust if your control topics use a different type
from tf.transformations import euler_from_quaternion
from functions_for_controllers import find_s_of_closest_point_on_global_path, produce_track,produce_marker_array_rviz, produce_marker_rviz, steer_angle_2_command
import numpy as np
from visualization_msgs.msg import MarkerArray, Marker
import tf
import time
from collections import deque

class SimulatorROS:
    def __init__(self, car_number):
        # — pose listener —
        self._lock = threading.Lock()
        self._latest_msg = None
        rospy.Subscriber("/vicon/jetracer"+str(car_number),
                         PoseWithCovarianceStamped,
                         self._pose_cb)
        self.car_number = car_number

        # — control publishers —
        self.steer_pub    = rospy.Publisher(f"/steering_{car_number}",
                                            Float32, queue_size=1)
        self.throttle_pub = rospy.Publisher(f"/throttle_{car_number}",
                                            Float32, queue_size=1)



        self.rviz_global_path_publisher = rospy.Publisher('rviz_global_path_' + str(self.car_number), MarkerArray,
                                                          queue_size=10)
        self.tf_listener = tf.TransformListener()

        self.rviz_closest_point_on_path = rospy.Publisher('rviz_closest_point_on_path_' + str(self.car_number), Marker,
                                                          queue_size=10)

        self.w = 0.0
        self.v = 0.0
        self.previous_path_index = 0  # initial index for closest point in global path

        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0

        rospy.Subscriber(f"/vx_{car_number}", Float32, self._vx_cb)
        rospy.Subscriber(f"/vy_{car_number}", Float32, self._vy_cb)
        rospy.Subscriber(f"/omega_{car_number}", Float32, self._omega_cb)

        # Set velocity target
        # self.V_target = 0.5

        self.rviz_rollouts_pub = rospy.Publisher(
            f"/rviz_rollouts_{self.car_number}",
            MarkerArray,
            queue_size=1
        )
        self.rviz_controls_pub = rospy.Publisher(
            f"/rviz_controls_{self.car_number}",
            MarkerArray,
            queue_size=1
        )

        # Are set in the run_mppi_ros.py script
        self.dynamics = None
        self.vizdynamics = None



        self.buffer_len = rospy.get_param("~buffer_len", 5)            # finite-difference window
        self.delay_compensation = rospy.get_param("~delay_comp", False)
        self.delay = rospy.get_param("~delay_sec", 0.05)               # seconds

        # --- ring buffers (oldest -> newest) ---
        self.past_x_vicon   = np.zeros(5)
        self.past_y_vicon   = np.zeros(5)
        self.past_yaw_vicon = np.zeros(5)
        self.past_time_vicon= np.zeros(5)


        self.vx_publisher = rospy.Publisher("vx_1", Float32, queue_size=1)
        self.vy_publisher = rospy.Publisher("vy_1", Float32, queue_size=1)
        self.w_publisher  = rospy.Publisher("omega_1",   Float32, queue_size=1)

    def _pose_cb(self, msg):
        with self._lock:
            self._latest_msg = msg

    def _vx_cb(self, msg):
        self._vx = msg.data

    def _vy_cb(self, msg):
        self._vy = msg.data

    def _omega_cb(self, msg):
        self._omega = msg.data

    def vicon_pos_2_vel(self, x, y, yaw, time, stamp):



        # update past states
        # shift them back by 1 step and update the last value
        self.past_x_vicon[:-1] = self.past_x_vicon[1:]
        self.past_y_vicon[:-1] = self.past_y_vicon[1:]
        self.past_yaw_vicon[:-1] = self.past_yaw_vicon[1:]
        self.past_time_vicon[:-1] = self.past_time_vicon[1:]

        # add last entry
        self.past_x_vicon[-1] = x
        self.past_y_vicon[-1] = y
        self.past_yaw_vicon[-1] = yaw
        self.past_time_vicon[-1] = time

        # evalaute velocities using finite differences on last values

        vx_abs = (self.past_x_vicon[-1] - self.past_x_vicon[0]) / (self.past_time_vicon[-1] - self.past_time_vicon[0])
        vy_abs = (self.past_y_vicon[-1] - self.past_y_vicon[0]) / (self.past_time_vicon[-1] - self.past_time_vicon[0])

        # convert to body frame
        vx = +vx_abs * np.cos(yaw) + vy_abs * np.sin(yaw)
        vy = -vx_abs * np.sin(yaw) + vy_abs * np.cos(yaw)

        # unwrap past angles to avoid jumps when flipping from - pi to + pi
        delta_yaw = self.past_yaw_vicon[-1] - self.past_yaw_vicon[0]
        if delta_yaw > np.pi:
            delta_yaw -= 2 * np.pi
        elif delta_yaw < -np.pi:
            delta_yaw += 2 * np.pi
        omega = delta_yaw / (self.past_time_vicon[-1] - self.past_time_vicon[0])

        # update the pose
        # if delay compensation is used, forward propagate the current state into the future
        if self.delay_compensation:
            # print('delay=',self.delay)
            # determine absolute velocities
            x_y_yaw_state = [x + vx_abs * self.delay,
                                  y + vy_abs * self.delay,
                                  yaw + omega * self.delay]
        else:
            x_y_yaw_state = [x, y, yaw]

        pose_msg_time = stamp

        # publish velocity states for rviz
        self.vx_publisher.publish(Float32(vx))
        self.vy_publisher.publish(Float32(vy))
        self.w_publisher.publish(Float32(omega))

        return vx, vy, omega


    def get_current_state(self, device):
        """
        Returns a torch.Tensor of shape [1,3] = [x, y, yaw],
        or None if we haven't received a pose yet.
        """
        with self._lock:
            msg = self._latest_msg
        if msg is None:
            return None

        # Extract position
        pos = msg.pose.pose.position
        # Extract yaw from the quaternion
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

        vx, vy , omega = self.vicon_pos_2_vel(pos.x, pos.y, yaw, msg.header.stamp.to_sec(), msg.header.stamp)

        # Build a 1×6 tensor: [x, y, yaw, vx, vy, omega]
        state = torch.tensor([[pos.x, pos.y, yaw, vx, vy, omega]], dtype=torch.float32, device=device)
        return state


    def send_control(self, action):
        """
        action: torch.Tensor of shape (2,) or (nu,)
          action[0] → throttle
          action[1] → steering
        """
        # Convert to float

        throttle = float(action[0].item())
        steering = float(action[1].item())

        if throttle > 2.0:
            throttle = 2
        elif throttle < -2.0:
            throttle = -2

        if steering > 2.0:
            steering = 2
        elif steering < -2.0:
            steering = -2

        # Publish
        self.throttle_pub.publish(throttle)
        self.steer_pub.publish(steering)

        #rospy.loginfo(f"throttle={throttle:.3f}, steering={steering:.3f}")

    def generate_track(self, x_vals_global_path, y_vals_global_path):
        # produce and send out global path message to rviz, which contains information about the track (i.e. the global path)
        rgba = [219.0, 0.0, 204.0, 0.6]
        marker_type = 4
        global_path_message = produce_marker_array_rviz(x_vals_global_path, y_vals_global_path, rgba, marker_type)

        return global_path_message

    def publish_rollouts(self, rollouts: np.ndarray):
        """
        rollouts: shape (K, T, nx), where rollouts[k, t, 0:2] = [x, y]
        """
        ma = MarkerArray()
        now = rospy.Time.now()

        # Optional: clear old markers first
        #clear = Marker()
        #clear.action = Marker.DELETEALL
        #ma.markers.append(clear)

        K, T, _ = rollouts.shape
        for k in range(K):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = now
            m.ns = f"rollouts_{self.car_number}"
            m.id = k
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD

            # light-blue, see-through
            m.scale.x = 0.01
            m.color.r = 0.5
            m.color.g = 0.75
            m.color.b = 1.0
            m.color.a = 0.3

            # build the geometry_msgs/Point list
            pts = []
            for t in range(T):
                x, y = float(rollouts[k, t, 0]), float(rollouts[k, t, 1])
                p = Point(x=x, y=y, z=0.0)
                pts.append(p)
            m.points = pts

            ma.markers.append(m)

        self.rviz_rollouts_pub.publish(ma)



    def publish_path(self, U: np.array):
        """
        Visualize the full MPPI-mean control sequence by rolling out
        through the dynamics.step dynamics.

        U: shape (T,2) array of [throttle, steering].
        Requires self.dynamics(state, action, t) → (next_state, _).
        """
        # 1) Setup MarkerArray
        ma = MarkerArray()
        now = rospy.Time.now()
        clear = Marker();
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)

        # 2) Get the full starting state [x,y,yaw,vx,vy,w]
        s0 = self.get_current_state("cpu")  # torch.Tensor [1,5]

        state = s0

        T = U.shape[0]

        pts = []
        p = Point(x = state[0,0].item(), y = state[0,1].item(), z =0.0)
        pts.append(p)
        self.states = []
        for t in range(T):
            state, _ = self.vizdynamics.step(state, torch.tensor([[U[t, 0], U[t, 1]]], dtype=torch.float32), t)
            x = state[0,0].item()
            y = state[0,1].item()
            p = Point(x=x, y=y, z=0.0)
            pts.append(p)
            self.states.append(state)

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = now
        m.ns = f"control_path_{self.car_number}"
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD

        # visual style
        m.scale.x = 0.02  # line width
        m.color.r = 1.0
        m.color.g = 0.5
        m.color.b = 0.0
        m.color.a = 1.0  # fully opaque

        m.points = pts
        ma.markers.append(m)

        # 3) publish
        self.rviz_controls_pub.publish(ma)








