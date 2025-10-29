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


        self.vx_publisher = rospy.Publisher("vx_est", Float32, queue_size=1)
        self.vy_publisher = rospy.Publisher("vy_est", Float32, queue_size=1)
        self.w_publisher  = rospy.Publisher("omega_est",   Float32, queue_size=1)

        self.vx_error = 0.0
        self.vy_error = 0.0
        self.omega_error = 0.0

        # extra publishers for direct comparison
        self.vx_publisher_fd = rospy.Publisher("vx_est_fd", Float32, queue_size=1)
        self.vy_publisher_fd = rospy.Publisher("vy_est_fd", Float32, queue_size=1)
        self.w_publisher_fd = rospy.Publisher("omega_est_fd", Float32, queue_size=1)

        self.vx_publisher_ls = rospy.Publisher("vx_est_ls", Float32, queue_size=1)
        self.vy_publisher_ls = rospy.Publisher("vy_est_ls", Float32, queue_size=1)
        self.w_publisher_ls = rospy.Publisher("omega_est_ls", Float32, queue_size=1)

        # --- α–β filter params (tuneable) ---
        self.ab_alpha = rospy.get_param("~ab_alpha", 0.2)  # position gain
        self.ab_beta = rospy.get_param("~ab_beta", 0.4)  # velocity gain (≈ 2*alpha is a good start)

        # filter state (world frame)
        self._ab_x = None
        self._ab_y = None
        self._ab_yaw_unwrapped = None
        self._ab_vx = 0.0
        self._ab_vy = 0.0
        self._ab_omega = 0.0
        self._ab_t_last = None

        # publishers for α–β
        self.vx_publisher_ab = rospy.Publisher("vx_est_ab", Float32, queue_size=1)
        self.vy_publisher_ab = rospy.Publisher("vy_est_ab", Float32, queue_size=1)
        self.w_publisher_ab = rospy.Publisher("omega_est_ab", Float32, queue_size=1)

        self.obs_pub = rospy.Publisher('obstacles', MarkerArray, queue_size=1, latch=True)

        # Example: one obstacle placed at a path position (s0) with lateral offset d0 (m)
        # You can also set absolute (cx, cy) below if you prefer.
        self.obst_s0 = 8.0  # metres along global path
        self.obst_d0 = -0.2  # lateral offset from centreline (+left, -right)
        self.obst_a, self.obst_b = 0.5, 0.25  # ellipse semi-axes (m): x'-axis=a, y'-axis=b
        self.obst_yaw = 0.0  # orientation of ellipse in world (rad)

        self.obst_margin = 0.05  # safety buffer added to "solid" interior

        self.obst_cy = 0.0
        self.obst_cx = 0.0

        self.x_vals_global_path = None
        self.y_vals_global_path = None
        self.s_vals_global_path = None

        self.vel_mode = None


    def _pose_cb(self, msg):
        with self._lock:
            self._latest_msg = msg

    def _vx_cb(self, msg):
        self._vx = msg.data

    def _vy_cb(self, msg):
        self._vy = msg.data

    def _omega_cb(self, msg):
        self._omega = msg.data

    def _ab_update(self, x_meas, y_meas, yaw_meas, t_now):
        """
        α–β filter on x(t), y(t), yaw(t) in WORLD frame.
        Returns body-frame (vx, vy) and omega at time t_now.
        """
        alpha, beta = self.ab_alpha, self.ab_beta

        # unwrap yaw measurement consistently
        if self._ab_yaw_unwrapped is None:
            yaw_unw = yaw_meas
        else:
            # incremental unwrap against last unwrapped
            dy = yaw_meas - (self._ab_yaw_unwrapped % (2 * np.pi))
            if dy > np.pi:  dy -= 2 * np.pi
            if dy < -np.pi:  dy += 2 * np.pi
            yaw_unw = self._ab_yaw_unwrapped + dy

        # init on first call
        if self._ab_x is None or self._ab_t_last is None:
            self._ab_x, self._ab_y = float(x_meas), float(y_meas)
            self._ab_yaw_unwrapped = float(yaw_unw)
            self._ab_vx = 0.0;
            self._ab_vy = 0.0;
            self._ab_omega = 0.0
            self._ab_t_last = float(t_now)
            # return zeros (no estimate yet)
            return 0.0, 0.0, 0.0

        dt = float(t_now - self._ab_t_last)
        if dt <= 1e-6 or dt > 0.5:  # guard weird stamps; reset if too large a gap
            self._ab_t_last = float(t_now)
            return self._ab_vx, self._ab_vy, self._ab_omega

        # ---- Predict
        x_pred = self._ab_x + self._ab_vx * dt
        y_pred = self._ab_y + self._ab_vy * dt
        yaw_pred = self._ab_yaw_unwrapped + self._ab_omega * dt

        # Residuals
        rx = float(x_meas) - x_pred
        ry = float(y_meas) - y_pred
        ryaw = float(yaw_unw) - yaw_pred

        # ---- Correct
        self._ab_x = x_pred + alpha * rx
        self._ab_y = y_pred + alpha * ry
        self._ab_yaw_unwrapped = yaw_pred + alpha * ryaw

        self._ab_vx = self._ab_vx + (beta / dt) * rx
        self._ab_vy = self._ab_vy + (beta / dt) * ry
        self._ab_omega = self._ab_omega + (beta / dt) * ryaw

        self._ab_t_last = float(t_now)

        # rotate to BODY using the current measured yaw (fastest, consistent per tick)
        c, s = np.cos(yaw_meas), np.sin(yaw_meas)
        vx_body = c * self._ab_vx + s * self._ab_vy
        vy_body = -s * self._ab_vx + c * self._ab_vy
        return vx_body, vy_body, self._ab_omega

    def ls_slope(self, val, tt):
        v0 = val.mean()
        vv = val - v0
        denom = (tt * tt).sum()
        # guard
        if denom <= 1e-9:
            return 0.0
        return float((tt * vv).sum() / denom)

    def vicon_pos_2_vel(self, x, y, yaw, time, stamp):
        # shift ring buffers
        self.past_x_vicon[:-1] = self.past_x_vicon[1:]
        self.past_y_vicon[:-1] = self.past_y_vicon[1:]
        self.past_yaw_vicon[:-1] = self.past_yaw_vicon[1:]
        self.past_time_vicon[:-1] = self.past_time_vicon[1:]

        # append newest
        self.past_x_vicon[-1] = x
        self.past_y_vicon[-1] = y
        self.past_yaw_vicon[-1] = yaw
        self.past_time_vicon[-1] = time

        # ----- FD (end-point) -----
        dt_fd = (self.past_time_vicon[-1] - self.past_time_vicon[0])
        if dt_fd <= 1e-9:
            vx_abs_fd, vy_abs_fd, omega_fd = 0.0, 0.0, 0.0
        else:
            vx_abs_fd = (self.past_x_vicon[-1] - self.past_x_vicon[0]) / dt_fd
            vy_abs_fd = (self.past_y_vicon[-1] - self.past_y_vicon[0]) / dt_fd

            # unwrap only for FD omega across the window:
            delta_yaw = self.past_yaw_vicon[-1] - self.past_yaw_vicon[0]
            if delta_yaw > np.pi:
                delta_yaw -= 2 * np.pi
            elif delta_yaw < -np.pi:
                delta_yaw += 2 * np.pi
            omega_fd = delta_yaw / dt_fd

        # rotate FD using current yaw (your original behaviour)
        vx_fd = vx_abs_fd * np.cos(yaw) + vy_abs_fd * np.sin(yaw)
        vy_fd = -vx_abs_fd * np.sin(yaw) + vy_abs_fd * np.cos(yaw)

        # ----- LS (Method A) -----
        yaw_unwrapped = np.unwrap(self.past_yaw_vicon)
        t0 = self.past_time_vicon.mean()
        tt = self.past_time_vicon - t0
        denom = float((tt * tt).sum())

        if denom <= 1e-9:
            vx_abs_ls, vy_abs_ls, omega_ls = 0.0, 0.0, 0.0
        else:
            def slope(arr):
                return float(((arr - arr.mean()) * tt).sum() / denom)

            vx_abs_ls = slope(self.past_x_vicon)
            vy_abs_ls = slope(self.past_y_vicon)
            omega_ls = slope(yaw_unwrapped)

        # use mean yaw over window for LS rotation
        yaw_avg = np.arctan2(np.sin(self.past_yaw_vicon).mean(),
                             np.cos(self.past_yaw_vicon).mean())
        c, s = np.cos(yaw_avg), np.sin(yaw_avg)
        vx_ls = c * vx_abs_ls + s * vy_abs_ls
        vy_ls = -s * vx_abs_ls + c * vy_abs_ls

        # # publish for comparison
        # self.vx_publisher_fd.publish(Float32(vx_fd))
        # self.vy_publisher_fd.publish(Float32(vy_fd))
        # self.w_publisher_fd.publish(Float32(omega_fd))
        #
        # self.vx_publisher_ls.publish(Float32(vx_ls))
        # self.vy_publisher_ls.publish(Float32(vy_ls))
        # self.w_publisher_ls.publish(Float32(omega_ls))

        # keep your original outputs driving MPPI (choose which to return):
        # return LS to actually test it in control loop, or FD to keep baseline stable.
        if self.vel_mode == "gt":
            return self._vx, self._vy, self._omega
        elif self.vel_mode == "fd":
            return vx_fd, vy_fd, omega_fd
        else:
            return vx_ls, vy_ls, omega_ls

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

        # vx = self._vx
        # vy = self._vy
        # omega = self._omega

        vx, vy , omega = self.vicon_pos_2_vel(pos.x, pos.y, yaw, msg.header.stamp.to_sec(), msg.header.stamp)

        # if abs(self._vx - vx) > self.vx_error:
        #     print(f"max vx error: {self._vx - vx}")
        #     self.vx_error = abs(self._vx - vx)
        #
        # if abs(self._vy - vy) > self.vy_error:
        #     print(f"vy error: {self._vy - vy}")
        #     self.vy_error = abs(self._vy - vy)
        #
        # if abs(self._omega - omega) > self.omega_error:
        #     print(f"omega error: {self._omega - omega}")
        #     self.omega_error = abs(self._omega - omega)

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

    def generate_track(self, x_vals_global_path, y_vals_global_path, s_vals_global_path):
        # produce and send out global path message to rviz, which contains information about the track (i.e. the global path)
        rgba = [219.0, 0.0, 204.0, 0.6]
        marker_type = 4

        self.x_vals_global_path = x_vals_global_path
        self.y_vals_global_path = y_vals_global_path
        self.s_vals_global_path = s_vals_global_path

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
        clear = Marker()
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

    def _lane_point_from_s_and_offset(self, s_query: float, d: float):
        # Interpolate (x(s), y(s)) at s_query and shift laterally by d using path normal
        s_max = self.s_vals_global_path[-1]
        s_q = np.mod(s_query, s_max)
        x = np.interp(s_q, self.s_vals_global_path, self.x_vals_global_path)
        y = np.interp(s_q, self.s_vals_global_path, self.y_vals_global_path)

        # approximate heading via gradients on the *global path arrays*
        # (good enough for placing static stuff)
        dx_ds = np.gradient(self.x_vals_global_path, self.s_vals_global_path)
        dy_ds = np.gradient(self.y_vals_global_path, self.s_vals_global_path)
        heading = np.arctan2(np.interp(s_q, self.s_vals_global_path, dy_ds),
                             np.interp(s_q, self.s_vals_global_path, dx_ds))
        nx, ny = -np.sin(heading), np.cos(heading)  # left normal
        return x + d * nx, y + d * ny

    def _publish_obstacles_rviz(self):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.type = Marker.CYLINDER  # flat cylinder as an ellipse footprint
        m.id = 1
        m.pose.position.x = float(self.obst_cx)
        m.pose.position.y = float(self.obst_cy)
        m.pose.position.z = 0.05

        # yaw -> quaternion (z-rotation)
        cy = np.cos(self.obst_yaw * 0.5)
        sy = np.sin(self.obst_yaw * 0.5)
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = 0.0
        m.pose.orientation.z = sy
        m.pose.orientation.w = cy

        m.scale.x = 2.0 * self.obst_a # X diameter
        m.scale.y = 2.0 * self.obst_b  # Y diameter
        m.scale.z = 0.1  # thin "puck"

        m.color.r, m.color.g, m.color.b, m.color.a = 0.9, 0.15, 0.15, 0.6
        m.lifetime = rospy.Duration(0)  # persistent
        ma.markers.append(m)
        self.obs_pub.publish(ma)

    def get_obstacle(self):

        self.obst_cx, self.obst_cy = self._lane_point_from_s_and_offset(self.obst_s0, self.obst_d0)

        self._publish_obstacles_rviz()

        c = torch.tensor([self.obst_cx, self.obst_cy], dtype=torch.float32)
        ca = float(self.obst_a)
        cb = float(self.obst_b)
        th = float(self.obst_yaw)

        R = torch.tensor([[np.cos(th), -np.sin(th)],
                          [np.sin(th), np.cos(th)]], dtype=torch.float32)
        D = torch.tensor([[1.0 / (ca * ca), 0.0],
                          [0.0, 1.0 / (cb * cb)]],
                          dtype=torch.float32)
        Q = R @ D @ R.T

        return Q, c









