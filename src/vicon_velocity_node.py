#!/usr/bin/env python3
import numpy as np
from collections import deque

import rospy
from nav_msgs.msg import Odometry              # assuming Vicon publishes Odometry
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseStamped      # optional if your Vicon uses PoseStamped
from tf.transformations import euler_from_quaternion

class ViconVelocityEstimator:
    def __init__(self):
        # --- params ---
        self.buffer_len = rospy.get_param("~buffer_len", 5)            # finite-difference window
        self.delay_compensation = rospy.get_param("~delay_comp", False)
        self.delay = rospy.get_param("~delay_sec", 0.05)               # seconds

        self.topic_in = rospy.get_param("~pose_topic", "/vicon/jetracer_1")  # Odometry by default

        # --- ring buffers (oldest -> newest) ---
        self.past_x_vicon   = deque(maxlen=self.buffer_len)
        self.past_y_vicon   = deque(maxlen=self.buffer_len)
        self.past_yaw_vicon = deque(maxlen=self.buffer_len)
        self.past_time_vicon= deque(maxlen=self.buffer_len)

        # public state
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        self.x_y_yaw_state = [0.0, 0.0, 0.0]
        self.pose_msg_time = rospy.Time(0)

        # --- pubs ---
        self.vx_pub = rospy.Publisher("vx_1", Float32, queue_size=1)
        self.vy_pub = rospy.Publisher("vy_1", Float32, queue_size=1)
        self.w_pub  = rospy.Publisher("omega_1",   Float32, queue_size=1)

        # --- sub ---
        # Choose correct message type:
        msg_type = Odometry if rospy.get_param("~vicon_is_odometry", True) else PoseStamped
        self.sub = rospy.Subscriber(self.topic_in, msg_type, self.vicon_subscriber_callback, queue_size=10)

        rospy.loginfo("ViconVelocityEstimator up. Subscribing to %s", self.topic_in)

    def vicon_subscriber_callback(self, msg):
        # --- extract pose / orientation for both Odometry and PoseStamped ---
        if isinstance(msg, Odometry):
            p = msg.pose.pose.position
            o = msg.pose.pose.orientation
            t = msg.header.stamp.to_sec()
        else:  # PoseStamped
            p = msg.pose.position
            o = msg.pose.orientation
            t = msg.header.stamp.to_sec()

        q = [o.x, o.y, o.z, o.w]
        roll, pitch, yaw = euler_from_quaternion(q)

        # --- push into buffers ---
        self.past_x_vicon.append(p.x)
        self.past_y_vicon.append(p.y)
        self.past_yaw_vicon.append(yaw)
        self.past_time_vicon.append(t)

        # need at least 2 samples (better: full buffer) to compute a derivative
        if len(self.past_time_vicon) < 2:
            return

        t0, t1 = self.past_time_vicon[0], self.past_time_vicon[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return  # avoid div-by-zero and junk

        # --- finite-difference over the window (robust to noise) ---
        vx_abs = (self.past_x_vicon[-1] - self.past_x_vicon[0]) / dt
        vy_abs = (self.past_y_vicon[-1] - self.past_y_vicon[0]) / dt

        # --- convert world->body frame (2D) ---
        cy, sy = np.cos(yaw), np.sin(yaw)
        self.vx =  +vx_abs * cy + vy_abs * sy
        self.vy =  -vx_abs * sy + vy_abs * cy

        # --- unwrap yaw difference for omega ---
        dyaw = self.past_yaw_vicon[-1] - self.past_yaw_vicon[0]
        if   dyaw >  np.pi: dyaw -= 2*np.pi
        elif dyaw < -np.pi: dyaw += 2*np.pi
        self.omega = dyaw / dt

        # --- optional delay compensation in world frame, then yaw ---
        if self.delay_compensation:
            self.x_y_yaw_state = [p.x + vx_abs*self.delay,
                                  p.y + vy_abs*self.delay,
                                  yaw + self.omega*self.delay]
        else:
            self.x_y_yaw_state = [p.x, p.y, yaw]

        self.pose_msg_time = msg.header.stamp


        print("test")
        # --- publish ---
        self.vx_pub.publish(Float32(self.vx))
        self.vy_pub.publish(Float32(self.vy))
        self.w_pub.publish(Float32(self.omega))

def main():
    rospy.init_node("vicon_velocity_node")
    ViconVelocityEstimator()
    rospy.spin()

if __name__ == "__main__":
    main()
