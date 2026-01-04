#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32, Int32
from dart_dynamic_models import model_functions
import numpy as np
import threading
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion

mf = model_functions()

# -------------------------
# Safety callback container
# -------------------------
class SafetyState:
    def __init__(self):
        self.safe = False
        self.t_start = None

    def callback_safety(self, msg: Float32):
        if msg.data == 1.0 and not self.safe:
            # Rising edge → start sinusoid at t = 0
            self.safe = True
            self.t_start = rospy.Time.now().to_sec()
            rospy.loginfo("Safety active → starting sinusoidal steering")
        elif msg.data != 1.0 and self.safe:
            # Falling edge → stop sinusoid
            self.safe = False
            self.t_start = None
            rospy.loginfo("Safety inactive → stopping sinusoidal steering")
            
            
class Velocity:
    def __init__(self):
        self.past_x_vicon   = np.zeros(5)
        self.past_y_vicon   = np.zeros(5)
        self.past_yaw_vicon = np.zeros(5)
        self.past_time_vicon= np.zeros(5)
        
        self._lock = threading.Lock()

    def _pose_cb(self, msg):
        with self._lock:

            # Extract position
            pos = msg.pose.pose.position
            # Extract yaw from the quaternion
            q = msg.pose.pose.orientation
            yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]

            self.past_x_vicon[:-1] = self.past_x_vicon[1:]
            self.past_y_vicon[:-1] = self.past_y_vicon[1:]
            self.past_yaw_vicon[:-1] = self.past_yaw_vicon[1:]
            self.past_time_vicon[:-1] = self.past_time_vicon[1:]

            # append newest
            self.past_x_vicon[-1] = pos.x
            self.past_y_vicon[-1] = pos.y
            self.past_yaw_vicon[-1] = yaw
            self.past_time_vicon[-1] = msg.header.stamp.to_sec()

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

            self.vx = vx_fd
            self.vy = vy_fd
            self.omega = omega_fd



def main():
    rospy.init_node("sinusoidal_steering_test")
    
    car_number = 4


    # Publishers
    pub_throttle = rospy.Publisher(f"/throttle_{car_number}", Float32, queue_size=1)
    pub_steering = rospy.Publisher(f"/steering_{car_number}", Float32, queue_size=1)
    lap_count = 0
    lap_pub = rospy.Publisher("lap_count", Int32, queue_size=1, latch=True)
    lap_pub.publish(1)
    
    
    
    vx_publisher = rospy.Publisher(f"/vx_{car_number}", Float32, queue_size=1)
    vy_publisher = rospy.Publisher(f"/vy_{car_number}", Float32, queue_size=1)
    w_publisher  = rospy.Publisher(f"/omega_{car_number}",   Float32, queue_size=1)

    safety = SafetyState()
    rospy.Subscriber("/safety_value", Float32, safety.callback_safety)
    
    velocity = Velocity()
    rospy.Subscriber("/vicon/jetracer"+str(car_number), PoseWithCovarianceStamped,velocity._pose_cb)

    # -------- Tunable parameters --------
    throttle_cmd = rospy.get_param("~throttle", 0.3)
    steer_amp    = rospy.get_param("~steer_amp", 1.0)
    steer_freq   = rospy.get_param("~steer_freq", 0.8)  # Hz
    rate_hz      = rospy.get_param("~rate", 10.0)
    # -----------------------------------

    rate = rospy.Rate(rate_hz)
    
    speed_up_time = 1.0

    while not rospy.is_shutdown():

        if safety.safe:
            # Time since safety became active
            t = rospy.Time.now().to_sec() - safety.t_start
            if t > speed_up_time:
                steering = steer_amp * np.sin(2.0 * np.pi * steer_freq * (t - speed_up_time))
            else:
                steering = 0.0
        else:
            steering = 0.0  # hold zero steering when unsafe

        # --- Your existing steering transformations ---
        desired_steering_angle = mf.steering_2_steering_angle(
            steering,
            mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        )

        delta_max = mf.steering_2_steering_angle_actual(
            1.0, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        )
        delta_min = mf.steering_2_steering_angle_actual(
            -1.0, mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self
        )

        desired_steering_angle = np.clip(
            desired_steering_angle, delta_min, delta_max
        )

        transformed_steer = mf.steering_angle_2_steering_command(
            desired_steering_angle,
            mf.a_s_self, mf.b_s_self, mf.c_s_self, mf.d_s_self, mf.e_s_self,
            -1, 1
        )

        pub_throttle.publish(Float32(throttle_cmd))
        pub_steering.publish(Float32(transformed_steer)) # steering, transformed_steer
        #vx_publisher.publish(velocity.vx)
        #vy_publisher.publish(velocity.vy)
        #w_publisher.publish(velocity.omega)

        rate.sleep()


if __name__ == "__main__":
    main()
