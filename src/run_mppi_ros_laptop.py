#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from dart_dynamic_models import model_functions
import numpy as np

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


def main():
    rospy.init_node("sinusoidal_steering_test")

    # Publishers
    pub_throttle = rospy.Publisher("/throttle_1", Float32, queue_size=1)
    pub_steering = rospy.Publisher("/steering_1", Float32, queue_size=1)

    safety = SafetyState()
    rospy.Subscriber("/safety_value", Float32, safety.callback_safety)

    # -------- Tunable parameters --------
    throttle_cmd = rospy.get_param("~throttle", 0.3)
    steer_amp    = rospy.get_param("~steer_amp", 0.5)
    steer_freq   = rospy.get_param("~steer_freq", 0.5)  # Hz
    rate_hz      = rospy.get_param("~rate", 10.0)
    # -----------------------------------

    rate = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():

        if safety.safe:
            # Time since safety became active
            t = rospy.Time.now().to_sec() - safety.t_start
            if t > 1.0:
                steering = steer_amp * np.sin(2.0 * np.pi * steer_freq * (t - 1.0))
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
        pub_steering.publish(Float32(transformed_steer))

        rate.sleep()


if __name__ == "__main__":
    main()
