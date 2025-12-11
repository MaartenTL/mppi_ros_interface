#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32, Int32
import pandas as pd
import numpy as np

def load_csv(path):
    """
    Load columns t, throttle, steering from CSV using pandas.
    Returns three numpy arrays of equal length.
    """
    df = pd.read_csv(path)  # add sep='\t' if file is tab-separated

    required_cols = {'t', 'throttle', 'steering'}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(f"CSV missing one of required columns: {required_cols}, found: {df.columns}")

    t = df['t'].to_numpy(dtype=float)
    throttle = df['throttle'].to_numpy(dtype=float)
    steering = df['steering'].to_numpy(dtype=float)

    if t.size == 0:
        raise RuntimeError("CSV file is empty or contains no valid rows.")

    return t, throttle, steering

def main():
    rospy.init_node('throttle_steering_replayer')

    # Parameters
    csv_path       = rospy.get_param('~csv_path', '/home/maarten/Documents/Thesis/log_Dart/lab_mpc_straight_v15_simtuned_20251209_162136.csv')
    throttle_topic = rospy.get_param('~throttle_topic', '/throttle_1')
    steering_topic = rospy.get_param('~steering_topic', '/steering_1')
    time_scale     = rospy.get_param('~time_scale', 1.0)  # >1 = slower, <1 = faster

    rospy.loginfo(f"Loading commands from {csv_path}")
    t_arr, throttle_arr, steering_arr = load_csv(csv_path)
    n_samples = t_arr.shape[0]
    rospy.loginfo(f"Loaded {n_samples} command samples")

    throttle_pub = rospy.Publisher(throttle_topic, Float32, queue_size=10)
    steering_pub = rospy.Publisher(steering_topic, Float32, queue_size=10)
    lap_pub = rospy.Publisher("lap_count", Int32, queue_size=1, latch=True)

    lap_pub.publish(1)

    # Give publishers a moment to connect
    rospy.sleep(1.0)

    # Initial sample
    t_prev = float(t_arr[0])
    throttle_prev = float(throttle_arr[0])
    steering_prev = float(steering_arr[0])

    rospy.loginfo("Starting replay...")

    throttle_pub.publish(Float32(throttle_prev))
    steering_pub.publish(Float32(steering_prev))

    # Replay the rest with original timing
    for i in range(1, n_samples):
        if rospy.is_shutdown():
            break

        t_i = float(t_arr[i])
        throttle_i = float(throttle_arr[i])
        steering_i = float(steering_arr[i])

        dt = (t_i - t_prev) * time_scale
        if dt < 0:
            rospy.logwarn(f"Negative dt ({dt}) between sample {i-1} and {i}, clamping to 0.")
            dt = 0.0

        rospy.sleep(dt)

        throttle_pub.publish(Float32(throttle_i))
        steering_pub.publish(Float32(steering_i))

        t_prev = t_i

    rospy.loginfo("Finished replaying commands.")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
