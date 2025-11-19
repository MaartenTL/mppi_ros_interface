#!/usr/bin/env python3
import rospy
import rosbag
import os
from roslib.message import get_message_class
import time
from std_msgs.msg import String, Float32, MultiArrayDimension, Float32MultiArray, Int32

BAG_PATH = "/home/maarten/Documents/Thesis/log_Dart/ROSbag/run__2025-11-17-15-18-35_0.bag"
TOPIC_A = "/steering_1"
TOPIC_B = "/throttle_1"

PUBLISH_INTERVAL = 0.05   # 50 ms

def main():
    rospy.init_node("fixed_rate_republisher", anonymous=True)

    if not os.path.isfile(BAG_PATH):
        raise FileNotFoundError(f"Bag not found: {BAG_PATH}")

    bag = rosbag.Bag(BAG_PATH, 'r')

    meta = bag.get_type_and_topic_info().topics
    msg_type_a = get_message_class(meta[TOPIC_A].msg_type)
    msg_type_b = get_message_class(meta[TOPIC_B].msg_type)

    pub_a = rospy.Publisher(TOPIC_A, msg_type_a, queue_size=10)
    pub_b = rospy.Publisher(TOPIC_B, msg_type_b, queue_size=10)

    messages_a = (m for _, m, _ in bag.read_messages(topics=[TOPIC_A]))
    messages_b = (m for _, m, _ in bag.read_messages(topics=[TOPIC_B]))

    safety_value = rospy.Publisher('safety_value', Float32)
    rate = rospy.Rate(1.0 / PUBLISH_INTERVAL)

    try:
        for msg_a, msg_b in zip(messages_a, messages_b):

            # allow Ctrl+C to interrupt immediately
            if rospy.is_shutdown():
                break

            # update header timestamp if present
            if hasattr(msg_a, "header"):
                msg_a.header.stamp = rospy.Time.now()
            if hasattr(msg_b, "header"):
                msg_b.header.stamp = rospy.Time.now()

            safety_value.publish(1.0)
            pub_a.publish(msg_a)
            pub_b.publish(msg_b)
            rate.sleep()

    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user (Ctrl+C). Stopping republishing...")

    finally:
        bag.close()
        rospy.loginfo("Bag closed. Node shutting down.")


if __name__ == "__main__":
    main()