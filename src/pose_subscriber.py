#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import threading

class PoseListener:
    def __init__(self):
        # Subscribe to the Vicon pose stream
        topic_name = "/vicon/jetracer1"
        rospy.Subscriber(topic_name,
                         PoseWithCovarianceStamped,
                         self.pose_callback)
        self._lock = threading.Lock()
        self.current_pose = None  # will hold the last received msg.pose

    def pose_callback(self, msg):
        # Save the latest pose (position + orientation)
        with self._lock:
            self.current_pose = msg.pose
        # Log once per second
        rospy.loginfo_throttle(1.0,
            f"[PoseListener] x={msg.pose.pose.position.x:.3f} "
            f"y={msg.pose.pose.position.y:.3f} "
            f"z={msg.pose.pose.position.z:.3f}")

    def get_pose(self):
        # Thread‐safe getter; returns None until first message arrives
        with self._lock:
            return self.current_pose

if __name__ == "__main__":
    rospy.init_node("pose_listener_node", anonymous=True)
    listener = PoseListener()
    rospy.loginfo("PoseListener started, waiting for /vicon/jetracer1 …")
    rospy.spin()