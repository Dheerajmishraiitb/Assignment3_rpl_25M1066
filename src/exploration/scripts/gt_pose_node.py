#!/usr/bin/env python3
"""Ground-truth pose out of Gazebo
"""
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

GT_FRAME = "map"


class GtPose(Node):
    def __init__(self):
        super().__init__("gt_pose_node")
        self.declare_parameter("max_rate_hz", 20.0)
      
        self.declare_parameter("world", "default")
        self.declare_parameter("model", "burger_cam")

        self.pub = self.create_publisher(PoseStamped, "/ground_truth/pose", 10)
        self.sub = self.create_subscription(Odometry, "/ground_truth/odom",
                                            self.on_odom, 20)
        self.n = 0
        self.n_in = 0
        self._last_pub = 0.0
        self.create_timer(5.0, self.report)
        self.get_logger().info(
            "waiting for /ground_truth/odom from libgazebo_ros_p3d.so")

    def on_odom(self, msg):
        self.n_in += 1

        now = time.monotonic()
        min_dt = 1.0 / max(self.get_parameter("max_rate_hz").value, 1e-6)
        if now - self._last_pub < min_dt:
            return
        self._last_pub = now

        out = PoseStamped()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = GT_FRAME
        out.pose = msg.pose.pose
        self.pub.publish(out)
        self.n += 1

    def report(self):
        if self.n:
            self.get_logger().info(
                f"{self.n} ground-truth poses published ({self.n_in} received)")
        else:
            self.get_logger().warning(
                "no ground truth yet -- is the sim running, and does the robot model "
                "still declare the libgazebo_ros_p3d.so plugin?")


def main():
    rclpy.init()
    node = GtPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
