#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


class OdomTFNode(Node):
    """Broadcast the simulator's /odom pose as odom -> base_footprint TF."""

    def __init__(self):
        super().__init__("odom_tf_node")

        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            Odometry,
            "/odom",
            self.on_odom,
            50,
        )

    def on_odom(self, msg: Odometry):
        tf = TransformStamped()

        tf.header.stamp = msg.header.stamp
        tf.header.frame_id = msg.header.frame_id or "odom"
        tf.child_frame_id = msg.child_frame_id or "base_footprint"

        tf.transform.translation.x = msg.pose.pose.position.x
        tf.transform.translation.y = msg.pose.pose.position.y
        tf.transform.translation.z = msg.pose.pose.position.z

        tf.transform.rotation = msg.pose.pose.orientation

        self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)

    node = OdomTFNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
