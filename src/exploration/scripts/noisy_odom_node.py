#!/usr/bin/env python3
"""Adds realistic noise to the wheel odometry.

This is the drift your filter has to correct for.
"""
import math
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

MIN_TRANS_M = 1e-4
MIN_ROT_RAD = 1e-4


NOISE_STEP_TRANS_M = 0.05
NOISE_STEP_ROT_RAD = math.radians(5.0)


ROT1_MIN_TRANS_M = 0.01


PATH_APPEND_HZ = 10.0
PATH_PUB_HZ = 5.0
PATH_MAX_POSES = 4000

class NoisyOdom(Node):
    def __init__(self):
        super().__init__("noisy_odom_node")
        self.declare_parameter("alpha", [0.02, 0.02, 0.05, 0.02])
        self.declare_parameter("enable_noise", True)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("seed", 0)

        self.alpha = [float(a) for a in self.get_parameter("alpha").value]
        self.rng = np.random.default_rng(self.get_parameter("seed").value)

        self.prev = None          # last clean pose  [x, y, theta]
        self.noisy = None         # accumulated noisy pose
        self.travelled = 0.0
        self._path_appended = -1e9
        self._path_published = -1e9

        self.pub = self.create_publisher(Odometry, "/odom_noisy", 20)
        self.pub_path = self.create_publisher(Path, "/odom_noisy/path", 1)
        self.pub_cleanpath = self.create_publisher(Path, "/odom_clean/path", 1)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_timer(5.0, self.report)

        self.path = Path()
        self.path.header.frame_id = "odom"
        self.cleanpath = Path()
        self.cleanpath.header.frame_id = "odom"

        self.get_logger().info(
            f"alpha = {self.alpha}, noise "
            f"{'ON' if self.get_parameter('enable_noise').value else 'OFF'}")

    def sample(self, std):
        return float(self.rng.normal(0.0, abs(std))) if std > 1e-12 else 0.0

    def on_odom(self, msg):
    
        p = msg.pose.pose.position
        cur = np.array([p.x, p.y, yaw_of(msg.pose.pose.orientation)])

        if self.prev is None:
            self.prev = cur              # clean pose at the last noise event
            self.noisy = cur.copy()      # published noisy pose
            self.anchor = cur.copy()     # noisy pose at the last noise event
            return

        dx, dy = cur[0] - self.prev[0], cur[1] - self.prev[1]
        trans = math.hypot(dx, dy)
        dyaw = wrap(cur[2] - self.prev[2])

        if trans < MIN_TRANS_M and abs(dyaw) < MIN_ROT_RAD:
            self.publish(msg)
            return

        rot1 = (wrap(math.atan2(dy, dx) - self.prev[2])
                if trans > ROT1_MIN_TRANS_M else 0.0)
        rot2 = wrap(dyaw - rot1)

        stepped = (trans >= NOISE_STEP_TRANS_M
                   or abs(dyaw) >= NOISE_STEP_ROT_RAD)

        if stepped and self.get_parameter("enable_noise").value:
            a1, a2, a3, a4 = self.alpha
            rot1 += self.sample(a1 * abs(rot1) + a2 * trans)
            trans += self.sample(a3 * trans + a4 * (abs(rot1) + abs(rot2)))
            rot2 += self.sample(a1 * abs(rot2) + a2 * trans)

        th = self.anchor[2]
        pose = self.anchor + np.array([trans * math.cos(th + rot1),
                                       trans * math.sin(th + rot1),
                                       rot1 + rot2])
        pose[2] = wrap(pose[2])
        self.noisy = pose

        if stepped:
            
            self.anchor = pose.copy()
            self.prev = cur
            self.travelled += abs(trans)
            self.n_steps = getattr(self, "n_steps", 0) + 1

        self.publish(msg)

    def publish(self, src):
        out = Odometry()
        out.header = src.header
        out.child_frame_id = src.child_frame_id
        out.pose.pose.position.x = float(self.noisy[0])
        out.pose.pose.position.y = float(self.noisy[1])
        out.pose.pose.orientation.z = math.sin(self.noisy[2] / 2)
        out.pose.pose.orientation.w = math.cos(self.noisy[2] / 2)
        out.twist = src.twist
        self.pub.publish(out)

        if self.get_parameter("publish_tf").value:
            t = TransformStamped()
            t.header = src.header
            t.header.frame_id = "odom"
            t.child_frame_id = src.child_frame_id or "base_footprint"
            t.transform.translation.x = float(self.noisy[0])
            t.transform.translation.y = float(self.noisy[1])
            t.transform.rotation = out.pose.pose.orientation
            self.tf.sendTransform(t)

        t_src = (src.header.stamp.sec + src.header.stamp.nanosec * 1e-9)
        append = t_src - self._path_appended >= 1.0 / PATH_APPEND_HZ
        emit = t_src - self._path_published >= 1.0 / PATH_PUB_HZ
        if not (append or emit):
            return
        if append:
            self._path_appended = t_src
        if emit:
            self._path_published = t_src

        for path, pub, pose in ((self.path, self.pub_path, self.noisy),
                                (self.cleanpath, self.pub_cleanpath, self.prev)):
            if append:
                ps = PoseStamped()
                ps.header = src.header
                ps.header.frame_id = "odom"
                ps.pose.position.x = float(pose[0])
                ps.pose.position.y = float(pose[1])
                path.poses.append(ps)
                if len(path.poses) > PATH_MAX_POSES:
                    path.poses.pop(0)
            if emit:
                path.header.stamp = src.header.stamp
                pub.publish(path)

    def report(self):
        if self.prev is not None and self.travelled > 0.01:
            d = math.hypot(self.noisy[0] - self.prev[0], self.noisy[1] - self.prev[1])
            self.get_logger().info(
                f"travelled {self.travelled:.2f} m, dead-reckoning has drifted "
                f"{d:.3f} m ({100*d/max(self.travelled,1e-9):.1f} %), "
                f"heading off {math.degrees(abs(wrap(self.noisy[2]-self.prev[2]))):.1f} deg")


def main():
    rclpy.init()
    node = NoisyOdom()
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
