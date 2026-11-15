#!/usr/bin/env python3
"""Publishes everything RViz draws, plus the /diag/* accuracy signals.
"""
import math
import os
import sys

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from landmark_msgs.msg import LandmarkArray
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Float32
from visualization_msgs.msg import Marker, MarkerArray

PLATE_RGB = (0.88, 0.88, 0.90)      # light: the view background is black
UNSEEN_RGB = (1.0, 0.85, 0.1)
SEEN_RGB = (0.2, 1.0, 0.35)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def anchor_of(gt0, od0):

    rot = wrap(gt0[2] - od0[2])
    c, s = math.cos(rot), math.sin(rot)
    return (gt0[0] - (c * od0[0] - s * od0[1]),
            gt0[1] - (s * od0[0] + c * od0[1]), rot)


def apply_anchor(a, p):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * p[0] - s * p[1], a[1] + s * p[0] + c * p[1],
            wrap(a[2] + p[2]))


class Viz(Node):
    def __init__(self):
        super().__init__("ekf_viz_node")
        share = get_package_share_directory("exploration_assignment")
        self.declare_parameter("config", os.path.join(share, "config", "markers.yaml"))
        cfg = yaml.safe_load(open(self.get_parameter("config").value))
        self.markers = cfg["markers"]
        self.mz = float(cfg["height"])
        self.side = float(cfg["marker"]["size"])

        self.est = None
        self.gt = None
        self.errs = []
        self.known = {int(s["id"]) for s in self.markers}
        self.seen = set()
        self.anchor = {}            # topic -> frozen map<-odom transform

        self.diag = {n: self.create_publisher(Float32, f"/diag/{n}", 10)
                     for n in ("ekf_err_pos", "ekf_err_yaw_deg", "dr_err_pos",
                               "dr_err_yaw_deg", "clean_err_pos",
                               "sqrt_trace_S", "markers_in_view")}

        self.pub_lm = self.create_publisher(MarkerArray, "/viz/landmarks", 1)
        self.pub_cov = self.create_publisher(Marker, "/viz/covariance", 1)
        self.pub_gtpath = self.create_publisher(Path, "/ground_truth/path", 1)
        self.gtpath = Path()
        self.gtpath.header.frame_id = "map"

        self.create_subscription(PoseWithCovarianceStamped, "/ekf/pose",
                                 self.on_est, 10)
        self.create_subscription(PoseStamped, "/ground_truth/pose",
                                 self.on_gt, 10)
        self.create_subscription(LandmarkArray, "/landmarks", self.on_lm, 10)
        self.create_subscription(Odometry, "/odom_noisy",
                                 lambda m: self.on_odom(m, "/odom_noisy",
                                                        "dr_err_pos",
                                                        "dr_err_yaw_deg"), 20)
        self.create_subscription(Odometry, "/odom",
                                 lambda m: self.on_odom(m, "/odom",
                                                        "clean_err_pos", None), 20)
        self.create_timer(1.0, self.publish_landmarks)
        self.create_timer(3.0, self.report)

    def emit(self, name, value):
        self.diag[name].publish(Float32(data=float(value)))

    def on_lm(self, msg):
        self.seen |= {int(l.id) for l in msg.landmarks} & self.known
        self.emit("markers_in_view", len(msg.landmarks))

    def on_odom(self, msg, topic, pos_name, yaw_name):
      
        if self.gt is None:
            return
        p = msg.pose.pose.position
        od = (p.x, p.y, yaw_of(msg.pose.pose.orientation))
        if topic not in self.anchor:
            self.anchor[topic] = anchor_of(tuple(self.gt), od)
        x, y, th = apply_anchor(self.anchor[topic], od)
        self.emit(pos_name, math.hypot(x - self.gt[0], y - self.gt[1]))
        if yaw_name:
            self.emit(yaw_name, abs(math.degrees(wrap(th - self.gt[2]))))

    def publish_landmarks(self):
        arr = MarkerArray()
        for s in self.markers:
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns, m.id, m.type, m.action = "aruco", int(s["id"]), Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y = float(s["x"]), float(s["y"])
            m.pose.position.z = self.mz
            yaw = math.radians(float(s["yaw"]))
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.scale.x, m.scale.y, m.scale.z = 0.02, self.side, self.side
            m.color.r, m.color.g, m.color.b = PLATE_RGB
            m.color.a = 0.95
            arr.markers.append(m)

            t = Marker()
            t.header = m.header
            t.ns, t.id, t.type, t.action = "aruco_id", int(s["id"]), Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y = float(s["x"]), float(s["y"])
            t.pose.position.z = self.mz + 0.18
            t.scale.z = 0.14
            t.color.r, t.color.g, t.color.b = (
                SEEN_RGB if int(s["id"]) in self.seen else UNSEEN_RGB)
            t.color.a = 1.0
            t.text = str(s["id"])
            arr.markers.append(t)
        self.pub_lm.publish(arr)

    def on_est(self, msg):
        p = msg.pose.pose.position
        self.est = np.array([p.x, p.y, yaw_of(msg.pose.pose.orientation)])
        cov = np.array(msg.pose.covariance).reshape(6, 6)
        self.publish_cov(msg.header, cov[:2, :2])
        self.emit("sqrt_trace_S",
                  math.sqrt(max(cov[0, 0] + cov[1, 1] + cov[5, 5], 0.0)))
        if self.gt is not None:
            ep = math.hypot(self.est[0] - self.gt[0], self.est[1] - self.gt[1])
            eh = abs(wrap(self.est[2] - self.gt[2]))
            self.errs.append((ep, eh))
            self.emit("ekf_err_pos", ep)
            self.emit("ekf_err_yaw_deg", math.degrees(eh))

    def publish_cov(self, header, cov2):
        vals, vecs = np.linalg.eigh(cov2)
        order = np.argsort(vals)[::-1]
        vals, vecs = np.maximum(vals[order], 0.0), vecs[:, order]
        ang = math.atan2(vecs[1, 0], vecs[0, 0])
        m = Marker()
        m.header = header
        m.ns, m.id, m.type, m.action = "cov", 0, Marker.CYLINDER, Marker.ADD
        m.pose.position.x, m.pose.position.y = float(self.est[0]), float(self.est[1])
        m.pose.position.z = 0.01
        m.pose.orientation.z, m.pose.orientation.w = math.sin(ang / 2), math.cos(ang / 2)
        # 2-sigma, so it is actually visible when the filter is confident
        m.scale.x = max(4 * math.sqrt(vals[0]), 0.02)
        m.scale.y = max(4 * math.sqrt(vals[1]), 0.02)
        m.scale.z = 0.005
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.9, 0.2, 0.35
        self.pub_cov.publish(m)

    def on_gt(self, msg):
        p = msg.pose.position
        self.gt = np.array([p.x, p.y, yaw_of(msg.pose.orientation)])
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose = msg.pose
        self.gtpath.header.stamp = msg.header.stamp
        self.gtpath.poses.append(ps)
        if len(self.gtpath.poses) > 4000:
            self.gtpath.poses.pop(0)
        self.pub_gtpath.publish(self.gtpath)

    def report(self):
        if not self.errs:
            waiting = [n for n, v in (("/ekf/pose", self.est),
                                      ("/ground_truth/pose", self.gt)) if v is None]
            if waiting:
                self.get_logger().info(f"waiting for {', '.join(waiting)}")
            return
        e = np.array(self.errs[-200:])
        self.get_logger().info(
            f"EKF vs ground truth over last {len(e)} updates: "
            f"position mean {e[:,0].mean():.3f} m  max {e[:,0].max():.3f} m | "
            f"heading mean {math.degrees(e[:,1].mean()):.2f} deg")


def main():
    rclpy.init()
    node = Viz()
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
