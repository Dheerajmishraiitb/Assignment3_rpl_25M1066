#!/usr/bin/env python3

import math
import os
import sys

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from std_msgs.msg import Float32
from tf2_ros import TransformBroadcaster

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ekf_localization as EKF         

from ament_index_python.packages import get_package_share_directory  
from landmark_msgs.msg import LandmarkArray  

BUFFER_SEC = 3.0
ANCHOR_LAG_SEC = 1.0    


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class EkfNode(Node):
    def __init__(self):
        super().__init__("ekf_node")
        share = get_package_share_directory("exploration_assignment")
        self.declare_parameter("map_npy", os.path.join(share, "config", "map_M.npy"))
        self.declare_parameter("config", os.path.join(share, "config", "markers.yaml"))

        self.declare_parameter("max_detection_age", 1.0)     # [s]

        self.declare_parameter("min_translation", EKF.STEP_TRANS_M)
        self.declare_parameter("min_rotation_deg", EKF.STEP_ROT_DEG)
        self.declare_parameter("seed_from_ground_truth", True)
        self.declare_parameter("innovation_gate", float(EKF.INNOVATION_GATE))

        self.M = np.load(self.get_parameter("map_npy").value)
        if np.isnan(self.M).any():
            self.get_logger().warn("map M contains NaN rows (non-contiguous ids)")

    
        _mu0, self.S, self.R, self.Q = EKF.init_params()
        cfg = yaml.safe_load(open(self.get_parameter("config").value))
        sp = cfg["spawn"]
        self.mu = np.array([float(sp["x"]), float(sp["y"]),
                            math.radians(float(sp["yaw"]))])
        self.seed = None
        self.seeded_from_gt = False

        # --- timing state -------------------------------------------------
        self.buf = []              # [(t, odom_pose)], newest last
        self.anchor_t = None       # committed belief is valid at this instant
        self.anchor_odom = None    # odometry pose the committed belief sits at
        self.last_odom = None
        self.started = False

        self.can_correct = True
        self.n_predict = 0
        self.n_correct = 0
        self.n_dropped = 0
        self.n_rejected = 0
        self.n_stale = 0
        self.n_out_of_order = 0
        self.n_no_motion = 0

        self.pub_pose = self.create_publisher(PoseWithCovarianceStamped,
                                              "/ekf/pose", 10)
        self.pub_path = self.create_publisher(Path, "/ekf/path", 1)
        self.pub_corr = self.create_publisher(Float32, "/diag/corrected", 10)
        self.tf = TransformBroadcaster(self)

        if self.get_parameter("seed_from_ground_truth").value:
            self.create_subscription(PoseStamped, "/ground_truth/pose",
                                     self.on_gt_seed, 10)
        self.create_subscription(LandmarkArray, "/landmarks", self.on_landmarks, 10)
        self.create_subscription(Odometry, "/odom_noisy", self.on_odom, 20)
        self.create_timer(5.0, self.report)
        self.create_timer(1.0 / 20.0, self.broadcast)

        self.path = Path()
        self.path.header.frame_id = "map"
        self.get_logger().info(
            f"loaded {self.M.shape[0]} landmarks; using ekf_localization.py from "
            f"{EKF.__file__}")
        self.get_logger().info(
            f"step {EKF.STEP_TRANS_M} m / {EKF.STEP_ROT_DEG} deg, "
            f"gate {EKF.INNOVATION_GATE}, alpha {EKF.ALPHA}")

    # ------------------------------------------------------------------ timing

    def pose_at(self, t_query):
        
        if not self.buf:
            return None
        best = min(self.buf, key=lambda r: abs(r[0] - t_query))
        return best if abs(best[0] - t_query) <= 0.10 else None

    def _replay(self, mu, S, ref, samples, commit):
        """Apply prediction steps from `ref` through `samples`."""
        min_t = self.get_parameter("min_translation").value
        min_r = math.radians(self.get_parameter("min_rotation_deg").value)
        for _t, pose in samples:
            moved = math.hypot(pose[0] - ref[0], pose[1] - ref[1])
            turned = abs(wrap(pose[2] - ref[2]))
            if moved < min_t and turned < min_r:
                continue
            u = EKF.inverse_motion_model(ref, pose)
            mu, S = EKF.ekf_predict(mu, S, u, self.R)
            ref = pose
            if commit:
                self.n_predict += 1
                self.can_correct = True     # the belief moved: new evidence is due
        return mu, S, ref

    def advance_committed(self, t_target):
       
        samples = [s for s in self.buf if self.anchor_t < s[0] <= t_target]
        self.mu, self.S, self.anchor_odom = self._replay(
            self.mu, self.S, self.anchor_odom, samples, commit=True)
        self.anchor_t = t_target

    def belief_now(self):
       
        samples = [s for s in self.buf if s[0] > self.anchor_t]
        mu, S, _ = self._replay(self.mu.copy(), self.S.copy(),
                                self.anchor_odom, samples, commit=False)
        return mu, S

    # ----------------------------------------------------------- subscriptions

    def on_gt_seed(self, msg):
        if self.seed is None and not self.started:
            p = msg.pose.position
            self.seed = np.array([p.x, p.y, yaw_of(msg.pose.orientation)])
            self.seeded_from_gt = True

    def on_odom(self, msg):
        p = msg.pose.pose.position
        cur = np.array([p.x, p.y, yaw_of(msg.pose.pose.orientation)])
        t = stamp_s(msg.header.stamp)
        self.last_odom = cur
        self.buf.append((t, cur))
        while self.buf and t - self.buf[0][0] > BUFFER_SEC:
            self.buf.pop(0)

        if not self.started:
            self.started = True
            if self.seed is not None:
                self.mu = self.seed.copy()
            self.anchor_t = t
            self.anchor_odom = cur
            self.get_logger().info(
                f"seeded at map pose [{self.mu[0]:+.3f} {self.mu[1]:+.3f} "
                f"{math.degrees(self.mu[2]):+.1f} deg] "
                f"({'ground truth' if self.seeded_from_gt else 'configured spawn'})"
                f"; odom starts at [{cur[0]:+.3f} {cur[1]:+.3f}]")
            return

        if t - self.anchor_t > ANCHOR_LAG_SEC:
            self.advance_committed(t - ANCHOR_LAG_SEC)

    def on_landmarks(self, msg):
        good = [lm for lm in msg.landmarks if 0 <= lm.id < self.M.shape[0]]
        if len(good) != len(msg.landmarks):
            self.n_dropped += len(msg.landmarks) - len(good)
        if not good or not self.started:
            return

        t_m = stamp_s(msg.header.stamp)
        now = self.get_clock().now().nanoseconds / 1e9
        if now - t_m > self.get_parameter("max_detection_age").value:
            self.n_stale += 1
            return
        if t_m <= self.anchor_t:
            self.n_out_of_order += 1
            return
        if self.pose_at(t_m) is None:
            self.n_stale += 1
            return

        z = np.array([[lm.range for lm in good],
                      [lm.bearing for lm in good],
                      [float(lm.id) for lm in good]])

        self.advance_committed(t_m)

        if not self.can_correct:
            self.n_no_motion += 1
            self.pub_corr.publish(Float32(data=0.0))
            return

        self.log_rejections(z)
        before = self.mu.copy()
        self.mu, self.S = EKF.ekf_correct(self.mu, self.S, z, self.Q, self.M,
                                          gate=EKF.INNOVATION_GATE)
        fired = not np.allclose(before, self.mu)
        if fired:
            self.n_correct += 1
            self.can_correct = False
        self.pub_corr.publish(Float32(data=1.0 if fired else 0.0))
        self.append_path(msg.header.stamp)

    # --------------------------------------------------------------- diagnostics

    def log_rejections(self, z):
  
        limit = self.get_parameter("innovation_gate").value
        for k in range(z.shape[1]):
            lm_range, lm_bearing, lm_id = z[0, k], z[1, k], int(z[2, k])
            m = self.M[lm_id]
            dx, dy = m[0] - self.mu[0], m[1] - self.mu[1]
            q = dx * dx + dy * dy
            if q < 1e-9:
                continue
            sq = math.sqrt(q)
            z_hat = np.array([sq, wrap(math.atan2(dy, dx) - self.mu[2])])
            H = (1.0 / q) * np.array([[-sq * dx, -sq * dy, 0.0],
                                      [dy, -dx, -q]])
            Sz = H @ self.S @ H.T + self.Q
            innov = np.array([lm_range - z_hat[0], wrap(lm_bearing - z_hat[1])])
            try:
                d2 = float(innov @ np.linalg.solve(Sz, innov))
            except np.linalg.LinAlgError:
                continue
            if d2 > limit:
                self.n_rejected += 1
                self.get_logger().warn(
                    f"rejected id {lm_id}: mahalanobis^2 = {d2:.1f} > {limit} "
                    f"(range {lm_range:.2f} vs expected {z_hat[0]:.2f} m, "
                    f"bearing {math.degrees(lm_bearing):+.1f} vs "
                    f"{math.degrees(z_hat[1]):+.1f} deg)",
                    throttle_duration_sec=2.0)

    # ------------------------------------------------------------------ outputs

    def broadcast(self):
        
        if not self.started:
            return
        mu, S = self.belief_now()
        stamp = self.get_clock().now().to_msg()
        self.publish(stamp, self.last_odom, mu, S)
        self.path.header.stamp = stamp
        self.pub_path.publish(self.path)

    def append_path(self, stamp):
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = "map"
        ps.pose.position.x = float(self.mu[0])
        ps.pose.position.y = float(self.mu[1])
        ps.pose.orientation.z = math.sin(self.mu[2] / 2)
        ps.pose.orientation.w = math.cos(self.mu[2] / 2)
        self.path.poses.append(ps)
        if len(self.path.poses) > 4000:
            self.path.poses.pop(0)

    def publish(self, stamp, odom_pose, mu, S):
        out = PoseWithCovarianceStamped()
        out.header.stamp = stamp
        out.header.frame_id = "map"
        out.pose.pose.position.x = float(mu[0])
        out.pose.pose.position.y = float(mu[1])
        out.pose.pose.orientation.z = math.sin(mu[2] / 2)
        out.pose.pose.orientation.w = math.cos(mu[2] / 2)
        cov = np.zeros((6, 6))
        cov[:2, :2] = S[:2, :2]
        cov[5, 5] = S[2, 2]
        cov[0, 5] = cov[5, 0] = S[0, 2]
        cov[1, 5] = cov[5, 1] = S[1, 2]
        out.pose.covariance = cov.flatten().tolist()
        self.pub_pose.publish(out)

        # map -> odom, so that (map->odom)(odom->base) puts the robot at mu
        dth = mu[2] - odom_pose[2]
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = float(
            mu[0] - (math.cos(dth) * odom_pose[0] - math.sin(dth) * odom_pose[1]))
        t.transform.translation.y = float(
            mu[1] - (math.sin(dth) * odom_pose[0] + math.cos(dth) * odom_pose[1]))
        t.transform.rotation.z = math.sin(dth / 2)
        t.transform.rotation.w = math.cos(dth / 2)
        self.tf.sendTransform(t)

    def report(self):
        if self.n_predict:
            pct = 100.0 * self.n_correct / max(self.n_predict, 1)
            mu, S = self.belief_now()
            self.get_logger().info(
                f"mu = [{mu[0]:+.3f} {mu[1]:+.3f} "
                f"{math.degrees(mu[2]):+7.2f} deg]  "
                f"sqrt(trace S) = {math.sqrt(max(np.trace(S), 0)):.4f}  "
                f"predict {self.n_predict}, correct {self.n_correct} ({pct:.0f} %)"
                + (f", dropped {self.n_dropped} out-of-map ids" if self.n_dropped else "")
                + (f", GATED {self.n_rejected}" if self.n_rejected else "")
                + (f", stale {self.n_stale}" if self.n_stale else "")
                + (f", out-of-order {self.n_out_of_order}" if self.n_out_of_order else "")
                + (f", held {self.n_no_motion} awaiting motion" if self.n_no_motion else ""))


def main():
    rclpy.init()
    node = EkfNode()
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
