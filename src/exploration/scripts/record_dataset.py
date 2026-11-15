#!/usr/bin/env python3
"""Record a localization dataset from the Gazebo simulation."""
import argparse
import math
import os
import pickle
import sys
import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from landmark_msgs.msg import LandmarkArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter


TOUR_BUDGET_S = 300.0         


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y),
                      1 - 2 * (q.y * q.y + q.z * q.z))


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec / 1e9


class Recorder(Node):
    def __init__(self, args):
        super().__init__("record_dataset",
                         parameter_overrides=[
                             Parameter("use_sim_time", value=True)])
        share = get_package_share_directory("exploration_assignment")
        self.declare_parameter("config", os.path.join(share, "config", "markers.yaml"))
        self.cfg = yaml.safe_load(open(self.get_parameter("config").value))
        self.M = np.load(os.path.join(share, "config", "map_M.npy"))

        self.rate = args.rate
        self.min_move = args.min_move
        self.max_age = args.max_detection_age
        self.out = args.out

        self.odom = None            # noisy
        self.odom_true = None       # clean
        self.gt = None
        self.lm = None
        self.lm_stamp = None

        self.rows = []
        self.buf = []               # (t, odom, odom_true, gt), for pose lookup
        self.last_used_stamp = None
        self.skipped_stale = 0
        self.skipped_idle = 0
        self.last_logged = None
        self.last_pace = None       # (elapsed, distance) at the previous report

        self.create_subscription(Odometry, "/odom_noisy", self._odom, 50)
        self.create_subscription(Odometry, "/odom", self._odom_true, 50)
        self.create_subscription(PoseStamped, "/ground_truth/pose", self._gt, 50)
        self.create_subscription(LandmarkArray, "/landmarks", self._lm, 20)
        self.create_timer(0.02, self._buffer)      # 50 Hz pose buffer
        self.create_timer(1.0 / self.rate, self.tick)
        self.create_timer(5.0, self.report)
        if args.autosave > 0:
            self.create_timer(args.autosave,
                              lambda: self.save(self.out, quiet=True))

        self.get_logger().info(
            f"recording at {self.rate} Hz; a detection counts only if fresher "
            f"than {self.max_age:.3f} s")

    def _odom(self, m):
        p = m.pose.pose.position
        self.odom = np.array([p.x, p.y, yaw_of(m.pose.pose.orientation)])

    def _odom_true(self, m):
        p = m.pose.pose.position
        self.odom_true = np.array([p.x, p.y, yaw_of(m.pose.pose.orientation)])

    def _gt(self, m):
        p = m.pose.position
        self.gt = np.array([p.x, p.y, yaw_of(m.pose.orientation)])

    def _lm(self, m):
        if m.landmarks:
            self.lm = m.landmarks
            self.lm_stamp = stamp_s(m.header.stamp)

    def _buffer(self):
        
        if self.odom is None or self.gt is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9
        self.buf.append((now, self.odom.copy(),
                         self.odom_true.copy() if self.odom_true is not None
                         else self.odom.copy(), self.gt.copy()))
        while self.buf and now - self.buf[0][0] > 3.0:
            self.buf.pop(0)

    def pose_at(self, t_query):
        
        if not self.buf:
            return None
        best = min(self.buf, key=lambda r: abs(r[0] - t_query))
        return best if abs(best[0] - t_query) <= 0.25 else None

    def tick(self):
        if self.odom is None or self.gt is None:
            return
        now = self.get_clock().now().nanoseconds / 1e9

        if self.last_logged is not None and self.min_move > 0:
            if (math.hypot(self.odom[0] - self.last_logged[0],
                           self.odom[1] - self.last_logged[1]) < self.min_move
                    and abs(self.odom[2] - self.last_logged[2]) < math.radians(2.0)):
                self.skipped_idle += 1
                return

        z = np.zeros((3, 0))
        row_t, row = now, None
        if (self.lm is not None and self.lm_stamp is not None
                and self.lm_stamp != self.last_used_stamp
                and now - self.lm_stamp <= self.max_age):
            at = self.pose_at(self.lm_stamp)
            if at is not None:
                valid = [l for l in self.lm if 0 <= l.id < self.M.shape[0]]
                if valid:
                    z = np.array([[l.range for l in valid],
                                  [l.bearing for l in valid],
                                  [float(l.id) for l in valid]])
                    row_t, row = self.lm_stamp, at
                    self.last_used_stamp = self.lm_stamp
            else:
                self.skipped_stale += 1
        elif (self.lm_stamp is not None
              and now - self.lm_stamp > self.max_age):
            self.skipped_stale += 1

        if row is None:
            row = (now, self.odom.copy(),
                   self.odom_true.copy() if self.odom_true is not None
                   else self.odom.copy(), self.gt.copy())

        if self.rows and abs(row_t - self.rows[-1][0]) < 1e-6:
            return                       # same instant, nothing new
        self.rows.append((row_t, row[1], row[2], row[3], z))
        self.last_logged = self.odom.copy()

    def report(self):
        if not self.rows:
            self.get_logger().info("waiting for /odom_noisy and /ground_truth/pose")
            return
        n_obs = sum(1 for r in self.rows if r[4].shape[1] > 0)
        gt = np.array([r[3] for r in self.rows])
        dist = float(np.sum(np.hypot(np.diff(gt[:, 0]), np.diff(gt[:, 1]))))
        el = self.rows[-1][0] - self.rows[0][0]

        v_now = 0.0
        if self.last_pace is not None and el - self.last_pace[0] > 0.5:
            v_now = (dist - self.last_pace[1]) / (el - self.last_pace[0])
        self.last_pace = (el, dist)

        self.get_logger().info(
            f"[{el:4.0f}/{TOUR_BUDGET_S:.0f} s] {dist:5.2f} m  "
            f"now {v_now:4.2f} m/s  avg {dist / max(el, 1e-6):5.3f} m/s  |  "
            f"{len(self.rows)} ticks, {100*n_obs/len(self.rows):.0f} % with a "
            f"landmark, {self.skipped_stale} stale, {self.skipped_idle} idle"
            + ("   <-- NO LANDMARKS: dataset would be unusable"
               if n_obs == 0 and len(self.rows) > 50 else ""))

    def save(self, path, quiet=False):
        if len(self.rows) < 20:
            if not quiet:
                print(f"only {len(self.rows)} ticks -- refusing to save",
                      file=sys.stderr)
            return False
        rows = sorted(self.rows, key=lambda r: r[0])
        gt = np.array([r[3] for r in rows])
        dist = float(np.sum(np.hypot(np.diff(gt[:, 0]), np.diff(gt[:, 1]))))
        seen = sorted({int(i) for r in rows for i in r[4][2, :]})
        data = {
            "M": self.M,
            "odom": [r[1] for r in rows],
            "odom_true": [r[2] for r in rows],
            "gt": [r[3] for r in rows],
            "z": [r[4] for r in rows],
            "t": [r[0] for r in rows],
            "meta": {
                "source": "gazebo turtlebot3_house_aruco",
                "rate_hz": self.rate,
                "marker": self.cfg["marker"],
                "marker_height": self.cfg["height"],
                "sensor": self.cfg.get("sensor", {}),
                "spawn": self.cfg["spawn"],
                "distance_m": dist,
                "duration_s": rows[-1][0] - rows[0][0],
                "ids_seen": seen,
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp, path)          # atomic -- never a half-written dataset
        n_obs = sum(1 for r in rows if r[4].shape[1] > 0)
        if quiet:
            self.get_logger().info(
                f"autosaved {len(self.rows)} ticks, {dist:.2f} m, "
                f"{n_obs} with landmarks -> {os.path.basename(path)}")
            return True
        print(f"\nwrote {path}")
        print(f"  {len(self.rows)} ticks at {self.rate} Hz "
              f"({data['meta']['duration_s']:.1f} s of sim time)")
        print(f"  {dist:.2f} m driven")
        print(f"  {n_obs} ticks carry a landmark ({100*n_obs/len(self.rows):.0f} %), "
              f"{len(self.rows)-n_obs} are prediction-only")
        print(f"  {len(seen)} distinct markers seen: {seen}")
        print(f"  {self.skipped_stale} detections dropped as stale, "
              f"{self.skipped_idle} idle ticks skipped")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="my_run.p")
    ap.add_argument("--rate", type=float, default=10.0, help="tick rate [Hz]")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after this much wall time (0 = until Ctrl-C)")
    ap.add_argument("--min-move", type=float, default=0.01,
                    help="skip ticks with less motion than this [m]")
    ap.add_argument("--max-detection-age", type=float, default=0.60,
                    help="attach a detection only if fresher than this [s]")
    ap.add_argument("--autosave", type=float, default=5.0,
                    help="rewrite the output file every N seconds (0 = off)")
    args = ap.parse_args()

    rclpy.init()
    node = Recorder(args)
    t0 = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.seconds and time.time() - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    ok = node.save(args.out)
    try:
        node.destroy_node()
    except Exception:
        pass
    if rclpy.ok():
        rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
