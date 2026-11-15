#!/usr/bin/env python3
"""Turns camera frames into range/bearing observations of the ArUco markers.

This is what feeds your filter's correction step.
"""
import os
import sys

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from landmark_msgs.msg import Landmark, LandmarkArray
from rclpy.node import Node
from rclpy.qos import (HistoryPolicy, QoSPresetProfiles, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

CAM_FWD = 0.076


def cap_opencv_threads():
   
    n = None
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = max(1, int(float(quota) / float(period)))
    except Exception:
        pass
    if n is None:
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                n = max(1, q // p)
        except Exception:
            pass
    if n is None:
        n = os.cpu_count() or 2
   
    cv2.setNumThreads(max(1, min(4, n - 2)))
    return cv2.getNumThreads(), n

def build_detector_params(cfg):
    p = (cv2.aruco.DetectorParameters_create()
         if hasattr(cv2.aruco, "DetectorParameters_create")
         else cv2.aruco.DetectorParameters())
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    sensor = cfg.get("sensor", {})
    if "error_correction_rate" in sensor:
        p.errorCorrectionRate = float(sensor["error_correction_rate"])
    return p


class ArucoDetector(Node):
    def __init__(self):
        super().__init__("aruco_detector_node")
        _cv_threads, _cv_quota = cap_opencv_threads()
        self.get_logger().info(
            f"OpenCV threads capped to {_cv_threads} "
            f"(cpu quota {_cv_quota}, os.cpu_count {os.cpu_count()})")
        share = get_package_share_directory("exploration_assignment")
        self.declare_parameter("config", os.path.join(share, "config", "markers.yaml"))
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("debug_compressed", False)
        self.declare_parameter("use_compressed", False)
        self.declare_parameter("max_image_age", 0.060)   # [s]
 
        self.declare_parameter("blocked_ids", [-1])

        cfg_path = self.get_parameter("config").value
        cfg = yaml.safe_load(open(cfg_path))
        m, sensor = cfg["marker"], cfg.get("sensor", {})

        self.side = float(m["size"])
        self.solve_side = float(sensor.get("effective_size_for_solvepnp", self.side))
        self.min_px = float(sensor.get("min_apparent_px", 0.0))
        self.blocked_ids = {int(i) for i
                            in (self.get_parameter("blocked_ids").value or [])
                            if int(i) >= 0}
        if self.blocked_ids:
            self.get_logger().warn(
                f"DEGRADED MODE: refusing marker ids {sorted(self.blocked_ids)}")
        self.range_scale = float(sensor.get("range_scale", 1.0))
        self.range_offset = float(sensor.get("range_offset", 0.0))

        dict_id = getattr(cv2.aruco, m["dictionary"])
        self.dictionary = (cv2.aruco.getPredefinedDictionary(dict_id)
                           if hasattr(cv2.aruco, "getPredefinedDictionary")
                           else cv2.aruco.Dictionary_get(dict_id))
        self.params = build_detector_params(cfg)

        h = self.solve_side / 2.0
        self.objp = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
                             np.float32)

        self.valid_ids = {int(s["id"]) for s in cfg["markers"]}

        self.K = None
        self.D = None
        self.n_frames = 0
        self.n_obs = 0
        self.n_gated = 0
        self.n_unknown = 0
        self.n_blocked = 0
        self.n_late = 0
        self.unknown_ids = {}

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(CameraInfo, "/camera/camera_info",
                                 self.on_info, sensor_qos)
        img_qos = QoSProfile(depth=1,
                             history=HistoryPolicy.KEEP_LAST,
                             reliability=ReliabilityPolicy.BEST_EFFORT)

        if self.get_parameter("use_compressed").value:
            self.create_subscription(CompressedImage,
                                     "/camera/image_raw/compressed",
                                     self.on_compressed, img_qos)
            self.get_logger().info("subscribed to /camera/image_raw/compressed")
        else:
            self.create_subscription(Image, "/camera/image_raw",
                                     self.on_image, img_qos)
            self.get_logger().info("subscribed to /camera/image_raw (raw)")
        self.pub = self.create_publisher(LandmarkArray, "/landmarks", 10)

        if self.get_parameter("debug_compressed").value:
            self.pub_dbg = self.create_publisher(
                CompressedImage, "/aruco/debug_image/compressed", 1)
        else:
            self.pub_dbg = self.create_publisher(Image, "/aruco/debug_image", 1)
        self.create_timer(5.0, self.report)

        self.get_logger().info(
            f"marker {self.side*100:.0f} cm ({m['dictionary']}), solvePnP side "
            f"{self.solve_side*100:.1f} cm, gate >= {self.min_px:.0f} px, "
            f"AprilTag refinement")

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k, float).reshape(3, 3)
            self.D = np.array(msg.d, float)
            self.get_logger().info(
                f"intrinsics fx={self.K[0,0]:.1f} cx={self.K[0,2]:.0f} "
                f"cy={self.K[1,2]:.0f}  {msg.width}x{msg.height}")

    def on_compressed(self, msg):

        if self.K is None:
            return
        buf = np.frombuffer(msg.data, np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            self.n_decode_fail = getattr(self, "n_decode_fail", 0) + 1
            return
        shim = Image()
        shim.header = msg.header
        shim.height, shim.width = bgr.shape[0], bgr.shape[1]
        shim.data = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes()
        shim.encoding = "rgb8"
        shim.step = shim.width * 3
        self.on_image(shim)

    def on_image(self, msg):
        if self.K is None:
            return
        max_age = self.get_parameter("max_image_age").value
        if max_age > 0:
            age = (self.get_clock().now().nanoseconds / 1e9
                   - (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9))
            if age > max_age:
                self.n_late += 1
                return
        self.n_frames += 1
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.shape[2] == 3 else arr[:, :, 0]

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary,
                                                  parameters=self.params)
        out = LandmarkArray()
        out.header = msg.header
        kept, flagged = [], []
        if ids is not None:
            for c, i in zip(corners, ids.ravel()):
                ip = c.reshape(4, 2).astype(np.float32)
                px = float(np.mean([np.linalg.norm(ip[k] - ip[(k + 1) % 4])
                                    for k in range(4)]))
                self.n_obs += 1
                if int(i) not in self.valid_ids:
                    self.n_unknown += 1
                    self.unknown_ids[int(i)] = self.unknown_ids.get(int(i), 0) + 1
                    flagged.append((ip, int(i), "not in map"))
                    continue
                if int(i) in self.blocked_ids:
                    self.n_blocked += 1
                    flagged.append((ip, int(i), "blocked"))
                    continue
                if px < self.min_px:
                    self.n_gated += 1
                    flagged.append((ip, int(i), f"{px:.0f}px < {self.min_px:.0f}"))
                    continue
                ok, _, tvec = cv2.solvePnP(self.objp, ip, self.K, self.D,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not ok:
                    flagged.append((ip, int(i), "solvePnP failed"))
                    continue
                X, _, Z = tvec.ravel()
                # optical frame (x right, y down, z forward) -> base frame
                fwd, left = Z + CAM_FWD, -X
                lm = Landmark()
                lm.id = int(i)
                raw_range = float(np.hypot(fwd, left))
                lm.range = (raw_range - self.range_offset) / self.range_scale
                lm.bearing = float(np.arctan2(left, fwd))
                lm.apparent_px = px
                out.landmarks.append(lm)
                kept.append((ip, int(i), lm.range))
        self.pub.publish(out)

        if self.get_parameter("publish_debug_image").value:
            self.publish_debug(arr, kept, flagged, msg.header)

    def publish_debug(self, rgb, kept, flagged, header):
        """GREEN = sent to the EKF. RED = detected but discarded, with the reason."""
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        for ip, i, why in flagged:
            cv2.polylines(img, [ip.astype(np.int32)], True, (0, 0, 255), 2)
            c = ip.mean(axis=0).astype(int)
            cv2.putText(img, f"{i} ({why})", tuple(c), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        for ip, i, rng in kept:
            cv2.polylines(img, [ip.astype(np.int32)], True, (0, 255, 0), 2)
            c = ip.mean(axis=0).astype(int)
            cv2.putText(img, f"{i} {rng:.2f}m", tuple(c), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        cv2.putText(img, "green = used by EKF    red = detected but rejected",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        if self.get_parameter("debug_compressed").value:
            ok, enc = cv2.imencode(".jpg", img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return
            out = CompressedImage()
            out.header = header
            out.format = "bgr8; jpeg compressed bgr8"
            out.data = enc.tobytes()
        else:
            out = Image()
            out.header = header
            out.height, out.width = img.shape[:2]
            out.encoding = "bgr8"
            out.step = img.shape[1] * 3
            out.data = img.tobytes()
        self.pub_dbg.publish(out)

    def report(self):
        if not self.n_frames:
            return
        published = (self.n_obs - self.n_gated - self.n_unknown
                     - self.n_blocked)
        self.get_logger().info(
            f"{self.n_frames} frames, {self.n_obs} raw detections -> "
            f"{published} published "
            f"({self.n_gated} under the {self.min_px:.0f} px gate, "
            + (f"{self.n_late} dropped as late, " if self.n_late else "")
            + (f"{self.n_blocked} blocked, " if self.n_blocked else "")
            + f"{self.n_unknown} unknown ids)")
        if self.unknown_ids:
            worst = sorted(self.unknown_ids.items(), key=lambda kv: -kv[1])[:5]
            self.get_logger().warn(
                "misdecoded ids not in the map (dropped): "
                + ", ".join(f"{i}x{n}" for i, n in worst))


def main():
    rclpy.init()
    node = ArucoDetector()
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
