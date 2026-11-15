#!/usr/bin/env python3
"""ROS 2 infrastructure for CS7001 Assignment 3.

Students should edit ONLY ``exploration.py``.  This node supplies:
  * Gazebo/EKF/lidar interfaces and OccupancyGrid publication,
  * map save/load (PNG + CSV + YAML metadata),
  * obstacle inflation and waypoint following,
  * random reachable checkpoint generation,
  * experiment timing/logging,
  * orchestration for manual mapping, A* checkpoint navigation,
    largest-frontier greedy exploration and MCTS exploration.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import deque
from pathlib import Path as FsPath
from typing import List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exploration as student
from tf2_ros import Buffer, TransformListener, TransformException

GridCell = Tuple[int, int]
WorldPoint = Tuple[float, float]


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of(q) -> float:
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    )


class Assignment3Node(Node):
    def __init__(self):
        super().__init__("assignment3_node")

        # ---------------- assignment / experiment mode
        self.declare_parameter("mode", "mapping")
        self.declare_parameter("output_dir", "~/.ros/cs7001_assignment3")
        self.declare_parameter("map_csv", "~/.ros/cs7001_assignment3/manual_map.csv")
        self.declare_parameter("map_name", "manual_map")
        self.declare_parameter("random_seed", 7)

        # ---------------- occupancy map geometry
        self.declare_parameter("resolution", 0.05)
        self.declare_parameter("origin_x", -8.5)
        self.declare_parameter("origin_y", -4.8)
        self.declare_parameter("width", 340)
        self.declare_parameter("height", 216)
        self.declare_parameter("sensor_range", 3.5)

        # Mapping stability: do not integrate scans while the EKF pose is
        # changing too rapidly. This avoids smearing walls during turns or
        # immediately after a localization correction.
        self.declare_parameter("mapping_max_angular_rate", 0.45)  # rad/s
        self.declare_parameter("mapping_max_pose_jump", 0.08)     # m / scan

        # ---------------- navigation / checkpoint settings
        self.declare_parameter("inflation_radius", 0.15)
        self.declare_parameter("checkpoint_clearance", 0.15)
        self.declare_parameter("checkpoint_count", 6)
        self.declare_parameter("checkpoint_min_separation", 2.50)
        self.declare_parameter("goal_tolerance", 0.22)
        self.declare_parameter("path_point_tolerance", 0.14)
        self.declare_parameter("max_linear_speed", 0.16)
        self.declare_parameter("max_angular_speed", 1.20)
        self.declare_parameter("emergency_stop_range", 0.20)
        self.declare_parameter("replan_period", 1.0)

        # ---------------- exploration settings
        self.declare_parameter("min_frontier_size", 4)
        self.declare_parameter("goal_timeout", 35.0)
        self.declare_parameter("no_frontier_timeout", 6.0)
        self.declare_parameter("min_runtime_before_finish", 15.0)
        self.declare_parameter("mcts_iterations", 180)
        self.declare_parameter("mcts_rollout_depth", 4)
        self.declare_parameter("mcts_exploration_constant", 1.4)
        self.declare_parameter("mcts_gain_weight", 0.05)
        self.declare_parameter("mcts_distance_weight", 1.0)
        self.declare_parameter("mcts_max_frontiers", 18)

        self.mode = str(self.get_parameter("mode").value).strip().lower()
        if self.mode not in ("mapping", "checkpoint", "greedy", "mcts"):
            raise ValueError("mode must be mapping, checkpoint, greedy, or mcts")

        self.resolution = float(self.get_parameter("resolution").value)
        self.origin = (
            float(self.get_parameter("origin_x").value),
            float(self.get_parameter("origin_y").value),
        )
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)
        self.occupancy = np.full((self.height, self.width), -1, dtype=np.int8)

        self.output_dir = FsPath(
            os.path.expanduser(str(self.get_parameter("output_dir").value)))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(int(self.get_parameter("random_seed").value))

        # ---------------- live state
        self.pose: Optional[np.ndarray] = None

        # Pose supplied by the SLAM localization backend through TF.
        # Its occupancy map is NOT used by the assignment mapper.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Explicit pose output for RViz. This is the same localization pose
        # used by mapping/planning, expressed in the map frame.
        self.current_pose_pub = self.create_publisher(
            PoseStamped,
            "/assignment3/current_pose",
            10,
        )

        self.robot_marker_pub = self.create_publisher(
            Marker,
            "/assignment3/robot_marker",
            10,
        )
        self.scan: Optional[LaserScan] = None
        self.home: Optional[WorldPoint] = None
        self.previous_pose_xy: Optional[WorldPoint] = None
        self.travel_distance = 0.0

        # Pose history used only to decide whether a laser scan is stable
        # enough to integrate into the occupancy grid.
        self.mapping_previous_pose: Optional[np.ndarray] = None
        self.mapping_previous_time: Optional[float] = None
        self.last_mapping_skip_log_time = -1e9

        # Task 1 trajectory log. Poses are recorded in the same map/world frame
        # used by the occupancy grid so the path can be overlaid later.
        self.mapping_trajectory_start_time: Optional[float] = None
        self.mapping_trajectory_rows: List[Tuple[float, float, float, float, int, int]] = []

        self.goal: Optional[WorldPoint] = None
        self.goal_cell: Optional[GridCell] = None
        self.goal_set_time = -1e9
        self.path_cells: List[GridCell] = []
        self.path_index = 0
        self.last_plan_time = -1e9
        self.blacklisted_until = {}  # GridCell -> simulation time

        self.phase = "mapping" if self.mode == "mapping" else (
            "checkpoint_wait" if self.mode == "checkpoint" else "exploring")
        self.exploration_start_time: Optional[float] = None
        self.exploration_start_distance = 0.0
        self.exploration_end_time: Optional[float] = None
        self.exploration_end_distance: Optional[float] = None
        self.frontier_missing_since: Optional[float] = None
        self.coverage_rows: List[
            Tuple[float, int, float, float, int, int]
        ] = []

        self.mission_waypoints: List[WorldPoint] = []
        self.mission_labels: List[str] = []
        self.mission_index = 0
        self.mission_started_time: Optional[float] = None
        self.mission_start_distance = 0.0

        # Task 2 / Task 6 autonomous mission trajectory.
        self.mission_trajectory_rows = []

        # ---------------- publishers/subscribers/services
        self.pub_cmd = self.create_publisher(Twist, "/cmd_vel", 10)
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        # Static/saved maps should remain available to RViz even when RViz
        # subscribes after the map was initially loaded.
        self.pub_map = self.create_publisher(
            OccupancyGrid,
            "/map",
            map_qos,
        )
        self.pub_path = self.create_publisher(Path, "/exploration/path", 1)
        self.pub_goal = self.create_publisher(Marker, "/exploration/goal", 1)
        self.pub_waypoint = self.create_publisher(
            Marker,
            "/exploration/waypoint",
            1,
        )
        self.pub_frontiers = self.create_publisher(
            Marker,
            "/exploration/frontiers",
            1,
        )

        self.pub_mcts_debug = self.create_publisher(
            MarkerArray,
            "/exploration/mcts_debug",
            1,
        )
        self.pub_checkpoints = self.create_publisher(
            MarkerArray, "/assignment3/checkpoints", 1)
        self.pub_elapsed = self.create_publisher(Float32, "/exploration/elapsed", 10)
        self.pub_known = self.create_publisher(Float32, "/exploration/known_cells", 10)
        self.pub_status = self.create_publisher(String, "/exploration/status", 10)

        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_service(Trigger, "/assignment3/save_map", self.on_save_map)

        self.create_timer(0.10, self.control_tick)
        self.create_timer(0.50, self.diagnostic_tick)

        if self.mode == "checkpoint":
            self._load_manual_map()

        self.get_logger().info(
            f"Assignment 3 node ready: mode={self.mode}, "
            f"map={self.width}x{self.height}@{self.resolution:.2f}m")
        if self.mode == "mapping":
            self.get_logger().info(
                "Teleoperate the robot. Save with: "
                "ros2 service call /assignment3/save_map std_srvs/srv/Trigger '{}' ")

    # ------------------------------------------------------------------ time

    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------------------- callbacks

    def on_pose(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        cur = np.array([float(p.x), float(p.y), float(yaw_of(q))], dtype=float)

        if self.home is None:
            self.home = (cur[0], cur[1])
            self.get_logger().info(
                f"Home recorded at ({self.home[0]:+.2f}, {self.home[1]:+.2f})")

        if self.previous_pose_xy is not None:
            step = math.hypot(
                cur[0] - self.previous_pose_xy[0],
                cur[1] - self.previous_pose_xy[1],
            )
            # Ignore rare EKF discontinuities in physical travel accounting.
            if step < 0.5:
                self.travel_distance += step
        self.previous_pose_xy = (cur[0], cur[1])
        self.pose = cur

        if self.mode == "mapping":
            now = self.now_s()
            if self.mapping_trajectory_start_time is None:
                self.mapping_trajectory_start_time = now

            row, col = student.world_to_grid(
                cur[0], cur[1], self.resolution, self.origin)
            self.mapping_trajectory_rows.append((
                now - self.mapping_trajectory_start_time,
                float(cur[0]),
                float(cur[1]),
                float(cur[2]),
                int(row),
                int(col),
            ))

    def get_localization_pose(self):
        """Return [x, y, yaw] from slam_toolbox TF."""
        try:
            tf = self.tf_buffer.lookup_transform(
                "map",
                "base_footprint",
                rclpy.time.Time(),
            )
        except TransformException:
            return None

        x = float(tf.transform.translation.x)
        y = float(tf.transform.translation.y)

        q = tf.transform.rotation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        yaw = math.atan2(siny_cosp, cosy_cosp)

        return np.array([x, y, yaw], dtype=float)


    def on_scan(self, msg: LaserScan):
        self.scan = msg

        localization_pose = self.get_localization_pose()
        if localization_pose is None:
            return

        # From this point onward, all Assignment 3 mapping/planning code
        # sees the pose estimated by the SLAM backend.
        self.pose = localization_pose

        # -------------------------------------------------------------
        # Common pose bookkeeping
        #
        # SLAM TF is now the localization source for Assignment 3, so
        # home position, travelled distance and Task 1 trajectory must
        # be derived from this pose rather than the old /ekf/pose topic.
        # -------------------------------------------------------------
        cur = self.pose

        if self.home is None:
            self.home = (float(cur[0]), float(cur[1]))
            self.get_logger().info(
                f"Home recorded at ({self.home[0]:+.2f}, {self.home[1]:+.2f})"
            )

        if self.previous_pose_xy is not None:
            step = math.hypot(
                float(cur[0]) - self.previous_pose_xy[0],
                float(cur[1]) - self.previous_pose_xy[1],
            )

            # Ignore rare localization discontinuities in travel accounting.
            if step < 0.5:
                self.travel_distance += step

        self.previous_pose_xy = (float(cur[0]), float(cur[1]))

        if self.mode == "mapping":
            now = self.now_s()

            if self.mapping_trajectory_start_time is None:
                self.mapping_trajectory_start_time = now

            row, col = student.world_to_grid(
                cur[0],
                cur[1],
                self.resolution,
                self.origin,
            )

            self.mapping_trajectory_rows.append((
                now - self.mapping_trajectory_start_time,
                float(cur[0]),
                float(cur[1]),
                float(cur[2]),
                int(row),
                int(col),
            ))

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"

        pose_msg.pose.position.x = float(self.pose[0])
        pose_msg.pose.position.y = float(self.pose[1])
        pose_msg.pose.position.z = 0.0

        half_yaw = 0.5 * float(self.pose[2])
        pose_msg.pose.orientation.z = math.sin(half_yaw)
        pose_msg.pose.orientation.w = math.cos(half_yaw)

        # Keep publishing the robot pose during active operation.
        # Once Task 4/5 exploration is complete, stop updating the RViz pose.
        if not (
            self.mode in ("greedy", "mcts")
            and self.phase == "done"
        ):
            self.current_pose_pub.publish(pose_msg)

        # Blue top-down robot marker for RViz.
        # Do not keep updating it after Task 4/5 exploration is complete.
        if (
            self.mode in ("greedy", "mcts")
            and self.phase == "done"
        ):
            return

        robot_marker = Marker()
        robot_marker.header.stamp = pose_msg.header.stamp
        robot_marker.header.frame_id = "map"

        robot_marker.ns = "assignment3_robot"
        robot_marker.id = 0
        robot_marker.type = Marker.CYLINDER
        robot_marker.action = Marker.ADD

        robot_marker.pose.position.x = float(self.pose[0])
        robot_marker.pose.position.y = float(self.pose[1])
        robot_marker.pose.position.z = 0.05
        robot_marker.pose.orientation.w = 1.0

        robot_marker.scale.x = 0.22
        robot_marker.scale.y = 0.22
        robot_marker.scale.z = 0.06

        robot_marker.color.r = 0.05
        robot_marker.color.g = 0.25
        robot_marker.color.b = 1.0
        robot_marker.color.a = 1.0

        self.robot_marker_pub.publish(robot_marker)

        # Record autonomous checkpoint-mission trajectory.
        if (
            self.mission_started_time is not None
            and self.phase == "checkpoint_mission"
        ):
            row, col = student.world_to_grid(
                self.pose[0],
                self.pose[1],
                self.resolution,
                self.origin,
            )

            # PNG is vertically flipped relative to occupancy-grid row order.
            image_u = int(col)
            image_v = int(self.height - 1 - row)

            if self.mission_index < len(self.mission_labels):
                target = self.mission_labels[self.mission_index]
            else:
                target = "DONE"

            self.mission_trajectory_rows.append((
                self.now_s() - self.mission_started_time,
                float(self.pose[0]),
                float(self.pose[1]),
                float(self.pose[2]),
                int(row),
                int(col),
                image_u,
                image_v,
                target,
            ))

        # Checkpoint mode uses the saved Task 1 map and does not update
        # occupancy online. The scan is still stored above for obstacle
        # safety in the waypoint controller.
        if self.mode == "checkpoint":
            return

        # -----------------------------------------------------------------
        # Mapping pose-stability gate
        #
        # A laser scan is only useful if it is projected using a reasonably
        # stable pose. During a fast turn, or immediately after a sizeable
        # EKF correction, inserting the scan can smear otherwise straight
        # walls across several occupancy-grid cells.
        # -----------------------------------------------------------------
        pose_for_scan = self.pose.copy()
        now = self.now_s()
        skip_mapping_scan = False
        skip_reason = ""

        if (
            self.mapping_previous_pose is not None
            and self.mapping_previous_time is not None
        ):
            dt = max(now - self.mapping_previous_time, 1e-3)

            dx = float(pose_for_scan[0] - self.mapping_previous_pose[0])
            dy = float(pose_for_scan[1] - self.mapping_previous_pose[1])
            position_change = math.hypot(dx, dy)

            yaw_change = wrap(
                float(pose_for_scan[2] - self.mapping_previous_pose[2])
            )
            angular_rate = abs(yaw_change) / dt

            max_angular_rate = float(
                self.get_parameter("mapping_max_angular_rate").value
            )
            max_pose_jump = float(
                self.get_parameter("mapping_max_pose_jump").value
            )

            if angular_rate > max_angular_rate:
                skip_mapping_scan = True
                skip_reason = (
                    f"turning at {angular_rate:.2f} rad/s "
                    f"(limit {max_angular_rate:.2f})"
                )

            elif position_change > max_pose_jump:
                skip_mapping_scan = True
                skip_reason = (
                    f"EKF pose changed {position_change:.3f} m "
                    f"(limit {max_pose_jump:.3f} m)"
                )

        # Always advance the reference pose. This allows mapping to resume
        # immediately once a turn/correction has settled.
        self.mapping_previous_pose = pose_for_scan
        self.mapping_previous_time = now

        if skip_mapping_scan:
            # Avoid flooding the terminal while the robot is continuously
            # rotating. At most one message per second is printed.
            if now - self.last_mapping_skip_log_time > 1.0:
                self.get_logger().warn(
                    f"Skipping occupancy update: {skip_reason}"
                )
                self.last_mapping_skip_log_time = now
            return

        mapping_active = (
            self.mode == "mapping"
            or (
                self.mode in ("greedy", "mcts")
                and self.phase == "exploring"
            )
        )

        if mapping_active:
            effective_max = min(
                float(msg.range_max),
                float(self.get_parameter("sensor_range").value),
            )
            self.log_odds = student.update_occupancy_grid(
                self.log_odds,
                pose_for_scan,
                msg.ranges,
                float(msg.angle_min),
                float(msg.angle_increment),
                float(msg.range_min),
                effective_max,
                self.resolution,
                self.origin,
            )
            self.occupancy = student.occupancy_from_log_odds(self.log_odds)
            self.publish_map(msg.header.stamp)

            # Update the visible frontier set whenever the live map changes.
            # Goal selection remains separate, so the robot does not constantly
            # switch targets while driving toward the current frontier.
            if self.mode in ("greedy", "mcts") and self.phase == "exploring":
                live_clusters = student.frontier_clustering(
                    self.occupancy,
                    min_cluster_size=int(
                        self.get_parameter("min_frontier_size").value
                    ),
                )
                self.publish_frontiers(live_clusters)

            if self.mode in ("greedy", "mcts") and self.exploration_start_time is None:
                self.exploration_start_time = self.now_s()
                self.exploration_start_distance = self.travel_distance

    # ------------------------------------------------------------ map I/O

    def on_save_map(self, request, response):
        del request
        try:
            stem = str(self.get_parameter("map_name").value).strip() or "manual_map"
            paths = list(self.save_map(stem))
            if self.mode == "mapping":
                paths.append(self.save_mapping_trajectory())
            response.success = True
            response.message = "Saved: " + ", ".join(str(p) for p in paths)
            self.get_logger().info(response.message)
        except Exception as exc:  # pragma: no cover - ROS error path
            response.success = False
            response.message = f"Map save failed: {exc}"
            self.get_logger().error(response.message)
        return response

    def save_map(self, stem: str) -> Tuple[FsPath, FsPath, FsPath]:
        stem = stem.replace(" ", "_")
        csv_path = self.output_dir / f"{stem}.csv"
        png_path = self.output_dir / f"{stem}.png"
        yaml_path = self.output_dir / f"{stem}.yaml"

        np.savetxt(csv_path, self.occupancy.astype(np.int16), fmt="%d", delimiter=",")

        image = np.full(self.occupancy.shape, 205, dtype=np.uint8)
        image[self.occupancy == 0] = 254
        image[self.occupancy == 100] = 0
        # Image row 0 is top; occupancy row 0 is the map's minimum y.
        cv2.imwrite(str(png_path), np.flipud(image))

        metadata = {
            "resolution": float(self.resolution),
            "origin": [float(self.origin[0]), float(self.origin[1])],
            "width": int(self.width),
            "height": int(self.height),
            "occupancy_values": {"unknown": -1, "free": 0, "occupied": 100},
            "csv": csv_path.name,
            "png": png_path.name,
        }
        yaml_path.write_text(yaml.safe_dump(metadata, sort_keys=False))
        return png_path, csv_path, yaml_path

    def save_mapping_trajectory(self) -> FsPath:
        path = self.output_dir / "trajectory.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s",
                "x_m",
                "y_m",
                "yaw_rad",
                "grid_row",
                "grid_col",
            ])
            writer.writerows(self.mapping_trajectory_rows)
        return path

    def _load_manual_map(self):
        csv_path = FsPath(
            os.path.expanduser(str(self.get_parameter("map_csv").value)))
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Saved map not found: {csv_path}. Run mapping.launch.py and save "
                "the map first, or pass map_csv:=/path/to/manual_map.csv")

        yaml_path = csv_path.with_suffix(".yaml")
        if yaml_path.exists():
            meta = yaml.safe_load(yaml_path.read_text()) or {}
            self.resolution = float(meta.get("resolution", self.resolution))
            ori = meta.get("origin", self.origin)
            self.origin = (float(ori[0]), float(ori[1]))

        occ = np.loadtxt(csv_path, delimiter=",", dtype=np.int16)
        if occ.ndim != 2:
            raise ValueError(f"Invalid occupancy CSV shape: {occ.shape}")
        self.occupancy = occ.astype(np.int8)
        self.height, self.width = self.occupancy.shape
        self.log_odds = np.zeros_like(self.occupancy, dtype=np.float32)
        self.publish_map(self.get_clock().now().to_msg())
        self.get_logger().info(f"Loaded saved map: {csv_path}")

    # ---------------------------------------------------------- map geometry

    def _inflated_blocked(self, radius_m: Optional[float] = None) -> np.ndarray:
        occupied = self.occupancy == 100
        radius_m = (float(self.get_parameter("inflation_radius").value)
                    if radius_m is None else float(radius_m))
        radius = max(0, int(math.ceil(radius_m / self.resolution)))
        if radius == 0:
            return occupied.copy()

        blocked = occupied.copy()
        offsets = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr * dr + dc * dc <= radius * radius:
                    offsets.append((dr, dc))
        for r, c in np.argwhere(occupied):
            for dr, dc in offsets:
                rr, cc = int(r + dr), int(c + dc)
                if 0 <= rr < self.height and 0 <= cc < self.width:
                    blocked[rr, cc] = True
        return blocked

    def _planning_map(self, start: GridCell, goal: GridCell) -> np.ndarray:
        planning = self.occupancy.copy()
        planning[self._inflated_blocked()] = 100
        if student.in_bounds(start, planning.shape):
            planning[start] = 0
        if student.in_bounds(goal, planning.shape):
            planning[goal] = 0
        return planning

    def _component_from(self, start: GridCell, planning: np.ndarray) -> Set[GridCell]:
        if not student.in_bounds(start, planning.shape):
            return set()
        free = planning == 0
        free[start] = True
        q = deque([start])
        seen: Set[GridCell] = {start}
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nxt = (r + dr, c + dc)
                if not student.in_bounds(nxt, planning.shape) or nxt in seen:
                    continue
                if not free[nxt]:
                    continue
                if dr != 0 and dc != 0:
                    if not free[r + dr, c] or not free[r, c + dc]:
                        continue
                seen.add(nxt)
                q.append(nxt)
        return seen

    def _nearest_free_cell(self, target: GridCell, planning: np.ndarray,
                           max_radius_cells: int = 10) -> Optional[GridCell]:
        if student.in_bounds(target, planning.shape) and planning[target] == 0:
            return target
        tr, tc = target
        best = None
        best_d2 = float("inf")
        for radius in range(1, max(1, int(max_radius_cells)) + 1):
            for r in range(tr - radius, tr + radius + 1):
                for c in range(tc - radius, tc + radius + 1):
                    if not student.in_bounds((r, c), planning.shape):
                        continue
                    if planning[r, c] != 0:
                        continue
                    d2 = (r - tr) ** 2 + (c - tc) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (r, c)
            if best is not None:
                return best
        return None

    # --------------------------------------------------------------- planning

    @staticmethod
    def _compress_path(path: Sequence[GridCell]) -> List[GridCell]:
        """Keep corners of an A* grid path; semantics still come from student A*."""
        path = list(path)
        if len(path) <= 2:
            return path
        out = [path[0]]
        prev_dir = None
        for i in range(1, len(path)):
            dr = path[i][0] - path[i - 1][0]
            dc = path[i][1] - path[i - 1][1]
            d = (int(np.sign(dr)), int(np.sign(dc)))
            if prev_dir is not None and d != prev_dir:
                out.append(path[i - 1])
            prev_dir = d
        out.append(path[-1])
        return out

    def plan_to(self, goal: WorldPoint) -> bool:
        if self.pose is None:
            return False
        start = student.world_to_grid(
            self.pose[0], self.pose[1], self.resolution, self.origin)
        goal_cell = student.world_to_grid(
            goal[0], goal[1], self.resolution, self.origin)
        if not student.in_bounds(start, self.occupancy.shape):
            return False
        if not student.in_bounds(goal_cell, self.occupancy.shape):
            return False
        # Goals are always chosen from known free space. If a frontier cell later
        # becomes occupied as the map improves, abandon it instead of clearing a
        # newly discovered obstacle for the planner.
        if self.occupancy[goal_cell] != 0:
            return False

        planning = self._planning_map(start, goal_cell)
        raw = student.astar(planning, start, goal_cell)
        self.path_cells = self._compress_path(raw)
        self.path_index = 1 if len(self.path_cells) > 1 else 0
        self.goal = goal
        self.goal_cell = goal_cell
        self.publish_path()
        return bool(raw)

    # ------------------------------------------------------ checkpoint mission

    def _generate_checkpoints(self, count: int) -> List[WorldPoint]:
        """Generate checkpoints distributed across the whole known map.

        The bounding box of all known free space is divided into a 2 x 3 grid.
        One safe, A*-reachable checkpoint is selected from each region.
        """

        if self.pose is None:
            return []

        start = student.world_to_grid(
            self.pose[0],
            self.pose[1],
            self.resolution,
            self.origin,
        )

        # --------------------------------------------------------------
        # Use the ENTIRE known free map to define the six regions.
        # Do not restrict the region layout to Home's connected component.
        # --------------------------------------------------------------
        all_free = [
            (int(r), int(c))
            for r, c in np.argwhere(self.occupancy == 0)
        ]

        if not all_free:
            return []

        rows = [cell[0] for cell in all_free]
        cols = [cell[1] for cell in all_free]

        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)

        row_edges = np.linspace(
            r_min,
            r_max + 1,
            3,
            dtype=int,
        )

        col_edges = np.linspace(
            c_min,
            c_max + 1,
            4,
            dtype=int,
        )

        # --------------------------------------------------------------
        # Reachability uses the NORMAL navigation inflation radius.
        #
        # The larger checkpoint-clearance radius must not be used here:
        # it can artificially seal doorways and split an otherwise
        # traversable map into disconnected regions.
        # --------------------------------------------------------------
        navigation_map = self.occupancy.copy()
        navigation_map[self._inflated_blocked()] = 100

        if student.in_bounds(start, navigation_map.shape):
            navigation_map[start] = 0

        reachable = self._component_from(
            start,
            navigation_map,
        )

        # --------------------------------------------------------------
        # Candidate safety is handled separately using the larger
        # checkpoint clearance.
        # --------------------------------------------------------------
        checkpoint_blocked = self._inflated_blocked(
            float(
                self.get_parameter(
                    "checkpoint_clearance"
                ).value
            )
        )

        min_home_distance = 1.0
        selected: List[GridCell] = []

        def distance_m(a: GridCell, b: GridCell) -> float:
            return self.resolution * math.hypot(
                a[0] - b[0],
                a[1] - b[1],
            )

        for region_row in range(2):
            for region_col in range(3):

                if len(selected) >= count:
                    break

                r0 = int(row_edges[region_row])
                r1 = int(row_edges[region_row + 1])

                c0 = int(col_edges[region_col])
                c1 = int(col_edges[region_col + 1])

                # Candidate must:
                #   1. lie inside this spatial region,
                #   2. be known free,
                #   3. be reachable using normal navigation inflation,
                #   4. have additional clearance from walls,
                #   5. not be right next to Home.
                region_candidates = [
                    cell
                    for cell in all_free
                    if r0 <= cell[0] < r1
                    and c0 <= cell[1] < c1
                    and cell in reachable
                    and not checkpoint_blocked[cell]
                    and distance_m(cell, start) >= min_home_distance
                ]

                if not region_candidates:
                    self.get_logger().warn(
                        f"No safe reachable checkpoint in region "
                        f"({region_row}, {region_col})"
                    )
                    continue

                # Prefer the spatial centre of the region so that each
                # checkpoint represents a genuinely different part of map.
                centre_r = 0.5 * (r0 + r1 - 1)
                centre_c = 0.5 * (c0 + c1 - 1)

                region_candidates.sort(
                    key=lambda cell:
                        (cell[0] - centre_r) ** 2
                        + (cell[1] - centre_c) ** 2
                )

                # Keep some randomness while preferring central candidates.
                pool_size = max(
                    1,
                    int(math.ceil(
                        0.30 * len(region_candidates)
                    )),
                )

                pool = region_candidates[:pool_size]

                index = int(
                    self.rng.integers(
                        0,
                        len(pool),
                    )
                )

                selected.append(pool[index])

        if len(selected) < count:
            self.get_logger().warn(
                f"Generated {len(selected)}/{count} distributed "
                "checkpoints because some regions contained no "
                "safe reachable free cells."
            )

        return [
            student.grid_to_world(
                r,
                c,
                self.resolution,
                self.origin,
            )
            for r, c in selected[:count]
        ]

    def _home_goal_on_map(self) -> Optional[WorldPoint]:
        if self.home is None or self.pose is None:
            return None
        start = student.world_to_grid(
            self.pose[0], self.pose[1], self.resolution, self.origin)
        home_cell = student.world_to_grid(
            self.home[0], self.home[1], self.resolution, self.origin)

        planning = self.occupancy.copy()
        planning[self._inflated_blocked()] = 100
        if student.in_bounds(start, planning.shape):
            planning[start] = 0
        component = self._component_from(start, planning)
        if not component:
            return None

        # Prefer the exact spawn cell when it remains safely free; otherwise use
        # the closest reachable free cell. This avoids inventing free space at home.
        if home_cell in component and planning[home_cell] == 0:
            safe_home = home_cell
        else:
            safe_home = min(
                component,
                key=lambda c: (c[0] - home_cell[0]) ** 2 + (c[1] - home_cell[1]) ** 2,
            )
        return student.grid_to_world(
            safe_home[0], safe_home[1], self.resolution, self.origin)

    def _begin_checkpoint_mission(self):
        if self.pose is None or self.scan is None or self.home is None:
            return
        count = int(
            self.get_parameter(
                "checkpoint_count"
            ).value
        )

        checkpoints = self._generate_checkpoints(count)
        if len(checkpoints) < count:
            self.get_logger().error(
                f"Could only generate {len(checkpoints)}/{count} reachable checkpoints. "
                "Map may be incomplete or too conservative.")
            self.phase = "done"
            self.stop()
            return

        home_goal = self._home_goal_on_map()
        if home_goal is None:
            self.get_logger().error("Could not find a safe reachable home cell on the map.")
            self.phase = "done"
            self.stop()
            return

        self.mission_waypoints = checkpoints + [home_goal]
        self.mission_labels = [f"CP{i+1}" for i in range(len(checkpoints))] + ["HOME"]
        self.mission_index = 0
        self.mission_started_time = self.now_s()
        self.mission_start_distance = self.travel_distance
        self.mission_trajectory_rows = []
        self.phase = "checkpoint_mission"
        self.publish_checkpoints(checkpoints, home_goal)

        self.get_logger().info(
            "Mission: " + " -> ".join(
                ["START"] + [
                    f"{name}({p[0]:+.2f},{p[1]:+.2f})"
                    for name, p in zip(self.mission_labels, self.mission_waypoints)
                ]))
        self._plan_current_mission_goal()

    def _plan_current_mission_goal(self) -> bool:
        if self.mission_index >= len(self.mission_waypoints):
            return False
        goal = self.mission_waypoints[self.mission_index]
        label = self.mission_labels[self.mission_index]
        if not self.plan_to(goal):
            self.get_logger().error(
                f"Student A* could not plan to {label} at {goal}. Mission halted.")
            self.stop()
            self.phase = "done"
            return False
        self.goal_set_time = self.now_s()
        self.last_plan_time = self.now_s()
        self.publish_goal(label)
        self.get_logger().info(f"Navigating to {label}")
        return True

    def _mission_tick(self):
        if self.pose is None or not self.mission_waypoints:
            return
        if self.mission_index >= len(self.mission_waypoints):
            self._complete_mission()
            return

        goal = self.mission_waypoints[self.mission_index]
        if self._goal_reached(goal):
            label = self.mission_labels[self.mission_index]
            self.get_logger().info(f"Reached {label}")
            self.stop()
            self.mission_index += 1
            if self.mission_index >= len(self.mission_waypoints):
                self._complete_mission()
            else:
                self._plan_current_mission_goal()
            return

        now = self.now_s()
        if now - self.last_plan_time >= float(self.get_parameter("replan_period").value):
            self.last_plan_time = now
            if not self.plan_to(goal):
                self.get_logger().error("Replanning failed during checkpoint mission.")
                self.phase = "done"
                self.stop()
                return
        self.follow_path()

    def _save_mission_checkpoints(self) -> FsPath:
        path = self.output_dir / "checkpoints.csv"

        with path.open("w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "name",
                "x_m",
                "y_m",
                "grid_row",
                "grid_col",
                "image_u",
                "image_v",
            ])

            # Initial Home position.
            all_points = []

            if self.home is not None:
                all_points.append(("HOME_START", self.home))

            for label, point in zip(
                self.mission_labels,
                self.mission_waypoints,
            ):
                all_points.append((label, point))

            for label, point in all_points:
                row, col = student.world_to_grid(
                    point[0],
                    point[1],
                    self.resolution,
                    self.origin,
                )

                writer.writerow([
                    label,
                    float(point[0]),
                    float(point[1]),
                    int(row),
                    int(col),
                    int(col),
                    int(self.height - 1 - row),
                ])

        return path


    def _save_mission_trajectory(self) -> FsPath:
        path = self.output_dir / "trajectory.csv"

        with path.open("w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "time_s",
                "x_m",
                "y_m",
                "yaw_rad",
                "grid_row",
                "grid_col",
                "image_u",
                "image_v",
                "target",
            ])

            writer.writerows(self.mission_trajectory_rows)

        return path


    def _complete_mission(self):
        elapsed = (self.now_s() - self.mission_started_time
                   if self.mission_started_time is not None else 0.0)
        distance = self.travel_distance - self.mission_start_distance
        phase_was = self.phase
        self.phase = "done"
        self.stop()
        self.clear_exploration_visuals()

        checkpoint_path = self._save_mission_checkpoints()
        trajectory_path = self._save_mission_trajectory()

        self.get_logger().info(
            f"Saved mission checkpoints: {checkpoint_path}"
        )
        self.get_logger().info(
            f"Saved mission trajectory: {trajectory_path}"
        )

        msg = (f"MISSION_COMPLETE mode={self.mode} phase={phase_was} "
               f"elapsed={elapsed:.2f}s travel={distance:.2f}m")

        # Make mission completion visually obvious in the terminal.
        GREEN = "\033[92m"
        BOLD = "\033[1m"
        RESET = "\033[0m"

        self.get_logger().info(
            f"{GREEN}{BOLD}{msg}{RESET}"
        )

        # Keep the ROS status topic plain text.
        self.pub_status.publish(String(data=msg))

    # ------------------------------------------------------------ exploration

    def _frontiers_for_navigation(self) -> List[List[GridCell]]:
        clusters = student.frontier_clustering(
            self.occupancy,
            min_cluster_size=int(self.get_parameter("min_frontier_size").value),
        )

        # Visualize the raw Task 3 frontier clusters before navigation
        # filtering so the frontier detector can be inspected directly.

        if not clusters:
            return []

        blocked = self._inflated_blocked()
        now = self.now_s()
        # Expire old blacklisting records.
        self.blacklisted_until = {
            c: t for c, t in self.blacklisted_until.items() if t > now
        }

        safe: List[List[GridCell]] = []
        for cluster in clusters:
            valid = []
            for cell in cluster:
                if blocked[cell]:
                    continue
                # Avoid retrying the vicinity of a recently unreachable goal.
                too_close = any(
                    abs(cell[0] - b[0]) <= 2 and abs(cell[1] - b[1]) <= 2
                    for b in self.blacklisted_until
                )
                if not too_close:
                    valid.append(cell)
            if valid:
                safe.append(valid)
        return safe

    def _choose_exploration_goal(self) -> Optional[WorldPoint]:
        if self.pose is None:
            return None
        clusters = self._frontiers_for_navigation()
        if not clusters:
            return None

        if self.mode == "greedy":
            return student.largest_frontier(
                clusters,
                self.resolution,
                self.origin,
            )

        goal = student.mcts_frontier(
            clusters,
            self.occupancy,
            self.pose,
            self.resolution,
            self.origin,
            sensor_range=float(self.get_parameter("sensor_range").value),
            num_iterations=int(self.get_parameter("mcts_iterations").value),
            rollout_depth=int(self.get_parameter("mcts_rollout_depth").value),
            exploration_constant=float(
                self.get_parameter("mcts_exploration_constant").value),
            gain_weight=float(self.get_parameter("mcts_gain_weight").value),
            distance_weight=float(self.get_parameter("mcts_distance_weight").value),
            max_frontiers=int(self.get_parameter("mcts_max_frontiers").value),
            rng=self.rng,
        )

        self.publish_mcts_debug()

        return goal

    def _exploration_tick(self):
        if self.pose is None or self.scan is None or self.exploration_start_time is None:
            return
        now = self.now_s()

        if self.goal is not None:
            if self._goal_reached(self.goal):
                self.stop()
                self.goal = None
                self.goal_cell = None
                self.path_cells = []
            elif now - self.goal_set_time > float(self.get_parameter("goal_timeout").value):
                if self.goal_cell is not None:
                    self.blacklisted_until[self.goal_cell] = now + 30.0
                self.get_logger().warn("Exploration goal timed out; choosing another frontier.")
                self.stop()
                self.goal = None
                self.goal_cell = None
                self.path_cells = []

        if self.goal is None:
            candidate = self._choose_exploration_goal()
            if candidate is None:
                self.stop()
                if self.frontier_missing_since is None:
                    self.frontier_missing_since = now
                runtime = now - self.exploration_start_time
                if (runtime >= float(self.get_parameter("min_runtime_before_finish").value)
                        and now - self.frontier_missing_since >= float(
                            self.get_parameter("no_frontier_timeout").value)):
                    self._finish_exploration()
                return

            self.frontier_missing_since = None
            if not self.plan_to(candidate):
                bad = student.world_to_grid(
                    candidate[0], candidate[1], self.resolution, self.origin)
                self.blacklisted_until[bad] = now + 30.0
                self.goal = None
                self.goal_cell = None
                self.path_cells = []
                self.stop()
                return
            self.goal_set_time = now
            self.last_plan_time = now
            self.publish_goal("FRONTIER")

        if self.goal is not None and now - self.last_plan_time >= float(
                self.get_parameter("replan_period").value):
            self.last_plan_time = now
            current_goal = self.goal
            if not self.plan_to(current_goal):
                if self.goal_cell is not None:
                    self.blacklisted_until[self.goal_cell] = now + 30.0
                self.goal = None
                self.goal_cell = None
                self.path_cells = []
                self.stop()
                return

        self.follow_path()

    def _finish_exploration(self):
        if self.phase != "exploring":
            return
        self.stop()
        self.exploration_end_time = self.now_s()
        self.exploration_end_distance = self.travel_distance
        elapsed = self.exploration_end_time - float(self.exploration_start_time)
        travel = self.exploration_end_distance - self.exploration_start_distance
        known = int(np.count_nonzero(self.occupancy != -1))

        stem = f"exploration_{self.mode}"
        self.save_map(stem)
        self._save_coverage_csv(stem)
        msg = (f"EXPLORATION_COMPLETE method={self.mode} elapsed={elapsed:.2f}s "
               f"travel={travel:.2f}m known_cells={known}")
        self.get_logger().info(msg)
        self.pub_status.publish(String(data=msg))

        # Exploration ends here. The final map and experiment trace
        # have already been saved above.
        self.goal = None
        self.goal_cell = None
        self.path_cells = []
        self.phase = "done"
        self.clear_exploration_visuals()

    # --------------------------------------------------------------- control

    def control_tick(self):
        if self.pose is None:
            return

        if self.mode == "mapping":
            # Manual teleop owns /cmd_vel in this mode.
            return

        if self.mode == "checkpoint":
            if self.phase == "checkpoint_wait" and self.scan is not None:
                self._begin_checkpoint_mission()
            elif self.phase == "checkpoint_mission":
                self._mission_tick()
            elif self.phase == "done":
                self.stop()
            return

        if self.phase == "exploring":
            self._exploration_tick()
        elif self.phase == "done":
            self.stop()

    def _goal_reached(self, goal: WorldPoint) -> bool:
        return (self.pose is not None and
                math.hypot(goal[0] - self.pose[0], goal[1] - self.pose[1])
                <= float(self.get_parameter("goal_tolerance").value))

    def _sector_min(self, center: float, half_width: float) -> float:
        if self.scan is None:
            return float("inf")
        best = float("inf")
        for i, rr in enumerate(self.scan.ranges):
            r = float(rr)
            if not math.isfinite(r) or r < float(self.scan.range_min):
                continue
            a = float(self.scan.angle_min) + i * float(self.scan.angle_increment)
            if abs(wrap(a - center)) <= half_width:
                best = min(best, r)
        return best

    def follow_path(self):
        if self.pose is None or not self.path_cells or self.path_index >= len(self.path_cells):
            self.publish_waypoint(None)
            self.stop()
            return

        tol = float(self.get_parameter("path_point_tolerance").value)
        while self.path_index < len(self.path_cells):
            r, c = self.path_cells[self.path_index]
            tx, ty = student.grid_to_world(r, c, self.resolution, self.origin)
            if math.hypot(tx - self.pose[0], ty - self.pose[1]) > tol:
                break
            self.path_index += 1
        if self.path_index >= len(self.path_cells):
            self.publish_waypoint(None)
            self.stop()
            return

        r, c = self.path_cells[self.path_index]

        # Visualize the waypoint currently being tracked.
        self.publish_waypoint((r, c))

        tx, ty = student.grid_to_world(r, c, self.resolution, self.origin)
        dx, dy = tx - self.pose[0], ty - self.pose[1]
        distance = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        err = wrap(heading - self.pose[2])

        max_v = float(
            self.get_parameter("max_linear_speed").value
        )

        max_w = float(self.get_parameter("max_angular_speed").value)
        cmd = Twist()
        cmd.angular.z = float(np.clip(2.4 * err, -max_w, max_w))
        if abs(err) < 0.75:
            cmd.linear.x = min(max_v, 0.9 * distance) * max(0.15, math.cos(err))

        # Provided safety layer only: it does not choose a route.  It prevents the
        # waypoint controller driving into a newly observed obstacle between A*
        # replans.
        emergency = float(self.get_parameter("emergency_stop_range").value)
        front = self._sector_min(0.0, math.radians(20.0))
        if front < emergency:
            cmd.linear.x = 0.0
            left = self._sector_min(math.radians(45.0), math.radians(25.0))
            right = self._sector_min(-math.radians(45.0), math.radians(25.0))
            direction = 1.0 if left >= right else -1.0
            cmd.angular.z = direction * max(0.45, abs(cmd.angular.z))

        self.pub_cmd.publish(cmd)

    def stop(self):
        self.pub_cmd.publish(Twist())

    # -------------------------------------------------------------- publishing

    def publish_map(self, stamp):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.info.resolution = float(self.resolution)
        msg.info.width = int(self.width)
        msg.info.height = int(self.height)
        msg.info.origin.position.x = float(self.origin[0])
        msg.info.origin.position.y = float(self.origin[1])
        msg.info.origin.orientation.w = 1.0
        msg.data = self.occupancy.astype(np.int16).ravel().tolist()
        self.pub_map.publish(msg)

    def publish_path(self):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        for r, c in self.path_cells:
            x, y = student.grid_to_world(r, c, self.resolution, self.origin)
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.pub_path.publish(msg)

    def publish_frontiers(
            self,
            clusters: Sequence[Sequence[GridCell]]) -> None:
        """Publish currently detected frontier cells for RViz."""

        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"

        m.ns = "assignment3_frontiers"
        m.id = 0
        m.type = Marker.POINTS

        # Reuse the same marker ID every update. Publishing an empty
        # marker therefore also clears old frontier points.
        m.action = Marker.ADD

        m.pose.orientation.w = 1.0

        # Slightly smaller than one occupancy-grid cell.
        point_size = max(0.04, 0.75 * float(self.resolution))
        m.scale.x = point_size
        m.scale.y = point_size

        # Cyan frontier cells.
        m.color.r = 0.0
        m.color.g = 0.9
        m.color.b = 1.0
        m.color.a = 0.95

        for cluster in clusters:
            for r, c in cluster:
                x, y = student.grid_to_world(
                    int(r),
                    int(c),
                    self.resolution,
                    self.origin,
                )

                p = Point()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.06

                m.points.append(p)

        self.pub_frontiers.publish(m)


    def publish_mcts_debug(self) -> None:
        """Visualize the MCTS search used for the current decision."""

        if self.mode != "mcts" or self.pose is None:
            return

        debug = getattr(student, "MCTS_DEBUG", None)
        if not debug:
            return

        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Clear the previous decision's visualization.
        clear = Marker()
        clear.header.stamp = stamp
        clear.header.frame_id = "map"
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        marker_id = 1

        # ------------------------------------------------------
        # Sampled rollout sequences: thin grey lines.
        # ------------------------------------------------------
        for rollout in debug.get("rollouts", []):
            cells = rollout.get("cells", [])
            if not cells:
                continue

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "mcts_rollouts"
            m.id = marker_id
            marker_id += 1

            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.025

            m.color.r = 0.55
            m.color.g = 0.55
            m.color.b = 0.55
            m.color.a = 0.35

            p0 = Point()
            p0.x = float(self.pose[0])
            p0.y = float(self.pose[1])
            p0.z = 0.08
            m.points.append(p0)

            for cell in cells:
                x, y = student.grid_to_world(
                    cell[0],
                    cell[1],
                    self.resolution,
                    self.origin,
                )

                pt = Point()
                pt.x = float(x)
                pt.y = float(y)
                pt.z = 0.08
                m.points.append(pt)

            arr.markers.append(m)

        # ------------------------------------------------------
        # Most-visited tree branch: thick magenta line.
        # ------------------------------------------------------
        best_sequence = debug.get("best_sequence", [])

        if best_sequence:
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "mcts_best_sequence"
            m.id = marker_id
            marker_id += 1

            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.075

            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 1.0
            m.color.a = 0.95

            p0 = Point()
            p0.x = float(self.pose[0])
            p0.y = float(self.pose[1])
            p0.z = 0.11
            m.points.append(p0)

            for cell in best_sequence:
                x, y = student.grid_to_world(
                    cell[0],
                    cell[1],
                    self.resolution,
                    self.origin,
                )

                pt = Point()
                pt.x = float(x)
                pt.y = float(y)
                pt.z = 0.11
                m.points.append(pt)

            arr.markers.append(m)

        # ------------------------------------------------------
        # Root action statistics.
        # ------------------------------------------------------
        for stat in debug.get("root_stats", []):
            cell = stat["cell"]

            x, y = student.grid_to_world(
                cell[0],
                cell[1],
                self.resolution,
                self.origin,
            )

            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = "map"
            m.ns = "mcts_root_stats"
            m.id = marker_id
            marker_id += 1

            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD

            m.pose.position.x = float(x)
            m.pose.position.y = float(y)
            m.pose.position.z = 0.30
            m.pose.orientation.w = 1.0

            m.scale.z = 0.12

            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 1.0

            m.text = (
                f"N={stat['visits']} "
                f"Q={stat['mean_value']:.1f}"
            )

            arr.markers.append(m)

        self.pub_mcts_debug.publish(arr)


    def clear_exploration_visuals(self) -> None:
        """Clear Task 4/5 visualization after exploration completes."""

        stamp = self.get_clock().now().to_msg()

        # ------------------------------------------------------
        # Clear frontier markers.
        # DELETEALL avoids depending on marker IDs/namespaces.
        # ------------------------------------------------------
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "map"
        marker.action = Marker.DELETEALL
        self.pub_frontiers.publish(marker)

        # ------------------------------------------------------
        # Clear selected frontier goal.
        # ------------------------------------------------------
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "map"
        marker.action = Marker.DELETEALL
        self.pub_goal.publish(marker)

        # ------------------------------------------------------
        # Clear current A* waypoint.
        # ------------------------------------------------------
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "map"
        marker.action = Marker.DELETEALL
        self.pub_waypoint.publish(marker)

        # ------------------------------------------------------
        # Clear A* path.
        # Publishing an empty Path removes the displayed route.
        # ------------------------------------------------------
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = "map"
        self.pub_path.publish(path)

        # ------------------------------------------------------
        # Clear custom blue robot marker.
        # ------------------------------------------------------
        if hasattr(self, "robot_marker_pub"):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = "map"
            marker.ns = "assignment3_robot"
            marker.id = 0
            marker.action = Marker.DELETE
            self.robot_marker_pub.publish(marker)

        # ------------------------------------------------------
        # Clear MCTS search visualization, if enabled.
        # ------------------------------------------------------
        if hasattr(self, "pub_mcts_debug"):
            arr = MarkerArray()

            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = "map"
            marker.action = Marker.DELETEALL

            arr.markers.append(marker)
            self.pub_mcts_debug.publish(arr)


    def publish_waypoint(
            self,
            cell: Optional[GridCell]) -> None:
        """Publish the current A* waypoint being tracked by the controller."""

        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"

        m.ns = "assignment3_waypoint"
        m.id = 0
        m.type = Marker.SPHERE

        if cell is None:
            m.action = Marker.DELETE
            self.pub_waypoint.publish(m)
            return

        m.action = Marker.ADD

        x, y = student.grid_to_world(
            int(cell[0]),
            int(cell[1]),
            self.resolution,
            self.origin,
        )

        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = 0.12
        m.pose.orientation.w = 1.0

        m.scale.x = 0.16
        m.scale.y = 0.16
        m.scale.z = 0.16

        # Magenta: current local A* waypoint.
        m.color.r = 1.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.color.a = 1.0

        self.pub_waypoint.publish(m)


    def publish_goal(self, text: str):
        if self.goal is None:
            return
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "map"
        m.ns = "assignment3_goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(self.goal[0])
        m.pose.position.y = float(self.goal[1])
        m.pose.position.z = 0.10
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.22
        m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.25, 0.0, 0.95
        self.pub_goal.publish(m)

    def publish_checkpoints(
            self,
            checkpoints: Sequence[WorldPoint],
            home: WorldPoint) -> None:
        """Publish checkpoint state for RViz.

        Blue circle : checkpoint not yet reached
        Black X     : checkpoint already reached
        Green circle: Home
        """

        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # ----------------------------------------------------------
        # Checkpoints
        # ----------------------------------------------------------
        for i, point in enumerate(checkpoints):

            # mission_index is the waypoint currently being pursued.
            # Therefore every checkpoint with index < mission_index
            # has already been reached.
            reached = i < self.mission_index

            if reached:
                # Remove the old blue checkpoint marker first.
                delete_old = Marker()
                delete_old.header.stamp = stamp
                delete_old.header.frame_id = "map"
                delete_old.ns = "assignment3_pending_checkpoints"
                delete_old.id = i
                delete_old.action = Marker.DELETE
                arr.markers.append(delete_old)

                # --------------------------------------------------
                # Black X for completed checkpoint.
                # --------------------------------------------------
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = "map"
                m.ns = "assignment3_completed_checkpoints"
                m.id = i
                m.type = Marker.LINE_LIST
                m.action = Marker.ADD

                m.scale.x = 0.06

                m.color.r = 0.0
                m.color.g = 0.0
                m.color.b = 0.0
                m.color.a = 1.0

                half = 0.14

                p1 = Point()
                p1.x = float(point[0] - half)
                p1.y = float(point[1] - half)
                p1.z = 0.10

                p2 = Point()
                p2.x = float(point[0] + half)
                p2.y = float(point[1] + half)
                p2.z = 0.10

                p3 = Point()
                p3.x = float(point[0] - half)
                p3.y = float(point[1] + half)
                p3.z = 0.10

                p4 = Point()
                p4.x = float(point[0] + half)
                p4.y = float(point[1] - half)
                p4.z = 0.10

                m.points = [
                    p1, p2,
                    p3, p4,
                ]

                arr.markers.append(m)

            else:
                # --------------------------------------------------
                # Blue circle for checkpoint not yet reached.
                # --------------------------------------------------
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = "map"

                m.ns = "assignment3_pending_checkpoints"
                m.id = i

                m.type = Marker.CYLINDER
                m.action = Marker.ADD

                m.pose.position.x = float(point[0])
                m.pose.position.y = float(point[1])
                m.pose.position.z = 0.04
                m.pose.orientation.w = 1.0

                m.scale.x = 0.24
                m.scale.y = 0.24
                m.scale.z = 0.08

                m.color.r = 0.1
                m.color.g = 0.4
                m.color.b = 1.0
                m.color.a = 0.95

                arr.markers.append(m)

        # ----------------------------------------------------------
        # Home stays green.
        # ----------------------------------------------------------
        home_marker = Marker()
        home_marker.header.stamp = stamp
        home_marker.header.frame_id = "map"

        home_marker.ns = "assignment3_home"
        home_marker.id = 2000

        home_marker.type = Marker.CYLINDER
        home_marker.action = Marker.ADD

        home_marker.pose.position.x = float(home[0])
        home_marker.pose.position.y = float(home[1])
        home_marker.pose.position.z = 0.04
        home_marker.pose.orientation.w = 1.0

        home_marker.scale.x = 0.24
        home_marker.scale.y = 0.24
        home_marker.scale.z = 0.08

        home_marker.color.r = 0.1
        home_marker.color.g = 0.9
        home_marker.color.b = 0.2
        home_marker.color.a = 0.95

        arr.markers.append(home_marker)

        self.pub_checkpoints.publish(arr)


    def _map_entropy(self) -> float:
        """Return mean binary occupancy entropy in bits per grid cell."""

        l = np.asarray(self.log_odds, dtype=np.float64)

        # Convert log-odds to occupancy probability.
        p = 1.0 / (1.0 + np.exp(-l))

        eps = 1e-12
        p = np.clip(p, eps, 1.0 - eps)

        h = -(
            p * np.log2(p)
            + (1.0 - p) * np.log2(1.0 - p)
        )

        return float(np.mean(h))


    def diagnostic_tick(self):
        # Checkpoint mode uses the static Task 1 map. Keep it available
        # independently of whether localization TF has initialized yet.
        if self.mode == "checkpoint":
            self.publish_map(self.get_clock().now().to_msg())

        if self.pose is None:
            return
        if len(self.mission_waypoints) >= 2:
            cps = self.mission_waypoints[:-1]
            self.publish_checkpoints(cps, self.mission_waypoints[-1])
        known = int(np.count_nonzero(self.occupancy != -1))
        if self.exploration_start_time is not None and self.phase in (
                "exploring", "done"):
            elapsed = max(0.0, self.now_s() - self.exploration_start_time)
            self.pub_elapsed.publish(Float32(data=float(elapsed)))
            self.pub_known.publish(Float32(data=float(known)))
            if self.phase == "exploring":
                exp_travel = (
                    self.travel_distance - self.exploration_start_distance
                )

                entropy = self._map_entropy()

                clusters = student.frontier_clustering(
                    self.occupancy,
                    min_cluster_size=int(
                        self.get_parameter("min_frontier_size").value
                    ),
                )

                frontier_clusters = len(clusters)
                frontier_cells = sum(len(c) for c in clusters)

                self.coverage_rows.append((
                    elapsed,
                    known,
                    exp_travel,
                    entropy,
                    frontier_cells,
                    frontier_clusters,
                ))

                self.pub_status.publish(String(
                    data=(
                        f"RUNNING method={self.mode} "
                        f"elapsed={elapsed:.1f}s "
                        f"known_cells={known} "
                        f"travel={exp_travel:.2f}m "
                        f"entropy={entropy:.4f}"
                    )
                ))

    def _save_coverage_csv(self, stem: str):
        path = self.output_dir / f"{stem}_coverage.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s",
                "known_cells",
                "travel_m",
                "map_entropy",
                "frontier_cells",
                "frontier_clusters",
            ])
            writer.writerows(self.coverage_rows)
        self.get_logger().info(f"Saved experiment trace: {path}")


def main(args=None):
    rclpy.init(args=args)
    node = Assignment3Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
