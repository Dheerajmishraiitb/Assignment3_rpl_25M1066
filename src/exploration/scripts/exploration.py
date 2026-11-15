#!/usr/bin/env python3
"""Algorithms students implement for CS7001 Assignment 3.

This is the only source file students need to edit for the assessed
algorithmic components of Assignment 3.

Complete only the designated student implementation regions marked
with the INSERT CODE HERE and CODE ENDS HERE comments.

Do not modify the provided function signatures or surrounding infrastructure.

Grid convention
---------------
``grid[row, col]`` maps to world coordinates

    x = origin_x + (col + 0.5) * resolution
    y = origin_y + (row + 0.5) * resolution

Occupancy convention
--------------------
``-1`` unknown, ``0`` free, ``100`` occupied.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

GridCell = Tuple[int, int]          # (row, col)
WorldPoint = Tuple[float, float]    # (x, y)


# ---------------------------------------------------------------------------
# Provided geometry / map helpers
# ---------------------------------------------------------------------------

def world_to_grid(x: float, y: float, resolution: float,
                  origin: Sequence[float]) -> GridCell:
    col = int(math.floor((float(x) - float(origin[0])) / float(resolution)))
    row = int(math.floor((float(y) - float(origin[1])) / float(resolution)))
    return row, col


def grid_to_world(row: int, col: int, resolution: float,
                  origin: Sequence[float]) -> WorldPoint:
    x = float(origin[0]) + (int(col) + 0.5) * float(resolution)
    y = float(origin[1]) + (int(row) + 0.5) * float(resolution)
    return x, y


def in_bounds(cell: GridCell, shape: Sequence[int]) -> bool:
    r, c = int(cell[0]), int(cell[1])
    return 0 <= r < int(shape[0]) and 0 <= c < int(shape[1])


def bresenham(start: GridCell, end: GridCell) -> List[GridCell]:
    """Integer grid cells crossed by a line, including both endpoints."""
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    cells: List[GridCell] = []

    dc = abs(c1 - c0)
    dr = -abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc + dr

    r, c = r0, c0
    while True:
        cells.append((r, c))
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 >= dr:
            err += dr
            c += sc
        if e2 <= dc:
            err += dc
            r += sr
    return cells


def occupancy_from_log_odds(log_odds: np.ndarray,
                            unknown_band: float = 0.15,
                            free_probability: float = 0.42,
                            occupied_probability: float = 0.65) -> np.ndarray:
    """Convert log-odds to ROS-style {-1, 0, 100} occupancy values."""
    p = 1.0 / (1.0 + np.exp(-np.clip(log_odds, -20.0, 20.0)))
    out = np.full(log_odds.shape, -1, dtype=np.int8)
    observed = np.abs(log_odds) >= float(unknown_band)
    out[observed & (p <= float(free_probability))] = 0
    out[observed & (p >= float(occupied_probability))] = 100
    return out


def _protect_scan_from_known_walls(
        log_odds: np.ndarray,
        pose: Sequence[float],
        ranges: Sequence[float],
        angle_min: float,
        angle_increment: float,
        range_min: float,
        range_max: float,
        resolution: float,
        origin: Sequence[float],
        occupied_stop_log_odds: float = 1.5) -> List[float]:
    """Prevent free-space carving through already-established walls.

    This is provided infrastructure and is intentionally outside the
    student implementation region.
    """
    x = float(pose[0])
    y = float(pose[1])
    yaw = float(pose[2])

    robot_cell = world_to_grid(x, y, resolution, origin)

    if not in_bounds(robot_cell, log_odds.shape):
        return list(ranges)

    protected = [float(v) for v in ranges]

    for i, measured in enumerate(protected):

        if math.isnan(measured) or measured < float(range_min):
            continue

        if math.isfinite(measured):
            ray_range = min(measured, float(range_max))
        else:
            ray_range = float(range_max)

        if ray_range <= float(range_min):
            continue

        beam_angle = (
            yaw
            + float(angle_min)
            + i * float(angle_increment)
        )

        end_x = x + ray_range * math.cos(beam_angle)
        end_y = y + ray_range * math.sin(beam_angle)

        end_cell = world_to_grid(
            end_x,
            end_y,
            resolution,
            origin,
        )

        ray = bresenham(robot_cell, end_cell)

        # Ignore the robot's own cell.
        for r, c in ray[1:]:

            if not in_bounds((r, c), log_odds.shape):
                break

            # A sufficiently established occupied cell blocks the beam.
            if float(log_odds[r, c]) >= float(occupied_stop_log_odds):

                wall_x = (
                    float(origin[0])
                    + (float(c) + 0.5) * float(resolution)
                )
                wall_y = (
                    float(origin[1])
                    + (float(r) + 0.5) * float(resolution)
                )

                wall_range = math.hypot(
                    wall_x - x,
                    wall_y - y,
                )

                # Do not alter the measurement if this occupied cell is
                # effectively the actual measured endpoint.
                if (
                    wall_range + 0.5 * float(resolution)
                    < ray_range
                ):
                    protected[i] = max(
                        float(range_min),
                        wall_range,
                    )

                break

    return protected


# =============================================================================
# Q1 -- Occupancy-grid mapping
# =============================================================================

def update_occupancy_grid(log_odds: np.ndarray,
                          pose: Sequence[float],
                          ranges: Sequence[float],
                          angle_min: float,
                          angle_increment: float,
                          range_min: float,
                          range_max: float,
                          resolution: float,
                          origin: Sequence[float],
                          l_free: float = -0.40,
                          l_occ: float = 0.85,
                          min_log_odds: float = -4.0,
                          max_log_odds: float = 4.0) -> np.ndarray:
    """Fuse one planar lidar scan into a log-odds occupancy map."""
    # Provided robustness layer: prevent rays from clearing free
    # space through already-established occupied cells.
    ranges = _protect_scan_from_known_walls(
        log_odds,
        pose,
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        resolution,
        origin,
    )

    # INSERT CODE HERE
    # CODE ENDS HERE
    return log_odds


# =============================================================================
# Q2 -- A* path planning
# =============================================================================

def astar(occupancy: np.ndarray,
          start: GridCell,
          goal: GridCell) -> List[GridCell]:
    """Plan an 8-connected collision-free path through known free cells.

    ``occupancy`` is already obstacle-inflated by the provided framework before this function
    is called.  Unknown cells are deliberately not traversable.
    """
    path: List[GridCell] = []
    # INSERT CODE HERE
    # CODE ENDS HERE
    return path


# =============================================================================
# Q3 -- Frontier detection and clustering
# =============================================================================

def frontier_clustering(occupancy: np.ndarray,
                        min_cluster_size: int = 4) -> List[List[GridCell]]:
    """Return 8-connected clusters of frontier cells.

    A frontier cell is a *known free* cell with at least one 4-connected unknown
    neighbour.  Connected frontier cells are grouped with an 8-connected BFS.
    """
    clusters: List[List[GridCell]] = []
    # INSERT CODE HERE
    # CODE ENDS HERE
    return clusters


# ---------------------------------------------------------------------------
# Provided frontier / MCTS helpers
# ---------------------------------------------------------------------------

def representative_cell(cluster: Sequence[GridCell]) -> GridCell:
    """Frontier cell nearest the component centroid."""
    pts = np.asarray(cluster, dtype=float)
    centroid = pts.mean(axis=0)
    idx = int(np.argmin(np.sum((pts - centroid) ** 2, axis=1)))
    return tuple(map(int, cluster[idx]))


def _unknown_visible_cells_from(cell: GridCell,
                                occupancy: np.ndarray,
                                radius_cells: int,
                                n_rays: int = 72) -> Set[GridCell]:
    """Optimistic visible-unknown set using only the *current* occupancy map.

    Known occupied cells stop a simulated ray.  Unknown cells are counted but do
    not stop the ray because their true occupancy is unavailable to the planner.
    This deliberately does not access Gazebo ground truth.
    """
    visible: Set[GridCell] = set()
    r0, c0 = map(int, cell)
    for k in range(max(8, int(n_rays))):
        a = 2.0 * math.pi * k / max(8, int(n_rays))
        rr = int(round(r0 + int(radius_cells) * math.sin(a)))
        cc = int(round(c0 + int(radius_cells) * math.cos(a)))
        for r, c in bresenham((r0, c0), (rr, cc))[1:]:
            if not in_bounds((r, c), occupancy.shape):
                break
            if occupancy[r, c] == 100:
                break
            if occupancy[r, c] == -1:
                visible.add((r, c))
    return visible


@dataclass
class _MCTSNode:
    """Small provided node container; students implement the MCTS loop."""
    position: GridCell
    remaining: Tuple[int, ...]
    revealed: frozenset
    parent: Optional["_MCTSNode"] = None
    action: Optional[int] = None
    step_reward: float = 0.0
    children: List["_MCTSNode"] = field(default_factory=list)
    untried: List[int] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0

    def __post_init__(self):
        if not self.untried:
            self.untried = list(self.remaining)


# =============================================================================
# Q4 -- Greedy largest-frontier exploration
# =============================================================================

def largest_frontier(
    frontier_clusters: Sequence[Sequence[GridCell]],
    resolution: float,
    origin: Sequence[float],
) -> Optional[WorldPoint]:
    """Return the representative point of the largest frontier cluster."""
    selected_frontier: Optional[WorldPoint] = None

    # INSERT CODE HERE
    # CODE ENDS HERE

    return selected_frontier



# MCTS diagnostics used only for visualization.
MCTS_DEBUG = {
    "candidate_cells": [],
    "rollouts": [],
    "best_sequence": [],
    "root_stats": [],
}


# =============================================================================
# Q5 -- Monte Carlo Tree Search exploration
# =============================================================================

def mcts_frontier(frontier_clusters: Sequence[Sequence[GridCell]],
                  occupancy: np.ndarray,
                  robot_pose: Sequence[float],
                  resolution: float,
                  origin: Sequence[float],
                  sensor_range: float,
                  num_iterations: int = 180,
                  rollout_depth: int = 4,
                  exploration_constant: float = 1.4,
                  gain_weight: float = 0.1,
                  distance_weight: float = 2.0,
                  max_frontiers: int = 18,
                  rng: Optional[np.random.Generator] = None
                  ) -> Optional[WorldPoint]:
    """Choose the next frontier using genuine MCTS.

    Tree state: current hypothetical frontier, unvisited frontier actions, and the
    set of unknown cells that previous actions in the simulated sequence would
    reveal.  The one-step reward is

        gain_weight * marginal_information_gain - distance_weight * travel_cost

    This makes later rewards *marginal*: two nearby viewpoints do not both receive
    credit for the same currently-unknown cells.
    """
    selected_frontier: Optional[WorldPoint] = None
    
    # INSERT CODE HERE
    # CODE ENDS HERE
    
    return selected_frontier
