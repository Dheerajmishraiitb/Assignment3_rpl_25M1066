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
    x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
    robot_cell = world_to_grid(x, y, resolution, origin)
    if not in_bounds(robot_cell, log_odds.shape):
        return log_odds

    # Collect the cells each beam passes through (free) and ends on (occupied)
    # first, so that every cell is updated at most once per scan no matter how
    # many beams cross it -- otherwise the cells near the robot would receive
    # hundreds of "free" updates from a single scan.
    free_cells: Set[GridCell] = set()
    occ_cells: Set[GridCell] = set()
    for i, measured in enumerate(ranges):
        measured = float(measured)
        if math.isnan(measured) or measured < float(range_min):
            continue                                  # invalid reading
        hit = math.isfinite(measured) and measured < float(range_max)
        ray_range = measured if hit else float(range_max)
        if ray_range <= float(range_min):
            continue

        beam = yaw + float(angle_min) + i * float(angle_increment)
        end_cell = world_to_grid(x + ray_range * math.cos(beam),
                                 y + ray_range * math.sin(beam),
                                 resolution, origin)
        ray = bresenham(robot_cell, end_cell)

        # every cell before the endpoint was seen through -> free
        for cell in ray[:-1]:
            if not in_bounds(cell, log_odds.shape):
                break
            free_cells.add(cell)
        else:
            last = ray[-1]
            if in_bounds(last, log_odds.shape):
                if hit:
                    occ_cells.add(last)               # the beam stopped here
                else:
                    free_cells.add(last)              # max range: no obstacle seen

    # A cell that is the endpoint of some beam is evidence of an obstacle even if
    # a neighbouring beam grazed through it.
    free_cells -= occ_cells
    if free_cells:
        r, c = zip(*free_cells)
        log_odds[r, c] += float(l_free)
    if occ_cells:
        r, c = zip(*occ_cells)
        log_odds[r, c] += float(l_occ)
    np.clip(log_odds, float(min_log_odds), float(max_log_odds), out=log_odds)
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
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    shape = occupancy.shape
    if not (in_bounds(start, shape) and in_bounds(goal, shape)):
        return path
    if occupancy[goal] != 0:
        return path                                  # goal must be known free
    if start == goal:
        return [start]

    def passable(cell: GridCell) -> bool:
        # only known-free cells; the start cell is allowed regardless, since
        # the robot is physically there even if inflation marked it
        return cell == start or occupancy[cell] == 0

    def h(cell: GridCell) -> float:
        # octile distance: admissible and consistent for 8-connected moves
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return (dr + dc) + (math.sqrt(2.0) - 2.0) * min(dr, dc)

    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
             (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]

    g_cost: Dict[GridCell, float] = {start: 0.0}
    came_from: Dict[GridCell, GridCell] = {}
    closed: Set[GridCell] = set()
    counter = 0                                      # FIFO tie-break in the heap
    open_heap = [(h(start), counter, start)]

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue                                 # stale heap entry
        if cur == goal:
            node = cur
            path = [node]
            while node in came_from:
                node = came_from[node]
                path.append(node)
            path.reverse()
            break
        closed.add(cur)

        for dr, dc, step in moves:
            nb = (cur[0] + dr, cur[1] + dc)
            if not in_bounds(nb, shape) or nb in closed or not passable(nb):
                continue
            # no corner cutting: a diagonal move needs both side cells free
            if dr != 0 and dc != 0:
                if not (passable((cur[0] + dr, cur[1])) and passable((cur[0], cur[1] + dc))):
                    continue
            ng = g_cost[cur] + step
            if ng < g_cost.get(nb, float("inf")):
                g_cost[nb] = ng
                came_from[nb] = cur
                counter += 1
                heapq.heappush(open_heap, (ng + h(nb), counter, nb))
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
    occ = np.asarray(occupancy)
    rows, cols = occ.shape

    # 1) frontier cells: known free with at least one 4-connected unknown neighbour
    free = occ == 0
    unknown = occ == -1
    unknown_nb = np.zeros_like(free)
    unknown_nb[1:, :] |= unknown[:-1, :]     # neighbour above (row - 1)
    unknown_nb[:-1, :] |= unknown[1:, :]     # neighbour below (row + 1)
    unknown_nb[:, 1:] |= unknown[:, :-1]     # neighbour left  (col - 1)
    unknown_nb[:, :-1] |= unknown[:, 1:]     # neighbour right (col + 1)
    frontier = free & unknown_nb

    # 2) group frontier cells with an 8-connected BFS; each cell joins one cluster
    visited = np.zeros_like(frontier)
    for r0, c0 in zip(*np.nonzero(frontier)):
        if visited[r0, c0]:
            continue
        visited[r0, c0] = True
        queue = deque([(int(r0), int(c0))])
        cluster: List[GridCell] = []
        while queue:
            r, c = queue.popleft()
            cluster.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols \
                            and frontier[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        queue.append((rr, cc))
        # 3) discard clusters that are too small to be worth visiting
        if len(cluster) >= int(min_cluster_size):
            clusters.append(cluster)
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
    best_cluster = None
    best_size = 0
    for cluster in frontier_clusters:
        # strict '>' keeps the FIRST cluster on equal-size ties; empty clusters
        # (size 0) can never be selected
        if len(cluster) > best_size:
            best_cluster, best_size = cluster, len(cluster)
    if best_cluster is not None:
        r, c = representative_cell(best_cluster)
        selected_frontier = grid_to_world(r, c, resolution, origin)
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
    if rng is None:
        rng = np.random.default_rng()
    for key in MCTS_DEBUG:
        MCTS_DEBUG[key] = []

    # ---- candidate actions: one representative viewpoint per frontier cluster.
    # Keep the largest clusters if there are more than max_frontiers.
    clusters = [c for c in frontier_clusters if len(c) > 0]
    if not clusters:
        return selected_frontier
    clusters.sort(key=len, reverse=True)
    clusters = clusters[:max(1, int(max_frontiers))]
    cand_cells: List[GridCell] = [representative_cell(c) for c in clusters]
    n_actions = len(cand_cells)
    MCTS_DEBUG["candidate_cells"] = list(cand_cells)

    # Unknown cells each viewpoint would (optimistically) reveal, from the
    # CURRENT map only -- computed once per decision and reused in the search.
    radius_cells = max(1, int(round(float(sensor_range) / float(resolution))))
    visible = [frozenset(_unknown_visible_cells_from(c, occupancy, radius_cells))
               for c in cand_cells]

    robot_cell = world_to_grid(robot_pose[0], robot_pose[1], resolution, origin)

    def travel_cost(a: GridCell, b: GridCell) -> float:
        # straight-line distance in metres (cheap proxy for path length)
        return math.hypot(a[0] - b[0], a[1] - b[1]) * float(resolution)

    def step(position: GridCell, revealed: frozenset, action: int):
        """One simulated frontier visit: reward and the new revealed set.
        Only cells NOT already revealed earlier in this sequence count
        (marginal information gain)."""
        new_cells = visible[action] - revealed
        reward = (float(gain_weight) * len(new_cells)
                  - float(distance_weight) * travel_cost(position, cand_cells[action]))
        return reward, revealed | new_cells

    def depth_of(node: _MCTSNode) -> int:
        d = 0
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    horizon = max(1, int(rollout_depth))
    root = _MCTSNode(position=robot_cell,
                     remaining=tuple(range(n_actions)),
                     revealed=frozenset())
    rng.shuffle(root.untried)

    def uct_child(node: _MCTSNode) -> _MCTSNode:
        # UCT with the exploitation term min-max normalised over the siblings,
        # so the exploration constant is meaningful whatever the reward scale.
        means = [ch.value / ch.visits for ch in node.children]
        lo, hi = min(means), max(means)
        span = hi - lo if hi > lo else 1.0
        log_n = math.log(max(node.visits, 1))
        best, best_score = None, -float("inf")
        for ch, m in zip(node.children, means):
            score = ((m - lo) / span
                     + float(exploration_constant) * math.sqrt(log_n / ch.visits))
            if score > best_score:
                best, best_score = ch, score
        return best

    max_debug_rollouts = 40
    for _ in range(max(1, int(num_iterations))):
        # 1) SELECTION: descend through fully-expanded nodes by UCT
        node = root
        while not node.untried and node.children:
            node = uct_child(node)

        # 2) EXPANSION: add one untried action (within the planning horizon)
        if node.untried and depth_of(node) < horizon:
            a = node.untried.pop()
            reward, revealed = step(node.position, node.revealed, a)
            child = _MCTSNode(position=cand_cells[a],
                              remaining=tuple(x for x in node.remaining if x != a),
                              revealed=revealed,
                              parent=node, action=a, step_reward=reward)
            rng.shuffle(child.untried)
            node.children.append(child)
            node = child

        # 3) ROLLOUT: random continuation of the frontier sequence up to the horizon
        rollout_return = 0.0
        position, revealed = node.position, node.revealed
        remaining = list(node.remaining)
        rollout_cells: List[GridCell] = []
        for _d in range(horizon - depth_of(node)):
            if not remaining:
                break
            a = remaining.pop(int(rng.integers(len(remaining))))
            reward, revealed = step(position, revealed, a)
            rollout_return += reward
            position = cand_cells[a]
            rollout_cells.append(position)
        if len(MCTS_DEBUG["rollouts"]) < max_debug_rollouts:
            seq, n = [], node
            while n.parent is not None:
                seq.append(n.position)
                n = n.parent
            MCTS_DEBUG["rollouts"].append({"cells": seq[::-1] + rollout_cells,
                                           "return": rollout_return})

        # 4) BACKPROPAGATION: each node stores the return from its own action on
        G = rollout_return
        while node is not None:
            G += node.step_reward
            node.visits += 1
            node.value += G
            node = node.parent

    if not root.children:
        return selected_frontier

    # executed action: the most-visited root child (mean value breaks ties)
    best = max(root.children, key=lambda ch: (ch.visits, ch.value / ch.visits))
    selected_frontier = grid_to_world(best.position[0], best.position[1],
                                      resolution, origin)

    # diagnostics for RViz
    MCTS_DEBUG["root_stats"] = [
        {"cell": ch.position, "visits": ch.visits,
         "mean_value": ch.value / max(ch.visits, 1)}
        for ch in root.children]
    seq, n = [], best
    while n is not None:
        seq.append(n.position)
        n = max(n.children, key=lambda ch: ch.visits) if n.children else None
    MCTS_DEBUG["best_sequence"] = seq
    # CODE ENDS HERE

    return selected_frontier
