#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Import the student's exploration.py
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "src" / "exploration" / "scripts"

sys.path.insert(0, str(SCRIPT_DIR))

import exploration


def normalize_clusters(clusters):
    """Convert clusters to sets so BFS/output ordering does not matter."""
    return [set((int(r), int(c)) for r, c in cluster) for cluster in clusters]


def show_map(occupancy, clusters):
    """Print a compact ASCII visualization of occupancy and frontier clusters."""

    display = np.full(occupancy.shape, "?", dtype="<U1")

    display[occupancy == 0] = "."
    display[occupancy == 100] = "#"

    for i, cluster in enumerate(clusters):
        symbol = str((i + 1) % 10)

        for r, c in cluster:
            if (
                0 <= int(r) < occupancy.shape[0]
                and 0 <= int(c) < occupancy.shape[1]
            ):
                display[int(r), int(c)] = symbol

    print()

    # Reverse row order so the display matches the usual map view:
    # larger grid row / larger y appears toward the top.
    for row in reversed(display):
        print(" ".join(row))

    print()


def run_test(name, fn):
    try:
        fn()
        print(f"[PASS] {name}")
        return True
    except AssertionError as exc:
        print(f"[FAIL] {name}")
        print(f"       {exc}")
        return False
    except Exception as exc:
        print(f"[ERROR] {name}")
        print(f"        {type(exc).__name__}: {exc}")
        return False


# ===========================================================================
# TEST 1
# One rectangular explored region surrounded by unknown space.
#
# Expected:
#   - exactly one frontier cluster
#   - frontier cells form the outer free-cell boundary
# ===========================================================================

def test_single_frontier_cluster():

    occ = np.full((10, 14), -1, dtype=np.int8)

    # 4 x 6 known-free rectangle
    occ[3:7, 4:10] = 0

    clusters = exploration.frontier_clustering(
        occ,
        min_cluster_size=1,
    )

    show_map(occ, clusters)

    clusters = normalize_clusters(clusters)

    assert len(clusters) == 1, (
        f"expected 1 frontier cluster, got {len(clusters)}"
    )

    cluster = clusters[0]

    expected = set()

    for r in range(3, 7):
        for c in range(4, 10):

            # A frontier cell is free and has at least one
            # 4-connected unknown neighbour.
            for dr, dc in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ]:
                nr = r + dr
                nc = c + dc

                if (
                    0 <= nr < occ.shape[0]
                    and 0 <= nc < occ.shape[1]
                    and occ[nr, nc] == -1
                ):
                    expected.add((r, c))
                    break

    assert cluster == expected, (
        "frontier-cell membership is incorrect"
    )


# ===========================================================================
# TEST 2
# Two disconnected explored regions.
#
# Expected:
#   - exactly two frontier clusters
# ===========================================================================

def test_two_frontier_clusters():

    occ = np.full((12, 18), -1, dtype=np.int8)

    occ[2:5, 2:6] = 0
    occ[7:10, 11:16] = 0

    clusters = exploration.frontier_clustering(
        occ,
        min_cluster_size=1,
    )

    show_map(occ, clusters)

    clusters = normalize_clusters(clusters)

    assert len(clusters) == 2, (
        f"expected 2 frontier clusters, got {len(clusters)}"
    )

    sizes = sorted(len(c) for c in clusters)

    assert all(size > 0 for size in sizes), (
        "one or more returned clusters are empty"
    )


# ===========================================================================
# TEST 3
# Tiny frontier cluster should be removed by min_cluster_size.
#
# Expected:
#   - only the larger cluster remains
# ===========================================================================

def test_min_cluster_size_filter():

    occ = np.full((12, 18), 100, dtype=np.int8)

    # Large free strip with unknown directly below it.
    occ[3:6, 2:9] = 0
    occ[6, 2:9] = -1

    # Tiny isolated free cell with one unknown neighbour.
    occ[9, 14] = 0
    occ[10, 14] = -1

    clusters = exploration.frontier_clustering(
        occ,
        min_cluster_size=3,
    )

    show_map(occ, clusters)

    clusters = normalize_clusters(clusters)

    assert len(clusters) == 1, (
        f"expected tiny cluster to be filtered, got "
        f"{len(clusters)} clusters"
    )

    assert len(clusters[0]) >= 3, (
        "remaining cluster is smaller than min_cluster_size"
    )


# ===========================================================================
# TEST 4
# Fully known map.
#
# Expected:
#   - no frontier cells
# ===========================================================================

def test_no_frontiers():

    occ = np.zeros((10, 10), dtype=np.int8)

    # Add a known occupied wall. There is still no unknown space.
    occ[5, :] = 100

    clusters = exploration.frontier_clustering(
        occ,
        min_cluster_size=1,
    )

    show_map(occ, clusters)

    assert len(clusters) == 0, (
        f"expected no frontier clusters, got {len(clusters)}"
    )


# ===========================================================================
# TEST 5
# Verify that diagonal unknown cells alone do NOT create a frontier.
#
# Frontier detection uses 4-connected unknown neighbours.
#
# Expected:
#   - centre free cell is not a frontier if its only unknown neighbour
#     is diagonal
# ===========================================================================

def test_frontier_uses_four_connected_unknown():

    occ = np.full((5, 5), 100, dtype=np.int8)

    occ[2, 2] = 0

    # Only diagonal unknown.
    occ[1, 1] = -1

    clusters = exploration.frontier_clustering(
        occ,
        min_cluster_size=1,
    )

    show_map(occ, clusters)

    flattened = set()

    for cluster in clusters:
        for cell in cluster:
            flattened.add(tuple(cell))

    assert (2, 2) not in flattened, (
        "a free cell with only a diagonal unknown neighbour "
        "must not be classified as frontier"
    )


# ===========================================================================
# Main
# ===========================================================================

def main():

    tests = [
        (
            "single frontier cluster",
            test_single_frontier_cluster,
        ),
        (
            "two disconnected frontier clusters",
            test_two_frontier_clusters,
        ),
        (
            "minimum cluster-size filtering",
            test_min_cluster_size_filter,
        ),
        (
            "fully known map has no frontiers",
            test_no_frontiers,
        ),
        (
            "frontier detection uses 4-connected unknown neighbours",
            test_frontier_uses_four_connected_unknown,
        ),
    ]

    print()
    print("=" * 62)
    print("CS7001 Assignment 3 — Task 3 Frontier Self-Check")
    print("=" * 62)
    print()

    passed = 0

    for name, fn in tests:
        if run_test(name, fn):
            passed += 1

    print()
    print("-" * 62)
    print(f"Passed {passed}/{len(tests)} tests.")

    if passed == len(tests):
        print("ALL TASK 3 SELF-CHECKS PASSED")
        print("-" * 62)
        return 0

    print("TASK 3 SELF-CHECK FAILED")
    print("-" * 62)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
