#!/usr/bin/env python3

"""Offline self-checks for Task 4 largest-frontier greedy selection."""

from pathlib import Path
import importlib.util
import sys


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "exploration" / "scripts" / "exploration.py"

spec = importlib.util.spec_from_file_location(
    "exploration",
    MODULE,
)

exploration = importlib.util.module_from_spec(spec)
sys.modules["exploration"] = exploration
spec.loader.exec_module(exploration)


RESOLUTION = 1.0
ORIGIN = (0.0, 0.0)


def expected_world(cluster):
    cell = exploration.representative_cell(cluster)
    return exploration.grid_to_world(
        cell[0],
        cell[1],
        RESOLUTION,
        ORIGIN,
    )


def assert_point_close(actual, expected, name):
    if actual is None:
        raise AssertionError(
            f"{name}: expected {expected}, got None"
        )

    if (
        abs(actual[0] - expected[0]) > 1e-9
        or abs(actual[1] - expected[1]) > 1e-9
    ):
        raise AssertionError(
            f"{name}: expected {expected}, got {actual}"
        )


def test_selects_largest_cluster():
    small = [
        (1, 1),
        (1, 2),
    ]

    large = [
        (5, 5),
        (5, 6),
        (5, 7),
        (6, 5),
        (6, 6),
    ]

    result = exploration.largest_frontier(
        [small, large],
        RESOLUTION,
        ORIGIN,
    )

    assert_point_close(
        result,
        expected_world(large),
        "selects largest cluster",
    )


def test_size_not_distance():
    # The small frontier is deliberately much closer to the origin.
    # Largest-frontier greedy must still select the larger cluster.
    close_small = [
        (0, 0),
        (0, 1),
    ]

    far_large = [
        (20, 20),
        (20, 21),
        (20, 22),
        (21, 20),
        (21, 21),
        (21, 22),
    ]

    result = exploration.largest_frontier(
        [close_small, far_large],
        RESOLUTION,
        ORIGIN,
    )

    assert_point_close(
        result,
        expected_world(far_large),
        "frontier size is primary criterion",
    )


def test_first_cluster_wins_equal_size_tie():
    first = [
        (2, 2),
        (2, 3),
        (3, 2),
    ]

    second = [
        (8, 8),
        (8, 9),
        (9, 8),
    ]

    result = exploration.largest_frontier(
        [first, second],
        RESOLUTION,
        ORIGIN,
    )

    assert_point_close(
        result,
        expected_world(first),
        "equal-size tie",
    )


def test_empty_clusters_are_ignored():
    largest = [
        (4, 4),
        (4, 5),
        (4, 6),
        (5, 4),
    ]

    result = exploration.largest_frontier(
        [[], largest, []],
        RESOLUTION,
        ORIGIN,
    )

    assert_point_close(
        result,
        expected_world(largest),
        "empty clusters ignored",
    )


def test_no_frontiers_returns_none():
    result = exploration.largest_frontier(
        [],
        RESOLUTION,
        ORIGIN,
    )

    if result is not None:
        raise AssertionError(
            f"expected None, got {result}"
        )


TESTS = [
    ("selects largest frontier cluster", test_selects_largest_cluster),
    ("size overrides geometric distance", test_size_not_distance),
    ("equal-size tie is deterministic", test_first_cluster_wins_equal_size_tie),
    ("empty clusters are ignored", test_empty_clusters_are_ignored),
    ("no frontiers returns None", test_no_frontiers_returns_none),
]


def main():
    passed = 0

    print("\nTASK 4 — LARGEST-FRONTIER SELF-CHECK")
    print("=" * 60)

    for name, test in TESTS:
        try:
            test()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f"[FAIL] {name}")
            print(f"       {exc}")

    print("\n" + "-" * 60)
    print(f"Passed {passed}/{len(TESTS)} tests.")

    if passed == len(TESTS):
        print("ALL TASK 4 SELF-CHECKS PASSED")
        print("-" * 60)
        return 0

    print("TASK 4 SELF-CHECK FAILED")
    print("-" * 60)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
