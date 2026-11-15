#!/usr/bin/env python3
"""CS7001 Assignment 1 -- EKF localization with ArUco landmarks."""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

FLAG_SENSOR_FAILURE = -1
FLAG_TIME_MISSMATCH = -2

HERE = os.path.dirname(os.path.realpath(__file__))


def _find_dataset(name="my_run.p"):
    """Locate the recording whether this file sits beside data/ or in scripts/."""
    for candidate in (os.path.join(HERE, "data", name),                     # flat
                      os.path.join(os.path.dirname(HERE), "data", name)):    # package
        if os.path.exists(candidate):
            return candidate
    return os.path.join(os.path.dirname(HERE), "data", name)   # best guess for the error


DEFAULT_DATASET = _find_dataset()

ALPHA = [0.30, 0.0, 0.1109, 0.0445]

STEP_TRANS_M = 0.05
STEP_ROT_DEG = 5.0

RANGE_SCALE, RANGE_OFFSET = 1.02558, -0.00970

INNOVATION_GATE = 80.0

MIN_DATASET_IDS = 20
MIN_DATASET_M = 30.0

ROOM_F_IDS = [7, 11, 16]


# =============================================================================
# TASK 1 -- Motion model and inverse motion model
# =============================================================================

def inverse_motion_model(pose_prev, pose):
    """Relative motion between two poses: (delta_rot1, delta_trans, delta_rot2)."""
    x_prev, y_prev, theta_prev = pose_prev[0], pose_prev[1], pose_prev[2]
    x, y, theta = pose[0], pose[1], pose[2]

    trans = np.sqrt((x - x_prev) ** 2 + (y - y_prev) ** 2)

    if trans < 1e-3:
        rot1 = 0.0
        rot2 = wrapToPi(theta - theta_prev)
    else:
        rot1 = wrapToPi(np.arctan2(y - y_prev, x - x_prev) - theta_prev)
        rot2 = wrapToPi(theta - theta_prev - rot1)
    return np.array([rot1, trans, rot2])


def probability_density(mean, variance):
    """Zero-mean normal density evaluated at `mean`, with the given VARIANCE."""
    variance = max(variance, 0.00001)          # avoid division by zero
    return (1.0 / np.sqrt(2.0 * np.pi * variance)) \
        * np.exp(-0.5 * (mean ** 2) / variance)
    return density


def motion_model(x_t, x_t_1, u_t, alpha, marginalise_p3=False):
    """p(x_t | u_t, x_t_1) for the odometry motion model."""
    rot1, trans, rot2 = inverse_motion_model(u_t[0], u_t[1])
    rot1_hat, trans_hat, rot2_hat = inverse_motion_model(x_t_1, x_t)

    a1, a2, a3, a4 = alpha[0], alpha[1], alpha[2], alpha[3]

    std1 = a1 * np.abs(rot1_hat) + a2 * np.abs(trans_hat)
    std2 = a3 * np.abs(trans_hat) + a4 * (np.abs(rot1_hat) + np.abs(rot2_hat))
    std3 = a1 * np.abs(rot2_hat) + a2 * np.abs(trans_hat)

    p1 = probability_density(rot1 - rot1_hat, std1 ** 2)
    p2 = probability_density(trans - trans_hat, std2 ** 2)
    p3 = probability_density(rot2 - rot2_hat, std3 ** 2)
    return p1 * p2 if marginalise_p3 else p1 * p2 * p3


def plot_posterior_belief(x_t_1, u_t, alpha, ret=False, marginalise_p3=False,
                          N_theta=8, size=150, res=0.01):
    """Evaluate the posterior over a grid centred on x_t_1 and show it."""
    gridmap = np.zeros([size, size])
    origin = [np.floor(size / 2), np.floor(size / 2)]

    if not marginalise_p3:
        dtheta = 2 * np.pi / N_theta
        for i in range(size):
            for j in range(size):
                x_t = list(x_t_1).copy()
                x_t[0] += res * (j - origin[0])
                x_t[1] += res * (i - origin[1])
                for k in range(N_theta):
                    x_t[2] = k * dtheta
                    gridmap[i, j] += motion_model(x_t, x_t_1, u_t, alpha) * dtheta
    else:
        for i in range(size):
            for j in range(size):
                x_t = list(x_t_1).copy()
                x_t[0] += res * (j - origin[0])
                x_t[1] += res * (i - origin[1])
                gridmap[i, j] = motion_model(x_t, x_t_1, u_t, alpha,
                                             marginalise_p3=True)

    if ret:
        return gridmap

    assert np.sum(gridmap) != 0, "Sum over gridmap is zero -- not implemented?"
    gridmap = gridmap / np.sum(gridmap)
    plt.figure(figsize=(6, 6))
    plt.imshow(1 - gridmap, cmap="gray",
               extent=[-res * (size - origin[0]), res * (size - origin[0]),
                       -res * (size - origin[1]), res * (size - origin[1])])
    plt.title("Posterior belief p(x_t | u_t, x_t-1)"
              + ("  [p3 marginalised]" if marginalise_p3 else
                 f"  [integrated over {N_theta} orientations]"))
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    return gridmap


# =============================================================================
# TASK 2 -- Observation model and likelihood field
# =============================================================================

def landmark_observation_model(z, sigma_r, b, x):
    """p(z | x, m) for a range-only beacon at b observed from robot pose x."""
    return normpdf(z, dist(b, x), sigma_r)
    return density


def cell_to_world(i, j, gridmap, extent):
    """Map a cell index to a world (x, y) in metres.

    extent = (xmin, xmax, ymin, ymax). Pass extent=None to get the cell index back
    unchanged, which is the plain-grid behaviour of the original exercises.
    """
    if extent is None:
        return np.array([i, j], dtype=float)
    xmin, xmax, ymin, ymax = extent
    nx, ny = gridmap.shape
    return np.array([xmin + (xmax - xmin) * i / max(nx - 1, 1),
                     ymin + (ymax - ymin) * j / max(ny - 1, 1)])


def observation_likelihood(z, sigma_z, b, gridmap, extent=None):
    """Evaluate the observation model at every cell, normalised to sum to 1.

    With extent given, cells are real positions in METRES and `b` is a real marker
    position from the map -- so the field is over the actual house the robot drives in,
    not an abstract grid.
    """
    size_x, size_y = gridmap.shape
    likelihood = np.zeros_like(gridmap, dtype=float)
    for i in range(size_x):
        for j in range(size_y):
            likelihood[i, j] = landmark_observation_model(
                z, sigma_z, b, cell_to_world(i, j, gridmap, extent))
    return likelihood / np.sum(likelihood)
    assert np.sum(likelihood) != 0, \
        "Sum over likelihood is zero -- error or no implementation!"
    return likelihood / np.sum(likelihood)


def joint_observation_likelihood(z, sigma_z, b, gridmap, extent=None):
    """Joint likelihood of several independent range observations."""
    joint = np.ones_like(gridmap, dtype=float)
    for i in range(np.shape(z)[0]):
        joint = np.multiply(joint, observation_likelihood(
            z[i], sigma_z[i], b[i], gridmap, extent))
    return joint / np.sum(joint)
    assert np.sum(joint) != 0, \
        "Sum over joint likelihood is zero -- error or no implementation!"
    return joint / np.sum(joint)


def landmark_observation_model_rb(z_range, z_bearing, sigma_r, sigma_phi, b, x, theta):
    """p(z | x, m) for a RANGE-AND-BEARING observation -- the sensor ekf_correct uses.

    This is the same h(x) that the correction step linearises:
        range   = ||b - x||
        bearing = atan2(by - y, bx - x) - theta
    Bearing only carries information if the heading `theta` is known, which is exactly
    why heading is part of the filter's state.
    """
    dx, dy = b[0] - x[0], b[1] - x[1]
    r_hat = np.hypot(dx, dy)
    phi_hat = wrapToPi(np.arctan2(dy, dx) - theta)
    return (normpdf(z_range, r_hat, sigma_r)
            * normpdf(wrapToPi(z_bearing - phi_hat), 0.0, sigma_phi))


# =============================================================================
# TASK 3 -- EKF prediction
# =============================================================================
    return density


# =============================================================================
# TASK 3 -- EKF prediction
# =============================================================================

def ekf_predict(mu, S, u, R):
    """Propagate the Gaussian belief through the odometry motion model."""
    d_rot1, d_trans, d_rot2 = u[0], u[1], u[2]
    theta = mu[2]

    mu = mu + np.array([d_trans * np.cos(theta + d_rot1),
                        d_trans * np.sin(theta + d_rot1),
                        d_rot1 + d_rot2])
    mu[2] = wrapToPi(mu[2])

    # dg/dx
    G = np.array([[1, 0, -d_trans * np.sin(theta + d_rot1)],
                  [0, 1, d_trans * np.cos(theta + d_rot1)],
                  [0, 0, 1]])

    # dg/du
    V = np.array([[-d_trans * np.sin(theta + d_rot1), np.cos(theta + d_rot1), 0],
                  [d_trans * np.cos(theta + d_rot1), np.sin(theta + d_rot1), 0],
                  [1, 0, 1]])

    a1, a2, a3, a4 = ALPHA
    s_rot1 = a1 * abs(d_rot1) + a2 * abs(d_trans)
    s_trans = a3 * abs(d_trans) + a4 * (abs(d_rot1) + abs(d_rot2))
    s_rot2 = a1 * abs(d_rot2) + a2 * abs(d_trans)
    M = np.diag(np.array([s_rot1, s_trans, s_rot2]) ** 2)

    S = G @ S @ G.T + V @ M @ V.T + R
    return mu, S


# =============================================================================
# TASK 4 -- EKF correction
# =============================================================================

def ekf_correct(mu, S, z, Q, M, gate=INNOVATION_GATE):
    """Fold range/bearing observations of known landmarks into the belief."""
    # One update per observed landmark in this measurement.
    for i in range(z.shape[1]):
        landmark_id = int(z[2, i])
        m = M[landmark_id, :]

        delta = np.array([m[0] - mu[0], m[1] - mu[1]])
        q = float(delta @ delta)
        sq = np.sqrt(q)

        z_hat = np.array([sq, wrapToPi(np.arctan2(delta[1], delta[0]) - mu[2])])

        H = (1.0 / q) * np.array(
            [[-sq * delta[0], -sq * delta[1], 0.0],
             [delta[1], -delta[0], -q]])

        S_inn = H @ S @ H.T + Q
        innovation = np.array([z[0, i] - z_hat[0],
                               wrapToPi(z[1, i] - z_hat[1])])

        if gate is not None:
            d2 = float(innovation @ np.linalg.solve(S_inn, innovation))
            if d2 > gate:
                continue

        K = S @ H.T @ np.linalg.inv(S_inn)
        mu = mu + K @ innovation
        mu[2] = wrapToPi(mu[2])
        S = (np.eye(3) - K @ H) @ S

    return mu, S


# =============================================================================
# TASK 5 -- EKF localization, full loop
# =============================================================================

def init_params():
    """Initial belief and the two noise matrices, all measured from the recording."""
    sigma_x, sigma_y, sigma_theta = 0.01, 0.01, np.deg2rad(0.5)
    R = np.diag(np.array([sigma_x, sigma_y, sigma_theta]) ** 2)

    sigma_r, sigma_phi = 0.02, np.deg2rad(0.5)
    Q = np.diag(np.array([sigma_r, sigma_phi]) ** 2)

    mu = np.array([-2.507, 2.800, 0.0])
    S = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(2.0) ** 2])
    return mu, S, R, Q


def run_ekf_localization(dataset_path=DEFAULT_DATASET, plot=True,
                        step_size=1, last_step=None, sensor_specs=None,
                        params=None, gate=INNOVATION_GATE, blocked_ids=None):
  
    dataset = load_dataset(dataset_path)
    if blocked_ids:
        blocked = {int(i) for i in blocked_ids}
        kept = []
        for z in dataset["z"]:
            arr = np.asarray(z)
            if arr.size == 0 or arr.ndim != 2:
                kept.append(z)
                continue
            cols = [k for k in range(arr.shape[1])
                    if int(arr[2, k]) not in blocked]
            # An empty (3,0) array reads as no observation below, which is the intent.
            kept.append(arr[:, cols] if cols else np.empty((3, 0)))
        dataset["z"] = kept
    max_steps = len(dataset["odom"])

    odom_timesteps, sensor_timesteps = get_timesteps_system(
        sensor_specs, step_size, last_step, max_steps, plot=plot)

    mu, S, R, Q = init_params() if params is None else params
    if params is None:
        mu = np.asarray(dataset["gt"][0], dtype=float).copy()
    M = dataset["M"]
    seam = dataset.get("seam")

    axes = plot_state(M) if plot else None
    trace = [mu.copy()]

    for now, prev, sensor_t in zip(odom_timesteps[1:], odom_timesteps[:-1],
                                   sensor_timesteps[1:]):
        if seam is not None and now == seam:
            trace.append(mu.copy())
            continue

        u = inverse_motion_model(dataset["odom"][prev], dataset["odom"][now])
        mu, S = ekf_predict(mu, S, u, R)

        corrected = (sensor_t != FLAG_SENSOR_FAILURE
                     and sensor_t != FLAG_TIME_MISSMATCH
                     and np.size(dataset["z"][sensor_t]) > 0)
        if corrected:
            mu, S = ekf_correct(mu, S, dataset["z"][sensor_t], Q, M, gate=gate)

        trace.append(mu.copy())

        if plot:
            color = "green" if corrected else "blue"
            plt.plot(mu[0], mu[1], "x", color=color)
            plot_2dcov_axes(axes, mu, S, color=color)

    trace = np.array(trace)
    if plot:
        gt = np.stack(dataset["gt"], axis=0)
        plt.plot(gt[:last_step, 0], gt[:last_step, 1], color="red")
        plt.tight_layout()
    return mu, S, trace


# =============================================================================
# TASK 7 -- Localization under landmark degradation
# =============================================================================

def get_timesteps_system(sensor_specs, odom_step_size, last_step, max_steps,
                         plot=False):
    if sensor_specs is None:
        odom_timesteps = get_timesteps(odom_step_size, last_step, max_steps)
        sensor_timesteps = get_timesteps(odom_step_size, last_step, max_steps)
    else:
        raise NotImplementedError(
            "Task 7 (sensor degradation) is not part of this release.")

    assert odom_timesteps.size == sensor_timesteps.size, \
        "sensor_timesteps must be the same length as odom_timesteps"
    return odom_timesteps, sensor_timesteps


# =============================================================================
# =============================================================================

def wrapToPi(theta):
    """Wrap an angle into [-pi, pi]."""
    return (theta + np.pi) % (2 * np.pi) - np.pi


def dist(x, y):
    return np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def normpdf(x, mu, sigma):
    u = (x - mu) / abs(sigma)
    return (1.0 / (np.sqrt(2 * np.pi) * abs(sigma))) * np.exp(-u * u / 2)


def load_marker_map():
    
    for candidate in (os.path.join(HERE, "config", "map_M.npy"),
                      os.path.join(os.path.dirname(HERE), "config", "map_M.npy")):
        if os.path.exists(candidate):
            return np.load(candidate)
    raise FileNotFoundError("config/map_M.npy not found next to this script")


def check_dataset_quality(d, path):

    meta = d.get("meta", {})
    ids = sorted(set(meta.get("ids_seen", [])))
    dist = float(meta.get("distance_m", 0.0))
    if len(ids) >= MIN_DATASET_IDS and dist >= MIN_DATASET_M:
        return

    missing = [i for i in range(MIN_DATASET_IDS) if i not in ids]
    raise SystemExit(
        f"\nThis recording is not good enough to judge a filter on.\n\n"
        f"  {os.path.basename(path)}: {len(ids)} of {MIN_DATASET_IDS} markers seen, "
        f"{dist:.1f} m driven\n"
        + (f"  markers never seen: {missing}\n" if missing else "")
        + f"  needed: all {MIN_DATASET_IDS} markers, at least "
          f"{MIN_DATASET_M:.0f} m driven\n\n"
        "With this little coverage a CORRECT filter can still score worse than dead\n"
        "reckoning, so the result would tell you nothing about your code.\n\n"
        "Drive the full route B-A-F-A-B-G-D-E-D-C and watch the recorder's marker\n"
        "count until all 20 have appeared, then stop. See 'Record your dataset' in\n"
        "the README.\n")


def load_dataset(path=DEFAULT_DATASET, calibrate=False,
                 thr_m=STEP_TRANS_M, thr_deg=STEP_ROT_DEG):
  
    if not os.path.exists(path):
        raise SystemExit(
            f"\nNo dataset found at {path}\n\n"
            "Tasks 3-5 run on a recording you make yourself. To record one:\n"
            "  1. ros2 launch exploration_assignment ekf_demo.launch.py\n"
            "  2. drive the route B-A-F-A-B-G-D-E-D-C with teleop\n"
            "  3. ros2 run exploration_assignment record_dataset.py --out data/my_run.p\n\n"
            "See 'Record your dataset' in the README for the full constraints.\n")
    with open(path, "rb") as f:
        d = pickle.load(f)

    check_dataset_quality(d, path)

    odom, z = np.asarray(d["odom"]), d["z"]
    has_obs = [zi is not None and np.size(zi) > 0 for zi in z]

    keep, last, i, n = [0], 0, 1, len(odom)
    while i < n:
        moved = np.hypot(odom[i, 0] - odom[last, 0], odom[i, 1] - odom[last, 1])
        turned = abs(wrapToPi(odom[i, 2] - odom[last, 2]))
        if moved >= thr_m or turned >= np.deg2rad(thr_deg):
            pick = i
            if not has_obs[i]:
                for j in range(i, min(n, i + 6)):
                    if has_obs[j]:
                        pick = j
                        break
            keep.append(pick)
            last, i = pick, pick + 1
        else:
            i += 1
    keep = np.array(keep)

    out = {"M": np.asarray(d["M"]), "meta": d.get("meta", {})}
    for key in ("odom", "odom_true", "gt"):
        out[key] = np.asarray(d[key])[keep]
    out["t"] = np.asarray(d["t"])[keep]
    out["z"] = []
    for k in keep:
        zi = z[k]
        if zi is None or np.size(zi) == 0:
            out["z"].append(np.zeros((3, 0)))
            continue
        zi = np.asarray(zi, dtype=float).copy()
        if calibrate:
            zi[0, :] = (zi[0, :] - RANGE_OFFSET) / RANGE_SCALE
        out["z"].append(zi)

    raw_seam = d.get("meta", {}).get("seam_index")
    out["seam"] = int(np.searchsorted(keep, raw_seam)) if raw_seam else None
    return out


def get_timesteps(step_size, last_step, total_steps):
    last_step = total_steps if last_step is None else min(total_steps, last_step)
    return np.array(list(range(0, last_step, step_size)))


def cov_ellipse_params(cov):
    vals, vecs = np.linalg.eigh(cov[:2, :2])
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals, 0.0)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return 2 * np.sqrt(vals[0]), 2 * np.sqrt(vals[1]), angle


def plot_2dcov_axes(axes, mu, cov, color="blue"):
    w, h, angle = cov_ellipse_params(cov)
    axes.add_patch(Ellipse((mu[0], mu[1]), width=w, height=h, angle=angle,
                           edgecolor=color, facecolor="none", alpha=0.5))


def plot_state(M):
    plt.figure(figsize=(8, 6))
    axes = plt.gca()
    plt.plot(M[:, 0], M[:, 1], "^", color="black")
    axes.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    axes.set_ylim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
    plt.title("EKF Localization")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend(handles=[
        Line2D([0], [0], marker="^", color="black", linestyle="", label="Map"),
        Line2D([0], [0], color="r", label="GroundTruth"),
        Line2D([0], [0], marker="x", color="b", linestyle="",
               label="Prediction only"),
        Line2D([0], [0], marker="x", color="green", linestyle="",
               label="Prediction & Correction"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    return axes


def plot_result_ekf(mode="predict", dataset_path=DEFAULT_DATASET, plot=True):

    assert mode in ("predict", "predict&correct"), f"invalid mode {mode}"
    dataset = load_dataset(dataset_path)
    M = dataset["M"]

    R = np.diag(np.array([0.25, 0.25, np.deg2rad(10)]) ** 2)
    Q = np.diag(np.array([0.3, np.deg2rad(5)]) ** 2)

    mu = np.array(dataset["gt"][0], dtype=float)
    S = np.zeros([3, 3])

    plt.figure(figsize=(8, 6))
    axes = plt.gca()
    plt.title("EKF prediction" + (" and correction" if "correct" in mode else ""))

    u = [inverse_motion_model(dataset["odom"][0], dataset["odom"][50]),
         inverse_motion_model(dataset["odom"][50], dataset["odom"][100])]
    z = [dataset["z"][50], dataset["z"][100]]
    gt = [dataset["gt"][50], dataset["gt"][100]]

    for u_i, z_i, gt_i, first in zip(u, z, gt, (True, False)):
        mu, S = ekf_predict(mu, S, u_i, R)
        plt.plot(mu[0], mu[1], "x", color="blue",
                 label="Prediction" if first else "_")
        plot_2dcov_axes(axes, mu, S, color="blue")
        print(f"  after predict : mu = [{mu[0]:+.3f} {mu[1]:+.3f} "
              f"{np.degrees(mu[2]):+7.2f} deg]  trace(S) = {np.trace(S):.5f}")

        if "correct" in mode:
            mu, S = ekf_correct(mu, S, z_i, Q, M)
            plt.plot(mu[0], mu[1], "x", color="green",
                     label="Correction" if first else "_")
            plot_2dcov_axes(axes, mu, S, color="green")
            print(f"  after correct : mu = [{mu[0]:+.3f} {mu[1]:+.3f} "
                  f"{np.degrees(mu[2]):+7.2f} deg]  trace(S) = {np.trace(S):.5f}")

        plt.plot(gt_i[0], gt_i[1], ".r", label="GroundTruth" if first else "_")

    plt.legend()
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.tight_layout()
    return mu, S


# =============================================================================
# =============================================================================

def _task1(args):
    ut = [[0.0, 0.0, 0.0], [0.5, 0.0, np.pi / 2]]
    x_t_1 = [0.0, 0.0, 0.0]
    alpha = [0.5, 0.5, 0.01, 0.01]

    print("  inverse_motion_model([0,0,0] -> [4,3,0])")
    rot1, trans, rot2 = inverse_motion_model([0, 0, 0], [4, 3, 0])
    print(f"     rot1={rot1:+.4f}  trans={trans:.4f}  rot2={rot2:+.4f}"
          f"   (expect trans=5, rot1=-rot2)")

    print("  motion_model")
    print(f"     p = {motion_model([4, 3, 0], [0, 0, 0], [[0.0, 0.0, 0.0], [-4, -3, 0]], [1.0, 1.0, 0.01, 0.01]):.6e}")

    n = 40 if args.fast else 150
    print(f"  plot_posterior_belief (grid {n}x{n}"
          f"{', --fast' if args.fast else ''})")
    plot_posterior_belief(x_t_1, ut, alpha, size=n)
    plot_posterior_belief(x_t_1, ut, alpha, size=n, marginalise_p3=True)


def _task2(args):

    M = np.asarray(load_marker_map())

    # A real pose in the house, near the spawn, with several markers in range.
    x_true = np.array([-2.50, 2.80])
    theta_true = 0.0
    ids = [2, 0, 9]                            

    sensor = {"sigma_r": 0.0165, "sigma_phi": np.deg2rad(0.243)}
 
    vis = {"sigma_r": 0.30, "sigma_phi": np.deg2rad(4.0)}
    print(f"  landmark_observation_model   (sigma_r = {sensor['sigma_r']} m)")
  
    for mid in ids:
        r = float(np.hypot(*(M[mid] - x_true)))
        p_ok = landmark_observation_model(r, sensor["sigma_r"], M[mid], x_true)
        p_off = landmark_observation_model(r + 0.05, sensor["sigma_r"], M[mid], x_true)
        print(f"     marker {mid:2d} at [{M[mid][0]:+.2f} {M[mid][1]:+.2f}]  "
              f"true range {r:.3f} m  ->  p = {p_ok:8.4f}   "
              f"5 cm off -> {p_off:.2e}  ({p_ok/max(p_off, 1e-300):.0f}x lower)")

    pad, step = 1.0, 0.05
    extent = (M[:, 0].min() - pad, M[:, 0].max() + pad,
              M[:, 1].min() - pad, M[:, 1].max() + pad)
    nx = int((extent[1] - extent[0]) / step) + 1
    ny = int((extent[3] - extent[2]) / step) + 1
    gridmap = np.zeros([nx, ny])
    print(f"\n  likelihood fields over the real map "
          f"({nx}x{ny} cells at {step} m)")

    z = np.array([np.hypot(*(M[i] - x_true)) for i in ids])
    sig = np.full(len(ids), vis["sigma_r"])

    fig, axs = plt.subplots(1, 4, figsize=(19, 4.6))
    for n in (1, 2, 3):
        lk = joint_observation_likelihood(z[:n], sig[:n], M[ids[:n]], gridmap, extent)
        _plot_field(axs[n - 1], lk, extent, M, ids[:n], x_true,
                    f"{n} marker(s), RANGE only")

 
    lk = np.zeros_like(gridmap)
    b, mid = M[ids[0]], ids[0]
    z_r = float(np.hypot(*(b - x_true)))
    z_phi = float(wrapToPi(np.arctan2(b[1] - x_true[1], b[0] - x_true[0]) - theta_true))
    for i in range(nx):
        for j in range(ny):
            lk[i, j] = landmark_observation_model_rb(
                z_r, z_phi, vis["sigma_r"], vis["sigma_phi"], b,
                cell_to_world(i, j, gridmap, extent), theta_true)
    lk /= lk.sum()
    _plot_field(axs[3], lk, extent, M, [mid], x_true,
                "1 marker, RANGE + BEARING\n(heading known)")
    plt.suptitle(f"Task 2 -- observation likelihood over the real marker map "
                 f"(sigma widened to {vis['sigma_r']} m for visibility; "
                 f"real sigma_r = {sensor['sigma_r']} m)")
    plt.tight_layout()


def _plot_field(ax, lk, extent, M, used, x_true, title):
   
    ax.imshow(np.transpose(lk), origin="lower", cmap="viridis",
              extent=(extent[0], extent[1], extent[2], extent[3]), aspect="equal")
    # the argmax, so a sub-pixel blob is still locatable
    pk = np.unravel_index(np.argmax(lk), lk.shape)
    pw = cell_to_world(pk[0], pk[1], lk, extent)
    ax.plot(pw[0], pw[1], "o", mfc="none", mec="red", ms=13, mew=1.6)
    ax.scatter(M[:, 0], M[:, 1], c="white", s=8, alpha=0.45, label="all markers")
    ax.scatter(M[used, 0], M[used, 1], c="red", s=55, marker="s", label="used")
    ax.plot(x_true[0], x_true[1], "w*", ms=15, label="true pose")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)


def dead_reckoning(dataset):
    """Where the robot would think it is on noisy odometry alone, in map frame."""
    o, gt = np.asarray(dataset["odom"]), np.asarray(dataset["gt"])
    th0 = gt[0, 2] - o[0, 2]
    c, s = np.cos(th0), np.sin(th0)
    R = np.array([[c, -s], [s, c]])
    xy = (R @ (o[:, :2] - o[0, :2]).T).T + gt[0, :2]
    return np.column_stack([xy, wrapToPi(o[:, 2] + th0)])


def _run_and_report(args, blocked_ids=None, label=""):

    _, _, trace = run_ekf_localization(
        args.dataset, plot=args.task != 7, step_size=args.step_size,
        last_step=args.last_step, blocked_ids=blocked_ids)
    dataset = load_dataset(args.dataset)
    gt = np.array(dataset["gt"])
    ts = get_timesteps(args.step_size, args.last_step, len(dataset["odom"]))
    err = np.hypot(trace[:, 0] - gt[ts, 0], trace[:, 1] - gt[ts, 1])
    ang = np.degrees(np.abs(wrapToPi(trace[:, 2] - gt[ts, 2])))
    tag = f"{label} " if label else ""
    print(f"  {tag}position error over {len(ts)} steps: mean {err.mean():.3f} m  "
          f"median {np.median(err):.3f} m  max {err.max():.3f} m")
    print(f"  {tag}heading error: mean {ang.mean():.2f} deg  max {ang.max():.2f} deg")

    dr = dead_reckoning(dataset)[ts]
    dr_err = np.hypot(dr[:, 0] - gt[ts, 0], dr[:, 1] - gt[ts, 1])
    dr_ang = np.degrees(np.abs(wrapToPi(dr[:, 2] - gt[ts, 2])))
    print(f"  {tag}dead reckoning: mean {dr_err.mean():.3f} m  "
          f"{dr_ang.mean():.2f} deg")
    print(f"  {tag}-> your filter is {dr_err.mean()/max(err.mean(), 1e-12):.0f}x better "
          f"in position, {dr_ang.mean()/max(ang.mean(), 1e-12):.0f}x in heading")
    return err, ang


def main():
    ap = argparse.ArgumentParser(
        description="CS7001 Assignment 1 -- EKF localization with ArUco landmarks")
    ap.add_argument("--task", type=int, required=True,
                    choices=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--step-size", type=int, default=1)
    ap.add_argument("--last-step", type=int, default=None)
    ap.add_argument("--fast", action="store_true",
                    help="Task 1: coarsen the posterior grid (150x150x8 is ~180k "
                         "model evaluations and takes a while)")
    ap.add_argument("--save", metavar="DIR",
                    help="write figures here instead of showing them")
    args = ap.parse_args()

    print(f"=== Task {args.task} ===")
    if args.task == 1:
        _task1(args)
    elif args.task == 2:
        _task2(args)
    elif args.task == 3:
        plot_result_ekf("predict", args.dataset)
    elif args.task == 4:
        plot_result_ekf("predict&correct", args.dataset)
    elif args.task in (5, 6):
        _run_and_report(args)
        if args.task == 6:
            if args.dataset == DEFAULT_DATASET:
                print("\n  NOTE: that was the PROVIDED recording. Task 6 is about your"
                      " own run:")
                print("    python3 scripts/ekf_localization.py --task 6 "
                      "--dataset data/my_run.p")
            print("\n  Task 6 also requires, from your own teleop run:")
            print("    the two trajectory CSVs:  "
                  "evaluate.py --dataset <your.p> --csv <dir>")
            print("    a bag of the /diag/* topics (see the README; never "
                  "`ros2 bag record -a`)")
    elif args.task == 7:
       
        print(f"  blocking room F, marker ids {ROOM_F_IDS}\n")
        base_err, base_ang = _run_and_report(args, label="intact  ")
        deg_err, deg_ang = _run_and_report(args, blocked_ids=ROOM_F_IDS,
                                          label="degraded")
        print(f"\n  degradation cost: position {deg_err.mean() - base_err.mean():+.3f} m "
              f"mean ({deg_err.mean()/max(base_err.mean(), 1e-12):.1f}x), "
              f"peak {deg_err.max():.3f} m")
        print(f"                    heading  {deg_ang.mean() - base_ang.mean():+.2f} deg "
              f"mean, peak {deg_ang.max():.2f} deg")
       
        gap = deg_err - base_err
        k = int(np.argmax(gap))
        print(f"\n  worst divergence at step {k}: {deg_err[k]:.3f} m degraded vs "
              f"{base_err[k]:.3f} m intact  (+{gap[k]:.3f} m)")
        print("  Look at this window, not the run mean -- the mean averages the "
              "blackout\n  away over the whole route. Plot the error against step "
              "to see the episode.")

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for i in plt.get_fignums():
            p = os.path.join(args.save, f"task{args.task}_fig{i}.png")
            plt.figure(i).savefig(p, dpi=110)
            print(f"  wrote {p}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
